//! Parallel schedules: integer scan/sort/sum, any-type filter and concat
//! lower to multi-pass schedules, and every emitted shader validates.
use ink_lowering_gpu::{compute, core::CheckedModule, syntax::parse};
use serde_json::Value;
fn emit(s: &str) -> ink_lowering_gpu::gpu::Emission {
    compute::lower(&CheckedModule::from_source(parse(s).unwrap()).unwrap()).unwrap()
}
fn function<'a>(e: &'a ink_lowering_gpu::gpu::Emission, name: &str) -> &'a Value {
    e.manifest["functions"].as_array().unwrap().iter().find(|f| f["name"] == name).unwrap()
}
fn validate(name: &str, code: &str) {
    let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
}
#[test]
fn parallel_stages_carry_complete_schedules() {
    let e = emit(include_str!("fixtures/parallel.ink"));
    for (name, op, first) in [
        ("scan_u32", "scan", "scan0"),
        ("scan_u64", "scan", "scan0"),
        ("sort_i32", "sort", "copy"),
        ("sort_u64", "sort", "copy"),
        ("sum_u64", "sum", "sum0"),
        ("keep_items", "filter", "flags"),
        ("keep_odd", "filter", "flags"),
        ("cat_items", "concat", "concat"),
    ] {
        let f = function(&e, name);
        assert_eq!(f["graph_required"], true, "{name}");
        let node = &f["graph"]["nodes"][0];
        assert_eq!(node["op"], op, "{name}");
        assert_eq!(node["parallel"], true, "{name}");
        assert!(node["shader"].is_null(), "{name}");
        let schedule = &node["schedule"];
        let passes = schedule["passes"].as_array().unwrap();
        assert!(passes[0]["shader"].as_str().unwrap().ends_with(&format!("{first}.wgsl")), "{name}");
        let scratch = schedule["scratch"].as_array().unwrap().len() as u64;
        let args = node["args"].as_array().unwrap().len() as u64;
        for pass in passes {
            let shader = pass["shader"].as_str().unwrap();
            assert!(e.shaders.contains_key(shader), "{name}: {shader}");
            assert!(["b1", "b2", "r1", "one"].contains(&pass["groups"].as_str().unwrap()), "{name}");
            let mut out = 0;
            for b in pass["bind"].as_array().unwrap() {
                if let Some(a) = b["arg"].as_u64() {
                    assert!(a < args, "{name}");
                } else if let Some(j) = b["scratch"].as_u64() {
                    assert!(j < scratch, "{name}");
                } else {
                    assert_eq!(b["out"], true, "{name}");
                    out += 1;
                }
            }
            assert!(out <= 1, "{name}");
        }
        // The result is written by the final pass.
        assert!(passes.last().unwrap()["bind"].as_array().unwrap().iter().any(|b| b["out"] == true), "{name}");
    }
    // Radix sort: four 8-bit digits per 32-bit lane, ping-ponging buffers.
    let digits = |name: &str| {
        function(&e, name)["graph"]["nodes"][0]["schedule"]["passes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p["shader"].as_str().unwrap().contains("-place"))
            .count()
    };
    assert_eq!(digits("sort_u32"), 4);
    assert_eq!(digits("sort_u64"), 8);
    // Float sums keep their sequential source order (no reduction tree).
    let f = emit("module Floats;\nfn total(xs: List<f32>) -> f32 { return sum(xs); }");
    let total = function(&f, "total");
    assert!(total["graph"]["nodes"].as_array().unwrap().iter().all(|n| n["schedule"].is_null()));
}
#[test]
fn every_emitted_shader_validates() {
    for (fixture, source) in [
        ("parallel", include_str!("fixtures/parallel.ink")),
        ("literal-graph", include_str!("fixtures/literal-graph.ink")),
    ] {
        let e = emit(source);
        assert!(!e.shaders.is_empty(), "{fixture}");
        for (name, code) in &e.shaders {
            validate(&format!("{fixture}/{name}"), code);
        }
    }
}
#[test]
fn browser_reduction_shaders_validate() {
    // The browser device generates read reductions at run time; check every
    // kind, width and operation it can request. Skipped without node.
    let script = "import {reduceShader} from './devices/compute.mjs';const out={};for(const k of ['u32','i32','f32','u64'])for(const c of [1,2,3,4])for(const o of ['sum','min','max'])if(k!=='u64'||c===1)out[`${k}-${c}-${o}`]=reduceShader(k,c,o);console.log(JSON.stringify(out));";
    let Ok(out) = std::process::Command::new("node")
        .args(["--input-type=module", "-e", script])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
    else {
        eprintln!("node unavailable; skipped");
        return;
    };
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let shaders: serde_json::Map<String, Value> = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(shaders.len(), 39);
    for (name, code) in shaders {
        validate(&name, code.as_str().unwrap());
    }
}
