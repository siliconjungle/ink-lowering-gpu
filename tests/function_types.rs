use ink_lowering_gpu::{compute, core::CheckedModule, syntax::parse};
fn emission(source: &str) -> ink_lowering_gpu::gpu::Emission {
    compute::lower(&CheckedModule::from_source(parse(source).unwrap()).unwrap()).unwrap()
}
fn entry<'a>(e: &'a ink_lowering_gpu::gpu::Emission, name: &str) -> &'a serde_json::Value {
    e.manifest["functions"].as_array().unwrap().iter().find(|f| f["name"] == name).unwrap()
}
fn validate(e: &ink_lowering_gpu::gpu::Emission) {
    assert!(!e.shaders.is_empty());
    for (name, code) in &e.shaders {
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
    }
}
#[test]
fn unrelated_owned_and_binary64_records_do_not_block_literal_device_functions() {
    let e = emission(r#"module PerFunction;
record Unused{items:List<f64>,}
record Precise{value:f64,}
record Point{x:f32,}
record Pair{point:Point,index:u32,}
fn value(x:f32)->f32{let p=Pair{point:Point{x:x},index:3};return p.point.x;}
fn mapped(xs:List<f32>)->List<f32>{return xs.map(fn(x)=>value(x)+1.0);}
fn precise(x:Precise)->f64{return x.value;}
"#);
    assert!(entry(&e, "mapped")["fallback_reason"].is_null(), "{}", entry(&e,"mapped"));
    assert!(entry(&e, "precise")["fallback_reason"].is_string());
    for code in e.shaders.values() {
        assert!(!code.contains("struct R_Unused"));
        assert!(!code.contains("struct R_Precise"));
    }
    validate(&e);
}
#[test]
fn local_annotations_and_helpers_retain_types_without_constructors() {
    let e = emission(r#"module Annotated;
record Item{value:f32,}
record Unused{items:List<f64>,}
fn helper(x:u32)->u32{let empty:List<Item>=[];return length(empty)+x;}
fn mapped(xs:List<u32>)->List<u32>{return xs.map(fn(x)=>helper(x));}
"#);
    assert!(entry(&e,"mapped")["fallback_reason"].is_null(), "{}", entry(&e,"mapped"));
    validate(&e);
}
#[test]
fn reachable_binary64_fields_and_strict_argument_work_stay_cpu_only() {
    let e = emission(r#"module StrictTypes;
record Mixed{value:f32,wide:f64,}
fn projection(xs:List<Mixed>)->List<f32>{return xs.map(fn(x)=>x.value);}
fn ignore(x:f64)->f32{return 1.0;}
fn strict(xs:List<f64>)->List<f32>{return xs.map(fn(x)=>ignore(x));}
"#);
    for name in ["projection", "strict"] {
        let f=entry(&e,name);
        assert!(f["fallback_reason"].is_string(), "{name}: {f}");
        assert!(f["shader"].is_null() && f["graph"].is_null(), "{name}: {f}");
    }
}
