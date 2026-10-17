use ink_lowering_gpu::{compute, core::CheckedModule, syntax::parse};
fn emit(source: &str, name: &str) -> ink_lowering_gpu::gpu::Emission {
    compute::lower_entry(
        &CheckedModule::from_source(parse(source).unwrap()).unwrap(),
        name,
    )
    .unwrap()
}
fn check(e: &ink_lowering_gpu::gpu::Emission) {
    assert!(
        e.manifest["functions"][0]["fallback_reason"].is_null(),
        "{}",
        e.manifest["functions"][0]
    );
    assert!(!e.shaders.is_empty());
    for (name, code) in &e.shaders {
        let module = naga::front::wgsl::parse_str(code)
            .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(code)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{name}: {error:?}"));
    }
}
#[test]
fn records_carry_private_buffer_bounds_through_helpers_fields_branches_and_loops() {
    let e = emit(
        r#"module RecordBuffers;
record Node{value:f32,index:u32,}
record Store{nodes:Buffer<Node>,}
record Work{store:Store,steps:u32,}
fn initial(x:f32)->Work{return Work{store:Store{nodes:buffer_fill(2,Node{value:x,index:1})},steps:0};}
fn update(work:Work,index:u32,x:f32)->Work{return Work{store:Store{nodes:work.store.nodes.set(index,Node{value:x,index:index})},steps:work.steps+1};}
fn element(x:f32,flag:Bool)->f32{
 let w=initial(x);
 let w=choose(flag,w,initial(-x));
 let w=repeat(3,w,fn(i)=>fn(w)=>update(w,i,x));
 return w.store.nodes.at_or(1,Node{value:-0.0,index:0}).value;
}
fn outer(xs:List<f32>,flag:Bool)->List<f32>{return xs.map(fn(x)=>element(x,flag));}
"#,
        "outer",
    );
    check(&e);
}
#[test]
fn anonymous_record_and_vector_buffers_have_exact_private_declarations() {
    for source in [
        "module Anonymous;record P{v:Vec3<u32>,wide:u64,}fn output(xs:List<u32>)->List<u64>{return xs.map(fn(x)=>buffer_fill(2,P{v:vec3(x,x,x),wide:1}).at_or(1,P{v:vec3(0,0,0),wide:0}).wide);}",
        "module Vectors;fn output(xs:List<u32>)->List<u32>{return xs.map(fn(x)=>buffer_fill(2,vec3(x,x,x)).at_or(0,vec3(0,0,0)).y);}",
    ] {check(&emit(source,"output"));}
}
#[test]
fn loop_checks_cover_each_owned_record_field() {
    for expression in [
        "repeat(100,w,fn(i)=>fn(w)=>Work{items:w.items.push(x)})",
        "repeat_until(100,w,fn(w)=>false,fn(i)=>fn(w)=>Work{items:w.items.push(x)})",
        "foldl(xs,w,fn(x)=>fn(w)=>Work{items:w.items.push(x)})",
        "foldr(xs,w,fn(x)=>fn(w)=>Work{items:w.items.push(x)})",
        "fold_until(xs,w,fn(x)=>fn(w)=>false,fn(x)=>fn(w)=>Work{items:w.items.push(x)})",
    ] {
        let source=format!("module Growing;record Work{{items:Buffer<u32>,}}fn element(x:u32,xs:List<u32>)->u32{{let w=Work{{items:buffer_fill(0,x)}};let w={expression};return w.items.len();}}fn output(xs:List<u32>)->List<u32>{{return xs.map(fn(x)=>element(x,xs));}}");
        let e = emit(&source, "output");
        let f = &e.manifest["functions"][0];
        assert!(
            f["shader"].is_null() && f["graph"].is_null(),
            "{expression}: {f}"
        );
        assert!(
            f["fallback_reason"]
                .as_str()
                .unwrap()
                .contains("loop may grow"),
            "{expression}: {f}"
        );
    }
}

#[test]
fn full_record_tape_observations_and_ordered_buffer_scans_are_device_eligible() {
    let source = include_str!("fixtures/record-tape.ink");
    for name in [
        "tape_observation",
        "mapped",
        "partitioned",
        "scanned",
        "scanned_wide",
        "selected_wide",
        "empty_owned",
    ] {
        check(&emit(source, name));
    }
}

#[test]
fn private_scan_classification_does_not_relax_list_scan_work_limits() {
    let source="module ScanBounds;fn bounded(x:u32,c:Bool)->u32{let values=buffer_fill(4,x);return choose(c,values.scan().at_or(3,0),7);}fn unbounded(xs:List<u32>,c:Bool)->u32{return choose(c,xs.scan().at_or(0,0),7);}";
    check(&emit(source, "bounded"));
    let e = emit(source, "unbounded");
    assert!(
        e.manifest["functions"][0]["shader"].is_null()
            && e.manifest["functions"][0]["graph"].is_null()
    );
}

#[test]
fn local_and_input_output_capacities_keep_static_floor_and_lazy_arms() {
    let source = include_str!("fixtures/record-tape.ink");
    let e = emit(source, "selected_wide");
    check(&e);
    let nodes = e.manifest["functions"][0]["graph"]["nodes"]
        .as_array()
        .unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0]["capacity"]["kind"], "max");
    assert_eq!(nodes[0]["capacity"]["floor"], 4);
    let slot = nodes[0]["capacity"]["args"][0].as_u64().unwrap() as usize;
    assert_eq!(nodes[0]["args"][slot], serde_json::json!({"param":0}));
    let e = emit(source, "empty_owned");
    check(&e);
    assert_eq!(
        e.manifest["functions"][0]["graph"]["nodes"][0]["capacity"],
        serde_json::json!({"kind":"fixed","value":1,"args":[]})
    );
}
