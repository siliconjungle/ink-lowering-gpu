use ink_lowering_gpu::{compute, core::CheckedModule, syntax::parse};

fn emit(source: &str) -> ink_lowering_gpu::gpu::Emission {
    compute::lower(&CheckedModule::from_source(parse(source).unwrap()).unwrap()).unwrap()
}

fn function<'a>(e: &'a ink_lowering_gpu::gpu::Emission, name: &str) -> &'a serde_json::Value {
    e.manifest["functions"].as_array().unwrap().iter()
        .find(|f| f["name"] == name).unwrap()
}

#[test]
fn local_lists_cross_helpers_loops_captures_and_scalar_record_elements() {
    let e = emit(r#"module LocalLists;
record Pair{value:f32,index:u32,}
fn layer(x:f32)->List<f32>{return [x*x,x+1.0,-x];}
fn use_layer(x:f32,i:u32)->f32{
 let first=layer(x);
 let second=layer(first.at_or(i,0.0));
 let empty:List<f32>=[];
 return foldl(second,first.at_or(0,0.0),fn(x)=>fn(total)=>total+x)+empty.at_or(i,-0.0);
}
fn mapped(xs:List<f32>,i:u32)->List<f32>{return xs.map(fn(x)=>use_layer(x,i));}
fn records(xs:List<f32>)->List<f32>{return xs.map(fn(x)=>consume(x));}
fn consume(x:f32)->f32{
 let pairs=[Pair{value:x,index:3},Pair{value:-x,index:7}];
 return repeat(2,0.0,fn(i)=>fn(total)=>total+pairs.at_or(i,Pair{value:0.0,index:0}).value);
}
"#);
    for name in ["mapped", "records"] {
        let f = function(&e, name);
        assert!(f["fallback_reason"].is_null(), "{name}: {}", f["fallback_reason"]);
        assert!(f["shader"].is_string() || f["graph"].is_object());
    }
    for (name, code) in &e.shaders {
        let module = naga::front::wgsl::parse_str(code)
            .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
            .validate(&module).unwrap_or_else(|error| panic!("{name}: {error:?}"));
    }
}

#[test]
fn local_storage_exhaustion_retains_cpu_without_publishing_partial_shaders() {
    let items = vec!["x"; 4097].join(",");
    let source = format!("module LargeLocal;fn inner(x:f32)->f32{{let xs=[{items}];return sum(xs);}}fn outer(xs:List<f32>)->List<f32>{{return xs.map(fn(x)=>inner(x));}}");
    let e = emit(&source);
    let f = function(&e, "outer");
    assert!(f["shader"].is_null());
    assert!(f["graph"].is_null());
    assert!(f["fallback_reason"].as_str().unwrap().contains("local list storage budget"));
}

#[test]
fn lazy_list_control_has_one_bounded_graph_stage_without_eager_arm_stages() {
    let e = emit("module LazyLists;fn branch(xs:List<f32>,ys:List<f32>,left:Bool)->List<f32>{return choose(left,xs.map(fn(x)=>x*3.0),ys.map(fn(y)=>y+7.0));}");
    let f = function(&e, "branch");
    assert!(f["fallback_reason"].is_null(), "{}", f["fallback_reason"]);
    let nodes = f["graph"]["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0]["op"], "literal");
    assert_eq!(nodes[0]["capacity"]["kind"], "max");
    assert_eq!(nodes[0]["capacity"]["args"].as_array().unwrap().len(), 2);
    let code = &e.shaders[nodes[0]["shader"].as_str().unwrap()];
    let module = naga::front::wgsl::parse_str(code).unwrap();
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
        .validate(&module).unwrap();
}

#[test]
fn private_list_budget_includes_wgsl_record_and_vector_padding() {
    // Packed P has five words; WGSL P has eight words after alignment.
    // 700 elements fit a packed-only estimate but exceed the 4096-word bound.
    let items = vec!["x"; 700].join(",");
    let source = format!("module Padded;record P{{v:Vec3<u32>,tag:u64,}}fn inner(x:P)->u32{{let ys=[{items}];return length(ys);}}fn outer(xs:List<P>)->List<u32>{{return xs.map(fn(x)=>inner(x));}}");
    let e = emit(&source);
    let f = function(&e, "outer");
    assert!(f["shader"].is_null());
    assert!(f["graph"].is_null());
    assert!(f["fallback_reason"].as_str().unwrap().contains("local list storage budget"));
}

#[test]
fn literal_graph_wrapping_cannot_remove_direct_collection_work_limits() {
    let e = emit("module BoundedControl;fn branch(xs:List<u32>,ys:List<u32>,left:Bool)->List<u32>{return choose(left,xs.sort(),ys);}");
    let f = function(&e, "branch");
    assert!(f["graph"].is_null());
    assert!(f["shader"].is_string());
    assert_eq!(f["gpu_max_elements"], 128);
}

#[test]
fn unchecked_helper_preconditions_reject_before_device_emission() {
    let source = r#"module ContractLists;
spec fn positive(x:u32)->u32 requires x>0; ensures result==x;
implementation direct for positive{return x;}
fn hidden(x:u32)->List<u32>{return [positive(x),x+1];}
fn mapped(xs:List<u32>)->List<u32>{return xs.map(fn(x)=>hidden(x).at_or(0,0));}
"#;
    let error = CheckedModule::from_source(parse(source).unwrap()).err().expect("unchecked precondition admitted");
    assert!(error.contains("requires"), "{error}");
}
