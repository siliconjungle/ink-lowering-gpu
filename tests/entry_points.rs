use ink_lowering_gpu::{compute, core::CheckedModule, syntax::parse};
#[test]
fn internal_owned_helpers_do_not_require_host_transport() {
    let checked=CheckedModule::from_source(parse("module Entries;fn helper(x:u32)->Buffer<u32>{return buffer_fill(4,x);}fn unused(x:String)->String{return x;}fn output(xs:List<u32>)->List<u32>{return xs.map(fn(x)=>helper(x).at_or(0,0));}").unwrap()).unwrap();
    let e=compute::lower_entry(&checked,"output").unwrap();
    assert_eq!(e.manifest["functions"].as_array().unwrap().len(),1);
    let f=&e.manifest["functions"][0];
    assert_eq!(f["name"],"output");assert_eq!(f["index"],2);
    assert!(f["fallback_reason"].is_null(),"{f}");
    assert_eq!(e.manifest["core_sha256"],checked.identity().unwrap());
    assert!(compute::lower_entry(&checked,"missing").err().unwrap().contains("unknown"));
    for (name,code) in &e.shaders {
        let module=naga::front::wgsl::parse_str(code).unwrap_or_else(|error|panic!("{name}: {}",error.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(),naga::valid::Capabilities::empty()).validate(&module).unwrap();
    }
}
#[test]
fn entry_transport_does_not_hide_strict_unsupported_helper_work() {
    let checked=CheckedModule::from_source(parse("module Strict;fn hidden()->f64{return 1.0000000000000002f64;}fn ignore(x:f64)->f32{return 1.0;}fn output(xs:List<f32>)->List<f32>{return xs.map(fn(x)=>ignore(hidden())+x);}").unwrap()).unwrap();
    let e=compute::lower_entry(&checked,"output").unwrap();let f=&e.manifest["functions"][0];
    assert!(f["fallback_reason"].is_string());assert!(f["shader"].is_null()&&f["graph"].is_null());
}

#[test]
fn private_helper_folds_cannot_grow_beyond_a_proven_temporary_bound() {
    for fold in ["foldl(xs,buffer_fill(0,x),fn(item)=>fn(b)=>b.push(item))",
                 "foldr(xs,buffer_fill(0,x),fn(item)=>fn(b)=>b.push(item))",
                 "fold_until(xs,buffer_fill(0,x),fn(item)=>fn(b)=>false,fn(item)=>fn(b)=>b.push(item))"] {
        let source=format!("module Growing;fn helper(xs:List<u32>,x:u32)->Buffer<u32>{{return {fold};}}fn output(xs:List<u32>)->List<u32>{{return xs.map(fn(x)=>helper(xs,x).len());}}");
        let checked=CheckedModule::from_source(parse(&source).unwrap()).unwrap();
        let e=compute::lower_entry(&checked,"output").unwrap();let f=&e.manifest["functions"][0];
        assert!(f["shader"].is_null()&&f["graph"].is_null(),"{fold}: {f}");
        assert!(f["fallback_reason"].as_str().unwrap().contains("loop may grow"),"{fold}: {f}");
    }
}

#[test]
fn scalar_entries_keep_owned_helpers_local_and_preserve_lazy_work() {
    let checked=CheckedModule::from_source(parse("module Scalar;fn helper(x:u32)->Buffer<u32>{return buffer_fill(4,x);}fn output(x:u32,enabled:Bool)->u32{return choose(enabled,helper(x).at_or(3,7),9);}").unwrap()).unwrap();
    let e=compute::lower_entry(&checked,"output").unwrap();let f=&e.manifest["functions"][0];
    assert!(f["fallback_reason"].is_null(),"{f}");
    let nodes=f["graph"]["nodes"].as_array().unwrap();assert_eq!(nodes.len(),1);assert_eq!(nodes[0]["op"],"scalar");
    let code=&e.shaders[nodes[0]["shader"].as_str().unwrap()];
    let module=naga::front::wgsl::parse_str(code).unwrap();
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(),naga::valid::Capabilities::empty()).validate(&module).unwrap();
    let heavy=CheckedModule::from_source(parse("module Heavy;fn selected(xs:List<u32>,enabled:Bool)->u32{return choose(enabled,xs.sort().at_or(0,7),9);}").unwrap()).unwrap();
    let e=compute::lower_entry(&heavy,"selected").unwrap();assert!(e.manifest["functions"][0]["fallback_reason"].is_string());
}
