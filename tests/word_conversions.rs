use ink_lowering_gpu::{compute,core::CheckedModule,syntax::parse};
#[test]
fn fixed_word_conversions_emit_valid_direct_and_graph_shaders() {
    let source="module Conversions;
        fn widen(xs:List<u32>)->List<u64>{return xs.map(fn(x)=>word_widen_u32_to_u64(x));}
        fn narrow(xs:List<u64>)->List<u32>{return xs.map(fn(x)=>word_narrow_u64_to_u32(x));}
        fn roundtrip(xs:List<u64>)->List<u64>{return xs.map(fn(x)=>word_widen_u32_to_u64(word_narrow_u64_to_u32(x)));}";
    let p=CheckedModule::from_source(parse(source).unwrap()).unwrap();let e=compute::lower(&p).unwrap();
    for f in e.manifest["functions"].as_array().unwrap() {
        assert!(f["shader"].is_string(),"{}",f["fallback_reason"]);
        assert!(f["graph"].is_object(),"{}",f["fallback_reason"]);
    }
    for (name,source) in &e.shaders {
        let module=naga::front::wgsl::parse_str(source).unwrap_or_else(|error|panic!("{name}: {}",error.emit_to_string(source)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(),naga::valid::Capabilities::all())
            .validate(&module).unwrap_or_else(|error|panic!("{name}: {error}"));
    }
}
