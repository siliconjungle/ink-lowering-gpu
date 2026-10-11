use ink_lowering_gpu::{compute, core::CheckedModule, syntax::parse};
#[test]
fn concat_materializes_both_word_widths_with_the_shared_layout() {
    let module=CheckedModule::from_source(parse("module t; fn join(a:List<u32>,b:List<u32>)->List<u32>{return a.concat(b);} fn wide(a:List<u64>,b:List<u64>)->List<u64>{return a.concat(b);}").unwrap()).unwrap();
    let emission = compute::lower(&module).unwrap();
    let join = &emission.manifest["functions"][0];
    assert!(join["shader"].is_null());
    assert_eq!(join["graph"]["nodes"][0]["op"], "concat");
    assert_eq!(join["graph"]["nodes"][0]["capacity"]["kind"], "sum");
    assert_eq!(join["graph_required"], true);
    let wide = &emission.manifest["functions"][1];
    assert!(wide["shader"].is_null());
    assert_eq!(wide["graph"]["nodes"][0]["op"], "concat");
    assert_eq!(wide["graph_required"], true);
    assert_eq!(wide["params"][0]["type"]["element"]["kind"], "u64");
    assert!(wide["fallback_reason"].is_null());
}
