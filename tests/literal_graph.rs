use ink_lowering_gpu::{compute, core::CheckedModule, eval, syntax::parse};
use serde_json::json;
fn emit(s: &str) -> ink_lowering_gpu::gpu::Emission {
    compute::lower(&CheckedModule::from_source(parse(s).unwrap()).unwrap()).unwrap()
}
fn ops(e: &ink_lowering_gpu::gpu::Emission, name: &str) -> Vec<String> {
    e.manifest["functions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["name"] == name)
        .unwrap()["graph"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["op"].as_str().unwrap().to_string())
        .collect()
}
#[test]
fn materializes_each_literal_stage_and_ordered_reducer() {
    let source = include_str!("fixtures/literal-graph.ink");
    let e = emit(source);
    for (name, expected) in [
        ("mapped_filter", vec!["map", "filter", "map_indexed"]),
        ("ordered", vec!["filter", "map", "foldr"]),
        ("paired", vec!["map", "filter", "zip", "map"]),
        ("joined", vec!["scan", "map", "concat"]),
        ("helper_pipeline", vec!["filter", "map"]),
        ("count_filtered", vec!["filter", "count"]),
        ("safe_read", vec!["map", "scalar"]),
    ] {
        assert_eq!(ops(&e, name), expected, "{name}");
    }
    let ordered = e.manifest["functions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["name"] == "ordered")
        .unwrap();
    let last = ordered["graph"]["nodes"]
        .as_array()
        .unwrap()
        .last()
        .unwrap();
    let code = &e.shaders[last["shader"].as_str().unwrap()];
    assert!(code.contains("cursor--"));
    assert!(code.contains("let i=cursor"));
    assert!(last["ordered"].as_bool().unwrap());
    assert_eq!(last["workgroup_size"], 1);
    // Real interpreter value establishes an order-sensitive expected result; a
    // forward accumulation is different. Actual GPU runs use the same fixtures.
    let p = parse(source).unwrap();
    let args = vec![
        eval::Value::List(vec![
            eval::Value::U32(2),
            eval::Value::U32(9),
            eval::Value::U32(3),
        ]),
        eval::Value::U32(8),
        eval::Value::U32(3),
    ];
    let result = eval::call(&p, "ordered", args, &mut 100000).unwrap().json();
    assert_eq!(result, json!(7));
    // Optional fixture export feeds the separate actual browser/native tests.
    if let Ok(path) = std::env::var("INK_GPU_GRAPH_EXPORT") {
        let dir = std::path::Path::new(&path);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec_pretty(&e.manifest).unwrap(),
        )
        .unwrap();
        for (n, s) in &e.shaders {
            std::fs::write(dir.join(n), s).unwrap();
        }
        let mut frames = Vec::new();
        for f in &p.functions {
            for profile in 0..7 {
                let mut args = Vec::new();
                for (slot, (name, t)) in f.params.iter().enumerate() {
                    args.push(match t {
   ink_lowering_gpu::syntax::Type::List(et)=>match et.as_ref(){
    ink_lowering_gpu::syntax::Type::U32=>match profile {0=>json!([]),1=>json!([3]),2=>json!([2,9,3,0,7]),3=>json!([4294967295u32,0,2147483648u32,7]),4=>if slot==0{json!([5,1,3,9])}else{json!([2,8])},5=>json!([0,1,2,3]),_=>json!([0,4294967295u32])},
    ink_lowering_gpu::syntax::Type::I32=>match profile {0=>json!([]),1=>json!([-3]),3=>json!([-2147483648i64,2147483647,0,-1]),4=>if slot==0{json!([4,-3,2])}else{json!([-7])},_=>json!([3,-5,8])},
    ink_lowering_gpu::syntax::Type::F32=>match profile {0=>json!([]),1=>json!([-0.0]),5=>json!([16777216.0,1.0,-16777216.0]),6=>json!([{"F32Bits":2143289345u32},{"F32Bits":2139095040u32},{"F32Bits":2147483648u32}]),_=>json!([3.5,-2.0,0.5])},_=>unreachable!()},
   ink_lowering_gpu::syntax::Type::U32=>if profile==3{json!(0)}else if name=="k"{json!(3)}else{json!(8)},
   ink_lowering_gpu::syntax::Type::F32=>json!(1.0),_=>unreachable!()
  });
                }
                let typed = args
                    .iter()
                    .zip(&f.params)
                    .map(|(v, (_, t))| eval::Value::from_program_json(v, t, &p).unwrap())
                    .collect();
                let value = eval::call(&p, &f.name, typed, &mut 100000).unwrap();
                frames.push(json!({"function":f.name,"profile":profile,"args":args,"expected":value.json(),"expect_gpu":profile!=6 || !f.params.iter().any(|(_,t)|matches!(t,ink_lowering_gpu::syntax::Type::List(e) if **e==ink_lowering_gpu::syntax::Type::F32)),"source":"semantic interpreter, complete outputs"}));
            }
        }
        std::fs::write(
            dir.join("oracle-frames.json"),
            serde_json::to_vec_pretty(&frames).unwrap(),
        )
        .unwrap();
    }
}
#[test]
fn lazy_collection_branches_are_explicit_fallback_and_scalar_choices_stay_lazy() {
    let e=emit("module Lazy;fn lazy(xs:List<u32>,c:Bool)->u32{return choose(c,sum(xs.map(fn(x)=>x*7)),9);} fn scalar(xs:List<u32>,k:u32)->List<u32>{return xs.map(fn(x)=>choose(x<k,quot_or(x,0,7),x+1)).map(fn(x)=>x);}");
    assert!(e.manifest["functions"][0]["graph"].is_null());
    assert!(e.manifest["functions"][0]["fallback_reason"]
        .as_str()
        .unwrap()
        .contains("lazy collection"));
    let g = &e.manifest["functions"][1]["graph"];
    let code = &e.shaders[g["nodes"][0]["shader"].as_str().unwrap()];
    assert!(code.contains("if ("));
    assert!(code.contains("else"));
}
#[test]
fn sharing_and_capture_bounds_do_not_change_variable_scopes() {
    // Construct an expression Let because source statements don't expose sibling
    // expression Lets. The left shadow is only in scope in its own body.
    let mut p=parse("module Scope;fn scope(x:u32)->u32{return x+x;} fn unused(xs:List<u32>,a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32)->List<u32>{return xs.map(fn(x)=>x).map(fn(x)=>x+1);}").unwrap();
    use ink_lowering_gpu::syntax::Expr;
    p.functions[0].body = Expr::Binary(
        "+".into(),
        Box::new(Expr::Let(
            "x".into(),
            Some(Box::new(ink_lowering_gpu::syntax::Type::U32)),
            Box::new(Expr::Num(7)),
            Box::new(Expr::Var("x".into())),
        )),
        Box::new(Expr::Var("x".into())),
    );
    let e = compute::lower(&CheckedModule::from_source(p.clone()).unwrap()).unwrap();
    let r = eval::call(&p, "scope", vec![eval::Value::U32(11)], &mut 10000)
        .unwrap()
        .json();
    assert_eq!(r, json!(18));
    let scope = &e.manifest["functions"][0]["graph"];
    let last = scope["nodes"].as_array().unwrap().last().unwrap();
    assert!(last["args"]
        .as_array()
        .unwrap()
        .iter()
        .any(|r| r == &json!({"param":0})));
    assert!(last["args"]
        .as_array()
        .unwrap()
        .iter()
        .any(|r| r.get("node").is_some()));
    for n in e.manifest["functions"][1]["graph"]["nodes"]
        .as_array()
        .unwrap()
    {
        assert_eq!(n["args"].as_array().unwrap().len(), 1);
    }
}
#[test]
fn unsupported_width_emits_no_partial_graph_and_sort_remains_literal() {
    let e=emit("module Unsupported;fn sort(xs:List<u32>)->List<u32>{return xs.sort().map(fn(x)=>x+1);}fn wide(xs:List<u64>)->List<u64>{return xs.map(fn(x)=>x+1);}");
    let sort = &e.manifest["functions"][0];
    assert!(sort["graph"].is_object()); // Literal ordered sort has a graph stage.
    assert!(sort["graph"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|n| n["op"] == "sort"));
    assert!(sort["gpu_max_elements"].is_null());
    assert_eq!(e.manifest["gpu_list_abi"], "ink-gpu-list-v3");
    let wide = &e.manifest["functions"][1];
    assert!(wide["graph"].is_null());
    assert!(wide["shader"].is_null());
    assert!(wide["fallback_reason"].is_string());
    let supported=emit("module Unsupported;fn sort(xs:List<u32>)->List<u32>{return xs.sort().map(fn(x)=>x+1);}");
    assert_eq!(e.shaders,supported.shaders); // The rejected wide path adds no artifacts.
}

#[test]
fn standalone_ordered_and_aggregate_contexts_select_packed_graph_abi() {
    let fold=CheckedModule::from_source(parse("module Plain;fn ordered(xs:List<u32>,z:u32)->u32{return foldr(xs,z,fn(x)=>fn(a)=>x-a);}").unwrap()).unwrap();
    assert!(!ink_lowering_gpu::compute_values::needed(
        fold.pure_program().unwrap()
    ));
    assert!(compute::needed(fold.pure_program().unwrap()));
    let e = ink_lowering_gpu::gpu::lower(&fold).unwrap();
    assert_eq!(e.manifest["schema"], 2);
    assert_eq!(e.manifest["functions"][0]["graph_required"], true);
    let contextual = CheckedModule::from_source(
        parse(
            "module Context;fn total(xs:List<u32>,k:u32)->u32{return sum(xs.map(fn(x)=>x*k))+k;}",
        )
        .unwrap(),
    )
    .unwrap();
    assert!(compute::needed(contextual.pure_program().unwrap()));
    assert_eq!(
        ink_lowering_gpu::gpu::lower(&contextual).unwrap().manifest["schema"],
        2
    );
    let legacy = CheckedModule::from_source(
        parse("module Legacy;fn total(xs:List<u32>)->u32{return sum(xs.map(fn(x)=>x*3));}")
            .unwrap(),
    )
    .unwrap();
    assert!(!compute::needed(legacy.pure_program().unwrap()));
    assert_eq!(
        ink_lowering_gpu::gpu::lower(&legacy).unwrap().manifest["schema"],
        1
    );
}
