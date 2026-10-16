use ink_lowering_gpu::{compute, core::CheckedModule, eval, syntax::parse};
use serde_json::{json, Value};

fn emission(source: &str) -> ink_lowering_gpu::gpu::Emission {
    compute::lower(&CheckedModule::from_source(parse(source).unwrap()).unwrap()).unwrap()
}

fn function<'a>(e: &'a ink_lowering_gpu::gpu::Emission, name: &str) -> &'a Value {
    e.manifest["functions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["name"] == name)
        .unwrap()
}

#[test]
fn balanced_scalar_calls_keep_one_stage_and_valid_exact_word_shaders() {
    let mut source = String::from("module ScalarStatements;record Coeff{scale:f32,bias:f32,}fn leaf(w:Coeff,x:f32)->f32{let a=x*w.scale;let b=a+w.bias;return choose(b<0.0,-b,b);}fn word(x:u64)->u64{return (x*18446744073709551615)+7;}fn signed(x:i32)->i32{return (x*2147483647)+1;}fn scoped(x:u32)->u32{let y=quot_or(x,0,7);return choose(x==0,y,x+y);} ");
    let mut previous = "leaf".to_string();
    for index in 0..8 {
        let name = format!("level_{index}");
        source.push_str(&format!(
            "fn {name}(w:Coeff,x:f32)->f32{{return {previous}(w,{previous}(w,x));}}"
        ));
        previous = name;
    }
    let e = emission(&source);
    let root = function(&e, &previous);
    assert!(root["graph"].is_object(), "{}", root["fallback_reason"]);
    let nodes = root["graph"]["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0]["op"], "scalar");
    assert_eq!(nodes[0]["capacity"]["kind"], "one");
    for (name, code) in &e.shaders {
        let module = naga::front::wgsl::parse_str(code)
            .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(code)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{name}: {error:?}"));
    }
    let p = parse(&source).unwrap();
    for (name, args, expected) in [
        ("word", vec![eval::Value::U64(u64::MAX)], json!(8)),
        (
            "signed",
            vec![eval::Value::I32(i32::MIN)],
            json!(-2147483647),
        ),
        ("scoped", vec![eval::Value::U32(0)], json!(7)),
        ("scoped", vec![eval::Value::U32(u32::MAX)], json!(6)),
    ] {
        assert_eq!(
            eval::call(&p, name, args, &mut 100000).unwrap().json(),
            expected
        );
        assert_eq!(
            function(&e, name)["graph"]["nodes"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }
}

#[test]
fn transitive_collection_callees_retain_eager_stages_and_lazy_fallback() {
    let e = emission("module CollectionBoundaries;fn inner(xs:List<u32>)->u32{return sum(xs.map(fn(x)=>x+1));}fn middle(xs:List<u32>)->u32{return inner(xs);}fn outer(xs:List<u32>)->u32{return middle(xs)+7;}fn guarded(xs:List<u32>,c:Bool)->u32{return choose(c,inner(xs),9);}");
    let outer = function(&e, "outer");
    let ops: Vec<_> = outer["graph"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["op"].as_str().unwrap())
        .collect();
    assert_eq!(ops, ["map", "sum", "scalar"]);
    assert!(function(&e, "guarded")["graph"].is_null());
}
