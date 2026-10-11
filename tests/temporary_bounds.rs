use ink_lowering_gpu::{compute, core::CheckedModule, syntax::parse};
fn eligible(body: &str) -> (bool, String) {
    let source = format!(
        "module Bounds;fn work(xs:List<u32>,at:u32)->List<u32>{{return xs.map(fn(x)=>{body});}}"
    );
    let module = CheckedModule::from_source(parse(&source).unwrap()).unwrap();
    let emission = compute::lower(&module).unwrap();
    let f = &emission.manifest["functions"][0];
    (
        f["shader"].is_string() || f["graph"].is_object(),
        f["fallback_reason"].to_string(),
    )
}
#[test]
fn split_provenance_allows_bounded_rejoin_and_rejects_growth() {
    for body in [
        "buffer_fill(64,x).split_at(at,fn(a)=>fn(b)=>a.map_parallel(1,fn(i)=>fn(v)=>v+i).append(b)).at_or(63,7)",
        "buffer_fill(64,x).split_at(at,fn(a)=>fn(b)=>b.append(a)).len()",
        "buffer_fill(64,x).split_at(at,fn(a)=>fn(b)=>a.split_at(at,fn(c)=>fn(d)=>c.append(d)).append(b)).len()",
        "buffer_fill(64,x).split_at(at,fn(a)=>fn(b)=>repeat(2,a,fn(i)=>fn(v)=>v.pop()).append(b)).len()",
        "buffer_fill(64,x).split_at(at,fn(a)=>fn(b)=>choose(x>0,a.pop(),a).append(b)).len()",
        "buffer_fill(64,x).split_at(at,fn(a)=>fn(b)=>a.append(buffer_fill(0,x)).append(b)).len()",
        "buffer_fill(0,x).push(x).append(buffer_fill(63,x)).len()",
    ] {
        let (ok, reason) = eligible(body);
        assert!(ok, "{body}: {reason}");
    }
    for (body, reason) in [
        ("buffer_fill(64,x).append(buffer_fill(1,x)).len()", "append capacity"),
        ("buffer_fill(64,x).push(x).len()", "push capacity"),
        ("repeat(100,buffer_fill(0,x),fn(i)=>fn(b)=>b.push(x)).len()", "loop may grow"),
        ("repeat_until(100,buffer_fill(0,x),fn(b)=>b.len()>70,fn(i)=>fn(b)=>b.push(x)).len()", "loop may grow"),
        ("buffer_fill(64,x).split_at(at,fn(a)=>fn(b)=>repeat(2,a,fn(i)=>fn(v)=>buffer_fill(64,x)).append(b)).len()", "loop may grow"),
        ("buffer_fill(64,x).split_at(at,fn(a)=>fn(b)=>choose(x>0,a,buffer_fill(64,x)).append(b)).len()", "append capacity"),
    ] {
        let (ok, fallback) = eligible(body);
        assert!(!ok, "unsafe bound admitted: {body}");
        assert!(fallback.contains(reason), "{body}: {fallback}");
    }
}
