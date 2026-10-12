//! Parallel multi-pass schedules for graph stages that were previously run by
//! a single GPU thread: inclusive integer scan, stable filter (stream
//! compaction), stable LSD radix sort, concatenation and exact integer sums.
//!
//! A schedule is data. Each pass names a WGSL shader, its bindings in order
//! (`{"arg":k}` node input k, read-only; `{"out":true}` the node result;
//! `{"scratch":j}` a node-private buffer) and a workgroup count expressed in
//! the node capacity `c`: `b1 = ceil(c/256)`, `b2 = ceil(b1/256)`,
//! `r1 = min(b1, 256)` or `one`. Scratch sizes are `[mult, q, extra]` words
//! with q one of `c`, `b1`, `b2` or `one`. Hosts evaluate these formulas and
//! must reject `c > 256^3` (the three-level scan bound) by falling back.
//!
//! Every pass reads logical lengths from list headers, so the same buffers
//! serve any length up to capacity across resident iterations. Integer scans
//! and sums wrap exactly, so their results equal the ordered CPU semantics.
use serde_json::{json, Value};

/// Integer lane kinds that the parallel algorithms combine.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Kind {
    U32,
    I32,
    U64,
}
impl Kind {
    pub(super) fn lanes(self) -> usize {
        if self == Kind::U64 {
            2
        } else {
            1
        }
    }
    fn ty(self) -> &'static str {
        match self {
            Kind::U32 => "u32",
            Kind::I32 => "i32",
            Kind::U64 => "vec2<u32>",
        }
    }
    /// Helpers: zero, wrapping add, and load/store at a lane offset.
    fn ops(self) -> String {
        let t = self.ty();
        let (zero, add) = match self {
            Kind::U32 => ("0u", "a+b"),
            Kind::I32 => ("0i", "a+b"),
            Kind::U64 => ("vec2<u32>(0u,0u)", "vec2<u32>(a.x+b.x,a.y+b.y+select(0u,1u,a.x+b.x<a.x))"),
        };
        format!("fn kzero()->{t}{{return {zero};}}\nfn kadd(a:{t},b:{t})->{t}{{return {add};}}\n")
    }
    fn load(self, buf: &str, at: &str) -> String {
        match self {
            Kind::U32 => format!("{buf}[{at}]"),
            Kind::I32 => format!("bitcast<i32>({buf}[{at}])"),
            Kind::U64 => format!("vec2<u32>({buf}[{at}],{buf}[({at})+1u])"),
        }
    }
    fn store(self, buf: &str, at: &str, v: &str) -> String {
        match self {
            Kind::U32 => format!("{buf}[{at}]={v};"),
            Kind::I32 => format!("{buf}[{at}]=bitcast<u32>({v});"),
            Kind::U64 => format!("{buf}[{at}]={v}.x;{buf}[({at})+1u]={v}.y;"),
        }
    }
}

pub(super) struct Pass {
    pub name: String,
    pub code: String,
    pub bind: Vec<Value>,
    pub groups: &'static str,
}
pub(super) struct Schedule {
    pub passes: Vec<Pass>,
    pub scratch: Vec<Value>,
}
impl Schedule {
    fn scratch(&mut self, mult: usize, q: &str, extra: usize) -> usize {
        self.scratch.push(json!({"words": [mult, q, extra]}));
        self.scratch.len() - 1
    }
    pub(super) fn descriptor(&self) -> Value {
        json!({
            "passes": self.passes.iter().map(|p| json!({"shader": p.name, "bind": p.bind, "groups": p.groups})).collect::<Vec<_>>(),
            "scratch": self.scratch,
        })
    }
}
fn arg(k: usize) -> Value {
    json!({"arg": k})
}
fn out() -> Value {
    json!({"out": true})
}
fn scratch(j: usize) -> Value {
    json!({"scratch": j})
}
/// Binding declarations in order; `rw[i]` selects read_write storage.
fn decls(names: &[(&str, bool)]) -> String {
    names
        .iter()
        .enumerate()
        .map(|(i, (n, rw))| {
            format!(
                "@group(0) @binding({i}) var<storage,{}> {n}:array<u32>;\n",
                if *rw { "read_write" } else { "read" }
            )
        })
        .collect()
}
const MAIN: &str = "@compute @workgroup_size(256) fn main(@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) wid:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>)";
fn block_scan(kind: Kind) -> String {
    let t = kind.ty();
    format!(
        "var<workgroup> sh:array<{t},256>;
fn block_scan(l:u32,v:{t})->{t}{{
sh[l]=v;workgroupBarrier();
for(var off=1u;off<256u;off=off<<1u){{var x=sh[l];if(l>=off){{x=kadd(sh[l-off],x);}}workgroupBarrier();sh[l]=x;workgroupBarrier();}}
return sh[l];
}}
"
    )
}

/// Where a scanned array lives: a buffer name in the pass, its first word,
/// and whether the pass that seeds it copies from a separate source.
struct ScanTarget<'a> {
    /// Binding of the array being scanned, in the first pass.
    target: Value,
    /// Word offset of element 0 in the target.
    offset: usize,
    /// Optional separate read-only source (node input with a header).
    source: Option<(Value, usize)>,
    /// Count binding (a node input whose header gives n) and the WGSL count
    /// expression in terms of `cnt`.
    count: (Value, &'a str),
    /// Write `n` into word 0 of the target.
    header: bool,
}
/// Five passes: block scans at three levels (the top level is one group,
/// so capacity is bounded by 256^3) and two passes adding block prefixes.
fn scan(s: &mut Schedule, tag: &str, kind: Kind, t: ScanTarget) {
    let l = kind.lanes();
    let s1 = s.scratch(l, "b1", 0);
    let s2 = s.scratch(l, "b2", 0);
    let common = format!("{}{}", kind.ops(), block_scan(kind));
    let (cnt_ref, count) = (t.count.0.clone(), t.count.1);
    let at = |i: &str| format!("{}u+({i})*{l}u", t.offset);
    // P0: seed block scans from the source (or in place).
    let (bindings, src, src_at): (Vec<(&str, bool)>, &str, String) = match &t.source {
        Some((_, off)) => (
            vec![("cnt", false), ("src", false), ("dst", true), ("s1", true)],
            "src",
            format!("{off}u+i*{l}u"),
        ),
        None => (vec![("cnt", false), ("dst", true), ("s1", true)], "dst", at("i")),
    };
    let mut bind = vec![cnt_ref.clone()];
    if let Some((r, _)) = &t.source {
        bind.push(r.clone());
    }
    bind.push(t.target.clone());
    bind.push(scratch(s1));
    s.passes.push(Pass {
        name: format!("{tag}-scan0.wgsl"),
        code: format!(
            "{}{common}{MAIN}{{
let n={count};let i=wid.x*256u+lid.x;
{}
var v=kzero();if(i<n){{v={};}}
let r=block_scan(lid.x,v);
if(i<n){{{}}}
if(lid.x==255u){{{}}}
}}
",
            decls(&bindings),
            if t.header { "if(i==0u){dst[0]=n;}" } else { "" },
            kind.load(src, &src_at),
            kind.store("dst", &at("i"), "r"),
            kind.store("s1", &format!("wid.x*{l}u"), "r"),
        ),
        bind,
        groups: "b1",
    });
    // P1: scan block totals, emitting second-level totals.
    s.passes.push(Pass {
        name: format!("{tag}-scan1.wgsl"),
        code: format!(
            "{}{common}{MAIN}{{
let n={count};let m=(n+255u)/256u;let i=wid.x*256u+lid.x;
var v=kzero();if(i<m){{v={};}}
let r=block_scan(lid.x,v);
if(i<m){{{}}}
if(lid.x==255u){{{}}}
}}
",
            decls(&[("cnt", false), ("s1", true), ("s2", true)]),
            kind.load("s1", &format!("i*{l}u")),
            kind.store("s1", &format!("i*{l}u"), "r"),
            kind.store("s2", &format!("wid.x*{l}u"), "r"),
        ),
        bind: vec![cnt_ref.clone(), scratch(s1), scratch(s2)],
        groups: "b2",
    });
    // P2: one group scans every second-level total (at most 256).
    s.passes.push(Pass {
        name: format!("{tag}-scan2.wgsl"),
        code: format!(
            "{}{common}{MAIN}{{
let n={count};let m=((n+255u)/256u+255u)/256u;let i=lid.x;
var v=kzero();if(i<m){{v={};}}
let r=block_scan(lid.x,v);
if(i<m){{{}}}
}}
",
            decls(&[("cnt", false), ("s2", true)]),
            kind.load("s2", &format!("i*{l}u")),
            kind.store("s2", &format!("i*{l}u"), "r"),
        ),
        bind: vec![cnt_ref.clone(), scratch(s2)],
        groups: "one",
    });
    // Q1, Q0: add each block's exclusive prefix.
    s.passes.push(Pass {
        name: format!("{tag}-scan3.wgsl"),
        code: format!(
            "{}{}{MAIN}{{
let n={count};let m=(n+255u)/256u;let i=wid.x*256u+lid.x;
if(i<m&&wid.x>0u){{let v=kadd({},{});{}}}
}}
",
            decls(&[("cnt", false), ("s1", true), ("s2", true)]),
            kind.ops(),
            kind.load("s2", &format!("(wid.x-1u)*{l}u")),
            kind.load("s1", &format!("i*{l}u")),
            kind.store("s1", &format!("i*{l}u"), "v"),
        ),
        bind: vec![cnt_ref.clone(), scratch(s1), scratch(s2)],
        groups: "b2",
    });
    s.passes.push(Pass {
        name: format!("{tag}-scan4.wgsl"),
        code: format!(
            "{}{}{MAIN}{{
let n={count};let i=wid.x*256u+lid.x;
if(i<n&&wid.x>0u){{let v=kadd({},{});{}}}
}}
",
            decls(&[("cnt", false), ("dst", true), ("s1", true)]),
            kind.ops(),
            kind.load("s1", &format!("(wid.x-1u)*{l}u")),
            kind.load("dst", &at("i")),
            kind.store("dst", &at("i"), "v"),
        ),
        bind: vec![cnt_ref, t.target, scratch(s1)],
        groups: "b1",
    });
}

/// Inclusive wrapping prefix sum of an integer list (node input `src`).
pub(super) fn scan_list(tag: &str, kind: Kind, src: usize) -> Schedule {
    let mut s = Schedule { passes: vec![], scratch: vec![] };
    scan(
        &mut s,
        tag,
        kind,
        ScanTarget {
            target: out(),
            offset: 1,
            source: Some((arg(src), 1)),
            count: (arg(src), "cnt[0]"),
            header: true,
        },
    );
    s
}

/// Stable filter: flags from the predicate shader (written by the caller's
/// first pass), an in-place scan of the flags, then a scatter of kept
/// elements. `width` is the element width in lanes.
pub(super) fn filter_list(tag: &str, predicate: Pass, src: usize, width: usize) -> Schedule {
    let mut s = Schedule { passes: vec![], scratch: vec![] };
    let flags = s.scratch(1, "c", 0);
    let mut predicate = predicate;
    predicate.bind.push(scratch(flags));
    s.passes.push(predicate);
    scan(
        &mut s,
        tag,
        Kind::U32,
        ScanTarget {
            target: scratch(flags),
            offset: 0,
            source: None,
            count: (arg(src), "cnt[0]"),
            header: false,
        },
    );
    s.passes.push(Pass {
        name: format!("{tag}-scatter.wgsl"),
        code: format!(
            "{}{MAIN}{{
let n=src[0];let i=wid.x*256u+lid.x;
if(i==0u){{if(n==0u){{out[0]=0u;}}else{{out[0]=flags[n-1u];}}}}
if(i<n){{
let before=select(0u,flags[i-1u],i>0u);
if(flags[i]!=before){{
let d=flags[i]-1u;
for(var k=0u;k<{width}u;k++){{out[1u+d*{width}u+k]=src[1u+i*{width}u+k];}}
}}
}}
}}
",
            decls(&[("src", false), ("flags", true), ("out", true)]),
        ),
        bind: vec![arg(src), scratch(flags), out()],
        groups: "b1",
    });
    s
}

/// Stable LSD radix sort of an integer list, eight bits per pass. Signed
/// keys flip the top bit of their most significant digit.
pub(super) fn sort_list(tag: &str, kind: Kind, src: usize) -> Schedule {
    let mut s = Schedule { passes: vec![], scratch: vec![] };
    let l = kind.lanes();
    let tmp = s.scratch(l, "c", 0);
    let hist = s.scratch(256, "b1", 0);
    s.passes.push(Pass {
        name: format!("{tag}-copy.wgsl"),
        code: format!(
            "{}{MAIN}{{
let n=src[0];let i=wid.x*256u+lid.x;
if(i==0u){{out[0]=n;}}
if(i<n){{for(var k=0u;k<{l}u;k++){{out[1u+i*{l}u+k]=src[1u+i*{l}u+k];}}}}
}}
",
            decls(&[("src", false), ("out", true)])
        ),
        bind: vec![arg(src), out()],
        groups: "b1",
    });
    let digits = 4 * l;
    for d in 0..digits {
        // Even passes read the result buffer (header at word 0), odd passes
        // the scratch copy; an even number of passes ends in the result.
        let (from, from_off, to, to_off) = if d % 2 == 0 {
            (out(), 1, scratch(tmp), 0)
        } else {
            (scratch(tmp), 0, out(), 1)
        };
        let lane = if d < 4 { 0 } else { 1 };
        let shift = 8 * (d % 4);
        let flip = if kind == Kind::I32 && d == 3 { "^128u" } else { "" };
        let digit = format!("((sortsrc[{from_off}u+i*{l}u+{lane}u]>>{shift}u)&255u){flip}");
        s.passes.push(Pass {
            name: format!("{tag}-hist{d}.wgsl"),
            code: format!(
                "{}var<workgroup> counts:array<atomic<u32>,256>;
{MAIN}{{
let n=cnt[0];let nb=(n+255u)/256u;let i=wid.x*256u+lid.x;
atomicStore(&counts[lid.x],0u);
workgroupBarrier();
if(i<n){{atomicAdd(&counts[{digit}],1u);}}
workgroupBarrier();
if(wid.x<nb){{hist[lid.x*nb+wid.x]=atomicLoad(&counts[lid.x]);}}
}}
",
                decls(&[("cnt", false), ("sortsrc", true), ("hist", true)])
            ),
            bind: vec![arg(src), from.clone(), scratch(hist)],
            groups: "b1",
        });
        scan(
            &mut s,
            &format!("{tag}-d{d}"),
            Kind::U32,
            ScanTarget {
                target: scratch(hist),
                offset: 0,
                source: None,
                count: (arg(src), "256u*((cnt[0]+255u)/256u)"),
                header: false,
            },
        );
        s.passes.push(Pass {
            name: format!("{tag}-place{d}.wgsl"),
            code: format!(
                "{}var<workgroup> keys:array<u32,256>;
{MAIN}{{
let n=cnt[0];let nb=(n+255u)/256u;let i=wid.x*256u+lid.x;
var mine=256u;if(i<n){{mine={digit};}}
keys[lid.x]=mine;
workgroupBarrier();
if(i<n){{
var rank=0u;var total=0u;
for(var j=0u;j<256u;j++){{if(keys[j]==mine){{total++;if(j<lid.x){{rank++;}}}}}}
let d=hist[mine*nb+wid.x]-total+rank;
for(var k=0u;k<{l}u;k++){{sortdst[{to_off}u+d*{l}u+k]=sortsrc[{from_off}u+i*{l}u+k];}}
}}
}}
",
                decls(&[("cnt", false), ("sortsrc", true), ("sortdst", true), ("hist", true)])
            ),
            bind: vec![arg(src), from, to, scratch(hist)],
            groups: "b1",
        });
    }
    s
}

/// Parallel concatenation of two lists of `width`-lane elements.
pub(super) fn concat_lists(tag: &str, a: usize, b: usize, width: usize) -> Schedule {
    Schedule {
        passes: vec![Pass {
            name: format!("{tag}-concat.wgsl"),
            code: format!(
                "{}{MAIN}{{
let n1=a[0];let n2=b[0];let i=wid.x*256u+lid.x;
if(i==0u){{out[0]=n1+n2;}}
if(i<n1){{for(var k=0u;k<{width}u;k++){{out[1u+i*{width}u+k]=a[1u+i*{width}u+k];}}}}
else if(i<n1+n2){{let j=i-n1;for(var k=0u;k<{width}u;k++){{out[1u+i*{width}u+k]=b[1u+j*{width}u+k];}}}}
}}
",
                decls(&[("a", false), ("b", false), ("out", true)])
            ),
            bind: vec![arg(a), arg(b), out()],
            groups: "b1",
        }],
        scratch: vec![],
    }
}

/// Exact wrapping integer sum: grid-stride partials in at most 256 groups,
/// then one group combines them. The scalar result has no header.
pub(super) fn sum_list(tag: &str, kind: Kind, src: usize) -> Schedule {
    let mut s = Schedule { passes: vec![], scratch: vec![] };
    let l = kind.lanes();
    let part = s.scratch(l, "r1", 1);
    let t = kind.ty();
    let tree = format!(
        "var<workgroup> sh:array<{t},256>;
fn tree(l:u32,v:{t})->{t}{{
sh[l]=v;workgroupBarrier();
for(var s=128u;s>0u;s=s>>1u){{if(l<s){{sh[l]=kadd(sh[l],sh[l+s]);}}workgroupBarrier();}}
return sh[0];
}}
"
    );
    s.passes.push(Pass {
        name: format!("{tag}-sum0.wgsl"),
        code: format!(
            "{}{}{tree}{MAIN}{{
let n=src[0];let total=nwg.x*256u;
var acc=kzero();
for(var e=wid.x*256u+lid.x;e<n;e+=total){{acc=kadd(acc,{});}}
let r=tree(lid.x,acc);
if(lid.x==0u){{{}if(wid.x==0u){{part[0]=nwg.x;}}}}
}}
",
            decls(&[("src", false), ("part", true)]),
            kind.ops(),
            kind.load("src", &format!("1u+e*{l}u")),
            kind.store("part", &format!("1u+wid.x*{l}u"), "r"),
        ),
        bind: vec![arg(src), scratch(part)],
        groups: "r1",
    });
    s.passes.push(Pass {
        name: format!("{tag}-sum1.wgsl"),
        code: format!(
            "{}{}{tree}{MAIN}{{
let g=part[0];
var acc=kzero();
for(var e=lid.x;e<g;e+=256u){{acc=kadd(acc,{});}}
let r=tree(lid.x,acc);
if(lid.x==0u){{{}}}
}}
",
            decls(&[("part", true), ("out", true)]),
            kind.ops(),
            kind.load("part", &format!("1u+e*{l}u")),
            kind.store("out", "0u", "r"),
        ),
        bind: vec![scratch(part), out()],
        groups: "one",
    });
    s
}
