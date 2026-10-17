//! WGSL pure compute lowering. Logical values have a versioned packed ABI;
//! shader eligibility does not assert or select an optimisation law.
use crate::{check, core, syntax::*, LangResult};
/// Target-owned packed ABI classification for literal ordered stage graphs.
/// The semantic core's value classification stays unchanged.
pub fn needed(p: &Program) -> bool {
    fn full_width(t: &Type) -> bool {
        match t {
            Type::U64 => true,
            Type::List(t) | Type::Option(t) => full_width(t),
            _ => false,
        }
    }
    fn aggregate(e: &Expr) -> bool {
        match e {
            Expr::Call(n, a) => {
                ["sum", "count", "foldr", "foldl", "fold_until"].contains(&n.as_str())
                    || a.iter().any(aggregate)
            }
            Expr::Binary(_, a, b) | Expr::Let(_, _, a, b) => aggregate(a) || aggregate(b),
            Expr::Method(x, _, a) => aggregate(x) || a.iter().any(aggregate),
            Expr::Lambda(_, x) | Expr::Neg(x) | Expr::Field(x, _) => aggregate(x),
            Expr::Record(_, fs) => fs.iter().any(|(_, x)| aggregate(x)),
            _ => false,
        }
    }
    fn ordered(e: &Expr) -> bool {
        match e {
            Expr::Call(n, a) => (n == "foldr" || n == "foldl" || n == "fold_until") || a.iter().any(ordered),
            Expr::Binary(_, a, b) | Expr::Let(_, _, a, b) => ordered(a) || ordered(b),
            Expr::Method(x, _, a) => ordered(x) || a.iter().any(ordered),
            Expr::Lambda(_, x) | Expr::Neg(x) | Expr::Field(x, _) => ordered(x),
            Expr::Record(_, fs) => fs.iter().any(|(_, x)| ordered(x)),
            _ => false,
        }
    }
    ink_core::recursion::needed(p)
        || p.functions
            .iter()
            .any(|f| full_width(&f.result) || f.params.iter().any(|(_, t)| full_width(t)))
        || ink_core::local_collections::needed(p)
        || ink_core::compute_values::needed(p)
        || p.functions.iter().any(|f| {
            ordered(&f.body)
                || (aggregate(&f.body)
                    && !matches!(&f.body,Expr::Call(n,_) if n=="sum" || n=="count"))
        })
}
use serde_json::{json, Value};
use std::collections::BTreeMap;
type Env = BTreeMap<String, (String, Type)>;
#[path = "compute_graph.rs"]
mod graph;
#[path = "parallel.rs"]
mod parallel;

fn wg(t: &Type) -> LangResult<String> {
    Ok(match t {
        Type::U32 => "u32".into(),
        Type::U64 => "InkU64".into(),
        Type::I32 => "i32".into(),
        Type::F32 => "u32".into(),
        Type::F64 => return Err("binary64 device arithmetic is unsupported; retain the CPU implementation".into()),
        Type::Bool => "bool".into(),
        Type::Vector(t, n) => format!("vec{n}<{}>", wg(t)?),
        Type::Named(n) => format!("R_{n}"),
        Type::Buffer(inner)
            if matches!(
                inner.as_ref(),
                Type::U32 | Type::U64 | Type::I32 | Type::F32 | Type::Bool
            ) =>
        {
            format!("InkBuffer_{}", wg(inner)?)
        }
        _ => {
            return Err("GPU scalar requires a 32-bit numeric, Bool, vector or flat record".into())
        }
    })
}
fn desc(t: &Type, p: &Program) -> LangResult<Value> {
    ink_abi::describe(t, &p.records, &p.enums, &p.ids)
}
fn lanes(t: &Type, p: &Program, _gpu: bool) -> LangResult<usize> {
    ink_abi::flat_width(&ink_abi::describe(t, &p.records, &p.enums, &p.ids)?)
}

// Function-private arrays use WGSL alignment, not the packed transport ABI.
// Count vec3/record tail padding in the local allocation admission bound.
fn local_layout(t: &Type, p: &Program, depth: usize, budget: &mut usize) -> LangResult<(usize, usize)> {
    if depth >= 48 || *budget == 0 {
        return Err("GPU local list layout budget exceeded; CPU fallback".into());
    }
    *budget -= 1;
    match t {
        Type::U32 | Type::I32 | Type::F32 | Type::Bool => Ok((1, 1)),
        Type::U64 => Ok((1, 2)),
        Type::Vector(inner, n) if matches!(inner.as_ref(), Type::U32 | Type::I32 | Type::F32 | Type::Bool) =>
            Ok((if *n == 2 { 2 } else { 4 }, *n as usize)),
        Type::Named(name) => {
            let mut alignment = 1;
            let mut size = 0usize;
            for (_, t) in p.records.get(name).ok_or("GPU local list record layout")? {
                let (a, s) = local_layout(t, p, depth + 1, budget)?;
                size = size.checked_add(a - 1).map(|n| n / a * a)
                    .and_then(|n| n.checked_add(s)).ok_or("GPU local list layout overflow")?;
                alignment = alignment.max(a);
            }
            let size = size.checked_add(alignment - 1).map(|n| n / alignment * alignment)
                .ok_or("GPU local list layout overflow")?;
            Ok((alignment, size))
        }
        _ => Err("GPU local list element layout is unsupported; CPU fallback".into()),
    }
}

// Upper length bounds follow owners through literal emission. Complementary
// partitions retain their common origin, so rejoining even dynamically split
// halves stays bounded by the original length. This is emission eligibility,
// not optimisation or proof authority.
#[derive(Clone, Debug, PartialEq, Eq)]
enum BufferBound {
    Fixed(u32),
    Partition(Box<BufferBound>, usize, bool),
}
impl BufferBound {
    fn upper(&self) -> u32 {
        match self {
            Self::Fixed(n) => *n,
            Self::Partition(parent, _, _) => parent.upper(),
        }
    }
    fn append(&self, other: &Self) -> Self {
        if *other == Self::Fixed(0) {
            return self.clone();
        }
        if *self == Self::Fixed(0) {
            return other.clone();
        }
        if let (Self::Partition(a, id, side), Self::Partition(b, other_id, other_side)) =
            (self, other)
        {
            if id == other_id && side != other_side && a == b {
                return *a.clone();
            }
        }
        Self::Fixed(self.upper() + other.upper())
    }
}
struct Emitter<'a> {
    p: &'a Program,
    lines: String,
    next: usize,
    budget: usize,
    depth: usize,
    views: Vec<(Expr, Env)>,
    local_lists: BTreeMap<String, (String, usize)>,
    local_list_words: usize,
    buffers: BTreeMap<String, BufferBound>,
}
impl Emitter<'_> {
    fn fresh(&mut self) -> String {
        let n = format!("t{}", self.next);
        self.next += 1;
        n
    }
    fn line(&mut self, s: impl AsRef<str>) {
        self.lines.push_str(s.as_ref());
        self.lines.push('\n');
    }
    fn types(env: &Env) -> check::Env {
        env.iter()
            .map(|(n, (_, t))| (n.clone(), t.clone()))
            .collect()
    }
    fn ty(&self, t: &Type) -> LangResult<String> {
        {
            wg(t)
        }
    }
    fn bind(&mut self, t: &Type, v: &str) -> LangResult<String> {
        let n = self.fresh();
        let ty = self.ty(t)?;
        self.line(format!("var {n}:{ty}={v};"));
        if matches!(t, Type::Buffer(_)) {
            let bound = self.buffer_bound(v)?;
            self.buffers.insert(n.clone(), bound);
        }

        Ok(n)
    }
    fn buffer_bound(&self, value: &str) -> LangResult<BufferBound> {
        self.buffers
            .get(value)
            .cloned()
            .ok_or("GPU temporary buffer has no static bound".into())
    }
    fn loop_buffer_bound(
        &mut self,
        initial: Option<BufferBound>,
        out: &str,
        step: &str,
    ) -> LangResult<()> {
        if let Some(initial) = initial {
            let step = self.buffer_bound(step)?;
            // Maintaining a partition bound requires preserving its provenance;
            // an independently filled buffer of the same maximum can be longer
            // than the actual partition. Fixed owners may shrink freely.
            if initial != step && !matches!(initial, BufferBound::Fixed(n) if step.upper() <= n) {
                return Err("GPU temporary loop may grow its buffer bound; CPU fallback".into());
            }
            self.buffers.insert(out.into(), initial);
        }
        Ok(())
    }
    fn add(&self, op: &str, a: &str, b: &str, t: &Type) -> String {
        if let Type::Vector(inner, n) = t {
            if **inner == Type::F32 {
                return format!(
                    "vec{n}<u32>({})",
                    ["x", "y", "z", "w"]
                        .iter()
                        .take(*n as usize)
                        .map(|field| self.add(
                            op,
                            &format!("({a}).{field}"),
                            &format!("({b}).{field}"),
                            inner
                        ))
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
        }
        if *t == Type::F32 {
            return match op {
                "+" => format!("ink_f32_add({a},{b})"),
                "-" => format!("ink_f32_sub({a},{b})"),
                "*" => format!("ink_f32_mul({a},{b})"),
                "==" => format!("ink_f32_eq({a},{b})"),
                "!=" => format!("!ink_f32_eq({a},{b})"),
                "<" => format!("ink_f32_lt({a},{b})"),
                "<=" => format!("(ink_f32_lt({a},{b})||ink_f32_eq({a},{b}))"),
                ">" => format!("ink_f32_lt({b},{a})"),
                ">=" => format!("(ink_f32_lt({b},{a})||ink_f32_eq({a},{b}))"),
                _ => unreachable!("checked f32 operation"),
            };
        }
        if *t == Type::U64 {
            return match op {
                "+" => format!("ink_u64_add({a},{b})"),
                "-" => format!("ink_u64_sub({a},{b})"),
                "*" => format!("ink_u64_mul({a},{b})"),
                "==" => format!("ink_u64_eq({a},{b})"),
                "!=" => format!("!ink_u64_eq({a},{b})"),
                "<" => format!("ink_u64_lt({a},{b})"),
                "<=" => format!("!ink_u64_lt({b},{a})"),
                ">" => format!("ink_u64_lt({b},{a})"),
                ">=" => format!("!ink_u64_lt({a},{b})"),
                _ => unreachable!("checked u64 operation"),
            };
        }
        if *t == Type::I32 && ["+", "-", "*"].contains(&op) {
            {
                format!("bitcast<i32>(bitcast<u32>({a}) {op} bitcast<u32>({b}))")
            }
        } else {
            format!("({a} {op} {b})")
        }
    }
    fn scalar(&mut self, e: &Expr, env: &Env, t: &Type) -> LangResult<String> {
        if self.depth >= 48 {
            return Err(
                "scalar lowering depth budget exceeded; use shallower helpers for GPU execution"
                    .into(),
            );
        }
        self.depth += 1;
        let result = self.scalar_inner(e, env, t);
        self.depth -= 1;
        result
    }
    fn scalar_inner(&mut self, e: &Expr, env: &Env, t: &Type) -> LangResult<String> {
        if self.budget == 0 {
            return Err("compute expression budget exceeded".into());
        }
        self.budget -= 1;
        check::infer_as(e, &Self::types(env), self.p, t)?;
        match e {
            Expr::Num(n) => Ok(match t {
                Type::F32 => {
                    format!("{}u", (*n as f32).to_bits())
                }
                Type::I32 => format!("{n}{}", { "i" }),
                Type::U64 => format!("InkU64({}u,{}u)", *n as u32, (*n >> 32) as u32),
                _ => format!("{n}u"),
            }),
            Expr::Float(bits) => Ok(format!("{bits}u")),
            Expr::Bool(b) => Ok(b.to_string()),
            Expr::Var(n) => env
                .get(n)
                .map(|(v, _)| v.clone())
                .ok_or("unknown compute binding".into()),
            Expr::Neg(x) => {
                if *t == Type::I32 {
                    if let Expr::Num(n) = **x {
                        if n == 2147483648 {
                            return Ok("bitcast<i32>(2147483648u)".into());
                        }
                    }
                }
                let a = self.scalar(x, env, t)?;
                Ok(if *t == Type::I32 {
                    self.add("-", "0", &a, t)
                } else if *t == Type::U64 {
                    format!("ink_u64_sub(InkU64(0u,0u),{a})")
                } else if *t == Type::F32 {
                    format!("(({a}) ^ 0x80000000u)")
                } else if *t == Type::U32 {
                    format!("(0u - ({a}))")
                } else {
                    format!("(-({a}))")
                })
            }
            Expr::Let(n, annotation, x, b) => {
                let xt = check::infer_context(x, &Self::types(env), self.p, annotation.as_deref())?;
                let a = if matches!(xt, Type::List(_)) {
                    self.list(x, env, Some(&xt))?
                } else {
                    self.scalar(x, env, &xt)?
                };
                let v = if matches!(xt, Type::List(_)) {
                    a
                } else {
                    self.bind(&xt, &a)?
                };
                let mut local = env.clone();
                local.insert(n.clone(), (v, xt));
                self.scalar(b, &local, t)
            }
            Expr::Field(x, n) => {
                let xt = check::infer(x, &Self::types(env), self.p)?;
                let a = self.scalar(x, env, &xt)?;
                Ok(format!(
                    "({a}).{}",
                    if matches!(xt, Type::Named(_)) {
                        format!("ink_f_{n}")
                    } else {
                        n.clone()
                    }
                ))
            }
            Expr::Record(n, fields) => {
                let schema = self.p.records[n].clone();
                let mut xs = Vec::new();
                for (k, t) in &schema {
                    let e = &fields
                        .iter()
                        .find(|(f, _)| f == k)
                        .ok_or("record field missing")?
                        .1;
                    let a = self.scalar(e, env, t)?;
                    xs.push(self.bind(t, &a)?);
                }
                Ok(format!("R_{n}({})", xs.join(",")))
            }
            Expr::Binary(op, a, b) => {
                let ot = if check::is_number(t) {
                    t.clone()
                } else {
                    check::hint(a, &Self::types(env), self.p)
                        .or_else(|| check::hint(b, &Self::types(env), self.p))
                        .unwrap_or(Type::Bool)
                };
                let x = self.scalar(a, env, &ot)?;
                let x = self.bind(&ot, &x)?;
                if op == "&&" || op == "||" {
                    let out = self.bind(t, &x)?;
                    self.line(format!(
                        "if ({}{out}) {{",
                        if op == "||" { "!" } else { "" }
                    ));
                    let y = self.scalar(b, env, &ot)?;
                    self.line(format!("{out}={y};\n}}"));
                    Ok(out)
                } else {
                    let y = self.scalar(b, env, &ot)?;
                    if matches!(ot, Type::Vector(..)) && ["==", "!="].contains(&op.as_str()) {
                        return Ok(format!(
                            "{}({x} {op} {y})",
                            if op == "==" { "all" } else { "any" }
                        ));
                    }
                    Ok(self.add(op, &x, &y, &ot))
                }
            }
            Expr::Call(n,a) if ink_core::word::primitive(n) => {
                let params=ink_core::word::arguments(n,t);
                let mut values=Vec::new();
                for (arg,ty) in a.iter().zip(params) {
                    let value=self.scalar(arg,env,&ty)?;
                    values.push(self.bind(&ty,&value)?);
                }
                crate::word_emit::expression(n,&values,t)
            }
            Expr::Call(n,a) if ink_core::primitives::primitive(n) => {
                let first=if ink_core::primitives::bit_count(n){a.first().map(|a|check::infer(a,&Self::types(env),self.p)).transpose()?}else{None};
                let (params,_)=ink_core::primitives::signature(n,first.as_ref())?;
                let values=a.iter().zip(params).map(|(a,t)|self.scalar(a,env,&t)).collect::<LangResult<Vec<_>>>()?;
                crate::primitive_emit::expression(n,&values,first.as_ref())
            }
            Expr::Call(n, a) if n == "choose" => {
                let c = self.scalar(&a[0], env, &Type::Bool)?;
                let out = self.fresh();
                let ty = self.ty(t)?;
                let before = self.buffers.clone();
                self.line(format!("var {out}:{ty};"));
                self.line(format!("if ({c}) {{"));
                let x = self.scalar(&a[1], env, t)?;
                let xb = self.buffers.get(&x).cloned();
                self.line(format!("{out}={x};\n}} else {{"));
                self.buffers = before.clone();
                let y = self.scalar(&a[2], env, t)?;
                let yb = self.buffers.get(&y).cloned();
                self.line(format!("{out}={y};\n}}"));
                self.buffers = before;
                if matches!(t, Type::Buffer(_)) {
                    let x = xb.ok_or("GPU branch buffer bound")?;
                    let y = yb.ok_or("GPU branch buffer bound")?;
                    let merged = if x == y {
                        x
                    } else {
                        BufferBound::Fixed(x.upper().max(y.upper()))
                    };
                    self.buffers.insert(out.clone(), merged);
                }
                Ok(out)
            }
            Expr::Call(n, a) if ["vec2", "vec3", "vec4"].contains(&n.as_str()) => {
                let Type::Vector(et, _) = t else {
                    return Err("vector type missing".into());
                };
                let mut xs = Vec::new();
                for x in a {
                    let v = self.scalar(x, env, et)?;
                    xs.push(self.bind(et, &v)?);
                }
                Ok(format!("{}({})", wg(t)?, xs.join(",")))
            }
            Expr::Call(n, a) if n == "repeat" => {
                let Expr::Num(bound) = a[0] else {
                    return Err("literal loop bound missing".into());
                };
                let Expr::Lambda(i, inner) = &a[2] else {
                    return Err("loop lambda missing".into());
                };
                let Expr::Lambda(acc, b) = inner.as_ref() else {
                    return Err("loop accumulator missing".into());
                };
                let x = self.scalar(&a[1], env, t)?;
                let out = self.bind(t, &x)?;
                let initial_bound = self.buffers.get(&out).cloned();
                let idx = self.fresh();
                self.line(format!("for(var {idx}=0u;{idx}<{bound}u;{idx}++){{"));
                let mut local = env.clone();
                local.insert(i.clone(), (idx, Type::U32));
                local.insert(acc.clone(), (out.clone(), t.clone()));
                let v = self.scalar(b, &local, t)?;
                self.loop_buffer_bound(initial_bound, &out, &v)?;
                self.line(format!("{out}={v};\n}}"));
                Ok(out)
            }
            Expr::Call(n, a) if n == "quot_or" || n == "rem_or" => {
                let x = self.scalar(&a[0], env, t)?;
                let x = self.bind(t, &x)?;
                let y = self.scalar(&a[1], env, t)?;
                let y = self.bind(t, &y)?;
                let out = self.fresh();
                let ty = self.ty(t)?;
                self.line(format!("var {out}:{ty};"));
                let overflow = if *t == Type::I32 {
                    format!(" || ({x}=={} && {y}==-1)", { "bitcast<i32>(2147483648u)" })
                } else {
                    String::new()
                };
                let zero = if *t == Type::U64 {
                    format!("ink_u64_eq({y},InkU64(0u,0u))")
                } else {
                    format!("{y}==0{overflow}")
                };
                self.line(format!("if ({zero}) {{"));
                let fallback = self.scalar(&a[2], env, t)?;
                let division = if *t == Type::U64 {
                    format!(
                        "ink_u64_div({x},{y}).{}",
                        if n == "quot_or" {
                            "quotient"
                        } else {
                            "remainder"
                        }
                    )
                } else {
                    format!("{x} {} {y}", if n == "quot_or" { "/" } else { "%" })
                };
                self.line(format!(
                    "{out}={fallback};\n}} else {{ {out}={division}; }}"
                ));
                Ok(out)
            }
            Expr::Call(n, a) if n == "buffer_fill" => {
                let Type::Buffer(inner) = t else {
                    unreachable!()
                };
                let Expr::Num(size) = a[0] else {
                    return Err("GPU temporary buffers require literal capacity".into());
                };
                if size > 64 {
                    return Err("GPU temporary buffer capacity exceeds 64 elements".into());
                };
                let value = self.scalar(&a[1], env, inner)?;
                let value = self.bind(inner, &value)?;
                let out = self.fresh();
                let i = self.fresh();
                self.line(format!("var {out}:{};{out}.length={size}u;for(var {i}=0u;{i}<{size}u;{i}++){{{out}.data[{i}]={value};}}",wg(t)?));
                self.buffers
                    .insert(out.clone(), BufferBound::Fixed(size as u32));
                Ok(out)
            }
            Expr::Call(n, a) if n == "length" => self.length(&a[0], env),
            Expr::Call(n, a) if n == "repeat_until" => {
                let Expr::Num(limit) = a[0] else {
                    return Err("GPU stopping loops require a literal limit".into());
                };
                if limit > 65536 {
                    return Err("GPU stopping loop bound exceeds 65536".into());
                };
                let initial = self.scalar(&a[1], env, t)?;
                let out = self.bind(t, &initial)?;
                let initial_bound = self.buffers.get(&out).cloned();
                let i = self.fresh();
                self.line(format!("for(var {i}=0u;{i}<{limit}u;{i}++){{"));
                let stop = self.apply(&a[2], env, vec![(out.clone(), t.clone())], &Type::Bool)?;
                self.line(format!("if({stop}){{break;}}"));
                let step = self.apply(
                    &a[3],
                    env,
                    vec![(i, Type::U32), (out.clone(), t.clone())],
                    t,
                )?;
                self.loop_buffer_bound(initial_bound, &out, &step)?;
                self.line(format!("{out}={step};}}"));
                Ok(out)
            }
            Expr::Method(xs, n, _)
                if matches!(xs.as_ref(), Expr::Var(v) if v == "Buffer") && n == "empty" =>
            {
                let out = self.fresh();
                self.line(format!("var {out}:{};{out}.length=0u;", wg(t)?));
                self.buffers.insert(out.clone(), BufferBound::Fixed(0));
                Ok(out)
            }
            Expr::Method(xs, n, a)
                if matches!(
                    check::infer(xs, &Self::types(env), self.p)?,
                    Type::Buffer(_)
                ) =>
            {
                let xt = check::infer(xs, &Self::types(env), self.p)?;
                let Type::Buffer(inner) = &xt else {
                    unreachable!()
                };
                let b = self.scalar(xs, env, &xt)?;
                if n == "len" {
                    return Ok(format!("{b}.length"));
                }
                if n == "pop" {
                    self.line(format!("if({b}.length>0u){{{b}.length--;}}"));
                    return Ok(b);
                }
                if n == "split_at" {
                    let at = self.scalar(&a[0], env, &Type::U32)?;
                    let at = self.bind(&Type::U32, &at)?;
                    let left = self.fresh();
                    let right = self.fresh();
                    let i = self.fresh();
                    let parent = self.buffer_bound(&b)?;
                    let partition = self.next; // fresh owner identity, never a source-name alias
                    self.line(format!("var {left}:{};var {right}:{};{left}.length=min({at},{b}.length);{right}.length={b}.length-{left}.length;", wg(&xt)?, wg(&xt)?));
                    self.line(format!("for(var {i}=0u;{i}<{left}.length;{i}++){{{left}.data[{i}]={b}.data[{i}];}}"));
                    let j = self.fresh();
                    self.line(format!("for(var {j}=0u;{j}<{right}.length;{j}++){{{right}.data[{j}]={b}.data[{left}.length+{j}];}}"));
                    self.buffers.insert(
                        left.clone(),
                        BufferBound::Partition(Box::new(parent.clone()), partition, false),
                    );
                    self.buffers.insert(
                        right.clone(),
                        BufferBound::Partition(Box::new(parent), partition, true),
                    );
                    return self.apply(
                        &a[1],
                        env,
                        vec![(left, xt.clone()), (right, xt.clone())],
                        &xt,
                    );
                }
                if n == "append" {
                    let other = self.scalar(&a[0], env, &xt)?;
                    let bound = self.buffer_bound(&b)?.append(&self.buffer_bound(&other)?);
                    if bound.upper() > 64 {
                        return Err(
                            "GPU temporary append capacity exceeds 64 elements; CPU fallback"
                                .into(),
                        );
                    }
                    let i = self.fresh();
                    self.line(format!("for(var {i}=0u;{i}<{other}.length;{i}++){{{b}.data[{b}.length+{i}]={other}.data[{i}];}}{b}.length+={other}.length;"));
                    self.buffers.insert(b.clone(), bound);
                    return Ok(b);
                }
                if n == "push" {
                    let upper = self.buffer_bound(&b)?.upper() + 1;
                    if upper > 64 {
                        return Err(
                            "GPU temporary push capacity exceeds 64 elements; CPU fallback".into(),
                        );
                    }
                    let value = self.scalar(&a[0], env, inner)?;
                    self.line(format!("{b}.data[{b}.length]={value};{b}.length++;"));
                    self.buffers.insert(b.clone(), BufferBound::Fixed(upper));
                    return Ok(b);
                }
                if n == "filter" {
                    let i = self.fresh();
                    let kept = self.fresh();
                    self.line(format!(
                        "var {kept}=0u;for(var {i}=0u;{i}<{b}.length;{i}++){{"
                    ));
                    let item = self.bind(inner, &format!("{b}.data[{i}]"))?;
                    let pred = self.apply(
                        &a[0],
                        env,
                        vec![(item.clone(), *inner.clone())],
                        &Type::Bool,
                    )?;
                    self.line(format!(
                        "if({pred}){{{b}.data[{kept}]={item};{kept}++;}}}}{b}.length={kept};"
                    ));
                    return Ok(b);
                }
                if ["map", "map_indexed", "scan"].contains(&n.as_str()) {
                    let i = self.fresh();
                    let acc = self.fresh();
                    if n == "scan" {
                        self.line(format!("var {acc}:{}=0;", self.ty(inner)?));
                    }
                    self.line(format!("for(var {i}=0u;{i}<{b}.length;{i}++){{"));
                    let item = format!("{b}.data[{i}]");
                    let value = if n == "scan" {
                        self.line(format!("{acc}={acc}+{item};"));
                        acc
                    } else {
                        let params = if n == "map" {
                            vec![(item, *inner.clone())]
                        } else {
                            vec![(i.clone(), Type::U32), (item, *inner.clone())]
                        };
                        self.apply(&a[0], env, params, inner)?
                    };
                    self.line(format!("{b}.data[{i}]={value};}}"));
                    return Ok(b);
                }
                if n == "map_parallel" {
                    self.scalar(&a[0], env, &Type::U32)?;
                    let i = self.fresh();
                    self.line(format!("for(var {i}=0u;{i}<{b}.length;{i}++){{"));
                    let value = self.apply(
                        &a[1],
                        env,
                        vec![
                            (i.clone(), Type::U32),
                            (format!("{b}.data[{i}]"), *inner.clone()),
                        ],
                        inner,
                    )?;
                    self.line(format!("{b}.data[{i}]={value};}}"));
                    return Ok(b);
                }
                if !["at_or", "set", "swap", "truncate"].contains(&n.as_str()) {
                    return Err("GPU temporary method uses CPU fallback".into());
                };
                let i = self.scalar(&a[0], env, &Type::U32)?;
                let i = self.bind(&Type::U32, &i)?;
                match n.as_str() {
                    "at_or" => {
                        let out = self.fresh();
                        self.line(format!(
                            "var {out}:{};if({i}<{b}.length){{{out}={b}.data[{i}];}}else{{",
                            wg(inner)?
                        ));
                        let value = self.scalar(&a[1], env, inner)?;
                        self.line(format!("{out}={value};}}"));
                        Ok(out)
                    }
                    "set" => {
                        let value = self.scalar(&a[1], env, inner)?;
                        self.line(format!("if({i}<{b}.length){{{b}.data[{i}]={value};}}"));
                        Ok(b)
                    }
                    "swap" => {
                        let j = self.scalar(&a[1], env, &Type::U32)?;
                        let j = self.bind(&Type::U32, &j)?;
                        let tmp = self.fresh();
                        self.line(format!("if({i}<{b}.length&&{j}<{b}.length){{let {tmp}={b}.data[{i}];{b}.data[{i}]={b}.data[{j}];{b}.data[{j}]={tmp};}}"));
                        Ok(b)
                    }
                    "truncate" => {
                        self.line(format!("{b}.length=min({b}.length,{i});"));
                        Ok(b)
                    }
                    _ => unreachable!(),
                }
            }
            Expr::Method(xs, n, a) if n == "at_or" => {
                let idx = self.scalar(&a[0], env, &Type::U32)?;
                let idx = self.bind(&Type::U32, &idx)?;
                let len = self.length(xs, env)?;
                let out = self.fresh();
                self.line(format!("var {out}:{};if({idx}<{len}){{", wg(t)?));
                let v = self.element_as(xs, &idx, env, Some(t))?;
                self.line(format!("{out}={v};}}else{{"));
                let v = self.scalar(&a[1], env, t)?;
                self.line(format!("{out}={v};}}"));
                Ok(out)
            }
            Expr::Call(n, a) if n == "fold_until" => {
                let len = self.length(&a[0], env)?;
                let initial = self.scalar(&a[1], env, t)?;
                let out = self.bind(t, &initial)?;
                let initial_bound = self.buffers.get(&out).cloned();
                let i = self.fresh();
                self.line(format!("for(var {i}=0u;{i}<{len};{i}++){{"));
                let item = self.element(&a[0], &i, env)?;
                let Type::List(inner) = check::infer(&a[0], &Self::types(env), self.p)? else {
                    return Err("fold input".into());
                };
                let values = vec![(item, *inner), (out.clone(), t.clone())];
                let stop = self.apply(&a[2], env, values.clone(), &Type::Bool)?;
                let stop = self.bind(&Type::Bool, &stop)?;
                let value = self.apply(&a[3], env, values, t)?;
                self.loop_buffer_bound(initial_bound, &out, &value)?;
                self.line(format!("{out}={value};if({stop}){{break;}}}}"));
                Ok(out)
            }
            Expr::Call(n, a) if n == "count" => self.length(&a[0], env),
            Expr::Call(n, a) if n == "sum" || n == "foldr" || n == "foldl" => {
                let len = self.length(&a[0], env)?;
                let init = if n == "sum" {
                    if *t == Type::F32 {
                        "0u".into()
                    } else if *t == Type::I32 {
                        "0i".into()
                    } else if *t == Type::U64 {
                        "InkU64(0u,0u)".into()
                    } else {
                        "0u".into()
                    }
                } else {
                    self.scalar(&a[1], env, t)?
                };
                let out = self.bind(t, &init)?;
                let initial_bound = self.buffers.get(&out).cloned();
                let i = self.fresh();
                self.line(if n == "sum" || n == "foldl" {
                    format!("for(var {i}=0u;{i}<{len};{i}++){{")
                } else {
                    format!("for(var {i}={len};{i}>0u;){{{i}--; ")
                });
                let item = self.element(&a[0], &i, env)?;
                let v = if n == "sum" {
                    self.add("+", &out, &item, t)
                } else {
                    let Type::List(et) = check::infer(&a[0], &Self::types(env), self.p)? else {
                        return Err("fold input".into());
                    };
                    self.apply(&a[2], env, vec![(item, *et), (out.clone(), t.clone())], t)?
                };
                self.loop_buffer_bound(initial_bound, &out, &v)?;
                self.line(format!("{out}={v};}}"));
                Ok(out)
            }

            Expr::Call(n, a) => {
                let f = self
                    .p
                    .functions
                    .iter()
                    .find(|f| &f.name == n)
                    .ok_or("GPU aggregate is unsupported")?
                    .clone();

                let mut local = Env::new();
                for ((n, t), a) in f.params.iter().zip(a) {
                    if matches!(t, Type::List(_)) {
                        let v = self.list(a, env, Some(t))?;
                        local.insert(n.clone(), (v, t.clone()));
                    } else {
                        let v = if matches!(t, Type::List(_)) {
                            self.list(a, env, Some(t))?
                        } else {
                            self.scalar(a, env, t)?
                        };
                        let v = self.bind(t, &v)?;
                        local.insert(n.clone(), (v, t.clone()));
                    }
                }
                self.scalar(&f.body, &local, t)
            }
            _ => Err("unsupported scalar compute expression".into()),
        }
    }
    fn list(&mut self, e: &Expr, env: &Env, want: Option<&Type>) -> LangResult<String> {
        if self.depth >= 48 {
            return Err("GPU local collection lowering depth budget".into());
        }
        self.depth += 1;
        let result = self.list_inner(e, env, want);
        self.depth -= 1;
        result
    }
    fn list_inner(&mut self, e: &Expr, env: &Env, want: Option<&Type>) -> LangResult<String> {
        if self.budget == 0 {
            return Err("GPU local collection emission budget".into());
        }
        self.budget -= 1;
        if let Expr::Var(n) = e {
            return env.get(n).map(|(v, _)| v.clone()).ok_or("GPU list binding".into());
        }
        if let Expr::Call(n, items) = e {
            if n == "list" {
                let Type::List(et) = check::infer_context(e, &Self::types(env), self.p, want)? else {
                    return Err("GPU local list type".into());
                };
                lanes(&et, self.p, true)?;
                let (alignment, size) = local_layout(&et, self.p, 0, &mut self.budget)?;
                let stride = size.checked_add(alignment - 1).map(|n| n / alignment * alignment)
                    .ok_or("GPU local list size overflow")?;
                let words = stride.checked_mul(items.len().max(1))
                    .ok_or("GPU local list size overflow")?;
                if words > 4096usize.saturating_sub(self.local_list_words) {
                    return Err("GPU local list storage budget exceeded; CPU fallback".into());
                }
                self.local_list_words += words;
                let array = self.fresh();
                self.line(format!("var {array}:array<{},{}>;", wg(&et)?, items.len().max(1)));
                // A source constructor is strict and evaluates its elements once,
                // in order, before a subsequent use of the immutable list.
                for (i, item) in items.iter().enumerate() {
                    let value = self.scalar(item, env, &et)?;
                    self.line(format!("{array}[{i}u]={value};"));
                }
                let slot = format!("local_list_{array}");
                self.local_lists.insert(slot.clone(), (array, items.len()));
                return Ok(slot);
            }
        }
        if let Some((body, local)) = self.with_list_body(e, env)? {
            return self.list(&body, &local, want);
        }
        let i = self.views.len();
        self.views.push((e.clone(), env.clone()));
        Ok(format!("view{i}"))
    }
    fn apply(
        &mut self,
        e: &Expr,
        env: &Env,
        values: Vec<(String, Type)>,
        t: &Type,
    ) -> LangResult<String> {
        let mut env = env.clone();
        let mut body = e;
        for (v, t) in values {
            let Expr::Lambda(n, b) = body else {
                return Err("lambda arity".into());
            };
            let v = self.bind(&t, &v)?;
            env.insert(n.clone(), (v, t));
            body = b;
        }
        self.scalar(body, &env, t)
    }
    fn resolve(&self, e: &Expr, env: &Env) -> Option<(Expr, Env)> {
        if let Expr::Var(n) = e {
            let (v, _) = env.get(n)?;
            if let Some(i) = v.strip_prefix("view") {
                return self.views.get(i.parse::<usize>().ok()?).cloned();
            }
        }
        None
    }
    fn with_list_body(&mut self, e: &Expr, env: &Env) -> LangResult<Option<(Expr, Env)>> {
        if let Some(v) = self.resolve(e, env) {
            return Ok(Some(v));
        }
        match e {
            Expr::Call(n, _) if n == "list" => {
                let ty = check::infer(e, &Self::types(env), self.p)?;
                let v = self.list(e, env, Some(&ty))?;
                Ok(Some((Expr::Var("ink_local_list".into()),
                    BTreeMap::from([("ink_local_list".into(), (v, ty))]))))
            }
            Expr::Let(n, annotation, x, b) => {
                let t = check::infer_context(x, &Self::types(env), self.p, annotation.as_deref())?;
                let v = if matches!(t, Type::List(_)) {
                    self.list(x, env, Some(&t))?
                } else {
                    let v = self.scalar(x, env, &t)?;
                    self.bind(&t, &v)?
                };
                let mut local = env.clone();
                local.insert(n.clone(), (v, t));
                Ok(Some((*b.clone(), local)))
            }
            Expr::Call(n, args) if n == "repeat" => {
                let Expr::Num(bound) = args[0] else {
                    return Err("GPU repeat bound".into());
                };
                if bound > 16 {
                    return Err("GPU collection unrolling work limit; CPU fallback".into());
                }
                let Type::List(_) = check::infer(e, &Self::types(env), self.p)? else {
                    return Err("list repeat".into());
                };
                let Expr::Lambda(i, inner) = &args[2] else {
                    return Err("repeat index".into());
                };
                let Expr::Lambda(acc, body) = inner.as_ref() else {
                    return Err("repeat accumulator".into());
                };
                let t = check::infer(&args[1], &Self::types(env), self.p)?;
                let mut v = self.list(&args[1], env, Some(&t))?;
                for index in 0..bound {
                    let mut local = env.clone();
                    local.insert(i.clone(), (format!("{index}u"), Type::U32));
                    local.insert(acc.clone(), (v, t.clone()));
                    v = self.list(body, &local, Some(&t))?;
                }
                Ok(Some((
                    Expr::Var("ink_repeat_value".into()),
                    BTreeMap::from([("ink_repeat_value".into(), (v, t))]),
                )))
            }
            Expr::Call(n, args) if n == "parallel_map" => Ok(None),
            Expr::Call(n, args) if n == "choose" => Ok(None),
            Expr::Call(n, args) => {
                let f = self
                    .p
                    .functions
                    .iter()
                    .find(|f| &f.name == n)
                    .ok_or("GPU list call")?
                    .clone();
                let mut local = Env::new();
                for ((n, t), a) in f.params.iter().zip(args) {
                    let v = if matches!(t, Type::List(_)) {
                        self.list(a, env, Some(t))?
                    } else {
                        let v = self.scalar(a, env, t)?;
                        self.bind(t, &v)?
                    };
                    local.insert(n.clone(), (v, t.clone()));
                }
                Ok(Some((f.body, local)))
            }
            _ => Ok(None),
        }
    }
    fn length(&mut self, e: &Expr, env: &Env) -> LangResult<String> {
        if self.depth >= 48 {
            return Err("GPU collection lowering depth budget".into());
        }
        self.depth += 1;
        let r = self.length_inner(e, env);
        self.depth -= 1;
        r
    }
    fn length_inner(&mut self, e: &Expr, env: &Env) -> LangResult<String> {
        if let Expr::Call(n, a) = e {
            if n == "parallel_map" {
                self.scalar(&a[1], env, &Type::U32)?;
                return self.length(&a[0], env);
            }
        }
        if let Some((e, env)) = self.with_list_body(e, env)? {
            return self.length(&e, &env);
        }
        match e {
            Expr::Var(n) => {
                let (slot, _) = env.get(n).ok_or("list binding")?;
                if let Some((_, length)) = self.local_lists.get(slot) {
                    return Ok(format!("{length}u"));
                }
                Ok(format!("a{slot}[0u]"))
            }
            Expr::Call(n, a) if n == "choose" => {
                let c = self.scalar(&a[0], env, &Type::Bool)?;
                let out = self.fresh();
                self.line(format!("var {out}:u32;if({c}){{"));
                let x = self.length(&a[1], env)?;
                self.line(format!("{out}={x};}}else{{"));
                let x = self.length(&a[2], env)?;
                self.line(format!("{out}={x};}}"));
                Ok(out)
            }
            Expr::Method(xs, n, a) => {
                let len = self.length(xs, env)?;
                if n == "take" || n == "drop" {
                    let count = self.scalar(&a[0], env, &Type::U32)?;
                    let clipped = format!("min({len},{count})");
                    Ok(if n == "take" { clipped } else { format!("({len}-{clipped})") })
                } else if n == "zip" {
                    let other = self.length(&a[0], env)?;
                    Ok(format!("min({len},{other})"))
                } else if n == "filter" {
                    let out = self.bind(&Type::U32, "0u")?;
                    let i = self.fresh();
                    self.line(format!("for(var {i}=0u;{i}<{len};{i}++){{"));
                    let item = self.element(xs, &i, env)?;
                    let Type::List(et) = check::infer(xs, &Self::types(env), self.p)? else {
                        return Err("list type".into());
                    };
                    let keep = self.apply(&a[0], env, vec![(item, *et)], &Type::Bool)?;
                    self.line(format!("if({keep}){{{out}++;}}}}"));
                    Ok(out)
                } else {
                    Ok(len)
                }
            }
            _ => Err("GPU collection shape requires flat input-derived arrays".into()),
        }
    }
    fn element(&mut self, e: &Expr, index: &str, env: &Env) -> LangResult<String> {
        self.element_as(e, index, env, None)
    }
    // Literal callbacks inherit the checked result carrier. Re-inferring a
    // number-only branch without its context can silently select U64 here.
    fn element_as(&mut self, e: &Expr, index: &str, env: &Env, want: Option<&Type>) -> LangResult<String> {
        if self.depth >= 48 {
            return Err("GPU collection lowering depth budget".into());
        }
        self.depth += 1;
        let r = self.element_inner(e, index, env, want);
        self.depth -= 1;
        r
    }
    fn element_inner(&mut self, e: &Expr, index: &str, env: &Env, want: Option<&Type>) -> LangResult<String> {
        let list = want.map(|t| Type::List(Box::new(t.clone())));
        if let Expr::Call(n, a) = e {
            if n == "parallel_map" {
                let Type::List(input) = check::infer(&a[0], &Self::types(env), self.p)? else {
                    unreachable!()
                };
                let Type::List(output) = check::infer_context(e, &Self::types(env), self.p, list.as_ref())? else {
                    unreachable!()
                };
                let v = self.element(&a[0], index, env)?;
                return self.apply(
                    &a[2],
                    env,
                    vec![(index.into(), Type::U32), (v, *input)],
                    &output,
                );
            }
        }
        if self.budget == 0 {
            return Err("GPU collection emission budget".into());
        }
        self.budget -= 1;
        let Type::List(et) = check::infer_context(e, &Self::types(env), self.p, list.as_ref())? else {
            return Err("GPU list type".into());
        };
        if let Some((e, env)) = self.with_list_body(e, env)? {
            return self.element_as(&e, index, &env, Some(&et));
        }
        match e {
            Expr::Var(n) => {
                let (slot, _) = env.get(n).ok_or("list binding")?;
                if let Some((array, _)) = self.local_lists.get(slot) {
                    return Ok(format!("{array}[{index}]"));
                }
                load_list(&et, &format!("a{slot}"), index, self.p)
            }
            Expr::Call(n, a) if n == "choose" => {
                let c = self.scalar(&a[0], env, &Type::Bool)?;
                let out = self.fresh();
                self.line(format!("var {out}:{};if({c}){{", wg(&et)?));
                let x = self.element_as(&a[1], index, env, Some(&et))?;
                self.line(format!("{out}={x};}}else{{"));
                let x = self.element_as(&a[2], index, env, Some(&et))?;
                self.line(format!("{out}={x};}}"));
                Ok(out)
            }
            Expr::Method(xs, n, a) => {
                let same_carrier = ["take", "drop", "scan", "filter", "sort"].contains(&n.as_str());
                let source = same_carrier.then(|| Type::List(et.clone()));
                let Type::List(st) = check::infer_context(xs, &Self::types(env), self.p, source.as_ref())? else {
                    return Err("list type".into());
                };
                if n == "take" { return self.element_as(xs, index, env, Some(&st)); }
                if n == "drop" {
                    let len = self.length(xs, env)?;
                    let count = self.scalar(&a[0], env, &Type::U32)?;
                    return self.element_as(xs, &format!("({index}+min({len},{count}))"), env, Some(&st));
                }
                if n == "scan" {
                    let out = self.bind(
                        &et,
                        if *et == Type::F32 {
                            "0u"
                        } else if *et == Type::U64 {
                            "InkU64(0u,0u)"
                        } else if *et == Type::I32 {
                            "0i"
                        } else {
                            "0u"
                        },
                    )?;
                    let i = self.fresh();
                    self.line(format!("for(var {i}=0u;{i}<={index};{i}++){{"));
                    let v = self.element_as(xs, &i, env, Some(&st))?;
                    let v = self.add("+", &out, &v, &et);
                    self.line(format!("{out}={v};}}"));
                    return Ok(out);
                }
                if n == "filter" {
                    let len = self.length(xs, env)?;
                    let seen = self.bind(&Type::U32, "0u")?;
                    let out = self.fresh();
                    self.line(format!("var {out}:{};", wg(&et)?));
                    let i = self.fresh();
                    self.line(format!("for(var {i}=0u;{i}<{len};{i}++){{"));
                    let v = self.element_as(xs, &i, env, Some(&st))?;
                    let v = self.bind(&et, &v)?;
                    let keep = self.apply(&a[0], env, vec![(v.clone(), *st)], &Type::Bool)?;
                    self.line(format!(
                        "if({keep}){{if({seen}=={index}){{{out}={v};break;}}{seen}++;}}}}"
                    ));
                    return Ok(out);
                }
                if n == "sort" {
                    let len = self.length(xs, env)?;
                    let out = self.fresh();
                    self.line(format!("var {out}:{};", wg(&et)?));
                    let i = self.fresh();
                    let j = self.fresh();
                    self.line(format!("for(var {i}=0u;{i}<{len};{i}++){{"));
                    let v = self.element_as(xs, &i, env, Some(&st))?;
                    let v = self.bind(&et, &v)?;
                    let rank = self.bind(&Type::U32, "0u")?;
                    self.line(format!("for(var {j}=0u;{j}<{len};{j}++){{"));
                    let other = self.element_as(xs, &j, env, Some(&st))?;
                    let other = self.bind(&et, &other)?;
                    let less = self.add("<", &other, &v, &et);
                    let equal = self.add("==", &other, &v, &et);
                    self.line(format!("if({less}||({equal}&&{j}<{i})){{{rank}++;}}}}if({rank}=={index}){{{out}={v};break;}}}}"));
                    return Ok(out);
                }
                let v = self.element_as(xs, index, env, Some(&st))?;
                match n.as_str() {
                    "map" => self.apply(&a[0], env, vec![(v, *st)], &et),
                    "map_indexed" => {
                        self.apply(&a[0], env, vec![(index.into(), Type::U32), (v, *st)], &et)
                    }
                    "zip" => {
                        let other = self.element(&a[0], index, env)?;
                        let Type::List(ot) = check::infer(&a[0], &Self::types(env), self.p)? else {
                            return Err("zip list".into());
                        };
                        self.apply(&a[1], env, vec![(v, *st), (other, *ot)], &et)
                    }
                    _ => Err("GPU collection operation".into()),
                }
            }
            _ => Err("GPU collection requires flat input-derived arrays".into()),
        }
    }
    fn bound_slots(&mut self, e: &Expr, env: &Env) -> LangResult<Vec<usize>> {
        if self.depth >= 48 || self.budget == 0 {
            return Err("GPU collection lowering depth budget".into());
        }
        self.budget -= 1;
        self.depth += 1;
        let r = self.bound_slots_inner(e, env);
        self.depth -= 1;
        r
    }
    fn bound_slots_inner(&mut self, e: &Expr, env: &Env) -> LangResult<Vec<usize>> {
        if let Expr::Call(n, a) = e {
            if n == "parallel_map" {
                return self.bound_slots(&a[0], env);
            }
        }
        if let Some((e, env)) = self.with_list_body(e, env)? {
            return self.bound_slots(&e, &env);
        }
        match e {
            Expr::Call(n, a) if n == "choose" => {
                let mut v = self.bound_slots(&a[1], env)?;
                v.extend(self.bound_slots(&a[2], env)?);
                Ok(v)
            }
            Expr::Var(n) => Ok(vec![env
                .get(n)
                .ok_or("list binding")?
                .0
                .parse()
                .map_err(|_| "input slot")?]),
            Expr::Method(xs, n, a) => {
                let mut v = self.bound_slots(xs, env)?;
                if n == "zip" {
                    v.extend(self.bound_slots(&a[0], env)?)
                }
                Ok(v)
            }
            _ => Err("GPU output capacity is not input-bounded".into()),
        }
    }
}
// Packed storage avoids vec3 padding, Bool storage restrictions and host-specific
// struct alignment. The descriptor defines word order independently of placement.
fn load(t: &Type, buf: &str, index: &str, p: &Program) -> LangResult<String> {
    load_with_header(t, buf, index, p, 0)
}
fn load_with_header(t: &Type, buf: &str, index: &str, p: &Program, header: usize) -> LangResult<String> {
    let width = lanes(t, p, true)?;
    fn go(t: &Type, buf: &str, base: &str, offset: &mut usize, p: &Program) -> LangResult<String> {
        let v = match t {
            Type::U32 | Type::I32 | Type::F32 | Type::Bool => {
                let n = *offset;
                *offset += 1;
                let x = format!("{buf}[{base}+{n}u]");
                match t {
                    Type::U32 | Type::F32 => x,
                    Type::Bool => format!("({x}!=0u)"),
                    _ => format!("bitcast<{}>({x})", wg(t)?),
                }
            }
            Type::U64 => {
                let n = *offset;
                *offset += 2;
                format!("InkU64({buf}[{base}+{n}u],{buf}[{base}+{}u])", n + 1)
            }
            Type::Vector(et, n) => {
                let a = (0..*n)
                    .map(|_| go(et, buf, base, offset, p))
                    .collect::<LangResult<Vec<_>>>()?;
                format!("{}({})", wg(t)?, a.join(","))
            }
            Type::Named(n) => {
                let a = p.records[n]
                    .iter()
                    .map(|(_, t)| go(t, buf, base, offset, p))
                    .collect::<LangResult<Vec<_>>>()?;
                format!("R_{n}({})", a.join(","))
            }
            _ => return Err("unsupported GPU layout".into()),
        };
        Ok(v)
    }
    go(t, buf, &format!("{header}u+({index})*{width}u"), &mut 0, p)
}
fn load_list(t: &Type, buf: &str, index: &str, p: &Program) -> LangResult<String> {
    load_with_header(t, buf, index, p, 1)
}
fn store(t: &Type, value: &str, base: &str, off: &mut usize, p: &Program) -> LangResult<String> {
    Ok(match t {
        Type::U32 | Type::I32 | Type::F32 | Type::Bool => {
            let n = *off;
            *off += 1;
            let v = match t {
                Type::U32 => value.to_string(),
                Type::Bool => format!("select(0u,1u,{value})"),
                _ => format!("bitcast<u32>({value})"),
            };
            format!("output[{base}+{n}u]={v};\n")
        }
        Type::U64 => {
            let n = *off;
            *off += 2;
            format!(
                "output[{base}+{n}u]=({value}).lo;output[{base}+{}u]=({value}).hi;\n",
                n + 1
            )
        }
        Type::Vector(et, n) => (0..*n)
            .map(|i| {
                store(
                    et,
                    &format!("({value}).{}", ["x", "y", "z", "w"][i as usize]),
                    base,
                    off,
                    p,
                )
            })
            .collect::<LangResult<Vec<_>>>()?
            .join(""),
        Type::Named(n) => p.records[n]
            .iter()
            .map(|(n, t)| store(t, &format!("({value}).ink_f_{n}"), base, off, p))
            .collect::<LangResult<Vec<_>>>()?
            .join(""),
        _ => return Err("unsupported GPU output".into()),
    })
}
fn types(p: &Program, needed: &std::collections::BTreeSet<String>) -> LangResult<String> {
    let mut out = include_str!("u64.wgsl").to_owned();
    out.push_str(include_str!("f32.wgsl"));
    for t in ["u32", "InkU64", "i32", "bool"] {
        out.push_str(&format!(
            "struct InkBuffer_{t}{{data:array<{t},64>,length:u32,}}\n"
        ));
    }
    for (n, fields) in &p.records {
        if !needed.contains(n) { continue; }
        out.push_str(&format!("struct R_{n}{{"));
        for (n, t) in fields {
            out.push_str(&format!("ink_f_{n}:{},", wg(t)?));
        }
        out.push_str("}\n");
    }
    Ok(out)
}

fn shader(f: &Function, p: &Program, declarations: &str) -> LangResult<(String, Vec<usize>)> {
    let Type::List(et) = &f.result else {
        return Err(
            "GPU v2 currently requires an array result; ordered scalar reductions use CPU".into(),
        );
    };
    if matches!(&f.body, Expr::Method(_, m, _) if m == "concat") {
        return Err("GPU concatenation currently uses the compiled CPU fallback".into());
    }
    lanes(et, p, true)?;
    let mut env = Env::new();
    let mut decl = declarations.to_owned();
    for (i, (n, t)) in f.params.iter().enumerate() {
        let leaf = if let Type::List(t) = t { t.as_ref() } else { t };
        lanes(leaf, p, true)?;
        decl.push_str(&format!(
            "@group(0) @binding({i}) var<storage,read> a{i}:array<u32>;\n"
        ));
        env.insert(
            n.clone(),
            (
                if matches!(t, Type::List(_)) {
                    i.to_string()
                } else {
                    load(t, &format!("a{i}"), "0u", p)?
                },
                t.clone(),
            ),
        );
    }
    let nb = f.params.len();
    decl.push_str(&format!("@group(0) @binding({nb}) var<storage,read_write> output:array<u32>;\n@group(0) @binding({}) var<storage,read> ink_meta:array<u32>;\n",nb+1));
    let mut em = Emitter {
        p,
        lines: String::new(),
        next: 0,
        budget: 100000,
        depth: 0,
        views: vec![],
        local_lists: BTreeMap::new(),
        local_list_words: 0,
        buffers: BTreeMap::new(),
    };
    let lengths = em.bound_slots(&f.body, &env)?;
    em.lines.clear();
    em.views.clear();
    em.local_lists.clear();
    em.local_list_words = 0;
    em.buffers.clear();
    let length = em.length(&f.body, &env)?;
    em.line(format!(
        "if(i==0u){{output[0u]={length};}}if(i>={length}){{return;}}"
    ));
    let v = em.element_as(&f.body, "i", &env, Some(et))?;
    let v = em.bind(et, &v)?;
    em.line(store(
        et,
        &v,
        &format!("1u+i*{}u", lanes(et, p, true)?),
        &mut 0,
        p,
    )?);
    decl.push_str(&format!("@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){{let i=gid.x;if(i>=ink_meta[{nb}]){{return;}}\n{}\n}}",em.lines));
    Ok((decl, lengths))
}
/// Per-element collection work in a direct shader: 0 none, 1 filter/scan
/// (linear per invocation), 2 sort (quadratic stable rank per invocation).
fn heavy(e: &Expr, p: &Program, seen: &mut std::collections::BTreeSet<String>) -> u8 {
    let all = |es: &mut dyn Iterator<Item = &Expr>, seen: &mut std::collections::BTreeSet<String>| {
        es.map(|e| heavy(e, p, seen)).max().unwrap_or(0)
    };
    match e {
        Expr::Method(x, n, a) => {
            let own = match n.as_str() {
                "sort" => 2,
                "filter" | "scan" => 1,
                _ => 0,
            };
            own.max(heavy(x, p, seen)).max(all(&mut a.iter(), seen))
        }
        Expr::Call(n, a) => {
            let callee = p
                .functions
                .iter()
                .find(|f| &f.name == n)
                .filter(|_| seen.insert(n.clone()))
                .map_or(0, |f| heavy(&f.body, p, seen));
            callee.max(all(&mut a.iter(), seen))
        }
        Expr::Let(_, _, a, b) | Expr::Binary(_, a, b) => heavy(a, p, seen).max(heavy(b, p, seen)),
        Expr::Lambda(_, b) | Expr::Neg(b) | Expr::Field(b, _) => heavy(b, p, seen),
        Expr::Record(_, fs) => all(&mut fs.iter().map(|(_, e)| e), seen),
        _ => 0,
    }
}
/// Element bound for a direct shader whose invocations each repeat
/// collection work: linear work per element up to 1,024 elements, quadratic
/// sort per element up to 128. Graph stages have no such bound.
fn heavy_limit(level: u8) -> Option<u32> {
    match level {
        0 => None,
        1 => Some(1024),
        _ => Some(128),
    }
}
pub fn lower(checked: &core::CheckedModule) -> LangResult<crate::gpu::Emission> {
    lower_filtered(checked, None)
}
/// Keep internal typed helpers in the checked program, exposing only the chosen
/// entry through the packed ABI. Eligibility still examines all executed work.
pub fn lower_entry(checked: &core::CheckedModule, entry: &str) -> LangResult<crate::gpu::Emission> {
    lower_filtered(checked, Some(entry))
}
fn lower_filtered(checked: &core::CheckedModule, entry: Option<&str>) -> LangResult<crate::gpu::Emission> {
    let p = checked.pure_program()?;
    if entry.is_some_and(|name| !p.functions.iter().any(|f| f.name == name)) {
        return Err("unknown packed GPU entry function".into());
    }
    let mut files = BTreeMap::new();
    let mut functions = Vec::new();
    for (i, f) in p.functions.iter().enumerate().filter(|(_,f)|entry.map_or(true,|name|f.name==name)) {
        let guard=p.contracts.precondition_checks.get(&f.name);
        let declarations = if guard.is_some() {
            Err("contract precondition ingress uses the CPU host".to_owned())
        } else {
            ink_core::analysis::function_dependencies(p, &f.name, 100000)
                .and_then(|dependencies| types(p, &dependencies.named_types))
        };
        let compiled = declarations.as_ref().map_err(Clone::clone)
            .and_then(|declarations| shader(f, p, declarations));
        let (shader, lengths, direct_reason) = match compiled {
            Ok((s, l)) => {
                let path = format!("compute-{i}.wgsl");
                files.insert(path.clone(), s);
                (Some(path), l, None)
            }
            Err(e) => (None, vec![], Some(e)),
        };
        let (graph, graph_reason) = match declarations.as_ref().map_err(Clone::clone)
            .and_then(|declarations| graph::lower(f, p, i, declarations)) {
            Ok((descriptor, shaders)) => {
                files.extend(shaders);
                (Some(descriptor), None)
            }
            Err(e) => (None, Some(e)),
        };
        let level = heavy(&f.body, p, &mut std::collections::BTreeSet::new());
        let required = graph.is_some() && (shader.is_none() || level > 0);
        let reason = if shader.is_some() || graph.is_some() {
            None
        } else {
            Some(format!(
                "{}; literal graph: {}",
                direct_reason.unwrap_or_default(),
                graph_reason.unwrap_or_default()
            ))
        };
        functions.push(json!({"name":f.name,"index":i,"precondition":guard.map(|name|json!({"index":p.functions.iter().position(|g|&g.name==name).unwrap(),"result":{"kind":"bool"}})),"params":f.params.iter().map(|(n,t)|Ok(json!({"name":n,"type":desc(t,p)?}))).collect::<LangResult<Vec<_>>>()?,"result":desc(&f.result,p)?,"shader":shader,"lengths":lengths,"graph":graph,"graph_required":required,"capacity_mode":"max","gpu_max_elements":if required{None}else{heavy_limit(level)},"fallback_reason":reason}));
    }
    use sha2::{Digest, Sha256};
    let manifest = json!({"schema":2,"backend":"webgpu-wgpu-v2","core_semantics":checked.semantics(),"core_sha256":checked.identity()?,"shaders":files.iter().map(|(n,s)|(n.clone(),format!("{:x}",Sha256::digest(s.as_bytes())))).collect::<BTreeMap<_,_>>(),"functions":functions,"abi":"packed little-endian 32-bit lanes; GPU lists carry a length header; records in declaration order; vectors unpadded","gpu_list_abi":"ink-gpu-list-v3","collection_strategy":"parallel graph schedules for integer scan, sum and radix sort, stable filter and concat; other stages retain literal source order; profitability must be measured","float_contract":"integer-lane binary32; round-to-nearest ties-to-even, gradual underflow, signed zero, source operation order and canonical arithmetic NaN; pass-through/negation retain payload bits","trust":["type checker","literal C and WGSL lowering","packed ABI and host pipeline execution","C/LLVM, WebGPU/wgpu and drivers"],"proof_claim":"no backend correctness proof or optimisation theorem; saved profiles estimate profitability only"});
    Ok(crate::gpu::Emission {
        manifest,
        shaders: files,
    })
}
