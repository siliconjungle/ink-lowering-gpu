//! WGSL pure compute lowering. Logical values have a versioned packed ABI;
//! shader eligibility does not assert or select an optimisation law.
use crate::{check, core, syntax::*, LangResult};
pub use ink_core::compute_values::needed;
use serde_json::{json, Value};
use std::collections::BTreeMap;
type Env = BTreeMap<String, (String, Type)>;

fn wg(t: &Type) -> LangResult<String> {
    Ok(match t {
        Type::U32 => "u32".into(),
        Type::I32 => "i32".into(),
        Type::F32 => "f32".into(),
        Type::Bool => "bool".into(),
        Type::Vector(t, n) => format!("vec{n}<{}>", wg(t)?),
        Type::Named(n) => format!("R_{n}"),
        _ => {
            return Err("GPU scalar requires a 32-bit numeric, Bool, vector or flat record".into())
        }
    })
}
fn desc(t: &Type, p: &Program) -> LangResult<Value> {
    Ok(match t {
        Type::List(t) => json!({"kind":"list","element":desc(t,p)?}),
        Type::Vector(t, n) => json!({"kind":"vector","element":desc(t,p)?,"size":n}),
        Type::Named(n) => {
            json!({"kind":"record","name":n,"fields":p.records.get(n).ok_or("unknown record")?.iter().map(|(n,t)|Ok(json!({"name":n,"type":desc(t,p)?}))).collect::<LangResult<Vec<_>>>()?})
        }
        Type::U32 => json!({"kind":"u32"}),
        Type::U64 => json!({"kind":"u64"}),
        Type::I32 => json!({"kind":"i32"}),
        Type::F32 => json!({"kind":"f32"}),
        Type::Bool => json!({"kind":"bool"}),
        _ => return Err("unsupported compute ABI".into()),
    })
}
fn lanes(t: &Type, p: &Program, gpu: bool) -> LangResult<usize> {
    match t {
        Type::U32 | Type::I32 | Type::F32 | Type::Bool => Ok(1),
        Type::U64 if !gpu => Ok(2),
        Type::Vector(t, n) => Ok(lanes(t, p, gpu)? * (*n as usize)),
        Type::Named(n) => p
            .records
            .get(n)
            .ok_or("unknown record")?
            .iter()
            .try_fold(0, |a, (_, t)| Ok(a + lanes(t, p, gpu)?)),
        _ => {
            Err("packed compute values must be flat; GPU values use only 32-bit components".into())
        }
    }
}
struct Emitter<'a> {
    p: &'a Program,
    lines: String,
    next: usize,
    budget: usize,
    depth: usize,
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

        Ok(n)
    }
    fn add(&self, op: &str, a: &str, b: &str, t: &Type) -> String {
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
                    format!("{n}.0f")
                }
                Type::I32 => format!("{n}{}", { "i" }),
                Type::U64 => return Err("GPU scalar u64 is unsupported".into()),
                _ => format!("{n}u"),
            }),
            Expr::Float(bits) => Ok(format!("bitcast<f32>({bits}u)")),
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
                let v = self.bind(&xt, &a)?;
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
                    if op == "==" && matches!(ot, Type::Vector(..)) {
                        return Ok(format!("all({x} == {y})"));
                    }
                    Ok(self.add(op, &x, &y, &ot))
                }
            }
            Expr::Call(n, a) if n == "choose" => {
                let c = self.scalar(&a[0], env, &Type::Bool)?;
                let out = self.fresh();
                let ty = self.ty(t)?;
                self.line(format!("var {out}:{ty};"));
                self.line(format!("if ({c}) {{"));
                let x = self.scalar(&a[1], env, t)?;
                self.line(format!("{out}={x};\n}} else {{"));
                let y = self.scalar(&a[2], env, t)?;
                self.line(format!("{out}={y};\n}}"));
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
                let idx = self.fresh();
                self.line(format!("for(var {idx}=0u;{idx}<{bound}u;{idx}++){{"));
                let mut local = env.clone();
                local.insert(i.clone(), (idx, Type::U32));
                local.insert(acc.clone(), (out.clone(), t.clone()));
                let v = self.scalar(b, &local, t)?;
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
                self.line(format!("if ({y}==0{overflow}) {{"));
                let fallback = self.scalar(&a[2], env, t)?;
                self.line(format!(
                    "{out}={fallback};\n}} else {{ {out}={x} {} {y}; }}",
                    if n == "quot_or" { "/" } else { "%" }
                ));
                Ok(out)
            }
            Expr::Method(xs, n, a) if n == "at_or" => {
                let Type::List(et) = check::infer(xs, &Self::types(env), self.p)? else {
                    return Err("list required".into());
                };
                let idx = self.scalar(&a[0], env, &Type::U32)?;
                let idx = self.bind(&Type::U32, &idx)?;
                let out = self.fresh();
                let ty = self.ty(t)?;
                self.line(format!("var {out}:{ty};"));
                let (len, read) = {
                    let Expr::Var(n) = xs.as_ref() else {
                        return Err("GPU indexed reads require an input list".into());
                    };
                    let (slot, _) = env.get(n).ok_or("unknown input list")?;
                    (
                        format!("ink_meta[{slot}]"),
                        load(&et, &format!("a{slot}"), &idx, self.p)?,
                    )
                };
                self.line(format!("if ({idx} < {len}) {{ {out}={read}; }} else {{"));
                let v = self.scalar(&a[1], env, t)?;
                self.line(format!("{out}={v};\n}}"));
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
                        let Expr::Var(v) = a else {
                            return Err("shader helper list arguments must be input lists".into());
                        };
                        local.insert(n.clone(), env.get(v).ok_or("unknown list")?.clone());
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
    fn list(&mut self, _e: &Expr, _env: &Env, _want: Option<&Type>) -> LangResult<String> {
        Err("nested GPU collections require another pipeline step".into())
    }
}
// Packed storage avoids vec3 padding, Bool storage restrictions and host-specific
// struct alignment. The descriptor defines word order independently of placement.
fn load(t: &Type, buf: &str, index: &str, p: &Program) -> LangResult<String> {
    let width = lanes(t, p, true)?;
    fn go(t: &Type, buf: &str, base: &str, offset: &mut usize, p: &Program) -> LangResult<String> {
        let v = match t {
            Type::U32 | Type::I32 | Type::F32 | Type::Bool => {
                let n = *offset;
                *offset += 1;
                let x = format!("{buf}[{base}+{n}u]");
                match t {
                    Type::U32 => x,
                    Type::Bool => format!("({x}!=0u)"),
                    _ => format!("bitcast<{}>({x})", wg(t)?),
                }
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
    go(t, buf, &format!("({index})*{width}u"), &mut 0, p)
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
fn types(p: &Program, _gpu: bool) -> LangResult<String> {
    let mut out = String::new();
    for (n, fields) in &p.records {
        out.push_str(&format!("struct R_{n}{{"));
        for (n, t) in fields {
            out.push_str(&format!("ink_f_{n}:{},", wg(t)?));
        }
        out.push_str("}\n");
    }
    Ok(out)
}

fn shader(f: &Function, p: &Program) -> LangResult<(String, Vec<usize>)> {
    let Type::List(et) = &f.result else {
        return Err(
            "GPU v2 currently requires an array result; ordered scalar reductions use CPU".into(),
        );
    };
    lanes(et, p, true)?;
    let mut env = Env::new();
    let mut decl = types(p, true)?;
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
    let Expr::Method(xs, m, a) = &f.body else {
        return Err("GPU array function must directly map, zip or map_indexed input arrays".into());
    };
    let Expr::Var(src) = xs.as_ref() else {
        return Err(
            "GPU source must be an input array; use separate pipeline steps for collection stages"
                .into(),
        );
    };
    let (slot, Type::List(st)) = env.get(src).ok_or("GPU input array missing")? else {
        return Err("GPU input list missing".into());
    };
    let slot: usize = slot.parse().map_err(|_| "invalid array binding")?;
    let mut lengths = vec![slot];
    let Expr::Lambda(n, b) = a
        .get(if m == "zip" { 1 } else { 0 })
        .ok_or("GPU collection lambda missing")?
    else {
        return Err("GPU lambda missing".into());
    };
    let body = if m == "map" {
        env.insert(
            n.clone(),
            (load(st, &format!("a{slot}"), "i", p)?, st.as_ref().clone()),
        );
        b.as_ref()
    } else if m == "map_indexed" {
        let Expr::Lambda(v, b) = b.as_ref() else {
            return Err("GPU indexed lambda missing".into());
        };
        let st = st.as_ref().clone();
        env.insert(n.clone(), ("i".into(), Type::U32));
        env.insert(v.clone(), (load(&st, &format!("a{slot}"), "i", p)?, st));
        b.as_ref()
    } else if m == "zip" {
        let Expr::Var(other) = &a[0] else {
            return Err("GPU zip requires an input array".into());
        };
        let (os, Type::List(ot)) = env.get(other).ok_or("GPU zip input missing")? else {
            return Err("GPU zip list missing".into());
        };
        let os: usize = os.parse().map_err(|_| "invalid zip binding")?;
        lengths.push(os);
        let Expr::Lambda(v, b) = b.as_ref() else {
            return Err("GPU zip lambda missing".into());
        };
        let ot = ot.as_ref().clone();
        let st = st.as_ref().clone();
        env.insert(n.clone(), (load(&st, &format!("a{slot}"), "i", p)?, st));
        env.insert(v.clone(), (load(&ot, &format!("a{os}"), "i", p)?, ot));
        b.as_ref()
    } else {
        return Err(
            "GPU filters, scans and sorts require additional checked primitives; CPU fallback"
                .into(),
        );
    };
    let mut em = Emitter {
        p,
        lines: String::new(),
        next: 0,
        budget: 100000,
        depth: 0,
    };
    let v = em.scalar(body, &env, et)?;
    let v = em.bind(et, &v)?;
    em.line(store(
        et,
        &v,
        &format!("i*{}u", lanes(et, p, true)?),
        &mut 0,
        p,
    )?);
    decl.push_str(&format!("@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){{let i=gid.x;if(i>=ink_meta[{nb}]){{return;}}\n{}\n}}",em.lines));
    Ok((decl, lengths))
}
pub fn lower(checked: &core::CheckedModule) -> LangResult<crate::gpu::Emission> {
    let p = checked.pure_program()?;
    let mut files = BTreeMap::new();
    let mut functions = Vec::new();
    for (i, f) in p.functions.iter().enumerate() {
        let compiled = shader(f, p);
        let (shader, lengths, reason) = match compiled {
            Ok((s, l)) => {
                let path = format!("compute-{i}.wgsl");
                files.insert(path.clone(), s);
                (Some(path), l, None)
            }
            Err(e) => (None, vec![], Some(e)),
        };
        functions.push(json!({"name":f.name,"index":i,"params":f.params.iter().map(|(n,t)|Ok(json!({"name":n,"type":desc(t,p)?}))).collect::<LangResult<Vec<_>>>()?,"result":desc(&f.result,p)?,"shader":shader,"lengths":lengths,"fallback_reason":reason}));
    }
    use sha2::{Digest, Sha256};
    let manifest = json!({"schema":2,"backend":"webgpu-wgpu-v2","core_semantics":checked.semantics(),"core_sha256":checked.identity()?,"shaders":files.iter().map(|(n,s)|(n.clone(),format!("{:x}",Sha256::digest(s.as_bytes())))).collect::<BTreeMap<_,_>>(),"functions":functions,"abi":"packed little-endian 32-bit lanes; records in declaration order; vectors unpadded","float_contract":"portable f32 follows WGSL permitted floating evaluation, including rounding, reassociation, fusion, flush-to-zero and finite-math assumptions; lowering emits source order; CPU reference uses binary32; bitwise equivalence is not promised","trust":["type checker","literal C and WGSL lowering","packed ABI and host pipeline execution","C/LLVM, WebGPU/wgpu and drivers"],"proof_claim":"no backend correctness proof or optimisation theorem; profiling establishes profitability only"});
    Ok(crate::gpu::Emission {
        manifest,
        shaders: files,
    })
}
