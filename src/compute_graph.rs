//! Literal materialized stage graph. No rewrite, fusion, or reduction reassociation.
use super::*;
#[derive(Clone)]
struct Ref {
    id: Value,
    ty: Type,
}
type GraphEnv = BTreeMap<String, Ref>;
struct Graph<'a> {
    p: &'a Program,
    index: usize,
    nodes: Vec<Value>,
    shaders: BTreeMap<String, String>,
    remaining: usize,
    depth: usize,
    alias: usize,
}

pub(super) fn load_list(t: &Type, buf: &str, index: &str, p: &Program) -> LangResult<String> {
    // load() multiplies the element index by its flat width. Offset each actual
    // storage access, rather than offsetting the index (which fails for records).
    Ok(load(t, buf, index, p)?.replace(&format!("{buf}["), &format!("{buf}[1u+")))
}
fn leaf(t: &Type) -> &Type {
    if let Type::List(t) = t {
        t
    } else {
        t
    }
}
fn has_collection(e: &Expr, p: &Program) -> bool {
    match e {
        Expr::Method(..) => true,
        Expr::Call(n, a) => {
            ["sum", "count", "foldr"].contains(&n.as_str())
                || p.functions.iter().any(|f| {
                    &f.name == n && (matches!(f.result, Type::List(_)) || contains_method(&f.body))
                })
                || a.iter().any(|x| has_collection(x, p))
        }
        Expr::Let(_, _, x, b) | Expr::Binary(_, x, b) => {
            has_collection(x, p) || has_collection(b, p)
        }
        Expr::Neg(x) | Expr::Field(x, _) => has_collection(x, p),
        Expr::Record(_, fs) => fs.iter().any(|(_, x)| has_collection(x, p)),
        _ => false,
    }
}
fn contains_method(e: &Expr) -> bool {
    match e {
        Expr::Method(..) => true,
        Expr::Call(n, a) => {
            ["sum", "count", "foldr"].contains(&n.as_str()) || a.iter().any(contains_method)
        }
        Expr::Let(_, _, x, b) | Expr::Binary(_, x, b) => contains_method(x) || contains_method(b),
        Expr::Neg(x) | Expr::Field(x, _) | Expr::Lambda(_, x) => contains_method(x),
        Expr::Record(_, fs) => fs.iter().any(|(_, x)| contains_method(x)),
        _ => false,
    }
}
fn free_vars(
    e: &Expr,
    bound: &mut std::collections::BTreeSet<String>,
    out: &mut std::collections::BTreeSet<String>,
) {
    match e {
        Expr::Var(n) => {
            if !bound.contains(n) {
                out.insert(n.clone());
            }
        }
        Expr::Lambda(n, b) => {
            let mut local = bound.clone();
            local.insert(n.clone());
            free_vars(b, &mut local, out);
        }
        Expr::Let(n, _, x, b) => {
            free_vars(x, bound, out);
            let mut local = bound.clone();
            local.insert(n.clone());
            free_vars(b, &mut local, out);
        }
        Expr::Binary(_, a, b) => {
            free_vars(a, bound, out);
            free_vars(b, bound, out);
        }
        Expr::Call(_, a) => {
            for x in a {
                free_vars(x, bound, out)
            }
        }
        Expr::Method(x, _, a) => {
            free_vars(x, bound, out);
            for x in a {
                free_vars(x, bound, out)
            }
        }
        Expr::Neg(x) | Expr::Field(x, _) | Expr::Try(x) => free_vars(x, bound, out),
        Expr::Record(_, fs) => {
            for (_, x) in fs {
                free_vars(x, bound, out)
            }
        }
        _ => {}
    }
}
impl Graph<'_> {
    fn types(env: &GraphEnv) -> check::Env {
        env.iter().map(|(n, r)| (n.clone(), r.ty.clone())).collect()
    }
    fn alias(&mut self, r: Ref, env: &mut GraphEnv) -> Expr {
        let n = loop {
            let n = format!("ink_graph_value_{}", self.alias);
            self.alias += 1;
            if !env.contains_key(&n) {
                break n;
            }
        };
        env.insert(n.clone(), r);
        Expr::Var(n)
    }
    fn walk(&mut self, e: &Expr, env: &GraphEnv, want: Option<&Type>) -> LangResult<Ref> {
        if self.remaining == 0 || self.depth >= 24 {
            return Err("literal graph traversal budget exceeded".into());
        }
        self.remaining -= 1;
        self.depth += 1;
        let result = self.walk_inner(e, env, want);
        self.depth -= 1;
        result
    }
    fn walk_inner(&mut self, e: &Expr, env: &GraphEnv, want: Option<&Type>) -> LangResult<Ref> {
        let ty = check::infer_context(e, &Self::types(env), self.p, want)?;
        if let Expr::Var(n) = e {
            return env.get(n).cloned().ok_or("unknown graph binding".into());
        }
        if let Expr::Let(n, annotation, x, b) = e {
            let xt = check::infer_context(x, &Self::types(env), self.p, annotation.as_deref())?;
            let v = self.walk(x, env, Some(&xt))?;
            let mut local = env.clone();
            local.insert(n.clone(), v);
            return self.walk(b, &local, Some(&ty));
        }
        if let Expr::Call(n, args) = e {
            if let Some(f) = self.p.functions.iter().find(|f| &f.name == n).cloned() {
                let mut local = GraphEnv::new();
                for ((name, t), arg) in f.params.iter().zip(args) {
                    local.insert(name.clone(), self.walk(arg, env, Some(t))?);
                }
                return self.walk(&f.body, &local, Some(&f.result));
            }
            if ["sum", "count", "foldr"].contains(&n.as_str()) {
                let list_ty = if n == "sum" {
                    Some(Type::List(Box::new(ty.clone())))
                } else {
                    None
                };
                let src = self.walk(&args[0], env, list_ty.as_ref())?;
                let mut local = env.clone();
                self.alias(src.clone(), &mut local);
                let mut initial = None;
                if n == "foldr" {
                    let r = self.walk(&args[1], env, Some(&ty))?;
                    initial = Some(self.alias(r, &mut local));
                }
                return self.node(
                    n,
                    &ty,
                    &local,
                    vec![src],
                    None,
                    initial.as_ref(),
                    args.get(2),
                );
            }
        }
        if let Expr::Method(xs, n, args) = e {
            if [
                "map",
                "map_indexed",
                "filter",
                "zip",
                "concat",
                "scan",
                "sort",
            ]
            .contains(&n.as_str())
            {
                let src = self.walk(
                    xs,
                    env,
                    if ["filter", "concat", "scan", "sort"].contains(&n.as_str()) {
                        Some(&ty)
                    } else {
                        None
                    },
                )?;
                let mut local = env.clone();
                self.alias(src.clone(), &mut local);
                let mut inputs = vec![src];
                if n == "zip" || n == "concat" {
                    let other =
                        self.walk(&args[0], env, if n == "concat" { Some(&ty) } else { None })?;
                    self.alias(other.clone(), &mut local);
                    inputs.push(other);
                }
                let callback = if n == "zip" {
                    args.get(1)
                } else {
                    args.first()
                };
                return self.node(n, &ty, &local, inputs, None, None, callback);
            }
        }
        if matches!(ty, Type::List(_)) {
            return Err("literal graph does not support this list expression (dynamic list choice remains CPU)".into());
        }
        let mut local = env.clone();
        let expression = self.lift(e, &mut local, Some(&ty))?;
        self.node("scalar", &ty, &local, vec![], Some(&expression), None, None)
    }
    // Extract eager collection dependencies, leaving primitive scalar grouping
    // and lazy scalar branch evaluation to the existing statement emitter.
    fn lift(&mut self, e: &Expr, env: &mut GraphEnv, want: Option<&Type>) -> LangResult<Expr> {
        let ty = check::infer_context(e, &Self::types(env), self.p, want)?;
        match e {
            Expr::Let(..) => {
                let r = self.walk(e, env, Some(&ty))?;
                Ok(self.alias(r, env))
            }
            Expr::Call(n, _)
                if ["sum", "count", "foldr"].contains(&n.as_str())
                    || self.p.functions.iter().any(|f| &f.name == n) =>
            {
                let r = self.walk(e, env, Some(&ty))?;
                Ok(self.alias(r, env))
            }
            Expr::Method(xs, n, a) if n == "at_or" => {
                let r = self.walk(xs, env, None)?;
                let x = self.alias(r, env);
                Ok(Expr::Method(
                    Box::new(x),
                    n.clone(),
                    vec![self.lift(&a[0], env, Some(&Type::U32))?, {
                        if has_collection(&a[1], self.p) {
                            return Err("lazy collection fallback requires control-flow graph; CPU fallback".into());
                        }
                        a[1].clone()
                    }],
                ))
            }
            Expr::Binary(op, a, b) => {
                if (op == "&&" || op == "||") && has_collection(b, self.p) {
                    return Err(
                        "lazy collection branches require control-flow graphs; CPU fallback".into(),
                    );
                }
                let ot = if check::is_number(&ty) {
                    ty.clone()
                } else {
                    check::hint(a, &Self::types(env), self.p)
                        .or_else(|| check::hint(b, &Self::types(env), self.p))
                        .unwrap_or(Type::Bool)
                };
                Ok(Expr::Binary(
                    op.clone(),
                    Box::new(self.lift(a, env, Some(&ot))?),
                    Box::new(if op == "&&" || op == "||" {
                        b.as_ref().clone()
                    } else {
                        self.lift(b, env, Some(&ot))?
                    }),
                ))
            }
            Expr::Neg(x) => Ok(Expr::Neg(Box::new(self.lift(x, env, Some(&ty))?))),
            Expr::Field(x, n) => Ok(Expr::Field(Box::new(self.lift(x, env, None)?), n.clone())),
            Expr::Call(n, a) if n == "choose" => {
                if has_collection(&a[1], self.p) || has_collection(&a[2], self.p) {
                    return Err(
                        "lazy collection branches require control-flow graphs; CPU fallback".into(),
                    );
                }
                Ok(Expr::Call(
                    n.clone(),
                    vec![
                        self.lift(&a[0], env, Some(&Type::Bool))?,
                        a[1].clone(),
                        a[2].clone(),
                    ],
                ))
            }
            // Remaining scalar forms (repeat, quotient, records, constructors)
            // are emitted literally. Nested collection reductions inside those
            // unsupported shapes cause an explicit fallback, never fusion.
            _ => Ok(e.clone()),
        }
    }
    fn node(
        &mut self,
        op: &str,
        ty: &Type,
        env: &GraphEnv,
        inputs: Vec<Ref>,
        expression: Option<&Expr>,
        initial: Option<&Expr>,
        callback: Option<&Expr>,
    ) -> LangResult<Ref> {
        if self.nodes.len() >= 64 {
            return Err("literal graph stage budget exceeded".into());
        }
        if *ty != Type::U64 || op != "count" {
            lanes(leaf(ty), self.p, true)?;
        }
        let mut refs: Vec<Ref> = Vec::new();
        let mut bindings = Env::new();
        let mut decl = types(self.p, true)?;
        let mut needed = std::collections::BTreeSet::new();
        for e in [expression, initial, callback].into_iter().flatten() {
            free_vars(e, &mut std::collections::BTreeSet::new(), &mut needed);
        }
        for (name, r) in env {
            if !needed.contains(name) && !inputs.iter().any(|x| x.id == r.id) {
                continue;
            }
            let i = if let Some(i) = refs.iter().position(|v| v.id == r.id) {
                i
            } else {
                refs.push(r.clone());
                refs.len() - 1
            };
            bindings.insert(
                name.clone(),
                (
                    if matches!(r.ty, Type::List(_)) {
                        i.to_string()
                    } else {
                        load(&r.ty, &format!("a{i}"), "0u", self.p)?
                    },
                    r.ty.clone(),
                ),
            );
        }
        let slots: Vec<usize> = inputs
            .iter()
            .map(|r| refs.iter().position(|v| v.id == r.id).unwrap())
            .collect();
        let nb = refs.len();
        for i in 0..nb {
            decl.push_str(&format!(
                "@group(0) @binding({i}) var<storage,read> a{i}:array<u32>;\n"
            ));
        }
        decl.push_str(&format!(
            "@group(0) @binding({nb}) var<storage,read_write> output:array<u32>;\n"
        ));
        let mut em = Emitter {
            p: self.p,
            lines: String::new(),
            next: 0,
            budget: 100000,
            depth: 0,
            graph: true,
            views: vec![],
        };
        let et = leaf(ty);
        let scalar = !matches!(ty, Type::List(_));
        let serial = [
            "filter", "scan", "concat", "sum", "count", "foldr", "scalar",
        ]
        .contains(&op);
        let src = slots.first().map(|i| format!("a{i}"));
        let len = src.as_ref().map(|b| format!("{b}[0]"));
        let mut local = bindings.clone();
        match op {
            "map" | "filter" | "map_indexed" | "zip" => {
                let Expr::Lambda(a, b) = callback.ok_or("stage callback missing")? else {
                    return Err("stage needs lambda".into());
                };
                let src_ref = &inputs[0];
                let value = load_list(leaf(&src_ref.ty), src.as_ref().unwrap(), "i", self.p)?;
                let body = if op == "map_indexed" {
                    let Expr::Lambda(x, body) = b.as_ref() else {
                        return Err("indexed nested lambda missing".into());
                    };
                    local.insert(a.clone(), ("i".into(), Type::U32));
                    local.insert(x.clone(), (value, leaf(&src_ref.ty).clone()));
                    body.as_ref()
                } else if op == "zip" {
                    let Expr::Lambda(x, body) = b.as_ref() else {
                        return Err("zip nested lambda missing".into());
                    };
                    local.insert(a.clone(), (value, leaf(&src_ref.ty).clone()));
                    local.insert(
                        x.clone(),
                        (
                            load_list(leaf(&inputs[1].ty), &format!("a{}", slots[1]), "i", self.p)?,
                            leaf(&inputs[1].ty).clone(),
                        ),
                    );
                    body.as_ref()
                } else {
                    local.insert(a.clone(), (value, leaf(&src_ref.ty).clone()));
                    b.as_ref()
                };
                let v = em.scalar(body, &local, if op == "filter" { &Type::Bool } else { et })?;
                if op == "filter" {
                    em.line(format!("if ({v}) {{"));
                    let v = load_list(et, src.as_ref().unwrap(), "i", self.p)?;
                    em.line(store(
                        et,
                        &v,
                        &format!("1u+used*{}u", lanes(et, self.p, true)?),
                        &mut 0,
                        self.p,
                    )?);
                    em.line("used++;\n}");
                } else {
                    let v = em.bind(et, &v)?;
                    em.line(store(
                        et,
                        &v,
                        &format!("1u+i*{}u", lanes(et, self.p, true)?),
                        &mut 0,
                        self.p,
                    )?);
                }
            }
            "sort" => {
                if !matches!(et, Type::U32 | Type::I32) {
                    return Err("literal graph sort currently supports u32/i32".into());
                }
                let v = load_list(et, src.as_ref().unwrap(), "i", self.p)?;
                let other = load_list(et, src.as_ref().unwrap(), "j", self.p)?;
                em.line(format!("let item={v};var rank=0u;for(var j=0u;j<{};j++){{let other={other};if(other<item||(other==item&&j<i)){{rank++;}}}}",len.as_ref().unwrap()));
                em.line(store(et, "item", "1u+rank", &mut 0, self.p)?);
            }
            "scan" => {
                let v = load_list(et, src.as_ref().unwrap(), "i", self.p)?;
                em.line(format!("acc={};", em.add("+", "acc", &v, et)));
                em.line(store(
                    et,
                    "acc",
                    &format!("1u+i*{}u", lanes(et, self.p, true)?),
                    &mut 0,
                    self.p,
                )?);
            }
            "concat" => {
                let v = load_list(et, src.as_ref().unwrap(), "i", self.p)?;
                em.line(store(
                    et,
                    &v,
                    &format!("1u+i*{}u", lanes(et, self.p, true)?),
                    &mut 0,
                    self.p,
                )?);
            }
            "sum" => {
                let v = load_list(et, src.as_ref().unwrap(), "i", self.p)?;
                em.line(format!("acc={};", em.add("+", "acc", &v, et)));
            }
            "count" => {}
            "foldr" => {
                let Expr::Lambda(x, b) = callback.ok_or("fold callback missing")? else {
                    return Err("fold lambda missing".into());
                };
                let Expr::Lambda(a, body) = b.as_ref() else {
                    return Err("fold nested lambda missing".into());
                };
                local.insert(
                    x.clone(),
                    (
                        load_list(leaf(&inputs[0].ty), src.as_ref().unwrap(), "i", self.p)?,
                        leaf(&inputs[0].ty).clone(),
                    ),
                );
                local.insert(a.clone(), ("acc".into(), ty.clone()));
                let v = em.scalar(body, &local, ty)?;
                em.line(format!("acc={v};"));
            }
            "scalar" => {
                let v = em.scalar(expression.ok_or("scalar expression missing")?, &local, ty)?;
                let v = em.bind(ty, &v)?;
                em.line(store(ty, &v, "0u", &mut 0, self.p)?);
            }
            _ => return Err("unknown literal graph stage".into()),
        }
        let body = if !serial {
            let size = if op == "zip" {
                format!("min({},a{}[0])", len.as_ref().unwrap(), slots[1])
            } else {
                len.clone().unwrap()
            };
            format!(
                "let size={size};let i=gid.x;if(i==0u){{output[0]=size;}}if(i<size){{\n{}\n}}",
                em.lines
            )
        } else {
            match op {
            "filter"=>format!("if(gid.x!=0u){{return;}}var used=0u;for(var i=0u;i<{};i++){{\n{}\n}}output[0]=used;",len.unwrap(),em.lines),
            "concat"=>{let second=load_list(et,&format!("a{}",slots[1]),"i",self.p)?;format!("if(gid.x!=0u){{return;}}let n={};output[0]=n+a{}[0];for(var i=0u;i<n;i++){{{}}}for(var i=0u;i<a{}[0];i++){{{}}}",len.unwrap(),slots[1],em.lines,slots[1],store(et,&second,&format!("1u+(n+i)*{}u",lanes(et,self.p,true)?),&mut 0,self.p)?)},
            "sum"|"scan"=>format!("if(gid.x!=0u){{return;}}var acc:{}={};{}for(var i=0u;i<{};i++){{\n{}\n}}{}",wg(et)?,if *et==Type::F32{"0.0f"}else if *et==Type::I32{"0i"}else{"0u"},if op=="scan"{format!("output[0]={};",len.as_ref().unwrap())}else{String::new()},len.unwrap(),em.lines,if op=="sum"{store(ty,"acc","0u",&mut 0,self.p)?}else{String::new()}),
            "foldr"=>{let mut start=Emitter{p:self.p,lines:String::new(),next:10000,budget:100000,depth:0,graph:true,views:vec![]};let init=start.scalar(initial.ok_or("fold initial missing")?,&bindings,ty)?;format!("if(gid.x!=0u){{return;}}{}var acc:{}={init};var cursor={};loop{{if(cursor==0u){{break;}}cursor--;let i=cursor;\n{}\n}}{}",start.lines,wg(ty)?,len.unwrap(),em.lines,store(ty,"acc","0u",&mut 0,self.p)?)},
            "count"=>format!("if(gid.x!=0u){{return;}}output[0]={};output[1]=0u;",len.unwrap()),
            _=>format!("if(gid.x!=0u){{return;}}{}",em.lines)
        }
        };
        decl.push_str(&format!("@compute @workgroup_size({}) fn main(@builtin(global_invocation_id) gid:vec3<u32>){{\n{body}\n}}",if serial{1}else{64}));
        let id = self.nodes.len();
        let shader = format!("graph-{}-{id}.wgsl", self.index);
        self.shaders.insert(shader.clone(), decl);
        let capacity = if scalar {
            json!({"kind":"one","args":[]})
        } else {
            json!({"kind":if op=="zip"{"min"}else if op=="concat"{"sum"}else{"same"},"args":slots})
        };
        self.nodes.push(json!({"id":id,"op":op,"args":refs.iter().map(|r|r.id.clone()).collect::<Vec<_>>(),"arg_types":refs.iter().map(|r|desc(&r.ty,self.p)).collect::<LangResult<Vec<_>>>()?,"result":desc(ty,self.p)?,"capacity":capacity,"shader":shader,"workgroup_size":if serial{1}else{64},"ordered":serial}));
        Ok(Ref {
            id: json!({"node":id}),
            ty: ty.clone(),
        })
    }
}
pub(super) fn lower(
    f: &Function,
    p: &Program,
    index: usize,
) -> LangResult<(Value, BTreeMap<String, String>)> {
    let mut env = GraphEnv::new();
    for (i, (n, t)) in f.params.iter().enumerate() {
        if *t != Type::U64 {
            lanes(leaf(t), p, true)?;
        }
        env.insert(
            n.clone(),
            Ref {
                id: json!({"param":i}),
                ty: t.clone(),
            },
        );
    }
    let mut g = Graph {
        p,
        index,
        nodes: Vec::new(),
        shaders: BTreeMap::new(),
        remaining: 10000,
        depth: 0,
        alias: 0,
    };
    let out = g.walk(&f.body, &env, Some(&f.result))?;
    // A raw u64 input may only pass through/count; no unsupported WGSL math.
    Ok((
        json!({"schema":1,"nodes":g.nodes,"output":out.id,"result":desc(&f.result,p)?,"semantics":"literal eager array stages; ordered sums forward, folds last-to-first; no reassociation or fusion","limitations":["bounded 64 stages and 24 traversal depth","serial stable filter/scan/concat/ordered reductions","portable WGSL f32; exact CPU bit identity not promised","feedback requires matching bounded capacities and all stages to have graph descriptors"]}),
        g.shaders,
    ))
}
