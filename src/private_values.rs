//! Private WGSL carriers and source-bound owner paths. No host ABI or rewrite facts.
use super::*;

pub(super) const WORDS: usize = 4096;
type Paths = Vec<Vec<String>>;
type Bounds = BTreeMap<Vec<String>, BufferBound>;
#[derive(Default)]
pub(super) struct Private {
    pub(super) words: usize,
    paths: BTreeMap<String, Paths>,
    sizes: BTreeMap<String, usize>,
}
pub(super) fn project(root: &str, path: &[String]) -> String {
    path.iter()
        .fold(root.to_owned(), |v, field| format!("({v}).ink_f_{field}"))
}
pub(super) fn buffer_name(inner: &Type) -> LangResult<String> {
    let name = match inner {
        Type::U32 | Type::U64 | Type::I32 | Type::F32 | Type::Bool | Type::Named(_) => wg(inner)?,
        Type::Vector(t, n)
            if matches!(t.as_ref(), Type::U32 | Type::I32 | Type::F32 | Type::Bool) =>
        {
            format!("Vec{n}_{}", wg(t)?)
        }
        _ => return Err("GPU temporary buffer element requires a supported flat value".into()),
    };
    Ok(format!("InkBuffer_{name}"))
}
impl Emitter<'_> {
    pub(super) fn charge_bound_work(&mut self, count: usize) -> LangResult<()> {
        self.budget = self
            .budget
            .checked_sub(count)
            .ok_or("GPU private owner analysis budget exceeded; CPU fallback")?;
        Ok(())
    }
    fn owner_paths(&mut self, ty: &Type) -> LangResult<Paths> {
        if !matches!(ty, Type::Buffer(_) | Type::Named(_)) {
            return Ok(vec![]);
        }
        if let Type::Named(name) = ty {
            if let Some(paths) = self.private.paths.get(name) {
                let work = paths.iter().map(|p| p.len() + 1).sum();
                self.charge_bound_work(work)?;
                return Ok(self.private.paths[name].clone());
            }
        }
        fn find(
            ty: &Type,
            p: &Program,
            depth: usize,
            remaining: &mut usize,
            cache: &mut BTreeMap<String, Paths>,
        ) -> LangResult<Paths> {
            if depth >= 48 {
                return Err("GPU private owner type depth exceeded; CPU fallback".into());
            }
            *remaining = remaining
                .checked_sub(1)
                .ok_or("GPU private owner type budget exceeded; CPU fallback")?;
            match ty {
                Type::Buffer(_) => Ok(vec![vec![]]),
                Type::Named(name) => {
                    if let Some(paths) = cache.get(name) {
                        let work = paths.iter().map(|p| p.len() + 1).sum();
                        *remaining = remaining
                            .checked_sub(work)
                            .ok_or("GPU private owner summary budget exceeded; CPU fallback")?;
                        return Ok(paths.clone());
                    }
                    let mut paths = vec![];
                    if let Some(fields) = p.records.get(name) {
                        for (field, ty) in fields {
                            for mut path in find(ty, p, depth + 1, remaining, cache)? {
                                path.insert(0, field.clone());
                                paths.push(path);
                            }
                        }
                    }
                    cache.insert(name.clone(), paths.clone());
                    Ok(paths)
                }
                _ => Ok(vec![]),
            }
        }
        find(ty, self.p, 0, &mut self.budget, &mut self.private.paths)
    }
    pub(super) fn owned_bounds(&mut self, ty: &Type, value: &str) -> LangResult<Bounds> {
        self.owner_paths(ty)?
            .into_iter()
            .map(|path| {
                self.charge_bound_work(path.len() + 1)?;
                let bound = self.buffer_bound(&project(value, &path))?;
                Ok((path, bound))
            })
            .collect()
    }
    pub(super) fn inherit_bounds(&mut self, ty: &Type, from: &str, to: &str) -> LangResult<()> {
        for (path, bound) in self.owned_bounds(ty, from)? {
            self.buffers.insert(project(to, &path), bound);
        }
        Ok(())
    }
    pub(super) fn snapshot_buffer_bounds(&mut self) -> LangResult<BTreeMap<String, BufferBound>> {
        self.charge_bound_work(self.buffers.len())?;
        Ok(self.buffers.clone())
    }
    pub(super) fn loop_buffer_bounds(
        &mut self,
        initial: Bounds,
        out: &str,
        step: &str,
    ) -> LangResult<()> {
        for (path, before) in initial {
            self.charge_bound_work(path.len() + 1)?;
            let after = self.buffer_bound(&project(step, &path))?;
            // A partition must keep its exact origin. An independently filled
            // owner with the same upper bound can exceed the actual partition.
            if before != after && !matches!(before,BufferBound::Fixed(n) if after.upper()<=n) {
                return Err("GPU temporary loop may grow its buffer bound; CPU fallback".into());
            }
            self.buffers.insert(project(out, &path), before);
        }
        Ok(())
    }
    pub(super) fn reserve_private(&mut self, ty: &Type) -> LangResult<()> {
        if self.owner_paths(ty)?.is_empty() {
            return Ok(());
        }
        let key = wg(ty)?;
        let size = if let Some(size) = self.private.sizes.get(&key) {
            *size
        } else {
            let (_, size) = local_layout(ty, self.p, 0, &mut self.budget)?;
            self.private.sizes.insert(key, size);
            size
        };
        if size > WORDS.saturating_sub(self.private.words) {
            return Err("GPU private array storage budget exceeded; CPU fallback".into());
        }
        self.private.words += size;
        Ok(())
    }
}

pub(super) fn declarations(
    p: &Program,
    dependencies: &ink_core::analysis::FunctionDependencies,
) -> LangResult<String> {
    fn add_buffer(
        ty: &Type,
        p: &Program,
        remaining: &mut usize,
        buffers: &mut BTreeMap<String, String>,
    ) -> LangResult<()> {
        let Type::Buffer(inner) = ty else {
            return Ok(());
        };
        let name = buffer_name(inner)?;
        if !buffers.contains_key(&name) {
            let (_, size) = local_layout(ty, p, 0, remaining)?;
            if size > WORDS {
                return Err("GPU private buffer type exceeds storage budget; CPU fallback".into());
            }
            buffers.insert(name, wg(inner)?);
        }
        Ok(())
    }
    let mut out = include_str!("u64.wgsl").to_owned();
    out.push_str(include_str!("f32.wgsl"));
    let mut buffers = BTreeMap::from([
        ("InkBuffer_u32".to_owned(), "u32".to_owned()),
        ("InkBuffer_InkU64".to_owned(), "InkU64".to_owned()),
        ("InkBuffer_i32".to_owned(), "i32".to_owned()),
        ("InkBuffer_bool".to_owned(), "bool".to_owned()),
    ]);
    let mut pending = vec![];
    let mut remaining = 100000usize;
    for (name, fields) in &p.records {
        if !dependencies.named_types.contains(name) {
            continue;
        }
        out.push_str(&format!("struct R_{name}{{"));
        for (field, ty) in fields {
            out.push_str(&format!("ink_f_{field}:{},", wg(ty)?));
            pending.push(ty);
        }
        out.push_str("}\n");
    }
    for f in &p.functions {
        if !dependencies.functions.contains(&f.name) {
            continue;
        }
        pending.extend(f.params.iter().map(|(_, t)| t));
        pending.push(&f.result);
        let count = ink_core::analysis::node_count(&f.body, remaining)
            .ok_or("GPU private declaration work budget exceeded")?;
        remaining -= count;
        let mut failure = None;
        ink_core::analysis::typed_walk_context(p, f, |e, env, want| {
            if failure.is_some() {
                return;
            }
            // The fixed constructors can introduce a buffer element type with
            // no signature/annotation. Reuse checked contextual type judgments.
            if matches!(e,Expr::Call(n,_) if n=="buffer_fill")
                || matches!(e,Expr::Method(base,n,_) if n=="empty" && matches!(base.as_ref(),Expr::Var(v) if v=="Buffer"))
            {
                let result = check::infer_context(e, env, p, want)
                    .and_then(|ty| add_buffer(&ty, p, &mut remaining, &mut buffers));
                if let Err(error) = result {
                    failure = Some(error);
                }
            }
        })?;
        if let Some(error) = failure {
            return Err(error);
        }
    }
    while let Some(ty) = pending.pop() {
        remaining = remaining
            .checked_sub(1)
            .ok_or("GPU private declaration type budget exceeded")?;
        match ty {
            Type::Buffer(_) => add_buffer(ty, p, &mut remaining, &mut buffers)?,
            Type::List(inner) | Type::Vector(inner, _) | Type::Option(inner) => pending.push(inner),
            Type::Map(a, b) | Type::Table(a, b) | Type::Result(a, b) => {
                pending.extend([a.as_ref(), b.as_ref()])
            }
            _ => {}
        }
    }
    for (name, element) in buffers {
        out.push_str(&format!(
            "struct {name}{{data:array<{element},64>,length:u32,}}\n"
        ));
    }
    Ok(out)
}
