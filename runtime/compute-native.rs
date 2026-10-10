#[cfg(target_endian = "big")]
compile_error!("packed compute CPU ABI currently requires little endian");
// Compiled CPU and resident wgpu pipelines using the same packed ABI and WGSL
// as the browser host. Profiles measure whole executions, never correctness.
mod assets;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    ffi::c_void,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};
type Result<T> = std::result::Result<T, String>;
const MAX_BYTES: usize = 256 * 1024 * 1024;
extern "C" {
    fn ink_compute_call(
        f: u32,
        data: *const *const c_void,
        lengths: *const usize,
        out: *mut *mut c_void,
        count: *mut usize,
    );
    fn ink_compute_reset();
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Auto,
    Cpu,
    Gpu,
}
impl Backend {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            _ => Err("backend must be auto, cpu or gpu".into()),
        }
    }
}
fn width(t: &Value) -> usize {
    match t["kind"].as_str().unwrap() {
        "list" => width(&t["element"]),
        "vector" => t["size"].as_u64().unwrap() as usize * width(&t["element"]),
        "record" => t["fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| width(&f["type"]))
            .sum(),
        "u64" => 2,
        _ => 1,
    }
}
#[derive(Clone)]
struct Data {
    words: Arc<Vec<u32>>,
    count: usize,
    finite: bool,
}
fn pack(v: &Value, t: &Value) -> Result<Data> {
    fn go(v: &Value, t: &Value, w: &mut Vec<u32>, finite: &mut bool) -> Result<()> {
        match t["kind"].as_str().unwrap() {
            "list" => {
                for x in v.as_array().ok_or("expected list")? {
                    go(x, &t["element"], w, finite)?;
                }
            }
            "vector" => {
                let a = v.as_array().ok_or("expected vector")?;
                if a.len() != t["size"].as_u64().unwrap() as usize {
                    return Err("vector dimension mismatch".into());
                }
                for x in a {
                    go(x, &t["element"], w, finite)?
                }
            }
            "record" => {
                let a = v.as_object().ok_or("expected record")?;
                let fields = t["fields"].as_array().unwrap();
                if a.len() != fields.len() {
                    return Err("record fields mismatch".into());
                }
                for f in fields {
                    go(
                        a.get(f["name"].as_str().unwrap())
                            .ok_or("missing record field")?,
                        &f["type"],
                        w,
                        finite,
                    )?
                }
            }
            "bool" => w.push(v.as_bool().ok_or("expected Bool")? as u32),
            "u32" => w.push(
                u32::try_from(v.as_u64().ok_or("expected u32")?)
                    .map_err(|_| "u32 outside range")?,
            ),
            "i32" => w.push(
                i32::try_from(v.as_i64().ok_or("expected i32")?).map_err(|_| "i32 outside range")?
                    as u32,
            ),
            "u64" => {
                let n = if let Some(s) = v.as_str() {
                    s.parse::<u64>().map_err(|_| "u64 outside range")?
                } else {
                    v.as_u64().ok_or("expected u64")?
                };
                w.push(n as u32);
                w.push((n >> 32) as u32);
            }
            "f32" => {
                let bits = if let Some(n) = v.as_f64() {
                    let f = n as f32;
                    if !f.is_finite() {
                        return Err("f32 input overflow".into());
                    }
                    f.to_bits()
                } else {
                    if v.as_object().is_none_or(|o| o.len() != 1) {
                        return Err("expected finite f32 or explicit F32Bits".into());
                    }
                    u32::try_from(v["F32Bits"].as_u64().ok_or("invalid F32Bits")?)
                        .map_err(|_| "invalid F32Bits")?
                };
                *finite &= f32::from_bits(bits).is_finite();
                w.push(bits);
            }
            _ => return Err("invalid compiled type".into()),
        };
        if w.len() * 4 > MAX_BYTES {
            return Err("input exceeds 256 MiB budget".into());
        }
        Ok(())
    }
    let mut words = Vec::new();
    let mut finite = true;
    go(v, t, &mut words, &mut finite)?;
    let count = if t["kind"] == "list" {
        v.as_array().unwrap().len()
    } else {
        1
    };
    Ok(Data {
        words: Arc::new(words),
        count,
        finite,
    })
}
fn unpack(d: &Data, t: &Value) -> Value {
    fn go(w: &[u32], i: &mut usize, t: &Value) -> Value {
        match t["kind"].as_str().unwrap() {
            "list" => Value::Array(
                (0..w.len() / width(&t["element"]))
                    .map(|_| go(w, i, &t["element"]))
                    .collect(),
            ),
            "vector" => Value::Array(
                (0..t["size"].as_u64().unwrap())
                    .map(|_| go(w, i, &t["element"]))
                    .collect(),
            ),
            "record" => Value::Object(
                t["fields"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|f| (f["name"].as_str().unwrap().into(), go(w, i, &f["type"])))
                    .collect(),
            ),
            "u64" => {
                let n = w[*i] as u64 + ((w[*i + 1] as u64) << 32);
                *i += 2;
                if n > 9007199254740991 {
                    json!(n.to_string())
                } else {
                    json!(n)
                }
            }
            k => {
                let n = w[*i];
                *i += 1;
                match k {
                    "u32" => json!(n),
                    "i32" => json!(n as i32),
                    "bool" => json!(n != 0),
                    "f32" => {
                        let f = f32::from_bits(n);
                        if f.is_finite() {
                            json!(f)
                        } else {
                            json!({"F32Bits":n})
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
    go(&d.words, &mut 0, t)
}
#[derive(Clone)]
struct Ref {
    step: bool,
    index: usize,
    input: Option<String>,
    ty: Value,
    count: Option<usize>,
}
struct Step {
    id: String,
    f: Value,
    args: Vec<Ref>,
    count: Option<usize>,
}
struct Plan {
    constants: Vec<Data>,
    inputs: BTreeMap<String, Ref>,
    steps: Vec<Step>,
    outputs: Vec<usize>,
    feedback: BTreeMap<String, usize>,
    iterations: usize,
    reason: Option<String>,
    bytes: usize,
}
fn compile(v: &Value, m: &Value) -> Result<Plan> {
    let source = v["steps"]
        .as_array()
        .ok_or("pipeline steps must be an array")?;
    if source.is_empty() || source.len() > 64 {
        return Err("pipeline requires 1..64 steps".into());
    }
    let iterations = v
        .get("iterations")
        .map(|n| n.as_u64().ok_or("invalid iterations"))
        .transpose()?
        .unwrap_or(1) as usize;
    if iterations == 0 || iterations > 10000 || iterations * source.len() > 65536 {
        return Err("pipeline iteration/dispatch limit".into());
    }
    let mut constants = Vec::new();
    let mut inputs: BTreeMap<String, Ref> = BTreeMap::new();
    let mut steps: Vec<Step> = Vec::new();
    let mut ids = BTreeMap::new();
    let mut reason = None;
    for s in source {
        let id = s["id"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or("invalid step id")?
            .to_string();
        if ids.contains_key(&id) {
            return Err("duplicate step id".into());
        }
        let f = m["functions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["name"] == s["call"])
            .ok_or("unknown function")?
            .clone();
        let a = s["args"].as_array().ok_or("arguments must be an array")?;
        let params = f["params"].as_array().unwrap();
        if a.len() != params.len() {
            return Err("argument count mismatch".into());
        }
        let mut args = Vec::new();
        for (a, param) in a.iter().zip(params) {
            let t = &param["type"];
            let reference = a.as_object().filter(|o| o.len() == 1);
            let r = if let Some(id) = reference.and_then(|o| o.get("step")) {
                let id = id.as_str().ok_or("invalid step reference")?;
                let index = *ids
                    .get(id)
                    .ok_or("step reference must name an earlier step")?;
                let s: &Step = &steps[index];
                if s.f["result"] != *t {
                    return Err("step reference type mismatch".into());
                }
                Ref {
                    step: true,
                    index,
                    input: None,
                    ty: t.clone(),
                    count: s.count,
                }
            } else if let Some(input) = reference.and_then(|o| o.get("input")) {
                let input = input.as_str().ok_or("invalid input reference")?;
                if let Some(r) = inputs.get(input) {
                    if r.ty != *t {
                        return Err("pipeline input used with different types".into());
                    }
                    r.clone()
                } else {
                    let value = v["inputs"].get(input).ok_or("unknown pipeline input")?;
                    let d = pack(value, t)?;
                    let count = d.count;
                    constants.push(d);
                    let r = Ref {
                        step: false,
                        index: constants.len() - 1,
                        input: Some(input.into()),
                        ty: t.clone(),
                        count: Some(count),
                    };
                    inputs.insert(input.into(), r.clone());
                    r
                }
            } else {
                let d = pack(a, t)?;
                let count = d.count;
                constants.push(d);
                Ref {
                    step: false,
                    index: constants.len() - 1,
                    input: None,
                    ty: t.clone(),
                    count: Some(count),
                }
            };
            args.push(r);
        }
        let count = if f["shader"].is_string() {
            f["lengths"]
                .as_array()
                .unwrap()
                .iter()
                .map(|i| args[i.as_u64().unwrap() as usize].count)
                .collect::<Option<Vec<_>>>()
                .and_then(|a| a.into_iter().min())
        } else {
            reason.get_or_insert_with(|| {
                f["fallback_reason"]
                    .as_str()
                    .unwrap_or("CPU-only function")
                    .into()
            });
            None
        };
        ids.insert(id.clone(), steps.len());
        steps.push(Step { id, f, args, count });
    }
    let output_ids = v
        .get("outputs")
        .cloned()
        .unwrap_or(json!([steps.last().unwrap().id]));
    let output_ids = output_ids.as_array().ok_or("outputs must be an array")?;
    let mut outputs = Vec::new();
    for id in output_ids {
        let i = *ids
            .get(id.as_str().ok_or("invalid output id")?)
            .ok_or("unknown output")?;
        if outputs.contains(&i) {
            return Err("duplicate output".into());
        }
        outputs.push(i);
    }
    if outputs.is_empty() {
        return Err("pipeline requires outputs".into());
    }
    let mut feedback = BTreeMap::new();
    if let Some(f) = v.get("feedback") {
        for (input, id) in f.as_object().ok_or("feedback must be an object")? {
            let r = inputs.get(input).ok_or("unknown feedback input")?;
            let i = *ids
                .get(id.as_str().ok_or("invalid feedback step")?)
                .ok_or("unknown feedback step")?;
            if r.ty != steps[i].f["result"] {
                return Err("feedback type mismatch".into());
            }
            if r.count != steps[i].count {
                reason.get_or_insert("feedback changes array shape; compiled CPU pipeline".into());
            }
            feedback.insert(input.clone(), i);
        }
    }
    if constants.iter().any(|c| !c.finite) {
        reason.get_or_insert("non-finite f32 inputs require CPU".into());
    }
    let bytes = constants
        .iter()
        .map(|c| (c.words.len() * 4).max(4))
        .sum::<usize>()
        + steps
            .iter()
            .map(|s| {
                2 * (s.count.unwrap_or(0) * width(&s.f["result"]) * 4).max(4)
                    + 4 * (s.args.len() + 1)
            })
            .sum::<usize>()
        + outputs
            .iter()
            .map(|i| (steps[*i].count.unwrap_or(0) * width(&steps[*i].f["result"]) * 4).max(4))
            .sum::<usize>();
    if bytes > MAX_BYTES {
        reason.get_or_insert("GPU live-buffer budget exceeds 256 MiB".into());
    }
    Ok(Plan {
        constants,
        inputs,
        steps,
        outputs,
        feedback,
        iterations,
        reason,
        bytes,
    })
}
fn cpu_call(f: &Value, args: &[Data]) -> Result<Data> {
    let pointers: Vec<*const c_void> = args.iter().map(|d| d.words.as_ptr().cast()).collect();
    let lengths: Vec<usize> = args.iter().map(|d| d.count).collect();
    unsafe {
        ink_compute_reset();
        let mut out = std::ptr::null_mut();
        let mut count = 0;
        ink_compute_call(
            f["index"].as_u64().unwrap() as u32,
            pointers.as_ptr(),
            lengths.as_ptr(),
            &mut out,
            &mut count,
        );
        if count > MAX_BYTES / 4 {
            ink_compute_reset();
            return Err("CPU output exceeds 256 MiB".into());
        }
        let words = std::slice::from_raw_parts(out.cast::<u32>(), count).to_vec();
        ink_compute_reset();
        let count = if f["result"]["kind"] == "list" {
            count / width(&f["result"]["element"])
        } else {
            1
        };
        Ok(Data {
            words: Arc::new(words),
            count,
            finite: true,
        })
    }
}
fn cpu(p: &Plan) -> Result<Value> {
    let mut inputs: BTreeMap<String, Data> = p
        .inputs
        .iter()
        .map(|(n, r)| (n.clone(), p.constants[r.index].clone()))
        .collect();
    let mut values: Vec<Option<Data>> = vec![None; p.steps.len()];
    for _ in 0..p.iterations {
        for (i, s) in p.steps.iter().enumerate() {
            let args = s
                .args
                .iter()
                .map(|r| {
                    if let Some(n) = &r.input {
                        inputs[n].clone()
                    } else if r.step {
                        values[r.index].as_ref().unwrap().clone()
                    } else {
                        p.constants[r.index].clone()
                    }
                })
                .collect::<Vec<_>>();
            values[i] = Some(cpu_call(&s.f, &args)?);
        }
        for (n, i) in &p.feedback {
            inputs.insert(n.clone(), values[*i].as_ref().unwrap().clone());
        }
    }
    let values: serde_json::Map<_, _> = p
        .outputs
        .iter()
        .map(|i| {
            (
                p.steps[*i].id.clone(),
                unpack(values[*i].as_ref().unwrap(), &p.steps[*i].f["result"]),
            )
        })
        .collect();
    Ok(json!({"values":values,"counters":{"upload_bytes":0,"readback_bytes":0,"dispatches":0}}))
}
struct Gpu {
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: BTreeMap<String, wgpu::ComputePipeline>,
    lost: Arc<AtomicBool>,
}
impl Gpu {
    fn new() -> Result<Self> {
        if std::env::var_os("INK_GPU_DISABLE").is_some() {
            return Err("GPU disabled by INK_GPU_DISABLE".into());
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(|e| e.to_string())?;
        let lost = Arc::new(AtomicBool::new(false));
        let flag = lost.clone();
        device.set_device_lost_callback(move |_, _| {
            flag.store(true, Ordering::Relaxed);
        });
        Ok(Self {
            _instance: instance,
            device,
            queue,
            pipelines: BTreeMap::new(),
            lost,
        })
    }
    fn prepare(&mut self, p: &Plan) -> Result<()> {
        if self.lost.load(Ordering::Relaxed) {
            return Err("GPU device lost".into());
        }
        let limits = self.device.limits();
        for s in &p.steps {
            if s.args.len() + 2 > limits.max_storage_buffers_per_shader_stage as usize
                || s.count.unwrap().div_ceil(64)
                    > limits.max_compute_workgroups_per_dimension as usize
            {
                return Err("GPU binding/workgroup limits".into());
            }
            let name = s.f["shader"].as_str().unwrap();
            if !self.pipelines.contains_key(name) {
                let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
                let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
                let code = assets::SHADERS
                    .iter()
                    .find(|(n, _)| *n == name)
                    .ok_or("missing compiled shader")?
                    .1;
                let module = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(name),
                        source: wgpu::ShaderSource::Wgsl(code.into()),
                    });
                let entries: Vec<_> = (0..s.args.len() + 2)
                    .map(|i| wgpu::BindGroupLayoutEntry {
                        binding: i as u32,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: i != s.args.len(),
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect();
                let bindings =
                    self.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: None,
                            entries: &entries,
                        });
                let layout = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&bindings)],
                        immediate_size: 0,
                    });
                let pipe = self
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(name),
                        layout: Some(&layout),
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    });
                let allocation = pollster::block_on(oom.pop());
                let validation = pollster::block_on(scope.pop());
                if let Some(e) = allocation.or(validation) {
                    return Err(e.to_string());
                }
                self.pipelines.insert(name.into(), pipe);
            }
        }
        Ok(())
    }
    fn buffer(
        &self,
        bytes: usize,
        usage: wgpu::BufferUsages,
        words: Option<&[u32]>,
    ) -> Result<wgpu::Buffer> {
        let bytes = bytes.max(4);
        let l = self.device.limits();
        if bytes > l.max_buffer_size as usize
            || usage.contains(wgpu::BufferUsages::STORAGE)
                && bytes > l.max_storage_buffer_binding_size as usize
        {
            return Err("GPU buffer limits".into());
        }
        let b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes as u64,
            usage,
            mapped_at_creation: words.is_some(),
        });
        if let Some(w) = words {
            let mut mapped = b
                .slice(..)
                .get_mapped_range_mut()
                .map_err(|e| e.to_string())?;
            for (i, n) in w.iter().enumerate() {
                mapped
                    .slice(i * 4..i * 4 + 4)
                    .copy_from_slice(&n.to_le_bytes());
            }
            drop(mapped);
            b.unmap();
        }
        Ok(b)
    }
    fn execute(&self, p: &Plan) -> Result<Value> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let result = self.execute_inner(p);
        let allocation = pollster::block_on(oom.pop());
        let validation = pollster::block_on(scope.pop());
        if let Some(e) = allocation.or(validation) {
            return Err(e.to_string());
        }
        if self.lost.load(Ordering::Relaxed) {
            return Err("GPU device lost".into());
        }
        result
    }
    fn execute_inner(&self, p: &Plan) -> Result<Value> {
        use wgpu::BufferUsages as U;
        let input = p
            .constants
            .iter()
            .map(|c| self.buffer(c.words.len() * 4, U::STORAGE, Some(&c.words)))
            .collect::<Result<Vec<_>>>()?;
        let banks = (0..2)
            .map(|_| {
                p.steps
                    .iter()
                    .map(|s| {
                        self.buffer(
                            s.count.unwrap() * width(&s.f["result"]) * 4,
                            U::STORAGE | U::COPY_SRC,
                            None,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let metadata = p
            .steps
            .iter()
            .map(|s| {
                let mut w: Vec<u32> = s.args.iter().map(|a| a.count.unwrap() as u32).collect();
                w.push(s.count.unwrap() as u32);
                self.buffer(w.len() * 4, U::STORAGE, Some(&w))
            })
            .collect::<Result<Vec<_>>>()?;
        let groups = p
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| {
                (0..3)
                    .map(|phase| {
                        let bank = if phase == 2 { 1 } else { 0 };
                        let mut entries = s
                            .args
                            .iter()
                            .enumerate()
                            .map(|(binding, a)| {
                                let buffer = if let Some(step) = a
                                    .input
                                    .as_ref()
                                    .filter(|_| phase != 0)
                                    .and_then(|n| p.feedback.get(n))
                                {
                                    &banks[1 - bank][*step]
                                } else if a.step {
                                    &banks[bank][a.index]
                                } else {
                                    &input[a.index]
                                };
                                wgpu::BindGroupEntry {
                                    binding: binding as u32,
                                    resource: buffer.as_entire_binding(),
                                }
                            })
                            .collect::<Vec<_>>();
                        entries.push(wgpu::BindGroupEntry {
                            binding: s.args.len() as u32,
                            resource: banks[bank][i].as_entire_binding(),
                        });
                        entries.push(wgpu::BindGroupEntry {
                            binding: s.args.len() as u32 + 1,
                            resource: metadata[i].as_entire_binding(),
                        });
                        let pipe = &self.pipelines[s.f["shader"].as_str().unwrap()];
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &pipe.get_bind_group_layout(0),
                            entries: &entries,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for iteration in 0..p.iterations {
            let phase = if iteration == 0 {
                0
            } else if iteration % 2 == 1 {
                2
            } else {
                1
            };
            for (i, s) in p.steps.iter().enumerate() {
                if s.count == Some(0) {
                    continue;
                }
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines[s.f["shader"].as_str().unwrap()]);
                pass.set_bind_group(0, &groups[i][phase], &[]);
                pass.dispatch_workgroups(s.count.unwrap().div_ceil(64) as u32, 1, 1);
            }
        }
        let reads = p
            .outputs
            .iter()
            .map(|i| {
                let bytes = p.steps[*i].count.unwrap() * width(&p.steps[*i].f["result"]) * 4;
                let b = self.buffer(bytes, U::MAP_READ | U::COPY_DST, None)?;
                if bytes > 0 {
                    encoder.copy_buffer_to_buffer(
                        &banks[(p.iterations - 1) % 2][*i],
                        0,
                        &b,
                        0,
                        bytes as u64,
                    );
                }
                Ok((*i, b, bytes))
            })
            .collect::<Result<Vec<_>>>()?;
        let submitted = self.queue.submit([encoder.finish()]);
        let mut values = serde_json::Map::new();
        let mut readback = 0;
        for (i, b, bytes) in reads {
            let (tx, rx) = std::sync::mpsc::channel();
            b.map_async(wgpu::MapMode::Read, .., move |r| {
                let _ = tx.send(r);
            });
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submitted.clone()),
                    timeout: Some(std::time::Duration::from_secs(30)),
                })
                .map_err(|e| e.to_string())?;
            rx.recv_timeout(std::time::Duration::from_secs(30))
                .map_err(|e| e.to_string())?
                .map_err(|e| e.to_string())?;
            let mapped = b.slice(..).get_mapped_range().map_err(|e| e.to_string())?;
            let words = mapped[..bytes]
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let d = Data {
                words: Arc::new(words),
                count: p.steps[i].count.unwrap(),
                finite: true,
            };
            values.insert(p.steps[i].id.clone(), unpack(&d, &p.steps[i].f["result"]));
            drop(mapped);
            b.unmap();
            readback += bytes;
        }
        Ok(
            json!({"values":values,"counters":{"upload_bytes":p.constants.iter().map(|c|c.words.len()*4).sum::<usize>()+p.steps.iter().map(|s|4*(s.args.len()+1)).sum::<usize>(),"readback_bytes":readback,"dispatches":p.steps.iter().filter(|s|s.count!=Some(0)).count()*p.iterations,"live_buffer_bytes":p.bytes}}),
        )
    }
}
#[derive(Clone)]
struct Profile {
    cpu_ms: f64,
    gpu_ms: f64,
    setup_ms: f64,
    uses: usize,
    gpu: bool,
}
impl Profile {
    fn json(&self) -> Value {
        json!({"cpu_ms":self.cpu_ms,"gpu_ms":self.gpu_ms,"setup_ms":self.setup_ms,"uses":self.uses,"gpu":self.gpu})
    }
}
fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[1]
}
pub struct Engine {
    manifest: Value,
    gpu: Option<Gpu>,
    gpu_error: Option<String>,
    profiles: BTreeMap<String, Profile>,
}
impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}
impl Engine {
    pub fn new() -> Self {
        Self {
            manifest: serde_json::from_str(assets::MANIFEST).unwrap(),
            gpu: None,
            gpu_error: None,
            profiles: BTreeMap::new(),
        }
    }
    pub fn manifest(&self) -> &Value {
        &self.manifest
    }
    pub fn call(&mut self, name: &str, args: &Value, backend: Backend) -> Result<Value> {
        let mut r = self.pipeline(
            &json!({"steps":[{"id":"result","call":name,"args":args}],"outputs":["result"]}),
            backend,
        )?;
        r["value"] = r["values"]["result"].clone();
        Ok(r)
    }
    pub fn pipeline(&mut self, source: &Value, backend: Backend) -> Result<Value> {
        let p = compile(source, &self.manifest)?;
        if backend == Backend::Cpu || p.reason.is_some() {
            let mut r = cpu(&p)?;
            r["backend"] = json!("cpu");
            r["reason"] = json!(if backend == Backend::Cpu {
                "explicit CPU selection"
            } else {
                p.reason.as_deref().unwrap()
            });
            return Ok(r);
        }
        let key = json!([
            p.steps
                .iter()
                .map(|s| json!([
                    s.f["index"],
                    s.args
                        .iter()
                        .map(|a| json!([a.step, a.index, a.count]))
                        .collect::<Vec<_>>()
                ]))
                .collect::<Vec<_>>(),
            p.iterations,
            p.feedback,
            p.outputs,
            p.constants
                .iter()
                .map(|c| json!([c.count, c.words.iter().take(8).collect::<Vec<_>>()]))
                .collect::<Vec<_>>()
        ])
        .to_string();
        let selected = (|| -> Result<Value> {
            if let Some(e) = &self.gpu_error {
                return Err(e.clone());
            }
            let setup = Instant::now();
            if self.gpu.is_none() {
                match Gpu::new() {
                    Ok(g) => self.gpu = Some(g),
                    Err(e) => {
                        self.gpu_error = Some(e.clone());
                        return Err(e);
                    }
                }
            }
            self.gpu.as_mut().unwrap().prepare(&p)?;
            let setup_ms = ms(setup);
            if backend == Backend::Gpu {
                let mut r = self.gpu.as_ref().unwrap().execute(&p)?;
                r["backend"] = json!("gpu");
                r["reason"] = json!("explicit GPU selection");
                return Ok(r);
            }
            if self.profiles.get(&key).is_none_or(|p| p.uses >= 32) {
                let mut ct = Vec::new();
                let mut gt = Vec::new();
                for trial in 0..3 {
                    if trial % 2 == 0 {
                        let t = Instant::now();
                        cpu(&p)?;
                        ct.push(ms(t));
                        let t = Instant::now();
                        self.gpu.as_ref().unwrap().execute(&p)?;
                        gt.push(ms(t));
                    } else {
                        let t = Instant::now();
                        self.gpu.as_ref().unwrap().execute(&p)?;
                        gt.push(ms(t));
                        let t = Instant::now();
                        cpu(&p)?;
                        ct.push(ms(t));
                    }
                }
                let cpu_ms = median(ct);
                let gpu_ms = median(gt);
                if self.profiles.len() >= 32 {
                    self.profiles.pop_first();
                }
                self.profiles.insert(
                    key.clone(),
                    Profile {
                        cpu_ms,
                        gpu_ms,
                        setup_ms,
                        uses: 0,
                        gpu: gpu_ms + setup_ms / 32.0 < cpu_ms * 0.9,
                    },
                );
            }
            let profile = self.profiles.get_mut(&key).unwrap();
            profile.uses += 1;
            let profile = profile.clone();
            let mut r = if profile.gpu {
                self.gpu.as_ref().unwrap().execute(&p)?
            } else {
                cpu(&p)?
            };
            r["backend"] = json!(if profile.gpu { "gpu" } else { "cpu" });
            r["reason"]=json!("measured whole pipeline including transfers; setup amortised over 32 uses, 10% minimum saving");
            r["profile"] = profile.json();
            Ok(r)
        })();
        match selected {
            Ok(r) => Ok(r),
            Err(e) => {
                self.profiles.remove(&key);
                let mut r = cpu(&p)?;
                r["backend"] = json!("cpu");
                r["reason"] = json!(format!("GPU fallback: {e}"));
                Ok(r)
            }
        }
    }
}
