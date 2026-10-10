//! wgpu device adapter for the versioned runtime ABI.
use super::*;
pub(super) struct Gpu {
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: BTreeMap<String, wgpu::ComputePipeline>,
    lost: Arc<AtomicBool>,
}
impl Gpu {
    pub(super) fn new() -> Result<Self> {
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
        let notify = lost.clone();
        device.set_device_lost_callback(move |_, _| {
            notify.store(true, Ordering::Relaxed);
        });
        Ok(Self {
            _instance: instance,
            device,
            queue,
            pipelines: BTreeMap::new(),
            lost,
        })
    }
    fn pipeline(&mut self, name: &str, access: &[bool]) -> Result<()> {
        if self.pipelines.contains_key(name) {
            return Ok(());
        }
        let code = assets::SHADERS
            .iter()
            .find(|(n, _)| *n == name)
            .ok_or("missing compiled shader")?
            .1;
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let entries: Vec<_> = access
            .iter()
            .enumerate()
            .map(|(i, write)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: !*write },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let bindings = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(name),
                entries: &entries,
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &[Some(&bindings)],
                immediate_size: 0,
            });
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let validation = pollster::block_on(scope.pop());
        let allocation = pollster::block_on(oom.pop());
        if let Some(error) = validation.or(allocation) {
            return Err(error.to_string());
        }
        self.pipelines.insert(name.into(), pipeline);
        Ok(())
    }
    pub(super) fn prepare(&mut self, f: &Value, args: &[Arg]) -> Result<()> {
        if self.lost.load(Ordering::Relaxed) {
            return Err("GPU device lost".into());
        }
        let cap = list(f, args)?.len().max(1);
        let groups = cap.div_ceil(256);
        let limits = self.device.limits();
        if cap as u64 * 4 > limits.max_storage_buffer_binding_size as u64
            || cap as u64 * 4 > limits.max_buffer_size
            || groups > limits.max_compute_workgroups_per_dimension as usize
        {
            return Err("input exceeds GPU buffer/dispatch limits".into());
        }
        let stages = f["gpu"]["stages"].as_array().unwrap();
        let filter = stages.iter().any(|s| s["kind"] == "filter");
        let filters = stages.iter().filter(|s| s["kind"] == "filter").count();
        if cap as u64 * 4 * (1 + stages.len() + filters) as u64 + groups as u64 * 8 * filters as u64
            > 268435456
        {
            return Err("GPU working buffers exceed runtime allocation budget".into());
        }
        if filter && cap > 1048576 {
            return Err("input exceeds this backend's stable-filter prefix budget".into());
        }
        for stage in stages {
            self.pipeline(
                stage["shader"].as_str().unwrap(),
                if stage["kind"] == "filter" {
                    &[false, true, false, true, false, true, true]
                } else {
                    &[false, true, false, true, false]
                },
            )?;
        }
        if filter {
            self.pipeline("filter-offsets.wgsl", &[false, true, true])?;
            self.pipeline("filter-scatter.wgsl", &[false, true, false, false, false])?;
        }
        self.pipeline(
            if f["gpu"]["aggregate"] == "sum" {
                "sum.wgsl"
            } else {
                "count.wgsl"
            },
            if f["gpu"]["aggregate"] == "sum" {
                &[false, true, false, true]
            } else {
                &[false, true]
            },
        )?;
        Ok(())
    }
    fn buffer(&self, size: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size.max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }
    fn upload(&self, words: &[u32]) -> wgpu::Buffer {
        let b = self.buffer(words.len() * 4);
        let bytes: Vec<_> = words.iter().flat_map(|x| x.to_le_bytes()).collect();
        if !bytes.is_empty() {
            self.queue.write_buffer(&b, 0, &bytes);
        }
        b
    }
    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
        buffers: &[&wgpu::Buffer],
        groups: usize,
    ) {
        let pipeline = &self.pipelines[name];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(groups.max(1) as u32, 1, 1);
    }
    pub(super) fn execute(&mut self, f: &Value, args: &[Arg]) -> Result<u64> {
        self.prepare(f, args)?;
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = (|| -> Result<u64> {
            let xs = list(f, args)?;
            let mut cap = xs.len().max(1);
            let mut data = self.upload(xs);
            let mut length = self.upload(&[xs.len() as u32]);
            let parameters = self.upload(
                &args
                    .iter()
                    .map(|v| if let Arg::Scalar(n) = v { *n as u32 } else { 0 })
                    .collect::<Vec<_>>(),
            );
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let mut resources = vec![data.clone(), length.clone(), parameters.clone()];
            for stage in f["gpu"]["stages"].as_array().unwrap() {
                let output = self.buffer(cap * 4);
                let outlen = self.buffer(4);
                let groups = cap.div_ceil(256);
                if stage["kind"] == "map" {
                    self.dispatch(
                        &mut encoder,
                        stage["shader"].as_str().unwrap(),
                        &[&data, &output, &length, &outlen, &parameters],
                        groups,
                    );
                } else {
                    let prefixes = self.buffer(cap * 4);
                    let counts = self.buffer(groups * 4);
                    let offsets = self.buffer(groups * 4);
                    self.dispatch(
                        &mut encoder,
                        stage["shader"].as_str().unwrap(),
                        &[
                            &data,
                            &output,
                            &length,
                            &outlen,
                            &parameters,
                            &prefixes,
                            &counts,
                        ],
                        groups,
                    );
                    self.dispatch(
                        &mut encoder,
                        "filter-offsets.wgsl",
                        &[&counts, &offsets, &outlen],
                        groups.div_ceil(64),
                    );
                    self.dispatch(
                        &mut encoder,
                        "filter-scatter.wgsl",
                        &[&data, &output, &length, &prefixes, &offsets],
                        groups,
                    );
                    resources.extend([prefixes, counts, offsets]);
                }
                resources.extend([output.clone(), outlen.clone()]);
                data = output;
                length = outlen;
            }
            if f["gpu"]["aggregate"] == "count" {
                let output = self.buffer(4);
                self.dispatch(&mut encoder, "count.wgsl", &[&length, &output], 1);
                data = output;
            } else {
                loop {
                    let groups = cap.div_ceil(256);
                    let output = self.buffer(groups * 4);
                    let outlen = self.buffer(4);
                    self.dispatch(
                        &mut encoder,
                        "sum.wgsl",
                        &[&data, &output, &length, &outlen],
                        groups,
                    );
                    resources.extend([output.clone(), outlen.clone()]);
                    data = output;
                    length = outlen;
                    cap = groups;
                    if cap == 1 {
                        break;
                    }
                }
            }
            let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&data, 0, &readback, 0, 4);
            let submission = self.queue.submit([encoder.finish()]);
            let (sender, receiver) = std::sync::mpsc::channel();
            readback.map_async(wgpu::MapMode::Read, .., move |r| {
                let _ = sender.send(r);
            });
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: Some(Duration::from_secs(10)),
                })
                .map_err(|e| e.to_string())?;
            receiver
                .recv_timeout(Duration::from_secs(1))
                .map_err(|e| e.to_string())?
                .map_err(|e| e.to_string())?;
            let bytes = readback.get_mapped_range(..).map_err(|e| e.to_string())?;
            let value = u32::from_le_bytes(bytes[..4].try_into().unwrap());
            drop(bytes);
            readback.unmap();
            Ok(value as u64)
        })();
        let validation = pollster::block_on(scope.pop());
        let allocation = pollster::block_on(oom.pop());
        if let Some(error) = validation.or(allocation) {
            return Err(error.to_string());
        }
        result
    }
}
