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
    pub(super) fn prepare(&mut self, p: &Plan) -> Result<()> {
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
    pub(super) fn execute(&self, p: &Plan) -> Result<Value> {
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
