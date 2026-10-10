export function createDevice({root, enabled, onLost, fail, width, unpack, MAX_BYTES}) {
  let device=null, gpuError=enabled?null:"GPU disabled by host option", lost=false;
  const pipelines=new Map();
  async function prepare(p) {
    if (p.reason) fail(p.reason);
    if (gpuError) fail(gpuError);
    if (lost) fail("GPU device lost");
    if (!device) {
      if (!globalThis.navigator?.gpu) fail("WebGPU unavailable");
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) fail("WebGPU adapter unavailable");
      device = await adapter.requestDevice();
      device.lost.then(() => {
        lost = true;
        gpuError = "GPU device lost";
        onLost();
      });
    }
    for (const s of p.steps) {
      if (s.args.length + 2 > device.limits.maxStorageBuffersPerShaderStage)
        fail("GPU binding limits");
      if (
        Math.ceil(s.count / 64) > device.limits.maxComputeWorkgroupsPerDimension
      )
        fail("GPU workgroup limits");
      if (!pipelines.has(s.f.shader)) {
        device.pushErrorScope("validation");
        device.pushErrorScope("out-of-memory");
        let pipeline, error;
        try {
          const code = await (await fetch(new URL(s.f.shader, root))).text();
          const bindings = device.createBindGroupLayout({
            entries: Array.from(
              { length: s.args.length + 2 },
              (_, binding) => ({
                binding,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                  type:
                    binding === s.args.length ? "storage" : "read-only-storage",
                },
              }),
            ),
          });
          pipeline = await device.createComputePipelineAsync({
            layout: device.createPipelineLayout({
              bindGroupLayouts: [bindings],
            }),
            compute: {
              module: device.createShaderModule({ code }),
              entryPoint: "main",
            },
          });
        } catch (e) {
          error = e;
        }
        const oom = await device.popErrorScope(),
          validation = await device.popErrorScope();
        if (error || oom || validation)
          fail(String(error ?? oom?.message ?? validation?.message));
        pipelines.set(s.f.shader, pipeline);
      }
    }
  }
  async function runGpu(p) {
    const owned = [];
    device.pushErrorScope("validation");
    device.pushErrorScope("out-of-memory");
    let result, error;
    try {
      const make = (bytes, usage, words) => {
        bytes = Math.max(4, bytes);
        if (
          bytes > device.limits.maxBufferSize ||
          (usage & GPUBufferUsage.STORAGE &&
            bytes > device.limits.maxStorageBufferBindingSize)
        )
          fail("GPU buffer limits");
        const b = device.createBuffer({
          size: bytes,
          usage,
          mappedAtCreation: !!words,
        });
        owned.push(b);
        if (words) {
          new Uint32Array(b.getMappedRange()).set(words);
          b.unmap();
        }
        return b;
      };
      const input = p.constants.map((c) =>
        make(c.words.byteLength, GPUBufferUsage.STORAGE, c.words),
      );
      const banks = [0, 1].map(() =>
        p.steps.map((s) =>
          make(
            s.count * width(s.f.result) * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          ),
        ),
      );
      const metadata = p.steps.map((s) =>
        make(
          (s.args.length + 1) * 4,
          GPUBufferUsage.STORAGE,
          Uint32Array.from([...s.args.map((a) => a.count), s.count]),
        ),
      );
      const groups = p.steps.map((s, i) =>
        [0, 1, 2].map((phase) => {
          const bank = phase === 2 ? 1 : 0;
          const entries = s.args.map((a, binding) => {
            let buffer =
              a.kind === "step" ? banks[bank][a.index] : input[a.index];
            if (a.input && phase !== 0 && p.feedback.has(a.input))
              buffer = banks[1 - bank][p.feedback.get(a.input)];
            return { binding, resource: { buffer } };
          });
          entries.push(
            { binding: s.args.length, resource: { buffer: banks[bank][i] } },
            { binding: s.args.length + 1, resource: { buffer: metadata[i] } },
          );
          return device.createBindGroup({
            layout: pipelines.get(s.f.shader).getBindGroupLayout(0),
            entries,
          });
        }),
      );
      const encoder = device.createCommandEncoder();
      for (let iter = 0; iter < p.iterations; iter++) {
        const phase = iter === 0 ? 0 : iter % 2 ? 2 : 1;
        for (let i = 0; i < p.steps.length; i++) {
          const s = p.steps[i];
          if (!s.count) continue;
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipelines.get(s.f.shader));
          pass.setBindGroup(0, groups[i][phase]);
          pass.dispatchWorkgroups(Math.ceil(s.count / 64));
          pass.end();
        }
      }
      const reads = p.outputs.map((id) => {
        const i = p.ids.get(id),
          bytes = p.steps[i].count * width(p.steps[i].f.result) * 4;
        const b = make(
          bytes,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        );
        if (bytes)
          encoder.copyBufferToBuffer(
            banks[(p.iterations - 1) % 2][i],
            0,
            b,
            0,
            bytes,
          );
        return { id, b, bytes };
      });
      device.queue.submit([encoder.finish()]);
      const values = {};
      for (const { id, b, bytes } of reads) {
        await b.mapAsync(GPUMapMode.READ);
        values[id] = unpack(
          new Uint32Array(b.getMappedRange(), 0, bytes / 4).slice(),
          p.steps[p.ids.get(id)].f.result,
        );
        b.unmap();
      }
      result = {
        values,
        counters: {
          upload_bytes:
            p.constants.reduce((n, c) => n + c.words.byteLength, 0) +
            p.steps.reduce((n, s) => n + (s.args.length + 1) * 4, 0),
          readback_bytes: reads.reduce((n, r) => n + r.bytes, 0),
          dispatches: p.steps.filter((s) => s.count).length * p.iterations,
          live_buffer_bytes: p.bytes,
        },
      };
    } catch (e) {
      error = e;
    } finally {
      const oom = await device.popErrorScope(),
        validation = await device.popErrorScope();
      for (const b of owned) b.destroy();
      if (error || oom || validation)
        fail(String(error ?? oom?.message ?? validation?.message));
    }
    if (lost) fail("GPU device lost");
    return result;
  }

  return {prepare, run:runGpu, close(){device?.destroy();lost=true;gpuError="engine disposed";pipelines.clear();}};
}
