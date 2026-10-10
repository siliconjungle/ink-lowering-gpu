// Version 2: packed logical values, compiled CPU fallback, resident GPU pipelines.
const MAX_BYTES = 256 * 1024 * 1024,
  MAX_STEPS = 64,
  MAX_DISPATCHES = 65536;
const fail = (s) => {
  throw Error(s);
};
const array = (v) =>
  Array.isArray(v) || (ArrayBuffer.isView(v) && typeof v.length === "number");
const width = (t) =>
  t.kind === "list"
    ? width(t.element)
    : t.kind === "vector"
      ? t.size * width(t.element)
      : t.kind === "record"
        ? t.fields.reduce((n, f) => n + width(f.type), 0)
        : t.kind === "u64"
          ? 2
          : 1;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
function pack(v, t) {
  const words = [],
    view = new DataView(new ArrayBuffer(4));
  let finite = true;
  function go(v, t) {
    switch (t.kind) {
      case "list":
        if (!array(v)) fail("expected list");
        for (const x of v) go(x, t.element);
        break;
      case "vector":
        if (!array(v) || v.length !== t.size) fail("vector dimension mismatch");
        for (const x of v) go(x, t.element);
        break;
      case "record":
        if (
          !v ||
          Array.isArray(v) ||
          typeof v !== "object" ||
          Object.keys(v).length !== t.fields.length
        )
          fail("record fields mismatch");
        for (const f of t.fields) {
          if (!Object.hasOwn(v, f.name)) fail("missing record field");
          go(v[f.name], f.type);
        }
        break;
      case "bool":
        if (typeof v !== "boolean") fail("expected Bool");
        words.push(+v);
        break;
      case "u32":
      case "i32":
        if (
          !Number.isInteger(v) ||
          v < (t.kind === "i32" ? -2147483648 : 0) ||
          v > (t.kind === "i32" ? 2147483647 : 4294967295)
        )
          fail(`${t.kind} outside range`);
        words.push(v >>> 0);
        break;
      case "u64": {
        if (typeof v === "number" && !Number.isSafeInteger(v))
          fail("u64 above safe JSON number range requires decimal string");
        let n;
        try {
          n = BigInt(v);
        } catch {
          fail("expected u64");
        }
        if (n < 0n || n > 18446744073709551615n) fail("u64 outside range");
        words.push(Number(n & 4294967295n), Number(n >> 32n));
        break;
      }
      case "f32":
        if (typeof v === "number" && Number.isFinite(v)) {
          view.setFloat32(0, v, true);
          if (!Number.isFinite(view.getFloat32(0, true)))
            fail("f32 input overflow");
          words.push(view.getUint32(0, true));
        } else if (
          v &&
          Object.keys(v).length === 1 &&
          Number.isInteger(v.F32Bits) &&
          v.F32Bits >= 0 &&
          v.F32Bits <= 4294967295
        ) {
          view.setUint32(0, v.F32Bits, true);
          finite &&= Number.isFinite(view.getFloat32(0, true));
          words.push(v.F32Bits);
        } else fail("expected finite f32 or explicit F32Bits");
        break;
      default:
        fail("unsupported packed type");
    }
  }
  go(v, t);
  if (words.length * 4 > MAX_BYTES) fail("input exceeds 256 MiB budget");
  return {
    words: Uint32Array.from(words),
    type: t,
    count: t.kind === "list" ? v.length : 1,
    finite,
  };
}
function unpack(words, t) {
  let i = 0;
  const view = new DataView(new ArrayBuffer(4));
  function go(t) {
    switch (t.kind) {
      case "list": {
        const n = words.length / width(t.element);
        if (!Number.isInteger(n)) fail("invalid result layout");
        return Array.from({ length: n }, () => go(t.element));
      }
      case "vector":
        return Array.from({ length: t.size }, () => go(t.element));
      case "record":
        return Object.fromEntries(t.fields.map((f) => [f.name, go(f.type)]));
      case "bool":
        return words[i++] !== 0;
      case "u32":
        return words[i++];
      case "i32":
        return words[i++] | 0;
      case "u64": {
        const n = BigInt(words[i++]) + (BigInt(words[i++]) << 32n);
        return n <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(n) : n.toString();
      }
      case "f32":
        view.setUint32(0, words[i++], true);
        return Number.isFinite(view.getFloat32(0, true))
          ? view.getFloat32(0, true)
          : { F32Bits: view.getUint32(0, true) };
      default:
        fail("invalid result type");
    }
  }
  return go(t);
}
function compilePlan(plan, manifest) {
  if (
    !plan ||
    !Array.isArray(plan.steps) ||
    !plan.steps.length ||
    plan.steps.length > MAX_STEPS
  )
    fail("pipeline requires 1..64 steps");
  const iterations = plan.iterations ?? 1;
  if (
    !Number.isInteger(iterations) ||
    iterations < 1 ||
    iterations > 10000 ||
    iterations * plan.steps.length > MAX_DISPATCHES
  )
    fail("pipeline iteration/dispatch limit");
  const constants = [],
    inputs = new Map(),
    steps = [],
    ids = new Map();
  let reason = null;
  const constant = (v, t) => {
    const a = pack(v, t);
    constants.push(a);
    return {
      kind: "constant",
      index: constants.length - 1,
      type: t,
      count: a.count,
    };
  };
  for (const step of plan.steps) {
    if (typeof step.id !== "string" || !step.id || ids.has(step.id))
      fail("duplicate or invalid step id");
    const f = manifest.functions.find((f) => f.name === step.call);
    if (!f) fail(`unknown function ${step.call}`);
    if (!Array.isArray(step.args) || step.args.length !== f.params.length)
      fail("argument count mismatch");
    const args = step.args.map((v, i) => {
      const t = f.params[i].type;
      if (
        v &&
        typeof v === "object" &&
        !Array.isArray(v) &&
        Object.keys(v).length === 1 &&
        Object.hasOwn(v, "step")
      ) {
        const previous = ids.get(v.step);
        if (previous === undefined)
          fail("step reference must name an earlier step");
        const s = steps[previous];
        if (!same(t, s.f.result)) fail("step reference type mismatch");
        return { kind: "step", index: previous, type: t, count: s.count };
      }
      if (
        v &&
        typeof v === "object" &&
        !Array.isArray(v) &&
        Object.keys(v).length === 1 &&
        Object.hasOwn(v, "input")
      ) {
        if (
          typeof v.input !== "string" ||
          !Object.hasOwn(plan.inputs ?? {}, v.input)
        )
          fail("unknown pipeline input");
        if (!inputs.has(v.input))
          inputs.set(v.input, constant(plan.inputs[v.input], t));
        const ref = inputs.get(v.input);
        if (!same(t, ref.type))
          fail("pipeline input used with different types");
        return { ...ref, input: v.input };
      }
      return constant(v, t);
    });
    const count = f.shader
      ? Math.min(...f.lengths.map((i) => args[i].count ?? 0))
      : null;
    if (!f.shader) reason ??= f.fallback_reason ?? "CPU-only function";
    ids.set(step.id, steps.length);
    steps.push({ id: step.id, f, args, count });
  }
  const outputs = plan.outputs ?? [steps.at(-1).id];
  if (
    !Array.isArray(outputs) ||
    !outputs.length ||
    new Set(outputs).size !== outputs.length ||
    outputs.some((x) => !ids.has(x))
  )
    fail("invalid pipeline outputs");
  const feedback = new Map();
  for (const [input, id] of Object.entries(plan.feedback ?? {})) {
    if (!inputs.has(input) || !ids.has(id))
      fail("unknown feedback input or step");
    const ref = inputs.get(input),
      step = steps[ids.get(id)];
    if (!same(ref.type, step.f.result)) fail("feedback type mismatch");
    if (ref.count !== step.count)
      reason ??= "feedback changes array shape; compiled CPU pipeline";
    feedback.set(input, ids.get(id));
  }
  if (constants.some((c) => !c.finite))
    reason ??= "non-finite f32 inputs require CPU";
  let bytes = constants.reduce(
    (n, c) => n + Math.max(4, c.words.byteLength),
    0,
  );
  bytes += steps.reduce(
    (n, s) =>
      n +
      2 * Math.max(4, (s.count ?? 0) * width(s.f.result) * 4) +
      4 * (s.args.length + 1),
    0,
  );
  bytes += outputs.reduce(
    (n, id) =>
      n +
      Math.max(
        4,
        (steps[ids.get(id)].count ?? 0) *
          width(steps[ids.get(id)].f.result) *
          4,
      ),
    0,
  );
  if (bytes > MAX_BYTES) reason ??= "GPU live-buffer budget exceeds 256 MiB";
  return {
    constants,
    inputs,
    steps,
    outputs,
    ids,
    feedback,
    iterations,
    reason,
    bytes,
  };
}
const median = (a) => a.sort((a, b) => a - b)[1];
export async function loadInk(
  base,
  { gpu = true, backend: defaultBackend = "auto" } = {},
) {
  const root = new URL(base, globalThis.location?.href ?? "http://localhost/");
  if (!root.pathname.endsWith("/")) root.pathname += "/";
  const manifest = await (await fetch(new URL("manifest.json", root))).json();
  if (manifest.schema !== 2) fail("expected schema 2 compute bundle");
  const wasm = await WebAssembly.instantiate(
    await (await fetch(new URL("cpu.wasm", root))).arrayBuffer(),
    {},
  );
  const cpu = wasm.instance.exports;
  let device = null,
    gpuError = gpu ? null : "GPU disabled by host option",
    lost = false,
    tail = Promise.resolve();
  const pipelines = new Map(),
    profiles = new Map();
  function cpuCall(f, args) {
    cpu.ink_compute_reset();
    try {
      const ptrs = cpu.ink_alloc(args.length, 4),
        lengths = cpu.ink_alloc(args.length, 4),
        header = cpu.ink_alloc(2, 4);
      for (let i = 0; i < args.length; i++) {
        const a = args[i],
          ptr = cpu.ink_alloc(Math.max(1, a.words.length), 4);
        new Uint32Array(cpu.memory.buffer, ptr, a.words.length).set(a.words);
        new DataView(cpu.memory.buffer).setUint32(ptrs + 4 * i, ptr, true);
        new DataView(cpu.memory.buffer).setUint32(
          lengths + 4 * i,
          a.count,
          true,
        );
      }
      cpu.ink_compute_call(f.index, ptrs, lengths, header, header + 4);
      const view = new DataView(cpu.memory.buffer),
        ptr = view.getUint32(header, true),
        count = view.getUint32(header + 4, true);
      if (count * 4 > MAX_BYTES) fail("CPU output exceeds 256 MiB budget");
      const words = new Uint32Array(cpu.memory.buffer, ptr, count).slice();
      return {
        words,
        type: f.result,
        count: f.result.kind === "list" ? count / width(f.result.element) : 1,
      };
    } finally {
      cpu.ink_compute_reset();
    }
  }
  function runCpu(p) {
    const inputs = new Map(p.inputs),
      values = [];
    for (let iteration = 0; iteration < p.iterations; iteration++) {
      for (let i = 0; i < p.steps.length; i++) {
        const s = p.steps[i];
        const args = s.args.map((a) =>
          a.input ? resolve(inputs.get(a.input)) : resolve(a),
        );
        values[i] = cpuCall(s.f, args);
      }
      for (const [input, step] of p.feedback)
        inputs.set(input, { kind: "value", value: values[step] });
    }
    function resolve(a) {
      return a.kind === "constant"
        ? p.constants[a.index]
        : a.kind === "value"
          ? a.value
          : values[a.index];
    }
    return {
      values: Object.fromEntries(
        p.outputs.map((id) => [
          id,
          unpack(values[p.ids.get(id)].words, p.steps[p.ids.get(id)].f.result),
        ]),
      ),
      counters: { upload_bytes: 0, readback_bytes: 0, dispatches: 0 },
    };
  }
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
        profiles.clear();
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
  async function execute(plan, { backend = defaultBackend } = {}) {
    if (!["auto", "cpu", "gpu"].includes(backend))
      fail("backend must be auto, cpu or gpu");
    const p = plan;
    if (backend === "cpu" || p.reason)
      return {
        ...runCpu(p),
        backend: "cpu",
        reason: backend === "cpu" ? "explicit CPU selection" : p.reason,
      };
    let key = JSON.stringify([
      p.steps.map((s) => [
        s.f.index,
        s.args.map((a) => [a.kind, a.index, a.count]),
      ]),
      p.iterations,
      [...p.feedback],
      p.outputs,
      p.constants.map((c) => [
        c.type,
        c.count,
        Array.from(c.words.subarray(0, 8)),
      ]),
    ]);
    let setup = performance.now();
    try {
      await prepare(p);
      setup = performance.now() - setup;
      if (backend === "gpu")
        return {
          ...(await runGpu(p)),
          backend: "gpu",
          reason: "explicit GPU selection",
        };
      let profile = profiles.get(key);
      if (!profile || profile.uses >= 32) {
        const ct = [],
          gt = [];
        for (let trial = 0; trial < 3; trial++) {
          const c = () => {
            const t = performance.now();
            runCpu(p);
            ct.push(performance.now() - t);
          };
          const g = async () => {
            const t = performance.now();
            await runGpu(p);
            gt.push(performance.now() - t);
          };
          if (trial % 2) {
            await g();
            c();
          } else {
            c();
            await g();
          }
        }
        profile = {
          cpu_ms: median(ct),
          gpu_ms: median(gt),
          setup_ms: setup,
          uses: 0,
        };
        profile.gpu = profile.gpu_ms + setup / 32 < profile.cpu_ms * 0.9;
        if (profiles.size >= 32) profiles.delete(profiles.keys().next().value);
        profiles.set(key, profile);
      }
      profile.uses++;
      return {
        ...(profile.gpu ? await runGpu(p) : runCpu(p)),
        backend: profile.gpu ? "gpu" : "cpu",
        reason:
          "measured whole pipeline including transfers; setup amortised over 32 uses, 10% minimum saving",
        profile: { ...profile },
      };
    } catch (e) {
      profiles.delete(key);
      return {
        ...runCpu(p),
        backend: "cpu",
        reason: `GPU fallback: ${e.message ?? e}`,
      };
    }
  }
  function serial(fn) {
    const task = tail.then(fn);
    tail = task.catch(() => {});
    return task;
  }
  function submit(plan, options) {
    let prepared;
    try {
      prepared = compilePlan(plan, manifest);
    } catch (e) {
      return Promise.reject(e);
    }
    return serial(() => execute(prepared, { ...options }));
  }
  const runtime = {
    manifest,
    pipeline: submit,
    call: (name, args, options) =>
      submit(
        { steps: [{ id: "result", call: name, args }], outputs: ["result"] },
        options,
      ).then((r) => ({ ...r, value: r.values.result })),
    dispose: () =>
      serial(() => {
        device?.destroy();
        lost = true;
        gpuError = "engine disposed";
        pipelines.clear();
        profiles.clear();
      }),
  };
  runtime.run = runtime.call;
  runtime.close = runtime.dispose;
  return runtime;
}
