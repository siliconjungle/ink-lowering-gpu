# Ink GPU lowering

Shared literal WGSL lowering plus WebGPU (browser) and wgpu (native) runtimes.
The existing u32 map/filter and sum/count fragment retains each source stage.
All functions have a compiled C CPU fallback, compiled to Wasm in the browser.
Capability checks identify supported emission; they are not optimisation proofs.

The runtime measures allocation/upload/dispatch/readback costs before selecting
whole-function CPU or GPU execution. GPU intermediates remain resident within
its supported collection pipeline. Device loss, unsupported functions and limits
fall back to compiled CPU code. Compiler-checked literal source graphs can now route calls across CPU/GPU:
`emit_routed` accepts only ink-core's immutable source-routing witness. Browser
`loadRoute` and native `Engine::call_route` execute the embedded graph with typed
word/Bool/list inputs and scalar host results. Native inputs are captured once
and shared immutably. Invalid original arguments reject before execution;
unavailable GPU or failed shader execution retains compiled CPU fallback.

`c/cpu` resolves to Wasm or native C; `gpu/gpu` resolves to WebGPU or wgpu.
Placement and complete-cost candidate selection are external knowledge tools.
Cross-function GPU-resident buffer bridges, effects/state placement and formal
physical transport/code-generation proofs remain unfinished.

Rust/C already compile to native machine code or Wasm through existing toolchains.
This package does not introduce architecture-specific assembly compilers.
Backend/host protocols, shader primitives, drivers and target toolchains remain
trusted. See ink-lang/docs/gpu-backend.md for supported semantics and ABI scope.

Pure compute v2 supports multiple typed arrays and array outputs, numeric
vectors/records, safe indexed reads and bounded loops. Explicit host pipelines
retain intermediate GPU buffers across steps and iterations. Whole-pipeline
measurements include transfer and host costs; unavailable or unsupported GPU
execution replays on the separately versioned compiled C backend.
Portable f32 follows WGSL evaluation rules; CPU/GPU bit identity is not promised.
See ink-lang/docs/gpu-compute.md for the versioned ABI, limits and evidence.

`gpu::emit_selected` emits the exact module accepted by the shared semantic selection checker. GPU capability and measured profitability remain separate from proof validity. The legacy mixed-route executor rejects compute-v2 transport rather than emitting an incompatible host ABI.
