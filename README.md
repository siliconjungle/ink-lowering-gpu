# Ink GPU lowering

Shared literal WGSL lowering plus WebGPU (browser) and wgpu (native) runtimes.
The existing u32 map/filter and sum/count fragment retains each source stage.
All functions have a compiled C CPU fallback, compiled to Wasm in the browser.
Capability checks identify supported emission; they are not optimisation proofs.

The runtime measures allocation/upload/dispatch/readback costs before selecting
whole-function CPU or GPU execution. GPU intermediates remain resident within
its supported collection pipeline. Device loss, unsupported functions and limits
fall back to compiled CPU code. Arbitrary cross-backend graph execution is not
implemented yet; ink-core's checked pure routing interface is its foundation.

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
