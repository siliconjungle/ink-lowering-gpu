# Ink GPU lowering

Literal WGSL emission and WebGPU/wgpu device adapters for Ink's checked pure programs. `gpu::lower(&CheckedModule)` returns a manifest and shader artifacts. It performs capability checks, without selecting optimisation laws, assembling CPU fallback, invoking toolchains or scheduling mixed targets.

`shaders/` owns stable filter/reduction primitives. `devices/` owns buffer allocation, uploads, dispatch, pipelines, validation, device loss and readback for the versioned scalar and packed compute runtime ABIs. Browser adapters expose `createDevice`; native adapters are included as a device module in generated projects. Host scheduling, profiling, Wasm/native C fallback and bundle assembly live in [ink-runtime](https://github.com/siliconjungle/ink-runtime). This package depends only on the semantic core and serialization/hash libraries, not on C or runtime.

Supported scalar kernels use u32 map/filter followed by sum/count; unsupported functions retain a CPU artifact through runtime orchestration. Packed compute supports multiple typed arrays, vector/record layouts, indexed reads and bounded loops. Order-sensitive reductions and unsupported primitives remain on CPU. Portable f32 follows WGSL evaluation rules; CPU/GPU bit identity is not promised.

The checker, literal C/WGSL emission, packed ABI, shader primitives, driver and target toolchains remain trusted. No end-to-end backend correctness proof or all-device performance guarantee is claimed. See Ink's `docs/gpu-backend.md` and `docs/gpu-compute.md` for supported semantics and limits.
