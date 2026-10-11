# Ink GPU lowering

Literal WGSL emission and WebGPU/wgpu device adapters for Ink's checked pure programs. `gpu::lower(&CheckedModule)` returns a manifest and shader artifacts. It performs capability checks, without selecting optimisation laws, assembling CPU fallback, invoking toolchains or scheduling mixed targets.

`shaders/` owns stable filter/reduction primitives. `devices/` owns buffer allocation, uploads, dispatch, pipelines, validation, device loss and readback for the versioned scalar and packed compute runtime ABIs. Browser adapters expose `createDevice`; native adapters are included as a device module in generated projects. Host scheduling, profiling, Wasm/native C fallback and bundle assembly live in [ink-runtime](https://github.com/siliconjungle/ink-runtime). This package depends only on the semantic core and serialization/hash libraries, not on C or runtime.

Supported scalar kernels use u32 map/filter followed by sum/count; unsupported functions retain a CPU artifact through runtime orchestration. Packed compute supports multiple typed arrays, vector/record layouts, indexed reads and bounded loops. Literal packed stage graphs additionally support materialized map/indexed map/filter/zip/concat/scan stages and ordered scalar sum/count/fold/lookup consumers. Map/indexed/zip nodes run in parallel; stable filter, scan, concat and ordered reductions currently use one GPU invocation. List length headers remain resident between stages, including across mixed function pipelines; buffers are reused across requested iterations. Fold execution visits elements last-to-first and sum visits first-to-last without tree reassociation. No source stages are fused by this backend.

Direct-shader descriptors retain their fast path; optional graph descriptors allow complete mixed graph execution. Resident graph feedback and stable u32/i32 rank sorting execute literally. Simultaneous feedback captures every source before updating any input, including aliased identity outputs. Heavy graph operations retain the 128-element device bound. 64-bit element arithmetic and lazy collection branches explicitly fall back. Bounded known helper traversal preserves argument evaluation and lexical scopes. Portable WGSL floating-point allowances still apply; source grouping and stage stores do not establish CPU/GPU bit identity on all devices. Browser and native adapters report actual upload/readback/dispatch/peak resident buffer bytes, including internal list headers. Unsupported primitives remain on CPU. Portable f32 follows WGSL evaluation rules; CPU/GPU bit identity is not promised.

The checker, literal C/WGSL emission, packed ABI, shader primitives, driver and target toolchains remain trusted. No end-to-end backend correctness proof or all-device performance guarantee is claimed. See Ink's `docs/gpu-backend.md` and `docs/gpu-compute.md` for supported semantics and limits.

Current complete execution coverage, GPU exclusions and integration evidence are documented in the [lowering parity contract](https://github.com/siliconjungle/ink-lang/blob/main/docs/lowering-parity.md).
Eligible flat 32-bit array kernels include composed map/filter/zip, inclusive integer scan, stable rank sort, lazy list choice and bounded list repeat. These literal algorithms have work/expansion admission limits and can select CPU. Device buffers carry actual length through resident pipelines.

When literal collection work has a supported stage graph, emission selects that eager graph rather than a quadratic per-output direct traversal. Unsupported graph forms can retain the bounded direct shader path; direct list transport uses `ink-gpu-list-v3` headers.

Standalone proof-selected representations use the compile-time checked witness
and emit literal functions with native state retained between calls. Generated
applications carry no proof package, checker context, source AST or Ink runtime
dependency. See ink-lang/docs/database-representations.md and the shared
`representation_parity` conformance gate. Recursive GPU state uses the complete
host lane; device eligibility remains separate.


Generated entry points trust the caller's declared argument types, arity, shapes,
tags and canonical ranges. Bad data is unsafe and may crash, trap, corrupt state
or produce incorrect results; raw native pointers also require valid alignment,
length and lifetime. No malformed-input diagnostic or rollback is promised.
Representation conversion/JSON parsing and explicit portable checkpoint integrity
checks remain. Source/proof checking happens during compilation. Standalone
representation calls skip whole-state checkpoint validation; read-only observation
envelopes are erased when existing definitional computation establishes unchanged
logical state/version and no events.
