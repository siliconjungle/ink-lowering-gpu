# Ink GPU lowering

Literal WGSL emission and WebGPU/wgpu device adapters for Ink's checked pure programs. `gpu::lower(&CheckedModule)` returns a manifest and shader artifacts. It performs capability checks, without selecting optimisation laws, assembling CPU fallback, invoking toolchains or scheduling mixed targets.

`shaders/` owns stable filter/reduction primitives. `devices/` owns buffer allocation, uploads, dispatch, pipelines, validation, device loss and readback for the versioned scalar and packed compute runtime ABIs. Browser adapters expose `createDevice`; native adapters are included as a device module in generated projects. Host scheduling, saved-model selection, Wasm/native C fallback and bundle assembly live in [ink-runtime](https://github.com/siliconjungle/ink-runtime). This package depends on the semantic core, ink-abi and serialization/hash libraries, not on C or runtime.

Supported scalar kernels use u32 map/filter followed by sum/count; unsupported functions retain a CPU artifact through runtime orchestration. Packed compute supports multiple typed arrays, vector/record layouts, indexed reads and bounded loops. Literal packed stage graphs additionally support materialized map/indexed map/filter/zip/concat/scan stages and ordered scalar sum/count/fold/lookup consumers. Map/indexed/zip nodes run in parallel; stable filter, scan, concat and ordered reductions currently use one GPU invocation. List length headers remain resident between stages, including across mixed function pipelines; buffers are reused across requested iterations. Fold execution visits elements last-to-first and sum visits first-to-last without tree reassociation. No source stages are fused by this backend.

Direct-shader descriptors retain their fast path; optional graph descriptors allow complete mixed graph execution. Browser and native graph feedback preserve type/capacity bounds through simultaneous device snapshots, including aliased identity outputs; growing feedback falls back. Integer rank sort has a literal quadratic graph stage. 64-bit element arithmetic and lazy collection branches explicitly fall back. Bounded known helper traversal preserves argument evaluation and lexical scopes. Portable WGSL floating-point allowances still apply; source grouping and stage stores do not establish CPU/GPU bit identity on all devices. Browser and native adapters report actual upload/readback/dispatch/peak resident buffer bytes, including internal list headers. Unsupported primitives remain on CPU. Portable f32 follows WGSL evaluation rules; CPU/GPU bit identity is not promised.

The checker, literal C/WGSL emission, packed ABI, shader primitives, driver and target toolchains remain trusted. No end-to-end backend correctness proof or all-device performance guarantee is claimed. See Ink's `docs/gpu-backend.md` and `docs/gpu-compute.md` for supported semantics and limits.

Current complete execution coverage, GPU exclusions and integration evidence are documented in the [lowering parity contract](https://github.com/siliconjungle/ink-lang/blob/main/docs/lowering-parity.md).
Eligible flat 32-bit array kernels include composed map/filter/zip, inclusive integer scan, stable rank sort, lazy list choice and bounded list repeat. These literal algorithms have work/expansion admission limits and can select CPU. Device buffers carry actual length through resident pipelines.

When literal collection work has a supported stage graph, emission selects that eager graph rather than a quadratic per-output direct traversal. Unsupported graph forms can retain the bounded direct shader path; direct list transport uses `ink-gpu-list-v3` headers.

Packed browser device adapters additionally support fixed-input `capture(plan)`
sessions owned by the runtime host. Active required graphs and legacy shaders
reuse input/stage/readback buffers. Setup reports one upload; repeated runs
report actual dispatch/readback/live counters and zero upload. Dynamic list
lengths stay in headers. No feedback or public mutable device handle is exposed.
Host-owned serialization and close/device-loss boundaries remain explicit.

Private scalar buffers use fixed capacity 64, with bounded construction/push, split/rejoin, mapping and stopping loops. Complementary split provenance bounds dynamic rejoining; growing loop bounds and oversized independent appends explicitly fall back. See `tests/temporary_bounds.rs` and `tests/device_graph.mjs`.

All targets consume [ink-abi](https://github.com/siliconjungle/ink-abi) for shared
type/layout contracts. Early termination, owned-buffer transforms and helper
unfolding are checked database choices. The backend emits the resulting general
operations literally; it does not infer those transformations from source patterns.
