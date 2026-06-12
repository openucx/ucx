# Agent Guide: src/uct/cuda

NVIDIA GPU transport family. `cuda_copy`, `cuda_ipc`, and the shared
`base/` helpers are built into a single `uct_cuda` module; `gdr_copy` is
a separate optional module. Modules are dlopened at runtime, so the core
library never links CUDA directly. A CUDA-capable build (`--with-cuda`)
is required and the modules degrade silently when prerequisites are
absent.

## Subdirectory Map

- `base/` — shared CUDA helpers:
  - `cuda_iface.[ch]` and `cuda_md.[ch]` — common iface/MD scaffolding.
  - `cuda_ctx.[ch]`/`.inl` — primary-context handling, stream pools.
  - `cuda_nvml.[ch]` — NVML queries (topology, NVLink connectivity).
  - `cuda_util.[ch]` — driver-API wrappers, error translation,
    pointer-attribute lookup.
- `cuda_copy/` — device↔host and intra-process device↔device copies via
  the CUDA driver API (`cuMemcpyAsync`, async streams). Default GPU
  staging path.
- `cuda_ipc/` — inter-process device memory sharing using `cuIpcGetMemHandle`
  / `cuIpcOpenMemHandle`. Includes a per-handle cache (`cuda_ipc_cache.[ch]`)
  to amortize the open/close cost. `cuda_ipc.cuh` is a CUDA-syntax header
  pulled in by callers that need device-side declarations.
- `gdr_copy/` — GPUDirect RDMA via the gdrcopy library. Maps device memory
  into host VA space for low-latency CPU-driven transfers. Built as its
  own module only when `--with-gdrcopy` is enabled (own `configure.m4`
  and `Makefile.am`).

## Conventions

- All driver-API calls must be funneled through `cuda_util.h` wrappers so
  errors are logged consistently and the `CUcontext` is set/popped
  correctly. Direct calls to `cuCtxPushCurrent` outside `base/` are a
  smell.
- Memory-type classification of pointers happens through the global
  `ucs_memtype_cache` (`src/ucs/memory/memtype_cache.c`); CUDA modules
  feed it via the `ucm/cuda` event hooks rather than calling
  `cuPointerGetAttributes` per request on the fast path.
- IPC (`cuda_ipc_cache`) entries are keyed per-handle and invalidated on
  the matching `cudaFree` event from `ucm/cuda`. When invalidating, drop
  the full cache entry — never a partial mapping.
- `gdr_copy` registration is configurable via `UCX_GDR_COPY_*` env vars
  parsed in `gdr_copy_md.c` (notably `RCACHE`, `MEM_REG_OVERHEAD`,
  `MEM_REG_GROWTH`). Pinning falls back across `GDR_PIN_FLAG_*` modes,
  not to a different transport, so respect the flags rather than
  expecting an automatic switch to `cuda_copy`.
- DMABUF support: recent upstream fixes (`2582d8d2e` UCT/CUDA: Fix dmabuf
  offset for interior addresses, `50c59cd0c` UCT/GDA: Fix dmabuf offset)
  tightened offset handling for interior addresses and the GDA path.
  When adding DMABUF to a CUDA module, cross-reference those commits.

## Pointers

- Memory-event source: `src/ucm/cuda/cudamem.c` feeds `MEM_TYPE_ALLOC`/`FREE`
  to `ucs/memory/memtype_cache.c`. Without UCM, IPC cache invalidation
  cannot fire.
- Sister GPU families: `src/uct/rocm/` (HIP), `src/uct/ze/` (Level Zero),
  `src/uct/gaudi/`. Mirror their conventions for cross-vendor parity.
- Tests: `test/gtest/uct/cuda/`, `test/gtest/ucp/cuda/`, plus the
  parametrized fixtures.
- Tool: `ucx_perftest` GPU plug-in lives at `src/tools/perf/cuda/`.
