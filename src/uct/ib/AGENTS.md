# Agent Guide: src/uct/ib

InfiniBand transport family. The biggest UCT family by code volume and the
default fast-path for HPC/RDMA workloads. Each subdir is built as a
separate UCX module (its own `Makefile.am` and `configure.m4`) and dlopened
at runtime.

## Subdirectory Map

- `base/` — shared IB primitives consumed by every IB transport:
  - `ib_device.[ch]` — port enumeration, link-layer detection, GID/LID,
    `ibv_context` lifecycle.
  - `ib_md.[ch]` — IB memory domain (registration, rkey packing, ODP, DMABUF).
  - `ib_iface.[ch]` — common iface state (CQ, AH cache, QP attrs).
  - `ib_log.[ch]` — verbs/WC pretty-printing for traces.
  - `ib_verbs.h` — compatibility shim across `libibverbs` versions.
- `mlx5/` — Mellanox/NVIDIA fast path. Uses Direct Verbs (DV) plus DEVX to
  bypass libibverbs for performance:
  - `dv/` — DV-based MD (`ib_mlx5dv_md.c`), `ib_mlx5_ifc.h` PRM definitions.
  - `rc/` — RC over mlx5 (DV/DEVX QPs).
  - `ud/` — UD over mlx5.
  - `dc/` — Dynamically-Connected transport (Mellanox-only). DC ep + iface
    plus DEVX management (`dc_mlx5_devx.c`).
  - `gga/` — Generic GPU-DMA accelerator (`gga_mlx5.c`).
  - `gdaki/` — GPU Direct Async / Kernel Initiated; integrates with
    DOCA-GPUNetIO via the `gpunetio` git submodule. Includes a
    CUDA-syntax header (`gdaki.cuh`) consumed by GPUNetIO-side code.
  - `ib_mlx5.[ch]`/`.inl` and `ib_mlx5_log.*` — shared mlx5 helpers.
- `rc/` — verbs-based Reliable Connected transport, used when DV is
  unavailable: `base/` (shared `rc_iface`/`rc_ep`/`rc_def.h`) + `verbs/`.
- `ud/` — verbs-based Unreliable Datagram transport: `base/` + `verbs/`.
- `efa/` — AWS Elastic Fabric Adapter (SRD). Subdirs: `base/`, `srd/`.
- `rdmacm/` — sockaddr connection manager (`librdmacm`). Provides UCT CM
  for IB and RoCE via TCP/IP rendezvous.

## Conventions

- Each module contributes a `uct_component_t`. Capability discovery happens
  in `ib_device.c` and per-transport `*_iface_query` — capability flags
  must match the `uct_iface_ops_t` populated for that iface.
- `mlx5` modules degrade gracefully: if DV/DEVX support is missing at
  configure time, only `rc/verbs` and `ud/verbs` are built.
- New IB transports go under their own subdir with `configure.m4` + a
  `Makefile.am` whose `xxx.la` is added to module flags. Do not add
  IB-specific code to `src/uct/Makefile.am` — that file only enumerates
  the always-built `sm` and `tcp` transports.
- DEVX/PRM structures are versioned via `ib_mlx5_ifc.h`; when adding new
  commands match the kernel/firmware ABI rather than copying ad-hoc layouts.
- `ib_md` registration: prefer the existing helpers
  (`uct_ib_reg_mr` / `uct_ib_dereg_mr` in `ib_md.c`) — they handle ODP,
  multi-MR, and DMABUF correctly. Recent fixes around DMABUF offsets live
  in `ib_md.c`.
- DMABUF FD ownership: when `mem_reg`/`mem_advise` accepts a DMABUF FD, the
  caller retains ownership. Register paths must compute the offset from
  the original mapping base, not the registration base.
- `rdmacm` CM wraps `rdma_cm_event_t` callbacks and has its own state
  machine — keep it self-contained.

## Pointers

- High-level selection: `src/ucp/wireup/select.c` queries IB ifaces.
- Tests: `test/gtest/uct/ib/` (transport-specific) and `test/gtest/uct/`
  (parametrized fixtures iterate IB transports).
- Mock harness for IB: `contrib/ibmock/` enables CI runs without real HW.
- Build flags: `contrib/configure-devel` enables DEVX/DV by default; check
  `configure.m4` files in each subdir for the relevant `--with-*` knobs.
