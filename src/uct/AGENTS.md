# Agent Guide: src/uct

UCT is the low-level Unified Communication Transport layer (`libuct.so`). It
abstracts hardware/software transports behind a uniform component / memory
domain / interface / endpoint model that `ucp` builds on top of.

## Core Abstractions

Defined in `api/uct.h` and `api/uct_def.h`; implemented per-transport.

- **Component** (`uct_component_t`) — registry entry for a transport family
  (e.g. `ib`, `cuda_copy`). Discovered via `uct_query_components()`.
- **Memory Domain / MD** (`uct_md_*`) — owns memory registration, allocation,
  and rkey packing for a component.
- **Interface / iface** (`uct_iface_*`) — an open communication endpoint set on
  a worker. Has capability flags (AM, PUT/GET, atomics, tag matching, …).
- **Endpoint / ep** (`uct_ep_*`) — connected peer reachable through an iface.
- **Worker** (`uct_worker_t`) — async progress context (one per UCP worker).
- **Connection Manager / CM** (`uct_cm_*`) — sockaddr-based listener/connect.
- **Operation flavors:** `_short` (inline), `_bcopy` (buffered copy with
  packer callback), `_zcopy` (zero-copy with completion).

## Subdirectory Map

- `api/` — public C headers (`uct.h`, `uct_def.h`, `tl.h`, `v2/uct_v2.h`,
  `device/`). Keep ABI and API stable.
- `base/` — shared implementation: `uct_component`, `uct_md`, `uct_iface`,
  `uct_cm`, `uct_worker`, `uct_mem`, VFS attribute exposure. Every transport
  inherits from these C++-style classes (`UCS_CLASS_*`).
- `ib/` — InfiniBand family: shared `base/`, `mlx5/` (DV-based fast path),
  `rc/` and `ud/` (verbs + mlx5 variants), `efa/`, `rdmacm/`. **See
  `ib/AGENTS.md` before editing.**
- `cuda/` — NVIDIA GPU transports: `cuda_copy`, `cuda_ipc`, `gdr_copy`,
  shared `base/`. **See `cuda/AGENTS.md`.**
- `rocm/` — AMD GPU: `copy/`, `ipc/`, `base/`.
- `ze/` — Intel Level Zero: `copy/`, `base/`.
- `gaudi/` — Habana Gaudi: `gaudi_gdr/`, `base/`.
- `sm/` — host shared-memory family: `mm/` (POSIX/SysV/xpmem), `scopy/`
  (CMA, KNEM), `self/` (loopback). Compiled into `libuct` directly (see
  `Makefile.am`).
- `tcp/` — sockets transport plus the sockaddr `tcp_sockcm` CM. Compiled into
  `libuct` directly.
- `ugni/` — Cray uGNI (`rdma`, `smsg`, `udt`); legacy/optional.

## Conventions

- New transports plug in as a **module** under one of the families with their
  own `configure.m4` and `Makefile.am`. Module `.so`s are dlopened by
  `ucs/sys/module.c`; nothing in `libuct` should `dlsym` them directly.
- `sm/` and `tcp/` are the exceptions — both contribute sources to
  `libuct_la_SOURCES` directly. `sm/` is also listed in `SUBDIRS` so its
  `mm`/`scopy` submodules can build as separate modules; `tcp/` is not in
  `SUBDIRS`. See `Makefile.am`.
- Use `UCS_CLASS_DEFINE*` to derive a transport's `iface`/`ep`/`md` from the
  base classes. Capability flags advertised at iface construction must match
  the operations actually wired in the `uct_iface_ops_t` table — wrong flags
  cause UCP to pick a transport that then fails at runtime.
- Every operation must return `UCS_OK`, `UCS_INPROGRESS`, `UCS_ERR_NO_RESOURCE`
  (back-pressure), or another `ucs_status_t` per the API contract. Returning
  `UCS_ERR_NO_RESOURCE` requires that the request be re-driven from the
  pending queue.
- Headers and sources for a transport must be added in the family's
  `Makefile.am`, not the parent. The top-level `Makefile.am` only enumerates
  the IB-less sm/tcp built-ins.
- Logging: prefer `ucs_*` macros; transport-specific helpers in
  `ib/base/ib_log.h` and friends.

## Pointers

- High-level callers: `src/ucp/wireup/select.c` chooses transports;
  `src/ucp/core/ucp_worker.c` opens the workers/ifaces.
- Tests: `test/gtest/uct/` — the parametrized `uct_test`/`uct_p2p_test`
  fixtures iterate every available transport.
- Tools: `src/tools/info/tl_info.c` (`ucx_info -d`) prints transport
  capabilities; `src/tools/perf/` benchmarks them.
