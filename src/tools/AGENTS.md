# Agent Guide: src/tools

Command-line binaries shipped with UCX. Each subdir is an independent tool
with its own `Makefile.am`. Nothing in `src/tools` is linked from
`libucs`/`libuct`/`libucp` — the dependency direction is one-way.

## Subdirectory Map

- `info/` — `ucx_info`. Single binary (`ucx_info.c`) that introspects the
  build (`build_info.c`, `version_info.c`), platform (`sys_info.c`),
  installed transports (`tl_info.c`), public types (`type_info.c`), and the
  UCP protocol matrix (`proto_info.c`). Common helpers in `ucx_info.h`.
- `perf/` — `ucx_perftest`, the canonical microbenchmark tool plus its
  reusable library `libperf`:
  - `api/libperf.h` is the public API; `lib/libperf*` is the implementation
    (UCP and UCT test drivers, memory and threading helpers).
  - Top-level `perftest.c` + `perftest_run.c` + `perftest_params.c` is the
    CLI; `perftest_daemon.c` + `perftest_context.h` host the daemon mode.
  - GPU plug-ins under `cuda/`, `rocm/`, `ze/` provide host-managed device
    buffers; `mad/` adds InfiniBand MAD-based control plane.
- `profile/` — `read_profile`, the offline reader for samples produced by
  the `ucs/profile` runtime. One file (`read_profile.c`).
- `vfs/` — `ucx_vfs`, the FUSE daemon that mounts the per-process VFS tree
  exposed by `ucs/vfs/`. `vfs_main.c` is the entry point; `vfs_server.c`
  serves the Unix-socket protocol; `vfs_daemon.h` defines the wire format
  shared with `ucs/vfs/sock`.

## Conventions

- Each tool is an independent consumer of the public APIs and runs against
  whichever build it links against — either the in-tree build directory or
  an installed prefix.
- `ucx_perftest` adds tests by appending to the `tests[]` table in
  `perftest.h`/`.c`. New flags need a slot in `TEST_PARAMS_ARGS` (the
  `getopt` short-option string) and an entry in `TEST_PARAMS_ARGS_LONG`.
- `ucx_info -p` calls into `proto_info.c`, which pulls live data from
  `src/ucp/proto/*` — when the protocol-selection layer changes, this
  file usually needs an update.

## Pointers

- Build helpers and configure flags: `contrib/configure-*` (see
  `contrib/AGENTS.md`).
- VFS protocol on the library side: `src/ucs/vfs/`.
- Profile sampling on the library side: `src/ucs/profile/`.
