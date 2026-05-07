# Agent Guide: src/ucs

Common-services library (`libucs.la`) used by every other UCX layer. Pure C,
no transport or protocol logic — only foundational utilities.

## Subdirectory Map

- `api/` — *(none; public headers are exposed directly from each subdir via
  `nobase_dist_libucs_la_HEADERS` in `Makefile.am`)*
- `algorithm/` — `crc`, `qsort_r`, `string_distance`.
- `arch/` — per-CPU primitives (`x86_64`, `aarch64`, `ppc64`, `rv64`, `generic`):
  atomics, bitops, cycle counter, `global_opts`, optimized memcpy.
- `async/` — async event dispatch, signal/thread/eventfd/pipe backends. Drives
  worker progress for higher layers.
- `config/` — runtime config parser (`UCX_*` env vars), INI loader, global opts.
- `datastruct/` — generic containers: `mpool`, `mpool_set`, `ptr_array`,
  `ptr_map`, `khash`, `list`/`hlist`/`queue`, `callbackq`, `arbiter`,
  `pgtable`, `bitmap`/`static_bitmap`/`dynamic_bitmap`, `frag_list`,
  `interval_tree`, `lru`, `mpmc`, `strided_alloc`, `string_buffer`,
  `string_set`, `array`, `usage_tracker`, `conn_match`,
  `linear_func`/`piecewise_func`, `sglib`.
- `debug/` — `assert`, `log`, `memtrack` (allocation accounting), `debug` (BFD
  symbol resolution, backtraces).
- `memory/` — `rcache` (registration cache), `memtype_cache`, `memory_type`
  (CPU/CUDA/ROCm/ZE classification), `numa`.
- `profile/` — sampling and on/off profiling primitives consumed by
  `tools/profile/read_profile`.
- `signal/` — signal handler installation built as a shared subdir for ordering.
- `stats/` — counter/timer infrastructure; gated by `HAVE_STATS`. Adds the
  `ucs_stats_parser` binary.
- `sys/` — OS abstraction: `event_set` (epoll), `sock`, `string`, `math`,
  `topo`, `module` (dlopen modules), `iovec`, `netlink`, `lib`, `uid`,
  `compiler*`, `device_code` (CUDA/HIP host-side helpers).
- `time/` — `time_def`/`time` (high-resolution clock), `timerq`, `timer_wheel`.
- `type/` — `class` (UCX OO macros), `status`, `spinlock`, `rwlock`,
  `init_once`, `cpu_set`, `thread_mode`, `param`, `serialize`, `float8`.
- `vfs/` — VFS object tree (`base/`) plus `sock` and `fuse` mount adapters
  (built as ordered SUBDIRS: `vfs/sock . vfs/fuse signal`).

## Conventions

- New headers and sources must be listed explicitly in `Makefile.am`
  (`nobase_dist_libucs_la_HEADERS` for installed, `noinst_HEADERS` for
  internal). Forgetting either side breaks dist or hides the file from the
  build.
- `AUTOMAKE_OPTIONS = nostdinc` — header lookups go through `-I` paths only,
  to avoid colliding with the bundled `debug.h`.
- `libucs` links `libucm` and `libBFD` (when available); do not introduce
  reverse dependencies into `uct`/`ucp`/transports.
- Use the `UCS_CLASS_*` macros from `type/class.h` for any reference-counted
  or virtual-dispatch object; do not roll your own vtables.
- Logging: prefer `ucs_*` log macros from `debug/log.h`; honor the level
  conventions in `docs/LoggingStyle.md`.
- Performance-sensitive helpers belong in `.inl` files alongside the matching
  `.h`. Follow `docs/OptimizationStyle.md`.

## Pointers

- Public API headers ship under `$(includedir)/ucs/...` — keep them ABI-stable.
- Tests: `test/gtest/ucs/` mirrors this layout one-test-per-subsystem.
- For style and logging rules see `docs/CodeStyle.md`,
  `docs/LoggingStyle.md`, `docs/OptimizationStyle.md`.
