# Agent Guide: src/ucm

UCM (`libucm.la`) intercepts memory-management events so the rest of UCX can
invalidate registration caches, track GPU allocations, and react to
`mmap`/`munmap`/`brk`/etc. without changes from the application. Public API
is `api/ucm.h`; everything else is internal.

## Subdirectory Map

- `api/` — `ucm.h`, the only installed header. Defines the
  `ucm_event_type_t` bitset (mmap/munmap/mremap/shmat/shmdt/sbrk/madvise/brk
  plus aggregate `VM_MAPPED`/`VM_UNMAPPED` and per-memtype
  `MEM_TYPE_ALLOC`/`MEM_TYPE_FREE`).
- `event/` — event dispatch core (`event.c`, `event.h`). Manages handler
  chains, priority ordering, and synthetic events generated for already-
  mapped memory.
- `malloc/` — `malloc_hook` glue and the in-tree `ptmalloc` allocator
  (sibling dir `ptmalloc286/`, gated by `HAVE_UCM_PTMALLOC286`). Replaces
  `malloc`/`free`/`realloc`/`memalign` so VM events fire on heap growth.
- `mmap/` — installation of mmap-family hooks; uses `bistro/` to patch the
  loader-resolved symbols when `LD_PRELOAD` is unavailable.
- `bistro/` — per-arch binary-instrumentation patcher
  (`bistro_x86_64.c`, `bistro_aarch64.c`, `bistro_ppc64.c`, `bistro_rv64.c`).
  Used to redirect glibc syscalls without `dlsym`.
- `util/` — supporting helpers: `replace.c` (PLT/GOT relocation),
  `reloc.c` (ELF relocation walking), `log.c`, `sys.c`, `khash_safe.h`.
- `cuda/` — `cudamem` hooks for `cudaMalloc`/`cudaFree` and friends. Built
  as a module; emits `MEM_TYPE_ALLOC`/`FREE` events.
- `rocm/` — analogous hooks for `hipMalloc`/`hipFree`.
- `ze/` — Intel Level Zero allocations.
- `ptmalloc286/` — third-party allocator, only built when in-tree malloc
  replacement is enabled.

## Conventions

- Three interception strategies are available: bistro binary patching
  (preferred), dynamic loader symbol replacement, and PLT override. Add
  new hooks to `event/` and prefer `bistro/` unless an alternative is
  clearly needed.
- The library defines `UCM_MALLOC_PREFIX=ucm_dl` (see `Makefile.am`) so its
  own malloc symbols don't collide with the application's libc.
- GPU allocator hooks must report into `event/` via the existing
  `MEM_TYPE_ALLOC`/`FREE` codes — `ucs/memory/memtype_cache` consumes them
  and decides per-pointer the memory type.
- Per-arch bistro code uses raw machine-code emission; `docs/CodeStyle.md`
  applies but expect inline comments describing instruction encodings.
- `CFLAGS_NO_DEPRECATED` is added to `libucm_la_CFLAGS`; do not rely on
  recent libc deprecations being warned about here.

## Pointers

- Consumers: `ucs/memory/rcache.c` (registration cache invalidation),
  `ucs/memory/memtype_cache.c`, the GPU UCT transports (`uct/cuda/*`,
  `uct/rocm/*`, `uct/ze/*`).
- Tests: `test/gtest/ucm/`, plus `test/apps/test_hooks.c`,
  `test/apps/test_cuda_hook.c`.
- Build: optional ptmalloc and per-GPU support is gated by `configure.m4`
  files in each subdir.
