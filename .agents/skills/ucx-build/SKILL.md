---
name: ucx-build
description: Configure, build, and rebuild UCX. Picks the right `contrib/configure-*` helper and additional `--enable-*` / `--with-*` flags based on the goal of the build (development, release, profiling, gtest, CUDA, IB, bindings, etc.). Use when asked to bootstrap, configure, compile, rebuild, run tests, or troubleshoot a UCX build.
---

# UCX Build

## Bootstrap (once per fresh checkout)

```sh
./autogen.sh
./contrib/configure-<flavor> [extra flags]
make -j
```

These three steps run only on a fresh tree. The recipes below show flavor + flag combinations for that initial setup.

## Incremental rebuilds (the common case)

After source edits — `.c`, `.h`, `.inl`, `.cc`, even `Makefile.am` or `configure.m4` — just rebuild:

```sh
make -j                    # if you didn't install, or just running gtest
make -j && make install    # if you previously installed to --prefix
```

Autotools auto-regenerates `Makefile`/`configure` when their inputs change, so you do **not** re-run `autogen.sh` or `./configure` for ordinary edits. Re-run them only after:
- pulling new m4 macros or new `config/m4/*.m4` files,
- changing the configure flavor (e.g. switching from `devel` to `opt`),
- moving to a different build directory,
- a major build-system change (rare).

## Pick the right `contrib/configure-*` helper

Don't invent flag combinations — start from one of these and append extras through `"$@"`. They all live in `contrib/`.

| Goal | Helper |
|---|---|
| Develop, debug, run gtest, run under valgrind | `configure-devel` |
| Production performance, portable | `configure-release` |
| Production performance + multi-thread support | `configure-release-mt` |
| Microbench on this exact CPU (non-portable) | `configure-opt` |
| Profiling / perf debugging (frame pointers, samples) | `configure-prof` |

What each helper actually turns on:

- **`configure-devel`** — `--enable-gtest --enable-examples
  --enable-test-apps --enable-stats --enable-profiling
  --enable-frame-pointer --enable-debug-data --enable-mt
  --with-valgrind=guess`. Logging, assertions, and params-check are at their defaults (enabled). It does **not** pass `--enable-debug`; add that explicitly if you want `-D_DEBUG` and `-O0`.
- **`configure-release`** — `--disable-logging --disable-debug
  --disable-assertions --disable-params-check`. C is built with `-O3`, C\+\+ with `-O0` (UCX convention; tests are the main C\+\+ consumer).
- **`configure-release-mt`** — `configure-release` + `--enable-mt`.
- **`configure-opt`** — `configure-release` + `--enable-optimizations` (adds `-march=native`-style CPU-specific flags). Don't ship the result.
- **`configure-prof`** — `--disable-logging --disable-debug --disable-assertions --disable-params-check --enable-profiling --enable-frame-pointer --enable-stats`. Like `configure-release` but keeps profiling and frame pointers; assertions stay off.

## Add flags based on the goal

Append these to the helper. Pick by purpose, not by habit.

### Tests, examples, dev binaries

- `--enable-gtest` — needed for `make -C test/gtest test`. See
  `test/gtest/AGENTS.md` for harness conventions and run options (`GTEST_FILTER=...`).
- `--enable-test-apps` — builds `test/apps/` standalone binaries.
- `--enable-examples` — builds `examples/`.

(All three are already on in `configure-devel`.)

### Logging / asserts / param checks

- `--enable-logging[=LEVEL]` — caps the *compiled-in* log level. LEVEL ∈ `no, warn, diag, info, debug, trace, trace_data, trace_async,  trace_func, trace_poll`. Default is the most verbose.
- `--disable-logging` — strips everything below DEBUG.
- `--enable-assertions` / `--disable-assertions`.
- `--enable-debug-data` — bigger object headers, more debug info in packets and structs. Used heavily by gtest assertions; significant  runtime cost.

### Profiling / perf

- `--enable-profiling` — turns `ucs_profile_*` macros into real samples (read with `src/tools/profile/read_profile`).
- `--enable-stats` — `UCX_STATS_*` runtime counters and the `ucs_stats_parser` binary.
- `--enable-frame-pointer` — keeps `-fno-omit-frame-pointer` so external profilers (perf, vtune) get good stacks.
- `--enable-optimizations` — non-portable host-CPU flags.

### Sanitizers / coverage

- `--enable-asan` — AddressSanitizer.
- `--enable-gcov` — coverage instrumentation.

### Threading

- `--enable-mt` — required for `UCS_THREAD_MODE_MULTI` / multi-threaded
  UCP.

### CUDA / GPU

- `--with-cuda[=DIR]` — default `guess`. Disable with `--without-cuda`.
- `--with-nvcc-gencode='-gencode=arch=compute_90,code=sm_90'` — limit GPU archs to speed up NVCC; default builds many. Big build-time win when you only care about one device.
- `--with-gdrcopy[=DIR]` — GPUDirect copy (`uct_gdr_copy`).
- `--with-gda[=DIR]` — GPU Direct Async / Kernel-Initiated (mlx5 GDAKI).
- `--with-rocm[=DIR]` — AMD HIP.
- `--with-ze[=DIR]` — Intel Level Zero.
- `--with-gaudi[=DIR]` — Habana.

When working on a GPU transport, also consult `src/uct/cuda/AGENTS.md` (or `rocm/`/`ze/`) for transport-specific knobs.

### InfiniBand / RDMA

- `--with-verbs[=DIR]` — libibverbs (default `/usr`).
- `--with-mlx5` / `--with-rc` / `--with-ud` / `--with-dc` — default `yes`
  / `guess`.
- `--with-devx` — DEVX (Mellanox direct command path), default `check`.
- `--with-ib-hw-tm` — IB hardware tag matching.
- `--with-rdmacm[=DIR]` — sockaddr CM.
- `--with-efa[=DIR]` — AWS EFA.

See `src/uct/ib/AGENTS.md`.

### Shared memory / sockaddr / FUSE

- `--enable-cma` — Linux cross-memory-attach.
- `--with-knem[=DIR]`, `--with-xpmem[=DIR]` — alternative SM kernels.
- `--with-fuse3[=DIR]` — FUSE for `ucx_vfs`.

### Bindings

- `--with-go[=DIR]`, `--with-java[=DIR]` — see `bindings/AGENTS.md`.

### Misc

- `--with-valgrind[=DIR|guess]` — valgrind client requests in libucs.
- `--with-bfd[=DIR|guess]` — BFD-based detailed backtraces.
- `--with-mad[=DIR]` — IB MAD-based perftest control plane.
- `--enable-experimental-api` — installs `ucp/api/ucpx.h`.
- `--enable-devel-headers` — installs internal headers.

## Recipes

`make install` is needed when something outside the build tree consumes
UCX (bindings, OMPI, an installed `ucx_perftest`, downstream projects).
It is **not** needed for `test/gtest`, `examples/`, or `test/apps/` —
they build and run from the build tree directly. As a rule of thumb: if
the recipe sets `--prefix=...`, follow `make` with `make install`.

```sh
# Fresh dev tree, build, install
./autogen.sh && ./contrib/configure-devel --prefix=$PWD/install-debug && \
    make -j && make install

# Dev iteration without installing (gtest/examples run from build tree)
./contrib/configure-devel && make -j

# Dev build + true debug (-O0, _DEBUG)
./contrib/configure-devel --enable-debug && make -j

# Faster dev build by limiting CUDA archs to the host GPU
./contrib/configure-devel --with-nvcc-gencode='-gencode=arch=compute_90,code=sm_90' && \
    make -j

# Just gtest (no install needed)
./contrib/configure-devel && make -j -C test/gtest test GTEST_FILTER='test_ucp_tag*'

# Microbench build for this machine (non-portable), installed for use
./contrib/configure-opt --prefix=$PWD/install-opt && make -j && make install

# Profiling-friendly (frame pointers, profile samples)
./contrib/configure-prof --prefix=$PWD/install-prof && make -j && make install

# Production multi-thread (system-wide install)
./contrib/configure-release-mt --prefix=/opt/ucx && make -j && sudo make install

# Build without a flaky transport
./contrib/configure-devel --without-rocm && make -j

# Pin to a specific CUDA install
./contrib/configure-devel --with-cuda=/opt/cuda-12.4 && make -j
```

## When something fails

- If a probe fails (`configure: error:` or "build will proceed without X"), report the missing capability rather than masking it with `--without-X`. Disable a transport only if the user explicitly wants it gone.
- For commands inside a subtree (e.g. `make -C test/gtest`), check that subtree's `AGENTS.md` first — most list the expected commands and env-var conventions.
- After a failed build, fix the root cause; don't pass `--no-verify`-style shortcuts.

## Sources of truth

- `autogen.sh`, `configure.ac`, `config/m4/*.m4`, and per-component `src/*/configure.m4` define every option (search there before
  inventing flags).
- `contrib/configure-*` are the canonical helper combinations.
- `contrib/AGENTS.md` covers the dev-script ecosystem around them.