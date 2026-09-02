# Agent Guide: test/gtest

The main UCX correctness suite. Built as a single `gtest` binary that
loads the in-tree `libucs`/`libuct`/`libucp`. Driven from
`Makefile.am` — only built when `--enable-gtest` is configured (set by
`contrib/configure-devel`).

## Layout

- `common/` — shared harness:
  - `test.h`/`test.cc` — `ucs::test_base` (the C\+\+ fixture all tests
    inherit), `scoped_log_handler`, config push/pop helpers,
    skip/abort plumbing.
  - `test_helpers.[h|cc]` — assertion macros (`UCS_TEST_SKIP_R` etc.),
    fork helpers, pseudo-random generators, and other shared utilities.
  - `mem_buffer.[h|cc]` — multi-memtype buffer abstraction (host, CUDA,
    ROCm, ZE) used everywhere a test needs to vary memory type.
  - `test_obj_size.cc`, `test_perf.[h|cc]`, `test_watchdog.cc` — invariant
    checks (struct sizes), inline perf assertions, hang detection.
  - `main.cc` — gtest main + UCX-specific argument parsing.
  - `googletest/` — vendored gtest (do not modify).
- `ucs/` — unit tests for `src/ucs` subsystems. Roughly one `.cc` file
  per subsystem (`test_async.cc`, `test_rcache.cc`, `test_pgtable.cc`,
  …). `arch/` covers CPU-specific helpers; `test_module/` exercises
  module loading.
- `uct/` — UCT transport tests:
  - `uct_test.[h|cc]`, `uct_p2p_test.[h|cc]` — parametrized fixtures
    that iterate every available transport.
  - `ib/`, `sm/`, `tcp/`, `cuda/`, `v2/` — transport-specific tests.
  - Top-level `test_amo*.cc`, `test_p2p_*.cc`, `test_md.cc`, etc. share
    the parametrized fixtures.
- `ucp/` — UCP feature tests organized by area (`test_ucp_tag*`,
  `test_ucp_rma*`, `test_ucp_proto*`, `test_ucp_wireup.cc`,
  `test_ucp_sockaddr.cc`, …). `ucp_test.[h|cc]` is the base fixture and
  `ucp_datatype.[h|cc]` provides the dt parametrization. `cuda/` for
  GPU-specific cases.
- `ucm/` — memory-event hook tests, including a dedicated `test_dlopen`
  subdir with a separate Makefile target.

## Conventions

- Tests inherit from `ucs::test_base` (or a higher-level fixture like
  `uct_test` / `ucp_test`); never call gtest's `TEST(...)` directly with
  no fixture — the base class sets up the UCX config sandbox.
- Use the test-base config helpers (`set_config`, `modify_config`,
  `push_config` / `pop_config`) instead of `setenv` to avoid leaking
  state across tests.
- `mem_buffer` is the canonical way to allocate test buffers — it
  honors the parametrized memory type and frees through the right
  allocator. Don't `malloc` in tests that should run for GPU memtypes.
- most fixtures support memtype skipping via
  `check_skip_test()` patterns; prefer `UCS_TEST_SKIP_R(...)` over silent
  short-circuits so the runner reports skipped cases.
- New tests must be listed in the relevant `Makefile.am` source list.
  The Makefile already adds `lsan.supp`/`valgrind.supp` and ASAN env
  setup — don't redefine those locally.
- The default env (`UCX_HANDLE_ERRORS=freeze`) freezes a process on
  fatal error so a debugger can attach. Don't rely on test-process
  termination for cleanup.
- `GTEST_FILTER=...` and `GTEST_EXTRA_ARGS=...` are the supported knobs
  for running a subset; use them via `make test`.
- Use `UCS_TEST_MESSAGE` to print useful diagnostic information during
  a test.
- Tests that emit error or warning log messages will fail. Wrap any
  intentional error path with `scoped_log_handler` to suppress those
  messages while still letting the test run.
- Each test must clean up every resource it allocates before completing
  — leaks count as failures.

## Pointers

- Run a test: `make -C test/gtest test GTEST_FILTER=test_ucp_tag*`.
- Style: `docs/CodeStyle.md` (UCX C\+\+ idioms differ subtly from
  upstream gtest examples).
- Sister test trees: `test/apps/` (standalone binaries),
  `test/mpi/` (MPI-launched correctness).
