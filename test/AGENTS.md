# Agent Guide for `test`

This subtree contains UCX tests. The main suite is `test/gtest`, written in
C++11 on top of GoogleTest.

## Test Layout

- `test/gtest/ucs`: tests for UCS services and data structures.
- `test/gtest/uct`: transport and transport API tests.
- `test/gtest/ucp`: high-level protocol tests.
- `test/gtest/ucm`: memory hook and allocator interception tests.
- `test/gtest/common`: shared test helpers and GoogleTest integration.
- `test/apps`: standalone app tests.
- `test/mpi`: MPI-oriented tests.

## Adding or Editing Tests

- Add new gtest source files to `test/gtest/Makefile.am`.
- Add or adjust tests for behavior changes. Regression tests should fail
  without the fix whenever practical.
- Keep tests close to the layer being changed. A UCP behavior change usually
  belongs under `test/gtest/ucp`, not in a lower-layer transport test.
- Prefer deterministic tests. Avoid sleeps, timing thresholds, and reliance on
  specific hardware unless the test is already in a hardware-specific area.
- Reuse helpers from `test/gtest/common` and existing fixtures before adding a
  new fixture shape.
- Before adding component-specific tests, ask whether generic infrastructure
  such as `test_md`, `mem_buffer`, or `rkey_ptr` can cover the case.
- Use `mem_buffer` for generic memory allocation/comparison across memory
  types, and `ucs::handle<T>` / `UCS_TEST_CREATE_HANDLE` for RAII resources.
