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
- Keep tests close to the layer being changed. A UCP behavior change usually
  belongs under `test/gtest/ucp`, not in a lower-layer transport test.
- Bug-fix tests should fail without the fix whenever practical.
- Prefer deterministic tests. Avoid sleeps, timing thresholds, and reliance on
  specific hardware unless the test is already in a hardware-specific area.
- Reuse helpers from `test/gtest/common` and existing fixtures before adding a
  new fixture shape.
