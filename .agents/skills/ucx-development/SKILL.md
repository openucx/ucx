---
name: ucx-development
description: Develop, build, and test UCX code. Use when asked to change UCX internals, compile UCX, run UCX tests, or report verification coverage.
---

# UCX Development

## Overview

Use this skill for changing UCX itself. Durable code rules live in `AGENTS.md`
files and project docs; this skill defines the end-to-end development workflow,
including build and test commands.

## Required Reading

Before editing or judging code:

- Follow the `AGENTS.md` discovery rule from the root guide.
- Read `docs/CodeStyle.md`, `docs/LoggingStyle.md`, or
  `docs/OptimizationStyle.md` when the change touches style, logging, or
  performance-sensitive code.
- Read `REVIEW.md` when matching UCX reviewer expectations matters.

## Edit Workflow

1. Identify the owning subtree and nearby implementation pattern before
   editing.
2. Apply the rules from the applicable `AGENTS.md` files and project docs.
3. Make the code or test change.
4. Build UCX or explain why the local environment cannot build it.
5. Run focused tests that cover the changed behavior.
6. Report the build/test commands used and any hardware, optional dependency,
   or coverage gaps.

## Build

The two main helpers are `configure-devel` and `configure-release`: pick `configure-devel` for failure investigation, debug-oriented and feature work (logging, assertions, gtest, and valgrind enabled), `configure-release` for performance work (those checks disabled and optimized for speed).

From a fresh checkout, create an out-of-source build:

```sh
./autogen.sh
mkdir -p build-<flavor>
cd build-<flavor>
../contrib/configure-<flavor> --prefix=$PWD/install-<flavor>
make -j$(nproc)
make install
```

Run `autogen.sh` and configure once per build directory. For incremental
builds, run `make -j$(nproc)` again from `build-<flavor>`; use `make clean`
before a fresh rebuild when necessary.

Prefer repository configure helpers over ad hoc flag sets. If a dependency or optional transport is unavailable, report the missing capability instead of editing around it.

When build configuration details matter, inspect `autogen.sh`,
`contrib/configure-devel`, `configure.ac`, and `config/m4`.

## Unit Tests

From `build-devel`, run the C++ unit test target with:

```sh
make -C test/gtest test
```

Use `make check` when broader automake test coverage is needed.

## Result Interpretation

- Capture the command, failing test name, relevant log lines, and environment
  variables.
- Distinguish test failures from unavailable optional dependencies or hardware.
- Do not hide flakes behind broad reruns. If a rerun is useful, say whether the
  failure reproduced.
