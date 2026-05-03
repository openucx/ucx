---
name: ucx-testing
description: Select, run, and interpret focused UCX tests after builds or code changes. Use when asked to test UCX changes, choose GTEST_FILTER values, run gtest, app, MPI, docs, binding, or hardware-specific tests, inspect failures, or explain local test coverage.
---

# UCX Testing

## Overview

Use this skill to choose focused UCX verification commands and interpret their
results. It assumes the tree has already been configured and built, or uses the
`ucx-build` skill to get there.

## Sources

Use these as needed:

- `test/AGENTS.md`
- Relevant subtree `AGENTS.md`
- `test/gtest/Makefile.am` for supported variables and test organization
- Existing nearby tests before adding or changing coverage

## Common Commands

```sh
make -C test/gtest test
make -C test/gtest test GTEST_FILTER='ucs_*'
make -C test/gtest test GTEST_FILTER='uct_*'
make -C test/gtest test GTEST_FILTER='ucp_*'
make -C test/gtest test GTEST_FILTER='*tag*'
```

For docs or bindings, use the commands in `docs/AGENTS.md` or
`bindings/AGENTS.md`.

## Test Selection

- Prefer the smallest test surface that covers the changed behavior.
- For bug fixes, add or run a regression test that would fail without the fix
  whenever practical.
- For UCP protocol changes, favor targeted `ucp_*` filters and any specific
  feature filter visible in nearby tests.
- For UCT transport changes, run generic UCT coverage plus the relevant
  transport-specific tests when local hardware supports them.
- For UCS utilities, run focused `ucs_*` tests.
- For hardware-specific changes, clearly state untested hardware assumptions.

## Result Interpretation

- Capture the command, failing test name, relevant log lines, and environment
  variables.
- Distinguish test failure from unavailable optional dependency or hardware.
- Do not hide flakes behind broad reruns. If a rerun is useful, say whether the
  failure reproduced.
- Use `UCX_LOG_LEVEL`, `GTEST_FILTER`, `GTEST_EXTRA_ARGS`, `LAUNCHER`, and
  related variables documented in `test/AGENTS.md` before inventing wrappers.
