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

From a fresh checkout:

```sh
./autogen.sh
./contrib/configure-devel --prefix=$PWD/install-debug
make -j8
```

Prefer repository configure helpers over ad hoc flag sets. If a dependency or
optional transport is unavailable, report the missing capability instead of
editing around it.

When build configuration details matter, inspect `autogen.sh`,
`contrib/configure-devel`, `configure.ac`, and `config/m4`.

## Result Interpretation

- Capture the command, failing test name, relevant log lines, and environment
  variables.
- Distinguish test failures from unavailable optional dependencies or hardware.
- Do not hide flakes behind broad reruns. If a rerun is useful, say whether the
  failure reproduced.
