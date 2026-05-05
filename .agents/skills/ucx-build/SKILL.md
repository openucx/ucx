---
name: ucx-build
description: Configure and build UCX using the repository-supported autotools flow. Use when asked to bootstrap, configure, compile, rebuild after source changes, or choose UCX build commands.
---

# UCX Build

## Overview

Use this skill for UCX bootstrap, configure, and build commands.

## Standard Developer Build

From a fresh checkout:

```sh
./autogen.sh
./contrib/configure-devel --prefix=$PWD/install-debug
make -j8
```

Prefer the existing configure helpers over inventing ad hoc flag sets. If a
dependency or optional transport is unavailable, report the missing capability
instead of editing around it.

When build configuration details matter, inspect `autogen.sh`,
`contrib/configure-devel`, `configure.ac`, and `config/m4`.
