# Agent Guide for UCX

This file is for coding agents working in this repository. It summarizes the
project shape and points to the source-of-truth docs that should be followed
when changing code.

## Agentic File Budget

- Keep this root `AGENTS.md` under about 120 lines. It should summarize
  universal rules and point to detailed resources, not duplicate them.
- Keep subtree `AGENTS.md` files under about 100 lines and focused on local
  ownership, commands, and pitfalls.
- Put repeatable workflows, review procedures, and extended examples in
  agent skills or regular docs, then link to them from the relevant guide.

## Agent Skills

Repository skills live under `.agents/skills/<skill-name>/SKILL.md`. Keep
`.agents` as the canonical location and add tool-specific adapters only when a
tool requires them.

## Project Map

UCX is a C communication framework with C++ unit tests. The major runtime
layers are:

- `src/ucs`: common services, data structures, logging, stats, config, async,
  memory, and platform utilities.
- `src/uct`: low-level transports and transport APIs for IB, TCP, shared memory,
  GPU transports, and related hardware backends.
- `src/ucp`: high-level protocol layer for tag matching, streams, RMA, AM,
  rendezvous, wireup, endpoint/context/worker logic, and protocol selection.
- `src/ucm`: memory event hooks and allocation interception.
- `src/tools`: command-line tools such as `ucx_perftest`, `ucx_info`, profiling,
  and VFS helpers.
- `test/gtest`: main C++ unit and integration test suite.
- `test/apps`, `test/mpi`: application-level and MPI-oriented tests.
- `examples`: small programs that demonstrate public APIs.
- `docs`: Sphinx, Doxygen, style, and user documentation.
- `bindings`: Go and Java bindings over UCX APIs.
- `buildlib`: build, packaging, and CI helper scripts.
- `config`: autotools helpers and m4 feature checks.
- `debian`: Debian packaging metadata and scripts.

## Common Workflows

Use repository skills for repeatable workflows:

- `.agents/skills/ucx-build/SKILL.md` for configuring and building UCX.

## Source-of-Truth Docs

Follow these project docs instead of duplicating their contents:

- `docs/CodeStyle.md` for C/C++ formatting and naming.
- `docs/LoggingStyle.md` for log levels and message style.
- `docs/OptimizationStyle.md` for performance-sensitive changes.
- `.github/CONTRIBUTING.md` for contribution workflow expectations.
