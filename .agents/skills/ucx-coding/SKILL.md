---
name: ucx-coding
description: Implement or review UCX source changes according to project architecture, public API rules, naming, style, error handling, abstraction patterns, performance, build, and testing rules. Use when asked to change UCX internals, fix bugs, add features, add transports, extend APIs, refactor C/C++ code, add tests, update Makefile.am, write commits, or judge what good UCX code looks like.
---

# UCX Coding

## Overview

Use this skill for UCX code changes. It composes the build and testing skills
with UCX architecture, style, layering, ABI, and performance expectations.

## Sources

Use these as needed:

- `AGENTS.md`
- Relevant subtree `AGENTS.md`
- `docs/CodeStyle.md`
- `docs/LoggingStyle.md`
- `docs/OptimizationStyle.md`
- `.github/CONTRIBUTING.md`
- `REVIEW.md` when matching reviewer expectations matters

## Architecture

Preserve dependency direction:

- UCP may use UCT and UCS.
- UCT may use UCS and must not use UCP.
- UCS must not use UCT or UCP.
- UCM must remain independent of UCT and UCP.

Place code by responsibility:

- `ucs/`: portable utilities, data structures, logging, memory, atomics, async,
  config, and platform support.
- `uct/`: transport abstraction and transport implementations.
- `ucp/`: user-facing protocol logic such as wireup, rendezvous, AM, tag, RMA,
  streams, endpoints, workers, and flow control.
- `ucm/`: memory hooks and allocator interception.

When adding a shared utility, prefer UCS. When adding transport-agnostic
protocol behavior, prefer UCP. Keep transport-private logic inside the matching
transport directory.

## Public API And Headers

Treat public API and ABI as compatibility-sensitive. Stable public API lives in
`src/ucp/api/ucp.h`, `src/uct/api/uct.h`, and the headers they include. Add to
public API headers only when application or middleware code needs it and it
cannot be expressed by existing APIs.

- Use the extensible struct pattern: `field_mask` is the first field of params
  or attr structs, new optional fields get new `UCS_BIT(n)` enum values, and
  bits are never removed or renumbered.
- Do not change public field types or reorder public struct fields.
- Put deprecated symbols in the compatibility headers with the local
  deprecation pattern; do not remove public symbols casually.
- Keep internal shared headers within their layer and transport-private headers
  within their transport. Do not include transport-private headers across
  transport boundaries or from higher layers.
- Public API changes need Doxygen comments and should include only the minimal
  non-API changes needed to compile.

## Naming And Style

- Use layer prefixes consistently: `ucs_`, `uct_`, `ucp_`, `ucm_` for
  functions/types and uppercase equivalents for macros and enum constants.
- Static file-private helpers do not need a layer prefix when local style does
  not use one.
- Types use `_t`, opaque handles use `_h`, function pointer types use
  `_func_t`, and ops tables use `_ops_t`.
- Function names follow `<layer>_<object>_<verb>` and should describe what the
  function does, not how it happens internally.
- Output parameters go last and use `_p`.
- Use `ucs_status_t` for UCX failures and name the variable `status`. Reserve
  `ret` for plain integer return codes.
- In C code, declare local variables at the beginning of the function body.
- Prefer `sizeof(*ptr)` or `sizeof(variable)` over `sizeof(type)`.
- Use `ucs_container_of` and `ucs_derived_of` instead of open-coded pointer
  arithmetic.
- Use `ucs_likely` and `ucs_unlikely` only where branch hints protect hot paths.
- Use `static UCS_F_ALWAYS_INLINE` or local inline macros for hot-path header
  helpers; do not use plain `inline`.
- Keep formatting churn out of unrelated code. UCX often aligns declarations,
  assignments, and struct fields intentionally.

Commit messages usually follow `COMPONENT/SUBCOMPONENT: Imperative message`,
for example `UCP/CORE: Fix endpoint flush completion` or `GTEST/UCP: Add rndv
error test`.

## Error Handling And Logging

- Functions that can fail should return `ucs_status_t` or a documented UCX
  status-pointer form. Do not encode UCX errors as ad hoc `-1` values.
- Distinguish `UCS_OK`, `UCS_INPROGRESS`, and negative `UCS_ERR_*` statuses.
- Use cleanup labels in reverse initialization order, named for what they undo
  such as `err_free_desc` or `err_cleanup_md`.
- Set `status` before every cleanup `goto`.
- Only set output parameters after success.
- Use UCX logging macros, not `printf` or raw `assert`.
- Choose log levels according to `docs/LoggingStyle.md`: expected fallback or
  back-pressure must not be logged as an error.
- Use `ucs_assert` or `ucs_assertv` for invariants, `ucs_assert_always` for
  production-critical invariants, and `ucs_fatal` only for truly unrecoverable
  conditions.

## Abstraction Patterns

- Reuse existing helpers before adding a new abstraction. Search nearby code
  and shared UCS helpers first.
- Prefer functions over macros when logic is involved.
- For UCT transport polymorphism, use ops tables and typed function pointers.
  Avoid conditionals in hot-path ops when separate ops tables or setup-time
  selection can encode the mode.
- For UCX object-style code, keep `super` as the first member, call the super
  constructor first, clean up in reverse order, and use `ucs_derived_of` for
  downcasts.
- Config tables should use string defaults, descriptive help strings, and
  `ucs_offsetof`. User-facing config environment names use the `UCX_` prefix.
- Preserve symmetry: create/destroy, init/cleanup, pack/unpack, and
  enable/disable pairs should be named and implemented consistently.

## Memory And Performance

- Use UCX allocation primitives such as `ucs_malloc`, `ucs_calloc`, and
  `ucs_free`, with meaningful allocation tags.
- Use memory pools for fixed-size objects allocated on hot paths.
- Do not use stack allocation for unbounded sizes.
- Keep memory-type handling explicit; do not assume host memory in data paths.
- Avoid allocations, locks, atomics, logging, string formatting, and device
  queries in hot paths unless the local code already pays that cost.
- Cache expensive capability queries when the result is stable.
- Treat wire formats, endpoint configuration, request layout, and public struct
  layout as compatibility-sensitive.

## Change Scope And Tests

- Keep changes scoped to the requested behavior. Do not mix bug fixes,
  refactors, formatting churn, and feature work unless explicitly asked.
- Do not commit generated build output, Sphinx output, Doxygen XML/HTML, or
  local install directories.
- Avoid broad formatting churn in mature code paths.
- Add or adjust tests for behavior changes. Regression tests should fail without
  the fix whenever practical.
- Update the relevant `Makefile.am` when adding, removing, or renaming files.
- Treat hardware-specific paths conservatively. If a change affects IB, CUDA,
  ROCm, Level Zero, Gaudi, or shared memory transports, report any untested
  hardware assumptions.

## Edit Workflow

1. Identify the owning layer and read nearby code before editing.
2. Search for existing helpers and similar tests.
3. Make the smallest code change that preserves local patterns.
4. Update the relevant `Makefile.am` for new, removed, or renamed files.
5. Use `ucx-build` for compile verification and `ucx-testing` for focused tests.
6. Report any tests or hardware coverage that could not be run locally.
