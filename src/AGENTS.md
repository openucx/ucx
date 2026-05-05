# Agent Guide for `src`

This subtree contains UCX runtime code. Keep source edits aligned with the
layer boundaries below.

## Layer Boundaries

- `ucs`: shared infrastructure only. Avoid pulling protocol or transport policy
  into UCS. UCS must not depend on UCT or UCP.
- `uct`: transport-facing primitives and hardware-specific implementations.
  UCT may use UCS and must not use UCP. Keep backend-specific behavior inside
  the relevant transport directory.
- `ucp`: protocol, endpoint, worker, request, tag, AM, stream, RMA, rendezvous,
  and wireup logic. UCP should compose UCT capabilities rather than baking in a
  specific transport backend unless the existing code already does so.
- `ucm`: memory allocation/event interception and memory hook logic. Be careful
  with initialization order, async-signal-safety assumptions, and interactions
  with external allocators.
- `tools`: user-facing utilities. Keep command-line behavior and output stable
  unless the task explicitly changes it.

## Source Changes

- Update the local `Makefile.am` whenever adding, removing, or renaming source
  or header files.
- Shared utilities usually belong in UCS. Transport-agnostic protocol behavior
  usually belongs in UCP. Transport-private logic should stay in the matching
  transport directory.
- Stable public API lives in `src/ucp/api/ucp.h`, `src/uct/api/uct.h`, and the
  headers they include. Add public API only when application or middleware code
  needs it and existing APIs cannot express it.
- Public API structs use `field_mask` as the first field; new optional fields
  get new `UCS_BIT(n)` values. Do not remove or renumber bits, reorder public
  struct fields, or change public field types.
- Put deprecated public symbols in compatibility headers with the local
  deprecation pattern. Public API changes need Doxygen comments and should
  include only the minimal non-API changes needed to compile.
- Keep internal shared headers within their layer and transport-private headers
  within their transport. Do not include transport-private headers across
  transport boundaries or from higher layers.
- Inline helpers normally use `.inl`; forward declarations use `_fwd.h`; type
  declarations use `_types.h`; macro definition headers use `_def.h`.

## Naming, Style, and Errors

- Use layer prefixes consistently: `ucs_`, `uct_`, `ucp_`, `ucm_` for
  functions/types and uppercase equivalents for macros and enum constants.
- Types use `_t`, opaque handles use `_h`, function pointer types use
  `_func_t`, and ops tables use `_ops_t`.
- Function names follow `<layer>_<object>_<verb>` and should describe what the
  function does, not how it happens internally.
- Output parameters go last and use `_p`. Set output parameters only after
  success.
- Use `ucs_status_t` for UCX failures and name the variable `status`. Reserve
  `ret` for plain integer return codes.
- In C code, declare local variables at the beginning of the function body.
- Prefer `sizeof(*ptr)` or `sizeof(variable)` over `sizeof(type)`.
- Use `ucs_container_of` and `ucs_derived_of` instead of open-coded pointer
  arithmetic.
- Use UCX logging macros, not `printf` or raw `assert`. Choose log levels using
  `docs/LoggingStyle.md`.
- Keep error handling local: log at the first point that determines the real
  error cause, set `status` before every cleanup `goto`, and propagate
  `ucs_status_t` without duplicating logs in callers.
- Use cleanup labels in reverse initialization order, named for what they undo,
  such as `err_free_desc` or `err_cleanup_md`.
- Use `ucs_assert*` for internal invariants, not user-input validation.

## Performance and Concurrency

- Many paths are latency-sensitive. Avoid extra allocations, locks, atomics,
  string formatting, or logging in fast paths unless the local code already
  pays that cost.
- Check existing progress, async, callback queue, and worker locking patterns
  before changing concurrency behavior.
- Use `ucs_likely` and `ucs_unlikely` only where branch hints protect hot paths.
- Use `static UCS_F_ALWAYS_INLINE` or local inline macros for hot-path header
  helpers; do not use plain `inline`.
- For UCT transport polymorphism, use ops tables and typed function pointers.
  Avoid hot-path conditionals when setup-time selection can encode the mode.
- For UCX object-style code, keep `super` as the first member, call the super
  constructor first, clean up in reverse order, and use `ucs_derived_of`.
- Config tables should use string defaults, descriptive help strings, and
  `ucs_offsetof`. User-facing config environment names use the `UCX_` prefix.
- Preserve symmetry: create/destroy, init/cleanup, pack/unpack, and
  enable/disable pairs should be named and implemented consistently.
- Use UCX allocation primitives and memory pools for fixed-size hot-path
  objects. Do not use stack allocation for unbounded sizes.
- Keep memory-type handling explicit; do not assume host memory in data paths.
- Cache expensive capability queries when the result is stable.
- Treat wire formats, endpoint configuration, request layout, and public struct
  layout as compatibility-sensitive.
