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
- Header suffixes, naming, error handling, assertions, and cleanup style live
  in `docs/CodeStyle.md`. Log levels and message style live in
  `docs/LoggingStyle.md`.

## Runtime Notes

- Follow `docs/OptimizationStyle.md` for latency-sensitive source paths.
- Check existing progress, async, callback queue, and worker locking patterns
  before changing concurrency behavior.
- For UCX object-style code, keep `super` as the first member, call the super
  constructor first, clean up in reverse order, and use `ucs_derived_of`.
- Config tables should use string defaults, descriptive help strings, and
  `ucs_offsetof`. User-facing config environment names use the `UCX_` prefix.
- Preserve symmetry: create/destroy, init/cleanup, pack/unpack, and
  enable/disable pairs should be named and implemented consistently.
- Keep memory-type handling explicit; do not assume host memory in data paths.
- Treat wire formats, endpoint configuration, request layout, and public struct
  layout as compatibility-sensitive.
