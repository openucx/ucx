# Agent Guide: src/ucp/proto

The modern protocol-selection framework. For every operation flavor (tag
send, AM recv, RMA put, rendezvous fragment, …) and every endpoint
configuration, this layer enumerates candidate protocols, models their
performance, and picks one per message-size range. All new UCP operations
should plug in here rather than going through the legacy direct paths.

## Core Concepts

- **Protocol** (`ucp_proto_t`) — a strategy with `probe` (decide whether it
  applies and at what cost), `query` (describe), `progress[]` (per-stage
  progress callbacks), and `abort`/`reset` hooks. Stages numbered
  `UCP_PROTO_STAGE_START..LAST` (max 8). See `proto.h`.
- **Selection key** (`ucp_proto_select_param_t`) — operation ID, datatype
  class, sysdev, memtype, op flags. Maps to a `ucp_proto_select_elem_t`
  cached on the ep config.
- **Performance model** (`proto_perf.[ch]`) — piecewise linear cost
  functions over message size, broken into ranges (max
  `UCP_PROTO_MAX_PERF_RANGES = 24`). Each protocol contributes ranges; the
  envelope is computed and the cheapest protocol wins per range.
- **Single vs. multi** — `proto_single.[ch]` for one-lane protocols,
  `proto_multi.[ch]` for striped/multi-lane (e.g. multi-rail). Most rndv
  protocols layer on top of `proto_multi`.
- **Reconfig** (`proto_reconfig.c`) — re-resolves selection after lane
  changes (peer reachability, fault tolerance, memory-region updates).

## File Map

- `proto.[ch]` — the `ucp_proto_t` vtable, the `ucp_protocols[]` extern
  array of all registered protocols, common types.
- `proto_init.[ch]` — orchestrates the per-config initialization: runs
  every protocol's `probe` and populates the selection cache.
- `proto_select.[ch]`/`.inl` — the kHash-based cache from selection-param
  to chosen `ucp_proto_t` plus its private data; selection main loop.
- `proto_perf.[ch]` — perf node tree, range arithmetic, envelope.
- `proto_common.[ch]`/`.inl` — helpers shared across protos: lane choice
  utilities, memory-type checks, completion plumbing. Hosts
  `ucp_proto_request_send_op`, the canonical entry into the proto layer.
- `proto_single.[ch]`/`.inl`, `proto_multi.[ch]`/`.inl` — base protocols
  most concrete protos derive from.
- `proto_am.[ch]`/`.inl` — AM-specific shared bits (eager headers).
- `proto_debug.[ch]` — `ucx_info -p` output, perf-curve dumps.
- `proto_reconfig.c` — re-running selection on config change.
- `lane_type.[ch]` — semantic lane identifiers used to map proto choices
  to actual UCT lanes on an ep.

## Conventions

- Protocols are registered through the `UCP_PROTO_FOR_EACH(_macro)`
  X-macro in `proto.c`, which expands (via `UCP_PROTO_DECL` /
  `UCP_PROTO_ENTRY`) into the `ucp_protocols[]` array declared in
  `proto.h`. To add a proto: define a `ucp_proto_t` in your `.c` file,
  append `_macro(your_proto_name)` to `UCP_PROTO_FOR_EACH` in `proto.c`,
  and list the `.c` file in `src/ucp/Makefile.am`.
- A proto's `probe` must be a pure function of its inputs; it runs once
  per ep-config × selection-key and the result is cached. Reading worker
  state mutably from `probe` is a bug.
- Performance ranges must be monotonic in start size and use
  `linear_func`/`piecewise_func` from `ucs/datastruct`. Use
  `UCP_PROTO_PERF_EPSILON` (1e-15) for equality checks.
- When two protos tie, the earlier-listed (lower index in
  `ucp_protocols[]`, i.e. earlier in the `UCP_PROTO_FOR_EACH` list) wins;
  rely on this for deterministic test output.
- Stage callbacks must drain all pending `UCS_INPROGRESS` on completion
  before transitioning. The framework will call `progress[stage]` again
  on `UCS_ERR_NO_RESOURCE`.
- New op IDs go in `core/ucp_types.h` (`ucp_operation_id_t`) — keep the
  bitwidth in mind: it's packed into the selection-key kHash.
- Wire-visible state must round-trip through `wireup/address.c` or a
  protocol header — never rely on cached pointers across reconfig.

## Pointers

- Caller side: `ucp_proto_request_send_op` in `proto_common.inl` is the
  entry point; high-level call sites are in `tag/`, `am/`, `rma/`,
  `stream/`, `rndv/`.
- Tests: `test/gtest/ucp/test_ucp_proto.cc` and `test_ucp_proto_mock.cc`
  exercise the framework directly.
- Introspection: `ucx_info -p`/`-e` calls into `proto_debug.c` and prints
  the selection matrix for a configuration.
- Style/perf: `docs/OptimizationStyle.md` is especially relevant here.
