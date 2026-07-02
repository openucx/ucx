# Agent Guide: src/ucp/wireup

Connection establishment and lane configuration for UCP endpoints. Anything
related to "how does an ep go from `ucp_ep_create` to fully-connected" lives
here. There are two paths — address-based wireup (peer worker addresses
exchanged out-of-band) and connection-manager wireup (sockaddr listener).

## File Map

- `wireup.[ch]` — the wireup state machine. Defines the on-wire message
  types `UCP_WIREUP_MSG_{PRE_REQUEST, REQUEST, REPLY, ACK, EP_CHECK,
  EP_REMOVED, REPLY_RECONFIG, QUERY_LANE_STATE, LANE_STATE}` and drives
  transitions on send/recv.
- `address.[ch]` — packing and unpacking of `ucp_address_t`: per-device
  records, per-iface records, atomic capability bits, system-device IDs,
  and connection info. ABI-sensitive: any layout change must add a version
  bit, never re-purpose an existing field. Peer-name handling defaults to
  `UCP_WIREUP_EMPTY_PEER_NAME` when debug data is absent.
- `select.c` — the lane-selection algorithm (no header; the entry points
  live in `wireup.h`). Given two addresses, picks which UCT transports
  are paired into the semantic lanes defined in `proto/lane_type.h`
  (`AM`, `AM_BW`, `RMA`, `RMA_BW`, `RKEY_PTR`, `AMO`, `TAG`, `CM`,
  `KEEPALIVE`, `DEVICE`, plus the `FAILED` state) using the criteria
  tuples in `ucp_wireup_criteria_t` (mandatory and optional flag masks).
  Ranks candidates by distance and bandwidth costs. The list of eligible
  transports is already filtered upstream (e.g. by `UCX_TLS` in
  `core/ucp_context.c`).
- `wireup_ep.[ch]` — the proxy ep used during handshake. Buffers user
  requests until lanes are ready, then transparently re-issues them
  through the real lanes. Subclass of `uct_ep` via `UCS_CLASS_*`.
- `wireup_cm.[ch]` — sockaddr/CM path. Wraps `uct_cm_*` events
  (`CONNECT_REQUEST`/`REPLY`/etc.) and feeds them into the same lane
  selection, with a sockaddr-based address exchange instead of the
  pre-shared worker address.
- `ep_match.[ch]` — server-side reconcile of incoming wireup requests
  with already-created eps (avoids double-creating an ep when both sides
  initiated concurrently).

## Conventions

- The wireup msg format is part of the wire compat surface — additions go
  through reserved bits/length fields, never field reordering. See the
  `UCP_WIREUP_MSG_*` constants in `wireup.h` and the matching pack/unpack
  in `wireup.c`.
- New transport capabilities advertised via address records require both
  a packer in `address.c` and a consumer in `select.c`. Forgetting one
  half silently disables the capability for the new build vs. old peers.
- `select.c` decisions must be deterministic for given inputs and
  symmetric across peers: given the same pair of local/remote addresses,
  both sides must pick the same lane assignment, otherwise the connection
  will mismatch. Many tests assert specific lane assignments, so use
  stable tie-breakers.
- The proxy-ep path (`wireup_ep.c`) holds user requests in a queue; never
  short-circuit it for a fast path that might run before all lanes are
  active. Use `ucp_wireup_ep_progress_pending` on transition.
- CM and non-CM paths share the same `select.c` — keep the address-based
  vs. CM-based differences confined to entry points.
- Reconfig (`UCP_WIREUP_MSG_REPLY_RECONFIG`) is driven by the proto layer;
  this dir is only the messaging mechanism.

## Pointers

- High-level entry: `core/ucp_ep.c`'s `ucp_ep_create` and friends call into
  `ucp_wireup_send_request` / `wireup_cm_*`.
- Tests: `test/gtest/ucp/test_ucp_wireup.cc`,
  `test_ucp_ep_reconfig.cc`, `test_ucp_sockaddr.cc`,
  `test_ucp_peer_failure.cc`, `test_ucp_fault_tolerance.cc`.
- Lane semantics consumed by: `core/ucp_ep.h` lane-id helpers and the
  proto framework (`proto/lane_type.h`).
