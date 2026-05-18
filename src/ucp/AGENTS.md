# Agent Guide: src/ucp

UCP is the high-level protocol layer (`libucp.so`). It composes UCT
transports into a single endpoint that exposes tag matching, streams, RMA,
atomics, active messages, and sockaddr-based connection establishment to
applications. Almost every meaningful UCX feature lives here.

## Core Object Graph

- `ucp_context_t` (`core/ucp_context.[ch]`) — global resources, MD list,
  configuration. One per process (typically).
- `ucp_worker_t` (`core/ucp_worker.[ch]`) — async progress engine. Owns one
  `uct_worker_t` plus per-resource `ucp_worker_iface_t` slots.
- `ucp_ep_t` (`core/ucp_ep.[ch]`) — peer endpoint composed of multiple
  *lanes*. Each lane is one `uct_ep_t` chosen for one or more roles
  (AM, RMA, AMO, tag, CM, …). Lane roles are encoded in
  `proto/lane_type.h`.
- `ucp_request_t` (`core/ucp_request.[ch]`) — operation in flight. Pooled,
  reused; carries a state machine specific to the protocol that owns it.
- `ucp_rkey_t` (`core/ucp_rkey.[ch]`) — packed remote key for RMA/AMO.
- `ucp_listener_t` / `ucp_proxy_ep_t` — sockaddr listener and the wireup
  proxy endpoint used during handshake.

## Subdirectory Map

- `api/` — public C headers (`ucp.h`, `ucp_def.h`, `ucp_compat.h`,
  experimental `ucpx.h`, `device/`). ABI-stable. `ucpx.h` only ships when
  `--enable-experimental-api`.
- `core/` — implementation of UCP core objects such as context, worker,
  endpoint, etc. Inline fast paths live in `*.inl`.
- `proto/` — modern protocol-selection framework (init / select / single /
  multi / perf / debug / reconfig). **See `proto/AGENTS.md`** before
  touching protocol cost models or selection logic.
- `wireup/` — handshake state machine, address packing/unpacking, transport
  selection, CM path. **See `wireup/AGENTS.md`.**
- `rndv/` — rendezvous algorithms (`get`, `put`, `am`, `ats`, `rtr`, `ppln`,
  `rkey_ptr`, `mtype`) and their `proto_rndv` integration. **See
  `rndv/AGENTS.md`.**
- `tag/` — tag-matching send/recv, eager flavors (`single`/`multi`), tag
  rendezvous trigger, hardware tag offload (`offload/`).
- `am/` — active-message eager paths (`single`/`multi`) and AM-rndv glue.
- `rma/` — PUT/GET (`am` and `offload` paths), atomics (`basic`, `offload`,
  `sw`), `rma_send`, `flush`.
- `stream/` — stream send/recv (`stream_send`, `stream_multi`,
  `stream_recv`).
- `dt/` — datatype iterators: contiguous, IOV, generic (user callbacks).
  All higher protocols pack/unpack through `datatype_iter_t`.

## Conventions

- Every endpoint operation goes through a **proto** chosen by `proto_select`
  at first use; legacy direct paths exist for some flows but new code should
  plug into the proto framework (see `proto/AGENTS.md`).
- Lanes are numbered per-ep; mapping from semantic role (AM lane, tag lane,
  rkey-pointer lane, CM lane, …) is stored on the ep config and must be
  re-resolved after reconfiguration (`proto_reconfig.c`).
- `ucp_request_t` is allocated from a pool — never store pointers into it
  across `ucp_worker_progress()` calls without taking a ref. Use the
  helpers in `core/ucp_request.inl`.
- For data sent during wireup (worker addresses, capability bits, etc.),
  the pack/unpack lives in `wireup/address.c` — anything that needs to
  cross the wire at handshake time must round-trip through it.

## Pointers

- Lane selection: see `wireup/AGENTS.md`.
- Protocol selection and cost model: see `proto/AGENTS.md`.
- Tests: `test/gtest/ucp/` — most files target one feature
  (`test_ucp_tag*`, `test_ucp_rma*`, `test_ucp_proto*`, etc.).
- Tools: `ucx_info -e` introspects the ep proto matrix; profiling consumes
  `ucs/profile`.
