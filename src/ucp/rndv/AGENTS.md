# Agent Guide: src/ucp/rndv

Rendezvous protocols. Used for transfers above the eager threshold and for
zero-copy paths that want to bypass intermediate buffering. All
implementations plug into the `proto/` framework — when adding a new rndv
algorithm, register it as a `ucp_proto_t` and let proto-select decide when
to use it.

## On-Wire Messages

Defined in `rndv.h`:

- **RTS** (`ucp_rndv_rts_hdr_t`) — Ready-to-Send. Carries the sender's
  address, size, packed rkeys, and an opcode (`UCP_RNDV_RTS_TAG_OK` or
  `UCP_RNDV_RTS_AM`).
- **RTR** — Ready-to-Receive. Sent by the receiver when it can pull data,
  with its own address and rkey.
- **ATS** — acknowledgement marking the end of a GET-driven rendezvous
  (sent by the receiver after the GET completes).
- **ATP** — acknowledgement marking the end of a PUT-driven rendezvous
  (sent by the sender after the PUT completes).

## Algorithm Map

- `rndv.[ch]`/`.inl`, `rndv.c` — legacy proto-v1 dispatch core (RTS/RTR/
  ATS/ATP encode/decode, completion plumbing, AM handlers). New rndv work
  should plug into the proto framework instead — see `proto_rndv.[ch]`
  and the per-algorithm files below.
- `proto_rndv.[ch]`/`.inl` — base class shared by all rndv-aware protos.
  Owns common state (sreq/rreq IDs, rkey unpacking, fragmentation).
- `rndv_get.c` — receiver pulls data with `uct_ep_get_zcopy` (the default
  GET-driven rendezvous). Uses RTS → GET → ATS.
- `rndv_put.c` — sender pushes with `uct_ep_put_zcopy` (RTR-driven).
  Common when GET is unavailable (e.g. some sm transports).
- `rndv_am.c` — software fallback: sender breaks the payload into
  active-message fragments. Used when no zcopy lane is available.
- `rndv_rtr.c` — RTR generation/handling utilities consumed by the
  PUT-style protocols.
- `rndv_ats.c` — rendezvous protocol that completes with only an ATS
  message and no data transfer. Used for 0-length receives and for the
  `UCP_OP_ID_RNDV_RECV_DROP` "ignore data" path.
- `rndv_ppln.c` — pipelined rndv: overlaps a GPU staging copy with the
  network transfer by chunking. Selection uses
  `UCP_PROTO_SELECT_OP_FLAG_PPLN_FRAG` for sub-fragments.
- `rndv_rkey_ptr.c` — direct-pointer rndv: when sender memory is
  reachable as a local virtual address (e.g. via CUDA IPC or shared mem),
  do a memcpy/`cudaMemcpy` instead of a network op.
- `rndv_mtype.inl` — memory-type-aware helpers shared by the GPU paths.

## Conventions

- New algorithms plug into the proto framework: define a `ucp_proto_t`
  and add it to `ucp_protocols[]` (declared in `src/ucp/proto/proto.h`).
  Don't add ad-hoc dispatch code in `rndv.c`.
- Wire-visible header layouts are versioned; if you must extend an
  existing message, append fields and gate them on a flag, never reorder.
- Request IDs (`sreq_id`, `rreq_id`) are assigned via `ucs/datastruct/ptr_map`
  and must be released exactly once; leaking them shows up as request
  pool growth.
- For GPU paths, always go through `rndv_mtype.inl` helpers and the
  `memtype_cache`. Direct CUDA driver-API calls in this dir are wrong.
- Pipelining (`ppln`) is a meta-protocol: it owns a parent request that
  spawns child requests selected with `UCP_PROTO_SELECT_OP_FLAG_PPLN_FRAG`.
  Never inline-progress a pipeline child — schedule it through the
  worker.
- Direct-pointer (`rkey_ptr`) is the cheapest rndv when applicable.
  Selection uses the rkey-config sysdev/mem-type — keep that in mind when
  changing rkey packing in `core/ucp_rkey.c`.

## Pointers

- Tag/AM trigger sites: `tag/tag_rndv.c`, `am/rndv.c`.
- Selection plumbing: `proto/proto_select.c` consumes the
  `UCP_PROTO_SELECT_OP_FLAG_*_RNDV` flags from `proto_select.h`.
- Tests: `test/gtest/ucp/test_ucp_tag_xfer.cc`, `test_ucp_tag_mem_type.cc`,
  `test_ucp_rma.cc`, `test_ucp_proto.cc`, and the GPU rndv tests under
  `test/gtest/ucp/cuda/`.
