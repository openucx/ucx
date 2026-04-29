# UCP EP Failed-Lane Recovery POC

## Summary

After a successful failover (`ucp_ep_failover_reconfig`), rebuild the UCT endpoints
for lanes marked `UCP_LANE_TYPE_FAILED` by replacing the failed stub endpoints with
wireup proxies and exchanging a new dedicated WIREUP message pair over the operable
AM lane. The protocol is shape-reusable and intentionally named generically
(`WIREUP_MSG_LANES_ADDR_*`) so the same wire format can later back on-demand lane
connection flows.

Both `CONNECT_TO_IFACE` transports (DC, UD, TCP iface mode, self...) and p2p
`CONNECT_TO_EP` transports (RC, etc.) are supported. The p2p case materializes the
inner transport EP (e.g. an RC QP) eagerly during prepare so its `ep_addr` can
travel in the LANES_ADDR_REQUEST; the two-way `uct_ep_connect_to_ep_v2` handshake
is finalized symmetrically on both sides as the peer's ep_addr arrives.

Asymmetric failure is a first-class case: if only one side detected the UCT error
and the other still has live UCT EPs for the affected lanes, the peer handler
trusts the initiator's `provided_lane_map`, flushes/cancels and destroys the
still-live UCT EPs via `ucp_worker_discard_uct_ep`, and replaces them with new
ones built from the initiator's addresses.

Partial recovery is a legal steady state. The message carries two distinct lane
masks (`requested_lane_map` vs `provided_lane_map`) so either side can say "I
asked for lanes {a,b,c} but only managed to prepare addresses for {a,c}". Lanes
not successfully rebuilt stay `UCP_LANE_TYPE_FAILED` and `ucp_ep_recovery_schedule`
is re-invoked from the REPLY handler to converge over multiple rounds.

## High-level flow

```
         Initiator A                 Operable AM lane                 Peer B
             |                             |                             |
ucp_ep_failover_reconfig                   |                             |
  mark FAILED, discard                     |                             |
  ucp_ep_recovery_schedule                 |                             |
             |                             |                             |
ucp_ep_recovery_progress                   |                             |
  replace failed stubs with                |                             |
  wireup proxies;                          |                             |
  for p2p: create inner UCT EP (QP)        |                             |
  pack iface/ep addresses                  |                             |
             |--LANES_ADDR_REQUEST-------->|---------------------------->|
             |                             |                  process_lanes_addr_request
             |                             |                    rebuild_lanes(provided_lane_map)
             |                             |                    for each lane:
             |                             |                      install_wireup_ep
             |                             |                      iface: uct_ep_create w/ peer addr
             |                             |                      p2p:   connect_to_ep_v2 w/ peer QPN
             |                             |                      mark READY|REMOTE_CONNECTED
             |                             |                    schedule eps_progress
             |                             |                    clear_failed_lanes(rebuilt)
             |<-------------------LANES_ADDR_REPLY--------------|
  process_lanes_addr_reply                 |                             |
    to_complete =                          |                             |
       provided & failed_lanes(ep)         |                             |
    rebuild_lanes(to_complete)             |                             |
    schedule eps_progress                  |                             |
    clear_failed_lanes(rebuilt)            |                             |
    if still failed -> recovery_schedule   |                             |
             |                             |                             |
 ucp_wireup_eps_progress (on each side) swaps proxies for real UCT EPs.
```

## Design choices

| Question | Decision |
|---|---|
| Where does recovery get triggered? | Per-EP retry state (`recovery_next_time`, `recovery_retries_left`) is armed at the end of `ucp_ep_failover_reconfig()`. Actual rounds fire from the worker keepalive progress via `ucp_ep_recovery_tick()`, rate-limited by `UCX_RECOVERY_INTERVAL` and bounded by `UCX_RECOVERY_RETRIES`. |
| How is the remote address obtained? | Via a new dedicated wireup message pair: `UCP_WIREUP_MSG_LANES_ADDR_REQUEST` / `REPLY`. |
| Is the message type recovery-specific? | No. The generic name `LANES_ADDR_*` lets the same format back a future on-demand lane connection flow (the only difference there: lanes in `requested_lane_map` aren't marked FAILED going in). |
| Is there extra state in `ucp_ep_ext_t`? | No. `ucp_ep_get_failed_lanes(ep)` (via the `UCP_LANE_TYPE_FAILED` bit in `cfg_key.lanes[l].lane_types`) is the single source of truth. |
| Which lane is used to send LANES_ADDR_*? | `key.am_lane` - the operable AM lane post-failover. `wireup_msg_lane` may itself be failed, so `ucp_wireup_get_msg_lane()` now prefers `am_lane` for LANES_ADDR_* messages (same as for `UCP_WIREUP_MSG_ACK`). |
| Partial recovery? | First-class: two masks, `requested_lane_map` and `provided_lane_map`. `ucp_ep_reconfig_clear_failed_lanes()` accepts any subset and is a no-op when nothing changed. |
| How are p2p lanes supported? | Inner UCT EP is created in `install_wireup_ep` during prepare so its address is available at packing time. `connect_to_ep_v2` runs when the peer's ep_addr arrives. |
| How is asymmetric failure handled? | The REQUEST handler trusts `provided_lane_map` without intersecting with local failed lanes. `install_wireup_ep` discards still-live UCT EPs via `ucp_worker_discard_uct_ep(CANCEL)` with `ucp_ep_err_pending_purge`. |
| Duplicate scheduling guard? | Not needed: recovery is driven by the periodic worker keepalive progress, which is a single producer per worker. |
| Timeout / retry policy? | Two new context config parameters: `UCX_RECOVERY_INTERVAL` (default `1s`) and `UCX_RECOVERY_RETRIES` (default `3`). When retries are exhausted the endpoint is scheduled for full failure via `ucp_ep_set_lanes_failed_schedule(ep, 0, UCS_ERR_ENDPOINT_TIMEOUT)`. |
| Does the retry timer also fix the ep_count discard race? | Yes. The first recovery round fires only after `recovery_interval`, by which time `ucp_ep_discard_lanes()`'s async callback has finished the `deactivate(old_cfg) + activate(new_cfg)` transition. `ucp_ep_reconfig_clear_failed_lanes()` then runs against a consistent `ep_count`. |

## Scope

### In scope
- `UCT_IFACE_FLAG_CONNECT_TO_IFACE` transports (DC, UD, self, TCP iface mode, cuda_copy, ...). Primary test target. Rebuild via `uct_ep_create` with `dev_addr` + `iface_addr` from the peer's packed address.
- `UCT_IFACE_FLAG_CONNECT_TO_EP` p2p transports (RC, ...). Inner UCT EP (QP) created eagerly; `connect_to_ep_v2` applied from the peer's packed ep_addr.
- Asymmetric failure: only one side detected the UCT error. Peer handler trusts `provided_lane_map`, cancel-flushes live UCT EPs, and rebuilds.
- Partial recovery: any subset of requested lanes may come back in a given round; remaining lanes are retried on the next `ucp_ep_recovery_schedule` driven by the REPLY handler.

### Out of scope
- CM (sockaddr) flow. Disabled in `ucp_ep_failover_reconfig` already; we mirror that.
- Retrying a lost LANES_ADDR_REQUEST at the transport level. We rely on the next keepalive/IO-driven failure detection to re-enter `failover_reconfig` and reschedule recovery.
- Asymmetric lane layouts. `remote_lane = lane` (symmetric assumption). If lane indices could diverge across peers, add a `lanes2remote[UCP_MAX_LANES]` to `ucp_wireup_msg_t` and pass it into `ucp_address_pack` and `ucp_wireup_find_remote_p2p_addr`.

## Files changed

### `src/ucp/wireup/wireup.h`

Added two new message types and two lane-map fields on the wire header:

```c
enum {
    ...
    UCP_WIREUP_MSG_REPLY_RECONFIG,
    UCP_WIREUP_MSG_LANES_ADDR_REQUEST,
    UCP_WIREUP_MSG_LANES_ADDR_REPLY,
    UCP_WIREUP_MSG_LAST
};

typedef struct ucp_wireup_msg {
    uint8_t                type;
    uint8_t                err_mode;
    ucp_ep_match_conn_sn_t conn_sn;
    uint64_t               src_ep_id;
    uint64_t               dst_ep_id;
    /* Only valid for UCP_WIREUP_MSG_LANES_ADDR_* */
    ucp_lane_map_t         requested_lane_map; /* lanes the sender asked about */
    ucp_lane_map_t         provided_lane_map;  /* lanes actually carried here */
    /* packed addresses follow */
} UCS_S_PACKED ucp_wireup_msg_t;
```

### `src/ucp/wireup/wireup.c`

Selection of the send lane now prefers the operable AM lane for LANES_ADDR_*:

```c
if ((msg_type == UCP_WIREUP_MSG_ACK) ||
    (msg_type == UCP_WIREUP_MSG_LANES_ADDR_REQUEST) ||
    (msg_type == UCP_WIREUP_MSG_LANES_ADDR_REPLY)) {
    lane          = ep_config->key.am_lane;
    fallback_lane = ep_config->key.wireup_msg_lane;
} else { ... }
```

`ucp_wireup_msg_prepare` zeros the two new header fields. A new
`ucp_wireup_msg_send_full` takes the two lane maps as parameters; the original
`ucp_wireup_msg_send` becomes a thin wrapper passing zeros.

`ucp_wireup_msg_str` dumps the two new types as `LANES_ADDR_REQ` / `LANES_ADDR_REP`.

The `ucp_wireup_msg_handler` dispatch is extended with:

```c
} else if (msg->type == UCP_WIREUP_MSG_LANES_ADDR_REQUEST) {
    ucs_assert(msg->dst_ep_id != UCS_PTR_MAP_KEY_INVALID);
    ucs_assert(ep != NULL);
    ucp_wireup_process_lanes_addr_request(worker, ep, msg, &remote_address);
} else if (msg->type == UCP_WIREUP_MSG_LANES_ADDR_REPLY) {
    ucs_assert(msg->dst_ep_id != UCS_PTR_MAP_KEY_INVALID);
    ucs_assert(ep != NULL);
    ucp_wireup_process_lanes_addr_reply(worker, ep, msg, &remote_address);
}
```

A new ~400-line block right before `ucp_wireup_msg_handler` provides the recovery
machinery:

- `ucp_ep_recovery_install_wireup_ep(ep, lane)` - handles three starting states:
  already wireup proxy (no-op), failed stub (`uct_ep_destroy`), live UCT EP
  (`ucp_worker_discard_uct_ep` with `CANCEL` flush + `ucp_ep_err_pending_purge`).
  For p2p lanes, also `uct_ep_create()` iface-only to materialize an inner
  transport EP whose ep_addr can be packed; `LOCAL_CONNECTED` flag is
  **not** set at this stage so `connect_to_ep_v2` later does its job.
- `ucp_ep_recovery_attach_next_ep(ep, lane, next_ep)` - CONNECT_TO_IFACE path:
  attaches a freshly-connected real UCT EP as `next_ep` of the proxy and marks
  the proxy `READY|REMOTE_CONNECTED`.
- `ucp_ep_recovery_mark_p2p_ready(ep, lane)` - same `READY|REMOTE_CONNECTED`
  step for p2p after `connect_to_ep_v2` succeeded.
- `ucp_ep_recovery_find_iface_addr(ep, lane, remote_address)` - scans the
  unpacked address list for an entry whose `tl_name_csum` matches the local
  lane's rsc TL.
- `ucp_ep_recovery_rebuild_iface_lane(ep, lane, remote_address)` - CONNECT_TO_IFACE
  per-lane rebuild: `install_wireup_ep` then `uct_ep_create` with
  `dev_addr` + `iface_addr` + `path_index`, then `attach_next_ep`.
- `ucp_ep_recovery_rebuild_p2p_lane(ep, lane, remote_address)` - p2p per-lane
  rebuild: `install_wireup_ep` (idempotent) then `ucp_wireup_find_remote_p2p_addr`
  (`remote_lane = lane` symmetric assumption), then `ucp_wireup_ep_connect_to_ep_v2`,
  then `mark_p2p_ready`.
- `ucp_ep_recovery_rebuild_lanes(ep, lanes_to_rebuild, remote_address)` -
  dispatcher; returns the bitmap of lanes successfully rebuilt.
- `ucp_ep_recovery_prepare_lanes(ep, lanes)` - installs wireup proxies for
  every lane in the set (both iface and p2p); returns what's in proxy state.
- `ucp_ep_recovery_send_lanes_addr_msg(ep, msg_type, requested, provided)` -
  computes `tl_bitmap` from `ucp_wireup_get_ep_tl_bitmap(ep, provided)` and
  calls `ucp_wireup_msg_send_full`.
- `ucp_ep_recovery_progress(arg)` - the one-shot driver:
  1. bail if `UCP_EP_FLAG_FAILED`, `cfg_key.am_lane == UCP_NULL_LANE`, or
     `ucp_ep_get_failed_lanes(ep) == 0`;
  2. `prepare_lanes(failed_lanes)` for every failed lane;
  3. send `LANES_ADDR_REQUEST` with `requested = failed_lanes`,
     `provided = prepared`.
- `ucp_ep_recovery_progress_pred` - callbackq predicate for dedup.
- `ucp_ep_recovery_schedule(ep)` - public entry point. Removes any already-queued
  recovery progress for this ep, then adds a oneshot and signals the worker.
- `ucp_wireup_process_lanes_addr_request(worker, ep, msg, remote_address)` -
  peer handler. **Does not intersect with local failed set** - trusts
  `msg->provided_lane_map` as authoritative (asymmetric-case correctness).
  Schedules `eps_progress` if anything was rebuilt, always sends a REPLY
  (even empty), then `reconfig_clear_failed_lanes` (no-op for lanes that
  weren't locally failed).
- `ucp_wireup_process_lanes_addr_reply(worker, ep, msg, remote_address)` -
  initiator handler. Rebuilds `to_complete = msg->provided_lane_map &
  ucp_ep_get_failed_lanes(ep)` (initiator only cares about lanes it locally
  flagged). Reschedules `ucp_ep_recovery_schedule` if any lanes remain failed
  after this round - this is what gives us convergence over multiple rounds
  for partial recovery.

### `src/ucp/core/ucp_ep.c`

New helper right before `ucp_ep_failover_reconfig`:

```c
ucs_status_t ucp_ep_reconfig_clear_failed_lanes(ucp_ep_h ep,
                                                ucp_lane_map_t lanes);
```

- Accepts any subset of lanes.
- Clears `UCP_LANE_TYPE_FAILED` in `cfg_key.lanes[l].lane_types` for every
  lane in `lanes & ucp_ep_get_failed_lanes(ep)`. Lanes outside that
  intersection are untouched.
- Re-runs the post-failover `am_lane` promotion to the earliest non-failed
  `AM_BW` lane, so if the operable AM lane was a fallback and the original
  candidate is back, we switch back.
- Reuses `ucp_worker_get_ep_config` + `ucp_ep_set_cfg_index` +
  `ucp_ep_config_reactivate_worker_ifaces` (same pattern as
  `ucp_ep_reconfig_internal` and `ucp_ep_update_rkey_config`).
- No-op when `cfg_key` ends up identical to the current one (repeated partial
  passes don't thrash cfg_index).

`ucp_ep_failover_reconfig` now ends with:

```c
ucp_ep_discard_lanes(ucp_ep, failed_lanes, discard_status, old_cfg_index);
ucp_ep_recovery_schedule(ucp_ep);
return UCS_OK;
```

### `src/ucp/core/ucp_ep.h`

Two new public declarations:

```c
ucs_status_t ucp_ep_reconfig_clear_failed_lanes(ucp_ep_h ep,
                                                ucp_lane_map_t lanes);
void         ucp_ep_recovery_schedule(ucp_ep_h ep);
```

## Detailed sequences

### CONNECT_TO_IFACE (DC, UD, ...)

```
A  failover_reconfig: mark L FAILED, replace QP0 with failed stub, schedule recovery
A  recovery_progress: install_wireup_ep(L) -> replace failed stub with wireup proxy
A                     pack iface addr for rsc of L (from ucp_wireup_get_ep_tl_bitmap)
A                     send LANES_ADDR_REQUEST(requested={L}, provided={L})
B  process_lanes_addr_request:
B    rebuild_iface_lane(L):
B      find iface addr in remote_address by matching local rsc tl_name_csum
B      install_wireup_ep(L) -> discards live EP (asymmetric) or failed stub
B      uct_ep_create(iface, dev_addr, iface_addr, path_index) -> real EP
B      wireup_ep_set_next_ep + mark READY|REMOTE_CONNECTED
B    schedule ucp_wireup_eps_progress
B    send LANES_ADDR_REPLY(requested={L}, provided={L})
B    reconfig_clear_failed_lanes({L}) (no-op if locally not failed)
A  process_lanes_addr_reply:
A    to_complete = {L} & failed_lanes(ep) = {L}
A    rebuild_iface_lane(L): same as B above
A    schedule ucp_wireup_eps_progress
A    reconfig_clear_failed_lanes({L}) -> UCP_LANE_TYPE_FAILED cleared on A
A    failed_lanes(ep) = 0 -> no follow-up schedule
```

### p2p CONNECT_TO_EP (RC)

```
A  failover_reconfig: mark L FAILED, discard QP1, schedule recovery
A  recovery_progress: install_wireup_ep(L):
A                       replace failed stub with wireup proxy
A                       uct_ep_create(iface-only) -> QP2 under the proxy
A                     pack iface + ep_addr for L
A                     send LANES_ADDR_REQUEST(requested={L}, provided={L},
A                                             carries QP2 addr)
B  process_lanes_addr_request:
B    rebuild_p2p_lane(L):
B      install_wireup_ep(L) -> replace live QP0 via discard(CANCEL) or stub
B                              create QP3 (iface-only) under the proxy
B      find_remote_p2p_addr(ep, lane=L) -> QP2 addr
B      ucp_wireup_ep_connect_to_ep_v2(proxy, QP2 addr)
B         -> inner EP (QP3) transitions INIT -> RTR -> RTS, targets QP2
B      mark_p2p_ready(L)
B    schedule ucp_wireup_eps_progress
B    send LANES_ADDR_REPLY(requested={L}, provided={L}, carries QP3 addr)
A  process_lanes_addr_reply:
A    to_complete = {L} & failed_lanes(ep) = {L}
A    rebuild_p2p_lane(L):
A      install_wireup_ep(L) -> no-op, already a proxy with QP2 inside
A      find_remote_p2p_addr(ep, lane=L) -> QP3 addr
A      connect_to_ep_v2(proxy, QP3 addr) -> QP2 transitions to RTR/RTS
A      mark_p2p_ready(L)
A    schedule ucp_wireup_eps_progress
A    reconfig_clear_failed_lanes({L}) -> FAILED cleared
A  eps_progress on both sides: ucp_proxy_ep_replace -> lane now holds the
                               real inner UCT EP (QP2/QP3)
```

### Asymmetric case

Same sequences as above, with one difference: `install_wireup_ep` on the peer
is called with a live UCT EP (not a failed stub) as `ucp_ep_get_lane(ep, L)`.
Because `ucp_is_uct_ep_failed(old_uct_ep)` returns false, the function calls:

```c
ucp_worker_discard_uct_ep(ep, old_uct_ep, old_rsc_index, UCT_FLUSH_FLAG_CANCEL,
                          ucp_ep_err_pending_purge,
                          UCS_STATUS_PTR(UCS_ERR_CANCELED),
                          (ucp_send_nbx_callback_t)ucs_empty_function, NULL);
```

so the old UCT EP is flushed-and-destroyed asynchronously and any outstanding
user requests pending on it are purged with `UCS_ERR_CANCELED`. The new EP is
built from the initiator's addresses and the REPLY is sent with the peer's
fresh addresses. The peer's `cfg_key` is not touched - the lane never goes
through a FAILED state in the local cfg, which is correct because from the
user's perspective the lane is always operational (just underneath a proxy
for a brief moment).

## Convergence

For symmetric cases both sides reach a final state where
`ucp_ep_get_failed_lanes(ep) == 0` and `ucp_wireup_ep_test(lane)` is false
for all lanes in one round-trip. The test condition
`do { short_progress_loop(); } while (ucp_ep_get_failed_lanes(...) != 0);`
and the subsequent "no lane is wireup_ep" poll in `test_recovery` are
satisfied.

For partial recovery, `ucp_wireup_process_lanes_addr_reply` calls
`ucp_ep_recovery_schedule(ep)` at the tail when any failed lanes remain, which
drives another request/reply round. This converges toward full recovery or
stops when nothing further can be rebuilt, at which point the remaining
FAILED lanes simply stay failed until the next failure-detection event.

## Test coverage

Existing test
`test/gtest/ucp/test_ucp_fault_tolerance.target_failure_and_recovery`
exercises the recovery path end-to-end on a `TEST_OP_PUT | TEST_OP_AM |
TEST_OP_FLUSH | TEST_OP_RECOVERY` variant. It:

1. injects UCT failure on the target side for every rma_bw / am_bw lane
   except the last,
2. waits for `ucp_ep_get_failed_lanes(sender_ep) == 0`,
3. waits for no lane on the sender side to be a `ucp_wireup_ep`,
4. sends one more AM and expects success,
5. asserts no error-callback invocation.

The earlier `initiator_failure` / `target_failure` variants (which do not set
`TEST_OP_RECOVERY`) must continue passing - for them the new code path is a
no-op beyond scheduling, since they don't exercise the wait-for-recovery
condition.
