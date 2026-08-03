/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_RMA_H_
#define UCP_RMA_H_

#include <ucp/core/ucp_types.h>
#include <ucp/proto/proto_am.h>
#include <uct/api/uct.h>


#define UCP_PROTO_RMA_EMULATION_DESC "software emulation"


/**
 * In current implementation a known bug exists in the process of
 * flushing multiple lanes. The flush operation can be scheduled and
 * completed while an RMA operation executed prior is still pending
 * completion and scheduled on a different lane.
 *
 * To address this, we're using a single bcopy RMA lane to mitigate these
 * issues.
 */
#define UCP_PROTO_RMA_MAX_BCOPY_LANES 1

#define UCP_EP_FENCE_SPIN_TIMEOUT_US  20   /* max microseconds to spin */

/**
 * Reconcile the lanes carrying pre-fence operations with the endpoint's
 * current live lanes. If a recorded lane disappeared, its replacement cannot
 * be identified by lane index alone, so conservatively flush every live lane.
 * Otherwise, keep the recorded subset to avoid flushing unrelated lanes.
 */
static UCS_F_ALWAYS_INLINE ucp_lane_map_t
ucp_ep_fence_lane_map_update(ucp_lane_map_t unflushed_lanes,
                             ucp_lane_map_t live_lanes)
{
    if (unflushed_lanes & ~live_lanes) {
        return live_lanes;
    }

    return unflushed_lanes;
}

/**
 * Normalize fence lane tracking after the endpoint topology changed.
 *
 * A dirty topology is handled conservatively because a replacement may use
 * the same lane index as the endpoint it replaced. Otherwise, preserve the
 * current lane subset unless one of its lanes disappeared.
 */
static UCS_F_ALWAYS_INLINE ucp_lane_map_t
ucp_ep_fence_lane_map_normalize(ucp_lane_map_t unflushed_lanes,
                                ucp_lane_map_t live_lanes, int lanes_dirty)
{
    if ((unflushed_lanes != 0) && lanes_dirty) {
        return live_lanes;
    }

    return ucp_ep_fence_lane_map_update(unflushed_lanes, live_lanes);
}


/**
 * Update an in-progress flush after the endpoint's live lanes changed.
 *
 * Lanes that disappeared before their flush started no longer need a
 * completion. Lanes already started remain accounted for by their completion
 * or discard flow. Newly created lanes need a completion and are added to the
 * requested lane mask, since they may replace a lane that carried pre-fence
 * operations.
 *
 * @return Change to apply to the flush completion count.
 */
static UCS_F_ALWAYS_INLINE int
ucp_ep_flush_lane_state_update(ucp_lane_map_t live_lanes,
                               int lane_generation_changed,
                               ucp_lane_map_t *started_lanes_p,
                               ucp_lane_map_t *all_lanes_p,
                               ucp_lane_map_t *lane_mask_p)
{
    ucp_lane_map_t unstarted_lanes;
    ucp_lane_map_t destroyed_lanes;
    ucp_lane_map_t new_lanes;

    if (lane_generation_changed) {
        unstarted_lanes  = *all_lanes_p & ~*started_lanes_p;
        *all_lanes_p     = live_lanes;
        *lane_mask_p    |= live_lanes;
        *started_lanes_p = 0;
        return ucs_popcount(live_lanes) - ucs_popcount(unstarted_lanes);
    }

    destroyed_lanes = *all_lanes_p & ~live_lanes & ~*started_lanes_p;
    new_lanes       = live_lanes & ~*all_lanes_p;

    *all_lanes_p = live_lanes;
    *lane_mask_p |= new_lanes;

    return ucs_popcount(new_lanes) - ucs_popcount(destroyed_lanes);
}

/**
 * Return whether an in-progress flush has not started on every live lane.
 * Historical started bits for lanes destroyed after starting are ignored.
 */
static UCS_F_ALWAYS_INLINE int
ucp_ep_flush_has_unstarted_lanes(ucp_lane_map_t live_lanes,
                                 ucp_lane_map_t started_lanes)
{
    return !!(live_lanes & ~started_lanes);
}

/**
 * Defines functions for AMO protocol
 */
struct ucp_amo_proto {
    const char                 *name;
    uct_pending_callback_t     progress_fetch;
    uct_pending_callback_t     progress_post;
};


/**
 * Atomic reply data
 */
typedef union {
    uint32_t           reply32; /* 32-bit reply */
    uint64_t           reply64; /* 64-bit reply */
} ucp_atomic_reply_t;


typedef struct {
    uint64_t                  address;
    uint64_t                  ep_id;
    ucs_memory_type_t         mem_type;
} UCS_S_PACKED ucp_put_hdr_t;


enum {
    UCP_CMPL_FLAG_RMA_RNDV = UCS_BIT(0)
};

typedef struct {
    uint64_t                  ep_id;
    uint8_t                   flags;
} UCS_S_PACKED ucp_cmpl_hdr_t;


typedef struct {
    uint64_t                  address;
    uint64_t                  length;
    ucp_request_hdr_t         req;
    ucs_memory_type_t         mem_type;
} UCS_S_PACKED ucp_get_req_hdr_t;


typedef struct {
    uint64_t                  req_id;
} UCS_S_PACKED ucp_rma_rep_hdr_t;


typedef struct {
    uint64_t                  address;
    ucp_request_hdr_t         req; /* invalid req_id if no reply */
    uint8_t                   length;
    uint8_t                   opcode;
} UCS_S_PACKED ucp_atomic_req_hdr_t;


extern ucp_amo_proto_t ucp_amo_basic_proto;
extern ucp_amo_proto_t ucp_amo_sw_proto;


extern const ucp_amo_proto_t *ucp_amo_proto_list[];


void ucp_ep_flush_remote_completed(ucp_request_t *req);

void ucp_rma_sw_send_cmpl(ucp_ep_h ep, uint8_t flags);

ucs_status_t ucp_ep_fence_weak(ucp_ep_h ep);

ucs_status_t ucp_ep_fence_strong(ucp_ep_h ep);

ucs_status_t ucp_ep_fence_strong_nb(ucp_ep_h ep, uint64_t fence_seq);

#endif
