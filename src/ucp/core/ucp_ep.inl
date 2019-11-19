/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_EP_INL_
#define UCP_EP_INL_

#include "ucp_ep.h"
#include "ucp_worker.h"
#include "ucp_context.h"

#include <ucp/wireup/wireup.h>
#include <ucs/arch/bitops.h>


static inline ucp_ep_config_t *ucp_ep_config(ucp_ep_h ep)
{
    return &ep->worker->ep_config[ep->cfg_index];
}

static inline ucp_lane_index_t ucp_ep_get_am_lane(ucp_ep_h ep)
{
    ucs_assert(ucp_ep_config(ep)->key.am_lane != UCP_NULL_LANE);
    return ep->am_lane;
}

static inline ucp_lane_index_t ucp_ep_get_wireup_msg_lane(ucp_ep_h ep)
{
    ucp_lane_index_t lane = ucp_ep_config(ep)->key.wireup_lane;
    return (lane == UCP_NULL_LANE) ? ucp_ep_get_am_lane(ep) : lane;
}

static inline ucp_lane_index_t ucp_ep_get_tag_lane(ucp_ep_h ep)
{
    ucs_assert(ucp_ep_config(ep)->key.tag_lane != UCP_NULL_LANE);
    return ucp_ep_config(ep)->key.tag_lane;
}

static inline int ucp_ep_is_tag_offload_enabled(ucp_ep_config_t *config)
{
    ucp_lane_index_t lane = config->key.tag_lane;

    if (lane != UCP_NULL_LANE) {
        ucs_assert(config->key.lanes[lane].rsc_index != UCP_NULL_RESOURCE);
        return 1;
    }
    return 0;
}

static inline uct_ep_h ucp_ep_get_am_uct_ep(ucp_ep_h ep)
{
    return ep->uct_eps[ucp_ep_get_am_lane(ep)];
}

static inline uct_ep_h ucp_ep_get_tag_uct_ep(ucp_ep_h ep)
{
    return ep->uct_eps[ucp_ep_get_tag_lane(ep)];
}

static inline ucp_rsc_index_t ucp_ep_get_rsc_index(ucp_ep_h ep, ucp_lane_index_t lane)
{
    return ucp_ep_config(ep)->key.lanes[lane].rsc_index;
}

static inline uct_iface_attr_t *ucp_ep_get_iface_attr(ucp_ep_h ep, ucp_lane_index_t lane)
{
    return ucp_worker_iface_get_attr(ep->worker, ucp_ep_get_rsc_index(ep, lane));
}

static inline size_t ucp_ep_get_max_bcopy(ucp_ep_h ep, ucp_lane_index_t lane)
{
    return ucp_ep_get_iface_attr(ep, lane)->cap.am.max_bcopy;
}

static inline size_t ucp_ep_get_max_zcopy(ucp_ep_h ep, ucp_lane_index_t lane)
{
    return ucp_ep_get_iface_attr(ep, lane)->cap.am.max_zcopy;
}

static inline size_t ucp_ep_get_max_iov(ucp_ep_h ep, ucp_lane_index_t lane)
{
    return ucp_ep_get_iface_attr(ep, lane)->cap.am.max_iov;
}

static inline ucp_lane_index_t ucp_ep_num_lanes(ucp_ep_h ep)
{
    return ucp_ep_config(ep)->key.num_lanes;
}

static inline int ucp_ep_is_lane_p2p(ucp_ep_h ep, ucp_lane_index_t lane)
{
    return ucp_ep_config(ep)->p2p_lanes & UCS_BIT(lane);
}

static inline ucp_lane_index_t ucp_ep_get_proxy_lane(ucp_ep_h ep,
                                                     ucp_lane_index_t lane)
{
    return ucp_ep_config(ep)->key.lanes[lane].proxy_lane;
}

static inline ucp_md_index_t ucp_ep_md_index(ucp_ep_h ep, ucp_lane_index_t lane)
{
    return ucp_ep_config(ep)->md_index[lane];
}

static inline const uct_md_attr_t* ucp_ep_md_attr(ucp_ep_h ep, ucp_lane_index_t lane)
{
    ucp_context_h context = ep->worker->context;
    return &context->tl_mds[ucp_ep_md_index(ep, lane)].attr;
}

static UCS_F_ALWAYS_INLINE ucp_ep_ext_gen_t* ucp_ep_ext_gen(ucp_ep_h ep)
{
    return (ucp_ep_ext_gen_t*)ucs_strided_elem_get(ep, 0, 1);
}

static UCS_F_ALWAYS_INLINE ucp_ep_ext_proto_t* ucp_ep_ext_proto(ucp_ep_h ep)
{
    return (ucp_ep_ext_proto_t*)ucs_strided_elem_get(ep, 0, 2);
}

static UCS_F_ALWAYS_INLINE ucp_ep_h ucp_ep_from_ext_gen(ucp_ep_ext_gen_t *ep_ext)
{
    return (ucp_ep_h)ucs_strided_elem_get(ep_ext, 1, 0);
}

static UCS_F_ALWAYS_INLINE ucp_ep_h ucp_ep_from_ext_proto(ucp_ep_ext_proto_t *ep_ext)
{
    return (ucp_ep_h)ucs_strided_elem_get(ep_ext, 2, 0);
}

static UCS_F_ALWAYS_INLINE ucp_ep_flush_state_t* ucp_ep_flush_state(ucp_ep_h ep)
{
    ucs_assert(ep->flags & UCP_EP_FLAG_FLUSH_STATE_VALID);
    ucs_assert(!(ep->flags & UCP_EP_FLAG_ON_MATCH_CTX));
    ucs_assert(!(ep->flags & UCP_EP_FLAG_LISTENER));
    ucs_assert(!(ep->flags & UCP_EP_FLAG_CLOSE_REQ_VALID));
    return &ucp_ep_ext_gen(ep)->flush_state;
}

static UCS_F_ALWAYS_INLINE uintptr_t ucp_ep_dest_ep_ptr(ucp_ep_h ep)
{
#if UCS_ENABLE_ASSERT
    if (!(ep->flags & UCP_EP_FLAG_DEST_EP)) {
        return 0; /* Let remote side assert if it gets NULL pointer */
    }
#endif
    return ucp_ep_ext_gen(ep)->dest_ep_ptr;
}

/*
 * Make sure we have a valid dest_ep_ptr value, so protocols which require a
 * reply from remote side could be used.
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_ep_resolve_dest_ep_ptr(ucp_ep_h ep, ucp_lane_index_t lane)
{
    if (ep->flags & UCP_EP_FLAG_DEST_EP) {
        return UCS_OK;
    }

    return ucp_wireup_connect_remote(ep, lane);
}

static inline void ucp_ep_update_dest_ep_ptr(ucp_ep_h ep, uintptr_t ep_ptr)
{
    if (ep->flags & UCP_EP_FLAG_DEST_EP) {
        ucs_assertv(ep_ptr == ucp_ep_ext_gen(ep)->dest_ep_ptr,
                    "ep=%p ep_ptr=0x%lx ep->dest_ep_ptr=0x%lx",
                    ep, ep_ptr, ucp_ep_ext_gen(ep)->dest_ep_ptr);
    }

    ucs_assert(ep_ptr != 0);
    ucs_trace("ep %p: set dest_ep_ptr to 0x%lx", ep, ep_ptr);
    ep->flags                      |= UCP_EP_FLAG_DEST_EP;
    ucp_ep_ext_gen(ep)->dest_ep_ptr = ep_ptr;
}

static inline const char* ucp_ep_peer_name(ucp_ep_h ep)
{
#if ENABLE_DEBUG_DATA
    return ep->peer_name;
#else
    return "<no debug data>";
#endif
}

static inline void ucp_ep_flush_state_reset(ucp_ep_h ep)
{
    ucp_ep_flush_state_t *flush_state = &ucp_ep_ext_gen(ep)->flush_state;

    ucs_assert(!(ep->flags & (UCP_EP_FLAG_ON_MATCH_CTX |
                              UCP_EP_FLAG_LISTENER)));
    ucs_assert(!(ep->flags & UCP_EP_FLAG_FLUSH_STATE_VALID) ||
               ((flush_state->send_sn == 0) &&
                (flush_state->cmpl_sn == 0) &&
                ucs_queue_is_empty(&flush_state->reqs)));

    flush_state->send_sn = 0;
    flush_state->cmpl_sn = 0;
    ucs_queue_head_init(&flush_state->reqs);
    ep->flags |= UCP_EP_FLAG_FLUSH_STATE_VALID;
}

static inline void ucp_ep_flush_state_invalidate(ucp_ep_h ep)
{
    ucp_ep_flush_state_t *flush_state UCS_V_UNUSED = ucp_ep_flush_state(ep);

    ucs_assert(ucs_queue_is_empty(&flush_state->reqs));
    ep->flags &= ~UCP_EP_FLAG_FLUSH_STATE_VALID;
}

/* get index of the local component which can reach a remote memory domain */
static inline ucp_rsc_index_t
ucp_ep_config_get_dst_md_cmpt(const ucp_ep_config_key_t *key,
                              ucp_md_index_t dst_md_index)
{
    unsigned index = ucs_popcount(key->reachable_md_map & UCS_MASK(dst_md_index));

    return key->dst_md_cmpts[index];
}

static inline ucp_lane_index_t ucp_ep_get_cm_lane(ucp_ep_h ep)
{
    return ucp_ep_config(ep)->key.cm_lane;
}

#endif
