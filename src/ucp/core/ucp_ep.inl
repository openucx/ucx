/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_EP_INL_
#define UCP_EP_INL_

#include "ucp_ep.h"
#include "ucp_worker.h"
#include "ucp_worker.inl"
#include "ucp_context.h"

#include <ucp/wireup/wireup.h>
#include <ucs/arch/bitops.h>
#include <ucs/datastruct/ptr_map.inl>


static inline ucp_ep_config_t *ucp_ep_config(ucp_ep_h ep)
{
    ucs_assert(ep->cfg_index != UCP_WORKER_CFG_INDEX_NULL);
    return &ep->worker->ep_config[ep->cfg_index];
}

static inline ucp_lane_index_t ucp_ep_get_am_lane(ucp_ep_h ep)
{
    ucs_assert(ucp_ep_config(ep)->key.am_lane != UCP_NULL_LANE);
    return ep->am_lane;
}

static inline ucp_lane_index_t ucp_ep_get_wireup_msg_lane(ucp_ep_h ep)
{
    ucp_lane_index_t lane = ucp_ep_config(ep)->key.wireup_msg_lane;
    return (lane == UCP_NULL_LANE) ? ucp_ep_get_am_lane(ep) : lane;
}

static inline ucp_lane_index_t ucp_ep_get_tag_lane(ucp_ep_h ep)
{
    ucs_assert(ucp_ep_config(ep)->key.tag_lane != UCP_NULL_LANE);
    return ucp_ep_config(ep)->key.tag_lane;
}

static inline int ucp_ep_config_key_has_tag_lane(const ucp_ep_config_key_t *key)
{
    ucp_lane_index_t lane = key->tag_lane;

    if (lane != UCP_NULL_LANE) {
        ucs_assert(key->lanes[lane].rsc_index != UCP_NULL_RESOURCE);
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
    ucs_assert(lane < UCP_MAX_LANES); /* to suppress coverity */
    return ucp_ep_config(ep)->key.lanes[lane].rsc_index;
}

static inline const uct_tl_resource_desc_t *
ucp_ep_get_tl_rsc(ucp_ep_h ep, ucp_lane_index_t lane)
{
    return &ep->worker->context->tl_rscs[ucp_ep_get_rsc_index(ep, lane)].tl_rsc;
}

static inline uint8_t ucp_ep_get_path_index(ucp_ep_h ep, ucp_lane_index_t lane)
{
    return ucp_ep_config(ep)->key.lanes[lane].path_index;
}

static inline uct_iface_attr_t *ucp_ep_get_iface_attr(ucp_ep_h ep, ucp_lane_index_t lane)
{
    return ucp_worker_iface_get_attr(ep->worker, ucp_ep_get_rsc_index(ep, lane));
}

static inline size_t ucp_ep_get_max_bcopy(ucp_ep_h ep, ucp_lane_index_t lane)
{
    size_t max_bcopy = ucp_ep_get_iface_attr(ep, lane)->cap.am.max_bcopy;

    return ucs_min(ucp_ep_config(ep)->key.lanes[lane].seg_size, max_bcopy);
}

static inline size_t ucp_ep_get_max_zcopy(ucp_ep_h ep, ucp_lane_index_t lane)
{
    size_t max_zcopy = ucp_ep_get_iface_attr(ep, lane)->cap.am.max_zcopy;

    return ucs_min(ucp_ep_config(ep)->key.lanes[lane].seg_size, max_zcopy);
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

static inline ucp_md_index_t ucp_ep_md_index(ucp_ep_h ep, ucp_lane_index_t lane)
{
    return ucp_ep_config(ep)->md_index[lane];
}

static inline uct_md_h ucp_ep_md(ucp_ep_h ep, ucp_lane_index_t lane)
{
    ucp_context_h context = ep->worker->context;
    return context->tl_mds[ucp_ep_md_index(ep, lane)].md;
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
    return &ucp_ep_ext_gen(ep)->flush_state;
}

static UCS_F_ALWAYS_INLINE ucp_ep_ext_control_t* ucp_ep_ext_control(ucp_ep_h ep)
{
    ucs_assert(ucp_ep_ext_gen(ep)->control_ext != NULL);
    return ucp_ep_ext_gen(ep)->control_ext;
}

static UCS_F_ALWAYS_INLINE void ucp_ep_update_flags(
        ucp_ep_h ep, uint32_t flags_add, uint32_t flags_remove)
{
    ucp_ep_flags_t ep_flags_add    = (ucp_ep_flags_t)flags_add;
    ucp_ep_flags_t ep_flags_remove = (ucp_ep_flags_t)flags_remove;

    UCP_WORKER_THREAD_CS_CHECK_IS_BLOCKED(ep->worker);
    ucs_assert((ep_flags_add & ep_flags_remove) == 0);

    ep->flags = (ep->flags | ep_flags_add) & ~ep_flags_remove;
}

static UCS_F_ALWAYS_INLINE ucs_ptr_map_key_t ucp_ep_remote_id(ucp_ep_h ep)
{
#if UCS_ENABLE_ASSERT
    if (!(ep->flags & UCP_EP_FLAG_REMOTE_ID)) {
        /* Let remote side assert if it gets invalid key */
        return UCS_PTR_MAP_KEY_INVALID;
    }
#endif
    return ucp_ep_ext_control(ep)->remote_ep_id;
}

static UCS_F_ALWAYS_INLINE ucs_ptr_map_key_t ucp_ep_local_id(ucp_ep_h ep)
{
    ucs_assert(ucp_ep_ext_control(ep)->local_ep_id != UCS_PTR_MAP_KEY_INVALID);
    return ucp_ep_ext_control(ep)->local_ep_id;
}

/*
 * Make sure we have a valid dest_ep_ptr value, so protocols which require a
 * reply from remote side could be used.
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_ep_resolve_remote_id(ucp_ep_h ep, ucp_lane_index_t lane)
{
    if (ep->flags & UCP_EP_FLAG_REMOTE_ID) {
        return UCS_OK;
    }

    return ucp_wireup_connect_remote(ep, lane);
}

static inline void ucp_ep_update_remote_id(ucp_ep_h ep,
                                           ucs_ptr_map_key_t remote_id)
{
    if (ep->flags & UCP_EP_FLAG_REMOTE_ID) {
        ucs_assertv(remote_id == ucp_ep_ext_control(ep)->remote_ep_id,
                    "ep=%p flags=0x%" PRIx32 " rkey=0x%" PRIxPTR
                    " ep->remote_id=0x%" PRIxPTR, ep, ep->flags, remote_id,
                    ucp_ep_ext_control(ep)->remote_ep_id);
    }

    ucs_assert(remote_id != UCS_PTR_MAP_KEY_INVALID);
    ucs_trace("ep %p: set remote_id to 0x%" PRIxPTR, ep, remote_id);
    ucp_ep_update_flags(ep, UCP_EP_FLAG_REMOTE_ID, 0);
    ucp_ep_ext_control(ep)->remote_ep_id = remote_id;
}

static inline const char* ucp_ep_peer_name(ucp_ep_h ep)
{
#if ENABLE_DEBUG_DATA
    return ep->peer_name;
#else
    return UCP_WIREUP_EMPTY_PEER_NAME;
#endif
}

/* get index of the local component which can reach a remote memory domain */
static inline ucp_rsc_index_t
ucp_ep_config_get_dst_md_cmpt(const ucp_ep_config_key_t *key,
                              ucp_md_index_t dst_md_index)
{
    unsigned idx = ucs_popcount(key->reachable_md_map & UCS_MASK(dst_md_index));

    return key->dst_md_cmpts[idx];
}

static inline int
ucp_ep_config_key_has_cm_lane(const ucp_ep_config_key_t *config_key)
{
    return config_key->cm_lane != UCP_NULL_LANE;
}

static inline int ucp_ep_has_cm_lane(ucp_ep_h ep)
{
    return (ep->cfg_index != UCP_WORKER_CFG_INDEX_NULL) &&
           ucp_ep_config_key_has_cm_lane(&ucp_ep_config(ep)->key);
}

static UCS_F_ALWAYS_INLINE ucp_lane_index_t ucp_ep_get_cm_lane(ucp_ep_h ep)
{
    return ucp_ep_config(ep)->key.cm_lane;
}

static UCS_F_ALWAYS_INLINE int
ucp_ep_config_connect_p2p(ucp_worker_h worker,
                          const ucp_ep_config_key_t *ep_config_key,
                          ucp_rsc_index_t rsc_index)
{
    return ucp_wireup_connect_p2p(worker, rsc_index,
                                  ucp_ep_config_key_has_cm_lane(ep_config_key));
}

static UCS_F_ALWAYS_INLINE int ucp_ep_use_indirect_id(ucp_ep_h ep)
{
    UCS_STATIC_ASSERT(sizeof(ep->flags) <= sizeof(int));
    return ep->flags & UCP_EP_FLAG_INDIRECT_ID;
}

#endif
