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
    return &ep->worker->ifaces[ucp_ep_get_rsc_index(ep, lane)].attr;
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

static inline const char* ucp_ep_peer_name(ucp_ep_h ep)
{
#if ENABLE_DEBUG_DATA
    return ep->peer_name;
#else
    return "<no debug data>";
#endif
}

#endif
