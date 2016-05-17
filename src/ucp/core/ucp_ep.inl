/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_EP_INL_
#define UCP_EP_INL_

#include "ucp_ep.h"
#include "ucp_worker.h"

#include <ucs/arch/bitops.h>


static inline ucp_ep_config_t *ucp_ep_config(ucp_ep_h ep)
{
    return &ep->worker->ep_config[ep->cfg_index];
}

static inline ucp_lane_index_t ucp_ep_get_am_lane(ucp_ep_h ep)
{
    ucs_assert(ucp_ep_config(ep)->key.am_lane != UCP_NULL_RESOURCE);
    return ep->am_lane;
}

static inline ucp_lane_index_t ucp_ep_get_wireup_msg_lane(ucp_ep_h ep)
{
    return ucp_ep_config(ep)->key.wireup_msg_lane;
}

static inline uct_ep_h ucp_ep_get_am_uct_ep(ucp_ep_h ep)
{
    return ep->uct_eps[ucp_ep_get_am_lane(ep)];
}

static inline ucp_rsc_index_t ucp_ep_get_rsc_index(ucp_ep_h ep, ucp_lane_index_t lane)
{
    return ucp_ep_config(ep)->key.lanes[lane];
}

static inline ucp_rsc_index_t ucp_ep_num_lanes(ucp_ep_h ep)
{
    return ucp_ep_config(ep)->key.num_lanes;
}

static inline ucp_rsc_index_t ucp_ep_pd_index(ucp_ep_h ep, ucp_lane_index_t lane)
{
    ucp_context_h context = ep->worker->context;
    return context->tl_rscs[ucp_ep_get_rsc_index(ep, lane)].pd_index;
}

static inline uct_pd_h ucp_ep_pd(ucp_ep_h ep, ucp_lane_index_t lane)
{
    ucp_context_h context = ep->worker->context;
    return context->pds[ucp_ep_pd_index(ep, lane)];
}

static inline ucp_pd_map_t ucp_lane_map_get_lane(ucp_pd_lane_map_t lane_map,
                                                 ucp_lane_index_t lane)
{
    return (lane_map >> (lane * UCP_PD_INDEX_BITS)) & UCS_MASK(UCP_PD_INDEX_BITS);
}

static inline const char* ucp_ep_peer_name(ucp_ep_h ep)
{
#if ENABLE_DEBUG_DATA
    return ep->peer_name;
#else
    return "<no debug data>";
#endif
}

static inline ucp_pd_lane_map_t ucp_ep_pd_map_expand(ucp_pd_map_t pd_map)
{
    /* "Broadcast" pd_map to all bit groups in ucp_pd_lane_map_t, so that
     * if would look like this: <pd_map><pd_map>....<pd_map>.
     * The expanded value can be used to select which lanes support a given pd_map.
     * TODO use SSE and support larger pd_lane_map with shuffle / broadcast
     */
    UCS_STATIC_ASSERT(UCP_PD_LANE_MAP_BITS == 64);
    UCS_STATIC_ASSERT(UCP_PD_INDEX_BITS    == 8);
    return pd_map * 0x0101010101010101ul;
}

/*
 * Calculate lane and rkey index based of the lane_map in ep configuration: the
 * lane_map holds the pd_index which each lane supports, so we do 'and' between
 * that, and the pd_map of the rkey (we duplicate the pd_map in rkey to fill the
 * mask for each possible lane). The first set bit in the 'and' product represents
 * the first matching lane and its pd_index.
 */
#define UCP_EP_RESOLVE_RKEY(_ep, _rkey, _name, _config, _lane, _uct_rkey) \
    { \
        ucp_pd_lane_map_t ep_lane_map, rkey_pd_map; \
        ucp_rsc_index_t dst_pd_index, rkey_index; \
        uint8_t bit_index; \
        \
        _config     = ucp_ep_config(_ep); \
        ep_lane_map = (_config)->key._name##_lane_map; \
        rkey_pd_map = ucp_ep_pd_map_expand((_rkey)->pd_map); \
        \
        if (ENABLE_PARAMS_CHECK && !(ep_lane_map & rkey_pd_map)) { \
            ucs_error("Remote memory is unreachable"); \
            return UCS_ERR_UNREACHABLE; \
        } \
        \
        /* Find the first lane which supports one of the remote pd's in the rkey*/ \
        bit_index    = ucs_ffs64(ep_lane_map & rkey_pd_map); \
        lane         = bit_index / UCP_PD_INDEX_BITS; \
        dst_pd_index = bit_index % UCP_PD_INDEX_BITS; \
        rkey_index   = ucs_count_one_bits(rkey_pd_map & UCS_MASK(dst_pd_index)); \
        _uct_rkey    = (_rkey)->uct[rkey_index].rkey; \
    }

#define UCP_EP_RESOLVE_RKEY_RMA(_ep, _rkey, _uct_ep, _uct_rkey, _rma_config) \
    { \
        ucp_ep_config_t *config; \
        ucp_lane_index_t lane; \
        \
        UCP_EP_RESOLVE_RKEY(_ep, _rkey, rma, config, lane, _uct_rkey); \
        _uct_ep      = (_ep)->uct_eps[lane]; \
        _rma_config  = &config->rma[lane]; \
    }

#define UCP_EP_RESOLVE_RKEY_AMO(_ep, _rkey, _uct_ep, _uct_rkey) \
    { \
        ucp_ep_config_t *config; \
        ucp_lane_index_t lane; \
        \
        UCP_EP_RESOLVE_RKEY(_ep, _rkey, amo, config, lane, _uct_rkey); \
        _uct_ep      = (_ep)->uct_eps[lane]; \
    }

#endif
