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
    ucp_lane_index_t lane = ucp_ep_config(ep)->key.wireup_msg_lane;
    return (lane == UCP_NULL_LANE) ? ucp_ep_get_am_lane(ep) : lane;
}

static inline ucp_lane_index_t ucp_ep_get_rndv_data_lane(ucp_ep_h ep)
{
    ucs_assert(ucp_ep_config(ep)->key.rndv_lane != UCP_NULL_RESOURCE);
    return ucp_ep_config(ep)->key.rndv_lane;
}

static inline uct_ep_h ucp_ep_get_am_uct_ep(ucp_ep_h ep)
{
    return ep->uct_eps[ucp_ep_get_am_lane(ep)];
}

static inline uct_ep_h ucp_ep_get_rndv_data_uct_ep(ucp_ep_h ep)
{
    return ep->uct_eps[ucp_ep_get_rndv_data_lane(ep)];
}

static inline ucp_rsc_index_t ucp_ep_get_rsc_index(ucp_ep_h ep, ucp_lane_index_t lane)
{
    return ucp_ep_config(ep)->key.lanes[lane];
}

static inline ucp_rsc_index_t ucp_ep_num_lanes(ucp_ep_h ep)
{
    return ucp_ep_config(ep)->key.num_lanes;
}

static inline ucp_rsc_index_t ucp_ep_md_index(ucp_ep_h ep, ucp_lane_index_t lane)
{
    ucp_context_h context = ep->worker->context;
    ucs_assert(ucp_ep_get_rsc_index(ep, lane) != UCP_NULL_RESOURCE);
    return context->tl_rscs[ucp_ep_get_rsc_index(ep, lane)].md_index;
}

static inline uct_md_h ucp_ep_md(ucp_ep_h ep, ucp_lane_index_t lane)
{
    ucp_context_h context = ep->worker->context;
    return context->mds[ucp_ep_md_index(ep, lane)];
}

static inline ucp_md_map_t ucp_lane_map_get_lane(ucp_md_lane_map_t lane_map,
                                                 ucp_lane_index_t lane)
{
    return (lane_map >> (lane * UCP_MD_INDEX_BITS)) & UCS_MASK(UCP_MD_INDEX_BITS);
}

static inline const uct_md_attr_t* ucp_ep_md_attr(ucp_ep_h ep, ucp_lane_index_t lane)
{
    ucp_context_h context = ep->worker->context;
    return &context->md_attrs[ucp_ep_md_index(ep, lane)];
}

static inline const char* ucp_ep_peer_name(ucp_ep_h ep)
{
#if ENABLE_DEBUG_DATA
    return ep->peer_name;
#else
    return "<no debug data>";
#endif
}

static inline ucp_md_lane_map_t ucp_ep_md_map_expand(ucp_md_map_t md_map)
{
    /* "Broadcast" md_map to all bit groups in ucp_md_lane_map_t, so that
     * if would look like this: <md_map><md_map>....<md_map>.
     * The expanded value can be used to select which lanes support a given md_map.
     * TODO use SSE and support larger md_lane_map with shuffle / broadcast
     */
    UCS_STATIC_ASSERT(UCP_MD_LANE_MAP_BITS == 64);
    UCS_STATIC_ASSERT(UCP_MD_INDEX_BITS    == 8);
    return md_map * 0x0101010101010101ul;
}

/*
 * Calculate lane and rkey index based of the lane_map in ep configuration: the
 * lane_map holds the md_index which each lane supports, so we do 'and' between
 * that, and the md_map of the rkey (we duplicate the md_map in rkey to fill the
 * mask for each possible lane). The first set bit in the 'and' product represents
 * the first matching lane and its md_index.
 */
#define UCP_EP_RESOLVE_RKEY(_ep, _rkey, _name, _config, _lane, _uct_rkey) \
    { \
        ucp_md_lane_map_t ep_lane_map, rkey_md_map; \
        ucp_rsc_index_t dst_md_index, rkey_index; \
        uint8_t bit_index; \
        \
        _config     = ucp_ep_config(_ep); \
        ep_lane_map = (_config)->key._name##_lane_map; \
        rkey_md_map = ucp_ep_md_map_expand((_rkey)->md_map); \
        \
        if (ENABLE_PARAMS_CHECK && !(ep_lane_map & rkey_md_map)) { \
            ucs_error("Remote memory is unreachable"); \
            return UCS_ERR_UNREACHABLE; \
        } \
        \
        /* Find the first lane which supports one of the remote md's in the rkey*/ \
        bit_index    = ucs_ffs64(ep_lane_map & rkey_md_map); \
        _lane        = bit_index / UCP_MD_INDEX_BITS; \
        dst_md_index = bit_index % UCP_MD_INDEX_BITS; \
        rkey_index   = ucs_count_one_bits(rkey_md_map & UCS_MASK(dst_md_index)); \
        _uct_rkey    = (_rkey)->uct[rkey_index].rkey; \
    }

#define UCP_EP_RESOLVE_RKEY_RMA(_ep, _rkey, _lane, _uct_rkey, _rma_config) \
    { \
        ucp_ep_config_t *config; \
        \
        UCP_EP_RESOLVE_RKEY(_ep, _rkey, rma, config, _lane, _uct_rkey); \
        _rma_config  = &config->rma[(_lane)]; \
    }

#define UCP_EP_RESOLVE_RKEY_AMO(_ep, _rkey, _lane, _uct_rkey) \
    { \
        ucp_ep_config_t *config; \
        ucp_lane_index_t amo_index; \
        \
        UCP_EP_RESOLVE_RKEY(_ep, _rkey, amo, config, amo_index, _uct_rkey); \
        _lane        = config->key.amo_lanes[amo_index]; \
    }

#endif
