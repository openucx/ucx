/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_MM_H_
#define UCP_MM_H_

#include <ucp/api/ucp_def.h>
#include <ucp/core/ucp_ep.h>
#include <uct/api/uct.h>
#include <ucs/arch/bitops.h>
#include <ucs/debug/log.h>

#include <inttypes.h>


/**
 * Remote memory key structure.
 * Contains remote keys for UCT MDs.
 * md_map specifies which MDs from the current context are present in the array.
 * The array itself contains only the MDs specified in md_map, without gaps.
 */
typedef struct ucp_rkey {
    /* cached values for the most recent endpoint configuration */
    struct {
        ucp_ep_cfg_index_t        ep_cfg_index; /* EP configuration relevant for the cache */
        ucp_lane_index_t          rma_lane;     /* Lane to use for RMAs */
        ucp_lane_index_t          amo_lane;     /* Lane to use for AMOs */
        unsigned                  max_put_short;/* Cached value of max_put_short */
        uct_rkey_t                rma_rkey;     /* Key to use for RMAs */
        uct_rkey_t                amo_rkey;     /* Key to use for AMOs */
    } cache;
    ucp_md_map_t                  md_map;  /* Which *remote* MDs have valid memory handles */
    uct_rkey_bundle_t             uct[0];  /* Remote key for every MD */
} ucp_rkey_t;


/**
 * Memory handle.
 * Contains general information, and a list of UCT handles.
 * md_map specifies which MDs from the current context are present in the array.
 * The array itself contains only the MDs specified in md_map, without gaps.
 */
typedef struct ucp_mem {
    void                          *address;     /* Region start address */
    size_t                        length;       /* Region length */
    uct_alloc_method_t            alloc_method; /* Method used to allocate the memory */
    uct_md_h                      alloc_md;     /* MD used to allocated the memory */
    ucp_md_map_t                  md_map;       /* Which MDs have valid memory handles */
    uct_mem_h                     uct[0];       /* Valid memory handles, as popcount(md_map) */
} ucp_mem_t;


void ucp_rkey_resolve_inner(ucp_rkey_h rkey, ucp_ep_h ep);


#define UCP_RKEY_RESOLVE(_rkey, _ep, _op_type) \
    ({ \
        ucs_status_t status = UCS_OK; \
        if (ucs_unlikely((_ep)->cfg_index != (_rkey)->cache.ep_cfg_index)) { \
            ucp_rkey_resolve_inner(rkey, ep); \
            if (ucs_unlikely((_rkey)->cache._op_type##_lane == UCP_NULL_LANE)) { \
                ucs_error("Remote memory is unreachable"); \
                status = UCS_ERR_UNREACHABLE; \
            } \
        } \
        status; \
    })

#endif
