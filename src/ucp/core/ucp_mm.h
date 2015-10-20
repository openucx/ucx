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
 * Contains remote keys for UCT PDs.
 * pd_map specifies which PDs from the current context are present in the array.
 * The array itself contains only the PDs specified in pd_map, without gaps.
 */
typedef struct ucp_rkey {
    uint64_t                      pd_map;  /* Which *remote* PDs have valid memory handles */
    uct_rkey_bundle_t             uct[0];  /* Remote key for every PD */
} ucp_rkey_t;


/**
 * Memory handle.
 * Contains general information, and a list of UCT handles.
 * pd_map specifies which PDs from the current context are present in the array.
 * The array itself contains only the PDs specified in pd_map, without gaps.
 */
typedef struct ucp_mem {
    void                          *address;     /* Region start address */
    size_t                        length;       /* Region length */
    uct_alloc_method_t            alloc_method; /* Method used to allocate the memory */
    uct_pd_h                      alloc_pd;     /* PD used to allocated the memory */
    uint64_t                      pd_map;       /* Which PDs have valid memory handles */
    uct_mem_h                     uct[0];       /* Valid memory handles, as popcount(pd_map) */
} ucp_mem_t;


static inline uct_rkey_t ucp_lookup_uct_rkey(ucp_ep_h ep, ucp_rkey_h rkey)
{
    unsigned rkey_index;

    /*
     * Calculate the rkey index inside the compact array. This is actually the
     * number of PDs in the map with index less-than ours. So mask pd_map to get
     * only the less-than indices, and then count them using popcount operation.
     * TODO save the mask in ep->uct, to avoid the shift operation.
     */
    rkey_index = ucs_count_one_bits(rkey->pd_map & UCS_MASK(ep->dst_pd_index));
    return rkey->uct[rkey_index].rkey;
}


#define UCP_RKEY_LOOKUP(_ep, _rkey) \
    ({ \
        if (ENABLE_PARAMS_CHECK && \
            !((_rkey)->pd_map & UCS_BIT((_ep)->dst_pd_index))) \
        { \
            ucs_fatal("Remote key does not support current transport " \
                       "(remote pd index: %d rkey map: 0x%"PRIx64")", \
                       (_ep)->dst_pd_index, (_rkey)->pd_map); \
            return UCS_ERR_UNREACHABLE; \
        } \
        \
        ucp_lookup_uct_rkey(_ep, _rkey); \
    })


#endif
