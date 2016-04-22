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
    ucp_pd_map_t                  pd_map;  /* Which *remote* PDs have valid memory handles */
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
    ucp_pd_map_t                  pd_map;       /* Which PDs have valid memory handles */
    uct_mem_h                     uct[0];       /* Valid memory handles, as popcount(pd_map) */
} ucp_mem_t;


#endif
