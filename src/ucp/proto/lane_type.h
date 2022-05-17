/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_LANE_TYPE_H_
#define UCP_LANE_TYPE_H_

#include <stdint.h>


/*
 * Defines how a lane should be selected and used
 */
typedef enum {
    UCP_LANE_TYPE_FIRST,                    /* First item in enum */
    UCP_LANE_TYPE_AM = UCP_LANE_TYPE_FIRST, /* Active messages */
    UCP_LANE_TYPE_AM_BW,                    /* High-BW active messages */
    UCP_LANE_TYPE_RMA,                      /* Remote memory access */
    UCP_LANE_TYPE_RMA_BW,                   /* High-BW remote memory access */
    UCP_LANE_TYPE_RKEY_PTR,                 /* Obtain remote memory pointer */
    UCP_LANE_TYPE_AMO,                      /* Atomic memory access */
    UCP_LANE_TYPE_TAG,                      /* Tag matching offload */
    UCP_LANE_TYPE_CM,                       /* CM wireup */
    UCP_LANE_TYPE_KEEPALIVE,                /* Checks connectivity */
    UCP_LANE_TYPE_LAST
} ucp_lane_type_t;


typedef struct ucp_lane_type_info {
    const char        *short_name;
} ucp_lane_type_info_t;


typedef uint32_t ucp_lane_type_mask_t;


extern const ucp_lane_type_info_t ucp_lane_type_info[];

#endif
