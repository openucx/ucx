/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "lane_type.h"

#include <stddef.h>


const ucp_lane_type_info_t ucp_lane_type_info[] = {
    [UCP_LANE_TYPE_AM] = {
        .short_name = "am"
    },
    [UCP_LANE_TYPE_AM_BW] = {
        .short_name = "am_bw"
    },
    [UCP_LANE_TYPE_RMA] = {
        .short_name = "rma"
    },
    [UCP_LANE_TYPE_RMA_BW] = {
        .short_name = "rma_bw"
    },
    [UCP_LANE_TYPE_RKEY_PTR] = {
        .short_name = "rkey_ptr"
    },
    [UCP_LANE_TYPE_AMO] = {
        .short_name = "amo"
    },
    [UCP_LANE_TYPE_TAG] = {
        .short_name = "tag"
    },
    [UCP_LANE_TYPE_CM] = {
        .short_name = "cm"
    },
    [UCP_LANE_TYPE_KEEPALIVE] = {
        .short_name = "keepalive"
    },
    [UCP_LANE_TYPE_LAST] = {
        .short_name = NULL
    }
};

