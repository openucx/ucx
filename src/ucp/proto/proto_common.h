/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_COMMON_H_
#define UCP_PROTO_COMMON_H_

#include "proto.h"


typedef enum {
    UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY = UCS_BIT(0) /* Send buffer is used by
                                                          zero-copy operations */
} ucp_proto_common_init_flags_t;


/* Protocol common initialization parameters which are used to calculate
 * thresholds, performance, etc. */
typedef struct {
    ucp_proto_init_params_t super;
    double                  latency;       /* protocol added latency */
    double                  overhead;      /* protocol overhead */
    size_t                  cfg_thresh;    /* user-configured threshold */
    size_t                  fragsz_offset; /* offset of maximal fragment
                                              size in uct_iface_attr_t */
    size_t                  hdr_size;      /* header size on first lane */
    unsigned                flags;         /* see ucp_proto_common_init_flags_t */
} ucp_proto_common_init_params_t;


ucp_rsc_index_t
ucp_proto_common_get_md_index(const ucp_proto_common_init_params_t *params,
                              ucp_lane_index_t lane);


/* @return number of lanes found */
ucp_lane_index_t
ucp_proto_common_find_lanes(const ucp_proto_common_init_params_t *params,
                            ucp_lane_type_t lane_type, uint64_t tl_cap_flags,
                            ucp_lane_index_t *lanes, ucp_lane_index_t max_lanes);


void ucp_proto_common_calc_perf(const ucp_proto_common_init_params_t *params,
                                ucp_lane_index_t lane);

#endif
