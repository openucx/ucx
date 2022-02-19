/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_SINGLE_H_
#define UCP_PROTO_SINGLE_H_

#include "proto.h"
#include "proto_common.h"


typedef struct {
    ucp_proto_common_lane_priv_t   super;
    ucp_md_index_t                 reg_md; /* memory domain to register on, or NULL */
} ucp_proto_single_priv_t;


typedef struct {
    ucp_proto_common_init_params_t super;
    ucp_lane_type_t                lane_type;    /* Type of lane to select */
    uint64_t                       tl_cap_flags; /* Required iface capabilities */
} ucp_proto_single_init_params_t;


ucs_status_t ucp_proto_single_init(const ucp_proto_single_init_params_t *params);


ucs_status_t
ucp_proto_single_init_priv(const ucp_proto_single_init_params_t *params,
                           ucp_proto_single_priv_t *spriv);


void ucp_proto_single_query(const ucp_proto_query_params_t *params,
                            ucp_proto_query_attr_t *attr);


typedef ucs_status_t (*ucp_proto_send_single_cb_t)(
        ucp_request_t *req, const ucp_proto_single_priv_t *spriv,
        uct_iov_t *iov);


typedef ucs_status_t (*ucp_proto_request_zcopy_init_cb_t)(
        ucp_request_t *req, ucp_md_map_t md_map,
        uct_completion_callback_t comp_func, unsigned uct_reg_flags,
        unsigned dt_mask);

#endif
