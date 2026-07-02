/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_FAILOVER_H_
#define UCP_PROTO_FAILOVER_H_

#include <ucp/core/ucp_types.h>
#include <uct/api/v2/uct_v2.h>
#include <ucs/datastruct/queue_types.h>


typedef struct ucp_proto_failover_replay_op {
    ucs_queue_elem_t queue;
    ucp_request_t    *req;
    uct_ep_op_info_t info;
    char             data[0];
} ucp_proto_failover_replay_op_t;


ucs_status_t ucp_proto_failover_replay_op_create(
        const uct_ep_op_info_t *op_info,
        ucp_proto_failover_replay_op_t **replay_op_p);

void ucp_proto_failover_replay_op_destroy(
        ucp_proto_failover_replay_op_t *op);

ucs_status_t
ucp_proto_failover_replay_op_progress(ucp_ep_h ep, ucp_lane_index_t failed_lane,
                                      ucp_proto_failover_replay_op_t *op);

#endif
