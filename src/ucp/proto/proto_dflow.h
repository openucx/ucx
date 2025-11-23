/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_DFLOW_H_
#define UCP_PROTO_DFLOW_H_

#include <ucp/core/ucp_types.h>
#include <ucs/time/time.h>
#include <uct/api/uct.h>

/* TODO: add const UCP_MAX_SELECTED_LANES 8 */
#define UCP_PROTO_DFLOW_MAX_LANES  8


/* TODO */
typedef enum {
    UCP_PROTO_DFLOW_MODE_DISABLED,
    UCP_PROTO_DFLOW_MODE_IDLE,
    UCP_PROTO_DFLOW_MODE_WAITING,
    UCP_PROTO_DFLOW_MODE_READY,
    UCP_PROTO_DFLOW_MODE_RUNNING,
} ucp_proto_dflow_mode_t;


/* TODO */
typedef struct {
    uct_completion_t comp;
    uct_completion_t *parent;
    ucs_time_t       start_time;
    ucs_time_t       latency_sum;
} ucp_proto_dflow_stats_t;


/* TODO */
typedef struct {
    ucp_proto_dflow_stats_t stats;
    ucp_proto_dflow_mode_t  mode;
    uint8_t                 weight;
} ucp_proto_dflow_lane_t;


/* TODO */
typedef struct {
    ucp_proto_dflow_mode_t  mode;
    size_t                  min_length;
    ucp_proto_dflow_stats_t stats;
    size_t                  length_sum;
    unsigned                num_samples;
    ucp_lane_index_t        num_lanes;
    ucp_proto_dflow_lane_t  *lanes[UCP_PROTO_DFLOW_MAX_LANES];
} ucp_proto_dflow_node_t;


/* TODO */
typedef struct {
    ucs_time_t             interval;
    uct_worker_cb_id_t     progress_cb_id;
    ucs_time_t             next_progress_time;
    unsigned               num_samples_interval;
    /* TODO: hash map */
    ucp_proto_dflow_node_t *node;
} ucp_proto_dflow_service_t;


ucs_status_t ucp_proto_dflow_service_init(ucp_worker_h worker,
                                          ucp_proto_dflow_service_t *service);

void ucp_proto_dflow_service_cleanup(ucp_worker_h worker,
                                     ucp_proto_dflow_service_t *service);

#endif
