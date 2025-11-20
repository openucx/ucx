/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_DFLOW_INL_
#define UCP_PROTO_DFLOW_INL_

#include "proto_dflow.h"

static UCS_F_ALWAYS_INLINE void
ucp_proto_dflow_node_init(ucp_proto_dflow_node_t *node, int enabled,
                          ucp_lane_index_t num_lanes)
{
    node->mode       = enabled && (num_lanes <= UCP_PROTO_DFLOW_MAX_LANES) ?
                                UCP_PROTO_DFLOW_MODE_READY :
                                UCP_PROTO_DFLOW_MODE_DISABLED;
    node->min_length = 0;
    node->length_sum = 0;
    node->num_lanes  = num_lanes;

    memset(&node->stats, 0, sizeof(node->stats));
    memset(&node->lanes, 0, sizeof(node->lanes));
}

static UCS_F_ALWAYS_INLINE int
ucp_proto_dflow_enabled(const ucp_proto_dflow_node_t *node,
                        const ucp_request_t *req,
                        ucp_lane_index_t lane_idx)
{
    /* This check should filter out most of the requests */
    if (ucs_likely(node->mode < UCP_PROTO_DFLOW_MODE_READY)) {
        return 0;
    } else if (node->mode == UCP_PROTO_DFLOW_MODE_READY) {
        /* Profile only requests that are split between all lanes */
        return (lane_idx == 0) &&
               (req->send.state.dt_iter.length > node->min_length);
    } else { /* UCP_PROTO_DFLOW_MODE_RUNNING */
        return (req->flags & UCP_REQUEST_FLAG_DFLOW);
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_dflow_stats_init(ucp_proto_dflow_stats_t *stats,
                           uct_completion_callback_t comp_func,
                           uct_completion_t *parent, int count)
{
    stats->comp.func   = comp_func;
    stats->comp.count  = count;
    stats->comp.status = UCS_OK;
    stats->parent      = parent;
    stats->start_time  = ucs_get_time();
}

static void
ucp_proto_dflow_node_comp(uct_completion_t *comp)
{
    ucp_proto_dflow_stats_t *stats = ucs_container_of(comp, ucp_proto_dflow_stats_t, comp);
    ucp_proto_dflow_node_t *node   = ucs_container_of(stats, ucp_proto_dflow_node_t, stats);
    ucs_time_t tx_time             = ucs_get_time() - stats->start_time;
    ucp_lane_index_t i;

    stats->latency_sum += tx_time;
    ucs_assert(stats->parent->count == node->num_lanes);
    stats->parent->count = 1;
    ucp_invoke_uct_completion(stats->parent, comp->status);

    /* TODO: upload data to LB */
    node->mode = UCP_PROTO_DFLOW_MODE_READY;

    if (++node->num_samples % 10000 == 0) {
        ucs_diag("dflow samples=%d latency_sum=%.2f", node->num_samples, ucs_time_to_msec(node->stats.latency_sum));
        for (i = 0; i < node->num_lanes; ++i) {
            ucs_diag("  lane %d latency_sum=%.2f", i, ucs_time_to_msec(node->lanes[i].stats.latency_sum));
            node->lanes[i].stats.latency_sum = 0;
        }
        node->num_samples = 0;
        node->stats.latency_sum = 0;
    }
}

static void
ucp_proto_dflow_lane_comp(uct_completion_t *comp)
{
    ucp_proto_dflow_stats_t *stats = ucs_container_of(comp, ucp_proto_dflow_stats_t, comp);
    ucs_time_t tx_time             = ucs_get_time() - stats->start_time;

    stats->latency_sum += tx_time;
    ucp_invoke_uct_completion(stats->parent, comp->status);
}

static UCS_F_ALWAYS_INLINE ucp_proto_dflow_lane_t *
ucp_proto_dflow_setup(const ucp_proto_dflow_node_t *cnode,
                      ucp_request_t *req,
                      ucp_lane_index_t lane_idx)
{
    /* TODO: get rid of const cast */
    ucp_proto_dflow_node_t *node = (ucp_proto_dflow_node_t *)cnode;
    ucp_proto_dflow_lane_t *lane;

    if (lane_idx == 0) {
        node->mode  = UCP_PROTO_DFLOW_MODE_RUNNING;
        req->flags |= UCP_REQUEST_FLAG_DFLOW;

        ucp_proto_dflow_stats_init(&node->stats, ucp_proto_dflow_node_comp,
                                   &req->send.state.uct_comp, node->num_lanes);
    } else {
        ucs_assert(req->flags & UCP_REQUEST_FLAG_DFLOW);
        ucs_assert(node->mode == UCP_PROTO_DFLOW_MODE_RUNNING);

        if (lane_idx == (node->num_lanes - 1)) {
            node->mode = UCP_PROTO_DFLOW_MODE_WAITING;
        }
    }

    lane = &node->lanes[lane_idx];
    ucp_proto_dflow_stats_init(&lane->stats, ucp_proto_dflow_lane_comp,
                               &node->stats.comp, 1);
    return lane;
}

static UCS_F_ALWAYS_INLINE uct_completion_t *
ucp_proto_dflow_get_completion(const ucp_proto_dflow_lane_t *lane, ucp_request_t *req)
{
    if (ucs_unlikely(req->flags & UCP_REQUEST_FLAG_DFLOW)) {
        /* TODO: get rid of const cast */
        return (uct_completion_t *)&lane->stats.comp;
    }

    return &req->send.state.uct_comp;
}

#endif
