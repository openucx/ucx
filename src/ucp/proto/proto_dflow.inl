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
    node->min_length  = 0;
    node->length_sum  = 0;
    node->num_lanes   = num_lanes;
    node->num_samples = 0;

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

static UCS_F_ALWAYS_INLINE ucp_proto_dflow_mode_t
ucp_proto_dflow_service_update(ucp_proto_dflow_service_t *service,
                               ucp_proto_dflow_node_t *node)
{
    service->node = node;

    if (node->num_samples >= service->num_samples_interval) {
        return UCP_PROTO_DFLOW_MODE_IDLE;
    }
    return UCP_PROTO_DFLOW_MODE_READY;
}

static UCS_F_ALWAYS_INLINE ucp_proto_dflow_service_t *
ucp_proto_dflow_service_get(ucp_request_t *req)
{
    return &req->send.ep->worker->dflow_service;
}

static void
ucp_proto_dflow_node_comp(uct_completion_t *comp)
{
    ucp_proto_dflow_stats_t *stats = ucs_container_of(comp, ucp_proto_dflow_stats_t, comp);
    ucp_proto_dflow_node_t *node   = ucs_container_of(stats, ucp_proto_dflow_node_t, stats);
    ucp_request_t *req             = ucs_container_of(stats->parent, ucp_request_t,
                                                      send.state.uct_comp);
    ucp_proto_dflow_service_t *srv = &req->send.ep->worker->dflow_service;
    ucs_time_t tx_time             = ucs_get_time() - stats->start_time;

    ucs_assert(req->flags & UCP_REQUEST_FLAG_DFLOW);
    ucs_assert(node->mode == UCP_PROTO_DFLOW_MODE_WAITING);

    stats->latency_sum   += tx_time;
    stats->parent->count -= node->num_lanes - 1;
    ucp_invoke_uct_completion(stats->parent, comp->status);

    ++node->num_samples;
    node->mode = ucp_proto_dflow_service_update(srv, node);
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
    lane->mode = UCP_PROTO_DFLOW_MODE_READY;
    ucp_proto_dflow_stats_init(&lane->stats, ucp_proto_dflow_lane_comp,
                               &node->stats.comp, 1);
    return lane;
}

static UCS_F_ALWAYS_INLINE uct_completion_t *
ucp_proto_dflow_get_completion(const ucp_proto_dflow_lane_t *clane,
                               ucp_request_t *req)
{
    /* TODO: get rid of const cast */
    ucp_proto_dflow_lane_t *lane = (ucp_proto_dflow_lane_t *)clane;

    if (ucs_unlikely(lane->mode == UCP_PROTO_DFLOW_MODE_READY)) {
        lane->mode = UCP_PROTO_DFLOW_MODE_WAITING;
        return &lane->stats.comp;
    }

    return &req->send.state.uct_comp;
}

#endif
