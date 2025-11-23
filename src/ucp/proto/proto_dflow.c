/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_dflow.h"
#include <ucp/core/ucp_worker.h>


/* TODO: Move config here */

static unsigned ucp_proto_dflow_service_progress(void *arg)
{
    ucp_proto_dflow_service_t *service = arg;
    ucs_time_t now                     = ucs_get_time();
    ucp_proto_dflow_node_t *node       = service->node;
    ucp_lane_index_t i;

    if (ucs_likely(now < service->next_progress_time)) {
        return 0;
    }

    if (node && node->mode == UCP_PROTO_DFLOW_MODE_IDLE) {
        ucs_diag("dflow samples=%d latency_sum=%.2f", node->num_samples, ucs_time_to_msec(node->stats.latency_sum));
        for (i = 0; i < node->num_lanes; ++i) {
            ucs_diag("  lane %d latency_sum=%.2f", i, ucs_time_to_msec(node->lanes[i].stats.latency_sum));
            node->lanes[i].stats.latency_sum = 0;
        }
        node->num_samples = 0;
        node->stats.latency_sum = 0;
        node->mode = UCP_PROTO_DFLOW_MODE_READY;
    }

    service->next_progress_time = now + service->interval;
    return 1;
}

ucs_status_t ucp_proto_dflow_service_init(ucp_worker_h worker,
                                          ucp_proto_dflow_service_t *service)
{
    service->interval             = ucs_time_from_sec(1.0);
    service->progress_cb_id       = UCS_CALLBACKQ_ID_NULL;
    service->next_progress_time   = ucs_get_time() + service->interval;
    service->num_samples_interval = 100;
    service->node                 = NULL;

    uct_worker_progress_register_safe(worker->uct,
                                      ucp_proto_dflow_service_progress, service,
                                      0, &service->progress_cb_id);
    return UCS_OK;
}

void ucp_proto_dflow_service_cleanup(ucp_worker_h worker,
                                     ucp_proto_dflow_service_t *service)
{
    uct_worker_progress_unregister_safe(worker->uct, &service->progress_cb_id);
}
