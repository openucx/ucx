/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rma_bw.inl"

#include <ucp/core/ucp_request.inl>
#include <ucp/proto/proto_common.inl>

void ucp_rma_bw_sample_start(ucp_request_t *req,
                             ucp_lane_index_t num_lanes)
{
    ucp_worker_h worker = req->send.ep->worker;
    ucp_rma_bw_sample_t *sample;
    ucs_time_t now;
    unsigned i;

    if (ucs_likely(!worker->context->config.ext.rma_bw_measure) ||
        (num_lanes < 2) ||
        (num_lanes > UCP_MAX_LANES) ||
        (req->send.state.dt_iter.length <
         UCP_RMA_BW_MIN_LANE_LENGTH * num_lanes)) {
        return;
    }

    now = ucs_get_time();
    if (now < worker->rma_bw_next_sample) {
        return;
    }

    for (i = 0; i < UCP_RMA_BW_MAX_ACTIVE; ++i) {
        sample = &worker->rma_bw_samples[i];
        if (ucp_rma_bw_sample_is_free(sample)) {
            break;
        }
    }
    if (i == UCP_RMA_BW_MAX_ACTIVE) {
        return;
    }

    memset(sample, 0, sizeof(*sample));
    sample->req                = req;
    sample->num_lanes          = num_lanes;
    req->send.rma.bw_sample    = sample;
    req->flags                |= UCP_REQUEST_FLAG_RMA_BW_SAMPLE;
    worker->rma_bw_next_sample = now +
                                 ucs_time_from_sec(UCP_RMA_BW_SAMPLE_INTERVAL);
}

void ucp_rma_bw_sample_complete(uct_completion_t *comp)
{
    ucp_request_t *req = ucs_container_of(comp, ucp_request_t,
                                          send.state.uct_comp);
    ucp_rma_bw_sample_t *sample = ucp_rma_bw_sample_get(req);
    ucp_rma_bw_lane_t *lane;
    unsigned i;

    if ((comp->status == UCS_OK) && !sample->invalid) {
        for (i = 0; i < sample->num_lanes; ++i) {
            lane = &sample->lanes[i];
            if ((lane->bytes == 0) || (lane->num_async == 0) ||
                (lane->last_comp <= lane->first_post)) {
                break;
            }
        }

        if (i == sample->num_lanes) {
            for (i = 0; i < sample->num_lanes; ++i) {
                ucp_trace_req(req, "rma bw sample lane %u/%u: %zu bytes / "
                              "%.3f us", i, sample->num_lanes,
                              sample->lanes[i].bytes,
                              ucs_time_to_sec(sample->lanes[i].last_comp -
                                              sample->lanes[i].first_post) *
                              1e6);
            }
        }
    }

    req->flags             &= ~UCP_REQUEST_FLAG_RMA_BW_SAMPLE;
    req->send.rma.bw_sample = NULL;
    sample->req             = NULL;
    ucs_assert(sample->pending == 0);
    ucp_proto_request_zcopy_complete(req, comp->status);
}

void ucp_rma_bw_frag_complete(uct_completion_t *comp)
{
    ucp_rma_bw_frag_t *frag = ucs_container_of(comp, ucp_rma_bw_frag_t, comp);
    ucp_rma_bw_sample_t *sample = frag->sample;
    ucp_rma_bw_lane_t *lane     = &sample->lanes[frag->lane_idx];

    lane->last_comp = ucs_get_time();
    sample->invalid |= (comp->status != UCS_OK);
    ucs_assert(sample->pending > 0);
    --sample->pending;
    if (sample->req == NULL) {
        return;
    }

    ucp_invoke_uct_completion(&sample->req->send.state.uct_comp, comp->status);
}

void ucp_rma_bw_sample_detach(ucp_request_t *req)
{
    ucp_rma_bw_sample_t *sample = ucp_rma_bw_sample_get(req);

    if (sample == NULL) {
        return;
    }

    sample->req = NULL;
    /* Overflow fragments may still use the request completion after reset. */
    req->send.state.uct_comp.func = ucp_proto_request_zcopy_completion;
    req->flags                   &= ~UCP_REQUEST_FLAG_RMA_BW_SAMPLE;
    req->send.rma.bw_sample       = NULL;
}
