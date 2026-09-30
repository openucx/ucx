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

void ucp_rma_bw_sample_complete(uct_completion_t *comp)
{
    ucp_request_t *req = ucs_container_of(comp, ucp_request_t,
                                          send.state.uct_comp);
    ucp_rma_bw_sample_t *sample = ucp_rma_bw_sample_get(req);
    ucp_rma_bw_lane_t *lane0     = &sample->lanes[0];
    ucp_rma_bw_lane_t *lane1     = &sample->lanes[1];
    int valid;

    valid = (comp->status == UCS_OK) && !sample->invalid &&
            (lane0->bytes != 0) && (lane1->bytes != 0) &&
            (lane0->last_comp > lane0->first_post) &&
            (lane1->last_comp > lane1->first_post) &&
            (lane0->num_async != 0) && (lane1->num_async != 0);
    if (valid) {
        ucs_trace("rma bw sample req %p: lane0 %zu bytes / %.3f us, "
                  "lane1 %zu bytes / %.3f us", req, lane0->bytes,
                  ucs_time_to_sec(lane0->last_comp - lane0->first_post) * 1e6,
                  lane1->bytes,
                  ucs_time_to_sec(lane1->last_comp - lane1->first_post) * 1e6);
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

void ucp_rma_bw_abort(ucp_request_t *req, ucs_status_t status)
{
    ucp_rma_bw_sample_t *sample = ucp_rma_bw_sample_get(req);

    if (sample != NULL) {
        sample->invalid = 1;
    }
    ucp_proto_request_zcopy_abort(req, status);
}
