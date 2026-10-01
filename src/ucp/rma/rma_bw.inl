/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_RMA_BW_INL_
#define UCP_RMA_BW_INL_

#include "rma_bw.h"

#include <ucp/proto/proto_multi.h>

/* A detached request keeps its slot until every UCT fragment completes. */
static UCS_F_ALWAYS_INLINE int
ucp_rma_bw_sample_is_free(const ucp_rma_bw_sample_t *sample)
{
    return (sample->req == NULL) && (sample->pending == 0);
}

static UCS_F_ALWAYS_INLINE ucp_rma_bw_sample_t *
ucp_rma_bw_sample_get(ucp_request_t *req)
{
    return (req->flags & UCP_REQUEST_FLAG_RMA_BW_SAMPLE) ?
           req->send.rma.bw_sample : NULL;
}

void ucp_rma_bw_sample_start(ucp_request_t *req,
                             const ucp_proto_multi_priv_t *mpriv);

static UCS_F_ALWAYS_INLINE void
ucp_rma_bw_sample_try_start(ucp_request_t *req,
                            const ucp_proto_multi_priv_t *mpriv)
{
    if (ucs_unlikely(req->send.ep->worker->context->config.ext.rma_bw_measure)) {
        ucp_rma_bw_sample_start(req, mpriv);
    }
}

static UCS_F_ALWAYS_INLINE uct_completion_t *
ucp_rma_bw_frag_start(ucp_request_t *req, ucp_lane_index_t lane_idx,
                      ucp_rma_bw_frag_t **frag_p)
{
    ucp_rma_bw_sample_t *sample = ucp_rma_bw_sample_get(req);
    ucp_rma_bw_frag_t *frag;

    *frag_p = NULL;
    if (sample == NULL) {
        return &req->send.state.uct_comp;
    }

    ucs_assert(lane_idx < sample->num_lanes);
    if (sample->num_frags == UCP_RMA_BW_MAX_FRAGS) {
        sample->invalid = 1;
        return &req->send.state.uct_comp;
    }

    frag              = &sample->frags[sample->num_frags];
    frag->sample      = sample;
    frag->lane_idx    = lane_idx;
    frag->post_time   = ucs_get_time();
    frag->comp.func   = ucp_rma_bw_frag_complete;
    frag->comp.count  = 1;
    frag->comp.status = UCS_OK;
    *frag_p           = frag;
    return &frag->comp;
}

static UCS_F_ALWAYS_INLINE void
ucp_rma_bw_frag_posted(ucp_rma_bw_frag_t *frag, size_t bytes,
                       ucs_status_t status)
{
    ucp_rma_bw_sample_t *sample;
    ucp_rma_bw_lane_t *lane;
    ucs_time_t now;

    if ((frag == NULL) || UCS_STATUS_IS_ERR(status)) {
        return;
    }

    sample = frag->sample;
    lane   = &sample->lanes[frag->lane_idx];
    if (lane->num_frags == 0) {
        lane->first_post = frag->post_time;
    }
    lane->bytes += bytes;
    ++lane->num_frags;
    if (ucs_likely(status == UCS_INPROGRESS)) {
        ++lane->num_async;
        ++sample->num_frags;
        ++sample->pending;
        return;
    }

    ucs_assert(status == UCS_OK);
    now = ucs_get_time();
    if (now > lane->last_comp) {
        lane->last_comp = now;
    }
}

#endif
