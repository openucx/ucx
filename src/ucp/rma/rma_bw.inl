/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_RMA_BW_INL_
#define UCP_RMA_BW_INL_

#include <ucp/core/ucp_request.inl>
#include <ucp/proto/proto_common.inl>
#include <ucp/proto/proto_multi.h>
#include <ucs/debug/memtrack_int.h>

/* The first measurement stage records completed payload throughput only. */
#define UCP_RMA_BW_MAX_FRAGS      128
#define UCP_RMA_BW_MAX_ACTIVE     4
#define UCP_RMA_BW_MIN_LENGTH     (256 * UCS_KBYTE)

typedef struct ucp_rma_bw_sample ucp_rma_bw_sample_t;

typedef struct {
    uct_completion_t    comp;
    ucp_rma_bw_sample_t *sample;
    ucs_time_t          post_time;
    uint8_t             lane_idx;
} ucp_rma_bw_frag_t;

typedef struct {
    size_t     bytes;
    ucs_time_t first_post;
    ucs_time_t last_comp;
    unsigned   num_frags;
    unsigned   num_async;
} ucp_rma_bw_lane_t;

struct ucp_rma_bw_sample {
    ucp_request_t     *req;
    ucp_worker_h      worker;
    unsigned          num_frags;
    unsigned          invalid;
    ucp_rma_bw_lane_t lanes[2];
    ucp_rma_bw_frag_t frags[UCP_RMA_BW_MAX_FRAGS];
};

static UCS_F_ALWAYS_INLINE ucp_rma_bw_sample_t *
ucp_rma_bw_sample_get(ucp_request_t *req)
{
    return (req->flags & UCP_REQUEST_FLAG_RMA_BW_SAMPLE) ?
           req->send.rma.bw_sample : NULL;
}

static UCS_F_ALWAYS_INLINE void
ucp_rma_bw_sample_complete(uct_completion_t *comp)
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
        ++sample->worker->rma_bw_valid;
        ucs_trace("rma bw sample req %p: lane0 %zu bytes / %.3f us, "
                  "lane1 %zu bytes / %.3f us", req, lane0->bytes,
                  ucs_time_to_sec(lane0->last_comp - lane0->first_post) * 1e6,
                  lane1->bytes,
                  ucs_time_to_sec(lane1->last_comp - lane1->first_post) * 1e6);
    } else {
        ++sample->worker->rma_bw_rejected;
    }

    --sample->worker->rma_bw_active;
    req->flags &= ~UCP_REQUEST_FLAG_RMA_BW_SAMPLE;
    req->send.rma.bw_sample = NULL;
    ucs_free(sample);
    ucp_proto_request_zcopy_complete(req, comp->status);
}

static void ucp_rma_bw_frag_complete(uct_completion_t *comp)
{
    ucp_rma_bw_frag_t *frag = ucs_container_of(comp, ucp_rma_bw_frag_t, comp);
    ucp_rma_bw_sample_t *sample = frag->sample;
    ucp_rma_bw_lane_t *lane     = &sample->lanes[frag->lane_idx];

    lane->last_comp = ucs_get_time();
    sample->invalid |= (comp->status != UCS_OK);
    ucp_invoke_uct_completion(&sample->req->send.state.uct_comp, comp->status);
}

static UCS_F_ALWAYS_INLINE void
ucp_rma_bw_sample_start(ucp_request_t *req,
                        const ucp_proto_multi_priv_t *mpriv)
{
    ucp_worker_h worker = req->send.ep->worker;
    ucp_rma_bw_sample_t *sample;
    ucs_time_t now;

    if (ucs_likely(!worker->context->config.ext.rma_bw_measure) ||
        (mpriv->num_lanes != 2) ||
        (req->send.state.dt_iter.length < UCP_RMA_BW_MIN_LENGTH) ||
        (worker->rma_bw_active >= UCP_RMA_BW_MAX_ACTIVE)) {
        return;
    }

    now = ucs_get_time();
    if (now < worker->rma_bw_next_sample) {
        return;
    }

    sample = ucs_malloc(sizeof(*sample), "rma_bw_sample");
    if (sample == NULL) {
        return;
    }

    memset(sample, 0, sizeof(*sample));
    sample->req                  = req;
    sample->worker               = worker;
    req->send.rma.bw_sample      = sample;
    req->flags                  |= UCP_REQUEST_FLAG_RMA_BW_SAMPLE;
    worker->rma_bw_next_sample   = now + ucs_time_from_sec(1.0);
    ++worker->rma_bw_active;
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

    if (sample->num_frags == UCP_RMA_BW_MAX_FRAGS) {
        sample->invalid = 1;
        return &req->send.state.uct_comp;
    }

    frag            = &sample->frags[sample->num_frags];
    frag->sample    = sample;
    frag->lane_idx  = lane_idx;
    frag->post_time = ucs_get_time();
    frag->comp.func   = ucp_rma_bw_frag_complete;
    frag->comp.count  = 1;
    frag->comp.status = UCS_OK;
    *frag_p         = frag;
    return &frag->comp;
}

static UCS_F_ALWAYS_INLINE void
ucp_rma_bw_frag_posted(ucp_rma_bw_frag_t *frag, size_t bytes,
                       ucs_status_t status)
{
    ucp_rma_bw_sample_t *sample;
    ucp_rma_bw_lane_t *lane;
    ucs_time_t now;

    if ((frag == NULL) || ((status != UCS_OK) &&
                           (status != UCS_INPROGRESS))) {
        return;
    }

    sample = frag->sample;
    lane   = &sample->lanes[frag->lane_idx];
    now    = ucs_get_time();
    if (lane->num_frags == 0) {
        lane->first_post = frag->post_time;
    }
    lane->bytes += bytes;
    ++lane->num_frags;
    if (status == UCS_INPROGRESS) {
        ++lane->num_async;
        ++sample->num_frags;
    } else if (now > lane->last_comp) {
        lane->last_comp = now;
    }
}

static void ucp_rma_bw_abort(ucp_request_t *req, ucs_status_t status)
{
    ucp_rma_bw_sample_t *sample = ucp_rma_bw_sample_get(req);

    if (sample != NULL) {
        sample->invalid = 1;
    }
    ucp_proto_request_zcopy_abort(req, status);
}

#endif
