/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rma_bw.inl"

#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_request.inl>
#include <ucp/core/ucp_worker.inl>
#include <ucp/proto/proto_common.inl>
#include <ucp/proto/proto_multi.h>

ucp_rma_bw_reject_t
ucp_rma_bw_estimator_update(ucp_rma_bw_estimator_t *estimator,
                            const ucp_rma_bw_sample_t *sample, ucs_time_t now)
{
    int warmup = 0;
    ucp_rma_bw_lane_estimate_t *lane;
    unsigned i;

    if (sample->invalid) {
        return UCP_RMA_BW_REJECT_INCOMPLETE;
    }

    if ((sample->epoch != estimator->epoch) ||
        (sample->generation != estimator->generation) ||
        (sample->num_lanes != estimator->num_lanes)) {
        return UCP_RMA_BW_REJECT_GENERATION;
    }

    lane = &estimator->lanes[0];
    if ((lane->samples > 0) && (now >= lane->last_update) &&
        (now - lane->last_update <
         ucs_time_from_sec(UCP_RMA_BW_SAMPLE_INTERVAL))) {
        return UCP_RMA_BW_REJECT_RATE_LIMIT;
    }

    /* Validate the whole transfer before modifying any lane estimate. */
    if (!sample->concurrent) {
        return UCP_RMA_BW_REJECT_UNSATURATED;
    }
    for (i = 0; i < sample->num_lanes; ++i) {
        const ucp_rma_bw_lane_t *observation = &sample->lanes[i];

        lane = &estimator->lanes[i];
        if ((lane->lane_id != sample->lane_ids[i]) || (lane->nominal <= 0.0)) {
            return UCP_RMA_BW_REJECT_GENERATION;
        }

        if ((observation->bytes < UCP_RMA_BW_MIN_LANE_LENGTH) ||
            (observation->num_async == 0) ||
            (observation->last_comp <= observation->first_post)) {
            return UCP_RMA_BW_REJECT_UNSATURATED;
        }

        if (lane->marker_time == 0) {
            warmup = 1;
        } else if ((observation->posted_total <= lane->marker_posted) ||
                   (observation->last_comp <= lane->marker_time)) {
            return UCP_RMA_BW_REJECT_WINDOW;
        }
    }

    if (warmup) {
        /* The first completed request sets a stream marker. Its queue wait
         * cannot be separated from service time without a prior marker. */
        for (i = 0; i < sample->num_lanes; ++i) {
            lane                = &estimator->lanes[i];
            lane->marker_posted = sample->lanes[i].posted_total;
            lane->marker_time   = sample->lanes[i].last_comp;
        }
        return UCP_RMA_BW_REJECT_WARMUP;
    }

    for (i = 0; i < sample->num_lanes; ++i) {
        const ucp_rma_bw_lane_t *observation = &sample->lanes[i];
        double request_rate, stream_rate, raw;

        lane         = &estimator->lanes[i];
        request_rate = observation->bytes /
                       ucs_time_to_sec(observation->last_comp -
                                       observation->first_post);
        stream_rate  = (observation->posted_total - lane->marker_posted) /
                       ucs_time_to_sec(observation->last_comp -
                                       lane->marker_time);
        /* The stream rate removes the old queue wait. An underfilled lane
         * can complete this request faster than the aggregate stream rate. */
        raw          = ucs_max(request_rate, stream_rate);
        if ((lane->samples == 0) || (now < lane->last_update) ||
            (now - lane->last_update >
             ucs_time_from_sec(UCP_RMA_BW_STALE_INTERVAL))) {
            lane->smoothed = lane->nominal;
        }

        /* A single low observation can reduce the previous estimate by
         * at most 27%; the EWMA uses the DOCA model's alpha of 0.3. */
        lane->smoothed      = 0.7 * lane->smoothed +
                              0.3 * ucs_max(raw, 0.1 * lane->smoothed);
        lane->last_update   = now;
        lane->marker_posted = observation->posted_total;
        lane->marker_time   = observation->last_comp;
        ++lane->samples;
    }

    return UCP_RMA_BW_REJECT_NONE;
}

static double
ucp_rma_bw_nominal(ucp_ep_h ep, ucp_lane_index_t lane_id, ucp_rma_bw_dir_t dir)
{
    ucp_worker_h worker = ep->worker;
    ucp_rsc_index_t rsc_index;
    ucp_worker_iface_t *wiface;
    uct_perf_attr_t perf_attr;
    ucs_status_t status;

    if (lane_id >= ucp_ep_config(ep)->key.num_lanes) {
        return 0.0;
    }

    rsc_index            = ucp_ep_config(ep)->key.lanes[lane_id].rsc_index;
    wiface               = ucp_worker_iface(worker, rsc_index);
    perf_attr.field_mask = UCT_PERF_ATTR_FIELD_OPERATION |
                           UCT_PERF_ATTR_FIELD_BANDWIDTH;
    perf_attr.operation  = (dir == UCP_RMA_BW_PUT) ? UCT_EP_OP_PUT_ZCOPY :
                                                      UCT_EP_OP_GET_ZCOPY;
    status               = ucp_worker_iface_estimate_perf(wiface, &perf_attr);
    if (status == UCS_OK) {
        return ucp_tl_iface_bandwidth(worker->context, &perf_attr.bandwidth);
    }

    return ucp_worker_iface_bandwidth(worker, rsc_index);
}

static ucp_rma_bw_estimator_t *
ucp_rma_bw_estimator_prepare(ucp_ep_h ep, const ucp_rma_bw_sample_t *sample)
{
    ucp_rma_bw_ep_state_t *state = ucp_worker_rma_bw_state_get(ep);
    ucp_rma_bw_estimator_t *estimator;
    ucp_rma_bw_lane_estimate_t *lane;
    kh_ucp_worker_rma_bw_t *hash;
    khiter_t it;
    int ret;
    size_t size;
    unsigned i;

    if (state == NULL) {
        state = ucs_calloc(1, sizeof(*state), "rma_bw_ep_state");
        if (state == NULL) {
            return NULL;
        }

        hash = &ep->worker->rma_bw_hash;
        it   = kh_put(ucp_worker_rma_bw, hash, ep, &ret);
        if (ret == UCS_KH_PUT_FAILED) {
            ucs_free(state);
            return NULL;
        }

        ucs_assert((ret == UCS_KH_PUT_BUCKET_EMPTY) ||
                   (ret == UCS_KH_PUT_BUCKET_CLEAR));
        kh_value(hash, it) = state;
    }

    estimator = state->dirs[sample->dir];
    if ((estimator != NULL) && (estimator->num_lanes != sample->num_lanes)) {
        ucs_free(estimator);
        estimator                = NULL;
        state->dirs[sample->dir] = NULL;
    }

    if (estimator == NULL) {
        size      = sizeof(*estimator) +
                    sample->num_lanes * sizeof(estimator->lanes[0]);
        estimator = ucs_calloc(1, size, "rma_bw_estimator");
        if (estimator == NULL) {
            return NULL;
        }

        estimator->num_lanes     = sample->num_lanes;
        state->dirs[sample->dir] = estimator;
    }

    if ((estimator->epoch != ep->worker->epoch) ||
        (estimator->generation != state->generation)) {
        memset(estimator->lanes, 0,
               estimator->num_lanes * sizeof(estimator->lanes[0]));
        estimator->epoch      = ep->worker->epoch;
        estimator->generation = state->generation;
    }

    for (i = 0; i < sample->num_lanes; ++i) {
        lane = &estimator->lanes[i];
        if ((lane->nominal == 0.0) || (lane->lane_id != sample->lane_ids[i])) {
            memset(lane, 0, sizeof(*lane));
            lane->nominal  = ucp_rma_bw_nominal(ep, sample->lane_ids[i],
                                                sample->dir);
            lane->smoothed = lane->nominal;
            lane->lane_id  = sample->lane_ids[i];
        }
    }

    return estimator;
}

void ucp_rma_bw_ep_state_cleanup(ucp_ep_h ep)
{
    kh_ucp_worker_rma_bw_t *hash = &ep->worker->rma_bw_hash;
    khiter_t it                  = kh_get(ucp_worker_rma_bw, hash, ep);
    ucp_rma_bw_ep_state_t *state;
    unsigned dir;

    if (it == kh_end(hash)) {
        return;
    }

    state = kh_value(hash, it);
    kh_del(ucp_worker_rma_bw, hash, it);
    for (dir = 0; dir < UCP_RMA_BW_DIR_LAST; ++dir) {
        ucs_free(state->dirs[dir]);
    }
    ucs_free(state);
}

void ucp_rma_bw_sample_start(ucp_request_t *req, ucp_lane_index_t num_lanes,
                             ucp_rma_bw_dir_t dir)
{
    ucp_worker_h worker = req->send.ep->worker;
    const ucp_proto_multi_priv_t *mpriv = req->send.proto_config->priv;
    ucp_rma_bw_sample_t *sample;
    ucp_rma_bw_estimator_t *estimator;
    ucs_time_t now;
    unsigned i;

    if ((num_lanes < 2) || (num_lanes > UCP_MAX_LANES) ||
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
    sample->req       = req;
    sample->num_lanes = num_lanes;
    sample->dir       = dir;
    sample->epoch     = worker->epoch;
    for (i = 0; i < num_lanes; ++i) {
        sample->lane_ids[i] = mpriv->lanes[i].super.lane;
    }

    estimator = ucp_rma_bw_estimator_prepare(req->send.ep, sample);
    if (estimator == NULL) {
        sample->req = NULL;
        return;
    }
    sample->generation = estimator->generation;

    req->send.rma.bw_sample    = sample;
    req->flags                |= UCP_REQUEST_FLAG_RMA_BW_SAMPLE;
    worker->rma_bw_next_sample = now +
                                 ucs_time_from_sec(UCP_RMA_BW_SAMPLE_INTERVAL);
}

static const char *ucp_rma_bw_reject_name(ucp_rma_bw_reject_t reject)
{
    static const char *names[] = {
        [UCP_RMA_BW_REJECT_NONE]        = "none",
        [UCP_RMA_BW_REJECT_INCOMPLETE]  = "incomplete",
        [UCP_RMA_BW_REJECT_UNSATURATED] = "unsaturated",
        [UCP_RMA_BW_REJECT_GENERATION]  = "generation",
        [UCP_RMA_BW_REJECT_RATE_LIMIT]  = "rate_limit",
        [UCP_RMA_BW_REJECT_WARMUP]      = "warmup",
        [UCP_RMA_BW_REJECT_WINDOW]      = "window"
    };

    ucs_assert(reject < ucs_static_array_size(names));
    return names[reject];
}

void ucp_rma_bw_sample_complete(uct_completion_t *comp)
{
    ucp_request_t *req = ucs_container_of(comp, ucp_request_t,
                                          send.state.uct_comp);
    ucp_rma_bw_sample_t *sample = ucp_rma_bw_sample_get(req);
    ucp_rma_bw_ep_state_t *state = ucp_worker_rma_bw_state_get(req->send.ep);
    ucp_rma_bw_estimator_t *estimator;
    ucp_rma_bw_reject_t reject;
    ucs_time_t now;
    unsigned i;

    sample->invalid |= (comp->status != UCS_OK);
    if ((state == NULL) || (sample->epoch != req->send.ep->worker->epoch) ||
        (sample->generation != state->generation)) {
        ucp_trace_req(req, "rma bw sample rejected after EP change");
        goto out;
    }

    estimator = ucp_rma_bw_estimator_prepare(req->send.ep, sample);
    if (estimator != NULL) {
        now    = ucs_get_time();
        reject = ucp_rma_bw_estimator_update(estimator, sample, now);
        if (reject == UCP_RMA_BW_REJECT_NONE) {
            for (i = 0; i < sample->num_lanes; ++i) {
                ucp_trace_req(req,
                              "rma bw %s lane %u/%u ep_lane %u: "
                              "%zu bytes, ewma %.2f MB/s",
                              sample->dir == UCP_RMA_BW_PUT ? "PUT" : "GET", i,
                              sample->num_lanes, sample->lane_ids[i],
                              sample->lanes[i].bytes,
                              estimator->lanes[i].smoothed / UCS_MBYTE);
            }
        } else {
            ucp_trace_req(req,
                          "rma bw %s sample rejected: reason %s, "
                          "epoch %lu/%lu",
                          sample->dir == UCP_RMA_BW_PUT ? "PUT" : "GET",
                          ucp_rma_bw_reject_name(reject), sample->epoch,
                          req->send.ep->worker->epoch);
        }
    }

out:
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
    ucs_assert(lane->pending > 0);
    --sample->pending;
    if (--lane->pending == 0) {
        sample->active_lanes &= ~UCS_BIT(frag->lane_idx);
    }

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
