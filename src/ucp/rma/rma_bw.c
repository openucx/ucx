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

static ucp_rma_bw_lane_estimate_t *
ucp_rma_bw_estimator_lane(ucp_rma_bw_estimator_t *estimator,
                          unsigned lane_idx)
{
    return &estimator->lanes[lane_idx];
}

int ucp_rma_bw_estimator_is_valid(const ucp_rma_bw_estimator_t *estimator,
                                  ucp_rma_bw_dir_t dir, uint64_t epoch,
                                  uint32_t generation,
                                  ucp_worker_cfg_index_t cfg_index,
                                  ucs_time_t now)
{
    const ucp_rma_bw_lane_estimate_t *lane;
    unsigned i;

    if ((estimator == NULL) || (dir >= UCP_RMA_BW_DIR_LAST) ||
        (estimator->dir != dir) || (estimator->epoch != epoch) ||
        (estimator->generation != generation) ||
        (estimator->cfg_index != cfg_index)) {
        return 0;
    }

    for (i = 0; i < estimator->num_lanes; ++i) {
        lane = &estimator->lanes[i];
        if ((lane->nominal <= 0.0) || (lane->samples == 0) ||
            (now < lane->last_update) ||
            (now - lane->last_update >
             ucs_time_from_sec(UCP_RMA_BW_STALE_INTERVAL))) {
            return 0;
        }
    }

    return 1;
}

ucp_rma_bw_reject_t
ucp_rma_bw_estimator_update(ucp_rma_bw_estimator_t *estimator,
                            const ucp_rma_bw_sample_t *sample, ucs_time_t now)
{
    ucp_rma_bw_lane_estimate_t *lane;
    ucp_rma_bw_reject_t reject = UCP_RMA_BW_REJECT_NONE;
    unsigned i;

    if (sample->invalid) {
        reject = UCP_RMA_BW_REJECT_INCOMPLETE;
        goto out;
    }

    if ((sample->epoch != estimator->epoch) ||
        (sample->generation != estimator->generation) ||
        (sample->cfg_index != estimator->cfg_index) ||
        (sample->num_lanes != estimator->num_lanes) ||
        (sample->dir != estimator->dir)) {
        reject = UCP_RMA_BW_REJECT_GENERATION;
        goto out;
    }

    lane = ucp_rma_bw_estimator_lane(estimator, 0);
    if ((lane->samples > 0) && (now >= lane->last_update) &&
        (now - lane->last_update <
         ucs_time_from_sec(UCP_RMA_BW_SAMPLE_INTERVAL))) {
        reject = UCP_RMA_BW_REJECT_RATE_LIMIT;
        goto out;
    }

    /* Validate the whole transfer before modifying any lane estimate. */
    if (!sample->concurrent) {
        reject = UCP_RMA_BW_REJECT_UNSATURATED;
        goto out;
    }
    for (i = 0; i < sample->num_lanes; ++i) {
        const ucp_rma_bw_lane_t *observation = &sample->lanes[i];

        lane = ucp_rma_bw_estimator_lane(estimator, i);
        if ((lane->lane_id != sample->lane_ids[i]) ||
            (lane->nominal <= 0.0)) {
            reject = UCP_RMA_BW_REJECT_GENERATION;
            goto out;
        }

        if ((observation->bytes < UCP_RMA_BW_MIN_LANE_LENGTH) ||
            (observation->num_async == 0) ||
            (observation->last_comp <= observation->first_post)) {
            reject = UCP_RMA_BW_REJECT_UNSATURATED;
            goto out;
        }
    }

    for (i = 0; i < sample->num_lanes; ++i) {
        const ucp_rma_bw_lane_t *observation = &sample->lanes[i];
        double raw;

        lane = ucp_rma_bw_estimator_lane(estimator, i);
        raw  = observation->bytes /
               ucs_time_to_sec(observation->last_comp -
                               observation->first_post);
        if ((lane->samples == 0) || (now < lane->last_update) ||
            (now - lane->last_update >
             ucs_time_from_sec(UCP_RMA_BW_STALE_INTERVAL))) {
            lane->smoothed = lane->nominal;
        }

        /* A single low observation can reduce the previous estimate by
         * at most 90%; the EWMA uses the DOCA model's alpha of 0.3. */
        lane->smoothed   = 0.7 * lane->smoothed +
                           0.3 * ucs_max(raw, 0.1 * lane->smoothed);
        lane->last_update = now;
        ++lane->samples;
    }

    ++estimator->accepted;

out:
    if (reject != UCP_RMA_BW_REJECT_NONE) {
        ++estimator->rejected[reject];
    }
    return reject;
}

static double ucp_rma_bw_nominal(ucp_ep_h ep, ucp_lane_index_t lane_id)
{
    ucp_worker_h worker = ep->worker;
    ucp_rsc_index_t rsc_index;
    ucp_worker_iface_t *wiface;
    uct_perf_attr_t perf_attr;
    ucs_status_t status;

    rsc_index            = ucp_ep_config(ep)->key.lanes[lane_id].rsc_index;
    wiface               = ucp_worker_iface(worker, rsc_index);
    perf_attr.field_mask = UCT_PERF_ATTR_FIELD_BANDWIDTH;
    status               = uct_iface_estimate_perf(wiface->iface, &perf_attr);
    if (status == UCS_OK) {
        return ucp_tl_iface_bandwidth(worker->context, &perf_attr.bandwidth);
    }

    return ucp_worker_iface_bandwidth(worker, rsc_index);
}

static ucp_rma_bw_estimator_t *
ucp_rma_bw_estimator_prepare(ucp_ep_h ep, const ucp_rma_bw_sample_t *sample)
{
    ucp_rma_bw_ep_state_t *state = ep->ext->rma_bw_state;
    ucp_rma_bw_estimator_t *estimator;
    ucp_rma_bw_lane_estimate_t *lane;
    size_t size;
    unsigned i;

    if (state == NULL) {
        state = ucs_calloc(1, sizeof(*state), "rma_bw_ep_state");
        if (state == NULL) {
            return NULL;
        }
        ep->ext->rma_bw_state = state;
    }

    estimator = state->dirs[sample->dir];
    if ((estimator != NULL) && (estimator->num_lanes != sample->num_lanes)) {
        ucs_free(estimator);
        estimator = NULL;
        state->dirs[sample->dir] = NULL;
    }

    if (estimator == NULL) {
        size = sizeof(*estimator) +
               sample->num_lanes * sizeof(estimator->lanes[0]);
        estimator = ucs_calloc(1, size, "rma_bw_estimator");
        if (estimator == NULL) {
            return NULL;
        }

        estimator->num_lanes     = sample->num_lanes;
        estimator->dir           = sample->dir;
        state->dirs[sample->dir] = estimator;
    }

    if ((estimator->epoch != ep->worker->epoch) ||
        (estimator->generation != ep->ext->rma_bw_generation) ||
        (estimator->cfg_index != ep->cfg_index)) {
        memset(estimator->lanes, 0,
               estimator->num_lanes * sizeof(estimator->lanes[0]));
        estimator->epoch      = ep->worker->epoch;
        estimator->generation = ep->ext->rma_bw_generation;
        estimator->cfg_index  = ep->cfg_index;
    }

    for (i = 0; i < sample->num_lanes; ++i) {
        lane = ucp_rma_bw_estimator_lane(estimator, i);
        if ((lane->nominal == 0.0) ||
            (lane->lane_id != sample->lane_ids[i])) {
            lane->nominal     = ucp_rma_bw_nominal(ep, sample->lane_ids[i]);
            lane->smoothed    = lane->nominal;
            lane->last_update = 0;
            lane->samples     = 0;
            lane->lane_id     = sample->lane_ids[i];
        }
    }

    return estimator;
}

void ucp_rma_bw_estimator_free(ucp_ep_h ep)
{
    ucp_rma_bw_ep_state_t *state = ep->ext->rma_bw_state;
    unsigned dir;

    if (state == NULL) {
        return;
    }

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
    sample->req        = req;
    sample->num_lanes  = num_lanes;
    sample->dir        = dir;
    sample->epoch      = worker->epoch;
    sample->generation = req->send.ep->ext->rma_bw_generation;
    sample->cfg_index  = req->send.ep->cfg_index;
    for (i = 0; i < num_lanes; ++i) {
        sample->lane_ids[i] = mpriv->lanes[i].super.lane;
    }

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
    ucp_rma_bw_estimator_t *estimator;
    ucp_rma_bw_reject_t reject;
    ucs_time_t now;
    unsigned i;

    sample->invalid |= (comp->status != UCS_OK);
    if ((sample->epoch != req->send.ep->worker->epoch) ||
        (sample->generation != req->send.ep->ext->rma_bw_generation) ||
        (sample->cfg_index != req->send.ep->cfg_index)) {
        ucs_debug("ep %p req %p: rma bw sample rejected after EP change",
                  req->send.ep, req);
        goto out;
    }

    estimator = ucp_rma_bw_estimator_prepare(req->send.ep, sample);
    if (estimator != NULL) {
        now    = ucs_get_time();
        reject = ucp_rma_bw_estimator_update(estimator, sample, now);
        if (reject == UCP_RMA_BW_REJECT_NONE) {
            for (i = 0; i < sample->num_lanes; ++i) {
                ucs_debug("ep %p req %p: rma bw %s lane %u/%u ep_lane %u: "
                              "%zu bytes / %.3f us, raw %.2f MB/s, "
                              "nominal %.2f MB/s, ewma %.2f MB/s",
                              req->send.ep, req,
                              sample->dir == UCP_RMA_BW_PUT ? "PUT" : "GET",
                              i, sample->num_lanes, sample->lane_ids[i],
                              sample->lanes[i].bytes,
                              ucs_time_to_sec(sample->lanes[i].last_comp -
                                              sample->lanes[i].first_post) * 1e6,
                              sample->lanes[i].bytes /
                              ucs_time_to_sec(sample->lanes[i].last_comp -
                                              sample->lanes[i].first_post) /
                              UCS_MBYTE,
                              ucp_rma_bw_estimator_lane(estimator, i)->nominal /
                              UCS_MBYTE,
                              ucp_rma_bw_estimator_lane(estimator, i)->smoothed /
                              UCS_MBYTE);
            }
        } else {
            ucs_debug("ep %p req %p: rma bw %s sample rejected: reason %u, "
                          "epoch %lu/%lu, cfg %u/%u",
                          req->send.ep, req,
                          sample->dir == UCP_RMA_BW_PUT ? "PUT" : "GET",
                          reject, sample->epoch, req->send.ep->worker->epoch,
                          sample->cfg_index, req->send.ep->cfg_index);
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
