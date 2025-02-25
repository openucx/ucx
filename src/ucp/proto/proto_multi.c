/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <float.h>

#include "proto_init.h"
#include "proto_debug.h"
#include "proto_common.inl"
#include "proto_debug.h"
#include "proto_multi.inl"

#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>


typedef struct {
    ucp_lane_index_t           lane;
    ucp_proto_common_tl_perf_t perf;
    ucp_proto_perf_node_t      *perf_node;
} ucp_proto_lane_perf_t;

typedef struct {
    ucp_worker_h               worker;
    ucp_proto_lane_perf_t      lanes_perf[UCP_PROTO_MAX_LANES];
    ucp_lane_index_t           length;
    int                        fixed_first_lane;
} ucp_proto_lane_storage_t;

typedef struct {
    ucp_lane_index_t           lanes[UCP_PROTO_MAX_LANES];
    ucp_lane_index_t           length;
    ucp_proto_perf_node_t      *root_node;
    ucp_proto_lane_storage_t   *storage;
} ucp_proto_lane_selection_t;


static void ucp_proto_storage_destroy(ucp_proto_lane_storage_t *storage)
{
    ucp_lane_index_t i;

    for (i = 0; i < storage->length; ++i) {
        ucp_proto_perf_node_deref(&storage->lanes_perf[i].perf_node);
    }

    ucs_free(storage);
}

static ucs_status_t
ucp_proto_storage_create(const ucp_proto_common_init_params_t *params,
                         const ucp_proto_lane_desc_t *lane_desc,
                         const ucp_proto_lane_desc_t *first_lane_desc,
                         ucp_lane_index_t max_lanes, ucp_lane_map_t exclude_map,
                         ucp_proto_lane_storage_t **storage)
{
    int has_first_lane = (first_lane_desc != NULL) &&
                         (first_lane_desc->lane_type != lane_desc->lane_type);
    ucp_lane_index_t lanes[UCP_PROTO_MAX_LANES];
    ucp_lane_index_t length, i;
    ucp_proto_lane_perf_t *lane_iter;
    ucs_status_t status;

    /* Find first lane */
    length = 0;
    if (has_first_lane) {
        length = ucp_proto_common_find_lanes_with_min_frag(
                    params, first_lane_desc->lane_type,
                    first_lane_desc->tl_cap_flags, 1, 0, lanes);
        if (length == 0) {
            ucs_trace("no lanes for %s",
                      ucp_proto_id_field(params->super.proto_id, name));
            return UCS_ERR_NO_ELEM;
        }

        exclude_map |= UCS_BIT(lanes[0]);
    }

    /* Find rest of the lanes */
    length += ucp_proto_common_find_lanes_with_min_frag(
                params, lane_desc->lane_type, lane_desc->tl_cap_flags,
                UCP_PROTO_MAX_LANES - length, exclude_map, lanes + length);

    /* Allocate and initialize storage */
    *storage = ucs_calloc(1, sizeof(ucp_proto_lane_storage_t),
                          "ucp_proto_lane_storage_t");
    if (*storage == NULL) {
        ucs_error("failed to allocate proto lane storage");
        return UCS_ERR_NO_MEMORY;
    }

    (*storage)->worker           = params->super.worker;
    (*storage)->length           = length;
    (*storage)->fixed_first_lane = has_first_lane;

    /* Evaluate individual lane performance */
    for (i = 0; i < length; ++i) {
        lane_iter       = &(*storage)->lanes_perf[i];
        lane_iter->lane = lanes[i];

        status = ucp_proto_common_get_lane_perf(params, lane_iter->lane,
                                                &lane_iter->perf,
                                                &lane_iter->perf_node);
        if (status != UCS_OK) {
            goto err;
        }
    }

    return UCS_OK;

err:
    ucp_proto_storage_destroy(*storage);
    *storage = NULL;
    return status;
}

static void
ucp_proto_storage_remove_slow_lanes(ucp_proto_lane_storage_t *storage)
{
    ucp_context_h context     = storage->worker->context;
    const double max_bw_ratio = context->config.ext.multi_lane_max_ratio;
    double max_bandwidth      = 0;
    ucp_proto_lane_perf_t *lane_iter;
    ucp_lane_index_t i, num_fast_lanes;

    for (i = 0; i < storage->length; ++i) {
        /* Calculate maximal bandwidth of all lanes, to skip slow lanes */
        lane_iter = &storage->lanes_perf[i];
        max_bandwidth = ucs_max(max_bandwidth, lane_iter->perf.bandwidth);
    }

    i = storage->fixed_first_lane ? 1 : 0;
    for (num_fast_lanes = i; i < storage->length; ++i) {
        lane_iter = &storage->lanes_perf[i];
        if ((lane_iter->perf.bandwidth * max_bw_ratio) < max_bandwidth) {
            /* Bandwidth on this lane is too low compared to the fastest
             * available lane, so it's not worth using it */
            ucp_proto_perf_node_deref(&lane_iter->perf_node);
            continue;
        }

        if (num_fast_lanes != i) {
            storage->lanes_perf[num_fast_lanes] = *lane_iter;
        }

        ++num_fast_lanes;
    }

    storage->length = num_fast_lanes;
}

static UCS_F_ALWAYS_INLINE const ucp_proto_lane_perf_t *
ucp_proto_select_get_lane_perf(ucp_proto_lane_selection_t *selection,
                               ucp_lane_index_t lane)
{
    return &selection->storage->lanes_perf[selection->lanes[lane]];
}

static void ucp_proto_select_destroy(ucp_proto_lane_selection_t *selection)
{
    ucp_proto_perf_node_deref(&selection->root_node);
    ucp_proto_storage_destroy(selection->storage);
    selection->storage = NULL;
}

static void
ucp_proto_select_bw_lanes(ucp_proto_lane_storage_t *storage,
                          ucp_proto_lane_selection_t *selection,
                          ucp_lane_index_t max_lanes)
{
    ucp_lane_index_t i;

    selection->storage   = storage;
    selection->length    = ucs_min(storage->length, max_lanes);
    selection->root_node = NULL;

    for (i = 0; i < selection->length; ++i) {
        selection->lanes[i] = i;
    }
}

static ucs_status_t
ucp_proto_select_lanes(const ucp_proto_common_init_params_t *params,
                       const ucp_proto_lane_desc_t *lane_desc,
                       const ucp_proto_lane_desc_t *first_lane_desc,
                       ucp_lane_index_t max_lanes, ucp_lane_map_t exclude_map,
                       ucp_proto_lane_selection_t *selection)
{
    ucp_proto_lane_storage_t *storage;
    const ucp_proto_lane_perf_t *lane_iter;
    ucp_lane_index_t i;
    ucs_status_t status;

    status = ucp_proto_storage_create(params, lane_desc, first_lane_desc,
                                      max_lanes, exclude_map, &storage);
    if (status != UCS_OK) {
        return status;
    }

    ucp_proto_storage_remove_slow_lanes(storage);
    ucp_proto_select_bw_lanes(storage, selection, max_lanes);

    if (selection->length == 1) {
        lane_iter = ucp_proto_select_get_lane_perf(selection, 0);
        ucp_proto_perf_node_ref(lane_iter->perf_node);
        selection->root_node = lane_iter->perf_node;
    } else {
        selection->root_node = ucp_proto_perf_node_new_data("multi", "%u lanes",
                                                            selection->length);
        for (i = 0; i < selection->length; ++i) {
            lane_iter = ucp_proto_select_get_lane_perf(selection, i);
            ucp_proto_perf_node_add_child(selection->root_node,
                                          lane_iter->perf_node);
        }
    }

    return UCS_OK;
}

ucs_status_t ucp_proto_multi_init(const ucp_proto_multi_init_params_t *params,
                                  const char *perf_name,
                                  ucp_proto_perf_t **perf_p,
                                  ucp_proto_multi_priv_t *mpriv)
{
    ucp_proto_lane_selection_t selection;
    const ucp_proto_lane_perf_t *lane_iter;
    const ucp_proto_common_tl_perf_t *lane_perf;
    ucp_proto_common_tl_perf_t perf;
    double max_frag_ratio, min_bandwidth;
    ucp_lane_map_t lane_map;
    ucp_proto_multi_lane_priv_t *lpriv;
    size_t max_frag, min_length, min_end_offset, min_chunk;
    ucp_md_map_t reg_md_map;
    uint32_t weight_sum;
    ucp_lane_index_t i, lane;
    ucs_status_t status;

    ucs_assert(params->max_lanes <= UCP_PROTO_MAX_LANES);

    if ((ucp_proto_select_op_flags(params->super.super.select_param) &
         UCP_PROTO_SELECT_OP_FLAG_RESUME) &&
        !(params->super.flags & UCP_PROTO_COMMON_INIT_FLAG_RESUME)) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (!ucp_proto_common_init_check_err_handling(&params->super) ||
        (params->max_lanes == 0)) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = ucp_proto_select_lanes(&params->super, &params->middle,
                                    &params->first, params->max_lanes, 0,
                                    &selection);
    if (status != UCS_OK) {
        return status;
    }

    /* Calculate lanes aggregate performance */
    perf.bandwidth          = 0;
    perf.send_pre_overhead  = 0;
    perf.send_post_overhead = 0;
    perf.recv_overhead      = 0;
    perf.latency            = 0;
    perf.sys_latency        = 0;
    lane_map                = 0;
    max_frag_ratio          = 0;
    min_bandwidth           = DBL_MAX;

    for (i = 0; i < selection.length; ++i) {
        lane_iter = ucp_proto_select_get_lane_perf(&selection, i);
        lane_perf = &lane_iter->perf;

        ucs_trace("lane[%d]" UCP_PROTO_TIME_FMT(send_pre_overhead)
                  UCP_PROTO_TIME_FMT(send_post_overhead)
                  UCP_PROTO_TIME_FMT(recv_overhead)
                  " bw " UCP_PROTO_PERF_FUNC_BW_FMT
                  UCP_PROTO_TIME_FMT(latency), lane_iter->lane,
                  UCP_PROTO_TIME_ARG(lane_perf->send_pre_overhead),
                  UCP_PROTO_TIME_ARG(lane_perf->send_post_overhead),
                  UCP_PROTO_TIME_ARG(lane_perf->recv_overhead),
                  (lane_perf->bandwidth / UCS_MBYTE),
                  UCP_PROTO_TIME_ARG(lane_perf->latency));

        /* Calculate maximal bandwidth-to-fragment-size ratio, which is used to
           adjust fragment sizes so they are proportional to bandwidth ratio and
           also do not exceed maximal supported size */
        max_frag_ratio = ucs_max(max_frag_ratio,
                                 lane_perf->bandwidth / lane_perf->max_frag);

        min_bandwidth = ucs_min(min_bandwidth, lane_perf->bandwidth);

        /* Update aggregated performance metric */
        perf.bandwidth          += lane_perf->bandwidth;
        perf.send_pre_overhead  += lane_perf->send_pre_overhead;
        perf.send_post_overhead += lane_perf->send_post_overhead;
        perf.recv_overhead      += lane_perf->recv_overhead;
        perf.latency             = ucs_max(perf.latency, lane_perf->latency);
        perf.sys_latency         = ucs_max(perf.sys_latency,
                                           lane_perf->sys_latency);
        lane_map                |= UCS_BIT(lane_iter->lane);
    }

    /* Initialize multi-lane private data and relative weights */
    reg_md_map          = ucp_proto_common_reg_md_map(&params->super, lane_map);
    mpriv->reg_md_map   = reg_md_map | params->initial_reg_md_map;
    mpriv->lane_map     = lane_map;
    mpriv->num_lanes    = 0;
    mpriv->min_frag     = 0;
    mpriv->max_frag_sum = 0;
    mpriv->align_thresh = 1;
    perf.max_frag       = 0;
    perf.min_length     = 0;
    weight_sum          = 0;
    min_end_offset      = 0;

    for (i = 0; i < selection.length; ++i) {
        lane_iter = ucp_proto_select_get_lane_perf(&selection, i);
        lane      = lane_iter->lane;
        ucs_assert(lane < UCP_MAX_LANES);

        lpriv     = &mpriv->lanes[mpriv->num_lanes++];
        lane_perf = &lane_iter->perf;

        ucp_proto_common_lane_priv_init(&params->super, mpriv->reg_md_map, lane,
                                        &lpriv->super);

        /* Calculate maximal fragment size according to lane relative bandwidth.
           Due to floating-point accuracy, max_frag may be too large. */
        max_frag = ucs_double_to_sizet(lane_perf->bandwidth / max_frag_ratio,
                                       lane_perf->max_frag);

        /* Make sure fragment is not zero */
        ucs_assert(max_frag > 0);

        /* Min chunk is scaled, but must be within HW limits */
        min_chunk       = ucs_min(lane_perf->bandwidth * params->min_chunk /
                                  min_bandwidth, lane_perf->max_frag);
        max_frag        = ucs_max(max_frag, min_chunk);
        lpriv->max_frag = max_frag;
        perf.max_frag  += max_frag;

        /* Calculate lane weight as a fixed-point fraction */
        lpriv->weight = ucs_proto_multi_calc_weight(lane_perf->bandwidth,
                                                    perf.bandwidth);
        ucs_assert(lpriv->weight > 0);
        ucs_assert(lpriv->weight <= UCP_PROTO_MULTI_WEIGHT_MAX);

        /* Calculate minimal message length according to lane's relative weight:
           When the message length is scaled by this lane's weight, it must not
           be lower than 'lane_perf->min_length'. Proof of the below formula:

            Since we calculated min_length by our formula:
                length >= (lane_perf->min_length << SHIFT) / weight

            Let's mark "(lane_perf->min_length << SHIFT)" as "M" for simplicity:
                length >= M / weight

            Since weight <= mask + 1, then weight - (mask + 1) <= 0, so we get:
                length >= (M + weight - (mask + 1)) / weight)

            Changing order:
                length >= ((M - mask) + (weight - 1)) / weight)

            The righthand expression is actually rounding up the result of
            "M - mask" divided by weight. So if we multiplied the righthand
            expression by weight, it would be >= "M - mask".
            We can multiply both sides by weight and get:
                length * weight >= "righthand expression" * weight >= M - mask
            So:
                length * weight >= M - mask

            Changing order:
                length * weight + mask >= M = lane_perf->min_length << SHIFT

            Shifting right, we get the needed inequality, which means the result
            of ucp_proto_multi_scaled_length() will be always greater or equal
            to min_length:
                (length * weight + mask) >> SHIFT >= lane_perf->min_length
        */
        min_length = (lane_perf->min_length << UCP_PROTO_MULTI_WEIGHT_SHIFT) /
                     lpriv->weight;
        ucs_assert(ucp_proto_multi_scaled_length(lpriv->weight, min_length) >=
                   lane_perf->min_length);
        perf.min_length = ucs_max(perf.min_length, min_length);

        weight_sum           += lpriv->weight;
        min_end_offset       += min_chunk;
        mpriv->min_frag       = ucs_max(mpriv->min_frag, lane_perf->min_length);
        mpriv->max_frag_sum  += lpriv->max_frag;
        lpriv->weight_sum     = weight_sum;
        lpriv->min_end_offset = min_end_offset;
        lpriv->max_frag_sum   = mpriv->max_frag_sum;
        lpriv->opt_align      = ucp_proto_multi_get_lane_opt_align(params, lane);
        mpriv->align_thresh   = ucs_max(mpriv->align_thresh, lpriv->opt_align);
    }
    ucs_assert(mpriv->num_lanes == ucs_popcount(lane_map));

    status = ucp_proto_init_perf(&params->super, &perf, selection.root_node,
                                 reg_md_map, perf_name, perf_p);
    ucp_proto_select_destroy(&selection);
    return status;
}

size_t ucp_proto_multi_priv_size(const ucp_proto_multi_priv_t *mpriv)
{
    return ucs_offsetof(ucp_proto_multi_priv_t, lanes) +
           (mpriv->num_lanes *
            ucs_field_sizeof(ucp_proto_multi_priv_t, lanes[0]));
}

void ucp_proto_multi_probe(const ucp_proto_multi_init_params_t *params)
{
    const char *proto_name = ucp_proto_id_field(params->super.super.proto_id,
                                                name);
    ucp_proto_multi_priv_t mpriv;
    ucp_proto_perf_t *perf;
    ucs_status_t status;

    status = ucp_proto_multi_init(params, proto_name, &perf, &mpriv);
    if (status != UCS_OK) {
        return;
    }

    ucp_proto_select_add_proto(&params->super.super, params->super.cfg_thresh,
                               params->super.cfg_priority, perf, &mpriv,
                               ucp_proto_multi_priv_size(&mpriv));
}

static const ucp_ep_config_key_lane_t *
ucp_proto_multi_ep_lane_cfg(const ucp_proto_query_params_t *params,
                            ucp_lane_index_t lane_index)
{
    const ucp_proto_multi_priv_t *mpriv = params->priv;
    const ucp_proto_multi_lane_priv_t *lpriv;

    ucs_assertv(lane_index < mpriv->num_lanes, "proto=%s lane_index=%d",
                params->proto->name, lane_index);
    lpriv = &mpriv->lanes[lane_index];

    ucs_assertv(lpriv->super.lane < UCP_MAX_LANES, "proto=%s lane=%d",
                params->proto->name, lpriv->super.lane);
    return &params->ep_config_key->lanes[lpriv->super.lane];
}

void ucp_proto_multi_query_config(const ucp_proto_query_params_t *params,
                                  ucp_proto_query_attr_t *attr)
{
    UCS_STRING_BUFFER_FIXED(strb, attr->config, sizeof(attr->config));
    const ucp_proto_multi_priv_t *mpriv = params->priv;
    const ucp_ep_config_key_lane_t *cfg_lane, *cfg_lane0;
    const ucp_proto_multi_lane_priv_t *lpriv;
    size_t percent, remaining;
    int same_rsc, same_path;
    ucp_lane_index_t i;

    ucs_assert(mpriv->num_lanes <= UCP_MAX_LANES);
    ucs_assert(mpriv->num_lanes >= 1);

    same_rsc  = 1;
    same_path = 1;
    cfg_lane0 = ucp_proto_multi_ep_lane_cfg(params, 0);
    for (i = 1; i < mpriv->num_lanes; ++i) {
        cfg_lane  = ucp_proto_multi_ep_lane_cfg(params, i);
        same_rsc  = same_rsc && (cfg_lane->rsc_index == cfg_lane0->rsc_index);
        same_path = same_path &&
                    (cfg_lane->path_index == cfg_lane0->path_index);
    }

    if (same_rsc) {
        ucp_proto_common_lane_priv_str(params, &mpriv->lanes[0].super, 1,
                                       same_path, &strb);
        ucs_string_buffer_appendf(&strb, " ");
    }

    remaining = 100;
    for (i = 0; i < mpriv->num_lanes; ++i) {
        lpriv      = &mpriv->lanes[i];
        percent    = ucs_min(remaining,
                             ucp_proto_multi_scaled_length(lpriv->weight, 100));
        remaining -= percent;

        if (percent != 100) {
            ucs_string_buffer_appendf(&strb, "%zu%% on ", percent);
        }

        ucp_proto_common_lane_priv_str(params, &lpriv->super, !same_rsc,
                                       !(same_rsc && same_path), &strb);

        /* Print a string like "30% on A, 40% on B, and 30% on C" */
        if (i != (mpriv->num_lanes - 1)) {
            if (i == (mpriv->num_lanes - 2)) {
                ucs_string_buffer_appendf(&strb, " and ");
            } else {
                ucs_string_buffer_appendf(&strb, ", ");
            }
        }
    }

    ucs_string_buffer_rtrim(&strb, NULL);
    attr->lane_map = mpriv->lane_map;
}

void ucp_proto_multi_query(const ucp_proto_query_params_t *params,
                           ucp_proto_query_attr_t *attr)
{
    ucp_proto_default_query(params, attr);
    ucp_proto_multi_query_config(params, attr);
}
