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


static UCS_F_ALWAYS_INLINE double
ucp_proto_multi_get_avail_bw(const ucp_proto_init_params_t *params,
                             ucp_lane_index_t lane,
                             const ucp_proto_common_tl_perf_t *lane_perf,
                             const ucp_proto_lane_selection_t *selection)
{
    /* Minimal path ratio */
    static const double MIN_RATIO = 0.01;
    ucp_context_h context         = params->worker->context;
    double multi_path_ratio       = context->config.ext.multi_path_ratio;
    ucp_rsc_index_t dev_index     = ucp_proto_common_get_dev_index(params, lane);
    uint8_t path_index            = selection->dev_count[dev_index];
    double ratio;

    if (UCS_CONFIG_DBL_IS_AUTO(multi_path_ratio)) {
        ratio = ucs_min(1.0 - (lane_perf->path_ratio * path_index),
                        lane_perf->path_ratio);
    } else {
        ratio = 1.0 - (multi_path_ratio * path_index);
    }

    if (ratio < MIN_RATIO) {
        /* The iface BW is entirely consumed by the selected paths. But we still
         * need to assign some minimal BW to the these extra paths in order to
         * select them. We divide min ratio of the iface BW by path_index so
         * that each additional path on the same device has lower bandwidth. */
        ratio = MIN_RATIO / path_index;
    }

    ucs_trace("ratio=%0.3f path_index=%u avail_bw=" UCP_PROTO_PERF_FUNC_BW_FMT
              " " UCP_PROTO_LANE_FMT, ratio, path_index,
              (lane_perf->bandwidth * ratio) / UCS_MBYTE,
              UCP_PROTO_LANE_ARG(params, lane, lane_perf));
    return lane_perf->bandwidth * ratio;
}

static ucp_lane_index_t
ucp_proto_multi_find_max_avail_bw_lane(const ucp_proto_init_params_t *params,
                                       const ucp_lane_index_t *lanes,
                                       const ucp_proto_common_tl_perf_t *lanes_perf,
                                       const ucp_proto_lane_selection_t *selection,
                                       ucp_lane_map_t index_map)
{
    /* Initial value is 1Bps, so we don't consider lanes with lower available
     * bandwidth. */
    double max_avail_bw        = 1.0;
    ucp_lane_index_t max_index = UCP_NULL_LANE;
    double avail_bw;
    const ucp_proto_common_tl_perf_t *lane_perf;
    ucp_lane_index_t lane, index;

    ucs_assert(index_map != 0);
    ucs_for_each_bit(index, index_map) {
        lane      = lanes[index];
        lane_perf = &lanes_perf[lane];
        avail_bw  = ucp_proto_multi_get_avail_bw(params, lane, lane_perf,
                                                 selection);
        if (avail_bw > max_avail_bw) {
            max_avail_bw = avail_bw;
            max_index    = index;
        }
    }

    return max_index;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_select_add_lane(ucp_proto_lane_selection_t *selection,
                          const ucp_proto_init_params_t *params,
                          ucp_lane_index_t lane)
{
    ucp_rsc_index_t dev_index = ucp_proto_common_get_dev_index(params, lane);

    ucs_assertv(selection->num_lanes < UCP_PROTO_MAX_LANES,
                "selection num_lanes=%u max_lanes=%u", selection->num_lanes,
                UCP_PROTO_MAX_LANES);
    selection->lanes[selection->num_lanes++] = lane;
    selection->lane_map                     |= UCS_BIT(lane);
    selection->dev_count[dev_index]++;
}

static void
ucp_proto_multi_select_bw_lanes(const ucp_proto_init_params_t *params,
                                const ucp_lane_index_t *lanes,
                                ucp_lane_index_t num_lanes,
                                ucp_lane_index_t max_lanes,
                                const ucp_proto_common_tl_perf_t *lanes_perf,
                                int fixed_first_lane,
                                ucp_proto_lane_selection_t *selection)
{
    ucp_lane_index_t i, lane_index;
    ucp_lane_map_t index_map;

    memset(selection, 0, sizeof(*selection));

    /* Select all available indexes */
    index_map = UCS_MASK(num_lanes);

    if (fixed_first_lane) {
        ucp_proto_select_add_lane(selection, params, lanes[0]);
        index_map &= ~UCS_BIT(0);
    }

    for (i = fixed_first_lane? 1 : 0; i < ucs_min(max_lanes, num_lanes); ++i) {
        /* Greedy algorithm: find the best option at every step */
        lane_index = ucp_proto_multi_find_max_avail_bw_lane(params, lanes,
                                                            lanes_perf, selection,
                                                            index_map);
        if (lane_index == UCP_NULL_LANE) {
            break;
        }

        ucp_proto_select_add_lane(selection, params, lanes[lane_index]);
        index_map &= ~UCS_BIT(lane_index);
    }

    /* TODO: Aggregate performance:
     * Split full iface bandwidth between selected paths, according to the total
     * path ratio */
}

static ucp_sys_dev_map_t
ucp_proto_multi_init_flush_sys_dev_mask(const ucp_rkey_config_key_t *key)
{
    if (key == NULL || !ucp_rkey_need_remote_flush(key)) {
        return 0;
    }

    return UCS_BIT(key->sys_dev & ~UCP_SYS_DEVICE_FLUSH_BIT);
}

static ucp_lane_index_t ucp_proto_multi_filter_net_devices(
        ucp_lane_index_t num_lanes, const ucp_proto_init_params_t *params,
        const ucp_proto_common_tl_perf_t *tl_perfs, int fixed_first_lane,
        ucp_lane_index_t *lanes, ucp_proto_perf_node_t **perf_nodes)
{
    ucp_lane_index_t num_max_bw_devs = 0;
    double max_bandwidth;
    ucp_lane_index_t i, lane, seed, num_filtered_lanes;
    ucp_lane_map_t lane_map;
    ucs_sys_device_t sys_dev;
    ucs_sys_device_t sys_devs[UCP_PROTO_MAX_LANES];
    const uct_tl_resource_desc_t *tl_rsc;

    for (lane_map = 0, max_bandwidth = 0, i = 0; i < num_lanes; ++i) {
        lane = lanes[i];
        if (!ucp_proto_common_is_net_dev(params, lane)) {
            continue;
        }

        lane_map     |= UCS_BIT(lane);
        max_bandwidth = ucs_max(max_bandwidth, tl_perfs[lane].bandwidth);
    }

    ucs_for_each_bit(lane, lane_map) {
        if (!ucp_proto_common_bandwidth_equal(tl_perfs[lane].bandwidth,
                                              max_bandwidth)) {
            continue;
        }

        sys_dev = ucp_proto_common_get_sys_dev(params, lane);
        ucp_proto_common_add_unique_sys_dev(sys_dev, sys_devs, &num_max_bw_devs,
                                            UCP_PROTO_MAX_LANES);
    }

    if (num_max_bw_devs == 0) {
        return num_lanes;
    }

    seed = ucp_proto_common_select_sys_dev_by_node_id(params, num_max_bw_devs);

    for (i = !!fixed_first_lane, num_filtered_lanes = i; i < num_lanes; ++i) {
        lane   = lanes[i];
        tl_rsc = ucp_proto_common_get_tl_rsc(params, lane);
        if ((tl_rsc->dev_type == UCT_DEVICE_TYPE_NET) &&
            (tl_rsc->sys_device != sys_devs[seed])) {
            ucp_proto_perf_node_deref(&perf_nodes[lane]);
            ucs_trace("filtered out " UCP_PROTO_LANE_FMT,
                      UCP_PROTO_LANE_ARG(params, lane, &tl_perfs[lane]));
        } else {
            lanes[num_filtered_lanes++] = lane;
        }
    }

    return num_filtered_lanes;
}

ucs_status_t ucp_proto_multi_init(const ucp_proto_multi_init_params_t *params,
                                  const char *perf_name,
                                  ucp_proto_perf_t **perf_p,
                                  ucp_proto_multi_priv_t *mpriv)
{
    ucp_context_h context         = params->super.super.worker->context;
    const double max_bw_ratio     = context->config.ext.multi_lane_max_ratio;
    ucp_proto_perf_node_t *lanes_perf_nodes[UCP_PROTO_MAX_LANES];
    ucp_proto_common_tl_perf_t lanes_perf[UCP_PROTO_MAX_LANES];
    ucp_proto_common_tl_perf_t *lane_perf, perf;
    ucp_lane_index_t lanes[UCP_PROTO_MAX_LANES];
    double max_bandwidth, max_frag_ratio, min_bandwidth;
    ucp_lane_index_t i, lane, num_lanes, num_fast_lanes;
    ucp_proto_multi_lane_priv_t *lpriv;
    ucp_proto_perf_node_t *perf_node;
    size_t max_frag, min_length, min_end_offset, min_chunk;
    ucp_proto_lane_selection_t selection;
    ucp_md_map_t reg_md_map;
    uint32_t weight_sum;
    ucs_status_t status;
    int fixed_first_lane;

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

    if (!ucp_proto_common_check_memtype_copy(&params->super)) {
        return UCS_ERR_UNSUPPORTED;
    }

    /* Find first lane */
    num_lanes = ucp_proto_common_find_lanes(
            &params->super.super, params->super.flags, params->first.lane_type,
            params->first.tl_cap_flags, 1, 0, ucp_proto_common_filter_min_frag,
            lanes);
    if (num_lanes == 0) {
        ucs_trace("no lanes for %s",
                  ucp_proto_id_field(params->super.super.proto_id, name));
        return UCS_ERR_NO_ELEM;
    }

    /* Find rest of the lanes */
    num_lanes += ucp_proto_common_find_lanes(
            &params->super.super, params->super.flags, params->middle.lane_type,
            params->middle.tl_cap_flags, UCP_PROTO_MAX_LANES - 1,
            UCS_BIT(lanes[0]), ucp_proto_common_filter_min_frag, lanes + 1);

    /* Get bandwidth of all lanes and max_bandwidth */
    max_bandwidth = 0;
    for (i = 0; i < num_lanes; ++i) {
        lane      = lanes[i];
        lane_perf = &lanes_perf[lane];

        status = ucp_proto_common_get_lane_perf(&params->super, lane, lane_perf,
                                                &lanes_perf_nodes[lane]);
        if (status != UCS_OK) {
            return status;
        }

        /* Calculate maximal bandwidth of all lanes, to skip slow lanes */
        max_bandwidth = ucs_max(max_bandwidth, lane_perf->bandwidth);
    }

    /* Select the lanes to use, and calculate their aggregate performance */
    perf.bandwidth          = 0;
    perf.send_pre_overhead  = 0;
    perf.send_post_overhead = 0;
    perf.recv_overhead      = 0;
    perf.latency            = 0;
    perf.sys_latency        = 0;
    max_frag_ratio          = 0;
    min_bandwidth           = DBL_MAX;

    /* Filter out slow lanes */
    fixed_first_lane = params->first.lane_type != params->middle.lane_type;
    for (i = fixed_first_lane ? 1 : 0, num_fast_lanes = i; i < num_lanes; ++i) {
        lane      = lanes[i];
        lane_perf = &lanes_perf[lane];
        if ((lane_perf->bandwidth * max_bw_ratio) < max_bandwidth) {
            /* Bandwidth on this lane is too low compared to the fastest
               available lane, so it's not worth using it */
            ucp_proto_perf_node_deref(&lanes_perf_nodes[lane]);
            ucs_trace("drop " UCP_PROTO_LANE_FMT,
                      UCP_PROTO_LANE_ARG(&params->super.super, lane, lane_perf));
        } else {
            lanes[num_fast_lanes++] = lane;
            ucs_trace("avail " UCP_PROTO_LANE_FMT,
                      UCP_PROTO_LANE_ARG(&params->super.super, lane, lane_perf));
        }
    }

    num_lanes = num_fast_lanes;
    if (context->config.ext.proto_use_single_net_device) {
        num_lanes = ucp_proto_multi_filter_net_devices(num_lanes,
                                                       &params->super.super,
                                                       lanes_perf,
                                                       fixed_first_lane, lanes,
                                                       lanes_perf_nodes);
    }

    ucp_proto_multi_select_bw_lanes(&params->super.super, lanes, num_lanes,
                                    params->max_lanes, lanes_perf,
                                    fixed_first_lane, &selection);

    ucs_trace("selected %u lanes for %s", selection.num_lanes,
              ucp_proto_id_field(params->super.super.proto_id, name));
    ucs_log_indent(1);

    for (i = 0; i < selection.num_lanes; ++i) {
        lane      = selection.lanes[i];
        lane_perf = &lanes_perf[lane];
        ucs_trace(UCP_PROTO_LANE_FMT UCP_PROTO_TIME_FMT(send_pre_overhead)
                  UCP_PROTO_TIME_FMT(send_post_overhead)
                  UCP_PROTO_TIME_FMT(recv_overhead),
                  UCP_PROTO_LANE_ARG(&params->super.super, lane, lane_perf),
                  UCP_PROTO_TIME_ARG(lane_perf->send_pre_overhead),
                  UCP_PROTO_TIME_ARG(lane_perf->send_post_overhead),
                  UCP_PROTO_TIME_ARG(lane_perf->recv_overhead));

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
    }

    ucs_log_indent(-1);

    /* Initialize multi-lane private data and relative weights */
    reg_md_map          = ucp_proto_common_reg_md_map(&params->super,
                                                      selection.lane_map);
    mpriv->reg_md_map   = reg_md_map | params->initial_reg_md_map;
    mpriv->lane_map     = selection.lane_map;
    mpriv->num_lanes    = 0;
    mpriv->min_frag     = 0;
    mpriv->max_frag_sum = 0;
    mpriv->align_thresh = 1;
    perf.max_frag       = 0;
    perf.min_length     = 0;
    weight_sum          = 0;
    min_end_offset      = 0;

    ucs_for_each_bit(lane, selection.lane_map) {
        ucs_assert(lane < UCP_MAX_LANES);

        lpriv     = &mpriv->lanes[mpriv->num_lanes++];
        lane_perf = &lanes_perf[lane];

        ucp_proto_common_lane_priv_init(&params->super, mpriv->reg_md_map, lane,
                                        &lpriv->super);

        /* Calculate maximal fragment size according to lane relative bandwidth.
           Due to floating-point accuracy, max_frag may be too large. */
        max_frag = ucs_double_to_sizet(lane_perf->bandwidth / max_frag_ratio,
                                       lane_perf->max_frag);

        /* Make sure fragment is not zero */
        ucs_assert(max_frag > 0);

        /* Min chunk is scaled, but must be within HW limits.
           Min chunk cannot be less than UCP_MIN_BCOPY, as it's not worth to
           split tiny messages. */
        min_chunk       = ucs_min(lane_perf->max_frag,
                                  ucs_max(UCP_MIN_BCOPY,
                                          lane_perf->bandwidth *
                                          params->min_chunk / min_bandwidth));
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
        lpriv->flush_sys_dev_mask = ucp_proto_multi_init_flush_sys_dev_mask(
                params->super.super.rkey_config_key);
    }
    ucs_assert(mpriv->num_lanes == ucs_popcount(selection.lane_map));

    /* After this block, 'perf_node' and 'lane_perf_nodes[]' have extra ref */
    if (mpriv->num_lanes == 1) {
        perf_node = lanes_perf_nodes[ucs_ffs64(selection.lane_map)];
        ucp_proto_perf_node_ref(perf_node);
    } else {
        perf_node = ucp_proto_perf_node_new_data("multi", "%u lanes",
                                                 mpriv->num_lanes);
        ucs_for_each_bit(lane, selection.lane_map) {
            ucs_assert(lane < UCP_MAX_LANES);
            ucp_proto_perf_node_add_child(perf_node, lanes_perf_nodes[lane]);
        }
    }

    status = ucp_proto_init_perf(&params->super, &perf, perf_node, reg_md_map,
                                 perf_name, perf_p);

    /* Deref unused nodes */
    for (i = 0; i < num_lanes; ++i) {
        ucp_proto_perf_node_deref(&lanes_perf_nodes[lanes[i]]);
    }
    ucp_proto_perf_node_deref(&perf_node);

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
