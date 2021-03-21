/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_common.inl"
#include "proto_multi.inl"

#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>


ucs_status_t ucp_proto_multi_init(const ucp_proto_multi_init_params_t *params)
{
    ucp_proto_multi_priv_t *mpriv = params->super.super.priv;
    ucp_lane_index_t lanes[UCP_PROTO_MAX_LANES];
    double lanes_bandwidth[UCP_PROTO_MAX_LANES];
    size_t lanes_max_frag[UCP_PROTO_MAX_LANES];
    ucp_proto_common_perf_params_t perf_params;
    ucp_lane_index_t i, lane, num_lanes;
    const uct_iface_attr_t *iface_attr;
    ucp_proto_multi_lane_priv_t *lpriv;
    ucp_lane_map_t lane_map;
    double total_bandwidth;

    ucs_assert(params->max_lanes >= 1);
    ucs_assert(params->max_lanes <= UCP_PROTO_MAX_LANES);

    /* Find first lane */
    num_lanes = ucp_proto_common_find_lanes(&params->super,
                                            params->first.lane_type,
                                            params->first.tl_cap_flags, 1, 0,
                                            lanes);
    if (num_lanes == 0) {
        ucs_trace("no lanes for %s", params->super.super.proto_name);
        return UCS_ERR_UNSUPPORTED;
    }

    /* Find rest of the lanes */
    num_lanes += ucp_proto_common_find_lanes(&params->super,
                                             params->middle.lane_type,
                                             params->middle.tl_cap_flags,
                                             params->max_lanes - 1,
                                             UCS_BIT(lanes[0]), lanes + 1);

    /* Get bandwidth of all lanes */
    total_bandwidth = 0;
    lane_map        = 0;
    for (i = 0; i < num_lanes; ++i) {
        lane       = lanes[i];
        iface_attr = ucp_proto_common_get_iface_attr(&params->super.super,
                                                     lane);

        lanes_bandwidth[lane] = ucp_proto_common_iface_bandwidth(&params->super,
                                                                 iface_attr);
        lanes_max_frag[lane]  = ucp_proto_common_get_max_frag(&params->super,
                                                              iface_attr);

        /* Calculate maximal bandwidth of all lanes, to skip slow lanes */
        total_bandwidth += lanes_bandwidth[lane];
        lane_map        |= UCS_BIT(lane);
    }

    /* Initialize multi-lane private data and relative weights */
    mpriv->reg_md_map = ucp_proto_common_reg_md_map(&params->super, lane_map);
    mpriv->num_lanes  = 0;
    ucs_for_each_bit(lane, lane_map) {
        ucs_assert(lane < UCP_MAX_LANES);
        lpriv = &mpriv->lanes[mpriv->num_lanes++];
        ucp_proto_common_lane_priv_init(&params->super, mpriv->reg_md_map, lane,
                                        &lpriv->super);
        lpriv->max_frag        = lanes_max_frag[lane];
        mpriv->lanes[i].weight = lanes_bandwidth[i] / total_bandwidth;
    }

    /* Fill the size of private data according to number of used lanes */
    *params->super.super.priv_size = sizeof(ucp_proto_multi_priv_t) +
                                     (mpriv->num_lanes * sizeof(*lpriv));

    /* Calculate protocol performance */
    perf_params.reg_md_map = mpriv->reg_md_map;
    perf_params.frag_size  = mpriv->lanes[0].max_frag;
    perf_params.lane_map   = lane_map;
    perf_params.bandwidth  = total_bandwidth;
    ucp_proto_common_calc_perf(&params->super, &perf_params);

    return UCS_OK;
}

void ucp_proto_multi_config_str(size_t min_length, size_t max_length,
                                const void *priv, ucs_string_buffer_t *strb)
{
    const ucp_proto_multi_priv_t *mpriv = priv;
    const ucp_proto_multi_lane_priv_t *lpriv;
    char frag_size_buf[64];
    ucp_lane_index_t i;

    for (i = 0; i < mpriv->num_lanes; ++i) {
        lpriv = &mpriv->lanes[i];
        ucs_string_buffer_appendf(strb, "%.0f%% ", 100.0 * lpriv->weight);
        ucp_proto_common_lane_priv_str(&lpriv->super, strb);

        /* Print fragment size if it's small enough. For large fragments we can
           skip the print because it has little effect on performance */
        if (lpriv->max_frag < UCS_MBYTE) {
            ucs_memunits_to_str(lpriv->max_frag, frag_size_buf,
                                sizeof(frag_size_buf));
            ucs_string_buffer_appendf(strb, "<=%s", frag_size_buf);
        }

        if ((i + 1) < mpriv->num_lanes) {
            ucs_string_buffer_appendf(strb, "|");
        }
    }
}
