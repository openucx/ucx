/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_single.h"
#include "proto_common.h"
#include "proto_init.h"

#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/sys/math.h>


ucs_status_t
ucp_proto_single_init_priv(const ucp_proto_single_init_params_t *params,
                           ucp_proto_single_priv_t *spriv)
{
    ucp_proto_common_tl_perf_t perf;
    ucp_lane_index_t num_lanes;
    ucp_md_map_t reg_md_map;
    ucp_lane_index_t lane;
    ucs_status_t status;

    num_lanes = ucp_proto_common_find_lanes(&params->super, params->lane_type,
                                            params->tl_cap_flags, 1, 0, &lane);
    if (num_lanes == 0) {
        ucs_trace("no lanes for %s", params->super.super.proto_name);
        return UCS_ERR_NO_ELEM;
    }

    ucs_assert(num_lanes == 1);

    reg_md_map = ucp_proto_common_reg_md_map(&params->super, UCS_BIT(lane));
    if (reg_md_map == 0) {
        spriv->reg_md = UCP_NULL_RESOURCE;
    } else {
        ucs_assert(ucs_popcount(reg_md_map) == 1);
        spriv->reg_md = ucs_ffs64(reg_md_map);
    }

    ucp_proto_common_lane_priv_init(&params->super, reg_md_map, lane,
                                    &spriv->super);
    status = ucp_proto_common_get_lane_perf(&params->super, lane, &perf);
    if (status != UCS_OK) {
        return status;
    }

    return ucp_proto_common_init_caps(&params->super, &perf, reg_md_map);
}

ucs_status_t ucp_proto_single_init(const ucp_proto_single_init_params_t *params)
{
    ucs_status_t status;

    status = ucp_proto_single_init_priv(params, params->super.super.priv);
    if (status != UCS_OK) {
        return status;
    }

    *params->super.super.priv_size = sizeof(ucp_proto_single_priv_t);
    return UCS_OK;
}

void ucp_proto_single_query(const ucp_proto_query_params_t *params,
                            ucp_proto_query_attr_t *attr)
{
    UCS_STRING_BUFFER_FIXED(config_strb, attr->config, sizeof(attr->config));
    const ucp_proto_single_priv_t *spriv = params->priv;

    ucp_proto_default_query(params, attr);
    ucp_proto_common_lane_priv_str(params, &spriv->super, 1, 1, &config_strb);
}
