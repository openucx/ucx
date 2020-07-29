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

#include <ucs/debug/log.h>


ucs_status_t ucp_proto_single_init(const ucp_proto_single_init_params_t *params)
{
    ucp_proto_single_priv_t *spriv = params->super.super.priv;
    ucp_lane_index_t num_lanes;

    num_lanes = ucp_proto_common_find_lanes(&params->super, params->lane_type,
                                            params->tl_cap_flags, &spriv->lane,
                                            1);
    if (num_lanes == 0) {
        ucs_trace("no lanes for %s", params->super.super.proto_name);
        return UCS_ERR_UNSUPPORTED;
    }

    *params->super.super.priv_size = sizeof(ucp_proto_single_priv_t);
    spriv->md_index                = ucp_proto_common_get_md_index(&params->super,
                                                                   spriv->lane);
    ucp_proto_common_calc_perf(&params->super, spriv->lane);
    return UCS_OK;
}

void ucp_proto_single_config_str(const void *priv, ucs_string_buffer_t *strb)
{
    const ucp_proto_single_priv_t *spriv = priv;

    ucs_string_buffer_init(strb);
    ucs_string_buffer_appendf(strb, "lane[%d]", spriv->lane);
}
