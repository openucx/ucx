/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "proto_rndv.h"

#include <ucp/core/ucp_request.inl>


static ucs_status_t
ucp_proto_rndv_get_zcopy_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = UCS_MEMUNITS_AUTO,
        .super.cfg_priority  = 0,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_RESPONSE,
        .super.overhead      = 0,
        .super.latency       = 0,
        .max_lanes           = context->config.ext.max_rndv_lanes,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_GET_ZCOPY,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.get.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.get.max_zcopy),
        .first.lane_type     = UCP_LANE_TYPE_RMA_BW,
        .super.hdr_size      = 0,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_GET_ZCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_RMA_BW
    };

    UCP_PROTO_RNDV_CHECK_PARAMS(init_params, UCP_OP_ID_RNDV_RECV,
                                UCP_RNDV_MODE_GET_ZCOPY);

    return ucp_proto_rndv_bulk_init(&params);
}

static ucp_proto_t ucp_rndv_get_zcopy_proto = {
    .name       = "rndv/get/zcopy",
    .flags      = 0,
    .init       = ucp_proto_rndv_get_zcopy_init,
    .config_str = ucp_proto_rndv_bulk_config_str,
    .progress   = (uct_pending_callback_t)ucs_empty_function_do_assert
};
UCP_PROTO_REGISTER(&ucp_rndv_get_zcopy_proto);
