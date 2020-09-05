/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "eager.h"

#include <ucp/proto/proto_multi.h>


static ucs_status_t
ucp_proto_eager_multi_init_common(ucp_proto_multi_init_params_t *params)
{
    if (params->super.super.select_param->op_id != UCP_OP_ID_TAG_SEND) {
        return UCS_ERR_UNSUPPORTED;
    }

    params->super.overhead   = 10e-9; /* for multiple lanes management */
    params->super.latency    = 0;
    params->first.lane_type  = UCP_LANE_TYPE_AM;
    params->super.hdr_size   = sizeof(ucp_eager_first_hdr_t);
    params->middle.lane_type = UCP_LANE_TYPE_AM_BW;
    params->max_lanes        =
            params->super.super.worker->context->config.ext.max_eager_lanes;

    return ucp_proto_multi_init(params);
}

static ucs_status_t
ucp_proto_eager_bcopy_multi_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.flags         = 0,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_BCOPY,
        .super.fragsz_offset = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_BCOPY,
    };

    return ucp_proto_eager_multi_init_common(&params);
}

static ucp_proto_t ucp_eager_bcopy_multi_proto = {
    .name       = "egr/multi/bcopy",
    .flags      = 0,
    .init       = ucp_proto_eager_bcopy_multi_init,
    .config_str = ucp_proto_multi_config_str,
    .progress   = (uct_pending_callback_t)ucs_empty_function_do_assert
};
UCP_PROTO_REGISTER(&ucp_eager_bcopy_multi_proto);

static ucs_status_t
ucp_proto_eager_zcopy_multi_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = 30,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_ZCOPY,
        .super.fragsz_offset = ucs_offsetof(uct_iface_attr_t, cap.am.max_zcopy),
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_ZCOPY,
    };

    return ucp_proto_eager_multi_init_common(&params);
}

static ucp_proto_t ucp_eager_zcopy_multi_proto = {
    .name       = "egr/multi/zcopy",
    .flags      = 0,
    .init       = ucp_proto_eager_zcopy_multi_init,
    .config_str = ucp_proto_multi_config_str,
    .progress   = (uct_pending_callback_t)ucs_empty_function_do_assert
};
UCP_PROTO_REGISTER(&ucp_eager_zcopy_multi_proto);
