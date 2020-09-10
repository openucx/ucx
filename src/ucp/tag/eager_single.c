/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "tag_match.h"

#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/proto/proto_single.h>
#include <ucs/sys/string.h>


static ucs_status_t
ucp_proto_eager_short_init(const ucp_proto_init_params_t *init_params)
{
    const ucp_proto_select_param_t *select_param = init_params->select_param;
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = -150e-9, /* no extra memory access to fetch data */
        .super.overhead      = 0,
        .super.cfg_thresh    = UCS_MEMUNITS_AUTO,
        .super.cfg_priority  = 0,
        .super.fragsz_offset = ucs_offsetof(uct_iface_attr_t, cap.am.max_short),
        .super.hdr_size      = sizeof(ucp_tag_hdr_t),
        .super.flags         = 0,
        .lane_type           = UCP_LANE_TYPE_AM,
        .tl_cap_flags        = UCT_IFACE_FLAG_AM_SHORT
    };

    if ((select_param->dt_class != UCP_DATATYPE_CONTIG) ||
        (select_param->op_id != UCP_OP_ID_TAG_SEND) ||
        !UCP_MEM_IS_HOST(select_param->mem_type)) {
        return UCS_ERR_UNSUPPORTED;
    }

    ucp_proto_single_init(&params);
    return UCS_ERR_UNSUPPORTED; /* TODO enable when progress is implemented */
}

static ucp_proto_t ucp_eager_short_proto = {
    .name       = "egr/short",
    .flags      = UCP_PROTO_FLAG_AM_SHORT,
    .init       = ucp_proto_eager_short_init,
    .config_str = ucp_proto_single_config_str,
    .progress   = (uct_pending_callback_t)ucs_empty_function_do_assert
};
UCP_PROTO_REGISTER(&ucp_eager_short_proto);

static ucs_status_t
ucp_proto_eager_bcopy_single_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context                = init_params->worker->context;
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 5e-9,
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.flags         = 0,
        .super.fragsz_offset = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.hdr_size      = sizeof(ucp_tag_hdr_t),
        .lane_type           = UCP_LANE_TYPE_AM,
        .tl_cap_flags        = UCT_IFACE_FLAG_AM_BCOPY
    };

    if (init_params->select_param->op_id != UCP_OP_ID_TAG_SEND) {
        return UCS_ERR_UNSUPPORTED;
    }

    ucp_proto_single_init(&params);
    return UCS_ERR_UNSUPPORTED; /* TODO enable when progress is implemented */
}

static ucp_proto_t ucp_eager_bcopy_single_proto = {
    .name       = "egr/single/bcopy",
    .flags      = 0,
    .init       = ucp_proto_eager_bcopy_single_init,
    .config_str = ucp_proto_single_config_str,
    .progress   = (uct_pending_callback_t)ucs_empty_function_do_assert,
};
UCP_PROTO_REGISTER(&ucp_eager_bcopy_single_proto);

static ucs_status_t
ucp_proto_eager_zcopy_single_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = 30,
        .super.overhead      = 0,
        .super.fragsz_offset = ucs_offsetof(uct_iface_attr_t, cap.am.max_zcopy),
        .super.hdr_size      = sizeof(ucp_tag_hdr_t),
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY,
        .lane_type           = UCP_LANE_TYPE_AM,
        .tl_cap_flags        = UCT_IFACE_FLAG_AM_ZCOPY
    };

    if (init_params->select_param->op_id != UCP_OP_ID_TAG_SEND) {
        return UCS_ERR_UNSUPPORTED;
    }

    ucp_proto_single_init(&params);
    return UCS_ERR_UNSUPPORTED; /* TODO enable when progress is implemented */
}

static ucp_proto_t ucp_eager_zcopy_single_proto = {
    .name       = "egr/single/zcopy",
    .flags      = 0,
    .init       = ucp_proto_eager_zcopy_single_init,
    .config_str = ucp_proto_single_config_str,
    .progress   = (uct_pending_callback_t)ucs_empty_function_do_assert
};
UCP_PROTO_REGISTER(&ucp_eager_zcopy_single_proto);
