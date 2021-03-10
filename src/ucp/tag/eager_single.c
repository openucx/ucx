/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "eager.h"

#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_worker.h>
#include <ucs/sys/string.h>

#include <ucp/core/ucp_request.inl>
#include <ucp/proto/proto_single.inl>
#include <ucp/proto/proto_common.inl>


static ucs_status_t ucp_eager_short_progress(uct_pending_req_t *self)
{
    ucp_request_t                   *req = ucs_container_of(self, ucp_request_t,
                                                            send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    ucs_status_t status;

    status = uct_ep_am_short(req->send.ep->uct_eps[spriv->super.lane],
                             UCP_AM_ID_EAGER_ONLY, req->send.msg_proto.tag,
                             req->send.state.dt_iter.type.contig.buffer,
                             req->send.state.dt_iter.length);
    if (ucs_unlikely(status == UCS_ERR_NO_RESOURCE)) {
        req->send.lane = spriv->super.lane; /* for pending add */
        return status;
    }

    ucp_datatype_iter_cleanup(&req->send.state.dt_iter,
                              UCS_BIT(UCP_DATATYPE_CONTIG));

    ucs_assert(status != UCS_INPROGRESS);
    ucp_request_complete_send(req, status);
    return UCS_OK;
}

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
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_short),
        .super.hdr_size      = sizeof(ucp_tag_hdr_t),
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_MAX_FRAG,
        .lane_type           = UCP_LANE_TYPE_AM,
        .tl_cap_flags        = UCT_IFACE_FLAG_AM_SHORT
    };

    /* AM based proto can not be used if tag offload lane configured */
    if (!ucp_proto_eager_check_op_id(init_params, 0) ||
        /* short protocol requires contig/host */
        (select_param->dt_class != UCP_DATATYPE_CONTIG) ||
        !UCP_MEM_IS_HOST(select_param->mem_type)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_single_init(&params);
}

static ucp_proto_t ucp_eager_short_proto = {
    .name       = "egr/short",
    .flags      = UCP_PROTO_FLAG_AM_SHORT,
    .init       = ucp_proto_eager_short_init,
    .config_str = ucp_proto_single_config_str,
    .progress   = ucp_eager_short_progress
};
UCP_PROTO_REGISTER(&ucp_eager_short_proto);

static size_t ucp_eager_single_pack(void *dest, void *arg)
{
    ucp_eager_hdr_t *hdr = dest;
    ucp_request_t *req   = arg;
    ucp_datatype_iter_t next_iter;
    size_t packed_size;

    ucs_assert(req->send.state.dt_iter.offset == 0);
    hdr->super.tag = req->send.msg_proto.tag;
    packed_size    = ucp_datatype_iter_next_pack(&req->send.state.dt_iter,
                                                 req->send.ep->worker,
                                                 SIZE_MAX, &next_iter, hdr + 1);
    return sizeof(*hdr) + packed_size;
}

static ucs_status_t ucp_eager_bcopy_single_progress(uct_pending_req_t *self)
{
    ucp_request_t                   *req = ucs_container_of(self, ucp_request_t,
                                                            send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;

    return ucp_proto_am_bcopy_single_progress(
            req, UCP_AM_ID_EAGER_ONLY, spriv->super.lane, ucp_eager_single_pack,
            req, SIZE_MAX, ucp_proto_request_bcopy_complete_success);
}

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
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.hdr_size      = sizeof(ucp_tag_hdr_t),
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_MAX_FRAG,
        .lane_type           = UCP_LANE_TYPE_AM,
        .tl_cap_flags        = UCT_IFACE_FLAG_AM_BCOPY
    };

    /* AM based proto can not be used if tag offload lane configured */
    if (!ucp_proto_eager_check_op_id(init_params, 0)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_single_init(&params);
}

static ucp_proto_t ucp_eager_bcopy_single_proto = {
    .name       = "egr/single/bcopy",
    .flags      = 0,
    .init       = ucp_proto_eager_bcopy_single_init,
    .config_str = ucp_proto_single_config_str,
    .progress   = ucp_eager_bcopy_single_progress,
};
UCP_PROTO_REGISTER(&ucp_eager_bcopy_single_proto);

static ucs_status_t
ucp_proto_eager_zcopy_single_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 0,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = 30,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_zcopy),
        .super.hdr_size      = sizeof(ucp_tag_hdr_t),
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_MAX_FRAG,
        .lane_type           = UCP_LANE_TYPE_AM,
        .tl_cap_flags        = UCT_IFACE_FLAG_AM_ZCOPY
    };

    /* AM based proto can not be used if tag offload lane configured */
    if (!ucp_proto_eager_check_op_id(init_params, 0)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_single_init(&params);
}

static ucs_status_t
ucp_proto_eager_zcopy_send_func(ucp_request_t *req,
                                const ucp_proto_single_priv_t *spriv,
                                const uct_iov_t *iov)
{
    ucp_eager_hdr_t hdr = {
        .super.tag = req->send.msg_proto.tag
    };

    return uct_ep_am_zcopy(req->send.ep->uct_eps[spriv->super.lane],
                           UCP_AM_ID_EAGER_ONLY, &hdr, sizeof(hdr), iov, 1, 0,
                           &req->send.state.uct_comp);
}

static ucs_status_t
ucp_proto_eager_zcopy_single_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    return ucp_proto_zcopy_single_progress(req, UCT_MD_MEM_ACCESS_LOCAL_READ,
                                           ucp_proto_eager_zcopy_send_func,
                                           "am_zcopy_only");
}

static ucp_proto_t ucp_eager_zcopy_single_proto = {
    .name       = "egr/single/zcopy",
    .flags      = 0,
    .init       = ucp_proto_eager_zcopy_single_init,
    .config_str = ucp_proto_single_config_str,
    .progress   = ucp_proto_eager_zcopy_single_progress,
};
UCP_PROTO_REGISTER(&ucp_eager_zcopy_single_proto);
