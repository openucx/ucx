/**
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "eager.inl"

#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_am.h>
#include <ucp/proto/proto_common.inl>
#include <ucp/proto/proto_single.h>
#include <ucp/proto/proto_single.inl>


static ucs_status_t ucp_eager_short_progress(uct_pending_req_t *self)
{
    ucp_request_t                   *req = ucs_container_of(self, ucp_request_t,
                                                            send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    uint32_t header_length               = req->send.msg_proto.am.header_length;
    size_t iov_cnt                       = 0ul;
    uct_iov_t iov[3];
    ucp_am_hdr_t am_hdr;
    ucs_status_t status;

    ucp_am_fill_header(&am_hdr, req);

    ucp_add_uct_iov_elem(iov, &am_hdr, sizeof(am_hdr), UCT_MEM_HANDLE_NULL,
                         &iov_cnt);
    ucp_add_uct_iov_elem(iov, req->send.state.dt_iter.type.contig.buffer,
                         req->send.state.dt_iter.length, UCT_MEM_HANDLE_NULL,
                         &iov_cnt);

    if (header_length != 0) {
        ucp_add_uct_iov_elem(iov,req->send.msg_proto.am.header, header_length,
                             UCT_MEM_HANDLE_NULL, &iov_cnt);
    }

    status = uct_ep_am_short_iov(req->send.ep->uct_eps[spriv->super.lane],
                                 UCP_AM_ID_AM_SINGLE, iov, iov_cnt);
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
    ucp_proto_single_init_params_t params        = {
        .super.super         = *init_params,
        .super.latency       = -150e-9,
        .super.overhead      = 0,
        .super.cfg_thresh    = UCS_MEMUNITS_AUTO,
        .super.cfg_priority  = 0,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        /* 3 iovs are needed to send a message. The iovs contain AM header,
           payload, user header. */
        .super.min_iov       = 3,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_short),
        .super.max_iov_offs  = ucs_offsetof(uct_iface_attr_t, cap.am.max_iov),
        .super.hdr_size      = sizeof(ucp_am_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_SHORT,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG,
        .lane_type           = UCP_LANE_TYPE_AM,
        .tl_cap_flags        = UCT_IFACE_FLAG_AM_SHORT
    };

    if ((init_params->select_param->op_id != UCP_OP_ID_AM_SEND) ||
        !ucp_proto_is_short_supported(select_param)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_single_init(&params);
}

ucp_proto_t ucp_am_eager_short_proto = {
    .name     = "egr/am/short",
    .desc     = UCP_PROTO_SHORT_DESC,
    .flags    = 0,
    .init     = ucp_proto_eager_short_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_eager_short_progress},
    .abort    = ucp_proto_request_bcopy_abort
};

static size_t ucp_eager_single_pack(void *dest, void *arg)
{
    ucp_am_hdr_t *hdr  = dest;
    ucp_request_t *req = arg;
    ucp_datatype_iter_t next_iter;
    size_t length;

    ucs_assert(req->send.state.dt_iter.offset == 0);

    ucp_am_fill_header(hdr, req);

    length = ucp_am_bcopy_pack_data(hdr + 1, req,
                                    ucp_am_send_req_total_size(req),
                                    &next_iter);

    ucs_assertv(length == ucp_am_send_req_total_size(req),
                "length %zu total_size %zu", length,
                ucp_am_send_req_total_size(req));

    return sizeof(*hdr) + length;
}

static ucs_status_t ucp_eager_bcopy_single_progress(uct_pending_req_t *self)
{
    ucp_request_t                   *req = ucs_container_of(self, ucp_request_t,
                                                            send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;

    return ucp_proto_am_bcopy_single_progress(
            req, UCP_AM_ID_AM_SINGLE, spriv->super.lane, ucp_eager_single_pack,
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
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = sizeof(ucp_am_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_BCOPY,
        .super.memtype_op    = UCT_EP_OP_GET_SHORT,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG,
        .lane_type           = UCP_LANE_TYPE_AM,
        .tl_cap_flags        = UCT_IFACE_FLAG_AM_BCOPY
    };

    if (init_params->select_param->op_id != UCP_OP_ID_AM_SEND) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_single_init(&params);
}

ucp_proto_t ucp_am_eager_bcopy_single_proto = {
    .name     = "egr/am/single/bcopy",
    .desc     = UCP_PROTO_COPY_IN_DESC,
    .flags    = 0,
    .init     = ucp_proto_eager_bcopy_single_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_eager_bcopy_single_progress},
    .abort    = ucp_request_complete_send
};

static ucs_status_t
ucp_proto_eager_am_zcopy_single_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context                = init_params->worker->context;
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 0,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = 30,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 2,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_zcopy),
        .super.max_iov_offs  = ucs_offsetof(uct_iface_attr_t, cap.am.max_iov),
        .super.hdr_size      = sizeof(ucp_am_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_ZCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG,
        .lane_type           = UCP_LANE_TYPE_AM,
        .tl_cap_flags        = UCT_IFACE_FLAG_AM_ZCOPY
    };

    if ((init_params->select_param->op_id != UCP_OP_ID_AM_SEND) ||
        (init_params->select_param->dt_class != UCP_DATATYPE_CONTIG)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_single_init(&params);
}

static ucs_status_t
ucp_proto_eager_am_zcopy_single_send_func(ucp_request_t *req,
                                          const ucp_proto_single_priv_t *spriv,
                                          uct_iov_t *iov)
{
    size_t iovcnt = 1;
    ucp_am_hdr_t hdr;

    ucp_am_fill_header(&hdr, req);

    ucp_proto_eager_am_zcopy_add_footer(req, 0, spriv->super.md_index, iov,
                                        &iovcnt,
                                        req->send.msg_proto.am.header_length);

    return uct_ep_am_zcopy(req->send.ep->uct_eps[spriv->super.lane],
                           UCP_AM_ID_AM_SINGLE, &hdr, sizeof(hdr), iov, iovcnt,
                           0, &req->send.state.uct_comp);
}

static ucs_status_t ucp_proto_request_eager_am_zcopy_single_init(
        ucp_request_t *req, ucp_md_map_t md_map,
        uct_completion_callback_t comp_func, unsigned uct_reg_flags,
        unsigned dt_mask)
{
    ucs_status_t status = ucp_proto_request_zcopy_init(req, md_map, comp_func,
                                                       uct_reg_flags, dt_mask);
    if (status != UCS_OK) {
        return status;
    }

    return ucp_eager_am_zcopy_pack_user_header(req);
}

static ucs_status_t
ucp_proto_eager_am_zcopy_single_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    return ucp_proto_zcopy_single_progress(
            req, UCT_MD_MEM_ACCESS_LOCAL_READ,
            ucp_proto_eager_am_zcopy_single_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_proto_request_eager_am_zcopy_completion,
            ucp_proto_request_eager_am_zcopy_single_init);
}

ucp_proto_t ucp_eager_am_zcopy_single_proto = {
    .name     = "egr/am/single/zcopy",
    .desc     = UCP_PROTO_ZCOPY_DESC,
    .flags    = 0,
    .init     = ucp_proto_eager_am_zcopy_single_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_proto_eager_am_zcopy_single_progress},
    .abort    = ucp_proto_request_bcopy_abort
};
