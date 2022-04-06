/**
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "eager.inl"
#include "ucp_am.inl"

#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_am.h>
#include <ucp/proto/proto_common.inl>
#include <ucp/proto/proto_single.h>
#include <ucp/proto/proto_single.inl>


static UCS_F_ALWAYS_INLINE void
ucp_am_eager_fill_reply_footer(ucp_am_reply_ftr_t *ftr, ucp_request_t *req)
{
    ftr->ep_id = ucp_send_request_get_ep_remote_id(req);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_eager_short_proto_progress_common(uct_pending_req_t *self, int is_reply)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    uint32_t header_length               = req->send.msg_proto.am.header_length;
    size_t iov_cnt                       = 0ul;
    uct_iov_t iov[3];
    ucp_am_hdr_t am_hdr;
    ucs_status_t status;
    uint8_t am_id;
    ucp_am_reply_ftr_t ftr;

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

    if (is_reply) {
        status = ucp_ep_resolve_remote_id(req->send.ep, spriv->super.lane);
        if (ucs_unlikely(status != UCS_OK)) {
            return status;
        }

        am_id = UCP_AM_ID_AM_SINGLE_REPLY;
        ucp_am_eager_fill_reply_footer(&ftr, req);
        ucp_add_uct_iov_elem(iov, &ftr, sizeof(ftr), UCT_MEM_HANDLE_NULL,
                             &iov_cnt);
    } else {
        am_id = UCP_AM_ID_AM_SINGLE;
    }

    status = uct_ep_am_short_iov(req->send.ep->uct_eps[spriv->super.lane],
                                 am_id, iov, iov_cnt);
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
ucp_am_eager_short_proto_init_common(const ucp_proto_init_params_t *init_params,
                                     size_t min_iov, ucp_operation_id_t op_id)
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
        .super.min_iov       = min_iov,
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

    if (!ucp_am_check_init_params(init_params, UCS_BIT(op_id),
                                  UCP_PROTO_SELECT_OP_FLAG_AM_RNDV) ||
        !ucp_proto_is_short_supported(select_param)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_single_init(&params);
}

static ucs_status_t
ucp_am_eager_short_proto_init(const ucp_proto_init_params_t *init_params)
{
    /* 3 iovs are needed to send a message. The iovs contain AM header,
       payload, user header. */
    return ucp_am_eager_short_proto_init_common(init_params, 3,
                                                UCP_OP_ID_AM_SEND);
}

static ucs_status_t ucp_am_eager_short_proto_progress(uct_pending_req_t *self)
{
    return ucp_am_eager_short_proto_progress_common(self, 0);
}

ucp_proto_t ucp_am_eager_short_proto = {
    .name     = "am/egr/short",
    .desc     = UCP_PROTO_SHORT_DESC,
    .flags    = UCP_PROTO_FLAG_AM_SHORT,
    .init     = ucp_am_eager_short_proto_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_am_eager_short_proto_progress},
    .abort    = ucp_proto_request_bcopy_abort
};

static ucs_status_t
ucp_am_eager_short_reply_proto_init(const ucp_proto_init_params_t *init_params)
{
    /* 4 iovs are needed to send a message. The iovs contain AM header,
       payload, user header, reply footer. */
    return ucp_am_eager_short_proto_init_common(init_params, 4,
                                                UCP_OP_ID_AM_SEND_REPLY);
}

static ucs_status_t
ucp_am_eager_short_reply_proto_progress(uct_pending_req_t *self)
{
    return ucp_am_eager_short_proto_progress_common(self, 1);
}

ucp_proto_t ucp_am_eager_short_reply_proto = {
    .name     = "am/egr/short/reply",
    .desc     = UCP_PROTO_SHORT_DESC,
    .flags    = UCP_PROTO_FLAG_AM_SHORT,
    .init     = ucp_am_eager_short_reply_proto_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_am_eager_short_reply_proto_progress},
    .abort    = ucp_proto_request_bcopy_abort
};

static UCS_F_ALWAYS_INLINE size_t
ucp_am_eager_single_bcopy_pack_common(void *dest, void *arg, int is_reply)
{
    ucp_am_hdr_t *hdr  = dest;
    ucp_request_t *req = arg;
    ucp_datatype_iter_t next_iter;
    size_t length;
    ucp_am_reply_ftr_t *ftr;

    ucs_assert(req->send.state.dt_iter.offset == 0);

    ucp_am_fill_header(hdr, req);

    length = ucp_am_eager_bcopy_pack_data(hdr + 1, req,
                                          ucp_am_send_req_total_size(req),
                                          &next_iter);

    ucs_assertv(length == ucp_am_send_req_total_size(req),
                "length %zu total_size %zu", length,
                ucp_am_send_req_total_size(req));

    if (is_reply) {
        ftr     = UCS_PTR_BYTE_OFFSET(hdr + 1, length);
        length += sizeof(*ftr);
        ucp_am_eager_fill_reply_footer(ftr, req);
    }

    return sizeof(*hdr) + length;
}

static size_t ucp_am_eager_single_bcopy_pack(void *dest, void *arg)
{
    return ucp_am_eager_single_bcopy_pack_common(dest, arg, 0);
}

static ucs_status_t
ucp_am_eager_single_bcopy_proto_progress(uct_pending_req_t *self)
{
    ucp_request_t                   *req = ucs_container_of(self, ucp_request_t,
                                                            send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;

    return ucp_proto_am_bcopy_single_progress(
            req, UCP_AM_ID_AM_SINGLE, spriv->super.lane,
            ucp_am_eager_single_bcopy_pack, req, SIZE_MAX,
            ucp_proto_request_bcopy_complete_success);
}

static ucs_status_t ucp_am_eager_single_bcopy_proto_init_common(
        const ucp_proto_init_params_t *init_params, ucp_operation_id_t op_id)
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

    if (!ucp_am_check_init_params(init_params, UCS_BIT(op_id),
                                  UCP_PROTO_SELECT_OP_FLAG_AM_RNDV)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_single_init(&params);
}

static ucs_status_t
ucp_am_eager_single_bcopy_proto_init(const ucp_proto_init_params_t *init_params)
{
    return ucp_am_eager_single_bcopy_proto_init_common(init_params,
                                                       UCP_OP_ID_AM_SEND);
}

ucp_proto_t ucp_am_eager_single_bcopy_proto = {
    .name     = "am/egr/single/bcopy",
    .desc     = UCP_PROTO_COPY_IN_DESC,
    .flags    = 0,
    .init     = ucp_am_eager_single_bcopy_proto_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_am_eager_single_bcopy_proto_progress},
    .abort    = ucp_request_complete_send
};

static ucs_status_t ucp_am_eager_single_bcopy_reply_proto_init(
        const ucp_proto_init_params_t *init_params)
{
    return ucp_am_eager_single_bcopy_proto_init_common(init_params,
                                                       UCP_OP_ID_AM_SEND_REPLY);
}

static size_t ucp_am_eager_single_bcopy_reply_pack(void *dest, void *arg)
{
    return ucp_am_eager_single_bcopy_pack_common(dest, arg, 1);
}

static ucs_status_t
ucp_am_eager_single_bcopy_reply_proto_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;

    return ucp_proto_am_bcopy_single_progress(
            req, UCP_AM_ID_AM_SINGLE_REPLY, spriv->super.lane,
            ucp_am_eager_single_bcopy_reply_pack, req, SIZE_MAX,
            ucp_proto_request_bcopy_complete_success);
}

ucp_proto_t ucp_am_eager_single_bcopy_reply_proto = {
    .name     = "am/egr/single/bcopy/reply",
    .desc     = UCP_PROTO_COPY_IN_DESC,
    .flags    = 0,
    .init     = ucp_am_eager_single_bcopy_reply_proto_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_am_eager_single_bcopy_reply_proto_progress},
    .abort    = ucp_request_complete_send
};

static ucs_status_t ucp_am_eager_single_zcopy_proto_init_common(
        const ucp_proto_init_params_t *init_params, ucp_operation_id_t op_id)
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

    if (!ucp_am_check_init_params(init_params, UCS_BIT(op_id),
                                  UCP_PROTO_SELECT_OP_FLAG_AM_RNDV) ||
        (init_params->select_param->dt_class != UCP_DATATYPE_CONTIG)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_single_init(&params);
}

static ucs_status_t
ucp_am_eager_single_zcopy_proto_init(const ucp_proto_init_params_t *init_params)
{
    return ucp_am_eager_single_zcopy_proto_init_common(init_params,
                                                       UCP_OP_ID_AM_SEND);
}

static ucs_status_t
ucp_am_eager_single_zcopy_send_func(ucp_request_t *req,
                                    const ucp_proto_single_priv_t *spriv,
                                    uct_iov_t *iov)
{
    size_t iovcnt = 1;
    ucp_am_hdr_t hdr;

    ucp_am_fill_header(&hdr, req);

    ucp_am_eager_zcopy_add_footer(req, 0, spriv->super.md_index, iov, &iovcnt,
                                  req->send.msg_proto.am.header_length);

    return uct_ep_am_zcopy(req->send.ep->uct_eps[spriv->super.lane],
                           UCP_AM_ID_AM_SINGLE, &hdr, sizeof(hdr), iov, iovcnt,
                           0, &req->send.state.uct_comp);
}

static ucs_status_t
ucp_am_eager_single_zcopy_init(ucp_request_t *req, ucp_md_map_t md_map,
                               uct_completion_callback_t comp_func,
                               unsigned uct_reg_flags, unsigned dt_mask)
{
    ucs_status_t status = ucp_proto_request_zcopy_init(req, md_map, comp_func,
                                                       uct_reg_flags, dt_mask);
    if (status != UCS_OK) {
        return status;
    }

    return ucp_am_eager_zcopy_pack_user_header(req);
}

static ucs_status_t
ucp_am_eager_single_zcopy_proto_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    return ucp_proto_zcopy_single_progress(
            req, UCT_MD_MEM_ACCESS_LOCAL_READ,
            ucp_am_eager_single_zcopy_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_am_eager_zcopy_completion, ucp_am_eager_single_zcopy_init);
}

ucp_proto_t ucp_am_eager_single_zcopy_proto = {
    .name     = "am/egr/single/zcopy",
    .desc     = UCP_PROTO_ZCOPY_DESC,
    .flags    = 0,
    .init     = ucp_am_eager_single_zcopy_proto_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_am_eager_single_zcopy_proto_progress},
    .abort    = ucp_proto_request_bcopy_abort
};

static ucs_status_t ucp_am_eager_single_zcopy_reply_proto_init(
        const ucp_proto_init_params_t *init_params)
{
    return ucp_am_eager_single_zcopy_proto_init_common(init_params,
                                                       UCP_OP_ID_AM_SEND_REPLY);
}

static ucs_status_t
ucp_am_eager_single_zcopy_reply_send_func(ucp_request_t *req,
                                          const ucp_proto_single_priv_t *spriv,
                                          uct_iov_t *iov)
{
    size_t iovcnt = 1;
    ucp_am_hdr_t hdr;
    ucp_am_reply_ftr_t *ftr;

    ucp_am_fill_header(&hdr, req);
    ucs_assert(req->send.msg_proto.am.reg_desc != NULL);

    ftr = UCS_PTR_BYTE_OFFSET(req->send.msg_proto.am.reg_desc + 1,
                              req->send.msg_proto.am.header_length);
    ucp_am_eager_fill_reply_footer(ftr, req);

    ucp_am_eager_zcopy_add_footer(req, 0, spriv->super.md_index, iov, &iovcnt,
                                  req->send.msg_proto.am.header_length +
                                          sizeof(*ftr));

    return uct_ep_am_zcopy(req->send.ep->uct_eps[spriv->super.lane],
                           UCP_AM_ID_AM_SINGLE_REPLY, &hdr, sizeof(hdr), iov,
                           iovcnt, 0, &req->send.state.uct_comp);
}

static ucs_status_t
ucp_am_eager_single_zcopy_reply_proto_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    return ucp_proto_zcopy_single_progress(
            req, UCT_MD_MEM_ACCESS_LOCAL_READ,
            ucp_am_eager_single_zcopy_reply_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_am_eager_zcopy_completion, ucp_am_eager_single_zcopy_init);
}

ucp_proto_t ucp_am_eager_single_zcopy_reply_proto = {
    .name     = "am/egr/single/zcopy/reply",
    .desc     = UCP_PROTO_ZCOPY_DESC,
    .flags    = 0,
    .init     = ucp_am_eager_single_zcopy_reply_proto_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_am_eager_single_zcopy_reply_proto_progress},
    .abort    = ucp_proto_request_bcopy_abort
};
