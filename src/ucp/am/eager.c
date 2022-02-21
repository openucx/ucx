/**
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_am.h>
#include <ucp/proto/proto_common.inl>
#include <ucp/proto/proto_single.h>
#include <ucp/proto/proto_single.inl>
#include <ucp/proto/proto_am.inl>


static UCS_F_ALWAYS_INLINE size_t
ucp_am_send_req_total_size(ucp_request_t *req)
{
    return req->send.state.dt_iter.length +
           req->send.msg_proto.am.header_length;
}

static UCS_F_ALWAYS_INLINE void
ucp_am_fill_header(ucp_am_hdr_t *hdr, ucp_request_t *req)
{
    hdr->am_id         = req->send.msg_proto.am.am_id;
    hdr->flags         = req->send.msg_proto.am.flags;
    hdr->header_length = req->send.msg_proto.am.header_length;
}

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
                                 UCP_AM_ID_SINGLE, iov, iov_cnt);
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

static UCS_F_ALWAYS_INLINE void
ucp_am_pack_user_header(void *buffer, ucp_request_t *req)
{
    ucp_dt_state_t hdr_state;

    hdr_state.offset = 0ul;

    ucp_dt_pack(req->send.ep->worker, ucp_dt_make_contig(1),
                UCS_MEMORY_TYPE_HOST, buffer, req->send.msg_proto.am.header,
                &hdr_state, req->send.msg_proto.am.header_length);
}

static UCS_F_ALWAYS_INLINE ssize_t
ucp_am_bcopy_pack_data(void *buffer, ucp_request_t *req, size_t length)
{
    unsigned user_header_length = req->send.msg_proto.am.header_length;
    size_t total_length;
    ucp_datatype_iter_t next_iter;
    void *user_hdr;

    ucs_assertv((req->send.state.dt_iter.length == 0) ||
                (length > user_header_length),
                "length %zu user_header length %u", length, user_header_length);

    total_length = ucp_datatype_iter_next_pack(&req->send.state.dt_iter,
                                               req->send.ep->worker, SIZE_MAX,
                                               &next_iter, buffer);
    if (user_header_length != 0) {
        /* Pack user header to the end of message/fragment */
        user_hdr = UCS_PTR_BYTE_OFFSET(buffer, total_length);
        ucp_am_pack_user_header(user_hdr, req);
        total_length += user_header_length;
    }

    return total_length;
}

static size_t ucp_eager_single_pack(void *dest, void *arg)
{
    ucp_am_hdr_t *hdr  = dest;
    ucp_request_t *req = arg;
    size_t length;

    ucs_assert(req->send.state.dt_iter.offset == 0);

    ucp_am_fill_header(hdr, req);

    length = ucp_am_bcopy_pack_data(hdr + 1, req,
                                    ucp_am_send_req_total_size(req));

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
            req, UCP_AM_ID_SINGLE, spriv->super.lane, ucp_eager_single_pack,
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

static void ucp_proto_eager_am_zcopy_add_footer(ucp_request_t *req,
                                                ucp_lane_index_t lane,
                                                uct_iov_t *iov, size_t *iovcnt)
{
    size_t user_hdr_size = req->send.msg_proto.am.header_length;
    ucp_mem_desc_t *user_hdr_desc;
    ucp_md_index_t md_idx;

    if (user_hdr_size == 0) {
        return;
    }

    user_hdr_desc = req->send.msg_proto.am.reg_desc;
    md_idx        = ucp_ep_md_index(req->send.ep, lane);

    ucp_add_uct_iov_elem(iov, user_hdr_desc + 1, user_hdr_size,
                         user_hdr_desc->memh->uct[md_idx], iovcnt);
}

static ucs_status_t
ucp_proto_eager_am_zcopy_single_send_func(ucp_request_t *req,
                                          const ucp_proto_single_priv_t *spriv,
                                          uct_iov_t *iov)
{
    size_t iovcnt = 1;
    ucp_am_hdr_t hdr;

    ucp_am_fill_header(&hdr, req);

    ucp_proto_eager_am_zcopy_add_footer(req, spriv->super.lane, iov, &iovcnt);

    return uct_ep_am_zcopy(req->send.ep->uct_eps[spriv->super.lane],
                           UCP_AM_ID_SINGLE, &hdr, sizeof(hdr), iov, iovcnt, 0,
                           &req->send.state.uct_comp);
}

void ucp_proto_request_eager_am_zcopy_single_completion(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);

    if (req->send.msg_proto.am.header_length != 0) {
        ucs_assert(req->send.msg_proto.am.reg_desc != NULL);
        ucs_mpool_put_inline(req->send.msg_proto.am.reg_desc);
    }
    ucp_proto_request_zcopy_completion(self);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_eager_am_zcopy_pack_user_header(ucp_request_t *req)
{
    ucp_mem_desc_t *reg_desc;

    if (req->send.msg_proto.am.header_length == 0) {
        return UCS_OK;
    }

    reg_desc = ucp_worker_mpool_get(&req->send.ep->worker->reg_mp);
    if (ucs_unlikely(reg_desc == NULL)) {
        return UCS_ERR_NO_MEMORY;
    }

    ucs_assert(req->send.msg_proto.am.header != NULL);
    ucp_am_pack_user_header(reg_desc + 1, req);
    req->send.msg_proto.am.reg_desc = reg_desc;

    return UCS_OK;
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
            ucp_proto_request_eager_am_zcopy_single_completion,
            ucp_proto_request_eager_am_zcopy_single_init);
}

ucp_proto_t ucp_eager_am_zcopy_single_proto = {
    .name     = "egr/am/single/zcopy",
    .flags    = 0,
    .init     = ucp_proto_eager_am_zcopy_single_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_proto_eager_am_zcopy_single_progress},
    .abort    = ucp_proto_request_bcopy_abort
};
