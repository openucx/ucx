/**
 * Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "eager.inl"
#include "ucp_am.inl"

#include <ucp/proto/proto.h>
#include <ucp/proto/proto_common.inl>
#include <ucp/proto/proto_multi.inl>


#define UCP_AM_FIRST_FRAG_META_LEN \
    (sizeof(ucp_am_hdr_t) + sizeof(ucp_am_first_ftr_t))


#define UCP_AM_MID_FRAG_META_LEN \
    (sizeof(ucp_am_hdr_t) + sizeof(ucp_am_mid_ftr_t))


static ucs_status_t
ucp_am_eager_multi_bcopy_proto_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 10e-9,
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = sizeof(ucp_am_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_BCOPY,
        .super.memtype_op    = UCT_EP_OP_GET_SHORT,
        .super.flags         = 0,
        .first.lane_type     = UCP_LANE_TYPE_AM,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_BCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_AM_BW,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_BCOPY,
        .max_lanes           = context->config.ext.max_eager_lanes
    };

    if (!ucp_am_check_init_params(init_params, UCP_AM_OP_ID_MASK_ALL,
                                  UCP_PROTO_SELECT_OP_FLAG_AM_RNDV)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_multi_init(&params, params.super.super.priv,
                                params.super.super.priv_size);
}

static UCS_F_ALWAYS_INLINE void
ucp_am_eager_fill_middle_footer(ucp_am_mid_ftr_t *ftr, ucp_request_t *req)
{
    ftr->msg_id = req->send.msg_proto.message_id;
    ftr->ep_id  = ucp_send_request_get_ep_remote_id(req);
}

static UCS_F_ALWAYS_INLINE void
ucp_am_eager_fill_first_footer(ucp_am_first_ftr_t *ftr, ucp_request_t *req)
{
    ucp_am_eager_fill_middle_footer(&ftr->super, req);
    ftr->total_size = req->send.state.dt_iter.length;
}

static size_t ucp_am_eager_multi_bcopy_pack_args_first(void *dest, void *arg)
{
    ucp_am_hdr_t *hdr                    = dest;
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;
    ucp_request_t *req                   = pack_ctx->req;
    size_t length, max_length;

    ucs_assertv(req->send.state.dt_iter.offset == 0, "offset %zu",
                req->send.state.dt_iter.offset);

    ucp_am_fill_header(hdr, req);

    max_length = ucs_min(ucp_am_send_req_total_size(req),
                         pack_ctx->max_payload);
    length     = ucp_am_eager_bcopy_pack_data(hdr + 1, req, max_length,
                                              pack_ctx->next_iter);
    ucp_am_eager_fill_first_footer(UCS_PTR_BYTE_OFFSET(hdr + 1, length), req);

    return UCP_AM_FIRST_FRAG_META_LEN + length;
}

static size_t ucp_am_eager_multi_bcopy_pack_args_mid(void *dest, void *arg)
{
    ucp_am_mid_hdr_t *hdr                = dest;
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;
    ucp_request_t *req                   = pack_ctx->req;
    size_t length;
    ucp_am_mid_ftr_t *mid_ftr;

    /* some amount of data should be packed in the first fragment */
    ucs_assert(req->send.state.dt_iter.offset > 0);

    hdr->offset = req->send.state.dt_iter.offset;
    length      = ucp_proto_multi_data_pack(pack_ctx, hdr + 1);
    mid_ftr     = UCS_PTR_BYTE_OFFSET(hdr + 1, length);

    ucp_am_eager_fill_middle_footer(mid_ftr, req);

    return UCP_AM_MID_FRAG_META_LEN + length;
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_am_eager_multi_bcopy_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter)
{
    return ucp_proto_eager_bcopy_multi_common_send_func(
            req, lpriv, next_iter, UCP_AM_ID_AM_FIRST,
            ucp_am_eager_multi_bcopy_pack_args_first,
            UCP_AM_FIRST_FRAG_META_LEN, UCP_AM_ID_AM_MIDDLE,
            ucp_am_eager_multi_bcopy_pack_args_mid, UCP_AM_MID_FRAG_META_LEN);
}

static ucs_status_t
ucp_am_eager_multi_bcopy_proto_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_bcopy_progress(
            req, req->send.proto_config->priv, ucp_proto_msg_multi_request_init,
            ucp_am_eager_multi_bcopy_send_func,
            ucp_proto_request_bcopy_complete_success);
}

ucp_proto_t ucp_am_eager_multi_bcopy_proto = {
    .name     = "am/egr/multi/bcopy",
    .desc     = UCP_PROTO_COPY_IN_DESC,
    .flags    = 0,
    .init     = ucp_am_eager_multi_bcopy_proto_init,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_am_eager_multi_bcopy_proto_progress},
    .abort    = ucp_request_complete_send
};

static ucs_status_t
ucp_am_eager_multi_zcopy_proto_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 10e-9,
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
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY,
        .first.lane_type     = UCP_LANE_TYPE_AM,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_ZCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_AM_BW,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_ZCOPY,
        .max_lanes           = context->config.ext.max_eager_lanes
    };

    if (!ucp_am_check_init_params(init_params, UCP_AM_OP_ID_MASK_ALL,
                                  UCP_PROTO_SELECT_OP_FLAG_AM_RNDV)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_multi_init(&params, params.super.super.priv,
                                params.super.super.priv_size);
}

static UCS_F_ALWAYS_INLINE size_t ucp_am_eager_multi_zcopy_add_payload(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        size_t meta_size, ucp_datatype_iter_t *next_iter, uct_iov_t *iov)
{
    size_t max_payload = ucp_proto_multi_max_payload(req, lpriv, meta_size);

    ucs_assert(lpriv->super.max_iov > 1);

    return ucp_datatype_iter_next_iov(&req->send.state.dt_iter, max_payload,
                                      lpriv->super.md_index,
                                      UCP_DT_MASK_CONTIG_IOV, next_iter, iov,
                                      lpriv->super.max_iov - 1);
}

static UCS_F_ALWAYS_INLINE void
ucp_am_eager_fill_middle_header(ucp_am_mid_hdr_t *hdr, ucp_request_t *req)
{
    hdr->offset = req->send.state.dt_iter.offset;
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_am_eager_multi_zcopy_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter)
{
    size_t user_hdr_size = req->send.msg_proto.am.header_length;
    union {
        ucp_am_hdr_t     first;
        ucp_am_mid_hdr_t middle;
    } hdr;
    ucp_am_id_t am_id;
    size_t footer_size, footer_offset, iov_count;
    ucp_am_first_ftr_t *ftr;
    uct_iov_t iov[UCP_MAX_IOV];

    UCS_STATIC_ASSERT(sizeof(hdr.first) == sizeof(ucp_am_hdr_t));
    UCS_STATIC_ASSERT(sizeof(hdr.middle) == sizeof(ucp_am_hdr_t));

    if (req->send.state.dt_iter.offset == 0) {
        am_id         = UCP_AM_ID_AM_FIRST;
        footer_size   = sizeof(*ftr) + user_hdr_size;
        footer_offset = 0;
        ucp_am_fill_header(&hdr.first, req);
        /* The method also fills middle/last fragment footer. The footer can be
           reused by all fragments because it is immutable. */
        ftr = UCS_PTR_BYTE_OFFSET(req->send.msg_proto.am.reg_desc + 1,
                                  user_hdr_size);
        ucp_am_eager_fill_first_footer(ftr, req);
    } else {
        am_id         = UCP_AM_ID_AM_MIDDLE;
        footer_size   = sizeof(ftr->super);
        footer_offset = user_hdr_size;
        ucp_am_eager_fill_middle_header(&hdr.middle, req);
    }

    iov_count = ucp_am_eager_multi_zcopy_add_payload(
            req, lpriv, sizeof(ucp_am_hdr_t) + footer_size, next_iter, iov);
    ucp_am_eager_zcopy_add_footer(req, footer_offset, lpriv->super.md_index,
                                  iov, &iov_count, footer_size);

    return uct_ep_am_zcopy(req->send.ep->uct_eps[lpriv->super.lane], am_id,
                           &hdr, sizeof(ucp_am_hdr_t), iov, iov_count, 0,
                           &req->send.state.uct_comp);
}

static void ucp_am_eager_multi_zcopy_init(ucp_request_t *req)
{
    ucp_am_eager_zcopy_pack_user_header(req);
    ucp_proto_msg_multi_request_init(req);
}

static ucs_status_t
ucp_am_eager_multi_zcopy_proto_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_zcopy_progress(
            req, req->send.proto_config->priv, ucp_am_eager_multi_zcopy_init,
            UCT_MD_MEM_ACCESS_LOCAL_READ, UCP_DT_MASK_CONTIG_IOV,
            ucp_am_eager_multi_zcopy_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_am_eager_zcopy_completion);
}

ucp_proto_t ucp_am_eager_multi_zcopy_proto = {
    .name     = "am/egr/multi/zcopy",
    .desc     = UCP_PROTO_ZCOPY_DESC,
    .flags    = 0,
    .init     = ucp_am_eager_multi_zcopy_proto_init,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_am_eager_multi_zcopy_proto_progress},
    .abort    = ucp_proto_request_bcopy_abort
};
