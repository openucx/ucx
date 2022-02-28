/**
 * Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "eager.inl"

#include <ucp/proto/proto.h>
#include <ucp/proto/proto_common.inl>
#include <ucp/proto/proto_multi.inl>


#define UCP_AM_FIRST_FRAG_META_LEN \
    (sizeof(ucp_am_hdr_t) + sizeof(ucp_am_first_ftr_t))


#define UCP_AM_MID_FRAG_META_LEN \
    (sizeof(ucp_am_hdr_t) + sizeof(ucp_am_mid_ftr_t))


static ucs_status_t
ucp_proto_eager_am_bcopy_multi_init(const ucp_proto_init_params_t *init_params)
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
        .max_lanes = init_params->worker->context->config.ext.max_eager_lanes
    };

    if (init_params->select_param->op_id != UCP_OP_ID_AM_SEND) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_multi_init(&params, params.super.super.priv,
                                params.super.super.priv_size);
}

static UCS_F_ALWAYS_INLINE void
ucp_am_fill_middle_footer(ucp_am_mid_ftr_t *ftr, ucp_request_t *req)
{
    ftr->msg_id = req->send.msg_proto.message_id;
    ftr->ep_id  = ucp_send_request_get_ep_remote_id(req);
}

static UCS_F_ALWAYS_INLINE void
ucp_am_fill_first_footer(ucp_am_first_ftr_t *ftr, ucp_request_t *req)
{
    ucp_am_fill_middle_footer(&ftr->super, req);
    ftr->total_size = req->send.state.dt_iter.length;
}

static size_t ucp_am_bcopy_pack_args_first(void *dest, void *arg)
{
    ucp_am_hdr_t *hdr                    = dest;
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;
    ucp_request_t *req                   = pack_ctx->req;
    ucp_am_first_ftr_t *first_ftr;
    size_t length, max_length;

    ucs_assertv(req->send.state.dt_iter.offset == 0, "offset %zu",
                req->send.state.dt_iter.offset);

    ucp_am_fill_header(hdr, req);

    max_length = ucs_min(ucp_am_send_req_total_size(req),
                         pack_ctx->max_payload);
    length     = ucp_am_bcopy_pack_data(hdr + 1, req, max_length,
                                        pack_ctx->next_iter);
    first_ftr  = UCS_PTR_BYTE_OFFSET(hdr + 1, length);

    ucp_am_fill_first_footer(first_ftr, req);

    return UCP_AM_FIRST_FRAG_META_LEN + length;
}

static size_t ucp_am_bcopy_pack_args_mid(void *dest, void *arg)
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

    ucp_am_fill_middle_footer(mid_ftr, req);

    return UCP_AM_MID_FRAG_META_LEN + length;
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_eager_am_bcopy_multi_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter)
{
    return ucp_proto_eager_bcopy_multi_common_send_func(
            req, lpriv, next_iter, UCP_AM_ID_AM_FIRST,
            ucp_am_bcopy_pack_args_first, UCP_AM_FIRST_FRAG_META_LEN,
            UCP_AM_ID_AM_MIDDLE, ucp_am_bcopy_pack_args_mid,
            UCP_AM_MID_FRAG_META_LEN);
}

static ucs_status_t
ucp_proto_eager_am_bcopy_multi_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_bcopy_progress(
            req, req->send.proto_config->priv, ucp_proto_msg_multi_request_init,
            ucp_proto_eager_am_bcopy_multi_send_func,
            ucp_proto_request_bcopy_complete_success);
}

ucp_proto_t ucp_eager_am_bcopy_multi_proto = {
    .name     = "egr/am/multi/bcopy",
    .desc     = UCP_PROTO_COPY_IN_DESC,
    .flags    = 0,
    .init     = ucp_proto_eager_am_bcopy_multi_init,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_eager_am_bcopy_multi_progress},
    .abort    = ucp_request_complete_send
};
