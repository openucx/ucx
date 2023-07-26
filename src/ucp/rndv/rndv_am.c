/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "proto_rndv.inl"


static ucs_status_t
ucp_proto_rndv_am_init_common(ucp_proto_multi_init_params_t *params)
{
    ucp_context_h context = params->super.super.worker->context;

    if (!ucp_proto_rndv_op_check(&params->super.super, UCP_OP_ID_RNDV_SEND,
                                 0)) {
        return UCS_ERR_UNSUPPORTED;
    }

    params->super.min_length = 0;
    params->super.max_length = SIZE_MAX;
    params->super.overhead   = 10e-9; /* for multiple lanes management */
    params->super.latency    = 0;
    params->first.lane_type  = UCP_LANE_TYPE_AM;
    params->middle.lane_type = UCP_LANE_TYPE_AM_BW;
    params->super.hdr_size   = sizeof(ucp_request_data_hdr_t);
    params->max_lanes        = context->config.ext.max_rndv_lanes;
    params->opt_align_offs   = UCP_PROTO_COMMON_OFFSET_INVALID;

    return ucp_proto_multi_init(params, params->super.super.priv,
                                params->super.super.priv_size);
}

static UCS_F_ALWAYS_INLINE void
ucp_rndv_am_fill_header(ucp_request_data_hdr_t *hdr, ucp_request_t *req)
{
    hdr->req_id = req->send.rndv.remote_req_id;
    hdr->offset = req->send.state.dt_iter.offset;
}

static UCS_F_ALWAYS_INLINE void ucp_rndv_am_destroy_rkey(ucp_request_t *req)
{
    if (req->send.rndv.rkey != NULL) {
        ucp_proto_rndv_rkey_destroy(req);
    }
}

static size_t ucp_proto_rndv_am_bcopy_pack(void *dest, void *arg)
{
    ucp_request_data_hdr_t *hdr          = dest;
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;

    ucp_rndv_am_fill_header(hdr, pack_ctx->req);

    return sizeof(*hdr) + ucp_proto_multi_data_pack(pack_ctx, hdr + 1);
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_rndv_am_bcopy_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter, ucp_lane_index_t *lane_shift)
{
    static const size_t hdr_size        = sizeof(ucp_request_data_hdr_t);
    ucp_ep_t *ep                        = req->send.ep;
    ucp_proto_multi_pack_ctx_t pack_ctx = {
        .req       = req,
        .next_iter = next_iter
    };
    ssize_t packed_size;

    pack_ctx.max_payload = ucp_proto_multi_max_payload(req, lpriv, hdr_size);

    packed_size = uct_ep_am_bcopy(ucp_ep_get_lane(ep, lpriv->super.lane),
                                  UCP_AM_ID_RNDV_DATA,
                                  ucp_proto_rndv_am_bcopy_pack, &pack_ctx, 0);
    if (ucs_unlikely(packed_size < 0)) {
        return (ucs_status_t)packed_size;
    }

    ucs_assert(packed_size >= hdr_size);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_am_bcopy_complete(ucp_request_t *req)
{
    ucp_rndv_am_destroy_rkey(req);
    return ucp_proto_request_bcopy_complete_success(req);
}

static ucs_status_t ucp_proto_rndv_am_bcopy_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_bcopy_progress(req, req->send.proto_config->priv,
                                          NULL,
                                          ucp_proto_rndv_am_bcopy_send_func,
                                          ucp_proto_rndv_am_bcopy_complete);
}

static ucs_status_t
ucp_proto_rndv_am_bcopy_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.send_op       = UCT_EP_OP_AM_BCOPY,
        .super.memtype_op    = UCT_EP_OP_GET_SHORT,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_CAP_SEG_SIZE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_BCOPY,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_BCOPY
    };

    return ucp_proto_rndv_am_init_common(&params);
}

static void
ucp_proto_rndv_am_bcopy_abort(ucp_request_t *req, ucs_status_t status)
{
    ucp_rndv_am_destroy_rkey(req);
    ucp_proto_request_bcopy_abort(req,status);
}

ucp_proto_t ucp_rndv_am_bcopy_proto = {
    .name     = "rndv/am/bcopy",
    .desc     = "fragmented " UCP_PROTO_COPY_IN_DESC " " UCP_PROTO_COPY_OUT_DESC,
    .flags    = 0,
    .init     = ucp_proto_rndv_am_bcopy_init,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_rndv_am_bcopy_progress},
    .abort    = ucp_proto_rndv_am_bcopy_abort,
    .reset    = (ucp_request_reset_func_t)ucp_proto_reset_fatal_not_implemented
};

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_rndv_am_zcopy_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter, ucp_lane_index_t *lane_shift)
{
    const size_t hdr_size    = sizeof(ucp_request_data_hdr_t);
    const size_t max_payload = ucp_proto_multi_max_payload(req, lpriv,
                                                           hdr_size);
    ucp_request_data_hdr_t hdr;
    uct_iov_t iov[UCP_MAX_IOV];
    size_t iov_count;

    ucp_rndv_am_fill_header(&hdr, req);

    ucs_assert(lpriv->super.max_iov > 0);
    iov_count = ucp_datatype_iter_next_iov(&req->send.state.dt_iter,
                                           max_payload, lpriv->super.md_index,
                                           UCP_DT_MASK_CONTIG_IOV, next_iter,
                                           iov, lpriv->super.max_iov);

    return uct_ep_am_zcopy(ucp_ep_get_lane(req->send.ep, lpriv->super.lane),
                           UCP_AM_ID_RNDV_DATA, &hdr, hdr_size, iov, iov_count,
                           0, &req->send.state.uct_comp);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_rndv_am_zcopy_complete(ucp_request_t *req)
{
    ucp_rndv_am_destroy_rkey(req);
    return ucp_request_invoke_uct_completion_success(req);
}

static ucs_status_t ucp_rndv_am_zcopy_proto_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_zcopy_progress(req, req->send.proto_config->priv,
                                          NULL, UCT_MD_MEM_ACCESS_LOCAL_READ,
                                          UCP_DT_MASK_CONTIG_IOV,
                                          ucp_rndv_am_zcopy_send_func,
                                          ucp_rndv_am_zcopy_complete,
                                          ucp_proto_request_zcopy_completion);
}

static ucs_status_t
ucp_rndv_am_zcopy_proto_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = 30,
        .super.min_iov       = 1,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_zcopy),
        .super.max_iov_offs  = ucs_offsetof(uct_iface_attr_t, cap.am.max_iov),
        .super.send_op       = UCT_EP_OP_AM_ZCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY   |
                               UCP_PROTO_COMMON_INIT_FLAG_CAP_SEG_SIZE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_ZCOPY,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_ZCOPY
    };

    return ucp_proto_rndv_am_init_common(&params);
}

static void
ucp_rndv_am_zcopy_proto_abort(ucp_request_t *req, ucs_status_t status)
{
    ucp_rndv_am_destroy_rkey(req);
    ucp_proto_request_zcopy_abort(req, status);
}

ucp_proto_t ucp_rndv_am_zcopy_proto = {
    .name     = "rndv/am/zcopy",
    .desc     = UCP_PROTO_ZCOPY_DESC,
    .flags    = 0,
    .init     = ucp_rndv_am_zcopy_proto_init,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_rndv_am_zcopy_proto_progress},
    .abort    = ucp_rndv_am_zcopy_proto_abort,
    .reset    = ucp_am_proto_request_zcopy_reset
};
