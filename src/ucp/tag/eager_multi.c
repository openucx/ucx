/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020-2021. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_eager.inl"

#include <ucp/core/ucp_request.inl>
#include <ucp/proto/proto_multi.inl>


static UCS_F_ALWAYS_INLINE void
ucp_proto_eager_set_first_hdr(ucp_request_t *req, ucp_eager_first_hdr_t *hdr)
{
    hdr->super.super.tag = req->send.msg_proto.tag;
    hdr->total_len       = req->send.state.dt_iter.length;
    hdr->msg_id          = req->send.msg_proto.message_id;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_eager_set_middle_hdr(ucp_request_t *req, ucp_eager_middle_hdr_t *hdr)
{
    hdr->msg_id = req->send.msg_proto.message_id;
    hdr->offset = req->send.state.dt_iter.offset;
}

static void
ucp_proto_eager_multi_probe_common(ucp_proto_multi_init_params_t *params,
                                   ucp_proto_id_t op_id)
{
    ucp_context_config_t *context_config;

    if (!ucp_tag_eager_check_op_id(&params->super.super, op_id, 0)) {
        return;
    }

    context_config           = &params->super.super.worker->context->config.ext;
    params->super.overhead   = context_config->proto_overhead_multi;
    params->super.latency    = 0;
    params->first.lane_type  = UCP_LANE_TYPE_AM;
    params->middle.lane_type = UCP_LANE_TYPE_AM_BW;
    params->max_lanes        = context_config->max_eager_lanes;
    ucp_proto_multi_probe(params);
}

static void ucp_proto_eager_bcopy_multi_common_probe(
        const ucp_proto_init_params_t *init_params, ucp_proto_id_t op_id,
        size_t hdr_size)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = hdr_size,
        .super.send_op       = UCT_EP_OP_AM_BCOPY,
        .super.memtype_op    = UCT_EP_OP_GET_SHORT,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_CAP_SEG_SIZE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING |
                               UCP_PROTO_COMMON_INIT_FLAG_RESUME,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_mem_info_unknown,
        .opt_align_offs      = UCP_PROTO_COMMON_OFFSET_INVALID,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_BCOPY,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_BCOPY
    };

    ucp_proto_eager_multi_probe_common(&params, op_id);
}

static size_t ucp_proto_eager_bcopy_pack_first(void *dest, void *arg)
{
    ucp_eager_first_hdr_t           *hdr = dest;
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;

    ucp_proto_eager_set_first_hdr(pack_ctx->req, hdr);
    return sizeof(*hdr) + ucp_proto_multi_data_pack(pack_ctx, hdr + 1);
}

static size_t ucp_proto_eager_bcopy_pack_middle(void *dest, void *arg)
{
    ucp_eager_middle_hdr_t          *hdr = dest;
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;

    ucp_proto_eager_set_middle_hdr(pack_ctx->req, hdr);
    return sizeof(*hdr) + ucp_proto_multi_data_pack(pack_ctx, hdr + 1);
}

static void
ucp_proto_eager_bcopy_multi_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_eager_bcopy_multi_common_probe(init_params, UCP_OP_ID_TAG_SEND,
                                             sizeof(ucp_eager_first_hdr_t));
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_eager_bcopy_multi_send_func(ucp_request_t *req,
                                      const ucp_proto_multi_lane_priv_t *lpriv,
                                      ucp_datatype_iter_t *next_iter,
                                      ucp_lane_index_t *lane_shift)
{
    return ucp_proto_am_bcopy_multi_common_send_func(
            req, lpriv, next_iter, UCP_AM_ID_EAGER_FIRST,
            ucp_proto_eager_bcopy_pack_first, sizeof(ucp_eager_first_hdr_t),
            UCP_AM_ID_EAGER_MIDDLE, ucp_proto_eager_bcopy_pack_middle,
            sizeof(ucp_eager_middle_hdr_t));
}

static ucs_status_t
ucp_proto_eager_bcopy_multi_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_bcopy_progress(
            req, req->send.proto_config->priv, ucp_proto_msg_multi_request_init,
            ucp_proto_eager_bcopy_multi_send_func,
            ucp_proto_request_bcopy_complete_success);
}

ucp_proto_t ucp_eager_bcopy_multi_proto = {
    .name     = "egr/multi/bcopy",
    .desc     = UCP_PROTO_MULTI_FRAG_DESC " " UCP_PROTO_EAGER_BCOPY_DESC,
    .flags    = 0,
    .probe    = ucp_proto_eager_bcopy_multi_probe,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_eager_bcopy_multi_progress},
    .abort    = ucp_proto_request_bcopy_abort,
    .reset    = ucp_proto_request_bcopy_reset
};

static void ucp_proto_eager_sync_bcopy_multi_probe(
        const ucp_proto_init_params_t *init_params)
{
    ucp_proto_eager_bcopy_multi_common_probe(init_params,
                                             UCP_OP_ID_TAG_SEND_SYNC,
                                             sizeof(ucp_eager_sync_first_hdr_t));
}

static size_t ucp_eager_sync_bcopy_pack_first(void *dest, void *arg)
{
    ucp_eager_sync_first_hdr_t *hdr      = dest;
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;
    ucp_request_t *req                   = pack_ctx->req;

    ucp_proto_eager_set_first_hdr(req, &hdr->super);
    hdr->req.ep_id  = ucp_send_request_get_ep_remote_id(req);
    hdr->req.req_id = ucp_send_request_get_id(req);

    return sizeof(*hdr) + ucp_proto_multi_data_pack(pack_ctx, hdr + 1);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_eager_sync_bcopy_multi_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter, ucp_lane_index_t *lane_shift)
{
    return ucp_proto_am_bcopy_multi_common_send_func(
            req, lpriv, next_iter, UCP_AM_ID_EAGER_SYNC_FIRST,
            ucp_eager_sync_bcopy_pack_first, sizeof(ucp_eager_sync_first_hdr_t),
            UCP_AM_ID_EAGER_MIDDLE, ucp_proto_eager_bcopy_pack_middle,
            sizeof(ucp_eager_middle_hdr_t));
}

void ucp_proto_eager_sync_ack_handler(ucp_worker_h worker,
                                      const ucp_reply_hdr_t *rep_hdr)
{
    ucp_request_t *req;

    UCP_SEND_REQUEST_GET_BY_ID(&req, worker, rep_hdr->req_id, 1, return,
                               "EAGER_S ACK %p", rep_hdr);

    req->flags |= UCP_REQUEST_FLAG_SYNC_REMOTE_COMPLETED;
    if (req->flags & UCP_REQUEST_FLAG_SYNC_LOCAL_COMPLETED) {
        ucp_request_complete_send(req, rep_hdr->status);
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_eager_sync_bcopy_request_init(ucp_request_t *req)
{
    if (!(req->flags & UCP_REQUEST_FLAG_SYNC_REMOTE_COMPLETED)) {
        ucp_send_request_id_alloc(req);
    }

    ucp_proto_msg_multi_request_init(req);
}

static ucs_status_t
ucp_proto_eager_sync_bcopy_multi_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_bcopy_progress(
            req, req->send.proto_config->priv,
            ucp_proto_eager_sync_bcopy_request_init,
            ucp_proto_eager_sync_bcopy_multi_send_func,
            ucp_proto_eager_sync_bcopy_send_completed);
}

ucp_proto_t ucp_eager_sync_bcopy_multi_proto = {
    .name     = "egrsnc/multi/bcopy",
    .desc     = UCP_PROTO_MULTI_FRAG_DESC " " UCP_PROTO_EAGER_BCOPY_DESC,
    .flags    = 0,
    .probe    = ucp_proto_eager_sync_bcopy_multi_probe,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_eager_sync_bcopy_multi_progress},
    .abort    = ucp_proto_request_bcopy_id_abort,
    .reset    = ucp_proto_request_bcopy_id_reset
};

static void
ucp_proto_eager_zcopy_multi_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = 30,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 1,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_zcopy),
        .super.max_iov_offs  = ucs_offsetof(uct_iface_attr_t, cap.am.max_iov),
        .super.hdr_size      = sizeof(ucp_eager_first_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_ZCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY   |
                               UCP_PROTO_COMMON_INIT_FLAG_CAP_SEG_SIZE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_proto_common_select_param_mem_info(
                                                     init_params->select_param),
        .opt_align_offs      = UCP_PROTO_COMMON_OFFSET_INVALID,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_ZCOPY,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_ZCOPY
    };

    ucp_proto_eager_multi_probe_common(&params, UCP_OP_ID_TAG_SEND);
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_eager_zcopy_multi_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter, ucp_lane_index_t *lane_shift)
{
    ucp_eager_middle_hdr_t hdr_middle;
    ucp_eager_first_hdr_t hdr_first;

    if (req->send.state.dt_iter.offset == 0) {
        ucp_proto_eager_set_first_hdr(req, &hdr_first);
    } else {
        ucp_proto_eager_set_middle_hdr(req, &hdr_middle);
    }

    return ucp_proto_am_zcopy_multi_common_send_func(
            req, lpriv, next_iter, UCP_AM_ID_EAGER_FIRST, &hdr_first,
            sizeof(hdr_first), UCP_AM_ID_EAGER_MIDDLE, &hdr_middle,
            sizeof(hdr_middle));
}

static ucs_status_t ucp_proto_eager_zcopy_multi_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_zcopy_progress(
            req, req->send.proto_config->priv, ucp_proto_msg_multi_request_init,
            UCT_MD_MEM_ACCESS_LOCAL_READ, UCP_DT_MASK_CONTIG_IOV,
            ucp_proto_eager_zcopy_multi_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_proto_request_zcopy_completion);
}

ucp_proto_t ucp_eager_zcopy_multi_proto = {
    .name     = "egr/multi/zcopy",
    .desc     = UCP_PROTO_MULTI_FRAG_DESC " " UCP_PROTO_EAGER_ZCOPY_DESC,
    .flags    = 0,
    .probe    = ucp_proto_eager_zcopy_multi_probe,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_eager_zcopy_multi_progress},
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_request_zcopy_reset
};
