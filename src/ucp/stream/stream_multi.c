/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "stream.h"

#include <ucp/proto/proto_init.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/proto/proto_multi.inl>


/* Convenience macros for setting eager protocols descriptions  */
#define UCP_PROTO_STREAM_DESC "stream"
#define UCP_PROTO_STREAM_BCOPY_DESC \
    UCP_PROTO_STREAM_DESC " " UCP_PROTO_COPY_IN_DESC " " UCP_PROTO_COPY_OUT_DESC
#define UCP_PROTO_STREAM_ZCOPY_DESC \
    UCP_PROTO_STREAM_DESC " " UCP_PROTO_ZCOPY_DESC " " UCP_PROTO_COPY_OUT_DESC

static void
ucp_stream_multi_common_probe(const ucp_proto_multi_init_params_t *params)
{
    if (!ucp_proto_init_check_op(&params->super.super,
                                 UCS_BIT(UCP_OP_ID_STREAM_SEND))) {
        return;
    }

    ucp_proto_multi_probe(params);
}

static UCS_F_ALWAYS_INLINE void
ucp_stream_set_hdr(ucp_request_t *req, ucp_stream_am_hdr_t *hdr)
{
    hdr->ep_id = ucp_send_request_get_ep_remote_id(req);
}

static size_t ucp_stream_bcopy_pack(void *dest, void *arg)
{
    ucp_stream_am_hdr_t *hdr             = dest;
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;

    ucp_stream_set_hdr(pack_ctx->req, hdr);
    return sizeof(*hdr) + ucp_proto_multi_data_pack(pack_ctx, hdr + 1);
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_stream_multi_bcopy_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter, ucp_lane_index_t *lane_shift)
{
    return ucp_proto_am_bcopy_multi_common_send_func(
            req, lpriv, next_iter, UCP_AM_ID_STREAM_DATA, ucp_stream_bcopy_pack,
            sizeof(ucp_stream_am_hdr_t), UCP_AM_ID_STREAM_DATA,
            ucp_stream_bcopy_pack, sizeof(ucp_stream_am_hdr_t));
}

static ucs_status_t ucp_stream_multi_bcopy_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_bcopy_progress(
            req, req->send.proto_config->priv, NULL,
            ucp_stream_multi_bcopy_send_func,
            ucp_proto_request_bcopy_complete_success);
}

static void
ucp_stream_multi_bcopy_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = context->config.ext.proto_overhead_multi,
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = sizeof(ucp_stream_am_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_BCOPY,
        .super.memtype_op    = UCT_EP_OP_GET_SHORT,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_CAP_SEG_SIZE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING |
                               UCP_PROTO_COMMON_INIT_FLAG_RESUME,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_mem_info_unknown,
        .max_lanes           = 1,
        .initial_reg_md_map  = 0,
        .opt_align_offs      = UCP_PROTO_COMMON_OFFSET_INVALID,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_BCOPY,
        .first.lane_type     = UCP_LANE_TYPE_AM,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_BCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_AM
    };

    ucp_stream_multi_common_probe(&params);
}

ucp_proto_t ucp_stream_multi_bcopy_proto = {
    .name     = "stream/multi/bcopy",
    .desc     = UCP_PROTO_MULTI_FRAG_DESC " " UCP_PROTO_STREAM_BCOPY_DESC,
    .flags    = 0,
    .probe    = ucp_stream_multi_bcopy_probe,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_stream_multi_bcopy_progress},
    .abort    = ucp_proto_request_bcopy_abort,
    .reset    = ucp_proto_request_bcopy_reset
};

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_stream_multi_zcopy_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter, ucp_lane_index_t *lane_shift)
{
    ucp_stream_am_hdr_t hdr;

    ucp_stream_set_hdr(req, &hdr);
    return ucp_proto_am_zcopy_multi_common_send_func(req, lpriv, next_iter,
                                                     UCP_AM_ID_STREAM_DATA,
                                                     &hdr, sizeof(hdr),
                                                     UCP_AM_ID_STREAM_DATA,
                                                     &hdr, sizeof(hdr));
}

static ucs_status_t ucp_stream_multi_zcopy_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_zcopy_progress(
            req, req->send.proto_config->priv, NULL,
            UCT_MD_MEM_ACCESS_LOCAL_READ, UCP_DT_MASK_CONTIG_IOV,
            ucp_stream_multi_zcopy_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_proto_request_zcopy_completion);
}

static void
ucp_stream_multi_zcopy_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = context->config.ext.proto_overhead_multi,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = 30,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 1,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_zcopy),
        .super.max_iov_offs  = ucs_offsetof(uct_iface_attr_t, cap.am.max_iov),
        .super.hdr_size      = sizeof(ucp_stream_am_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_ZCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_CAP_SEG_SIZE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_proto_common_select_param_mem_info(
                                                     init_params->select_param),
        .max_lanes           = 1,
        .initial_reg_md_map  = 0,
        .opt_align_offs      = UCP_PROTO_COMMON_OFFSET_INVALID,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_ZCOPY,
        .first.lane_type     = UCP_LANE_TYPE_AM,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_ZCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_AM
    };

    ucp_stream_multi_common_probe(&params);
}

ucp_proto_t ucp_stream_multi_zcopy_proto = {
    .name     = "stream/multi/zcopy",
    .desc     = UCP_PROTO_MULTI_FRAG_DESC " " UCP_PROTO_STREAM_ZCOPY_DESC,
    .flags    = 0,
    .probe    = ucp_stream_multi_zcopy_probe,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_stream_multi_zcopy_progress},
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_request_zcopy_reset
};
