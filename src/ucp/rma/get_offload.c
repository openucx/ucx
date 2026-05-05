/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rma.inl"

#include <ucp/core/ucp_request.inl>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto_init.h>
#include <ucp/proto/proto_multi.inl>


typedef struct {
    ucp_request_t *req;
    void          *dest;
} ucp_proto_get_bcopy_unpack_ctx_t;


static void ucp_proto_get_offload_bcopy_unpack(void *arg, const void *data,
                                               size_t length)
{
    ucp_proto_get_bcopy_unpack_ctx_t *unpack_ctx = arg;
    ucp_request_t *req                           = unpack_ctx->req;

    ucp_dt_contig_unpack(req->send.ep->worker, unpack_ctx->dest, data, length,
                         (ucs_memory_type_t)req->send.state.dt_iter.mem_info.type,
                         req->send.state.dt_iter.length);
}

static UCS_F_ALWAYS_INLINE size_t
ucp_proto_get_offload_bcopy_num_frags(const ucp_request_t *req,
                                      const ucp_proto_multi_lane_priv_t *lpriv)
{
    return ucs_div_round_up(req->send.state.dt_iter.length, lpriv->max_frag);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_get_offload_bcopy_send_func(ucp_request_t *req,
                                      const ucp_proto_multi_lane_priv_t *lpriv,
                                      ucp_datatype_iter_t *next_iter,
                                      ucp_lane_index_t *lane_shift)
{
    ucp_proto_get_bcopy_unpack_ctx_t *unpack_ctx;
    uct_rkey_t tl_rkey = ucp_rkey_get_tl_rkey(req->send.rma.rkey,
                                              lpriv->super.rkey_index);
    size_t max_length, length;
    size_t frag_idx;
    void *dest;

    max_length = ucp_proto_multi_max_payload(req, lpriv, 0);
    length     = ucp_datatype_iter_next_ptr(&req->send.state.dt_iter,
                                            max_length, next_iter, &dest);
    frag_idx   = req->send.state.dt_iter.offset / lpriv->max_frag;
    ucs_assertv(frag_idx < ucp_proto_get_offload_bcopy_num_frags(req, lpriv),
                "frag_idx=%zu offset=%zu max_frag=%zu length=%zu", frag_idx,
                req->send.state.dt_iter.offset, lpriv->max_frag,
                req->send.state.dt_iter.length);

    unpack_ctx                = req->send.buffer;
    unpack_ctx[frag_idx].req  = req;
    unpack_ctx[frag_idx].dest = dest;

    return uct_ep_get_bcopy(ucp_ep_get_lane(req->send.ep, lpriv->super.lane),
                            ucp_proto_get_offload_bcopy_unpack,
                            &unpack_ctx[frag_idx],
                            length,
                            req->send.rma.remote_addr +
                                    req->send.state.dt_iter.offset,
                            tl_rkey, &req->send.state.uct_comp);
}

static void ucp_proto_get_offload_bcopy_completion(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);
    ucs_free(req->send.buffer);
    req->send.buffer = NULL;
    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 0,
                              UCS_BIT(UCP_DATATYPE_CONTIG));
    ucp_request_complete_send(req, req->send.state.uct_comp.status);
}

static ucs_status_t ucp_proto_get_offload_bcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req                  = ucs_container_of(self, ucp_request_t,
                                                           send.uct);
    const ucp_proto_multi_priv_t *mpriv = req->send.proto_config->priv;
    size_t ctx_count;
    size_t num_frags;
    ucs_status_t status;

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        ucp_proto_multi_request_init(req);

        status = ucp_ep_rma_handle_fence(req->send.ep, req, mpriv->lane_map);
        if (status != UCS_OK) {
            ucp_proto_request_abort(req, status);
            return UCS_OK;
        }

        UCS_STATIC_ASSERT(UCP_PROTO_RMA_MAX_BCOPY_LANES == 1);
        ucs_assertv(mpriv->num_lanes == 1, "num_lanes=%u", mpriv->num_lanes);

        num_frags        = ucp_proto_get_offload_bcopy_num_frags(req,
                                                                 &mpriv->lanes[0]);
        ctx_count        = ucs_max(num_frags, (size_t)1);
        req->send.buffer = ucs_malloc(ctx_count *
                                      sizeof(ucp_proto_get_bcopy_unpack_ctx_t),
                                      "get_bcopy_unpack_ctx");
        if (req->send.buffer == NULL) {
            ucp_proto_request_abort(req, UCS_ERR_NO_MEMORY);
            return UCS_OK;
        }

        ucp_proto_completion_init(&req->send.state.uct_comp,
                                  ucp_proto_get_offload_bcopy_completion);
        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_progress(req, req->send.proto_config->priv,
                                    ucp_proto_get_offload_bcopy_send_func,
                                    ucp_request_invoke_uct_completion_success,
                                    UCS_BIT(UCP_DATATYPE_CONTIG));
}

static void
ucp_proto_get_offload_bcopy_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 0,
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.get.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_GET_BCOPY,
        .super.memtype_op    = UCT_EP_OP_PUT_SHORT,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_RESPONSE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_mem_info_unknown,
        .max_lanes           = UCP_PROTO_RMA_MAX_BCOPY_LANES,
        .min_chunk           = context->config.ext.min_rma_chunk_size,
        .initial_reg_md_map  = 0,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_GET_BCOPY,
        .first.lane_type     = UCP_LANE_TYPE_RMA_BW,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_GET_BCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_RMA_BW,
        .opt_align_offs      = UCP_PROTO_COMMON_OFFSET_INVALID
    };

    if ((init_params->select_param->dt_class != UCP_DATATYPE_CONTIG) ||
        !ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_GET))) {
        return;
    }

    ucp_proto_multi_probe(&params);
}

static void ucp_proto_get_offload_reset(ucp_request_t *req)
{
    /* get_am protocol will use this field after reset, so it must be
     * initialized */
    req->send.state.completed_size = 0;
}

static void ucp_proto_get_offload_bcopy_abort(ucp_request_t *req,
                                              ucs_status_t status)
{
    if (req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED) {
        ucs_free(req->send.buffer);
        req->send.buffer = NULL;
    }

    ucp_proto_request_bcopy_abort(req, status);
}

static ucs_status_t ucp_proto_get_offload_bcopy_reset(ucp_request_t *req)
{
    if (req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED) {
        ucs_free(req->send.buffer);
        req->send.buffer = NULL;
    }

    ucp_proto_get_offload_reset(req);
    return ucp_proto_request_bcopy_reset(req);
}

ucp_proto_t ucp_get_offload_bcopy_proto = {
    .name     = "get/bcopy",
    .desc     = UCP_PROTO_COPY_OUT_DESC,
    .flags    = 0,
    .probe    = ucp_proto_get_offload_bcopy_probe,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_get_offload_bcopy_progress},
    .abort    = ucp_proto_get_offload_bcopy_abort,
    .reset    = ucp_proto_get_offload_bcopy_reset
};

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_get_offload_zcopy_send_func(ucp_request_t *req,
                                      const ucp_proto_multi_lane_priv_t *lpriv,
                                      ucp_datatype_iter_t *next_iter,
                                      ucp_lane_index_t *lane_shift)
{
    uct_rkey_t tl_rkey = ucp_rkey_get_tl_rkey(req->send.rma.rkey,
                                              lpriv->super.rkey_index);
    size_t offset      = req->send.state.dt_iter.offset;
    const ucp_proto_multi_priv_t *mpriv;
    uct_iov_t iov;

    ucp_datatype_iter_next_iov(&req->send.state.dt_iter,
                               ucp_proto_multi_max_payload(req, lpriv, 0),
                               lpriv->super.md_index, UCP_DT_MASK_CONTIG_IOV,
                               next_iter, &iov, 1);

    mpriv = req->send.proto_config->priv;
    ucp_proto_common_zcopy_adjust_min_frag(req, mpriv->min_frag, iov.length,
                                           &iov, 1, &offset);
    return uct_ep_get_zcopy(ucp_ep_get_lane(req->send.ep, lpriv->super.lane),
                            &iov, 1, req->send.rma.remote_addr + offset,
                            tl_rkey, &req->send.state.uct_comp);
}

static ucs_status_t ucp_proto_get_offload_zcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_zcopy_progress(
            req, req->send.proto_config->priv, ucp_proto_multi_rma_init_func,
            UCT_MD_MEM_ACCESS_LOCAL_WRITE, UCP_DT_MASK_CONTIG_IOV,
            ucp_proto_get_offload_zcopy_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_proto_request_zcopy_completion);
}

static void
ucp_proto_get_offload_zcopy_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 0,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = 30,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 1,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.get.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.get.max_zcopy),
        .super.max_iov_offs  = ucs_offsetof(uct_iface_attr_t, cap.get.max_iov),
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_GET_ZCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_RESPONSE |
                               UCP_PROTO_COMMON_INIT_FLAG_MIN_FRAG |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING |
                               UCP_PROTO_COMMON_INIT_FLAG_FAILOVER,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_proto_common_select_param_mem_info(
                                                     init_params->select_param),
        .max_lanes           = context->config.ext.max_rma_lanes,
        .min_chunk           = context->config.ext.min_rma_chunk_size,
        .initial_reg_md_map  = 0,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_GET_ZCOPY,
        .first.lane_type     = UCP_LANE_TYPE_RMA_BW,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_GET_ZCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_RMA_BW,
        .opt_align_offs      = UCP_PROTO_COMMON_OFFSET_INVALID
    };

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_GET))) {
        return;
    }

    ucp_proto_multi_probe(&params);
}

static ucs_status_t ucp_proto_get_offload_zcopy_reset(ucp_request_t *req)
{
    ucp_proto_get_offload_reset(req);
    return ucp_proto_offload_zcopy_reset(req);
}

ucp_proto_t ucp_get_offload_zcopy_proto = {
    .name     = "get/zcopy",
    .desc     = UCP_PROTO_ZCOPY_DESC,
    .flags    = 0,
    .probe    = ucp_proto_get_offload_zcopy_probe,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_get_offload_zcopy_progress},
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_get_offload_zcopy_reset
};
