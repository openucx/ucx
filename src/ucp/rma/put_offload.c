/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rma.h"
#include "rma.inl"

#include <ucp/core/ucp_request.inl>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto_init.h>
#include <ucp/proto/proto_multi.inl>
#include <ucp/proto/proto_single.inl>


static ucs_status_t ucp_proto_put_offload_short_progress(uct_pending_req_t *self)
{
    ucp_request_t *req                   = ucs_container_of(self, ucp_request_t,
                                                            send.uct);
    ucp_ep_t *ep                         = req->send.ep;
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    ucs_status_t status;
    uct_rkey_t tl_rkey;

    tl_rkey = ucp_rkey_get_tl_rkey(req->send.rma.rkey, spriv->super.rkey_index);
    status  = uct_ep_put_short(ucp_ep_get_fast_lane(ep, spriv->super.lane),
                               req->send.state.dt_iter.type.contig.buffer,
                               req->send.state.dt_iter.length,
                               req->send.rma.remote_addr, tl_rkey);
    if (ucs_unlikely(status == UCS_ERR_NO_RESOURCE)) {
        req->send.lane = spriv->super.lane; /* for pending add */
        return status;
    }

    /* UCS_INPROGRESS is not expected */
    ucs_assert((status == UCS_OK) || UCS_STATUS_IS_ERR(status));

    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 0,
                              UCS_BIT(UCP_DATATYPE_CONTIG));
    ucp_request_complete_send(req, status);
    return UCS_OK;
}

static ucs_status_t
ucp_proto_put_offload_short_init(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 0,
        .super.cfg_thresh    = UCS_MEMUNITS_AUTO,
        .super.cfg_priority  = 0,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.put.max_short),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_PUT_SHORT,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY    |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG   |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .lane_type           = UCP_LANE_TYPE_RMA,
        .tl_cap_flags        = UCT_IFACE_FLAG_PUT_SHORT
    };

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_PUT)) ||
        !ucp_proto_is_short_supported(init_params->select_param)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_single_init(&params);
}

ucp_proto_t ucp_put_offload_short_proto = {
    .name     = "put/offload/short",
    .desc     = UCP_PROTO_SHORT_DESC,
    .flags    = UCP_PROTO_FLAG_PUT_SHORT,
    .init     = ucp_proto_put_offload_short_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_proto_put_offload_short_progress},
    .abort    = ucp_proto_request_bcopy_abort,
    .reset    = ucp_proto_request_bcopy_reset
};

static size_t ucp_proto_put_offload_bcopy_pack(void *dest, void *arg)
{
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;

    return ucp_proto_multi_data_pack(pack_ctx, dest);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_put_offload_bcopy_send_func(ucp_request_t *req,
                                      const ucp_proto_multi_lane_priv_t *lpriv,
                                      ucp_datatype_iter_t *next_iter,
                                      ucp_lane_index_t *lane_shift)
{
    ucp_ep_t *ep                        = req->send.ep;
    ucp_proto_multi_pack_ctx_t pack_ctx = {
        .req         = req,
        .max_payload = ucp_proto_multi_max_payload(req, lpriv, 0),
        .next_iter   = next_iter
    };
    ssize_t packed_size;
    uct_rkey_t tl_rkey;

    tl_rkey     = ucp_rkey_get_tl_rkey(req->send.rma.rkey,
                                       lpriv->super.rkey_index);
    packed_size = uct_ep_put_bcopy(ucp_ep_get_lane(ep, lpriv->super.lane),
                                   ucp_proto_put_offload_bcopy_pack, &pack_ctx,
                                   req->send.rma.remote_addr +
                                   req->send.state.dt_iter.offset,
                                   tl_rkey);

    return ucp_proto_bcopy_send_func_status(packed_size);
}

static ucs_status_t ucp_proto_put_offload_bcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        ucp_proto_multi_request_init(req);
        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_progress(req, req->send.proto_config->priv,
                                    ucp_proto_put_offload_bcopy_send_func,
                                    ucp_proto_request_bcopy_complete_success,
                                    UCP_DT_MASK_ALL);
}

static ucs_status_t
ucp_proto_put_offload_bcopy_init(const ucp_proto_init_params_t *init_params)
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
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                           cap.put.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_PUT_BCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY    |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .max_lanes           = UCP_PROTO_RMA_MAX_BCOPY_LANES,
        .initial_reg_md_map  = 0,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_PUT_BCOPY,
        .first.lane_type     = UCP_LANE_TYPE_RMA_BW,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_PUT_BCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_RMA_BW,
        .opt_align_offs      = UCP_PROTO_COMMON_OFFSET_INVALID
    };

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_PUT))) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_multi_init(&params, init_params->priv,
                                init_params->priv_size);
}

ucp_proto_t ucp_put_offload_bcopy_proto = {
    .name     = "put/offload/bcopy",
    .desc     = UCP_PROTO_COPY_IN_DESC,
    .flags    = 0,
    .init     = ucp_proto_put_offload_bcopy_init,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_put_offload_bcopy_progress},
    .abort    = ucp_proto_request_bcopy_abort,
    .reset    = ucp_proto_request_bcopy_reset
};

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_put_offload_zcopy_send_func(ucp_request_t *req,
                                      const ucp_proto_multi_lane_priv_t *lpriv,
                                      ucp_datatype_iter_t *next_iter,
                                      ucp_lane_index_t *lane_shift)
{
    uct_rkey_t tl_rkey = ucp_rkey_get_tl_rkey(req->send.rma.rkey,
                                              lpriv->super.rkey_index);
    uct_iov_t iov;

    ucp_datatype_iter_next_iov(&req->send.state.dt_iter,
                               ucp_proto_multi_max_payload(req, lpriv, 0),
                               lpriv->super.md_index, UCP_DT_MASK_CONTIG_IOV,
                               next_iter, &iov, 1);
    return uct_ep_put_zcopy(ucp_ep_get_lane(req->send.ep, lpriv->super.lane),
                            &iov, 1,
                            req->send.rma.remote_addr +
                            req->send.state.dt_iter.offset,
                            tl_rkey, &req->send.state.uct_comp);
}

static ucs_status_t
ucp_proto_put_offload_zcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_zcopy_progress(
            req, req->send.proto_config->priv, NULL,
            UCT_MD_MEM_ACCESS_LOCAL_READ, UCP_DT_MASK_CONTIG_IOV,
            ucp_proto_put_offload_zcopy_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_proto_request_zcopy_completion);
}

static ucs_status_t
ucp_proto_put_offload_zcopy_init(const ucp_proto_init_params_t *init_params)
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
        .super.min_iov       = 1,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                           cap.put.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.put.max_zcopy),
        .super.max_iov_offs  = ucs_offsetof(uct_iface_attr_t, cap.put.max_iov),
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_PUT_ZCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY    |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY    |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .max_lanes           = context->config.ext.max_rma_lanes,
        .initial_reg_md_map  = 0,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_PUT_ZCOPY,
        .first.lane_type     = UCP_LANE_TYPE_RMA_BW,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_PUT_ZCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_RMA_BW,
        .opt_align_offs      = UCP_PROTO_COMMON_OFFSET_INVALID,
    };

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_PUT))) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_multi_init(&params, init_params->priv,
                                init_params->priv_size);
}

ucp_proto_t ucp_put_offload_zcopy_proto = {
    .name     = "put/offload/zcopy",
    .desc     = UCP_PROTO_ZCOPY_DESC,
    .flags    = 0,
    .init     = ucp_proto_put_offload_zcopy_init,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_put_offload_zcopy_progress},
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_request_zcopy_reset
};
