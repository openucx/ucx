/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rma.inl"

#include <ucp/core/ucp_request.inl>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto_multi.inl>


static void ucp_proto_get_offload_bcopy_unpack(void *arg, const void *data,
                                               size_t length)
{
    void *dest = arg;
    ucs_memcpy_relaxed(dest, data, length);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_get_offload_bcopy_send_func(ucp_request_t *req,
                                      const ucp_proto_multi_lane_priv_t *lpriv,
                                      ucp_datatype_iter_t *next_iter)
{
    uct_rkey_t tl_rkey = ucp_rma_request_get_tl_rkey(req,
                                                     lpriv->super.rkey_index);
    size_t max_length, length;
    void *dest;

    max_length = ucp_proto_multi_max_payload(req, lpriv, 0);
    length     = ucp_datatype_iter_next_ptr(&req->send.dt_iter, max_length,
                                            next_iter, &dest);
    return uct_ep_get_bcopy(req->send.ep->uct_eps[lpriv->super.lane],
                            ucp_proto_get_offload_bcopy_unpack, dest, length,
                            req->send.rma.remote_addr + req->send.dt_iter.offset,
                            tl_rkey, &req->send.state.uct_comp);
}

static void ucp_proto_get_offload_bcopy_completion(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);
    ucp_proto_request_bcopy_complete(req, req->send.state.uct_comp.status);
}

static ucs_status_t ucp_proto_get_offload_bcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        ucp_proto_multi_request_init(req);
        ucp_proto_request_completion_init(req,
                                          ucp_proto_get_offload_bcopy_completion);
        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    return ucp_proto_multi_progress(req, ucp_proto_get_offload_bcopy_send_func,
                                    ucp_request_invoke_uct_completion,
                                    UCS_BIT(UCP_DATATYPE_CONTIG));
}

static ucs_status_t
ucp_proto_get_offload_bcopy_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 0,
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.get.max_bcopy),
        .super.hdr_size      = 0,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_RESPONSE,
        .max_lanes           = 1,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_GET_BCOPY,
        .first.lane_type     = UCP_LANE_TYPE_RMA,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_GET_BCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_RMA,
    };

    UCP_RMA_PROTO_INIT_CHECK(init_params, UCP_OP_ID_GET);

    return ucp_proto_multi_init(&params);
}

static ucp_proto_t ucp_get_offload_bcopy_proto = {
    .name       = "get/bcopy",
    .flags      = 0,
    .init       = ucp_proto_get_offload_bcopy_init,
    .config_str = ucp_proto_multi_config_str,
    .progress   = ucp_proto_get_offload_bcopy_progress
};
UCP_PROTO_REGISTER(&ucp_get_offload_bcopy_proto);

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_get_offload_zcopy_send_func(ucp_request_t *req,
                                      const ucp_proto_multi_lane_priv_t *lpriv,
                                      ucp_datatype_iter_t *next_iter)
{
    uct_rkey_t tl_rkey = ucp_rma_request_get_tl_rkey(req,
                                                     lpriv->super.rkey_index);
    uct_iov_t iov;

    ucp_datatype_iter_next_iov(&req->send.dt_iter, lpriv->super.memh_index,
                               ucp_proto_multi_max_payload(req, lpriv, 0),
                               next_iter, &iov);
    return uct_ep_get_zcopy(req->send.ep->uct_eps[lpriv->super.lane], &iov, 1,
                            req->send.rma.remote_addr + req->send.dt_iter.offset,
                            tl_rkey, &req->send.state.uct_comp);
}

static ucs_status_t ucp_proto_get_offload_zcopy_progress(uct_pending_req_t *self)
{
    return ucp_proto_multi_zcopy_progress(self, NULL,
                                          ucp_proto_get_offload_zcopy_send_func,
                                          ucp_proto_request_zcopy_completion);
}

static ucs_status_t
ucp_proto_get_offload_zcopy_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 0,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = 30,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.get.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.get.max_zcopy),
        .super.hdr_size      = 0,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_RESPONSE,
        .max_lanes           = 1,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_GET_ZCOPY,
        .first.lane_type     = UCP_LANE_TYPE_RMA,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_GET_ZCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_RMA
    };

    UCP_RMA_PROTO_INIT_CHECK(init_params, UCP_OP_ID_GET);

    return ucp_proto_multi_init(&params);
}

static ucp_proto_t ucp_get_offload_zcopy_proto = {
    .name       = "get/zcopy",
    .flags      = 0,
    .init       = ucp_proto_get_offload_zcopy_init,
    .config_str = ucp_proto_multi_config_str,
    .progress   = ucp_proto_get_offload_zcopy_progress
};
UCP_PROTO_REGISTER(&ucp_get_offload_zcopy_proto);
