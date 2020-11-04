/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
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

    tl_rkey = ucp_rma_request_get_tl_rkey(req, spriv->super.rkey_index);
    status  = uct_ep_put_short(ep->uct_eps[spriv->super.lane],
                               req->send.dt_iter.type.contig.buffer,
                               req->send.dt_iter.length,
                               req->send.rma.remote_addr, tl_rkey);
    if (ucs_unlikely(status == UCS_ERR_NO_RESOURCE)) {
        req->send.lane = spriv->super.lane; /* for pending add */
        return status;
    }

    /* UCS_INPROGRESS is not expected */
    ucs_assert((status == UCS_OK) || UCS_STATUS_IS_ERR(status));

    ucp_datatype_iter_cleanup(&req->send.dt_iter, UCS_BIT(UCP_DATATYPE_CONTIG));
    ucp_request_complete_send(req, status);
    return UCS_OK;
}

static ucs_status_t
ucp_proto_put_offload_short_init(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = -150e-9,
        .super.overhead      = 0,
        .super.cfg_thresh    = UCS_MEMUNITS_AUTO,
        .super.cfg_priority  = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.put.max_short),
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS,
        .lane_type           = UCP_LANE_TYPE_RMA,
        .tl_cap_flags        = UCT_IFACE_FLAG_PUT_SHORT,
        .super.hdr_size      = 0
    };

    UCP_RMA_PROTO_INIT_CHECK(init_params, UCP_OP_ID_PUT);

    return ucp_proto_single_init(&params);
}

static ucp_proto_t ucp_put_offload_short_proto = {
    .name       = "put/offload/short",
    .flags      = UCP_PROTO_FLAG_PUT_SHORT,
    .init       = ucp_proto_put_offload_short_init,
    .config_str = ucp_proto_single_config_str,
    .progress   = ucp_proto_put_offload_short_progress
};
UCP_PROTO_REGISTER(&ucp_put_offload_short_proto);

static size_t ucp_proto_put_offload_bcopy_pack(void *dest, void *arg)
{
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;

    return ucp_proto_multi_data_pack(pack_ctx, dest);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_put_offload_bcopy_send_func(ucp_request_t *req,
                                      const ucp_proto_multi_lane_priv_t *lpriv,
                                      ucp_datatype_iter_t *next_iter)
{
    ucp_ep_t *ep                        = req->send.ep;
    ucp_proto_multi_pack_ctx_t pack_ctx = {
        .req         = req,
        .max_payload = ucp_proto_multi_max_payload(req, lpriv, 0),
        .next_iter   = next_iter
    };
    ssize_t packed_size;
    uct_rkey_t tl_rkey;

    tl_rkey     = ucp_rma_request_get_tl_rkey(req, lpriv->super.rkey_index);
    packed_size = uct_ep_put_bcopy(ep->uct_eps[lpriv->super.lane],
                                   ucp_proto_put_offload_bcopy_pack, &pack_ctx,
                                   req->send.rma.remote_addr + req->send.dt_iter.offset,
                                   tl_rkey);
    if (ucs_likely(packed_size >= 0)) {
        return UCS_OK;
    } else {
        return (ucs_status_t)packed_size;
    }
}

static ucs_status_t ucp_proto_put_offload_bcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        ucp_proto_multi_request_init(req);
        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    return ucp_proto_multi_progress(req, ucp_proto_put_offload_bcopy_send_func,
                                    ucp_proto_request_bcopy_complete,
                                    UCS_BIT(UCP_DATATYPE_CONTIG));
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
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.put.max_bcopy),
        .super.hdr_size      = 0,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS,
        .max_lanes           = 1,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_PUT_BCOPY,
        .first.lane_type     = UCP_LANE_TYPE_RMA,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_PUT_BCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_RMA,
    };

    UCP_RMA_PROTO_INIT_CHECK(init_params, UCP_OP_ID_PUT);

    return ucp_proto_multi_init(&params);
}

static ucp_proto_t ucp_put_offload_bcopy_proto = {
    .name       = "put/offload/bcopy",
    .flags      = 0,
    .init       = ucp_proto_put_offload_bcopy_init,
    .config_str = ucp_proto_multi_config_str,
    .progress   = ucp_proto_put_offload_bcopy_progress
};
UCP_PROTO_REGISTER(&ucp_put_offload_bcopy_proto);

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_put_offload_zcopy_send_func(ucp_request_t *req,
                                      const ucp_proto_multi_lane_priv_t *lpriv,
                                      ucp_datatype_iter_t *next_iter)
{
    uct_rkey_t tl_rkey = ucp_rma_request_get_tl_rkey(req, lpriv->super.rkey_index);
    uct_iov_t iov;

    ucp_datatype_iter_next_iov(&req->send.dt_iter, lpriv->super.memh_index,
                               ucp_proto_multi_max_payload(req, lpriv, 0),
                               next_iter, &iov);
    return uct_ep_put_zcopy(req->send.ep->uct_eps[lpriv->super.lane], &iov, 1,
                            req->send.rma.remote_addr + req->send.dt_iter.offset,
                            tl_rkey, &req->send.state.uct_comp);
}

static ucs_status_t ucp_proto_put_offload_zcopy_progress(uct_pending_req_t *self)
{
    return ucp_proto_multi_zcopy_progress(self, NULL,
                                          ucp_proto_put_offload_zcopy_send_func,
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
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.put.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.put.max_zcopy),
        .super.hdr_size      = 0,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS,
        .max_lanes           = 1,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_PUT_ZCOPY,
        .first.lane_type     = UCP_LANE_TYPE_RMA,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_PUT_ZCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_RMA,
    };

    UCP_RMA_PROTO_INIT_CHECK(init_params, UCP_OP_ID_PUT);

    return ucp_proto_multi_init(&params);
}

static ucp_proto_t ucp_put_offload_zcopy_proto = {
    .name       = "put/offload/zcopy",
    .flags      = 0,
    .init       = ucp_proto_put_offload_zcopy_init,
    .config_str = ucp_proto_multi_config_str,
    .progress   = ucp_proto_put_offload_zcopy_progress
};
UCP_PROTO_REGISTER(&ucp_put_offload_zcopy_proto);
