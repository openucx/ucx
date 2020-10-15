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
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS,
        .super.overhead      = 10e-9,
        .super.latency       = 0,
        .max_lanes           = 1,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_PUT_BCOPY,
        .super.fragsz_offset = ucs_offsetof(uct_iface_attr_t, cap.put.max_bcopy),
        .first.lane_type     = UCP_LANE_TYPE_RMA,
        .super.hdr_size      = 0,
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
