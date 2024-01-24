/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rma.inl"

#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto_init.h>
#include <ucp/proto/proto_single.inl>


static size_t ucp_proto_get_am_bcopy_pack(void *dest, void *arg)
{
    ucp_request_t *req         = arg;
    ucp_get_req_hdr_t *getreqh = dest;

    getreqh->address    = req->send.rma.remote_addr;
    getreqh->length     = req->send.state.dt_iter.length;
    getreqh->req.ep_id  = ucp_send_request_get_ep_remote_id(req);
    getreqh->req.req_id = ucp_send_request_get_id(req);
    getreqh->mem_type   = req->send.rma.rkey->mem_type;

    return sizeof(*getreqh);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_get_am_bcopy_complete(ucp_request_t *req)
{
    ucp_ep_rma_remote_request_sent(req->send.ep);
    return UCS_OK;
}

static ucs_status_t ucp_proto_get_am_bcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t                   *req = ucs_container_of(self, ucp_request_t,
                                                            send.uct);
    ucp_worker_h                  worker = req->send.ep->worker;
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    ucs_status_t status;

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        status = ucp_ep_resolve_remote_id(req->send.ep, spriv->super.lane);
        if (status != UCS_OK) {
            return status;
        }

        /* initialize some request fields, for compatibility of get_reply
         * processing */
        req->send.buffer = req->send.state.dt_iter.type.contig.buffer;
        req->send.length = req->send.state.dt_iter.length;
        req->flags      |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
        ucp_send_request_id_alloc(req);
    }

    ucp_worker_flush_ops_count_add(worker, +1);
    status = ucp_proto_am_bcopy_single_progress(
            req, UCP_AM_ID_GET_REQ, spriv->super.lane,
            ucp_proto_get_am_bcopy_pack, req, sizeof(ucp_get_req_hdr_t),
            ucp_proto_get_am_bcopy_complete, 0);
    if (status != UCS_OK) {
        ucp_worker_flush_ops_count_add(worker, -1);
    }
    return status;
}

static ucs_status_t
ucp_proto_get_am_bcopy_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_h context                 = init_params->worker->context;
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 40e-9,
        .super.cfg_thresh    = ucp_proto_sw_rma_cfg_thresh(
                                   context, context->config.ext.bcopy_thresh),
        .super.cfg_priority  = 20,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = sizeof(ucp_get_req_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_BCOPY,
        .super.memtype_op    = UCT_EP_OP_PUT_SHORT,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RESPONSE     |
                               UCP_PROTO_COMMON_INIT_FLAG_CAP_SEG_SIZE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .lane_type           = UCP_LANE_TYPE_AM,
        .tl_cap_flags        = UCT_IFACE_FLAG_AM_BCOPY
    };

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_GET))) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_single_init(&params);
}

ucp_proto_t ucp_get_am_bcopy_proto = {
    .name     = "get/am/bcopy",
    .desc     = UCP_PROTO_RMA_EMULATION_DESC,
    .flags    = 0,
    .init     = ucp_proto_get_am_bcopy_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_proto_get_am_bcopy_progress},
    .abort    = ucp_proto_request_bcopy_id_abort,
    .reset    = ucp_proto_request_bcopy_id_reset
};
