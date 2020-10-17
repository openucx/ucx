/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
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
#include <ucp/proto/proto_single.inl>


static size_t ucp_proto_get_am_bcopy_pack(void *dest, void *arg)
{
    ucp_request_t *req         = arg;
    ucp_get_req_hdr_t *getreqh = dest;

    getreqh->address    = req->send.rma.remote_addr;
    getreqh->length     = req->send.dt_iter.length;
    getreqh->req.ep_id  = ucp_send_request_get_ep_remote_id(req);
    getreqh->req.req_id = ucp_send_request_get_id(req);
    getreqh->mem_type   = req->send.rma.rkey->mem_type;

    return sizeof(*getreqh);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_get_am_bcopy_complete(ucp_request_t *req, ucs_status_t status)
{
    ucs_assert(status == UCS_OK);
    ucp_ep_rma_remote_request_sent(req->send.ep);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_get_am_bcopy_error(ucp_request_t *req, ucs_status_t status)
{
    ucp_worker_flush_ops_count_dec(req->send.ep->worker);
    ucp_request_complete_send(req, status);
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
        req->send.buffer = req->send.dt_iter.type.contig.buffer;
        req->send.length = req->send.dt_iter.length;

        req->flags      |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    ucp_worker_flush_ops_count_inc(worker);
    status = ucp_proto_am_bcopy_single_progress(req, UCP_AM_ID_GET_REQ,
                                                spriv->super.lane,
                                                ucp_proto_get_am_bcopy_pack,
                                                req, sizeof(ucp_get_req_hdr_t),
                                                ucp_proto_get_am_bcopy_complete,
                                                ucp_proto_get_am_bcopy_error);
    if (status != UCS_OK) {
        ucp_worker_flush_ops_count_dec(worker);
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
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.hdr_size      = sizeof(ucp_get_req_hdr_t),
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RESPONSE |
                               UCP_PROTO_COMMON_INIT_FLAG_MEM_TYPE,
        .lane_type           = UCP_LANE_TYPE_AM,
        .tl_cap_flags        = UCT_IFACE_FLAG_AM_BCOPY
    };

    UCP_RMA_PROTO_INIT_CHECK(init_params, UCP_OP_ID_GET);

    return ucp_proto_single_init(&params);
}

static ucp_proto_t ucp_get_am_bcopy_proto = {
    .name       = "get/am/bcopy",
    .flags      = 0,
    .init       = ucp_proto_get_am_bcopy_init,
    .config_str = ucp_proto_single_config_str,
    .progress   = ucp_proto_get_am_bcopy_progress
};
UCP_PROTO_REGISTER(&ucp_get_am_bcopy_proto);
