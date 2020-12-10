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


static size_t ucp_proto_put_am_bcopy_pack(void *dest, void *arg)
{
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;
    ucp_request_t                   *req = pack_ctx->req;
    ucp_put_hdr_t                  *puth = dest;

    puth->address  = req->send.rma.remote_addr +
                     req->send.state.dt_iter.offset;
    puth->ep_id    = ucp_send_request_get_ep_remote_id(req);
    puth->mem_type = req->send.rma.rkey->mem_type;

    return sizeof(*puth) + ucp_proto_multi_data_pack(pack_ctx, puth + 1);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_put_am_bcopy_send_func(ucp_request_t *req,
                                 const ucp_proto_multi_lane_priv_t *lpriv,
                                 ucp_datatype_iter_t *next_iter)
{
    ucp_proto_multi_pack_ctx_t pack_ctx = {
        .req         = req,
        .max_payload = ucp_proto_multi_max_payload(req, lpriv,
                                                   sizeof(ucp_put_hdr_t)),
        .next_iter   = next_iter
    };

    return ucp_rma_sw_do_am_bcopy(req, UCP_AM_ID_PUT, lpriv->super.lane,
                                  ucp_proto_put_am_bcopy_pack, &pack_ctx, NULL);
}

static ucs_status_t ucp_proto_put_am_bcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req                  = ucs_container_of(self, ucp_request_t,
                                                           send.uct);
    const ucp_proto_multi_priv_t *mpriv = req->send.proto_config->priv;
    ucs_status_t status;

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        status = ucp_ep_resolve_remote_id(req->send.ep,
                                          mpriv->lanes[0].super.lane);
        if (status != UCS_OK) {
            return status;
        }

        ucp_proto_multi_request_init(req);
        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    return ucp_proto_multi_progress(req, ucp_proto_put_am_bcopy_send_func,
                                    ucp_proto_request_bcopy_complete,
                                    UCS_BIT(UCP_DATATYPE_CONTIG));
}

static ucs_status_t
ucp_proto_put_am_bcopy_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_h context                = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 40e-9,
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.hdr_size      = sizeof(ucp_put_hdr_t),
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_MEM_TYPE,
        .max_lanes           = 1,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_BCOPY,
        .first.lane_type     = UCP_LANE_TYPE_AM,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_BCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_AM,
    };

    UCP_RMA_PROTO_INIT_CHECK(init_params, UCP_OP_ID_PUT);

    return ucp_proto_multi_init(&params);
}

static ucp_proto_t ucp_put_am_bcopy_proto = {
    .name       = "put/am/bcopy",
    .flags      = 0,
    .init       = ucp_proto_put_am_bcopy_init,
    .config_str = ucp_proto_multi_config_str,
    .progress   = ucp_proto_put_am_bcopy_progress
};
UCP_PROTO_REGISTER(&ucp_put_am_bcopy_proto);

