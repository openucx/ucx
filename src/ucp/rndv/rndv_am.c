/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "proto_rndv.inl"


static ucs_status_t
ucp_proto_rdnv_am_init_common(ucp_proto_multi_init_params_t *params)
{
    ucp_context_h context = params->super.super.worker->context;

    if (!ucp_proto_rndv_op_check(&params->super.super, UCP_OP_ID_RNDV_SEND,
                                 0)) {
        return UCS_ERR_UNSUPPORTED;
    }

    params->super.cfg_thresh =
            ucp_proto_rndv_cfg_thresh(context, UCS_BIT(UCP_RNDV_MODE_AM));
    params->super.overhead   = 10e-9; /* for multiple lanes management */
    params->super.latency    = 0;
    params->first.lane_type  = UCP_LANE_TYPE_AM;
    params->middle.lane_type = UCP_LANE_TYPE_AM_BW;
    params->super.hdr_size   = sizeof(ucp_request_data_hdr_t);
    params->max_lanes        = context->config.ext.max_rndv_lanes;

    return ucp_proto_multi_init(params, params->super.super.priv,
                                params->super.super.priv_size);
}

static size_t ucp_proto_rndv_am_bcopy_pack(void *dest, void *arg)
{
    ucp_request_data_hdr_t *hdr          = dest;
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;
    ucp_request_t *req                   = pack_ctx->req;

    hdr->req_id  = req->send.rndv.remote_req_id;
    hdr->offset  = req->send.state.dt_iter.offset;

    return sizeof(*hdr) + ucp_proto_multi_data_pack(pack_ctx, hdr + 1);
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_rndv_am_bcopy_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter)
{
    static const size_t hdr_size        = sizeof(ucp_request_data_hdr_t);
    ucp_ep_t *ep                        = req->send.ep;
    ucp_proto_multi_pack_ctx_t pack_ctx = {
        .req       = req,
        .next_iter = next_iter
    };
    ssize_t packed_size;

    pack_ctx.max_payload = ucp_proto_multi_max_payload(req, lpriv, hdr_size);

    packed_size = uct_ep_am_bcopy(ep->uct_eps[lpriv->super.lane],
                                  UCP_AM_ID_RNDV_DATA,
                                  ucp_proto_rndv_am_bcopy_pack, &pack_ctx, 0);
    if (ucs_unlikely(packed_size < 0)) {
        return (ucs_status_t)packed_size;
    }

    ucs_assert(packed_size >= hdr_size);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_am_bcopy_complete(ucp_request_t *req)
{
    if (req->send.rndv.rkey != NULL) {
        ucp_proto_rndv_rkey_destroy(req);
    }
    ucp_datatype_iter_mem_dereg(req->send.ep->worker->context,
                                &req->send.state.dt_iter,
                                UCS_BIT(UCP_DATATYPE_CONTIG));
    return ucp_proto_request_bcopy_complete_success(req);
}

static ucs_status_t ucp_proto_rndv_am_bcopy_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_bcopy_progress(req, req->send.proto_config->priv,
                                          NULL,
                                          ucp_proto_rndv_am_bcopy_send_func,
                                          ucp_proto_rndv_am_bcopy_complete);
}

static ucs_status_t
ucp_proto_rdnv_am_bcopy_init(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = UCS_MEMUNITS_AUTO,
        .super.cfg_priority  = 0,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.send_op       = UCT_EP_OP_AM_BCOPY,
        .super.memtype_op    = UCT_EP_OP_GET_SHORT,
        .super.flags         = 0,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_BCOPY,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_BCOPY,
    };

    return ucp_proto_rdnv_am_init_common(&params);
}

ucp_proto_t ucp_rndv_am_bcopy_proto = {
    .name     = "rndv/am/bcopy",
    .desc     = "fragmented " UCP_PROTO_COPY_IN_DESC " " UCP_PROTO_COPY_OUT_DESC,
    .flags    = 0,
    .init     = ucp_proto_rdnv_am_bcopy_init,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_rndv_am_bcopy_progress},
    .abort    = (ucp_request_abort_func_t)ucs_empty_function_do_assert_void
};
