/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "eager.h"

#include <ucp/core/ucp_request.inl>
#include <ucp/proto/proto_multi.inl>


static UCS_F_ALWAYS_INLINE
void ucp_eager_multi_proto_request_init(ucp_request_t *req)
{
    req->send.msg_proto.message_id = req->send.ep->worker->am_message_id++;
}

static UCS_F_ALWAYS_INLINE void
ucp_eager_proto_set_first_hdr(ucp_request_t *req, ucp_eager_first_hdr_t *hdr)
{
    hdr->super.super.tag = req->send.msg_proto.tag.tag;
    hdr->total_len       = req->send.state.dt_iter.length;
    hdr->msg_id          = req->send.msg_proto.message_id;
}

static UCS_F_ALWAYS_INLINE void
ucp_eager_proto_set_middle_hdr(ucp_request_t *req, ucp_eager_middle_hdr_t *hdr)
{
    hdr->msg_id = req->send.msg_proto.message_id;
    hdr->offset = req->send.state.dt_iter.offset;
}

static ucs_status_t
ucp_proto_eager_multi_init_common(ucp_proto_multi_init_params_t *params)
{
    if (params->super.super.select_param->op_id != UCP_OP_ID_TAG_SEND) {
        return UCS_ERR_UNSUPPORTED;
    }

    params->super.overhead   = 10e-9; /* for multiple lanes management */
    params->super.latency    = 0;
    params->first.lane_type  = UCP_LANE_TYPE_AM;
    params->super.hdr_size   = sizeof(ucp_eager_first_hdr_t);
    params->middle.lane_type = UCP_LANE_TYPE_AM_BW;
    params->max_lanes        =
            params->super.super.worker->context->config.ext.max_eager_lanes;

    return ucp_proto_multi_init(params);
}

static ucs_status_t
ucp_proto_eager_bcopy_multi_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.flags         = 0,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_BCOPY,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_BCOPY,
    };

    return ucp_proto_eager_multi_init_common(&params);
}

static size_t ucp_eager_bcopy_pack_first(void *dest, void *arg)
{
    ucp_eager_first_hdr_t           *hdr = dest;
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;

    ucp_eager_proto_set_first_hdr(pack_ctx->req, hdr);
    return sizeof(*hdr) + ucp_proto_multi_data_pack(pack_ctx, hdr + 1);
}

static size_t ucp_eager_bcopy_pack_middle(void *dest, void *arg)
{
    ucp_eager_middle_hdr_t          *hdr = dest;
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;

    ucp_eager_proto_set_middle_hdr(pack_ctx->req, hdr);
    return sizeof(*hdr) + ucp_proto_multi_data_pack(pack_ctx, hdr + 1);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_eager_bcopy_multi_send_func(ucp_request_t *req,
                                const ucp_proto_multi_lane_priv_t *lpriv,
                                ucp_datatype_iter_t *next_iter)
{
    ucp_ep_t *ep                        = req->send.ep;
    ucp_proto_multi_pack_ctx_t pack_ctx = {
        .req       = req,
        .next_iter = next_iter
    };
    uct_pack_callback_t pack_cb;
    ssize_t packed_size;
    ucp_am_id_t am_id;
    size_t hdr_size;

    if (req->send.state.dt_iter.offset == 0) {
        am_id    = UCP_AM_ID_EAGER_FIRST;
        pack_cb  = ucp_eager_bcopy_pack_first;
        hdr_size = sizeof(ucp_eager_first_hdr_t);
    } else {
        am_id   = UCP_AM_ID_EAGER_MIDDLE;
        pack_cb = ucp_eager_bcopy_pack_middle;
        hdr_size = sizeof(ucp_eager_middle_hdr_t);
    }
    pack_ctx.max_payload = ucp_proto_multi_max_payload(req, lpriv, hdr_size);

    packed_size = uct_ep_am_bcopy(ep->uct_eps[lpriv->super.lane], am_id,
                                  pack_cb, &pack_ctx, 0);
    if (ucs_likely(packed_size >= 0)) {
        ucs_assert(packed_size >= hdr_size);
        return UCS_OK;
    } else {
        return (ucs_status_t)packed_size;
    }
}

static ucs_status_t
ucp_eager_bcopy_multi_proto_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        ucp_proto_multi_request_init(req);
        ucp_eager_multi_proto_request_init(req);
        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    return ucp_proto_multi_progress(req, ucp_eager_bcopy_multi_send_func,
                                    ucp_proto_request_bcopy_complete, UINT_MAX);
}

static ucp_proto_t ucp_eager_bcopy_multi_proto = {
    .name       = "egr/multi/bcopy",
    .flags      = 0,
    .init       = ucp_proto_eager_bcopy_multi_init,
    .config_str = ucp_proto_multi_config_str,
    .progress   = ucp_eager_bcopy_multi_proto_progress
};
UCP_PROTO_REGISTER(&ucp_eager_bcopy_multi_proto);

static ucs_status_t
ucp_proto_eager_zcopy_multi_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = 30,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_zcopy),
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_ZCOPY,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_ZCOPY,
    };

    return ucp_proto_eager_multi_init_common(&params);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_eager_zcopy_multi_send_func(ucp_request_t *req,
                                      const ucp_proto_multi_lane_priv_t *lpriv,
                                      ucp_datatype_iter_t *next_iter)
{
    union {
        ucp_eager_first_hdr_t  first;
        ucp_eager_middle_hdr_t middle;
    } hdr;
    ucp_am_id_t am_id;
    size_t hdr_size;
    uct_iov_t iov;

    if (req->send.state.dt_iter.offset == 0) {
        am_id    = UCP_AM_ID_EAGER_FIRST;
        hdr_size = sizeof(hdr.first);
        ucp_eager_proto_set_first_hdr(req, &hdr.first);
    } else {
        am_id    = UCP_AM_ID_EAGER_MIDDLE;
        hdr_size = sizeof(hdr.middle);
        ucp_eager_proto_set_middle_hdr(req, &hdr.middle);
    }

    ucp_datatype_iter_next_iov(&req->send.state.dt_iter, lpriv->super.memh_index,
                               ucp_proto_multi_max_payload(req, lpriv, hdr_size),
                               next_iter, &iov);
    return uct_ep_am_zcopy(req->send.ep->uct_eps[lpriv->super.lane], am_id, &hdr,
                           hdr_size, &iov, 1, 0, &req->send.state.uct_comp);
}

static ucs_status_t ucp_proto_eager_zcopy_multi_progress(uct_pending_req_t *self)
{
    return ucp_proto_multi_zcopy_progress(self,
                                          ucp_eager_multi_proto_request_init,
                                          ucp_proto_eager_zcopy_multi_send_func,
                                          ucp_proto_request_zcopy_completion);
}

static ucp_proto_t ucp_eager_zcopy_multi_proto = {
    .name       = "egr/multi/zcopy",
    .flags      = 0,
    .init       = ucp_proto_eager_zcopy_multi_init,
    .config_str = ucp_proto_multi_config_str,
    .progress   = ucp_proto_eager_zcopy_multi_progress
};
UCP_PROTO_REGISTER(&ucp_eager_zcopy_multi_proto);
