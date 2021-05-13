/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "eager.h"

#include <ucp/core/ucp_request.inl>
#include <ucp/proto/proto_multi.inl>


static UCS_F_ALWAYS_INLINE void
ucp_proto_eager_set_first_hdr(ucp_request_t *req, ucp_eager_first_hdr_t *hdr)
{
    hdr->super.super.tag = req->send.msg_proto.tag;
    hdr->total_len       = req->send.state.dt_iter.length;
    hdr->msg_id          = req->send.msg_proto.message_id;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_eager_set_middle_hdr(ucp_request_t *req, ucp_eager_middle_hdr_t *hdr)
{
    hdr->msg_id = req->send.msg_proto.message_id;
    hdr->offset = req->send.state.dt_iter.offset;
}

static ucs_status_t
ucp_proto_eager_multi_init_common(ucp_proto_multi_init_params_t *params,
                                  ucp_proto_id_t op_id)
{
    /* TODO: Disable AM based protocols if tag lane is present! It can be done
     * when tag offload rndv is implemented (so any msg size can be sent with
     * tag offload). I. e. would need to check one more condition below:
     * ucp_ep_config_key_has_tag_lane(params->super.super.ep_config_key)
     */
    if (params->super.super.select_param->op_id != op_id) {
        return UCS_ERR_UNSUPPORTED;
    }

    params->super.overhead   = 10e-9; /* for multiple lanes management */
    params->super.latency    = 0;
    params->first.lane_type  = UCP_LANE_TYPE_AM;
    params->middle.lane_type = UCP_LANE_TYPE_AM_BW;
    params->max_lanes        =
            params->super.super.worker->context->config.ext.max_eager_lanes;

    return ucp_proto_multi_init(params);
}

static ucs_status_t ucp_proto_eager_bcopy_multi_common_init(
        const ucp_proto_init_params_t *init_params, ucp_proto_id_t op_id,
        size_t hdr_size)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.hdr_size      = hdr_size,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_MEM_TYPE,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_BCOPY,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_BCOPY,
    };

    return ucp_proto_eager_multi_init_common(&params, op_id);
}

static size_t ucp_proto_eager_bcopy_pack_first(void *dest, void *arg)
{
    ucp_eager_first_hdr_t           *hdr = dest;
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;

    ucp_proto_eager_set_first_hdr(pack_ctx->req, hdr);
    return sizeof(*hdr) + ucp_proto_multi_data_pack(pack_ctx, hdr + 1);
}

static size_t ucp_proto_eager_bcopy_pack_middle(void *dest, void *arg)
{
    ucp_eager_middle_hdr_t          *hdr = dest;
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;

    ucp_proto_eager_set_middle_hdr(pack_ctx->req, hdr);
    return sizeof(*hdr) + ucp_proto_multi_data_pack(pack_ctx, hdr + 1);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_eager_bcopy_multi_common_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter, ucp_am_id_t am_id_first,
        uct_pack_callback_t pack_cb_first, size_t hdr_size_first)
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
        am_id    = am_id_first;
        pack_cb  = pack_cb_first;
        hdr_size = hdr_size_first;
    } else {
        am_id    = UCP_AM_ID_EAGER_MIDDLE;
        pack_cb  = ucp_proto_eager_bcopy_pack_middle;
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
ucp_proto_eager_bcopy_multi_init(const ucp_proto_init_params_t *init_params)
{
    return ucp_proto_eager_bcopy_multi_common_init(
            init_params, UCP_OP_ID_TAG_SEND, sizeof(ucp_eager_first_hdr_t));
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_eager_bcopy_multi_send_func(ucp_request_t *req,
                                      const ucp_proto_multi_lane_priv_t *lpriv,
                                      ucp_datatype_iter_t *next_iter)
{
    return ucp_proto_eager_bcopy_multi_common_send_func(
            req, lpriv, next_iter, UCP_AM_ID_EAGER_FIRST,
            ucp_proto_eager_bcopy_pack_first, sizeof(ucp_eager_first_hdr_t));
}

static ucs_status_t
ucp_proto_eager_bcopy_multi_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    return ucp_proto_multi_bcopy_progress(
            req, req->send.proto_config->priv, ucp_proto_msg_multi_request_init,
            ucp_proto_eager_bcopy_multi_send_func,
            ucp_proto_request_bcopy_complete_success);
}

static ucp_proto_t ucp_eager_bcopy_multi_proto = {
    .name       = "egr/multi/bcopy",
    .flags      = 0,
    .init       = ucp_proto_eager_bcopy_multi_init,
    .config_str = ucp_proto_multi_config_str,
    .progress   = ucp_proto_eager_bcopy_multi_progress
};
UCP_PROTO_REGISTER(&ucp_eager_bcopy_multi_proto);

static ucs_status_t
ucp_proto_eager_sync_bcopy_multi_init(const ucp_proto_init_params_t *init_params)
{
    return ucp_proto_eager_bcopy_multi_common_init(
            init_params, UCP_OP_ID_TAG_SEND_SYNC,
            sizeof(ucp_eager_sync_first_hdr_t));
}

static size_t ucp_eager_sync_bcopy_pack_first(void *dest, void *arg)
{
    ucp_eager_sync_first_hdr_t *hdr      = dest;
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;
    ucp_request_t *req                   = pack_ctx->req;

    ucp_proto_eager_set_first_hdr(req, &hdr->super);
    hdr->req.ep_id  = ucp_send_request_get_ep_remote_id(req);
    hdr->req.req_id = ucp_send_request_get_id(req);

    return sizeof(*hdr) + ucp_proto_multi_data_pack(pack_ctx, hdr + 1);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_eager_sync_bcopy_multi_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter)
{
    return ucp_proto_eager_bcopy_multi_common_send_func(
            req, lpriv, next_iter, UCP_AM_ID_EAGER_SYNC_FIRST,
            ucp_eager_sync_bcopy_pack_first,
            sizeof(ucp_eager_sync_first_hdr_t));
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_eager_sync_bcopy_send_completed(ucp_request_t *req)
{
    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, UINT_MAX);

    req->flags |= UCP_REQUEST_FLAG_SYNC_LOCAL_COMPLETED;
    if (req->flags & UCP_REQUEST_FLAG_SYNC_REMOTE_COMPLETED) {
        ucp_request_complete_send(req, UCS_OK);
    }
    return UCS_OK;
}

void ucp_proto_eager_sync_ack_handler(ucp_worker_h worker,
                                      const ucp_reply_hdr_t *rep_hdr)
{
    ucp_request_t *req;

    UCP_SEND_REQUEST_GET_BY_ID(&req, worker, rep_hdr->req_id, 1, return,
                               "EAGER_S ACK %p", rep_hdr);

    req->flags |= UCP_REQUEST_FLAG_SYNC_REMOTE_COMPLETED;
    if (req->flags & UCP_REQUEST_FLAG_SYNC_LOCAL_COMPLETED) {
        ucp_request_complete_send(req, rep_hdr->status);
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_eager_sync_bcopy_request_init(ucp_request_t *req)
{
    ucp_proto_msg_multi_request_init(req);
    ucp_send_request_id_alloc(req);
}

static ucs_status_t
ucp_proto_eager_sync_bcopy_multi_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    return ucp_proto_multi_bcopy_progress(
            req, req->send.proto_config->priv,
            ucp_proto_eager_sync_bcopy_request_init,
            ucp_proto_eager_sync_bcopy_multi_send_func,
            ucp_proto_eager_sync_bcopy_send_completed);
}

static ucp_proto_t ucp_eager_sync_bcopy_multi_proto = {
    .name       = "egrsnc/multi/bcopy",
    .flags      = 0,
    .init       = ucp_proto_eager_sync_bcopy_multi_init,
    .config_str = ucp_proto_multi_config_str,
    .progress   = ucp_proto_eager_sync_bcopy_multi_progress
};
UCP_PROTO_REGISTER(&ucp_eager_sync_bcopy_multi_proto);

static ucs_status_t
ucp_proto_eager_zcopy_multi_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = 30,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_zcopy),
        .super.hdr_size      = sizeof(ucp_eager_first_hdr_t),
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_ZCOPY,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_ZCOPY,
    };

    return ucp_proto_eager_multi_init_common(&params, UCP_OP_ID_TAG_SEND);
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
        ucp_proto_eager_set_first_hdr(req, &hdr.first);
    } else {
        am_id    = UCP_AM_ID_EAGER_MIDDLE;
        hdr_size = sizeof(hdr.middle);
        ucp_proto_eager_set_middle_hdr(req, &hdr.middle);
    }

    ucp_datatype_iter_next_iov(&req->send.state.dt_iter, lpriv->super.memh_index,
                               ucp_proto_multi_max_payload(req, lpriv, hdr_size),
                               next_iter, &iov);
    return uct_ep_am_zcopy(req->send.ep->uct_eps[lpriv->super.lane], am_id, &hdr,
                           hdr_size, &iov, 1, 0, &req->send.state.uct_comp);
}

static ucs_status_t ucp_proto_eager_zcopy_multi_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    return ucp_proto_multi_zcopy_progress(req, req->send.proto_config->priv,
                                          ucp_proto_msg_multi_request_init,
                                          UCT_MD_MEM_ACCESS_LOCAL_READ,
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
