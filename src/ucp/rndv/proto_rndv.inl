/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_RNDV_INL_
#define UCP_PROTO_RNDV_INL_

#include "proto_rndv.h"

#include <ucp/core/ucp_rkey.inl>
#include <ucp/proto/proto_am.inl>
#include <ucp/proto/proto_single.inl>
#include <ucp/proto/proto_multi.inl>


static UCS_F_ALWAYS_INLINE size_t
ucp_proto_rndv_cfg_thresh(ucp_context_h context, uint64_t rndv_modes)
{
    ucs_assert(!(rndv_modes & UCS_BIT(UCP_RNDV_MODE_AUTO)));

    if (context->config.ext.rndv_mode == UCP_RNDV_MODE_AUTO) {
        return UCS_MEMUNITS_AUTO; /* automatic threshold */
    } else if (rndv_modes & UCS_BIT(context->config.ext.rndv_mode)) {
        return 0; /* enabled by default */
    } else {
        return UCS_MEMUNITS_INF; /* used only as last resort */
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_rts_request_init(ucp_request_t *req)
{
    const ucp_proto_rndv_ctrl_priv_t *rpriv = req->send.proto_config->priv;
    ucp_ep_h ep                             = req->send.ep;
    ucs_status_t status;

    if (req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED) {
        return UCS_OK;
    }

    status = ucp_ep_resolve_remote_id(req->send.ep, rpriv->lane);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_datatype_iter_mem_reg(ep->worker->context,
                                       &req->send.state.dt_iter, rpriv->md_map,
                                       UCT_MD_MEM_ACCESS_RMA |
                                       UCT_MD_MEM_FLAG_HIDE_ERRORS,
                                       UCS_BIT(UCP_DATATYPE_CONTIG));
    if (status != UCS_OK) {
        return status;
    }

    ucp_send_request_id_alloc(req);
    req->flags                    |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    req->send.state.completed_size = 0;

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_ats_handler(void *arg, void *data, size_t length, unsigned flags)
{
    ucp_worker_h worker           = arg;
    const ucp_reply_hdr_t *rephdr = data;
    ucs_status_t status           = rephdr->status;
    const ucp_rndv_ack_hdr_t *ats;
    ucp_request_t *req;

    UCP_SEND_REQUEST_GET_BY_ID(&req, worker, rephdr->req_id, 0, return UCS_OK,
                               "ATS %p", rephdr);
    if (length >= sizeof(*ats)) {
        /* ATS message carries a size field */
        ats = ucs_derived_of(rephdr, ucp_rndv_ack_hdr_t);
        if (!ucp_proto_common_frag_complete(req, ats->size, "rndv_ats")) {
            return UCS_OK; /* Not completed */
        }
    }

    ucp_send_request_id_release(req);
    ucp_proto_request_zcopy_complete(req, status);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE size_t ucp_proto_rndv_rts_pack(
        ucp_request_t *req, ucp_rndv_rts_hdr_t *rts, size_t hdr_len)
{
    void *rkey_buffer = UCS_PTR_BYTE_OFFSET(rts, hdr_len);
    const ucp_proto_rndv_ctrl_priv_t *rpriv;
    size_t rkey_size;

    rts->sreq.req_id = ucp_send_request_get_id(req);
    rts->sreq.ep_id  = ucp_send_request_get_ep_remote_id(req);
    rts->size        = req->send.state.dt_iter.length;
    rpriv            = req->send.proto_config->priv;

    if ((rts->size == 0) || (rpriv->md_map == 0)) {
        rts->address = 0;
        rkey_size    = 0;
    } else {
        rts->address = (uintptr_t)req->send.state.dt_iter.type.contig.buffer;
        rkey_size    = UCS_PROFILE_CALL(ucp_proto_request_pack_rkey, req,
                                        rpriv->md_map, rpriv->sys_dev_map,
                                        rpriv->sys_dev_distance, rkey_buffer);
    }

    return hdr_len + rkey_size;
}

static size_t UCS_F_ALWAYS_INLINE ucp_proto_rndv_pack_ack(ucp_request_t *req,
                                                          void *dest,
                                                          size_t ack_size)
{
    ucp_rndv_ack_hdr_t *ack_hdr = dest;

    /* Sending 0-length ATS/ATP can lead to undefined behavior on remote side */
    if (req->send.state.dt_iter.length > 0) {
        ucs_assert(ack_size > 0);
    }

    ack_hdr->super.req_id = req->send.rndv.remote_req_id;
    ack_hdr->super.status = UCS_OK;
    ack_hdr->size         = ack_size;
    return sizeof(*ack_hdr);
}

static size_t UCS_F_ALWAYS_INLINE ucp_proto_rndv_ack_progress(
        ucp_request_t *req, const ucp_proto_rndv_ack_priv_t *apriv,
        ucp_am_id_t am_id, uct_pack_callback_t pack_func,
        ucp_proto_complete_cb_t complete_func)
{
    return ucp_proto_am_bcopy_single_progress(req, am_id, apriv->lane,
                                              pack_func, req,
                                              sizeof(ucp_rndv_ack_hdr_t),
                                              complete_func);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_rkey_destroy(ucp_request_t *req)
{
    ucs_assert(req->send.rndv.rkey != NULL);
    ucp_rkey_destroy(req->send.rndv.rkey);
#if UCS_ENABLE_ASSERT
    req->send.rndv.rkey = NULL;
#endif
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_recv_complete_with_ats(ucp_request_t *req, uint8_t ats_stage)
{
    ucp_proto_rndv_rkey_destroy(req);
    ucp_proto_request_set_stage(req, ats_stage);
    ucp_request_send(req);
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_rndv_frag_request_alloc(
        ucp_worker_h worker, ucp_request_t *req, ucp_request_t **freq_p)
{
    ucp_request_t *freq;

    freq = ucp_request_get(worker);
    if (freq == NULL) {
        ucs_error("failed to allocated rendezvous send fragment");
        return UCS_ERR_NO_MEMORY;
    }

    ucp_trace_req(req, "allocated rndv fragment %p", freq);
    freq->flags   = UCP_REQUEST_FLAG_RNDV_FRAG;
    freq->send.ep = req->send.ep;
    ucp_request_set_super(freq, req);

    *freq_p = freq;
    return UCS_OK;
}

/* @return Nonzero if the top-level rendezvous request 'req' is completed */
static UCS_F_ALWAYS_INLINE int ucp_proto_rndv_frag_complete(ucp_request_t *req,
                                                            ucp_request_t *freq,
                                                            const char *title)
{
    size_t frag_size = freq->send.state.dt_iter.length;

    ucs_assert(freq->flags & UCP_REQUEST_FLAG_RNDV_FRAG);
    ucp_request_put(freq);

    return ucp_proto_common_frag_complete(req, frag_size, title);
}

static UCS_F_ALWAYS_INLINE size_t
ucp_proto_rndv_request_total_length(ucp_request_t *req)
{
    ucp_request_t *super_req;

    if (ucs_unlikely(req->flags & UCP_REQUEST_FLAG_RNDV_FRAG)) {
        super_req = ucp_request_get_super(req);
        return super_req->send.state.dt_iter.length;
    }

    return req->send.state.dt_iter.length;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_bulk_request_init(ucp_request_t *req,
                                 const ucp_proto_rndv_bulk_priv_t *rpriv)
{
    if (req->send.rndv.offset == 0) {
        req->send.multi_lane_idx = 0;
    } else {
        ucp_proto_rndv_bulk_request_init_lane_idx(req, rpriv);
    }
    ucp_proto_multi_set_send_lane(req);
}

/**
 * Calculate how much data to send on the next lane in a rendezvous protocol,
 * including when the request is a fragment and starts from nonzero offset.
 */
static UCS_F_ALWAYS_INLINE size_t
ucp_proto_rndv_bulk_max_payload(ucp_request_t *req,
                                const ucp_proto_rndv_bulk_priv_t *rpriv,
                                const ucp_proto_multi_lane_priv_t *lpriv)
{
    size_t total_offset = req->send.rndv.offset +
                          req->send.state.dt_iter.offset;
    size_t total_length = ucp_proto_rndv_request_total_length(req);
    size_t max_frag_sum = rpriv->mpriv.max_frag_sum;
    size_t lane_offset, max_payload, scaled_length;

    if (ucs_likely(total_length < max_frag_sum)) {
        /* Each lane sends less than its maximal fragment size */
        scaled_length = ucp_proto_multi_scaled_length(lpriv->weight_sum,
                                                      total_length);

        ucs_assertv(scaled_length >= total_offset,
                    "req=%p scaled_length=%zu total_offset=%zu "
                    "total_length=%zu weight_sum=%zu%% ",
                    req, scaled_length, total_offset, total_length,
                    ucp_proto_multi_scaled_length(lpriv->weight_sum, 100));

        max_payload = scaled_length - total_offset;
    } else {
        /* Send in round-robin fashion, each lanes sends its maximal size */
        lane_offset = total_offset % max_frag_sum;
        ucs_assertv(lpriv->max_frag_sum >= lane_offset,
                    "req=%p lpriv->max_frag_sum=%zu lane_offset=%zu", req,
                    lpriv->max_frag_sum, lane_offset);

        max_payload = lpriv->max_frag_sum - lane_offset;
    }

    ucp_trace_req(req,
                  "offset %zu/%zu (start %zu/%zu) max_frag_sum %zu/%zu: "
                  "max_payload %zu",
                  req->send.state.dt_iter.offset,
                  req->send.state.dt_iter.length, req->send.rndv.offset,
                  total_length, lpriv->max_frag_sum, max_frag_sum, max_payload);

    /* Check that send length is not greater than maximal fragment size */
    ucs_assertv(max_payload <= lpriv->max_frag,
                "req=%p max_payload=%zu max_frag=%zu", req, max_payload,
                lpriv->max_frag);
    return max_payload;
}


static UCS_F_ALWAYS_INLINE int
ucp_proto_rndv_request_is_ppln_frag(ucp_request_t *req)
{
    return req->send.proto_config->select_param.op_flags &
           UCP_PROTO_SELECT_OP_FLAG_PPLN_FRAG;
}

static UCS_F_ALWAYS_INLINE int
ucp_proto_rndv_init_params_is_ppln_frag(const ucp_proto_init_params_t *params)
{
    return params->select_param->op_flags & UCP_PROTO_SELECT_OP_FLAG_PPLN_FRAG;
}

static UCS_F_ALWAYS_INLINE int
ucp_proto_rndv_op_check(const ucp_proto_init_params_t *params,
                        ucp_operation_id_t op_id, int support_ppln)
{
    return (params->select_param->op_id == op_id) &&
           (support_ppln || !ucp_proto_rndv_init_params_is_ppln_frag(params));
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_recv_complete_status(ucp_request_t *req, ucs_status_t status)
{
    ucp_request_t *rreq = ucp_request_get_super(req);

    ucp_trace_req(req, "rndv_recv_complete rreq=%p", rreq);

    /* Remote key should already be released */
    ucs_assert(req->send.rndv.rkey == NULL);

    ucs_assert(!ucp_proto_rndv_request_is_ppln_frag(req));

    if (rreq->flags & UCP_REQUEST_FLAG_RECV_AM) {
        ucp_request_complete_am_recv(rreq, status);
    } else {
        ucs_assert(rreq->flags & UCP_REQUEST_FLAG_RECV_TAG);
        ucp_request_complete_tag_recv(rreq, status);
    }

    ucp_request_put(req);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_recv_complete(ucp_request_t *req)
{
    ucp_request_t *rreq = ucp_request_get_super(req);

    return ucp_proto_rndv_recv_complete_status(req, rreq->status);
}

#endif
