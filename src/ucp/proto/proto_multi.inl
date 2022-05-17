/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_MULTI_INL_
#define UCP_PROTO_MULTI_INL_

#include "proto_multi.h"

#include <ucp/proto/proto_common.inl>


static UCS_F_ALWAYS_INLINE void
ucp_proto_multi_set_send_lane(ucp_request_t *req)
{
#if ENABLE_ASSERT
    req->send.lane = UCP_NULL_LANE;
#endif
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_multi_request_init(ucp_request_t *req)
{
    req->send.multi_lane_idx = 0;
    ucp_proto_multi_set_send_lane(req);
}

static UCS_F_ALWAYS_INLINE uint32_t
ucs_proto_multi_calc_weight(double lane_weight, double total_weight)
{
    return (uint32_t)(
            ((lane_weight * UCP_PROTO_MULTI_WEIGHT_MAX) / total_weight) + 0.5);
}

static UCS_F_ALWAYS_INLINE size_t
ucp_proto_multi_scaled_length(uint32_t weight, size_t length)
{
    return ((weight * length) + UCP_PROTO_MULTI_WEIGHT_MAX - 1) >>
           UCP_PROTO_MULTI_WEIGHT_SHIFT;
}

static UCS_F_ALWAYS_INLINE size_t
ucp_proto_multi_max_payload(ucp_request_t *req,
                            const ucp_proto_multi_lane_priv_t *lpriv,
                            size_t hdr_size)
{
    size_t length   = req->send.state.dt_iter.length;
    size_t max_frag = lpriv->max_frag - hdr_size;
    size_t max_payload;

    /* Do not split very small sends to chunks, it's not worth it, and
       generic datatype may not be able to pack to a smaller buffer */
    if (length < UCP_MIN_BCOPY) {
        return max_frag;
    }

    max_payload = ucs_min(ucp_proto_multi_scaled_length(lpriv->weight, length),
                          max_frag);
    ucs_assertv(max_payload > 0,
                "length=%zu weight=%zu%% lpriv->max_frag=%zu hdr_size=%zu",
                req->send.state.dt_iter.length,
                ucp_proto_multi_scaled_length(lpriv->weight, 100),
                lpriv->max_frag, hdr_size);
    return max_payload;
}

static size_t UCS_F_ALWAYS_INLINE
ucp_proto_multi_data_pack(ucp_proto_multi_pack_ctx_t *pack_ctx, void *dest)
{
    ucp_request_t *req = pack_ctx->req;

    return ucp_datatype_iter_next_pack(&req->send.state.dt_iter,
                                       req->send.ep->worker,
                                       pack_ctx->max_payload,
                                       pack_ctx->next_iter, dest);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_multi_no_resource(ucp_request_t *req, ucp_lane_index_t lane)
{
    ucs_status_t status;
    uct_ep_h uct_ep;

    if (lane == req->send.lane) {
        /* if we failed to send on same lane, return error */
        return UCS_ERR_NO_RESOURCE;
    }

    /* failed to send on another lane - add to its pending queue */
    uct_ep = req->send.ep->uct_eps[lane];
    status = uct_ep_pending_add(uct_ep, &req->send.uct, 0);
    if (status == UCS_ERR_BUSY) {
        /* try sending again */
        return UCS_INPROGRESS;
    }

    ucs_assert(status == UCS_OK);
    req->send.lane = lane;

    /* Remove the request from current pending queue because it was added to
     * other lane's pending queue.
     */
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_multi_handle_send_error(
        ucp_request_t *req, ucp_lane_index_t lane, ucs_status_t status)
{
    if (ucs_likely(status == UCS_ERR_NO_RESOURCE)) {
        return ucp_proto_multi_no_resource(req, lane);
    }

    /* failed to send - call common error handler */
    ucp_proto_request_abort(req, status);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_multi_progress(ucp_request_t *req,
                         const ucp_proto_multi_priv_t *mpriv,
                         ucp_proto_send_multi_cb_t send_func,
                         ucp_proto_complete_cb_t complete_func,
                         unsigned dt_mask)
{
    const ucp_proto_multi_lane_priv_t *lpriv;
    ucp_datatype_iter_t next_iter;
    ucp_lane_index_t lane_idx;
    ucs_status_t status;

    ucs_assertv(req->send.multi_lane_idx < mpriv->num_lanes,
                "lane_idx=%d num_lanes=%d", req->send.multi_lane_idx,
                mpriv->num_lanes);

    lane_idx = req->send.multi_lane_idx;
    lpriv    = &mpriv->lanes[lane_idx];

    /* send the next fragment */
    status = send_func(req, lpriv, &next_iter);
    if (ucs_likely(status == UCS_OK)) {
        /* fast path is OK */
    } else if (status == UCS_INPROGRESS) {
        /* operation started and completion will be called later */
        ++req->send.state.uct_comp.count;
    } else {
        return ucp_proto_multi_handle_send_error(req, lpriv->super.lane,
                                                 status);
    }

    /* advance position in send buffer */
    ucp_datatype_iter_copy_position(&req->send.state.dt_iter, &next_iter,
                                    dt_mask);
    if (ucp_datatype_iter_is_end(&req->send.state.dt_iter)) {
        return complete_func(req);
    }

    /* move to the next lane, in a round-robin fashion */
    lane_idx = req->send.multi_lane_idx + 1;
    if (lane_idx >= mpriv->num_lanes) {
        lane_idx = 0;
    }
    req->send.multi_lane_idx = lane_idx;

    return UCS_INPROGRESS;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_multi_bcopy_progress(ucp_request_t *req,
                               const ucp_proto_multi_priv_t *mpriv,
                               ucp_proto_init_cb_t init_func,
                               ucp_proto_send_multi_cb_t send_func,
                               ucp_proto_complete_cb_t comp_func)
{
    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        ucp_proto_multi_request_init(req);
        if (init_func != NULL) {
            init_func(req);
        }

        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    return ucp_proto_multi_progress(req, mpriv, send_func, comp_func, UINT_MAX);
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_multi_zcopy_progress(
        ucp_request_t *req, const ucp_proto_multi_priv_t *mpriv,
        ucp_proto_init_cb_t init_func, unsigned uct_mem_flags, unsigned dt_mask,
        ucp_proto_send_multi_cb_t send_func,
        ucp_proto_complete_cb_t complete_func,
        uct_completion_callback_t uct_comp_cb)
{
    ucs_status_t status;

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        status = ucp_proto_request_zcopy_init(req, mpriv->reg_md_map,
                                              uct_comp_cb, uct_mem_flags,
                                              dt_mask);
        if (status != UCS_OK) {
            ucp_proto_request_abort(req, status);
            return UCS_OK; /* remove from pending after request is completed */
        }

        ucp_proto_multi_request_init(req);
        if (init_func != NULL) {
            init_func(req);
        }

        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    return ucp_proto_multi_progress(req, mpriv, send_func, complete_func,
                                    dt_mask);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_multi_lane_map_progress(ucp_request_t *req, ucp_lane_map_t *lane_map,
                                   ucp_proto_multi_lane_send_func_t send_func)
{
    ucp_lane_index_t lane = ucs_ffs32(*lane_map);
    ucs_status_t status;

    ucs_assert(*lane_map != 0);

    status = send_func(req, lane);
    if (ucs_likely(status == UCS_OK)) {
        /* fast path is OK */
    } else if (status == UCS_INPROGRESS) {
        ++req->send.state.uct_comp.count;
    } else {
        return ucp_proto_multi_handle_send_error(req, lane, status);
    }

    *lane_map &= ~UCS_BIT(lane);
    if (*lane_map != 0) {
        return UCS_INPROGRESS; /* Not finished all lanes yet */
    }

    ucp_request_invoke_uct_completion_success(req);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_eager_bcopy_multi_common_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter, ucp_am_id_t am_id_first,
        uct_pack_callback_t pack_cb_first, size_t hdr_size_first,
        ucp_am_id_t am_id_middle, uct_pack_callback_t pack_cb_middle,
        size_t hdr_size_middle)
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
        am_id    = am_id_middle;
        pack_cb  = pack_cb_middle;
        hdr_size = hdr_size_middle;
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

#endif
