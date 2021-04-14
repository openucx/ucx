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

static UCS_F_ALWAYS_INLINE size_t
ucs_proto_multi_calc_weight(double lane_weight, double total_weight)
{
    return (size_t)(
            lane_weight * UCS_BIT(UCP_PROTO_MULTI_WEIGHT_SHIFT) / total_weight +
            0.5);
}

static UCS_F_ALWAYS_INLINE size_t
ucp_proto_multi_scaled_length(const ucp_proto_multi_lane_priv_t *lpriv,
                              size_t length)
{
    return (lpriv->weight * length + UCS_MASK(UCP_PROTO_MULTI_WEIGHT_SHIFT)) >>
           UCP_PROTO_MULTI_WEIGHT_SHIFT;
}

static UCS_F_ALWAYS_INLINE size_t
ucp_proto_multi_max_payload(ucp_request_t *req,
                            const ucp_proto_multi_lane_priv_t *lpriv,
                            size_t hdr_size)
{
    size_t scaled_length =
            ucp_proto_multi_scaled_length(lpriv, req->send.state.dt_iter.length);
    size_t max_payload   = ucs_min(scaled_length, lpriv->max_frag - hdr_size);

    ucs_assertv(max_payload > 0,
                "length=%zu weight=%.2f scaled_length=%zu max_frag=%zu "
                "hdr_size=%zu",
                req->send.state.dt_iter.length,
                lpriv->weight / (double)UCS_BIT(UCP_PROTO_MULTI_WEIGHT_SHIFT),
                scaled_length, lpriv->max_frag, hdr_size);
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
ucp_proto_multi_no_resource(ucp_request_t *req,
                            const ucp_proto_multi_lane_priv_t *lpriv)
{
    ucs_status_t status;
    uct_ep_h uct_ep;

    if (lpriv->super.lane == req->send.lane) {
        /* if we failed to send on same lane, return error */
        return UCS_ERR_NO_RESOURCE;
    }

    /* failed to send on another lane - add to its pending queue */
    uct_ep = req->send.ep->uct_eps[lpriv->super.lane];
    status = uct_ep_pending_add(uct_ep, &req->send.uct, 0);
    if (status == UCS_ERR_BUSY) {
        /* try sending again */
        return UCS_INPROGRESS;
    }

    ucs_assert(status == UCS_OK);
    req->send.lane = lpriv->super.lane;

    /* Remove the request from current pending queue because it was added to
     * other lane's pending queue.
     */
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
    ucs_assert(!ucp_datatype_iter_is_end(&req->send.state.dt_iter));

    lane_idx = req->send.multi_lane_idx;
    lpriv    = &mpriv->lanes[lane_idx];

    /* send the next fragment */
    status = send_func(req, lpriv, &next_iter);
    if (ucs_likely(status == UCS_OK)) {
        /* fast path is OK */
    } else if (status == UCS_INPROGRESS) {
        /* operation started and completion will be called later */
        ++req->send.state.uct_comp.count;
    } else if (status == UCS_ERR_NO_RESOURCE) {
        return ucp_proto_multi_no_resource(req, lpriv);
    } else {
        /* failed to send - call common error handler */
        ucp_proto_request_abort(req, status);
        return UCS_OK;
    }

    /* advance position in send buffer */
    ucp_datatype_iter_copy_from_next(&req->send.state.dt_iter, &next_iter,
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
        ucp_proto_init_cb_t init_func, unsigned uct_mem_flags,
        ucp_proto_send_multi_cb_t send_func,
        uct_completion_callback_t comp_func)
{
    ucs_status_t status;

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        status = ucp_proto_request_zcopy_init(req, mpriv->reg_md_map, comp_func,
                                              uct_mem_flags);
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

    return ucp_proto_multi_progress(req, mpriv, send_func,
                                    ucp_request_invoke_uct_completion_success,
                                    UCS_BIT(UCP_DATATYPE_CONTIG));
}

#endif
