/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_MULTI_INL_
#define UCP_PROTO_MULTI_INL_

#include "proto_multi.h"

#include <ucp/proto/proto_common.inl>


static UCS_F_ALWAYS_INLINE size_t
ucp_proto_multi_get_lane_opt_align(const ucp_proto_multi_init_params_t *params,
                                   ucp_lane_index_t lane)
{
    ucp_worker_h worker        = params->super.super.worker;
    ucp_rsc_index_t rsc_index  = ucp_proto_common_get_rsc_index(&params->super.super,
                                                                lane);
    ucp_worker_iface_t *wiface = ucp_worker_iface(worker, rsc_index);

    return ucp_proto_common_get_iface_attr_field(&wiface->attr,
                                                 params->opt_align_offs, 1);
}

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
    uint32_t weight =
        ucs_div_round_up(lane_weight * UCP_PROTO_MULTI_WEIGHT_MAX,
                         total_weight);

    return ucs_min(weight, UCP_PROTO_MULTI_WEIGHT_MAX);
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

    ucs_assertv(lpriv->max_frag > hdr_size, "max_frag=%zu hdr_size=%zu",
                lpriv->max_frag, hdr_size);

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
    uct_ep = ucp_ep_get_lane(req->send.ep, lane);
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

static UCS_F_ALWAYS_INLINE void
ucp_proto_multi_advance_lane_idx(ucp_request_t *req, ucp_lane_index_t num_lanes,
                                 ucp_lane_index_t lane_shift)
{
    ucp_lane_index_t lane_idx;

    ucs_assertv(req->send.multi_lane_idx < num_lanes,
                "req=%p lane_idx=%d num_lanes=%d", req,
                req->send.multi_lane_idx, num_lanes);

    lane_idx = req->send.multi_lane_idx + lane_shift;
    if (lane_idx >= num_lanes) {
        lane_idx = 0;
    }
    req->send.multi_lane_idx = lane_idx;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_multi_progress(ucp_request_t *req,
                         const ucp_proto_multi_priv_t *mpriv,
                         ucp_proto_send_multi_cb_t send_func,
                         ucp_proto_complete_cb_t complete_func,
                         unsigned dt_mask)
{
    ucp_lane_index_t lane_shift = 1;
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
    status = send_func(req, lpriv, &next_iter, &lane_shift);
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
    ucp_proto_multi_advance_lane_idx(req, mpriv->num_lanes, lane_shift);

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

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_multi_lane_map_progress(
        ucp_request_t *req, ucp_lane_index_t *lane_p, ucp_lane_map_t lane_map,
        ucp_proto_multi_lane_send_func_t send_func)
{
    ucp_lane_map_t remaining_lane_map = lane_map & ~UCS_MASK(*lane_p);
    ucp_lane_index_t lane;
    ucs_status_t status;

    ucs_assertv(remaining_lane_map != 0,
                "req=%p *lane_p=%d lane_map=0x%" PRIx64, req, *lane_p,
                lane_map);
    lane = ucs_ffs64(remaining_lane_map);

    status = send_func(req, lane);
    if (ucs_likely(status == UCS_OK)) {
        /* fast path is OK */
    } else if (status == UCS_INPROGRESS) {
        ++req->send.state.uct_comp.count;
    } else {
        return ucp_proto_multi_handle_send_error(req, lane, status);
    }

    if (ucs_is_pow2_or_zero(remaining_lane_map)) {
        /* This lane was the last one */
        ucp_request_invoke_uct_completion_success(req);
        return UCS_OK;
    }

    /* Not finished yet, so continue from next lane */
    *lane_p = lane + 1;
    return UCS_INPROGRESS;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_am_bcopy_multi_common_send_func(
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
    uct_ep_h uct_ep;

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

    uct_ep      = ucp_ep_get_lane(ep, lpriv->super.lane);
    packed_size = uct_ep_am_bcopy(uct_ep, am_id, pack_cb, &pack_ctx, 0);

    return ucp_proto_bcopy_send_func_status(packed_size);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_am_zcopy_multi_common_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter, ucp_am_id_t am_id_first,
        const void *hdr_first, size_t hdr_size_first, ucp_am_id_t am_id_middle,
        const void *hdr_middle, size_t hdr_size_middle)
{
    uct_iov_t iov[UCP_MAX_IOV];
    size_t max_payload;
    ucp_am_id_t am_id;
    size_t hdr_size;
    size_t iov_count;
    const void *hdr;

    if (req->send.state.dt_iter.offset == 0) {
        am_id    = am_id_first;
        hdr      = hdr_first;
        hdr_size = hdr_size_first;
    } else {
        am_id    = am_id_middle;
        hdr      = hdr_middle;
        hdr_size = hdr_size_middle;
    }

    max_payload = ucp_proto_multi_max_payload(req, lpriv, hdr_size);
    iov_count   = ucp_datatype_iter_next_iov(&req->send.state.dt_iter,
                                             max_payload, lpriv->super.md_index,
                                             UCP_DT_MASK_CONTIG_IOV, next_iter,
                                             iov, lpriv->super.max_iov);
    return uct_ep_am_zcopy(ucp_ep_get_lane(req->send.ep, lpriv->super.lane),
                           am_id, hdr, hdr_size, iov, iov_count, 0,
                           &req->send.state.uct_comp);
}

#endif
