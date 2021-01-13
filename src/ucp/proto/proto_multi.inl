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
ucp_proto_multi_request_init(ucp_request_t *req)
{
    req->send.multi_lane_idx = 0;
#if ENABLE_ASSERT
    req->send.lane           = UCP_NULL_LANE;
#endif
}

static UCS_F_ALWAYS_INLINE size_t
ucp_proto_multi_max_payload(ucp_request_t *req,
                            const ucp_proto_multi_lane_priv_t *lpriv,
                            size_t hdr_size)
{
    return lpriv->max_frag - hdr_size;
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
ucp_proto_multi_progress(ucp_request_t *req, ucp_proto_send_multi_cb_t send_func,
                         ucp_proto_complete_cb_t complete_func, unsigned dt_mask)
{
    const ucp_proto_multi_priv_t *mpriv = req->send.proto_config->priv;
    const ucp_proto_multi_lane_priv_t *lpriv;
    ucs_status_t pending_add_status;
    ucp_datatype_iter_t next_iter;
    ucp_lane_index_t lane_idx;
    ucs_status_t status;
    uct_ep_h uct_ep;

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
        if (lpriv->super.lane == req->send.lane) {
            /* if we failed to send on same lane, return error */
            return UCS_ERR_NO_RESOURCE;
        }

        /* failed to send on another lane - add to its pending queue */
        uct_ep             = req->send.ep->uct_eps[lpriv->super.lane];
        pending_add_status = uct_ep_pending_add(uct_ep, &req->send.uct, 0);
        if (pending_add_status == UCS_ERR_BUSY) {
            /* try sending again */
            return UCS_INPROGRESS;
        }

        ucs_assert(pending_add_status == UCS_OK);
        req->send.lane = lpriv->super.lane;

        /* remove the request from current pending queue because it was
         * added to other lane's pending queue
         * TODO return an indication, if the protocol needs to roll-back
         */
        return UCS_OK;
    } else {
        /* send failed - complete request with error */
        ucs_debug("send %s completed with status %s",
                  req->send.proto_config->proto->name, ucs_status_string(status));
        complete_func(req, status);
        return UCS_OK;
    }

    /* advance position in send buffer */
    ucp_datatype_iter_copy_from_next(&req->send.state.dt_iter, &next_iter,
                                     dt_mask);
    if (ucp_datatype_iter_is_end(&req->send.state.dt_iter)) {
        complete_func(req, UCS_OK);
        return UCS_OK;
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
ucp_proto_multi_zcopy_progress(uct_pending_req_t *uct_req,
                               ucp_proto_init_cb_t init_func,
                               ucp_proto_send_multi_cb_t send_func,
                               uct_completion_callback_t comp_func)
{
    ucp_request_t *req                 = ucs_container_of(uct_req, ucp_request_t,
                                                          send.uct);
    const ucp_proto_multi_priv_t *priv = req->send.proto_config->priv;
    ucs_status_t status;

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        status = ucp_proto_request_zcopy_init(req, priv->reg_md_map, comp_func);
        if (status != UCS_OK) {
            ucp_proto_request_zcopy_complete(req, status);
            return UCS_OK; /* remove from pending after request is completed */
        }

        ucp_proto_multi_request_init(req);
        if (init_func != NULL) {
            init_func(req);
        }

        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    return ucp_proto_multi_progress(req, send_func,
                                    ucp_request_invoke_uct_completion,
                                    UCS_BIT(UCP_DATATYPE_CONTIG));
}

#endif
