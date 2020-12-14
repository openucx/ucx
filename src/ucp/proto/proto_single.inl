/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_SINGLE_INL_
#define UCP_PROTO_SINGLE_INL_

#include "proto_single.h"
#include "proto_common.inl"

#include <ucp/dt/datatype_iter.inl>


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_am_bcopy_single_send(ucp_request_t *req, ucp_am_id_t am_id,
                               ucp_lane_index_t lane,
                               uct_pack_callback_t pack_func, void *pack_arg,
                               size_t max_packed_size)
{
    ucp_ep_t *ep                       = req->send.ep;
    const uct_iface_attr_t *iface_attr = ucp_ep_get_iface_attr(ep, lane);
    ssize_t packed_size;
    uint64_t *buffer;

    if ((max_packed_size <= UCS_ALLOCA_MAX_SIZE) &&
        (iface_attr->cap.flags & UCT_IFACE_FLAG_AM_SHORT) &&
        (max_packed_size <= iface_attr->cap.am.max_short)) {

        /* Send as inline if expected size is small enough */
        buffer      = ucs_alloca(max_packed_size);
        packed_size = pack_func(buffer, pack_arg);
        ucs_assertv((packed_size >= 0) && (packed_size <= max_packed_size),
                    "packed_size=%zd max_packed_size=%zu", packed_size,
                    max_packed_size);
        return uct_ep_am_short(ep->uct_eps[lane], am_id, buffer[0],
                               &buffer[1], packed_size - sizeof(buffer[0]));
    } else {
        /* Send as bcopy */
        packed_size = uct_ep_am_bcopy(ep->uct_eps[lane], am_id, pack_func,
                                      pack_arg, 0);
        return ucs_likely(packed_size >= 0) ? UCS_OK : packed_size;
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_am_bcopy_single_progress(ucp_request_t *req, ucp_am_id_t am_id,
                                   ucp_lane_index_t lane,
                                   uct_pack_callback_t pack_func, void *pack_arg,
                                   size_t max_packed_size,
                                   ucp_proto_complete_cb_t complete_func,
                                   ucp_proto_complete_cb_t error_func)
{
    ucs_status_t status;

    ucs_assert(error_func != NULL);

    status = ucp_proto_am_bcopy_single_send(req, am_id, lane, pack_func,
                                            pack_arg, max_packed_size);
    if (ucs_likely(status == UCS_OK)) {
        if (complete_func != NULL) {
            complete_func(req, status);
        }
    } else if (status == UCS_ERR_NO_RESOURCE) {
        req->send.lane = lane;
        return UCS_ERR_NO_RESOURCE;
    } else {
        ucs_assert(status != UCS_INPROGRESS);
        error_func(req, status);
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_am_zcopy_single_progress(ucp_request_t *req, ucp_am_id_t am_id,
                                   const void *hdr, size_t hdr_size)
{
    ucp_ep_t *ep                         = req->send.ep;
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    ucp_datatype_iter_t next_iter;
    ucs_status_t status;
    ucp_md_map_t md_map;
    uct_iov_t iov;

    ucs_assert(req->send.state.dt_iter.offset == 0);

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        md_map = (spriv->reg_md == UCP_NULL_RESOURCE) ? 0 : UCS_BIT(spriv->reg_md);
        status = ucp_proto_request_zcopy_init(req, md_map,
                                              ucp_proto_request_zcopy_completion);
        if (status != UCS_OK) {
            ucp_proto_request_zcopy_complete(req, status);
            return UCS_OK; /* remove from pending after request is completed */
        }

        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    ucp_datatype_iter_next_iov(&req->send.state.dt_iter,
                               spriv->super.memh_index,
                               SIZE_MAX, &next_iter, &iov);
    status = uct_ep_am_zcopy(ep->uct_eps[spriv->super.lane], am_id, hdr,
                             hdr_size, &iov, 1, 0, &req->send.state.uct_comp);
    UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(req, "am_zcopy_only", iov.length,
                                           status);
    if (ucs_likely(status == UCS_OK)) {
        /* fastpath is UCS_OK */
    } else if (status == UCS_INPROGRESS) {
        /* completion callback will be called */
        return UCS_OK;
    } else if (status == UCS_ERR_NO_RESOURCE) {
        /* keep on pending queue */
        req->send.lane = spriv->super.lane;
        return UCS_ERR_NO_RESOURCE;
    }

    /* complete the request with OK or error */
    ucp_proto_request_zcopy_complete(req, status);
    return UCS_OK;
}

#endif
