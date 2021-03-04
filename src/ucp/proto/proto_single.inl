/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
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
ucp_proto_single_status_handle(ucp_request_t *req,
                               ucp_proto_complete_cb_t complete_func,
                               ucp_lane_index_t lane, ucs_status_t status)
{
    if (ucs_likely(status == UCS_OK)) {
        if (complete_func != NULL) {
            complete_func(req);
        }
    } else if (status == UCS_ERR_NO_RESOURCE) {
        /* keep on pending queue */
        req->send.lane = lane;
        return UCS_ERR_NO_RESOURCE;
    } else if (status != UCS_INPROGRESS) {
        ucp_proto_request_abort(req, status);
    }
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_am_bcopy_single_progress(ucp_request_t *req, ucp_am_id_t am_id,
                                   ucp_lane_index_t lane,
                                   uct_pack_callback_t pack_func, void *pack_arg,
                                   size_t max_packed_size,
                                   ucp_proto_complete_cb_t complete_func)
{
    ucs_status_t status;

    status = ucp_proto_am_bcopy_single_send(req, am_id, lane, pack_func,
                                            pack_arg, max_packed_size);
    return ucp_proto_single_status_handle(req, complete_func, lane, status);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_zcopy_single_progress(ucp_request_t *req, unsigned uct_mem_flags,
                                ucp_proto_send_single_cb_t send_func,
                                const char *name)
{
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    ucp_datatype_iter_t next_iter;
    ucs_status_t status;
    ucp_md_map_t md_map;
    uct_iov_t iov;

    ucs_assert(req->send.state.dt_iter.offset == 0);

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        md_map = (spriv->reg_md == UCP_NULL_RESOURCE) ? 0 : UCS_BIT(spriv->reg_md);
        status = ucp_proto_request_zcopy_init(req, md_map,
                                              ucp_proto_request_zcopy_completion,
                                              uct_mem_flags);
        if (status != UCS_OK) {
            ucp_proto_request_abort(req, status);
            return UCS_OK; /* remove from pending after request is completed */
        }

        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    ucp_datatype_iter_next_iov(&req->send.state.dt_iter, spriv->super.memh_index,
                               SIZE_MAX, &next_iter, &iov);
    status = send_func(req, spriv, &iov);
    UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(req, name, iov.length, status);

    return ucp_proto_single_status_handle(
            req, ucp_proto_request_zcopy_complete_success, spriv->super.lane,
            status);
}

#endif
