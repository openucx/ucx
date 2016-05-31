/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_request.h"
#include "ucp_ep.inl"

#include <ucp/core/ucp_worker.h>
#include <ucp/dt/dt_generic.h>
#include <inttypes.h>


static UCS_F_ALWAYS_INLINE void
ucp_request_put(ucp_request_t *req)
{
    if ((req->flags |= UCP_REQUEST_FLAG_COMPLETED) & UCP_REQUEST_FLAG_RELEASED) {
        ucs_mpool_put(req);
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_request_complete_send(ucp_request_t *req, ucs_status_t status)
{
    ucs_trace_data("completing send request %p (%p), %s", req, req + 1,
                   ucs_status_string(status));
    req->send.cb(req + 1, status);
    ucp_request_put(req);
}

static UCS_F_ALWAYS_INLINE void
ucp_request_complete_recv(ucp_request_t *req, ucs_status_t status,
                          ucp_tag_recv_info_t *info)
{
    ucs_trace_data("completing recv request %p (%p) stag 0x%"PRIx64" len %zu, %s",
                   req, req + 1, info->sender_tag, info->length,
                   ucs_status_string(status));
    req->recv.cb(req + 1, status, info);
    ucp_request_put(req);
}

static UCS_F_ALWAYS_INLINE size_t
ucp_request_generic_dt_pack(ucp_request_t *req, void *dest, size_t length)
{
    ucp_dt_generic_t *dt = ucp_dt_generic(req->send.datatype);
    return dt->ops.pack(req->send.state.dt.generic.state,
                        req->send.state.offset, dest, length);
}

static UCS_F_ALWAYS_INLINE void
ucp_request_generic_dt_finish(ucp_request_t *req)
{
    ucp_dt_generic_t *dt = ucp_dt_generic(req->send.datatype);
    return dt->ops.finish(req->send.state.dt.generic.state);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_request_send_buffer_reg(ucp_request_t *req, ucp_lane_index_t lane)
{
    uct_pd_h uct_pd = ucp_ep_pd(req->send.ep, lane);
    ucs_status_t status;

    status = uct_pd_mem_reg(uct_pd, (void*)req->send.buffer, req->send.length,
                            &req->send.state.dt.contig.memh);
    if (status != UCS_OK) {
        ucs_error("failed to register user buffer: %s",
                  ucs_status_string(status));
    }
    return status;
}

static UCS_F_ALWAYS_INLINE void
ucp_request_send_buffer_dereg(ucp_request_t *req, ucp_lane_index_t lane)
{
    uct_pd_h uct_pd = ucp_ep_pd(req->send.ep, lane);
    (void)uct_pd_mem_dereg(uct_pd, req->send.state.dt.contig.memh);
}
