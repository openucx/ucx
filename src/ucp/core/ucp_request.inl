/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_request.h"
#include "ucp_worker.h"
#include "ucp_ep.inl"

#include <ucp/core/ucp_worker.h>
#include <ucp/dt/dt.h>
#include <ucs/debug/instrument.h>
#include <ucs/datastruct/mpool.inl>
#include <inttypes.h>


static UCS_F_ALWAYS_INLINE ucp_request_t*
ucp_request_get(ucp_worker_h worker)
{
    ucp_request_t *req = ucs_mpool_get_inline(&worker->req_mp);

    if (req != NULL) {
        VALGRIND_MAKE_MEM_DEFINED(req + 1,  worker->context->config.request.size);
    }
    return req;
}

static UCS_F_ALWAYS_INLINE void
ucp_request_put(ucp_request_t *req, ucs_status_t status)
{
    req->status = status;
    if ((req->flags |= UCP_REQUEST_FLAG_COMPLETED) & UCP_REQUEST_FLAG_RELEASED) {
        /* Release should not be called for external requests */
        ucs_assert(!(req->flags & UCP_REQUEST_FLAG_EXTERNAL));
        ucs_mpool_put(req);
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_request_complete_send(ucp_request_t *req, ucs_status_t status)
{
    ucs_trace_data("completing send request %p (%p), %s", req, req + 1,
                   ucs_status_string(status));
    req->send.cb(req + 1, status);

    UCS_INSTRUMENT_RECORD(UCS_INSTRUMENT_TYPE_UCP_TX,
                          "ucp_request_complete_send",
                          req, 0);
    ucp_request_put(req, status);
}

static UCS_F_ALWAYS_INLINE void
ucp_request_complete_recv(ucp_request_t *req, ucs_status_t status,
                          ucp_tag_recv_info_t *info)
{
    ucs_trace_data("completing recv request %p (%p) stag 0x%"PRIx64" len %zu, %s",
                   req, req + 1, info->sender_tag, info->length,
                   ucs_status_string(status));

    if (!(req->flags & UCP_REQUEST_FLAG_EXTERNAL)) {
        /* In the current API callbacks are defined for internal reqs only. */
        req->recv.cb(req + 1, status, info);
    }
    ucp_request_put(req, status);

    UCS_INSTRUMENT_RECORD(UCS_INSTRUMENT_TYPE_UCP_RX,
                          "ucp_request_complete_recv",
                          req, info->length);
}

static UCS_F_ALWAYS_INLINE
void ucp_request_send_generic_dt_finish(ucp_request_t *req)
{
    ucp_dt_generic_t *dt;
    if (UCP_DATATYPE_GENERIC == (req->send.datatype & UCP_DATATYPE_CLASS_MASK)) {
        dt = ucp_dt_generic(req->send.datatype);
        ucs_assert(NULL != dt);
        dt->ops.finish(req->send.state.dt.generic.state);
    }
}

static UCS_F_ALWAYS_INLINE
void ucp_request_recv_generic_dt_finish(ucp_request_t *req)
{
    ucp_dt_generic_t *dt;
    if (UCP_DATATYPE_GENERIC == (req->send.datatype & UCP_DATATYPE_CLASS_MASK)) {
        dt = ucp_dt_generic(req->recv.datatype);
        ucs_assert(NULL != dt);
        dt->ops.finish(req->recv.state.dt.generic.state);
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_request_send_buffer_reg(ucp_request_t *req, ucp_lane_index_t lane)
{
    uct_md_h uct_md = ucp_ep_md(req->send.ep, lane);
    ucs_status_t status;

    status = uct_md_mem_reg(uct_md, (void*)req->send.buffer, req->send.length, 0,
                            &req->send.state.dt.contig.memh);
    if (status != UCS_OK) {
        ucs_error("failed to register user buffer [address %p len %zu pd %s]: %s",
                  req->send.buffer, req->send.length,
                  ucp_ep_md_attr(req->send.ep, lane)->component_name,
                  ucs_status_string(status));
    }
    return status;
}

static UCS_F_ALWAYS_INLINE void
ucp_request_send_buffer_dereg(ucp_request_t *req, ucp_lane_index_t lane)
{
    uct_md_h uct_md = ucp_ep_md(req->send.ep, lane);
    (void)uct_md_mem_dereg(uct_md, req->send.state.dt.contig.memh);
}
