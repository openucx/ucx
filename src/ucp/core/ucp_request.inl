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
#include <ucs/debug/profile.h>
#include <ucs/datastruct/mpool.inl>
#include <inttypes.h>


#define UCP_REQUEST_FLAGS_FMT \
    "%c%c%c%c%c%c%c%c%c"

#define UCP_REQUEST_FLAGS_ARG(_flags) \
    (((_flags) & UCP_REQUEST_FLAG_COMPLETED)       ? 'd' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_RELEASED)        ? 'f' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_BLOCKING)        ? 'b' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_EXPECTED)        ? 'e' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_LOCAL_COMPLETED) ? 'L' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_CALLBACK)        ? 'c' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_RECV)            ? 'r' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_SYNC)            ? 's' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_RNDV)            ? 'v' : '-')


/* defined as a macro to print the call site */
#define ucp_request_get(_worker) \
    ({ \
        ucp_request_t *_req = ucs_mpool_get_inline(&(_worker)->req_mp); \
        if (_req != NULL) { \
            VALGRIND_MAKE_MEM_DEFINED(_req + 1, \
                                      (_worker)->context->config.request.size); \
            ucs_trace_req("allocated request %p", _req); \
            UCS_PROFILE_REQUEST_NEW(_req, "ucp_request", 0); \
        } \
        _req; \
    })

#define ucp_request_complete(_req, _cb, _status, ...) \
    { \
        (_req)->status = (_status); \
        if (ucs_likely((_req)->flags & UCP_REQUEST_FLAG_CALLBACK)) { \
            (_req)->_cb((_req) + 1, (_status), ## __VA_ARGS__); \
        } \
        if (ucs_unlikely(((_req)->flags  |= UCP_REQUEST_FLAG_COMPLETED) & \
                         UCP_REQUEST_FLAG_RELEASED)) { \
            ucp_request_put(_req); \
        } \
    }

#define ucp_request_set_callback(_req, _cb, _value) \
    { \
        (_req)->_cb    = _value; \
        (_req)->flags |= UCP_REQUEST_FLAG_CALLBACK; \
        ucs_trace_data("request %p %s set to %p", _req, #_cb, _value); \
    }


static UCS_F_ALWAYS_INLINE void
ucp_request_put(ucp_request_t *req)
{
    ucs_trace_req("put request %p", req);
    UCS_PROFILE_REQUEST_FREE(req);
    ucs_mpool_put_inline(req);
}

static UCS_F_ALWAYS_INLINE void
ucp_request_complete_send(ucp_request_t *req, ucs_status_t status)
{
    ucs_trace_req("completing send request %p (%p) "UCP_REQUEST_FLAGS_FMT" %s",
                  req, req + 1, UCP_REQUEST_FLAGS_ARG(req->flags),
                  ucs_status_string(status));
    UCS_PROFILE_REQUEST_EVENT(req, "complete_send", status);
    ucp_request_complete(req, send.cb, status);
}

static UCS_F_ALWAYS_INLINE void
ucp_request_complete_recv(ucp_request_t *req, ucs_status_t status)
{
    ucs_trace_req("completing receive request %p (%p) "UCP_REQUEST_FLAGS_FMT
                  " stag 0x%"PRIx64" len %zu, %s",
                  req, req + 1, UCP_REQUEST_FLAGS_ARG(req->flags),
                  req->recv.info.sender_tag, req->recv.info.length,
                  ucs_status_string(status));
    UCS_PROFILE_REQUEST_EVENT(req, "complete_recv", status);
    ucp_request_complete(req, recv.cb, status, &req->recv.info);
}

/*
 * @return Whether completed.
 *         *req_status if filled with the completion status if completed.
 */
static int UCS_F_ALWAYS_INLINE
ucp_request_try_send(ucp_request_t *req, ucs_status_t *req_status)
{
    ucs_status_t status;

    status = req->send.uct.func(&req->send.uct);
    if (status == UCS_OK) {
        /* Completed the operation */
        *req_status = UCS_OK;
        return 1;
    } else if (status == UCS_INPROGRESS) {
        /* Not completed, but made progress */
        return 0;
    } else if (status != UCS_ERR_NO_RESOURCE) {
        /* Unexpected error */
        ucp_request_complete_send(req, status);
        *req_status = status;
        return 1;
    }

    /* No send resources, try to add to pending queue */
    ucs_assert(status == UCS_ERR_NO_RESOURCE);
    return ucp_request_pending_add(req, req_status);
}

/**
 * Start sending a request.
 *
 * @param [in]  req   Request to start.
 *
 * @return UCS_OK - completed (callback will not be called)
 *         UCS_INPROGRESS - started but not completed
 *         other error - failure
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_request_start_send(ucp_request_t *req)
{
    ucs_status_t status = UCS_ERR_NOT_IMPLEMENTED;
    while (!ucp_request_try_send(req, &status));
    return status;
}

static UCS_F_ALWAYS_INLINE
void ucp_request_send_generic_dt_finish(ucp_request_t *req)
{
    ucp_dt_extended_t *dt;
    if (UCP_DT_IS_GENERIC(req->send.datatype)) {
        dt = ucp_dt_ptr(req->send.datatype);
        ucs_assert(NULL != dt);
        dt->generic.ops.finish(req->send.state.dt.generic.state);
    }
}

static UCS_F_ALWAYS_INLINE
void ucp_request_recv_generic_dt_finish(ucp_request_t *req)
{
    ucp_dt_extended_t *dt;
    if (UCP_DT_IS_GENERIC(req->recv.datatype)) {
        dt = ucp_dt_ptr(req->recv.datatype);
        ucs_assert(NULL != dt);
        dt->generic.ops.finish(req->recv.state.dt.generic.state);
    }
}

static UCS_F_ALWAYS_INLINE void 
ucp_request_wait_uct_comp(ucp_request_t *req)
{
    while (req->send.uct_comp.count > 0) {
        ucp_worker_progress(req->send.ep->worker);
    }
}

static UCS_F_ALWAYS_INLINE int
ucp_request_is_send_buffer_reg(ucp_request_t *req) {
    return req->send.reg_rsc != UCP_NULL_RESOURCE;
}
