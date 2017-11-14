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
ucp_request_complete_tag_recv(ucp_request_t *req, ucs_status_t status)
{
    ucs_trace_req("completing receive request %p (%p) "UCP_REQUEST_FLAGS_FMT
                  " stag 0x%"PRIx64" len %zu, %s",
                  req, req + 1, UCP_REQUEST_FLAGS_ARG(req->flags),
                  req->recv.tag.info.sender_tag, req->recv.tag.info.length,
                  ucs_status_string(status));
    UCS_PROFILE_REQUEST_EVENT(req, "complete_recv", status);
    if (req->flags & UCP_REQUEST_FLAG_BLOCK_OFFLOAD) {
        --req->recv.worker->context->tm.offload.sw_req_count;
    }
    ucp_request_complete(req, recv.tag.cb, status, &req->recv.tag.info);
}

static UCS_F_ALWAYS_INLINE void
ucp_request_complete_stream_recv(ucp_request_t *req,
                                 ucp_ep_ext_stream_t* ep_stream,
                                 ucs_status_t status)
{
    /* dequeue request before complete */
    ucp_request_t *check_req UCS_V_UNUSED =
            ucs_queue_pull_elem_non_empty(&ep_stream->reqs, ucp_request_t,
                                          recv.queue);
    ucs_assert(check_req == req);
    ucs_assert(req->recv.state.offset > 0);
    req->recv.stream.count = req->recv.state.offset;
    ucs_trace_req("completing stream receive request %p (%p) "
                  UCP_REQUEST_FLAGS_FMT" count %zu, %s",
                  req, req + 1, UCP_REQUEST_FLAGS_ARG(req->flags),
                  req->recv.stream.count, ucs_status_string(status));
    UCS_PROFILE_REQUEST_EVENT(req, "complete_recv", status);
    ucp_request_complete(req, recv.stream.cb, status, req->recv.stream.count);
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
ucp_request_send(ucp_request_t *req)
{
    ucs_status_t status = UCS_ERR_NOT_IMPLEMENTED;
    while (!ucp_request_try_send(req, &status));
    return status;
}

static UCS_F_ALWAYS_INLINE
void ucp_request_send_generic_dt_finish(ucp_request_t *req)
{
    ucp_dt_generic_t *dt;
    if (UCP_DT_IS_GENERIC(req->send.datatype)) {
        dt = ucp_dt_generic(req->send.datatype);
        ucs_assert(NULL != dt);
        dt->ops.finish(req->send.state.dt.dt.generic.state);
    }
}

static UCS_F_ALWAYS_INLINE
void ucp_request_recv_generic_dt_finish(ucp_request_t *req)
{
    ucp_dt_generic_t *dt;
    if (UCP_DT_IS_GENERIC(req->recv.datatype)) {
        dt = ucp_dt_generic(req->recv.datatype);
        ucs_assert(NULL != dt);
        dt->ops.finish(req->recv.state.dt.generic.state);
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_request_send_state_init(ucp_request_t *req, size_t dt_count)
{
    ucp_dt_generic_t *dt_gen;
    void             *state_gen;

    VALGRIND_MAKE_MEM_UNDEFINED(&req->send.state.uct_comp.count,
                                sizeof(req->send.state.uct_comp.count));
    VALGRIND_MAKE_MEM_UNDEFINED(&req->send.state.dt.offset,
                                sizeof(req->send.state.dt.offset));

    req->send.state.uct_comp.func = NULL;

    switch (req->send.datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        return;
    case UCP_DATATYPE_IOV:
        req->send.state.dt.dt.iov.iovcnt_offset = 0;
        req->send.state.dt.dt.iov.iov_offset    = 0;
        req->send.state.dt.dt.iov.iovcnt        = dt_count;
        return;
    case UCP_DATATYPE_GENERIC:
        dt_gen    = ucp_dt_generic(req->send.datatype);
        state_gen = dt_gen->ops.start_pack(dt_gen->context, req->send.buffer,
                                           dt_count);
        req->send.state.dt.dt.generic.state = state_gen;
        return;
    default:
        ucs_fatal("Invalid data type");
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_request_send_state_reset(ucp_request_t *req,
                             uct_completion_callback_t comp_cb, unsigned proto)
{
    switch (proto) {
    case UCP_REQUEST_SEND_PROTO_RMA:
        ucs_assert(UCP_DT_IS_CONTIG(req->send.datatype));
        /* Fall through */
    case UCP_REQUEST_SEND_PROTO_RNDV_GET:
        if (UCP_DT_IS_CONTIG(req->send.datatype)) {
            req->send.state.dt.dt.contig.memh = UCT_MEM_HANDLE_NULL;
        }
        /* Fall through */
    case UCP_REQUEST_SEND_PROTO_ZCOPY_AM:
        req->send.state.uct_comp.func       = comp_cb;
        req->send.state.uct_comp.count      = 0;
        /* Fall through */
    case UCP_REQUEST_SEND_PROTO_BCOPY_AM:
        req->send.state.dt.offset           = 0;
        break;
    default:
        ucs_fatal("unknown protocol");
    }

    /* offset is not used for RMA */
    ucs_assert((proto == UCP_REQUEST_SEND_PROTO_RMA) ||
               (req->send.state.dt.offset <= req->send.length));
}

/**
 * Advance state of send request after UCT operation. This function applies
 * @a new_dt_state to @a req request according to @a proto protocol. Also, UCT
 * completion counter will be incremented if @a proto requires it.
 *
 * @param [inout]   req             Send request.
 * @param [in]      new_dt_state    State which was progressed by
 *                                  @ref ucp_dt_pack or @ref ucp_dt_iov_copy_uct.
 * @param [in]      proto           Internal UCP protocol identifier
 *                                  (UCP_REQUEST_SEND_PROTO_*)
 * @param [in]      status          Status of the last UCT operation which
 *                                  progressed @a proto protocol.
 */
static UCS_F_ALWAYS_INLINE void
ucp_request_send_state_advance(ucp_request_t *req,
                               const ucp_dt_state_t *new_dt_state,
                               unsigned proto, ucs_status_t status)
{
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        /* Don't advance after failed operation in order to continue on next try
         * from last valid point.
         */
        return;
    }

    switch (proto) {
    case UCP_REQUEST_SEND_PROTO_RMA:
        if (status == UCS_INPROGRESS) {
            ++req->send.state.uct_comp.count;
        }
        break;
    case UCP_REQUEST_SEND_PROTO_ZCOPY_AM:
        /* Fall through */
    case UCP_REQUEST_SEND_PROTO_RNDV_GET:
        if (status == UCS_INPROGRESS) {
            ++req->send.state.uct_comp.count;
        }
        /* Fall through */
    case UCP_REQUEST_SEND_PROTO_BCOPY_AM:
        ucs_assert(new_dt_state != NULL);
        if (UCP_DT_IS_CONTIG(req->send.datatype)) {
            req->send.state.dt.offset = new_dt_state->offset;
        } else {
            req->send.state.dt        = *new_dt_state;
        }
        break;
    default:
        ucs_fatal("unknown protocol");
    }

    /* offset is not used for RMA */
    ucs_assert((proto == UCP_REQUEST_SEND_PROTO_RMA) ||
               (req->send.state.dt.offset <= req->send.length));
}

/* Fast-forward to data end */
static UCS_F_ALWAYS_INLINE void
ucp_request_send_state_ff(ucp_request_t *req, ucs_status_t status)
{
    if (req->send.state.uct_comp.func) {
        req->send.state.dt.offset = req->send.length;
        if (status == UCS_ERR_CANCELED) {
            req->send.state.uct_comp.count = 0;
        }
        req->send.state.uct_comp.func(&req->send.state.uct_comp, status);
    } else {
        ucp_request_complete_send(req, status);
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_request_wait_uct_comp(ucp_request_t *req)
{
    while (req->send.state.uct_comp.count > 0) {
        ucp_worker_progress(req->send.ep->worker);
    }
}

static UCS_F_ALWAYS_INLINE int
ucp_request_is_send_buffer_reg(ucp_request_t *req) {
    return req->send.reg_rsc != UCP_NULL_RESOURCE;
}

static UCS_F_ALWAYS_INLINE void ucp_request_send_tag_stat(ucp_request_t *req)
{
    if (req->flags & UCP_REQUEST_FLAG_RNDV) {
        UCP_EP_STAT_TAG_OP(req->send.ep, RNDV);
    } else if (req->flags & UCP_REQUEST_FLAG_SYNC) {
        UCP_EP_STAT_TAG_OP(req->send.ep, EAGER_SYNC);
    } else {
        UCP_EP_STAT_TAG_OP(req->send.ep, EAGER);
    }
}
