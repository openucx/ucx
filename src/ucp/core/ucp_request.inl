/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_REQUEST_INL_
#define UCP_REQUEST_INL_

#include "ucp_request.h"
#include "ucp_worker.h"
#include "ucp_ep.inl"

#include <ucp/core/ucp_worker.h>
#include <ucp/dt/dt.h>
#include <ucs/debug/profile.h>
#include <ucs/datastruct/mpool.inl>
#include <ucp/dt/dt.inl>
#include <inttypes.h>


#define UCP_REQUEST_FLAGS_FMT \
    "%c%c%c%c%c%c%c"

#define UCP_REQUEST_FLAGS_ARG(_flags) \
    (((_flags) & UCP_REQUEST_FLAG_COMPLETED)       ? 'd' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_RELEASED)        ? 'f' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_EXPECTED)        ? 'e' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_LOCAL_COMPLETED) ? 'L' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_CALLBACK)        ? 'c' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_RECV)            ? 'r' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_SYNC)            ? 's' : '-')

#define UCP_RECV_DESC_FMT \
    "rdesc %p %c%c%c%c%c%c len %u+%u"

#define UCP_RECV_DESC_ARG(_rdesc) \
    (_rdesc), \
    (((_rdesc)->flags & UCP_RECV_DESC_FLAG_UCT_DESC)      ? 't' : '-'), \
    (((_rdesc)->flags & UCP_RECV_DESC_FLAG_EAGER)         ? 'e' : '-'), \
    (((_rdesc)->flags & UCP_RECV_DESC_FLAG_EAGER_ONLY)    ? 'o' : '-'), \
    (((_rdesc)->flags & UCP_RECV_DESC_FLAG_EAGER_SYNC)    ? 's' : '-'), \
    (((_rdesc)->flags & UCP_RECV_DESC_FLAG_EAGER_OFFLOAD) ? 'f' : '-'), \
    (((_rdesc)->flags & UCP_RECV_DESC_FLAG_RNDV)          ? 'r' : '-'), \
    (_rdesc)->payload_offset, \
    ((_rdesc)->length - (_rdesc)->payload_offset)


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
    ucp_request_complete(req, recv.tag.cb, status, &req->recv.tag.info);
}

static UCS_F_ALWAYS_INLINE void
ucp_request_complete_stream_recv(ucp_request_t *req,
                                 ucp_ep_ext_stream_t* ep_stream,
                                 ucs_status_t status)
{
    /* dequeue request before complete */
    ucp_request_t *check_req UCS_V_UNUSED =
            ucs_queue_pull_elem_non_empty(&ep_stream->match_q, ucp_request_t,
                                          recv.queue);
    ucs_assert(check_req               == req);
    ucs_assert(req->recv.stream.offset >  0);

    req->recv.stream.length = req->recv.stream.offset;
    ucs_trace_req("completing stream receive request %p (%p) "
                  UCP_REQUEST_FLAGS_FMT" count %zu, %s",
                  req, req + 1, UCP_REQUEST_FLAGS_ARG(req->flags),
                  req->recv.stream.length, ucs_status_string(status));
    UCS_PROFILE_REQUEST_EVENT(req, "complete_recv", status);
    ucp_request_complete(req, recv.stream.cb, status, req->recv.stream.length);
}

static UCS_F_ALWAYS_INLINE int
ucp_request_can_complete_stream_recv(ucp_request_t *req)
{
    /* NOTE: first check is needed to avoid heavy "%" operation if request is
     *       completely filled */
    if (req->recv.stream.offset == req->recv.length) {
        return 1;
    }

    /* 0-length stream recv is meaningless if this was not requested explicitely */
    if (req->recv.stream.offset == 0) {
        return 0;
    }

    if (ucs_likely(UCP_DT_IS_CONTIG(req->recv.datatype))) {
        return req->recv.stream.offset %
               ucp_contig_dt_elem_size(req->recv.datatype) == 0;
    }

    /* Currently, all data types except contig has granularity 1 byte */
    return 1;
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
ucp_request_send_state_init(ucp_request_t *req, ucp_datatype_t datatype,
                            size_t dt_count)
{
    ucp_dt_generic_t *dt_gen;
    void             *state_gen;

    VALGRIND_MAKE_MEM_UNDEFINED(&req->send.state.uct_comp.count,
                                sizeof(req->send.state.uct_comp.count));
    VALGRIND_MAKE_MEM_UNDEFINED(&req->send.state.dt.offset,
                                sizeof(req->send.state.dt.offset));

    req->send.state.uct_comp.func = NULL;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        req->send.state.dt.dt.contig.md_map     = 0;
        return;
    case UCP_DATATYPE_IOV:
        req->send.state.dt.dt.iov.iovcnt_offset = 0;
        req->send.state.dt.dt.iov.iov_offset    = 0;
        req->send.state.dt.dt.iov.iovcnt        = dt_count;
        req->send.state.dt.dt.iov.dt_reg        = NULL;
        return;
    case UCP_DATATYPE_GENERIC:
        dt_gen    = ucp_dt_generic(datatype);
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
    case UCP_REQUEST_SEND_PROTO_RNDV_PUT:
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
    case UCP_REQUEST_SEND_PROTO_RNDV_PUT:
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
        req->send.state.uct_comp.count = 0;
        req->send.state.uct_comp.func(&req->send.state.uct_comp, status);
    } else {
        ucp_request_complete_send(req, status);
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_request_send_buffer_reg(ucp_request_t *req, ucp_md_map_t md_map)
{
    return ucp_request_memory_reg(req->send.ep->worker->context, md_map,
                                  (void*)req->send.buffer, req->send.length,
                                  req->send.datatype, &req->send.state.dt, req);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_request_send_buffer_reg_lane(ucp_request_t *req, ucp_lane_index_t lane)
{
    ucp_md_map_t md_map = UCS_BIT(ucp_ep_md_index(req->send.ep, lane));
    return ucp_request_send_buffer_reg(req, md_map);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_send_request_add_reg_lane(ucp_request_t *req, ucp_lane_index_t lane)
{
    /* add new lane to registration map */
    ucp_md_map_t md_map = UCS_BIT(ucp_ep_md_index(req->send.ep, lane)) |
                          req->send.state.dt.dt.contig.md_map;
    ucs_assert(ucs_count_one_bits(md_map) <= UCP_MAX_OP_MDS);
    return ucp_request_send_buffer_reg(req, md_map);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_request_recv_buffer_reg(ucp_request_t *req, ucp_md_map_t md_map)
{
    return ucp_request_memory_reg(req->recv.worker->context, md_map,
                                  req->recv.buffer, req->recv.length,
                                  req->recv.datatype, &req->recv.state, req);
}

static UCS_F_ALWAYS_INLINE void ucp_request_send_buffer_dereg(ucp_request_t *req)
{
    ucp_request_memory_dereg(req->send.ep->worker->context, req->send.datatype,
                             &req->send.state.dt, req);
}

static UCS_F_ALWAYS_INLINE void ucp_request_recv_buffer_dereg(ucp_request_t *req)
{
    ucp_request_memory_dereg(req->recv.worker->context, req->recv.datatype,
                             &req->recv.state, req);
}

static UCS_F_ALWAYS_INLINE void
ucp_request_wait_uct_comp(ucp_request_t *req)
{
    while (req->send.state.uct_comp.count > 0) {
        ucp_worker_progress(req->send.ep->worker);
    }
}

/**
 * Unpack receive data to a request
 *
 * req - receive request
 * data - data to unpack
 * length -
 * offset - offset of received data within the request, for OOO fragments
 *
 *
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_request_recv_data_unpack(ucp_request_t *req, const void *data,
                             size_t length, size_t offset, int last)
{
    ucp_dt_generic_t *dt_gen;
    ucs_status_t status;

    ucs_assert(req->status == UCS_OK);

    ucp_trace_req(req, "unpack recv_data req_len %zu data_len %zu offset %zu last: %s",
                  req->recv.length, length, offset, last ? "yes" : "no");

    if (ucs_unlikely((length + offset) > req->recv.length)) {
        ucs_debug("message truncated: recv_length %zu offset %zu buffer_size %zu",
                  length, offset, req->recv.length);
        if (UCP_DT_IS_GENERIC(req->recv.datatype)) {
            dt_gen = ucp_dt_generic(req->recv.datatype);
            UCS_PROFILE_NAMED_CALL_VOID("dt_finish", dt_gen->ops.finish,
                                         req->recv.state.dt.generic.state);
        }
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    switch (req->recv.datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        if (ucs_likely(UCP_MEM_IS_HOST(req->recv.mem_type))) {
            UCS_PROFILE_NAMED_CALL("memcpy_recv", memcpy, req->recv.buffer + offset,
                                   data, length);
        } else {
            ucp_mem_type_unpack(req->recv.worker, req->recv.buffer + offset,
                                data, length, req->recv.mem_type);
        }
        return UCS_OK;;

    case UCP_DATATYPE_IOV:
        if (offset != req->recv.state.offset) {
            ucp_dt_iov_seek(req->recv.buffer, req->recv.state.dt.iov.iovcnt,
                            offset - req->recv.state.offset,
                            &req->recv.state.dt.iov.iov_offset,
                            &req->recv.state.dt.iov.iovcnt_offset);
            req->recv.state.offset = offset;
        }
        UCS_PROFILE_CALL(ucp_dt_iov_scatter, req->recv.buffer,
                         req->recv.state.dt.iov.iovcnt, data, length,
                         &req->recv.state.dt.iov.iov_offset,
                         &req->recv.state.dt.iov.iovcnt_offset);
        req->recv.state.offset += length;
        return UCS_OK;

    case UCP_DATATYPE_GENERIC:
        dt_gen = ucp_dt_generic(req->recv.datatype);
        status = UCS_PROFILE_NAMED_CALL("dt_unpack", dt_gen->ops.unpack,
                                        req->recv.state.dt.generic.state,
                                        offset, data, length);
        if (last || (status != UCS_OK)) {
            UCS_PROFILE_NAMED_CALL_VOID("dt_finish", dt_gen->ops.finish,
                                        req->recv.state.dt.generic.state);
        }
        return status;

    default:
        ucs_fatal("unexpected datatype=%lx", req->recv.datatype);
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_recv_desc_init(ucp_worker_h worker, void *data, size_t length,
                   unsigned am_flags, uint16_t hdr_len, uint16_t rdesc_flags,
                   ucp_recv_desc_t **rdesc_p)
{
    ucp_recv_desc_t *rdesc;
    ucs_status_t status;

    if (ucs_unlikely(am_flags & UCT_CB_PARAM_FLAG_DESC)) {
        /* slowpath */
        rdesc        = (ucp_recv_desc_t *)data - 1;
        rdesc->flags = rdesc_flags | UCP_RECV_DESC_FLAG_UCT_DESC;
        status       = UCS_INPROGRESS;
    } else {
        rdesc = (ucp_recv_desc_t*)ucs_mpool_get_inline(&worker->am_mp);
        if (rdesc == NULL) {
            ucs_error("ucp recv descriptor is not allocated");
            return UCS_ERR_NO_MEMORY;
        }

        rdesc->flags = rdesc_flags;
        memcpy(rdesc + 1, data, length);
        status = UCS_OK;
    }

    rdesc->length         = length;
    rdesc->payload_offset = hdr_len;
    *rdesc_p              = rdesc;
    return status;
}

static UCS_F_ALWAYS_INLINE ucp_lane_index_t
ucp_send_request_get_next_am_bw_lane(ucp_request_t *req)
{
    ucp_lane_index_t lane;

    /* at least one lane must be initialized */
    ucs_assert(ucp_ep_config(req->send.ep)->key.am_bw_lanes[0] != UCP_NULL_LANE);

    lane = (req->send.tag.am_bw_index >= UCP_MAX_LANES) ?
           UCP_NULL_LANE :
           ucp_ep_config(req->send.ep)->key.am_bw_lanes[req->send.tag.am_bw_index];
    if (lane != UCP_NULL_LANE) {
        req->send.tag.am_bw_index++;
        return lane;
    } else {
        req->send.tag.am_bw_index = 1;
        return ucp_ep_config(req->send.ep)->key.am_bw_lanes[0];
    }
}

#endif
