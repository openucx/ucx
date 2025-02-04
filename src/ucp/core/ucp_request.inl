/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_REQUEST_INL_
#define UCP_REQUEST_INL_

#include "ucp_request.h"
#include "ucp_worker.h"
#include "ucp_ep.inl"
#include "ucp_mm.inl"

#include <ucp/dt/dt.h>
#include <ucs/profile/profile.h>
#include <ucs/datastruct/mpool.inl>
#include <ucs/datastruct/mpool_set.inl>
#include <ucs/datastruct/ptr_map.inl>
#include <ucs/debug/debug_int.h>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/dt/dt.inl>
#include <inttypes.h>


UCS_PTR_MAP_IMPL(request, 0);


#define UCP_REQUEST_FLAGS_FMT \
    "%c%c%c%c%c%c"

#define UCP_REQUEST_FLAGS_ARG(_flags) \
    (((_flags) & UCP_REQUEST_FLAG_COMPLETED)       ? 'd' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_RELEASED)        ? 'f' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_SYNC_LOCAL_COMPLETED) ? 'L' : '-'), \
    (((_flags) & UCP_REQUEST_FLAG_CALLBACK)        ? 'c' : '-'), \
    (((_flags) & (UCP_REQUEST_FLAG_RECV_TAG | \
                  UCP_REQUEST_FLAG_RECV_AM))       ? 'r' : '-'), \
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
            ucs_trace_req("allocated request %p", _req); \
            ucp_request_reset_internal(_req, _worker); \
            UCS_PROFILE_REQUEST_NEW(_req, "ucp_request", 0); \
        } \
        _req; \
    })

#define ucp_request_complete(_req, _cb, _status, ...) \
    { \
        /* NOTE: external request can't have RELEASE flag and we */ \
        /* will never put it into mpool */ \
        uint32_t _flags; \
        \
        ucs_assert(!((_req)->flags & UCP_REQUEST_FLAG_COMPLETED)); \
        ucs_assert((_status) != UCS_INPROGRESS); \
        \
        _flags         = ((_req)->flags |= UCP_REQUEST_FLAG_COMPLETED); \
        (_req)->status = (_status); \
        \
        ucp_request_id_check(_req, ==, UCS_PTR_MAP_KEY_INVALID); \
        \
        if (ucs_likely((_req)->flags & UCP_REQUEST_FLAG_CALLBACK)) { \
            (_req)->_cb((_req) + 1, (_status), ## __VA_ARGS__); \
        } \
        if (ucs_unlikely(_flags & UCP_REQUEST_FLAG_RELEASED)) { \
            ucp_request_put(_req); \
        } \
    }


#define ucp_request_set_callback(_req, _cb, _cb_value) \
    { \
        (_req)->_cb       = _cb_value; \
        (_req)->flags    |= UCP_REQUEST_FLAG_CALLBACK; \
    }


#define ucp_request_set_user_callback(_req, _cb, _cb_value, _user_data) \
    { \
        ucp_request_set_callback(_req, _cb, _cb_value); \
        (_req)->user_data = _user_data; \
        ucs_trace_data("request %p %s set to %p, user data: %p", \
                       _req, #_cb, _cb_value, _user_data); \
    }


#define ucp_request_get_param(_worker, _param, _failed) \
    ({ \
        ucp_request_t *__req; \
        if (!((_param)->op_attr_mask & UCP_OP_ATTR_FIELD_REQUEST)) { \
            __req = ucp_request_get(_worker); \
            if (ucs_unlikely((__req) == NULL)) { \
                _failed; \
            } \
        } else { \
            __req = ((ucp_request_t*)(_param)->request) - 1; \
            ucp_request_id_reset(__req); \
        } \
        __req; \
    })


#define ucp_request_id_check(_req, _cmp, _id) \
    ucs_assertv((_req)->id _cmp (_id), "req=%p req->id=0x%" PRIx64 " id=0x%" \
                PRIx64, \
                (_req), (_req)->id, (_id))


#define ucp_request_put_param(_param, _req) \
    if (!((_param)->op_attr_mask & UCP_OP_ATTR_FIELD_REQUEST)) { \
        ucp_request_put(_req); \
    } else { \
        ucp_request_id_check(_req, ==, UCS_PTR_MAP_KEY_INVALID); \
    }


#define ucp_request_cb_param(_param, _req, _cb, ...) \
    if ((_param)->op_attr_mask & UCP_OP_ATTR_FIELD_CALLBACK) { \
        (_param)->cb._cb((_req) + 1, (_req)->status, ##__VA_ARGS__, \
                         (_param)->user_data); \
    }


#define ucp_request_prevent_imm_cmpl(_param, _req, _cb, ...) \
    ({ \
        ucp_request_cb_param(_param, _req, _cb, ##__VA_ARGS__); \
        ucs_trace_req("request %p completed, but immediate completion is " \
                      "prohibited, status %s", \
                      _req, ucs_status_string((_req)->status)); \
        (_req) + 1; \
    })


#define ucp_request_imm_cmpl_param(_param, _req, _cb, ...) \
    if ((_param)->op_attr_mask & UCP_OP_ATTR_FLAG_NO_IMM_CMPL) { \
        return ucp_request_prevent_imm_cmpl(_param, _req, _cb, ##__VA_ARGS__); \
    } \
    { \
        ucs_status_t _status = (_req)->status; \
        ucp_request_put_param(_param, _req); \
        return UCS_STATUS_PTR(_status); \
    }


#define UCP_REQUEST_PARAM_FIELD(_param, _flag, _field, _default_value) \
    ((_param)->op_attr_mask & UCP_OP_ATTR_FIELD_##_flag) ? \
            (_param)->_field : (_default_value)


#define ucp_request_set_callback_param(_param, _param_cb, _req, _req_cb) \
    if ((_param)->op_attr_mask & UCP_OP_ATTR_FIELD_CALLBACK) { \
        ucp_request_set_user_callback(_req, _req_cb.cb, \
                                      (_param)->cb._param_cb, \
                                      ucp_request_param_user_data(_param)); \
    }


#define ucp_request_set_send_callback_param(_param, _req, _cb) \
    ucp_request_set_callback_param(_param, send, _req, _cb)


#define ucp_request_send_check_status(_status, _ret, _done) \
    if (ucs_likely((_status) != UCS_ERR_NO_RESOURCE)) { \
        _ret = UCS_STATUS_PTR(_status); /* UCS_OK also goes here */ \
        _done; \
    }


#define UCP_REQUEST_CHECK_PARAM(_param) \
    if (ENABLE_PARAMS_CHECK) { \
        if (((_param)->op_attr_mask & UCP_OP_ATTR_FIELD_MEMORY_TYPE) && \
            ((_param)->memory_type > UCS_MEMORY_TYPE_LAST)) { \
            ucs_error("invalid memory type parameter: %d", \
                      (_param)->memory_type); \
            return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM); \
        } \
        \
        if (ucs_test_all_flags((_param)->op_attr_mask, \
                               (UCP_OP_ATTR_FLAG_FAST_CMPL | \
                                UCP_OP_ATTR_FLAG_MULTI_SEND))) { \
            ucs_error("UCP_OP_ATTR_FLAG_FAST_CMPL and " \
                      "UCP_OP_ATTR_FLAG_MULTI_SEND are mutually exclusive"); \
            return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM); \
        } \
    }


#if UCS_ENABLE_ASSERT
#  define UCP_REQUEST_RESET(_req) \
    (_req)->send.uct.func = \
            (uct_pending_callback_t)ucs_empty_function_do_assert; \
    (_req)->send.state.uct_comp.func  = \
            (uct_completion_callback_t)ucs_empty_function_do_assert_void; \
    (_req)->send.state.uct_comp.count = 0;
#else /* UCS_ENABLE_ASSERT */
#  define UCP_REQUEST_RESET(_req)
#endif /* UCS_ENABLE_ASSERT */


static UCS_F_ALWAYS_INLINE void ucp_request_id_reset(ucp_request_t *req)
{
    req->id = UCS_PTR_MAP_KEY_INVALID;
}

static UCS_F_ALWAYS_INLINE void
ucp_request_reset_internal(ucp_request_t *req, ucp_worker_h worker)
{
    VALGRIND_MAKE_MEM_DEFINED(&req->id, sizeof(req->id));
    VALGRIND_MAKE_MEM_DEFINED(req + 1, worker->context->config.request.size);
    ucp_request_id_check(req, ==, UCS_PTR_MAP_KEY_INVALID);
}

static UCS_F_ALWAYS_INLINE void
ucp_request_put(ucp_request_t *req)
{
    ucs_trace_req("put request %p", req);
    ucp_request_id_check(req, ==, UCS_PTR_MAP_KEY_INVALID);
    UCS_PROFILE_REQUEST_FREE(req);
    UCP_REQUEST_RESET(req);
    ucs_mpool_put_inline(req);
}

static UCS_F_ALWAYS_INLINE void
ucp_request_complete_send(ucp_request_t *req, ucs_status_t status)
{
    ucs_trace_req("completing send request %p (%p) " UCP_REQUEST_FLAGS_FMT
                  " %s",
                  req, req + 1, UCP_REQUEST_FLAGS_ARG(req->flags),
                  ucs_status_string(status));
    UCS_PROFILE_REQUEST_EVENT(req, "complete_send", status);
    /* Coverity wrongly resolves completion callback function to
     * 'ucp_cm_client_connect_progress'/'ucp_cm_server_conn_request_progress'
     */
    /* coverity[offset_free] */
    ucp_request_complete(req, send.cb, status, req->user_data);
}

static UCS_F_ALWAYS_INLINE void
ucp_request_complete_tag_recv(ucp_request_t *req, ucs_status_t status)
{
    ucs_trace_req("completing receive request %p (%p) " UCP_REQUEST_FLAGS_FMT
                  " stag 0x%" PRIx64" len %zu, %s",
                  req, req + 1, UCP_REQUEST_FLAGS_ARG(req->flags),
                  req->recv.tag.info.sender_tag, req->recv.tag.info.length,
                  ucs_status_string(status));
    UCS_PROFILE_REQUEST_EVENT(req, "complete_tag_recv", status);
    /* coverity[address_free] */
    /* coverity[offset_free] */
    ucp_request_complete(req, recv.tag.cb, status, &req->recv.tag.info,
                         req->user_data);
}

static UCS_F_ALWAYS_INLINE ucp_request_t*
ucp_request_mem_alloc(const char *name)
{
    ucp_request_t *req = (ucp_request_t*)ucs_malloc(sizeof(*req), name);

    if (ucs_unlikely(req == NULL)) {
        return NULL;
    }

    ucs_trace_req("allocated request %p (%s)", req, name);
    ucp_request_id_reset(req);
    UCS_PROFILE_REQUEST_NEW(req, "ucp_request", 0);

    return req;
}

static UCS_F_ALWAYS_INLINE void ucp_request_mem_free(ucp_request_t *req)
{
    UCS_PROFILE_REQUEST_FREE(req);
    ucp_request_id_check(req, ==, UCS_PTR_MAP_KEY_INVALID);
    ucs_trace_req("freed request %p", req);
    ucs_free(req);
}

/*
 * @return Whether completed.
 *         *req_status if filled with the completion status if completed.
 */
static int UCS_F_ALWAYS_INLINE ucp_request_try_send(ucp_request_t *req)
{
    ucs_status_t status;

    /* coverity wrongly resolves (*req).send.uct.func to test_uct_pending::pending_send_op_ok */
    /* coverity[address_free] */
    status = req->send.uct.func(&req->send.uct);
    if (status == UCS_OK) {
        /* Completed the operation, error also goes here */
        return 1;
    } else if (status == UCS_INPROGRESS) {
        /* Not completed, but made progress */
        return 0;
    } else if (status == UCS_ERR_NO_RESOURCE) {
        /* No send resources, try to add to pending queue */
        return ucp_request_pending_add(req);
    }

    ucs_fatal("unexpected error: %s", ucs_status_string(status));
}

/**
 * Start sending a request.
 *
 * @param [in]  req             Request to start.
 * */
static UCS_F_ALWAYS_INLINE void
ucp_request_send(ucp_request_t *req)
{
    while (!ucp_request_try_send(req));
}

static UCS_F_ALWAYS_INLINE
void ucp_request_send_generic_dt_finish(ucp_request_t *req)
{
    ucp_dt_generic_t *dt;
    if (UCP_DT_IS_GENERIC(req->send.datatype)) {
        dt = ucp_dt_to_generic(req->send.datatype);
        ucs_assert(NULL != dt);
        dt->ops.finish(req->send.state.dt.dt.generic.state);
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_request_send_state_init(ucp_request_t *req, ucp_datatype_t datatype,
                            size_t dt_count)
{
    ucp_dt_generic_t *dt_gen;
    void             *state_gen;

    VALGRIND_MAKE_MEM_UNDEFINED(&req->send.state.uct_comp,
                                sizeof(req->send.state.uct_comp));
    VALGRIND_MAKE_MEM_UNDEFINED(&req->send.state.dt_iter.offset,
                                sizeof(req->send.state.dt_iter.offset));

    req->send.state.uct_comp.func = NULL;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        req->send.state.dt.dt.contig.memh       = NULL;
        return;
    case UCP_DATATYPE_IOV:
        req->send.state.dt.dt.iov.iovcnt_offset = 0;
        req->send.state.dt.dt.iov.iov_offset    = 0;
        req->send.state.dt.dt.iov.iovcnt        = dt_count;
        req->send.state.dt.dt.iov.memhs         = NULL;
        return;
    case UCP_DATATYPE_GENERIC:
        dt_gen    = ucp_dt_to_generic(datatype);
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
        req->send.state.uct_comp.count  = 0;
        req->send.state.uct_comp.status = UCS_OK;
        /* Fall through */
    case UCP_REQUEST_SEND_PROTO_BCOPY_AM:
        /* Always set completion function to make sure this value is initialized
         * when doing send fast-forwarding in ucp_request_send_state_ff() */
        req->send.state.uct_comp.func = comp_cb;
        req->send.state.dt.offset     = 0;

        if (proto == UCP_REQUEST_SEND_PROTO_BCOPY_AM) {
            ucs_assertv(comp_cb == NULL,
                        "completion function for AM Bcopy protocol must be NULL"
                        " instead of %s",
                        ucs_debug_get_symbol_name((void*)comp_cb));
        }
        break;
    default:
        ucs_fatal("unknown protocol");
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_send_request_invoke_uct_completion(ucp_request_t *req)
{
    if (req->send.state.uct_comp.count == 0) {
        req->send.state.uct_comp.func(&req->send.state.uct_comp);
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
    if (status == UCS_ERR_NO_RESOURCE) {
        /* Don't advance in order to continue on next try from last valid point
         */
        return;
    }

    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        ucp_request_send_state_ff(req, status);
        return;
    }

    switch (proto) {
    case UCP_REQUEST_SEND_PROTO_ZCOPY_AM:
    case UCP_REQUEST_SEND_PROTO_RNDV_GET:
    case UCP_REQUEST_SEND_PROTO_RNDV_PUT:
        ucs_assert(new_dt_state != NULL);
        if (UCP_DT_IS_CONTIG(req->send.datatype)) {
            /* cppcheck-suppress nullPointer */
            req->send.state.dt.offset = new_dt_state->offset;
        } else {
            /* cppcheck-suppress nullPointer */
            req->send.state.dt        = *new_dt_state;
        }
        /* Fall through */
    case UCP_REQUEST_SEND_PROTO_RMA:
        if (status == UCS_INPROGRESS) {
            ++req->send.state.uct_comp.count;
        }
        break;
    default:
        ucs_fatal("unknown protocol %d", proto);
    }

    /* offset is not used for RMA */
    ucs_assert((proto == UCP_REQUEST_SEND_PROTO_RMA) ||
               (req->send.state.dt.offset <= req->send.length));
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_request_send_buffer_reg(ucp_request_t *req, ucp_md_map_t md_map,
                            unsigned uct_flags)
{
    return ucp_request_memory_reg(req->send.ep->worker->context, md_map,
                                  (void*)req->send.buffer, req->send.length,
                                  req->send.datatype, &req->send.state.dt,
                                  (ucs_memory_type_t)req->send.mem_type, req,
                                  uct_flags);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_request_send_reg_lane(ucp_request_t *req, ucp_lane_index_t lane)
{
    ucp_md_map_t md_map;

    if (!(ucp_ep_md_attr(req->send.ep, lane)->flags & UCT_MD_FLAG_NEED_MEMH)) {
        return UCS_OK;
    }

    ucs_assert(ucp_ep_md_attr(req->send.ep, lane)->flags & UCT_MD_FLAG_REG);
    md_map = UCS_BIT(ucp_ep_md_index(req->send.ep, lane));
    return ucp_request_send_buffer_reg(req, md_map, 0);
}

static UCS_F_ALWAYS_INLINE void ucp_request_send_buffer_dereg(ucp_request_t *req)
{
    ucp_request_memory_dereg(req->send.datatype, &req->send.state.dt, req);
}

/* Returns whether user-provided memory handle can be used. If the result is 0,
   *status_p is set to the error code. */
static UCS_F_ALWAYS_INLINE int
ucp_request_is_user_memh_valid(ucp_request_t *req,
                               const ucp_request_param_t *param, void *buffer,
                               size_t length, ucp_datatype_t datatype,
                               ucs_memory_type_t mem_type,
                               ucs_status_t *status_p)
{
    /* User-provided memh supported only on contig type with proto_v1 */
    if (!(param->op_attr_mask & UCP_OP_ATTR_FIELD_MEMH) ||
        !UCP_DT_IS_CONTIG(datatype)) {
        *status_p = UCS_OK;
        return 0;
    }

    if (ENABLE_PARAMS_CHECK &&
        ((param->memh == NULL) || (buffer < ucp_memh_address(param->memh)) ||
         (UCS_PTR_BYTE_OFFSET(buffer, length) >
          UCS_PTR_BYTE_OFFSET(ucp_memh_address(param->memh),
                              ucp_memh_length(param->memh))) ||
         (param->memh->mem_type != mem_type))) {
        ucs_error("req %p: mismatched memory handle [buffer %p length %zu %s]"
                  " memh %p [address %p length %zu %s]",
                  req, buffer, length, ucs_memory_type_names[mem_type],
                  param->memh, ucp_memh_address(param->memh),
                  ucp_memh_length(param->memh),
                  ucs_memory_type_names[param->memh->mem_type]);
        *status_p = UCS_ERR_INVALID_PARAM;
        return 0;
    }

    ucs_assert(param->memh != NULL); /* For Coverity */
    return 1;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_send_request_set_user_memh(ucp_request_t *req, ucp_md_map_t md_map,
                               const ucp_request_param_t *param)
{
    ucs_status_t status;

    if (!ucp_request_is_user_memh_valid(req, param, req->send.buffer,
                                        req->send.length, req->send.datatype,
                                        (ucs_memory_type_t)req->send.mem_type,
                                        &status)) {
        return status;
    }

    /* req->send.state.dt should not be used with protov2 */
    ucs_assert(!req->send.ep->worker->context->config.ext.proto_enable);

    req->send.state.dt.dt.contig.memh = param->memh;
    return UCS_OK;
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
                             size_t length, size_t offset, int dereg, int last)
{
    ucs_status_t status;

    ucs_assertv(req->status == UCS_OK, "status: %s",
                ucs_status_string(req->status));

    ucp_trace_req(
            req, "unpack recv_data req_len %zu data_len %zu offset %zu%s",
            req->recv.dt_iter.length, length, offset, last ? " last" : "");

    if (ucs_unlikely((length + offset) > req->recv.dt_iter.length)) {
        return ucp_request_recv_msg_truncated(req, length, offset);
    }

    status = ucp_datatype_iter_unpack(&req->recv.dt_iter, req->recv.worker,
                                      length, offset, data);
    if (last || (status != UCS_OK)) {
        ucp_datatype_iter_cleanup(&req->recv.dt_iter, dereg, UCP_DT_MASK_ALL);
    }

    return status;
}

static UCS_F_ALWAYS_INLINE void
ucp_recv_desc_set_name(ucp_recv_desc_t *rdesc, const char *name)
{
#if ENABLE_DEBUG_DATA
    rdesc->name = name;
#endif
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_recv_desc_init(ucp_worker_h worker, void *data, size_t length,
                   int data_offset, unsigned am_flags, uint16_t hdr_len,
                   uint16_t rdesc_flags, int priv_length, size_t alignment,
                   const char *name, ucp_recv_desc_t **rdesc_p)
{
    ucp_recv_desc_t *rdesc;
    void *data_hdr;
    ucs_status_t status;
    size_t padding;

    if (ucs_unlikely(am_flags & UCT_CB_PARAM_FLAG_DESC)) {
        /* slowpath */
        ucs_assert(priv_length <= UCP_WORKER_HEADROOM_PRIV_SIZE);
        data_hdr                   = UCS_PTR_BYTE_OFFSET(data, -data_offset);
        rdesc                      = (ucp_recv_desc_t *)data_hdr - 1;
        rdesc->flags               = rdesc_flags | UCP_RECV_DESC_FLAG_UCT_DESC;
        rdesc->release_desc_offset = UCP_WORKER_HEADROOM_PRIV_SIZE - priv_length;
        status                     = UCS_INPROGRESS;
    } else {
        rdesc = (ucp_recv_desc_t*)ucs_mpool_set_get_inline(&worker->am_mps,
                                                           length);
        if (rdesc == NULL) {
            *rdesc_p = NULL; /* To suppress compiler warning */
            ucs_error("ucp recv descriptor is not allocated");
            return UCS_ERR_NO_MEMORY;
        }

        padding = ucs_padding((uintptr_t)(rdesc + 1), alignment);
        rdesc   = (ucp_recv_desc_t*)UCS_PTR_BYTE_OFFSET(rdesc, padding);
        rdesc->release_desc_offset = padding;

        /* No need to initialize rdesc->priv_length here, because it is only
         * needed for releasing UCT descriptor. */
        rdesc->flags = rdesc_flags;
        status       = UCS_OK;
        memcpy(UCS_PTR_BYTE_OFFSET(rdesc + 1, data_offset), data, length);
    }

    ucp_recv_desc_set_name(rdesc, name);
    rdesc->length         = length + data_offset;
    rdesc->payload_offset = hdr_len;
    *rdesc_p              = rdesc;
    return status;
}

static UCS_F_ALWAYS_INLINE void
ucp_recv_desc_release(ucp_recv_desc_t *rdesc)
{
    void *desc = UCS_PTR_BYTE_OFFSET(rdesc, -rdesc->release_desc_offset);

    ucs_trace_req("release receive descriptor %p", rdesc);

    if (ucs_unlikely(rdesc->flags & UCP_RECV_DESC_FLAG_UCT_DESC)) {
        /* uct desc is slowpath */
        uct_iface_release_desc(desc);
    } else {
        ucs_mpool_set_put_inline(desc);
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_request_complete_am_recv(ucp_request_t *req, ucs_status_t status)
{
    ucs_trace_req("completing AM receive request %p (%p) " UCP_REQUEST_FLAGS_FMT
                  " length %zu, %s",
                  req, req + 1, UCP_REQUEST_FLAGS_ARG(req->flags),
                  req->recv.dt_iter.length, ucs_status_string(status));
    UCS_PROFILE_REQUEST_EVENT(req, "complete_am_recv", status);

    if (req->recv.am.desc->flags & UCP_RECV_DESC_FLAG_AM_CB_INPROGRESS) {
        /* Descriptor is not initialized by UCT yet, therefore can not call
         * ucp_recv_desc_release() for it. Clear the flag to let UCT AM
         * callback know that this descriptor is not needed anymore.
         */
        req->recv.am.desc->flags &= ~UCP_RECV_DESC_FLAG_AM_CB_INPROGRESS;
    } else {
        ucp_recv_desc_release(req->recv.am.desc);
    }

    /* Coverity wrongly resolves completion callback function to
     * 'ucp_cm_server_conn_request_progress' */
    /* coverity[offset_free] */
    ucp_request_complete(req, recv.am.cb, status, req->recv.dt_iter.length,
                         req->user_data);
}

/*
 * process data, complete receive if done
 * @return UCS_OK/ERR - completed, UCS_INPROGRESS - not completed
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_request_process_recv_data(ucp_request_t *req, const void *data,
                              size_t length, size_t offset, int is_zcopy,
                              int is_am)
{
    ucs_status_t status;
    int last;

    last = (req->recv.remaining == length);

    /* process data only if the request is not in error state */
    if (ucs_likely(req->status == UCS_OK)) {
        req->status = ucp_request_recv_data_unpack(req, data, length, offset,
                                                   is_zcopy, last);
    }
    ucs_assertv(req->recv.remaining >= length,
                "req->recv.remaining=%zu length=%zu",
                req->recv.remaining, length);

    req->recv.remaining -= length;

    if (!last) {
        return UCS_INPROGRESS;
    }

    status = req->status;

    if (is_am) {
        ucp_request_complete_am_recv(req, status);
    } else {
        ucp_request_complete_tag_recv(req, status);
    }

    ucs_assert(status != UCS_INPROGRESS);

    return status;
}

static UCS_F_ALWAYS_INLINE ucp_lane_index_t
ucp_send_request_get_am_bw_lane(ucp_request_t *req)
{
    ucp_lane_index_t lane;

    lane = ucp_ep_config(req->send.ep)->
           key.am_bw_lanes[req->send.am_bw_index];
    ucs_assertv(lane != UCP_NULL_LANE, "req->send.am_bw_index=%d",
                req->send.am_bw_index);
    return lane;
}

static UCS_F_ALWAYS_INLINE void
ucp_send_request_next_am_bw_lane(ucp_request_t *req)
{
    ucp_lane_index_t am_bw_index = ++req->send.am_bw_index;
    ucp_ep_config_t *config      = ucp_ep_config(req->send.ep);

    if ((am_bw_index >= UCP_MAX_LANES) ||
        (config->key.am_bw_lanes[am_bw_index] == UCP_NULL_LANE)) {
        req->send.am_bw_index = 0;
    }
}

static UCS_F_ALWAYS_INLINE ucs_ptr_map_key_t
ucp_send_request_get_ep_remote_id(ucp_request_t *req)
{
    /* This function may return UCS_PTR_MAP_KEY_INVALID, but in such cases
     * the message should not be sent at all because the am_lane would point to
     * a wireup (proxy) endpoint. So only the receiver side has an assertion
     * that remote_id != UCS_PTR_MAP_KEY_INVALID.
     */
    return ucp_ep_remote_id(req->send.ep);
}

static UCS_F_ALWAYS_INLINE uint32_t
ucp_request_param_flags(const ucp_request_param_t *param)
{
    return UCP_REQUEST_PARAM_FIELD(param, FLAGS, flags, 0);
}

static UCS_F_ALWAYS_INLINE ucp_datatype_t
ucp_request_param_datatype(const ucp_request_param_t *param)
{
    return UCP_REQUEST_PARAM_FIELD(param, DATATYPE, datatype,
                                   ucp_dt_make_contig(1));
}

static UCS_F_ALWAYS_INLINE ucp_send_nbx_callback_t
ucp_request_param_send_callback(const ucp_request_param_t *param)
{
    return UCP_REQUEST_PARAM_FIELD(param, CALLBACK, cb.send, NULL);
}

static UCS_F_ALWAYS_INLINE void *
ucp_request_param_user_data(const ucp_request_param_t *param)
{
    return UCP_REQUEST_PARAM_FIELD(param, USER_DATA, user_data, NULL);
}

static UCS_F_ALWAYS_INLINE ucs_memory_type_t
ucp_request_get_memory_type(ucp_context_h context, const void *address,
                            size_t count, ucp_datatype_t datatype,
                            size_t contig_length,
                            const ucp_request_param_t *param)
{
    ucp_memory_info_t mem_info;
    uint8_t UCS_V_UNUSED dummy_sg_count;
    const uct_iov_t *iov;

    if ((param->op_attr_mask & UCP_OP_ATTR_FIELD_MEMORY_TYPE) &&
        (param->memory_type != UCS_MEMORY_TYPE_UNKNOWN)) {
        return param->memory_type;
    }

    if (param->op_attr_mask & UCP_OP_ATTR_FIELD_MEMH) {
        ucs_assert(param->memh != NULL);
        return param->memh->mem_type;
    }

    if (UCP_DT_IS_CONTIG(datatype)) {
        ucp_memory_detect(context, address, contig_length, &mem_info);
    } else if (UCP_DT_IS_IOV(datatype) && (count > 0)) {
        iov = (const uct_iov_t*)address;
        ucp_memory_detect(context, iov->buffer, iov->length, &mem_info);
    } else {
        ucp_memory_info_set_host(&mem_info);
    }

    ucs_assert((ucs_memory_type_t)mem_info.type < UCS_MEMORY_TYPE_UNKNOWN);
    return (ucs_memory_type_t)mem_info.type;
}

static UCS_F_ALWAYS_INLINE void
ucp_request_ptr_map_status_check(ucs_status_t status, const char *action_str,
                                 ucp_ep_h ep, void *ptr)
{
    ucs_assertv((status == UCS_OK) || (status == UCS_ERR_NO_PROGRESS),
                "ep %p: failed to %s id for %p: %s", ep, action_str, ptr,
                ucs_status_string(status));
}

static UCS_F_ALWAYS_INLINE void ucp_send_request_id_alloc(ucp_request_t *req)
{
    ucp_ep_h ep = req->send.ep;
    ucs_status_t status;

    ucp_request_id_check(req, ==, UCS_PTR_MAP_KEY_INVALID);
    status = UCS_PTR_MAP_PUT(request, &ep->worker->request_map, req,
                             ucp_ep_use_indirect_id(ep), &req->id);
    ucp_request_ptr_map_status_check(status, "put", ep, req);

    if (status == UCS_OK) {
        ucs_hlist_add_tail(&ep->ext->proto_reqs, &req->send.list);
    }
}

static UCS_F_ALWAYS_INLINE ucs_ptr_map_key_t
ucp_send_request_get_id(const ucp_request_t *req)
{
    ucp_request_id_check(req, !=, UCS_PTR_MAP_KEY_INVALID);
    return req->id;
}

/* Since release function resets request ID to @ref UCS_PTR_MAP_KEY_INVALID and
 * PTR MAP considers @ref UCS_PTR_MAP_KEY_INVALID as direct key, release request
 * ID is re-entrant function */
static UCS_F_ALWAYS_INLINE void ucp_send_request_id_release(ucp_request_t *req)
{
    ucp_ep_h ep;
    ucs_status_t status;

    ucs_assert(!(req->flags &
                 (UCP_REQUEST_FLAG_RECV_AM | UCP_REQUEST_FLAG_RECV_TAG)));
    ep = req->send.ep;

    status = UCS_PTR_MAP_DEL(request, &ep->worker->request_map, req->id);
    ucp_request_ptr_map_status_check(status, "delete", ep, req);

    if (status == UCS_OK) {
        ucs_hlist_del(&ep->ext->proto_reqs, &req->send.list);
    }

    ucp_request_id_reset(req);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_send_request_get_by_id(ucp_worker_h worker, ucs_ptr_map_key_t id,
                           ucp_request_t **req_p, int extract)
{
    ucs_status_t status;
    void *ptr;

    ucs_assert(id != UCS_PTR_MAP_KEY_INVALID);

    status = UCS_PTR_MAP_GET(request, &worker->request_map, id, extract, &ptr);
    if (ucs_unlikely((status != UCS_OK) && (status != UCS_ERR_NO_PROGRESS))) {
        *req_p = NULL; /* To suppress compiler warning */
        return status;
    }

    *req_p = (ucp_request_t*)ptr;
    ucp_request_id_check(*req_p, ==, id);

    if (extract) {
        /* If request ID was released, then need to reset the request ID to use
         * the value for checking whether the request ID should be put to PTR
         * map or not in case of error handling */
        ucp_request_id_reset(*req_p);

        if (status == UCS_OK) {
            ucs_hlist_del(&((*req_p)->send.ep->ext->proto_reqs),
                          &(*req_p)->send.list);
        }
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void ucp_request_set_super(ucp_request_t *req,
                                                      ucp_request_t *super_req)
{
    ucs_assertv(!(req->flags & UCP_REQUEST_FLAG_SUPER_VALID),
                "req=%p req->super_req=%p", req, req->super_req);
    req->super_req = super_req;
    req->flags    |= UCP_REQUEST_FLAG_SUPER_VALID;
}

static UCS_F_ALWAYS_INLINE void ucp_request_reset_super(ucp_request_t *req)
{
    req->flags &= ~UCP_REQUEST_FLAG_SUPER_VALID;
}

static UCS_F_ALWAYS_INLINE ucp_request_t*
ucp_request_get_super(ucp_request_t *req)
{
    ucs_assertv(req->flags & UCP_REQUEST_FLAG_SUPER_VALID,
                "req=%p req->super_req=%p", req, req->super_req);
    return req->super_req;
}

static UCS_F_ALWAYS_INLINE ucp_request_t *
ucp_request_user_data_get_super(void *request, void *user_data)
{
    ucp_request_t UCS_V_UNUSED *req = (ucp_request_t*)request - 1;
    ucp_request_t *super_req        = (ucp_request_t*)user_data;

    ucs_assert(super_req != NULL);
    ucs_assert(ucp_request_get_super(req) == super_req);
    return super_req;
}

static UCS_F_ALWAYS_INLINE void
ucp_request_param_rndv_thresh(ucp_request_t *req,
                              const ucp_request_param_t *param,
                              ucp_rndv_thresh_t *rma_thresh_config,
                              ucp_rndv_thresh_t *am_thresh_config,
                              size_t *rndv_rma_thresh, size_t *rndv_am_thresh)
{
    if ((param->op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) &&
        ucs_likely(UCP_MEM_IS_HOST(req->send.mem_type))) {
        *rndv_rma_thresh = rma_thresh_config->local;
        *rndv_am_thresh  = am_thresh_config->local;
    } else {
        *rndv_rma_thresh = rma_thresh_config->remote;
        *rndv_am_thresh  = am_thresh_config->remote;
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_invoke_uct_completion(uct_completion_t *comp, ucs_status_t status)
{
    ucs_assertv(comp->count > 0, "comp=%p count=%d func=%p status %s", comp,
                comp->count, comp->func, ucs_status_string(status));

    uct_completion_update_status(comp, status);
    if (--comp->count == 0) {
        comp->func(comp);
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_request_invoke_uct_completion_success(ucp_request_t *req)
{
    ucp_invoke_uct_completion(&req->send.state.uct_comp, UCS_OK);
    return UCS_OK;
}

/* The function can be used to complete any UCP send request */
static UCS_F_ALWAYS_INLINE void
ucp_request_complete_and_dereg_send(ucp_request_t *sreq, ucs_status_t status)
{
    ucs_assert(!sreq->send.ep->worker->context->config.ext.proto_enable);
    ucp_request_send_generic_dt_finish(sreq);
    ucp_request_send_buffer_dereg(sreq);
    ucp_request_complete_send(sreq, status);
}


#define UCP_SEND_REQUEST_GET_BY_ID(_req_p, _worker, _req_id, _extract, \
                                   _action, _fmt_str, ...) \
    { \
        ucs_status_t __status = ucp_send_request_get_by_id(_worker, _req_id, \
                                                           _req_p, _extract); \
        if (ucs_unlikely(__status != UCS_OK)) { \
            ucs_trace_data("worker %p: req id 0x%" PRIx64 " doesn't exist" \
                           " drop " _fmt_str, \
                           _worker, _req_id, ##__VA_ARGS__); \
            _action; \
        } \
    }

#endif
