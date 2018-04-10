/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (C) UT-Battelle, LLC. 2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_AMO_INL_
#define UCP_AMO_INL_

#include <inttypes.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_ep.inl>
#include <ucs/sys/preprocessor.h>
#include <ucs/debug/log.h>
#include <ucs/profile/profile.h>

static inline
ucs_status_t ucp_amo_check_send_status(ucp_request_t *req, ucs_status_t status);

#define UCP_UCT_AMO_OPCODE_POST(_opcode, _op)  \
    switch(_opcode) {                          \
    case UCP_ATOMIC_POST_OP_ADD:               \
        _op = UCT_ATOMIC_OP_ADD;               \
        break;                                 \
    case UCP_ATOMIC_POST_OP_AND:               \
        _op = UCT_ATOMIC_OP_AND;               \
        break;                                 \
    case UCP_ATOMIC_POST_OP_OR:                \
        _op = UCT_ATOMIC_OP_OR;                \
        break;                                 \
    case UCP_ATOMIC_POST_OP_XOR:               \
        _op = UCT_ATOMIC_OP_XOR;               \
        break;                                 \
    default:                                   \
        return UCS_ERR_UNSUPPORTED;            \
    }

#define UCP_UCT_AMO_OPCODE_FETCH(_opcode, _op) \
    switch(_opcode) {                          \
    case UCP_ATOMIC_FETCH_OP_FADD:             \
        _op = UCT_ATOMIC_OP_ADD;               \
        break;                                 \
    case UCP_ATOMIC_FETCH_OP_FAND:             \
        _op = UCT_ATOMIC_OP_AND;               \
        break;                                 \
    case UCP_ATOMIC_FETCH_OP_FOR:              \
        _op = UCT_ATOMIC_OP_OR;                \
        break;                                 \
    case UCP_ATOMIC_FETCH_OP_FXOR:             \
        _op = UCT_ATOMIC_OP_XOR;               \
        break;                                 \
    case UCP_ATOMIC_FETCH_OP_SWAP:             \
        _op = UCT_ATOMIC_OP_SWAP;              \
        break;                                 \
    case UCP_ATOMIC_FETCH_OP_CSWAP:            \
        /* do not set op */                    \
        break;                                 \
    default:                                   \
        return UCS_ERR_UNSUPPORTED;            \
    }

#define _UCP_PROGRESS_AMO_NAME(_function)           \
    UCS_PP_TOKENPASTE(ucp_amo_progress_, _function)

#define _UCP_PROGRESS_AMO_POST(_size)              \
    _UCP_PROGRESS_AMO_NAME(uct_ep_atomic##_size##_post)

#define _UCP_PROGRESS_AMO_FETCH(_size)              \
    _UCP_PROGRESS_AMO_NAME(uct_ep_atomic##_size##_fetch)

#define UCP_PROGRESS_AMO_CSWAP_DECL(_type, _function) \
    static ucs_status_t _UCP_PROGRESS_AMO_NAME(_function)(uct_pending_req_t *_self) \
    { \
        ucp_request_t *req = ucs_container_of(_self, ucp_request_t, send.uct); \
        ucp_rkey_h rkey    = req->send.amo.rkey; \
        ucp_ep_t *ep       = req->send.ep; \
        _type value = (_type)req->send.amo.value;         \
        _type *result = (_type *)req->send.amo.result;    \
        uint64_t remote_addr = req->send.amo.remote_addr; \
        ucs_status_t status; \
        \
        status = UCP_RKEY_RESOLVE(rkey, ep, amo); \
        if (status != UCS_OK) { \
            return UCS_ERR_UNREACHABLE; \
        } \
        \
        req->send.lane = rkey->cache.amo_lane; \
        status = _function(ep->uct_eps[req->send.lane], \
                           value, *result, \
                           remote_addr, rkey->cache.amo_rkey, result, \
                           &req->send.state.uct_comp); \
        return ucp_amo_check_send_status(req, status); \
    }

#define UCP_PROGRESS_AMO_FETCH_DECL(_size)                                        \
    static ucs_status_t _UCP_PROGRESS_AMO_FETCH(_size)(uct_pending_req_t *_self)  \
    {                                                                             \
        typedef uint##_size##_t type;                                             \
                                                                                  \
        ucp_request_t *req    = ucs_container_of(_self, ucp_request_t, send.uct); \
        ucp_rkey_h rkey       = req->send.amo.rkey;                               \
        ucp_ep_t *ep          = req->send.ep;                                     \
        type value            = (type)req->send.amo.value;                        \
        type *result          = (type *)req->send.amo.result;                     \
        uint64_t remote_addr  = req->send.amo.remote_addr;                        \
        uct_atomic_op_t op    = req->send.amo.uct_op;                             \
        ucs_status_t status;                                                      \
                                                                                  \
        status = UCP_RKEY_RESOLVE(rkey, ep, amo);                                 \
        if (status != UCS_OK) {                                                   \
            return UCS_ERR_UNREACHABLE;                                           \
        }                                                                         \
                                                                                  \
        req->send.lane = rkey->cache.amo_lane;                                    \
        status = uct_ep_atomic##_size##_fetch(ep->uct_eps[req->send.lane],        \
                                              op, value, result,                  \
                                              remote_addr,                        \
                                              rkey->cache.amo_rkey,               \
                                              &req->send.state.uct_comp);         \
        return ucp_amo_check_send_status(req, status);                            \
    }

#define UCP_PROGRESS_AMO_POST_DECL(_size) \
    static ucs_status_t _UCP_PROGRESS_AMO_POST(_size)(uct_pending_req_t *_self)  \
    {                                                                            \
        typedef uint##_size##_t type;                                            \
                                                                                 \
        ucp_request_t *req   = ucs_container_of(_self, ucp_request_t, send.uct); \
        ucp_rkey_h rkey      = req->send.amo.rkey;                               \
        ucp_ep_t *ep         = req->send.ep;                                     \
        type value           = (type)req->send.amo.value;                        \
        uint64_t remote_addr = req->send.amo.remote_addr;                        \
        uct_atomic_op_t op   = req->send.amo.uct_op;                             \
        ucs_status_t status;                                                     \
                                                                                 \
        status = UCP_RKEY_RESOLVE(rkey, ep, amo);                                \
        if (status != UCS_OK) {                                                  \
            return UCS_ERR_UNREACHABLE;                                          \
        }                                                                        \
                                                                                 \
        req->send.lane = rkey->cache.amo_lane;                                   \
        status = UCS_PROFILE_CALL(uct_ep_atomic##_size##_post,                   \
                                  ep->uct_eps[req->send.lane], op, value,        \
                                  remote_addr, rkey->cache.amo_rkey);            \
        return ucp_amo_check_send_status(req, status);                           \
    }

UCP_PROGRESS_AMO_FETCH_DECL(32)
UCP_PROGRESS_AMO_FETCH_DECL(64)

UCP_PROGRESS_AMO_CSWAP_DECL(uint32_t, uct_ep_atomic_cswap32)
UCP_PROGRESS_AMO_CSWAP_DECL(uint64_t, uct_ep_atomic_cswap64)

UCP_PROGRESS_AMO_POST_DECL(64)
UCP_PROGRESS_AMO_POST_DECL(32)

#define UCP_RMA_CHECK_ATOMIC_PTR(_addr, _op_size) \
    do { \
        ucs_status_t status = ucp_rma_check_atomic(_addr, _op_size); \
        \
        if (status != UCS_OK) { \
            return UCS_STATUS_PTR(status); \
        } \
    } while(0)

static inline 
ucs_status_t ucp_amo_check_send_status(ucp_request_t *req, ucs_status_t status)
{
    if (status == UCS_INPROGRESS) {
        return UCS_OK;
    }
    /* Complete for UCS_OK and unexpected errors */
    if (status != UCS_ERR_NO_RESOURCE) {
        ucp_request_complete_send(req, status);
    }
    return status;
}

static void ucp_amo_completed_single(uct_completion_t *self,
                                     ucs_status_t status)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);
    ucs_trace("Invoking completion on AMO request %p", req);
    ucp_request_complete_send(req, status);
}

static inline ucs_status_t ucp_rma_check_atomic(uint64_t remote_addr, size_t size)
{
    if (ENABLE_PARAMS_CHECK && ((remote_addr % size) != 0)) {
        ucs_debug("Error: Atomic variable must be naturally aligned "
                  "(got address 0x%"PRIx64", atomic size %zu)", (remote_addr),
                  (size));
        return UCS_ERR_INVALID_PARAM;
    }
    return UCS_OK;
}

static inline ucs_status_ptr_t 
ucp_amo_send_request(ucp_request_t *req, ucp_send_callback_t cb)
{
    ucs_status_t status = ucp_request_send(req);

    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        ucs_trace_req("releasing send request %p, returning status %s", req,
                      ucs_status_string(status));
        ucs_mpool_put(req);
        return UCS_STATUS_PTR(status);
    }
    ucs_trace_req("returning amo request %p, status %s", req,
                  ucs_status_string(status));
    ucp_request_set_callback(req, send.cb, cb);
    return req + 1;
}

static inline
uct_pending_callback_t ucp_amo_fetch_select_uct_func(ucp_atomic_fetch_op_t opcode, size_t op_size)
{
    if (op_size == sizeof(uint64_t)) {
        switch (opcode) {
        case UCP_ATOMIC_FETCH_OP_CSWAP:
            return _UCP_PROGRESS_AMO_NAME(uct_ep_atomic_cswap64);
        case UCP_ATOMIC_FETCH_OP_FADD:
        case UCP_ATOMIC_FETCH_OP_FAND:
        case UCP_ATOMIC_FETCH_OP_FOR:
        case UCP_ATOMIC_FETCH_OP_FXOR:
        case UCP_ATOMIC_FETCH_OP_SWAP:
            return _UCP_PROGRESS_AMO_NAME(uct_ep_atomic64_fetch);
        default:
            return NULL;
        }
    } else {
        ucs_assert(op_size == sizeof(uint32_t));

        switch (opcode) {
        case UCP_ATOMIC_FETCH_OP_CSWAP:
            return _UCP_PROGRESS_AMO_NAME(uct_ep_atomic_cswap32);
        case UCP_ATOMIC_FETCH_OP_FADD:
        case UCP_ATOMIC_FETCH_OP_FAND:
        case UCP_ATOMIC_FETCH_OP_FOR:
        case UCP_ATOMIC_FETCH_OP_FXOR:
        case UCP_ATOMIC_FETCH_OP_SWAP:
            return _UCP_PROGRESS_AMO_NAME(uct_ep_atomic32_fetch);
        default:
            return NULL;
        }
    }
}

static inline
uct_pending_callback_t ucp_amo_post_select_uct_func(ucp_atomic_post_op_t opcode, size_t op_size)
{
    if (op_size == sizeof(uint64_t)) {
        switch (opcode) {
        case UCP_ATOMIC_POST_OP_ADD:
        case UCP_ATOMIC_POST_OP_AND:
        case UCP_ATOMIC_POST_OP_OR:
        case UCP_ATOMIC_POST_OP_XOR:
            return _UCP_PROGRESS_AMO_NAME(uct_ep_atomic64_post);
        default:
            return NULL;
        }
    } else {
        ucs_assert(op_size == sizeof(uint32_t));

        switch (opcode) {
        case UCP_ATOMIC_POST_OP_ADD:
        case UCP_ATOMIC_POST_OP_AND:
        case UCP_ATOMIC_POST_OP_OR:
        case UCP_ATOMIC_POST_OP_XOR:
            return _UCP_PROGRESS_AMO_NAME(uct_ep_atomic32_post);
        default:
            return NULL;
        }
    }
}

static inline void init_amo_common(ucp_request_t *req, ucp_ep_h ep, uint64_t remote_addr,
                                   ucp_rkey_h rkey, uint64_t value)
{
    req->flags                = 0;
    req->send.ep              = ep;
    req->send.amo.remote_addr = remote_addr;
    req->send.amo.rkey        = rkey;
    req->send.amo.value       = value;
#if ENABLE_ASSERT
    req->send.lane            = UCP_NULL_LANE;
#endif
}

static inline ucs_status_t init_amo_fetch(ucp_request_t *req, ucp_ep_h ep, void *buffer,
                                          ucp_atomic_fetch_op_t op, size_t op_size,
                                          uint64_t remote_addr, ucp_rkey_h rkey, uint64_t value)
{
    init_amo_common(req, ep, remote_addr, rkey, value);
    UCP_UCT_AMO_OPCODE_FETCH(op, req->send.amo.uct_op);
    req->send.state.uct_comp.count  = 1;
    req->send.state.uct_comp.func   = ucp_amo_completed_single;
    req->send.amo.result            = buffer;
    req->send.uct.func              = ucp_amo_fetch_select_uct_func(op, op_size);
    return UCS_OK;
}

static inline ucs_status_t init_amo_post(ucp_request_t *req, ucp_ep_h ep,
                                         ucp_atomic_post_op_t op,
                                         size_t op_size, uint64_t remote_addr,
                                         ucp_rkey_h rkey, uint64_t value)
{
    init_amo_common(req, ep, remote_addr, rkey, value);
    UCP_UCT_AMO_OPCODE_POST(op, req->send.amo.uct_op);
    req->send.uct.func = ucp_amo_post_select_uct_func(op, op_size);
    return UCS_OK;
}
#endif
