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
#include <ucs/debug/profile.h>

static inline
ucs_status_t ucp_amo_check_send_status(ucp_request_t *req, ucs_status_t status);

#define UCP_AMO_ONE_PARAM (value)
#define UCP_AMO_TWO_PARAM (value, *result)

#define _UCP_PROGRESS_AMO_NAME(_function)           \
    UCS_PP_TOKENPASTE(ucp_amo_progress_, _function)

#define UCP_PROGRESS_AMO_DECL(_type, _function, _params) \
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
                           UCS_PP_TUPLE_BREAK _params, \
                           remote_addr, rkey->cache.amo_rkey, result, \
                           &req->send.state.uct_comp); \
        return ucp_amo_check_send_status(req, status); \
    }

UCP_PROGRESS_AMO_DECL(uint32_t, uct_ep_atomic_swap32, UCP_AMO_ONE_PARAM)

UCP_PROGRESS_AMO_DECL(uint32_t, uct_ep_atomic_fadd32, UCP_AMO_ONE_PARAM)

UCP_PROGRESS_AMO_DECL(uint32_t, uct_ep_atomic_cswap32, UCP_AMO_TWO_PARAM)

UCP_PROGRESS_AMO_DECL(uint64_t, uct_ep_atomic_swap64, UCP_AMO_ONE_PARAM)

UCP_PROGRESS_AMO_DECL(uint64_t, uct_ep_atomic_fadd64, UCP_AMO_ONE_PARAM)

UCP_PROGRESS_AMO_DECL(uint64_t, uct_ep_atomic_cswap64, UCP_AMO_TWO_PARAM)

#define UCP_POST_AMO_DECL(_type, _function) \
    static ucs_status_t _UCP_PROGRESS_AMO_NAME(_function)(uct_pending_req_t *_self) \
    { \
        ucp_request_t *req = ucs_container_of(_self, ucp_request_t, send.uct); \
        ucp_rkey_h rkey    = req->send.amo.rkey; \
        ucp_ep_t *ep       = req->send.ep; \
        _type value = (_type)req->send.amo.value;         \
        uint64_t remote_addr = req->send.amo.remote_addr; \
        ucs_status_t status; \
        \
        status = UCP_RKEY_RESOLVE(rkey, ep, amo); \
        if (status != UCS_OK) { \
            return UCS_ERR_UNREACHABLE; \
        } \
        \
        req->send.lane = rkey->cache.amo_lane; \
        status = UCS_PROFILE_CALL(_function, ep->uct_eps[req->send.lane], value, \
                                  remote_addr, rkey->cache.amo_rkey); \
        return ucp_amo_check_send_status(req, status); \
    }

UCP_POST_AMO_DECL(uint64_t, uct_ep_atomic_add64)

UCP_POST_AMO_DECL(uint32_t, uct_ep_atomic_add32)

#define UCP_AMO_WITHOUT_RESULT(_ep, _param, _remote_addr, _rkey, _uct_func, _size) \
    { \
        ucs_status_t status; \
        \
        status = ucp_rma_check_atomic(_remote_addr, _size); \
        if (status != UCS_OK) { \
            goto out; \
        } \
        \
        UCP_THREAD_CS_ENTER_CONDITIONAL(&(_ep)->worker->mt_lock); \
        for (;;) { \
            status = UCP_RKEY_RESOLVE(_rkey, _ep, amo); \
            if (status != UCS_OK) { \
                goto out_unlock; \
            } \
            \
            status = UCS_PROFILE_CALL(_uct_func, (_ep)->uct_eps[(_rkey)->cache.amo_lane], \
                                      _param, _remote_addr, (_rkey)->cache.amo_rkey); \
            if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) { \
                UCP_THREAD_CS_EXIT_CONDITIONAL(&(_ep)->worker->mt_lock); \
                return status; \
            } \
            ucp_worker_progress((_ep)->worker); \
        } \
        \
        status = UCS_OK; \
    out_unlock: \
        UCP_THREAD_CS_EXIT_CONDITIONAL(&(_ep)->worker->mt_lock); \
    out: \
        return status; \
    }

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
uct_pending_callback_t ucp_amo_select_uct_func(ucp_atomic_fetch_op_t opcode, size_t op_size)
{
    uct_pending_callback_t progress_func;

    if (op_size == sizeof(uint64_t)) {
        switch (opcode) {
        case UCP_ATOMIC_FETCH_OP_CSWAP:
            progress_func = _UCP_PROGRESS_AMO_NAME(uct_ep_atomic_cswap64);
            break;
        case UCP_ATOMIC_FETCH_OP_SWAP:
            progress_func = _UCP_PROGRESS_AMO_NAME(uct_ep_atomic_swap64);
            break;
        case UCP_ATOMIC_FETCH_OP_FADD:
            progress_func = _UCP_PROGRESS_AMO_NAME(uct_ep_atomic_fadd64);
            break;
        default:
            progress_func = NULL;
        }
    } else {
        switch (opcode) {
        case UCP_ATOMIC_FETCH_OP_CSWAP:
            progress_func = _UCP_PROGRESS_AMO_NAME(uct_ep_atomic_cswap32);
            break;
        case UCP_ATOMIC_FETCH_OP_SWAP:
            progress_func = _UCP_PROGRESS_AMO_NAME(uct_ep_atomic_swap32);
            break;
        case UCP_ATOMIC_FETCH_OP_FADD:
            progress_func = _UCP_PROGRESS_AMO_NAME(uct_ep_atomic_fadd32);
            break;
        default:
            progress_func = NULL;
        }
    }
    return progress_func;
}

static inline
uct_pending_callback_t ucp_amo_post_select_uct_func(ucp_atomic_post_op_t opcode, size_t op_size)
{
    uct_pending_callback_t progress_func;

    if (opcode != UCP_ATOMIC_POST_OP_ADD) {
        return NULL;
    }
    switch (op_size) {
    case sizeof(uint32_t):
        progress_func = _UCP_PROGRESS_AMO_NAME(uct_ep_atomic_add32);
        break;
    case sizeof(uint64_t):
        progress_func = _UCP_PROGRESS_AMO_NAME(uct_ep_atomic_add64);
        break;
    default:
        progress_func = NULL;
    }

    return progress_func;
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

static inline void init_amo_req(ucp_request_t *req, ucp_ep_h ep, void *buffer,
                                ucp_atomic_fetch_op_t op, size_t op_size, uint64_t remote_addr,
                                ucp_rkey_h rkey, uint64_t value)
{
    init_amo_common(req, ep, remote_addr, rkey, value);
    req->send.state.uct_comp.count  = 1;
    req->send.state.uct_comp.func   = ucp_amo_completed_single;
    req->send.amo.result            = buffer;
    req->send.uct.func              = ucp_amo_select_uct_func(op, op_size);
}

static inline void init_amo_post(ucp_request_t *req, ucp_ep_h ep, ucp_atomic_post_op_t op,
                                 size_t op_size, uint64_t remote_addr, ucp_rkey_h rkey,
                                 uint64_t value)
{
    init_amo_common(req, ep, remote_addr, rkey, value);
    req->send.uct.func = ucp_amo_post_select_uct_func(op, op_size);
}
#endif
