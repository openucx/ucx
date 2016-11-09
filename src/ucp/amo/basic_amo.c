/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (C) UT-Battelle, LLC. 2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "amo.inl"
#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_ep.inl>
#include <ucs/sys/preprocessor.h>
#include <ucs/debug/log.h>
#include <ucs/debug/profile.h>
#include <inttypes.h>

#define UCP_AMO_FINISH_REQUEST(_req, _worker) \
    { \
        if (UCS_PTR_IS_PTR(_req)) { \
            do { \
                ucp_worker_progress(_worker); \
            } while (!ucp_request_is_completed(_req)); \
            ucp_request_release(_req); \
        } else if (UCS_PTR_IS_ERR(_req)) { \
            ucs_warn("AMO failed: %s (%i)", \
                     ucs_status_string(UCS_PTR_STATUS(_req)), UCS_PTR_STATUS(_req)); \
            return UCS_PTR_STATUS(_req); \
        } \
    }

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_add32, (ep, add, remote_addr, rkey),
                 ucp_ep_h ep, uint32_t add, uint64_t remote_addr, ucp_rkey_h rkey)
{
    UCP_AMO_WITHOUT_RESULT(ep, (uint32_t)add, remote_addr, rkey,
                           uct_ep_atomic_add32, sizeof(uint32_t));
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_add64, (ep, add, remote_addr, rkey),
                 ucp_ep_h ep, uint64_t add, uint64_t remote_addr, ucp_rkey_h rkey)
{
    UCP_AMO_WITHOUT_RESULT(ep, add, remote_addr, rkey,
                           uct_ep_atomic_add64, sizeof(uint64_t));
}

static inline ucs_status_t
ucp_atomic_blocking_internal(ucp_ep_h ep, ucp_atomic_fetch_op_t opcode,
                             uint64_t value, void *result, size_t op_size,
                             uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucp_worker_h worker = ep->worker;
    ucs_status_ptr_t request;
    ucp_request_t *req;
    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);
    ucp_rma_check_atomic(remote_addr, op_size);
    req = ucp_request_get(ep->worker);
    if (ucs_unlikely(NULL == req)) {
        UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
        return UCS_ERR_NO_MEMORY;
    }
    init_amo_req(req, ep, result, opcode, op_size, remote_addr, rkey, value);
    request = ucp_amo_send_request(req, (ucp_send_callback_t)ucs_empty_function);
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
    UCP_AMO_FINISH_REQUEST(request, worker);
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_fadd32, (ep, add, remote_addr, rkey, result),
                 ucp_ep_h ep, uint32_t add, uint64_t remote_addr, ucp_rkey_h rkey,
                 uint32_t *result)
{
    return ucp_atomic_blocking_internal(ep, UCP_ATOMIC_FETCH_OP_FADD, add, result,
                                        sizeof(uint32_t), remote_addr, rkey);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_fadd64, (ep, add, remote_addr, rkey, result),
                 ucp_ep_h ep, uint64_t add, uint64_t remote_addr, ucp_rkey_h rkey,
                 uint64_t *result)
{
    return ucp_atomic_blocking_internal(ep, UCP_ATOMIC_FETCH_OP_FADD, add, result,
                                        sizeof(uint64_t), remote_addr, rkey);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_swap32, (ep, swap, remote_addr, rkey, result),
                 ucp_ep_h ep, uint32_t swap, uint64_t remote_addr, ucp_rkey_h rkey,
                 uint32_t *result)
{
    return ucp_atomic_blocking_internal(ep, UCP_ATOMIC_FETCH_OP_SWAP, swap, result,
                                        sizeof(uint32_t), remote_addr, rkey);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_swap64, (ep, swap, remote_addr, rkey, result),
                 ucp_ep_h ep, uint64_t swap, uint64_t remote_addr, ucp_rkey_h rkey,
                 uint64_t *result)
{
    return ucp_atomic_blocking_internal(ep, UCP_ATOMIC_FETCH_OP_SWAP, swap, result,
                                        sizeof(uint64_t), remote_addr, rkey);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_cswap32,
                 (ep, compare, swap, remote_addr, rkey, result),
                 ucp_ep_h ep, uint32_t compare, uint32_t swap,
                 uint64_t remote_addr, ucp_rkey_h rkey, uint32_t *result)
{
    *result = swap;
    return ucp_atomic_blocking_internal(ep, UCP_ATOMIC_FETCH_OP_CSWAP, compare, result,
                                        sizeof(uint32_t), remote_addr, rkey);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_cswap64,
                 (ep, compare, swap, remote_addr, rkey, result),
                 ucp_ep_h ep, uint64_t compare, uint64_t swap,
                 uint64_t remote_addr, ucp_rkey_h rkey, uint64_t *result)
{
    *result = swap;
    return ucp_atomic_blocking_internal(ep, UCP_ATOMIC_FETCH_OP_CSWAP, compare, result,
                                        sizeof(uint64_t), remote_addr, rkey);
}
