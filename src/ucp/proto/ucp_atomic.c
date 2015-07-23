/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ucp_int.h"

#include <ucs/sys/preprocessor.h>


#define UCP_RMA_CHECK_ATOMIC(_remote_addr, _size) \
    if (ENABLE_PARAMS_CHECK && (((_remote_addr) % (_size)) != 0)) { \
        ucs_debug("Error: Atomic variable must be naturally aligned " \
                  "(got address 0x%"PRIx64", atomic size %zu)", (_remote_addr),\
                  (_size)); \
        return UCS_ERR_INVALID_PARAM; \
    }

#define UCP_RMA_ATOMIC_WITHOUT_RESULT(_ep, _param, _remote_addr, _rkey, _uct_func, _size) \
    { \
        ucs_status_t status; \
        uct_rkey_t uct_rkey; \
        \
        UCP_RMA_CHECK_ATOMIC(_remote_addr, _size); \
        uct_rkey = UCP_RMA_RKEY_LOOKUP(_ep, _rkey); \
        for (;;) { \
            status = _uct_func((_ep)->uct_ep, _param, _remote_addr, uct_rkey); \
            if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) { \
                return status; \
            } \
            ucp_worker_progress((_ep)->worker); \
        } \
        return UCS_OK; \
    }

#define UCP_RMA_ATOMIC_WITH_RESULT(_ep, _params, _remote_addr, _rkey, _result, _uct_func, _size) \
    { \
        uct_completion_t comp; \
        ucs_status_t status; \
        uct_rkey_t uct_rkey; \
        \
        UCP_RMA_CHECK_ATOMIC(_remote_addr, _size); \
        uct_rkey   = UCP_RMA_RKEY_LOOKUP(_ep, _rkey); \
        comp.count = 2; \
        \
        for (;;) { \
            status = _uct_func((_ep)->uct_ep, \
                               UCS_PP_TUPLE_BREAK _params, \
                               _remote_addr, uct_rkey, \
                               _result, &comp); \
            if (ucs_likely(status == UCS_OK)) { \
                goto out; \
            } else if (status == UCS_INPROGRESS) { \
                goto out_wait; \
            } else if (status != UCS_ERR_NO_RESOURCE) { \
                return status; \
            } \
            ucp_worker_progress((_ep)->worker); \
        } \
    out_wait: \
        do { \
            ucp_worker_progress((_ep)->worker); \
        } while (comp.count != 1); \
    out: \
        return UCS_OK; \
    }

ucs_status_t ucp_atomic_add32(ucp_ep_h ep, uint32_t add,
                              uint64_t remote_addr, ucp_rkey_h rkey)
{
    UCP_RMA_ATOMIC_WITHOUT_RESULT(ep, add, remote_addr, rkey,
                                  uct_ep_atomic_add32, sizeof(uint32_t));
}

ucs_status_t ucp_atomic_add64(ucp_ep_h ep, uint64_t add,
                              uint64_t remote_addr, ucp_rkey_h rkey)
{
    UCP_RMA_ATOMIC_WITHOUT_RESULT(ep, add, remote_addr, rkey,
                                  uct_ep_atomic_add64, sizeof(uint64_t));
}

ucs_status_t ucp_atomic_fadd32(ucp_ep_h ep, uint32_t add, uint64_t remote_addr,
                               ucp_rkey_h rkey, uint32_t *result)
{
    UCP_RMA_ATOMIC_WITH_RESULT(ep, (add), remote_addr, rkey, result,
                               uct_ep_atomic_fadd32, sizeof(uint32_t));
}

ucs_status_t ucp_atomic_fadd64(ucp_ep_h ep, uint64_t add, uint64_t remote_addr,
                               ucp_rkey_h rkey, uint64_t *result)
{
    UCP_RMA_ATOMIC_WITH_RESULT(ep, (add), remote_addr, rkey, result,
                               uct_ep_atomic_fadd64, sizeof(uint64_t));
}

ucs_status_t ucp_atomic_swap32(ucp_ep_h ep, uint32_t swap, uint64_t remote_addr,
                               ucp_rkey_h rkey, uint32_t *result)
{
    UCP_RMA_ATOMIC_WITH_RESULT(ep, (swap), remote_addr, rkey, result,
                               uct_ep_atomic_swap32, sizeof(uint32_t));
}

ucs_status_t ucp_atomic_swap64(ucp_ep_h ep, uint64_t swap, uint64_t remote_addr,
                               ucp_rkey_h rkey, uint64_t *result)
{
    UCP_RMA_ATOMIC_WITH_RESULT(ep, (swap), remote_addr, rkey, result,
                               uct_ep_atomic_swap64, sizeof(uint64_t));
}

ucs_status_t ucp_atomic_cswap32(ucp_ep_h ep, uint32_t compare, uint32_t swap,
                                uint64_t remote_addr, ucp_rkey_h rkey, uint32_t *result)
{
    UCP_RMA_ATOMIC_WITH_RESULT(ep, (compare, swap), remote_addr, rkey, result,
                               uct_ep_atomic_cswap32, sizeof(uint32_t));
}

ucs_status_t ucp_atomic_cswap64(ucp_ep_h ep, uint64_t compare, uint64_t swap,
                                uint64_t remote_addr, ucp_rkey_h rkey, uint64_t *result)
{
    UCP_RMA_ATOMIC_WITH_RESULT(ep, (compare, swap), remote_addr, rkey, result,
                               uct_ep_atomic_cswap64, sizeof(uint64_t));
}
