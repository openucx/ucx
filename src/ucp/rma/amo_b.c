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
#include <ucs/profile/profile.h>
#include <inttypes.h>

#define UCP_AMO_WITHOUT_RESULT(_ep, _param, _remote_addr, _rkey, _op, _size) \
    { \
        ucs_status_t status; \
        \
        status = ucp_rma_check_atomic(_remote_addr, sizeof(uint##_size##_t)); \
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
            status = UCS_PROFILE_CALL(uct_ep_atomic##_size##_post, \
                                      (_ep)->uct_eps[(_rkey)->cache.amo_lane], _op, \
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

#define UCP_AMO_WITH_RESULT(_ep, _params, _remote_addr, _rkey, _result, _uct_func, _size) \
    { \
        uct_completion_t comp; \
        ucs_status_t status; \
        \
        status = ucp_rma_check_atomic(_remote_addr, _size); \
        if (status != UCS_OK) { \
            goto out; \
        } \
        \
        UCP_THREAD_CS_ENTER_CONDITIONAL(&(_ep)->worker->mt_lock); \
        comp.count = 2; \
        \
        for (;;) { \
            status = UCP_RKEY_RESOLVE(_rkey, _ep, amo); \
            if (status != UCS_OK) { \
                goto out_unlock; \
            } \
            \
            status = UCS_PROFILE_CALL(_uct_func, (_ep)->uct_eps[(_rkey)->cache.amo_lane], \
                                      UCS_PP_TUPLE_BREAK _params, _remote_addr, \
                                      (_rkey)->cache.amo_rkey, UCS_PP_TUPLE_BREAK _result ); \
            if (ucs_likely(status == UCS_OK)) { \
                goto out_unlock; \
            } else if (status == UCS_INPROGRESS) { \
                goto out_wait; \
            } else if (status != UCS_ERR_NO_RESOURCE) { \
                goto out_unlock; \
            } \
            ucp_worker_progress((_ep)->worker); \
        } \
    out_wait: \
        do { \
            ucp_worker_progress((_ep)->worker); \
        } while (comp.count != 1); \
        status = UCS_OK; \
    out_unlock: \
        UCP_THREAD_CS_EXIT_CONDITIONAL(&(_ep)->worker->mt_lock); \
    out: \
        return status; \
    }

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_add32, (ep, add, remote_addr, rkey),
                 ucp_ep_h ep, uint32_t add, uint64_t remote_addr, ucp_rkey_h rkey)
{
    UCP_AMO_WITHOUT_RESULT(ep, add, remote_addr, rkey, UCT_ATOMIC_OP_ADD, 32);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_add64, (ep, add, remote_addr, rkey),
                 ucp_ep_h ep, uint64_t add, uint64_t remote_addr, ucp_rkey_h rkey)
{
    UCP_AMO_WITHOUT_RESULT(ep, add, remote_addr, rkey, UCT_ATOMIC_OP_ADD, 64);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_fadd32, (ep, add, remote_addr, rkey, result),
                 ucp_ep_h ep, uint32_t add, uint64_t remote_addr, ucp_rkey_h rkey,
                 uint32_t *result)
{
    UCP_AMO_WITH_RESULT(ep, (UCT_ATOMIC_OP_ADD, add, result), remote_addr, rkey, (&comp),
                        uct_ep_atomic32_fetch, sizeof(uint32_t));
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_fadd64, (ep, add, remote_addr, rkey, result),
                 ucp_ep_h ep, uint64_t add, uint64_t remote_addr, ucp_rkey_h rkey,
                 uint64_t *result)
{
    UCP_AMO_WITH_RESULT(ep, (UCT_ATOMIC_OP_ADD, add, result), remote_addr, rkey, (&comp),
                        uct_ep_atomic64_fetch, sizeof(uint64_t));
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_swap32, (ep, swap, remote_addr, rkey, result),
                 ucp_ep_h ep, uint32_t swap, uint64_t remote_addr, ucp_rkey_h rkey,
                 uint32_t *result)
{
    UCP_AMO_WITH_RESULT(ep, (UCT_ATOMIC_OP_SWAP, swap, result), remote_addr, rkey, (&comp),
                        uct_ep_atomic32_fetch, sizeof(uint32_t));
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_swap64, (ep, swap, remote_addr, rkey, result),
                 ucp_ep_h ep, uint64_t swap, uint64_t remote_addr, ucp_rkey_h rkey,
                 uint64_t *result)
{
    UCP_AMO_WITH_RESULT(ep, (UCT_ATOMIC_OP_SWAP, swap, result), remote_addr, rkey, (&comp),
                        uct_ep_atomic64_fetch, sizeof(uint64_t));
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_cswap32,
                 (ep, compare, swap, remote_addr, rkey, result),
                 ucp_ep_h ep, uint32_t compare, uint32_t swap,
                 uint64_t remote_addr, ucp_rkey_h rkey, uint32_t *result)
{
    UCP_AMO_WITH_RESULT(ep, (compare, swap), remote_addr, rkey, (result, &comp),
                        uct_ep_atomic_cswap32, sizeof(uint32_t));
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_cswap64,
                 (ep, compare, swap, remote_addr, rkey, result),
                 ucp_ep_h ep, uint64_t compare, uint64_t swap,
                 uint64_t remote_addr, ucp_rkey_h rkey, uint64_t *result)
{
    UCP_AMO_WITH_RESULT(ep, (compare, swap), remote_addr, rkey, (result, &comp),
                        uct_ep_atomic_cswap64, sizeof(uint64_t));
}
