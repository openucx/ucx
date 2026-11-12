/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2016. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "sm_ep.h"

#include <ucs/arch/atomic.h>
#include <ucs/time/time.h>


#define uct_sm_ep_trace_data(_remote_addr, _rkey, _fmt, ...) \
     ucs_trace_data(_fmt " to 0x%"PRIx64"(%+ld)", ## __VA_ARGS__, (_remote_addr), \
                    (_rkey))

ucs_status_t uct_sm_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                 unsigned length, uint64_t remote_addr,
                                 uct_rkey_t rkey)
{
    if (ucs_likely(length != 0)) {
        memcpy((void *)(rkey + remote_addr), buffer, length);
        uct_sm_ep_trace_data(remote_addr, rkey, "PUT_SHORT [buffer %p size %u]",
                             buffer, length);
    } else {
        ucs_trace_data("PUT_SHORT [zero-length]");
    }
    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, SHORT, length);
    return UCS_OK;
}

ssize_t uct_sm_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                            void *arg, uint64_t remote_addr, uct_rkey_t rkey)
{
    size_t length;

    length = pack_cb((void *)(rkey + remote_addr), arg);
    uct_sm_ep_trace_data(remote_addr, rkey, "PUT_BCOPY [arg %p size %zu]",
    		             arg, length);
    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, BCOPY, length);
    return length;
}

ucs_status_t uct_sm_ep_get_bcopy(uct_ep_h tl_ep, uct_unpack_callback_t unpack_cb,
                                 void *arg, size_t length,
                                 uint64_t remote_addr, uct_rkey_t rkey,
                                 uct_completion_t *comp)
{
    if (ucs_likely(0 != length)) {
        unpack_cb(arg, (void *)(rkey + remote_addr), length);
        uct_sm_ep_trace_data(remote_addr, rkey, "GET_BCOPY [length %zu]", length);
    } else {
        ucs_trace_data("GET_BCOPY [zero-length]");
    }
    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, BCOPY, length);
    return UCS_OK;
}

#define UCT_SM_EP_DEFINE_ATOMIC_POST_FUNC(_bits) \
    ucs_status_t uct_sm_ep_atomic##_bits##_post(uct_ep_h ep, unsigned opcode, \
                                                uint##_bits##_t value, \
                                                uint64_t remote_addr, \
                                                uct_rkey_t rkey) \
    { \
        uint##_bits##_t *ptr = (uint##_bits##_t *)(rkey + remote_addr); \
        switch (opcode) { \
        case UCT_ATOMIC_OP_ADD: \
            ucs_atomic_add##_bits(ptr, value); \
            uct_sm_ep_trace_data(remote_addr, rkey, \
                                 "ATOMIC_ADD" #_bits \
                                 " [value %"PRIu##_bits"]", value); \
            break; \
        case UCT_ATOMIC_OP_AND: \
            ucs_atomic_and##_bits(ptr, value); \
            uct_sm_ep_trace_data(remote_addr, rkey, \
                                 "ATOMIC_AND" #_bits \
                                 " [value %"PRIu##_bits"]", value); \
            break; \
        case UCT_ATOMIC_OP_OR: \
            ucs_atomic_or##_bits(ptr, value); \
            uct_sm_ep_trace_data(remote_addr, rkey, \
                                 "ATOMIC_OR" #_bits \
                                 " [value %"PRIu##_bits"]", value); \
            break; \
        case UCT_ATOMIC_OP_XOR: \
            ucs_atomic_xor##_bits(ptr, value); \
            uct_sm_ep_trace_data(remote_addr, rkey, \
                                 "ATOMIC_XOR" #_bits \
                                 " [value %"PRIu##_bits"]", value); \
            break; \
        default: \
            ucs_assertv(0, "incorrect opcode: %d", opcode); \
            return UCS_ERR_UNSUPPORTED; \
        } \
        \
        UCT_TL_EP_STAT_ATOMIC(ucs_derived_of(ep, uct_base_ep_t)); \
        return UCS_OK; \
    }

#define UCT_SM_EP_DEFINE_ATOMIC_FETCH_FUNC(_bits) \
    ucs_status_t uct_sm_ep_atomic##_bits##_fetch(uct_ep_h ep, \
                                                 uct_atomic_op_t opcode, \
                                                 uint##_bits##_t value, \
                                                 uint##_bits##_t *result, \
                                                 uint64_t remote_addr, \
                                                 uct_rkey_t rkey, \
                                                 uct_completion_t *comp) \
    { \
        uint##_bits##_t *ptr = (uint##_bits##_t *)(rkey + remote_addr); \
        switch (opcode) { \
        case UCT_ATOMIC_OP_ADD: \
            *result = ucs_atomic_fadd##_bits(ptr, value); \
            uct_sm_ep_trace_data(remote_addr, rkey, \
                                 "ATOMIC_FADD" #_bits " [value %"PRIu##_bits \
                                 " result %"PRIu##_bits"]", value, *result); \
            break; \
        case UCT_ATOMIC_OP_AND: \
            *result = ucs_atomic_fand##_bits(ptr, value); \
            uct_sm_ep_trace_data(remote_addr, rkey, \
                                 "ATOMIC_FAND" #_bits " [value %"PRIu##_bits \
                                 " result %"PRIu##_bits"]", value, *result); \
            break; \
        case UCT_ATOMIC_OP_OR: \
            *result = ucs_atomic_for##_bits(ptr, value); \
            uct_sm_ep_trace_data(remote_addr, rkey, \
                                 "ATOMIC_FOR" #_bits " [value %"PRIu##_bits \
                                 " result %"PRIu##_bits"]", value, *result); \
            break; \
        case UCT_ATOMIC_OP_XOR: \
            *result = ucs_atomic_fxor##_bits(ptr, value); \
            uct_sm_ep_trace_data(remote_addr, rkey, \
                                 "ATOMIC_FXOR" #_bits " [value %"PRIu##_bits \
                                 " result %"PRIu##_bits"]", value, *result); \
            break; \
        case UCT_ATOMIC_OP_SWAP: \
            *result = ucs_atomic_swap##_bits(ptr, value); \
            uct_sm_ep_trace_data(remote_addr, rkey, \
                                 "ATOMIC_SWAP" #_bits " [value %"PRIu##_bits \
                                 " result %"PRIu##_bits"]", value, *result); \
            break; \
        default: \
            ucs_assertv(0, "incorrect opcode: %d", opcode); \
            return UCS_ERR_UNSUPPORTED; \
        } \
        \
        UCT_TL_EP_STAT_ATOMIC(ucs_derived_of(ep, uct_base_ep_t)); \
        return UCS_OK; \
    }

#define UCT_SM_EP_DEFINE_ATOMIC_CSWAP_FUNC(_bits) \
    ucs_status_t uct_sm_ep_atomic_cswap##_bits(uct_ep_h tl_ep, \
                                               uint##_bits##_t compare, \
                                               uint##_bits##_t swap, \
                                               uint64_t remote_addr, \
                                               uct_rkey_t rkey, \
                                               uint##_bits##_t *result, \
                                               uct_completion_t *comp) \
    { \
        uint##_bits##_t *ptr = (uint##_bits##_t *)(rkey + remote_addr); \
        *result = ucs_atomic_cswap##_bits(ptr, compare, swap); \
        uct_sm_ep_trace_data(remote_addr, rkey, \
                             "ATOMIC_CSWAP" #_bits " [compare %"PRIu##_bits \
                             " swap %"PRIu##_bits" result %"PRIu##_bits"]", \
                             compare, swap, *result); \
        UCT_TL_EP_STAT_ATOMIC(ucs_derived_of(tl_ep, uct_base_ep_t)); \
        return UCS_OK; \
    }

UCT_SM_EP_DEFINE_ATOMIC_POST_FUNC(32)
UCT_SM_EP_DEFINE_ATOMIC_POST_FUNC(64)
UCT_SM_EP_DEFINE_ATOMIC_FETCH_FUNC(32)
UCT_SM_EP_DEFINE_ATOMIC_FETCH_FUNC(64)
UCT_SM_EP_DEFINE_ATOMIC_CSWAP_FUNC(32)
UCT_SM_EP_DEFINE_ATOMIC_CSWAP_FUNC(64)
