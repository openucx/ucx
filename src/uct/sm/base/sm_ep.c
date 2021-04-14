/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
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

ucs_status_t uct_sm_ep_atomic32_post(uct_ep_h ep, unsigned opcode, uint32_t value,
                                     uint64_t remote_addr, uct_rkey_t rkey)
{
    uint32_t *ptr = (uint32_t *)(rkey + remote_addr);
    switch (opcode) {
    case UCT_ATOMIC_OP_ADD:
        ucs_atomic_add32(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_ADD32 [value %"PRIu32"]", value);
        break;
    case UCT_ATOMIC_OP_AND:
        ucs_atomic_and32(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_AND32 [value %"PRIu32"]", value);
        break;
    case UCT_ATOMIC_OP_OR:
        ucs_atomic_or32(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_OR32 [value %"PRIu32"]", value);
        break;
    case UCT_ATOMIC_OP_XOR:
        ucs_atomic_xor32(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_XOR32 [value %"PRIu32"]", value);
        break;
    default:
        ucs_assertv(0, "incorrect opcode: %d", opcode);
        return UCS_ERR_UNSUPPORTED;
    }

    UCT_TL_EP_STAT_ATOMIC(ucs_derived_of(ep, uct_base_ep_t));
    return UCS_OK;
}

ucs_status_t uct_sm_ep_atomic64_post(uct_ep_h ep, unsigned opcode, uint64_t value,
                                     uint64_t remote_addr, uct_rkey_t rkey)
{
    uint64_t *ptr = (uint64_t *)(rkey + remote_addr);
    switch (opcode) {
    case UCT_ATOMIC_OP_ADD:
        ucs_atomic_add64(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_ADD64 [value %"PRIu64"]", value);
        break;
    case UCT_ATOMIC_OP_AND:
        ucs_atomic_and64(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_AND64 [value %"PRIu64"]", value);
        break;
    case UCT_ATOMIC_OP_OR:
        ucs_atomic_or64(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_OR64 [value %"PRIu64"]", value);
        break;
    case UCT_ATOMIC_OP_XOR:
        ucs_atomic_xor64(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_XOR64 [value %"PRIu64"]", value);
        break;
    default:
        ucs_assertv(0, "incorrect opcode: %d", opcode);
        return UCS_ERR_UNSUPPORTED;
    }

    UCT_TL_EP_STAT_ATOMIC(ucs_derived_of(ep, uct_base_ep_t));
    return UCS_OK;
}

ucs_status_t uct_sm_ep_atomic64_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                      uint64_t value, uint64_t *result,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp)
{
    uint64_t *ptr = (uint64_t *)(rkey + remote_addr);
    switch (opcode) {
    case UCT_ATOMIC_OP_ADD:
        *result = ucs_atomic_fadd64(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_FADD64 [value %"PRIu64
                             " result %"PRIu64"]", value, *result);
        break;
    case UCT_ATOMIC_OP_AND:
        *result = ucs_atomic_fand64(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_FAND64 [value %"PRIu64
                             " result %"PRIu64"]", value, *result);
        break;
    case UCT_ATOMIC_OP_OR:
        *result = ucs_atomic_for64(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_FOR64 [value %"PRIu64
                             " result %"PRIu64"]", value, *result);
        break;
    case UCT_ATOMIC_OP_XOR:
        *result = ucs_atomic_fxor64(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_FXOR64 [value %"PRIu64
                             " result %"PRIu64"]", value, *result);
        break;
    case UCT_ATOMIC_OP_SWAP:
        *result = ucs_atomic_swap64(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_SWAP64 [value %"PRIu64
                             " result %"PRIu64"]", value, *result);
        break;
    default:
        ucs_assertv(0, "incorrect opcode: %d", opcode);
        return UCS_ERR_UNSUPPORTED;
    }

    UCT_TL_EP_STAT_ATOMIC(ucs_derived_of(ep, uct_base_ep_t));
    return UCS_OK;
}

ucs_status_t uct_sm_ep_atomic32_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                      uint32_t value, uint32_t *result,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp)
{
    uint32_t *ptr = (uint32_t *)(rkey + remote_addr);
    switch (opcode) {
    case UCT_ATOMIC_OP_ADD:
        *result = ucs_atomic_fadd32(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_FADD32 [value %"PRIu32
                             " result %"PRIu32"]", value, *result);
        break;
    case UCT_ATOMIC_OP_AND:
        *result = ucs_atomic_fand32(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_FAND32 [value %"PRIu32
                             " result %"PRIu32"]", value, *result);
        break;
    case UCT_ATOMIC_OP_OR:
        *result = ucs_atomic_for32(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_FOR32 [value %"PRIu32
                             " result %"PRIu32"]", value, *result);
        break;
    case UCT_ATOMIC_OP_XOR:
        *result = ucs_atomic_fxor32(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_FXOR32 [value %"PRIu32
                             " result %"PRIu32"]", value, *result);
        break;
    case UCT_ATOMIC_OP_SWAP:
        *result = ucs_atomic_swap32(ptr, value);
        uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_SWAP32 [value %"PRIu32
                             " result %"PRIu32"]", value, *result);
        break;
    default:
        ucs_assertv(0, "incorrect opcode: %d", opcode);
        return UCS_ERR_UNSUPPORTED;
    }

    UCT_TL_EP_STAT_ATOMIC(ucs_derived_of(ep, uct_base_ep_t));
    return UCS_OK;
}

ucs_status_t uct_sm_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare,
                                      uint64_t swap, uint64_t remote_addr,
                                      uct_rkey_t rkey, uint64_t *result,
                                      uct_completion_t *comp)
{
    uint64_t *ptr = (uint64_t *)(rkey + remote_addr);
    *result = ucs_atomic_cswap64(ptr, compare, swap);
    uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_CSWAP64 [compare %"PRIu64
    		             " swap %"PRIu64" result %"PRIu64"]", compare, swap,
    		             *result);
    UCT_TL_EP_STAT_ATOMIC(ucs_derived_of(tl_ep, uct_base_ep_t));
    return UCS_OK;
}

ucs_status_t uct_sm_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare,
                                      uint32_t swap, uint64_t remote_addr,
                                      uct_rkey_t rkey, uint32_t *result,
                                      uct_completion_t *comp)
{
    uint32_t *ptr = (uint32_t *)(rkey + remote_addr);
    *result = ucs_atomic_cswap32(ptr, compare, swap);
    uct_sm_ep_trace_data(remote_addr, rkey, "ATOMIC_CSWAP32 [compare %"PRIu32
    		             " swap %"PRIu32" result %"PRIu32"]", compare, swap,
    		             *result);
    UCT_TL_EP_STAT_ATOMIC(ucs_derived_of(tl_ep, uct_base_ep_t));
    return UCS_OK;
}
