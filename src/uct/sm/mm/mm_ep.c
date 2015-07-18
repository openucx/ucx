/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include "mm_ep.h"


static UCS_CLASS_INIT_FUNC(uct_mm_ep_t, uct_iface_t *tl_iface,
                           const struct sockaddr *addr)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_iface, uct_mm_iface_t);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_mm_ep_t)
{
    /* No op */
}

UCS_CLASS_DEFINE(uct_mm_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_mm_ep_t, uct_ep_t, uct_iface_t*,
                          const struct sockaddr *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_mm_ep_t, uct_ep_t);


#define uct_mm_trace_data(_remote_addr, _rkey, _fmt, ...) \
     ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_remote_addr), \
                    (_rkey))

ucs_status_t uct_mm_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                 unsigned length, uint64_t remote_addr,
                                 uct_rkey_t rkey)
{
    if (ucs_likely(length != 0)) {
        memcpy((void *)(rkey + remote_addr), buffer, length);
        uct_mm_trace_data(remote_addr, rkey, "PUT_SHORT [buffer %p size %u]",
                          buffer, length);
    } else {
        ucs_trace_data("PUT_SHORT [zero-length]");
    }
    return UCS_OK;
}

ucs_status_t uct_mm_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                 void *arg, size_t length, 
                                 uint64_t remote_addr, uct_rkey_t rkey)
{
    if (ucs_likely(length != 0)) {
        pack_cb((void *)(rkey + remote_addr), arg, length);
        uct_mm_trace_data(remote_addr, rkey, "PUT_BCOPY [size %zu]", length);
    } else {
        ucs_trace_data("PUT_BCOPY [zero-length]");
    }
    return UCS_OK;
}

ucs_status_t uct_mm_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                                const void *payload, unsigned length)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t uct_mm_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                    uint64_t remote_addr, uct_rkey_t rkey)
{
    uint64_t *ptr = (uint64_t *)(rkey + remote_addr);
    ucs_atomic_add64(ptr, add);
    uct_mm_trace_data(remote_addr, rkey, "ATOMIC_ADD64 [add %"PRIu64"]", add);
    return UCS_OK;
}

ucs_status_t uct_mm_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint64_t *result, uct_completion_t *comp)
{
    uint64_t *ptr = (uint64_t *)(rkey + remote_addr);
    *result = ucs_atomic_fadd64(ptr, add);
    uct_mm_trace_data(remote_addr, rkey,
                      "ATOMIC_FADD64 [add %"PRIu64" result %"PRIu64"]",
                      add, *result);
    return UCS_OK;
}

ucs_status_t uct_mm_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint64_t *result, uct_completion_t *comp)
{
    uint64_t *ptr = (uint64_t *)(rkey + remote_addr);
    *result = ucs_atomic_swap64(ptr, swap);
    uct_mm_trace_data(remote_addr, rkey,
                      "ATOMIC_SWAP64 [swap %"PRIu64" result %"PRIu64"]",
                      swap, *result);
    return UCS_OK;
}

ucs_status_t uct_mm_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare,
                                      uint64_t swap, uint64_t remote_addr, 
                                      uct_rkey_t rkey, uint64_t *result,
                                      uct_completion_t *comp)
{
    uint64_t *ptr = (uint64_t *)(rkey + remote_addr);
    *result = ucs_atomic_cswap64(ptr, compare, swap);
    uct_mm_trace_data(remote_addr, rkey,
                      "ATOMIC_CSWAP64 [compare %"PRIu64" swap %"PRIu64" result %"PRIu64"]",
                      compare, swap, *result);
    return UCS_OK;
}

ucs_status_t uct_mm_ep_atomic_add32(uct_ep_h tl_ep, uint32_t add,
                                    uint64_t remote_addr, uct_rkey_t rkey)
{
    uint32_t *ptr = (uint32_t *)(rkey + remote_addr);
    ucs_atomic_add32(ptr, add);
    uct_mm_trace_data(remote_addr, rkey, "ATOMIC_ADD32 [add %"PRIu32"]", add);
    return UCS_OK;
}

ucs_status_t uct_mm_ep_atomic_fadd32(uct_ep_h tl_ep, uint32_t add,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint32_t *result, uct_completion_t *comp)
{
    uint32_t *ptr = (uint32_t *)(rkey + remote_addr);
    *result = ucs_atomic_fadd32(ptr, add);
    uct_mm_trace_data(remote_addr, rkey,
                      "ATOMIC_FADD32 [add %"PRIu32" result %"PRIu32"]",
                      add, *result);
    return UCS_OK;
}

ucs_status_t uct_mm_ep_atomic_swap32(uct_ep_h tl_ep, uint32_t swap,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint32_t *result, uct_completion_t *comp)
{
    uint32_t *ptr = (uint32_t *)(rkey + remote_addr);
    *result = ucs_atomic_swap32(ptr, swap);
    uct_mm_trace_data(remote_addr, rkey,
                      "ATOMIC_SWAP32 [swap %"PRIu32" result %"PRIu32"]",
                      swap, *result);
    return UCS_OK;
}

ucs_status_t uct_mm_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare,
                                      uint32_t swap, uint64_t remote_addr, 
                                      uct_rkey_t rkey, uint32_t *result,
                                      uct_completion_t *comp)
{
    uint32_t *ptr = (uint32_t *)(rkey + remote_addr);
    *result = ucs_atomic_cswap32(ptr, compare, swap);
    uct_mm_trace_data(remote_addr, rkey,
                      "ATOMIC_CSWAP32 [compare %"PRIu32" swap %"PRIu32" result %"PRIu32"]",
                      compare, swap, *result);
    return UCS_OK;
}

ucs_status_t uct_mm_ep_get_bcopy(uct_ep_h tl_ep, uct_unpack_callback_t unpack_cb,
                                 void *arg, size_t length,
                                 uint64_t remote_addr, uct_rkey_t rkey,
                                 uct_completion_t *comp)
{
    if (ucs_likely(0 != length)) {
        unpack_cb(arg, (void *)(rkey + remote_addr), length);
        uct_mm_trace_data(remote_addr, rkey, "GET_BCOPY [length %zu]", length);
    } else {
        ucs_trace_data("GET_BCOPY [zero-length]");
    }
    return UCS_OK;
}
