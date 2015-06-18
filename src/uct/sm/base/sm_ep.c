/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include "sm_ep.h"


UCS_CLASS_INIT_FUNC(uct_sm_ep_t, uct_sm_iface_t *sm_iface)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &sm_iface->super)
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_sm_ep_t)
{
    /* No op */
}

UCS_CLASS_DEFINE(uct_sm_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_sm_ep_t, uct_sm_iface_t, uct_sm_iface_t*);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_sm_ep_t, uct_ep_t);

ucs_status_t uct_sm_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                 unsigned length, uint64_t remote_addr,
                                 uct_rkey_t rkey)
{
    uct_sm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_sm_iface_t); 

    if (0 == length) {
        ucs_trace_data("Zero length request: skip it");
        return UCS_OK;
    }
    /* FIXME make this user-configurable */
    UCT_CHECK_LENGTH(length <= iface->config.max_put, "put_short");

    /* FIXME add debug/assertion to check remote_addr within attached region */

    memcpy((void *)(rkey + remote_addr), buffer, length);

    ucs_trace_data("Posting PUT Short, memcpy of size %u to %p",
                    length,
                    (void *)(remote_addr));

    return UCS_OK;
}

ucs_status_t uct_sm_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                 void *arg, size_t length, 
                                 uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_sm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_sm_iface_t); 
    if (0 == length) {
        ucs_trace_data("Zero length request: skip it");
        return UCS_OK;
    }

    /* FIXME make this user-configurable */
    UCT_CHECK_LENGTH(length <= iface->config.max_bcopy, "put_bcopy");

    /* FIXME add debug/assertion to check remote_addr within attached region */

    pack_cb((void *)(rkey + remote_addr), arg, length);

    ucs_trace_data("Posting PUT BCOPY of size %zd to %p",
                    length,
                    (void *)(remote_addr));

    return UCS_OK;
}

ucs_status_t uct_sm_ep_put_zcopy(uct_ep_h tl_ep, const void *buffer, 
                                 size_t length, uct_mem_h memh, 
                                 uint64_t remote_addr, uct_rkey_t rkey, 
                                 uct_completion_t *comp)
{
    uct_sm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_sm_iface_t); 
    if (0 == length) {
        ucs_trace_data("Zero length request: skip it");
        return UCS_OK;
    }

    /* FIXME make this user-configurable */
    UCT_CHECK_LENGTH(length <= iface->config.max_zcopy, "put_zcopy");

    /* FIXME add debug/assertion to check remote_addr within attached region */

    memcpy((void *)(rkey + remote_addr), buffer, length);

    ucs_trace_data("Posting PUT ZCOPY of size %zd to %p",
                    length,
                    (void *)(remote_addr));

    return UCS_OK;
}


ucs_status_t uct_sm_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                                const void *payload, unsigned length)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t uct_sm_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                    uint64_t remote_addr, uct_rkey_t rkey)
{
    uint64_t *ptr = (uint64_t *)(rkey + remote_addr);
    ucs_atomic_add64(ptr, add);
    ucs_trace_data("Posting atomic_add64, value %"PRIx64" to %p",
                    add,
                    (void *)(remote_addr));
    return UCS_OK;
}

ucs_status_t uct_sm_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint64_t *result, uct_completion_t *comp)
{
    uint64_t *ptr = (uint64_t *)(rkey + remote_addr);
    *result = ucs_atomic_fadd64(ptr, add);
    ucs_trace_data("Posting atomic_fadd64, value %"PRIx64" to %p",
                    add,
                    (void *)(remote_addr));
    return UCS_OK;
}

ucs_status_t uct_sm_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint64_t *result, uct_completion_t *comp)
{
    uint64_t *ptr = (uint64_t *)(rkey + remote_addr);
    *result = ucs_atomic_swap64(ptr, swap);
    ucs_trace_data("Posting atomic_swap64, value %"PRIx64" to %p",
                    swap,
                    (void *)(remote_addr));
    return UCS_OK;
}

ucs_status_t uct_sm_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, 
                                      uint64_t swap, uint64_t remote_addr, 
                                      uct_rkey_t rkey, uint64_t *result,
                                      uct_completion_t *comp)
{
    uint64_t *ptr = (uint64_t *)(rkey + remote_addr);
    *result = ucs_atomic_cswap64(ptr, compare, swap);
    ucs_trace_data("Posting atomic_cswap64, value %"PRIx64" to %p",
                    swap,
                    (void *)(remote_addr));
    return UCS_OK;
}

ucs_status_t uct_sm_ep_atomic_add32(uct_ep_h tl_ep, uint32_t add,
                                    uint64_t remote_addr, uct_rkey_t rkey)
{
    uint32_t *ptr = (uint32_t *)(rkey + remote_addr);
    ucs_atomic_add32(ptr, add);
    ucs_trace_data("Posting atomic_add32, value %"PRIx32" to %p",
                    add,
                    (void *)remote_addr);
    return UCS_OK;
}

ucs_status_t uct_sm_ep_atomic_fadd32(uct_ep_h tl_ep, uint32_t add,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint32_t *result, uct_completion_t *comp)
{
    uint32_t *ptr = (uint32_t *)(rkey + remote_addr);
    *result = ucs_atomic_fadd32(ptr, add);
    ucs_trace_data("Posting atomic_fadd32, value %"PRIx32" to %p",
                    add,
                    (void *)(remote_addr));
    return UCS_OK;
}

ucs_status_t uct_sm_ep_atomic_swap32(uct_ep_h tl_ep, uint32_t swap,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint32_t *result, uct_completion_t *comp)
{
    uint32_t *ptr = (uint32_t *)(rkey + remote_addr);
    *result = ucs_atomic_swap32(ptr, swap);
    ucs_trace_data("Posting atomic_swap32, value %"PRIx32" to %p",
                    swap,
                    (void *)(remote_addr));
    return UCS_OK;
}

ucs_status_t uct_sm_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, 
                                      uint32_t swap, uint64_t remote_addr, 
                                      uct_rkey_t rkey, uint32_t *result,
                                      uct_completion_t *comp)
{
    uint32_t *ptr = (uint32_t *)(rkey + remote_addr);
    *result = ucs_atomic_cswap32(ptr, compare, swap);
    ucs_trace_data("Posting atomic_cswap32, value %"PRIx32" to %p",
                    swap,
                    (void *)(remote_addr));
    return UCS_OK;
}

ucs_status_t uct_sm_ep_get_bcopy(uct_ep_h tl_ep, uct_unpack_callback_t unpack_cb,
                                 void *arg, size_t length,
                                 uint64_t remote_addr, uct_rkey_t rkey,
                                 uct_completion_t *comp)
{
    uct_sm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_sm_iface_t); 
    if (0 == length) {
        ucs_trace_data("Zero length request: skip it");
        return UCS_OK;
    }

    /* FIXME make this user-configurable */
    UCT_CHECK_LENGTH(length <= iface->config.max_bcopy, "get_bcopy");

    /* FIXME add debug/assertion to check remote_addr within attached region */

    unpack_cb(arg, (void *)(rkey + remote_addr), length);

    ucs_trace_data("Posting GET BCOPY of size %zd to %p",
                    length,
                    (void *)(remote_addr));

    return UCS_OK;
}

ucs_status_t uct_sm_ep_get_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                 uct_mem_h memh, uint64_t remote_addr,
                                 uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_sm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_sm_iface_t); 
    if (0 == length) {
        ucs_trace_data("Zero length request: skip it");
        return UCS_OK;
    }

    /* FIXME make this user-configurable */
    UCT_CHECK_LENGTH(length <= iface->config.max_zcopy, "get_zcopy");

    /* FIXME add debug/assertion to check remote_addr within attached region */

    memcpy(buffer, (void *)(rkey + remote_addr), length);

    ucs_trace_data("Posting GET ZCOPY of size %zd to %p",
                    length,
                    (void *)(remote_addr));

    return UCS_OK;
}
