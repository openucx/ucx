/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_MM_INL_
#define UCP_MM_INL_

#include "ucp_mm.h"

#include <ucs/memory/rcache.inl>


static UCS_F_ALWAYS_INLINE int
ucp_memh_is_zero_length(const ucp_mem_h memh)
{
    return memh == &ucp_mem_dummy_handle.memh;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_memh_get(ucp_context_h context, void *address, size_t length,
             ucs_memory_type_t mem_type, ucp_md_map_t reg_md_map,
             unsigned uct_flags, ucp_mem_h *memh_p)
{
    ucs_rcache_region_t *rregion;
    ucp_mem_h memh;

    if (length == 0) {
        ucs_assert(ucp_memh_address(&ucp_mem_dummy_handle.memh) == NULL);
        ucs_assert(ucp_memh_length(&ucp_mem_dummy_handle.memh) == 0);

        *memh_p = &ucp_mem_dummy_handle.memh;
        ucs_trace("memh %p: address %p, obtained dummy", *memh_p, address);
        return UCS_OK;
    }

    if (ucs_likely(context->rcache != NULL)) {
        UCP_THREAD_CS_ENTER(&context->mt_lock);
        rregion = ucs_rcache_lookup_unsafe(context->rcache, address, length,
                                           PROT_READ | PROT_WRITE);
        if (rregion == NULL) {
            goto not_found;
        }

        memh = ucs_derived_of(rregion, ucp_mem_t);

        if (ucs_likely(ucs_test_all_flags(memh->md_map, reg_md_map))) {
            ucs_trace("memh %p: address %p/%p length %zu/%zu md_map %" PRIx64
                      "/%" PRIx64 " obtained from rcache", memh, address,
                      ucp_memh_address(memh), length, ucp_memh_length(memh),
                      reg_md_map, memh->md_map);
            *memh_p = memh;
            UCP_THREAD_CS_EXIT(&context->mt_lock);
            return UCS_OK;
        }

        ucs_rcache_region_put_unsafe(context->rcache, rregion);
not_found:
        UCP_THREAD_CS_EXIT(&context->mt_lock);
    }

    return ucp_memh_get_slow(context, address, length, mem_type, reg_md_map,
                             uct_flags, memh_p);
}

static UCS_F_ALWAYS_INLINE void
ucp_memh_put(ucp_context_h context, ucp_mem_h memh)
{
    ucs_trace("memh %p: release address %p length %zu md_map %" PRIx64,
              memh, ucp_memh_address(memh), ucp_memh_length(memh),
              memh->md_map);

    if (ucs_unlikely(ucp_memh_is_zero_length(memh))) {
        return;
    }

    /* User memory handle, or rcache was disabled */
    if (ucs_unlikely(memh->parent != NULL)) {
        ucp_memh_cleanup(context, memh);
        ucs_free(memh);
        return;
    }

    ucs_assert(context->rcache != NULL);

    UCP_THREAD_CS_ENTER(&context->mt_lock);
    ucs_rcache_region_put_unsafe(context->rcache, &memh->super);
    UCP_THREAD_CS_EXIT(&context->mt_lock);
}

#endif
