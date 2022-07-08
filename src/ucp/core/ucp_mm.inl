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
    ucs_status_t status;
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
        status = ucs_rcache_get_unsafe(context->rcache, address, length,
                                       PROT_READ | PROT_WRITE, NULL, &rregion);
        if (status != UCS_OK) {
            goto out_unlock;
        }

        memh = ucs_derived_of(rregion, ucp_mem_t);

        if (ucs_likely(ucs_test_all_flags(memh->md_map, reg_md_map))) {
            ucs_trace("memh %p: address %p/%p length %zu/%zu md_map %" PRIx64
                      "/%" PRIx64 " obtained from rcache", memh, address,
                      ucp_memh_address(memh), length, ucp_memh_length(memh),
                      reg_md_map, memh->md_map);
            *memh_p = memh;
            status = UCS_OK;
            goto out_unlock;
        }

        ucs_rcache_region_put_unsafe(context->rcache, rregion);
        UCP_THREAD_CS_EXIT(&context->mt_lock);
    }

    return ucp_memh_get_slow(context, address, length, mem_type, reg_md_map,
                             uct_flags, memh_p);
out_unlock:
    UCP_THREAD_CS_EXIT(&context->mt_lock);
    return status;
}

static UCS_F_ALWAYS_INLINE void
ucp_memh_put(ucp_context_h context, ucp_mem_h memh, int invalidate)
{
    ucs_trace("memh %p: release address %p length %zu md_map %" PRIx64
              " invalidate: %d", memh, ucp_memh_address(memh),
              ucp_memh_length(memh), memh->md_map, invalidate);

    if (ucs_unlikely(ucp_memh_is_zero_length(memh))) {
        return;
    }

    if (ucs_unlikely(context->rcache == NULL)) {
        ucp_memh_unmap(context, memh, memh->md_map);
        ucs_free(memh);
        return;
    }

    UCP_THREAD_CS_ENTER(&context->mt_lock);
    if (invalidate) {
        ucs_rcache_region_invalidate(context->rcache, &memh->super,
                (ucs_rcache_invalidate_comp_func_t)ucs_empty_function, NULL);
    }

    ucs_rcache_region_put_unsafe(context->rcache, &memh->super);
    UCP_THREAD_CS_EXIT(&context->mt_lock);
}

#endif
