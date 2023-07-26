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

static UCS_F_ALWAYS_INLINE void
ucp_memh_rcache_print(ucp_mem_h memh, void *address, size_t length)
{
    const char UCS_V_UNUSED *type = (memh->flags & UCP_MEMH_FLAG_IMPORTED) ?
                                    "imported " : "";

    ucs_trace("%smemh %p: address %p/%p length %zu/%zu md_map %" PRIx64
              " obtained from rcache", type, memh, address,
              ucp_memh_address(memh), length, ucp_memh_length(memh),
              memh->md_map);
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
            ucp_memh_rcache_print(memh, address, length);
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
    ucs_rcache_t *rcache;
    khiter_t iter;

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

    UCP_THREAD_CS_ENTER(&context->mt_lock);
    if (memh->flags & UCP_MEMH_FLAG_IMPORTED) {
        iter = kh_get(ucp_context_imported_mem_hash,
                      context->imported_mem_hash, memh->remote_uuid);
        ucs_assert(iter != kh_end(context->imported_mem_hash));

        rcache = kh_value(context->imported_mem_hash, iter);
        ucs_assert(rcache != NULL);
        ucs_rcache_region_put_unsafe(rcache, &memh->super);
    } else {
        ucs_rcache_region_put_unsafe(context->rcache, &memh->super);
    }
    UCP_THREAD_CS_EXIT(&context->mt_lock);
}

static UCS_F_ALWAYS_INLINE int ucp_memh_is_user_memh(ucp_mem_h memh)
{
    return (memh->parent != NULL) && (memh->parent != memh);
}

static UCS_F_ALWAYS_INLINE int ucp_memh_is_buffer_in_range(const ucp_mem_h memh,
                                                           const void *buffer,
                                                           size_t length)
{
    const void *memh_address = ucp_memh_address(memh);

    if ((buffer < memh_address) ||
        (UCS_PTR_BYTE_OFFSET(buffer, length) >
         UCS_PTR_BYTE_OFFSET(memh_address, ucp_memh_length(memh)))) {
        return 0;
    }

    return 1;
}

static UCS_F_ALWAYS_INLINE int
ucp_memh_is_iov_buffer_in_range(const ucp_mem_h memh, const void *buffer,
                                size_t iov_count, ucs_string_buffer_t *err_msg)
{
    const ucp_dt_iov_t *iov = (const ucp_dt_iov_t*)buffer;
    size_t iov_index;

    for (iov_index = 0; iov_index < iov_count; ++iov_index) {
        if (!ucp_memh_is_buffer_in_range(memh, iov[iov_index].buffer,
                                         iov[iov_index].length)) {
            ucs_string_buffer_appendf(err_msg,
                                      "iov[%zu] [buffer %p length %zu]",
                                      iov_index, iov[iov_index].buffer,
                                      iov[iov_index].length);
            return 0;
        }
    }

    return 1;
}

#endif
