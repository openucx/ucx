/**
 * Copyright (C) Huawei Technologies Co., Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <tools/perf/lib/libperf_int.h>

#include <ucs/sys/compiler.h>


static ucs_status_t ucx_perf_dm_init(ucx_perf_context_t *perf)
{
    return UCS_OK; // TODO: find the UCT MD of the device we want for DM
}

static ucs_status_t ucp_perf_dm_alloc(const ucx_perf_context_t *perf, size_t length,
                                        void **address_p, ucp_mem_h *memh_p,
                                        int non_blk_flag)
{
    return uct_ib_mem_dm_alloc();
}

static void ucp_perf_dm_free(const ucx_perf_context_t *perf,
                               void *address, ucp_mem_h memh)
{
    uct_ib_mem_dm_free();
}

static inline ucs_status_t
uct_perf_dm_alloc_reg_mem(const ucx_perf_context_t *perf,
                            size_t length,
                            ucs_memory_type_t mem_type,
                            unsigned flags,
                            uct_allocated_memory_t *alloc_mem)
{
    ucs_status_t status;

    status = ucx_perf_dm_alloc(length, mem_type, &alloc_mem->address);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_md_mem_reg(perf->uct.md, alloc_mem->address,
                            length, flags, &alloc_mem->memh);
    if (status != UCS_OK) {
        uct_md_mem_free(alloc_mem->address);
        ucs_error("failed to register memory");
        return status;
    }

    alloc_mem->mem_type = mem_type;
    alloc_mem->md       = perf->uct.md;

    return UCS_OK;
}

static ucs_status_t uct_perf_dm_alloc(const ucx_perf_context_t *perf,
                                        size_t length, unsigned flags,
                                        uct_allocated_memory_t *alloc_mem)
{
    return uct_perf_dm_alloc_reg_mem(perf, length, UCS_MEMORY_TYPE_RDMA_DM,
                                       flags, alloc_mem);
}

static void uct_perf_dm_free(const ucx_perf_context_t *perf,
                               uct_allocated_memory_t *alloc_mem)
{
    ucs_status_t status;

    ucs_assert(alloc_mem->md == perf->uct.md);

    status = uct_md_mem_dereg(perf->uct.md, alloc_mem->memh);
    if (status != UCS_OK) {
        ucs_error("failed to deregister memory");
    }

    uct_md_mem_free(alloc_mem->address);
}

static void ucx_perf_dm_memcpy(void *dst, ucs_memory_type_t dst_mem_type,
                               const void *src, ucs_memory_type_t src_mem_type,
                               size_t count)
{
    struct ibv_dm *dm;
    int ret;

    if (dst_mem_type == UCS_MEMORY_TYPE_RDMA_DM) {
        if (src_mem_type != UCS_MEMORY_TYPE_HOST) {
            goto memcpy_unsupported_error;
        }

        ret = ibv_memcpy_to_dm(dm, uint64_t dm_offset, const void *host_addr, size_t length);
    } else if (dst_mem_type == UCS_MEMORY_TYPE_HOST) {
        if (src_mem_type != UCS_MEMORY_TYPE_RDMA_DM) {
            goto memcpy_unsupported_error;
        }

        ret = ibv_memcpy_from_dm(dm, uint64_t dm_offset, const void *host_addr, size_t length);
    } else {
        goto memcpy_unsupported_error;
    }

    if (ret) {
        ucs_error("failed to copy RDMA device memory");
    }
    return;

memcpy_unsupported_error:
    ucs_error("failed to copy memory: can only copy from or to the device");
}

static void* ucx_perf_dm_memset(void *dst, int value, size_t count)
{
    void *tmp = ucs_malloc(count, "perf_dm_memset");
    if (tmp == NULL) {
        ucs_error("failed to allocate memory for memset");
        return NULL;
    }

    (void)memset(tmp, value, count);
    ucx_perf_dm_memcpy(dst, UCS_MEMORY_TYPE_RDMA_DM, tmp, UCS_MEMORY_TYPE_HOST,
                       count);

    ucs_free(tmp);
    return dst;
}

UCS_STATIC_INIT {
    static ucx_perf_allocator_t dm_allocator = {
        .mem_type  = UCS_MEMORY_TYPE_RDMA_DM,
        .init      = ucx_perf_dm_init,
        .ucp_alloc = ucp_perf_dm_alloc,
        .ucp_free  = ucp_perf_dm_free,
        .uct_alloc = uct_perf_dm_alloc,
        .uct_free  = uct_perf_dm_free,
        .memcpy    = ucx_perf_dm_memcpy,
        .memset    = ucx_perf_dm_memset
    };

    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_RDMA_DM] = &dm_allocator;
}
UCS_STATIC_CLEANUP {
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_RDMA_DM] = NULL;

}
