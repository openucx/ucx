/**
 * Copyright (C) Advanced Micro Devices, Inc. 2019.  ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <tools/perf/lib/libperf_int.h>

#include <hip/hip_runtime.h>
#include <ucs/sys/compiler.h>


static ucs_status_t ucx_perf_rocm_init(ucx_perf_context_t *perf)
{
    hipError_t ret;
    unsigned group_index;
    int num_gpus;
    int gpu_index;

    group_index = rte_call(perf, group_index);

    ret = hipGetDeviceCount(&num_gpus);
    if (ret != hipSuccess) {
        return UCS_ERR_NO_DEVICE;
    }

    gpu_index = group_index % num_gpus;

    ret = hipSetDevice(gpu_index);
    if (ret != hipSuccess) {
        return UCS_ERR_NO_DEVICE;
    }

    return UCS_OK;
}

static inline ucs_status_t ucx_perf_rocm_alloc(size_t length,
                                               ucs_memory_type_t mem_type,
                                               void **address_p)
{
    hipError_t ret;

    ucs_assert((mem_type == UCS_MEMORY_TYPE_ROCM) ||
               (mem_type == UCS_MEMORY_TYPE_ROCM_MANAGED));

    ret = ((mem_type == UCS_MEMORY_TYPE_ROCM) ?
            hipMalloc(address_p, length) :
            hipMallocManaged(address_p, length, hipMemAttachGlobal));
    if (ret != hipSuccess) {
        ucs_error("failed to allocate memory");
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

static inline ucs_status_t
uct_perf_rocm_alloc_reg_mem(const ucx_perf_context_t *perf,
                            size_t length,
                            ucs_memory_type_t mem_type,
                            unsigned flags,
                            uct_allocated_memory_t *alloc_mem)
{
    ucs_status_t status;

    status = ucx_perf_rocm_alloc(length, mem_type, &alloc_mem->address);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_md_mem_reg(perf->uct.md, alloc_mem->address,
                            length, flags, &alloc_mem->memh);
    if (status != UCS_OK) {
        hipFree(alloc_mem->address);
        ucs_error("failed to register memory");
        return status;
    }

    alloc_mem->mem_type = mem_type;
    alloc_mem->md       = perf->uct.md;

    return UCS_OK;
}

static ucs_status_t uct_perf_rocm_alloc(const ucx_perf_context_t *perf,
                                        size_t length, unsigned flags,
                                        uct_allocated_memory_t *alloc_mem)
{
    return uct_perf_rocm_alloc_reg_mem(perf, length, UCS_MEMORY_TYPE_ROCM,
                                       flags, alloc_mem);
}

static ucs_status_t uct_perf_rocm_managed_alloc(const ucx_perf_context_t *perf,
                                                size_t length, unsigned flags,
                                                uct_allocated_memory_t *alloc_mem)
{
    return uct_perf_rocm_alloc_reg_mem(perf, length, UCS_MEMORY_TYPE_ROCM_MANAGED,
                                       flags, alloc_mem);
}

static void uct_perf_rocm_free(const ucx_perf_context_t *perf,
                               uct_allocated_memory_t *alloc_mem)
{
    ucs_status_t status;

    ucs_assert(alloc_mem->md == perf->uct.md);

    status = uct_md_mem_dereg(perf->uct.md, alloc_mem->memh);
    if (status != UCS_OK) {
        ucs_error("failed to deregister memory");
    }

    hipFree(alloc_mem->address);
}

static void ucx_perf_rocm_memcpy(void *dst, ucs_memory_type_t dst_mem_type,
                                 const void *src, ucs_memory_type_t src_mem_type,
                                 size_t count)
{
    hipError_t ret;

    ret = hipMemcpy(dst, src, count, hipMemcpyDefault);
    if (ret != hipSuccess) {
        ucs_error("failed to copy memory: %s", hipGetErrorString(ret));
    }
}

static void* ucx_perf_rocm_memset(void *dst, int value, size_t count)
{
    hipError_t ret;

    ret = hipMemset(dst, value, count);
    if (ret != hipSuccess) {
        ucs_error("failed to set memory: %s", hipGetErrorString(ret));
    }

    return dst;
}

UCS_STATIC_INIT {
    static ucx_perf_allocator_t rocm_allocator = {
        .mem_type  = UCS_MEMORY_TYPE_ROCM,
        .init      = ucx_perf_rocm_init,
        .uct_alloc = uct_perf_rocm_alloc,
        .uct_free  = uct_perf_rocm_free,
        .memcpy    = ucx_perf_rocm_memcpy,
        .memset    = ucx_perf_rocm_memset
    };
    static ucx_perf_allocator_t rocm_managed_allocator = {
        .mem_type  = UCS_MEMORY_TYPE_ROCM_MANAGED,
        .init      = ucx_perf_rocm_init,
        .uct_alloc = uct_perf_rocm_managed_alloc,
        .uct_free  = uct_perf_rocm_free,
        .memcpy    = ucx_perf_rocm_memcpy,
        .memset    = ucx_perf_rocm_memset
    };

    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ROCM]         = &rocm_allocator;
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ROCM_MANAGED] = &rocm_managed_allocator;
}
UCS_STATIC_CLEANUP {
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ROCM]         = NULL;
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ROCM_MANAGED] = NULL;

}
