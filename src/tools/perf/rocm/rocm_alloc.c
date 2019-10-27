/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <tools/perf/lib/libperf_int.h>
#include <ucs/sys/compiler.h>
#include "hip/hip_runtime.h"

static ucs_status_t ucx_perf_rocm_init(ucx_perf_context_t *perf)
{
    hipError_t hiperr;
    unsigned group_index;
    int num_gpus;
    int gpu_index;

    group_index = rte_call(perf, group_index);

    hiperr = hipGetDeviceCount(&num_gpus);
    if (hiperr != hipSuccess) {
        return UCS_ERR_NO_DEVICE;
    }

    gpu_index = group_index % num_gpus;

    hiperr = hipSetDevice(gpu_index);
    if (hiperr != hipSuccess) {
        return UCS_ERR_NO_DEVICE;
    }

    return UCS_OK;
}

static ucs_status_t ucp_perf_rocm_alloc(ucx_perf_context_t *perf, size_t length,
                                        void **address_p, ucp_mem_h *memh_p,
                                        int non_blk_flag)
{
    hipError_t hiperr;

    hiperr = hipMalloc(address_p, length);
    if (hiperr != hipSuccess) {
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

static ucs_status_t ucp_perf_rocm_alloc_managed(ucx_perf_context_t *perf,
                                                size_t length, void **address_p,
                                                ucp_mem_h *memh_p, int non_blk_flag)
{
    hipError_t hiperr;

    hiperr = hipMallocManaged(address_p, length, hipMemAttachGlobal);
    if (hiperr != hipSuccess) {
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

static void ucp_perf_rocm_free(ucx_perf_context_t *perf, void *address,
                               ucp_mem_h memh)
{
    hipFree(address);
}

static void* ucp_perf_rocm_memset(void *s, int c, size_t len)
{
    hipMemset(s, c, len);
    return s;
}

UCS_STATIC_INIT {
    static ucx_perf_allocator_t hip_allocator = {
        .init      = ucx_perf_rocm_init,
        .ucp_alloc = ucp_perf_rocm_alloc,
        .ucp_free  = ucp_perf_rocm_free,
        .memset    = ucp_perf_rocm_memset
    };
    static ucx_perf_allocator_t hip_managed_allocator = {
        .init      = ucx_perf_rocm_init,
        .ucp_alloc = ucp_perf_rocm_alloc_managed,
        .ucp_free  = ucp_perf_rocm_free,
        .memset    = ucp_perf_rocm_memset
    };
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ROCM]         = &hip_allocator;
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ROCM_MANAGED] = &hip_managed_allocator;
}
UCS_STATIC_CLEANUP {
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ROCM]         = NULL;
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ROCM_MANAGED] = NULL;

}
