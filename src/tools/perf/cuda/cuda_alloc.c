/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <tools/perf/lib/libperf_int.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ucs/sys/compiler.h>


static ucs_status_t ucx_perf_cuda_init(ucx_perf_context_t *perf)
{
    cudaError_t cerr;
    unsigned group_index;
    int num_gpus;
    int gpu_index;

    group_index = rte_call(perf, group_index);

    cerr = cudaGetDeviceCount(&num_gpus);
    if (cerr != cudaSuccess) {
        return UCS_ERR_NO_DEVICE;
    }

    gpu_index = group_index % num_gpus;

    cerr = cudaSetDevice(gpu_index);
    if (cerr != cudaSuccess) {
        return UCS_ERR_NO_DEVICE;
    }

    /* actually set device context as calling cudaSetDevice may result in
     * context being initialized lazily */
    cudaFree(0);

    return UCS_OK;
}

static inline ucs_status_t ucx_perf_cuda_alloc(size_t length,
                                               ucs_memory_type_t mem_type,
                                               void **address_p)
{
    cudaError_t cerr;

    ucs_assert((mem_type == UCS_MEMORY_TYPE_CUDA) ||
               (mem_type == UCS_MEMORY_TYPE_CUDA_MANAGED));

    cerr = ((mem_type == UCS_MEMORY_TYPE_CUDA) ?
            cudaMalloc(address_p, length) :
            cudaMallocManaged(address_p, length, cudaMemAttachGlobal));
    if (cerr != cudaSuccess) {
        ucs_error("failed to allocate memory");
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

static inline ucs_status_t
uct_perf_cuda_alloc_reg_mem(const ucx_perf_context_t *perf,
                            size_t length,
                            ucs_memory_type_t mem_type,
                            unsigned flags,
                            uct_allocated_memory_t *alloc_mem)
{
    ucs_status_t status;

    status = ucx_perf_cuda_alloc(length, mem_type, &alloc_mem->address);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_md_mem_reg(perf->uct.md, alloc_mem->address,
                            length, flags, &alloc_mem->memh);
    if (status != UCS_OK) {
        cudaFree(alloc_mem->address);
        ucs_error("failed to register memory");
        return status;
    }

    alloc_mem->mem_type = mem_type;
    alloc_mem->md       = perf->uct.md;

    return UCS_OK;
}

static ucs_status_t uct_perf_cuda_alloc(const ucx_perf_context_t *perf,
                                        size_t length, unsigned flags,
                                        uct_allocated_memory_t *alloc_mem)
{
    return uct_perf_cuda_alloc_reg_mem(perf, length, UCS_MEMORY_TYPE_CUDA,
                                       flags, alloc_mem);
}

static ucs_status_t uct_perf_cuda_managed_alloc(const ucx_perf_context_t *perf,
                                                size_t length, unsigned flags,
                                                uct_allocated_memory_t *alloc_mem)
{
    return uct_perf_cuda_alloc_reg_mem(perf, length, UCS_MEMORY_TYPE_CUDA_MANAGED,
                                       flags, alloc_mem);
}

static void uct_perf_cuda_free(const ucx_perf_context_t *perf,
                               uct_allocated_memory_t *alloc_mem)
{
    ucs_status_t status;

    ucs_assert(alloc_mem->md == perf->uct.md);

    status = uct_md_mem_dereg(perf->uct.md, alloc_mem->memh);
    if (status != UCS_OK) {
        ucs_error("failed to deregister memory");
    }

    cudaFree(alloc_mem->address);
}

static void ucx_perf_cuda_memcpy(void *dst, ucs_memory_type_t dst_mem_type,
                                 const void *src, ucs_memory_type_t src_mem_type,
                                 size_t count)
{
    cudaError_t cerr;

    cerr = cudaMemcpy(dst, src, count, cudaMemcpyDefault);
    if (cerr != cudaSuccess) {
        ucs_error("failed to copy memory: %s", cudaGetErrorString(cerr));
    }

    cerr = cudaDeviceSynchronize();
    if (cerr != cudaSuccess) {
        ucs_error("failed to sync device: %s", cudaGetErrorString(cerr));
    }
}

static void* ucx_perf_cuda_memset(void *dst, int value, size_t count)
{
    cudaError_t cerr;

    cerr = cudaMemset(dst, value, count);
    if (cerr != cudaSuccess) {
        ucs_error("failed to set memory: %s", cudaGetErrorString(cerr));
    }

    return dst;
}

UCS_STATIC_INIT {
    static ucx_perf_allocator_t cuda_allocator = {
        .mem_type  = UCS_MEMORY_TYPE_CUDA,
        .init      = ucx_perf_cuda_init,
        .uct_alloc = uct_perf_cuda_alloc,
        .uct_free  = uct_perf_cuda_free,
        .memcpy    = ucx_perf_cuda_memcpy,
        .memset    = ucx_perf_cuda_memset
    };
    static ucx_perf_allocator_t cuda_managed_allocator = {
        .mem_type  = UCS_MEMORY_TYPE_CUDA_MANAGED,
        .init      = ucx_perf_cuda_init,
        .uct_alloc = uct_perf_cuda_managed_alloc,
        .uct_free  = uct_perf_cuda_free,
        .memcpy    = ucx_perf_cuda_memcpy,
        .memset    = ucx_perf_cuda_memset
    };

    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_CUDA]         = &cuda_allocator;
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_CUDA_MANAGED] = &cuda_managed_allocator;
}
UCS_STATIC_CLEANUP {
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_CUDA]         = NULL;
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_CUDA_MANAGED] = NULL;

}
