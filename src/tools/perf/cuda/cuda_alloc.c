/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

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

    return UCS_OK;
}

static ucs_status_t ucp_perf_cuda_alloc(ucx_perf_context_t *perf, size_t length,
                                        void **address_p, ucp_mem_h *memh_p,
                                        int non_blk_flag)
{
    cudaError_t cerr;

    cerr = cudaMalloc(address_p, length);
    if (cerr != cudaSuccess) {
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

static ucs_status_t ucp_perf_cuda_alloc_managed(ucx_perf_context_t *perf,
                                                size_t length, void **address_p,
                                                ucp_mem_h *memh_p, int non_blk_flag)
{
    cudaError_t cerr;

    cerr = cudaMallocManaged(address_p, length, cudaMemAttachGlobal);
    if (cerr != cudaSuccess) {
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

static void ucp_perf_cuda_free(ucx_perf_context_t *perf, void *address,
                               ucp_mem_h memh)
{
    cudaFree(address);
}

UCS_STATIC_INIT {
    static ucx_perf_allocator_t cuda_allocator = {
        .init      = ucx_perf_cuda_init,
        .ucp_alloc = ucp_perf_cuda_alloc,
        .ucp_free  = ucp_perf_cuda_free
    };
    static ucx_perf_allocator_t cuda_managed_allocator = {
        .init      = ucx_perf_cuda_init,
        .ucp_alloc = ucp_perf_cuda_alloc_managed,
        .ucp_free  = ucp_perf_cuda_free
    };
    ucx_perf_mem_type_allocators[UCT_MD_MEM_TYPE_CUDA]         = &cuda_allocator;
    ucx_perf_mem_type_allocators[UCT_MD_MEM_TYPE_CUDA_MANAGED] = &cuda_managed_allocator;
}
