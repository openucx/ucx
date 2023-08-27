/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>

#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

#if HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <tools/perf/lib/libperf_int.h>
#include <ucs/memory/memory_type.h>
#include <ucs/sys/compiler.h>


#if HAVE_CUDA
static ucs_status_t ucx_perf_cuda_init(const ucx_perf_context_t *perf)
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

static void *ucx_perf_cuda_memcpy(void *dst, const void *src, size_t length)
{
    cudaError_t cerr;

    cerr = cudaMemcpy(dst, src, length, cudaMemcpyDefault);
    if (cerr != cudaSuccess) {
        ucs_error("failed to copy memory: %s", cudaGetErrorString(cerr));
    }

    cerr = cudaDeviceSynchronize();
    if (cerr != cudaSuccess) {
        ucs_error("failed to sync device: %s", cudaGetErrorString(cerr));
    }
    return dst;
}
#endif

#if HAVE_ROCM
static ucs_status_t ucx_perf_rocm_init(const ucx_perf_context_t *perf)
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

static void *ucx_perf_rocm_memcpy(void *dst, const void *src, size_t length)
{
    hipError_t ret;

    ret = hipMemcpy(dst, src, length, hipMemcpyDefault);
    if (ret != hipSuccess) {
        ucs_error("failed to copy memory: %s", hipGetErrorString(ret));
    }
    return dst;
}
#endif

memcpy_func_t ucx_get_perf_memcpy(const ucx_perf_context_t *perf)
{
    ucs_memory_type_t send_mem_type = perf->params.send_mem_type,
                      recv_mem_type = perf->params.recv_mem_type;
    ucs_memory_type_t mem_type;
#if HAVE_CUDA || HAVE_ROCM
    ucs_status_t status;
#endif

    if (send_mem_type == recv_mem_type) {
        mem_type = send_mem_type;
    } else if (send_mem_type != UCS_MEMORY_TYPE_HOST) {
        mem_type = send_mem_type;
    } else {
        mem_type = recv_mem_type;
    }

    switch (mem_type) {
    case UCS_MEMORY_TYPE_HOST:
    case UCS_MEMORY_TYPE_RDMA:
        return memcpy;
#if HAVE_CUDA
    case UCS_MEMORY_TYPE_CUDA:
    case UCS_MEMORY_TYPE_CUDA_MANAGED:
        status = ucx_perf_cuda_init(perf);
        if(status != UCS_OK) {
            break;
        }
        return ucx_perf_cuda_memcpy;
#endif
#if HAVE_ROCM
    case UCS_MEMORY_TYPE_ROCM:
    case UCS_MEMORY_TYPE_ROCM_MANAGED:
        status = ucx_perf_rocm_init(perf);
        if(status != UCS_OK) {
            break;
        }
        return ucx_perf_rocm_memcpy;
#endif
    default:
        ucs_error("memcpy was not defined for memory type: "
                  "%s", ucs_memory_type_names[mem_type]);
        return NULL;
    }

    ucs_error("Failed to init memory type: "
              "%s", ucs_memory_type_names[mem_type]);
    return NULL;
}
