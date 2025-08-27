/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "gdaki_mem.h"
#include <cuda_runtime.h>

#define CUDA_CALL(_func, ...) \
    do { \
        cudaError_t _cerr = _func(__VA_ARGS__); \
        if (_cerr != cudaSuccess) { \
            ucs_error("%s() failed: %d (%s)", UCS_PP_MAKE_STRING(_func), \
                      _cerr, cudaGetErrorString(_cerr)); \
        } \
    } while (0)


gdaki_mem::gdaki_mem(size_t size) : m_size(size)
{
    CUDA_CALL(cudaSetDeviceFlags, cudaDeviceMapHost |
                                  cudaDeviceScheduleBlockingSync);
    CUDA_CALL(cudaHostAlloc, &m_cpu_ptr, size, cudaHostAllocMapped);
    CUDA_CALL(cudaHostGetDevicePointer, &m_gpu_ptr, m_cpu_ptr, 0);
}

gdaki_mem::~gdaki_mem()
{
    CUDA_CALL(cudaFreeHost, m_cpu_ptr);
}
