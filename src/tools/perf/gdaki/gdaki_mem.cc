/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "gdaki_mem.h"
#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CALL(_handler, _func, ...) \
    do { \
        cudaError_t _cerr = _func(__VA_ARGS__); \
        if (_cerr != cudaSuccess) { \
            char _msg[256]; \
            snprintf(_msg, sizeof(_msg), "%s() failed: %d (%s)", \
                     UCS_PP_MAKE_STRING(_func), (int)_cerr, \
                     cudaGetErrorString(_cerr)); \
            _handler(_msg); \
        } \
    } while (0)

#define CUDA_CALL_THROW(_func, ...) \
    CUDA_CALL(throw std::runtime_error, _func, __VA_ARGS__)

gdaki_mem::gdaki_mem(size_t size) : m_size(size)
{
    CUDA_CALL_THROW(cudaSetDeviceFlags, cudaDeviceMapHost |
                                        cudaDeviceScheduleBlockingSync);
    CUDA_CALL_THROW(cudaHostAlloc, &m_cpu_ptr, size, cudaHostAllocMapped);
    CUDA_CALL([this](const char *msg) {
        CUDA_CALL(ucs_warn, cudaFreeHost, m_cpu_ptr);
        throw std::runtime_error(msg);
    }, cudaHostGetDevicePointer, &m_gpu_ptr, m_cpu_ptr, 0);
}

gdaki_mem::~gdaki_mem()
{
    CUDA_CALL(ucs_warn, cudaFreeHost, m_cpu_ptr);
}
