/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "device_mem.h"
#include <cuda_runtime.h>

#define CUDA_CALL(_handler, _ret, _func, ...) \
    do { \
        cudaError_t _cerr = _func(__VA_ARGS__); \
        if (_cerr != cudaSuccess) { \
            char _msg[256]; \
            snprintf(_msg, sizeof(_msg), "%s() failed: %d (%s)", \
                     UCS_PP_MAKE_STRING(_func), (int)_cerr, \
                     cudaGetErrorString(_cerr)); \
            _handler(_msg); \
            return _ret; \
        } \
    } while (0)

ucs_status_t device_mem_create(device_mem_t *mem, size_t size)
{
    CUDA_CALL(ucs_error, UCS_ERR_IO_ERROR,
              cudaSetDeviceFlags, cudaDeviceMapHost |
                                  cudaDeviceScheduleBlockingSync);
    CUDA_CALL(ucs_error, UCS_ERR_NO_MEMORY,
              cudaHostAlloc, &mem->cpu_ptr, size, cudaHostAllocMapped);

#define ERR_HANDLER(_msg) \
    ucs_error(_msg); \
    cudaFreeHost(mem->cpu_ptr);

    CUDA_CALL(ERR_HANDLER, UCS_ERR_IO_ERROR,
              cudaHostGetDevicePointer, &mem->gpu_ptr, mem->cpu_ptr, 0);

    mem->size = size;
    return UCS_OK;
}

void device_mem_destroy(device_mem_t *mem)
{
    CUDA_CALL(ucs_warn, , cudaFreeHost, mem->cpu_ptr);
}
