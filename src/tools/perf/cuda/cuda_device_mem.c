/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_device_mem.h"

#include <cuda_runtime.h>

ucs_status_t cuda_device_mem_create(cuda_device_mem_t *mem, size_t size)
{
    CUDA_CALL(UCS_ERR_NO_MEMORY, cudaHostAlloc, &mem->cpu_ptr, size,
              cudaHostAllocMapped);

#define ERR_HANDLER(fmt, ...) \
    ucs_error(fmt, __VA_ARGS__); \
    cudaFreeHost(mem->cpu_ptr); \
    mem->cpu_ptr = NULL;

    CUDA_CALL_HANDLER(ERR_HANDLER, UCS_ERR_IO_ERROR, cudaHostGetDevicePointer,
                      &mem->gpu_ptr, mem->cpu_ptr, 0);

    return UCS_OK;
}

void cuda_device_mem_destroy(cuda_device_mem_t *mem)
{
    CUDA_CALL_HANDLER(ucs_warn, , cudaFreeHost, mem->cpu_ptr);
}
