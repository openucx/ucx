/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <cstdint>
#include <cuda_runtime.h>

#include "test_kernels.h"

namespace cuda {
static __global__ void memcmp_kernel(const void* s1, const void* s2,
                                     int* result, size_t size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = idx; i < size; i += blockDim.x * gridDim.x) {
        if (reinterpret_cast<const uint8_t*>(s1)[i]
            != reinterpret_cast<const uint8_t*>(s2)[i]) {
            *result = 1;
            break;
        }
    }
}

// Compare generic CUDA buffers without copying them
int memcmp(const void *s1, const void *s2, size_t size)
{
    int *h_result, *d_result;
    int result;

    if (cudaHostAlloc(&h_result, sizeof(*h_result), cudaHostAllocMapped)
        != cudaSuccess) {
        return -1;
    }

    if (cudaHostGetDevicePointer(&d_result, h_result, 0) != cudaSuccess) {
        result = 1;
        goto out;
    }

    *h_result = 0;
    memcmp_kernel<<<16, 64>>>(s1, s2, d_result, size);
    cudaDeviceSynchronize();
    result = *h_result;

out:
    cudaFreeHost(h_result);
    return result;
}
}
