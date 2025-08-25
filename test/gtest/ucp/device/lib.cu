/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <cuda_runtime.h>
#include <ucp/device/lib.h>

#include <cstdint>

static __global__ void memcmp_kernel(const void* a, const void* b,
                                     int* result, size_t size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = idx; i < size; i += blockDim.x * gridDim.x) {
        if (reinterpret_cast<const uint8_t*>(a)[i]
            != reinterpret_cast<const uint8_t*>(b)[i]) {
            *result = 1;
            break;
        }
    }
}

// Compare generic CUDA buffers without copying them
int test_ucp_cuda_memcmp(const void *a, const void *b, size_t size)
{
    int *h_result, *d_result;
    int result;

    if ((cudaHostAlloc(&h_result, sizeof(h_result), cudaHostAllocMapped)
         != cudaSuccess) ||
        (cudaHostGetDevicePointer(&d_result, h_result, 0)
         != cudaSuccess)) {
        return -1;
    }

    *h_result = 0;
    memcmp_kernel<<<16, 64>>>(a, b, d_result, size);
    cudaDeviceSynchronize();
    result = *h_result;

    cudaFree(h_result);
    return result;
}
