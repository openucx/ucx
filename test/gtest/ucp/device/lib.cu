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
            result[0] = 1;
            break;
        }
    }
}

// Compare generic CUDA buffers without copying them
int test_ucp_cuda_memcmp(const void *a, const void *b, size_t size)
{
    int h_result = 0;
    int* d_result;

    if (cudaMalloc(&d_result, sizeof(*d_result)) != cudaSuccess) {
        return -1;
    }

    if (cudaMemcpy(d_result, &h_result, sizeof(h_result),
                   cudaMemcpyHostToDevice)
        != cudaSuccess) {
        h_result = -1;
        goto out;
    }

    memcmp_kernel<<<16, 64>>>(a, b, d_result, size);
    cudaDeviceSynchronize();
    if (cudaMemcpy(&h_result, d_result, sizeof(h_result),
                   cudaMemcpyDeviceToHost)
        != cudaSuccess) {
        h_result = -1;
    }

out:
    cudaFree(d_result);
    return h_result;
}
