/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucp/cuda/test_kernels.h>

#include <cstdint>

static __global__ void cuda_memcmp_kernel(const void* a, const void* b,
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

void test_cuda_memcmp(const void* a, const void* b,
                      int* result, size_t size) {
    cuda_memcmp_kernel<<<16, 64>>>(a, b, result, size);
}
