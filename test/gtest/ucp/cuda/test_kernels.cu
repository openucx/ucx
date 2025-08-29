/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_kernels.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

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

/**
 * @brief Compares two blocks of device memory.
 *
 * Compares @a size bytes of the memory areas pointed to by @a s1 and @a s2,
 * which must both point to device memory.
 *
 * @param s1   Pointer to the first block of device memory.
 * @param s2   Pointer to the second block of device memory.
 * @param size Number of bytes to compare.
 *
 * @return int Returns 0 only if the memory blocks are equal.
 */
int memcmp(const void *s1, const void *s2, size_t size)
{
    int *h_result, *d_result;
    int result;

    if (cudaHostAlloc(&h_result, sizeof(*h_result), cudaHostAllocMapped)
        != cudaSuccess) {
        throw std::bad_alloc();
    }

    if (cudaHostGetDevicePointer(&d_result, h_result, 0) != cudaSuccess) {
        cudaFreeHost(h_result);
        throw std::runtime_error("cudaHostGetDevicePointer() failure");
    }

    *h_result = 0;
    memcmp_kernel<<<16, 64>>>(s1, s2, d_result, size);

    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFreeHost(h_result);
        throw std::runtime_error("cudaDeviceSynchronize() failure");
    }

    result = *h_result;
    cudaFreeHost(h_result);
    return result;
}
}
