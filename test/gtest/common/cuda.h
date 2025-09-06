/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_TEST_CUDA_H
#define UCS_TEST_CUDA_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
#include <memory>

namespace ucx_cuda {

/**
 * Wrapper class for a host memory result variable, that can be mapped to device
 * memory and passed to a Cuda kernel.
 */
template<typename T> class device_result_ptr {
public:
    device_result_ptr() : m_ptr(allocate(), release)
    {
    }

    device_result_ptr(const T &value) : m_ptr(allocate(), release)
    {
        *m_ptr.get() = value;
    }

    T &operator*() const
    {
        return *m_ptr;
    }

    T *device_ptr()
    {
        T *device_ptr;
        if (cudaHostGetDevicePointer(&device_ptr, m_ptr.get(), 0) !=
            cudaSuccess) {
            throw std::runtime_error("cudaHostGetDevicePointer() failure");
        }
        return device_ptr;
    }

private:
    static T *allocate()
    {
        T *ptr = nullptr;
        if (cudaHostAlloc(&ptr, sizeof(T), cudaHostAllocMapped) !=
            cudaSuccess) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    static void release(T *ptr)
    {
        cudaFreeHost(ptr);
    }

    std::unique_ptr<T, decltype(&release)> m_ptr;
};

static inline void synchronize()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::stringstream ss;
        ss << "kernel launch failure: " << cudaGetErrorString(err);
        throw std::runtime_error(ss.str());
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::stringstream ss;
        ss << "cudaDeviceSynchronize(): " << cudaGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
}

} // namespace ucx_cuda

#endif
