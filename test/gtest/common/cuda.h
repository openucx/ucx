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
#include <vector>


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

template<typename T> class device_vector {
public:
    device_vector(const std::vector<T> &vec)
    {
        if (cudaMalloc(&m_device_ptr, vec.size() * sizeof(T)) != cudaSuccess) {
            throw std::bad_alloc();
        }

        cudaMemcpy(m_device_ptr, vec.data(), vec.size() * sizeof(T),
                   cudaMemcpyHostToDevice);
    }

    ~device_vector()
    {
        cudaFree(m_device_ptr);
    }

    T *ptr() const
    {
        return reinterpret_cast<T*>(m_device_ptr);
    }

private:
    void *m_device_ptr;
};

template<typename T>
device_vector<T> make_device_vector(const std::vector<T> &vec)
{
    return device_vector<T>(vec);
}

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
