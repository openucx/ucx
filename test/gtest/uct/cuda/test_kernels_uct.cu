/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_kernels_uct.h"

#include <uct/api/device/uct_device_impl.h>
#include <uct/api/device/uct_device_types.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <memory>

namespace cuda_uct {
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

     T &operator*()
     {
         return *m_ptr.get();
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

 static void synchronize()
 {
     if (cudaDeviceSynchronize() != cudaSuccess) {
         throw std::runtime_error("cudaDeviceSynchronize() failure");
     }
 }

 template<uct_device_level_t level>
 static __global__ void
 uct_put_single_kernel(uct_device_ep_h device_ep,
                       const uct_device_mem_element_t *mem_elem,
                       const void *address, uint64_t remote_address,
                       size_t length, ucs_status_t *status)
 {
    uct_device_completion_t comp;
    uct_device_completion_init(&comp);

    *status = uct_device_ep_put_single<level>(device_ep, mem_elem,
                                        address, remote_address,
                                        length, 0, &comp);
 }

 /**
  * Basic single element put operation.
  */
 ucs_status_t launch_uct_put_single(uct_device_ep_h device_ep,
                                    const uct_device_mem_element_t *mem_elem,
                                    const void *address, uint64_t remote_address,
                                    size_t length,
                                    uct_device_level_t level,
                                    unsigned num_threads,
                                    unsigned num_blocks)
 {
     device_result_ptr<ucs_status_t> status = UCS_ERR_NOT_IMPLEMENTED;

    switch (level) {
        case UCT_DEVICE_LEVEL_THREAD:
            uct_put_single_kernel<UCT_DEVICE_LEVEL_THREAD><<<num_blocks, num_threads>>>(
                device_ep, mem_elem, address, remote_address, length,
                status.device_ptr());
            break;
        case UCT_DEVICE_LEVEL_WARP:
            uct_put_single_kernel<UCT_DEVICE_LEVEL_WARP><<<num_blocks, num_threads>>>(
                device_ep, mem_elem, address, remote_address, length,
                status.device_ptr());
            break;
        case UCT_DEVICE_LEVEL_BLOCK:
            uct_put_single_kernel<UCT_DEVICE_LEVEL_BLOCK><<<num_blocks, num_threads>>>(
                device_ep, mem_elem, address, remote_address, length,
                status.device_ptr());
            break;
        case UCT_DEVICE_LEVEL_GRID:
            uct_put_single_kernel<UCT_DEVICE_LEVEL_GRID><<<num_blocks, num_threads>>>(
                device_ep, mem_elem, address, remote_address, length,
                status.device_ptr());
            break;
        default:
            throw std::runtime_error("Unsupported level");
    }
     synchronize();

     return *status;
 }

 } // namespace cuda_uct
