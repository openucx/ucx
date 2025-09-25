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

#define UCS_DEVICE_LEVEL_EXEC_ID 1

#define UCS_DEVICE_LEVEL_EXEC_SELECT(scope_ok, count, id) \
    ((scope_ok) ? (((count) > UCS_DEVICE_LEVEL_EXEC_ID) ? \
                    ((id) == UCS_DEVICE_LEVEL_EXEC_ID) : true) : false)

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

 static __device__ bool is_op_enabled(ucs_device_level_t level)
 {
    unsigned int thread_id   = threadIdx.x;
    unsigned int num_threads = blockDim.x;
    unsigned int warp_id     = thread_id / 32;
    unsigned int num_warps   = num_threads / 32;
    unsigned int block_id    = blockIdx.x;
    unsigned int num_blocks  = gridDim.x;

    switch (level) {
    case UCS_DEVICE_LEVEL_THREAD:
        return UCS_DEVICE_LEVEL_EXEC_SELECT(block_id == 0, num_threads, thread_id);
    case UCS_DEVICE_LEVEL_WARP:
        return UCS_DEVICE_LEVEL_EXEC_SELECT(block_id == 0, num_warps, warp_id);
    case UCS_DEVICE_LEVEL_BLOCK:
        return UCS_DEVICE_LEVEL_EXEC_SELECT(true, num_blocks, block_id);
    case UCS_DEVICE_LEVEL_GRID:
        return true;
    }
    return false;
 }

 template<ucs_device_level_t level>
 static __global__ void
 uct_put_single_kernel(uct_device_ep_h device_ep,
                       const uct_device_mem_element_t *mem_elem,
                       const void *address, uint64_t remote_address,
                       size_t length, ucs_status_t *status)
 {
    uct_device_completion_t comp;

    if (is_op_enabled(level)) {
        uct_device_completion_init(&comp);
        *status = uct_device_ep_put_single<level>(device_ep, mem_elem,
                                                  address, remote_address,
                                                  length, 0, &comp);
    }
}

 /**
  * Basic single element put operation.
  */
ucs_status_t launch_uct_put_single(uct_device_ep_h device_ep,
                                   const uct_device_mem_element_t *mem_elem,
                                   const void *address, uint64_t remote_address,
                                   size_t length,
                                   ucs_device_level_t level,
                                   unsigned num_threads,
                                   unsigned num_blocks)
 {
    device_result_ptr<ucs_status_t> status = UCS_ERR_NOT_IMPLEMENTED;
    cudaError_t st;

     switch (level) {
     case UCS_DEVICE_LEVEL_THREAD:
         uct_put_single_kernel<UCS_DEVICE_LEVEL_THREAD>
                 <<<num_blocks, num_threads>>>(device_ep, mem_elem, address,
                                               remote_address, length,
                                               status.device_ptr());
         break;
     case UCS_DEVICE_LEVEL_WARP:
         uct_put_single_kernel<UCS_DEVICE_LEVEL_WARP>
                 <<<num_blocks, num_threads>>>(device_ep, mem_elem, address,
                                               remote_address, length,
                                               status.device_ptr());
         break;
     case UCS_DEVICE_LEVEL_BLOCK:
         uct_put_single_kernel<UCS_DEVICE_LEVEL_BLOCK>
                 <<<num_blocks, num_threads>>>(device_ep, mem_elem, address,
                                               remote_address, length,
                                               status.device_ptr());
         break;
     case UCS_DEVICE_LEVEL_GRID:
         uct_put_single_kernel<UCS_DEVICE_LEVEL_GRID>
                 <<<num_blocks, num_threads>>>(device_ep, mem_elem, address,
                                               remote_address, length,
                                               status.device_ptr());
         break;
     default:
         throw std::runtime_error("Unsupported level");
     }

     st = cudaGetLastError();
     if (st != cudaSuccess) {
         throw std::runtime_error(cudaGetErrorString(st));
     }

     synchronize();

    return *status;
}

template<ucs_device_level_t level>
static __global__ void
uct_atomic_kernel(uct_device_ep_h ep,
                  const uct_device_mem_element_t *mem_elem,
                  uint64_t rva, uint64_t add, ucs_status_t *status_p)
{
    uct_device_completion_t comp;

    if (is_op_enabled(level)) {
        uct_device_completion_init(&comp);
        *status_p = uct_device_ep_atomic_add<level>(ep, mem_elem, add, rva,
                                                    UCT_DEVICE_FLAG_NODELAY, &comp);
    }
}

ucs_status_t launch_uct_atomic(uct_device_ep_h device_ep,
                               const uct_device_mem_element_t *mem_elem,
                               uint64_t rva,
                               uint64_t add,
                               ucs_device_level_t level,
                               unsigned num_threads,
                               unsigned num_blocks)
{
    device_result_ptr<ucs_status_t> status = UCS_ERR_NOT_IMPLEMENTED;
    cudaError_t st;

    switch (level) {
        case UCS_DEVICE_LEVEL_THREAD:
            uct_atomic_kernel<UCS_DEVICE_LEVEL_THREAD><<<num_blocks, num_threads>>>(
                device_ep, mem_elem, rva, add, status.device_ptr());
            break;
        case UCS_DEVICE_LEVEL_WARP:
            uct_atomic_kernel<UCS_DEVICE_LEVEL_WARP><<<num_blocks, num_threads>>>(
                device_ep, mem_elem, rva, add, status.device_ptr());
            break;
        case UCS_DEVICE_LEVEL_BLOCK:
            uct_atomic_kernel<UCS_DEVICE_LEVEL_BLOCK><<<num_blocks, num_threads>>>(
                device_ep, mem_elem, rva, add, status.device_ptr());
            break;
        case UCS_DEVICE_LEVEL_GRID:
            uct_atomic_kernel<UCS_DEVICE_LEVEL_GRID><<<num_blocks, num_threads>>>(
                device_ep, mem_elem, rva, add, status.device_ptr());
            break;
        default:
            throw std::runtime_error("Unsupported level");
    }

    st = cudaGetLastError();
    if (st != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(st));
    }

    synchronize();
    return *status;
}

template<ucs_device_level_t level>
static __global__ void
uct_put_multi_kernel(uct_device_ep_h ep,
                  const uct_device_mem_element_t *mem_list,
                  size_t mem_list_count, void *const *addresses,
                  const uint64_t *remote_addresses, const size_t *lengths,
                  uint64_t counter_inc_value, uint64_t counter_remote_address,
                  ucs_status_t *status_p)
{
    uct_device_completion_t comp;

    if (is_op_enabled(level)) {
        uct_device_completion_init(&comp);
        *status_p = uct_device_ep_put_multi<level>(ep, mem_list, mem_list_count, addresses,
                                                   remote_addresses, lengths,
                                                   counter_inc_value, counter_remote_address,
                                                   UCT_DEVICE_FLAG_NODELAY, &comp);
    }
}

ucs_status_t launch_uct_put_multi(uct_device_ep_h device_ep,
                                  const uct_device_mem_element_t *mem_list,
                                  size_t mem_list_count, void *const *addresses,
                                  const uint64_t *remote_addresses, const size_t *lengths,
                                  uint64_t counter_inc_value, uint64_t counter_remote_address,
                                  ucs_device_level_t level,
                                  unsigned num_threads, unsigned num_blocks)
{
    device_result_ptr<ucs_status_t> status = UCS_ERR_NOT_IMPLEMENTED;
    cudaError_t st;

    switch (level) {
        case UCS_DEVICE_LEVEL_THREAD:
            uct_put_multi_kernel<UCS_DEVICE_LEVEL_THREAD><<<num_blocks, num_threads>>>(
                device_ep, mem_list, mem_list_count, addresses, remote_addresses, lengths,
                counter_inc_value, counter_remote_address, status.device_ptr());
            break;
        case UCS_DEVICE_LEVEL_WARP:
            uct_put_multi_kernel<UCS_DEVICE_LEVEL_WARP><<<num_blocks, num_threads>>>(
                device_ep, mem_list, mem_list_count, addresses, remote_addresses, lengths,
                counter_inc_value, counter_remote_address, status.device_ptr());
            break;
        case UCS_DEVICE_LEVEL_BLOCK:
            uct_put_multi_kernel<UCS_DEVICE_LEVEL_BLOCK><<<num_blocks, num_threads>>>(
                device_ep, mem_list, mem_list_count, addresses, remote_addresses, lengths,
                counter_inc_value, counter_remote_address, status.device_ptr());
            break;
        case UCS_DEVICE_LEVEL_GRID:
            uct_put_multi_kernel<UCS_DEVICE_LEVEL_GRID><<<num_blocks, num_threads>>>(
                device_ep, mem_list, mem_list_count, addresses, remote_addresses, lengths,
                counter_inc_value, counter_remote_address, status.device_ptr());
            break;
        default:
            throw std::runtime_error("Unsupported level");
    }

    st = cudaGetLastError();
    if (st != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(st));
    }

    synchronize();
    return *status;
}

template<ucs_device_level_t level>
static __global__ void
uct_put_multi_partial_kernel(uct_device_ep_h ep,
                             const uct_device_mem_element_t *mem_list,
                             const unsigned *mem_list_indices, unsigned mem_list_count,
                             void *const *addresses, const uint64_t *remote_addresses,
                             const size_t *lengths, unsigned counter_index,
                             uint64_t counter_inc_value, uint64_t counter_remote_address,
                             ucs_status_t *status_p)
{
    uct_device_completion_t comp;

    if (is_op_enabled(level)) {
        uct_device_completion_init(&comp);
        *status_p = uct_device_ep_put_multi_partial<level>(ep, mem_list, mem_list_indices, mem_list_count,
                                                           addresses, remote_addresses, lengths, counter_index,
                                                           counter_inc_value, counter_remote_address,
                                                           UCT_DEVICE_FLAG_NODELAY, &comp);
    }
}

ucs_status_t launch_uct_put_multi_partial(uct_device_ep_h device_ep,
                                           const uct_device_mem_element_t *mem_list,
                                           const unsigned *mem_list_indices, unsigned mem_list_count,
                                           void *const *addresses, const uint64_t *remote_addresses,
                                           const size_t *lengths, unsigned counter_index,
                                           uint64_t counter_inc_value, uint64_t counter_remote_address,
                                           ucs_device_level_t level,
                                           unsigned num_threads, unsigned num_blocks)
{
    device_result_ptr<ucs_status_t> status = UCS_ERR_NOT_IMPLEMENTED;
    cudaError_t st;

    switch (level) {
        case UCS_DEVICE_LEVEL_THREAD:
            uct_put_multi_partial_kernel<UCS_DEVICE_LEVEL_THREAD><<<num_blocks, num_threads>>>(
                device_ep, mem_list, mem_list_indices, mem_list_count, addresses,
                remote_addresses, lengths, counter_index,
                counter_inc_value, counter_remote_address, status.device_ptr());
            break;
        case UCS_DEVICE_LEVEL_WARP:
            uct_put_multi_partial_kernel<UCS_DEVICE_LEVEL_WARP><<<num_blocks, num_threads>>>(
                device_ep, mem_list, mem_list_indices, mem_list_count, addresses,
                remote_addresses, lengths, counter_index,
                counter_inc_value, counter_remote_address, status.device_ptr());
            break;
        case UCS_DEVICE_LEVEL_BLOCK:
            uct_put_multi_partial_kernel<UCS_DEVICE_LEVEL_BLOCK><<<num_blocks, num_threads>>>(
                device_ep, mem_list, mem_list_indices, mem_list_count, addresses,
                remote_addresses, lengths, counter_index,
                counter_inc_value, counter_remote_address, status.device_ptr());
            break;
        case UCS_DEVICE_LEVEL_GRID:
            uct_put_multi_partial_kernel<UCS_DEVICE_LEVEL_GRID><<<num_blocks, num_threads>>>(
                device_ep, mem_list, mem_list_indices, mem_list_count, addresses,
                remote_addresses, lengths, counter_index,
                counter_inc_value, counter_remote_address, status.device_ptr());
            break;
        default:
            throw std::runtime_error("Unsupported level");
    }

    st = cudaGetLastError();
    if (st != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(st));
    }

    synchronize();
    return *status;
}

} // namespace cuda_uct
