/**
 * Copyright (c) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_kernels_uct.h"

#include <uct/api/device/uct_device_impl.h>
#include <uct/api/device/uct_device_types.h>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <memory>

#define UCS_DEVICE_LEVEL_EXEC_ID 1

#define UCS_DEVICE_LEVEL_EXEC_SELECT(scope_ok, count, id) \
    ((scope_ok) ? (((count) > UCS_DEVICE_LEVEL_EXEC_ID) ? \
                    ((id) == UCS_DEVICE_LEVEL_EXEC_ID) : true) : false)

namespace rocm_uct {

/**
 * Wrapper class for a host memory result variable, that can be mapped to device
 * memory and passed to a HIP kernel.
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
         void *device_ptr_void;
         if (hipHostGetDevicePointer(&device_ptr_void, m_ptr.get(), 0) !=
             hipSuccess) {
             throw std::runtime_error("hipHostGetDevicePointer() failure");
         }
         return static_cast<T*>(device_ptr_void);
     }

 private:
     static T *allocate()
     {
         T *ptr = nullptr;
         if (hipHostMalloc(&ptr, sizeof(T), hipHostMallocMapped) !=
             hipSuccess) {
             throw std::bad_alloc();
         }
         return ptr;
     }

     static void release(T *ptr)
     {
         hipHostFree(ptr);
     }

     std::unique_ptr<T, decltype(&release)> m_ptr;
 };

 static void synchronize()
 {
     if (hipDeviceSynchronize() != hipSuccess) {
         throw std::runtime_error("hipDeviceSynchronize() failure");
     }
 }

 static __device__ bool is_op_enabled(ucs_device_level_t level)
 {
    unsigned int thread_id   = threadIdx.x;
    unsigned int num_threads = blockDim.x;
    unsigned int warp_id     = thread_id / 64;  // ROCm wavefront size
    unsigned int num_warps   = num_threads / 64;
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
__global__ void
test_put_kernel(uct_device_ep_h ep,
                const uct_device_local_mem_list_elem_t *src_elem,
                const uct_device_mem_element_t *mem_elem, const void *va,
                uint64_t rva, size_t length, ucs_status_t *status_p)
{
    uct_device_completion_t comp;

    if (is_op_enabled(level)) {
        ucs_status_t status = uct_device_ep_put<level>(ep, src_elem, mem_elem,
                                                       va, rva, length, 0,
                                                       UCT_DEVICE_FLAG_NODELAY,
                                                       &comp);
        while (status == UCS_INPROGRESS) {
            uct_device_ep_progress<level>(ep);
            status = uct_device_ep_check_completion<level>(ep, &comp);
        }
        *status_p = status;
    }
}

ucs_status_t launch_uct_put(uct_device_ep_h device_ep,
                            const uct_device_local_mem_list_elem_t *src_elem,
                            const uct_device_mem_element_t *mem_elem,
                            const void *va, uint64_t rva, size_t length,
                            ucs_device_level_t level, unsigned num_threads,
                            unsigned num_blocks)
{
    device_result_ptr<ucs_status_t> status = UCS_ERR_NOT_IMPLEMENTED;
    hipError_t st;

    switch (level) {
    case UCS_DEVICE_LEVEL_THREAD:
        hipLaunchKernelGGL(test_put_kernel<UCS_DEVICE_LEVEL_THREAD>,
                          dim3(num_blocks), dim3(num_threads), 0, 0,
                          device_ep, src_elem, mem_elem, va, rva, length,
                          status.device_ptr());
        break;
    case UCS_DEVICE_LEVEL_WARP:
        hipLaunchKernelGGL(test_put_kernel<UCS_DEVICE_LEVEL_WARP>,
                          dim3(num_blocks), dim3(num_threads), 0, 0,
                          device_ep, src_elem, mem_elem, va, rva, length,
                          status.device_ptr());
        break;
    case UCS_DEVICE_LEVEL_BLOCK:
        hipLaunchKernelGGL(test_put_kernel<UCS_DEVICE_LEVEL_BLOCK>,
                          dim3(num_blocks), dim3(num_threads), 0, 0,
                          device_ep, src_elem, mem_elem, va, rva, length,
                          status.device_ptr());
        break;
    case UCS_DEVICE_LEVEL_GRID:
        hipLaunchKernelGGL(test_put_kernel<UCS_DEVICE_LEVEL_GRID>,
                          dim3(num_blocks), dim3(num_threads), 0, 0,
                          device_ep, src_elem, mem_elem, va, rva, length,
                          status.device_ptr());
        break;
    default:
        throw std::runtime_error("Unsupported level");
    }

    st = hipGetLastError();
    if (st != hipSuccess) {
        throw std::runtime_error(hipGetErrorString(st));
    }

    synchronize();
    return *status;
}

// Explicit template instantiations for all device levels
template __global__ void test_put_kernel<UCS_DEVICE_LEVEL_THREAD>(
    uct_device_ep_h, const uct_device_local_mem_list_elem_t*,
    const uct_device_mem_element_t*, const void*, uint64_t, size_t, ucs_status_t*);
template __global__ void test_put_kernel<UCS_DEVICE_LEVEL_WARP>(
    uct_device_ep_h, const uct_device_local_mem_list_elem_t*,
    const uct_device_mem_element_t*, const void*, uint64_t, size_t, ucs_status_t*);
template __global__ void test_put_kernel<UCS_DEVICE_LEVEL_BLOCK>(
    uct_device_ep_h, const uct_device_local_mem_list_elem_t*,
    const uct_device_mem_element_t*, const void*, uint64_t, size_t, ucs_status_t*);
template __global__ void test_put_kernel<UCS_DEVICE_LEVEL_GRID>(
    uct_device_ep_h, const uct_device_local_mem_list_elem_t*,
    const uct_device_mem_element_t*, const void*, uint64_t, size_t, ucs_status_t*);

} // namespace rocm_uct
