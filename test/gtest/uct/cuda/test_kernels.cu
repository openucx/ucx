/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_kernels.h"

#include <uct/api/device/uct_device_impl.h>
#include <common/cuda.h>


namespace ucx_cuda {

static __global__ void
uct_put_single_kernel(uct_device_ep_h ep, uct_device_mem_element_t *mem_elem,
                      const void *va, uint64_t rva, size_t length,
                      ucs_status_t *status_p)
{
    uct_device_completion_t comp;

    comp.count          = 1;
    comp.status         = UCS_OK;
    ucs_status_t status = uct_device_ep_put_single<UCS_DEVICE_LEVEL_THREAD>(
            ep, mem_elem, va, rva, length, UCT_DEVICE_FLAG_NODELAY, &comp);
    if (status != UCS_OK) {
        *status_p = status;
        return;
    }

    while (comp.count != 0) {
        uct_device_ep_progress<UCS_DEVICE_LEVEL_THREAD>(ep);
    }
    *status_p = UCS_OK;
}

/**
 * Basic single element put operation.
 */
ucs_status_t launch_uct_put_single(uct_device_ep_h ep,
                                   uct_device_mem_element_t *mem_elem,
                                   const void *va, uint64_t rva, size_t length)
{
    device_result_ptr<ucs_status_t> status = UCS_ERR_NOT_IMPLEMENTED;

    uct_put_single_kernel<<<1, 1>>>(ep, mem_elem, va, rva, length,
                                    status.device_ptr());
    synchronize();
    return *status;
}

static __global__ void
uct_atomic_kernel(uct_device_ep_h ep, uct_device_mem_element_t *mem_elem,
                  uint64_t rva, uint64_t add, ucs_status_t *status_p)
{
    uct_device_completion_t comp;

    comp.count          = 1;
    comp.status         = UCS_OK;
    ucs_status_t status = uct_device_ep_atomic_add<UCS_DEVICE_LEVEL_THREAD>(
            ep, mem_elem, add, rva, UCT_DEVICE_FLAG_NODELAY, &comp);
    if (status != UCS_OK) {
        *status_p = status;
        return;
    }

    while (comp.count != 0) {
        uct_device_ep_progress<UCS_DEVICE_LEVEL_THREAD>(ep);
    }
    *status_p = UCS_OK;
}

/**
 * Atomic operation.
 */
ucs_status_t launch_uct_atomic(uct_device_ep_h ep,
                               uct_device_mem_element_t *mem_elem, uint64_t rva,
                               uint64_t add)
{
    device_result_ptr<ucs_status_t> status = UCS_ERR_NOT_IMPLEMENTED;

    uct_atomic_kernel<<<1, 1>>>(ep, mem_elem, rva, add, status.device_ptr());
    synchronize();
    return *status;
}

template<size_t iovcnt>
static __global__ void
uct_put_multi_kernel(uct_device_ep_h ep, uct_device_mem_element_t *mem_list,
                     const void *va, uint64_t rva, uint64_t atomic_rva,
                     size_t length, ucs_status_t *status_p)
{
    __shared__ uct_device_completion_t comp;
    __shared__ size_t sizes[iovcnt];
    __shared__ void *src[iovcnt];
    __shared__ uint64_t dst[iovcnt];
    int lane_id = threadIdx.x;
    ucs_status_t status;

    if (lane_id < iovcnt) {
        /* TODO: what if iovcnt > num_lanes? */
        sizes[lane_id] = length / iovcnt;
        src[lane_id]   = (void*)((uintptr_t)va + length / iovcnt * lane_id);
        dst[lane_id]   = rva + length / iovcnt * lane_id;
    }

    __syncwarp();
    comp.count  = 1;
    comp.status = UCS_OK;
    status      = uct_device_ep_put_multi<UCS_DEVICE_LEVEL_WARP>(
            ep, mem_list, iovcnt + 1, src, dst, sizes, 4, atomic_rva,
            UCT_DEVICE_FLAG_NODELAY, &comp);
    if (status != UCS_OK) {
        *status_p = status;
        return;
    }

    while (comp.count != 0) {
        uct_device_ep_progress<UCS_DEVICE_LEVEL_THREAD>(ep);
    }
    *status_p = UCS_OK;
}

/**
 * Multi element put and atomic operation.
 */
template<size_t iovcnt>
ucs_status_t
launch_uct_put_multi(uct_device_ep_h ep, uct_device_mem_element_t *mem_list,
                     const void *va, uint64_t rva, uint64_t atomic_rva,
                     size_t length)
{
    device_result_ptr<ucs_status_t> status = UCS_ERR_NOT_IMPLEMENTED;

    uct_put_multi_kernel<iovcnt>
            <<<1, UCS_DEVICE_NUM_THREADS_IN_WARP>>>(ep, mem_list, va, rva,
                                                    atomic_rva, length,
                                                    status.device_ptr());
    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(st));
    }
    synchronize();
    return *status;
}

template ucs_status_t launch_uct_put_multi<UCS_DEVICE_NUM_THREADS_IN_WARP - 1>(
        uct_device_ep_h ep, uct_device_mem_element_t *mem_list, const void *va,
        uint64_t rva, uint64_t atomic_rva, size_t length);


template<size_t iovcnt>
static __global__ void
uct_put_partial_kernel(uct_device_ep_h ep, uct_device_mem_element_t *mem_list,
                       const void *va, uint64_t rva, uint64_t atomic_rva,
                       size_t length, ucs_status_t *status_p)
{
    __shared__ uct_device_completion_t comp;
    __shared__ unsigned indices[iovcnt];
    __shared__ size_t sizes[iovcnt];
    __shared__ void *src[iovcnt];
    __shared__ uint64_t dst[iovcnt];
    int lane_id = threadIdx.x;
    ucs_status_t status;

    if (lane_id < iovcnt) {
        indices[lane_id] = lane_id;
        sizes[lane_id]   = length / iovcnt;
        src[lane_id]     = (void*)((uintptr_t)va + length / iovcnt * lane_id);
        dst[lane_id]     = rva + length / iovcnt * lane_id;
    }

    __syncwarp();
    comp.count  = 1;
    comp.status = UCS_OK;
    status      = uct_device_ep_put_multi_partial<UCS_DEVICE_LEVEL_WARP>(
            ep, mem_list, indices, iovcnt, src, dst, sizes, iovcnt, 4,
            atomic_rva, UCT_DEVICE_FLAG_NODELAY, &comp);
    if (status != UCS_OK) {
        *status_p = status;
        return;
    }

    while (comp.count != 0) {
        uct_device_ep_progress<UCS_DEVICE_LEVEL_THREAD>(ep);
    }
    *status_p = UCS_OK;
}

/**
 * Multi element partial put and atomic operation.
 */
template<size_t iovcnt>
ucs_status_t
launch_uct_put_partial(uct_device_ep_h ep, uct_device_mem_element_t *mem_list,
                       const void *va, uint64_t rva, uint64_t atomic_rva,
                       size_t length)
{
    device_result_ptr<ucs_status_t> status = UCS_ERR_NOT_IMPLEMENTED;

    uct_put_partial_kernel<iovcnt>
            <<<1, UCS_DEVICE_NUM_THREADS_IN_WARP>>>(ep, mem_list, va, rva,
                                                    atomic_rva, length,
                                                    status.device_ptr());
    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(st));
    }
    synchronize();
    return *status;
}

template ucs_status_t
launch_uct_put_partial<UCS_DEVICE_NUM_THREADS_IN_WARP - 1>(
        uct_device_ep_h ep, uct_device_mem_element_t *mem_list, const void *va,
        uint64_t rva, uint64_t atomic_rva, size_t length);

} // namespace ucx_cuda
