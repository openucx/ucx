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
uct_put_kernel(uct_device_ep_h ep,
               const uct_device_local_mem_list_elem_t *src_elem,
               const uct_device_mem_element_t *mem_elem, const void *va,
               uint64_t rva, size_t length, ucs_status_t *status_p)
{
    uct_device_completion_t comp;

    ucs_status_t status = uct_device_ep_put<UCS_DEVICE_LEVEL_THREAD>(
            ep, src_elem, mem_elem, va, rva, length, 0, UCT_DEVICE_FLAG_NODELAY,
            &comp);
    if (status != UCS_INPROGRESS) {
        *status_p = status;
        return;
    }

    while ((status = uct_device_ep_check_completion<UCS_DEVICE_LEVEL_THREAD>(
                    ep, &comp)) == UCS_INPROGRESS) {
        uct_device_ep_progress<UCS_DEVICE_LEVEL_THREAD>(ep);
    }
    *status_p = status;
}

/**
 * Basic single element put operation (V2 API).
 */
ucs_status_t launch_uct_put(uct_device_ep_h ep,
                            const uct_device_local_mem_list_elem_t *src_elem,
                            const uct_device_mem_element_t *mem_elem,
                            const void *va, uint64_t rva, size_t length)
{
    device_result_ptr<ucs_status_t> status = UCS_ERR_NOT_IMPLEMENTED;

    uct_put_kernel<<<1, 1>>>(ep, src_elem, mem_elem, va, rva, length,
                             status.device_ptr());
    synchronize();
    return *status;
}

static __global__ void
uct_atomic_kernel(uct_device_ep_h ep, uct_device_mem_element_t *mem_elem,
                  uint64_t rva, uint64_t add, ucs_status_t *status_p)
{
    uct_device_completion_t comp;

    ucs_status_t status = uct_device_ep_atomic_add<UCS_DEVICE_LEVEL_THREAD>(
            ep, mem_elem, add, rva, 0, UCT_DEVICE_FLAG_NODELAY, &comp);
    if (status != UCS_INPROGRESS) {
        *status_p = status;
        return;
    }

    while ((status = uct_device_ep_check_completion<UCS_DEVICE_LEVEL_THREAD>(
                    ep, &comp)) == UCS_INPROGRESS) {
        uct_device_ep_progress<UCS_DEVICE_LEVEL_THREAD>(ep);
    }
    *status_p = status;
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

} // namespace ucx_cuda
