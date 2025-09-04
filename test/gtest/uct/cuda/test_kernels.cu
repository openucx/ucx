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

    uct_device_completion_init(&comp);
    ucs_status_t status = uct_device_ep_put_single<UCT_DEVICE_LEVEL_THREAD>(
            ep, mem_elem, va, rva, length, UCT_DEVICE_FLAG_NODELAY, &comp);
    if (status != UCS_OK) {
        *status_p = status;
        return;
    }

    while (comp.count != 0) {
        uct_device_ep_progress<UCT_DEVICE_LEVEL_THREAD>(ep);
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

} // namespace ucx_cuda
