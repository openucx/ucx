/**
 * Copyright (c) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef ROCM_TEST_KERNELS_H_
#define ROCM_TEST_KERNELS_H_

#include <hip/hip_runtime.h>
#include <uct/api/device/uct_device_types.h>
#include <uct/api/uct.h>
#include <ucs/sys/device_code.h>
#include <stdexcept>

namespace rocm_uct {

/**
 * Kernel that calls the generic uct_device_ep_put API
 */
template<ucs_device_level_t level>
__global__ void test_put_kernel(
    uct_device_ep_h ep,
    const uct_device_local_mem_list_elem_t *src_elem,
    const uct_device_mem_element_t *mem_elem,
    const void *va,
    uint64_t rva,
    size_t length,
    ucs_status_t *status_p);

/**
 * Host function to launch the PUT kernel
 */
ucs_status_t launch_uct_put(uct_device_ep_h device_ep,
                            const uct_device_local_mem_list_elem_t *src_elem,
                            const uct_device_mem_element_t *mem_elem,
                            const void *va, uint64_t rva, size_t length,
                            ucs_device_level_t level, unsigned num_threads,
                            unsigned num_blocks);

} // namespace rocm_uct

#endif
