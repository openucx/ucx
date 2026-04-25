/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef CUDA_UCT_TEST_KERNELS_H_
#define CUDA_UCT_TEST_KERNELS_H_

#include <uct/api/uct_def.h>
#include <uct/api/device/uct_device_types.h>
#include <ucs/sys/device_code.h>

namespace ucx_cuda {

ucs_status_t launch_uct_put(uct_device_ep_h ep,
                            const uct_device_mem_elem_t *src_elem,
                            const uct_device_mem_elem_t *mem_elem,
                            const void *va, uint64_t rva, size_t length);

ucs_status_t launch_uct_atomic(uct_device_ep_h ep,
                               uct_device_mem_elem_t *mem_elem, uint64_t rva,
                               uint64_t add);

}; // namespace ucx_cuda

#endif
