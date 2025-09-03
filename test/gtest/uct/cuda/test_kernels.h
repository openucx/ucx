/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef CUDA_UCT_TEST_KERNELS_H_
#define CUDA_UCT_TEST_KERNELS_H_

#include <uct/api/uct_def.h>
#include <uct/api/device/uct_device_types.h>

namespace uct_cuda {

ucs_status_t launch_single_kernel(uct_device_ep_h ep,
                                  uct_device_mem_element_t *mem_elem,
                                  const void *va, uint64_t rva, size_t length);

}; // namespace uct_cuda

#endif
