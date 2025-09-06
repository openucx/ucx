/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef CUDA_TEST_KERNELS_H_
#define CUDA_TEST_KERNELS_H_

#include <uct/api/device/uct_device_types.h>
#include <uct/api/v2/uct_v2.h>

 namespace cuda_uct {

 int launch_memcmp(const void *s1, const void *s2, size_t size);

 ucs_status_t launch_uct_put_single(uct_device_ep_h device_ep,
                                    const uct_device_mem_element_t *mem_elem,
                                    const void *address, uint64_t remote_address,
                                    size_t length,
                                    uct_device_level_t level,
                                    unsigned num_threads,
                                    unsigned num_blocks);

 }; // namespace cuda

 #endif
