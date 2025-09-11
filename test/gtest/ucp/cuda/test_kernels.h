/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef CUDA_TEST_KERNELS_H_
#define CUDA_TEST_KERNELS_H_

#include <ucp/api/device/ucp_host.h>

namespace ucx_cuda {

int launch_memcmp(const void *s1, const void *s2, size_t size);

ucs_status_t launch_ucp_put_single(ucp_device_mem_list_handle_h mem_list,
                                   unsigned mem_list_index, const void *address,
                                   uint64_t remote_address, size_t length);

}; // namespace ucx_cuda

#endif
