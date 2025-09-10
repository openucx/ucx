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

ucs_status_t launch_ucp_put_multi_partial(
                            ucp_device_mem_list_handle_h handle,
                            const unsigned *mem_list_indices,
                            unsigned mem_list_count,
                            void *const *addresses,
                            const uint64_t *remote_addresses,
                            const size_t *lengths,
                            uint64_t counter_inc_value,
                            uint64_t counter_remote_address);

ucs_status_t launch_ucp_put_multi(
                            ucp_device_mem_list_handle_h handle,
                            void *const *addresses,
                            const uint64_t *remote_addresses,
                            const size_t *lengths,
                            uint64_t counter_inc_value,
                            uint64_t counter_remote_address);

ucs_status_t launch_ucp_counter_inc(ucp_device_mem_list_handle_h handle,
                                    const unsigned mem_list_index,
                                    uint64_t counter_inc_value,
                                    uint64_t counter_remote_address);
}; // namespace ucx_cuda

#endif
