/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef CUDA_TEST_KERNELS_H_
#define CUDA_TEST_KERNELS_H_

#include <uct/api/device/uct_device_types.h>
#include <uct/api/uct.h>
#include <ucs/sys/device_code.h>

namespace cuda_uct {

int launch_memcmp(const void *s1, const void *s2, size_t size);

ucs_status_t launch_uct_put_single(uct_device_ep_h device_ep,
                                   const uct_device_mem_element_t *mem_elem,
                                   const void *address, uint64_t remote_address,
                                   size_t length, ucs_device_level_t level,
                                   unsigned num_threads, unsigned num_blocks);

ucs_status_t launch_uct_atomic(uct_device_ep_h device_ep,
                               const uct_device_mem_element_t *mem_elem,
                               uint64_t rva,
                               uint64_t add,
                               ucs_device_level_t level,
                               unsigned num_threads,
                               unsigned num_blocks);

ucs_status_t launch_uct_put_multi(uct_device_ep_h device_ep,
                                  const uct_device_mem_element_t *mem_list,
                                  size_t mem_list_count, void *const *addresses,
                                  const uint64_t *remote_addresses, const size_t *lengths,
                                  uint64_t counter_inc_value, uint64_t counter_remote_address,
                                  ucs_device_level_t level,
                                  unsigned num_threads, unsigned num_blocks);

ucs_status_t launch_uct_put_multi_partial(uct_device_ep_h device_ep,
                                          const uct_device_mem_element_t *mem_list,
                                          const unsigned *mem_list_indices, unsigned mem_list_count,
                                          void *const *addresses, const uint64_t *remote_addresses,
                                          const size_t *lengths, unsigned counter_index,
                                          uint64_t counter_inc_value, uint64_t counter_remote_address,
                                          ucs_device_level_t level,
                                          unsigned num_threads, unsigned num_blocks);
}; // namespace cuda_uct

#endif
