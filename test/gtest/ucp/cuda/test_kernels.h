/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef CUDA_TEST_KERNELS_H_
#define CUDA_TEST_KERNELS_H_

#include <ucp/api/device/ucp_host.h>

namespace ucx_cuda {

struct kernel_params {
    ucp_device_mem_list_handle_h mem_list;
    union {
        struct {
            unsigned   mem_list_index;
            const void *address;
            uint64_t   remote_address;
            size_t     length;
        } single;
        struct {
            unsigned mem_list_index;
            void* address;
            uint64_t value;
            uint64_t inc_value;
            uint64_t remote_address;
        } counter;
        struct {
            void *const    *addresses;
            const uint64_t *remote_addresses;
            const size_t   *lengths;
            uint64_t       counter_inc_value;
            uint64_t       counter_remote_address;
        } multi;
        struct {
            const unsigned *mem_list_indices;
            unsigned       mem_list_count;
            void *const    *addresses;
            const uint64_t *remote_addresses;
            const size_t   *lengths;
            unsigned       counter_index;
            uint64_t       counter_inc_value;
            uint64_t       counter_remote_address;
        } partial;
    };
};


int launch_memcmp(const void *s1, const void *s2, size_t size);

ucs_status_t launch_ucp_put_single(const kernel_params &params);

ucs_status_t launch_ucp_put_multi(const kernel_params &params);

ucs_status_t launch_ucp_put_multi_partial(const kernel_params &params);

ucs_status_t launch_ucp_counter_inc(const kernel_params &params);

ucs_status_t launch_ucp_counter_write(const kernel_params &params);

ucs_status_t launch_ucp_counter_read(const kernel_params &params);

}; // namespace ucx_cuda

#endif
