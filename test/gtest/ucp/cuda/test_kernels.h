/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef CUDA_TEST_KERNELS_H_
#define CUDA_TEST_KERNELS_H_

#include <ucp/api/device/ucp_host.h>
#include <ucs/sys/device_code.h>

typedef enum {
    TEST_UCP_DEVICE_KERNEL_PUT_SINGLE,
    TEST_UCP_DEVICE_KERNEL_PUT_MULTI,
    TEST_UCP_DEVICE_KERNEL_PUT_MULTI_PARTIAL,
    TEST_UCP_DEVICE_KERNEL_COUNTER_INC,
    TEST_UCP_DEVICE_KERNEL_COUNTER_WRITE,
    TEST_UCP_DEVICE_KERNEL_COUNTER_READ
} test_ucp_device_operation_t;

typedef struct {
    unsigned                     num_threads;
    unsigned                     num_blocks;
    test_ucp_device_operation_t  operation;
    ucs_device_level_t           level;
    bool                         with_no_delay;
    bool                         with_request;
    size_t                       num_iters;
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
            uint64_t inc_value;
            uint64_t remote_address;
        } counter_inc;
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
        struct {
            void     *address;
            uint64_t value;
        } local_counter;
    };
} test_ucp_device_kernel_params_t;

ucs_status_t
launch_test_ucp_device_kernel(const test_ucp_device_kernel_params_t &params);

#endif
