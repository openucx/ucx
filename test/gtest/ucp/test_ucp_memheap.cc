/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"

#include <common/test_helpers.h>
#include <ucs/sys/sys.h>


std::vector<ucp_test_param>
test_ucp_memheap::enum_test_params(const ucp_params_t& ctx_params,
                                   const std::string& name,
                                   const std::string& test_case_name,
                                   const std::string& tls)
{
    std::vector<ucp_test_param> result;
    generate_test_params_variant(ctx_params, name,
                                 test_case_name, tls, 0, result);
    generate_test_params_variant(ctx_params, name,
                                 test_case_name + "/map_nb",
                                 tls, UCP_MEM_MAP_NONBLOCK, result);
    return result;
}

void test_ucp_memheap::test_nonblocking_implicit_stream_xfer(nonblocking_send_func_t send,
                                                             size_t size, int max_iter,
                                                             size_t alignment,
                                                             bool malloc_allocate,
                                                             bool is_ep_flush)
{
    void *memheap;
    size_t memheap_size;
    ucp_mem_map_params_t params;
    ucp_mem_attr_t mem_attr;
    ucs_status_t status;

    memheap = NULL;
    memheap_size = max_iter * size + alignment;

    if (max_iter == DEFAULT_ITERS) {
        max_iter = 300 / ucs::test_time_multiplier();
    }

    if (size == DEFAULT_SIZE) {
        size = ucs_max((size_t)ucs::rand() % (12*1024), alignment);
    }
    memheap_size = max_iter * size + alignment;

    sender().connect(&receiver(), get_ep_params());

    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.length     = memheap_size;
    params.flags      = GetParam().variant;
    if (malloc_allocate) {
        memheap = malloc(memheap_size);
        params.address = memheap;
        params.flags   = params.flags & (~(UCP_MEM_MAP_ALLOCATE|UCP_MEM_MAP_FIXED));
    } else if (params.flags & UCP_MEM_MAP_FIXED) {
        params.address = ucs::mmap_fixed_address();
    } else {
        params.address = NULL;
        params.flags  |= UCP_MEM_MAP_ALLOCATE;
    }

    ucp_mem_h memh;
    status = ucp_mem_map(receiver().ucph(), &params, &memh);
    ASSERT_UCS_OK(status);

    mem_attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS |
                          UCP_MEM_ATTR_FIELD_LENGTH;
    status = ucp_mem_query(memh, &mem_attr);
    ASSERT_UCS_OK(status);

    EXPECT_GE(mem_attr.length, memheap_size);
    if (!malloc_allocate) {
        memheap = mem_attr.address;
    }
    memset(memheap, 0, memheap_size);

    void *rkey_buffer;
    size_t rkey_buffer_size;
    status = ucp_rkey_pack(receiver().ucph(), memh, &rkey_buffer, &rkey_buffer_size);
    ASSERT_UCS_OK(status);

    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack(sender().ep(), rkey_buffer, &rkey);
    ASSERT_UCS_OK(status);

    std::string expected_data[300];
    assert (max_iter <= 300);

    for (int i = 0; i < max_iter; ++i) {
        expected_data[i].resize(size);

        ucs::fill_random(expected_data[i]);

        ucs_assert(size * i + alignment <= memheap_size);

        (this->*send)(&sender(), size,
                      (void*)((uintptr_t)memheap + alignment + i * size),
                      rkey, expected_data[i]);

        ASSERT_UCS_OK(status);

    }

    if (is_ep_flush) {
        flush_ep(sender());
    } else {
        flush_worker(sender());
    }

    for (int i = 0; i < max_iter; ++i) {
        EXPECT_EQ(expected_data[i],
                  std::string((char *)((uintptr_t)memheap + alignment + i * size),
                              expected_data[i].length()));
    }

    ucp_rkey_destroy(rkey);

    disconnect(sender());

    ucp_rkey_buffer_release(rkey_buffer);
    status = ucp_mem_unmap(receiver().ucph(), memh);
    ASSERT_UCS_OK(status);

    if (malloc_allocate) {
        free(memheap);
    }
}

/* NOTE: alignment is ignored if memheap_size is not default */
void test_ucp_memheap::test_blocking_xfer(blocking_send_func_t send,
                                          size_t memheap_size, int max_iter,
                                          size_t alignment,
                                          bool malloc_allocate, 
                                          bool is_ep_flush)
{
    ucp_mem_map_params_t params;
    ucp_mem_attr_t mem_attr;
    ucs_status_t status;
    size_t size;
    int zero_offset = 0;

    if (max_iter == DEFAULT_ITERS) {
        max_iter = 300 / ucs::test_time_multiplier();
    }

    if (memheap_size == DEFAULT_SIZE) {
        memheap_size = 3 * 1024;
        zero_offset = 1;
    }

    sender().connect(&receiver(), get_ep_params());

    ucp_mem_h memh;
    void *memheap = NULL;

    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.length     = memheap_size;
    params.flags      = GetParam().variant;
    if (malloc_allocate) {
        memheap = malloc(memheap_size);
        params.address = memheap;
        params.flags   = params.flags & (~(UCP_MEM_MAP_ALLOCATE|UCP_MEM_MAP_FIXED));
    } else if (params.flags & UCP_MEM_MAP_FIXED) {
        params.address = ucs::mmap_fixed_address();
        params.flags |= UCP_MEM_MAP_ALLOCATE;
    } else {
        params.address = NULL;
        params.flags |= UCP_MEM_MAP_ALLOCATE;
    }

    status = ucp_mem_map(receiver().ucph(), &params, &memh);
    ASSERT_UCS_OK(status);

    mem_attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS |
                          UCP_MEM_ATTR_FIELD_LENGTH;
    status = ucp_mem_query(memh, &mem_attr);
    ASSERT_UCS_OK(status);
    EXPECT_GE(mem_attr.length, memheap_size);
    if (!memheap) {
        memheap = mem_attr.address;
    }
    memset(memheap, 0, memheap_size);

    void *rkey_buffer;
    size_t rkey_buffer_size;
    status = ucp_rkey_pack(receiver().ucph(), memh, &rkey_buffer, &rkey_buffer_size);
    ASSERT_UCS_OK(status);

    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack(sender().ep(), rkey_buffer, &rkey);
    ASSERT_UCS_OK(status);

    ucp_rkey_buffer_release(rkey_buffer);

    for (int i = 0; i < max_iter; ++i) {
        size_t offset;

        if (!zero_offset) {
            size = ucs_max(ucs::rand() % (memheap_size - alignment - 1), alignment);
            offset = ucs::rand() % (memheap_size - size - alignment);
        } else {
            size = memheap_size;
            offset = 0;
        }

        offset = ucs_align_up(offset, alignment);

        ucs_assert(((((uintptr_t)memheap + offset)) % alignment) == 0);
        ucs_assert(size + offset <= memheap_size);

        std::string expected_data;
        expected_data.resize(size);

        ucs::fill_random(expected_data);

        (this->*send)(&sender(), size, (void*)((uintptr_t)memheap + offset),
                      rkey, expected_data);

        if (is_ep_flush) {
            flush_ep(sender());
        } else {
            flush_worker(sender());
        }

        EXPECT_EQ(expected_data,
                  std::string((char*)memheap + offset, expected_data.length()));

        expected_data.clear();
    }

    ucp_rkey_destroy(rkey);

    disconnect(sender());

    status = ucp_mem_unmap(receiver().ucph(), memh);
    ASSERT_UCS_OK(status);

    if (malloc_allocate) {
        free(memheap);
    }
}
