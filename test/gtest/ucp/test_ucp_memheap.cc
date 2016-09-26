/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"


void test_ucp_memheap::test_nonblocking_implicit_stream_xfer(nonblocking_send_func_t send,
                                                             size_t alignment,
                                                             bool malloc_allocate,
                                                             bool is_ep_flush)
{
    static const int max_iter = 300 / ucs::test_time_multiplier();
    static const size_t size = ucs_max((size_t)rand() % (12*1024), alignment);
    static const size_t memheap_size = max_iter * size + alignment;
    ucs_status_t status;

    sender().connect(&receiver());
    if (&sender() != &receiver()) {
        receiver().connect(&sender());
    }

    ucp_mem_h memh;
    void *memheap;

    if (malloc_allocate) {
        memheap = malloc(memheap_size);
    } else {
        memheap = NULL;
    }

    status = ucp_mem_map(receiver().ucph(), &memheap, memheap_size, 0, &memh);
    ASSERT_UCS_OK(status);

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

        ucs::fill_random(expected_data[i].begin(), expected_data[i].end());

        ucs_assert(size * i + alignment <= memheap_size);

        (this->*send)(&sender(), size, (void*)((uintptr_t)memheap + alignment + i * size),
                      rkey, expected_data[i]);

        ASSERT_UCS_OK(status);

    }

    if (is_ep_flush) {
        sender().flush_ep();
    } else {
        sender().flush_worker();
    }

    for (int i = 0; i < max_iter; ++i) {
        EXPECT_EQ(expected_data[i],
                  std::string((char *)((uintptr_t)memheap + alignment + i * size), expected_data[i].length()));
    }

    ucp_rkey_destroy(rkey);
    receiver().flush_worker();

    disconnect(sender());
    disconnect(receiver());

    ucp_rkey_buffer_release(rkey_buffer);
    status = ucp_mem_unmap(receiver().ucph(), memh);
    ASSERT_UCS_OK(status);

    if (malloc_allocate) {
        free(memheap);
    }
}

void test_ucp_memheap::test_blocking_xfer(blocking_send_func_t send, size_t alignment,
                                          bool malloc_allocate, bool is_ep_flush)
{
    static const size_t memheap_size = 3 * 1024;
    ucs_status_t status;
    size_t size;

    sender().connect(&receiver());
    if (&sender() != &receiver()) {
        receiver().connect(&sender());
    }

    ucp_mem_h memh;
    void *memheap;

    if (malloc_allocate) {
        memheap = malloc(memheap_size);
    } else {
        memheap = NULL;
    }

    status = ucp_mem_map(receiver().ucph(), &memheap, memheap_size, 0, &memh);
    ASSERT_UCS_OK(status);

    memset(memheap, 0, memheap_size);

    void *rkey_buffer;
    size_t rkey_buffer_size;
    status = ucp_rkey_pack(receiver().ucph(), memh, &rkey_buffer, &rkey_buffer_size);
    ASSERT_UCS_OK(status);

    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack(sender().ep(), rkey_buffer, &rkey);
    ASSERT_UCS_OK(status);

    ucp_rkey_buffer_release(rkey_buffer);

    for (int i = 0; i < 300 / ucs::test_time_multiplier(); ++i) {

        size = ucs_max(rand() % (memheap_size - alignment - 1), alignment);

        size_t offset = rand() % (memheap_size - size - alignment);
        offset = ucs_align_up(offset, alignment);

        ucs_assert(((((uintptr_t)memheap + offset)) % alignment) == 0);
        ucs_assert(size + offset <= memheap_size);

        std::string expected_data;
        expected_data.resize(size);

        ucs::fill_random(expected_data.begin(), expected_data.end());

        (this->*send)(&sender(), size, (void*)((uintptr_t)memheap + offset),
                      rkey, expected_data);

        if (is_ep_flush) {
            sender().flush_ep();
        } else {
            sender().flush_worker();
        }

        EXPECT_EQ(expected_data,
                  std::string((char*)memheap + offset, expected_data.length()));

        expected_data.clear();
    }

    ucp_rkey_destroy(rkey);
    receiver().flush_worker();

    disconnect(sender());
    disconnect(receiver());

    status = ucp_mem_unmap(receiver().ucph(), memh);
    ASSERT_UCS_OK(status);

    if (malloc_allocate) {
        free(memheap);
    }
}

