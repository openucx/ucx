/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"


void test_ucp_memheap::test_nonblocking_implicit_xfer(blocking_send_func_t send, size_t alignment)
{
    static const int max_iter = 300 / ucs::test_time_multiplier();
    static const size_t size = ucs_max((size_t)rand() % 64, alignment);
    static const size_t memheap_size = max_iter * size + alignment;
    entity *pe0 = create_entity();
    entity *pe1 = create_entity();
    ucs_status_t status;

    ucp_mem_h memh;
    void *memheap = NULL;
    status = ucp_mem_map(pe1->ucph(), &memheap, memheap_size, 0, &memh);
    ASSERT_UCS_OK(status);

    memset(memheap, 0, memheap_size);

    void *rkey_buffer;
    size_t rkey_buffer_size;
    status = ucp_rkey_pack(pe1->ucph(), memh, &rkey_buffer, &rkey_buffer_size);
    ASSERT_UCS_OK(status);

    pe0->connect(pe1);
    pe1->connect(pe0);

    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack(pe0->ep(), rkey_buffer, &rkey);
    ASSERT_UCS_OK(status);

    std::string expected_data[300];
    assert (max_iter <= 300);

    for (int i = 0; i < max_iter; ++i) {
        ucs_assert(size*i + alignment <= memheap_size);

        (this->*send)(pe0, size, (void*)((uintptr_t)memheap + alignment + i * size),
                        rkey, expected_data[i]);
        ASSERT_UCS_OK(status);

    }

    status = ucp_worker_flush(pe0->worker());

    for (int i = 0; i < max_iter; ++i) {
        EXPECT_EQ(expected_data[i],
                  std::string((char*)memheap + alignment +i * size, expected_data[i].length()));
    }

    ucp_rkey_destroy(rkey);

    status = ucp_worker_flush(pe1->worker());
    ASSERT_UCS_OK(status);

    pe0->disconnect();
    pe1->disconnect();

    ucp_rkey_buffer_release(rkey_buffer);
    status = ucp_mem_unmap(pe1->ucph(), memh);
    ASSERT_UCS_OK(status);
}

void test_ucp_memheap::test_blocking_xfer(blocking_send_func_t send, size_t alignment)
{
    static const size_t memheap_size = 512 * 1024;
    entity *pe0 = create_entity();
    entity *pe1 = create_entity();
    ucs_status_t status;
    size_t size;

    ucp_mem_h memh;
    void *memheap = NULL;
    status = ucp_mem_map(pe1->ucph(), &memheap, memheap_size, 0, &memh);
    ASSERT_UCS_OK(status);

    memset(memheap, 0, memheap_size);

    void *rkey_buffer;
    size_t rkey_buffer_size;
    status = ucp_rkey_pack(pe1->ucph(), memh, &rkey_buffer, &rkey_buffer_size);
    ASSERT_UCS_OK(status);

    pe0->connect(pe1);
    pe1->connect(pe0);

    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack(pe0->ep(), rkey_buffer, &rkey);
    ASSERT_UCS_OK(status);

    for (int i = 0; i < 300 / ucs::test_time_multiplier(); ++i) {

        size = ucs_max(rand() % (memheap_size - alignment - 1), alignment);

        size_t offset = rand() % (memheap_size - size - alignment);
        offset = ucs_align_up(offset, alignment);

        ucs_assert(((((uintptr_t)memheap + offset)) % alignment) == 0);
        ucs_assert(size + offset <= memheap_size);

        std::string expected_data;
        (this->*send)(pe0, size, (void*)((uintptr_t)memheap + offset),
                        rkey, expected_data);

        status = ucp_worker_flush(pe0->worker());
        ASSERT_UCS_OK(status);

        EXPECT_EQ(expected_data,
                  std::string((char*)memheap + offset, expected_data.length()));
    }

    ucp_rkey_destroy(rkey);

    status = ucp_worker_flush(pe1->worker());
    ASSERT_UCS_OK(status);

    pe0->disconnect();
    pe1->disconnect();

    ucp_rkey_buffer_release(rkey_buffer);
    status = ucp_mem_unmap(pe1->ucph(), memh);
    ASSERT_UCS_OK(status);
}

void test_ucp_memheap::get_params(ucp_params_t& params) const {
    ucp_test::get_params(params);
    params.features |= UCP_FEATURE_RMA;
}
