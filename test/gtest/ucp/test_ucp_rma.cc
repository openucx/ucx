/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ucp_test.h"
extern "C" {
#include <ucp/proto/ucp_int.h>
}

class test_ucp_rma : public ucp_test {
protected:
    void test_mapped_memory(entity &e, ucp_mem_h memh,
                            void *ptr, size_t size)
    {
        EXPECT_EQ(ptr, memh->address);
        EXPECT_GE(memh->length, size);
        EXPECT_NE(0ull, memh->pd_map);

        size_t rkey_size;
        void *rkey_buffer;
        ucs_status_t status;

        status = ucp_rkey_pack(e.ucph(), memh, &rkey_buffer, &rkey_size);
        ASSERT_UCS_OK(status);

        ucp_rkey_h rkey;
        status = ucp_ep_rkey_unpack(e.ep(), rkey_buffer, &rkey);
        ASSERT_UCS_OK(status);

        ucp_rkey_buffer_release(rkey_buffer);
        ucp_rkey_destroy(rkey);
    }
};

UCS_TEST_F(test_ucp_rma, mem_alloc) {

    ucs_status_t status;
    entity e;

    e.connect(e);

    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        size_t size = rand() % (1024 * 1024);

        ucp_mem_h memh;
        void *ptr = NULL;
        status = ucp_mem_map(e.ucph(), &ptr, size, 0, &memh);
        ASSERT_UCS_OK(status);

        test_mapped_memory(e, memh, ptr, size);

        status = ucp_mem_unmap(e.ucph(), memh);
        ASSERT_UCS_OK(status);
    }
}

UCS_TEST_F(test_ucp_rma, mem_reg) {

    ucs_status_t status;
    entity e;

    e.connect(e);

    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        size_t size = rand() % (1024 * 1024);

        void *ptr = malloc(size);

        ucp_mem_h memh;
        status = ucp_mem_map(e.ucph(), &ptr, size, 0, &memh);
        ASSERT_UCS_OK(status);

        test_mapped_memory(e, memh, ptr, size);

        status = ucp_mem_unmap(e.ucph(), memh);
        ASSERT_UCS_OK(status);

        free(ptr);
    }
}

UCS_TEST_F(test_ucp_rma, put) {
    static const size_t memheap_size = 4096;
    ucs_status_t status;
    entity pes[2];

    ucp_mem_h memh;
    void *memheap = NULL;
    status = ucp_mem_map(pes[1].ucph(), &memheap, memheap_size, 0, &memh);
    ASSERT_UCS_OK(status);

    memset(memheap, 0, memheap_size);

    void *rkey_buffer;
    size_t rkey_buffer_size;
    status = ucp_rkey_pack(pes[1].ucph(), memh, &rkey_buffer, &rkey_buffer_size);
    ASSERT_UCS_OK(status);

    pes[0].connect(pes[1]);
    pes[1].connect(pes[0]);

    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack(pes[0].ep(), rkey_buffer, &rkey);
    ASSERT_UCS_OK(status);

    uint64_t send_data = 0x12345678abcdefull;
    status = ucp_rma_put(pes[0].ep(), &send_data, sizeof(send_data),
                         (uintptr_t)memheap, rkey);
    ASSERT_UCS_OK(status);

    status = ucp_rma_flush(pes[0].worker());
    ASSERT_UCS_OK(status);

    EXPECT_EQ(send_data, *(uint64_t*)memheap);

    ucp_rkey_destroy(rkey);

    pes[0].disconnect();
    pes[1].disconnect();

    ucp_rkey_buffer_release(rkey_buffer);
    status = ucp_mem_unmap(pes[1].ucph(), memh);
    ASSERT_UCS_OK(status);
}
