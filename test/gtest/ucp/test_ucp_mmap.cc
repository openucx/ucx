/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"
extern "C" {
#include <ucp/proto/ucp_int.h> /* To validate the contents of a memory handle */
}


class test_ucp_mmap : public test_ucp_memheap {
protected:
    void test_mapped_memory(entity *e, ucp_mem_h memh, void *ptr, size_t size);

    virtual uint64_t features() const {
        return UCP_FEATURE_RMA;
    }
};


void test_ucp_mmap::test_mapped_memory(entity *e, ucp_mem_h memh,
                                       void *ptr, size_t size)
{
    EXPECT_EQ(ptr, memh->address);
    EXPECT_GE(memh->length, size);

    size_t rkey_size;
    void *rkey_buffer;
    ucs_status_t status;

    status = ucp_rkey_pack(e->ucph(), memh, &rkey_buffer, &rkey_size);
    ASSERT_UCS_OK(status);

    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack(e->ep(), rkey_buffer, &rkey);
    ASSERT_UCS_OK(status);

    ucp_rkey_buffer_release(rkey_buffer);
    ucp_rkey_destroy(rkey);
}


UCS_TEST_F(test_ucp_mmap, alloc) {
    ucs_status_t status;
    entity *e = create_entity();

    e->connect(e);

    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        size_t size = rand() % (1024 * 1024);

        ucp_mem_h memh;
        void *ptr = NULL;
        status = ucp_mem_map(e->ucph(), &ptr, size, 0, &memh);
        if (size == 0) {
            EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
            continue;
        }

        ASSERT_UCS_OK(status);

        test_mapped_memory(e, memh, ptr, size);

        status = ucp_mem_unmap(e->ucph(), memh);
        ASSERT_UCS_OK(status);
    }
}

UCS_TEST_F(test_ucp_mmap, reg) {

    ucs_status_t status;
    entity *e = create_entity();

    e->connect(e);

    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        size_t size = rand() % (1024 * 1024);

        void *ptr = malloc(size);

        ucp_mem_h memh;
        status = ucp_mem_map(e->ucph(), &ptr, size, 0, &memh);
        if (size == 0) {
            EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
            continue;
        }

        ASSERT_UCS_OK(status);

        test_mapped_memory(e, memh, ptr, size);

        status = ucp_mem_unmap(e->ucph(), memh);
        ASSERT_UCS_OK(status);

        free(ptr);
    }
}
