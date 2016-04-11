/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"


class test_ucp_mmap : public test_ucp_memheap {
public:
    using test_ucp_memheap::get_ctx_params;
protected:
    void test_rkey_management(entity *e, ucp_mem_h memh, bool is_dummy);
};


void test_ucp_mmap::test_rkey_management(entity *e, ucp_mem_h memh, bool is_dummy)
{
    size_t rkey_size;
    void *rkey_buffer;
    ucs_status_t status;

    /* Some transports don't support memory registration, so the memory
     * can be inaccessible remotely. But it should always be possible
     * to pack/unpack a key for dummy memh. */

    status = ucp_rkey_pack(e->ucph(), memh, &rkey_buffer, &rkey_size);
    if (status == UCS_ERR_UNREACHABLE && !is_dummy) {
        return;
    }
    ASSERT_UCS_OK(status);

    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack(e->ep(), rkey_buffer, &rkey);
    if (status == UCS_ERR_UNREACHABLE && !is_dummy) {
        return;
    }
    ASSERT_UCS_OK(status);

    ucp_rkey_destroy(rkey);
    ucp_rkey_buffer_release(rkey_buffer);
}


UCS_TEST_P(test_ucp_mmap, alloc) {
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

        test_rkey_management(e, memh, false);

        status = ucp_mem_unmap(e->ucph(), memh);
        ASSERT_UCS_OK(status);
    }
}

UCS_TEST_P(test_ucp_mmap, reg) {

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

        test_rkey_management(e, memh, false);

        status = ucp_mem_unmap(e->ucph(), memh);
        ASSERT_UCS_OK(status);

        free(ptr);
    }
}

UCS_TEST_P(test_ucp_mmap, dummy_mem) {

    ucs_status_t status;
    entity *e    = create_entity();
    int buf_num = 2;
    ucp_mem_h memh[buf_num];
    int dummy[1];
    void *ptr = NULL;
    int i;

    e->connect(e);

    status = ucp_mem_map(e->ucph(), &ptr, 0, 0, &memh[0]);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    /* Check that ucp_mem_map accepts any value for buffer if size is 0 and
     * UCP_MEM_FLAG_ZERO_REG flag is passed to it. */
    status = ucp_mem_map(e->ucph(), &ptr, 0, UCP_MEM_FLAG_ZERO_REG, &memh[0]);
    ASSERT_UCS_OK(status);
    status = ucp_mem_map(e->ucph(), (void**)&dummy, 0, UCP_MEM_FLAG_ZERO_REG, &memh[1]);
    ASSERT_UCS_OK(status);

    for (i = 0; i < buf_num; i++) {
        test_rkey_management(e, memh[i], true);
        status = ucp_mem_unmap(e->ucph(), memh[i]);
        ASSERT_UCS_OK(status);
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_mmap)
