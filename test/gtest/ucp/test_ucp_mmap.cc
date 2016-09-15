/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"


class test_ucp_mmap : public test_ucp_memheap {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_RMA;
        return params;
    }

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
    if (status == UCS_ERR_UNSUPPORTED && !is_dummy) {
        return;
    }
    ASSERT_UCS_OK(status);

    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack(e->ep(), rkey_buffer, &rkey);
    if (status == UCS_ERR_UNREACHABLE && !is_dummy) {
        ucp_rkey_buffer_release(rkey_buffer);
        return;
    }
    ASSERT_UCS_OK(status);

    ucp_rkey_destroy(rkey);
    ucp_rkey_buffer_release(rkey_buffer);
}


UCS_TEST_P(test_ucp_mmap, alloc) {
    ucs_status_t status;
    bool is_dummy;

    sender().connect(&sender());

    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        size_t size = rand() % (1024 * 1024);

        ucp_mem_h memh;
        void *ptr = NULL;
        status = ucp_mem_map(sender().ucph(), &ptr, size, 0, &memh);
        ASSERT_UCS_OK(status);

        is_dummy = (size == 0);
        test_rkey_management(&sender(), memh, is_dummy);

        status = ucp_mem_unmap(sender().ucph(), memh);
        ASSERT_UCS_OK(status);
    }
}

UCS_TEST_P(test_ucp_mmap, reg) {

    ucs_status_t status;
    bool is_dummy;

    sender().connect(&sender());

    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        size_t size = rand() % (1024 * 1024);

        void *ptr = malloc(size);

        ucp_mem_h memh;
        status = ucp_mem_map(sender().ucph(), &ptr, size, 0, &memh);
        ASSERT_UCS_OK(status);

        is_dummy = (size == 0);
        test_rkey_management(&sender(), memh, is_dummy);

        status = ucp_mem_unmap(sender().ucph(), memh);
        ASSERT_UCS_OK(status);

        free(ptr);
    }
}

UCS_TEST_P(test_ucp_mmap, dummy_mem) {

    ucs_status_t status;
    int buf_num = 2;
    ucp_mem_h memh[buf_num];
    int dummy[1];
    void *ptr = NULL;
    int i;

    sender().connect(&sender());

    /* Check that ucp_mem_map accepts any value for buffer if size is 0 and
     * UCP_MEM_FLAG_ZERO_REG flag is passed to it. */
    status = ucp_mem_map(sender().ucph(), &ptr, 0, 0, &memh[0]);
    ASSERT_UCS_OK(status);
    ptr = dummy;
    status = ucp_mem_map(sender().ucph(), &ptr, 0, 0, &memh[1]);
    ASSERT_UCS_OK(status);

    for (i = 0; i < buf_num; i++) {
        test_rkey_management(&sender(), memh[i], true);
        status = ucp_mem_unmap(sender().ucph(), memh[i]);
        ASSERT_UCS_OK(status);
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_mmap)
