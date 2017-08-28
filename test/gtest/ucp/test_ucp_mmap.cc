/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"
#include "ucp/core/ucp_mm.h"


class test_ucp_mmap : public test_ucp_memheap {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_RMA;
        return params;
    }

    static int rand_flags() {
        if ((ucs::rand() % 2) == 0) {
            return 0;
        } else {
            return UCP_MEM_MAP_NONBLOCK;
        }
    }

protected:
    void test_rkey_management(entity *e, ucp_mem_h memh, bool is_dummy);
};


void test_ucp_mmap::test_rkey_management(entity *e, ucp_mem_h memh, bool is_dummy)
{
    size_t rkey_size;
    void *rkey_buffer;
    ucs_status_t status;
    void *ptr;

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

    status = ucp_rkey_ptr(rkey, (uint64_t)memh->address, &ptr);
    if (status == UCS_OK) {
        EXPECT_EQ(0, memcmp(memh->address, ptr, memh->length));
    } else {
        EXPECT_EQ(UCS_ERR_UNREACHABLE, status);
    }

    ucp_rkey_destroy(rkey);
    ucp_rkey_buffer_release(rkey_buffer);
}


UCS_TEST_P(test_ucp_mmap, alloc) {
    ucs_status_t status;
    bool is_dummy;

    sender().connect(&sender(), get_ep_params());

    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        size_t size = ucs::rand() % (1024 * 1024);

        ucp_mem_h memh;
        ucp_mem_map_params_t params;

        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        params.address    = NULL;
        params.length     = size;
        params.flags      = rand_flags() | UCP_MEM_MAP_ALLOCATE;

        status = ucp_mem_map(sender().ucph(), &params, &memh);
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

    sender().connect(&sender(), get_ep_params());

    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        size_t size = ucs::rand() % (1024 * 1024);

        void *ptr = malloc(size);

        ucp_mem_h memh;
        ucp_mem_map_params_t params;

        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        params.address    = ptr;
        params.length     = size;
        params.flags      = rand_flags();

        status = ucp_mem_map(sender().ucph(), &params, &memh);
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
    ucp_mem_map_params_t params;
    int i;

    sender().connect(&sender(), get_ep_params());

    /* Check that ucp_mem_map accepts any value for buffer if size is 0 and
     * UCP_MEM_FLAG_ZERO_REG flag is passed to it. */

    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.address    = NULL;
    params.length     = 0;
    params.flags      = rand_flags() | UCP_MEM_MAP_ALLOCATE;

    status = ucp_mem_map(sender().ucph(), &params, &memh[0]);
    ASSERT_UCS_OK(status);

    params.address = dummy;
    status = ucp_mem_map(sender().ucph(), &params, &memh[1]);
    ASSERT_UCS_OK(status);

    for (i = 0; i < buf_num; i++) {
        test_rkey_management(&sender(), memh[i], true);
        status = ucp_mem_unmap(sender().ucph(), memh[i]);
        ASSERT_UCS_OK(status);
    }
}

UCS_TEST_P(test_ucp_mmap, alloc_advise) {
    ucs_status_t status;
    bool is_dummy;

    sender().connect(&sender(), get_ep_params());

    size_t size = 128 * (1024 * 1024);

    ucp_mem_h memh;
    ucp_mem_map_params_t params;
    ucp_mem_attr_t attr;
    ucp_mem_advise_params_t advise_params;

    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.address    = NULL;
    params.length     = size;
    params.flags      = UCP_MEM_MAP_NONBLOCK | UCP_MEM_MAP_ALLOCATE;

    status = ucp_mem_map(sender().ucph(), &params, &memh);
    ASSERT_UCS_OK(status);

    attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH;
    status = ucp_mem_query(memh, &attr);
    ASSERT_UCS_OK(status);
    EXPECT_GE(attr.length, size);

    advise_params.field_mask = UCP_MEM_ADVISE_PARAM_FIELD_ADDRESS |
                               UCP_MEM_ADVISE_PARAM_FIELD_LENGTH |
                               UCP_MEM_ADVISE_PARAM_FIELD_ADVICE;
    advise_params.address    = attr.address;
    advise_params.length     = size;
    advise_params.advice     = UCP_MADV_WILLNEED;
    status = ucp_mem_advise(sender().ucph(), memh, &advise_params);
    ASSERT_UCS_OK(status);

    is_dummy = (size == 0);
    test_rkey_management(&sender(), memh, is_dummy);

    status = ucp_mem_unmap(sender().ucph(), memh);
    ASSERT_UCS_OK(status);
}

UCS_TEST_P(test_ucp_mmap, reg_advise) {

    ucs_status_t status;
    bool is_dummy;

    sender().connect(&sender(), get_ep_params());

    size_t size = 128 * 1024 * 1024;

    void *ptr = malloc(size);

    ucp_mem_h               memh;
    ucp_mem_map_params_t    params;
    ucp_mem_attr_t          mem_attr;
    ucp_mem_advise_params_t advise_params;

    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.address    = ptr;
    params.length     = size;
    params.flags      = UCP_MEM_MAP_NONBLOCK;

    status = ucp_mem_map(sender().ucph(), &params, &memh);
    ASSERT_UCS_OK(status);

    mem_attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS;
    status = ucp_mem_query(memh, &mem_attr);
    ASSERT_UCS_OK(status);

    advise_params.field_mask = UCP_MEM_ADVISE_PARAM_FIELD_ADDRESS |
                               UCP_MEM_ADVISE_PARAM_FIELD_LENGTH |
                               UCP_MEM_ADVISE_PARAM_FIELD_ADVICE;
    advise_params.address    = mem_attr.address;
    advise_params.length     = size;
    advise_params.advice     = UCP_MADV_WILLNEED;
    status = ucp_mem_advise(sender().ucph(), memh, &advise_params); 
    ASSERT_UCS_OK(status);
    is_dummy = (size == 0);
    test_rkey_management(&sender(), memh, is_dummy);

    status = ucp_mem_unmap(sender().ucph(), memh);
    ASSERT_UCS_OK(status);

    free(ptr);
}

UCS_TEST_P(test_ucp_mmap, fixed) {
    ucs_status_t status;
    bool         is_dummy;

    sender().connect(&sender(), get_ep_params());

    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        size_t size = (i + 1) * ((i % 2) ? 1000 : 1);
        uintptr_t uptr = 0xFF000000;
        void *ptr = (void *)uptr;

        ucp_mem_h memh;
        ucp_mem_map_params_t params;

        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        params.address    = ptr;
        params.length     = size;
        params.flags      = UCP_MEM_MAP_FIXED | UCP_MEM_MAP_ALLOCATE;

        status = ucp_mem_map(sender().ucph(), &params, &memh);
        ASSERT_UCS_OK(status);
        EXPECT_EQ(memh->address, ptr);
        EXPECT_GE(memh->length, size);

        is_dummy = (size == 0);
        test_rkey_management(&sender(), memh, is_dummy);

        status = ucp_mem_unmap(sender().ucph(), memh);
        ASSERT_UCS_OK(status);
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_mmap)
