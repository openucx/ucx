/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"
extern "C" {
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_mm.h>
}

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
    bool resolve_rma(entity *e, ucp_rkey_h rkey);
    bool resolve_amo(entity *e, ucp_rkey_h rkey);
    bool resolve_rma_bw(entity *e, ucp_rkey_h rkey);
    void test_rkey_management(entity *e, ucp_mem_h memh, bool is_dummy);
};

bool test_ucp_mmap::resolve_rma(entity *e, ucp_rkey_h rkey)
{
    hide_errors();
    ucs_status_t status = UCP_RKEY_RESOLVE(rkey, e->ep(), rma);
    restore_errors();
    if (status == UCS_OK) {
        EXPECT_NE(UCP_NULL_LANE, rkey->cache.rma_lane);
        return true;
    } else if (status == UCS_ERR_UNREACHABLE) {
        EXPECT_EQ(UCP_NULL_LANE, rkey->cache.rma_lane);
        return false;
    } else {
        UCS_TEST_ABORT("Invalid status from UCP_RKEY_RESOLVE");
    }
}

bool test_ucp_mmap::resolve_amo(entity *e, ucp_rkey_h rkey)
{
    hide_errors();
    ucs_status_t status = UCP_RKEY_RESOLVE(rkey, e->ep(), amo);
    restore_errors();
    if (status == UCS_OK) {
        EXPECT_NE(UCP_NULL_LANE, rkey->cache.amo_lane);
        return true;
    } else if (status == UCS_ERR_UNREACHABLE) {
        EXPECT_EQ(UCP_NULL_LANE, rkey->cache.amo_lane);
        return false;
    } else {
        UCS_TEST_ABORT("Invalid status from UCP_RKEY_RESOLVE");
    }
}

bool test_ucp_mmap::resolve_rma_bw(entity *e, ucp_rkey_h rkey)
{
    ucp_lane_index_t lane;
    uct_rkey_t uct_rkey;

    lane = ucp_rkey_get_rma_bw_lane(rkey, e->ep(), &uct_rkey, 0);
    if (lane != UCP_NULL_LANE) {
        return true;
    } else {
        return false;
    }
}

void test_ucp_mmap::test_rkey_management(entity *e, ucp_mem_h memh, bool is_dummy)
{
    size_t rkey_size;
    void *rkey_buffer;
    ucs_status_t status;

    /* Some transports don't support memory registration, so the memory
     * can be inaccessible remotely. But it should always be possible
     * to pack/unpack a key, even if empty. */
    status = ucp_rkey_pack(e->ucph(), memh, &rkey_buffer, &rkey_size);
    if (status == UCS_ERR_UNSUPPORTED && !is_dummy) {
        return;
    }
    ASSERT_UCS_OK(status);

    EXPECT_EQ(ucp_rkey_packed_size(e->ucph(), memh->md_map), rkey_size);

    /* Unpack remote key buffer */
    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack(e->ep(), rkey_buffer, &rkey);
    if (status == UCS_ERR_UNREACHABLE && !is_dummy) {
        ucp_rkey_buffer_release(rkey_buffer);
        return;
    }
    ASSERT_UCS_OK(status);

    bool have_rma    = resolve_rma(e, rkey);
    bool have_amo    = resolve_amo(e, rkey);
    bool have_rma_bw = resolve_rma_bw(e, rkey);

    /* Test that lane resolution on the remote key returns consistent results */
    for (int i = 0; i < 10; ++i) {
        switch (ucs::rand() % 3) {
        case 0:
            EXPECT_EQ(have_rma, resolve_rma(e, rkey));
            break;
        case 1:
            EXPECT_EQ(have_amo, resolve_amo(e, rkey));
            break;
        case 2:
            EXPECT_EQ(have_rma_bw, resolve_rma_bw(e, rkey));
            break;
        }
    }

    /* Test obtaining direct-access pointer */
    void *ptr;
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
        ucs::fill_random(ptr, size);

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
    ucs::fill_random(ptr, size);

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
        void *ptr = ucs::mmap_fixed_address();

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
