/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "uct_test.h"

class test_pd : public uct_test {
public:
    virtual void init() {
        uct_test::init();
        m_entities.push_back(uct_test::create_entity(0));
    }

protected:
    const uct_pd_h pd() {
        return ent(0).pd();
    }
};

UCS_TEST_P(test_pd, alloc) {
    size_t size, orig_size;
    ucs_status_t status;
    uct_pd_attr_t pd_attr;
    void *address;
    uct_mem_h memh;

    status = uct_pd_query(pd(), &pd_attr);
    if (!(pd_attr.cap.flags & UCT_PD_FLAG_ALLOC)) {
        UCS_TEST_SKIP_R("allocation is not supported");
    }

    for (unsigned i = 0; i < 300; ++i) {
        size = orig_size = rand() % 65536;

        status = uct_pd_mem_alloc(pd(), &size, &address, "test", &memh);
        if (size == 0) {
            EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
            continue;
        }

        ASSERT_UCS_OK(status);
        EXPECT_GE(size, orig_size);
        EXPECT_TRUE(address != NULL);
        EXPECT_TRUE(memh != UCT_INVALID_MEM_HANDLE);

        memset(address, 0xBB, size);
        uct_pd_mem_free(pd(), memh);
    }
}

UCS_TEST_P(test_pd, reg) {
    size_t size;
    ucs_status_t status;
    uct_pd_attr_t pd_attr;
    void *address;
    uct_mem_h memh;

    status = uct_pd_query(pd(), &pd_attr);
    if (!(pd_attr.cap.flags & UCT_PD_FLAG_REG)) {
        UCS_TEST_SKIP_R("allocation is not supported");
    }

    for (unsigned i = 0; i < 300; ++i) {
        size = rand() % 65536;

        address = malloc(size);
        ASSERT_TRUE(address != NULL);

        memset(address, 0xBB, size);

        status = uct_pd_mem_reg(pd(), address, size, &memh);
        if (size == 0) {
            EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
            continue;
        }

        ASSERT_UCS_OK(status);
        ASSERT_TRUE(memh != UCT_INVALID_MEM_HANDLE);
        EXPECT_EQ('\xBB', *((char*)address + size - 1));

        status = uct_pd_mem_dereg(pd(), memh);
        ASSERT_UCS_OK(status);
        EXPECT_EQ('\xBB', *((char*)address + size - 1));

        free(address);
    }
}

UCT_INSTANTIATE_TEST_CASE(test_pd)
