/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
}
#include <ucs/gtest/test.h>

class test_mem : public testing::TestWithParam<uct_alloc_method_t>,
public ucs::test_base {
public:
    UCS_TEST_BASE_IMPL;

protected:

    void check_mem(const uct_allocated_memory &mem, size_t min_length) {
        EXPECT_TRUE(mem.address != 0);
        EXPECT_GE(mem.length, min_length);
        if (mem.method == UCT_ALLOC_METHOD_PD) {
            EXPECT_TRUE(mem.pd != NULL);
            EXPECT_TRUE(mem.memh != UCT_INVALID_MEM_HANDLE);
        } else {
            EXPECT_TRUE((mem.method == GetParam()) ||
                        (mem.method == UCT_ALLOC_METHOD_HEAP));
        }
    }

    static const size_t min_length = 1234557;
};


UCS_TEST_P(test_mem, nopd_alloc) {
    uct_alloc_method_t methods[2];
    uct_allocated_memory mem;
    ucs_status_t status;

    methods[0] = GetParam();
    methods[1] = UCT_ALLOC_METHOD_HEAP;

    status = uct_mem_alloc(min_length, methods, 2, NULL, 0, "test", &mem);
    ASSERT_UCS_OK(status);

    check_mem(mem, min_length);

    uct_mem_free(&mem);
}

UCS_TEST_P(test_mem, pd_alloc) {
    uct_alloc_method_t methods[3];
    uct_allocated_memory mem;
    uct_pd_resource_desc_t *pd_resources;
    uct_pd_attr_t pd_attr;
    unsigned i, num_pd_resources;
    ucs_status_t status;
    uct_pd_h pd;

    status = uct_query_pd_resources(&pd_resources, &num_pd_resources);
    ASSERT_UCS_OK(status);

    methods[0] = UCT_ALLOC_METHOD_PD;
    methods[1] = GetParam();
    methods[2] = UCT_ALLOC_METHOD_HEAP;

    for (i = 0; i < num_pd_resources; ++i) {

        status = uct_pd_open(pd_resources[i].pd_name, &pd);
        ASSERT_UCS_OK(status);

        status = uct_pd_query(pd, &pd_attr);
        ASSERT_UCS_OK(status);

        status = uct_mem_alloc(min_length, methods, 3, &pd, 1, "test", &mem);
        ASSERT_UCS_OK(status);

        if (pd_attr.cap.flags & UCT_PD_FLAG_ALLOC) {
            EXPECT_EQ(UCT_ALLOC_METHOD_PD, mem.method);
        } else {
            EXPECT_NE(UCT_ALLOC_METHOD_PD, mem.method);
        }

        check_mem(mem, min_length);

        uct_mem_free(&mem);

        uct_pd_close(pd);
    }

    uct_release_pd_resource_list(pd_resources);
}

INSTANTIATE_TEST_CASE_P(alloc_methods, test_mem,
                        ::testing::Values(UCT_ALLOC_METHOD_HEAP,
                                          UCT_ALLOC_METHOD_MMAP,
                                          UCT_ALLOC_METHOD_HUGE));

