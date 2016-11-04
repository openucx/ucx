/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
}
#include <common/test.h>

class test_mem : public testing::TestWithParam<uct_alloc_method_t>,
public ucs::test_base {
public:
    UCS_TEST_BASE_IMPL;

protected:

    void check_mem(const uct_allocated_memory &mem, size_t min_length) {
        EXPECT_TRUE(mem.address != 0);
        EXPECT_GE(mem.length, min_length);
        if (mem.method == UCT_ALLOC_METHOD_MD) {
            EXPECT_TRUE(mem.md != NULL);
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

    status = uct_mem_alloc(min_length, 0, methods, 2, NULL, 0, "test", &mem);
    ASSERT_UCS_OK(status);

    check_mem(mem, min_length);

    uct_mem_free(&mem);
}

UCS_TEST_P(test_mem, pd_alloc) {
    uct_alloc_method_t methods[3];
    uct_allocated_memory mem;
    uct_md_resource_desc_t *md_resources;
    uct_md_attr_t md_attr;
    unsigned i, num_md_resources;
    ucs_status_t status;
    uct_md_h pd;
    uct_md_config_t *md_config;
    int nonblock;

    status = uct_query_md_resources(&md_resources, &num_md_resources);
    ASSERT_UCS_OK(status);

    methods[0] = UCT_ALLOC_METHOD_MD;
    methods[1] = GetParam();
    methods[2] = UCT_ALLOC_METHOD_HEAP;

    for (i = 0; i < num_md_resources; ++i) {

        status = uct_md_config_read(md_resources[i].md_name, NULL, NULL, &md_config);
        ASSERT_UCS_OK(status);

        status = uct_md_open(md_resources[i].md_name, md_config, &pd);
        uct_config_release(md_config);
        ASSERT_UCS_OK(status);

        status = uct_md_query(pd, &md_attr);
        ASSERT_UCS_OK(status);

        for (nonblock = 0; nonblock <= 1; ++nonblock) {
            int flags = nonblock ? UCT_MD_MEM_FLAG_NONBLOCK : 0;
            status = uct_mem_alloc(min_length, flags, methods, 3, &pd, 1, "test", &mem);
            ASSERT_UCS_OK(status);

            if (md_attr.cap.flags & UCT_MD_FLAG_ALLOC) {
                EXPECT_EQ(UCT_ALLOC_METHOD_MD, mem.method);
            } else {
                EXPECT_NE(UCT_ALLOC_METHOD_MD, mem.method);
            }

            check_mem(mem, min_length);

            uct_mem_free(&mem);
        }

        uct_md_close(pd);
    }

    uct_release_md_resource_list(md_resources);
}

INSTANTIATE_TEST_CASE_P(alloc_methods, test_mem,
                        ::testing::Values(UCT_ALLOC_METHOD_HEAP,
                                          UCT_ALLOC_METHOD_MMAP,
                                          UCT_ALLOC_METHOD_HUGE));

