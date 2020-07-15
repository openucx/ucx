/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_test.h"

#include <ucs/sys/sys.h>


class test_mem : public testing::TestWithParam<uct_alloc_method_t>,
                 public uct_test_base {
public:
    UCS_TEST_BASE_IMPL;

    virtual void init() {
        ucs::skip_on_address_sanitizer();
        uct_test_base::init();
    }

protected:

    void check_mem(const uct_allocated_memory &mem, size_t min_length) {
        EXPECT_TRUE(mem.address != 0);
        EXPECT_GE(mem.length, min_length);
        if (mem.method == UCT_ALLOC_METHOD_MD) {
            EXPECT_TRUE(mem.md != NULL);
            EXPECT_TRUE(mem.memh != UCT_MEM_HANDLE_NULL);
        } else {
            EXPECT_TRUE((mem.method == GetParam()) ||
                        (mem.method == UCT_ALLOC_METHOD_HEAP));
        }
    }

    static const size_t min_length = 1234557;
};


UCS_TEST_P(test_mem, nomd_alloc) {
    void *addr = NULL;
    size_t length;
    uct_alloc_method_t methods[2];
    uct_allocated_memory mem;
    uct_mem_alloc_params_t param;
    ucs_status_t status;

    methods[0]            = GetParam();
    methods[1]            = UCT_ALLOC_METHOD_HEAP;
    length                = min_length;

    param.field_mask  = UCT_MEM_ALLOC_PARAM_FIELD_FLAGS       |
                        UCT_MEM_ALLOC_PARAM_FIELD_ADDR_PTR    |
                        UCT_MEM_ALLOC_PARAM_FIELD_LENGTH_PTR  |
                        UCT_MEM_ALLOC_PARAM_FIELD_METHODS     |
                        UCT_MEM_ALLOC_PARAM_FIELD_MDS         |
                        UCT_MEM_ALLOC_PARAM_FIELD_NAME;

    param.flags           = UCT_MD_MEM_ACCESS_ALL;
    param.address_p       = &addr;
    param.length_p        = &length;
    param.methods.methods = methods;
    param.methods.count   = 2;
    param.mds.mds         = NULL;
    param.mds.count       = 0;
    param.name            = "test";

    status = uct_mem_alloc(&param, &mem);
    ASSERT_UCS_OK(status);

    check_mem(mem, min_length);

    uct_mem_free(&mem);
}

UCS_TEST_P(test_mem, md_alloc) {
    uct_alloc_method_t methods[3];
    uct_allocated_memory mem;
    std::vector<md_resource> md_resources;
    uct_md_attr_t md_attr;
    ucs_status_t status;
    uct_md_h md;
    uct_md_config_t *md_config;
    int nonblock;

    methods[0] = UCT_ALLOC_METHOD_MD;
    methods[1] = GetParam();
    methods[2] = UCT_ALLOC_METHOD_HEAP;

    md_resources = enum_md_resources();
    for (std::vector<md_resource>::iterator iter = md_resources.begin();
         iter != md_resources.end(); ++iter) {

        status = uct_md_config_read(iter->cmpt, NULL, NULL, &md_config);
        ASSERT_UCS_OK(status);

        status = uct_md_open(iter->cmpt, iter->rsc_desc.md_name, md_config, &md);
        uct_config_release(md_config);
        ASSERT_UCS_OK(status);

        status = uct_md_query(md, &md_attr);
        ASSERT_UCS_OK(status);

        for (nonblock = 0; nonblock <= 1; ++nonblock) {
            int flags  = nonblock ? UCT_MD_MEM_FLAG_NONBLOCK : 0;
            void *addr = NULL;
            size_t length;
            uct_mem_alloc_params_t param;
            flags                |= UCT_MD_MEM_ACCESS_ALL;

            length            = min_length;
            param.field_mask  = UCT_MEM_ALLOC_PARAM_FIELD_FLAGS       |
                                UCT_MEM_ALLOC_PARAM_FIELD_ADDR_PTR    |
                                UCT_MEM_ALLOC_PARAM_FIELD_LENGTH_PTR  |
                                UCT_MEM_ALLOC_PARAM_FIELD_METHODS     |
                                UCT_MEM_ALLOC_PARAM_FIELD_MDS         |
                                UCT_MEM_ALLOC_PARAM_FIELD_NAME;

            param.flags           = flags;
            param.address_p       = &addr;
            param.length_p        = &length;
            param.methods.methods = methods;
            param.methods.count   = 3;
            param.mds.mds         = &md;
            param.mds.count       = 1;
            param.name            = "test";

            status = uct_mem_alloc(&param, &mem);
            ASSERT_UCS_OK(status);

            if (md_attr.cap.flags & UCT_MD_FLAG_ALLOC) {
                EXPECT_EQ(UCT_ALLOC_METHOD_MD, mem.method);
            } else {
                EXPECT_NE(UCT_ALLOC_METHOD_MD, mem.method);
            }

            check_mem(mem, min_length);

            uct_mem_free(&mem);
        }

        uct_md_close(md);
    }
}

UCS_TEST_P(test_mem, md_fixed) {
    std::vector<md_resource> md_resources;
    uct_md_attr_t           md_attr;
    uct_md_config_t         *md_config;
    uct_md_h                md;
    unsigned                j;

    const size_t            page_size   = ucs_get_page_size();
    const size_t            n_tryes     = 101;
    uct_alloc_method_t      meth;
    void*                   p_addr      = ucs::mmap_fixed_address();
    size_t                  length      = 1;
    size_t                  n_success;

    uct_allocated_memory_t  uct_mem;
    uct_mem_alloc_params_t   param;
    ucs_status_t            status;

    md_resources = enum_md_resources();
    for (std::vector<md_resource>::iterator iter = md_resources.begin();
         iter != md_resources.end(); ++iter) {

        status = uct_md_config_read(iter->cmpt, NULL, NULL, &md_config);
        ASSERT_UCS_OK(status);

        status = uct_md_open(iter->cmpt, iter->rsc_desc.md_name, md_config, &md);
        uct_config_release(md_config);
        ASSERT_UCS_OK(status);

        status = uct_md_query(md, &md_attr);
        ASSERT_UCS_OK(status);

        if ((md_attr.cap.flags & UCT_MD_FLAG_ALLOC) &&
            (md_attr.cap.flags & UCT_MD_FLAG_FIXED)) {
            n_success = 0;

            for (j = 0; j < n_tryes; ++j) {
                meth              = UCT_ALLOC_METHOD_MD;
                param.field_mask  = UCT_MEM_ALLOC_PARAM_FIELD_FLAGS       |
                                    UCT_MEM_ALLOC_PARAM_FIELD_ADDR_PTR    |
                                    UCT_MEM_ALLOC_PARAM_FIELD_LENGTH_PTR  |
                                    UCT_MEM_ALLOC_PARAM_FIELD_METHODS     |
                                    UCT_MEM_ALLOC_PARAM_FIELD_MDS         |
                                    UCT_MEM_ALLOC_PARAM_FIELD_NAME;

                param.flags           = UCT_MD_MEM_FLAG_FIXED |
                                        UCT_MD_MEM_ACCESS_ALL;
                param.address_p       = &p_addr;
                param.length_p        = &length;
                param.methods.methods = &meth;
                param.methods.count   = 1;
                param.mds.mds         = &md;
                param.mds.count       = 1;
                param.name            = "test";

                status = uct_mem_alloc(&param, &uct_mem);
                if (status == UCS_OK) {
                    ++n_success;
                    EXPECT_EQ(meth, uct_mem.method);
                    EXPECT_EQ(p_addr, uct_mem.address);
                    EXPECT_GE(uct_mem.length, (size_t)1);
                    /* touch the page*/
                    memset(uct_mem.address, 'c', uct_mem.length);
                    EXPECT_EQ(*(char*)p_addr, 'c');
                    status = uct_mem_free(&uct_mem);
                } else {
                    EXPECT_EQ(status, UCS_ERR_NO_MEMORY);
                }

                p_addr = (char*)p_addr + (2 * page_size);
            }

            EXPECT_GT(n_success, (size_t)0);
        }

        uct_md_close(md);
    }
}


UCS_TEST_P(test_mem, mmap_fixed) {
    unsigned                i;

    const size_t            page_size   = ucs_get_page_size();
    const size_t            n_tryes     = 101;
    uct_alloc_method_t      meth;
    void*                   p_addr      = ucs::mmap_fixed_address();
    size_t                  length      = 1;
    size_t                  n_success;

    uct_allocated_memory_t  uct_mem;
    uct_mem_alloc_params_t   param;
    ucs_status_t            status;

    n_success = 0;

    for (i = 0; i < n_tryes; ++i) {
        meth = (i % 2) ? UCT_ALLOC_METHOD_MMAP : UCT_ALLOC_METHOD_HUGE;

        param.field_mask  = UCT_MEM_ALLOC_PARAM_FIELD_FLAGS       |
                            UCT_MEM_ALLOC_PARAM_FIELD_ADDR_PTR    |
                            UCT_MEM_ALLOC_PARAM_FIELD_LENGTH_PTR  |
                            UCT_MEM_ALLOC_PARAM_FIELD_METHODS     |
                            UCT_MEM_ALLOC_PARAM_FIELD_MDS         |
                            UCT_MEM_ALLOC_PARAM_FIELD_NAME;

        param.flags           = UCT_MD_MEM_FLAG_FIXED| UCT_MD_MEM_ACCESS_ALL;
        param.address_p       = &p_addr;
        param.length_p        = &length;
        param.methods.methods = &meth;
        param.methods.count   = 1;
        param.mds.mds         = NULL;
        param.mds.count       = 0;
        param.name            = "test";

        status = uct_mem_alloc(&param, &uct_mem);
        if (status == UCS_OK) {
            ++n_success;
            EXPECT_EQ(meth, uct_mem.method);
            EXPECT_EQ(p_addr, uct_mem.address);
            EXPECT_GE(uct_mem.length, (size_t)1);
            /* touch the page*/
            memset(uct_mem.address, 'c', uct_mem.length);
            EXPECT_EQ(*(char*)p_addr, 'c');
            status = uct_mem_free(&uct_mem);
        } else {
            EXPECT_EQ(status, UCS_ERR_NO_MEMORY);
        }
        p_addr = (char*)p_addr + (2 * page_size);
    }
}

INSTANTIATE_TEST_CASE_P(alloc_methods, test_mem,
                        ::testing::Values(UCT_ALLOC_METHOD_THP,
                                          UCT_ALLOC_METHOD_HEAP,
                                          UCT_ALLOC_METHOD_MMAP,
                                          UCT_ALLOC_METHOD_HUGE));
