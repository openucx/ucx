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
    const entity& e() {
        return ent(0);
    }
};

UCS_TEST_P(test_pd, alloc) {
    const size_t alignment = 128;
    size_t size, orig_size;
    ucs_status_t status;
    uct_alloc_method_t alloc_method;
    void *address;
    uct_mem_h memh;

    for (unsigned i = 0; i < 300; ++i) {
        size = orig_size = rand() % 65536;
        alloc_method      = (uct_alloc_method_t)(rand() % (UCT_ALLOC_METHOD_LAST + 1));

        status = uct_pd_mem_alloc(e().pd(), alloc_method, &size, alignment,
                                  &address, &memh, "test");
        if ((status != UCS_OK) && (alloc_method != UCT_ALLOC_METHOD_DEFAULT)) {
            /* Ignore allocation failure for specific method */
            continue;
        }

        ASSERT_UCS_OK(status);
        EXPECT_GE(size, orig_size);
        EXPECT_TRUE(address != NULL);
        EXPECT_TRUE(memh != UCT_INVALID_MEM_HANDLE);

        memset(address, 0xBB, size);
        uct_pd_mem_free(e().pd(), address, memh);
    }
}

UCT_INSTANTIATE_TEST_CASE(test_pd)
