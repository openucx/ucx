/**
* Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/datastruct/strided_alloc.h>
}

#include <limits.h>
#include <vector>
#include <queue>

class test_strided_alloc : public ucs::test {
protected:
    static const size_t area_size   = 64;
    static const unsigned num_areas = 3;
};


UCS_TEST_F(test_strided_alloc, basic) {

    ucs_strided_alloc_t sa;

    ucs_strided_alloc_init(&sa, area_size, num_areas);

    std::vector<void*> objs;
    for (size_t i = 0; i < 2; ++i) {
        /* allocate */
        void *base = ucs_strided_alloc_get(&sa, "test");

        for (unsigned j = 0; j < num_areas; ++j) {
            void *area = ucs_strided_elem_get(base, 0, j);
            memset(area, i*j, area_size);
        }

        /* save in a vector */
        objs.push_back(base);
    }

    /* check data integrity */
    char buf[area_size];
    for (size_t i = 0; i < objs.size(); ++i) {
        void *base = objs[i];

        for (unsigned j = 0; j < num_areas; ++j) {
            void *area = ucs_strided_elem_get(base, 0, j);
            memset(buf, i*j, area_size);
            EXPECT_EQ(0, memcmp(area, buf, area_size));
        }
    }

    /* release */
    while (!objs.empty()) {
        void *base = objs.back();
        objs.pop_back();
        ucs_strided_alloc_put(&sa, base);
    }

    ucs_strided_alloc_cleanup(&sa);
}
