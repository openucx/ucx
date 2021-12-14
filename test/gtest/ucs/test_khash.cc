/**
* Copyright (C) NVIDIA Corporation. 2021.  ALL RIGHTS RESERVED.
*/

#include <common/test.h>
#include <ucs/datastruct/khash.h>
#include <string.h>

KHASH_MAP_INIT_INT64(test_khash, size_t)

class test_khash : public ucs::test {
};

UCS_TEST_F(test_khash, init_inplace) {
    khash_t(test_khash) kh_static_init = KHASH_STATIC_INITIALIZER;
    khash_t(test_khash) kh_init_inplace;

    memset(&kh_init_inplace, -1, sizeof(kh_init_inplace));
    kh_init_inplace(test_khash, &kh_init_inplace);

    ASSERT_EQ(sizeof(kh_static_init), sizeof(kh_init_inplace));

    /* Check that static initializer produces same result as kh_init_inplace */
    EXPECT_EQ(0, memcmp(&kh_static_init, &kh_init_inplace,
              sizeof(kh_static_init)));
}

