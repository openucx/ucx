/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

extern "C" {
#include <ucm/util/sys.h>
}


class test_ucm_sys : public ucs::test {
};


UCS_TEST_F(test_ucm_sys, concat_path_zero_capacity)
{
    char storage[] = {'x', 'y'};
    char *buffer   = &storage[1];

    EXPECT_EQ(buffer, ucm_concat_path(buffer, 0, "dir", "file"));
    EXPECT_EQ('x', storage[0]);
    EXPECT_EQ('y', storage[1]);
}


UCS_TEST_F(test_ucm_sys, concat_path_one_byte_empty_dir)
{
    char storage[] = {'x', 'y'};
    char *buffer   = &storage[1];

    EXPECT_EQ(buffer, ucm_concat_path(buffer, 1, "", "file"));
    EXPECT_EQ('x', storage[0]);
    EXPECT_EQ('\0', storage[1]);
}


UCS_TEST_F(test_ucm_sys, concat_path)
{
    char buffer[16];

    EXPECT_EQ(buffer, ucm_concat_path(buffer, sizeof(buffer), "/usr/", "/lib"));
    EXPECT_STREQ("/usr/lib", buffer);
}
