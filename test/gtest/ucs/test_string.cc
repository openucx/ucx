/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/sys/string.h>
}

class test_string : public ucs::test {
};

UCS_TEST_F(test_string, trim) {
    char str1[] = " foo ";
    EXPECT_EQ("foo", std::string(ucs_strtrim(str1)));

    char str2[] = " foo foo   ";
    EXPECT_EQ("foo foo", std::string(ucs_strtrim(str2)));
}
