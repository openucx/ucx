/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/datastruct/string_buffer.h>
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

class test_string_buffer : public ucs::test {
};


UCS_TEST_F(test_string_buffer, appendf) {
    ucs_string_buffer_t strb;

    ucs_string_buffer_init(&strb);

    ucs_string_buffer_appendf(&strb, "%s", "We,");
    ucs_string_buffer_appendf(&strb, "%s", " Created,");
    ucs_string_buffer_appendf(&strb, "%s-%s", " The", "Monster");

    EXPECT_EQ("We, Created, The-Monster",
              std::string(ucs_string_buffer_cstr(&strb)));

    ucs_string_buffer_cleanup(&strb);
}
