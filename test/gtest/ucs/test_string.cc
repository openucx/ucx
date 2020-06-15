/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/datastruct/string_buffer.h>
#include <ucs/datastruct/string_set.h>
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

UCS_TEST_F(test_string_buffer, append_long) {
    ucs_string_buffer_t strb;
    std::string str;

    str.resize(100);
    std::fill(str.begin(), str.end(), 'e');

    ucs_string_buffer_init(&strb);

    ucs_string_buffer_appendf(&strb, "%s", str.c_str());
    EXPECT_EQ(str.c_str(), std::string(ucs_string_buffer_cstr(&strb)));

    ucs_string_buffer_appendf(&strb, "%s", str.c_str());
    EXPECT_EQ((str + str).c_str(), std::string(ucs_string_buffer_cstr(&strb)));

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, rtrim) {
    static const char *test_string = "wabbalubbadabdab";
    ucs_string_buffer_t strb;

    ucs_string_buffer_init(&strb);
    ucs_string_buffer_appendf(&strb, "%s%s", test_string, ",,");
    ucs_string_buffer_rtrim(&strb, ",");
    EXPECT_EQ(std::string(test_string), ucs_string_buffer_cstr(&strb));
    ucs_string_buffer_cleanup(&strb);

    ucs_string_buffer_init(&strb);
    ucs_string_buffer_appendf(&strb, "%s%s", test_string, " \t  \n \r  ");
    ucs_string_buffer_rtrim(&strb, NULL);
    EXPECT_EQ(std::string(test_string), ucs_string_buffer_cstr(&strb));
    ucs_string_buffer_cleanup(&strb);
}

class test_string_set : public ucs::test {
};

UCS_TEST_F(test_string_set, add) {
    ucs_string_set_t sset;

    ucs_string_set_init(&sset);

    ucs_string_set_add(&sset, "We");
    ucs_string_set_addf(&sset, "%s", "Created");
    ucs_string_set_addf(&sset, "%s-%s", "The", "Monster");

    EXPECT_TRUE (ucs_string_set_contains(&sset, "We"));
    EXPECT_FALSE(ucs_string_set_contains(&sset, "Created "));
    EXPECT_TRUE (ucs_string_set_contains(&sset, "Created"));

    ucs_string_buffer_t strb;
    ucs_string_buffer_init(&strb);
    ucs_string_set_print_sorted(&sset, &strb, ",");

    EXPECT_EQ("Created,The-Monster,We",
              std::string(ucs_string_buffer_cstr(&strb)));

    ucs_string_buffer_cleanup(&strb);

    ucs_string_set_cleanup(&sset);
}
