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
protected:
    void check_mask_str(uint64_t mask, const std::string &exp_str) const {
        ucs_string_buffer_t mask_str;
        ucs_string_buffer_init(&mask_str);
        EXPECT_EQ(exp_str,
                  static_cast<std::string>(
                      ucs_mask_str(mask, &mask_str)));
        ucs_string_buffer_cleanup(&mask_str);
    }
};

UCS_TEST_F(test_string, trim) {
    char str1[] = " foo ";
    EXPECT_EQ("foo", std::string(ucs_strtrim(str1)));

    char str2[] = " foo foo   ";
    EXPECT_EQ("foo foo", std::string(ucs_strtrim(str2)));
}

UCS_TEST_F(test_string, snprintf_safe) {
    char buf[4];

    ucs_snprintf_safe(buf, sizeof(buf), "12");
    EXPECT_EQ(std::string("12"), buf);

    ucs_snprintf_safe(buf, sizeof(buf), "123");
    EXPECT_EQ(std::string("123"), buf);

    ucs_snprintf_safe(buf, sizeof(buf), "1234");
    EXPECT_EQ(std::string("123"), buf);
}

UCS_TEST_F(test_string, mask_str) {
    const uint64_t empty_mask = 0;

    check_mask_str(empty_mask, "<none>");

    uint64_t mask = empty_mask;
    std::string exp_str;
    for (int i = 0; i < 64; ++i) {
        mask |= UCS_BIT(i);

        if (!exp_str.empty()) {
            exp_str += ", ";
        }
        exp_str     += ucs::to_string(i);

        check_mask_str(mask, exp_str);
    }
}

UCS_TEST_F(test_string, range_str) {
    char buf[64];
    EXPECT_EQ(std::string("1..10"),
              ucs_memunits_range_str(1, 10, buf, sizeof(buf)));
    EXPECT_EQ(std::string("10"),
              ucs_memunits_range_str(10, 10, buf, sizeof(buf)));
}

class test_string_buffer : public ucs::test {
protected:
    void test_fixed(ucs_string_buffer_t *strb, size_t capacity);
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
    std::string str, exp_str;

    str.resize(100);
    std::fill(str.begin(), str.end(), 'e');

    ucs_string_buffer_init(&strb);

    for (unsigned i = 0; i < 10; ++i) {
        ucs_string_buffer_appendf(&strb, "%s", str.c_str());
        exp_str += str;
        EXPECT_EQ(exp_str.c_str(), std::string(ucs_string_buffer_cstr(&strb)));
    }

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

void test_string_buffer::test_fixed(ucs_string_buffer_t *strb, size_t capacity)
{
    ucs_string_buffer_appendf(strb, "%s", "im");
    ucs_string_buffer_appendf(strb, "%s", "mrmeeseeks");
    ucs_string_buffer_appendf(strb, "%s", "lookatme");

    EXPECT_LE(ucs_string_buffer_length(strb), capacity - 1);
    EXPECT_EQ(std::string("immrmeeseeksloo"), ucs_string_buffer_cstr(strb));
}

UCS_TEST_F(test_string_buffer, fixed_static) {
    char buf[17];
    UCS_STRING_BUFFER_STATIC(strb, buf);
    test_fixed(&strb, sizeof(buf));
}

UCS_TEST_F(test_string_buffer, fixed_init) {
    ucs_string_buffer_t strb;
    char buf[17];

    ucs_string_buffer_init_fixed(&strb, buf, sizeof(buf));
    test_fixed(&strb, sizeof(buf));
}

UCS_TEST_F(test_string_buffer, fixed_onstack) {
    const size_t num_elems = 17;
    UCS_STRING_BUFFER_ONSTACK(strb, num_elems);
    test_fixed(&strb, num_elems);
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
