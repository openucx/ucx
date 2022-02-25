/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/debug/memtrack_int.h>
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

UCS_TEST_F(test_string, is_empty) {
    EXPECT_TRUE(ucs_string_is_empty(""));
    EXPECT_FALSE(ucs_string_is_empty("aaa"));
}

UCS_TEST_F(test_string, count_char) {
    static const char *str1 = "/foo";
    static const char *str2 = "/foo/bar";
    size_t count;

    count = ucs_string_count_char(str1, '/');
    EXPECT_EQ(1, count);

    count = ucs_string_count_char((const char*)UCS_PTR_BYTE_OFFSET(str1, 1),
                                  '/');
    EXPECT_EQ(0, count);

    count = ucs_string_count_char(str2, '/');
    EXPECT_EQ(2, count);

    count = ucs_string_count_char((const char*)UCS_PTR_BYTE_OFFSET(str2, 1),
                                  '/');
    EXPECT_EQ(1, count);
}

UCS_TEST_F(test_string, common_prefix_len) {
    static const char *str1 = "/foo";
    static const char *str2 = "/foobar";
    static const char *str3 = "foo/bar";
    size_t common_length;

    common_length = ucs_string_common_prefix_len(str1, str2);
    EXPECT_EQ(4, common_length);

    common_length = ucs_string_common_prefix_len(str1, str3);
    EXPECT_EQ(0, common_length);
}

UCS_TEST_F(test_string, path) {
    char path[PATH_MAX];
    ucs_path_get_common_parent("/sys/dev/one", "/sys/dev/two", path);
    EXPECT_STREQ("/sys/dev", path);

    EXPECT_EQ(4, ucs_path_calc_distance("/root/foo/bar", "/root/charlie/fox"));
    EXPECT_EQ(2, ucs_path_calc_distance("/a/b/c/d", "/a/b/c/e"));
    EXPECT_EQ(0, ucs_path_calc_distance("/a/b/c", "/a/b/c"));
    EXPECT_EQ(1, ucs_path_calc_distance("/a/b/c", "/a/b"));
    EXPECT_EQ(2, ucs_path_calc_distance("/a/b/cd", "/a/b/ce"));
    EXPECT_EQ(3, ucs_path_calc_distance("/a/b/c", "/a/b_c"));
}

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

UCS_TEST_F(test_string, split) {
    // No remainder
    {
        char str1[] = "foo,bar";
        char *p1, *p2;
        char *ret = ucs_string_split(str1, ",", 2, &p1, &p2);
        EXPECT_EQ(std::string("foo"), p1);
        EXPECT_EQ(std::string("bar"), p2);
        EXPECT_EQ(NULL, ret);
    }
    // Have a remainder
    {
        char str1[] = "foo,bar,baz,a,b,c";
        char *p1, *p2;
        char *ret = ucs_string_split(str1, ",", 2, &p1, &p2);
        EXPECT_EQ(std::string("foo"), p1);
        EXPECT_EQ(std::string("bar"), p2);
        EXPECT_EQ(std::string("baz,a,b,c"), ret);
    }
    // Less tokens than requested, and some are empty
    {
        char str1[] = "foo,:bar";
        char *p1, *p2, *p3, *p4;
        char *ret = ucs_string_split(str1, ",:", 4, &p1, &p2, &p3, &p4);
        EXPECT_EQ(std::string("foo"), p1);
        EXPECT_EQ(std::string(""), p2);
        EXPECT_EQ(std::string("bar"), p3);
        EXPECT_EQ(NULL, p4);
        EXPECT_EQ(NULL, ret);
    }
}

class test_string_buffer : public ucs::test {
protected:
    void test_fixed(ucs_string_buffer_t *strb, size_t capacity);
    void check_extract_mem(ucs_string_buffer_t *strb);
};

UCS_TEST_F(test_string_buffer, appendf) {
    ucs_string_buffer_t strb;

    ucs_string_buffer_init(&strb);

    ucs_string_buffer_appendf(&strb, "%s", "We,");
    ucs_string_buffer_appendf(&strb, "%s", " Created,");
    ucs_string_buffer_appendf(&strb, "%s-%s", " The", "Monster");

    EXPECT_EQ("We, Created, The-Monster",
              std::string(ucs_string_buffer_cstr(&strb)));

    ucs_string_buffer_reset(&strb);
    EXPECT_EQ("", std::string(ucs_string_buffer_cstr(&strb)));

    ucs_string_buffer_appendf(&strb, "%s", "Clean slate");
    EXPECT_EQ("Clean slate", std::string(ucs_string_buffer_cstr(&strb)));

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
    ucs_string_buffer_rtrim(&strb, "x");
    EXPECT_EQ(std::string(""), ucs_string_buffer_cstr(&strb));
    ucs_string_buffer_cleanup(&strb);

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

UCS_TEST_F(test_string_buffer, append_hex) {
    static const uint8_t hexbytes[] = {0xde, 0xad, 0xbe, 0xef,
                                       0xba, 0xdc, 0xf,  0xee};
    UCS_STRING_BUFFER_ONSTACK(strb, 128);
    ucs_string_buffer_append_hex(&strb, hexbytes,
                                 ucs_static_array_size(hexbytes), SIZE_MAX);
    EXPECT_EQ(std::string("deadbeef:badc0fee"), ucs_string_buffer_cstr(&strb));
}

UCS_TEST_F(test_string_buffer, append_iovec) {
    static const struct iovec iov[3] = {{NULL, 0},
                                        {(void*)0x1234, 100},
                                        {(void*)0x4567, 200}};
    UCS_STRING_BUFFER_ONSTACK(strb, 128);
    ucs_string_buffer_append_iovec(&strb, iov, ucs_static_array_size(iov));
    EXPECT_EQ(std::string("(nil),0|0x1234,100|0x4567,200"),
              ucs_string_buffer_cstr(&strb));
}

UCS_TEST_F(test_string_buffer, flags) {
    static const char *flag_names[] = {"zero", "one", "two", "three", "four"};
    UCS_STRING_BUFFER_ONSTACK(strb, 128);
    /* coverity[overrun-buffer-val] */
    ucs_string_buffer_append_flags(&strb, UCS_BIT(1) | UCS_BIT(3), flag_names);
    EXPECT_EQ(std::string("one|three"), ucs_string_buffer_cstr(&strb));
}

UCS_TEST_F(test_string_buffer, dump) {
    UCS_STRING_BUFFER_ONSTACK(strb, 128);
    ucs_string_buffer_appendf(&strb, "hungry\n");
    ucs_string_buffer_appendf(&strb, "for\n");
    ucs_string_buffer_appendf(&strb, "apples\n");
    ucs_string_buffer_dump(&strb, "[ TEST     ] ", stdout);
}

UCS_TEST_F(test_string_buffer, tokenize) {
    UCS_STRING_BUFFER_ONSTACK(strb, 128);
    ucs_string_buffer_appendf(&strb, "nova&noob|crocubot+ants&&rails");

    std::vector<std::string> names;
    char *name;
    ucs_string_buffer_for_each_token(name, &strb, "&|+") {
        names.push_back(name);
    }

    EXPECT_EQ(std::vector<std::string>(
                      {"nova", "noob", "crocubot", "ants", "", "rails"}),
              names);
}

UCS_TEST_F(test_string_buffer, appendc) {
    UCS_STRING_BUFFER_ONSTACK(strb, 8);

    ucs_string_buffer_appendc(&strb, '0', 0);
    ucs_string_buffer_appendc(&strb, '1', 1);
    ucs_string_buffer_appendc(&strb, '2', 2);
    ucs_string_buffer_appendc(&strb, '3', 3);
    ucs_string_buffer_appendc(&strb, '4', 4);
    ucs_string_buffer_appendc(&strb, '5', 5);

    // The string buffer should not exceed its limit (8)
    EXPECT_EQ(std::string("1223334"), ucs_string_buffer_cstr(&strb));
}

void test_string_buffer::check_extract_mem(ucs_string_buffer_t *strb)
{
    char test_str[] = "test";
    ucs_string_buffer_appendf(strb, "%s", test_str);
    char *c_str = ucs_string_buffer_extract_mem(strb);
    EXPECT_STREQ(test_str, c_str);
    ucs_free(c_str);
}

UCS_TEST_F(test_string_buffer, extract_mem) {
    ucs_string_buffer_t strb;
    char buf[8];

    ucs_string_buffer_init_fixed(&strb, buf, sizeof(buf));
    check_extract_mem(&strb);

    ucs_string_buffer_init(&strb);
    check_extract_mem(&strb);
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
