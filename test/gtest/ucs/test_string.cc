/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/debug/memtrack_int.h>
#include <ucs/datastruct/string_buffer.h>
#include <ucs/datastruct/string_set.h>
#include <ucs/sys/compiler.h>
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
    std::string path(PATH_MAX, '\0');

    ucs_path_get_common_parent("/sys/dev/one", "/sys/dev/two", &path[0]);
    EXPECT_STREQ("/sys/dev", path.c_str());

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

UCS_TEST_F(test_string, to_memunits) {
    size_t value;

    // Just a number
    {
        EXPECT_EQ(UCS_OK, ucs_str_to_memunits("123", &value));
        EXPECT_EQ(123, value);

        EXPECT_EQ(UCS_OK, ucs_str_to_memunits("0", &value));
        EXPECT_EQ(0, value);
    }
    // Invalid values
    {
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, ucs_str_to_memunits("abc", &value));
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, ucs_str_to_memunits("", &value));
    }

    // Number and 'b'
    {
        EXPECT_EQ(UCS_OK, ucs_str_to_memunits("123B", &value));
        EXPECT_EQ(123, value);

        EXPECT_EQ(UCS_OK, ucs_str_to_memunits("123b", &value));
        EXPECT_EQ(123, value);
    }
    // Invalid values
    {
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, ucs_str_to_memunits("123!", &value));
    }

    // Number and multiplier
    {
        EXPECT_EQ(UCS_OK, ucs_str_to_memunits("2k", &value));
        EXPECT_EQ(2 * UCS_KBYTE, value);

        EXPECT_EQ(UCS_OK, ucs_str_to_memunits("2M", &value));
        EXPECT_EQ(2 * UCS_MBYTE, value);

        EXPECT_EQ(UCS_OK, ucs_str_to_memunits("1G", &value));
        EXPECT_EQ(1 * UCS_GBYTE, value);
    }

    // Number and multiplier and 'b'
    {
        EXPECT_EQ(UCS_OK, ucs_str_to_memunits("2kb", &value));
        EXPECT_EQ(2 * UCS_KBYTE, value);

        EXPECT_EQ(UCS_OK, ucs_str_to_memunits("2KB", &value));
        EXPECT_EQ(2 * UCS_KBYTE, value);

        EXPECT_EQ(UCS_OK, ucs_str_to_memunits("2kB", &value));
        EXPECT_EQ(2 * UCS_KBYTE, value);

        EXPECT_EQ(UCS_OK, ucs_str_to_memunits("2MB", &value));
        EXPECT_EQ(2 * UCS_MBYTE, value);

        EXPECT_EQ(UCS_OK, ucs_str_to_memunits("1GB", &value));
        EXPECT_EQ(1 * UCS_GBYTE, value);
    }

    // Special values
    {
        EXPECT_EQ(UCS_OK, ucs_str_to_memunits("inf", &value));
        EXPECT_EQ(UCS_MEMUNITS_INF, value);

        EXPECT_EQ(UCS_OK, ucs_str_to_memunits("auto", &value));
        EXPECT_EQ(UCS_MEMUNITS_AUTO, value);
    }
}

class test_string_buffer : public ucs::test {
protected:
    void test_fixed(ucs_string_buffer_t *strb, size_t capacity);
    void check_extract_mem(ucs_string_buffer_t *strb);
    static char make_lowercase_remove_underscores(char ch);
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

UCS_TEST_F(test_string_buffer, rbrk) {
    UCS_STRING_BUFFER_ONSTACK(strb, 128);

    ucs_string_buffer_appendf(&strb, "one.two.three..");

    ucs_string_buffer_rbrk(&strb, ".");
    EXPECT_EQ(std::string("one.two.three."), ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_rbrk(&strb, ".");
    EXPECT_EQ(std::string("one.two.three"), ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_rbrk(&strb, ".");
    EXPECT_EQ(std::string("one.two"), ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_rbrk(&strb, ".");
    EXPECT_EQ(std::string("one"), ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_rbrk(&strb, ".");
    EXPECT_EQ(std::string("one"), ucs_string_buffer_cstr(&strb));
}

UCS_TEST_F(test_string_buffer, rbrk_empty) {
    UCS_STRING_BUFFER_ONSTACK(strb, 128);

    ucs_string_buffer_rbrk(&strb, ".");
    EXPECT_EQ(std::string(""), ucs_string_buffer_cstr(&strb));
}

void test_string_buffer::test_fixed(ucs_string_buffer_t *strb, size_t capacity)
{
    ucs_string_buffer_appendf(strb, "%s", "im");
    ucs_string_buffer_appendf(strb, "%s", "mrmeeseeks");
    ucs_string_buffer_appendf(strb, "%s", "lookatme");

    EXPECT_EQ(ucs_string_buffer_length(strb), capacity - 1);
    EXPECT_EQ(std::string("immrmeeseekslook"), ucs_string_buffer_cstr(strb));
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
    EXPECT_EQ(std::string(""), ucs_string_buffer_cstr(&strb));
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

UCS_TEST_F(test_string_buffer, array) {
    static const char *str_array[] = {"once", "upon", "a", "time"};
    UCS_STRING_BUFFER_ONSTACK(strb, 128);
    ucs_string_buffer_append_array(&strb, " ", "%s", str_array,
                                   ucs_static_array_size(str_array));
    EXPECT_EQ(std::string("once upon a time"), ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_reset(&strb);
    static int num_array[] = {1, 2, 3, 4};
    ucs_string_buffer_append_array(&strb, ",", "%d", num_array,
                                   ucs_static_array_size(num_array));
    EXPECT_EQ(std::string("1,2,3,4"), ucs_string_buffer_cstr(&strb));
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

char test_string_buffer::make_lowercase_remove_underscores(char ch)
{
    if (isupper(ch)) {
        return tolower(ch);
    } else if (ch == '_') {
        return '\0';
    } else {
        return ch;
    }
}

UCS_TEST_F(test_string_buffer, ucs_string_buffer_translate) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

    ucs_string_buffer_appendf(&strb, "Camel_Case_With_Underscores1234");

    ucs_string_buffer_translate(&strb, make_lowercase_remove_underscores);
    EXPECT_EQ(std::string("camelcasewithunderscores1234"),
              ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_range_prefix_suffix) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

    ASSERT_EQ(UCS_OK, ucs_string_buffer_expand_range(&strb, "prefix[0-2]suffix",
                                                     ',', SIZE_MAX, NULL));
    EXPECT_EQ(std::string("prefix0suffix,prefix1suffix,prefix2suffix"),
              ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_range_prefix) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

    ASSERT_EQ(UCS_OK, ucs_string_buffer_expand_range(&strb, "prefix[0-2]", ',',
                                                     SIZE_MAX, NULL));
    EXPECT_EQ(std::string("prefix0,prefix1,prefix2"),
              ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_range_suffix) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

    ASSERT_EQ(UCS_OK, ucs_string_buffer_expand_range(&strb, "[3-5]suffix", ',',
                                                     SIZE_MAX, NULL));
    EXPECT_EQ(std::string("3suffix,4suffix,5suffix"),
              ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_range_no_prefix_suffix) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

    ASSERT_EQ(UCS_OK, ucs_string_buffer_expand_range(&strb, "[0-2]", ',',
                                                     SIZE_MAX, NULL));
    EXPECT_EQ(std::string("0,1,2"), ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_range_no_bracket) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

    ASSERT_EQ(UCS_OK, ucs_string_buffer_expand_range(&strb, "no_bracket", ',',
                                                     SIZE_MAX, NULL));
    EXPECT_EQ(std::string("no_bracket"), ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_range_single) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

    ASSERT_EQ(UCS_OK, ucs_string_buffer_expand_range(&strb, "dev[99-99]", ',',
                                                     SIZE_MAX, NULL));
    EXPECT_EQ(std::string("dev99"), ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_range_multi_digit) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

    ASSERT_EQ(UCS_OK, ucs_string_buffer_expand_range(&strb, "dev[98-101]", ',',
                                                     SIZE_MAX, NULL));
    EXPECT_EQ(std::string("dev98,dev99,dev100,dev101"),
              ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_range_malformed) {
    /* Tokens without an opening bracket are treated as literals */
    const std::string malformed[] = {"no_bracket", "prefix2-]", "prefix]abc",
                                     "hello]]",    "]]-",       "]--"};

    for (const std::string &token : malformed) {
        ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

        ASSERT_EQ(UCS_OK,
                  ucs_string_buffer_expand_range(&strb, token.c_str(), ',',
                                                 SIZE_MAX, NULL))
                << "token: " << token;
        EXPECT_EQ(token, ucs_string_buffer_cstr(&strb))
                << "token: " << token;

        ucs_string_buffer_cleanup(&strb);
    }
}

UCS_TEST_F(test_string_buffer, expand_range_empty) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    size_t count;

    ASSERT_EQ(UCS_OK,
              ucs_string_buffer_expand_range(&strb, "", ',', SIZE_MAX, &count));
    EXPECT_EQ(std::string(""), ucs_string_buffer_cstr(&strb));
    EXPECT_EQ(0ul, count);

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_range_invalid) {
    const std::string invalid[] = {"a[5-2]b",    "a[-1-2]b", "a[-2-]b",
                                   "a[2-3-]b",   "a[4]b",    "a[[]b",
                                   "a[[2-4]b",   "a[2-]b",   "a[2-3-4]b",
                                   "a[b-c]d",    "a[0-A]b",  "a[-]b",
                                   "[]",         "[-1-2-]",  "[--]",
                                   "[1-2][3-4]", "[4-8][",   "[5-6]]",
                                   "][4-5]",     "[[0-4]]",  "a[0-4]b[6-8]c"};

    for (const std::string &token : invalid) {
        ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

        {
            const scoped_log_handler slh(hide_errors_logger);
            EXPECT_EQ(UCS_ERR_INVALID_PARAM,
                      ucs_string_buffer_expand_range(&strb, token.c_str(), ',',
                                                     SIZE_MAX, NULL))
                    << "token: " << token;
        }

        ucs_string_buffer_cleanup(&strb);
    }
}

UCS_TEST_F(test_string_buffer, expand_range_append) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

    ucs_string_buffer_appendf(&strb, "previous_data,");
    ASSERT_EQ(UCS_OK, ucs_string_buffer_expand_range(&strb, "data[0-1]", ',',
                                                     SIZE_MAX, NULL));
    EXPECT_EQ(std::string("previous_data,data0,data1"),
              ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_range_max_elements) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    size_t count;

    ASSERT_EQ(UCS_OK,
              ucs_string_buffer_expand_range(&strb, "dev[0-9]", ',', 3,
                                             &count));
    EXPECT_EQ(std::string("dev0,dev1,dev2"), ucs_string_buffer_cstr(&strb));
    EXPECT_EQ(3ul, count);

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_range_max_elements_one) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    size_t count;

    ASSERT_EQ(UCS_OK,
              ucs_string_buffer_expand_range(&strb, "dev[0-9]", ',', 1,
                                             &count));
    EXPECT_EQ(std::string("dev0"), ucs_string_buffer_cstr(&strb));
    EXPECT_EQ(1ul, count);

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_range_max_elements_zero) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    size_t count;

    ASSERT_EQ(UCS_OK,
              ucs_string_buffer_expand_range(&strb, "dev[0-9]", ',', 0,
                                             &count));
    EXPECT_EQ(std::string(""), ucs_string_buffer_cstr(&strb));
    EXPECT_EQ(0ul, count);

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_range_max_elements_exceeds_range) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    size_t count;

    ASSERT_EQ(UCS_OK,
              ucs_string_buffer_expand_range(&strb, "dev[0-2]", ',', 100,
                                             &count));
    EXPECT_EQ(std::string("dev0,dev1,dev2"), ucs_string_buffer_cstr(&strb));
    EXPECT_EQ(3ul, count);

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_range_leading_zeros) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    size_t count;

    ASSERT_EQ(UCS_OK, ucs_string_buffer_expand_range(&strb, "dev[01-03]", ',',
                                                     SIZE_MAX, &count));

    /* Leading zeros are not preserved in the output */
    EXPECT_EQ(std::string("dev1,dev2,dev3"), ucs_string_buffer_cstr(&strb));
    EXPECT_EQ(3ul, count);

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_range_long_prefix_suffix) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    const std::string long_prefix(200, 'p');
    const std::string long_suffix(200, 's');
    const std::string token = long_prefix + "[0-2]" + long_suffix;
    size_t count;

    ASSERT_EQ(UCS_OK, ucs_string_buffer_expand_range(&strb, token.c_str(), ',',
                                                     SIZE_MAX, &count));
    const std::string expected = long_prefix + "0" + long_suffix + "," +
                                 long_prefix + "1" + long_suffix + "," +
                                 long_prefix + "2" + long_suffix;
    EXPECT_EQ(expected, ucs_string_buffer_cstr(&strb));
    EXPECT_EQ(3ul, count);

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_range_different_delimiters) {
    const struct {
        const char *token;
        char       delim;
        const char *expected;
    } test_cases[] = {
        {"dev[0-2]", ';', "dev0;dev1;dev2"},
        {"a[3-5]b", '@', "a3b@a4b@a5b"},
        {"[0-3]", ' ', "0 1 2 3"},
        {"eth[10-12]", '|', "eth10|eth11|eth12"},
    };

    for (const auto &tc : test_cases) {
        ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

        ASSERT_EQ(UCS_OK,
                  ucs_string_buffer_expand_range(&strb, tc.token, tc.delim,
                                                 SIZE_MAX, NULL))
                << "token: " << tc.token << " delim: '" << tc.delim << "'";
        EXPECT_EQ(std::string(tc.expected), ucs_string_buffer_cstr(&strb))
                << "token: " << tc.token << " delim: '" << tc.delim << "'";

        ucs_string_buffer_cleanup(&strb);
    }
}

UCS_TEST_F(test_string_buffer, expand_ranges_mixed) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

    ASSERT_EQ(UCS_OK,
              ucs_string_buffer_expand_ranges(&strb, "mlx5_[0-2],eth0,ib[3-5]",
                                              ',', SIZE_MAX, NULL));
    EXPECT_EQ(std::string("mlx5_0,mlx5_1,mlx5_2,eth0,ib3,ib4,ib5"),
              ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_ranges_empty) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    size_t count;

    ASSERT_EQ(UCS_OK, ucs_string_buffer_expand_ranges(&strb, "", ',', SIZE_MAX,
                                                      &count));
    EXPECT_EQ(std::string(""), ucs_string_buffer_cstr(&strb));
    EXPECT_EQ(0ul, count);

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_ranges_no_ranges) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

    ASSERT_EQ(UCS_OK,
              ucs_string_buffer_expand_ranges(&strb, "eth0,lo,ib0", ',', SIZE_MAX,
                                              NULL));
    EXPECT_EQ(std::string("eth0,lo,ib0"), ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_ranges_single_token) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

    ASSERT_EQ(UCS_OK,
              ucs_string_buffer_expand_ranges(&strb, "dev[0-3]", ',', SIZE_MAX,
                                              NULL));
    EXPECT_EQ(std::string("dev0,dev1,dev2,dev3"),
              ucs_string_buffer_cstr(&strb));

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_ranges_invalid) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

    {
        const scoped_log_handler slh(hide_errors_logger);
        EXPECT_EQ(UCS_ERR_INVALID_PARAM,
                  ucs_string_buffer_expand_ranges(&strb, "a[0-1],b[5-2]", ',',
                                                  SIZE_MAX, NULL));
    }

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_ranges_max_elements_zero) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    size_t count;

    ASSERT_EQ(UCS_OK,
              ucs_string_buffer_expand_ranges(&strb, "dev[0-4],eth0", ',', 0,
                                              &count));
    EXPECT_EQ(std::string(""), ucs_string_buffer_cstr(&strb));
    EXPECT_EQ(0ul, count);

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_ranges_max_elements_cuts_range) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    size_t count;

    ASSERT_EQ(UCS_OK, ucs_string_buffer_expand_ranges(&strb, "lo,dev[0-4],eth0",
                                                      ',', 3, &count));
    EXPECT_EQ(std::string("lo,dev0,dev1"), ucs_string_buffer_cstr(&strb));
    EXPECT_EQ(3ul, count);

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_ranges_max_elements_across_tokens) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    size_t count;

    ASSERT_EQ(UCS_OK,
              ucs_string_buffer_expand_ranges(&strb, "dev[0-2],eth[0-2]", ',',
                                              4, &count));
    EXPECT_EQ(std::string("dev0,dev1,dev2,eth0"), ucs_string_buffer_cstr(&strb));
    EXPECT_EQ(4ul, count);

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_ranges_max_elements_exact) {
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    size_t count;

    ASSERT_EQ(UCS_OK,
              ucs_string_buffer_expand_ranges(&strb, "a[0-1],b[0-1]", ',', 4,
                                              &count));
    EXPECT_EQ(std::string("a0,a1,b0,b1"), ucs_string_buffer_cstr(&strb));
    EXPECT_EQ(4ul, count);

    ucs_string_buffer_cleanup(&strb);
}

UCS_TEST_F(test_string_buffer, expand_ranges_different_delimiters) {
    const struct {
        const char *input;
        char       delim;
        const char *expected;
    } test_cases[] = {
        {"mlx5_[11-12];eth0;ib[2-4]", ';', "mlx5_11;mlx5_12;eth0;ib2;ib3;ib4"},
        {"dev[0-1]@lo", '@', "dev0@dev1@lo"},
        {"a[0-1] b[2-3]", ' ', "a0 a1 b2 b3"},
    };

    for (const auto &tc : test_cases) {
        ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

        ASSERT_EQ(UCS_OK,
                  ucs_string_buffer_expand_ranges(&strb, tc.input, tc.delim,
                                                  SIZE_MAX, NULL))
                << "input: " << tc.input << " delim: '" << tc.delim << "'";
        EXPECT_EQ(std::string(tc.expected), ucs_string_buffer_cstr(&strb))
                << "input: " << tc.input << " delim: '" << tc.delim << "'";

        ucs_string_buffer_cleanup(&strb);
    }
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
