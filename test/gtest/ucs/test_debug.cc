/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/debug/debug.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>
}

#include <dlfcn.h>

extern "C" {

int UCS_F_NOINLINE my_cool_function(unsigned *lineno, void **address)
{
    int a;

    a = 5;

    ucs_compiler_fence();
label1: *lineno = __LINE__;
    ucs_compiler_fence();
    *address = &&label1;

    ++a;
    return a;
}

}

class test_debug : public ucs::test {
};

std::string __basename(const std::string& path) {
    char *p = strdup(path.c_str());
    std::string bn(::basename(p));
    free(p);
    return bn;
}

UCS_TEST_F(test_debug, lookup_ucs_func) {
    const char sym[] = "ucs_log_flush";

    ucs_debug_address_info info;
    ucs_status_t status = ucs_debug_lookup_address(dlsym(RTLD_DEFAULT, sym), &info);
    ASSERT_UCS_OK(status);

    EXPECT_NE(std::string::npos, std::string(info.file.path).find("libucs.so"));
#ifdef HAVE_DETAILED_BACKTRACE
    EXPECT_EQ(sym, std::string(info.function));
#endif
}

UCS_TEST_F(test_debug, lookup_invalid) {
    ucs_debug_address_info info;
    ucs_status_t status = ucs_debug_lookup_address((void*)0xffffffffffff, &info);
    EXPECT_EQ(UCS_ERR_NO_ELEM, status);
}

UCS_TEST_F(test_debug, lookup_address) {
    unsigned lineno;
    void *address;

    if (BULLSEYE_ON) {
        UCS_TEST_SKIP;
    }

    my_cool_function(&lineno, &address);

    ucs_debug_address_info info;
    ucs_status_t status = ucs_debug_lookup_address(address, &info);
    ASSERT_UCS_OK(status);

    EXPECT_NE(std::string::npos, std::string(info.file.path).find("gtest"));

#ifdef HAVE_DETAILED_BACKTRACE
    EXPECT_EQ("my_cool_function", std::string(info.function));
    EXPECT_EQ(lineno, info.line_number);
    EXPECT_EQ(__basename(__FILE__), __basename(info.source_file));
#else
    EXPECT_EQ(0u, info.line_number);
    EXPECT_EQ("???", std::string(info.source_file));
#endif
}

UCS_TEST_F(test_debug, print_backtrace) {
    char *data;
    size_t size;

    FILE *f = open_memstream(&data, &size);
    ucs_debug_print_backtrace(f, 0);
    fclose(f);

    /* Some functions that should appear */
    EXPECT_TRUE(strstr(data, "print_backtrace") != NULL);
#ifdef HAVE_DETAILED_BACKTRACE
    EXPECT_TRUE(strstr(data, "main") != NULL);
#endif

    free(data);
}
