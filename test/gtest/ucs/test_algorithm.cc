/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/algorithm/crc.h>
#include <ucs/algorithm/qsort_r.h>
}
#include <vector>

class test_algorithm : public ucs::test {
protected:

    static int compare_func(const void *elem1, const void *elem2)
    {
        return *(const int*)elem1 - *(const int*)elem2;
    }

    static int compare_func_r(const void *elem1, const void *elem2, void *arg)
    {
        EXPECT_TRUE(MAGIC == arg);
        return compare_func(elem1, elem2);
    }

    static void *MAGIC;
};

void *test_algorithm::MAGIC = (void*)0xdeadbeef1ee7a880ull;

UCS_TEST_F(test_algorithm, qsort_r) {
    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        unsigned nmemb = ucs::rand() % 100;

        std::vector<int> vec;
        for (unsigned j = 0; j < nmemb; ++j) {
            vec.push_back(ucs::rand() % 200);
        }

        std::vector<int> vec2 = vec;
        qsort(&vec2[0], nmemb, sizeof(int), compare_func);

        ucs_qsort_r(&vec[0], nmemb, sizeof(int), compare_func_r, MAGIC);
        ASSERT_EQ(vec2, vec);
    }
}

UCS_TEST_F(test_algorithm, crc16) {
    std::string test_str;

    test_str = "";
    EXPECT_EQ(0u, ucs_crc16_string(test_str.c_str()));

    test_str = "0";
    EXPECT_EQ(0xc1fbu, ucs_crc16_string(test_str.c_str()));

    test_str = "01";
    EXPECT_EQ(0x99efu, ucs_crc16_string(test_str.c_str()));

    test_str = "012";
    EXPECT_EQ(0xfd89u, ucs_crc16_string(test_str.c_str()));

    test_str = "0123";
    EXPECT_EQ(0xea54u, ucs_crc16_string(test_str.c_str()));

    test_str = "01234";
    EXPECT_EQ(0x9394u, ucs_crc16_string(test_str.c_str()));

    test_str = "012345";
    EXPECT_EQ(0x4468u, ucs_crc16_string(test_str.c_str()));

    test_str = "0123456";
    EXPECT_EQ(0x4bc7u, ucs_crc16_string(test_str.c_str()));

    test_str = "01234567";
    EXPECT_EQ(0x07bcu, ucs_crc16_string(test_str.c_str()));

    test_str = "012345678";
    EXPECT_EQ(0x3253u, ucs_crc16_string(test_str.c_str()));

    test_str = "0123456789";
    EXPECT_EQ(0x3c16u, ucs_crc16_string(test_str.c_str()));
}

UCS_TEST_F(test_algorithm, crc32) {
    std::string test_str;

    test_str = "";
    EXPECT_EQ(0u, ucs_crc32(0, test_str.c_str(), test_str.size()));

    test_str = "0";
    EXPECT_EQ(0xf4dbdf21ul, ucs_crc32(0, test_str.c_str(), test_str.size()));

    test_str = "01";
    EXPECT_EQ(0xcf412436ul, ucs_crc32(0, test_str.c_str(), test_str.size()));

    test_str = "012";
    EXPECT_EQ(0xd5a06ab0ul, ucs_crc32(0, test_str.c_str(), test_str.size()));

    test_str = "0123";
    EXPECT_EQ(0xa6669d7dul, ucs_crc32(0, test_str.c_str(), test_str.size()));

    test_str = "01234";
    EXPECT_EQ(0xdda47024ul, ucs_crc32(0, test_str.c_str(), test_str.size()));

    test_str = "012345";
    EXPECT_EQ(0xb86f6b0ful, ucs_crc32(0, test_str.c_str(), test_str.size()));

    test_str = "0123456";
    EXPECT_EQ(0x8dbf08eeul, ucs_crc32(0, test_str.c_str(), test_str.size()));

    test_str = "01234567";
    EXPECT_EQ(0x2d803af5ul, ucs_crc32(0, test_str.c_str(), test_str.size()));

    test_str = "012345678";
    EXPECT_EQ(0x37fad1baul, ucs_crc32(0, test_str.c_str(), test_str.size()));

    test_str = "0123456789";
    EXPECT_EQ(0xa684c7c6ul, ucs_crc32(0, test_str.c_str(), test_str.size()));
}
