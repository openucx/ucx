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



    static void * MAGIC;
};

void * test_algorithm::MAGIC = (void*)0xdeadbeef1ee7a880ull;

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

UCS_TEST_F(test_algorithm, crc16_string) {
    UCS_TEST_MESSAGE << "crc16 of '123456789' is 0x" << std::hex <<
                    ucs_crc16_string("123456789") << std::dec;
    EXPECT_NE(ucs_crc16_string("123456789"),
              ucs_crc16_string("12345"));
}
