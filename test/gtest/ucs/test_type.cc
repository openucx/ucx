/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <ucs/type/cpu_set.h>
}

#include <time.h>

class test_type : public ucs::test {
};

UCS_TEST_F(test_type, cpu_set) {
    ucs_cpu_set_t cpu_mask;

    UCS_CPU_ZERO(&cpu_mask);
    EXPECT_FALSE(ucs_cpu_is_set(117, &cpu_mask));
    EXPECT_FALSE(ucs_cpu_is_set(127, &cpu_mask));
    EXPECT_EQ(0, ucs_cpu_set_find_lcs(&cpu_mask));

    UCS_CPU_SET(127, &cpu_mask);
    UCS_CPU_SET(117, &cpu_mask);
    EXPECT_TRUE(ucs_cpu_is_set(117, &cpu_mask));
    EXPECT_TRUE(ucs_cpu_is_set(127, &cpu_mask));
    EXPECT_EQ(117, ucs_cpu_set_find_lcs(&cpu_mask));

    UCS_CPU_CLR(117, &cpu_mask);
    EXPECT_FALSE(ucs_cpu_is_set(117, &cpu_mask));
    EXPECT_TRUE(ucs_cpu_is_set(127, &cpu_mask));
    EXPECT_EQ(127, ucs_cpu_set_find_lcs(&cpu_mask));

    UCS_CPU_CLR(127, &cpu_mask);
    EXPECT_FALSE(ucs_cpu_is_set(117, &cpu_mask));
    EXPECT_FALSE(ucs_cpu_is_set(127, &cpu_mask));
    EXPECT_EQ(0, ucs_cpu_set_find_lcs(&cpu_mask));
}

