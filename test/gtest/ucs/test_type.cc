/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <ucs/type/cpu_set.h>
#include <ucs/type/init_once.h>
#include <ucs/type/status.h>
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

UCS_TEST_F(test_type, status) {
    void *ptr = (void*)0xff00000000ul;
    EXPECT_TRUE(UCS_PTR_IS_PTR(ptr));
    EXPECT_FALSE(UCS_PTR_IS_PTR(NULL));
    EXPECT_NE(UCS_OK, UCS_PTR_STATUS(ptr));
}

class test_init_once: public test_type {
protected:
    test_init_once() : m_once(INIT_ONCE_INIT), m_count(0) {};

    /* counter is not atomic, we expect the lock of init_once will protect it */
    ucs_init_once_t m_once;
    int             m_count;

private:
    static const ucs_init_once_t INIT_ONCE_INIT;
};

const ucs_init_once_t test_init_once::INIT_ONCE_INIT = UCS_INIT_ONCE_INITIALIZER;

UCS_MT_TEST_F(test_init_once, init_once, 10) {

    for (int i = 0; i < 100; ++i) {
        UCS_INIT_ONCE(&m_once) {
            ++m_count;
        }
    }

    EXPECT_EQ(1, m_count);
}

