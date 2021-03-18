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
#include <ucs/type/float8.h>
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

/* Represents latency (in ns) */
UCS_FP8_DECLARE_TYPE(LATENCY, UCS_BIT(7), UCS_BIT(20))

UCS_TEST_F(test_type, pack_float) {
    const std::size_t values_size    = 10;
    double values_array[values_size] = {
        130, 135.1234, 140, 200, 400, 1000, 10000, 100000, 1000000, 1000000
    };
    std::vector<double> values(values_array, values_array + values_size);
    float unpacked;

    /* 0 -> 0 */
    unpacked = UCS_FP8_UNPACK(LATENCY, UCS_FP8_PACK(LATENCY, 0));
    EXPECT_EQ(unpacked, 0);

    /* NaN -> NaN */
    unpacked = UCS_FP8_UNPACK(LATENCY, UCS_FP8_PACK(LATENCY, NAN));
    EXPECT_TRUE(isnan(unpacked));

    /* Below min -> min */
    EXPECT_EQ(UCS_FP8_UNPACK(LATENCY, UCS_FP8_PACK(LATENCY, UCS_BIT(7))),
              UCS_FP8_UNPACK(LATENCY, UCS_FP8_PACK(LATENCY, 15)));

    /* Precision test throughout the whole range */
    for (std::vector<double>::const_iterator it = values.begin();
         it < values.end(); it++) {
        unpacked = UCS_FP8_UNPACK(LATENCY, UCS_FP8_PACK(LATENCY, *it));
        ucs_assert((UCS_FP8_PRECISION < unpacked / *it) &&
                   (unpacked / *it <= 1));
    }

    /* Above max -> max */
    EXPECT_EQ(UCS_FP8_UNPACK(LATENCY, UCS_FP8_PACK(LATENCY, UCS_BIT(20))),
              UCS_FP8_UNPACK(LATENCY, UCS_FP8_PACK(LATENCY, 200000000)));
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

