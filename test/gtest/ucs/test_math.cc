/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>

#include <ucs/arch/bitops.h>
#include <ucs/arch/atomic.h>
#include <ucs/datastruct/linear_func.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>

#include <vector>

#define FLAG1 0x100
#define FLAG2 0x200
#define FLAG3 0x400

class test_math : public ucs::test {
protected:
    static const unsigned ATOMIC_COUNT = 50;
};

UCS_TEST_F(test_math, convert_flag) {
    volatile uint32_t value = FLAG1 | FLAG3;
    volatile uint32_t tmp = ucs_convert_flag(value, FLAG1, 0x1);

    EXPECT_EQ(0x1u, tmp);
    EXPECT_EQ(0x0u, ucs_convert_flag(value, FLAG2, 0x2u));
    EXPECT_EQ(0x4u, ucs_convert_flag(value, FLAG3, 0x4u));

    EXPECT_EQ(0x10000u, ucs_convert_flag(value, FLAG1, 0x10000u));
    EXPECT_EQ(0x00000u, ucs_convert_flag(value, FLAG2, 0x20000u));
    EXPECT_EQ(0x40000u, ucs_convert_flag(value, FLAG3, 0x40000u));
}

UCS_TEST_F(test_math, test_flag) {
    uint32_t value = FLAG2;
    EXPECT_TRUE(  ucs_test_flags(value, FLAG1, FLAG2) );
    EXPECT_TRUE(  ucs_test_flags(value, FLAG2, FLAG3) );
    EXPECT_FALSE( ucs_test_flags(value, FLAG1, FLAG3) );
}

UCS_TEST_F(test_math, circular_compare) {
    EXPECT_TRUE(  UCS_CIRCULAR_COMPARE32(0x000000001, <,  0x000000002) );
    EXPECT_TRUE(  UCS_CIRCULAR_COMPARE32(0x000000001, ==, 0x000000001) );
    EXPECT_TRUE(  UCS_CIRCULAR_COMPARE32(0xffffffffU, >,  0xfffffffeU) );
    EXPECT_TRUE(  UCS_CIRCULAR_COMPARE32(0xffffffffU, <,  0x00000000U) );
    EXPECT_TRUE(  UCS_CIRCULAR_COMPARE32(0xffffffffU, <,  0x00000001U) );
    EXPECT_TRUE(  UCS_CIRCULAR_COMPARE32(0xffffffffU, <,  0x00000001U) );
    EXPECT_TRUE(  UCS_CIRCULAR_COMPARE32(0x80000000U, >,  0x7fffffffU) );
    EXPECT_TRUE(  UCS_CIRCULAR_COMPARE32(0xffffffffU, <,  0x7fffffffU) );
}

#define TEST_ATOMIC_ADD(_bitsize) \
    { \
        typedef uint##_bitsize##_t inttype; \
        const inttype var_value = ucs::random_upper<inttype>(); \
        const inttype add_value = ucs::random_upper<inttype>(); \
        inttype var = var_value; \
        ucs_atomic_add##_bitsize(&var, add_value); \
        EXPECT_EQ(static_cast<inttype>(var_value + add_value), var); \
    }

#define TEST_ATOMIC_FADD(_bitsize) \
    { \
        typedef uint##_bitsize##_t inttype; \
        const inttype var_value = ucs::random_upper<inttype>(); \
        const inttype add_value = ucs::random_upper<inttype>(); \
        inttype var = var_value; \
        inttype oldvar = ucs_atomic_fadd##_bitsize(&var, add_value); \
        EXPECT_EQ(static_cast<inttype>(var_value + add_value), var); \
        EXPECT_EQ(var_value, oldvar); \
    }

#define TEST_ATOMIC_SWAP(_bitsize) \
    { \
        typedef uint##_bitsize##_t inttype; \
        const inttype var_value = ucs::random_upper<inttype>(); \
        const inttype swap_value = ucs::random_upper<inttype>(); \
        inttype var = var_value; \
        inttype oldvar = ucs_atomic_swap##_bitsize(&var, swap_value); \
        EXPECT_EQ(var_value, oldvar); \
        EXPECT_EQ(swap_value, var); \
    }

#define TEST_ATOMIC_CSWAP(_bitsize, is_eq) \
    { \
        typedef uint##_bitsize##_t inttype; \
        const inttype var_value = ucs::random_upper<inttype>(); \
        const inttype cmp_value = (is_eq) ? var_value : (var_value + 10); \
        const inttype swap_value = ucs::random_upper<inttype>(); \
        inttype var = var_value; \
        inttype oldvar = ucs_atomic_cswap##_bitsize(&var, cmp_value, swap_value); \
        EXPECT_EQ(var_value, oldvar); \
        if (is_eq) { \
            EXPECT_EQ(swap_value, var); \
        } else { \
            EXPECT_EQ(var_value, var); \
        } \
    }

UCS_TEST_F(test_math, atomic_add) {
    for (unsigned count = 0; count < ATOMIC_COUNT; ++count) {
        TEST_ATOMIC_ADD(8);
        TEST_ATOMIC_ADD(16);
        TEST_ATOMIC_ADD(32);
        TEST_ATOMIC_ADD(64);
    }
}

UCS_TEST_F(test_math, atomic_fadd) {
    for (unsigned count = 0; count < ATOMIC_COUNT; ++count) {
        TEST_ATOMIC_FADD(8);
        TEST_ATOMIC_FADD(16);
        TEST_ATOMIC_FADD(32);
        TEST_ATOMIC_FADD(64);
    }
}

UCS_TEST_F(test_math, atomic_swap) {
    for (unsigned count = 0; count < ATOMIC_COUNT; ++count) {
        TEST_ATOMIC_SWAP(8);
        TEST_ATOMIC_SWAP(16);
        TEST_ATOMIC_SWAP(32);
        TEST_ATOMIC_SWAP(64);
    }
}

UCS_TEST_F(test_math, atomic_cswap_success) {
    for (unsigned count = 0; count < ATOMIC_COUNT; ++count) {
        TEST_ATOMIC_CSWAP(8,  0);
        TEST_ATOMIC_CSWAP(16, 0);
        TEST_ATOMIC_CSWAP(32, 0);
        TEST_ATOMIC_CSWAP(64, 0);
    }
}

UCS_TEST_F(test_math, atomic_cswap_fail) {
    for (unsigned count = 0; count < ATOMIC_COUNT; ++count) {
        TEST_ATOMIC_CSWAP(8,  1);
        TEST_ATOMIC_CSWAP(16, 1);
        TEST_ATOMIC_CSWAP(32, 1);
        TEST_ATOMIC_CSWAP(64, 1);
    }
}

UCS_TEST_F(test_math, for_each_bit) {
    uint64_t gen_mask = 0;
    uint64_t mask;
    int idx;

    mask = ucs_generate_uuid(0);

    ucs_for_each_bit(idx, mask) {
        EXPECT_EQ(gen_mask & UCS_BIT(idx), 0ull);
        gen_mask |= UCS_BIT(idx);
    }

    EXPECT_EQ(mask, gen_mask);

    ucs_for_each_bit(idx, 0) {
        EXPECT_EQ(1, 0); /* should not be here */
    }

    gen_mask = 0;
    ucs_for_each_bit(idx, UCS_BIT(0)) {
        EXPECT_EQ(gen_mask & UCS_BIT(idx), 0ull);
        gen_mask |= UCS_BIT(idx);
    }
    EXPECT_EQ(1ull, gen_mask);

    gen_mask = 0;
    ucs_for_each_bit(idx, UCS_BIT(63)) {
        EXPECT_EQ(gen_mask & UCS_BIT(idx), 0ull);
        gen_mask |= UCS_BIT(idx);
    }
    EXPECT_EQ(UCS_BIT(63), gen_mask);
}

UCS_TEST_F(test_math, for_each_submask) {
    /* Generate mask values to test */
    std::vector<int64_t> masks;
    masks.push_back(0);
    masks.push_back(1);
    masks.push_back(65536);
    for (int i = 0; i < 100; ++i) {
        masks.push_back((ucs::rand() % 65536) + 2);
    }

    for (std::vector<int64_t>::const_iterator iter = masks.begin();
         iter != masks.end(); ++iter) {
        int64_t mask         = *iter;
        int64_t prev_submask = -1;
        unsigned count       = 0;
        int64_t submask;
        ucs_for_each_submask(submask, mask) {
            EXPECT_GT(submask, prev_submask); /* expect strictly monotonic series */
            EXPECT_EQ(0u, submask & ~mask);   /* sub-mask contained in the mask */
            prev_submask = submask;
            ++count;
        }

        /* expect to get all possible values */
        EXPECT_EQ(UCS_BIT(ucs_popcount(mask)), count);
    }
}

UCS_TEST_F(test_math, linear_func) {
    ucs_linear_func_t func[3];
    double x, y[3];

    /* Generate 2 random functions */
    x = ucs::rand() / (double)RAND_MAX;
    for (unsigned i = 0; i < 3; ++i) {
        func[i] = ucs_linear_func_make(ucs::rand() / (double)RAND_MAX,
                                       ucs::rand() / (double)RAND_MAX);
        y[i]    = ucs_linear_func_apply(func[i], x);
    }

    /* Add */
    ucs_linear_func_t sum_func = ucs_linear_func_add(func[0], func[1]);
    double y_sum               = ucs_linear_func_apply(sum_func, x);
    EXPECT_NEAR(y[0] + y[1], y_sum, 1e-6);

    /* Add */
    ucs_linear_func_t sum3_func = ucs_linear_func_add3(func[0], func[1],
                                                       func[2]);
    double y_sum3               = ucs_linear_func_apply(sum3_func, x);
    EXPECT_NEAR(y[0] + y[1] + y[2], y_sum3, 1e-6);


    /* Add in-place */
    ucs_linear_func_t sum_func_inplace = func[0];
    ucs_linear_func_add_inplace(&sum_func_inplace, func[1]);
    double y_sum_inplace = ucs_linear_func_apply(sum_func_inplace, x);
    EXPECT_NEAR(y[0] + y[1], y_sum_inplace, 1e-6);

    /* Subtract */
    ucs_linear_func_t diff_func = ucs_linear_func_sub(func[0], func[1]);
    double y_diff               = ucs_linear_func_apply(diff_func, x);
    EXPECT_NEAR(y[0] - y[1], y_diff, 1e-6);

    /* Intersect */
    double x_intersect = 0;
    ucs_status_t status;
    status = ucs_linear_func_intersect(func[0], func[1], &x_intersect);
    ASSERT_EQ(UCS_OK, status);
    double y_intersect[2];
    for (unsigned i = 0; i < 2; ++i) {
        y_intersect[i] = ucs_linear_func_apply(func[i], x_intersect);
    }
    EXPECT_NEAR(y_intersect[0], y_intersect[1], 1e-6);

    /* Invalid intersect - parallel functions */
    ucs_linear_func_t tmp_func = func[0];
    tmp_func.c = func[0].c + 1.0;
    status     = ucs_linear_func_intersect(func[0], tmp_func, &x_intersect);
    ASSERT_EQ(UCS_ERR_INVALID_PARAM, status);

    /* Invalid intersect - infinite point */
    ucs_linear_func_t tmp_func1 = ucs_linear_func_make(1000, DBL_MIN * 3);
    ucs_linear_func_t tmp_func2 = ucs_linear_func_make(2000, DBL_MIN * 2);
    status                      = ucs_linear_func_intersect(tmp_func1, tmp_func2,
                                                            &x_intersect);
    ASSERT_EQ(UCS_ERR_INVALID_PARAM, status) << x_intersect;

    /* Compare */
    EXPECT_FALSE(ucs_linear_func_is_equal(tmp_func1, tmp_func2, 1e-20));
    EXPECT_TRUE (ucs_linear_func_is_equal(tmp_func1, tmp_func1, 1e-20));
    EXPECT_TRUE (ucs_linear_func_is_equal(tmp_func2, tmp_func2, 1e-20));

    /* Compose */
    ucs_linear_func_t compose_func = ucs_linear_func_compose(func[0], func[1]);
    double y_compose               = ucs_linear_func_apply(compose_func, x);
    double y_compose_exp           = ucs_linear_func_apply(func[0], y[1]);
    EXPECT_NEAR(y_compose_exp, y_compose, 1e-6);

    /* Add value of */
    ucs_linear_func_t added_func = func[0];
    ucs_linear_func_add_value_at(&added_func, func[1], x);
    double y_added_func = ucs_linear_func_apply(added_func, x);
    EXPECT_NEAR(y[0] + y[1], y_added_func, 1e-6);
}

UCS_TEST_F(test_math, double_to_sizet) {
    EXPECT_EQ(SIZE_MAX, ucs_double_to_sizet(1e20, SIZE_MAX));
    EXPECT_EQ(SIZE_MAX, ucs_double_to_sizet(1e30, SIZE_MAX));
    EXPECT_EQ(SIZE_MAX, ucs_double_to_sizet((double)SIZE_MAX, SIZE_MAX));
    EXPECT_EQ(10, ucs_double_to_sizet(10.0, SIZE_MAX));
    EXPECT_EQ(UCS_MBYTE, ucs_double_to_sizet(UCS_MBYTE, SIZE_MAX));
}
