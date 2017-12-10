/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>

#include <ucs/arch/bitops.h>
#include <ucs/arch/atomic.h>
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

UCS_TEST_F(test_math, bitops) {
    EXPECT_EQ((unsigned)0,  ucs_ffs64(0xfffff));
    EXPECT_EQ((unsigned)16, ucs_ffs64(0xf0000));
    EXPECT_EQ((unsigned)1,  ucs_ffs64(0x4002));
    EXPECT_EQ((unsigned)41, ucs_ffs64(1ull<<41));

    EXPECT_EQ((unsigned)0, ucs_ilog2(1));
    EXPECT_EQ((unsigned)2, ucs_ilog2(4));
    EXPECT_EQ((unsigned)2, ucs_ilog2(5));
    EXPECT_EQ((unsigned)2, ucs_ilog2(7));
    EXPECT_EQ((unsigned)14, ucs_ilog2(17000));
    EXPECT_EQ((unsigned)40, ucs_ilog2(1ull<<40));
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

    ucs_for_each_bit (idx, mask) {
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
