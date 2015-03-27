/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/


#ifndef UCS_MATH_H
#define UCS_MATH_H

#include <ucs/sys/compiler.h>

#include <stdint.h>
#include <math.h>


#define UCS_KBYTE    (1ull << 10)
#define UCS_MBYTE    (1ull << 20)
#define UCS_GBYTE    (1ull << 30)

#define ucs_min(a, b) \
({ \
    typeof(a) _a = (a);  \
    typeof(b) _b = (b);  \
    _a < _b ? _a : _b;   \
})

#define ucs_max(a, b) \
({ \
    typeof(a) _a = (a);  \
    typeof(b) _b = (b);  \
    _a > _b ? _a : _b;   \
})

#define ucs_is_pow2(_n) \
    ( ((_n) > 0) && !((_n) & ((_n) - 1)) )

#define ucs_padding(_n, _alignment) \
    ( ((_alignment) - (_n) % (_alignment)) % (_alignment) )

#define ucs_align_up(_n, _alignment) \
    ( (_n) + ucs_padding(_n, _alignment) )

#define ucs_align_up_pow2(_n, _alignment) \
    ( ((_n) + (_alignment) - 1) & ~((_alignment) - 1) )

#define ucs_align_down(_n, _alignment) \
    ( (_n) - ((_n) % (_alignment)) )

#define ucs_roundup_pow2(n) \
    ({ \
        typeof(n) pow2; \
        ucs_assert((n) >= 1); \
        for (pow2 = 1; pow2 < (n); pow2 <<= 1); \
        pow2; \
    })

/* The i-th bit */
#define UCS_BIT(i)               (1ull << (i))

/* Mask of bits 0..i-1 */
#define UCS_MASK(i)              (UCS_BIT(i) - 1)

#define UCS_MASK_SAFE(i) \
    (((i) >= 64) ? ((uint64_t)(-1)) : UCS_MASK(i))

#define ucs_div_round_up(_n, _d) \
    (((_n) + (_d) - 1) / (_d))

static inline double ucs_log2(double x)
{
    return log(x) / log(2.0);
}

#define ucs_ilog2(n)                   \
(                                      \
    __builtin_constant_p(n) ? (        \
             (n) < 1 ? 0 :             \
             (n) & (1ULL << 63) ? 63 : \
             (n) & (1ULL << 62) ? 62 : \
             (n) & (1ULL << 61) ? 61 : \
             (n) & (1ULL << 60) ? 60 : \
             (n) & (1ULL << 59) ? 59 : \
             (n) & (1ULL << 58) ? 58 : \
             (n) & (1ULL << 57) ? 57 : \
             (n) & (1ULL << 56) ? 56 : \
             (n) & (1ULL << 55) ? 55 : \
             (n) & (1ULL << 54) ? 54 : \
             (n) & (1ULL << 53) ? 53 : \
             (n) & (1ULL << 52) ? 52 : \
             (n) & (1ULL << 51) ? 51 : \
             (n) & (1ULL << 50) ? 50 : \
             (n) & (1ULL << 49) ? 49 : \
             (n) & (1ULL << 48) ? 48 : \
             (n) & (1ULL << 47) ? 47 : \
             (n) & (1ULL << 46) ? 46 : \
             (n) & (1ULL << 45) ? 45 : \
             (n) & (1ULL << 44) ? 44 : \
             (n) & (1ULL << 43) ? 43 : \
             (n) & (1ULL << 42) ? 42 : \
             (n) & (1ULL << 41) ? 41 : \
             (n) & (1ULL << 40) ? 40 : \
             (n) & (1ULL << 39) ? 39 : \
             (n) & (1ULL << 38) ? 38 : \
             (n) & (1ULL << 37) ? 37 : \
             (n) & (1ULL << 36) ? 36 : \
             (n) & (1ULL << 35) ? 35 : \
             (n) & (1ULL << 34) ? 34 : \
             (n) & (1ULL << 33) ? 33 : \
             (n) & (1ULL << 32) ? 32 : \
             (n) & (1ULL << 31) ? 31 : \
             (n) & (1ULL << 30) ? 30 : \
             (n) & (1ULL << 29) ? 29 : \
             (n) & (1ULL << 28) ? 28 : \
             (n) & (1ULL << 27) ? 27 : \
             (n) & (1ULL << 26) ? 26 : \
             (n) & (1ULL << 25) ? 25 : \
             (n) & (1ULL << 24) ? 24 : \
             (n) & (1ULL << 23) ? 23 : \
             (n) & (1ULL << 22) ? 22 : \
             (n) & (1ULL << 21) ? 21 : \
             (n) & (1ULL << 20) ? 20 : \
             (n) & (1ULL << 19) ? 19 : \
             (n) & (1ULL << 18) ? 18 : \
             (n) & (1ULL << 17) ? 17 : \
             (n) & (1ULL << 16) ? 16 : \
             (n) & (1ULL << 15) ? 15 : \
             (n) & (1ULL << 14) ? 14 : \
             (n) & (1ULL << 13) ? 13 : \
             (n) & (1ULL << 12) ? 12 : \
             (n) & (1ULL << 11) ? 11 : \
             (n) & (1ULL << 10) ? 10 : \
             (n) & (1ULL <<  9) ?  9 : \
             (n) & (1ULL <<  8) ?  8 : \
             (n) & (1ULL <<  7) ?  7 : \
             (n) & (1ULL <<  6) ?  6 : \
             (n) & (1ULL <<  5) ?  5 : \
             (n) & (1ULL <<  4) ?  4 : \
             (n) & (1ULL <<  3) ?  3 : \
             (n) & (1ULL <<  2) ?  2 : \
             (n) & (1ULL <<  1) ?  1 : \
             (n) & (1ULL <<  0) ?  0 : \
             0                         \
                                ) :    \
    (sizeof(n) <= 4) ?                 \
    __ucs_ilog2_u32((uint32_t)(n)) :   \
    __ucs_ilog2_u64((uint64_t)(n))     \
)


/**
 * Convert flags without a branch
 * @return 'newflag' oldflag is set in 'value', otherwise - 0
 */
#define ucs_convert_flag(value, oldflag, newflag) \
    ({ \
        UCS_STATIC_ASSERT(ucs_is_constant(oldflag)); \
        UCS_STATIC_ASSERT(ucs_is_constant(newflag)); \
        UCS_STATIC_ASSERT(ucs_is_pow2(oldflag)); \
        UCS_STATIC_ASSERT(ucs_is_pow2(newflag)); \
        (((value) & (oldflag)) ? (newflag) : 0); \
    })


/**
 * Test if a value is one of a specified list of values, assuming all possible
 * values are powers of 2.
 */
#define __ucs_test_flags(__value, __f1, __f2, __f3, __f4, __f5, __f6, __f7, __f8, __f9, ...) \
    (__value & ((__f1) | (__f2) | (__f3) | (__f4) | (__f5) | (__f6) | (__f7) | (__f8) | (__f9)))
#define ucs_test_flags(__value, ...) \
    __ucs_test_flags((__value), __VA_ARGS__, 0, 0, 0, 0, 0, 0, 0, 0, 0)

/*
 * Check if all given flags are on
 */
#define ucs_test_all_flags(__value, __mask) \
    ( ((__value) & (__mask)) == (__mask) )

/* Returns the number of 1-bits in x */
#define ucs_count_one_bits(x)      __builtin_popcount(x)

/* Returns the number of trailing 0-bits in x, starting at the least
 * significant bit position.  If x is 0, the result is undefined.
 */
#define ucs_count_zero_bits(x)     __builtin_ctz(x)

/**
 * Compare unsigned numbers which can wrap-around, assuming the wrap-around
 * distance can be at most the maximal value of the signed type.
 *
 * @param __a            First number
 * @param __op           Operator (e.g >=)
 * @param __b            Second number
 * @param _signed_type   Signed type of __a/__b (e.g int_32_t)
 *
 * @return value of the expression "__a __op __b".
 */
#define UCS_CIRCULAR_COMPARE(__a, __op, __b, __signed_type)  \
    ((__signed_type)((__a) - (__b)) __op 0)

#define UCS_CIRCULAR_COMPARE8(__a, __op, __b)  UCS_CIRCULAR_COMPARE(__a, __op, __b, int8_t)
#define UCS_CIRCULAR_COMPARE16(__a, __op, __b)  UCS_CIRCULAR_COMPARE(__a, __op, __b, int16_t)
#define UCS_CIRCULAR_COMPARE32(__a, __op, __b)  UCS_CIRCULAR_COMPARE(__a, __op, __b, int32_t)
#define UCS_CIRCULAR_COMPARE64(__a, __op, __b)  UCS_CIRCULAR_COMPARE(__a, __op, __b, int64_t)


/**
 * Calculate CRC32 of a buffer.
 */
uint32_t ucs_calc_crc32(uint32_t crc, const void *buf, size_t size);


/*
 * Generate a large prime number
 */
uint64_t ucs_get_prime(unsigned index);


#endif /* MACROS_H_ */
