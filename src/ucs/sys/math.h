/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#ifndef UCS_MATH_H
#define UCS_MATH_H

#include "compiler_def.h"

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <ucs/arch/bitops.h>
#include <ucs/type/status.h>

BEGIN_C_DECLS

/** @file math.h */

#define UCS_KBYTE    (1ull << 10)
#define UCS_MBYTE    (1ull << 20)
#define UCS_GBYTE    (1ull << 30)
#define UCS_TBYTE    (1ull << 40)
#define UCS_PBYTE    (1ull << 50)

#define ucs_min(_a, _b) \
({ \
    ucs_typeof(_a) _min_a = (_a); \
    ucs_typeof(_b) _min_b = (_b); \
    (_min_a < _min_b) ? _min_a : _min_b; \
})

#define ucs_max(_a, _b) \
({ \
    ucs_typeof(_a) _max_a = (_a); \
    ucs_typeof(_b) _max_b = (_b); \
    (_max_a > _max_b) ? _max_a : _max_b; \
})

#define ucs_is_pow2_or_zero(_n) \
    !((_n) & ((_n) - 1))

#define ucs_is_pow2(_n) \
    (((_n) > 0) && ucs_is_pow2_or_zero(_n))

#define ucs_signum(_n) \
    (((_n) > (ucs_typeof(_n))0) - ((_n) < (ucs_typeof(_n))0))

#define ucs_div_round_up(_n, _d) \
    (((_n) + (_d) - 1) / (_d))

static inline double ucs_log2(double x)
{
    return log(x) / log(2.0);
}

static UCS_F_ALWAYS_INLINE size_t ucs_double_to_sizet(double value, size_t max)
{
    double round_value = value + 0.5;
    return (round_value < (double)max) ? ((size_t)round_value) : max;
}


/**
 * Convert flags without a branch
 * @return '_newflag' if '_oldflag' is set in '_value', otherwise - 0
 */
#define ucs_convert_flag(_value, _oldflag, _newflag) \
    ({ \
        UCS_STATIC_ASSERT(ucs_is_constant(_oldflag)); \
        UCS_STATIC_ASSERT(ucs_is_constant(_newflag)); \
        UCS_STATIC_ASSERT(ucs_is_pow2(_oldflag)); \
        UCS_STATIC_ASSERT(ucs_is_pow2(_newflag)); \
        (((_value) & (_oldflag)) ? (_newflag) : 0); \
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

/**
 * Compare unsigned numbers which can wrap-around, assuming the wrap-around
 * distance can be at most the maximal value of the signed type.
 *
 * @param __a            First number
 * @param __op           Operator (e.g >=)
 * @param __b            Second number
 * @param _signed_type   Signed type of __a/__b (e.g int32_t)
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
 * Enumerate on all bit values in the bitmap '_map'
 */
#define ucs_for_each_bit(_index, _map) \
    for ((_index) = ucs_ffs64_safe(_map); (_index) < 64; \
         (_index) = ucs_ffs64_safe((uint64_t)(_map) & (-2ull << (uint64_t)(_index))))


/**
 * Generate all sub-masks of the given mask, from 0 to _mask inclusive.
 *
 * @param _submask   Variable to iterate over the sub-masks
 * @param _mask      Generate sub-masks of this value
 */
#define ucs_for_each_submask(_submask, _mask) \
    for (/* start with 0 */ \
         (_submask) = 0; \
         /* end when reaching _mask + 1 */ \
         (_submask) <= (_mask); \
         /* Increment _submask by 1. If it became larger than _mask, do nothing \
          * here, and next condition check will exit the loop. Otherwise, add \
          * ~mask to fast-forward the carry (from ++ operation) to the next \
          * valid bit in _mask, and then do "& _mask" to remove any bits which \
          * are not in the mask. \
          */ \
         (_submask)++, \
         ((_submask) <= (_mask)) ? \
                 ((_submask) = ((_submask )+ ~(_mask)) & (_mask)) : 0)


/*
 * Generate a large prime number
 */
uint64_t ucs_get_prime(unsigned index);


/*
 * Generate a random seed
 */
void ucs_rand_seed_init();


/*
 * Generate a random number in the range 0..RAND_MAX
 */
int ucs_rand();


/*
 * Generate a random number in the given range (inclusive)
 *
 * @param [in]  range_min       Beginning of the range
 * @param [in]  range_max       End of the range
 * @param [out] rand_val        The generated random number
 *
 * @return UCS_OK on success or an error status on failure.
 */
ucs_status_t ucs_rand_range(int range_min, int range_max, int *rand_val);

END_C_DECLS

#endif /* MACROS_H_ */
