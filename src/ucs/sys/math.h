/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
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

BEGIN_C_DECLS

#define UCS_KBYTE    (1ull << 10)
#define UCS_MBYTE    (1ull << 20)
#define UCS_GBYTE    (1ull << 30)
#define UCS_TBYTE    (1ull << 40)

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

#define ucs_is_pow2_or_zero(_n) \
    !((_n) & ((_n) - 1))

#define ucs_is_pow2(_n) \
    (((_n) > 0) && ucs_is_pow2_or_zero(_n))

#define ucs_padding(_n, _alignment) \
    ( ((_alignment) - (_n) % (_alignment)) % (_alignment) )

#define ucs_align_down(_n, _alignment) \
    ( (_n) - ((_n) % (_alignment)) )

#define ucs_align_up(_n, _alignment) \
    ( (_n) + ucs_padding(_n, _alignment) )

#define ucs_align_down_pow2(_n, _alignment) \
    ( (_n) & ~((_alignment) - 1) )

#define ucs_align_up_pow2(_n, _alignment) \
    ucs_align_down_pow2((_n) + (_alignment) - 1, _alignment)

#define ucs_align_down_pow2_ptr(_ptr, _alignment) \
    ((typeof(_ptr))ucs_align_down_pow2((uintptr_t)(_ptr), (_alignment)))

#define ucs_align_up_pow2_ptr(_ptr, _alignment) \
    ((typeof(_ptr))ucs_align_up_pow2((uintptr_t)(_ptr), (_alignment)))

#define ucs_roundup_pow2(n) \
    ({ \
        typeof(n) pow2; \
        ucs_assert((n) >= 1); \
        for (pow2 = 1; pow2 < (n); pow2 <<= 1); \
        pow2; \
    })

/* Return values: 0 - aligned, non-0 - unaligned */
#define ucs_check_if_align_pow2(n, p) ((n) & (p-1))

/* Return values: off-set from the alignment */
#define ucs_padding_pow2(n, p) ucs_check_if_align_pow2(n, p)

#define UCS_MASK_SAFE(i) \
    (((i) >= 64) ? ((uint64_t)(-1)) : UCS_MASK(i))

#define ucs_div_round_up(_n, _d) \
    (((_n) + (_d) - 1) / (_d))

static inline double ucs_log2(double x)
{
    return log(x) / log(2.0);
}

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

/* on some arch ffs64(0) returns 0, on other -1, let's unify this */
#define ucs_ffs64_safe(_val) ((_val) ? ucs_ffs64(_val) : 64)

#define ucs_for_each_bit(_index, _map)                   \
    for ((_index) = ucs_ffs64_safe(_map); (_index) < 64; \
         (_index) = ucs_ffs64_safe((uint64_t)(_map) & (-2ull << (uint64_t)(_index))))


/**
 * Calculate CRC32 of a buffer.
 */
uint32_t ucs_calc_crc32(uint32_t crc, const void *buf, size_t size);


/*
 * Generate a large prime number
 */
uint64_t ucs_get_prime(unsigned index);

END_C_DECLS

#endif /* MACROS_H_ */
