/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) Huawei Technologies Co., Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ARCH_BITOPS_H
#define UCS_ARCH_BITOPS_H

#include <ucs/sys/compiler_def.h>
#include <stdint.h>
#include <string.h>

BEGIN_C_DECLS

#if defined(__x86_64__)
#  include "x86_64/bitops.h"
#elif defined(__powerpc64__)
#  include "ppc64/bitops.h"
#elif defined(__aarch64__)
#  include "aarch64/bitops.h"
#else
#  error "Unsupported architecture"
#endif


#define ucs_ilog2(_n)                   \
(                                       \
    __builtin_constant_p(_n) ? (        \
             (_n) < 1 ? 0 :             \
             (_n) & (1ULL << 63) ? 63 : \
             (_n) & (1ULL << 62) ? 62 : \
             (_n) & (1ULL << 61) ? 61 : \
             (_n) & (1ULL << 60) ? 60 : \
             (_n) & (1ULL << 59) ? 59 : \
             (_n) & (1ULL << 58) ? 58 : \
             (_n) & (1ULL << 57) ? 57 : \
             (_n) & (1ULL << 56) ? 56 : \
             (_n) & (1ULL << 55) ? 55 : \
             (_n) & (1ULL << 54) ? 54 : \
             (_n) & (1ULL << 53) ? 53 : \
             (_n) & (1ULL << 52) ? 52 : \
             (_n) & (1ULL << 51) ? 51 : \
             (_n) & (1ULL << 50) ? 50 : \
             (_n) & (1ULL << 49) ? 49 : \
             (_n) & (1ULL << 48) ? 48 : \
             (_n) & (1ULL << 47) ? 47 : \
             (_n) & (1ULL << 46) ? 46 : \
             (_n) & (1ULL << 45) ? 45 : \
             (_n) & (1ULL << 44) ? 44 : \
             (_n) & (1ULL << 43) ? 43 : \
             (_n) & (1ULL << 42) ? 42 : \
             (_n) & (1ULL << 41) ? 41 : \
             (_n) & (1ULL << 40) ? 40 : \
             (_n) & (1ULL << 39) ? 39 : \
             (_n) & (1ULL << 38) ? 38 : \
             (_n) & (1ULL << 37) ? 37 : \
             (_n) & (1ULL << 36) ? 36 : \
             (_n) & (1ULL << 35) ? 35 : \
             (_n) & (1ULL << 34) ? 34 : \
             (_n) & (1ULL << 33) ? 33 : \
             (_n) & (1ULL << 32) ? 32 : \
             (_n) & (1ULL << 31) ? 31 : \
             (_n) & (1ULL << 30) ? 30 : \
             (_n) & (1ULL << 29) ? 29 : \
             (_n) & (1ULL << 28) ? 28 : \
             (_n) & (1ULL << 27) ? 27 : \
             (_n) & (1ULL << 26) ? 26 : \
             (_n) & (1ULL << 25) ? 25 : \
             (_n) & (1ULL << 24) ? 24 : \
             (_n) & (1ULL << 23) ? 23 : \
             (_n) & (1ULL << 22) ? 22 : \
             (_n) & (1ULL << 21) ? 21 : \
             (_n) & (1ULL << 20) ? 20 : \
             (_n) & (1ULL << 19) ? 19 : \
             (_n) & (1ULL << 18) ? 18 : \
             (_n) & (1ULL << 17) ? 17 : \
             (_n) & (1ULL << 16) ? 16 : \
             (_n) & (1ULL << 15) ? 15 : \
             (_n) & (1ULL << 14) ? 14 : \
             (_n) & (1ULL << 13) ? 13 : \
             (_n) & (1ULL << 12) ? 12 : \
             (_n) & (1ULL << 11) ? 11 : \
             (_n) & (1ULL << 10) ? 10 : \
             (_n) & (1ULL <<  9) ?  9 : \
             (_n) & (1ULL <<  8) ?  8 : \
             (_n) & (1ULL <<  7) ?  7 : \
             (_n) & (1ULL <<  6) ?  6 : \
             (_n) & (1ULL <<  5) ?  5 : \
             (_n) & (1ULL <<  4) ?  4 : \
             (_n) & (1ULL <<  3) ?  3 : \
             (_n) & (1ULL <<  2) ?  2 : \
             (_n) & (1ULL <<  1) ?  1 : \
             (_n) & (1ULL <<  0) ?  0 : \
             0                          \
                                 ) :    \
    (sizeof(_n) <= 4) ?                 \
    __ucs_ilog2_u32((uint32_t)(_n)) :   \
    __ucs_ilog2_u64((uint64_t)(_n))     \
)

#define ucs_ilog2_or0(_n) \
    ( ((_n) == 0) ? 0 : ucs_ilog2(_n) )

/* Returns the number of 1-bits in x */
#define ucs_popcount(_n) \
    ((sizeof(_n) <= 4) ? __builtin_popcount((uint32_t)(_n)) : \
                         __builtin_popcountl(_n))

/* On some arch ffs64(0) returns 0, on other -1, let's unify this */
#define ucs_ffs64_safe(_val) ((_val) ? ucs_ffs64(_val) : 64)

/* Returns the number of trailing 0-bits in x, starting at the least
 * significant bit position.  If x is 0, the result is undefined.
 */
#define ucs_count_trailing_zero_bits(_n) \
    ((sizeof(_n) <= 4) ? __builtin_ctz((uint32_t)(_n)) : __builtin_ctzl(_n))

/* Returns the number of leading 0-bits in _n.
 * If _n is 0, the result is undefined
 */
#define ucs_count_leading_zero_bits(_n) \
    ((sizeof(_n) <= 4) ? __builtin_clz((uint32_t)(_n)) : __builtin_clzl(_n))

/* Returns the number of 1-bits by _idx mask */
#define ucs_bitmap2idx(_map, _idx) \
    ucs_popcount((_map) & (UCS_MASK(_idx)))


/**
 * Count how many bits at the end of the buffer are equal to zero.
 *
 * @param [in] ptr         Pointer to the buffer.
 * @param [in] bit_length  Total Buffer length (in bits).
 *
 * @return The number of trailing zero bits.
 */
static inline unsigned
ucs_count_ptr_trailing_zero_bits(const void *ptr, uint64_t bit_length)
{
    uint64_t idx = bit_length;
    uint8_t tmp  = 0;

    if (idx == 0) {
        return 0;
    }

    /* Start from the end of the given buffer, with fractions of a bytes */
    if ((idx % 8) != 0) {
        tmp  = *(uint8_t*)UCS_PTR_BYTE_OFFSET(ptr, idx / 8);
        tmp &= ~UCS_MASK(8 - (idx % 8));
        if (idx < 8) {
            tmp |= UCS_BIT(idx % 8);
        }
        if ((idx < 8) || (tmp != 0)) {
            return __builtin_ctz(tmp | (((uint32_t)-1) << 8));
        }
    }

    /* from now on - offsets are in bytes */
    idx = (idx / 8) - 1;
    while (((tmp = *(uint8_t*)UCS_PTR_BYTE_OFFSET(ptr, idx)) == 0) &&
           (idx > 0)) {
        idx--;
    }

    return bit_length - ((idx + 1) * 8) +
           __builtin_ctz(tmp | (((uint32_t)-1) << 8));
}

/**
 * Check if two buffers are equal (for the given amount of bits).
 *
 * @param [in] ptr1        Pointer to the first buffer.
 * @param [in] ptr2        Pointer to the second buffer.
 * @param [in] bit_length  Buffer length (in bits).
 *
 * @return Whether the buffers are equal.
 */
static inline int
ucs_bitwise_is_equal(const void *ptr1, const void *ptr2, uint64_t bit_length)
{
    size_t length          = bit_length / 8;
    unsigned remainder_val = bit_length % 8;

    if (memcmp(ptr1, ptr2, length) != 0) {
        return 0;
    }

    if (remainder_val == 0) {
        return 1;
    }

    /* Compare up to 7 last bits */
    return ((*((uint8_t*)ptr1 + length) & ~UCS_MASK(8 - remainder_val)) ==
            (*((uint8_t*)ptr2 + length) & ~UCS_MASK(8 - remainder_val)));
}

END_C_DECLS

#endif
