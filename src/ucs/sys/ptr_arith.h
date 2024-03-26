/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_PTR_ARITH_H
#define UCS_PTR_ARITH_H

#include "compiler_def.h"

#include <ucs/debug/assert.h>

#include <stddef.h>
#include <stdint.h>

BEGIN_C_DECLS


#define ucs_padding(_n, _alignment) \
    (((_alignment) - (_n) % (_alignment)) % (_alignment))


#define ucs_align_down(_n, _alignment) ((_n) - ((_n) % (_alignment)))


#define ucs_align_up(_n, _alignment) ((_n) + ucs_padding(_n, _alignment))


#define ucs_align_down_pow2(_n, _alignment) ((_n) & ~((_alignment)-1))


#define ucs_align_up_pow2(_n, _alignment) \
    ucs_align_down_pow2((_n) + (_alignment)-1, _alignment)


#define ucs_align_down_pow2_ptr(_ptr, _alignment) \
    ((ucs_typeof(_ptr))ucs_align_down_pow2((uintptr_t)(_ptr), (_alignment)))


#define ucs_align_up_pow2_ptr(_ptr, _alignment) \
    ((ucs_typeof(_ptr))ucs_align_up_pow2((uintptr_t)(_ptr), (_alignment)))


#define ucs_roundup_pow2(_n) \
    ({ \
        ucs_typeof(_n) pow2; \
        ucs_assert((_n) >= 1); \
        for (pow2 = 1; pow2 < (_n); pow2 <<= 1) \
            ; \
        pow2; \
    })


#define ucs_rounddown_pow2(_n) (ucs_roundup_pow2(_n + 1) / 2)


#define ucs_roundup_pow2_or0(_n) (((_n) == 0) ? 0 : ucs_roundup_pow2(_n))


/* Return values: 0 - aligned, non-0 - unaligned */
#define ucs_check_if_align_pow2(_n, _p) ((_n) & ((_p)-1))


/* Return values: off-set from the alignment */
#define ucs_padding_pow2(_n, _p) ucs_check_if_align_pow2(_n, _p)


static UCS_F_ALWAYS_INLINE void
ucs_align_ptr_range(void **address_p, size_t *length_p, size_t alignment)
{
    void *start, *end;

    start = ucs_align_down_pow2_ptr(*address_p, alignment);
    end   = ucs_align_up_pow2_ptr(UCS_PTR_BYTE_OFFSET(*address_p, *length_p),
                                  alignment);

    *address_p = start;
    *length_p  = UCS_PTR_BYTE_DIFF(start, end);
}


static UCS_F_ALWAYS_INLINE void
ucs_ptr_check_align(void *address, size_t length, size_t alignment)
{
    ucs_assertv(ucs_padding((uintptr_t)address, alignment) == 0,
                "address=%p align=%zu", address, alignment);
    ucs_assertv(ucs_padding(length, alignment) == 0, "length=%zu align=%zu",
                length, alignment);
}


END_C_DECLS

#endif
