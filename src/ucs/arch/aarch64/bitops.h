/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_AARCH64_BITOPS_H_
#define UCS_AARCH64_BITOPS_H_

#include <ucs/sys/compiler_def.h>
#include <sys/types.h>
#include <stdint.h>


static UCS_F_ALWAYS_INLINE unsigned __ucs_ilog2_u32(uint32_t n)
{
    int bit;
    asm ("clz %w0, %w1" : "=r" (bit) : "r" (n));
    return 31 - bit;
}

static UCS_F_ALWAYS_INLINE unsigned __ucs_ilog2_u64(uint64_t n)
{
    int64_t bit;
    asm ("clz %0, %1" : "=r" (bit) : "r" (n));
    return 63 - bit;
}

static UCS_F_ALWAYS_INLINE unsigned ucs_ffs32(uint32_t n)
{
    return __ucs_ilog2_u32(n & -n);
}

static UCS_F_ALWAYS_INLINE unsigned ucs_ffs64(uint64_t n)
{
    return __ucs_ilog2_u64(n & -n);
}

#endif
