/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_X86_64_BITOPS_H_
#define UCS_X86_64_BITOPS_H_

#include <ucs/sys/compiler_def.h>
#include <stdint.h>


static UCS_F_ALWAYS_INLINE unsigned ucs_ffs32(uint32_t n)
{
    asm("bsfl %1,%0"
        : "=r" (n)
        : "r" (n));
    return n;
}

static UCS_F_ALWAYS_INLINE unsigned ucs_ffs64(uint64_t n)
{
    asm("bsfq %1,%0"
        : "=r" (n)
        : "r" (n));
    return n;
}

static UCS_F_ALWAYS_INLINE unsigned __ucs_ilog2_u32(uint32_t n)
{
    asm("bsrl %1,%0"
        : "=r" (n)
        : "r" (n));
    return n;
}

static UCS_F_ALWAYS_INLINE unsigned __ucs_ilog2_u64(uint64_t n)
{
    asm("bsrq %1,%0"
        : "=r" (n)
        : "r" (n));
    return n;
}

#endif
