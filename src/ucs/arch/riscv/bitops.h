/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_RISCV64_BITOPS_H_
#define UCS_RISCV64_BITOPS_H_

#include <ucs/sys/compiler_def.h>
#include <stdint.h>

static UCS_F_ALWAYS_INLINE unsigned ucs_ffs32(uint32_t n)
{
    /* bsfl */
    uint32_t result;
    asm("ctz %1,%0    \n\t\
        add 1,%1,%1"
        : "=r" (result)
        : "r" (n));
    return result;
}

static UCS_F_ALWAYS_INLINE unsigned ucs_ffs64(uint64_t n)
{
    /* bsfq */
    uint64_t result;
    asm("ctz %1,%0    \n\t\
        add 1,%1,%1"
        : "=r" (result)
        : "r" (n));
    return result;
}

static UCS_F_ALWAYS_INLINE unsigned __ucs_ilog2_u32(uint32_t n)
{
    /* bsrl */
    uint32_t result;
    asm("clz %1,%0"
        : "=r" (result)
        : "r" (n));
    return result;
}

static UCS_F_ALWAYS_INLINE unsigned __ucs_ilog2_u64(uint64_t n)
{
    /* bsrq */
    uint64_t result;
    asm("clz %1,%0"
        : "=r" (result)
        : "r" (n));
    return result;
}

#endif
