/**
* Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ARCH_RV64_BITOPS_H_
#define UCS_ARCH_RV64_BITOPS_H_

#include <ucs/sys/compiler_def.h>
#include <stdint.h>

static UCS_F_ALWAYS_INLINE unsigned __ucs_ilog2_u32(uint32_t n)
{
    return 31 - __builtin_clz(n);
}

static UCS_F_ALWAYS_INLINE unsigned __ucs_ilog2_u64(uint64_t n)
{
    return 63 - __builtin_clzll(n);
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
