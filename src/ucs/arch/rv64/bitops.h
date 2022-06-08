/**
* Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_RV64_BITOPS_H_
#define UCS_RV64_BITOPS_H_

#include <ucs/sys/compiler_def.h>
#include <stdint.h>

static UCS_F_ALWAYS_INLINE unsigned __ucs_ilog2_u32(uint32_t n)
{
    return __builtin_ctz(n);
}

static UCS_F_ALWAYS_INLINE unsigned __ucs_ilog2_u64(uint64_t n)
{
    union input {
        uint64_t value;
        struct pvalue {
            uint32_t rhs;
            uint32_t lhs;
        } pvalue;
    } n_sto = { .value = n };

    const int lhs = __builtin_ctz(n_sto.pvalue.lhs);
    const int rhs = __builtin_ctz(n_sto.pvalue.rhs);
    
    int val = rhs;
    if(rhs < 0) {
        val = 32 + lhs;
    }
    return val;
}

static UCS_F_ALWAYS_INLINE unsigned ucs_ffs32(uint32_t n)
{
    return __ucs_ilog2_u32(n);
}

static UCS_F_ALWAYS_INLINE unsigned ucs_ffs64(uint64_t n)
{
    return __ucs_ilog2_u64(n);
}

#endif
