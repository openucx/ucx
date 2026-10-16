/*
 * Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_MUSA_UTILS_H_
#define UCT_MUSA_UTILS_H_

#include <ucs/sys/compiler_def.h>

#include <stddef.h>
#include <stdint.h>

BEGIN_C_DECLS

static UCS_F_ALWAYS_INLINE int
uct_musa_range_is_valid(uintptr_t base, size_t alloc_length,
                        uintptr_t address, size_t length)
{
    uintptr_t alloc_end;
    uintptr_t address_end;

    if ((alloc_length == 0) || (length == 0) || (address < base)) {
        return 0;
    }

    if (base > (UINTPTR_MAX - alloc_length)) {
        return 0;
    }
    alloc_end = base + alloc_length;

    if (address > (UINTPTR_MAX - length)) {
        return 0;
    }
    address_end = address + length;

    return address_end <= alloc_end;
}

END_C_DECLS

#endif
