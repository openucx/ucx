/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_AARCH64_BITOPS_H_
#define UCS_AARCH64_BITOPS_H_

#include <ucs/debug/log.h>


static inline unsigned ucs_ffs64(uint64_t n)
{
    ucs_fatal("unimplemented");
    return 64;
}

static inline unsigned __ucs_ilog2_u32(uint32_t n)
{
    ucs_fatal("unimplemented");
    return 31;
}

static inline unsigned __ucs_ilog2_u64(uint64_t n)
{
    ucs_fatal("unimplemented");
    return 63;
}

#endif
