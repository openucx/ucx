/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "dt_contig.h"

#include <ucs/debug/profile.h>
#include <string.h>


size_t ucp_memcpy_pack(void *dest, void *arg)
{
    ucp_memcpy_pack_context_t *ctx = arg;
    size_t length = ctx->length;
    UCS_PROFILE_CALL(memcpy, dest, ctx->src, length);
    return length;
}
