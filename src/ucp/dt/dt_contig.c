/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "dt_contig.h"

#include <ucs/profile/profile.h>
#include <string.h>


size_t ucp_memcpy_pack_cb(void *dest, void *arg)
{
    ucp_memcpy_pack_context_t *ctx = arg;
    size_t length = ctx->length;
    UCS_PROFILE_CALL(memcpy, dest, ctx->src, length);
    return length;
}

void ucp_dt_contig_pack(ucp_worker_h worker, void *dest, const void *src,
                        size_t length, ucs_memory_type_t mem_type)
{
    if (UCP_MEM_IS_ACCESSIBLE_FROM_CPU(mem_type)) {
        ucp_memcpy_pack_unpack(worker, dest, src, length, mem_type);
    } else {
        ucp_mem_type_pack(worker, dest, src, length, mem_type);
    }
}

void ucp_dt_contig_unpack(ucp_worker_h worker, void *dest, const void *src,
                          size_t length, ucs_memory_type_t mem_type)
{
    if (UCP_MEM_IS_ACCESSIBLE_FROM_CPU(mem_type)) {
        ucp_memcpy_pack_unpack(worker, dest, src, length, mem_type);
    } else {
        ucp_mem_type_unpack(worker, dest, src, length, mem_type);
    }
}
