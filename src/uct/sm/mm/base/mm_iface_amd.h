/* Copyright (C) Advanced Micro Devices, Inc. 2023. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifndef _MM_IFACE_AMD_H
#define _MM_IFACE_AMD_H
#include <stdbool.h>
#include <stdint.h>

#ifdef ENABLE_AMD_BUFFER_TRANSFER
static UCS_F_ALWAYS_INLINE
void uct_amd_optimized_fifo_send(uint64_t *dst, uint64_t header, const void *payload,
                                size_t length)
{
    if (length > (UCS_SYS_CACHE_LINE_SIZE - (sizeof(uct_mm_fifo_element_t) + 8))) {
        ucs_nt_write_prefetch((char *)dst + (UCS_SYS_CACHE_LINE_SIZE - sizeof(uct_mm_fifo_element_t)));
    }
    *dst++ = header;
    ucs_x86_amd_memcpy(dst, payload, length);
}
#endif
#endif
