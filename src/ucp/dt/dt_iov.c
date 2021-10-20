/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "dt_iov.h"

#include <ucs/arch/cpu.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/math.h>
#include <ucs/profile/profile.h>
#include <ucp/core/ucp_mm.h>
#include <ucp/dt/dt.h>

#include <string.h>
#include <unistd.h>


void ucp_dt_iov_gather(ucp_worker_h worker, void *dest, const ucp_dt_iov_t *iov,
                       size_t length, size_t *iov_offset, size_t *iovcnt_offset,
                       ucs_memory_type_t mem_type)
{
    ucp_dt_pack_func_t pack = UCP_MEM_IS_ACCESSIBLE_FROM_CPU(mem_type) ?
                                      ucp_memcpy_pack_unpack :
                                      ucp_mem_type_pack;
    size_t length_it = 0;
    size_t item_len, item_reminder, item_len_to_copy;


    while (length_it < length) {
        item_len      = iov[*iovcnt_offset].length;
        item_reminder = item_len - *iov_offset;

        item_len_to_copy = item_reminder -
                           ucs_max((ssize_t)((length_it + item_reminder) - length), 0);

        pack(worker, UCS_PTR_BYTE_OFFSET(dest, length_it),
             UCS_PTR_BYTE_OFFSET(iov[*iovcnt_offset].buffer, *iov_offset),
             item_len_to_copy, mem_type);
        length_it += item_len_to_copy;

        ucs_assert(length_it <= length);
        if (length_it < length) {
            *iov_offset = 0;
            ++(*iovcnt_offset);
        } else {
            *iov_offset += item_len_to_copy;
        }
    }
}


size_t ucp_dt_iov_scatter(ucp_worker_h worker, const ucp_dt_iov_t *iov,
                          size_t iovcnt, const void *src, size_t length,
                          size_t *iov_offset, size_t *iovcnt_offset,
                          ucs_memory_type_t mem_type)
{
    ucp_dt_unpack_func_t unpack = UCP_MEM_IS_ACCESSIBLE_FROM_CPU(mem_type) ?
                                          ucp_memcpy_pack_unpack :
                                          ucp_mem_type_unpack;
    size_t length_it = 0;
    size_t item_len, item_len_to_copy;

    while ((length_it < length) && (*iovcnt_offset < iovcnt)) {
        item_len         = iov[*iovcnt_offset].length;
        item_len_to_copy = ucs_min(ucs_max((ssize_t)(item_len - *iov_offset), 0),
                                   length - length_it);
        ucs_assert(*iov_offset <= item_len);

        unpack(worker,
               UCS_PTR_BYTE_OFFSET(iov[*iovcnt_offset].buffer, *iov_offset),
               UCS_PTR_BYTE_OFFSET(src, length_it), item_len_to_copy, mem_type);
        length_it += item_len_to_copy;

        ucs_assert(length_it <= length);
        if (length_it < length) {
            *iov_offset = 0;
            ++(*iovcnt_offset);
        } else {
            *iov_offset += item_len_to_copy;
        }
    }
    return length_it;
}

void ucp_dt_iov_seek(ucp_dt_iov_t *iov, size_t iovcnt, ptrdiff_t distance,
                     size_t *iov_offset, size_t *iovcnt_offset)
{
    ssize_t new_iov_offset; /* signed, since it can be negative */
    size_t length_it;

    new_iov_offset = ((ssize_t)*iov_offset) + distance;

    if (new_iov_offset < 0) {
        /* seek backwards */
        do {
            ucs_assert(*iovcnt_offset > 0);
            --(*iovcnt_offset);
            new_iov_offset += iov[*iovcnt_offset].length;
        } while (new_iov_offset < 0);
    } else {
        /* seek forward */
        while (new_iov_offset >= (length_it = iov[*iovcnt_offset].length)) {
            new_iov_offset -= length_it;
            ++(*iovcnt_offset);
            ucs_assert(*iovcnt_offset < iovcnt);
        }
    }

    *iov_offset = new_iov_offset;
}

size_t ucp_dt_iov_count_nonempty(const ucp_dt_iov_t *iov, size_t iovcnt)
{
    size_t iov_it, count;

    count = 0;
    for (iov_it = 0; iov_it < iovcnt; ++iov_it) {
        count += iov[iov_it].length != 0;
    }
    return count;
}
