/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "dt_iov.h"
#include "dt.h"

#include <ucs/debug/log.h>
#include <ucs/sys/math.h>

#include <string.h>
#include <unistd.h>


void ucp_dt_iov_gather(void *dest, const ucp_dt_iov_t *iov, size_t length,
                       size_t *iov_offset, size_t *iovcnt_offset)
{
    size_t item_len, item_reminder, item_len_to_copy;
    size_t length_it = 0;

    ucs_assert(length > 0);
    while (length_it < length) {
        item_len      = ucp_contig_dt_length(iov[*iovcnt_offset].dt,
                                             iov[*iovcnt_offset].count);
        item_reminder = item_len - *iov_offset;

        item_len_to_copy = item_reminder -
                           ucs_max((ssize_t)((length_it + item_reminder) - length), 0);
        memcpy(dest + length_it, iov[*iovcnt_offset].buffer + *iov_offset,
               item_len_to_copy);
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

size_t ucp_dt_iov_scatter(ucp_dt_iov_t *iov, size_t iovcnt, const void *src,
                          size_t length, size_t *iov_offset, size_t *iovcnt_offset)
{
    size_t item_len, item_len_to_copy;
    size_t length_it = 0;

    while ((length_it < length) && (*iovcnt_offset < iovcnt)) {
        item_len         = ucp_contig_dt_length(iov[*iovcnt_offset].dt,
                                                iov[*iovcnt_offset].count);
        item_len_to_copy = ucs_min(ucs_max((ssize_t)(item_len - *iov_offset), 0),
                                   length - length_it);
        ucs_assert(*iov_offset <= item_len);

        memcpy(iov[*iovcnt_offset].buffer + *iov_offset, src + length_it,
               item_len_to_copy);
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
