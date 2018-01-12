/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "dt_iov.h"

#include <ucs/debug/assert.h>
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
        item_len      = iov[*iovcnt_offset].length;
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
        item_len         = iov[*iovcnt_offset].length;
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
