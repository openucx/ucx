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
    size_t item_len, item_remainder, item_len_to_copy;
    size_t length_it = 0;

    iov += *iovcnt_offset;
    ucs_assert(length > 0);
    while (length_it < length) {
        item_len = ucp_dt_length_recursive(iov->dt, iov->count, NULL, NULL, 0);
        item_remainder = item_len - *iov_offset;

        item_len_to_copy = item_remainder -
                           ucs_max((ssize_t)((length_it + item_remainder) - length), 0);
        memcpy(dest + length_it, iov->buffer + *iov_offset, item_len_to_copy);
        length_it += item_len_to_copy;

        ucs_assert(length_it <= length);
        if (length_it < length) {
            *iov_offset = 0;
            ++(*iovcnt_offset);
            ++iov;
        } else {
            *iov_offset += item_len_to_copy;
        }
    }
}

size_t ucp_dt_iov_scatter(const ucp_dt_iov_t *iov, size_t iovcnt, const void *src,
                          size_t length, size_t *iov_offset, size_t *iovcnt_offset)
{
    size_t item_len, item_len_to_copy;
    size_t length_it = 0;

    iov += *iovcnt_offset;
    while ((length_it < length) && (*iovcnt_offset < iovcnt)) {
        item_len = ucp_dt_length_recursive(iov->dt, iov->count, NULL, NULL, 0);
        item_len_to_copy = ucs_min(ucs_max((ssize_t)(item_len - *iov_offset), 0),
                                   length - length_it);
        ucs_assert(*iov_offset <= item_len);

        memcpy(iov->buffer + *iov_offset, src + length_it, item_len_to_copy);
        length_it += item_len_to_copy;

        ucs_assert(length_it <= length);
        if (length_it < length) {
            *iov_offset = 0;
            ++(*iovcnt_offset);
            ++iov;
        } else {
            *iov_offset += item_len_to_copy;
        }
    }
    return length_it;
}
