/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "dt_stride.h"
#include "dt.h"

#include <ucs/debug/log.h>
#include <ucs/sys/math.h>

#include <string.h>
#include <unistd.h>

void ucp_dt_stride_create(ucp_dt_stride_t *dt, va_list ap)
{
    unsigned dim_it;

    dt->dt = va_arg(ap, ucp_datatype_t);
    dt->total_extent = va_arg(ap, size_t);
    dt->ratio = va_arg(ap, unsigned);

    /* Read the stride dimensions */
    dt->total_length = 0;
    dt->dim_cnt = va_arg(ap, unsigned);
    ucs_assert(dt->dim_cnt && (dt->dim_cnt <= UCP_DT_STRIDE_MAX_DIMS));
    for (dim_it = 0; dim_it < dt->dim_cnt; dim_it++) {
        dt->dims[dim_it].extent = va_arg(ap, size_t);
        dt->dims[dim_it].count = va_arg(ap, size_t);
        dt->total_length += dt->dims[dim_it].count;
    }
    dt->item_length = ucp_dt_length_recursive(dt->dt, 1, NULL, NULL, 0);
    dt->total_length *= dt->item_length;
    ucs_assert(dt->total_length <= dt->total_extent);
}

/*
 * Note to self: problem with recursive datatype definitions.
 *
 * proposed solution: "state" has to be an iov_t* + bytes-written!
 * in case of stride - the offset will be arithmetically calculated from
 * bytes written, and then we move on to next iov_t* to copy...
 *
 */
void ucp_dt_stride_gather(void *dest, const void *src, size_t length,
                          const ucp_dt_stride_t *dt,
                          size_t *item_offset, size_t *dim_indexes)
{
    size_t length_it = 0;
    size_t item_len = dt->item_length;
    size_t item_remainder, item_len_to_copy;

    /* Aggregate offset across all dimensions */
    size_t dim, dim_offset = 0;
    for (dim = 0; dim < dt->dim_cnt; dim++) {
        dim_offset += dt->dims[dim].extent * dim_indexes[dim];
    }

    ucs_assert(length > 0);
    while (length_it < length) {
        ucs_assert(item_len >= *item_offset);
        item_remainder = item_len - *item_offset;
        item_len_to_copy = item_remainder -
                           ucs_max((ssize_t)((length_it + item_remainder) - length), 0);
        ucs_assert(item_len_to_copy <= item_len);

        memcpy(dest + length_it, src + *item_offset + dim_offset, item_len_to_copy);
        length_it += item_len_to_copy;

        ucs_assert(length_it <= length);
        if (length_it < length) {
            *item_offset = 0;
            do {
                dim--;
                if (++dim_indexes[dim] == dt->dims[dim].count) {
                    dim_indexes[dim] = 0;
                    dim_offset -= dt->dims[dim].count * dt->dims[dim].extent;
                } else {
                    dim_offset += dt->dims[dim].extent;
                }
            } while ((dim) && (dim_indexes[dim] == 0));
            dim = dt->dim_cnt;
        } else {
            *item_offset += item_len_to_copy;
        }
    }
}

size_t ucp_dt_stride_scatter(const ucp_dt_stride_t *dt, void *dest,
                             const void *src, size_t length,
                             size_t *item_offset, size_t *dim_indexes)
{
    size_t length_it = 0;
    size_t item_len = dt->item_length;
    size_t item_len_to_copy;

    /* Aggregate offset across all dimensions */
    size_t dim, dim_offset = 0;
    for (dim = 0; dim < dt->dim_cnt; dim++) {
        dim_offset += dt->dims[dim].extent * dim_indexes[dim];
    }

    while (length_it < length) {
        item_len_to_copy = ucs_min(ucs_max((ssize_t)(item_len - *item_offset), 0),
                                   length - length_it);
        ucs_assert(*item_offset <= item_len);

        memcpy(dest + *item_offset + dim_offset, src + length_it, item_len_to_copy);
        length_it += item_len_to_copy;

        ucs_assert(length_it <= length);
        if (length_it < length) {
            *item_offset = 0;
            do {
                dim--;
                if (++dim_indexes[dim] == dt->dims[dim].count) {
                    dim_indexes[dim] = 0;
                    dim_offset -= dt->dims[dim].count * dt->dims[dim].extent;
                } else {
                    dim_offset += dt->dims[dim].extent;
                }
            } while ((dim) && (dim_indexes[dim] == 0));
            dim = dt->dim_cnt;
        } else {
            *item_offset += item_len_to_copy;
        }
    }
    return length_it;
}
