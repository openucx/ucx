/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "array.h"

#include <ucs/sys/math.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>


/* Increase the array buffer length by this factor, whenever it needs to grow */
#define UCS_ARRAY_GROW_FACTOR   2


ucs_status_t ucs_array_grow(void **buffer_p, size_t *capacity_p,
                            size_t min_capacity, size_t value_size,
                            const char *array_name, const char *value_name)
{
    size_t new_capacity, aligned_capacity;
    void *new_buffer;

    new_capacity     = ucs_max(*capacity_p * UCS_ARRAY_GROW_FACTOR, min_capacity);
    aligned_capacity = (new_capacity + ~UCS_ARRAY_CAP_MASK) & UCS_ARRAY_CAP_MASK;

    new_buffer       = ucs_realloc(*buffer_p, value_size * aligned_capacity,
                                   array_name);
    if (new_buffer == NULL) {
        ucs_error("failed to grow %s from %zu to %zu elems of '%s'",
                  array_name, *capacity_p, new_capacity, value_name);
        return UCS_ERR_NO_MEMORY;
    }

    *buffer_p   = new_buffer;
    *capacity_p = aligned_capacity;
    return UCS_OK;
}
