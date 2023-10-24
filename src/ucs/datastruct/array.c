/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
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
#include <ucs/debug/assert.h>


/* Increase the array buffer length by this factor, whenever it needs to grow */
#define UCS_ARRAY_GROW_FACTOR   2


ucs_status_t ucs_array_grow(void **buffer_p, size_t *capacity_p,
                            size_t min_capacity, size_t value_size,
                            const char *array_name, const char *value_name)
{
    size_t new_capacity, aligned_capacity, current_capacity;
    void *new_buffer;

    current_capacity = *capacity_p;
    new_capacity     = ucs_max(current_capacity * UCS_ARRAY_GROW_FACTOR,
                               min_capacity);
    aligned_capacity = (new_capacity + ~UCS_ARRAY_CAP_MASK) & UCS_ARRAY_CAP_MASK;
    new_buffer       = ucs_malloc(aligned_capacity * value_size, array_name);
    if (new_buffer == NULL) {
        ucs_error("failed to grow %s from %zu to %zu elems of '%s'",
                  array_name, *capacity_p, new_capacity, value_name);
        return UCS_ERR_NO_MEMORY;
    }

    if (*buffer_p == NULL) {
        goto out;
    }

    ucs_assert(!(current_capacity & UCS_ARRAY_CAP_FLAG_FIXED));

    memcpy(new_buffer, *buffer_p, current_capacity * value_size);

out:
    *buffer_p   = new_buffer;
    *capacity_p = aligned_capacity;
    return UCS_OK;
}
