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
                            size_t min_capacity, size_t max_capacity,
                            size_t value_size, void **old_buffer_p,
                            const char *array_name)
{
    void *old_buffer        = *buffer_p;
    size_t current_capacity = *capacity_p;
    size_t new_capacity;
    void *new_buffer;

    if (min_capacity > max_capacity) {
        ucs_error("failed to grow %s from %zu to %zu, maximum capacity is %zu",
                  array_name, current_capacity, min_capacity, max_capacity);
        return UCS_ERR_EXCEEDS_LIMIT;
    }

    new_capacity = ucs_max(current_capacity * UCS_ARRAY_GROW_FACTOR,
                           min_capacity);
    new_capacity = ucs_min(new_capacity, max_capacity);
    new_buffer   = ucs_malloc(new_capacity * value_size, array_name);
    if (new_buffer == NULL) {
        ucs_error("failed to grow %s from %zu to %zu elems of %zu bytes",
                  array_name, current_capacity, new_capacity, value_size);
        return UCS_ERR_NO_MEMORY;
    }

    if (old_buffer_p == NULL) {
        if (*buffer_p != NULL) {
            memcpy(new_buffer, *buffer_p, current_capacity * value_size);
        }
        ucs_array_buffer_free(old_buffer);
    } else {
        *old_buffer_p = old_buffer;
    }
    *buffer_p   = new_buffer;
    *capacity_p = new_capacity;
    return UCS_OK;
}

void ucs_array_buffer_free(void *buffer)
{
    ucs_free(buffer);
}
