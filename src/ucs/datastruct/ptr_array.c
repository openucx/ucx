/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ptr_array.h"

#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <malloc.h>


/* Initial allocation size */
#define UCS_PTR_ARRAY_INITIAL_SIZE  8


static inline int ucs_ptr_array_is_free(ucs_ptr_array_t *ptr_array, unsigned index)
{
    return (index < ptr_array->size) &&
            __ucs_ptr_array_is_free(ptr_array->start[index]);
}

static inline uint32_t ucs_ptr_array_placeholder_get(ucs_ptr_array_elem_t elem)
{
    ucs_assert(__ucs_ptr_array_is_free(elem));
    return elem >> UCS_PTR_ARRAY_PLCHDR_SHIFT;
}

static inline void ucs_ptr_array_placeholder_set(ucs_ptr_array_elem_t *elem,
                                                 uint32_t placeholder)
{
    *elem = (*elem & ~UCS_PTR_ARRAY_PLCHDR_MASK) |
                    (((ucs_ptr_array_elem_t)placeholder) << UCS_PTR_ARRAY_PLCHDR_SHIFT);
}

static inline unsigned
ucs_ptr_array_freelist_get_next(ucs_ptr_array_elem_t elem)
{
    ucs_assert(__ucs_ptr_array_is_free(elem));
    return (elem & UCS_PTR_ARRAY_NEXT_MASK) >> UCS_PTR_ARRAY_NEXT_SHIFT;
}

static inline void
ucs_ptr_array_freelist_set_next(ucs_ptr_array_elem_t *elem, unsigned next)
{
    ucs_assert(next <= UCS_PTR_ARRAY_NEXT_MASK);
    *elem = (*elem & ~UCS_PTR_ARRAY_NEXT_MASK) |
                    (((ucs_ptr_array_elem_t)next) << UCS_PTR_ARRAY_NEXT_SHIFT);
}

static void UCS_F_MAYBE_UNUSED ucs_ptr_array_dump(ucs_ptr_array_t *ptr_array)
{
#if ENABLE_ASSERT
    ucs_ptr_array_elem_t elem;
    unsigned i;

    ucs_trace_data("ptr_array start %p size %u", ptr_array->start, ptr_array->size);
    for (i = 0; i < ptr_array->size; ++i) {
        elem = ptr_array->start[i];
        if (ucs_ptr_array_is_free(ptr_array, i)) {
            ucs_trace_data("[%u]=<free> (%u)", i,
                           ucs_ptr_array_placeholder_get(elem));
        } else {
            ucs_trace_data("[%u]=%p", i, (void*)elem);
        }
    }

    ucs_trace_data("freelist:");
    i = ptr_array->freelist;
    while (i != UCS_PTR_ARRAY_SENTINEL) {
        ucs_trace_data("[%u] %p", i, &ptr_array->start[i]);
        i = ucs_ptr_array_freelist_get_next(ptr_array->start[i]);
    }
#endif
}

static void ucs_ptr_array_clear(ucs_ptr_array_t *ptr_array)
{
    ptr_array->start            = NULL;
    ptr_array->size             = 0;
    ptr_array->freelist         = UCS_PTR_ARRAY_SENTINEL;
}

void ucs_ptr_array_init(ucs_ptr_array_t *ptr_array, uint32_t init_placeholder,
                        const char *name)
{
    ptr_array->init_placeholder = init_placeholder;
    ucs_ptr_array_clear(ptr_array);
#if ENABLE_MEMTRACK
    ucs_snprintf_zero(ptr_array->name, sizeof(ptr_array->name), "%s", name);
#endif
}

void ucs_ptr_array_cleanup(ucs_ptr_array_t *ptr_array)
{
    unsigned i, inuse;

    inuse = 0;
    for (i = 0; i < ptr_array->size; ++i) {
        if (!ucs_ptr_array_is_free(ptr_array, i)) {
            ++inuse;
            ucs_trace("ptr_array(%p) idx %d is not free during cleanup", ptr_array, i);
        }
    }

    if (inuse > 0) {
        ucs_warn("releasing ptr_array with %u used items", inuse);
    }

    ucs_free(ptr_array->start);
    ucs_ptr_array_clear(ptr_array);
}

static void ucs_ptr_array_grow(ucs_ptr_array_t *ptr_array UCS_MEMTRACK_ARG)
{
    ucs_ptr_array_elem_t *new_array;
    unsigned curr_size, new_size;
    unsigned i, next;

    curr_size = ptr_array->size;
    if (curr_size == 0) {
        new_size = UCS_PTR_ARRAY_INITIAL_SIZE;
    } else {
        new_size = curr_size * 2;
    }

    /* Allocate new array */
    new_array = ucs_malloc(new_size * sizeof(ucs_ptr_array_elem_t) UCS_MEMTRACK_VAL);
    ucs_assert_always(new_array != NULL);
    memcpy(new_array, ptr_array->start, curr_size * sizeof(ucs_ptr_array_elem_t));

    /* Link all new array items */
    for (i = curr_size; i < new_size; ++i) {
        new_array[i] = UCS_PTR_ARRAY_FLAG_FREE;
        ucs_ptr_array_placeholder_set(&new_array[i], ptr_array->init_placeholder);
        ucs_ptr_array_freelist_set_next(&new_array[i], i + 1);
    }
    ucs_ptr_array_freelist_set_next(&new_array[new_size - 1], UCS_PTR_ARRAY_SENTINEL);

    /* Find last free list element */
    if (ptr_array->freelist == UCS_PTR_ARRAY_SENTINEL) {
        ptr_array->freelist = curr_size;
    } else {
        next = ptr_array->freelist;
        do {
            i = next;
            next = ucs_ptr_array_freelist_get_next(ptr_array->start[i]);
        } while (next != UCS_PTR_ARRAY_SENTINEL);
        ucs_ptr_array_freelist_set_next(&ptr_array->start[i], curr_size);
    }

    /* Switch to new array */
    ucs_free(ptr_array->start);
    ptr_array->start = new_array;
    ptr_array->size  = new_size;
}

unsigned ucs_ptr_array_insert(ucs_ptr_array_t *ptr_array, void *value,
                              uint32_t *placeholder_p)
{
    ucs_ptr_array_elem_t *elem;
    unsigned index;

    ucs_assert_always(((uintptr_t)value & UCS_PTR_ARRAY_FLAG_FREE) == 0);

    if (ptr_array->freelist == UCS_PTR_ARRAY_SENTINEL) {
        ucs_ptr_array_grow(ptr_array UCS_MEMTRACK_NAME(ptr_array->name));
    }

    /* Get the first item on the free list */
    index = ptr_array->freelist;
    ucs_assert(index != UCS_PTR_ARRAY_SENTINEL);
    elem = &ptr_array->start[index];

    /* Remove from free list */
    ptr_array->freelist = ucs_ptr_array_freelist_get_next(*elem);

    /* Populate */
    *placeholder_p = ucs_ptr_array_placeholder_get(*elem);
    *elem = (uintptr_t)value;
    return index;
}

void ucs_ptr_array_remove(ucs_ptr_array_t *ptr_array, unsigned index,
                          uint32_t placeholder)
{
    ucs_ptr_array_elem_t *elem = &ptr_array->start[index];

    ucs_assert_always(!ucs_ptr_array_is_free(ptr_array, index));
    *elem = UCS_PTR_ARRAY_FLAG_FREE;
    ucs_ptr_array_placeholder_set(elem, placeholder);
    ucs_ptr_array_freelist_set_next(elem, ptr_array->freelist);
    ptr_array->freelist = index;
}

void *ucs_ptr_array_replace(ucs_ptr_array_t *ptr_array, unsigned index, void *new_val)
{
    void *old_elem;

    ucs_assert_always(!ucs_ptr_array_is_free(ptr_array, index));
    old_elem = (void *)ptr_array->start[index];
    ptr_array->start[index] = (uintptr_t)new_val;
    return old_elem;
}
