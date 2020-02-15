/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
* Copyright (C) Huawei Technologies Co., Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ptr_array.h"

#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>


#define UCS_PTR_ARRAY_INITIAL_SIZE (8) /* Initial allocation size */


static inline int ucs_ptr_array_is_free(ucs_ptr_array_t *ptr_array, unsigned index)
{
    ucs_assert(index < ptr_array->size);
    return UCS_PTR_ARRAY_IS_INDEX_FREE(ptr_array, index);
}

static void UCS_F_MAYBE_UNUSED ucs_ptr_array_dump(ucs_ptr_array_t *ptr_array)
{
#if UCS_ENABLE_ASSERT
    ucs_ptr_array_elem_t *elem = ptr_array->start;
    unsigned i;

    ucs_trace_data("ptr_array start %p size %u", ptr_array->start, ptr_array->size);
    for (i = 0; i < ptr_array->size; ++i, ++elem) {
        if (ucs_ptr_array_is_free(ptr_array, i)) {
            ucs_trace_data("[%u]=<free> (offset=%lu)", i,
                    UCS_PTR_ARRAY_GET_OFFSET(ptr_array, i));
        } else {
            ucs_trace_data("[%u]=%p", i, *(void**)elem);
        }
    }
    ucs_trace_data("first free is #%u", ptr_array->first_free);
#endif
}

static void ucs_ptr_array_clear(ucs_ptr_array_t *ptr_array)
{
    ptr_array->start            = NULL;
    ptr_array->size             = 0;
    ptr_array->first_free       = 0;
}

void ucs_ptr_array_init(ucs_ptr_array_t *ptr_array, const char *name)
{
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

static void ucs_ptr_array_grow(ucs_ptr_array_t *ptr_array, unsigned new_size
                               UCS_MEMTRACK_ARG)
{
    ucs_ptr_array_elem_t *new_array;
    unsigned i, curr_size;

    /* Allocate new array */
    curr_size = ptr_array->size;
    new_array = ucs_realloc(ptr_array->start,
            new_size * sizeof(ucs_ptr_array_elem_t) UCS_MEMTRACK_VAL);
    ucs_assert_always(new_array != NULL);

    /* Link all new array items */
    for (i = curr_size; i < new_size; i++) {
        UCS_PTR_ARRAY_SET_OFFSET(new_array + i, new_size - 1 - i);
    }
    for (i = curr_size; i && UCS_PTR_ARRAY_IS_ELEM_FREE(new_array[i - 1]); i--) {
        UCS_PTR_ARRAY_SET_OFFSET(new_array + i - 1, new_size - i);
    }

    /* Switch to new array */
    ptr_array->start = new_array;
    ptr_array->size  = new_size;
}

unsigned ucs_ptr_array_insert(ucs_ptr_array_t *ptr_array, void *value)
{
    unsigned findex, index;

    ucs_assert_always(((uintptr_t)value & UCS_PTR_ARRAY_FLAG_FREE) == 0);

    if (ptr_array->first_free == ptr_array->size)  {
        ucs_ptr_array_grow(ptr_array, (ptr_array->size == 0) ?
                UCS_PTR_ARRAY_INITIAL_SIZE : ptr_array->size * 2
                UCS_MEMTRACK_NAME(ptr_array->name));
    }

    /* Insert the new element */
    findex = index = ptr_array->first_free;
    ucs_assert(UCS_PTR_ARRAY_IS_INDEX_FREE(ptr_array, index));
    ucs_ptr_array_elem_t *elem_p = &UCS_PTR_ARRAY_GET_ELEM(ptr_array, index);
    *elem_p = (uintptr_t)value;

    /* Find the next free slot, to set the new free_index */
    do {
        findex++;
        elem_p++;
    }
    while ((findex < ptr_array->size) &&
            (!UCS_PTR_ARRAY_IS_ELEM_FREE(*elem_p)));

    /* wrap-around (if the end of the array has been reached) */
    if (ucs_unlikely(findex == ptr_array->size)) {
        /* Free slots have not been found after that index - try before... */
        findex = 0;
        elem_p = ptr_array->start;
        while ((findex < ptr_array->size) &&
                (!UCS_PTR_ARRAY_IS_ELEM_FREE(*elem_p))) {
            findex++;
            elem_p++;
        }
    }
    ptr_array->first_free = findex;
    return index;
}

void ucs_ptr_array_remove(ucs_ptr_array_t *ptr_array, unsigned index)
{
    ucs_assert_always(index < ptr_array->size);
    ucs_assert_always(!ucs_ptr_array_is_free(ptr_array, index));

    /* Replace the removed pointer with the correct "free-offset" */
    ucs_ptr_array_elem_t *elem_p = &UCS_PTR_ARRAY_GET_ELEM(ptr_array, index++);
    unsigned free_index = ((index == ptr_array->size) ||
            (!UCS_PTR_ARRAY_IS_INDEX_FREE(ptr_array, index))) ? 0 :
                    UCS_PTR_ARRAY_GET_OFFSET(ptr_array, index) + 1;

    /* Run back and amend the other "free-offsets" accordingly */
    do {
        UCS_PTR_ARRAY_SET_OFFSET(elem_p--, free_index++);
    } while (--index && UCS_PTR_ARRAY_IS_ELEM_FREE(*elem_p));

    ptr_array->first_free = index;
    ucs_assert(UCS_PTR_ARRAY_IS_INDEX_FREE(ptr_array, ptr_array->first_free));
}

void ucs_ptr_array_set(ucs_ptr_array_t *ptr_array, unsigned index, void *new_val)
{
    ucs_assert_always(((uintptr_t)new_val & UCS_PTR_ARRAY_FLAG_FREE) == 0);

    /* Grow the array if this exceeds array bounds */
    if (index >= ptr_array->size) {
        ucs_ptr_array_grow(ptr_array, index + 1  UCS_MEMTRACK_NAME(ptr_array->name));
    }

    /* Set the new value */
    ptr_array->start[index] = (uintptr_t)new_val;

    if (ptr_array->first_free == index) {
        /* Find the next free slot */
        do {
            index++;
        } while ((index < ptr_array->size) &&
                (UCS_PTR_ARRAY_IS_INDEX_FREE(ptr_array, index)));
        ptr_array->first_free = index;
    } else {
        /* Update the offset on preceding slots, if any */
        unsigned findex = 0;
        while (index && UCS_PTR_ARRAY_IS_INDEX_FREE(ptr_array, index - 1)) {
            UCS_PTR_ARRAY_SET_OFFSET(ptr_array->start + --index, findex++);
        }
    }
}

void *ucs_ptr_array_replace(ucs_ptr_array_t *ptr_array, unsigned index, void *new_val)
{
    ucs_assert_always(((uintptr_t)new_val & UCS_PTR_ARRAY_FLAG_FREE) == 0);
    ucs_assert_always(!UCS_PTR_ARRAY_IS_INDEX_FREE(ptr_array, index));
    void *old_elem = (void *)ptr_array->start[index];
    ptr_array->start[index] = (uintptr_t)new_val;
    return old_elem;
}
