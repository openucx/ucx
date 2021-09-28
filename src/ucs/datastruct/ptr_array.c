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
#include <ucs/debug/memtrack_int.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>


/* Initial allocation size */
#define UCS_PTR_ARRAY_INITIAL_SIZE  8


static UCS_F_ALWAYS_INLINE int
ucs_ptr_array_is_free(ucs_ptr_array_t *ptr_array, unsigned element_index)
{
    return (element_index < ptr_array->size) &&
            __ucs_ptr_array_is_free(ptr_array->start[element_index]);
}

static UCS_F_ALWAYS_INLINE uint32_t
ucs_ptr_array_size_free_get_free_ahead(ucs_ptr_array_elem_t elem)
{
    ucs_assert(__ucs_ptr_array_is_free(elem));
    return elem >> UCS_PTR_ARRAY_FREE_AHEAD_SHIFT;
}

static UCS_F_ALWAYS_INLINE unsigned
ucs_ptr_array_freelist_get_next(ucs_ptr_array_elem_t elem)
{
    ucs_assert(__ucs_ptr_array_is_free(elem));
    return (elem & UCS_PTR_ARRAY_NEXT_MASK) >> UCS_PTR_ARRAY_NEXT_SHIFT;
}

static UCS_F_ALWAYS_INLINE void
ucs_ptr_array_freelist_set_next(ucs_ptr_array_elem_t *elem, unsigned next)
{
    ucs_assert(next <= UCS_PTR_ARRAY_NEXT_MASK);
    *elem = (*elem & ~UCS_PTR_ARRAY_NEXT_MASK) |
                    (((ucs_ptr_array_elem_t)next) << UCS_PTR_ARRAY_NEXT_SHIFT);
}

/**
 * Sets all values of a free ptr array element
 *
 * @param [in/out] elem       Pointer to a free element in the ptr array.
 * @param [in]     free_ahead Number of free consecutive elements ahead.
 * @param [in]     next       Pointer to the next element in the ptr array.
 *
 * Complexity: O(1)
 */
static UCS_F_ALWAYS_INLINE void
ucs_ptr_array_freelist_element_set(ucs_ptr_array_elem_t *elem,
                                   uint32_t free_ahead,
                                   unsigned next)
{
    ucs_assert(next <= UCS_PTR_ARRAY_NEXT_MASK);

    *elem = UCS_PTR_ARRAY_FLAG_FREE |
            (((ucs_ptr_array_elem_t)free_ahead) << UCS_PTR_ARRAY_FREE_AHEAD_SHIFT) |
            (((ucs_ptr_array_elem_t)next) << UCS_PTR_ARRAY_NEXT_SHIFT);
}

static UCS_F_ALWAYS_INLINE void
ucs_ptr_array_freelist_element_set_free_ahead(ucs_ptr_array_elem_t *elem,
                                              uint32_t free_ahead)
{
    ucs_ptr_array_freelist_element_set(elem, free_ahead,
                                       ucs_ptr_array_freelist_get_next(*elem));
}

static void UCS_F_MAYBE_UNUSED ucs_ptr_array_dump(ucs_ptr_array_t *ptr_array)
{
#if UCS_ENABLE_ASSERT
    unsigned i;

    ucs_trace_data("ptr_array start %p size %u count %u",
                   ptr_array->start, ptr_array->size, ptr_array->count);
    for (i = 0; i < ptr_array->size; ++i) {
        if (ucs_ptr_array_is_free(ptr_array, i)) {
            ucs_trace_data("(%u) [%u]=<free> [%u]=<next>", i,
                           ucs_ptr_array_size_free_get_free_ahead(ptr_array->start[i]),
                           ucs_ptr_array_freelist_get_next(ptr_array->start[i]));
        } else {
            ucs_trace_data("[%u]=%p", i, (void*)ptr_array->start[i]);
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
    ptr_array->start    = NULL;
    ptr_array->size     = 0;
    ptr_array->count    = 0;
    ptr_array->freelist = UCS_PTR_ARRAY_SENTINEL;
    ptr_array->name     = NULL;
}

void ucs_ptr_array_init(ucs_ptr_array_t *ptr_array, const char *name)
{
    ucs_ptr_array_clear(ptr_array);
    ptr_array->name = name;
}

void ucs_ptr_array_cleanup(ucs_ptr_array_t *ptr_array, int leak_check)
{
    unsigned i;

    if (leak_check && (ptr_array->count > 0)) {
        ucs_warn("releasing ptr_array with %u used items", ptr_array->count);
        for (i = 0; i < ptr_array->size; ++i) {
            if (!ucs_ptr_array_is_free(ptr_array, i)) {
                ucs_trace("ptr_array(%p) idx %d is not free during cleanup:"
                          " 0x%"PRIx64, ptr_array, i, ptr_array->start[i]);
            }
        }
    }

    ucs_free(ptr_array->start);
    ucs_ptr_array_clear(ptr_array);
}

static void ucs_ptr_array_grow(ucs_ptr_array_t *ptr_array, unsigned new_size)
{
    ucs_ptr_array_elem_t *new_array;
    unsigned curr_size, i, next;

    /* Allocate new array */
    new_array = ucs_malloc(new_size * sizeof(ucs_ptr_array_elem_t),
                           ptr_array->name);
    ucs_assert_always(new_array != NULL);
    curr_size = ptr_array->size;
    memcpy(new_array, ptr_array->start, curr_size * sizeof(ucs_ptr_array_elem_t));

    /* Link all new array items */
    for (i = curr_size; i < new_size; ++i) {
        ucs_ptr_array_freelist_element_set(&new_array[i], new_size - i,
                                           i + 1);
    }
    ucs_ptr_array_freelist_set_next(&new_array[new_size - 1], UCS_PTR_ARRAY_SENTINEL);

    /* Find last free list element */
    if (ptr_array->freelist == UCS_PTR_ARRAY_SENTINEL) {
        ptr_array->freelist = curr_size;
    } else {
        next = ptr_array->freelist;
        do {
            i = next;
            next = ucs_ptr_array_freelist_get_next(new_array[i]);
        } while (next != UCS_PTR_ARRAY_SENTINEL);
        ucs_ptr_array_freelist_set_next(&new_array[i], curr_size);
    }

    /* Switch to new array */
    ucs_free(ptr_array->start);
    ptr_array->start = new_array;
    ptr_array->size  = new_size;
}

unsigned
ucs_ptr_array_bulk_alloc(ucs_ptr_array_t *ptr_array, unsigned element_count)
{
    unsigned free_iter, new_size, element_index;

    if (element_count == 0) {
        return 0;
    }

    element_index = ptr_array->freelist;
    if (element_index == UCS_PTR_ARRAY_SENTINEL) {
        goto alloc_grow;
    }

    free_iter = 1; /* first element from a free-list must be free */
    do {
        while ((free_iter < element_count) &&
               (ucs_ptr_array_is_free(ptr_array, element_index + free_iter))) {
            free_iter++;
        }

        if (free_iter == element_count) {
            goto alloc_init;
        }

        free_iter     = ptr_array->start[element_index];
        element_index = ucs_ptr_array_freelist_get_next(free_iter);
        free_iter     = 1;
    } while (element_index != UCS_PTR_ARRAY_SENTINEL);

alloc_grow:
    element_index = ptr_array->size;
    new_size      = ucs_max(2 * ptr_array->size,
                            ptr_array->size + element_count);
    ucs_ptr_array_grow(ptr_array, new_size);

alloc_init:
    for (free_iter = 0; free_iter < element_count; free_iter++) {
        /* set the value and remove from the free-list */
        ucs_ptr_array_set(ptr_array, element_index + free_iter, 0);
    }

    return element_index;
}

unsigned ucs_ptr_array_insert(ucs_ptr_array_t *ptr_array, void *value)
{
    unsigned ret = ucs_ptr_array_bulk_alloc(ptr_array, 1);

    ucs_assert(((uintptr_t)value & UCS_PTR_ARRAY_FLAG_FREE) == 0);

    ptr_array->start[ret] = (uintptr_t)value;

    return ret;
}

void ucs_ptr_array_set(ucs_ptr_array_t *ptr_array, unsigned element_index,
                       void *new_val)
{
    ucs_ptr_array_elem_t *elem;
    unsigned next, free_iter, free_ahead, new_size;

    ucs_assert(((uintptr_t)new_val & UCS_PTR_ARRAY_FLAG_FREE) == 0);

    if (ucs_unlikely(element_index >= ptr_array->size)) {
        new_size = ucs_max(ptr_array->size * 2, element_index + 1);
        ucs_ptr_array_grow(ptr_array, new_size);
    } else if (!__ucs_ptr_array_is_free(ptr_array->start[element_index])) {
        ptr_array->start[element_index] = (uintptr_t)new_val;
        return;
    }

    next = ucs_ptr_array_freelist_get_next(ptr_array->start[element_index]);
    ptr_array->start[element_index] = (uintptr_t)new_val;
    ptr_array->count++;

    /* update the "next index" in the free list (removing element_index from it) */
    free_iter = ptr_array->freelist;
    if (ucs_unlikely(free_iter == element_index)) {
        ptr_array->freelist = next;
    } else {
        while (element_index !=
               ucs_ptr_array_freelist_get_next(ptr_array->start[free_iter])) {
            free_iter =
                   ucs_ptr_array_freelist_get_next(ptr_array->start[free_iter]);
            ucs_assert(free_iter != UCS_PTR_ARRAY_SENTINEL);
        }
        ucs_ptr_array_freelist_set_next(ptr_array->start + free_iter, next);
    }

    /* update the "free-ahead" for the cells before me */
    free_ahead = 1;
    elem       = ptr_array->start + element_index - 1;
    while ((elem >= ptr_array->start) && (__ucs_ptr_array_is_free(*elem))) {
        ucs_ptr_array_freelist_element_set_free_ahead(elem, free_ahead);
        free_ahead++;
        elem--;
    }
}

void ucs_ptr_array_remove(ucs_ptr_array_t *ptr_array, unsigned element_index)
{
    ucs_ptr_array_elem_t *next_elem;
    uint32_t size_free_ahead;

    ucs_assert_always(!ucs_ptr_array_is_free(ptr_array, element_index));
    ucs_assert(ptr_array->count > 0);

    if (ucs_ptr_array_is_free(ptr_array, element_index + 1)) {
        next_elem = &ptr_array->start[element_index + 1];
        size_free_ahead = ucs_ptr_array_size_free_get_free_ahead(*next_elem) + 1;
    } else {
        size_free_ahead = 1;
    }

    ucs_ptr_array_freelist_element_set(&ptr_array->start[element_index],
                                       size_free_ahead, ptr_array->freelist);

    /* Make sure the next element is free */
    ucs_assert(__ucs_ptr_array_is_free(ptr_array->start[element_index + size_free_ahead - 1]));

    ptr_array->freelist = element_index;
    ptr_array->count--;
}

void *ucs_ptr_array_replace(ucs_ptr_array_t *ptr_array, unsigned element_index,
                            void *new_val)
{
    void *old_elem;

    ucs_assert_always(!ucs_ptr_array_is_free(ptr_array, element_index));
    old_elem                        = (void*)ptr_array->start[element_index];
    ptr_array->start[element_index] = (uintptr_t)new_val;
    return old_elem;
}


/*
 *  Locked interface functions implementation
 */

ucs_status_t
ucs_ptr_array_locked_init(ucs_ptr_array_locked_t *locked_ptr_array,
                          const char *name)
{
    ucs_status_t status;

    /* Initialize spinlock */
    status = ucs_recursive_spinlock_init(&locked_ptr_array->lock, 0);
    if (status != UCS_OK) {
       return status;
    }

    /* Call unlocked function */
    ucs_ptr_array_init(&locked_ptr_array->super, name);

    return UCS_OK;
}

void ucs_ptr_array_locked_cleanup(ucs_ptr_array_locked_t *locked_ptr_array,
                                  int leak_check)
{
    ucs_recursive_spin_lock(&locked_ptr_array->lock);
    /* Call unlocked function */
    ucs_ptr_array_cleanup(&locked_ptr_array->super, leak_check);
    ucs_recursive_spin_unlock(&locked_ptr_array->lock);

    /* Destroy spinlock */
    ucs_recursive_spinlock_destroy(&locked_ptr_array->lock);
}

unsigned ucs_ptr_array_locked_insert(ucs_ptr_array_locked_t *locked_ptr_array,
                                     void *value)
{
    unsigned element_index;

    ucs_recursive_spin_lock(&locked_ptr_array->lock);
    /* Call unlocked function */
    element_index = ucs_ptr_array_insert(&locked_ptr_array->super, value);
    ucs_recursive_spin_unlock(&locked_ptr_array->lock);

    return element_index;
}

unsigned
ucs_ptr_array_locked_bulk_alloc(ucs_ptr_array_locked_t *locked_ptr_array,
                                unsigned element_count)
{
    unsigned element_index;

    ucs_recursive_spin_lock(&locked_ptr_array->lock);
    /* Call unlocked function */
    element_index = ucs_ptr_array_bulk_alloc(&locked_ptr_array->super,
                                             element_count);
    ucs_recursive_spin_unlock(&locked_ptr_array->lock);

    return element_index;
}

void ucs_ptr_array_locked_set(ucs_ptr_array_locked_t *locked_ptr_array,
                              unsigned element_index, void *new_val)
{
    ucs_recursive_spin_lock(&locked_ptr_array->lock);
    /* Call unlocked function */
    ucs_ptr_array_set(&locked_ptr_array->super, element_index, new_val);
    ucs_recursive_spin_unlock(&locked_ptr_array->lock);
}

void ucs_ptr_array_locked_remove(ucs_ptr_array_locked_t *locked_ptr_array,
                                 unsigned element_index)
{
    ucs_recursive_spin_lock(&locked_ptr_array->lock);
    /* Call unlocked function */
    ucs_ptr_array_remove(&locked_ptr_array->super, element_index);
    ucs_recursive_spin_unlock(&locked_ptr_array->lock);
}

void *ucs_ptr_array_locked_replace(ucs_ptr_array_locked_t *locked_ptr_array,
                                   unsigned element_index, void *new_val)
{
    void *old_elem;

    ucs_recursive_spin_lock(&locked_ptr_array->lock);
    /* Call unlocked function */
    old_elem = ucs_ptr_array_replace(&locked_ptr_array->super, element_index,
                                     new_val);
    ucs_recursive_spin_unlock(&locked_ptr_array->lock);

    return old_elem;
}

