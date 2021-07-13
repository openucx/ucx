/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
 * Copyright (C) Huawei Technologies Co., Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_PTR_ARRAY_FWD_H_
#define UCS_PTR_ARRAY_FWD_H_

#include <stdint.h>

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

/** @file ptr_array_fwd.h */

#define UCS_PTR_ARRAY_NEXT_SHIFT (1)

/*
 * Array element layout:
 *
 *         64                 32               1   0
 *          +-----------------+----------------+---+
 * free:    |     free_ahead  |  next index    | 1 |
 *          +-----------------+----------------+---+
 * used:    |           user pointer           | 0 |
 *          +-----------------+----------------+---+
 *
 *
 * free_ahead is the number of consecutive free elements ahead.
 *
 * The remove / insert algorithm works as follows:
 * On remove of an index: If start[index+1] is free ==>
 * start[index].free_elements_ahead = start[index+1].free_elements_ahead+1
 * Then, the removed index is pushed to the HEAD of the freelist.
 * NOTE, that if start[index+1] is free ==> It's already in the freelist !!!
 *
 * On insert, we fetch the first entry of the freelist and we rely on the
 * fact that the remove/insert mechanism effectively implements a LIFO
 * freelist, i.e. the last item pushed into the freelist will be fetched
 * first ==> There is no chance that index+1 will be fetched before index,
 * since index+1 was already in the list before index was put into the list.
 *
 * Therefore, we can rely on the free_size_ahead field to tell how many free
 * elements are from any index in the freelist.
 *
 * To clarify, "free_ahead" is a best-effort optimization, so when it is not
 * updated on removal - the for-each code runs slower, but still correctly.
 * This decision was made in order to preserve the O(1) performance of
 * ucs_ptr_array_remove() - at the expense of ptr_array_for_each() performance.
 * If we wanted to favor ptr_array_for_each() we had to update "free_ahead"
 * values in all the empty cells before the changed one, a noticeable overhead.
 * Instead, the for-each checks if the cell is empty even if it's indicated as
 * such by "free_ahead". As for insert() - a new cell can be either inserted
 * right after an occupied cell (no need to update "free_ahead") or instead of
 * a removed cell (so that "free_ahead" already points to it). The resulting
 * effect is that "free_ahead" may have "false positives" but never "false
 * negatives". Set() is different, because it "messes" with this logic - and
 * can create that "false negative". This is why it requires such a complicated
 * update of the "free_ahead" (unless the set overwrites an occupied cell).
 *
 */
typedef uint64_t ucs_ptr_array_elem_t;


/**
 * A sparse array of pointers.
 */
typedef struct ucs_ptr_array {
    ucs_ptr_array_elem_t *start;   /* Pointer to the allocated array */
    unsigned             freelist; /* Index of first free slot (see above) */
    unsigned             size;     /* Number of allocated array slots */
    unsigned             count;    /* Actual number of occupied slots */
    const char           *name;    /* Name of this pointer array */
} ucs_ptr_array_t;


/**
 * Shift an input value so it will fit in a ptr array slot.
 *
 * @param [in] new_val        Value to be put into a slot (at some point).
 *
 * @return The same value, as it would be stored in the ptr array.
 */
static UCS_F_ALWAYS_INLINE uintptr_t ucs_ptr_array_shift(uintptr_t new_val)
{
    return new_val << UCS_PTR_ARRAY_NEXT_SHIFT;
}


/**
 * Shift back the value retrieved from a ptr array (@ref ucs_ptr_array_shift).
 *
 * @param [in] old_val        Value which has been stored in the ptr array slot.
 *
 * @return The original value stored in the ptr array slot (before shifting).
 */
static UCS_F_ALWAYS_INLINE uintptr_t ucs_ptr_array_unshift(uintptr_t old_val)
{
    return old_val >> UCS_PTR_ARRAY_NEXT_SHIFT;
}


/**
 * Pointer to the element location by its index in the ptr array.
 *
 * @param [in]  ptr_array     Pointer to a ptr array.
 * @param [in]  index         Index to return the point to.
 *
 * @return A pointer to the element inside the ptr array.
 */
static UCS_F_ALWAYS_INLINE ucs_ptr_array_elem_t*
ucs_ptr_array_at(ucs_ptr_array_t *ptr_array, uint64_t index)
{
    return &ptr_array->start[index];
}


/**
 * Initialize the array.
 *
 * @param [in] ptr_array          Pointer to a ptr array.
 * @param [in] name               The name of the ptr array.
 */
void ucs_ptr_array_init(ucs_ptr_array_t *ptr_array, const char *name);


/**
 * Cleanup the array.
 *
 * @param ptr_array  Pointer to a ptr array.
 * @param leak_check Whether to check for leaks (elements which were not
 *                   freed from this ptr array).
 */
void ucs_ptr_array_cleanup(ucs_ptr_array_t *ptr_array, int leak_check);


/**
 * Allocate a number of contiguous slots in the array.
 *
 * @param [in] locked_ptr_array  Pointer to a ptr array.
 * @param [in] element_count     Number of slots to allocate.
 * @param [in] init_value     Value to set in every allocated slot.
 *
 * @return The index of the requested amount of slots (initialized to zero).
 *
 * Complexity: O(n*n) - not recommended for data-path
 *
 * @note The array will grow if needed.
 * @note Use @ref ucs_ptr_array_bulk_remove to "deallocate" the slots.
 */
unsigned
ucs_ptr_array_bulk_alloc(ucs_ptr_array_t *locked_ptr_array,
                         unsigned element_count,
                         void *init_value);


/**
 * Release a number of contiguous array slots.
 *
 * @param [in] ptr_array      Pointer to a ptr array.
 * @param [in] element_index  Index to remove from.
 * @param [in] element_count  Number of slots to release.
 *
 * Complexity: O(n)
 */
void ucs_ptr_array_bulk_remove(ucs_ptr_array_t *locked_ptr_array,
                               unsigned element_index,
                               unsigned element_count);


END_C_DECLS

#endif /* UCS_PTR_ARRAY_FWD_H_ */
