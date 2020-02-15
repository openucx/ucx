/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
* Copyright (C) Huawei Technologies Co., Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef PTR_ARRAY_H_
#define PTR_ARRAY_H_

#include <ucs/sys/math.h>
#include <ucs/debug/memtrack.h>


/*
 * Array element layout:
 *
 *         64                 32               1   0
 *          +-----------------+----------------+---+
 * free:    | offset to next used (zero-based) | 1 |
 *          +-----------------+----------------+---+
 * used:    |           user pointer           | 0 |
 *          +-----------------+----------------+---+
 *
 * Below is an example array (U) for used, and the offset is indicated in
 * free slots, pointing at the next free slot:
 *
 *          +---+---+---+---+---+---+---+---+---+---+
 *          | U | 0 | U | 2 | 1 | 0 | U | U | 1 | 0 |
 *          +---+---+---+---+---+---+---+---+---+---+
 *
 */
typedef uint64_t ucs_ptr_array_elem_t;


/**
 * A sparse array of pointers.
 * Free slots can hold 32-bit placeholder value.
 */
typedef struct ucs_ptr_array {
    ucs_ptr_array_elem_t     *start;
    unsigned                 first_free;
    unsigned                 size;
#if ENABLE_MEMTRACK
    char                     name[64];
#endif
} ucs_ptr_array_t;


#define UCS_PTR_ARRAY_FLAG_FREE  (1)  /* Slot is free */
#define UCS_PTR_ARRAY_NEXT_SHIFT (1)
#define UCS_PTR_ARRAY_SET_OFFSET(_elem_p, _offset) *(uintptr_t*)(_elem_p) = \
        (((_offset) << UCS_PTR_ARRAY_NEXT_SHIFT) | UCS_PTR_ARRAY_FLAG_FREE)
#define UCS_PTR_ARRAY_GET_ELEM(_ptr_array, _index) ((_ptr_array)->start[_index])
#define UCS_PTR_ARRAY_IS_ELEM_FREE(_elem) \
        ((uintptr_t)(_elem) & UCS_PTR_ARRAY_FLAG_FREE)
#define UCS_PTR_ARRAY_GET_OFFSET(_ptr_array, _index) \
        (UCS_PTR_ARRAY_GET_ELEM((_ptr_array), (_index)) >> UCS_PTR_ARRAY_NEXT_SHIFT)
#define UCS_PTR_ARRAY_IS_INDEX_FREE(_ptr_array, _index) \
        UCS_PTR_ARRAY_IS_ELEM_FREE(UCS_PTR_ARRAY_GET_ELEM((_ptr_array), (_index)))

/**
 * Get the next non-empty element (exceeds array bounds after the last element).
 */
#define ucs_ptr_array_get_next_index(_ptr_array, _index) \
        (UCS_PTR_ARRAY_IS_INDEX_FREE((_ptr_array), (_index) + 1) ? (_index) + 2 \
                + (unsigned)UCS_PTR_ARRAY_GET_OFFSET((_ptr_array), (_index) + 1) : \
                (_index) + 1)

/**
 * Initialize the array.
 * Retrieve a value from the array.
 *
 * @param init_placeholder   Default placeholder value.
 * @param index   Index to retrieve the value from.
 * @param value   Filled with the value.
 * @return        Whether the value is present and valid.
 *
 * Complexity: O(1)
 */
#define ucs_ptr_array_lookup(_ptr_array, _index, _var) \
    (((_index) >= (_ptr_array)->size) ? \
                    (UCS_V_INITIALIZED(_var), 0) : \
                    !UCS_PTR_ARRAY_IS_ELEM_FREE(_var = (void*) \
                            UCS_PTR_ARRAY_GET_ELEM((_ptr_array), (_index))))


/**
 * Test if the array is empty.
 */
#define ucs_ptr_array_is_empty(_ptr_array) \
        (((_ptr_array)->size == 0) || \
         (UCS_PTR_ARRAY_IS_INDEX_FREE((_ptr_array), 0) && \
          (ucs_ptr_array_get_next_index((_ptr_array), 0) == (_ptr_array)->size)))


/**
 * Iterate over all valid elements in the array.
 */
#define ucs_ptr_array_for_each(_elem, _index, _ptr_array) \
    for ((_index) = (((_ptr_array)->size > 0) && \
                      (UCS_PTR_ARRAY_IS_INDEX_FREE((_ptr_array), 0))) ? \
                      ucs_ptr_array_get_next_index((_ptr_array), 0) : 0, \
         (_elem)  = ((_ptr_array)->size <= (_index)) ? NULL : \
                    (typeof(_elem))UCS_PTR_ARRAY_GET_ELEM((_ptr_array), (_index)); \
         (_index) < (_ptr_array)->size; \
         (_index) = ucs_ptr_array_get_next_index((_ptr_array), (_index)), \
         (_elem)  = ((_ptr_array)->size <= (_index)) ? NULL : \
                    (typeof(_elem))UCS_PTR_ARRAY_GET_ELEM((_ptr_array), (_index)))

/**
 * Initialize the array.
 */
void ucs_ptr_array_init(ucs_ptr_array_t *ptr_array, const char *name);


/**
 * Cleanup the array.
 * All values should already be removed from it.
 */
void ucs_ptr_array_cleanup(ucs_ptr_array_t *ptr_array);


/**
 * Insert a pointer to the array.
 *
 * @param value        Pointer to insert. Must be 8-byte aligned.
 * @return             The index to which the value was inserted.
 *
 * Complexity: amortized O(1)
 *
 * Note: The array will grow if needed.
 */
unsigned ucs_ptr_array_insert(ucs_ptr_array_t *ptr_array, void *value);


/**
 * Set a pointer in the array
 * @param  index    index of slot
 * @param  new_val  value to put into slot given by index
 */
void ucs_ptr_array_set(ucs_ptr_array_t *ptr_array, unsigned index, void *new_val);


/**
 * Remove a pointer from the array.
 *
 * @param index        Index to remove from.
 * @param placeholder  Value to put in the free slot.
 *
 */
void ucs_ptr_array_remove(ucs_ptr_array_t *ptr_array, unsigned index);


/**
 * Replace pointer in the array
 * @param  index    index of slot
 * @param  new_val  value to put into slot given by index
 * @return old value of the slot
 */
void *ucs_ptr_array_replace(ucs_ptr_array_t *ptr_array, unsigned index, void *new_val);

#endif /* PTR_ARRAY_H_ */
