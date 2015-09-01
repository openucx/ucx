/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
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
 * free:    |     placeholder |  next index    | 1 |
 *          +-----------------+----------------+---+
 * used:    |           user pointer           | 0 |
 *          +-----------------+----------------+---+
 *
 *
 */
typedef uint64_t ucs_ptr_array_elem_t;


/**
 * A sparse array of pointers.
 * Free slots can hold 32-bit placeholder value.
 */
typedef struct ucs_ptr_array {
    uint32_t                 init_placeholder;
    ucs_ptr_array_elem_t     *start;
    unsigned                 freelist;
    unsigned                 size;
#if ENABLE_MEMTRACK
    char                     name[64];
#endif
} ucs_ptr_array_t;


/* Flags added to lower bits of the value */
#define UCS_PTR_ARRAY_FLAG_FREE    ((unsigned long)0x01)  /* Slot is free */

#define UCS_PTR_ARRAY_PLCHDR_SHIFT 32
#define UCS_PTR_ARRAY_PLCHDR_MASK  (((ucs_ptr_array_elem_t)-1) & ~UCS_MASK(UCS_PTR_ARRAY_PLCHDR_SHIFT))
#define UCS_PTR_ARRAY_NEXT_SHIFT   1
#define UCS_PTR_ARRAY_NEXT_MASK    (UCS_MASK(UCS_PTR_ARRAY_PLCHDR_SHIFT) & ~UCS_MASK(UCS_PTR_ARRAY_NEXT_SHIFT))
#define UCS_PTR_ARRAY_SENTINEL     (UCS_PTR_ARRAY_NEXT_MASK >> UCS_PTR_ARRAY_NEXT_SHIFT)

#define __ucs_ptr_array_is_free(_elem) \
    ((uintptr_t)(_elem) & UCS_PTR_ARRAY_FLAG_FREE)


/**
 * Initialize the array.
 *
 * @param init_placeholder   Default placeholder value.
 */
void ucs_ptr_array_init(ucs_ptr_array_t *ptr_array, uint32_t init_placeholder,
                        const char *name);


/**
 * Cleanup the array.
 * All values should already be removed from it.
 */
void ucs_ptr_array_cleanup(ucs_ptr_array_t *ptr_array);


/**
 * Insert a pointer to the array.
 *
 * @param value        Pointer to insert. Must be 8-byte aligned.
 * @param placeholder  Filled with placeholder value.
 * @return             The index to which the value was inserted.
 *
 * Complexity: amortized O(1)
 *
 * Note: The array will grow if needed.
 */
unsigned ucs_ptr_array_insert(ucs_ptr_array_t *ptr_array, void *value,
                              uint32_t *placeholder_p);


/**
 * Remove a pointer from the array.
 *
 * @param index        Index to remove from.
 * @param placeholder  Value to put in the free slot.
 *
 * Complexity: O(1)
 */
void ucs_ptr_array_remove(ucs_ptr_array_t *ptr_array, unsigned index,
                          uint32_t placeholder);


/**
 * Replace pointer in the array
 * @param  index    index of slot
 * @param  new_val  value to put into slot given by index
 * @return old value of the slot
 */
void *ucs_ptr_array_replace(ucs_ptr_array_t *ptr_array, unsigned index, void *new_val);


/**
 * Retrieve a value from the array.
 *
 * @param index   Index to retrieve the value from.
 * @param value   Filled with the value.
 * @return        Whether the value is present and valid.
 *
 * Complexity: O(1)
 */
#define ucs_ptr_array_lookup(_ptr_array, _index, _var) \
    (((_index) >= (_ptr_array)->size) ? \
                    (UCS_V_INITIALIZED(_var), 0) : \
                    !__ucs_ptr_array_is_free(_var = (void*)((_ptr_array)->start[_index])))


/**
 * Iterate over all valid elements in the array.
 */
#define ucs_ptr_array_for_each(_var, _index, _ptr_array) \
    for (_index = 0; _index < (_ptr_array)->size; ++_index) \
         if (!__ucs_ptr_array_is_free(_var = (void*)((_ptr_array)->start[_index]))) \


#endif /* PTR_ARRAY_H_ */
