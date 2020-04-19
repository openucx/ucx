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
#include <ucs/type/spinlock.h>
#include <ucs/debug/assert.h>

/*
 * Array element layout:
 *
 *         64                 32               1   0
 *          +-----------------+----------------+---+
 * free:    |     size_free   |  next index    | 1 |
 *          +-----------------+----------------+---+
 * used:    |           user pointer           | 0 |
 *          +-----------------+----------------+---+
 *
 *
 * size_free indicates the number of consecutive free elements ahead.
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
 */
typedef uint64_t ucs_ptr_array_elem_t;


/**
 * A sparse array of pointers.
 */
typedef struct ucs_ptr_array {
    ucs_ptr_array_elem_t     *start;
    unsigned                 freelist;
    unsigned                 size;
#ifdef ENABLE_MEMTRACK
    char                     name[64];
#endif
} ucs_ptr_array_t;


/* Flags added to lower bits of the value */
#define UCS_PTR_ARRAY_FLAG_FREE    ((unsigned long)0x01)  /* Slot is free */

#define UCS_PTR_ARRAY_SIZE_FREE_SHIFT 32
#define UCS_PTR_ARRAY_SIZE_FREE_MASK  (((ucs_ptr_array_elem_t)-1) & ~UCS_MASK(UCS_PTR_ARRAY_SIZE_FREE_SHIFT))
#define UCS_PTR_ARRAY_NEXT_SHIFT      1
#define UCS_PTR_ARRAY_NEXT_MASK       (UCS_MASK(UCS_PTR_ARRAY_SIZE_FREE_SHIFT) & ~UCS_MASK(UCS_PTR_ARRAY_NEXT_SHIFT))
#define UCS_PTR_ARRAY_SENTINEL        (UCS_PTR_ARRAY_NEXT_MASK >> UCS_PTR_ARRAY_NEXT_SHIFT)

#define __ucs_ptr_array_is_free(_elem) \
    ((uintptr_t)(_elem) & UCS_PTR_ARRAY_FLAG_FREE)


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
 *
 * @note All values should already be removed from it.
 */
void ucs_ptr_array_cleanup(ucs_ptr_array_t *ptr_array);


/**
 * Insert a pointer to the array.
 *
 * @param [in] ptr_array     Pointer to a ptr array.
 * @param [in] value         Pointer to insert. Must be 8-byte aligned.
 *
 * @return The index to which the value was inserted.
 *
 * Complexity: amortized O(1)
 *
 * @note The array will grow if needed.
 */
unsigned ucs_ptr_array_insert(ucs_ptr_array_t *ptr_array, void *value);


/**
 * Remove a pointer from the array.
 *
 * @param [in] ptr_array      Pointer to a ptr array.
 * @param [in] element_index  Index to remove from.
 *
 * Complexity: O(1)
 */
void ucs_ptr_array_remove(ucs_ptr_array_t *ptr_array, unsigned element_index);


/**
 * Replace pointer in the array
 *
 * @param [in] ptr_array      Pointer to a ptr array.
 * @param [in] element_index  Index of slot.
 * @param [in] new_val        Value to put into slot given by index.
 *
 * @return Old value of the slot
 */
void *ucs_ptr_array_replace(ucs_ptr_array_t *ptr_array, unsigned element_index,
                            void *new_val);


/**
 * Get the current size of the ptr array
 *
 * @param [in] ptr_array      Pointer to a ptr array.
 *
 * @return Size of the ptr array.
 */
static UCS_F_ALWAYS_INLINE unsigned
ucs_ptr_array_get_size(ucs_ptr_array_t *ptr_array)
{
    return ptr_array->size;
}


/**
 * Retrieve a value from the array.
 *
 * @param [in]  _ptr_array  Pointer to a ptr array.
 * @param [in]  _index      Index to retrieve the value from.
 * @param [out] _var        Filled with the value.
 *
 * @return Whether the value is present and valid.
 *
 * Complexity: O(1)
 */
#define ucs_ptr_array_lookup(_ptr_array, _index, _var) \
    (((_index) >= (_ptr_array)->size) ? \
                    (UCS_V_INITIALIZED(_var), 0) : \
                    !__ucs_ptr_array_is_free(_var = (void*)((_ptr_array)->start[_index])))


/**
 * For-each user function: Calculates how many free elements are ahead.
 *
 * @param [in] ptr_array      Pointer to a ptr array.
 * @param [in] element_index  Index of slot
 *
 * @return size_elem - The number of free elements ahead if free, if not 1.
 */
static UCS_F_ALWAYS_INLINE uint32_t
__ucs_ptr_array_for_each_get_step_size(ucs_ptr_array_t *ptr_array,
                                       unsigned element_index)
{
    uint32_t size_elem;
    ucs_ptr_array_elem_t elem = ptr_array->start[element_index];

    if (ucs_unlikely(__ucs_ptr_array_is_free((ucs_ptr_array_elem_t)elem))) {
       size_elem = (elem >> UCS_PTR_ARRAY_SIZE_FREE_SHIFT);
    } else {
       size_elem = 1;
    }
    ucs_assert(size_elem > 0);

    return size_elem;
}


/**
 * Check if element is free.
 *
 * @param [in] _elem        An element in the ptr array.
 *
 * @return 1 if the element is free and 0 if it's occupied.
 */
#define __ucs_ptr_array_is_free(_elem) ((uintptr_t)(_elem) & UCS_PTR_ARRAY_FLAG_FREE)


/**
 * Iterate over all valid elements in the array.
 *
 * @param [out] _var        Pointer to current array element in the foreach.
 * @param [out] _index      Index variable to use as iterator (unsigned).
 * @param [in]  _ptr_array  Pointer to a ptr array.
 */
#define ucs_ptr_array_for_each(_var, _index, _ptr_array) \
    for ((_index) = 0; ((_index) < (_ptr_array)->size); \
         (_index) += __ucs_ptr_array_for_each_get_step_size((_ptr_array), (_index))) \
         if ((ucs_likely(!__ucs_ptr_array_is_free( \
             (ucs_ptr_array_elem_t)((_var) = (void *)((_ptr_array)->start[(_index)]))))))


/**
 *  Locked interface
 */


/* Locked ptr array */
typedef struct ucs_ptr_array_locked {
    ucs_ptr_array_t          super;
    ucs_recursive_spinlock_t lock;
} ucs_ptr_array_locked_t;


/**
 * Locked array init
 *
 * @param [in] locked_ptr_array  Pointer to a locked ptr array.
 * @param [in] name              The name of the ptr array.
 *
 * @return Success or failure.
 */
ucs_status_t
ucs_ptr_array_locked_init(ucs_ptr_array_locked_t *locked_ptr_array,
                          const char *name);


/**
 * Cleanup the locked array.
 *
 * @param [in] locked_ptr_array    Pointer to a locked ptr array.
 *
 * @note All values should already be removed from it.
 */
void ucs_ptr_array_locked_cleanup(ucs_ptr_array_locked_t *locked_ptr_array);


/**
 * Insert a pointer to the locked array.
 *
 * @param [in] locked_ptr_array  Pointer to a locked ptr array.
 * @param [in] value             Pointer to insert. Must be 8-byte aligned.
 *
 * @return The index to which the value was inserted.
 *
 * Complexity: Amortized O(1)
 *
 * @note The array will grow if needed.
 */
unsigned ucs_ptr_array_locked_insert(ucs_ptr_array_locked_t *locked_ptr_array,
                                     void *value);


/**
 * Remove a pointer from the locked array.
 *
 * @param [in] locked_ptr_array  Pointer to a locked ptr array.
 * @param [in] element_index     Index to remove from.
 *
 * Complexity: O(1)
 */
void ucs_ptr_array_locked_remove(ucs_ptr_array_locked_t *locked_ptr_array,
                                 unsigned element_index);


/**
 * Replace pointer in the locked array
 *
 * @param [in] locked_ptr_array  Pointer to a locked ptr array.
 * @param [in] element_index     Index of slot
 * @param [in] new_val           Value to put into slot given by index
 *
 * @return Old value of the slot
 *
 * Complexity: O(1)
 */
void *ucs_ptr_array_locked_replace(ucs_ptr_array_locked_t *locked_ptr_array,
                                   unsigned element_index, void *new_val);


/** Acquire the ptr_array lock
 *
 * @param [in] _locked_ptr_array  Pointer to a locked ptr array.
 */
#define ucs_ptr_array_locked_acquire_lock(_locked_ptr_array) \
    ucs_recursive_spin_lock(&(_locked_ptr_array)->lock)


/** Release the ptr_array lock
 *
 * @param [in] _locked_ptr_array  Pointer to a locked ptr array.
 */
#define ucs_ptr_array_locked_release_lock(_locked_ptr_array) \
    ucs_recursive_spin_unlock(&(_locked_ptr_array)->lock)


/**
 * Retrieves a value from the locked array.
 *
 * @param [in]  locked_ptr_array   Pointer to a locked ptr array.
 * @param [in]  element_index      Index to retrieve the value from.
 * @param [out] var                Filled with the value.
 *
 * @return Whether the value is present and valid.
 *
 * Complexity: O(1)
 */
static UCS_F_ALWAYS_INLINE int
ucs_ptr_array_locked_lookup(ucs_ptr_array_locked_t *locked_ptr_array,
                            unsigned element_index, void **var)
{
    int present;

    ucs_ptr_array_locked_acquire_lock(locked_ptr_array);
    present = ucs_ptr_array_lookup(&locked_ptr_array->super, element_index,
                                   *var);
    ucs_ptr_array_locked_release_lock(locked_ptr_array);

    return present;
}


/**
 * Get the current size of the locked ptr array
 *
 * @param [in] locked_ptr_array      Pointer to a locked ptr array.
 *
 * @return Size of the locked ptr array.
 */
static UCS_F_ALWAYS_INLINE unsigned
ucs_ptr_array_locked_get_size(ucs_ptr_array_locked_t *locked_ptr_array)
{
    return ucs_ptr_array_get_size(&locked_ptr_array->super);
}


/**
 * If foreach locked ptr_array is finalized, releases lock.
 *
 * @param [in] locked_ptr_array   Pointer to a locked ptr array.
 * @param [in] element_index      The current for loop index.
 *
 * @return is_continue_loop for the for() loop end condition.
 */
static UCS_F_ALWAYS_INLINE int
__ucx_ptr_array_locked_foreach_finalize(ucs_ptr_array_locked_t *locked_ptr_array,
                                        uint32_t element_index)
{
    if (element_index < locked_ptr_array->super.size) {
        return 1;
    }

    ucs_ptr_array_locked_release_lock(locked_ptr_array);
    return 0;
}


/**
 * Iterate over all valid elements in the locked array.
 *
 * Please notice that using break or return are not allowed in 
 * this implementation.
 * Using break or return would require releasing the lock before by calling,
 * ucs_ptr_array_locked_release_lock(_locked_ptr_array);
 *
 * @param [out] _var                Pointer to current array element in the foreach.
 * @param [out] _index              Index variable to use as iterator (unsigned).
 * @param [in]  _locked_ptr_array   Pointer to a locked ptr array.
 */
#define ucs_ptr_array_locked_for_each(_var, _index, _locked_ptr_array) \
    for ((_index) = 0, \
         ucs_ptr_array_locked_acquire_lock(_locked_ptr_array); \
         __ucx_ptr_array_locked_foreach_finalize(_locked_ptr_array, (_index)); \
         (_index) += __ucs_ptr_array_for_each_get_step_size((&(_locked_ptr_array)->super), (_index))) \
        if ((ucs_likely(!__ucs_ptr_array_is_free( \
            (ucs_ptr_array_elem_t)((_var) = \
            (void *)((&(_locked_ptr_array)->super)->start[(_index)]))))))

#endif /* PTR_ARRAY_H_ */

