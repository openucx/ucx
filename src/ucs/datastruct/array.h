/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_ARRAY_H_
#define UCS_ARRAY_H_

#include <ucs/sys/compiler_def.h>
#include <ucs/sys/math.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/type/status.h>
#include <stddef.h>
#include <limits.h>

BEGIN_C_DECLS


/**
 * Define an array structure.
 *
 * @param _index_type  Type of array index.
 * @param _value_type  Type of array value.
*/
#define ucs_array_s(_index_type, _value_type) \
    struct { \
        _value_type *buffer; \
        _index_type length; \
        _index_type capacity : ((CHAR_BIT * sizeof(_index_type)) - 1); \
        uint8_t     is_fixed : 1; \
    }


/**
 * Declare an array type.
 * The array keeps track of its capacity and length (current number of elements)
 * It can be either fixed-size or dynamically-growing.
 *
 * @param _array_type   Array type to define.
 * @param _index_type   Type of array's index (e.g size_t, int, ...).
 * @param _value_type   Type of array's values (could be anything).
 */
#define UCS_ARRAY_DECLARE_TYPE(_array_type, _index_type, _value_type) \
    typedef ucs_array_s(_index_type, _value_type) _array_type


/**
 * Dynamic array initializer. The array storage should be released explicitly by
 * calling @ref ucs_array_cleanup_dynamic()
 */
#define UCS_ARRAY_DYNAMIC_INITIALIZER \
    { NULL, 0, 0, 0 }


/**
 * Static initializer to create a fixed-length array with existing static buffer
 * as backing storage. Such array can track the number of elements and check for
 * overrun, and does not have to be released.
 *
 * @param _buffer    Buffer to use as backing store
 * @param _capacity  Buffer capacity
 *
 * Example:
 *
 * @code{.c}
 * int buffer[20];
 * int_array_t array =
 *          UCS_ARRAY_FIXED_INITIALIZER(buffer, ucs_static_array_size(buffer));
 * @endcode
 */
#define UCS_ARRAY_FIXED_INITIALIZER(_buffer, _capacity) \
    { (_buffer), 0, (_capacity), 1 }


/**
 * Helper macro to allocate fixed-size array on stack and check for max alloca
 * size.
 *
 * @param _array_type Array type name.
 * @param _capacity   Array capacity to allocate, must be compile-time constant.
 *
 * @return Pointer to allocated memory on stack
 */
#define UCS_ARRAY_ALLOC_ONSTACK(_array_type, _capacity) \
    ({ \
        typedef ucs_typeof(*((_array_type*)NULL)->buffer) value_t; \
        (value_t*)ucs_alloca((_capacity) * sizeof(value_t)); \
    })


/**
 * Define a fixed-size array backed by a buffer allocated on the stack.
 *
 * @param _array_type  Array type, as used in @ref UCS_ARRAY_DECLARE_TYPE
 * @param _var         Array variable
 * @param _capacity    Array capacity
 *
 * Example:
 *
 * @code{.c}
 * UCS_ARRAY_DECLARE_TYPE(int_array_t, unsigned, int)
 *
 * void my_func()
 * {
 *     UCS_ARRAY_DEFINE_ONSTACK(my_array, int_array, 20);
 * }
 * @endcode
 */
#define UCS_ARRAY_DEFINE_ONSTACK(_array_type, _var, _capacity) \
    _array_type _var = UCS_ARRAY_FIXED_INITIALIZER( \
            UCS_ARRAY_ALLOC_ONSTACK(_array_type, _capacity), (_capacity))


/**
 * Initialize a dynamic array. Such array can grow automatically to accommodate
 * for more elements.
 *
 * @param _array   Pointer to the array to initialize
 */
#define ucs_array_init_dynamic(_array) \
    { \
        (_array)->buffer   = NULL; \
        (_array)->length   = 0; \
        (_array)->capacity = 0; \
        (_array)->is_fixed = 0; \
    }


/**
 * Initialize a fixed-size array with existing buffer as backing storage.
 *
 * @param _array     Pointer to the array to initialize.
 * @param _buffer    Buffer to use as backing store.
 * @param _capacity  Buffer capacity.
 */
#define ucs_array_init_fixed(_array, _buffer, _capacity) \
    { \
        ucs_assertv((_capacity) <= ucs_array_max_capacity(_array), \
                    "capacity=%zu is out of range [0, %zu]", \
                    (size_t)(_capacity), \
                    (size_t)ucs_array_max_capacity(_array)); \
        (_array)->buffer   = (_buffer); \
        (_array)->length   = 0; \
        (_array)->capacity = (_capacity); \
        (_array)->is_fixed = 1; \
    }


/*
 * Cleanup a dynamic array.
 *
 * @param _array   Array to cleanup.
 */
#define ucs_array_cleanup_dynamic(_array) \
    { \
        ucs_assert(!ucs_array_is_fixed(_array)); \
        ucs_array_buffer_free((void*)(_array)->buffer); \
    }


/**
 * Helper function: set old buffer pointer to NULL.
 */
static UCS_F_ALWAYS_INLINE void
ucs_array_old_buffer_set_null(void **old_buffer_p)
{
    if (old_buffer_p != NULL) {
        *old_buffer_p = NULL;
    }
}


/*
 * Grow the array memory buffer to the at least required capacity. This does not
 * change the array length or existing contents, but the backing buffer can be
 * relocated to another memory area.
 *
 * @param _array         Array to reserve buffer for.
 * @param _min_capacity  Minimal capacity to reserve.
 * @param _old_buffer_p  If the array was reallocated, and this parameter is
 *                       non-NULL, the previous buffer will not be released,
 *                       instead it will be returned in *_old_buffer_p,
 *                       and the caller should copy the contents of the previous buffer 
 *                       to the new array buffer.
 *
 * @return UCS_OK if successful, UCS_ERR_NO_MEMORY if cannot allocate the array.
 */
#define ucs_array_reserve_safe(_array, _min_capacity, _old_buffer_p) \
    ({ \
        ucs_status_t _reserve_status; \
        size_t _capacity; \
        UCS_STATIC_ASSERT(ucs_is_unsigned_type(ucs_typeof((_array)->length))); \
        \
        if (ucs_likely((_min_capacity)) <= ucs_array_capacity(_array)) { \
            ucs_array_old_buffer_set_null((void**)(_old_buffer_p)); \
            _reserve_status = UCS_OK; \
        } else if (ucs_array_is_fixed(_array)) { \
            _reserve_status = UCS_ERR_NO_MEMORY; \
        } else { \
            _capacity       = (_array)->capacity; \
            _reserve_status = ucs_array_grow((void**)&(_array)->buffer, \
                                             &_capacity, _min_capacity, \
                                             ucs_array_max_capacity(_array), \
                                             sizeof(*(_array)->buffer), \
                                             (void**)(_old_buffer_p), \
                                             UCS_PP_MAKE_STRING(_array)); \
            if (_reserve_status == UCS_OK) { \
                (_array)->capacity = _capacity; \
            } \
        } \
        _reserve_status; \
    })


/*
 * Grow the array memory buffer to the at least required capacity. This does not
 * change the array length or existing contents, but the backing buffer can be
 * relocated to another memory area.
 *
 * @param _array         Array to reserve buffer for.
 * @param _min_capacity  Minimal capacity to reserve.
 *
 * @return UCS_OK if successful, UCS_ERR_NO_MEMORY if cannot allocate the array.
 */
#define ucs_array_reserve(_array, _min_capacity) \
    ucs_array_reserve_safe(_array, _min_capacity, NULL)


/*
 * Add an element to the end of the array possibly in a thread safe way.
 *
 * @param _array           Array to add element to
 * @param _old_buffer_p    If the array was reallocated, the pointer to the
 *                         previous backing buffer is stored in *_old_buffer_p,
 *                         and the user is responsible for releasing it
 *                         (for example, from a different thread). If this
 *                         parameter is NULL, the last backing buffer is
 *                         released internally by @a ucs_free.
 * @param _failed_actions  Actions to perform if the append operation failed
 *
 * @return If successful returns a pointer to the added element, otherwise NULL.
 */
#define ucs_array_append_safe(_array, _old_buffer_p, _failed_actions) \
    ({ \
        ucs_typeof((_array)->buffer) _append_elem; \
        ucs_status_t _append_status; \
        \
        _append_status = ucs_array_reserve_safe(_array, (_array)->length + 1, \
                                                _old_buffer_p); \
        if (ucs_likely(_append_status == UCS_OK)) { \
            ucs_array_set_length(_array, (_array)->length + 1); \
            _append_elem = ucs_array_last(_array); \
        } else { \
            _failed_actions; \
            _append_elem = NULL; \
        } \
        _append_elem; \
    })


/*
 * Add an element to the end of the array.
 *
 * @param _array           Array to add element to.
 * @param _failed_actions  Actions to perform if the append operation failed.
 *
 * @return If successful returns a pointer to the added element, otherwise NULL.
 */
#define ucs_array_append(_array, _failed_actions) \
    ucs_array_append_safe(_array, NULL, _failed_actions)


/*
 * Change the size of the array and initialize new elements.
 *
 * @param _array           Array to resize.
 * @param _new_length      New size for the array.
 * @param _init_value      Initialize new elements to this value.
 * @param _failed_actions  Actions to perform if the append operation failed.
 * @param _old_buffer_p    If the array was reallocated, and this parameter is
 *                         non-NULL, the previous buffer will not be released,
 *                         instead it will be returned in *_old_buffer_p,
 *                         and the caller should copy the contents of the previous buffer 
 *                         to the new array buffer.
 */
#define ucs_array_resize_safe(_array, _new_length, _init_value, \
                              _failed_actions, _old_buffer_p) \
    { \
        ucs_typeof((_array)->length) _extend_index; \
        ucs_status_t _extend_status; \
        \
        _extend_status = ucs_array_reserve_safe(_array, _new_length, \
                                                _old_buffer_p); \
        if (_extend_status == UCS_OK) { \
            for (_extend_index = (_array)->length; \
                 _extend_index < (_new_length); ++_extend_index) { \
                (_array)->buffer[_extend_index] = (_init_value); \
            } \
            ucs_array_set_length(_array, _new_length); \
        } else { \
            _failed_actions; \
        } \
    }


/*
 * Change the size of the array and initialize new elements.
 *
 * @param _array           Array to resize.
 * @param _new_length      New size for the array.
 * @param _init_value      Initialize new elements to this value.
 * @param _failed_actions  Actions to perform if the append operation failed.
 */
#define ucs_array_resize(_array, _new_length, _init_value, _failed_actions) \
    ucs_array_resize_safe(_array, _new_length, _init_value, _failed_actions, \
                          NULL)


/**
 * Add an element to the end of the array assuming it has enough space.
 *
 * @param _array    Array to add element to.
 */
#define ucs_array_append_fixed(_array) \
    ucs_array_append(_array, ucs_fatal("failed to grow array %s", \
                                       UCS_PP_MAKE_STRING(_array)))


/**
 * @return Number of elements in the array
 */
#define ucs_array_length(_array) \
    ((_array)->length)


/**
 * @return Current capacity of the array
 */
#define ucs_array_capacity(_array) \
    ((_array)->capacity)


/**
 * @return Maximum capacity of the array type.
 *
 * Since we borrow one bit from the capacity to indicate whether the array is
 * fixed-size or not, the maximum capacity range is reduced by 1 bit.
 */
#define ucs_array_max_capacity(_array) \
    UCS_MASK((CHAR_BIT * sizeof((_array)->length)) - 1)


/**
 * @return Whether this is a fixed-length array
 */
#define ucs_array_is_fixed(_array) \
    ((_array)->is_fixed)


/**
 * Get array element
 *
 * @param _array   Array whose element to retrieve
 * @param _index   Element index in the array
 *
 * @return L-value of a specified element in the array
 */
#define ucs_array_elem(_array, _index) \
    ((_array)->buffer[_index])


/**
 * @return Whether the array is empty
 */
#define ucs_array_is_empty(_array) \
    (ucs_array_length(_array) == 0)


/**
 * @return Pointer to the first array element
 */
#define ucs_array_begin(_array) \
    ((_array)->buffer)


/**
 * @return Pointer to last array element
 */
#define ucs_array_last(_array) \
    ({ \
        ucs_assert(ucs_array_length(_array) > 0); \
        (_array)->buffer + ucs_array_length(_array) - 1; \
    })


/**
 * @return Pointer to array end element (first non-valid element)
 */
#define ucs_array_end(_array) \
    ((_array)->buffer + ucs_array_length(_array))


/**
 * @return How many elements could be added within the current array capacity
 */
#define ucs_array_available_length(_array) \
    (ucs_array_capacity(_array) - ucs_array_length(_array))


/**
 * Modify array length (number of valid elements in the array)
 *
 * @param _array        Array whose length to set
 * @param _new_length   New length to set
 */
#define ucs_array_set_length(_array, _new_length) \
    { \
        ucs_assertv((_new_length) <= ucs_array_capacity(_array), \
                    "new_length=%zu capacity=%zu", \
                    (size_t)(_new_length), \
                    (size_t)ucs_array_capacity(_array)); \
        ucs_array_length(_array) = (_new_length); \
    }

/**
 * Remove all elements from array
 *
 * @param _array        Array to clean
 */
#define ucs_array_clear(_array) \
    { \
        ucs_array_length(_array) = 0; \
    }


/**
 * Remove the last element in the array (decrease length by 1)
 *
 * @param _array    Array from which to remove the last element
 */
#define ucs_array_pop_back(_array) \
    ({ \
        ucs_assert(ucs_array_length(_array) > 0); \
        --(_array)->length; \
    })


/**
 * Extract array contents and reset the array
 */
#define ucs_array_extract_buffer(_array) \
    ({ \
        ucs_typeof((_array)->buffer) _buffer = (_array)->buffer; \
        ucs_assert(!ucs_array_is_fixed(_array)); \
        ucs_array_init_dynamic(_array); \
        _buffer; \
    })


/**
 * Iterate over array elements
 *
 * @param _elem    Pointer variable to the current array element
 * @param _array   Array to iterate over
 */
#define ucs_array_for_each(_elem, _array) \
    ucs_carray_for_each(_elem, ucs_array_begin(_array), \
                        ucs_array_length(_array))


/* Internal helper function */
ucs_status_t ucs_array_grow(void **buffer_p, size_t *capacity_p,
                            size_t min_capacity, size_t max_capacity,
                            size_t value_size, void **old_buffer_p,
                            const char *array_name);


/* Internal helper function */
void ucs_array_buffer_free(void *buffer);

END_C_DECLS

#endif
