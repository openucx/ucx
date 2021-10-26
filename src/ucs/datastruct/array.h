/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_ARRAY_H_
#define UCS_ARRAY_H_

#include <ucs/sys/compiler_def.h>
#include <ucs/sys/preprocessor.h>

BEGIN_C_DECLS


/**
 * Declare an array type.
 * The array keeps track of its capacity and length (current number of elements)
 * It can be either fixed-size or dynamically-growing.
 *
 * @param _name         Array definition name (needed for some array operations)
 * @param _index_type   Type of array's index (e.g size_t, int, ...)
 * @param _value_type   Type of array's values (could be anything)
 */
#define UCS_ARRAY_DECLARE_TYPE(_name, _index_type, _value_type) \
    typedef struct { \
        _value_type       *buffer; \
        _index_type       length; \
        _index_type       capacity; \
    } ucs_array_t(_name); \
    \
    typedef _value_type UCS_ARRAY_IDENTIFIER(_name, _value_type_t);


/**
 * Declare the function prototypes of an array
 *
 * @param _name        Array name
 * @param _index_type  Type of array's index
 * @param _value_type  Type of array's values
 * @param _scope       Scope for array's functions (e.g 'static inline')
 */
#define UCS_ARRAY_DECLARE_FUNCS(_name, _index_type, _value_type, _scope) \
    \
    _scope ucs_status_t \
    UCS_ARRAY_IDENTIFIER(_name, _reserve)(ucs_array_t(_name) *array, \
                                          _index_type min_capacity); \
    \
    _scope ucs_status_t \
    UCS_ARRAY_IDENTIFIER(_name, _append)(ucs_array_t(_name) *array);


/**
 * Dynamic array initializer. The array storage should be released explicitly by
 * calling @ref ucs_array_cleanup_dynamic()
 */
#define UCS_ARRAY_DYNAMIC_INITIALIZER \
    { NULL, 0, 0 }


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
 * ucs_array_t(int_array) array =
 *          UCS_ARRAY_FIXED_INITIALIZER(buffer, ucs_static_array_size(buffer));
 * @endcode
 */
#define UCS_ARRAY_FIXED_INITIALIZER(_buffer, _capacity) \
    { (_buffer), 0, ucs_array_init_fixed_capacity(_capacity) }


/**
 * Helper macro to allocate fixed-size array on stack and check for max alloca
 * size.
 *
 * @param _name      Array name to take value type from.
 * @param _capacity  Array capacity to allocate, must be compile-time constant.
 *
 * @return Pointer to allocated memory on stack
 */
#define UCS_ARRAY_ALLOC_ONSTACK(_name, _capacity) \
    ({ \
        typedef UCS_ARRAY_IDENTIFIER(_name, _value_type_t) value_t; \
        UCS_STATIC_ASSERT((_capacity) * sizeof(value_t) <= UCS_ALLOCA_MAX_SIZE); \
        (value_t*)alloca((_capacity) * sizeof(value_t)); \
    })


/**
 * Define a fixed-size array backed by a buffer allocated on the stack.
 *
 * @param _var       Array variable
 * @param _name      Array name, as used in @ref UCS_ARRAY_DECLARE_TYPE
 * @param _capacity  Array capacity
 *
 * Example:
 *
 * @code{.c}
 * UCS_ARRAY_DEFINE_INLINE(int_array, unsigned, int)
 *
 * void my_func()
 * {
 *     UCS_ARRAY_DEFINE_ONSTACK(my_array, int_array, 20);
 * }
 * @endcode
 */
#define UCS_ARRAY_DEFINE_ONSTACK(_var, _name, _capacity) \
    ucs_array_t(_name) _var = \
        UCS_ARRAY_FIXED_INITIALIZER(UCS_ARRAY_ALLOC_ONSTACK(_name, _capacity), \
                                    (_capacity))


/**
 * Expands to array type definition
 *
 * @param _name Array name (as passed to UCS_ARRAY_DECLARE)
 *
 * Example:
 *
 * @code{.c}
 * ucs_array_t(int_array) my_array;
 * @endcode
 */
#define ucs_array_t(_name) \
    UCS_ARRAY_IDENTIFIER(_name, _t)


/**
 * Helper function to initialize capacity field of a fixed-size array
 */
#define ucs_array_init_fixed_capacity(_capacity) \
    (((_capacity) & UCS_ARRAY_CAP_MASK) | UCS_ARRAY_CAP_FLAG_FIXED)



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
    }


/**
 * Initialize a fixed-size array with existing buffer as backing storage.
 *
 * @param _array     Pointer to the array to initialize
 * @param _buffer    Buffer to use as backing store
 * @param _capacity  Buffer capacity
 */
#define ucs_array_init_fixed(_array, _buffer, _capacity) \
    { \
        (_array)->buffer   = (_buffer); \
        (_array)->length   = 0; \
        (_array)->capacity = ucs_array_init_fixed_capacity(_capacity); \
    }


/*
 * Cleanup a dynamic array.
 *
 * @param _array   Array to cleanup
 */
#define ucs_array_cleanup_dynamic(_array) \
    { \
        ucs_assert(!ucs_array_is_fixed(_array)); \
        ucs_free((_array)->buffer); \
    }


/*
 * Grow the array memory buffer to the at least required capacity. This does not
 * change the array length or existing contents, but the backing buffer can be
 * relocated to another memory area.
 *
 * @param _name    Array name
 * @param _array   Array to reserve buffer for
 *
 * @return UCS_OK if successful, UCS_ERR_NO_MEMORY if cannot grow the array
 */
#define ucs_array_reserve(_name, _array, _min_capacity) \
    UCS_ARRAY_IDENTIFIER(_name, _reserve)(_array, _min_capacity)


/*
 * Add an element to the end of the array.
 *
 * @param _name     Array name
 * @param _array    Array to add element to
 *
 * @return UCS_OK if added, UCS_ERR_NO_MEMORY if cannot grow the array
 */
#define ucs_array_append(_name, _array) \
   UCS_ARRAY_IDENTIFIER(_name, _append)(_array)


/*
 * Add an element to the end of the array assuming it has enough space.
 *
 * @param _name     Array name
 * @param _array    Array to add element to
 */
#define ucs_array_append_fixed(_name, _array) \
    ({ \
        ucs_assert_always(ucs_array_append(_name, _array) == UCS_OK); \
        ucs_array_last(_array); \
    })


/**
 * @return Number of elements in the array
 */
#define ucs_array_length(_array) \
    ((_array)->length)


/**
 * @return Current capacity of the array
 */
#define ucs_array_capacity(_array) \
    ((_array)->capacity & UCS_ARRAY_CAP_MASK)


/**
 * @return Whether this is a fixed-length array
 */
#define ucs_array_is_fixed(_array) \
    ((_array)->capacity & UCS_ARRAY_CAP_FLAG_FIXED)


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
    ((_array)->buffer + ucs_array_length(_array) - 1)


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
 * Extract array contents and reset the array
 */
#define ucs_array_extract_buffer(_name, _array) \
    ({ \
        UCS_ARRAY_IDENTIFIER(_name, _value_type_t) *buffer = (_array)->buffer; \
        ucs_assert(!ucs_array_is_fixed(_array)); \
        (_array)->buffer   = NULL; \
        (_array)->length   = 0; \
        (_array)->capacity = 0; \
        buffer; \
    })


/**
 * Iterate over array elements
 *
 * @param _elem    Pointer variable to the current array element
 * @param _array   Array to iterate over
 */
#define ucs_array_for_each(_elem, _array) \
    for (_elem = ucs_array_begin(_array); _elem < ucs_array_end(_array); ++_elem)


/* Internal flag to distinguish between fixed/dynamic array */
#define UCS_ARRAY_CAP_FLAG_FIXED   UCS_BIT(0)
#define UCS_ARRAY_CAP_MASK         (~UCS_ARRAY_CAP_FLAG_FIXED)


/* Internal macro to construct array identifier from name */
#define UCS_ARRAY_IDENTIFIER(_name, _suffix) \
    UCS_PP_TOKENPASTE(ucs_array_, UCS_PP_TOKENPASTE(_name, _suffix))


END_C_DECLS

#endif
