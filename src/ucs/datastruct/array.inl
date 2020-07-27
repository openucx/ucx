/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_ARRAY_INL_
#define UCS_ARRAY_INL_

#include <ucs/datastruct/array.h>
#include <ucs/sys/math.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>


/* Increase the array buffer length by this factor, whenever it needs to grow */
#define UCS_ARRAY_GROW_FACTOR   2


/**
 * Define the functions of an array
 *
 * @param _name        Array name
 * @param _index_type  Type of array's index
 * @param _value_type  Type of array's values
 * @param _scope       Scope for array's functions (e.g 'static inline')
 */
#define UCS_ARRAY_IMPL(_name, _index_type, _value_type, _scope) \
    \
    _scope UCS_F_MAYBE_UNUSED void \
    UCS_ARRAY_IDENTIFIER(_name, _init_dynamic)(ucs_array_t(_name) *array) \
    { \
        array->buffer   = NULL; \
        array->length   = 0; \
        array->capacity = 0; \
    } \
    \
    _scope UCS_F_MAYBE_UNUSED void \
    UCS_ARRAY_IDENTIFIER(_name, _cleanup_dynamic)(ucs_array_t(_name) *array) \
    { \
        ucs_assert(!ucs_array_is_fixed(array)); \
        ucs_free(array->buffer); \
    } \
    \
    _scope UCS_F_MAYBE_UNUSED ucs_status_t \
    UCS_ARRAY_IDENTIFIER(_name, _grow)(ucs_array_t(_name) *array, \
                                       _index_type min_capacity) \
    { \
        _index_type new_capacity; \
        _value_type *new_buffer; \
        size_t alloc_length; \
        \
        if (ucs_array_is_fixed(array)) { \
            return UCS_ERR_NO_MEMORY; \
        } \
        \
        new_capacity = ucs_max(array->capacity * UCS_ARRAY_GROW_FACTOR, \
                               min_capacity); \
        new_capacity = (new_capacity + ~UCS_ARRAY_CAP_MASK) & UCS_ARRAY_CAP_MASK; \
        \
        alloc_length = sizeof(_value_type) * new_capacity; \
        new_buffer   = (_value_type*)ucs_realloc(array->buffer, alloc_length, \
                                                 UCS_PP_MAKE_STRING(_name)); \
        if (new_buffer == NULL) { \
            ucs_error("failed to grow %s from %zu to %zu elems of '%s'", \
                      UCS_PP_MAKE_STRING(_name), (size_t)array->capacity, \
                      (size_t)new_capacity, UCS_PP_MAKE_STRING(_value_type)); \
            return UCS_ERR_NO_MEMORY; \
        } \
        \
        array->buffer   = new_buffer; \
        array->capacity = new_capacity; \
        ucs_assert(!ucs_array_is_fixed(array)); \
        return UCS_OK; \
    } \
    \
    _scope UCS_F_MAYBE_UNUSED ucs_status_t \
    UCS_ARRAY_IDENTIFIER(_name, _reserve)(ucs_array_t(_name) *array, \
                                          _index_type min_capacity) \
    { \
        if (ucs_likely(min_capacity <= ucs_array_capacity(array))) { \
        	return UCS_OK; \
        } \
        \
        return UCS_ARRAY_IDENTIFIER(_name, _grow)(array, min_capacity); \
    } \
    \
    _scope UCS_F_MAYBE_UNUSED ucs_status_t \
    UCS_ARRAY_IDENTIFIER(_name, _append)(ucs_array_t(_name) *array, \
                                         _index_type *index_p) \
    { \
        ucs_status_t status; \
        \
        status = UCS_ARRAY_IDENTIFIER(_name, _reserve)(array, array->length + 1); \
        if (ucs_unlikely(status != UCS_OK)) { \
            return status; \
        } \
        \
        *index_p = array->length++; \
        return UCS_OK; \
    }


/**
 * Declare an array and define its functions as 'static inline'
 */
#define UCS_ARRAY_DEFINE_INLINE(_name, _index_type, _value_type) \
    UCS_ARRAY_DECLARE_TYPE(_name, _index_type, _value_type) \
    UCS_ARRAY_IMPL(_name, _index_type, _value_type, static UCS_F_ALWAYS_INLINE)


#endif
