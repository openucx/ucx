/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_ARRAY_INL_
#define UCS_ARRAY_INL_

#include "array.h"


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
    _scope UCS_F_MAYBE_UNUSED ucs_status_t \
    UCS_ARRAY_IDENTIFIER(_name, _grow)(ucs_array_t(_name) *array, \
                                       _index_type min_capacity) \
    { \
        ucs_status_t status; \
        size_t capacity; \
        \
        if (ucs_array_is_fixed(array)) { \
            return UCS_ERR_NO_MEMORY; \
        } \
        \
        capacity = array->capacity; \
		status   = ucs_array_grow((void**)&array->buffer, &capacity, min_capacity, \
		                          sizeof(_value_type), UCS_PP_MAKE_STRING(_name), \
		                          UCS_PP_MAKE_STRING(_value_type)); \
		if (status != UCS_OK) { \
		    return status; \
		} \
        \
        array->capacity = capacity; \
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
    UCS_ARRAY_IDENTIFIER(_name, _append)(ucs_array_t(_name) *array) \
    { \
        ucs_status_t status; \
        \
        status = UCS_ARRAY_IDENTIFIER(_name, _reserve)(array, array->length + 1); \
        if (ucs_unlikely(status != UCS_OK)) { \
            return status; \
        } \
        \
        ++array->length; \
        return UCS_OK; \
    }


/**
 * Declare an array and define its functions as 'static inline'
 */
#define UCS_ARRAY_DEFINE_INLINE(_name, _index_type, _value_type) \
    UCS_ARRAY_DECLARE_TYPE(_name, _index_type, _value_type) \
    UCS_ARRAY_IMPL(_name, _index_type, _value_type, static UCS_F_ALWAYS_INLINE)


/* Internal helper function */
ucs_status_t ucs_array_grow(void **buffer_p, size_t *capacity_p,
                            size_t min_capacity, size_t value_size,
                            const char *array_name, const char *value_name);

#endif
