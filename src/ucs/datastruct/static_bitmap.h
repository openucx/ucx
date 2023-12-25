/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_STATIC_BITMAP_H_
#define UCS_STATIC_BITMAP_H_

#include "bitmap.h"

#include <ucs/debug/assert.h>
#include <ucs/sys/math.h>
#include <ucs/sys/preprocessor.h>

BEGIN_C_DECLS


/**
 * Define a static bitmap structure.
 *
 * @param _num_bits Number of bits in the bitmap.
 */
#define ucs_static_bitmap_s(_num_bits) \
    struct { \
        ucs_bitmap_word_t \
                bits[ucs_div_round_up(_num_bits, UCS_BITMAP_BITS_IN_WORD)]; \
    }


/**
 * Initialize a static bitmap structure to zero.
 */
#define UCS_STATIC_BITMAP_ZERO_INITIALIZER \
    { \
        .bits = { 0 } \
    }


/**
 * Get the number of words in a given bitmap
 *
 * @param _bitmap Bitmap variable to get words number
 *
 * @return Number of words
 */
#define UCS_STATIC_BITMAP_NUM_WORDS(_bitmap) \
    ucs_static_array_size((_bitmap).bits)


/* Helper macro to pass bitmap words array and number of words to a function. */
#define UCS_STATIC_BITMAP_BITS_ARGS(_bitmap_ptr) \
    ((ucs_bitmap_word_t*)(_bitmap_ptr)->bits), \
            UCS_STATIC_BITMAP_NUM_WORDS(*(_bitmap_ptr))


/* Helper macro to pass bitmap words as a constant array and number of words to
   a function. */
#define UCS_STATIC_BITMAP_BITS_CARGS(_bitmap_ptr) \
    ((const ucs_bitmap_word_t*)(_bitmap_ptr)->bits), \
            UCS_STATIC_BITMAP_NUM_WORDS(*(_bitmap_ptr))


/* Helper macro to a function on a bitmap words array and number of words, while
   converting the input argument from an r-value to an l-value */
#define _UCS_STATIC_BITMAP_FUNC_CALL(_uid, _bitmap, _func, ...) \
    ({ \
        ucs_typeof(_bitmap) _b_##_uid = (_bitmap); \
        \
        _func(UCS_STATIC_BITMAP_BITS_CARGS(&_b_##_uid), ##__VA_ARGS__); \
    })
#define UCS_STATIC_BITMAP_FUNC_CALL(_uid, _bitmap, _func, ...) \
    _UCS_STATIC_BITMAP_FUNC_CALL(_uid, _bitmap, _func, ##__VA_ARGS__)


/**
 * Reset all bitmap bits to 0.
 *
 * @param [inout] _bitmap_ptr   Bitmap to reset.
 */
#define UCS_STATIC_BITMAP_RESET_ALL(_bitmap_ptr) \
    ucs_bitmap_bits_reset_all(UCS_STATIC_BITMAP_BITS_ARGS(_bitmap_ptr))


/**
 * Set all bitmap bits to 1.
 *
 * @param [inout] _bitmap_ptr   Bitmap to set to 1.
 */
#define UCS_STATIC_BITMAP_SET_ALL(_bitmap_ptr) \
    ucs_bitmap_bits_set_all(UCS_STATIC_BITMAP_BITS_ARGS(_bitmap_ptr))


/**
 * Get the value of a bit in the bitmap.
 *
 * @param _bitmap    Read value from this bitmap.
 * @param _bit_index Bit index to read.
 *
 * @return Bit value (0 or 1)
 */
#define UCS_STATIC_BITMAP_GET(_bitmap, _bit_index) \
    UCS_STATIC_BITMAP_FUNC_CALL(UCS_PP_UNIQUE_ID, _bitmap, \
                                ucs_bitmap_bits_get, _bit_index)


/**
 * Set a bit in the bitmap to 1.
 *
 * @param _bitmap    Set value in this bitmap.
 * @param _bit_index Bit index to set.
 */
#define UCS_STATIC_BITMAP_SET(_bitmap_ptr, _bit_index) \
    ucs_bitmap_bits_set(UCS_STATIC_BITMAP_BITS_ARGS(_bitmap_ptr), _bit_index)


/**
 * Set a bit in the bitmap to 0.
 *
 * @param _bitmap    Set value in this bitmap.
 * @param _bit_index Bit index to set.
 */
#define UCS_STATIC_BITMAP_RESET(_bitmap_ptr, _bit_index) \
    ucs_bitmap_bits_reset(UCS_STATIC_BITMAP_BITS_ARGS(_bitmap_ptr), _bit_index)


/**
 * Check whether all bits of a given lvalue bitmap are set to 0.
 *
 * @param _bitmap Check bits of this bitmap.
 *
 * @return Whether this bitmap consists only of bits set to 0.
 */
#define UCS_STATIC_BITMAP_IS_ZERO(_bitmap) \
    UCS_STATIC_BITMAP_FUNC_CALL(UCS_PP_UNIQUE_ID, _bitmap, \
                                ucs_bitmap_bits_is_zero)

END_C_DECLS

#endif /* UCS_STATIC_BITMAP_H_ */
