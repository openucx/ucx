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


/**
 * Find first set bit in the bitmap. If the bitmap is all zero, the result is
 * the total number of bits in the bitmap.
 *
 * @param _bitmap  Find the set bit in this bitmap.
 *
 * @return Index of the first bit set to 1.
 */
#define UCS_STATIC_BITMAP_FFS(_bitmap) \
    UCS_STATIC_BITMAP_FUNC_CALL(UCS_PP_UNIQUE_ID, _bitmap, \
                                ucs_bitmap_bits_ffs, 0)


/**
 * Find the index of the n-th bit set to 1 in a given bitmap, starting from a
 * particular index (inclusive). If all bits are zero, returns the index past
 * the last bit (bitmap size).
 *
 * @param _bitmap  Find the n-th set bit in this bitmap.
 * @param _n       Number of set bits to look up.
 *
 * @return Bit index of the n-th set bit in the bitmap.
 */
#define UCS_STATIC_BITMAP_FNS(_bitmap, _n) \
    UCS_STATIC_BITMAP_FUNC_CALL(UCS_PP_UNIQUE_ID, _bitmap, \
                                ucs_bitmap_bits_fns, 0, _n)


/**
 * Count the number of bits set to 1 in the bitmap.
 *
 * @param _bitmap   Count set bits in this bitmap.
 *
 * @return Number of bits set to 1 in the bitmap.
 */
#define UCS_STATIC_BITMAP_POPCOUNT(_bitmap) \
    UCS_STATIC_BITMAP_FUNC_CALL(UCS_PP_UNIQUE_ID, _bitmap, \
                                ucs_bitmap_bits_popcount)


/**
 * Count the number of bits set to 1 in the bitmap up to a given index.
 *
 * @param _bitmap     Count set bits in this bitmap.
 * @param _bit_index  Maximal index to count bits.
 *
 * @return Number of bits set to 1 in the bitmap up to @a bit_index.
 */
#define UCS_STATIC_BITMAP_POPCOUNT_UPTO_INDEX(_bitmap, _bit_index) \
    UCS_STATIC_BITMAP_FUNC_CALL(UCS_PP_UNIQUE_ID, _bitmap, \
                                ucs_bitmap_bits_popcount_upto_index, \
                                _bit_index)


/**
 * Copy from one bitmap to another. Destination size must be at least the source
 * size.
 *
 * @param _dst_bitmap_ptr  Pointer to destination bitmap.
 * @param _src_bitmap      Source bitmap.
 */
#define UCS_STATIC_BITMAP_COPY(_dst_bitmap_ptr, _src_bitmap) \
    UCS_STATIC_BITMAP_FUNC_CALL(UCS_PP_UNIQUE_ID, _src_bitmap, \
                                ucs_bitmap_bits_copy, \
                                UCS_STATIC_BITMAP_BITS_ARGS(_dst_bitmap_ptr))


/**
 * Fill the bitmap with a bit-mask up to the given index: bits before the given
 * index will be set to 1, and bits starting from the index onward will be set
 * to 0.
 *
 * @param _bitmap_ptr  Bitmap to fill with the bit mask.
 * @param _bit_index   Index of the first bit to set to 0.
 */
#define UCS_STATIC_BITMAP_MASK(_bitmap_ptr, _bit_index) \
    ucs_bitmap_bits_mask(UCS_STATIC_BITMAP_BITS_ARGS(_bitmap_ptr), _bit_index)


/**
 * Inverse the bits of the bitmap in-place.
 *
 * @param _bitmap   Inverse the bits of this bitmap.
 */
#define UCS_STATIC_BITMAP_NOT_INPLACE(_bitmap_ptr) \
    ucs_bitmap_bits_not(UCS_STATIC_BITMAP_BITS_ARGS(_bitmap_ptr), \
                        UCS_STATIC_BITMAP_BITS_CARGS(_bitmap_ptr))


/* Helper macro for bitmap unary operation */
#define _UCS_STATIC_BITMAP_UNARY_OP(_bitmap, _op_name, _uid) \
    ({ \
        ucs_typeof(_bitmap) _b_##_uid = (_bitmap); \
        ucs_typeof(_bitmap) _r_##_uid; \
        \
        ucs_bitmap_bits_##_op_name(UCS_STATIC_BITMAP_BITS_ARGS(&_r_##_uid), \
                                   UCS_STATIC_BITMAP_BITS_CARGS(&_b_##_uid)); \
        _r_##_uid; \
    })
#define UCS_STATIC_BITMAP_UNARY_OP(_bitmap, _op_name, _uid) \
    _UCS_STATIC_BITMAP_UNARY_OP(_bitmap, _op_name, _uid)


/**
 * Inverse the bits of the bitmap and return the resulting bitmap.
 *
 * @param _bitmap   Inverse the bits of this bitmap.
 *
 * @return Inversed bitmap.
 */
#define UCS_STATIC_BITMAP_NOT(_bitmap) \
    UCS_STATIC_BITMAP_UNARY_OP(_bitmap, not, UCS_PP_UNIQUE_ID)


/* Helper macro for bitmap binary operation */
#define _UCS_STATIC_BITMAP_BINARY_OP(_bitmap1, _bitmap2, _op_name, _uid) \
    ({ \
        ucs_typeof(_bitmap1) _b1_##_uid = (_bitmap1); \
        ucs_typeof(_bitmap2) _b2_##_uid = (_bitmap2); \
        ucs_typeof(_bitmap1) _r_##_uid; \
        \
        ucs_bitmap_bits_binary_op(UCS_STATIC_BITMAP_BITS_ARGS(&_r_##_uid), \
                                  UCS_STATIC_BITMAP_BITS_CARGS(&_b1_##_uid), \
                                  UCS_STATIC_BITMAP_BITS_CARGS(&_b2_##_uid), \
                                  ucs_bitmap_word_##_op_name); \
        _r_##_uid; \
    })
#define UCS_STATIC_BITMAP_BINARY_OP(_bitmap1, _bitmap2, _op_name, _uid) \
    _UCS_STATIC_BITMAP_BINARY_OP(_bitmap1, _bitmap2, _op_name, _uid)


/**
 * Perform bitwise "and" operation of two bitmaps and return the result.
 *
 * @param _bitmap1   First bitmap for the bitwise and.
 * @param _bitmap2   Second bitmap for the bitwise and.
 *
 * @return Resulting bitmap of the bitwise and operation.
 */
#define UCS_STATIC_BITMAP_AND(_bitmap1, _bitmap2) \
    UCS_STATIC_BITMAP_BINARY_OP(_bitmap1, _bitmap2, and, UCS_PP_UNIQUE_ID)


/**
 * Perform bitwise "or" operation of two bitmaps and return the result.
 *
 * @param _bitmap1   First bitmap for the bitwise or.
 * @param _bitmap2   Second bitmap for the bitwise or.
 *
 * @return Resulting bitmap of the bitwise and operation.
 */
#define UCS_STATIC_BITMAP_OR(_bitmap1, _bitmap2) \
    UCS_STATIC_BITMAP_BINARY_OP(_bitmap1, _bitmap2, or, UCS_PP_UNIQUE_ID)


/**
 * Perform bitwise "xor" operation of two bitmaps and return the result.
 *
 * @param _bitmap1   First bitmap for the bitwise xor.
 * @param _bitmap2   Second bitmap for the bitwise xor.
 *
 * @return Resulting bitmap of the bitwise xor operation.
 */
#define UCS_STATIC_BITMAP_XOR(_bitmap1, _bitmap2) \
    UCS_STATIC_BITMAP_BINARY_OP(_bitmap1, _bitmap2, xor, UCS_PP_UNIQUE_ID)


/* Helper function for bitmap in-place binary operation */
#define _UCS_STATIC_BITMAP_BINARY_OP_INPLACE(_bitmap1_ptr, _bitmap2, _op_name, \
                                             _uid) \
    { \
        ucs_typeof(_bitmap2) _b_##_uid = (_bitmap2); \
        \
        ucs_bitmap_bits_binary_op(UCS_STATIC_BITMAP_BITS_ARGS(_bitmap1_ptr), \
                                  UCS_STATIC_BITMAP_BITS_CARGS(_bitmap1_ptr), \
                                  UCS_STATIC_BITMAP_BITS_CARGS(&_b_##_uid), \
                                  ucs_bitmap_word_##_op_name); \
    }
#define UCS_STATIC_BITMAP_BINARY_OP_INPLACE(_bitmap1, _bitmap2, _op_name, \
                                            _uid) \
    _UCS_STATIC_BITMAP_BINARY_OP_INPLACE(_bitmap1, _bitmap2, _op_name, _uid)


/**
 * Perform bitwise "and" operation of two bitmaps and place the result in the
 * first bitmap. The destination bitmap size must be at least the source bitmap
 * size.
 *
 * @param _bitmap1_ptr  First bitmap for the bitwise and operation; the result
 *                      is placed in this bitmap.
 * @param _bitmap2      Second bitmap for the bitwise and operation.
 */
#define UCS_STATIC_BITMAP_AND_INPLACE(_bitmap1_ptr, _bitmap2) \
    UCS_STATIC_BITMAP_BINARY_OP_INPLACE(_bitmap1_ptr, _bitmap2, and, \
                                        UCS_PP_UNIQUE_ID)


/**
 * Perform bitwise "or" operation of two bitmaps and place the result in the
 * first bitmap. The destination bitmap size must be at least the source bitmap
 * size.
 *
 * @param _bitmap1_ptr  First bitmap for the bitwise or operation; the result
 *                      is placed in this bitmap.
 * @param _bitmap2      Second bitmap for the bitwise or operation.
 */
#define UCS_STATIC_BITMAP_OR_INPLACE(_bitmap1_ptr, _bitmap2) \
    UCS_STATIC_BITMAP_BINARY_OP_INPLACE(_bitmap1_ptr, _bitmap2, or, \
                                        UCS_PP_UNIQUE_ID)


/**
 * Perform bitwise "xor" operation of two bitmaps and place the result in the
 * first bitmap. The destination bitmap size must be at least the source bitmap
 * size.
 *
 * @param _bitmap1_ptr  First bitmap for the bitwise xor operation; the result
 *                      is placed in this bitmap.
 * @param _bitmap2      Second bitmap for the bitwise xor operation.
 */
#define UCS_STATIC_BITMAP_XOR_INPLACE(_bitmap1_ptr, _bitmap2) \
    UCS_STATIC_BITMAP_BINARY_OP_INPLACE(_bitmap1_ptr, _bitmap2, xor, \
                                        UCS_PP_UNIQUE_ID)


/*
 * Iterate over all set (1) bits of a given bitmap.
 *
 * @param _bit_index   Bit index (global offset - relative to the whole bitmap).
 * @param _bitmap_ptr  Iterate over bits of this bitmap.
 */
#define UCS_STATIC_BITMAP_FOR_EACH_BIT(_bit_index, _bitmap_ptr) \
    UCS_BITMAP_BITS_FOR_EACH_BIT(_bit_index, (_bitmap_ptr)->bits, \
                                 UCS_STATIC_BITMAP_NUM_WORDS(*(_bitmap_ptr)))

END_C_DECLS

#endif /* UCS_STATIC_BITMAP_H_ */
