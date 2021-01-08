/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_BITMAP_H_
#define UCS_BITMAP_H_

#include <stdint.h>
#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS


typedef uint64_t ucs_bitmap_word_t;


/*
 * Bits number in a single bitmap word
 */
#define UCS_BITMAP_BITS_IN_WORD \
    (sizeof(ucs_bitmap_word_t) * 8)


/*
 * Word index of a bit in bitmap
 *
 * @param _bit_index Index of this bit relative to the bitmap
 *
 * @return Index of the word this bit belongs to
 */
#define UCS_BITMAP_WORD_INDEX(_bit_index) \
    ((_bit_index) / UCS_BITMAP_BITS_IN_WORD)


#define _UCS_BITMAP_BIT_IN_WORD_INDEX(_bit_index) \
    ((_bit_index) % UCS_BITMAP_BITS_IN_WORD)


#define _UCS_BITMAP_BITS_TO_WORDS(_length) \
    ((((_length) + (UCS_BITMAP_BITS_IN_WORD - 1)) / UCS_BITMAP_BITS_IN_WORD))


#define _UCS_BITMAP_WORD(_bitmap, _index) \
    ((_bitmap)->bits[UCS_BITMAP_WORD_INDEX(_index)])


#define _UCS_BITMAP_WORD_INDEX0(_bit_index) \
    ((_bit_index) & ~(UCS_BITMAP_BITS_IN_WORD - 1))


#define _UCS_BITMAP_GET_NEXT_BIT(_index) \
    (-2ull << (uint64_t)((_index) & (UCS_BITMAP_BITS_IN_WORD - 1)))


/*
 * Represents an n-bit bitmap, by using an array
 * of 64-bit unsigned long integers.
 *
 * @param _length Number of bits in the bitmap
 */
#define _UCS_BITMAP_DECLARE_TYPE(_length) \
    typedef struct { \
        ucs_bitmap_word_t bits[_UCS_BITMAP_BITS_TO_WORDS(_length)]; \
    } ucs_bitmap_t(_length);


/**
 * Expands to bitmap type definition
 *
 * @param _length Number of bits (as passed to _UCS_BITMAP_DECLARE_TYPE)
 *
 * Example:
 *
 * @code{.c}
 * ucs_bitmap_t(64) my_bitmap;
 * @endcode
 */
#define ucs_bitmap_t(_length) ucs_bitmap_##_length##_suffix
    

_UCS_BITMAP_DECLARE_TYPE(64)
_UCS_BITMAP_DECLARE_TYPE(128)
_UCS_BITMAP_DECLARE_TYPE(256)


/**
 * Get the value of a bit in the bitmap
 *
 * @param _bitmap Read value from this bitmap
 * @param _index  Bit index to read
 *
 * @return Bit value (0 or 1)
 */
#define UCS_BITMAP_GET(_bitmap, _index) \
    (!!(_UCS_BITMAP_WORD(_bitmap, _index) & \
        UCS_BIT(_UCS_BITMAP_BIT_IN_WORD_INDEX(_index))))


/**
 * Set the value of a bit in the bitmap
 *
 * @param _bitmap Set value in this bitmap
 * @param _index  Bit index to set
 */
 #define UCS_BITMAP_SET(_bitmap, _index) \
    (_UCS_BITMAP_WORD(_bitmap, _index) |= \
        (UCS_BIT(_UCS_BITMAP_BIT_IN_WORD_INDEX(_index))));


/**
 * Unset (clear) the value of a bit in the bitmap
 *
 * @param _bitmap Unset value in this bitmap
 * @param _index  Bit index to unset
 */
 #define UCS_BITMAP_UNSET(_bitmap, _index) \
     ((_bitmap)->bits)[UCS_BITMAP_WORD_INDEX(_index)] &= \
        ~((UCS_BIT(_UCS_BITMAP_BIT_IN_WORD_INDEX(_index))));


/**
 * Clear a bitmap by setting all its bits to zero
 *
 * @param _bitmap Clear all bits in this bitmap
 */
#define UCS_BITMAP_CLEAR(_bitmap) \
    memset((_bitmap)->bits, 0, sizeof((_bitmap)->bits))


/**
 * Iterate over all set (1) bits of a given bitmap
 *
 * @param _bitmap Iterate over bits of this bitmap
 * @param _index  Bit index (global offset - relative to the whole bitmap)
 */
#define UCS_BITMAP_FOR_EACH_BIT(_bitmap, _index) \
    for (_index = ucs_ffs64_safe((_bitmap)->bits[0]); \
         _index < ucs_static_array_size((_bitmap)->bits) * UCS_BITMAP_BITS_IN_WORD; \
         _index = _UCS_BITMAP_WORD_INDEX0(_index) + \
                  ucs_ffs64_safe(_UCS_BITMAP_WORD((_bitmap), (_index)) & \
                                 _UCS_BITMAP_GET_NEXT_BIT(_index))) \
        if (UCS_BITMAP_GET((_bitmap), (_index)))


/**
 * Perform bitwise NOT of a bitmap
 *
 * @param _bitmap Negate this bitmap
 */
#define UCS_BITMAP_NOT(_bitmap) \
    ({ \
        int _word; \
        for (_word = 0; _word < ucs_static_array_size((_bitmap)->bits); _word++) \
            (_bitmap)->bits[_word] = ~(_bitmap)->bits[_word]; \
    })


#define _UCS_BITMAP_OP(_bitmap1, _bitmap2, _op) \
    ({ \
        int _word; \
        for (_word = 0; _word < ucs_static_array_size((_bitmap1)->bits); _word++) \
            (_bitmap1)->bits[_word] = (_bitmap1)->bits[_word] _op \
                                      (_bitmap2)->bits[_word]; \
    })


/**
 * Perform bitwise AND of 2 bitmaps, storing the result in the first one
 *
 * @param _bitmap1 First operand
 * @param _bitmap1 Second operand
 */
#define UCS_BITMAP_AND(_bitmap1, _bitmap2) _UCS_BITMAP_OP(_bitmap1, _bitmap2, &)


/**
 * Perform bitwise OR of 2 bitmaps, storing the result in the first one
 *
 * @param _bitmap1 First operand
 * @param _bitmap1 Second operand
 */
#define UCS_BITMAP_OR(_bitmap1, _bitmap2) _UCS_BITMAP_OP(_bitmap1, _bitmap2, |)


/**
 * Perform bitwise XOR of 2 bitmaps, storing the result in the first one
 *
 * @param _bitmap1 First operand
 * @param _bitmap1 Second operand
 */
#define UCS_BITMAP_XOR(_bitmap1, _bitmap2) _UCS_BITMAP_OP(_bitmap1, _bitmap2, ^)


/**
 * Copy the whole contents of a bitmap
 *
 * @param _dest_bitmap Copy bits to this bitmap
 * @param _src_bitmap  Copy bits from this bitmap
 */
#define UCS_BITMAP_COPY(_dest_bitmap, _src_bitmap) \
    memcpy((_dest_bitmap)->bits, (_src_bitmap)->bits, \
           ucs_static_array_size((_src_bitmap)->bits));


END_C_DECLS

#endif /* BITMAP_H_ */
