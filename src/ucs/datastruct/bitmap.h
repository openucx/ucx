/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_BITMAP_H_
#define UCS_BITMAP_H_

#include <stdbool.h>
#include <stdint.h>
#include <ucs/arch/bitops.h>
#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS


typedef uint64_t ucs_bitmap_word_t;


/*
 * Bits number in a single bitmap word
 */
#define UCS_BITMAP_BITS_IN_WORD \
    (sizeof(ucs_bitmap_word_t) * 8)


/**
 * Get the number of words number in a given bitmap
 *
 * @param _bitmap Words number in this bitmap
 *
 * @return Number of words
 */
#define UCS_BITMAP_WORDS(_bitmap) ucs_static_array_size((_bitmap).bits)


/**
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


#define _UCS_BITMAP_WORD(_bitmap, _word_index) ((_bitmap).bits[_word_index])


#define _UCS_BITMAP_WORD_BY_BIT(_bitmap, _bit_index) \
    _UCS_BITMAP_WORD((_bitmap), UCS_BITMAP_WORD_INDEX(_bit_index))


#define _UCS_BITMAP_WORD_INDEX0(_bit_index) \
    ((_bit_index) & ~(UCS_BITMAP_BITS_IN_WORD - 1))


#define _UCS_BITMAP_GET_NEXT_BIT(_index) \
    (-2ull << (uint64_t)((_index) & (UCS_BITMAP_BITS_IN_WORD - 1)))


#define _UCS_BITMAP_FOR_EACH_WORD(_bitmap, _word_index) \
    for (_word_index = 0; _word_index < UCS_BITMAP_WORDS(_bitmap); _word_index++)


/**
 * Perform inplace bitwise NOT of a bitmap
 *
 * @param _bitmap Negate this bitmap
 */
#define UCS_BITMAP_INPLACE_NOT(_bitmap) \
    { \
        size_t _word_index; \
        _UCS_BITMAP_FOR_EACH_WORD(_bitmap, _word_index) { \
            _UCS_BITMAP_WORD(_bitmap, _word_index) = ~_UCS_BITMAP_WORD(_bitmap, _word_index); \
        } \
    }


#define _UCS_BITMAP_INPLACE_OP(_bitmap1, _bitmap2, _op) \
    { \
        size_t           _word_index; \
        typeof(_bitmap1) _bitmap2_copy = _bitmap2; \
        _UCS_BITMAP_FOR_EACH_WORD(_bitmap1, _word_index) { \
            _UCS_BITMAP_WORD(_bitmap1, _word_index) = _UCS_BITMAP_WORD(_bitmap1, _word_index) _op \
                                     _UCS_BITMAP_WORD(_bitmap2_copy, _word_index); \
        } \
    }


/**
 * Perform inplace bitwise AND of 2 bitmaps, storing the result in the first one
 *
 * @param _bitmap1 First operand
 * @param _bitmap1 Second operand
 */
#define UCS_BITMAP_INPLACE_AND(_bitmap1, _bitmap2) \
    _UCS_BITMAP_INPLACE_OP(_bitmap1, _bitmap2, &)


/**
 * Perform inplace bitwise OR of 2 bitmaps, storing the result in the first one
 *
 * @param _bitmap1 First operand
 * @param _bitmap1 Second operand
 */
#define UCS_BITMAP_INPLACE_OR(_bitmap1, _bitmap2) \
    _UCS_BITMAP_INPLACE_OP(_bitmap1, _bitmap2, |)


/**
 * Perform inplace bitwise XOR of 2 bitmaps, storing the result in the first one
 *
 * @param _bitmap1 First operand
 * @param _bitmap1 Second operand
 */
#define UCS_BITMAP_INPLACE_XOR(_bitmap1, _bitmap2) \
    _UCS_BITMAP_INPLACE_OP(_bitmap1, _bitmap2, ^)


static inline bool _ucs_bitmap_is_zero(const void *bitmap, size_t words)
{
    size_t i = 0;

    for (; i < words; i++) {
        if (((ucs_bitmap_word_t *)bitmap)[i]) {
            return false;
        }
    }

    return true;
}

/**
 * Check whether all bits of a given bitmap are set to 0
 *
 * @param _bitmap Check bits of this bitmap
 *
 * @return Whether this bitmap consists only of bits set to 0
 */
#define UCS_BITMAP_IS_ZERO(_bitmap) \
    _ucs_bitmap_is_zero(&_bitmap, UCS_BITMAP_WORDS(_bitmap))


/**
 * Represents an n-bit bitmap, by using an array
 * of 64-bit unsigned long integers.
 *
 * @param _length Number of bits in the bitmap
 */
#define _UCS_BITMAP_DECLARE_TYPE(_length) \
    typedef struct { \
        ucs_bitmap_word_t bits[_UCS_BITMAP_BITS_TO_WORDS(_length)]; \
    } ucs_bitmap_t(_length); \
\
    static inline bool ucs_bitmap_##_length##_is_zero(ucs_bitmap_t(_length) \
                                                          bitmap) \
    { \
        return UCS_BITMAP_IS_ZERO(bitmap); \
    } \
\
    /** \
     * Perform bitwise NOT of a bitmap \
     * \
     * @param _bitmap Negate this bitmap \
     * \
     * @return A new bitmap, which is the negation of the given one \
     */ \
    static inline ucs_bitmap_t(_length) \
        ucs_bitmap_##_length##_not(ucs_bitmap_t(_length) bitmap) \
    { \
        UCS_BITMAP_INPLACE_NOT(bitmap); \
        return bitmap; \
    } \
\
    /** \
     * Perform bitwise AND of 2 bitmaps and return the result \
     * \
     * @param _bitmap1 First operand \
     * @param _bitmap1 Second operand \
     * \
     * @return A new bitmap, which is the logical AND of the operands \
     */ \
    static inline ucs_bitmap_t(_length) ucs_bitmap_##_length##_and( \
        ucs_bitmap_t(_length) bitmap1, ucs_bitmap_t(_length) bitmap2) \
    { \
        UCS_BITMAP_INPLACE_AND(bitmap1, bitmap2); \
        return bitmap1; \
    } \
\
    /** \
     * Perform bitwise OR of 2 bitmaps and return the result \
     * \
     * @param _bitmap1 First operand \
     * @param _bitmap1 Second operand \
     * \
     * @return A new bitmap, which is the logical OR of the operands \
     */ \
    static inline ucs_bitmap_t(_length) ucs_bitmap_##_length##_or( \
        ucs_bitmap_t(_length) bitmap1, ucs_bitmap_t(_length) bitmap2) \
    { \
        UCS_BITMAP_INPLACE_OR(bitmap1, bitmap2); \
        return bitmap1; \
    } \
\
    /** \
     * Perform bitwise XOR of 2 bitmaps and return the result \
     * \
     * @param _bitmap1 First operand \
     * @param _bitmap1 Second operand \
     * \
     * @return A new bitmap, which is the logical XOR of the operands \
     */ \
    static inline ucs_bitmap_t(_length) ucs_bitmap_##_length##_xor( \
        ucs_bitmap_t(_length) bitmap1, ucs_bitmap_t(_length) bitmap2) \
    { \
        UCS_BITMAP_INPLACE_XOR(bitmap1, bitmap2); \
        return bitmap1; \
    }

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
    (!!(_UCS_BITMAP_WORD_BY_BIT(_bitmap, _index) & \
        UCS_BIT(_UCS_BITMAP_BIT_IN_WORD_INDEX(_index))))


/**
 * Set the value of a bit in the bitmap
 *
 * @param _bitmap Set value in this bitmap
 * @param _index  Bit index to set
 */
 #define UCS_BITMAP_SET(_bitmap, _index) \
    (_UCS_BITMAP_WORD_BY_BIT(_bitmap, _index) |= \
        (UCS_BIT(_UCS_BITMAP_BIT_IN_WORD_INDEX(_index))));


/**
 * Unset (clear) the value of a bit in the bitmap
 *
 * @param _bitmap Unset value in this bitmap
 * @param _index  Bit index to unset
 */
 #define UCS_BITMAP_UNSET(_bitmap, _index) \
     (_UCS_BITMAP_WORD_BY_BIT(_bitmap, _index) &= \
        ~(UCS_BIT(_UCS_BITMAP_BIT_IN_WORD_INDEX(_index))));


/**
 * Clear a bitmap by setting all its bits to zero
 *
 * @param _bitmap Clear all bits in this bitmap
 */
#define UCS_BITMAP_CLEAR(_bitmap) \
    memset((_bitmap).bits, 0, sizeof((_bitmap).bits))


/**
 * Find the index of the first bit set to 1 in a given bitmap
 *
 * @param _bitmap Look for the first bit in this bitmap
 */
#define UCS_BITMAP_FFS(_bitmap) \
    ({ \
        size_t _word_index, _index; \
        _UCS_BITMAP_FOR_EACH_WORD(_bitmap, _word_index) { \
            _index = ucs_ffs64_safe(_UCS_BITMAP_WORD(_bitmap, _word_index)); \
            if (_index != UCS_BITMAP_BITS_IN_WORD) { \
                _index += _word_index * UCS_BITMAP_BITS_IN_WORD; \
                break; \
            } \
        } \
        _index; \
    })


/**
 * Return the number of bits set to 1 in a given bitmap
 *
 * @param _bitmap Check bits number in this bitmap
 *
 * @return Number of bits set to 1
 */
#define UCS_BITMAP_POPCOUNT(_bitmap) \
    ({ \
        size_t _word_index = 0, _popcount = 0; \
        _UCS_BITMAP_FOR_EACH_WORD(_bitmap, _word_index) { \
            _popcount += ucs_popcount(_UCS_BITMAP_WORD(_bitmap, _word_index)); \
        } \
        _popcount; \
    })


/**
 *  Returns the number of bits set to 1 in a given bitmap,
 *  up to a particular index
 *
 * @param _bitmap Check bits number in this bitmap
 * @param _index  Check bits up to this index
 *
 * @return Number of bits set to 1
 */
#define UCS_BITMAP_POPCOUNT_UPTO_INDEX(_bitmap, _index) \
    ({ \
        size_t _word_index = 0, _popcount = 0; \
        _UCS_BITMAP_FOR_EACH_WORD(_bitmap, _word_index) { \
            if (_index >= (_word_index + 1) * UCS_BITMAP_BITS_IN_WORD) { \
                _popcount += ucs_popcount( \
                    _UCS_BITMAP_WORD(_bitmap, _word_index)); \
            } else { \
                _popcount += ucs_popcount( \
                    _UCS_BITMAP_WORD(_bitmap, _word_index) & \
                    (UCS_MASK(_index % UCS_BITMAP_BITS_IN_WORD))); \
            } \
        } \
        _popcount; \
    })


/**
 * Mask a bitmap by setting all bits up to a given index (excluding it) to 1
 *
 * @param _bitmap     Mask bits in this bitmap
 * @param _mask_index Mask all bits up to this index (excluding it)
 */
#define UCS_BITMAP_MASK(_bitmap, _mask_index) \
    { \
        size_t _word_index = 0; \
        _UCS_BITMAP_FOR_EACH_WORD(_bitmap, _word_index) { \
            _UCS_BITMAP_WORD(_bitmap, _word_index) = \
                _mask_index > _word_index * UCS_BITMAP_BITS_IN_WORD ? \
                    ((_mask_index >= \
                                (_word_index + 1) * UCS_BITMAP_BITS_IN_WORD ? \
                            -1 : \
                            UCS_MASK((_mask_index) % UCS_BITMAP_BITS_IN_WORD))) : \
                    0; \
        } \
    } \


/**
 * Set all bits of a given bitmap to 1
 *
 * @param _bitmap Set bits in this bitmap
 */
#define UCS_BITMAP_SET_ALL(_bitmap) \
    { \
        size_t _word_index = 0; \
        _UCS_BITMAP_FOR_EACH_WORD(_bitmap, _word_index) { \
            _UCS_BITMAP_WORD(_bitmap, _word_index) = -1; \
        } \
    }


/**
 * Iterate over all set (1) bits of a given bitmap
 *
 * @param _bitmap Iterate over bits of this bitmap
 * @param _index  Bit index (global offset - relative to the whole bitmap)
 */
#define UCS_BITMAP_FOR_EACH_BIT(_bitmap, _index) \
    for (_index = ucs_ffs64_safe(_UCS_BITMAP_WORD(_bitmap, 0)); \
         _index < UCS_BITMAP_WORDS(_bitmap) * UCS_BITMAP_BITS_IN_WORD; \
         _index = _UCS_BITMAP_WORD_INDEX0(_index) + \
                  ucs_ffs64_safe(_UCS_BITMAP_WORD_BY_BIT((_bitmap), (_index)) & \
                                 _UCS_BITMAP_GET_NEXT_BIT(_index))) \
        if (UCS_BITMAP_GET((_bitmap), (_index)))


/**
 * Copy the whole contents of a bitmap
 *
 * @param _dest_bitmap Copy bits to this bitmap
 * @param _src_bitmap  Copy bits from this bitmap
 */
#define UCS_BITMAP_COPY(_dest_bitmap, _src_bitmap) \
    memcpy((_dest_bitmap).bits, (_src_bitmap).bits, \
           UCS_BITMAP_WORDS(_src_bitmap));


END_C_DECLS

#endif /* BITMAP_H_ */
