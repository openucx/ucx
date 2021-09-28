/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_BITMAP_H_
#define UCS_BITMAP_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <ucs/arch/bitops.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/preprocessor.h>

BEGIN_C_DECLS


typedef uint64_t ucs_bitmap_word_t;


/*
 * Bits number in a single bitmap word
 */
#define UCS_BITMAP_BITS_IN_WORD \
    (sizeof(ucs_bitmap_word_t) * 8)


/*
 * Fully-set bitmap word
 */
#define UCS_BITMAP_WORD_MASK \
    (~((ucs_bitmap_word_t)0))


/**
 * Get the number of words in a given bitmap
 *
 * @param _bitmap Bitmap variable to get words number
 *
 * @return Number of words
 */
#define _UCS_BITMAP_NUM_WORDS(_bitmap) ucs_static_array_size((_bitmap).bits)


/**
 * Get the number of bits in a given bitmap
 *
 * @param _bitmap Bitmap variable to get bits number
 *
 * @return Number of bits
 */
#define UCS_BITMAP_NUM_BITS(_bitmap) \
    (_UCS_BITMAP_NUM_WORDS(_bitmap) * UCS_BITMAP_BITS_IN_WORD)


/**
 * Word index of a bit in bitmap. Assert the bitmap is big enough
 *
 * @param _bitmap    Index of this bit relative to the bitmap
 * @param _bit_index Index of this bit relative to the bitmap
 *
 * @return Index of the word this bit belongs to
 */
#define UCS_BITMAP_WORD_INDEX(_bitmap, _bit_index) \
    _ucs_bitmap_word_index(_UCS_BITMAP_NUM_WORDS(_bitmap), (_bit_index))


static UCS_F_ALWAYS_INLINE size_t
_ucs_bitmap_word_index(size_t bitmap_words, size_t bit_index)
{
    ucs_assert(bit_index < (bitmap_words * UCS_BITMAP_BITS_IN_WORD));
    return bit_index / UCS_BITMAP_BITS_IN_WORD;
}


#define _UCS_BITMAP_BIT_IN_WORD_INDEX(_bit_index) \
    ((_bit_index) % UCS_BITMAP_BITS_IN_WORD)


#define _UCS_BITMAP_BITS_TO_WORDS(_length) \
    ((((_length) + (UCS_BITMAP_BITS_IN_WORD - 1)) / UCS_BITMAP_BITS_IN_WORD))


#define _UCS_BITMAP_BIT_INDEX(_bit_in_word_index, _word_index) \
    ((_word_index) * UCS_BITMAP_BITS_IN_WORD + (_bit_in_word_index))


#define _UCS_BITMAP_WORD(_bitmap, _word_index) ((_bitmap).bits[_word_index])


#define _UCS_BITMAP_INDEX_IN_BOUNDS_CONDITION(_bitmap, _bit_index) \
    ((_bit_index) < _UCS_BITMAP_NUM_WORDS(_bitmap) * UCS_BITMAP_BITS_IN_WORD)


/**
 * Given a bitmap and a bit index, get the whole word that contains it
 *
 * @param _bitmap    Take the word from this bitmap
 * @param _bit_index Index of the bit for fetching the word
 *
 * @return The word which contains requested bit index
 */
#define _UCS_BITMAP_WORD_BY_BIT(_bitmap, _bit_index) \
    _UCS_BITMAP_WORD((_bitmap), UCS_BITMAP_WORD_INDEX(_bitmap, _bit_index))


#define _UCS_BITMAP_WORD_INDEX0(_bit_index) \
    ((_bit_index) & ~(UCS_BITMAP_BITS_IN_WORD - 1))


#define _UCS_BITMAP_GET_NEXT_BIT(_bit_index) \
    (-2ull << (uint64_t)((_bit_index) & (UCS_BITMAP_BITS_IN_WORD - 1)))


#define _UCS_BITMAP_FOR_EACH_WORD(_bitmap, _word_index) \
    for (_word_index = 0; _word_index < _UCS_BITMAP_NUM_WORDS(_bitmap); \
         _word_index++)


/**
 * Check whether all bits of a given lvalue bitmap are set to 0.
 *
 * @param _bitmap Check bits of this bitmap
 *
 * @return Whether this bitmap consists only of bits set to 0
 */
#define UCS_BITMAP_IS_ZERO_INPLACE(_bitmap) \
    ucs_bitmap_is_zero((_bitmap), _UCS_BITMAP_NUM_WORDS(*(_bitmap)))


/**
 * Perform inplace bitwise NOT of a bitmap
 *
 * @param _bitmap Negate this bitmap
 */
#define UCS_BITMAP_NOT_INPLACE(_bitmap) \
    { \
        size_t _word_index; \
        _UCS_BITMAP_FOR_EACH_WORD(*(_bitmap), _word_index) { \
            _UCS_BITMAP_WORD(*(_bitmap), _word_index) = \
                ~_UCS_BITMAP_WORD(*(_bitmap), _word_index); \
        } \
    }


#define _UCS_BITMAP_OP_INPLACE(_bitmap1, _bitmap2, _op) \
    { \
        ucs_typeof(*(_bitmap1)) _bitmap2_copy = (_bitmap2); \
        size_t              _word_index; \
        _UCS_BITMAP_FOR_EACH_WORD(*(_bitmap1), _word_index) { \
            _UCS_BITMAP_WORD(*(_bitmap1), _word_index) = \
                _UCS_BITMAP_WORD(*(_bitmap1), _word_index) _op \
                    _UCS_BITMAP_WORD(_bitmap2_copy, _word_index); \
        } \
    }


/**
 * Perform inplace bitwise AND of 2 bitmaps, storing the result in the first one
 *
 * @param _bitmap1 First operand
 * @param _bitmap2 Second operand
 */
#define UCS_BITMAP_AND_INPLACE(_bitmap1, _bitmap2) \
    _UCS_BITMAP_OP_INPLACE(_bitmap1, _bitmap2, &)


/**
 * Perform inplace bitwise OR of 2 bitmaps, storing the result in the first one
 *
 * @param _bitmap1 First operand
 * @param _bitmap2 Second operand
 */
#define UCS_BITMAP_OR_INPLACE(_bitmap1, _bitmap2) \
    _UCS_BITMAP_OP_INPLACE(_bitmap1, _bitmap2, |)


/**
 * Perform inplace bitwise XOR of 2 bitmaps, storing the result in the first one
 *
 * @param _bitmap1 First operand
 * @param _bitmap2 Second operand
 */
#define UCS_BITMAP_XOR_INPLACE(_bitmap1, _bitmap2) \
    _UCS_BITMAP_OP_INPLACE(_bitmap1, _bitmap2, ^)


/**
 * Check whether all bits of a given bitmap are set to 0
 *
 * @param _bitmap Check bits of this bitmap
 *
 * @return Whether this bitmap consists only of bits set to 0
 */
#define UCS_BITMAP_IS_ZERO(_bitmap, _length) \
    UCS_PP_TOKENPASTE3(_ucs_bitmap_, _length, _is_zero)(_bitmap)


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
    static inline ucs_bitmap_t(_length) \
            _ucs_bitmap_##_length##_not(ucs_bitmap_t(_length) bitmap) \
    { \
        UCS_BITMAP_NOT_INPLACE(&bitmap); \
        return bitmap; \
    } \
    \
    static inline bool _ucs_bitmap_##_length##_is_zero(ucs_bitmap_t(_length) \
                                                               bitmap) \
    { \
        return ucs_bitmap_is_zero(&bitmap, \
                                  _UCS_BITMAP_BITS_TO_WORDS(_length)); \
    } \
    \
    static inline ucs_bitmap_t(_length) \
            _ucs_bitmap_##_length##_and(ucs_bitmap_t(_length) bitmap1, \
                                        ucs_bitmap_t(_length) bitmap2) \
    { \
        UCS_BITMAP_AND_INPLACE(&bitmap1, bitmap2); \
        return bitmap1; \
    } \
    \
    static inline ucs_bitmap_t(_length) \
            _ucs_bitmap_##_length##_or(ucs_bitmap_t(_length) bitmap1, \
                                       ucs_bitmap_t(_length) bitmap2) \
    { \
        UCS_BITMAP_OR_INPLACE(&bitmap1, bitmap2); \
        return bitmap1; \
    } \
    \
    static inline ucs_bitmap_t(_length) \
            _ucs_bitmap_##_length##_xor(ucs_bitmap_t(_length) bitmap1, \
                                        ucs_bitmap_t(_length) bitmap2) \
    { \
        UCS_BITMAP_XOR_INPLACE(&bitmap1, bitmap2); \
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
#define ucs_bitmap_t(_length) UCS_PP_TOKENPASTE3(ucs_bitmap_, _length, _t)


/**
 * Get the value of a bit in the bitmap
 *
 * @param _bitmap    Read value from this bitmap
 * @param _bit_index Bit index to read
 *
 * @return Bit value (0 or 1)
 */
#define UCS_BITMAP_GET(_bitmap, _bit_index) \
    (!!(_UCS_BITMAP_WORD_BY_BIT(_bitmap, _bit_index) & \
        UCS_BIT(_UCS_BITMAP_BIT_IN_WORD_INDEX(_bit_index))))


/**
 * Set the value of a bit in the bitmap
 *
 * @param _bitmap     Set value in this bitmap
 * @param _bit_index  Bit index to set
 */
#define UCS_BITMAP_SET(_bitmap, _bit_index) \
    ({ \
        _UCS_BITMAP_WORD_BY_BIT(_bitmap, _bit_index) |= UCS_BIT( \
                _UCS_BITMAP_BIT_IN_WORD_INDEX(_bit_index)); \
    })


/**
 * Unset (clear) the value of a bit in the bitmap
 *
 * @param _bitmap    Unset value in this bitmap
 * @param _bit_index Bit index to unset
 */
#define UCS_BITMAP_UNSET(_bitmap, _bit_index) \
    ({ \
        _UCS_BITMAP_WORD_BY_BIT(_bitmap, _bit_index) &= ~( \
                UCS_BIT(_UCS_BITMAP_BIT_IN_WORD_INDEX(_bit_index))); \
    })


/**
 * Clear a bitmap by setting all its bits to zero
 *
 * @param _bitmap Clear all bits in this bitmap
 */
#define UCS_BITMAP_CLEAR(_bitmap) \
    memset((_bitmap)->bits, 0, sizeof((_bitmap)->bits))


/**
 * Initialize a bitmap by assigning all its bits to zero.
 * Use with an assignment operator
 */
#define UCS_BITMAP_ZERO \
    { \
        .bits = { 0 } \
    }


/**
 * Find the index of the first bit set to 1 in a given bitmap
 *
 * @param _bitmap Look for the first bit in this bitmap
 */
#define UCS_BITMAP_FFS(_bitmap) \
    ({ \
        size_t _bit_index = UCS_BITMAP_BITS_IN_WORD * \
                            _UCS_BITMAP_NUM_WORDS(_bitmap); \
        size_t _word_index, _temp; \
        _UCS_BITMAP_FOR_EACH_WORD(_bitmap, _word_index) { \
            _temp = _UCS_BITMAP_WORD(_bitmap, _word_index); \
            if (_temp != 0) { \
                _bit_index = ucs_ffs64(_temp) + (_word_index * \
                        UCS_BITMAP_BITS_IN_WORD); \
                break; \
            } \
        } \
        _bit_index; \
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
 *  up to a particular bit index
 *
 * @param _bitmap    Check bits number in this bitmap
 * @param _bit_index Check bits up to this bit
 *
 * @return Number of bits set to 1
 */
#define UCS_BITMAP_POPCOUNT_UPTO_INDEX(_bitmap, _bit_index) \
    ({ \
        size_t _word_index = 0, _popcount = 0; \
        _UCS_BITMAP_FOR_EACH_WORD(_bitmap, _word_index) { \
            if ((_bit_index) >= ((_word_index) + 1) * UCS_BITMAP_BITS_IN_WORD) { \
                _popcount += ucs_popcount( \
                    _UCS_BITMAP_WORD(_bitmap, _word_index)); \
            } else { \
                _popcount += ucs_popcount( \
                    _UCS_BITMAP_WORD(_bitmap, _word_index) & \
                    (UCS_MASK((_bit_index) % UCS_BITMAP_BITS_IN_WORD))); \
                    break; \
            } \
        } \
        _popcount; \
    })


/**
 *  Return a word-mask for the word at '_word_index' for all the bits up to
 *  (and not including) '_mask_index'.
 *
 * @param _bitmap     Mask bits in this bitmap
 * @param _word_index Index of the word to be masked
 * @param _mask_index Mask bits up to this bit index
 */
#define _UCS_BITMAP_MASK_WORD(_bitmap, _word_index, _mask_index) \
    ((_mask_index) > (_word_index) * UCS_BITMAP_BITS_IN_WORD) ? \
        ((((_mask_index) >= ((_word_index) + 1) * UCS_BITMAP_BITS_IN_WORD) ? \
              UCS_BITMAP_WORD_MASK : \
              UCS_MASK((_mask_index) % UCS_BITMAP_BITS_IN_WORD))) : 0; \


/**
 * Mask a bitmap by setting all bits up to a given index (excluding it) to 1
 *
 * @param _bitmap     Mask bits in this bitmap
 * @param _mask_index Mask all bits up to this index (excluding it)
 */
#define UCS_BITMAP_MASK(_bitmap, _mask_index) \
    { \
        size_t _word_index = 0; \
        \
        ucs_assert((_mask_index) < \
                   _UCS_BITMAP_NUM_WORDS(*_bitmap) * UCS_BITMAP_BITS_IN_WORD); \
        UCS_BITMAP_CLEAR(_bitmap); \
        _UCS_BITMAP_FOR_EACH_WORD(*_bitmap, _word_index) { \
            _UCS_BITMAP_WORD(*_bitmap, _word_index) = \
                    _UCS_BITMAP_MASK_WORD(*_bitmap, _word_index, (_mask_index)); \
        } \
    }


/**
 * Set all bits of a given bitmap to 1
 *
 * @param _bitmap Set bits in this bitmap
 */
#define UCS_BITMAP_SET_ALL(_bitmap) \
    { \
        size_t _word_index = 0; \
        _UCS_BITMAP_FOR_EACH_WORD(_bitmap, _word_index) { \
            _UCS_BITMAP_WORD(_bitmap, _word_index) = UCS_BITMAP_WORD_MASK; \
        } \
    }


/**
 * Iterate over all set (1) bits of a given bitmap
 *
 * @param _bitmap    Iterate over bits of this bitmap
 * @param _bit_index Bit index (global offset - relative to the whole bitmap)
 */
#define UCS_BITMAP_FOR_EACH_BIT(_bitmap, _bit_index) \
    for (_bit_index = ucs_bitmap_ffs((_bitmap).bits, \
                                     _UCS_BITMAP_NUM_WORDS(_bitmap), 0); \
         _bit_index < \
         _UCS_BITMAP_NUM_WORDS(_bitmap) * UCS_BITMAP_BITS_IN_WORD; \
         _bit_index = ucs_bitmap_ffs((_bitmap).bits, \
                                     _UCS_BITMAP_NUM_WORDS(_bitmap), \
                                     _bit_index + 1))


/**
 * Copy the whole contents of a bitmap
 *
 * @param _dest_bitmap Copy bits to this bitmap
 * @param _src_bitmap  Copy bits from this bitmap
 */
#define UCS_BITMAP_COPY(_dest_bitmap, _src_bitmap) \
    memcpy((_dest_bitmap).bits, (_src_bitmap).bits, \
           _UCS_BITMAP_NUM_WORDS(_src_bitmap));


/**
 * Perform bitwise NOT of a bitmap
 *
 * @param _bitmap Negate this bitmap
 * @param _length Length of the bitmaps (in bits)
 *
 * @return A new bitmap, which is the negation of the given one
 */
#define UCS_BITMAP_NOT(_bitmap, _length) \
    UCS_PP_TOKENPASTE3(_ucs_bitmap_, _length, _not)(_bitmap)


/**
 * Perform bitwise AND of 2 bitmaps and return the result
 *
 * @param _bitmap1 First operand
 * @param _bitmap2 Second operand
 * @param _length  Length of the bitmaps (in bits)
 *
 * @return A new bitmap, which is the logical AND of the operands
 */
#define UCS_BITMAP_AND(_bitmap1, _bitmap2, _length) \
    UCS_PP_TOKENPASTE3(_ucs_bitmap_, _length, _and) \
    (_bitmap1, _bitmap2)


/**
 * Perform bitwise OR of 2 bitmaps and return the result
 *
 * @param _bitmap1 First operand
 * @param _bitmap2 Second operand
 * @param _length  Length of the bitmaps (in bits)
 *
 * @return A new bitmap, which is the logical OR of the operands
 */
#define UCS_BITMAP_OR(_bitmap1, _bitmap2, _length) \
    UCS_PP_TOKENPASTE3(_ucs_bitmap_, _length, _or) \
    (_bitmap1, _bitmap2)


/**
 * Perform bitwise XOR of 2 bitmaps and return the result
 *
 * @param _bitmap1 First operand
 * @param _bitmap2 Second operand
 * @param _length  Length of the bitmaps (in bits)
 *
 * @return A new bitmap, which is the logical XOR of the operands
 */
#define UCS_BITMAP_XOR(_bitmap1, _bitmap2, _length) \
    UCS_PP_TOKENPASTE3(_ucs_bitmap_, _length, _xor) \
    (_bitmap1, _bitmap2)


static UCS_F_ALWAYS_INLINE bool
ucs_bitmap_is_zero(const void *bitmap, size_t num_words)
{
    size_t i;

    for (i = 0; i < num_words; i++) {
        if (((ucs_bitmap_word_t *)bitmap)[i]) {
            return 0;
        }
    }

    return 1;
}


/**
 * Find the index of the first bit set to 1 in a given bitmap, starting from
 * a particular index (excluding it). If all bits are zero, returns the index
 * past the last bit (bitmap size).
 *
 * @param bitmap_words Look for the first bit in the words of this bitmap
 * @param num_words    Number of words in the bitmap
 * @param start_index  The first bit to look from
 */
static UCS_F_ALWAYS_INLINE int
ucs_bitmap_ffs(const ucs_bitmap_word_t *bitmap_words, size_t num_words,
               size_t start_index)
{
    size_t word_index = start_index / UCS_BITMAP_BITS_IN_WORD;
    size_t mask       = ~UCS_MASK(start_index % UCS_BITMAP_BITS_IN_WORD);
    size_t first_bit_in_word;

    while (word_index < num_words) {
        if (bitmap_words[word_index] & mask) {
            first_bit_in_word = ucs_ffs64(bitmap_words[word_index] & mask);
            return _UCS_BITMAP_BIT_INDEX(first_bit_in_word, word_index);
        }

        mask = UCS_BITMAP_WORD_MASK;
        word_index++;
    }

    return _UCS_BITMAP_BIT_INDEX(0, word_index);
}


_UCS_BITMAP_DECLARE_TYPE(64)
_UCS_BITMAP_DECLARE_TYPE(128)
_UCS_BITMAP_DECLARE_TYPE(256)


END_C_DECLS

#endif /* BITMAP_H_ */
