/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
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


/* Binary operation on bitmap words */
typedef ucs_bitmap_word_t (*ucs_bitmap_binary_op_t)(ucs_bitmap_word_t,
                                                    ucs_bitmap_word_t);


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
 * Assert that bit index is inside the expected range.
 *
 * @param _bit_index  Index to check.
 * @param _num_words  Number of words in the bitmap
 * @param _cmp        Comparison operator - should be < or <=
 */
#define UCS_BITMAP_CHECK_INDEX(_bit_index, _num_words, _cmp) \
    ucs_assertv((_bit_index) _cmp (UCS_BITMAP_BITS_IN_WORD * (_num_words)), \
                "bit_index=%zu num_words=%zu", (_bit_index), (_num_words))


/**
 * Check that destination number of words is at least as large as the source.
 *
 * @param _dst_num_words   Number of words in the destination bitmap.
 * @param _src_num_words   Number of words in the source bitmap.
 */
#define UCS_BITMAP_CHECK_DST_NUM_WORDS(_dst_num_words, _src_num_words) \
    ucs_assertv((_dst_num_words) >= (_src_num_words), \
                "dst_num_words=%zu src_num_words=%zu", (_dst_num_words), \
                (_src_num_words));


/**
 * Helper macro to iterate over all set (1) bits of a given bitmap.
 *
 * @param _bit_index Bit index (global offset - relative to the whole bitmap).
 * @param _bits      Iterate over bits of this bitmap.
 * @param _num_words Number of words in the bitmap.
 */
#define UCS_BITMAP_BITS_FOR_EACH_BIT(_bit_index, _bits, _num_words) \
    for (_bit_index = ucs_bitmap_bits_ffs(_bits, _num_words, 0); \
         _bit_index < (_num_words)*UCS_BITMAP_BITS_IN_WORD; \
         _bit_index = ucs_bitmap_bits_ffs(_bits, _num_words, _bit_index + 1))


/**
 * Helper function to find the index of the first bit set to 1 in a given
 * bitmap, starting from the index @a start_index (inclusive). If all bits are
 * zero, returns the index past the last bit (bitmap size).
 *
 * @param bits         Look for the first bit in the words of this bitmap.
 * @param num_words    Number of words in the bitmap.
 * @param start_index  The first bit to look from.
 */
size_t ucs_bitmap_bits_ffs(const ucs_bitmap_word_t *bits, size_t num_words,
                           size_t start_index);


/**
 * Helper function to find the index of the @a bit_count-th bit set to 1 in a
 * given bitmap, starting from the index @a start_index (inclusive). If all bits
 * are zero, returns the index past the last bit (bitmap size).
 *
 * @param bits         Look for the first bit in the words of this bitmap.
 * @param num_words    Number of words in the bitmap.
 * @param start_index  The first bit to look from.
 * @param bit_count    Number of bits to look for. If @a bit_count is 0, returns
 *                     the first set bit after @a start_index (inclusive).
 */
size_t ucs_bitmap_bits_fns(const ucs_bitmap_word_t *bits, size_t num_words,
                           size_t start_index, size_t bit_count);


/* Helper function to set all bitmap bits to a given value, avoiding a call to
 * memset() if the value is known to be 0, to workaround a compiler warning.
 */
static UCS_F_ALWAYS_INLINE void
ucs_bitmap_bits_memset(ucs_bitmap_word_t *bits, int value, size_t num_words)
{
    if (__builtin_constant_p(num_words) && (num_words == 0)) {
        return;
    }

    memset(bits, value, num_words * sizeof(ucs_bitmap_word_t));
}


/* Helper function to set all bitmap bits to 0 */
static UCS_F_ALWAYS_INLINE void
ucs_bitmap_bits_reset_all(ucs_bitmap_word_t *bits, size_t num_words)
{
    ucs_bitmap_bits_memset(bits, 0, num_words);
}


/* Helper function to set all bitmap bits to 1 */
static UCS_F_ALWAYS_INLINE void
ucs_bitmap_bits_set_all(ucs_bitmap_word_t *bits, size_t num_words)
{
    ucs_bitmap_bits_memset(bits, 0xff, num_words);
}


/* Helper function to return work mask of bit at given index */
static UCS_F_ALWAYS_INLINE ucs_bitmap_word_t
ucs_bitmap_word_bit_mask(size_t bit_index)
{
    return UCS_BIT(bit_index % UCS_BITMAP_BITS_IN_WORD);
}


/* Helper function to return word index of a given bit index */
static UCS_F_ALWAYS_INLINE size_t ucs_bitmap_word_index(size_t bit_index,
                                                        size_t num_words)
{
    UCS_BITMAP_CHECK_INDEX(bit_index, num_words, <);
    return bit_index / UCS_BITMAP_BITS_IN_WORD;
}


/* Helper function to retrieve the value of a single bit */
static UCS_F_ALWAYS_INLINE int
ucs_bitmap_bits_get(const ucs_bitmap_word_t *bits, size_t num_words,
                    size_t bit_index)
{
    const ucs_bitmap_word_t mask = ucs_bitmap_word_bit_mask(bit_index);
    return !!(bits[ucs_bitmap_word_index(bit_index, num_words)] & mask);
}


/* Helper function to set the value of a single bit to 1 */
static UCS_F_ALWAYS_INLINE void
ucs_bitmap_bits_set(ucs_bitmap_word_t *bits, size_t num_words, size_t bit_index)
{
    const ucs_bitmap_word_t mask = ucs_bitmap_word_bit_mask(bit_index);
    bits[ucs_bitmap_word_index(bit_index, num_words)] |= mask;
}


/* Helper function to set the value of a single bit to 0 */
static UCS_F_ALWAYS_INLINE void ucs_bitmap_bits_reset(ucs_bitmap_word_t *bits,
                                                      size_t num_words,
                                                      size_t bit_index)

{
    const ucs_bitmap_word_t mask = ucs_bitmap_word_bit_mask(bit_index);
    bits[ucs_bitmap_word_index(bit_index, num_words)] &= ~mask;
}


/* Helper function to test if all bitmap bits are 0 */
static UCS_F_ALWAYS_INLINE int
ucs_bitmap_bits_is_zero(const ucs_bitmap_word_t *bits, size_t num_words)
{
    const ucs_bitmap_word_t *bits_word;

    ucs_carray_for_each(bits_word, bits, num_words) {
        if (*bits_word != 0) {
            return 0;
        }
    }
    return 1;
}


/* Helper function to count the number of bits that are set to 1 */
static UCS_F_ALWAYS_INLINE size_t
ucs_bitmap_bits_popcount(const ucs_bitmap_word_t *bits, size_t num_words)
{
    size_t popcount = 0;
    const ucs_bitmap_word_t *bits_word;

    ucs_carray_for_each(bits_word, bits, num_words) {
        popcount += ucs_popcount(*bits_word);
    }

    return popcount;
}


/* Helper function to return the number of bits that are set to 1 up to a
   given bit_index (excluding bit_index) */
static UCS_F_ALWAYS_INLINE size_t ucs_bitmap_bits_popcount_upto_index(
        const ucs_bitmap_word_t *bits, size_t num_words, size_t bit_index)
{
    const size_t last_word_index = bit_index / UCS_BITMAP_BITS_IN_WORD;
    ucs_bitmap_word_t mask       = ucs_bitmap_word_bit_mask(bit_index) - 1;

    UCS_BITMAP_CHECK_INDEX(bit_index, num_words, <=);
    return ucs_bitmap_bits_popcount(bits, last_word_index) +
           ((mask == 0) ? 0 : ucs_popcount(bits[last_word_index] & mask));
}


/* Helper function to copy between bitmap bit arrays. Destination size must
   be at least the source size. 'src' argument comes first to enable using
   this function with UCS_STATIC_BITMAP_FUNC_CALL macro. */
static UCS_F_ALWAYS_INLINE void
ucs_bitmap_bits_copy(const ucs_bitmap_word_t *src_bits, size_t src_num_words,
                     ucs_bitmap_word_t *dst_bits, size_t dst_num_words)
{
    UCS_BITMAP_CHECK_DST_NUM_WORDS(dst_num_words, src_num_words);
    /* Copy from source to destination */
    memmove(dst_bits, src_bits, src_num_words * sizeof(ucs_bitmap_word_t));
    if (dst_num_words > src_num_words) {
        /* Reset remaining bits in destination */
        ucs_bitmap_bits_reset_all(dst_bits + src_num_words,
                                  dst_num_words - src_num_words);
    }
}


/* Helper function to set the bitmap array to a mask up to the given index */
static UCS_F_ALWAYS_INLINE void ucs_bitmap_bits_mask(ucs_bitmap_word_t *bits,
                                                     size_t num_words,
                                                     size_t bit_index)
{
    size_t last_word_index;
    ucs_bitmap_word_t mask;

    UCS_BITMAP_CHECK_INDEX(bit_index, num_words, <=);

    last_word_index = bit_index / UCS_BITMAP_BITS_IN_WORD;
    ucs_bitmap_bits_set_all(bits, last_word_index);

    /* Add the mask remainder if needed */
    mask = ucs_bitmap_word_bit_mask(bit_index) - 1;
    if (mask != 0) {
        bits[last_word_index++] = mask;
    }

    ucs_assertv(num_words >= last_word_index,
                "num_words=%zu last_word_index=%zu", num_words,
                last_word_index);
    ucs_bitmap_bits_reset_all(bits + last_word_index,
                              num_words - last_word_index);
}


/* Helper function to inverse the bitmap bits */
static UCS_F_ALWAYS_INLINE void
ucs_bitmap_bits_not(ucs_bitmap_word_t *dst_bits, size_t dst_num_words,
                    const ucs_bitmap_word_t *src_bits, size_t src_num_words)
{
    size_t word_index;

    UCS_BITMAP_CHECK_DST_NUM_WORDS(dst_num_words, src_num_words);

    for (word_index = 0; word_index < src_num_words; ++word_index) {
        dst_bits[word_index] = ~src_bits[word_index];
    }

    if (dst_num_words > src_num_words) {
        /* Set remaining bits in destination to 1 */
        ucs_bitmap_bits_set_all(dst_bits + src_num_words,
                                dst_num_words - src_num_words);
    }
}


/* Helper function to do bitwise and between bitmap words */
static UCS_F_ALWAYS_INLINE ucs_bitmap_word_t
ucs_bitmap_word_and(ucs_bitmap_word_t word1, ucs_bitmap_word_t word2)
{
    return word1 & word2;
}


/* Helper function to do bitwise or between bitmap words */
static UCS_F_ALWAYS_INLINE ucs_bitmap_word_t
ucs_bitmap_word_or(ucs_bitmap_word_t word1, ucs_bitmap_word_t word2)
{
    return word1 | word2;
}


/* Helper function to do bitwise xor between bitmap words */
static UCS_F_ALWAYS_INLINE ucs_bitmap_word_t
ucs_bitmap_word_xor(ucs_bitmap_word_t word1, ucs_bitmap_word_t word2)
{
    return word1 ^ word2;
}


/* Helper function to apply a binary operation on two bitmap bit arrays and
   return the result in a third array. The destination size must be at least
   the size of each of the source arrays. */
static UCS_F_ALWAYS_INLINE void
ucs_bitmap_bits_binary_op(ucs_bitmap_word_t *dst_bits, size_t dst_num_words,
                          const ucs_bitmap_word_t *src1_bits,
                          size_t src1_num_words,
                          const ucs_bitmap_word_t *src2_bits,
                          size_t src2_num_words, ucs_bitmap_binary_op_t op)
{
    size_t word_index = 0;

    UCS_BITMAP_CHECK_DST_NUM_WORDS(dst_num_words, src1_num_words);
    UCS_BITMAP_CHECK_DST_NUM_WORDS(dst_num_words, src2_num_words);

    while ((word_index < src1_num_words) && (word_index < src2_num_words)) {
        dst_bits[word_index] = op(src1_bits[word_index], src2_bits[word_index]);
        ++word_index;
    }

    /* Non-existing bits in either src1 or src2 are considered to be 0.
       In practice, at most one of the below while loops will be executed. */
    while (word_index < src1_num_words) {
        dst_bits[word_index] = op(src1_bits[word_index], 0);
        ++word_index;
    }
    while (word_index < src2_num_words) {
        dst_bits[word_index] = op(0, src2_bits[word_index]);
        ++word_index;
    }

    /* Clear the remaining bits */
    ucs_bitmap_bits_reset_all(dst_bits + word_index,
                              dst_num_words - word_index);
}


/* Helper function to compare two bitmaps */
static UCS_F_ALWAYS_INLINE int
ucs_bitmap_bits_is_equal(const ucs_bitmap_word_t *bitmap1_bits,
                         size_t bitmap1_num_words,
                         const ucs_bitmap_word_t *bitmap2_bits,
                         size_t bitmap2_num_words)
{
    size_t min_len;
    int is_zero;

    if (bitmap1_num_words > bitmap2_num_words) {
        min_len = bitmap2_num_words;
        is_zero = ucs_bitmap_bits_is_zero(bitmap1_bits + min_len,
                                          bitmap1_num_words - min_len);
    } else {
        min_len = bitmap1_num_words;
        is_zero = ucs_bitmap_bits_is_zero(bitmap2_bits + min_len,
                                          bitmap2_num_words - min_len);
    }

    /* Remainder must be zero and overlapping words must be equal */
    return is_zero && ((min_len == 0) ||
                       (memcmp(bitmap1_bits, bitmap2_bits,
                               sizeof(ucs_bitmap_word_t) * min_len) == 0));
}

END_C_DECLS

#endif /* BITMAP_H_ */
