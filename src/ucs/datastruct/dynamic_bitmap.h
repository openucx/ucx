
/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_DYNAMIC_BITMAP_H_
#define UCS_DYNAMIC_BITMAP_H_

#include <ucs/datastruct/array.h>
#include <ucs/datastruct/bitmap.h>
#include <ucs/sys/math.h>

BEGIN_C_DECLS

/* Variable-size dynamic bitmap, backed by ucs_array. */
UCS_ARRAY_DECLARE_TYPE(ucs_dynamic_bitmap_t, size_t, ucs_bitmap_word_t);


/*
 * Iterate over all set (1) bits of a given bitmap.
 *
 * @param _bit_index   Bit index (global offset - relative to the whole bitmap).
 * @param _bitmap      Iterate over bits of this bitmap.
 */
#define UCS_DYNAMIC_BITMAP_FOR_EACH_BIT(_bit_index, _bitmap) \
    UCS_BITMAP_BITS_FOR_EACH_BIT(_bit_index, ucs_array_begin(_bitmap), \
                                 ucs_array_length(_bitmap))


/**
 * Initilaize a dynamic bitmap.
 *
 * @param [out] bitmap   Bitmap to initialize.
 */
static UCS_F_ALWAYS_INLINE void
ucs_dynamic_bitmap_init(ucs_dynamic_bitmap_t *bitmap)
{
    ucs_array_init_dynamic(bitmap);
}


/**
 * Cleanup a dynamic bitmap and release its memory.
 *
 * @param [inout] bitmap   Bitmap to clean up.
 */
static UCS_F_ALWAYS_INLINE void
ucs_dynamic_bitmap_cleanup(ucs_dynamic_bitmap_t *bitmap)
{
    ucs_array_cleanup_dynamic(bitmap)
}


/**
 * Reset all bitmap bits to 0.
 *
 * @param [inout] bitmap   Bitmap to reset.
 */
static UCS_F_ALWAYS_INLINE void
ucs_dynamic_bitmap_reset_all(ucs_dynamic_bitmap_t *bitmap)
{
    ucs_bitmap_bits_reset_all(ucs_array_begin(bitmap),
                              ucs_array_length(bitmap));
}


/**
 * Helper function to reserve space in a bitmap in word granularity, setting the
 * newly allocated words to 0.
 *
 * @param [inout] bitmap     Bitmap to reserve space in.
 * @param [in]    num_words  Number of words to reserve.
 */
static UCS_F_ALWAYS_INLINE void
ucs_dynamic_bitmap_reserve_words(ucs_dynamic_bitmap_t *bitmap, size_t num_words)
{
    if (num_words > ucs_array_length(bitmap)) {
        ucs_array_resize(bitmap, num_words, 0,
                         ucs_fatal("failed to reserve space in a bitmap"));
    }
}


/**
 * Reserve space in a bitmap, setting the newly allocated bits to 0.
 *
 * @param [inout] bitmap     Bitmap to reserve space in.
 * @param [in]    num_bits   Number of bits to reserve.
 */
static UCS_F_ALWAYS_INLINE void
ucs_dynamic_bitmap_reserve(ucs_dynamic_bitmap_t *bitmap, size_t num_bits)
{
    const size_t num_words = ucs_div_round_up(num_bits,
                                              UCS_BITMAP_BITS_IN_WORD);

    ucs_dynamic_bitmap_reserve_words(bitmap, num_words);
}


/**
 * Get the number of bits in the bitmap.
 *
 * @param [in] bitmap  Find the number of bits in this bitmap.
 *
 * @return Number of defined bits in the bitmap.
 */
static UCS_F_ALWAYS_INLINE size_t
ucs_dynamic_bitmap_num_bits(const ucs_dynamic_bitmap_t *bitmap)
{
    return UCS_BITMAP_BITS_IN_WORD * ucs_array_length(bitmap);
}


/* Helper function to check if a bit_index is within the allocated size of the
   bitmap */
static UCS_F_ALWAYS_INLINE int
ucs_dynamic_bitmap_is_in_range(const ucs_dynamic_bitmap_t *bitmap,
                               size_t bit_index)
{
    return bit_index < ucs_dynamic_bitmap_num_bits(bitmap);
}


/**
 * Get the value of a bit in the bitmap.
 *
 * @param [in] bitmap    Read value from this bitmap.
 * @param [in] bit_index Bit index to read.
 *
 * @return Bit value (0 or 1)
 */
static UCS_F_ALWAYS_INLINE size_t
ucs_dynamic_bitmap_get(const ucs_dynamic_bitmap_t *bitmap, size_t bit_index)
{
    if (!ucs_dynamic_bitmap_is_in_range(bitmap, bit_index)) {
        return 0;
    }

    return ucs_bitmap_bits_get(ucs_array_begin(bitmap),
                               ucs_array_length(bitmap), bit_index);
}


/**
 * Set a bit in the bitmap to 1.
 *
 * @param [inout] bitmap    Set value in this bitmap.
 * @param [in]    bit_index Bit index to set.
 */
static UCS_F_ALWAYS_INLINE void
ucs_dynamic_bitmap_set(ucs_dynamic_bitmap_t *bitmap, size_t bit_index)
{
    ucs_dynamic_bitmap_reserve(bitmap, bit_index + 1);
    ucs_bitmap_bits_set(ucs_array_begin(bitmap), ucs_array_length(bitmap),
                        bit_index);
}


/**
 * Set a bit in the bitmap to 0.
 *
 * @param [inout] bitmap    Set value in this bitmap.
 * @param [in]    bit_index Bit index to set.
 */
static UCS_F_ALWAYS_INLINE void
ucs_dynamic_bitmap_reset(ucs_dynamic_bitmap_t *bitmap, size_t bit_index)
{
    ucs_dynamic_bitmap_reserve(bitmap, bit_index + 1);
    ucs_bitmap_bits_reset(ucs_array_begin(bitmap), ucs_array_length(bitmap),
                          bit_index);
}


/**
 * Check whether all bits of a given bitmap are set to 0.
 *
 * @param [in] bitmap Check bits of this bitmap.
 *
 * @return Whether this bitmap consists only of bits set to 0.
 */
static UCS_F_ALWAYS_INLINE int
ucs_dynamic_bitmap_is_zero(const ucs_dynamic_bitmap_t *bitmap)
{
    return ucs_bitmap_bits_is_zero(ucs_array_begin(bitmap),
                                   ucs_array_length(bitmap));
}


/**
 * Find first set bit in the bitmap. If the bitmap is all zero, the result is
 * undefined.
 *
 * @param [in] bitmap  Find the set bit in this bitmap.
 *
 * @return Index of the first bit set to 1.
 */
static UCS_F_ALWAYS_INLINE size_t
ucs_dynamic_bitmap_ffs(const ucs_dynamic_bitmap_t *bitmap)
{
    return ucs_bitmap_bits_ffs(ucs_array_begin(bitmap),
                               ucs_array_length(bitmap), 0);
}


/**
 * Find the index of the n-th bit set to 1 in a given bitmap, starting from a
 * particular index (inclusive). If all bits are zero, returns the index past
 * the last bit (bitmap size).
 *
 * @param [in] bitmap Look for the first bit in the words of this bitmap.
 * @param [in] n      Number of set bits to look up.
 *
 * @return Bit index of the n-th set bit in the bitmap.
 */
static UCS_F_ALWAYS_INLINE size_t
ucs_dynamic_bitmap_fns(const ucs_dynamic_bitmap_t *bitmap, size_t n)
{
    return ucs_bitmap_bits_fns(ucs_array_begin(bitmap),
                               ucs_array_length(bitmap), 0, n);
}


/**
 * Count the number of bits set to 1 in the bitmap.
 *
 * @param [in] bitmap   Count set bits in this bitmap.
 *
 * @return Number of bits set to 1 in the bitmap.
 */
static UCS_F_ALWAYS_INLINE size_t
ucs_dynamic_bitmap_popcount(const ucs_dynamic_bitmap_t *bitmap)
{
    return ucs_bitmap_bits_popcount(ucs_array_begin(bitmap),
                                    ucs_array_length(bitmap));
}


/**
 * Count the number of bits set to 1 in the bitmap up to a given index (not
 * including the bit at that index).
 *
 * @param [in] bitmap     Count set bits in this bitmap.
 * @param [in] bit_index  Maximal index to count bits.
 *
 * @return Number of bits set to 1 in the bitmap up to @a bit_index.
 */
static UCS_F_ALWAYS_INLINE size_t ucs_dynamic_bitmap_popcount_upto_index(
        const ucs_dynamic_bitmap_t *bitmap, size_t bit_index)
{
    return ucs_bitmap_bits_popcount_upto_index(ucs_array_begin(bitmap),
                                               ucs_array_length(bitmap),
                                               bit_index);
}


/**
 * Inverse the bits of the bitmap.
 *
 * @param [inout]  bitmap   Inverse the bits of this bitmap.
 * @param [in]     num_bits Number of bits to inverse. If this number is larger
 *                          than the number of bits in the bitmap, the bitmap is
 *                          extended to that number of bits.
 *
 * @note The parameter @a num_bits is needed to make sure the bitmap is defined
 *       for the whole required range, and not just considered as 0.
 */
static UCS_F_ALWAYS_INLINE void
ucs_dynamic_bitmap_not_inplace(ucs_dynamic_bitmap_t *bitmap, size_t num_bits)
{
    ucs_dynamic_bitmap_reserve(bitmap, num_bits);
    ucs_bitmap_bits_not(ucs_array_begin(bitmap), ucs_array_length(bitmap),
                        ucs_array_begin(bitmap), ucs_array_length(bitmap));
}


/**
 * Helper function to perform an in-place binary operation on the bitmap
 *
 * @param [inout] dest  First bitmap for the binary operation; the result
 *                      is placed in this bitmap.
 * @param [in]    src   Second bitmap for the binary operation.
 * @param [in]    op    Binary operation function to perform.
 */
static UCS_F_ALWAYS_INLINE void
ucs_dynamic_bitmap_binary_op_inplace(ucs_dynamic_bitmap_t *dest,
                                     const ucs_dynamic_bitmap_t *src,
                                     ucs_bitmap_binary_op_t op)
{
    ucs_dynamic_bitmap_reserve_words(dest, ucs_array_length(src));
    ucs_bitmap_bits_binary_op(ucs_array_begin(dest), ucs_array_length(dest),
                              ucs_array_begin(dest), ucs_array_length(dest),
                              ucs_array_begin(src), ucs_array_length(src), op);
}


/**
 * Perform bitwise "and" operation of two bitmaps and place the result in the
 * first bitmap. The destination bitmap size must be at least the source bitmap
 * size. Non-allocated bits are considered to be 0.
 *
 * @param [inout] dest  First bitmap for the bitwise and operation; the result
 *                      is placed in this bitmap.
 * @param [in]    src   Second bitmap for the bitwise and operation.
 */
static UCS_F_ALWAYS_INLINE void
ucs_dynamic_bitmap_and_inplace(ucs_dynamic_bitmap_t *dest,
                               const ucs_dynamic_bitmap_t *src)
{
    ucs_dynamic_bitmap_binary_op_inplace(dest, src, ucs_bitmap_word_and);
}


/**
 * Perform bitwise "or" operation of two bitmaps and place the result in the
 * first bitmap. The destination bitmap size must be at least the source bitmap
 * size. Non-allocated bits are considered to be 0.
 *
 * @param [inout] dest  First bitmap for the bitwise or operation; the result
 *                      is placed in this bitmap.
 * @param [in]    src   Second bitmap for the bitwise or operation.
 */
static UCS_F_ALWAYS_INLINE void
ucs_dynamic_bitmap_or_inplace(ucs_dynamic_bitmap_t *dest,
                              const ucs_dynamic_bitmap_t *src)
{
    ucs_dynamic_bitmap_binary_op_inplace(dest, src, ucs_bitmap_word_or);
}


/**
 * Perform bitwise "xor" operation of two bitmaps and place the result in the
 * first bitmap. The destination bitmap size must be at least the source bitmap
 * size. Non-allocated bits are considered to be 0.
 *
 * @param [inout] dest  First bitmap for the bitwise xor operation; the result
 *                      is placed in this bitmap.
 * @param [in]    src   Second bitmap for the bitwise xor operation.
 */
static UCS_F_ALWAYS_INLINE void
ucs_dynamic_bitmap_xor_inplace(ucs_dynamic_bitmap_t *dest,
                               const ucs_dynamic_bitmap_t *src)
{
    ucs_dynamic_bitmap_binary_op_inplace(dest, src, ucs_bitmap_word_xor);
}


/**
 * Compare two bitmaps. Non-allocated bits are considered to be 0.
 *
 * @param [in] bitmap1   First bitmap to compare.
 * @param [in] bitmap2   Second bitmap to compare.
 *
 * @return Nonzero if the bitmaps are equal, zero if they are different.
*/
static UCS_F_ALWAYS_INLINE int
ucs_dynamic_bitmap_is_equal(const ucs_dynamic_bitmap_t *bitmap1,
                            const ucs_dynamic_bitmap_t *bitmap2)
{
    return ucs_bitmap_bits_is_equal(ucs_array_begin(bitmap1),
                                    ucs_array_length(bitmap1),
                                    ucs_array_begin(bitmap2),
                                    ucs_array_length(bitmap2));
}

END_C_DECLS

#endif
