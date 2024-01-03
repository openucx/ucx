
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
 * Reserve space in a bitmap, setting the newly allocated bits to 0.
 *
 * @param [inout] bitmap   Bitmap to reserve space in.
 */
static UCS_F_ALWAYS_INLINE void
ucs_dynamic_bitmap_reserve(ucs_dynamic_bitmap_t *bitmap, size_t num_bits)
{
    size_t num_words = ucs_div_round_up(num_bits, UCS_BITMAP_BITS_IN_WORD);

    if (num_words > ucs_array_length(bitmap)) {
        ucs_array_resize(bitmap, num_words, 0,
                         ucs_fatal("failed to reserve space in a bitmap"));
    }
}


/* Helper function to check if a bit_index is within the allocated size of the
   bitmap */
static UCS_F_ALWAYS_INLINE int
ucs_dynamic_bitmap_is_in_range(const ucs_dynamic_bitmap_t *bitmap,
                               size_t bit_index)
{
    return bit_index < (UCS_BITMAP_BITS_IN_WORD * ucs_array_length(bitmap));
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

END_C_DECLS

#endif
