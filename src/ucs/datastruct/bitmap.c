/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "bitmap.h"

#include <inttypes.h>


/* Helper function to find the index of the the 'bit_count'-th set bit in a
   single bitmap word. This function assumes there are at least 'bit_count' set
   bits in the word, and asserts if it isn't the case. */
static UCS_F_ALWAYS_INLINE size_t ucs_bitmap_word_fns(ucs_bitmap_word_t word,
                                                      size_t bit_count)
{
    size_t bit_index;

    for (;;) {
        ucs_assertv((bit_count < UCS_BITMAP_BITS_IN_WORD) && (word != 0),
                    "word=%" PRIx64 " bit_count=%zu", word, bit_count);
        bit_index = ucs_ffs64(word);
        if (bit_count == 0) {
            return bit_index;
        }

        --bit_count;
        word &= ~UCS_BIT(bit_index);
    }
}

static UCS_F_ALWAYS_INLINE size_t
ucs_bitmap_bits_fns_inline(const ucs_bitmap_word_t *bits, size_t num_words,
                           size_t start_index, size_t bit_count)
{
    size_t word_index = start_index / UCS_BITMAP_BITS_IN_WORD;
    size_t mask       = ~(ucs_bitmap_word_bit_mask(start_index) - 1);
    ucs_bitmap_word_t masked_word;
    unsigned popcount;

    UCS_BITMAP_CHECK_INDEX(start_index, num_words, <=);
    while (word_index < num_words) {
        masked_word = bits[word_index] & mask;
        if (masked_word != 0) {
            popcount = ucs_popcount(masked_word);
            if (bit_count < popcount) {
                return (word_index * UCS_BITMAP_BITS_IN_WORD) +
                       ucs_bitmap_word_fns(masked_word, bit_count);
            }

            bit_count -= popcount;
        }

        mask = UCS_BITMAP_WORD_MASK;
        ++word_index;
    }

    return num_words * UCS_BITMAP_BITS_IN_WORD;
}

size_t ucs_bitmap_bits_ffs(const ucs_bitmap_word_t *bits, size_t num_words,
                           size_t start_index)
{
    return ucs_bitmap_bits_fns(bits, num_words, start_index, 0);
}

size_t ucs_bitmap_bits_fns(const ucs_bitmap_word_t *bits, size_t num_words,
                           size_t start_index, size_t bit_count)
{
    return ucs_bitmap_bits_fns_inline(bits, num_words, start_index, bit_count);
}
