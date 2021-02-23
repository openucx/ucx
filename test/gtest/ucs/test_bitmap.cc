/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
#include <ucs/datastruct/bitmap.h>

class test_ucs_bitmap : public ucs::test {
public:
    virtual void init()
    {
        UCS_BITMAP_CLEAR(&bitmap);
    }

protected:
    void copy_bitmap(ucs_bitmap_t(128) *bitmap, uint64_t *dest)
    {
        int i;

        UCS_BITMAP_FOR_EACH_BIT(*bitmap, i) {
            dest[UCS_BITMAP_WORD_INDEX(*bitmap, i)] |= UCS_BIT(
                    i % UCS_BITMAP_BITS_IN_WORD);
        }
    }

protected:
    ucs_bitmap_t(128) bitmap;
};

void test_set_get_unset(ucs_bitmap_t(128) *bitmap, uint64_t offset)
{
    UCS_BITMAP_SET(*bitmap, offset);
    EXPECT_EQ(UCS_BITMAP_GET(*bitmap, offset), 1);
    EXPECT_EQ(bitmap->bits[offset >= UCS_BITMAP_BITS_IN_WORD], UCS_BIT(offset % 64));
    EXPECT_EQ(bitmap->bits[offset < UCS_BITMAP_BITS_IN_WORD], 0);

    UCS_BITMAP_UNSET(*bitmap, offset);
    EXPECT_EQ(bitmap->bits[0], 0);
    EXPECT_EQ(bitmap->bits[1], 0);
    EXPECT_EQ(UCS_BITMAP_GET(*bitmap, offset), 0);
}

UCS_TEST_F(test_ucs_bitmap, test_popcount) {
    int popcount = UCS_BITMAP_POPCOUNT(bitmap);
    EXPECT_EQ(popcount, 0);
    UCS_BITMAP_SET(bitmap, 12);
    UCS_BITMAP_SET(bitmap, 53);
    UCS_BITMAP_SET(bitmap, 71);
    UCS_BITMAP_SET(bitmap, 110);
    popcount = UCS_BITMAP_POPCOUNT(bitmap);
    EXPECT_EQ(popcount, 4);
}

UCS_TEST_F(test_ucs_bitmap, test_popcount_upto_index) {
    int popcount;
    UCS_BITMAP_SET(bitmap, 17);
    UCS_BITMAP_SET(bitmap, 71);
    UCS_BITMAP_SET(bitmap, 121);
    popcount = UCS_BITMAP_POPCOUNT_UPTO_INDEX(bitmap, 110);
    EXPECT_EQ(popcount, 2);

    popcount = UCS_BITMAP_POPCOUNT_UPTO_INDEX(bitmap, 20);
    EXPECT_EQ(popcount, 1);
}

UCS_TEST_F(test_ucs_bitmap, test_mask) {
    /* coverity[unsigned_compare] */
    UCS_BITMAP_MASK(&bitmap, 0);
    EXPECT_EQ(UCS_BITMAP_IS_ZERO_INPLACE(&bitmap), true);
    UCS_BITMAP_SET(bitmap, 64 + 42);
    UCS_BITMAP_MASK(&bitmap, 64 + 42);

    EXPECT_EQ(bitmap.bits[0], UINT64_MAX);
    EXPECT_EQ(bitmap.bits[1], (1ul << 42) - 1);
}

UCS_TEST_F(test_ucs_bitmap, test_set_all) {
    UCS_BITMAP_SET_ALL(bitmap);
    EXPECT_EQ(bitmap.bits[0], UINT64_MAX);
    EXPECT_EQ(bitmap.bits[1], UINT64_MAX);
}

UCS_TEST_F(test_ucs_bitmap, test_ffs) {
    size_t bit_index;

    bit_index = UCS_BITMAP_FFS(bitmap);
    EXPECT_EQ(bit_index, 128);

    UCS_BITMAP_SET(bitmap, 90);
    UCS_BITMAP_SET(bitmap, 100);
    bit_index = UCS_BITMAP_FFS(bitmap);
    EXPECT_EQ(bit_index, 90);

    UCS_BITMAP_CLEAR(&bitmap);
    UCS_BITMAP_SET(bitmap, 0);
    bit_index = UCS_BITMAP_FFS(bitmap);
    EXPECT_EQ(bit_index, 0);

    UCS_BITMAP_CLEAR(&bitmap);
    UCS_BITMAP_SET(bitmap, 64);
    bit_index = UCS_BITMAP_FFS(bitmap);
    EXPECT_EQ(bit_index, 64);
}

UCS_TEST_F(test_ucs_bitmap, test_is_zero) {
    EXPECT_TRUE(UCS_BITMAP_IS_ZERO_INPLACE(&bitmap));

    UCS_BITMAP_SET(bitmap, 71);
    EXPECT_FALSE(UCS_BITMAP_IS_ZERO_INPLACE(&bitmap));
}

UCS_TEST_F(test_ucs_bitmap, test_get_set_clear)
{
    const uint64_t offset = 15;

    EXPECT_EQ(bitmap.bits[0], 0);
    EXPECT_EQ(bitmap.bits[1], 0);
    EXPECT_EQ(UCS_BITMAP_GET(bitmap, offset), 0);

    test_set_get_unset(&bitmap, offset);
    test_set_get_unset(&bitmap, offset + 64);

    UCS_BITMAP_CLEAR(&bitmap);
    for (int i = 0; i < 128; i++) {
        EXPECT_EQ(UCS_BITMAP_GET(bitmap, i), 0);
    }
}

UCS_TEST_F(test_ucs_bitmap, test_foreach)
{
    uint64_t bitmap_words[2] = {};

    UCS_BITMAP_SET(bitmap, 1);
    UCS_BITMAP_SET(bitmap, 25);
    UCS_BITMAP_SET(bitmap, 61);

    UCS_BITMAP_SET(bitmap, UCS_BITMAP_BITS_IN_WORD + 0);
    UCS_BITMAP_SET(bitmap, UCS_BITMAP_BITS_IN_WORD + 37);
    UCS_BITMAP_SET(bitmap, UCS_BITMAP_BITS_IN_WORD + 58);

    copy_bitmap(&bitmap, bitmap_words);

    EXPECT_EQ(bitmap_words[0], UCS_BIT(1) | UCS_BIT(25) | UCS_BIT(61));
    EXPECT_EQ(bitmap_words[1], UCS_BIT(0) | UCS_BIT(37) | UCS_BIT(58));
}

UCS_TEST_F(test_ucs_bitmap, test_not)
{
    ucs_bitmap_t(128) bitmap2;

    UCS_BITMAP_SET(bitmap, 1);
    bitmap2 = UCS_BITMAP_NOT(bitmap, 128);
    UCS_BITMAP_NOT_INPLACE(&bitmap);

    EXPECT_EQ(bitmap.bits[0], -3ull);
    EXPECT_EQ(bitmap.bits[1], UINT64_MAX);
    EXPECT_EQ(bitmap2.bits[0], -3ull);
    EXPECT_EQ(bitmap2.bits[1], UINT64_MAX);
}

UCS_TEST_F(test_ucs_bitmap, test_and)
{
    ucs_bitmap_t(128) bitmap2, bitmap3;

    UCS_BITMAP_CLEAR(&bitmap2);
    UCS_BITMAP_SET(bitmap, 1);
    UCS_BITMAP_SET(bitmap, UCS_BITMAP_BITS_IN_WORD + 1);
    UCS_BITMAP_SET(bitmap, UCS_BITMAP_BITS_IN_WORD + 16);

    UCS_BITMAP_SET(bitmap2, 25);
    UCS_BITMAP_SET(bitmap2, UCS_BITMAP_BITS_IN_WORD + 1);
    UCS_BITMAP_SET(bitmap2, UCS_BITMAP_BITS_IN_WORD + 30);
    bitmap3 = UCS_BITMAP_AND(bitmap, bitmap2, 128);
    UCS_BITMAP_AND_INPLACE(&bitmap, bitmap2);

    EXPECT_EQ(bitmap.bits[0], 0);
    EXPECT_EQ(bitmap.bits[1], UCS_BIT(1));
    EXPECT_EQ(bitmap3.bits[0], 0);
    EXPECT_EQ(bitmap3.bits[1], UCS_BIT(1));
}

UCS_TEST_F(test_ucs_bitmap, test_or)
{
    ucs_bitmap_t(128) bitmap2, bitmap3;

    UCS_BITMAP_CLEAR(&bitmap2);
    UCS_BITMAP_SET(bitmap, 1);
    UCS_BITMAP_SET(bitmap, UCS_BITMAP_BITS_IN_WORD + 1);
    UCS_BITMAP_SET(bitmap, UCS_BITMAP_BITS_IN_WORD + 16);

    UCS_BITMAP_SET(bitmap2, 25);
    UCS_BITMAP_SET(bitmap2, UCS_BITMAP_BITS_IN_WORD + 1);
    UCS_BITMAP_SET(bitmap2, UCS_BITMAP_BITS_IN_WORD + 30);
    bitmap3 = UCS_BITMAP_OR(bitmap, bitmap2, 128);
    UCS_BITMAP_OR_INPLACE(&bitmap, bitmap2);

    EXPECT_EQ(bitmap.bits[0], UCS_BIT(1) | UCS_BIT(25));
    EXPECT_EQ(bitmap.bits[1], UCS_BIT(1) | UCS_BIT(16) | UCS_BIT(30));
    EXPECT_EQ(bitmap3.bits[0], UCS_BIT(1) | UCS_BIT(25));
    EXPECT_EQ(bitmap3.bits[1], UCS_BIT(1) | UCS_BIT(16) | UCS_BIT(30));
}


UCS_TEST_F(test_ucs_bitmap, test_xor)
{
    ucs_bitmap_t(128) bitmap2 = UCS_BITMAP_ZERO, bitmap3 = UCS_BITMAP_ZERO;

    bitmap.bits[0]  = 1;
    bitmap.bits[1]  = UINT64_MAX;
    bitmap2.bits[0] = UINT64_MAX;
    bitmap2.bits[1] = 1;
    bitmap3         = UCS_BITMAP_XOR(bitmap, bitmap2, 128);
    UCS_BITMAP_XOR_INPLACE(&bitmap, bitmap2);

    EXPECT_EQ(bitmap.bits[0], -2);
    EXPECT_EQ(bitmap.bits[1], -2);
    EXPECT_EQ(bitmap3.bits[0], -2);
    EXPECT_EQ(bitmap3.bits[1], -2);
}

UCS_TEST_F(test_ucs_bitmap, test_copy)
{
    ucs_bitmap_t(128) bitmap2 = UCS_BITMAP_ZERO;

    UCS_BITMAP_SET(bitmap, 1);
    UCS_BITMAP_SET(bitmap, 25);
    UCS_BITMAP_SET(bitmap, 61);

    UCS_BITMAP_SET(bitmap, UCS_BITMAP_BITS_IN_WORD + 0);
    UCS_BITMAP_SET(bitmap, UCS_BITMAP_BITS_IN_WORD + 37);
    UCS_BITMAP_SET(bitmap, UCS_BITMAP_BITS_IN_WORD + 58);

    UCS_BITMAP_COPY(bitmap2, bitmap);

    EXPECT_EQ(bitmap.bits[0], UCS_BIT(1) | UCS_BIT(25) | UCS_BIT(61));
    EXPECT_EQ(bitmap.bits[1], UCS_BIT(0) | UCS_BIT(37) | UCS_BIT(58));
}

UCS_TEST_F(test_ucs_bitmap, test_for_each_bit)
{
    int i = 0, bit_index;
    int bits[128] = {0};

    UCS_BITMAP_SET(bitmap, 0);
    UCS_BITMAP_SET(bitmap, 25);
    UCS_BITMAP_SET(bitmap, 64);
    UCS_BITMAP_SET(bitmap, 100);
    UCS_BITMAP_FOR_EACH_BIT(bitmap, bit_index) {
        i++;
        bits[bit_index]++;
    }
    
    EXPECT_EQ(i, 4);
    EXPECT_EQ(bits[0], 1);
    EXPECT_EQ(bits[25], 1);
    EXPECT_EQ(bits[64], 1);
    EXPECT_EQ(bits[100], 1);

    /* Test FOREACH on an empty bitmap */
    UCS_BITMAP_CLEAR(&bitmap);
    i = 0;
    
    UCS_BITMAP_FOR_EACH_BIT(bitmap, bit_index) {
        i++;
    }
    EXPECT_EQ(i, 0);
}

UCS_TEST_F(test_ucs_bitmap, test_for_each_bit_single_word) {
    int i         = 0;
    int bits[128] = {0};
    int bit_index;

    UCS_BITMAP_SET(bitmap, 0);
    UCS_BITMAP_SET(bitmap, 25);
    UCS_BITMAP_FOR_EACH_BIT(bitmap, bit_index) {
        i++;
        bits[bit_index]++;
    }

    EXPECT_EQ(i, 2);
    EXPECT_EQ(bits[0], 1);
    EXPECT_EQ(bits[25], 1);
}

UCS_TEST_F(test_ucs_bitmap, test_compose) {
    /* The result is irrelevant, the code only needs to compile */
    UCS_BITMAP_AND(UCS_BITMAP_NOT(bitmap, 128), bitmap, 128);
}
