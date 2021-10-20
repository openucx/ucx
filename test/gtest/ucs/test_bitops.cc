/**
 * Copyright (C) Huawei Technologies Co., Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <ucs/arch/bitops.h>
}

class test_bitops : public ucs::test {
};

UCS_TEST_F(test_bitops, ffs32) {
    EXPECT_EQ(0u, ucs_ffs32(0xfffff));
    EXPECT_EQ(16u, ucs_ffs32(0xf0000));
    EXPECT_EQ(1u, ucs_ffs32(0x4002));
    EXPECT_EQ(21u, ucs_ffs32(1ull << 21));
}

UCS_TEST_F(test_bitops, ffs64) {
    EXPECT_EQ(0u, ucs_ffs64(0xfffff));
    EXPECT_EQ(16u, ucs_ffs64(0xf0000));
    EXPECT_EQ(1u, ucs_ffs64(0x4002));
    EXPECT_EQ(41u, ucs_ffs64(1ull << 41));
}

UCS_TEST_F(test_bitops, ilog2) {
    EXPECT_EQ(0u, ucs_ilog2(1));
    EXPECT_EQ(2u, ucs_ilog2(4));
    EXPECT_EQ(2u, ucs_ilog2(5));
    EXPECT_EQ(2u, ucs_ilog2(7));
    EXPECT_EQ(14u, ucs_ilog2(17000));
    EXPECT_EQ(40u, ucs_ilog2(1ull << 40));
}

UCS_TEST_F(test_bitops, popcount) {
    EXPECT_EQ(0, ucs_popcount(0));
    EXPECT_EQ(2, ucs_popcount(5));
    EXPECT_EQ(16, ucs_popcount(0xffff));
    EXPECT_EQ(48, ucs_popcount(0xffffffffffffUL));
}

UCS_TEST_F(test_bitops, ctz) {
    EXPECT_EQ(0, ucs_count_trailing_zero_bits(1));
    EXPECT_EQ(28, ucs_count_trailing_zero_bits(0x10000000));
    EXPECT_EQ(32, ucs_count_trailing_zero_bits(0x100000000UL));
}

UCS_TEST_F(test_bitops, ptr_ctz) {
    uint8_t buffer[20] = {0};

    ASSERT_EQ(0, ucs_count_ptr_trailing_zero_bits(buffer, 0));
    ASSERT_EQ(1, ucs_count_ptr_trailing_zero_bits(buffer, 1));
    ASSERT_EQ(8, ucs_count_ptr_trailing_zero_bits(buffer, 8));
    ASSERT_EQ(10, ucs_count_ptr_trailing_zero_bits(buffer, 10));
    ASSERT_EQ(64, ucs_count_ptr_trailing_zero_bits(buffer, 64));
    ASSERT_EQ(70, ucs_count_ptr_trailing_zero_bits(buffer, 70));

    buffer[0] = 0x10; /* 00010000 */

    ASSERT_EQ(0, ucs_count_ptr_trailing_zero_bits(buffer, 0));
    ASSERT_EQ(1, ucs_count_ptr_trailing_zero_bits(buffer, 1));
    ASSERT_EQ(4, ucs_count_ptr_trailing_zero_bits(buffer, 8));
    ASSERT_EQ(6, ucs_count_ptr_trailing_zero_bits(buffer, 10));
    ASSERT_EQ(60, ucs_count_ptr_trailing_zero_bits(buffer, 64));
    ASSERT_EQ(66, ucs_count_ptr_trailing_zero_bits(buffer, 70));

    buffer[0] = 0x01; /* 00000001 */

    ASSERT_EQ(0, ucs_count_ptr_trailing_zero_bits(buffer, 0));
    ASSERT_EQ(1, ucs_count_ptr_trailing_zero_bits(buffer, 1));
    ASSERT_EQ(0, ucs_count_ptr_trailing_zero_bits(buffer, 8));
    ASSERT_EQ(2, ucs_count_ptr_trailing_zero_bits(buffer, 10));
    ASSERT_EQ(56, ucs_count_ptr_trailing_zero_bits(buffer, 64));
    ASSERT_EQ(62, ucs_count_ptr_trailing_zero_bits(buffer, 70));

    buffer[8] = 0x01; /* 00000001 */

    ASSERT_EQ(0, ucs_count_ptr_trailing_zero_bits(buffer, 0));
    ASSERT_EQ(1, ucs_count_ptr_trailing_zero_bits(buffer, 1));
    ASSERT_EQ(0, ucs_count_ptr_trailing_zero_bits(buffer, 8));
    ASSERT_EQ(2, ucs_count_ptr_trailing_zero_bits(buffer, 10));
    ASSERT_EQ(56, ucs_count_ptr_trailing_zero_bits(buffer, 64));
    ASSERT_EQ(62, ucs_count_ptr_trailing_zero_bits(buffer, 70));

    ASSERT_EQ(0, ucs_count_ptr_trailing_zero_bits(buffer, 72));
    ASSERT_EQ(8, ucs_count_ptr_trailing_zero_bits(buffer, 80));
    ASSERT_EQ(56, ucs_count_ptr_trailing_zero_bits(buffer, 128));
    ASSERT_EQ(88, ucs_count_ptr_trailing_zero_bits(buffer, 160));
}

UCS_TEST_F(test_bitops, is_equal) {
    uint8_t buffer1[20] = {0};
    uint8_t buffer2[20] = {0};

    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 0));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 1));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 8));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 64));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 65));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 128));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 130));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 159));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 160));

    buffer1[19] = 0x1; /* 00000001 */

    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 0));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 1));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 8));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 64));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 65));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 128));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 130));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 159));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 160));

    buffer1[19] = 0x10; /* 00010000 */

    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 0));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 1));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 8));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 64));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 65));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 128));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 130));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 159));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 160));

    buffer1[16] = 0xff; /* 11111111 */

    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 0));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 1));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 8));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 64));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 65));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 128));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 130));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 159));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 160));

    buffer1[9] = 0xff; /* 11111111 */

    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 0));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 1));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 8));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 64));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 65));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 128));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 130));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 159));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 160));

    buffer1[7] = 0xff; /* 11111111 */

    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 0));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 1));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 8));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 64));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 65));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 128));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 130));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 159));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 160));

    buffer1[1] = 0xff; /* 11111111 */

    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 0));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 1));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 8));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 64));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 65));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 128));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 130));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 159));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 160));

    buffer1[0] = 0x1; /* 00000001 */

    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 0));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 1));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 8));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 64));

    buffer2[0] = 0x1; /* 00000001 */

    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 0));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 1));
    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 8));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 64));

    buffer2[0] = 0xff; /* 11111111 */

    ASSERT_TRUE(ucs_bitwise_is_equal(buffer1, buffer2, 0));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 1));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 8));
    ASSERT_FALSE(ucs_bitwise_is_equal(buffer1, buffer2, 64));
}
