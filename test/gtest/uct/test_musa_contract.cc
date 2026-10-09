/*
 * Copyright (C) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

#include <ucs/memory/memory_type.h>
#include <uct/musa/musa_utils.h>

class test_musa_contract : public ucs::test {
};

UCS_TEST_F(test_musa_contract, memory_type_name_is_stable)
{
    EXPECT_STREQ(ucs_memory_type_names[UCS_MEMORY_TYPE_MUSA], "musa");
    EXPECT_NE(ucs_memory_type_descs[UCS_MEMORY_TYPE_MUSA], nullptr);
    EXPECT_EQ(UCS_MEMORY_TYPE_MUSA + 1, UCS_MEMORY_TYPE_LAST);
}

UCS_TEST_F(test_musa_contract, range_accepts_interior_subrange)
{
    EXPECT_TRUE(uct_musa_range_is_valid(0x1000, 0x1000, 0x1800, 0x400));
}

UCS_TEST_F(test_musa_contract, range_rejects_cross_allocation_subrange)
{
    EXPECT_FALSE(uct_musa_range_is_valid(0x1000, 0x1000, 0x1f00, 0x200));
}

UCS_TEST_F(test_musa_contract, range_rejects_integer_overflow)
{
    EXPECT_FALSE(uct_musa_range_is_valid(UINTPTR_MAX - 0x10, 0x20,
                                         UINTPTR_MAX - 0x10, 0x20));
}

UCS_TEST_F(test_musa_contract, registration_mode_values_are_distinct)
{
    EXPECT_NE(UCT_MUSA_REG_MODE_AUTO, UCT_MUSA_REG_MODE_DMABUF);
    EXPECT_NE(UCT_MUSA_REG_MODE_DMABUF, UCT_MUSA_REG_MODE_PEERMEM);
    EXPECT_NE(UCT_MUSA_REG_MODE_PEERMEM, UCT_MUSA_REG_MODE_STAGING);
}
