/*
 * Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

#include <ucp/core/ucp_mm.h>
#include <ucs/memory/memory_type.h>
#include <ucs/sys/topo/base/topo.h>
#include <uct/musa/base/musa_utils.h>

class test_musa_contract : public ucs::test {
};

UCS_TEST_F(test_musa_contract, memory_type_name_is_stable)
{
    EXPECT_STREQ(ucs_memory_type_names[UCS_MEMORY_TYPE_MUSA], "musa");
    EXPECT_NE(ucs_memory_type_descs[UCS_MEMORY_TYPE_MUSA], nullptr);
}

UCS_TEST_F(test_musa_contract, memory_type_is_gpu)
{
    EXPECT_NE(0, UCP_MEM_IS_GPU(UCS_MEMORY_TYPE_MUSA));
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

UCS_TEST_F(test_musa_contract, parse_pci_bus_id_with_function)
{
    ucs_sys_bus_id_t bus_id;

    ASSERT_TRUE(uct_musa_parse_pci_bus_id("0000:3b:00.2", &bus_id));
    EXPECT_EQ(0, bus_id.domain);
    EXPECT_EQ(0x3b, bus_id.bus);
    EXPECT_EQ(0, bus_id.slot);
    EXPECT_EQ(2, bus_id.function);
}

UCS_TEST_F(test_musa_contract, reject_invalid_pci_bus_id)
{
    ucs_sys_bus_id_t bus_id;

    EXPECT_FALSE(uct_musa_parse_pci_bus_id("0000:3b:00", &bus_id));
    EXPECT_FALSE(uct_musa_parse_pci_bus_id("0000:3b:00.0.trailing", &bus_id));
}
