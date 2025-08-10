/**
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

extern "C" {
#include <uct/api/uct.h>
}
#include "uct_test.h"


class test_uct_iface : public uct_test {
protected:
    void init()
    {
        uct_test::init();
        m_entities.push_back(uct_test::create_entity(0));
    }

    entity &get_entity()
    {
        return *m_entities.front();
    }

    void test_is_reachable();

    virtual bool is_self_reachable() const
    {
        return true;
    }
};

void test_uct_iface::test_is_reachable()
{
    const auto &iface_attr = get_entity().iface_attr();
    auto iface             = get_entity().iface();
    uct_iface_is_reachable_params_t params;
    ucs_status_t status;

    std::string dev_addr(ucs_max(iface_attr.device_addr_len, 4096), '\0');
    std::string iface_addr(ucs_max(iface_attr.iface_addr_len, 4096), '\0');

    char info_str[256];
    params.field_mask         = UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR |
                                UCT_IFACE_IS_REACHABLE_FIELD_IFACE_ADDR |
                                UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING |
                                UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING_LENGTH |
                                UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR_LENGTH |
                                UCT_IFACE_IS_REACHABLE_FIELD_IFACE_ADDR_LENGTH;
    params.info_string        = info_str;
    params.info_string_length = sizeof(info_str);
    params.device_addr        = (uct_device_addr_t*)&dev_addr[0];
    params.device_addr_length = iface_attr.device_addr_len;
    params.iface_addr         = (uct_iface_addr_t*)&iface_addr[0];
    params.iface_addr_length  = iface_attr.iface_addr_len;

    status = uct_iface_get_device_address(iface,
                                          (uct_device_addr_t*)&dev_addr[0]);
    ASSERT_UCS_OK(status);

    status = uct_iface_get_address(iface, (uct_iface_addr_t*)&iface_addr[0]);
    ASSERT_UCS_OK(status);

    bool is_reachable = uct_iface_is_reachable_v2(iface, &params);
    EXPECT_EQ(is_self_reachable(), is_reachable);

    // Allocate corrupted address buffers, make it larger than the correct
    // buffer size in case the corrupted data indicates a larger address length
    // Some random buffers could still be reachable, so we fail if too many of
    // them are reachable.
    params.device_addr_length = dev_addr.size();
    params.iface_addr_length  = iface_addr.size();
    bool found_unreachable    = false;
    for (int i = 0; i < 100; ++i) {
        std::generate(dev_addr.begin(), dev_addr.end(), ucs::rand);
        std::generate(iface_addr.begin(), iface_addr.end(), ucs::rand);

        // Corrupted device and iface address should not be reachable, and should
        // provide the reason in the info string
        is_reachable = uct_iface_is_reachable_v2(iface, &params);
        if (!is_reachable) {
            if (i < 3) {
                // Print only first 3 info strings to not flood the output
                UCS_TEST_MESSAGE << info_str;
            }
            ASSERT_FALSE(std::string(info_str).empty());
            found_unreachable = true;
        }
    }

    EXPECT_TRUE(found_unreachable);
}

UCS_TEST_P(test_uct_iface, is_reachable)
{
    test_is_reachable();
}

UCT_INSTANTIATE_TEST_CASE(test_uct_iface)

class test_uct_iface_self_unreachable : public test_uct_iface {
protected:
    bool is_self_reachable() const override
    {
        return false;
    }
};

UCS_TEST_P(test_uct_iface_self_unreachable, is_reachable)
{
    test_is_reachable();
}

UCT_INSTANTIATE_CUDA_IPC_TEST_CASE(test_uct_iface_self_unreachable)
