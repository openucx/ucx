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

private:
    entity *m_entity{nullptr};
};

void test_uct_iface::test_is_reachable()
{
    const auto &iface_attr = get_entity().iface_attr();
    auto iface             = get_entity().iface();
    uct_iface_is_reachable_params_t params;
    ucs_status_t status;

    char info_str[256];
    params.field_mask         = UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR |
                                UCT_IFACE_IS_REACHABLE_FIELD_IFACE_ADDR |
                                UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING |
                                UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING_LENGTH |
                                UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR_LENGTH |
                                UCT_IFACE_IS_REACHABLE_FIELD_IFACE_ADDR_LENGTH;
    params.info_string        = info_str;
    params.info_string_length = sizeof(info_str);
    params.device_addr_length = iface_attr.device_addr_len;
    params.iface_addr_length  = iface_attr.iface_addr_len;

    auto dev_addr = (uct_device_addr_t*)malloc(iface_attr.device_addr_len);
    status        = uct_iface_get_device_address(iface, dev_addr);
    ASSERT_UCS_OK(status);
    params.device_addr = dev_addr;

    auto iface_addr = (uct_iface_addr_t*)malloc(iface_attr.iface_addr_len);
    status          = uct_iface_get_address(iface, iface_addr);
    ASSERT_UCS_OK(status);
    params.iface_addr = iface_addr;

    bool is_reachable = uct_iface_is_reachable_v2(iface, &params);
    EXPECT_EQ(is_self_reachable(), is_reachable);

    free(iface_addr);
    free(dev_addr);

    // Allocate corrupted address buffers, make it larger than the correct
    // buffer size in case the corrupted data indicates a larger address length

    size_t invalid_dev_addr_len = ucs_min(4096, iface_attr.device_addr_len);
    dev_addr = (uct_device_addr_t*)malloc(invalid_dev_addr_len);
    ucs::fill_random((uint8_t*)dev_addr, invalid_dev_addr_len);
    params.device_addr = dev_addr;

    size_t invalid_iface_addr_len = ucs_min(4096, iface_attr.iface_addr_len);
    iface_addr = (uct_iface_addr_t*)malloc(invalid_iface_addr_len);
    ucs::fill_random((uint8_t*)iface_addr, invalid_iface_addr_len);
    params.iface_addr = iface_addr;

    // Corrupted device and iface address should not be reachable, and should
    // provide the reason in the info string
    is_reachable = uct_iface_is_reachable_v2(iface, &params);
    EXPECT_FALSE(is_reachable);
    UCS_TEST_MESSAGE << info_str;
    EXPECT_FALSE(std::string(info_str).empty());

    free(iface_addr);
    free(dev_addr);
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
