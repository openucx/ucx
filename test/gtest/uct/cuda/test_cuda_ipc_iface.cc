/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/uct_test.h>

extern "C" {
#include <uct/cuda/cuda_ipc/cuda_ipc_md.h>
}


class test_cuda_ipc_iface_nvlink_domain : public uct_test {
protected:
    void init() override
    {
        modify_config("CUDA_IPC_ENABLE_SAME_PROCESS", "y");
        uct_test::init();
    }

    entity *create_entity_in_domain(const std::string &nvlink_domain)
    {
        modify_config("CUDA_IPC_NVLINK_DOMAIN", nvlink_domain);
        entity *e = uct_test::create_entity(0);
        m_entities.push_back(e);

        if (!ucs_derived_of(e->md(), uct_cuda_ipc_md_t)->enable_mnnvl) {
            UCS_TEST_SKIP_R("MNNVL is not supported");
        }

        return e;
    }

    bool is_reachable(entity &sender, entity &receiver, std::string &info)
    {
        const auto &iface_attr = receiver.iface_attr();
        std::string dev_addr(iface_attr.device_addr_len, '\0');
        std::string iface_addr(iface_attr.iface_addr_len, '\0');
        uct_iface_is_reachable_params_t params;
        char info_str[256] = {};

        ASSERT_UCS_OK(uct_iface_get_device_address(
                receiver.iface(), (uct_device_addr_t*)&dev_addr[0]));
        ASSERT_UCS_OK(uct_iface_get_address(receiver.iface(),
                                            (uct_iface_addr_t*)&iface_addr[0]));

        params.field_mask         = UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR |
                                    UCT_IFACE_IS_REACHABLE_FIELD_IFACE_ADDR |
                                    UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING |
                                    UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING_LENGTH |
                                    UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR_LENGTH |
                                    UCT_IFACE_IS_REACHABLE_FIELD_IFACE_ADDR_LENGTH;
        params.info_string        = info_str;
        params.info_string_length = sizeof(info_str);
        params.device_addr        = (uct_device_addr_t*)&dev_addr[0];
        params.device_addr_length = dev_addr.size();
        params.iface_addr         = (uct_iface_addr_t*)&iface_addr[0];
        params.iface_addr_length  = iface_addr.size();

        bool result = uct_iface_is_reachable_v2(sender.iface(), &params);
        info        = info_str;
        return result;
    }
};

UCS_TEST_P(test_cuda_ipc_iface_nvlink_domain, same_domain)
{
    entity *e1 = create_entity_in_domain("1");
    entity *e2 = create_entity_in_domain("1");
    std::string info;

    EXPECT_TRUE(is_reachable(*e1, *e2, info)) << info;
    EXPECT_TRUE(is_reachable(*e2, *e1, info)) << info;
}

UCS_TEST_P(test_cuda_ipc_iface_nvlink_domain, different_domain)
{
    entity *e1 = create_entity_in_domain("1");
    entity *e2 = create_entity_in_domain("2");
    std::string info;

    EXPECT_FALSE(is_reachable(*e1, *e2, info));
    EXPECT_EQ("different NVLink domain", info);
    EXPECT_FALSE(is_reachable(*e2, *e1, info));
    EXPECT_EQ("different NVLink domain", info);
}

_UCT_INSTANTIATE_TEST_CASE(test_cuda_ipc_iface_nvlink_domain, cuda_ipc)
