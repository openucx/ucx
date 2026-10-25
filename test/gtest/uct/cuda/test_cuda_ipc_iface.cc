/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <uct/uct_test.h>

extern "C" {
#include <uct/cuda/cuda_ipc/cuda_ipc_iface.h>
#include <uct/cuda/cuda_ipc/cuda_ipc_md.h>
}


/* UCX_CUDA_IPC_NVLINK_DOMAIN="<domain>[:<host>]" makes an entity emulate the
 * given host in the given NVLink domain, with MNNVL enabled even without a
 * GPU fabric. <host> defaults to <domain>. */
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

        EXPECT_TRUE(ucs_derived_of(e->md(), uct_cuda_ipc_md_t)->enable_mnnvl);
        return e;
    }

    static std::string get_device_address(entity &e)
    {
        std::string dev_addr(e.iface_attr().device_addr_len, '\0');

        ASSERT_UCS_OK(uct_iface_get_device_address(
                e.iface(), (uct_device_addr_t*)&dev_addr[0]));
        EXPECT_EQ(sizeof(uct_cuda_ipc_mnnvl_device_addr_t), dev_addr.size());
        return dev_addr;
    }

    /* Device address as sent by a UCX which does not support NVLink domains */
    static std::string get_older_peer_device_address(entity &e)
    {
        std::string dev_addr = get_device_address(e);

        dev_addr.resize(sizeof(uct_cuda_ipc_device_addr_t));
        return dev_addr;
    }

    bool is_reachable(entity &sender, entity &receiver,
                      const std::string &dev_addr, std::string &info)
    {
        const uint64_t field_mask =
                UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR |
                UCT_IFACE_IS_REACHABLE_FIELD_IFACE_ADDR |
                UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING |
                UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING_LENGTH |
                UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR_LENGTH |
                UCT_IFACE_IS_REACHABLE_FIELD_IFACE_ADDR_LENGTH;
        std::string iface_addr(receiver.iface_attr().iface_addr_len, '\0');
        uct_iface_is_reachable_params_t params;
        char info_str[256] = {};

        ASSERT_UCS_OK(uct_iface_get_address(
                receiver.iface(), (uct_iface_addr_t*)&iface_addr[0]));

        params.field_mask         = field_mask;
        params.info_string        = info_str;
        params.info_string_length = sizeof(info_str);
        params.device_addr        = (const uct_device_addr_t*)dev_addr.data();
        params.device_addr_length = dev_addr.size();
        params.iface_addr         = (uct_iface_addr_t*)&iface_addr[0];
        params.iface_addr_length  = iface_addr.size();

        bool result = uct_iface_is_reachable_v2(sender.iface(), &params);
        info        = info_str;
        return result;
    }

    bool is_reachable(entity &sender, entity &receiver, std::string &info)
    {
        return is_reachable(sender, receiver, get_device_address(receiver),
                            info);
    }
};

/* Same host, same domain */
UCS_TEST_P(test_cuda_ipc_iface_nvlink_domain, same_domain)
{
    entity *e1 = create_entity_in_domain("1");
    entity *e2 = create_entity_in_domain("1");
    std::string info;

    EXPECT_TRUE(is_reachable(*e1, *e2, info)) << info;
    EXPECT_TRUE(is_reachable(*e2, *e1, info)) << info;
}

/* Different hosts, different domains: not reachable over MNNVL */
UCS_TEST_P(test_cuda_ipc_iface_nvlink_domain, different_domain)
{
    entity *e1 = create_entity_in_domain("1");
    entity *e2 = create_entity_in_domain("2");
    std::string info;

    EXPECT_FALSE(is_reachable(*e1, *e2, info));
    EXPECT_EQ(0u, info.find("different NVLink domain")) << info;
    EXPECT_FALSE(is_reachable(*e2, *e1, info));
    EXPECT_EQ(0u, info.find("different NVLink domain")) << info;
}

/* Different hosts, same domain: reachable over MNNVL */
UCS_TEST_P(test_cuda_ipc_iface_nvlink_domain, different_host_same_domain)
{
    entity *e1 = create_entity_in_domain("1:1");
    entity *e2 = create_entity_in_domain("1:2");
    std::string info;

    EXPECT_TRUE(is_reachable(*e1, *e2, info)) << info;
    EXPECT_TRUE(is_reachable(*e2, *e1, info)) << info;
}

/* Same host, different domains: the domain is a property of the host and is
 * not checked between peers on the same host */
UCS_TEST_P(test_cuda_ipc_iface_nvlink_domain, same_host_different_domain)
{
    entity *e1 = create_entity_in_domain("1:1");
    entity *e2 = create_entity_in_domain("2:1");
    std::string info;

    EXPECT_TRUE(is_reachable(*e1, *e2, info)) << info;
}

/* A peer that does not send its NVLink domain keeps the previous behavior:
 * reachable whenever both sides support MNNVL */
UCS_TEST_P(test_cuda_ipc_iface_nvlink_domain, older_peer)
{
    entity *e1 = create_entity_in_domain("1");
    entity *e2 = create_entity_in_domain("2");
    std::string info;

    EXPECT_TRUE(is_reachable(*e1, *e2, get_older_peer_device_address(*e2),
                             info)) << info;
}

_UCT_INSTANTIATE_TEST_CASE(test_cuda_ipc_iface_nvlink_domain, cuda_ipc)
