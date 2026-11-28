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
#include <uct/cuda/cuda_ipc/cuda_ipc_iface_address.h>
#include <uct/cuda/cuda_ipc/cuda_ipc_md.h>
}


/* UCX_CUDA_IPC_NVLINK_DOMAIN="<domain>[:<host>]" makes an entity emulate the
 * given host in the given NVLink domain, with MNNVL and fabric support enabled
 * even without a GPU fabric or an IMEX channel. <host> defaults to <domain>.
 * Other peer properties are emulated by editing the addresses of a local
 * entity before passing them to the check. */
class test_cuda_ipc_iface_reachability : public uct_test {
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

        uct_cuda_ipc_md_t *md = ucs_derived_of(e->md(), uct_cuda_ipc_md_t);
        EXPECT_TRUE(md->enable_mnnvl);
        EXPECT_TRUE(uct_cuda_ipc_md_fabric_supported(md));
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

    /* Same peer, but without access to an IMEX channel */
    static void set_no_fabric(std::string &dev_addr)
    {
        ((uct_cuda_ipc_device_addr_t*)&dev_addr[0])->flags &=
                ~UCT_CUDA_IPC_DEVICE_ADDR_FLAG_FABRIC;
    }

    /* Device address as sent by a UCX which does not support NVLink domains */
    static std::string get_older_peer_device_address(entity &e)
    {
        std::string dev_addr = get_device_address(e);

        dev_addr.resize(sizeof(uct_cuda_ipc_device_addr_t));
        set_no_fabric(dev_addr);
        return dev_addr;
    }

    static uct_cuda_ipc_iface_imex_address_t *
    imex_addr(std::string &iface_addr)
    {
        EXPECT_EQ(sizeof(uct_cuda_ipc_iface_imex_address_t),
                  iface_addr.size());
        return (uct_cuda_ipc_iface_imex_address_t*)&iface_addr[0];
    }

    static std::string get_iface_address(entity &e)
    {
        std::string iface_addr(e.iface_attr().iface_addr_len, '\0');

        ASSERT_UCS_OK(uct_iface_get_address(
                e.iface(), (uct_iface_addr_t*)&iface_addr[0]));
        return iface_addr;
    }

    /* Same peer, but with another IMEX channel */
    static void set_other_imex_channel(std::string &iface_addr)
    {
        imex_addr(iface_addr)->imex_channel ^= 1;
    }

    /* Same peer, but in another PID namespace */
    static void set_other_pid_ns(std::string &iface_addr)
    {
        imex_addr(iface_addr)->super.pid_ns ^= 1;
    }

    /* Same peer, but running UCX v1.23, which does not send the IMEX channel */
    static void set_v1_23_peer(std::string &iface_addr)
    {
        iface_addr.resize(sizeof(uct_cuda_ipc_iface_address_t));
    }

    bool is_reachable(entity &sender, const std::string &dev_addr,
                      const std::string &iface_addr, std::string &info)
    {
        const uint64_t field_mask =
                UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR |
                UCT_IFACE_IS_REACHABLE_FIELD_IFACE_ADDR |
                UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING |
                UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING_LENGTH |
                UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR_LENGTH |
                UCT_IFACE_IS_REACHABLE_FIELD_IFACE_ADDR_LENGTH;
        uct_iface_is_reachable_params_t params;
        char info_str[256] = {};

        params.field_mask         = field_mask;
        params.info_string        = info_str;
        params.info_string_length = sizeof(info_str);
        params.device_addr        = (const uct_device_addr_t*)dev_addr.data();
        params.device_addr_length = dev_addr.size();
        params.iface_addr         = (const uct_iface_addr_t*)iface_addr.data();
        params.iface_addr_length  = iface_addr.size();

        bool result = uct_iface_is_reachable_v2(sender.iface(), &params);
        info        = info_str;
        return result;
    }

    bool is_reachable(entity &sender, entity &receiver,
                      const std::string &dev_addr, std::string &info)
    {
        return is_reachable(sender, dev_addr, get_iface_address(receiver),
                            info);
    }

    bool is_reachable(entity &sender, entity &receiver, std::string &info)
    {
        return is_reachable(sender, receiver, get_device_address(receiver),
                            info);
    }
};

/* Same host, same domain */
UCS_TEST_P(test_cuda_ipc_iface_reachability, same_domain)
{
    entity *e1 = create_entity_in_domain("1");
    entity *e2 = create_entity_in_domain("1");
    std::string info;

    EXPECT_TRUE(is_reachable(*e1, *e2, info)) << info;
    EXPECT_TRUE(is_reachable(*e2, *e1, info)) << info;
}

/* Different hosts, different domains: not reachable over MNNVL */
UCS_TEST_P(test_cuda_ipc_iface_reachability, different_domain)
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
UCS_TEST_P(test_cuda_ipc_iface_reachability, different_host_same_domain)
{
    entity *e1 = create_entity_in_domain("1:1");
    entity *e2 = create_entity_in_domain("1:2");
    std::string info;

    EXPECT_TRUE(is_reachable(*e1, *e2, info)) << info;
    EXPECT_TRUE(is_reachable(*e2, *e1, info)) << info;
}

/* Same host, different domains: the domain is a property of the host and is
 * not checked between peers on the same host */
UCS_TEST_P(test_cuda_ipc_iface_reachability, same_host_different_domain)
{
    entity *e1 = create_entity_in_domain("1:1");
    entity *e2 = create_entity_in_domain("2:1");
    std::string info;

    EXPECT_TRUE(is_reachable(*e1, *e2, info)) << info;
}

/* A peer that does not send its NVLink domain keeps the previous behavior:
 * reachable whenever both sides support MNNVL */
UCS_TEST_P(test_cuda_ipc_iface_reachability, older_peer)
{
    entity *e1 = create_entity_in_domain("1");
    entity *e2 = create_entity_in_domain("2");
    std::string info;

    EXPECT_TRUE(is_reachable(*e1, *e2, get_older_peer_device_address(*e2),
                             info)) << info;
}

/* A peer on another host running v1.23 does not send the IMEX channel: fabric
 * support is checked, and the channel is not */
UCS_TEST_P(test_cuda_ipc_iface_reachability, v1_23_peer)
{
    entity *e1             = create_entity_in_domain("1:1");
    entity *e2             = create_entity_in_domain("1:2");
    std::string dev_addr   = get_device_address(*e2);
    std::string iface_addr = get_iface_address(*e2);
    std::string info;

    set_v1_23_peer(iface_addr);
    EXPECT_TRUE(is_reachable(*e1, dev_addr, iface_addr, info)) << info;

    set_no_fabric(dev_addr);
    EXPECT_FALSE(is_reachable(*e1, dev_addr, iface_addr, info));
    EXPECT_EQ(0u, info.find("different machine and FABRIC support local=1 "
                            "remote=0")) << info;
}

/* Peers on another host in the same NVLink domain are reachable only if both
 * have access to the same IMEX channel */
UCS_TEST_P(test_cuda_ipc_iface_reachability, different_host_imex)
{
    entity *e1             = create_entity_in_domain("1:1");
    entity *e2             = create_entity_in_domain("1:2");
    std::string dev_addr   = get_device_address(*e2);
    std::string iface_addr = get_iface_address(*e2);
    std::string remote_addr;
    std::string info;

    remote_addr = iface_addr;
    set_other_imex_channel(remote_addr);
    EXPECT_FALSE(is_reachable(*e1, dev_addr, remote_addr, info));
    EXPECT_EQ(0u, info.find("different machine and different IMEX channel"))
            << info;

    remote_addr = dev_addr;
    set_no_fabric(remote_addr);
    EXPECT_FALSE(is_reachable(*e1, remote_addr, iface_addr, info));
    EXPECT_EQ(0u, info.find("different machine and FABRIC support local=1 "
                            "remote=0")) << info;

    ucs_derived_of(e1->md(), uct_cuda_ipc_md_t)->imex_channel =
            UCT_CUDA_IPC_IMEX_CHANNEL_INVALID;
    EXPECT_FALSE(is_reachable(*e1, dev_addr, iface_addr, info));
    EXPECT_EQ(0u, info.find("different machine and FABRIC support local=0 "
                            "remote=1")) << info;
}

/* Peers on the same host in different PID namespaces exchange fabric handles,
 * so they are reachable only if both have access to the same IMEX channel */
UCS_TEST_P(test_cuda_ipc_iface_reachability, other_pid_ns_imex)
{
    entity *e1             = create_entity_in_domain("1");
    entity *e2             = create_entity_in_domain("1");
    std::string dev_addr   = get_device_address(*e2);
    std::string iface_addr = get_iface_address(*e2);
    std::string remote_addr;
    std::string info;

    /* The channel is not checked in the same PID namespace */
    remote_addr = iface_addr;
    set_other_imex_channel(remote_addr);
    EXPECT_TRUE(is_reachable(*e1, dev_addr, remote_addr, info)) << info;

    set_other_pid_ns(iface_addr);
    EXPECT_TRUE(is_reachable(*e1, dev_addr, iface_addr, info)) << info;

    set_other_imex_channel(iface_addr);
    EXPECT_FALSE(is_reachable(*e1, dev_addr, iface_addr, info));
    EXPECT_EQ(0u, info.find("different PID namespace and different IMEX "
                            "channel")) << info;

    /* The channel is sent in the interface address, so it is checked also
     * without MNNVL */
    ucs_derived_of(e1->md(), uct_cuda_ipc_md_t)->enable_mnnvl = 0;
    EXPECT_FALSE(is_reachable(*e1, dev_addr, iface_addr, info));
    EXPECT_EQ(0u, info.find("different PID namespace and different IMEX "
                            "channel")) << info;
}

_UCT_INSTANTIATE_TEST_CASE(test_cuda_ipc_iface_reachability, cuda_ipc)
