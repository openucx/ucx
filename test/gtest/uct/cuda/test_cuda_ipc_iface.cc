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


/* The reachability decision depends only on the bytes of the peer's device
 * address, so a peer on another host or in another NVLink domain is emulated
 * by editing the address of a local entity before passing it to the check.
 * On systems without MNNVL, MNNVL support and the NVLink domain of the MD are
 * emulated as well, so that the check runs on any CUDA system. */
class test_cuda_ipc_iface_nvlink_domain : public uct_test {
protected:
    void init() override
    {
        modify_config("CUDA_IPC_ENABLE_SAME_PROCESS", "y");
        uct_test::init();
    }

    entity *create_mnnvl_entity()
    {
        entity *e = uct_test::create_entity(0);
        m_entities.push_back(e);

        uct_cuda_ipc_md_t *md = ucs_derived_of(e->md(), uct_cuda_ipc_md_t);
        if (!md->enable_mnnvl) {
            UCS_TEST_MESSAGE << "MNNVL is not supported, emulating it";
            md->enable_mnnvl  = 1;
            md->nvlink_domain = 0xA5A5A5A5A5A5A5A5ULL;
        }

        return e;
    }

    static std::string get_device_address(entity &e)
    {
        std::string dev_addr(sizeof(uct_cuda_ipc_mnnvl_device_addr_t), '\0');
        uct_iface_attr_t iface_attr;

        /* Query again, since the cached entity attributes were taken before
         * MNNVL was emulated */
        ASSERT_UCS_OK(uct_iface_query(e.iface(), &iface_attr));
        EXPECT_EQ(dev_addr.size(), iface_attr.device_addr_len);

        ASSERT_UCS_OK(uct_iface_get_device_address(
                e.iface(), (uct_device_addr_t*)&dev_addr[0]));
        return dev_addr;
    }

    static uct_cuda_ipc_mnnvl_device_addr_t *mnnvl_addr(std::string &dev_addr)
    {
        return (uct_cuda_ipc_mnnvl_device_addr_t*)&dev_addr[0];
    }

    /* Same peer, but on another host */
    static void set_remote_host(std::string &dev_addr)
    {
        mnnvl_addr(dev_addr)->super.system_uuid ^= 0xDEADBEEFDEADBEEFULL;
    }

    /* Same peer, but in another NVLink domain */
    static void set_other_domain(std::string &dev_addr)
    {
        mnnvl_addr(dev_addr)->nvlink_domain ^= 1;
    }

    /* Same peer, but running a UCX which does not send the NVLink domain */
    static void set_older_peer(std::string &dev_addr)
    {
        dev_addr.resize(sizeof(uct_cuda_ipc_device_addr_t));
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
};

/* Peers on the same host are reachable regardless of the NVLink domain, which
 * is checked only against other hosts */
UCS_TEST_P(test_cuda_ipc_iface_nvlink_domain, same_host)
{
    entity *e1           = create_mnnvl_entity();
    entity *e2           = create_mnnvl_entity();
    std::string dev_addr = get_device_address(*e2);
    std::string info;

    EXPECT_TRUE(is_reachable(*e1, *e2, dev_addr, info)) << info;
    EXPECT_TRUE(is_reachable(*e2, *e1, get_device_address(*e1), info))
            << info;

    set_other_domain(dev_addr);
    EXPECT_TRUE(is_reachable(*e1, *e2, dev_addr, info)) << info;
}

/* A remote peer that does not send its NVLink domain must keep the previous
 * behavior: reachable whenever both sides support MNNVL. */
UCS_TEST_P(test_cuda_ipc_iface_nvlink_domain, older_peer)
{
    entity *e1           = create_mnnvl_entity();
    entity *e2           = create_mnnvl_entity();
    std::string dev_addr = get_device_address(*e2);
    std::string info;

    set_remote_host(dev_addr);
    set_older_peer(dev_addr);
    EXPECT_TRUE(is_reachable(*e1, *e2, dev_addr, info)) << info;
}

/* Peers on another host are reachable only within the same NVLink domain */
UCS_TEST_P(test_cuda_ipc_iface_nvlink_domain, remote_host)
{
    entity *e1           = create_mnnvl_entity();
    entity *e2           = create_mnnvl_entity();
    std::string dev_addr = get_device_address(*e2);
    std::string info;

    set_remote_host(dev_addr);
    EXPECT_TRUE(is_reachable(*e1, *e2, dev_addr, info)) << info;

    set_other_domain(dev_addr);
    EXPECT_FALSE(is_reachable(*e1, *e2, dev_addr, info));
    EXPECT_EQ(0u, info.find("different NVLink domain")) << info;
}

_UCT_INSTANTIATE_TEST_CASE(test_cuda_ipc_iface_nvlink_domain, cuda_ipc)
