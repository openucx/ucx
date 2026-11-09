/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef TEST_UCP_WIREUP_H_
#define TEST_UCP_WIREUP_H_

#include "ucp_test.h"

extern "C" {
#include <ucp/wireup/address.h>
#include <ucp/wireup/wireup.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_worker.inl>
}


/* Worker address of a remote worker, unpacked on a local worker */
class unpacked_address {
public:
    unpacked_address(ucp_worker_h remote_worker) :
        address(), m_remote_worker(remote_worker), m_address(NULL)
    {
    }

    ~unpacked_address()
    {
        if (m_address != NULL) {
            ucp_worker_release_address(m_remote_worker, m_address);
        }
        ucs_free(address.address_list);
    }

    void unpack(ucp_worker_h local_worker)
    {
        size_t address_length;
        ucs_status_t status;

        ASSERT_TRUE(m_address == NULL);

        status = ucp_worker_get_address(m_remote_worker, &m_address,
                                        &address_length);
        ASSERT_UCS_OK(status);
        ASSERT_GT(address_length, 0ul);

        status = ucp_address_unpack(
                local_worker, m_address,
                ucp_worker_default_address_pack_flags(m_remote_worker),
                &address);
        ASSERT_UCS_OK(status);
    }

    /* Unpack an address with only the given transports, like the address of a
     * wireup message. The unpacked entries point to the packed buffer, which is
     * kept until the destructor */
    void unpack(ucp_worker_h local_worker, const ucp_tl_bitmap_t &tl_bitmap)
    {
        const unsigned flags = ucp_worker_default_address_pack_flags(
                m_remote_worker);
        size_t address_length;
        void *buffer;

        ASSERT_TRUE(m_address == NULL);

        ASSERT_UCS_OK(ucp_address_pack(
                m_remote_worker, NULL, &tl_bitmap, flags,
                m_remote_worker->context->config.ext.worker_addr_version, NULL,
                UINT_MAX, &address_length, &buffer));
        m_address = static_cast<ucp_address_t*>(buffer);
        ASSERT_UCS_OK(
                ucp_address_unpack(local_worker, m_address, flags, &address));
    }

    ucp_unpacked_address_t address;

private:
    unpacked_address(const unpacked_address&);
    unpacked_address& operator=(const unpacked_address&);

    ucp_worker_h m_remote_worker;
    ucp_address_t *m_address;
};


/* Select the lanes to the remote address like ucp_wireup_try_select_lanes(),
 * without connecting */
static inline void select_lanes(ucp_test_base::entity &e,
                                const ucp_unpacked_address_t &remote_address,
                                ucp_ep_config_key_t &key)
{
    unsigned addr_indices[UCP_MAX_LANES];
    ucs_status_t status;
    ucp_ep_h ep;

    ucp_ep_config_key_reset(&key);
    key.dst_version = remote_address.dst_version;

    ucs::scoped_async_lock lock(e.worker()->async);
    ASSERT_UCS_OK(ucp_ep_create_base(e.worker(), UCP_EP_INIT_FLAG_INTERNAL,
                                     "select-lanes", "select-lanes", &ep));
    status = ucp_wireup_select_lanes(ep, 0, ucp_tl_bitmap_max, &remote_address,
                                     addr_indices, &key, 1);
    ucp_ep_delete(ep);
    ASSERT_UCS_OK(status);
}

#endif
