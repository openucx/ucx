/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "uct_test.h"

#include <ucs/gtest/test_helpers.h>


std::vector<uct_resource_desc_t> uct_test::enum_resources(const std::string& tl_name) {
    std::vector<uct_resource_desc_t> result;
    uct_resource_desc_t *resources;
    unsigned num_resources;
    ucs_status_t status;
    uct_context_h ucth;

    status = uct_init(&ucth);
    ASSERT_UCS_OK(status);

    status = uct_query_resources(ucth, &resources, &num_resources);
    ASSERT_UCS_OK(status);

    for (unsigned i = 0; i < num_resources; ++i) {
        if (tl_name.empty() || (std::string(resources[i].tl_name) == tl_name)) {
            result.push_back(resources[i]);
        }
    }

    uct_release_resource_list(resources);
    uct_cleanup(ucth);
    return result;
}

uct_test::entity::entity(const uct_resource_desc_t& resource) {
    ucs_status_t status;

    status = uct_init(&m_ucth);
    ASSERT_UCS_OK(status);

    uct_iface_config_t *iface_config;
    status = uct_iface_config_read(m_ucth, resource.tl_name, NULL, NULL,
                                   &iface_config);
    ASSERT_UCS_OK(status);

    status = uct_iface_open(m_ucth, resource.tl_name, resource.dev_name,
                            iface_config, &m_iface);
    ASSERT_UCS_OK(status);

    status = uct_ep_create(m_iface, &m_ep);
    ASSERT_UCS_OK(status);

    uct_iface_config_release(iface_config);
}

uct_test::entity::~entity() {
    uct_ep_destroy(m_ep);
    uct_iface_close(m_iface);
    uct_cleanup(m_ucth);
}

void uct_test::entity::connect(const uct_test::entity& other) {
    ucs_status_t status;

    uct_iface_attr_t iface_attr;
    status = uct_iface_query(other.m_iface, &iface_attr);
    ASSERT_UCS_OK(status);

    uct_iface_addr_t *iface_addr = (uct_iface_addr_t*)malloc(iface_attr.iface_addr_len);
    uct_ep_addr_t *ep_addr       = (uct_ep_addr_t*)malloc(iface_attr.ep_addr_len);

    status = uct_iface_get_address(other.m_iface, iface_addr);
    ASSERT_UCS_OK(status);

    status = uct_ep_get_address(other.m_ep, ep_addr);
    ASSERT_UCS_OK(status);

    status = uct_ep_connect_to_ep(m_ep, iface_addr, ep_addr);
    ASSERT_UCS_OK(status);

    free(ep_addr);
    free(iface_addr);
}

uct_rkey_bundle_t uct_test::entity::mem_map(void *address, size_t length, uct_lkey_t *lkey_p) const {
    ucs_status_t status;
    void *rkey_buffer;
    uct_pd_attr_t pd_attr;
    uct_rkey_bundle_t rkey;

    status = uct_mem_map(m_iface->pd, address, length, 0, lkey_p);
    ASSERT_UCS_OK(status);

    status = uct_pd_query(m_iface->pd, &pd_attr);
    ASSERT_UCS_OK(status);

    rkey_buffer = malloc(pd_attr.rkey_packed_size);

    status = uct_rkey_pack(m_iface->pd, *lkey_p, rkey_buffer);
    ASSERT_UCS_OK(status);

    status = uct_rkey_unpack(m_ucth, rkey_buffer, &rkey);
    ASSERT_UCS_OK(status);

    free(rkey_buffer);
    return rkey;
}

void uct_test::entity::mem_unmap(uct_lkey_t lkey, const uct_rkey_bundle_t& rkey) const {
    ucs_status_t status;
    uct_rkey_release(m_ucth, const_cast<uct_rkey_bundle_t*>(&rkey));
    status = uct_mem_unmap(m_iface->pd, lkey);
    ASSERT_UCS_OK(status);
}

uct_ep_h uct_test::entity::ep() const {
    return m_ep;
}

void uct_test::entity::flush() const {
    ucs_status_t status;
    do {
        uct_progress(m_ucth);
        status = uct_iface_flush(m_iface, NULL, NULL);
    } while (status == UCS_ERR_WOULD_BLOCK);
    ASSERT_UCS_OK(status);
}

std::ostream& operator<<(std::ostream& os, const uct_resource_desc_t& resource) {
    return os << resource.tl_name << "/" << resource.dev_name;
}
