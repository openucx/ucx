/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include "ucp_test.h"

#include <ucs/gtest/test_helpers.h>


ucp_test::entity::entity() {
    ucs_status_t status;
    ucp_config_t *config;

    status = ucp_config_read(NULL, NULL, &config);
    ASSERT_UCS_OK(status);

    status = ucp_init(config, 0, &m_ucph);
    ASSERT_UCS_OK(status);

    ucp_config_release(config);

    status = ucp_worker_create(m_ucph, UCS_THREAD_MODE_MULTI, &m_worker);
    ASSERT_UCS_OK(status);

    m_ep = NULL;
}

ucp_test::entity::~entity() {
    disconnect();
    ucp_worker_destroy(m_worker);
    ucp_cleanup(m_ucph);
}

void ucp_test::entity::connect(const ucp_test::entity& other) {
    ucs_status_t status;
    ucp_address_t *address;
    size_t address_length;

    status = ucp_worker_get_address(other.m_worker, &address, &address_length);
    ASSERT_UCS_OK(status);

    status = ucp_ep_create(m_worker, address, &m_ep);
    ASSERT_UCS_OK(status);

    ucp_worker_release_address(other.m_worker, address);
}

ucp_ep_h ucp_test::entity::ep() const {
    return m_ep;
}

ucp_worker_h ucp_test::entity::worker() const {
    return m_worker;
}

ucp_context_h ucp_test::entity::ucph() const {
    return m_ucph;
}

void ucp_test::entity::disconnect() {
    if (m_ep != NULL) {
        ucp_ep_destroy(m_ep);
        m_ep = NULL;
    }
}
