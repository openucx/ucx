/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include "ucp_test.h"

#include <ucs/gtest/test_helpers.h>


ucp_test::entity::entity() {
    ucs::handle<ucp_config_t*> config;

    UCS_TEST_CREATE_HANDLE(ucp_config_t*, config, ucp_config_release,
                           ucp_config_read, NULL, NULL);

    UCS_TEST_CREATE_HANDLE(ucp_context_h, m_ucph, ucp_cleanup, ucp_init, config, 0);

    UCS_TEST_CREATE_HANDLE(ucp_worker_h, m_worker, ucp_worker_destroy,
                           ucp_worker_create, m_ucph, UCS_THREAD_MODE_MULTI);
}

void ucp_test::entity::connect(const ucp_test::entity& other) {
    ucs_status_t status;
    ucp_address_t *address;
    size_t address_length;

    status = ucp_worker_get_address(other.m_worker, &address, &address_length);
    ASSERT_UCS_OK(status);

    UCS_TEST_CREATE_HANDLE(ucp_ep_h, m_ep, ucp_ep_destroy,
                           ucp_ep_create, m_worker, address);

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
    m_ep.reset();
}
