/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ucp_test.h"

#include <ucs/gtest/test_helpers.h>


ucp_test::entity::entity() {
    ucs_status_t status;

    status = ucp_init(&m_ucph);
    ASSERT_UCS_OK(status);

    status = ucp_iface_create(m_ucph, &m_iface);
    ASSERT_UCS_OK(status);

    status = ucp_ep_create(m_iface, &m_ep);
    ASSERT_UCS_OK(status);

}

ucp_test::entity::~entity() {
    ucp_ep_destroy(m_ep);
    ucp_iface_close(m_iface);
    ucp_cleanup(m_ucph);
}

void ucp_test::entity::connect(const ucp_test::entity& other) {

}

ucp_ep_h ucp_test::entity::ep() const {
    return m_ep;
}

void ucp_test::entity::flush() const {

}
