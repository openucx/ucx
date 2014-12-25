/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include <ucs/gtest/test.h>
extern "C" {
#include <ucp/api/ucp.h>
}

class test_ucp : public ucs::test {
};

UCS_TEST_F(test_ucp, open_ep) {
    ucs_status_t status;
    ucp_context_h ucph;
    ucp_iface_h iface;
    ucp_ep_h ep;

    ucph = NULL;
    iface = NULL;
    ep = NULL;

    /* initialize ucp context, interface and ep */
    status = ucp_init(&ucph);
    ASSERT_UCS_OK(status);
    ASSERT_TRUE(ucph != NULL);

    status = ucp_iface_create(ucph, NULL, &iface);
    ASSERT_UCS_OK(status);

    status = ucp_ep_create(iface, &ep);
    ASSERT_UCS_OK(status);

    /* release everything */
    ucp_ep_destroy(ep);
    ucp_iface_close(iface);
    ucp_cleanup (ucph);
}
