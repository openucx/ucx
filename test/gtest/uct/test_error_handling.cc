
/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.All rights reserved.
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
}
#include <common/test.h>
#include "uct_test.h"

class test_error_handling : public uct_test {
public:
    virtual void init() {
        uct_test::init();

        set_config("LOG_LEVEL=fatal");

        /* To reduce test execution time decrease retransmition timeouts
         * where it is relevant */
        if (GetParam()->tl_name == "rc" || GetParam()->tl_name == "rc_mlx5") {
            set_config("RC_TIMEOUT=0.0001"); /* 100 us should be enough */
            set_config("RC_RETRY_COUNT=2");
        }

        m_e1 = uct_test::create_entity(0);
        m_entities.push_back(m_e1);

        m_e2 = uct_test::create_entity(0);
        m_entities.push_back(m_e2);

        connect();
    }

    static ucs_status_t am_dummy_handler(void *arg, void *data, size_t length, void *desc) {
        return UCS_OK;
    }

    static ucs_status_t pending_cb(uct_pending_req_t *self)
    {
        req_count++;
        return UCS_OK;
    }

    static void purge_cb(uct_pending_req_t *self, void *arg)
    {
        req_count++;
    }

    void connect() {
        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);

        uct_iface_set_am_handler(m_e1->iface(), 0, am_dummy_handler,
                                 NULL, UCT_AM_CB_FLAG_SYNC);
        uct_iface_set_am_handler(m_e2->iface(), 0, am_dummy_handler,
                                 NULL, UCT_AM_CB_FLAG_SYNC);
    }

protected:
    static int req_count;
    entity *m_e1, *m_e2;
};

int test_error_handling::req_count = 0;

UCS_TEST_P(test_error_handling, peer_failure)
{
    check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE);

    m_e2->destroy_ep(0);
    EXPECT_EQ(uct_ep_put_short(m_e1->ep(0), NULL, 0, 0, 0), UCS_OK);

    m_e1->flush();

    UCS_TEST_GET_BUFFER_IOV(iov, iovlen, NULL, 0, NULL, 1);

    /* Check that all ep operations return pre-defined error code */
    EXPECT_EQ(uct_ep_am_short(m_e1->ep(0), 0, 0, NULL, 0),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_am_bcopy(m_e1->ep(0), 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_am_zcopy(m_e1->ep(0), 0, NULL, 0, iov, iovlen, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_put_short(m_e1->ep(0), NULL, 0, 0, 0),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_put_bcopy(m_e1->ep(0), NULL, NULL, 0, 0),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_put_zcopy(m_e1->ep(0), iov, iovlen, 0, 0, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_get_bcopy(m_e1->ep(0), NULL, NULL, 0, 0, 0, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_get_zcopy(m_e1->ep(0), iov, iovlen, 0, 0, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_add64(m_e1->ep(0), 0, 0, 0),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_add32(m_e1->ep(0), 0, 0, 0),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_fadd64(m_e1->ep(0), 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_fadd32(m_e1->ep(0), 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_swap64(m_e1->ep(0), 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_swap32(m_e1->ep(0), 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_cswap64(m_e1->ep(0), 0, 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_cswap32(m_e1->ep(0), 0, 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_flush(m_e1->ep(0), 0, NULL), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_get_address(m_e1->ep(0), NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_pending_add(m_e1->ep(0), NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_connect_to_ep(m_e1->ep(0), NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
}

UCS_TEST_P(test_error_handling, purge_failed_ep)
{
    check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE);

    ucs_status_t status;
    int num_pend_sends = 3;
    uct_pending_req_t reqs[num_pend_sends];

    req_count = 0;

    m_e2->destroy_ep(0);

    do {
          status = uct_ep_put_short(m_e1->ep(0), NULL, 0, 0, 0);
    } while (status == UCS_OK);

    for (int i = 0; i < num_pend_sends; i ++) {
        reqs[i].func = pending_cb;
        EXPECT_EQ(uct_ep_pending_add(m_e1->ep(0), &reqs[i]), UCS_OK);
    }

    m_e1->flush();

    EXPECT_EQ(uct_ep_am_short(m_e1->ep(0), 0, 0, NULL, 0),
              UCS_ERR_ENDPOINT_TIMEOUT);

    uct_ep_pending_purge(m_e1->ep(0), purge_cb, NULL);
    EXPECT_EQ(num_pend_sends, req_count);
}

UCT_INSTANTIATE_TEST_CASE(test_error_handling)

