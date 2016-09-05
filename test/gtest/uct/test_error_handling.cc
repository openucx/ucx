
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
#include "ud_base.h"

enum {
    PEER_TYPE_UD,
    PEER_TYPE_RC
};

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
            peer_type = PEER_TYPE_RC;
        } else if (GetParam()->tl_name == "ud" ||
                   GetParam()->tl_name == "ud_mlx5") {
            peer_type = PEER_TYPE_UD;
        }

        orig = uct_test::create_entity(0);
        m_entities.push_back(orig);

        peer = uct_test::create_entity(0);
        m_entities.push_back(peer);

        connect();
        peer->destroy_ep(0);
    }

    void disable_peer() {
        if (peer_type == PEER_TYPE_RC) {
            orig->flush();
        } else if (peer_type == PEER_TYPE_UD) {
            uct_ud_iface_t *peer_if = ucs_derived_of(peer->iface(), uct_ud_iface_t);
            uct_ud_iface_t *orig_if = ucs_derived_of(orig->iface(), uct_ud_iface_t);
            ucs_async_remove_timer(peer_if->async.timer_id);
            orig_if->config.peer_timeout = ucs_time_from_msec(50);
            twait(300);
        } else {
            UCS_TEST_ABORT("Error: unsupported transport " << peer_type);
        }
    }

    static ucs_status_t am_dummy_handler(void *arg, void *data, size_t length,
                                         void *desc) {
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

    static void fail_completion_cb(uct_completion_t *self, ucs_status_t status)
    {
        if (status == UCS_ERR_ENDPOINT_TIMEOUT) {
            req_count++;
        }
    }

    void connect() {
        orig->connect(0, *peer, 0);
        peer->connect(0, *orig, 0);

        uct_iface_set_am_handler(orig->iface(), 0, am_dummy_handler,
                                 NULL, UCT_AM_CB_FLAG_SYNC);
        uct_iface_set_am_handler(peer->iface(), 0, am_dummy_handler,
                                 NULL, UCT_AM_CB_FLAG_SYNC);
    }

protected:
    static int req_count;
    entity *orig, *peer;
    int peer_type;
};

int test_error_handling::req_count = 0;

#define CHECK_IF_SUPPORTED(actual, expected, cap) \
        if (flags & cap) { \
            EXPECT_EQ(actual,expected); \
        } \

UCS_TEST_P(test_error_handling, peer_failure)
{
    check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE | UCT_IFACE_FLAG_AM_SHORT);

    EXPECT_EQ(uct_ep_am_short(orig->ep(0), 0, 0, NULL, 0), UCS_OK);
    disable_peer();

    unsigned flags = orig->iface_attr().cap.flags;

    /* Check that all ep operations return pre-defined error code */
    EXPECT_EQ(uct_ep_am_short(orig->ep(0), 0, 0, NULL, 0), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_flush(orig->ep(0), 0, NULL), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_get_address(orig->ep(0), NULL), UCS_ERR_ENDPOINT_TIMEOUT);

    CHECK_IF_SUPPORTED(uct_ep_am_bcopy(orig->ep(0), 0, NULL, NULL),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_AM_BCOPY);
    CHECK_IF_SUPPORTED(uct_ep_am_zcopy(orig->ep(0), 0, NULL, 0, NULL, 0, NULL, NULL),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_AM_ZCOPY);
    CHECK_IF_SUPPORTED(uct_ep_put_short(orig->ep(0), NULL, 0, 0, 0),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_PUT_SHORT);
    CHECK_IF_SUPPORTED(uct_ep_put_bcopy(orig->ep(0), NULL, NULL, 0, 0),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_PUT_BCOPY);
    CHECK_IF_SUPPORTED(uct_ep_put_zcopy(orig->ep(0), NULL, 0, NULL, 0, 0, NULL),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_PUT_ZCOPY);
    CHECK_IF_SUPPORTED(uct_ep_get_bcopy(orig->ep(0), NULL, NULL, 0, 0, 0, NULL),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_GET_BCOPY);
    CHECK_IF_SUPPORTED(uct_ep_get_zcopy(orig->ep(0), NULL, 0, NULL, 0, 0, NULL),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_GET_ZCOPY);
    CHECK_IF_SUPPORTED(uct_ep_atomic_add64(orig->ep(0), 0, 0, 0),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_ATOMIC_ADD64);
    CHECK_IF_SUPPORTED(uct_ep_atomic_add32(orig->ep(0), 0, 0, 0),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_ATOMIC_ADD32);
    CHECK_IF_SUPPORTED(uct_ep_atomic_fadd64(orig->ep(0), 0, 0, 0, NULL, NULL),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_ATOMIC_FADD64);
    CHECK_IF_SUPPORTED(uct_ep_atomic_fadd32(orig->ep(0), 0, 0, 0, NULL, NULL),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_ATOMIC_FADD32);
    CHECK_IF_SUPPORTED(uct_ep_atomic_swap64(orig->ep(0), 0, 0, 0, NULL, NULL),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_ATOMIC_SWAP64);
    CHECK_IF_SUPPORTED(uct_ep_atomic_swap32(orig->ep(0), 0, 0, 0, NULL, NULL),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_ATOMIC_SWAP32);
    CHECK_IF_SUPPORTED(uct_ep_atomic_cswap64(orig->ep(0), 0, 0, 0, 0, NULL, NULL),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_ATOMIC_CSWAP64);
    CHECK_IF_SUPPORTED(uct_ep_atomic_cswap32(orig->ep(0), 0, 0, 0, 0, NULL, NULL),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_ATOMIC_CSWAP32);
    CHECK_IF_SUPPORTED(uct_ep_pending_add(orig->ep(0), NULL),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_PENDING);
    CHECK_IF_SUPPORTED(uct_ep_connect_to_ep(orig->ep(0), NULL, NULL),
                       UCS_ERR_ENDPOINT_TIMEOUT, UCT_IFACE_FLAG_CONNECT_TO_EP);
}

UCS_TEST_P(test_error_handling, purge_failed_ep)
{
    check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE |
               UCT_IFACE_FLAG_AM_SHORT | UCT_IFACE_FLAG_PENDING);

    ucs_status_t status;
    int num_pend_sends = 3;
    uct_pending_req_t reqs[num_pend_sends];

    req_count = 0;

    do {
          status = uct_ep_am_short(orig->ep(0), 0, 0, NULL, 0);
    } while (status == UCS_OK);

    for (int i = 0; i < num_pend_sends; i ++) {
        reqs[i].func = pending_cb;
        EXPECT_EQ(uct_ep_pending_add(orig->ep(0), &reqs[i]), UCS_OK);
    }

    disable_peer();
    orig->flush();

    EXPECT_EQ(uct_ep_am_short(orig->ep(0), 0, 0, NULL, 0),
              UCS_ERR_ENDPOINT_TIMEOUT);

    uct_ep_pending_purge(orig->ep(0), purge_cb, NULL);
    EXPECT_EQ(num_pend_sends, req_count);
}


UCS_TEST_P(test_error_handling, check_comp) {
    int num_comp = 2;
    uct_completion_t comp[num_comp];
    ucs_status_t status;

    req_count = 0;

    check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE | UCT_IFACE_FLAG_AM_ZCOPY);

    for (int i = 0; i < num_comp; i++) {
        comp[i].count = 1;
        comp[i].func = fail_completion_cb;
        status = uct_ep_am_zcopy(orig->ep(0), 0, NULL, 0, NULL, 0,
                                 NULL, &comp[i]);
        EXPECT_EQ(status, UCS_INPROGRESS);
    }

    disable_peer();

    short_progress_loop();

    EXPECT_EQ(req_count, num_comp);
}

UCT_INSTANTIATE_TEST_CASE(test_error_handling)

