
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

class test_uct_peer_failure : public uct_test {
public:
    virtual void init();

    virtual uct_error_handler_t get_err_handler() const {
        return err_cb;
    }

    static ucs_status_t am_dummy_handler(void *arg, void *data, size_t length,
                                         unsigned flags) {
        reinterpret_cast<test_uct_peer_failure*>(arg)->m_am_count++;
        return UCS_OK;
    }

    static ucs_status_t pending_cb(uct_pending_req_t *self)
    {
        m_req_count++;
        return UCS_OK;
    }

    static void purge_cb(uct_pending_req_t *self, void *arg)
    {
        m_req_count++;
    }

    static void err_cb(void *arg, uct_ep_h ep, ucs_status_t status)
    {
        EXPECT_EQ(UCS_ERR_ENDPOINT_TIMEOUT, status);
        reinterpret_cast<test_uct_peer_failure*>(arg)->m_err_count++;
    }

    void kill_receiver0()
    {
        m_entities.remove(m_receiver1);
        ucs_assert(m_entities.size() == 2);
    }

    void set_am_handlers()
    {
        check_caps(UCT_IFACE_FLAG_AM_CB_SYNC);
        uct_iface_set_am_handler(m_receiver1->iface(), 0, am_dummy_handler,
                                 reinterpret_cast<void*>(this), UCT_AM_CB_FLAG_SYNC);
        uct_iface_set_am_handler(m_receiver2->iface(), 0, am_dummy_handler,
                                 reinterpret_cast<void*>(this), UCT_AM_CB_FLAG_SYNC);
    }

    ucs_status_t send_am(int index)
    {
        ucs_status_t status;
        do {
            progress();
            status = uct_ep_am_short(m_sender->ep(index), 0, 0, NULL, 0);
        } while (status == UCS_ERR_NO_RESOURCE);
        return status;
    }

    void send_recv_am(int index, ucs_status_t exp_status = UCS_OK)
    {
        m_am_count = 0;

        ucs_status_t status = send_am(index);
        EXPECT_EQ(exp_status, status);

        if (exp_status == UCS_OK) {
            wait_for_flag(&m_am_count);
            EXPECT_EQ(m_am_count, 1ul);
        }
    }

    uct_ep_h ep0() {
        return m_sender->ep(0);
    }

    void flush_ep1() {
        uct_completion_t comp;
        ucs_status_t status;

        comp.count = 2;
        comp.func  = NULL;
        do {
            progress();
            status = uct_ep_flush(m_sender->ep(1), 0, &comp);
        } while (status == UCS_ERR_NO_RESOURCE);
        ASSERT_UCS_OK_OR_INPROGRESS(status);
        if (status == UCS_OK) {
            return;
        }
        /* coverity[loop_condition] */
        while (comp.count != 1) {
            progress();
        }
    }

protected:
    entity *m_sender, *m_receiver1, *m_receiver2;
    size_t m_err_count;
    size_t m_am_count;
    static size_t m_req_count;
};

size_t test_uct_peer_failure::m_req_count = 0ul;

void test_uct_peer_failure::init()
{
    uct_test::init();

    /* To reduce test execution time decrease retransmition timeouts
     * where it is relevant */
    if (GetParam()->tl_name == "rc" || GetParam()->tl_name == "rc_mlx5" ||
        GetParam()->tl_name == "dc" || GetParam()->tl_name == "dc_mlx5") {
        set_config("RC_TIMEOUT=100us"); /* 100 us should be enough */
        set_config("RC_RETRY_COUNT=2");
    } else if (GetParam()->tl_name == "ud" || GetParam()->tl_name == "ud_mlx5") {
        set_config("UD_TIMEOUT=1s");
    }

    uct_iface_params_t params;
    memset(&params, 0, sizeof(params));
    params.err_handler     = get_err_handler();
    params.err_handler_arg = reinterpret_cast<void*>(this);

    m_sender = uct_test::create_entity(params);
    m_entities.push_back(m_sender);

    m_receiver1 = uct_test::create_entity(params);
    m_entities.push_back(m_receiver1);

    m_receiver2 = uct_test::create_entity(params);
    m_entities.push_back(m_receiver2);

    m_sender->connect(0, *m_receiver1, 0);
    m_sender->connect(1, *m_receiver2, 0);

    check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE);

    m_err_count = 0;
    m_req_count = 0;
    m_am_count  = 0;
}

UCS_TEST_P(test_uct_peer_failure, peer_failure)
{
    check_caps(UCT_IFACE_FLAG_PUT_SHORT);

    wrap_errors();

    kill_receiver0();
    EXPECT_EQ(uct_ep_put_short(ep0(), NULL, 0, 0, 0), UCS_OK);

    flush();

    restore_errors();

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, NULL, 0, NULL, 1, 0);

    /* Check that all ep operations return pre-defined error code */
    EXPECT_EQ(uct_ep_am_short(ep0(), 0, 0, NULL, 0), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_am_bcopy(ep0(), 0, NULL, NULL, 0), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_am_zcopy(ep0(), 0, NULL, 0, iov, iovcnt, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_put_short(ep0(), NULL, 0, 0, 0), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_put_bcopy(ep0(), NULL, NULL, 0, 0), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_put_zcopy(ep0(), iov, iovcnt, 0, 0, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_get_bcopy(ep0(), NULL, NULL, 0, 0, 0, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_get_zcopy(ep0(), iov, iovcnt, 0, 0, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_add64(ep0(), 0, 0, 0), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_add32(ep0(), 0, 0, 0), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_fadd64(ep0(), 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_fadd32(ep0(), 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_swap64(ep0(), 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_swap32(ep0(), 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_cswap64(ep0(), 0, 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_cswap32(ep0(), 0, 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_flush(ep0(), 0, NULL), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_get_address(ep0(), NULL), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_pending_add(ep0(), NULL), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_connect_to_ep(ep0(), NULL, NULL), UCS_ERR_ENDPOINT_TIMEOUT);

    EXPECT_GT(m_err_count, 0ul);
}

UCS_TEST_P(test_uct_peer_failure, purge_failed_peer)
{
    check_caps(UCT_IFACE_FLAG_AM_SHORT | UCT_IFACE_FLAG_PENDING);

    set_am_handlers();

    send_recv_am(0);
    send_recv_am(1);

    wrap_errors();
    kill_receiver0();

    ucs_status_t status;
    do {
        status = uct_ep_am_short(ep0(), 0, 0, NULL, 0);
    } while (status == UCS_OK);

    const size_t num_pend_sends = 3ul;
    uct_pending_req_t reqs[num_pend_sends];
    for (size_t i = 0; i < num_pend_sends; i ++) {
        reqs[i].func = pending_cb;
        EXPECT_EQ(uct_ep_pending_add(ep0(), &reqs[i]), UCS_OK);
    }

    flush();
    restore_errors();

    EXPECT_EQ(uct_ep_am_short(ep0(), 0, 0, NULL, 0), UCS_ERR_ENDPOINT_TIMEOUT);

    uct_ep_pending_purge(ep0(), purge_cb, NULL);
    EXPECT_EQ(num_pend_sends, m_req_count);
    EXPECT_GE(m_err_count, 0ul);
}

UCS_TEST_P(test_uct_peer_failure, two_pairs_send)
{
    check_caps(UCT_IFACE_FLAG_AM_SHORT | UCT_IFACE_FLAG_PENDING);

    set_am_handlers();

    /* queue sends on 1st pair */
    for (int i = 0; i < 100; ++i) {
        send_am(0);
    }

    /* kill the 1st receiver while sending on 2nd pair */
    wrap_errors();
    kill_receiver0();
    send_am(0);
    send_recv_am(1);
    flush();
    restore_errors();

    /* test flushing one operations */
    send_recv_am(0, UCS_ERR_ENDPOINT_TIMEOUT);
    send_recv_am(1, UCS_OK);
    flush();

    /* test flushing many operations */
    for (int i = 0; i < (1000 / ucs::test_time_multiplier()); ++i) {
        send_recv_am(0, UCS_ERR_ENDPOINT_TIMEOUT);
        send_recv_am(1, UCS_OK);
    }
    flush();
}


UCS_TEST_P(test_uct_peer_failure, two_pairs_send_after)
{
    check_caps(UCT_IFACE_FLAG_AM_SHORT | UCT_IFACE_FLAG_PENDING);

    set_am_handlers();

    wrap_errors();
    kill_receiver0();
    for (int i = 0; i < 100; ++i) {
        send_am(0);
    }
    flush();
    restore_errors();

    send_recv_am(0, UCS_ERR_ENDPOINT_TIMEOUT);

    m_am_count = 0;
    send_am(1);
    ucs_debug("flushing");
    flush_ep1();
    ucs_debug("flushed");
    wait_for_flag(&m_am_count);
    EXPECT_EQ(m_am_count, 1ul);
}

UCT_INSTANTIATE_TEST_CASE(test_uct_peer_failure)

class test_uct_peer_failure_cb : public test_uct_peer_failure {
public:
    virtual uct_error_handler_t get_err_handler() const {
        return err_cb_ep_destroy;
    }

    static void err_cb_ep_destroy(void *arg, uct_ep_h ep, ucs_status_t status) {
        test_uct_peer_failure_cb *self(reinterpret_cast<test_uct_peer_failure_cb*>(arg));
        EXPECT_EQ(self->ep0(), ep);
        self->m_sender->destroy_ep(0);
    }
};

UCS_TEST_P(test_uct_peer_failure_cb, desproy_ep_cb)
{
    check_caps(UCT_IFACE_FLAG_PUT_SHORT);

    wrap_errors();
    kill_receiver0();
    EXPECT_EQ(uct_ep_put_short(ep0(), NULL, 0, 0, 0), UCS_OK);
    flush();
    restore_errors();
}

UCT_INSTANTIATE_TEST_CASE(test_uct_peer_failure_cb)
