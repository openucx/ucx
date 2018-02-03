
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
private:
    struct am_handler_setter
    {
        am_handler_setter(test_uct_peer_failure *test) : m_test(test) {}

        void operator() (test_uct_peer_failure::entity *e) {
            uct_iface_set_am_handler(e->iface(), 0,
                                     am_dummy_handler,
                                     reinterpret_cast<void*>(m_test),
                                     UCT_CB_FLAG_SYNC);
        }

        test_uct_peer_failure* m_test;
    };

public:

    test_uct_peer_failure() : m_sender(NULL),  m_nreceivers(2),
                              m_tx_window(100), m_err_count(0), m_am_count(0) {}

    virtual void init();

    inline uct_iface_params_t entity_params() {
        static uct_iface_params_t params;

        memset(&params, 0, sizeof(params));
        params.err_handler     = get_err_handler();
        params.err_handler_arg = reinterpret_cast<void*>(this);
        return params;
    }

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

    static ucs_status_t err_cb(void *arg, uct_ep_h ep, ucs_status_t status)
    {
        EXPECT_EQ(UCS_ERR_ENDPOINT_TIMEOUT, status);
        reinterpret_cast<test_uct_peer_failure*>(arg)->m_err_count++;
        return UCS_OK;
    }

    void kill_receiver()
    {
        ucs_assert(!m_receivers.empty());
        m_entities.remove(m_receivers.front());
        ucs_assert(m_entities.size() == m_receivers.size());
        m_receivers.erase(m_receivers.begin());
    }

    void new_receiver()
    {
        uct_iface_params_t p = entity_params();
        p.open_mode = UCT_IFACE_OPEN_MODE_DEVICE;
        m_receivers.push_back(uct_test::create_entity(p));
        m_entities.push_back(m_receivers.back());
        m_sender->connect(m_receivers.size() - 1, *m_receivers.back(), 0);

        m_entities.back()->check_caps(UCT_IFACE_FLAG_AM_SHORT   |
                                      UCT_IFACE_FLAG_PENDING    |
                                      UCT_IFACE_FLAG_CB_SYNC    |
                                      UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE);

        am_handler_setter(this)(m_receivers.back());
        /* Make sure that TL is up and has resources */
        send_recv_am(m_receivers.size() - 1);
    }

    void set_am_handlers()
    {
        check_caps(UCT_IFACE_FLAG_CB_SYNC);
        std::for_each(m_receivers.begin(), m_receivers.end(),
                      am_handler_setter(this));
    }

    ucs_status_t send_am(int index)
    {
        ucs_status_t status;
        while ((status = uct_ep_am_short(m_sender->ep(index), 0, 0, NULL, 0)) ==
               UCS_ERR_NO_RESOURCE) {
            progress();
        };
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

    ucs_status_t flush_ep(size_t index, ucs_time_t deadline = ULONG_MAX) {
        uct_completion_t    comp;
        ucs_status_t        status;
        int                 is_time_out;

        comp.count = 2;
        comp.func  = NULL;
        do {
            progress();
            status = uct_ep_flush(m_sender->ep(index), 0, &comp);
            is_time_out = (ucs_get_time() > deadline);
        } while ((status == UCS_ERR_NO_RESOURCE) && !is_time_out);

        if (!is_time_out) {
            ASSERT_UCS_OK_OR_INPROGRESS(status);
        }

        if (status == UCS_OK) {
            return UCS_OK;
        } else if (is_time_out) {
            return UCS_ERR_TIMED_OUT;
        }

        /* coverity[loop_condition] */
        while ((comp.count == 2) && !is_time_out) {
            progress();
            is_time_out = (ucs_get_time() > deadline);
        }

        return (comp.count == 1) ? UCS_OK :
               (is_time_out ? UCS_ERR_TIMED_OUT : UCS_ERR_OUT_OF_RANGE);
    }

protected:
    entity                *m_sender;
    std::vector<entity *> m_receivers;
    size_t                m_nreceivers;
    size_t                m_tx_window;
    size_t                m_err_count;
    size_t                m_am_count;
    static size_t         m_req_count;
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

    uct_iface_params_t p = entity_params();
    p.open_mode = UCT_IFACE_OPEN_MODE_DEVICE;
    m_sender = uct_test::create_entity(p);
    m_entities.push_back(m_sender);

    for (size_t i = 0; i < 2; ++i) {
        new_receiver();
    }

    m_err_count = 0;
    m_req_count = 0;
    m_am_count  = 0;
}

UCS_TEST_P(test_uct_peer_failure, peer_failure)
{
    check_caps(UCT_IFACE_FLAG_PUT_SHORT);

    wrap_errors();

    kill_receiver();
    EXPECT_EQ(uct_ep_put_short(ep0(), NULL, 0, 0, 0), UCS_OK);

    flush();

    restore_errors();

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, NULL, 0, NULL, 1);

    /* Check that all ep operations return pre-defined error code */
    EXPECT_EQ(uct_ep_am_short(ep0(), 0, 0, NULL, 0), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_am_bcopy(ep0(), 0, NULL, NULL, 0), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_am_zcopy(ep0(), 0, NULL, 0, iov, iovcnt, 0, NULL),
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
    kill_receiver();

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
    for (size_t i = 0; i < m_tx_window; ++i) {
        send_am(0);
    }

    /* kill the 1st receiver while sending on 2nd pair */
    wrap_errors();
    kill_receiver();
    send_am(0);
    send_recv_am(1);
    flush();
    restore_errors();

    /* test flushing one operations */
    send_recv_am(0, UCS_ERR_ENDPOINT_TIMEOUT);
    send_recv_am(1, UCS_OK);
    flush();

    /* test flushing many operations */
    for (size_t i = 0; i < (m_tx_window * 10 / ucs::test_time_multiplier()); ++i) {
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
    kill_receiver();
    for (int i = 0; i < 100; ++i) {
        send_am(0);
    }
    flush();
    restore_errors();

    send_recv_am(0, UCS_ERR_ENDPOINT_TIMEOUT);

    m_am_count = 0;
    send_am(1);
    ucs_debug("flushing");
    flush_ep(1);
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

    static ucs_status_t err_cb_ep_destroy(void *arg, uct_ep_h ep, ucs_status_t status) {
        test_uct_peer_failure_cb *self(reinterpret_cast<test_uct_peer_failure_cb*>(arg));
        EXPECT_EQ(self->ep0(), ep);
        self->m_sender->destroy_ep(0);
        return UCS_OK;
    }
};

UCS_TEST_P(test_uct_peer_failure_cb, desproy_ep_cb)
{
    check_caps(UCT_IFACE_FLAG_PUT_SHORT);

    wrap_errors();
    kill_receiver();
    EXPECT_EQ(uct_ep_put_short(ep0(), NULL, 0, 0, 0), UCS_OK);
    flush();
    restore_errors();
}

UCT_INSTANTIATE_TEST_CASE(test_uct_peer_failure_cb)

class test_uct_peer_failure_multiple : public test_uct_peer_failure
{
public:
    virtual void init();

protected:
    size_t get_tx_queue_len() const;
};

void test_uct_peer_failure_multiple::init()
{
    if (RUNNING_ON_VALGRIND) {
        /* See https://bugs.kde.org/show_bug.cgi?id=352742 */
        UCS_TEST_SKIP_R("skipping on valgrind because \"brk segment overflow\"");
    }

    size_t tx_queue_len = get_tx_queue_len();

    if (ucs_get_page_size() > 4096) {
        /* NOTE: Too much receivers may cause failure of ibv_open_device */
        test_uct_peer_failure::m_nreceivers = 10;
    } else {
        test_uct_peer_failure::m_nreceivers = tx_queue_len;
    }

    test_uct_peer_failure::m_tx_window  = tx_queue_len / 3;

    test_uct_peer_failure::init();

    m_receivers.reserve(m_nreceivers);
    while (m_receivers.size() < m_nreceivers) {
        new_receiver();
    }
}

size_t test_uct_peer_failure_multiple::get_tx_queue_len() const
{
    const std::string   &tl_name = GetParam()->tl_name;
    std::string         name, val;
    size_t              tx_queue_len;

    if ((tl_name == "rc") || (tl_name == "rc_mlx5")) {
        name = "RC_IB_TX_QUEUE_LEN";
    } else if ((tl_name == "dc") || (tl_name == "dc_mlx5")) {
        name = "DC_RC_IB_TX_QUEUE_LEN";
    } else if ((tl_name == "ud") || (tl_name == "ud_mlx5")) {
        name = "UD_IB_TX_QUEUE_LEN";
    } else {
        name = "TX_QUEUE_LEN";
    }

    if (get_config(name, val)) {
        tx_queue_len = ucs::from_string<size_t>(val);
        EXPECT_LT(0ul, tx_queue_len);
    } else {
        tx_queue_len = 256;
        UCS_TEST_MESSAGE << name << " setting not found, "
                         << "taken test default value: " << tx_queue_len;
    }

    return tx_queue_len;
}

UCS_TEST_P(test_uct_peer_failure_multiple, test, "RC_TM_ENABLE?=n")
{
    ucs_time_t timeout  = ucs_get_time() +
                          ucs_time_from_sec(200 * ucs::test_time_multiplier());

    wrap_errors();
    for (size_t idx = 0; idx < m_nreceivers - 1; ++idx) {
        for (size_t i = 0; i < m_tx_window; ++i) {
            send_am(idx);
        }
        kill_receiver();
    }
    flush(timeout);

    /* if EPs are not failed yet, these ops should trigger that */
    for (size_t idx = 0; idx < m_nreceivers - 1; ++idx) {
        for (size_t i = 0; i < m_tx_window; ++i) {
            send_am(idx);
        }
    }

    flush(timeout);
    restore_errors();

    for (size_t idx = 0; idx < m_nreceivers - 1; ++idx) {
        send_recv_am(idx, UCS_ERR_ENDPOINT_TIMEOUT);
    }

    m_am_count = 0;
    send_am(m_nreceivers - 1);
    ucs_debug("flushing");
    flush_ep(m_nreceivers - 1);
    ucs_debug("flushed");
    wait_for_flag(&m_am_count);
    EXPECT_EQ(m_am_count, 1ul);
}

UCT_INSTANTIATE_TEST_CASE(test_uct_peer_failure_multiple)
