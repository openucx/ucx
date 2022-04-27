/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ud_base.h"

#include <uct/uct_test.h>

extern "C" {
#include <ucs/time/time.h>
#include <ucs/datastruct/queue.h>
#include <ucs/arch/atomic.h>
#include <ucs/arch/bitops.h>
#include <uct/ib/ud/base/ud_ep.h>
#include <uct/ib/ud/verbs/ud_verbs.h>
}


class test_ud : public ud_base_test {
public:

    static ucs_status_t clear_ack_req(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        neth->packet_type &= ~UCT_UD_PACKET_FLAG_ACK_REQ;
        return UCS_OK;
    }

    static ucs_status_t drop_ctl(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        if (neth->packet_type & UCT_UD_PACKET_FLAG_CTL) {
            return UCS_ERR_BUSY;
        }
        return UCS_OK;
    }

    static ucs_status_t count_rx_acks(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        if (UCT_UD_PSN_COMPARE(neth->ack_psn, >, ep->tx.acked_psn)) {
            ucs_atomic_add32(&rx_ack_count, 1);
        }
        return UCS_OK;
    }

    static ucs_status_t save_tx_ackreqs(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        if (neth->packet_type & UCT_UD_PACKET_FLAG_ACK_REQ) {
            tx_ackreq_psn = neth->psn;
        }
        return UCS_OK;
    }

    static ucs_status_t drop_rx(uct_ud_ep_t *ep, uct_ud_neth_t *neth) {
        ucs_atomic_add32(&rx_drop_count, 1);
        if (neth->packet_type & UCT_UD_PACKET_FLAG_ACK_REQ) {
            tx_ack_psn = neth->psn;
            ucs_atomic_add32(&ack_req_tx_cnt, 1);
            ucs_debug("RX: psn %u ack_req", neth->psn);
        }
        return UCS_ERR_BUSY;
    }

    static ucs_status_t ack_req_count_tx(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        if (neth->packet_type & UCT_UD_PACKET_FLAG_ACK_REQ) {
            tx_ack_psn = neth->psn;
            ucs_atomic_add32(&ack_req_tx_cnt, 1);
        }
        return UCS_OK;
    }

    static ucs_status_t count_tx(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        ucs_atomic_add32(&tx_count, 1);
        return UCS_OK;
    }

    static ucs_status_t invalidate_creq_tx(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        if ((neth->packet_type & UCT_UD_PACKET_FLAG_CTL) &&
            (uct_ud_neth_get_dest_id(neth) == UCT_UD_EP_NULL_ID)) {
            uct_ud_neth_set_dest_id(neth, 0xbeef);
        }
        return UCS_OK;
    }

    static ucs_status_t drop_ack(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        if (!(neth->packet_type & (UCT_UD_PACKET_FLAG_CTL|UCT_UD_PACKET_FLAG_AM))) {
            return UCS_ERR_BUSY;
        }
        return UCS_OK;
    }

    static ucs_status_t drop_creq(uct_ud_iface_t *iface, uct_ud_neth_t *neth)
    {
        if ((neth->packet_type & UCT_UD_PACKET_FLAG_CTL) &&
            ((uct_ud_ctl_hdr_t *)(neth + 1))->type == UCT_UD_PACKET_CREQ)
        {
            return UCS_ERR_BUSY;
        }

        return UCS_OK;
    }

    void connect_to_iface(unsigned index = 0)
    {
        m_e1->connect_to_iface(index, *m_e2);
        m_e2->connect_to_iface(index, *m_e1);
    }

    void validate_connect(uct_ud_ep_t *ep, unsigned value,
                          double timeout_sec=TEST_UD_LINGER_TIMEOUT_IN_SEC) {
        ucs_time_t timeout = ucs_get_time() + ucs_time_from_sec(timeout_sec);
        while ((ep->dest_ep_id != value) && (ucs_get_time() < timeout)) {
            progress();
        }
        EXPECT_EQ(value, ep->dest_ep_id);
        EXPECT_EQ(value, ep->conn_sn);
        EXPECT_EQ(value, ep->ep_id);
    }

    unsigned no_creq_cnt(uct_ud_ep_t *ep) {
        return (ep->flags & UCT_UD_EP_FLAG_CREQ_NOTSENT) ? 1 : 0;
    }

    void validate_send(uct_ud_ep_t *ep, unsigned value) {
        EXPECT_GE(ep->tx.acked_psn, value - no_creq_cnt(ep));
    }

    void validate_recv(uct_ud_ep_t *ep, unsigned value,
                       double timeout_sec=TEST_UD_LINGER_TIMEOUT_IN_SEC) {
        ucs_time_t timeout = ucs_get_time() + ucs_time_from_sec(timeout_sec);
        while ((ucs_frag_list_sn(&ep->rx.ooo_pkts) < value - no_creq_cnt(ep)) &&
               (ucs_get_time() < timeout)) {
            progress();
        }
        EXPECT_EQ(value - no_creq_cnt(ep), ucs_frag_list_sn(&ep->rx.ooo_pkts));
    }

    void validate_flush(unsigned base_psn = 1u) {
        /* 1 packets transmitted, 1 packets received */
        EXPECT_EQ(base_psn + 1, ep(m_e1)->tx.psn);
        EXPECT_EQ(base_psn, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));

        /* no data transmitted back */
        EXPECT_EQ(base_psn, ep(m_e2)->tx.psn);

        /* one packet was acked */
        EXPECT_EQ(0U, ucs_queue_length(&ep(m_e1)->tx.window));
        EXPECT_EQ(base_psn, ep(m_e1)->tx.acked_psn);
        EXPECT_EQ(base_psn, ep(m_e2)->rx.acked_psn);
    }

    void check_connection() {
        /* make sure that connection is good */
        EXPECT_UCS_OK(tx(m_e1));
        EXPECT_UCS_OK(tx(m_e1));
        flush();
        EXPECT_EQ(4, ep(m_e1, 0)->tx.psn);
        EXPECT_EQ(3, ep(m_e1)->tx.acked_psn);
    }


    static volatile uint32_t     rx_ack_count;
    static volatile uint32_t     rx_drop_count;
    static volatile uint32_t     ack_req_tx_cnt;
    static volatile uint32_t     tx_count;
    static volatile uct_ud_psn_t tx_ackreq_psn;
    static volatile uct_ud_psn_t tx_ack_psn;
};

volatile uint32_t      test_ud::ack_req_tx_cnt = 0;
volatile uint32_t      test_ud::rx_ack_count   = 0;
volatile uint32_t      test_ud::rx_drop_count  = 0;
volatile uint32_t      test_ud::tx_count  = 0;
volatile uct_ud_psn_t  test_ud::tx_ackreq_psn = 0;
volatile uct_ud_psn_t  test_ud::tx_ack_psn = 0;

UCS_TEST_SKIP_COND_P(test_ud, basic_tx,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    unsigned i, N = 13;

    disable_async(m_e1);
    disable_async(m_e2);
    connect();
    set_tx_win(m_e1, 1024);
    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(m_e1));
    }
    short_progress_loop();

    /* N packets transmitted, N packets received */
    EXPECT_EQ(N+1, ep(m_e1)->tx.psn);
    validate_recv(ep(m_e2), N);

    /* no data transmitted back */
    EXPECT_EQ(1, ep(m_e2)->tx.psn);

    /* nothing was acked */
    EXPECT_EQ(N, ucs_queue_length(&ep(m_e1)->tx.window));
    EXPECT_EQ(0, ep(m_e1)->tx.acked_psn);
    EXPECT_EQ(0, ep(m_e2)->rx.acked_psn);
}

UCS_TEST_SKIP_COND_P(test_ud, duplex_tx,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    unsigned i, N = 5;

    disable_async(m_e1);
    disable_async(m_e2);
    connect();
    set_tx_win(m_e1, 1024);
    set_tx_win(m_e2, 1024);
    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(m_e1));
        short_progress_loop();
        EXPECT_UCS_OK(tx(m_e2));
        short_progress_loop();
    }

    /* N packets transmitted, N packets received */
    EXPECT_EQ(N+1, ep(m_e1)->tx.psn);
    validate_recv(ep(m_e2), N);

    EXPECT_EQ(N+1, ep(m_e2)->tx.psn);
    validate_recv(ep(m_e1), N);

    /* everything but last packet from e2 is acked */
    EXPECT_EQ(N, ep(m_e1)->tx.acked_psn);
    EXPECT_EQ(N-1, ep(m_e2)->tx.acked_psn);
    EXPECT_EQ(N-1, ep(m_e1)->rx.acked_psn);
    EXPECT_EQ(N, ep(m_e2)->rx.acked_psn);
    EXPECT_EQ(1U, ucs_queue_length(&ep(m_e2)->tx.window));
    EXPECT_TRUE(ucs_queue_is_empty(&ep(m_e1)->tx.window));
}

/* send full window, rcv ack after progreess, send some more */
UCS_TEST_SKIP_COND_P(test_ud, tx_window1,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    unsigned i, N = 13;

    disable_async(m_e1);
    disable_async(m_e2);
    connect();
    set_tx_win(m_e1, N+1);
    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(m_e1));
    }
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, tx(m_e1));

    /* wait for ack */
    ucs_time_t timeout =
            ucs_get_time() + ucs_time_from_sec(TEST_UD_LINGER_TIMEOUT_IN_SEC);
    while ((ucs_get_time() < timeout) &&
            uct_ud_ep_no_window(ep(m_e1))) {
        short_progress_loop();
    }
    EXPECT_UCS_OK(tx(m_e1));
    EXPECT_UCS_OK(tx(m_e1));
    EXPECT_UCS_OK(tx(m_e1));
}

/* basic flush */
/* send packet, flush, wait till flush ended */

UCS_TEST_SKIP_COND_P(test_ud, flush_ep,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    connect();
    EXPECT_UCS_OK(tx(m_e1));
    EXPECT_UCS_OK(ep_flush_b(m_e1));

    validate_flush();
}

UCS_TEST_SKIP_COND_P(test_ud, flush_iface,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    connect();
    EXPECT_UCS_OK(tx(m_e1));
    EXPECT_UCS_OK(iface_flush_b(m_e1));

    validate_flush();
}

#if UCT_UD_EP_DEBUG_HOOKS

/* disable ack req,
 * send full window,
 * should not be able to send some more
 */
UCS_TEST_SKIP_COND_P(test_ud, tx_window2,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    unsigned i, N = 13;

    disable_async(m_e1);
    disable_async(m_e2);
    connect();
    set_tx_win(m_e1, N+1);
    ep(m_e1)->tx.tx_hook = clear_ack_req;

    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(m_e1));
    }
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, tx(m_e1));
    short_progress_loop();
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, tx(m_e1));
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, tx(m_e1));
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, tx(m_e1));
    EXPECT_EQ(N, ucs_queue_length(&ep(m_e1)->tx.window));
}


/* last packet in window must have ack_req
 * answered with ack control message
 */
UCS_TEST_SKIP_COND_P(test_ud, ack_req_single,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    connect();
    disable_async(m_e1);
    disable_async(m_e2);
    set_tx_win(m_e1, 2);
    ack_req_tx_cnt = 0;
    tx_ack_psn = 0;
    rx_ack_count = 0;
    ep(m_e1)->tx.tx_hook = ack_req_count_tx;
    ep(m_e1)->rx.rx_hook = count_rx_acks;
    ep(m_e2)->rx.rx_hook = ack_req_count_tx;

    EXPECT_UCS_OK(tx(m_e1));
    EXPECT_EQ(1, ack_req_tx_cnt);
    EXPECT_EQ(1, tx_ack_psn);

    wait_for_flag(&rx_ack_count);
    EXPECT_EQ(2, ack_req_tx_cnt);
    EXPECT_EQ(1, tx_ack_psn);
    EXPECT_TRUE(ucs_queue_is_empty(&ep(m_e1)->tx.window));
}

/* test that ack request is sent on 1/4 of window */
UCS_TEST_SKIP_COND_P(test_ud, ack_req_window,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    unsigned i, N = 16;

    disable_async(m_e1);
    disable_async(m_e2);
    connect();
    set_tx_win(m_e1, N);
    ack_req_tx_cnt = 0;
    tx_ack_psn = 0;
    rx_ack_count = 0;
    ep(m_e1)->tx.tx_hook = ack_req_count_tx;
    ep(m_e1)->rx.rx_hook = count_rx_acks;
    ep(m_e2)->rx.rx_hook = ack_req_count_tx;

    for (i = 0; i < N/4; i++) {
        EXPECT_UCS_OK(tx(m_e1));
    }
    EXPECT_EQ(1, ack_req_tx_cnt);
    EXPECT_EQ(N/4, tx_ack_psn);

    wait_for_flag(&rx_ack_count);
    EXPECT_EQ(2, ack_req_tx_cnt);
    EXPECT_EQ(N/4, tx_ack_psn);
    EXPECT_TRUE(ucs_queue_is_empty(&ep(m_e1)->tx.window));
}

/* simulate retransmission of the CREQ packet */
UCS_TEST_SKIP_COND_P(test_ud, crep_drop1,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    m_e1->connect_to_iface(0, *m_e2);
    /* setup filter to drop crep */
    ep(m_e1, 0)->rx.rx_hook = drop_ctl;
    short_progress_loop(50);
    /* remove filter. Go to sleep. CREQ will be retransmitted */
    ep(m_e1, 0)->rx.rx_hook = uct_ud_ep_null_hook;
    twait(500);

    /* CREQ resend and connection shall be fully functional */
    validate_connect(ep(m_e1), 0U);

    EXPECT_EQ(2, ep(m_e1, 0)->tx.psn);
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(m_e1, 0)->rx.ooo_pkts));

    check_connection();
}

/* check that creq is not left on tx window if
 * both sides connect simultaniously.
 */
UCS_TEST_SKIP_COND_P(test_ud, crep_drop2,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    connect_to_iface();

    ep(m_e1)->rx.rx_hook = drop_ctl;
    ep(m_e2)->rx.rx_hook = drop_ctl;

    short_progress_loop(100);

    /* Remove filter for CREP to be handled and TX win to be freed. */
    ep(m_e1)->rx.rx_hook = uct_ud_ep_null_hook;
    ep(m_e2)->rx.rx_hook = uct_ud_ep_null_hook;

    validate_connect(ep(m_e1), 0U);
    validate_connect(ep(m_e2), 0U);

    /* Expect that creq (and maybe crep already) are sent */
    validate_send(ep(m_e1), 1);
    validate_send(ep(m_e2), 1);
    EXPECT_GE(ep(m_e1)->tx.psn, 2);
    EXPECT_GE(ep(m_e2)->tx.psn, 2);

    /* Wait for TX win to be empty (which means that all
     * CONN packets are handled) */
    ucs_time_t timeout =
            ucs_get_time() + ucs_time_from_sec(TEST_UD_LINGER_TIMEOUT_IN_SEC);
    while (ucs_get_time() < timeout) {
        if(ucs_queue_is_empty(&ep(m_e1)->tx.window) &&
           ucs_queue_is_empty(&ep(m_e2)->tx.window)) {
            break;
        }
        short_progress_loop();
    }
    EXPECT_TRUE(ucs_queue_is_empty(&ep(m_e1)->tx.window));
    EXPECT_TRUE(ucs_queue_is_empty(&ep(m_e2)->tx.window));
}

UCS_TEST_P(test_ud, crep_ack_drop) {
    ucs_status_t status;

    connect_to_iface();

    /* drop ACK from CERQ/CREP */
    ep(m_e1, 0)->rx.rx_hook = drop_ack;
    ep(m_e2, 0)->rx.rx_hook = drop_ack;

    short_progress_loop();

    status = uct_iface_set_am_handler(m_e2->iface(), 0,
                                      (uct_am_callback_t)ucs_empty_function_return_success,
                                      NULL, UCT_CB_FLAG_ASYNC);
    ASSERT_UCS_OK(status);

    /* allow sending the active message, in case the congestion window is
     * already reduced to minimum (=2) by the slow timer, since CREP ACK
     * was not received.
     */
    set_tx_win(m_e1, 10);

    do {
        status = send_am_message(m_e1);
        progress();
    } while (status == UCS_ERR_NO_RESOURCE);
    ASSERT_UCS_OK(status);

    validate_recv(ep(m_e2), 3u - no_creq_cnt(ep(m_e1)));

    ep(m_e1, 0)->rx.rx_hook = uct_ud_ep_null_hook;
    ep(m_e2, 0)->rx.rx_hook = uct_ud_ep_null_hook;

    /* Should receive both CREP and the active message */

    short_progress_loop();
    twait(500);
    short_progress_loop();

    status = send_am_message(m_e1);
    ASSERT_UCS_OK(status);

    short_progress_loop();

    m_e1->flush();
    m_e2->flush();
}

static const char resend_iov_buf[] = "abc";

static ucs_status_t
test_ud_am_handler(void *arg, void *data, size_t length, unsigned flags)
{
    *(uint32_t*)arg = 1;
    EXPECT_STREQ(resend_iov_buf, (char*)data);
    return UCS_OK;
}

UCS_TEST_P(test_ud, resend_iov)
{
    connect_to_iface();

    ep(m_e2, 0)->rx.rx_hook = drop_rx;
    rx_drop_count           = 0;

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, resend_iov_buf, sizeof(resend_iov_buf),
                            UCT_MEM_HANDLE_NULL, 1);

    uint32_t received = 0;
    ASSERT_UCS_OK(uct_iface_set_am_handler(m_e2->iface(), 0, test_ud_am_handler,
                                           &received, UCT_CB_FLAG_ASYNC));

    ucs_status_t status;
    do {
        status = uct_ep_am_short_iov(m_e1->ep(0), 0, iov, iovcnt);
        progress();
    } while (status == UCS_ERR_NO_RESOURCE);
    ASSERT_UCS_OK(status);

    wait_for_flag(&rx_drop_count);
    ep(m_e2, 0)->rx.rx_hook = uct_ud_ep_null_hook;
    wait_for_flag(&received);

    m_e1->flush();
    m_e2->flush();
}

UCS_TEST_P(test_ud, creq_flush) {
    ucs_status_t status;

    m_e1->connect_to_iface(0, *m_e2);
    /* Setup filter to drop all packets. We have to drop CREP
     * and ACK_REQ packets. */
    ep(m_e1, 0)->rx.rx_hook = drop_rx;
    short_progress_loop();
    /* do flush while ep is being connected it must return in progress */
    status = uct_iface_flush(m_e1->iface(), 0, NULL);
    EXPECT_EQ(UCS_INPROGRESS, status);
}

UCS_TEST_SKIP_COND_P(test_ud, ca_ai,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    ucs_status_t status;
    int prev_cwnd;
    int max_window;

    /* check initial window */
    disable_async(m_e1);
    disable_async(m_e2);
    /* only test up to 'small' window when on valgrind
     * valgrind drops rx packets when window is too big and resends are disabled in this test
     */
    max_window = RUNNING_ON_VALGRIND ? 128 : UCT_UD_CA_MAX_WINDOW;
    connect();
    EXPECT_EQ(UCT_UD_CA_MIN_WINDOW, ep(m_e1)->ca.cwnd);
    EXPECT_EQ(UCT_UD_CA_MIN_WINDOW, ep(m_e2)->ca.cwnd);

    ep(m_e1, 0)->rx.rx_hook = count_rx_acks;
    ep(m_e1, 0)->tx.tx_hook = save_tx_ackreqs;
    prev_cwnd = ep(m_e1)->ca.cwnd;
    rx_ack_count = 0;

    /* window increase upto max window should
     * happen when we receive acks */
    while (ep(m_e1)->ca.cwnd < max_window) {
       status = tx(m_e1);
       if (status != UCS_OK) {

           /* progress until getting all acks for our requests */
           do {
               progress();
           } while (UCT_UD_PSN_COMPARE(ep(m_e1)->tx.acked_psn, <, tx_ackreq_psn));

           /* it is possible to get no acks if tx queue is full.
            * But no more than 2 acks per window.
            * One at 1/4 and one at the end
            *
            * every new ack should cause window increase
            */
           EXPECT_LE(rx_ack_count, 2);
           EXPECT_EQ(rx_ack_count,
                     UCT_UD_CA_AI_VALUE * (ep(m_e1)->ca.cwnd - prev_cwnd));
           prev_cwnd = ep(m_e1)->ca.cwnd;
           rx_ack_count = 0;
       }
    }
}

/* skip valgrind for now */
UCS_TEST_SKIP_COND_P(test_ud, ca_md,
                     (RUNNING_ON_VALGRIND ||
                      !check_caps(UCT_IFACE_FLAG_AM_SHORT)),
                     "IB_TX_QUEUE_LEN=" UCS_PP_MAKE_STRING(UCT_UD_CA_MAX_WINDOW)) {

    unsigned prev_cwnd, new_cwnd;
    uint32_t new_tx_count;
    ucs_status_t status;
    unsigned num_sent;

    connect();

    validate_connect(ep(m_e1), 0U);

    /* assume we are at the max window
     * on receive drop all packets. After several retransmission
     * attempts the window will be reduced to the minimum
     */
    uct_ud_enter(iface(m_e1));
    set_tx_win(m_e1, UCT_UD_CA_MAX_WINDOW);
    ep(m_e2, 0)->rx.rx_hook = drop_rx;
    uct_ud_leave(iface(m_e1));

    num_sent = 0;
    while (num_sent < UCT_UD_CA_MAX_WINDOW) {
        status = tx(m_e1);
        if (status == UCS_ERR_NO_RESOURCE) {
            // the congestion window can shrink by async timer if ACKs are
            // not received fast enough
            break;
        }
        ASSERT_UCS_OK(status);
        progress();
        ++num_sent;
    }
    short_progress_loop();

    UCS_TEST_MESSAGE << "sent " << num_sent << " packets";
    EXPECT_GE(num_sent, 1u); /* at least one packet should be sent */

    ep(m_e1)->tx.tx_hook = count_tx;
    do {
        uct_ud_enter(iface(m_e1));
        tx_count  = 0;
        prev_cwnd = ep(m_e1, 0)->ca.cwnd;
        uct_ud_leave(iface(m_e1));

        do {
            progress();
        } while (ep(m_e1, 0)->ca.cwnd > (prev_cwnd / UCT_UD_CA_MD_FACTOR));
        short_progress_loop();

        uct_ud_enter(iface(m_e1));
        new_cwnd     = ep(m_e1, 0)->ca.cwnd;
        new_tx_count = tx_count;
        uct_ud_leave(iface(m_e1));

        EXPECT_GE(new_tx_count, ucs_min(new_cwnd - 1, num_sent));
        if (new_cwnd > UCT_UD_CA_MIN_WINDOW) {
           /* up to 3 additional ack_reqs per each resend */
           int order = ucs_ilog2(prev_cwnd / new_cwnd);
           EXPECT_LE(new_tx_count, (prev_cwnd - new_cwnd + 3) * order);
        }

    } while (ep(m_e1, 0)->ca.cwnd > UCT_UD_CA_MIN_WINDOW);
}

UCS_TEST_SKIP_COND_P(test_ud, ca_resend,
                     (RUNNING_ON_VALGRIND ||
                      !check_caps(UCT_IFACE_FLAG_AM_SHORT))) {

    int max_window = 9;
    int i;
    ucs_status_t status;

    connect();
    set_tx_win(m_e1, max_window);

    ep(m_e2, 0)->rx.rx_hook = drop_rx;
    for (i = 1; i < max_window; i++) {
        status = tx(m_e1);
        EXPECT_UCS_OK(status);
    }
    short_progress_loop();
    rx_drop_count = 0;
    ack_req_tx_cnt = 0;
    do {
        progress();
    } while(ep(m_e1)->ca.cwnd > max_window/2);
    /* expect at least 1 drop and 1 ack req */
    short_progress_loop(100);
    EXPECT_GE(rx_drop_count, 1u);
    EXPECT_GE(ack_req_tx_cnt, 1u);
}

UCS_TEST_P(test_ud, connect_iface_single_drop_creq) {
    /* single connect */
    iface(m_e2)->rx.hook = drop_creq;

    connect_to_iface();
    short_progress_loop(50);

    iface(m_e2)->rx.hook = uct_ud_iface_null_hook;

    validate_connect(ep(m_e2), 0U);
}
#endif

UCS_TEST_SKIP_COND_P(test_ud, connect_iface_single,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    /* single connect */
    m_e1->connect_to_iface(0, *m_e2);
    short_progress_loop(TEST_UD_PROGRESS_TIMEOUT);
    validate_connect(ep(m_e1), 0U);

    EXPECT_EQ(2, ep(m_e1, 0)->tx.psn);
    EXPECT_EQ(1, ep(m_e1, 0)->tx.acked_psn);
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(m_e1, 0)->rx.ooo_pkts));

    check_connection();
}

UCS_TEST_P(test_ud, connect_iface_2to1) {
    /* 2 to 1 connect */
    m_e1->connect_to_iface(0, *m_e2);
    m_e1->connect_to_iface(1, *m_e2);

    validate_connect(ep(m_e1), 0U);
    EXPECT_EQ(2, ep(m_e1,0)->tx.psn);
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(m_e1, 0)->rx.ooo_pkts));

    validate_connect(ep(m_e1, 1), 1U);
    EXPECT_EQ(2, ep(m_e1,1)->tx.psn);
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(m_e1, 1)->rx.ooo_pkts));
}

UCS_TEST_SKIP_COND_P(test_ud, connect_iface_seq,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    /* sequential connect from both sides */
    m_e1->connect_to_iface(0, *m_e2);
    validate_connect(ep(m_e1), 0U);
    EXPECT_EQ(2, ep(m_e1)->tx.psn);
    /* one becase of crep */
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(m_e1)->rx.ooo_pkts));

    /* now side two connects. existing ep will be reused */
    m_e2->connect_to_iface(0, *m_e1);
    validate_connect(ep(m_e2), 0U);
    EXPECT_EQ(2, ep(m_e2)->tx.psn);
    /* one becase creq sets initial psn */
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));

    check_connection();
}

UCS_TEST_P(test_ud, connect_iface_sim) {
    /* simultanious connect from both sides */
    connect_to_iface();

    validate_connect(ep(m_e1), 0U);
    validate_connect(ep(m_e2), 0U);

    /* psns are not checked because it really depends on scheduling */
}

UCS_TEST_P(test_ud, connect_iface_sim2v2) {
    /* simultanious connect from both sides */
    connect_to_iface(0);
    connect_to_iface(1);

    validate_connect(ep(m_e1),    0U);
    validate_connect(ep(m_e2),    0U);
    validate_connect(ep(m_e1, 1), 1U);
    validate_connect(ep(m_e2, 1), 1U);
    /* psns are not checked because it really depends on scheduling */
}

/*
 * check that:
 * - connect is not blocking when we run out of iface resources
 * - flush() will also progress pending CREQs
 */
UCS_TEST_P(test_ud, connect_iface_2k) {

    unsigned i;
    unsigned cids[2000];
    unsigned count = 2000 / ucs::test_time_multiplier();

    /* create 2k connections */
    for (i = 0; i < count; i++) {
        m_e1->connect_to_iface(i, *m_e2);
        cids[i] = UCT_UD_EP_NULL_ID;
    }

    flush();

    for (i = 0; i < count; i++) {
        ASSERT_EQ(cids[i], (unsigned)UCT_UD_EP_NULL_ID);
        cids[i] = ep(m_e1,i)->dest_ep_id;
        ASSERT_NE((unsigned)UCT_UD_EP_NULL_ID, ep(m_e1,i)->dest_ep_id);
        EXPECT_EQ(i, ep(m_e1,i)->conn_sn);
        EXPECT_EQ(i, ep(m_e1,i)->ep_id);
    }
}

UCS_TEST_P(test_ud, ep_destroy_simple) {
    uct_ep_h ep;
    ucs_status_t status;
    uct_ud_ep_t *ud_ep1, *ud_ep2;
    uct_ep_params_t ep_params;

    ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
    ep_params.iface      = m_e1->iface();

    status = uct_ep_create(&ep_params, &ep);
    EXPECT_UCS_OK(status);
    ud_ep1 = ucs_derived_of(ep, uct_ud_ep_t);
    uct_ep_destroy(ep);

    ep_params.iface = m_e1->iface();
    status = uct_ep_create(&ep_params, &ep);
    EXPECT_UCS_OK(status);
    /* coverity[use_after_free] */
    ud_ep2 = ucs_derived_of(ep, uct_ud_ep_t);
    uct_ep_destroy(ep);

    EXPECT_EQ(0U, ud_ep1->ep_id);
    EXPECT_EQ(1U, ud_ep2->ep_id);
}

UCS_TEST_SKIP_COND_P(test_ud, ep_destroy_flush,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    uct_ep_h ep;
    ucs_status_t status;
    uct_ud_ep_t *ud_ep1;
    uct_ep_params_t ep_params;

    connect();
    EXPECT_UCS_OK(tx(m_e1));
    short_progress_loop();

    /* m_e1::ep[0] has to be revoked at the end of the testing */
    uct_ep_destroy(m_e1->ep(0));
    /* ep destroy should try to flush outstanding packets */
    short_progress_loop();
    validate_flush();

    /* next created ep must not reuse old id */
    ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
    ep_params.iface      = m_e1->iface();
    status = uct_ep_create(&ep_params, &ep);
    EXPECT_UCS_OK(status);
    ud_ep1 = ucs_derived_of(ep, uct_ud_ep_t);
    EXPECT_EQ(1U, ud_ep1->ep_id);
    uct_ep_destroy(ep);

    /* revoke m_e1::ep[0] as it was destroyed manually */
    m_e1->revoke_ep(0);
}

UCS_TEST_SKIP_COND_P(test_ud, ep_destroy_passive,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    connect_to_iface(0);
    short_progress_loop(TEST_UD_PROGRESS_TIMEOUT);

    /* m_e2::ep[0] has to be revoked at the end of the testing */
    uct_ep_destroy(m_e2->ep(0));

    /* destroyed ep must still accept data */
    EXPECT_UCS_OK(tx(m_e1));
    EXPECT_UCS_OK(ep_flush_b(m_e1));

    validate_flush(3);

    /* revoke m_e2::ep[0] as it was destroyed manually */
    m_e2->revoke_ep(0);
}

UCS_TEST_P(test_ud, ep_destroy_creq) {
    uct_ep_h ep;
    ucs_status_t status;
    uct_ud_ep_t *ud_ep;
    uct_ep_params ep_params;

    /* single connect */
    m_e1->connect_to_iface(0, *m_e2);
    short_progress_loop(TEST_UD_PROGRESS_TIMEOUT);

    m_e1->destroy_ep(0);

    /* check that ep id are not reused on both sides */
    ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
    ep_params.iface      = m_e1->iface();
    status = uct_ep_create(&ep_params, &ep);
    EXPECT_UCS_OK(status);
    ud_ep = ucs_derived_of(ep, uct_ud_ep_t);
    uct_ep_destroy(ep);
    EXPECT_EQ(1U, ud_ep->ep_id);

    ep_params.iface = m_e2->iface();
    status = uct_ep_create(&ep_params, &ep);
    EXPECT_UCS_OK(status);
    /* coverity[use_after_free] */
    ud_ep = ucs_derived_of(ep, uct_ud_ep_t);
    uct_ep_destroy(ep);
    EXPECT_EQ(1U, ud_ep->ep_id);
}

#if UCT_UD_EP_DEBUG_HOOKS
/* Simulate loss of ctl packets during simultaneous CREQs.
 * Use-case: CREQ and CREP packets from m_e2 to m_e1 are lost.
 * Check: that both eps (m_e1 and m_e2) are connected finally */
UCS_TEST_SKIP_COND_P(test_ud, ctls_loss,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    iface(m_e2)->tx.available = 0;

    connect_to_iface();

    /* Simulate loss of CREQ to m_e1 */
    ep(m_e2)->tx.tx_hook = invalidate_creq_tx;
    iface(m_e2)->tx.available = 128;
    iface(m_e1)->tx.available = 128;

    /* Simulate loss of CREP to m_e1 */
    ep(m_e1)->rx.rx_hook = drop_ctl;
    short_progress_loop(300);

    /* m_e2 ep should be in connected state now, as it received CREQ which is
     * counter to its own CREQ. So, send a packet to m_e1 (which is not in
     * connected state yet) */
    ep(m_e2)->tx.tx_hook = uct_ud_ep_null_hook;
    set_tx_win(m_e2, 128);
    EXPECT_UCS_OK(tx(m_e2));
    short_progress_loop();
    ep(m_e1)->rx.rx_hook = uct_ud_ep_null_hook;
    twait(500);

    validate_connect(ep(m_e1), 0U);
    validate_connect(ep(m_e2), 0U);
}
#endif

UCT_INSTANTIATE_UD_TEST_CASE(test_ud)

#ifdef HAVE_MLX5_HW
extern "C" {
#include <uct/ib/mlx5/ib_mlx5.h>
}
#endif

class test_ud_iface_attrs : public test_uct_iface_attrs {
public:
    attr_map_t get_num_iov() {
        attr_map_t iov_map;
#ifdef HAVE_MLX5_HW
        if (has_transport("ud_mlx5")) {
            // For am zcopy just small constant number of iovs is allowed
            // (to preserve some inline space for AM zcopy header)
            iov_map["am"] = UCT_IB_MLX5_AM_ZCOPY_MAX_IOV;

        } else
#endif
        {
            EXPECT_TRUE(has_transport("ud_verbs"));
            uct_ud_verbs_iface_t *iface = ucs_derived_of(m_e->iface(),
                                                         uct_ud_verbs_iface_t);
            size_t max_sge = 0;
            EXPECT_UCS_OK(uct_ud_verbs_qp_max_send_sge(iface, &max_sge));
            iov_map["am"]  = max_sge;
        }

        return iov_map;
    }
};

UCS_TEST_P(test_ud_iface_attrs, iface_attrs)
{
    basic_iov_test();
}

UCT_INSTANTIATE_UD_TEST_CASE(test_ud_iface_attrs)

