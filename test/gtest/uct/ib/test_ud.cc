/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
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
        sender().connect_to_iface(index, receiver());
        receiver().connect_to_iface(index, sender());
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
        EXPECT_EQ(base_psn + 1, ep(sender())->tx.psn);
        EXPECT_EQ(base_psn, ucs_frag_list_sn(&ep(receiver())->rx.ooo_pkts));

        /* no data transmitted back */
        EXPECT_EQ(base_psn, ep(receiver())->tx.psn);

        /* one packet was acked */
        EXPECT_EQ(0U, ucs_queue_length(&ep(sender())->tx.window));
        EXPECT_EQ(base_psn, ep(sender())->tx.acked_psn);
        EXPECT_EQ(base_psn, ep(receiver())->rx.acked_psn);
    }

    void check_connection() {
        /* make sure that connection is good */
        EXPECT_UCS_OK(tx(sender()));
        EXPECT_UCS_OK(tx(sender()));
        flush();
        EXPECT_EQ(4, ep(sender(), 0)->tx.psn);
        EXPECT_EQ(3, ep(sender())->tx.acked_psn);
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

    disable_async(sender());
    disable_async(receiver());
    connect();
    set_tx_win(sender(), 1024);
    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(sender()));
    }
    short_progress_loop();

    /* N packets transmitted, N packets received */
    EXPECT_EQ(N+1, ep(sender())->tx.psn);
    validate_recv(ep(receiver()), N);

    /* no data transmitted back */
    EXPECT_EQ(1, ep(receiver())->tx.psn);

    /* nothing was acked */
    EXPECT_EQ(N, ucs_queue_length(&ep(sender())->tx.window));
    EXPECT_EQ(0, ep(sender())->tx.acked_psn);
    EXPECT_EQ(0, ep(receiver())->rx.acked_psn);
}

UCS_TEST_SKIP_COND_P(test_ud, duplex_tx,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    unsigned i, N = 5;

    disable_async(sender());
    disable_async(receiver());
    connect();
    set_tx_win(sender(), 1024);
    set_tx_win(receiver(), 1024);
    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(sender()));
        short_progress_loop();
        EXPECT_UCS_OK(tx(receiver()));
        short_progress_loop();
    }

    /* N packets transmitted, N packets received */
    EXPECT_EQ(N+1, ep(sender())->tx.psn);
    validate_recv(ep(receiver()), N);

    EXPECT_EQ(N+1, ep(receiver())->tx.psn);
    validate_recv(ep(sender()), N);

    /* everything but last packet from e2 is acked */
    EXPECT_EQ(N, ep(sender())->tx.acked_psn);
    EXPECT_EQ(N-1, ep(receiver())->tx.acked_psn);
    EXPECT_EQ(N-1, ep(sender())->rx.acked_psn);
    EXPECT_EQ(N, ep(receiver())->rx.acked_psn);
    EXPECT_EQ(1U, ucs_queue_length(&ep(receiver())->tx.window));
    EXPECT_TRUE(ucs_queue_is_empty(&ep(sender())->tx.window));
}

/* send full window, rcv ack after progreess, send some more */
UCS_TEST_SKIP_COND_P(test_ud, tx_window1,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    unsigned i, N = 13;

    disable_async(sender());
    disable_async(receiver());
    connect();
    set_tx_win(sender(), N+1);
    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(sender()));
    }
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, tx(sender()));

    /* wait for ack */
    ucs_time_t timeout =
            ucs_get_time() + ucs_time_from_sec(TEST_UD_LINGER_TIMEOUT_IN_SEC);
    while ((ucs_get_time() < timeout) &&
            uct_ud_ep_no_window(ep(sender()))) {
        short_progress_loop();
    }
    EXPECT_UCS_OK(tx(sender()));
    EXPECT_UCS_OK(tx(sender()));
    EXPECT_UCS_OK(tx(sender()));
}

/* basic flush */
/* send packet, flush, wait till flush ended */

UCS_TEST_SKIP_COND_P(test_ud, flush_ep,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    connect();
    EXPECT_UCS_OK(tx(sender()));
    EXPECT_UCS_OK(ep_flush_b(sender()));

    validate_flush();
}

UCS_TEST_SKIP_COND_P(test_ud, flush_iface,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    connect();
    EXPECT_UCS_OK(tx(sender()));
    EXPECT_UCS_OK(iface_flush_b(sender()));

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

    disable_async(sender());
    disable_async(receiver());
    connect();
    set_tx_win(sender(), N+1);
    ep(sender())->tx.tx_hook = clear_ack_req;

    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(sender()));
    }
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, tx(sender()));
    short_progress_loop();
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, tx(sender()));
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, tx(sender()));
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, tx(sender()));
    EXPECT_EQ(N, ucs_queue_length(&ep(sender())->tx.window));
}


/* last packet in window must have ack_req
 * answered with ack control message
 */
UCS_TEST_SKIP_COND_P(test_ud, ack_req_single,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    connect();
    disable_async(sender());
    disable_async(receiver());
    set_tx_win(sender(), 2);
    ack_req_tx_cnt = 0;
    tx_ack_psn = 0;
    rx_ack_count = 0;
    ep(sender())->tx.tx_hook = ack_req_count_tx;
    ep(sender())->rx.rx_hook = count_rx_acks;
    ep(receiver())->rx.rx_hook = ack_req_count_tx;

    EXPECT_UCS_OK(tx(sender()));
    EXPECT_EQ(1, ack_req_tx_cnt);
    EXPECT_EQ(1, tx_ack_psn);

    wait_for_flag(&rx_ack_count);
    EXPECT_EQ(2, ack_req_tx_cnt);
    EXPECT_EQ(1, tx_ack_psn);
    EXPECT_TRUE(ucs_queue_is_empty(&ep(sender())->tx.window));
}

/* test that ack request is sent on 1/4 of window */
UCS_TEST_SKIP_COND_P(test_ud, ack_req_window,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    unsigned i, N = 16;

    disable_async(sender());
    disable_async(receiver());
    connect();
    set_tx_win(sender(), N);
    ack_req_tx_cnt = 0;
    tx_ack_psn = 0;
    rx_ack_count = 0;
    ep(sender())->tx.tx_hook   = ack_req_count_tx;
    ep(sender())->rx.rx_hook   = count_rx_acks;
    ep(receiver())->rx.rx_hook = ack_req_count_tx;

    for (i = 0; i < N/4; i++) {
        EXPECT_UCS_OK(tx(sender()));
    }
    EXPECT_EQ(1, ack_req_tx_cnt);
    EXPECT_EQ(N/4, tx_ack_psn);

    wait_for_flag(&rx_ack_count);
    EXPECT_EQ(2, ack_req_tx_cnt);
    EXPECT_EQ(N/4, tx_ack_psn);
    EXPECT_TRUE(ucs_queue_is_empty(&ep(sender())->tx.window));
}

/* simulate retransmission of the CREQ packet */
UCS_TEST_SKIP_COND_P(test_ud, crep_drop1,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    sender().connect_to_iface(0, receiver());
    /* setup filter to drop crep */
    ep(sender(), 0)->rx.rx_hook = drop_ctl;
    short_progress_loop(50);
    /* remove filter. Go to sleep. CREQ will be retransmitted */
    ep(sender(), 0)->rx.rx_hook = uct_ud_ep_null_hook;
    twait(500);

    /* CREQ resend and connection shall be fully functional */
    validate_connect(ep(sender()), 0U);

    EXPECT_EQ(2, ep(sender(), 0)->tx.psn);
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(sender(), 0)->rx.ooo_pkts));

    check_connection();
}

/* check that creq is not left on tx window if
 * both sides connect simultaniously.
 */
UCS_TEST_SKIP_COND_P(test_ud, crep_drop2,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    connect_to_iface();

    ep(sender())->rx.rx_hook   = drop_ctl;
    ep(receiver())->rx.rx_hook = drop_ctl;

    short_progress_loop(100);

    /* Remove filter for CREP to be handled and TX win to be freed. */
    ep(sender())->rx.rx_hook = uct_ud_ep_null_hook;
    ep(receiver())->rx.rx_hook = uct_ud_ep_null_hook;

    validate_connect(ep(sender()), 0U);
    validate_connect(ep(receiver()), 0U);

    /* Expect that creq (and maybe crep already) are sent */
    validate_send(ep(sender()), 1);
    validate_send(ep(receiver()), 1);
    EXPECT_GE(ep(sender())->tx.psn, 2);
    EXPECT_GE(ep(receiver())->tx.psn, 2);

    /* Wait for TX win to be empty (which means that all
     * CONN packets are handled) */
    ucs_time_t timeout =
            ucs_get_time() + ucs_time_from_sec(TEST_UD_LINGER_TIMEOUT_IN_SEC);
    while (ucs_get_time() < timeout) {
        if(ucs_queue_is_empty(&ep(sender())->tx.window) &&
           ucs_queue_is_empty(&ep(receiver())->tx.window)) {
            break;
        }
        short_progress_loop();
    }
    EXPECT_TRUE(ucs_queue_is_empty(&ep(sender())->tx.window));
    EXPECT_TRUE(ucs_queue_is_empty(&ep(receiver())->tx.window));
}

UCS_TEST_P(test_ud, crep_ack_drop) {
    ucs_status_t status;

    connect_to_iface();

    /* drop ACK from CERQ/CREP */
    ep(sender(), 0)->rx.rx_hook   = drop_ack;
    ep(receiver(), 0)->rx.rx_hook = drop_ack;

    short_progress_loop();

    status = uct_iface_set_am_handler(receiver().iface(), 0,
                                      (uct_am_callback_t)ucs_empty_function_return_success,
                                      NULL, UCT_CB_FLAG_ASYNC);
    ASSERT_UCS_OK(status);

    /* allow sending the active message, in case the congestion window is
     * already reduced to minimum (=2) by the slow timer, since CREP ACK
     * was not received.
     */
    set_tx_win(sender(), 10);

    do {
        status = send_am_message(sender());
        progress();
    } while (status == UCS_ERR_NO_RESOURCE);
    ASSERT_UCS_OK(status);

    validate_recv(ep(receiver()), 3u - no_creq_cnt(ep(sender())));

    ep(sender(), 0)->rx.rx_hook   = uct_ud_ep_null_hook;
    ep(receiver(), 0)->rx.rx_hook = uct_ud_ep_null_hook;

    /* Should receive both CREP and the active message */

    short_progress_loop();
    twait(500);
    short_progress_loop();

    status = send_am_message(sender());
    ASSERT_UCS_OK(status);

    short_progress_loop();

    sender().flush();
    receiver().flush();
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

    ep(receiver(), 0)->rx.rx_hook = drop_rx;
    rx_drop_count                 = 0;

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, resend_iov_buf, sizeof(resend_iov_buf),
                            UCT_MEM_HANDLE_NULL, 1);

    uint32_t received = 0;
    ASSERT_UCS_OK(uct_iface_set_am_handler(receiver().iface(), 0,
                                           test_ud_am_handler, &received,
                                           UCT_CB_FLAG_ASYNC));

    ucs_status_t status;
    do {
        status = uct_ep_am_short_iov(sender().ep(0), 0, iov, iovcnt);
        progress();
    } while (status == UCS_ERR_NO_RESOURCE);
    ASSERT_UCS_OK(status);

    wait_for_flag(&rx_drop_count);
    ep(receiver(), 0)->rx.rx_hook = uct_ud_ep_null_hook;
    wait_for_flag(&received);

    sender().flush();
    receiver().flush();
}

UCS_TEST_P(test_ud, creq_flush) {
    ucs_status_t status;

    sender().connect_to_iface(0, receiver());
    /* Setup filter to drop all packets. We have to drop CREP
     * and ACK_REQ packets. */
    ep(sender(), 0)->rx.rx_hook = drop_rx;
    short_progress_loop();
    /* do flush while ep is being connected it must return in progress */
    status = uct_iface_flush(sender().iface(), 0, NULL);
    EXPECT_EQ(UCS_INPROGRESS, status);
}

UCS_TEST_SKIP_COND_P(test_ud, ca_ai,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    ucs_status_t status;
    int prev_cwnd;
    int max_window;

    /* check initial window */
    disable_async(sender());
    disable_async(receiver());
    /* only test up to 'small' window when on valgrind
     * valgrind drops rx packets when window is too big and resends are disabled in this test
     */
    max_window = RUNNING_ON_VALGRIND ? 128 : UCT_UD_CA_MAX_WINDOW;
    connect();
    EXPECT_EQ(UCT_UD_CA_MIN_WINDOW, ep(sender())->ca.cwnd);
    EXPECT_EQ(UCT_UD_CA_MIN_WINDOW, ep(receiver())->ca.cwnd);

    ep(sender(), 0)->rx.rx_hook = count_rx_acks;
    ep(sender(), 0)->tx.tx_hook = save_tx_ackreqs;
    prev_cwnd = ep(sender())->ca.cwnd;
    rx_ack_count = 0;

    /* window increase upto max window should
     * happen when we receive acks */
    while (ep(sender())->ca.cwnd < max_window) {
       status = tx(sender());
       if (status != UCS_OK) {

           /* progress until getting all acks for our requests */
           do {
               progress();
           } while (UCT_UD_PSN_COMPARE(ep(sender())->tx.acked_psn, <, tx_ackreq_psn));

           /* it is possible to get no acks if tx queue is full.
            * But no more than 2 acks per window.
            * One at 1/4 and one at the end
            *
            * every new ack should cause window increase
            */
           EXPECT_LE(rx_ack_count, 2);
           EXPECT_EQ(rx_ack_count,
                     UCT_UD_CA_AI_VALUE * (ep(sender())->ca.cwnd - prev_cwnd));
           prev_cwnd = ep(sender())->ca.cwnd;
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

    validate_connect(ep(sender()), 0U);

    /* assume we are at the max window
     * on receive drop all packets. After several retransmission
     * attempts the window will be reduced to the minimum
     */
    uct_ud_enter(iface(sender()));
    set_tx_win(sender(), UCT_UD_CA_MAX_WINDOW);
    ep(receiver(), 0)->rx.rx_hook = drop_rx;
    uct_ud_leave(iface(sender()));

    num_sent = 0;
    while (num_sent < UCT_UD_CA_MAX_WINDOW) {
        status = tx(sender());
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

    ep(sender())->tx.tx_hook = count_tx;
    do {
        uct_ud_enter(iface(sender()));
        tx_count  = 0;
        prev_cwnd = ep(sender(), 0)->ca.cwnd;
        uct_ud_leave(iface(sender()));

        do {
            progress();
        } while (ep(sender(), 0)->ca.cwnd > (prev_cwnd / UCT_UD_CA_MD_FACTOR));
        short_progress_loop();

        uct_ud_enter(iface(sender()));
        new_cwnd     = ep(sender(), 0)->ca.cwnd;
        new_tx_count = tx_count;
        uct_ud_leave(iface(sender()));

        EXPECT_GE(new_tx_count, ucs_min(new_cwnd - 1, num_sent));
        if (new_cwnd > UCT_UD_CA_MIN_WINDOW) {
           /* up to 3 additional ack_reqs per each resend */
           int order = ucs_ilog2(prev_cwnd / new_cwnd);
           EXPECT_LE(new_tx_count, (prev_cwnd - new_cwnd + 3) * order);
        }

    } while (ep(sender(), 0)->ca.cwnd > UCT_UD_CA_MIN_WINDOW);
}

UCS_TEST_SKIP_COND_P(test_ud, ca_resend,
                     (RUNNING_ON_VALGRIND ||
                      !check_caps(UCT_IFACE_FLAG_AM_SHORT))) {

    int max_window = 9;
    int i;
    ucs_status_t status;

    connect();
    set_tx_win(sender(), max_window);

    ep(receiver(), 0)->rx.rx_hook = drop_rx;
    for (i = 1; i < max_window; i++) {
        status = tx(sender());
        EXPECT_UCS_OK(status);
    }
    short_progress_loop();
    rx_drop_count = 0;
    ack_req_tx_cnt = 0;
    do {
        progress();
    } while(ep(sender())->ca.cwnd > max_window/2);
    /* expect at least 1 drop and 1 ack req */
    short_progress_loop(100);
    EXPECT_GE(rx_drop_count, 1u);
    EXPECT_GE(ack_req_tx_cnt, 1u);
}

UCS_TEST_P(test_ud, connect_iface_single_drop_creq) {
    /* single connect */
    iface(receiver())->rx.hook = drop_creq;

    connect_to_iface();
    short_progress_loop(50);

    iface(receiver())->rx.hook = uct_ud_iface_null_hook;

    validate_connect(ep(receiver()), 0U);
}
#endif

UCS_TEST_SKIP_COND_P(test_ud, connect_iface_single,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    /* single connect */
    sender().connect_to_iface(0, receiver());
    short_progress_loop(TEST_UD_PROGRESS_TIMEOUT);
    validate_connect(ep(sender()), 0U);

    EXPECT_EQ(2, ep(sender(), 0)->tx.psn);
    EXPECT_EQ(1, ep(sender(), 0)->tx.acked_psn);
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(sender(), 0)->rx.ooo_pkts));

    check_connection();
}

UCS_TEST_P(test_ud, connect_iface_2to1) {
    /* 2 to 1 connect */
    sender().connect_to_iface(0, receiver());
    sender().connect_to_iface(1, receiver());

    validate_connect(ep(sender()), 0U);
    EXPECT_EQ(2, ep(sender(), 0)->tx.psn);
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(sender(), 0)->rx.ooo_pkts));

    validate_connect(ep(sender(), 1), 1U);
    EXPECT_EQ(2, ep(sender(), 1)->tx.psn);
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(sender(), 1)->rx.ooo_pkts));
}

UCS_TEST_SKIP_COND_P(test_ud, connect_iface_seq,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    /* sequential connect from both sides */
    sender().connect_to_iface(0, receiver());
    validate_connect(ep(sender()), 0U);
    EXPECT_EQ(2, ep(sender())->tx.psn);
    /* one becase of crep */
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(sender())->rx.ooo_pkts));

    /* now side two connects. existing ep will be reused */
    receiver().connect_to_iface(0, sender());
    validate_connect(ep(receiver()), 0U);
    EXPECT_EQ(2, ep(receiver())->tx.psn);
    /* one becase creq sets initial psn */
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(receiver())->rx.ooo_pkts));

    check_connection();
}

UCS_TEST_P(test_ud, connect_iface_sim) {
    /* simultanious connect from both sides */
    connect_to_iface();

    validate_connect(ep(sender()), 0U);
    validate_connect(ep(receiver()), 0U);

    /* psns are not checked because it really depends on scheduling */
}

UCS_TEST_P(test_ud, connect_iface_sim2v2) {
    /* simultanious connect from both sides */
    connect_to_iface(0);
    connect_to_iface(1);

    validate_connect(ep(sender()),      0U);
    validate_connect(ep(receiver()),    0U);
    validate_connect(ep(sender(),   1), 1U);
    validate_connect(ep(receiver(), 1), 1U);
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
        sender().connect_to_iface(i, receiver());
        cids[i] = UCT_UD_EP_NULL_ID;
    }

    flush();

    for (i = 0; i < count; i++) {
        ASSERT_EQ(cids[i], (unsigned)UCT_UD_EP_NULL_ID);
        cids[i] = ep(sender(), i)->dest_ep_id;
        ASSERT_NE((unsigned)UCT_UD_EP_NULL_ID, ep(sender(), i)->dest_ep_id);
        EXPECT_EQ(i, ep(sender(), i)->conn_sn);
        EXPECT_EQ(i, ep(sender(), i)->ep_id);
    }
}

UCS_TEST_P(test_ud, ep_destroy_simple) {
    uct_ep_h ep;
    ucs_status_t status;
    uct_ud_ep_t *ud_ep1, *ud_ep2;
    uct_ep_params_t ep_params;

    ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
    ep_params.iface      = sender().iface();

    status = uct_ep_create(&ep_params, &ep);
    EXPECT_UCS_OK(status);
    ud_ep1 = ucs_derived_of(ep, uct_ud_ep_t);
    uct_ep_destroy(ep);

    ep_params.iface = sender().iface();
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
    uct_ud_ep_t *ud_ep;
    uct_ep_params_t ep_params;

    connect();
    EXPECT_UCS_OK(tx(sender()));
    short_progress_loop();

    /* m_e1::ep[0] has to be revoked at the end of the testing */
    uct_ep_destroy(sender().ep(0));
    /* ep destroy should try to flush outstanding packets */
    short_progress_loop();
    validate_flush();

    /* next created ep must not reuse old id */
    ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
    ep_params.iface      = sender().iface();
    status = uct_ep_create(&ep_params, &ep);
    EXPECT_UCS_OK(status);
    ud_ep = ucs_derived_of(ep, uct_ud_ep_t);
    EXPECT_EQ(1U, ud_ep->ep_id);
    uct_ep_destroy(ep);

    /* revoke sender::ep[0] as it was destroyed manually */
    sender().revoke_ep(0);
}

UCS_TEST_SKIP_COND_P(test_ud, ep_destroy_passive,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT)) {
    connect_to_iface(0);
    short_progress_loop(TEST_UD_PROGRESS_TIMEOUT);

    /* receiver::ep[0] has to be revoked at the end of the testing */
    uct_ep_destroy(receiver().ep(0));

    /* destroyed ep must still accept data */
    EXPECT_UCS_OK(tx(sender()));
    EXPECT_UCS_OK(ep_flush_b(sender()));

    validate_flush(3);

    /* revoke receiver::ep[0] as it was destroyed manually */
    receiver().revoke_ep(0);
}

UCS_TEST_P(test_ud, ep_destroy_creq) {
    uct_ep_h ep;
    ucs_status_t status;
    uct_ud_ep_t *ud_ep;
    uct_ep_params ep_params;

    /* single connect */
    sender().connect_to_iface(0, receiver());
    short_progress_loop(TEST_UD_PROGRESS_TIMEOUT);

    sender().destroy_ep(0);

    /* check that ep id are not reused on both sides */
    ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
    ep_params.iface      = sender().iface();
    status = uct_ep_create(&ep_params, &ep);
    EXPECT_UCS_OK(status);
    ud_ep = ucs_derived_of(ep, uct_ud_ep_t);
    uct_ep_destroy(ep);
    EXPECT_EQ(1U, ud_ep->ep_id);

    ep_params.iface = receiver().iface();
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
    iface(receiver())->tx.available = 0;

    connect_to_iface();

    /* Simulate loss of CREQ to m_e1 */
    ep(receiver())->tx.tx_hook      = invalidate_creq_tx;
    iface(receiver())->tx.available = 128;
    iface(sender())->tx.available   = 128;

    /* Simulate loss of CREP to m_e1 */
    ep(sender())->rx.rx_hook = drop_ctl;
    short_progress_loop(300);

    /* m_e2 ep should be in connected state now, as it received CREQ which is
     * counter to its own CREQ. So, send a packet to m_e1 (which is not in
     * connected state yet) */
    ep(receiver())->tx.tx_hook = uct_ud_ep_null_hook;
    set_tx_win(receiver(), 128);
    EXPECT_UCS_OK(tx(receiver()));
    short_progress_loop();
    ep(sender())->rx.rx_hook = uct_ud_ep_null_hook;
    twait(500);

    validate_connect(ep(sender()), 0U);
    validate_connect(ep(receiver()), 0U);
}
#endif

UCT_INSTANTIATE_UD_TEST_CASE(test_ud)


class test_ud_peer_failure : public ud_base_test {
public:
    test_ud_peer_failure()
    {
        m_err_count = 0;
    }

    void init()
    {
        set_config("UD_TIMEOUT=3s");
        ud_base_test::init();
        connect();
    }

    void kill_receiver()
    {
        m_entities.remove(&receiver());
    }

    static ucs_status_t err_cb(void *arg, uct_ep_h ep, ucs_status_t status)
    {
        m_err_count++;
        return UCS_OK;
    }

    uct_error_handler_t get_err_handler() const
    {
        return err_cb;
    }

protected:
    static size_t m_err_count;
};

size_t test_ud_peer_failure::m_err_count = 0;

UCS_TEST_SKIP_COND_P(test_ud_peer_failure, send_am_after_kill,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT |
                                 UCT_IFACE_FLAG_EP_CHECK),
                     // Increase UD timer tick to not get NO_RESOURCE from AM
                     "UD_TIMER_TICK?=2s") {
    ucs_status_t status;

    flush();

    status = uct_ep_check(sender().ep(0), 0, NULL);
    ASSERT_UCS_OK(status);
    flush();
    // Allow keepalive request to complete
    short_progress_loop();

    // Set TX window to big enough value to not get UCS_ERR_NO_RESOURCE from
    // AM SHORT operations
    set_tx_win(sender(), 1024);

    // We are still alive
    EXPECT_EQ(0, m_err_count);

    kill_receiver();

    status = uct_ep_check(sender().ep(0), 0, NULL);
    ASSERT_UCS_OK(status);

    // Post AM operation to check that an error could be detected by EP_CHECK
    // when an endpoint has an in-flight AM operation
    const ucs_time_t loop_end_limit = ucs::get_deadline(10.0);
    while ((m_err_count == 0) && (ucs_get_time() < loop_end_limit)) {
        const uint64_t send_data = 0;
        status = uct_ep_am_short(sender().ep(0), 0, 0, &send_data,
                                 sizeof(send_data));
        if (m_err_count == 0) {
            ASSERT_UCS_OK(status);
            // Have a 2-second deadline to wait for peer failure and ensure
            // scheduling new AM operations while no error was detected
            wait_for_flag(&m_err_count, 2.0);
        }
    }

    EXPECT_EQ(1, m_err_count);
}

UCT_INSTANTIATE_UD_TEST_CASE(test_ud_peer_failure)


#ifdef HAVE_MLX5_DV
extern "C" {
#include <uct/ib/mlx5/ib_mlx5.h>
}
#endif

class test_ud_iface_attrs : public test_uct_iface_attrs {
public:
    attr_map_t get_num_iov() {
        attr_map_t iov_map;
#ifdef HAVE_MLX5_DV
        if (has_transport("ud_mlx5")) {
            // For am zcopy just small constant number of iovs is allowed
            // (to preserve some inline space for AM zcopy header)
            iov_map["am"] = UCT_IB_MLX5_AM_ZCOPY_MAX_IOV;

        } else
#endif
        {
            EXPECT_TRUE(has_transport("ud_verbs"));
            uct_ud_verbs_iface_t *iface = ucs_derived_of(e(0).iface(),
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
