/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_test.h"
#include "ud_base.h"
extern "C" {
#include <ucs/time/time.h>
#include <ucs/datastruct/queue.h>
#include <uct/ib/ud/base/ud_ep.h>
};

class test_ud : public ud_base_test {
public:

    static ucs_status_t clear_ack_req(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        neth->packet_type &= ~UCT_UD_PACKET_FLAG_ACK_REQ;
        return UCS_OK;
    }

    static ucs_status_t drop_crep(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        if (neth->packet_type & UCT_UD_PACKET_FLAG_CTL) {
            return UCS_ERR_BUSY;
        }
        return UCS_OK;
    }

    static int rx_ack_count;

    static ucs_status_t count_rx_acks(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        if (UCT_UD_PSN_COMPARE(neth->ack_psn, >, ep->tx.acked_psn)) {
            rx_ack_count++;
        }
        return UCS_OK;
    }

    static int rx_drop_count;

    static ucs_status_t drop_rx(uct_ud_ep_t *ep, uct_ud_neth_t *neth) {
        rx_drop_count++;
        return UCS_ERR_BUSY;
    }

    static int ack_req_tx_cnt;

    static uct_ud_psn_t tx_ack_psn;

    static ucs_status_t ack_req_count_tx(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        if (neth->packet_type & UCT_UD_PACKET_FLAG_ACK_REQ) {
            tx_ack_psn = neth->psn;
            ack_req_tx_cnt++;
        }
        return UCS_OK;
    }

    void validate_flush() {
        /* 1 packets transmitted, 1 packets received */
        EXPECT_EQ(2, ep(m_e1)->tx.psn);
        EXPECT_EQ(1, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));

        /* no data transmitted back */
        EXPECT_EQ(1, ep(m_e2)->tx.psn);

        /* one packet was acked */
        EXPECT_EQ(0U, ucs_queue_length(&ep(m_e1)->tx.window));
        EXPECT_EQ(1, ep(m_e1)->tx.acked_psn);
        EXPECT_EQ(1, ep(m_e2)->rx.acked_psn);
    }

    void check_connection() {
        /* make sure that connection is good */
        EXPECT_UCS_OK(tx(m_e1));
        EXPECT_UCS_OK(tx(m_e1));
        flush();
        EXPECT_EQ(4, ep(m_e1, 0)->tx.psn);
        EXPECT_EQ(3, ep(m_e1)->tx.acked_psn);
    }
};

int test_ud::ack_req_tx_cnt = 0;
int test_ud::rx_ack_count   = 0;
int test_ud::rx_drop_count  = 0;

uct_ud_psn_t test_ud::tx_ack_psn = 0;

UCS_TEST_P(test_ud, basic_tx) {
    unsigned i, N=13;

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
    EXPECT_EQ(N, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));

    /* no data transmitted back */
    EXPECT_EQ(1, ep(m_e2)->tx.psn);

    /* nothing was acked */
    EXPECT_EQ(N, ucs_queue_length(&ep(m_e1)->tx.window));
    EXPECT_EQ(0, ep(m_e1)->tx.acked_psn);
    EXPECT_EQ(0, ep(m_e2)->rx.acked_psn);
}

UCS_TEST_P(test_ud, duplex_tx) {
    unsigned i, N=5;

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
    EXPECT_EQ(N, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
    EXPECT_EQ(N+1, ep(m_e2)->tx.psn);
    EXPECT_EQ(N, ucs_frag_list_sn(&ep(m_e1)->rx.ooo_pkts));

    /* everything but last packet from e2 is acked */
    EXPECT_EQ(N, ep(m_e1)->tx.acked_psn);
    EXPECT_EQ(N-1, ep(m_e2)->tx.acked_psn);
    EXPECT_EQ(N-1, ep(m_e1)->rx.acked_psn);
    EXPECT_EQ(N, ep(m_e2)->rx.acked_psn);
    EXPECT_EQ(1U, ucs_queue_length(&ep(m_e2)->tx.window));
    EXPECT_TRUE(ucs_queue_is_empty(&ep(m_e1)->tx.window));
}

/* send full window, rcv ack after progreess, send some more */
UCS_TEST_P(test_ud, tx_window1) {
    unsigned i, N=13;

    connect();
    set_tx_win(m_e1, N+1);
    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(m_e1));
    }
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, tx(m_e1));
    short_progress_loop();
    EXPECT_UCS_OK(tx(m_e1));
    EXPECT_UCS_OK(tx(m_e1));
    EXPECT_UCS_OK(tx(m_e1));
}

/* basic flush */
/* send packet, flush, wait till flush ended */

UCS_TEST_P(test_ud, flush_ep) {

    connect();
    EXPECT_UCS_OK(tx(m_e1));
    EXPECT_UCS_OK(ep_flush_b(m_e1));

    validate_flush();
}

UCS_TEST_P(test_ud, flush_iface) {

    connect();
    EXPECT_UCS_OK(tx(m_e1));
    EXPECT_UCS_OK(iface_flush_b(m_e1));

    validate_flush();
}

#ifdef UCT_UD_EP_DEBUG_HOOKS

/* disable ack req,
 * send full window, 
 * should not be able to send some more 
 */
UCS_TEST_P(test_ud, tx_window2) {
    unsigned i, N=13;

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
UCS_TEST_P(test_ud, ack_req_single) {

    connect();
    set_tx_win(m_e1, 2);
    ack_req_tx_cnt = 0;
    tx_ack_psn = 0;
    ep(m_e1)->tx.tx_hook = ack_req_count_tx;
    ep(m_e2)->rx.rx_hook = ack_req_count_tx;

    EXPECT_UCS_OK(tx(m_e1));
    EXPECT_EQ(1, ack_req_tx_cnt);
    EXPECT_EQ(1, tx_ack_psn);
    short_progress_loop();
    EXPECT_EQ(2, ack_req_tx_cnt);
    EXPECT_EQ(1, tx_ack_psn);
    EXPECT_TRUE(ucs_queue_is_empty(&ep(m_e1)->tx.window));
}

/* test that ack request is sent on 1/4 of window */
UCS_TEST_P(test_ud, ack_req_window) {
    unsigned i, N=16;

    disable_async(m_e1);
    disable_async(m_e2);
    connect();
    set_tx_win(m_e1, N);
    ack_req_tx_cnt = 0;
    tx_ack_psn = 0;
    ep(m_e1)->tx.tx_hook = ack_req_count_tx;
    ep(m_e2)->rx.rx_hook = ack_req_count_tx;

    for (i = 0; i < N/4; i++) {
        EXPECT_UCS_OK(tx(m_e1));
    }
    EXPECT_EQ(1, ack_req_tx_cnt);
    EXPECT_EQ(N/4, tx_ack_psn);
    short_progress_loop();
    EXPECT_EQ(2, ack_req_tx_cnt);
    EXPECT_EQ(N/4, tx_ack_psn);
    EXPECT_TRUE(ucs_queue_is_empty(&ep(m_e1)->tx.window));
}

/* simulate retransmission of the CREQ packet */
UCS_TEST_P(test_ud, creq_drop) {
    m_e1->connect_to_iface(0, *m_e2);
    /* setup filter to drop crep */
    ep(m_e1, 0)->rx.rx_hook = drop_crep;
    short_progress_loop();
    /* remove filter. Go to sleep. CREQ will be retransmitted */
    ep(m_e1, 0)->rx.rx_hook = uct_ud_ep_null_hook;
    twait(500);

    /* CREQ resend and connection shall be fully functional */
    EXPECT_EQ(0U, ep(m_e1, 0)->dest_ep_id);
    EXPECT_EQ(0U, ep(m_e1, 0)->conn_id);

    EXPECT_EQ(2, ep(m_e1, 0)->tx.psn);
    EXPECT_EQ(0, ucs_frag_list_sn(&ep(m_e1, 0)->rx.ooo_pkts));
    
    check_connection();
}

UCS_TEST_P(test_ud, creq_flush) {
    m_e1->connect_to_iface(0, *m_e2);
    /* setup filter to drop crep */
    ep(m_e1, 0)->rx.rx_hook = drop_crep;
    short_progress_loop();
    /* do flush while ep is not yet connected - should work */
    flush();
}

UCS_TEST_P(test_ud, ca_ai) {
    ucs_status_t status;
    int prev_cwnd;

    /* check initial window */
    disable_async(m_e1);
    disable_async(m_e2);
    connect();
    EXPECT_EQ(UCT_UD_CA_MIN_WINDOW, ep(m_e1)->ca.cwnd);
    EXPECT_EQ(UCT_UD_CA_MIN_WINDOW, ep(m_e2)->ca.cwnd);
    
    ep(m_e1, 0)->rx.rx_hook = count_rx_acks;
    prev_cwnd = ep(m_e1)->ca.cwnd;
    rx_ack_count = 0;

    /* window increase upto max window should 
     * happen when we receive acks */
    while(ep(m_e1)->ca.cwnd < UCT_UD_CA_MAX_WINDOW) {
       status = tx(m_e1);
       if (status != UCS_OK) {
           progress();
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

UCS_TEST_P(test_ud, ca_md) {

    ucs_status_t status;
    int new_cwnd;
    int i;

    connect();

    /* assume we are at the max window 
     * on receive drop all packets. After several retransmission
     * attempts the window will be reduced to the minumum 
     */
    set_tx_win(m_e1, UCT_UD_CA_MAX_WINDOW);
    ep(m_e2, 0)->rx.rx_hook = drop_rx;
    for (i = 1; i < UCT_UD_CA_MAX_WINDOW; i++) {
        status = tx(m_e1);
        EXPECT_UCS_OK(status);
        progress();
    }

    ep(m_e1)->tx.tx_hook = ack_req_count_tx;
    do {
        new_cwnd = ep(m_e1, 0)->ca.cwnd / UCT_UD_CA_MD_FACTOR;
        rx_drop_count = 0;
        ack_req_tx_cnt = 0;
        do {
            progress();
        } while (ep(m_e1, 0)->ca.cwnd != new_cwnd);
        /* more progress because resends may be waiting
         * for tx queue to become available */
        do { 
            progress();
        } while (ack_req_tx_cnt + 1 != new_cwnd);

        EXPECT_EQ(new_cwnd-1, ack_req_tx_cnt);
        EXPECT_EQ(ep(m_e1, 0)->ca.cwnd, new_cwnd);

    } while (ep(m_e1, 0)->ca.cwnd > UCT_UD_CA_MIN_WINDOW);
}

#endif

UCS_TEST_P(test_ud, connect_iface_single) {
    /* single connect */
    m_e1->connect_to_iface(0, *m_e2);
    short_progress_loop();
    EXPECT_EQ(0U, ep(m_e1, 0)->dest_ep_id);
    EXPECT_EQ(0U, ep(m_e1, 0)->conn_id);

    EXPECT_EQ(2, ep(m_e1, 0)->tx.psn);
    EXPECT_EQ(0, ucs_frag_list_sn(&ep(m_e1, 0)->rx.ooo_pkts));

    check_connection();
}

UCS_TEST_P(test_ud, connect_iface_2to1) {
    /* 2 to 1 connect */
    m_e1->connect_to_iface(0, *m_e2);
    m_e1->connect_to_iface(1, *m_e2);
    short_progress_loop();

    EXPECT_EQ(0U, ep(m_e1,0)->dest_ep_id);
    EXPECT_EQ(0U, ep(m_e1,0)->conn_id);
    EXPECT_EQ(2, ep(m_e1,0)->tx.psn);
    EXPECT_EQ(0, ucs_frag_list_sn(&ep(m_e1, 0)->rx.ooo_pkts));

    EXPECT_EQ(1U, ep(m_e1,1)->dest_ep_id);
    EXPECT_EQ(1U, ep(m_e1,1)->conn_id);
    EXPECT_EQ(2, ep(m_e1,1)->tx.psn);
    EXPECT_EQ(0, ucs_frag_list_sn(&ep(m_e1, 1)->rx.ooo_pkts));
}

UCS_TEST_P(test_ud, connect_iface_seq) {
    /* sequential connect from both sides */
    m_e1->connect_to_iface(0, *m_e2);
    short_progress_loop();
    EXPECT_EQ(0U, ep(m_e1)->dest_ep_id);
    EXPECT_EQ(0U, ep(m_e1)->conn_id);
    EXPECT_EQ(2, ep(m_e1)->tx.psn);
    EXPECT_EQ(0, ucs_frag_list_sn(&ep(m_e1)->rx.ooo_pkts));

    /* now side two connects. existing ep will be reused */
    m_e2->connect_to_iface(0, *m_e1);
    short_progress_loop();
    EXPECT_EQ(0U, ep(m_e2)->dest_ep_id);
    EXPECT_EQ(0U, ep(m_e2)->ep_id);
    EXPECT_EQ(0U, ep(m_e2)->conn_id);
    EXPECT_EQ(1, ep(m_e2)->tx.psn);
    /* one becase creq sets initial psn */
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));

    check_connection();
}

UCS_TEST_P(test_ud, connect_iface_sim) {
    /* simultanious connect from both sides */
    m_e1->connect_to_iface(0, *m_e2);
    m_e2->connect_to_iface(0, *m_e1);
    short_progress_loop();

    EXPECT_EQ(0U, ep(m_e1)->dest_ep_id);
    EXPECT_EQ(0U, ep(m_e1)->conn_id);
    EXPECT_EQ(0U, ep(m_e1)->ep_id);

    EXPECT_EQ(0U, ep(m_e2)->dest_ep_id);
    EXPECT_EQ(0U, ep(m_e2)->ep_id);
    EXPECT_EQ(0U, ep(m_e2)->conn_id);
    
    /* psns are not checked because it really depends on scheduling */
}

UCS_TEST_P(test_ud, connect_iface_sim2v2) {
    /* simultanious connect from both sides */
    m_e1->connect_to_iface(0, *m_e2);
    m_e2->connect_to_iface(0, *m_e1);
    m_e1->connect_to_iface(1, *m_e2);
    m_e2->connect_to_iface(1, *m_e1);
    short_progress_loop();

    EXPECT_EQ(0U, ep(m_e1)->dest_ep_id);
    EXPECT_EQ(0U, ep(m_e1)->conn_id);
    EXPECT_EQ(0U, ep(m_e1)->ep_id);

    EXPECT_EQ(0U, ep(m_e2)->dest_ep_id);
    EXPECT_EQ(0U, ep(m_e2)->ep_id);
    EXPECT_EQ(0U, ep(m_e2)->conn_id);
    
    EXPECT_EQ(1U, ep(m_e1,1)->dest_ep_id);
    EXPECT_EQ(1U, ep(m_e1,1)->conn_id);
    EXPECT_EQ(1U, ep(m_e1,1)->ep_id);

    EXPECT_EQ(0U, ep(m_e2)->dest_ep_id);
    EXPECT_EQ(0U, ep(m_e2)->ep_id);
    EXPECT_EQ(0U, ep(m_e2)->conn_id);

    EXPECT_EQ(1U, ep(m_e2,1)->dest_ep_id);
    EXPECT_EQ(1U, ep(m_e2,1)->ep_id);
    EXPECT_EQ(1U, ep(m_e2,1)->conn_id);
    /* psns are not checked because it really depends on scheduling */
}


UCS_TEST_P(test_ud, ep_destroy_simple) {
    uct_ep_h ep;
    ucs_status_t status;
    uct_ud_ep_t *ud_ep1, *ud_ep2;

    status = uct_ep_create(m_e1->iface(), &ep);
    EXPECT_UCS_OK(status);
    ud_ep1 = ucs_derived_of(ep, uct_ud_ep_t);
    uct_ep_destroy(ep);

    status = uct_ep_create(m_e1->iface(), &ep);
    EXPECT_UCS_OK(status);
    ud_ep2 = ucs_derived_of(ep, uct_ud_ep_t);
    uct_ep_destroy(ep);

    EXPECT_EQ(0U, ud_ep1->ep_id);
    EXPECT_EQ(1U, ud_ep2->ep_id);
}

UCS_TEST_P(test_ud, ep_destroy_flush) {
    uct_ep_h ep;
    ucs_status_t status;
    uct_ud_ep_t *ud_ep1;

    connect();
    EXPECT_UCS_OK(tx(m_e1));
    short_progress_loop();
    uct_ep_destroy(m_e1->ep(0));
    /* ep destroy should try to flush outstanding packets */
    short_progress_loop();
    validate_flush();

    /* next created ep must not reuse old id */
    status = uct_ep_create(m_e1->iface(), &ep);
    EXPECT_UCS_OK(status);
    ud_ep1 = ucs_derived_of(ep, uct_ud_ep_t);
    EXPECT_EQ(1U, ud_ep1->ep_id);
    uct_ep_destroy(ep);
}

UCS_TEST_P(test_ud, ep_destroy_passive) {
    connect();
    uct_ep_destroy(m_e2->ep(0));
    /* destroyed ep must still accept data */
    EXPECT_UCS_OK(tx(m_e1));
    EXPECT_UCS_OK(ep_flush_b(m_e1));

    validate_flush();
}

UCS_TEST_P(test_ud, ep_destroy_creq) {
    uct_ep_h ep;
    ucs_status_t status;
    uct_ud_ep_t *ud_ep;

    /* single connect */
    m_e1->connect_to_iface(0, *m_e2);
    short_progress_loop();

    uct_ep_destroy(m_e1->ep(0));

    /* check that ep id are not reused on both sides */
    status = uct_ep_create(m_e1->iface(), &ep);
    EXPECT_UCS_OK(status);
    ud_ep = ucs_derived_of(ep, uct_ud_ep_t);
    uct_ep_destroy(ep);
    EXPECT_EQ(1U, ud_ep->ep_id);

    status = uct_ep_create(m_e2->iface(), &ep);
    EXPECT_UCS_OK(status);
    ud_ep = ucs_derived_of(ep, uct_ud_ep_t);
    uct_ep_destroy(ep);
    EXPECT_EQ(1U, ud_ep->ep_id);
}

_UCT_INSTANTIATE_TEST_CASE(test_ud, ud)
_UCT_INSTANTIATE_TEST_CASE(test_ud, ud_mlx5)

