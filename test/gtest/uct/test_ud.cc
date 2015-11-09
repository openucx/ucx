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
};

int test_ud::ack_req_tx_cnt = 0;

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

UCS_TEST_P(test_ud, ep_flush_basic) {
    //unsigned i, N=13;

    connect();
    EXPECT_UCS_OK(tx(m_e1));
    EXPECT_UCS_OK(tx(m_e1));
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
#endif

UCS_TEST_P(test_ud, connect_iface_single) {
    /* single connect */
    m_e1->connect_to_iface(0, *m_e2);
    short_progress_loop();
    EXPECT_EQ(0U, ep(m_e1, 0)->dest_ep_id);
    EXPECT_EQ(0U, ep(m_e1, 0)->conn_id);

    EXPECT_EQ(2, ep(m_e1, 0)->tx.psn);
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(m_e1, 0)->rx.ooo_pkts));
}

UCS_TEST_P(test_ud, connect_iface_2to1) {
    /* 2 to 1 connect */
    m_e1->connect_to_iface(0, *m_e2);
    m_e1->connect_to_iface(1, *m_e2);
    short_progress_loop();

    EXPECT_EQ(0U, ep(m_e1,0)->dest_ep_id);
    EXPECT_EQ(0U, ep(m_e1,0)->conn_id);
    EXPECT_EQ(2, ep(m_e1,0)->tx.psn);
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(m_e1, 0)->rx.ooo_pkts));

    EXPECT_EQ(1U, ep(m_e1,1)->dest_ep_id);
    EXPECT_EQ(1U, ep(m_e1,1)->conn_id);
    EXPECT_EQ(2, ep(m_e1,1)->tx.psn);
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(m_e1, 1)->rx.ooo_pkts));
}

UCS_TEST_P(test_ud, connect_iface_seq) {
    /* sequential connect from both sides */
    m_e1->connect_to_iface(0, *m_e2);
    short_progress_loop();
    EXPECT_EQ(0U, ep(m_e1)->dest_ep_id);
    EXPECT_EQ(0U, ep(m_e1)->conn_id);
    EXPECT_EQ(2, ep(m_e1)->tx.psn);
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(m_e1)->rx.ooo_pkts));

    /* now side two connects. existing ep will be reused */
    m_e2->connect_to_iface(0, *m_e1);
    short_progress_loop();
    EXPECT_EQ(0U, ep(m_e2)->dest_ep_id);
    EXPECT_EQ(0U, ep(m_e2)->ep_id);
    EXPECT_EQ(0U, ep(m_e2)->conn_id);
    EXPECT_EQ(1, ep(m_e2)->tx.psn);
    EXPECT_EQ(ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts), 1);
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

_UCT_INSTANTIATE_TEST_CASE(test_ud, ud)
_UCT_INSTANTIATE_TEST_CASE(test_ud, ud_mlx5)

