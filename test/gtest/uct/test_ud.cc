/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "uct_test.h"
extern "C" {
#include <ucs/time/time.h>
#include <ucs/datastruct/queue.h>
#include "uct/ib/ud/ud_ep.h"
};
/* TODO: disable ud async progress for most tests once we have it */
class test_ud : public uct_test {
public:
    virtual void init() {
        uct_test::init();

        m_e1 = uct_test::create_entity(0);
        m_e2 = uct_test::create_entity(0);

        m_e1->add_ep();
        m_e2->add_ep();

        m_entities.push_back(m_e1);
        m_entities.push_back(m_e2);
    }

    uct_ud_ep_t *ep(entity *e) {
        return ucs_derived_of(e->ep(0), uct_ud_ep_t);
    }

    uct_ud_ep_t *ep(entity *e, int i) {
        return ucs_derived_of(e->ep(i), uct_ud_ep_t);
    }

    void short_progress_loop() {
        ucs_time_t end_time = ucs_get_time() + ucs_time_from_msec(10.0* ucs::test_time_multiplier());
        while (ucs_get_time() < end_time) {
            progress();
        }
    }

    void connect() {
        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);
    }

    void cleanup() {
        uct_test::cleanup();
    }

    ucs_status_t tx(entity *e) {
        ucs_status_t err;
        err = uct_ep_put_short(e->ep(0), &m_dummy, sizeof(m_dummy), (uint64_t)&m_dummy, 0);
        return err;
    }

    void set_tx_win(entity *e, int size) {
        /* force window */
        ep(e)->tx.max_psn = size;
    }


protected:
    entity *m_e1, *m_e2;
    uint64_t m_dummy;
};

UCS_TEST_P(test_ud, basic_tx) {
    unsigned i, N=13;

    connect();
    set_tx_win(m_e1, 1024);
    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(m_e1));
    }
    short_progress_loop();

    /* N packets transmitted, N packets received */
    EXPECT_EQ(ep(m_e1)->tx.psn, N+1);
    EXPECT_EQ(ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts), N);

    /* no data transmitted back */
    EXPECT_EQ(ep(m_e2)->tx.psn, 1);

    /* nothing was acked */
    EXPECT_EQ(ucs_queue_length(&ep(m_e1)->tx.window), N);
    EXPECT_EQ(ep(m_e1)->tx.acked_psn, 0);
    EXPECT_EQ(ep(m_e2)->rx.acked_psn, 0);
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
    EXPECT_EQ(ep(m_e1)->tx.psn, N+1);
    EXPECT_EQ(ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts), N);
    EXPECT_EQ(ep(m_e2)->tx.psn, N+1);
    EXPECT_EQ(ucs_frag_list_sn(&ep(m_e1)->rx.ooo_pkts), N);

    /* everything but last packet from e2 is acked */
    EXPECT_EQ(ep(m_e1)->tx.acked_psn, N);
    EXPECT_EQ(ep(m_e2)->tx.acked_psn, N-1);
    EXPECT_EQ(ep(m_e1)->rx.acked_psn, N-1);
    EXPECT_EQ(ep(m_e2)->rx.acked_psn, N);
    EXPECT_EQ(ucs_queue_length(&ep(m_e2)->tx.window), 1U);
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
    EXPECT_EQ(tx(m_e1), UCS_ERR_NO_RESOURCE);
    short_progress_loop();
    EXPECT_UCS_OK(tx(m_e1));
    EXPECT_UCS_OK(tx(m_e1));
    EXPECT_UCS_OK(tx(m_e1));
}

#ifdef UCT_UD_EP_DEBUG_HOOKS
static ucs_status_t clear_ack_req(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
{
    neth->packet_type &= ~UCT_UD_PACKET_FLAG_ACK_REQ;
    return UCS_OK;
}

/* disable ack req,
 * send full window, 
 * should not be able to send some more 
 */
UCS_TEST_P(test_ud, tx_window2) {
    unsigned i, N=13;

    connect();
    set_tx_win(m_e1, N+1);
    ep(m_e1)->tx.tx_hook = clear_ack_req;

    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(m_e1));
    }
    EXPECT_EQ(tx(m_e1), UCS_ERR_NO_RESOURCE);
    short_progress_loop();
    EXPECT_EQ(tx(m_e1), UCS_ERR_NO_RESOURCE);
    EXPECT_EQ(tx(m_e1), UCS_ERR_NO_RESOURCE);
    EXPECT_EQ(tx(m_e1), UCS_ERR_NO_RESOURCE);
    EXPECT_EQ(ucs_queue_length(&ep(m_e1)->tx.window), N);
}

static int ack_req_tx_cnt;
uct_ud_psn_t tx_ack_psn;
static ucs_status_t ack_req_count_tx(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
{
    if (neth->packet_type & UCT_UD_PACKET_FLAG_ACK_REQ) {
        tx_ack_psn = neth->psn;
        ack_req_tx_cnt++;
    }
    return UCS_OK;
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
    EXPECT_EQ(ack_req_tx_cnt, 1);
    EXPECT_EQ(tx_ack_psn, 1);
    short_progress_loop();
    EXPECT_EQ(ack_req_tx_cnt, 2);
    EXPECT_EQ(tx_ack_psn, 1);
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
    EXPECT_EQ(ack_req_tx_cnt, 1);
    EXPECT_EQ(tx_ack_psn, N/4);
    short_progress_loop();
    EXPECT_EQ(ack_req_tx_cnt, 2);
    EXPECT_EQ(tx_ack_psn, N/4);
    EXPECT_TRUE(ucs_queue_is_empty(&ep(m_e1)->tx.window));
}
#endif

UCS_TEST_P(test_ud, connect_iface_single) {
    /* single connect */
    m_e1->connect_to_iface(0, *m_e2);
    short_progress_loop();
    EXPECT_EQ(ep(m_e1)->dest_ep_id, 1U);
    EXPECT_EQ(ep(m_e1)->conn_id, 0U);
    EXPECT_EQ(ep(m_e2)->dest_ep_id, (unsigned)UCT_UD_EP_NULL_ID);

    EXPECT_EQ(ep(m_e1)->tx.psn, 2);
    EXPECT_EQ(ucs_frag_list_sn(&ep(m_e1)->rx.ooo_pkts), 1);
    /* second active ep will not be used */
    EXPECT_EQ(ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts), 0);
}

UCS_TEST_P(test_ud, connect_iface_2to1) {
    /* 2 to 1 connect */
    m_e1->connect_to_iface(0, *m_e2);
    m_e1->add_ep();
    m_e1->connect_to_iface(1, *m_e2);
    short_progress_loop();

    EXPECT_EQ(ep(m_e1,0)->dest_ep_id, 1U);
    EXPECT_EQ(ep(m_e1,0)->conn_id, 0U);
    EXPECT_EQ(ep(m_e1,0)->tx.psn, 2);
    EXPECT_EQ(ucs_frag_list_sn(&ep(m_e1, 0)->rx.ooo_pkts), 1);

    EXPECT_EQ(ep(m_e1,1)->dest_ep_id, 2U);
    EXPECT_EQ(ep(m_e1,1)->conn_id, 1U);
    EXPECT_EQ(ep(m_e1,1)->tx.psn, 2);
    EXPECT_EQ(ucs_frag_list_sn(&ep(m_e1, 1)->rx.ooo_pkts), 1);
    EXPECT_EQ(ep(m_e2)->dest_ep_id, (unsigned)UCT_UD_EP_NULL_ID);
}

UCS_TEST_P(test_ud, connect_iface_seq) {
    /* sequential connect from both sides */
    m_e1->connect_to_iface(0, *m_e2);
    short_progress_loop();
    EXPECT_EQ(ep(m_e1)->dest_ep_id, 1U);
    EXPECT_EQ(ep(m_e1)->conn_id, 0U);
    EXPECT_EQ(ep(m_e1)->tx.psn, 2);
    EXPECT_EQ(ucs_frag_list_sn(&ep(m_e1)->rx.ooo_pkts), 1);
    EXPECT_EQ(ep(m_e2)->dest_ep_id, (unsigned)UCT_UD_EP_NULL_ID);

    /* now side two connects. existing ep will be reused */
    m_e2->connect_to_iface(0, *m_e1);
    short_progress_loop();
    EXPECT_EQ(ep(m_e2)->dest_ep_id, 0);
    EXPECT_EQ(ep(m_e2)->ep_id, 1);
    EXPECT_EQ(ep(m_e2)->conn_id, 0U);
    EXPECT_EQ(ep(m_e2)->tx.psn, 2);
    EXPECT_EQ(ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts), 1);
}

UCS_TEST_P(test_ud, connect_iface_sim) {
    /* simultanious connect from both sides */
    m_e1->connect_to_iface(0, *m_e2);
    m_e2->connect_to_iface(0, *m_e1);
    short_progress_loop();

    EXPECT_EQ(ep(m_e1)->dest_ep_id, 0);
    EXPECT_EQ(ep(m_e1)->conn_id, 0U);
    EXPECT_EQ(ep(m_e1)->ep_id, 0);

    EXPECT_EQ(ep(m_e2)->dest_ep_id, 0);
    EXPECT_EQ(ep(m_e2)->ep_id, 0);
    EXPECT_EQ(ep(m_e2)->conn_id, 0U);
    
    /* psns are not checked because it really depends on scheduling */
}

UCS_TEST_P(test_ud, connect_iface_sim2v2) {
    /* simultanious connect from both sides */
    m_e1->connect_to_iface(0, *m_e2);
    m_e2->connect_to_iface(0, *m_e1);
    m_e1->add_ep();
    m_e2->add_ep();
    m_e1->connect_to_iface(1, *m_e2);
    m_e2->connect_to_iface(1, *m_e1);
    short_progress_loop();

    EXPECT_EQ(ep(m_e1)->dest_ep_id, 0);
    EXPECT_EQ(ep(m_e1)->conn_id, 0U);
    EXPECT_EQ(ep(m_e1)->ep_id, 0);

    EXPECT_EQ(ep(m_e2)->dest_ep_id, 0);
    EXPECT_EQ(ep(m_e2)->ep_id, 0);
    EXPECT_EQ(ep(m_e2)->conn_id, 0U);
    
    EXPECT_EQ(ep(m_e1,1)->dest_ep_id, 1U);
    EXPECT_EQ(ep(m_e1,1)->conn_id, 1U);
    EXPECT_EQ(ep(m_e1,1)->ep_id, 1);

    EXPECT_EQ(ep(m_e2)->dest_ep_id, 0);
    EXPECT_EQ(ep(m_e2)->ep_id, 0);
    EXPECT_EQ(ep(m_e2)->conn_id, 0U);

    EXPECT_EQ(ep(m_e2,1)->dest_ep_id, 1U);
    EXPECT_EQ(ep(m_e2,1)->ep_id, 1);
    EXPECT_EQ(ep(m_e2,1)->conn_id, 1U);
    /* psns are not checked because it really depends on scheduling */
}

_UCT_INSTANTIATE_TEST_CASE(test_ud, ud)

