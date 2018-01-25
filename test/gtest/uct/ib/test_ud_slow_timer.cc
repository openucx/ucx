/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ud_base.h"

#include <uct/uct_test.h>

extern "C" {
#include <ucs/time/time.h>
#include <ucs/datastruct/queue.h>
#include <uct/ib/ud/base/ud_ep.h>
#include <uct/ib/ud/base/ud_iface.h>
}


class test_ud_slow_timer : public ud_base_test {
public:
    /* ack while doing retransmit */
    static int packet_count, rx_limit;
    static ucs_status_t rx_npackets(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        if (packet_count++ < rx_limit) {
            return UCS_OK;
        }
        else { 
            return UCS_ERR_INVALID_PARAM;
        }
    }
    /* test slow timer and restransmit */
    static int tick_count;

    static ucs_status_t tick_counter(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                uct_ud_iface_t);

        /* hack to disable retransmit */
        ep->tx.send_time = ucs_twheel_get_time(&iface->async.slow_timer);
        tick_count++;
        return UCS_OK;
    }

    static ucs_status_t drop_packet(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        return UCS_ERR_INVALID_PARAM;
    }

    void wait_for_rx_sn(unsigned sn)
    {
        ucs_time_t deadline = ucs_get_time() +
                              ucs_time_from_sec(10) * ucs::test_time_multiplier();
        while ((ucs_get_time() < deadline) && (ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts) < sn)) {
            usleep(1000);
        }
    }
};

int test_ud_slow_timer::rx_limit = 10;
int test_ud_slow_timer::packet_count = 0;
int test_ud_slow_timer::tick_count = 0;


/* single packet received without progress */
UCS_TEST_P(test_ud_slow_timer, tx1) {
    connect();
    EXPECT_UCS_OK(tx(m_e1));
    twait(200);
    EXPECT_EQ(2, ep(m_e1)->tx.psn);
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
}

/* multiple packets received without progress */
UCS_TEST_P(test_ud_slow_timer, txn) {
    unsigned i, N=42;
    connect();
    set_tx_win(m_e1, 1024);
    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(m_e1));
    }
    wait_for_rx_sn(N);
    EXPECT_EQ(N+1, ep(m_e1)->tx.psn);
    EXPECT_EQ(N, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
}

#ifdef UCT_UD_EP_DEBUG_HOOKS
/* no traffic - no ticks */
UCS_TEST_P(test_ud_slow_timer, tick1) {
    connect();
    tick_count = 0;
    ep(m_e1)->timer_hook = tick_counter;
    twait(500);
    EXPECT_EQ(0, tick_count);
}

/* ticks while tx  window is not empty */
UCS_TEST_P(test_ud_slow_timer, tick2) {
    connect();
    tick_count = 0;
    ep(m_e1)->timer_hook = tick_counter;
    EXPECT_UCS_OK(tx(m_e1));
    twait(500);
    EXPECT_LT(0, tick_count);
}

/* retransmit one packet */

UCS_TEST_P(test_ud_slow_timer, retransmit1) {

    connect();
    ep(m_e2)->rx.rx_hook = drop_packet;
    EXPECT_UCS_OK(tx(m_e1));
    short_progress_loop();
    EXPECT_EQ(0, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
    ep(m_e2)->rx.rx_hook = uct_ud_ep_null_hook;
    EXPECT_EQ(2, ep(m_e1)->tx.psn);
    wait_for_rx_sn(1);
    EXPECT_EQ(2, ep(m_e1)->tx.psn);
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
}

/* retransmit many packets */
UCS_TEST_P(test_ud_slow_timer, retransmitn) {

    unsigned i, N=42;

    connect();
    set_tx_win(m_e1, 1024);
    ep(m_e2)->rx.rx_hook = drop_packet;
    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(m_e1));
    }
    short_progress_loop();
    EXPECT_EQ(0, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
    ep(m_e2)->rx.rx_hook = uct_ud_ep_null_hook;
    EXPECT_EQ(N+1, ep(m_e1)->tx.psn);
    wait_for_rx_sn(N);
    EXPECT_EQ(N+1, ep(m_e1)->tx.psn);
    EXPECT_EQ(N, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
}


UCS_TEST_P(test_ud_slow_timer, partial_drop) {

    unsigned i, N=24;
    int orig_avail;

    connect();
    set_tx_win(m_e1, 1024);
    packet_count = 0;
    rx_limit = 10;
    ep(m_e2)->rx.rx_hook = rx_npackets;
    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(m_e1));
    }
    short_progress_loop();
    EXPECT_EQ(rx_limit, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
    ep(m_e2)->rx.rx_hook = uct_ud_ep_null_hook;
    EXPECT_EQ(N+1, ep(m_e1)->tx.psn);
    orig_avail = iface(m_e1)->tx.available;
    /* allow only 6 outgoing packets. It will allow to get ack
     * from receiver
     */
    iface(m_e1)->tx.available = 6;
    twait(500);
    iface(m_e1)->tx.available = orig_avail-6;
    short_progress_loop();
    
    EXPECT_EQ(N+1, ep(m_e1)->tx.psn);
    wait_for_rx_sn(N);
    EXPECT_EQ(N, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
}
#endif

_UCT_INSTANTIATE_TEST_CASE(test_ud_slow_timer, ud)
_UCT_INSTANTIATE_TEST_CASE(test_ud_slow_timer, ud_mlx5)

