/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.All rights reserved.
* See file LICENSE for terms.
*/

#include "test_rc.h"


#define UCT_RC_INSTANTIATE_TEST_CASE(_test_case) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, rc) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, rc_mlx5)


void test_rc::init()
{
    uct_test::init();

    m_e1 = uct_test::create_entity(0);
    m_entities.push_back(m_e1);

    m_e2 = uct_test::create_entity(0);
    m_entities.push_back(m_e2);

    connect();
}

void test_rc::connect()
{
    m_e1->connect(0, *m_e2, 0);
    m_e2->connect(0, *m_e1, 0);

    uct_iface_set_am_handler(m_e1->iface(), 0, am_dummy_handler,
                             NULL, UCT_AM_CB_FLAG_SYNC);
    uct_iface_set_am_handler(m_e2->iface(), 0, am_dummy_handler,
                             NULL, UCT_AM_CB_FLAG_SYNC);
}


class test_rc_max_wr : public test_rc {
protected:
    virtual void init() {
        ucs_status_t status1, status2;
        status1 = uct_config_modify(m_iface_config, "TX_MAX_WR", "32");
        status2 = uct_config_modify(m_iface_config, "TX_MAX_BB", "32");
        if (status1 != UCS_OK && status2 != UCS_OK) {
            UCS_TEST_ABORT("Error: cannot set rc max wr/bb");
        }
        test_rc::init();
    }
};

/* Check that max_wr stops from sending */
UCS_TEST_P(test_rc_max_wr, send_limit)
{
    /* first 32 messages should be OK */
    send_am_messages(m_e1, 32, UCS_OK);

    /* next message - should fail */
    send_am_messages(m_e1, 1, UCS_ERR_NO_RESOURCE);

    progress_loop();
    send_am_messages(m_e1, 1, UCS_OK);
}

UCT_RC_INSTANTIATE_TEST_CASE(test_rc_max_wr)


int test_rc_flow_control::req_count = 0;

ucs_status_t test_rc_flow_control::am_send(uct_pending_req_t *self)
{

    pending_send_request_t *req = ucs_container_of(self,
                                                   pending_send_request_t,
                                                   uct);

    ucs_status_t status = uct_ep_am_short(req->ep, 0, 0, NULL,0);
    ucs_debug("sending short with grant %d ",status);
    return status;
}

/* Check that FC window works as expected:
 * - If FC enabled, only 'wnd' messages can be sent in a row
 * - If FC is disabled 'wnd' does not limit senders flow  */
void test_rc_flow_control::test_general(int wnd, bool is_fc_enabled)
{
     set_fc_attributes(m_e1, is_fc_enabled, wnd,
                       ucs_max((int)(wnd*0.5), 1),
                       ucs_max((int)(wnd*0.25), 1));

     send_am_messages(m_e1, wnd, UCS_OK);
     send_am_messages(m_e1, 1, is_fc_enabled ?  UCS_ERR_NO_RESOURCE : UCS_OK);

     progress_loop();
     send_am_messages(m_e1, 1, UCS_OK);

     if (!is_fc_enabled) {
         /* Make valgrind happy, need to enable FC for proper cleanup */
         set_fc_attributes(m_e1, true, wnd, wnd, 1);
     }
}

void test_rc_flow_control::test_pending_grant(int wnd)
{
    /* Block send capabilities of m_e2 for fc grant to be
     * added to the pending queue. */
    disable_entity(m_e2);
    set_fc_attributes(m_e1, true, wnd, wnd, 1);

    send_am_messages(m_e1, wnd, UCS_OK);
    progress_loop();

    /* Now m_e1 should be blocked by FC window and FC grant
     * should be in pending queue of m_e2. */
    send_am_messages(m_e1, 1, UCS_ERR_NO_RESOURCE);
    EXPECT_EQ(get_fc_ptr(m_e1)->fc_wnd, 0);

    /* Enable send capabilities of m_e2 and send AM message
     * to force pending queue dispatch */
    enable_entity(m_e2);
    set_tx_moderation(m_e2, 1);
    send_am_messages(m_e2, 1, UCS_OK);
    progress_loop();

    /* Check that m_e1 got grant */
    send_am_messages(m_e1, 1, UCS_OK);
}

void test_rc_flow_control::test_pending_purge(int wnd, int num_pend_sends)
{
    uct_pending_req_t reqs[num_pend_sends];

    disable_entity(m_e2);
    set_fc_attributes(m_e1, true, wnd, wnd, 1);

    req_count = 0;

    send_am_messages(m_e1, wnd, UCS_OK);
    progress_loop();

    /* Now m2 ep should have FC grant message in the pending queue.
     * Add some user pending requests as well */
    for (int i = 0; i < num_pend_sends; i ++) {
        reqs[i].func = NULL; /* make valgrind happy */
        EXPECT_EQ(uct_ep_pending_add(m_e2->ep(0), &reqs[i]), UCS_OK);
    }
    uct_ep_pending_purge(m_e2->ep(0), purge_cb, NULL);
    EXPECT_EQ(num_pend_sends, req_count);
}


/* Check that FC window works as expected */
UCS_TEST_P(test_rc_flow_control, general_enabled)
{
    test_general(8, true);
}

UCS_TEST_P(test_rc_flow_control, general_disabled)
{
    test_general(8, true);
}

/* Test the scenario when ep is being destroyed while there is
 * FC grant message in the pending queue */
UCS_TEST_P(test_rc_flow_control, pending_only_fc)
{
    int wnd = 2;

    disable_entity(m_e2);
    set_fc_attributes(m_e1, true, wnd, wnd, 1);

    send_am_messages(m_e1, wnd, UCS_OK);
    progress_loop();

    m_e2->destroy_ep(0);
    ASSERT_TRUE(rc_iface(m_e2)->tx.arbiter.current == NULL);
}

/* Check that user callback passed to uct_ep_pending_purge is not
 * invoked for FC grant message */
UCS_TEST_P(test_rc_flow_control, pending_purge)
{
    test_pending_purge(2, 5);
}

UCS_TEST_P(test_rc_flow_control, pending_grant)
{
    test_pending_grant(5);
}

UCT_RC_INSTANTIATE_TEST_CASE(test_rc_flow_control)


#if ENABLE_STATS

void test_rc_flow_control_stats::test_general(int wnd)
{
    uint64_t v;

    set_fc_attributes(m_e1, true, wnd,
                      ucs_max((int)(wnd*0.5), 1),
                      ucs_max((int)(wnd*0.25), 1));

    send_am_messages(m_e1, wnd, UCS_OK);
    send_am_messages(m_e1, 1, UCS_ERR_NO_RESOURCE);

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_NO_CRED);
    EXPECT_EQ(1ul, v);

    progress_loop();
    send_am_messages(m_e1, 1, UCS_OK);

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_TX_HARD_REQ);
    EXPECT_EQ(1ul, v);

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_RX_PURE_GRANT);
    EXPECT_EQ(1ul, v);
}


UCS_TEST_P(test_rc_flow_control_stats, general)
{
    test_general(5);
}

UCS_TEST_P(test_rc_flow_control_stats, soft_request)
{
    uint64_t v;
    int wnd = 8;
    int s_thresh = 4;
    int h_thresh = 1;

    set_fc_attributes(m_e1, true, wnd, s_thresh, h_thresh);
    send_am_messages(m_e1, wnd - (s_thresh - 1), UCS_OK);
    progress_loop();

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_TX_SOFT_REQ);
    EXPECT_EQ(1ul, v);
    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e2)->stats, UCT_RC_FC_STAT_RX_SOFT_REQ);
    EXPECT_EQ(1ul, v);

    send_am_messages(m_e2, 1, UCS_OK);
    progress_loop();
    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_RX_GRANT);
    EXPECT_EQ(1ul, v);
    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e2)->stats, UCT_RC_FC_STAT_TX_GRANT);
    EXPECT_EQ(1ul, v);
}

UCT_RC_INSTANTIATE_TEST_CASE(test_rc_flow_control_stats)

#endif
