/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
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
                             NULL, UCT_CB_FLAG_SYNC);
    uct_iface_set_am_handler(m_e2->iface(), 0, am_dummy_handler,
                             NULL, UCT_CB_FLAG_SYNC);
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


uint32_t test_rc_flow_control::m_am_rx_count = 0;

void test_rc_flow_control::init()
{
    /* For correct testing FC needs to be initialized during interface creation */
    if (UCS_OK != uct_config_modify(m_iface_config, "RC_FC_ENABLE", "y")) {
        UCS_TEST_ABORT("Error: cannot enable flow control");
    }
    test_rc::init();

    ucs_assert(rc_iface(m_e1)->config.fc_enabled);
    ucs_assert(rc_iface(m_e2)->config.fc_enabled);

    uct_iface_set_am_handler(m_e1->iface(), FLUSH_AM_ID, am_handler,
                             NULL, UCT_CB_FLAG_SYNC);
    uct_iface_set_am_handler(m_e2->iface(), FLUSH_AM_ID, am_handler,
                             NULL, UCT_CB_FLAG_SYNC);

}

void test_rc_flow_control::cleanup()
{
    /* Restore FC state to enabled, so iface cleanup will destroy the grant mpool */
    rc_iface(m_e1)->config.fc_enabled = 1;
    rc_iface(m_e2)->config.fc_enabled = 1;
    test_rc::cleanup();
}

void test_rc_flow_control::send_am_and_flush(entity *e, int num_msg)
{
    m_am_rx_count = 0;

    send_am_messages(e, num_msg - 1, UCS_OK);
    send_am_messages(e, 1, UCS_OK, FLUSH_AM_ID); /* send last msg with FLUSH id */
    wait_for_flag(&m_am_rx_count);
    EXPECT_EQ(m_am_rx_count, 1ul);
}

void test_rc_flow_control::validate_grant(entity *e)
{
    wait_for_flag(&get_fc_ptr(e)->fc_wnd);
    EXPECT_GT(get_fc_ptr(e)->fc_wnd, 0);
}

/* Check that FC window works as expected:
 * - If FC enabled, only 'wnd' messages can be sent in a row
 * - If FC is disabled 'wnd' does not limit senders flow  */
void test_rc_flow_control::test_general(int wnd, int soft_thresh,
                                        int hard_thresh, bool is_fc_enabled)
{
    set_fc_attributes(m_e1, is_fc_enabled, wnd, soft_thresh, hard_thresh);

    send_am_messages(m_e1, wnd, UCS_OK);
    send_am_messages(m_e1, 1, is_fc_enabled ?  UCS_ERR_NO_RESOURCE : UCS_OK);

    validate_grant(m_e1);
    send_am_messages(m_e1, 1, UCS_OK);

    if (!is_fc_enabled) {
        /* Make valgrind happy, need to enable FC for proper cleanup */
        set_fc_attributes(m_e1, true, wnd, wnd, 1);
    }
    flush();
}

void test_rc_flow_control::test_pending_grant(int wnd)
{
    /* Block send capabilities of m_e2 for fc grant to be
     * added to the pending queue. */
    disable_entity(m_e2);
    set_fc_attributes(m_e1, true, wnd, wnd, 1);

    send_am_and_flush(m_e1, wnd);

    /* Now m_e1 should be blocked by FC window and FC grant
     * should be in pending queue of m_e2. */
    send_am_messages(m_e1, 1, UCS_ERR_NO_RESOURCE);
    EXPECT_LE(get_fc_ptr(m_e1)->fc_wnd, 0);

    /* Enable send capabilities of m_e2 and send AM message
     * to force pending queue dispatch */
    enable_entity(m_e2);
    set_tx_moderation(m_e2, 0);
    send_am_messages(m_e2, 1, UCS_OK);

    /* Check that m_e1 got grant */
    validate_grant(m_e1);
    send_am_messages(m_e1, 1, UCS_OK);
}

void test_rc_flow_control::test_flush_fc_disabled()
{
    set_fc_disabled(m_e1);
    ucs_status_t status;

    /* If FC is disabled, wnd=0 should not prevent the flush */
    get_fc_ptr(m_e1)->fc_wnd = 0;
    status = uct_ep_flush(m_e1->ep(0), 0, NULL);
    EXPECT_EQ(UCS_OK, status);

    /* send active message should be OK */
    get_fc_ptr(m_e1)->fc_wnd = 1;
    status = uct_ep_am_short(m_e1->ep(0), 0, 0, NULL, 0);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(0, get_fc_ptr(m_e1)->fc_wnd);

    /* flush must have resources */
    status = uct_ep_flush(m_e1->ep(0), 0, NULL);
    EXPECT_FALSE(UCS_STATUS_IS_ERR(status)) << ucs_status_string(status);
}

void test_rc_flow_control::test_pending_purge(int wnd, int num_pend_sends)
{
    pending_send_request_t reqs[num_pend_sends];

    disable_entity(m_e2);
    set_fc_attributes(m_e1, true, wnd, wnd, 1);

    send_am_and_flush(m_e1, wnd);

    /* Now m2 ep should have FC grant message in the pending queue.
     * Add some user pending requests as well */
    for (int i = 0; i < num_pend_sends; i++) {
        reqs[i].uct.func    = NULL; /* make valgrind happy */
        reqs[i].purge_count = 0;
        EXPECT_EQ(uct_ep_pending_add(m_e2->ep(0), &reqs[i].uct), UCS_OK);
    }
    uct_ep_pending_purge(m_e2->ep(0), purge_cb, NULL);

    for (int i = 0; i < num_pend_sends; i++) {
        EXPECT_EQ(1, reqs[i].purge_count);
    }
}


/* Check that FC window works as expected */
UCS_TEST_P(test_rc_flow_control, general_enabled)
{
    test_general(8, 4, 2, true);
}

UCS_TEST_P(test_rc_flow_control, general_disabled)
{
    test_general(8, 4, 2, false);
}

/* Test the scenario when ep is being destroyed while there is
 * FC grant message in the pending queue */
UCS_TEST_P(test_rc_flow_control, pending_only_fc)
{
    int wnd = 2;

    disable_entity(m_e2);
    set_fc_attributes(m_e1, true, wnd, wnd, 1);

    send_am_and_flush(m_e1, wnd);

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

UCS_TEST_P(test_rc_flow_control, fc_disabled_flush)
{
    test_flush_fc_disabled();
}

UCT_RC_INSTANTIATE_TEST_CASE(test_rc_flow_control)


#if ENABLE_STATS

void test_rc_flow_control_stats::test_general(int wnd, int soft_thresh,
                                              int hard_thresh)
{
    uint64_t v;

    set_fc_attributes(m_e1, true, wnd, soft_thresh, hard_thresh);

    send_am_messages(m_e1, wnd, UCS_OK);
    send_am_messages(m_e1, 1, UCS_ERR_NO_RESOURCE);

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_NO_CRED);
    EXPECT_EQ(1ul, v);

    validate_grant(m_e1);
    send_am_messages(m_e1, 1, UCS_OK);

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_TX_HARD_REQ);
    EXPECT_EQ(1ul, v);

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_RX_PURE_GRANT);
    EXPECT_EQ(1ul, v);
    flush();
}


UCS_TEST_P(test_rc_flow_control_stats, general)
{
    test_general(5, 2, 1);
}

UCS_TEST_P(test_rc_flow_control_stats, soft_request)
{
    uint64_t v;
    int wnd = 8;
    int s_thresh = 4;
    int h_thresh = 1;

    set_fc_attributes(m_e1, true, wnd, s_thresh, h_thresh);
    send_am_and_flush(m_e1, wnd - (s_thresh - 1));

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_TX_SOFT_REQ);
    EXPECT_EQ(1ul, v);
    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e2)->stats, UCT_RC_FC_STAT_RX_SOFT_REQ);
    EXPECT_EQ(1ul, v);

    send_am_and_flush(m_e2, wnd - (s_thresh - 1));
    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_RX_GRANT);
    EXPECT_EQ(1ul, v);
    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e2)->stats, UCT_RC_FC_STAT_TX_GRANT);
    EXPECT_EQ(1ul, v);
}

UCT_RC_INSTANTIATE_TEST_CASE(test_rc_flow_control_stats)

#endif
