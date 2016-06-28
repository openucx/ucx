/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.All rights reserved.
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
#include <uct/ib/rc/base/rc_ep.h>
#include <uct/ib/rc/base/rc_iface.h>
}
#include <common/test.h>
#include "uct_test.h"

#define UCT_RC_INSTANTIATE_TEST_CASE(_test_case) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, rc) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, rc_mlx5)


class test_rc : public uct_test {
public:
    virtual void init() {
        uct_test::init();

        m_e1 = uct_test::create_entity(0);
        m_entities.push_back(m_e1);

        m_e2 = uct_test::create_entity(0);
        m_entities.push_back(m_e2);

        connect();
    }

    void connect() {
        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);

        uct_iface_set_am_handler(m_e1->iface(), 0, am_dummy_handler,
                                 NULL, UCT_AM_CB_FLAG_SYNC);
        uct_iface_set_am_handler(m_e2->iface(), 0, am_dummy_handler,
                                 NULL, UCT_AM_CB_FLAG_SYNC);
    }

    uct_rc_iface_t* rc_iface(entity *e) {
        return ucs_derived_of(e->iface(), uct_rc_iface_t);
    }

    uct_rc_ep_t* rc_ep(entity *e) {
        return ucs_derived_of(e->ep(0), uct_rc_ep_t);
    }

    void send_am_messages(entity *e, int wnd, ucs_status_t expected) {
        for (int i = 0; i < wnd; i++) {
            EXPECT_EQ(expected, uct_ep_am_short(e->ep(0), 0, 0, NULL, 0));
        }
    }

    void progress_loop(double delta_ms=10.0) {
        uct_test::short_progress_loop(delta_ms);
    }

    static ucs_status_t am_dummy_handler(void *arg, void *data, size_t length, void *desc) {
        return UCS_OK;
    }

protected:
    entity *m_e1, *m_e2;
};


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


class test_rc_flow_control : public test_rc {
public:
    typedef struct pending_send_request {
        uct_ep_h          ep;
        uct_pending_req_t uct;
    } pending_send_request_t;

    void set_fc_attributes(entity *e, bool enabled, int wnd, int s_thresh, int h_thresh) {
        rc_iface(e)->config.fc_enabled     = enabled;
        rc_iface(e)->config.fc_wnd_size    = rc_ep(e)->fc.fc_wnd = wnd;
        rc_iface(e)->config.fc_soft_thresh = s_thresh;
        rc_iface(e)->config.fc_hard_thresh = h_thresh;

    }

    void progress_loop(double delta_ms=10.0) {
        uct_test::short_progress_loop(delta_ms);
    }

    static ucs_status_t am_dummy_handler(void *arg, void *data, size_t length, void *desc) {
        return UCS_OK;
    }

    static ucs_status_t am_send(uct_pending_req_t *self) {

        pending_send_request_t *req = ucs_container_of(self, pending_send_request_t, uct);
        ucs_status_t status;

        status = uct_ep_am_short(req->ep, 0, 0, NULL,0);
        ucs_debug("sending short with grant %d ",status);
        return status;
    }

    static void purge_cb(uct_pending_req_t *self, void *arg)
    {
        req_count++;
    }

protected:
    static int req_count;
};

int test_rc_flow_control::req_count = 0;

/* Check that FC window works as expected */
UCS_TEST_P(test_rc_flow_control, general)
{
    int test_wnd = 8;

    set_fc_attributes(m_e1, true, test_wnd,
                      ucs_max((int)(test_wnd*0.5), 1),
                      ucs_max((int)(test_wnd*0.25), 1));

    send_am_messages(m_e1, test_wnd, UCS_OK);
    send_am_messages(m_e1, 1, UCS_ERR_NO_RESOURCE);

    progress_loop();
    send_am_messages(m_e1, 1, UCS_OK);
}

/* Check that FC does not stop us when disabled */
UCS_TEST_P(test_rc_flow_control, disabled)
{
    int test_wnd = 8;

    set_fc_attributes(m_e1, false, test_wnd,
                      ucs_max((int)(test_wnd*0.5), 1),
                      ucs_max((int)(test_wnd*0.25), 1));

    send_am_messages(m_e1, test_wnd, UCS_OK);
    send_am_messages(m_e1, 1, UCS_OK);

    progress_loop();
    send_am_messages(m_e1, 1, UCS_OK);
}

/* Test the scenario when ep is being destroyed while there is
 * FC grant message in the pending queue */
UCS_TEST_P(test_rc_flow_control, pending_only_fc)
{
    int test_wnd = 2;

    /* Set tx resources of m2 to 0 for FC grant message
     * to be added to the pending group */
    rc_ep(m_e2)->txqp.available = 0;

    set_fc_attributes(m_e1, true, test_wnd, test_wnd, 1);

    send_am_messages(m_e1, test_wnd, UCS_OK);
    progress_loop();

    m_e2->destroy_ep(0);
    ASSERT_TRUE(rc_iface(m_e2)->tx.arbiter.current == NULL);
}

/* Check that user callback passed to uct_ep_pending_purge is not
 * invoked for FC grant message */
UCS_TEST_P(test_rc_flow_control, pending_purge)
{
    int test_wnd = 2;
    int num_pend_sends = 5;
    uct_pending_req_t reqs[num_pend_sends];

    /* Set tx resources of m2 to 0 for FC grant message
     * to be added to the pending group*/
    rc_ep(m_e2)->txqp.available = 0;

    set_fc_attributes(m_e1, true, test_wnd, test_wnd, 1);

    req_count = 0;

    send_am_messages(m_e1, test_wnd, UCS_OK);
    progress_loop();

    /* Now m2 ep should have FC grant message in the pending queue.
     * Add some user pending requests as well */
    for (int i = 0; i < num_pend_sends; i ++) {
        EXPECT_EQ(uct_ep_pending_add(m_e2->ep(0), &reqs[i]), UCS_OK);
    }

    uct_ep_pending_purge(m_e2->ep(0), purge_cb, NULL);
    EXPECT_EQ(num_pend_sends, req_count);
}

/* Check that FC grant message is not added to the pending queue
 * if it is already present there. Potentially it may happen if
 * endpoint answered on soft request with some AM message, but
 * grant message was added to pending due to lack of resources. */
UCS_TEST_P(test_rc_flow_control, pending_fc_req)
{
    int test_wnd = 4;
    int h_thresh = 1;
    int num_pend = 2;
    int available = rc_ep(m_e2)->txqp.available;
    pending_send_request_t req[num_pend];

    req_count = 0;

    /* Disable send capabilities of m_e2 */
    rc_iface(m_e2)->tx.cq_available = 0;

    set_fc_attributes(m_e1, true, test_wnd, test_wnd, h_thresh);

    for (int i = 0; i < num_pend; i++) {
        req[i].uct.func = am_send;
        req[i].ep = m_e2->ep(0);
        EXPECT_EQ(uct_ep_pending_add(m_e2->ep(0), &req[i].uct), UCS_OK);
        rc_ep(m_e1)->fc.fc_wnd = h_thresh;
        send_am_messages(m_e1, 1, UCS_OK); /* send AM with FC hard request */
        progress_loop();
    }

    /* Now pending group of m_e2 endpoint should look like:
     * AM-0->FC_grant->AM-1
     * The second FC grant should not be added to the group! */

    /* Now force dispatching of m_e2 pending group */
    rc_iface(m_e2)->tx.cq_available = 2;

    /* Make sure this AM message is signalled to force pendings dispatch */
    rc_iface(m_e2)->config.tx_moderation = 0;
    send_am_messages(m_e2, 1, UCS_OK);

    /* Avoid modifying of ep->available by send completions */
    rc_iface(m_e2)->config.tx_moderation = 10;
    progress_loop();

    /* Check that only num_pend + 1 (for FC grant) messages were sent */
    EXPECT_EQ(rc_ep(m_e2)->txqp.available, available - (num_pend + 1));
}

UCT_RC_INSTANTIATE_TEST_CASE(test_rc_flow_control)

/* Check that FC window works as expected */
#if ENABLE_STATS
class test_rc_flow_control_stats : public test_rc_flow_control {
public:
    void init() {
        stats_activate();
        test_rc_flow_control::init();
    }

    void cleanup() {
        test_rc_flow_control::cleanup();
        stats_restore();
    }
};

UCS_TEST_P(test_rc_flow_control_stats, general)
{
    int test_wnd = 8;
    uint64_t v;

    set_fc_attributes(m_e1, true, test_wnd,
                      ucs_max((int)(test_wnd*0.5), 1),
                      ucs_max((int)(test_wnd*0.25), 1));

    send_am_messages(m_e1, test_wnd, UCS_OK);
    send_am_messages(m_e1, 1, UCS_ERR_NO_RESOURCE);

    v = UCS_STATS_GET_COUNTER(rc_ep(m_e1)->fc.stats, UCT_RC_FC_STAT_NO_CRED);
    EXPECT_EQ(1ul, v);

    progress_loop();
    send_am_messages(m_e1, 1, UCS_OK);

    v = UCS_STATS_GET_COUNTER(rc_ep(m_e1)->fc.stats, UCT_RC_FC_STAT_TX_HARD_REQ);
    EXPECT_EQ(1ul, v);

    v = UCS_STATS_GET_COUNTER(rc_ep(m_e1)->fc.stats, UCT_RC_FC_STAT_RX_GRANT);
    EXPECT_EQ(1ul, v);
}

UCT_RC_INSTANTIATE_TEST_CASE(test_rc_flow_control_stats)
#endif
