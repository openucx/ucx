/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.All rights reserved.
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
#include <uct/ib/dc/base/dc_iface.h>
#include <uct/ib/dc/base/dc_ep.h>
}
#include <common/test.h>
#include "uct_test.h"
#include "test_rc.h"

class test_dc : public uct_test {
public:
    virtual void init() {
        uct_test::init();

        m_e1 = uct_test::create_entity(0);
        m_entities.push_back(m_e1);

        m_e2 = uct_test::create_entity(0);
        m_entities.push_back(m_e2);

        uct_iface_set_am_handler(m_e1->iface(), 0, am_dummy_handler,
                                 NULL, UCT_AM_CB_FLAG_SYNC);
        uct_iface_set_am_handler(m_e2->iface(), 0, am_dummy_handler,
                                 NULL, UCT_AM_CB_FLAG_SYNC);
    }

    static uct_dc_iface_t* dc_iface(entity *e) {
        return ucs_derived_of(e->iface(), uct_dc_iface_t);
    }

    static uct_dc_ep_t* dc_ep(entity *e, int idx) {
        return ucs_derived_of(e->ep(idx), uct_dc_ep_t);
    }

    static ucs_status_t am_dummy_handler(void *arg, void *data, size_t length, void *desc) {
        return UCS_OK;
    }

    virtual void cleanup() {
        uct_test::cleanup();
    }

    static int n_warnings;

    static ucs_log_func_rc_t
    log_ep_destroy(const char *file, unsigned line, const char *function,
                   ucs_log_level_t level, const char *prefix, const char *message,
                   va_list ap)
    {
        if (level != UCS_LOG_LEVEL_WARN) {
            /* debug messages are ignored */
            return UCS_LOG_FUNC_RC_CONTINUE;
        }
        if (strcmp("ep (%p) is destroyed with %d outstanding ops", message) == 0) {
            n_warnings++;
        }
        return UCS_LOG_FUNC_RC_STOP;
    }

protected:
    entity *m_e1, *m_e2;

    struct dcs_comp {
        uct_completion_t uct_comp;
        entity *e;
    } comp;

    static void uct_comp_cb(uct_completion_t *uct_comp, ucs_status_t status)
    {
        struct dcs_comp *comp = (struct dcs_comp *)uct_comp;
        uct_dc_ep_t *ep;

        ASSERT_UCS_OK(status);

        ep = dc_ep(comp->e, 0);
        /* dci must be released before completion cb is called */
        EXPECT_EQ(UCT_DC_EP_NO_DCI, ep->dci);
        comp->e->destroy_eps();
    }

    struct dcs_pending {
        uct_pending_req_t uct_req;
        entity *e;
        int is_done;
    } preq;

    static ucs_status_t uct_pending_flush(uct_pending_req_t *uct_req) 
    {
        struct dcs_pending *preq = (struct dcs_pending *)uct_req;
        ucs_status_t status;
        uct_dc_ep_t *ep;

        ep = dc_ep(preq->e, 0);
        EXPECT_NE(UCT_DC_EP_NO_DCI, ep->dci);

        status = uct_ep_flush(preq->e->ep(0), 0, NULL);
        if (status == UCS_OK) {
            preq->is_done = 1;
        }
        return status;
    }

    static ucs_status_t uct_pending_dummy(uct_pending_req_t *uct_req) 
    {
        struct dcs_pending *preq = (struct dcs_pending *)uct_req;
        uct_dc_ep_t *ep;
        uct_dc_iface_t *iface;

        ep    = dc_ep(preq->e, 0);
        iface = dc_iface(preq->e);

        EXPECT_NE(UCT_DC_EP_NO_DCI, ep->dci);

        /* simulate arbiter stop because lack of global resorce
         * operation still stands on pending
         */
        preq->is_done = 1;
        iface->super.tx.cq_available = 0;
        return UCS_INPROGRESS;
    }

    static void purge_cb(uct_pending_req_t *uct_req, void *arg)
    {
        struct dcs_pending *preq = (struct dcs_pending *)uct_req;
        uct_dc_ep_t *ep;
        uct_dc_iface_t *iface;

        ep    = dc_ep(preq->e, 0);
        iface = dc_iface(preq->e);
        EXPECT_NE(UCT_DC_EP_NO_DCI, ep->dci);
        iface->super.tx.cq_available = 8;
    }

};

int test_dc::n_warnings = 0;

UCS_TEST_P(test_dc, dcs_single) {
    ucs_status_t status;
    uct_dc_ep_t *ep;
    uct_dc_iface_t *iface;

    m_e1->connect_to_iface(0, *m_e2);
    ep = dc_ep(m_e1, 0);
    iface = dc_iface(m_e1);
    EXPECT_EQ(UCT_DC_EP_NO_DCI, ep->dci);
    status = uct_ep_am_short(m_e1->ep(0), 0, 0, NULL, 0);
    EXPECT_UCS_OK(status);
    /* dci 0 must be assigned to the ep */
    EXPECT_EQ(iface->tx.dcis_stack[0], ep->dci);
    EXPECT_EQ(1, iface->tx.stack_top);
    EXPECT_EQ(ep, iface->tx.dcis[ep->dci].ep);

    flush();

    /* after the flush dci must be released */
    EXPECT_EQ(UCT_DC_EP_NO_DCI, ep->dci);
    EXPECT_EQ(0, iface->tx.stack_top);
    EXPECT_EQ(0, iface->tx.dcis_stack[0]);
}

UCS_TEST_P(test_dc, dcs_multi) {
    ucs_status_t status;
    uct_dc_ep_t *ep;
    uct_dc_iface_t *iface;
    unsigned i;

    iface = dc_iface(m_e1);
    for (i = 0; i <= iface->tx.ndci; i++) {
        m_e1->connect_to_iface(i, *m_e2);
    }

    for (i = 0; i < iface->tx.ndci; i++) {
        ep = dc_ep(m_e1, i);
        EXPECT_EQ(UCT_DC_EP_NO_DCI, ep->dci);
        status = uct_ep_am_short(m_e1->ep(i), 0, 0, NULL, 0);
        EXPECT_UCS_OK(status);

        /* dci on free LIFO must be assigned to the ep */
        EXPECT_EQ(iface->tx.dcis_stack[i], ep->dci);
        EXPECT_EQ(i+1, iface->tx.stack_top);
        EXPECT_EQ(ep, iface->tx.dcis[ep->dci].ep);
    }

    /* this should fail because there are no free dci */
    status = uct_ep_am_short(m_e1->ep(i), 0, 0, NULL, 0);
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);

    flush();

    /* after the flush dci must be released */

    EXPECT_EQ(0, iface->tx.stack_top);
    for (i = 0; i < iface->tx.ndci; i++) {
        ep = dc_ep(m_e1, i);
        EXPECT_EQ(UCT_DC_EP_NO_DCI, ep->dci);
    }
}

/** 
 * send message, destroy ep while it is still holding dci.
 * This is against the rules so there should be a warning. 
 * Do not crash.
 */ 
UCS_TEST_P(test_dc, dcs_ep_destroy) {

    ucs_status_t status;
    uct_dc_ep_t *ep;
    uct_dc_iface_t *iface;


    ucs_log_push_handler(log_ep_destroy);
    UCS_TEST_SCOPE_EXIT() { ucs_log_pop_handler(); } UCS_TEST_SCOPE_EXIT_END

    m_e1->connect_to_iface(0, *m_e2);
    ep = dc_ep(m_e1, 0);
    iface = dc_iface(m_e1);
    n_warnings = 0;
    EXPECT_EQ(UCT_DC_EP_NO_DCI, ep->dci);
    status = uct_ep_am_short(m_e1->ep(0), 0, 0, NULL, 0);
    EXPECT_UCS_OK(status);
    /* dci 0 must be assigned to the ep */
    EXPECT_EQ(iface->tx.dcis_stack[0], ep->dci);
    EXPECT_EQ(1, iface->tx.stack_top);
    EXPECT_EQ(ep, iface->tx.dcis[ep->dci].ep);

    m_e1->destroy_eps();
    EXPECT_EQ(1, iface->tx.stack_top);
    EXPECT_EQ(1, n_warnings);

    flush();
    EXPECT_EQ(0, iface->tx.stack_top);
}

/**
 * destroy ep from the flush completion. It may not work in general but
 * it must work with dc ep
 */
UCS_TEST_P(test_dc, dcs_ep_flush_destroy) {

    ucs_status_t status;
    uct_dc_ep_t *ep;
    uct_dc_iface_t *iface;

    m_e1->connect_to_iface(0, *m_e2);
    ep = dc_ep(m_e1, 0);
    iface = dc_iface(m_e1);
    EXPECT_EQ(UCT_DC_EP_NO_DCI, ep->dci);
    status = uct_ep_am_short(m_e1->ep(0), 0, 0, NULL, 0);
    EXPECT_UCS_OK(status);

    EXPECT_EQ(iface->tx.dcis_stack[0], ep->dci);
    EXPECT_EQ(1, iface->tx.stack_top);
    EXPECT_EQ(ep, iface->tx.dcis[ep->dci].ep);

    comp.uct_comp.count = 1;
    comp.uct_comp.func  = uct_comp_cb;
    comp.e              = m_e1;

    status = uct_ep_flush(m_e1->ep(0), 0, &comp.uct_comp);
    do {
        progress();
    } while (comp.uct_comp.count > 0);

    EXPECT_EQ(0, iface->tx.stack_top);
}

/* Check that flushing ep from pending releases dci */
UCS_TEST_P(test_dc, dcs_ep_flush_pending) {

    ucs_status_t status;
    uct_dc_iface_t *iface;

    m_e1->connect_to_iface(0, *m_e2);
    m_e1->connect_to_iface(1, *m_e2);

    /* use all iface resources */
    iface = dc_iface(m_e1);
    iface->super.tx.cq_available = 8;
    do {
        status = uct_ep_am_short(m_e1->ep(1), 0, 0, NULL, 0);
    } while (status == UCS_OK);

    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);

    /* flush another ep. Flush fails because there are no cqes */
    status = uct_ep_flush(m_e1->ep(0), 0, NULL);
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);

    /* put flush op on pending */
    preq.is_done = 0;
    preq.e = m_e1;
    preq.uct_req.func = uct_pending_flush;
    status = uct_ep_pending_add(m_e1->ep(0), &preq.uct_req);
    EXPECT_UCS_OK(status);
    
    /* progress till ep is flushed */
    do {
        progress();
    } while (!preq.is_done);

    /* flush the other active ep */
    flush();

    /* check that ep does not hold dci */
    EXPECT_EQ(0, iface->tx.stack_top);
}

/* check that ep does not hold dci after
 * purge
 */
UCS_TEST_P(test_dc, dcs_ep_purge_pending) {

    ucs_status_t status;
    uct_dc_iface_t *iface;
    uct_dc_ep_t *ep;

    m_e1->connect_to_iface(0, *m_e2);
    m_e1->connect_to_iface(1, *m_e2);

    /* use all iface resources */
    iface = dc_iface(m_e1);
    ep = dc_ep(m_e1, 0);
    iface->super.tx.cq_available = 8;

    do {
        status = uct_ep_am_short(m_e1->ep(1), 0, 0, NULL, 0);
    } while (status == UCS_OK);

    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);

    /* flush another ep. Flush fails because there are no cqes */
    status = uct_ep_flush(m_e1->ep(0), 0, NULL);
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);

    /* put flush op on pending */
    preq.is_done = 0;
    preq.e = m_e1;
    preq.uct_req.func = uct_pending_dummy;
    status = uct_ep_pending_add(m_e1->ep(0), &preq.uct_req);
    EXPECT_UCS_OK(status);

    do {
        progress();
    } while (!preq.is_done);

    EXPECT_LE(1, iface->tx.stack_top);
    uct_ep_pending_purge(m_e1->ep(0), purge_cb, NULL);
    EXPECT_EQ(UCT_DC_EP_NO_DCI, ep->dci);
    flush();
    EXPECT_EQ(0, iface->tx.stack_top);
}

_UCT_INSTANTIATE_TEST_CASE(test_dc, dc)
_UCT_INSTANTIATE_TEST_CASE(test_dc, dc_mlx5)


class test_dc_flow_control : public test_rc_flow_control {

public:

    void init() {
        if (UCS_OK != uct_config_modify(m_iface_config, "RC_FC_ENABLE", "y")) {
            UCS_TEST_ABORT("Error: cannot enable flow control");
        }
        test_rc_flow_control::init();
    }

    /* virtual */
    uct_rc_fc_t* get_fc_ptr(entity *e) {
        return &ucs_derived_of(e->ep(0), uct_dc_ep_t)->fc;
    }
};

UCS_TEST_P(test_dc_flow_control, general_enabled)
{
    test_general(8, true);
}

UCS_TEST_P(test_dc_flow_control, general_disabled)
{
    test_general(8, false);
}

/* Check that user callback passed to uct_ep_pending_purge is not
 * invoked for FC grant message */
UCS_TEST_P(test_dc_flow_control, pending_purge)
{
    test_pending_purge(2, 5);
}

UCS_TEST_P(test_dc_flow_control, pending_grant)
{
    test_pending_grant(5);
}

/* Check that soft request is not handled by DC */
UCS_TEST_P(test_dc_flow_control, soft_request)
{
    int wnd = 8;
    int s_thresh = 4;
    int h_thresh = 1;

    set_fc_attributes(m_e1, true, wnd, s_thresh, h_thresh);

    send_am_messages(m_e1, wnd - (s_thresh - 1), UCS_OK);
    progress_loop();

    set_tx_moderation(m_e2, 1);
    send_am_messages(m_e2, 1, UCS_OK);
    progress_loop();

    /* Check that window is not updated */
    EXPECT_EQ(get_fc_ptr(m_e1)->fc_wnd, s_thresh - 1);
}

/* Check that flush returns UCS_INPROGRESS if there is
 * an outgoing grant request */
UCS_TEST_P(test_dc_flow_control, flush)
{
    int wnd = 5;

    disable_entity(m_e2);

    set_fc_attributes(m_e1, true, wnd,
                      ucs_max((int)(wnd*0.5), 1),
                      ucs_max((int)(wnd*0.25), 1));

    send_am_messages(m_e1, wnd, UCS_OK);
    progress_loop();

    EXPECT_EQ(uct_ep_flush(m_e1->ep(0), 0, NULL), UCS_INPROGRESS);

    /* Enable send capabilities of m_e2 and send AM message
     * to force pending queue dispatch */
    enable_entity(m_e2);
    set_tx_moderation(m_e2, 1);
    send_am_messages(m_e2, 1, UCS_OK);
    progress_loop();

    /* Grant should be received by now. Flush should return ok */
    EXPECT_UCS_OK(uct_ep_flush(m_e1->ep(0), 0, NULL));
}

_UCT_INSTANTIATE_TEST_CASE(test_dc_flow_control, dc)


#if ENABLE_STATS

class test_dc_flow_control_stats : public test_rc_flow_control_stats {
public:
    /* virtual */
    void init() {
        if (UCS_OK != uct_config_modify(m_iface_config, "RC_FC_ENABLE", "y")) {
            UCS_TEST_ABORT("Error: cannot enable flow control");
        }
        test_rc_flow_control_stats::init();
    }

    uct_rc_fc_t* get_fc_ptr(entity *e) {
        return &ucs_derived_of(e->ep(0), uct_dc_ep_t)->fc;
    }

    uct_rc_fc_t* fake_ep_fc_ptr(entity *e) {
        return &ucs_derived_of(e->iface(), uct_dc_iface_t)->tx.fake_fc_ep->fc;
    }
};

UCS_TEST_P(test_dc_flow_control_stats, general)
{
    test_general(5);
}

UCS_TEST_P(test_dc_flow_control_stats, fake_fc_ep)
{
    uint64_t v;
    int wnd = 5;

    set_fc_attributes(m_e1, true, wnd,
                      ucs_max((int)(wnd*0.5), 1),
                      ucs_max((int)(wnd*0.25), 1));

    send_am_messages(m_e1, wnd, UCS_OK);

    progress_loop();
    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_TX_HARD_REQ);
    EXPECT_EQ(1ul, v);
    v = UCS_STATS_GET_COUNTER(fake_ep_fc_ptr(m_e2)->stats, UCT_RC_FC_STAT_RX_HARD_REQ);
    EXPECT_EQ(1ul, v);

    progress_loop();

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_RX_PURE_GRANT);
    EXPECT_EQ(1ul, v);
    v = UCS_STATS_GET_COUNTER(fake_ep_fc_ptr(m_e2)->stats, UCT_RC_FC_STAT_TX_PURE_GRANT);
    EXPECT_EQ(1ul, v);
}

_UCT_INSTANTIATE_TEST_CASE(test_dc_flow_control_stats, dc)

#endif

