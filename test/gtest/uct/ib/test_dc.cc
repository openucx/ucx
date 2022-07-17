/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.All rights reserved.
* See file LICENSE for terms.
*/

#include "test_rc.h"

#include <common/test.h>
#include <uct/uct_test.h>

extern "C" {
#include <uct/api/uct.h>
#include <uct/ib/dc/dc_mlx5.h>
#include <uct/ib/dc/dc_mlx5_ep.h>
}


#define UCT_DC_INSTANTIATE_TEST_CASE(_test_case) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, dc_mlx5)


class test_dc : public test_rc {
public:
    virtual void init() {
        uct_test::init();

        m_e1 = uct_test::create_entity(0);
        m_entities.push_back(m_e1);

        m_e2 = uct_test::create_entity(0);
        m_entities.push_back(m_e2);

        uct_iface_set_am_handler(m_e1->iface(), 0, am_dummy_handler, NULL, 0);
        uct_iface_set_am_handler(m_e2->iface(), 0, am_dummy_handler, NULL, 0);
    }

    entity* create_rand_entity() {
        if (UCS_OK != uct_config_modify(m_iface_config, "DC_TX_POLICY", "rand")) {
            UCS_TEST_ABORT("Error: cannot enable random DCI policy");
        }
        entity *rand_e = uct_test::create_entity(0);
        m_entities.push_back(rand_e);
        return rand_e;
    }

    static uct_dc_mlx5_iface_t* dc_iface(entity *e) {
        return ucs_derived_of(e->iface(), uct_dc_mlx5_iface_t);
    }

    static uct_dc_mlx5_ep_t* dc_ep(entity *e, int idx) {
        return ucs_derived_of(e->ep(idx), uct_dc_mlx5_ep_t);
    }

    virtual void cleanup() {
        uct_test::cleanup();
    }

    static int n_warnings;

    static ucs_log_func_rc_t
    log_ep_destroy(const char *file, unsigned line, const char *function,
                   ucs_log_level_t level,
                   const ucs_log_component_config_t *comp_conf,
                   const char *message, va_list ap)
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
    static uint32_t m_am_rx_count;
    static int      m_purge_count;

    struct dcs_comp {
        uct_completion_t uct_comp;
        entity *e;
    } comp;

    static void uct_comp_cb(uct_completion_t *uct_comp)
    {
        struct dcs_comp *comp = (struct dcs_comp *)uct_comp;

        EXPECT_UCS_OK(uct_comp->status);

        comp->e->destroy_eps();
    }

    static ucs_status_t am_handler(void *arg, void *data, size_t length,
                                   unsigned flags)
    {
        ++m_am_rx_count;
        return UCS_OK;
    }

    static ucs_status_t am_dummy_handler(void *arg, void *data, size_t length,
                                         unsigned flags) {
        return UCS_OK;
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
        uct_dc_mlx5_ep_t *ep;

        ep = dc_ep(preq->e, 0);
        EXPECT_NE(UCT_DC_MLX5_EP_NO_DCI, ep->dci);

        status = uct_ep_flush(preq->e->ep(0), 0, NULL);
        if (status == UCS_OK) {
            preq->is_done = 1;
        }
        return status;
    }

    static ucs_status_t uct_pending_dummy(uct_pending_req_t *uct_req)
    {
        struct dcs_pending *preq = (struct dcs_pending *)uct_req;
        uct_dc_mlx5_ep_t *ep;

        ep = dc_ep(preq->e, 0);

        EXPECT_NE(UCT_DC_MLX5_EP_NO_DCI, ep->dci);

        /* simulate arbiter stop because lack of global resource
         * operation still stands on pending
         */
        preq->is_done = 1;
        return UCS_INPROGRESS;
    }

    static void purge_cb(uct_pending_req_t *uct_req, void *arg)
    {
        struct dcs_pending *preq = (struct dcs_pending *)uct_req;

        EXPECT_NE(UCT_DC_MLX5_EP_NO_DCI, dc_ep(preq->e, 0)->dci);
    }

    static void purge_count_cb(uct_pending_req_t *uct_req, void *arg)
    {
        ++m_purge_count;
    }

};

int test_dc::n_warnings         = 0;
int test_dc::m_purge_count      = 0;
uint32_t test_dc::m_am_rx_count = 0;

UCS_TEST_P(test_dc, dcs_single) {
    ucs_status_t status;
    uct_dc_mlx5_ep_t *ep;
    uct_dc_mlx5_iface_t *iface;

    m_e1->connect_to_iface(0, *m_e2);
    ep = dc_ep(m_e1, 0);
    iface = dc_iface(m_e1);
    EXPECT_EQ(UCT_DC_MLX5_EP_NO_DCI, ep->dci);
    status = uct_ep_am_short(m_e1->ep(0), 0, 0, NULL, 0);
    EXPECT_UCS_OK(status);
    /* dci 0 must be assigned to the ep */
    EXPECT_EQ(iface->tx.dci_pool[0].stack[0], ep->dci);
    EXPECT_EQ(1, iface->tx.dci_pool[0].stack_top);
    EXPECT_EQ(ep, iface->tx.dcis[ep->dci].ep);

    flush();

    /* after the flush dci must be released */
    EXPECT_EQ(UCT_DC_MLX5_EP_NO_DCI, ep->dci);
    EXPECT_EQ(0, iface->tx.dci_pool[0].stack_top);
    EXPECT_EQ(0, iface->tx.dci_pool[0].stack[0]);
}

UCS_TEST_P(test_dc, dcs_multi) {
    ucs_status_t status;
    uct_dc_mlx5_ep_t *ep;
    uct_dc_mlx5_iface_t *iface;
    unsigned i;

    iface = dc_iface(m_e1);
    for (i = 0; i <= iface->tx.ndci; i++) {
        m_e1->connect_to_iface(i, *m_e2);
    }

    for (i = 0; i < iface->tx.ndci; i++) {
        ep = dc_ep(m_e1, i);
        EXPECT_EQ(UCT_DC_MLX5_EP_NO_DCI, ep->dci);
        status = uct_ep_am_short(m_e1->ep(i), 0, 0, NULL, 0);
        EXPECT_UCS_OK(status);

        /* dci on free LIFO must be assigned to the ep */
        EXPECT_EQ(iface->tx.dci_pool[0].stack[i], ep->dci);
        EXPECT_EQ(i+1, iface->tx.dci_pool[0].stack_top);
        EXPECT_EQ(ep, iface->tx.dcis[ep->dci].ep);
    }

    /* this should fail because there are no free dci */
    status = uct_ep_am_short(m_e1->ep(i), 0, 0, NULL, 0);
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);

    flush();

    /* after the flush dci must be released */

    EXPECT_EQ(0, iface->tx.dci_pool[0].stack_top);
    for (i = 0; i < iface->tx.ndci; i++) {
        ep = dc_ep(m_e1, i);
        EXPECT_EQ(UCT_DC_MLX5_EP_NO_DCI, ep->dci);
    }
}

/**
 * send message, destroy ep while it is still holding dci.
 * Do not crash.
 */
UCS_TEST_P(test_dc, dcs_ep_destroy) {

    uct_dc_mlx5_ep_t *ep;
    uct_dc_mlx5_iface_t *iface;

    ucs_log_push_handler(log_ep_destroy);
    UCS_TEST_SCOPE_EXIT() { ucs_log_pop_handler(); } UCS_TEST_SCOPE_EXIT_END

    m_e1->connect_to_iface(0, *m_e2);
    ep = dc_ep(m_e1, 0);
    iface = dc_iface(m_e1);
    n_warnings = 0;
    EXPECT_EQ(UCT_DC_MLX5_EP_NO_DCI, ep->dci);
    send_am_messages(m_e1, 2, UCS_OK);
    /* dci 0 must be assigned to the ep */
    EXPECT_EQ(iface->tx.dci_pool[0].stack[0], ep->dci);
    EXPECT_EQ(1, iface->tx.dci_pool[0].stack_top);
    EXPECT_EQ(ep, iface->tx.dcis[ep->dci].ep);

    m_e1->destroy_eps();
    EXPECT_EQ(1, iface->tx.dci_pool[0].stack_top);

    flush();
    EXPECT_EQ(0, iface->tx.dci_pool[0].stack_top);
}

/**
 * destroy ep from the flush completion. It may not work in general but
 * it must work with dc ep
 */
UCS_TEST_P(test_dc, dcs_ep_flush_destroy) {

    ucs_status_t status;
    uct_dc_mlx5_ep_t *ep;
    uct_dc_mlx5_iface_t *iface;

    m_e1->connect_to_iface(0, *m_e2);
    ep = dc_ep(m_e1, 0);
    iface = dc_iface(m_e1);
    EXPECT_EQ(UCT_DC_MLX5_EP_NO_DCI, ep->dci);
    status = uct_ep_am_short(m_e1->ep(0), 0, 0, NULL, 0);
    EXPECT_UCS_OK(status);

    EXPECT_EQ(iface->tx.dci_pool[0].stack[0], ep->dci);
    EXPECT_EQ(1, iface->tx.dci_pool[0].stack_top);
    EXPECT_EQ(ep, iface->tx.dcis[ep->dci].ep);

    comp.uct_comp.count  = 1;
    comp.uct_comp.func   = uct_comp_cb;
    comp.uct_comp.status = UCS_OK;
    comp.e               = m_e1;

    status = uct_ep_flush(m_e1->ep(0), 0, &comp.uct_comp);
    do {
        progress();
    } while (comp.uct_comp.count > 0);

    EXPECT_EQ(0, iface->tx.dci_pool[0].stack_top);
}

UCS_TEST_P(test_dc, dcs_ep_flush_pending, "DC_NUM_DCI=1") {
    ucs_status_t status;
    uct_dc_mlx5_iface_t *iface;

    m_e1->connect_to_iface(0, *m_e2);
    m_e1->connect_to_iface(1, *m_e2);

    iface = dc_iface(m_e1);

    /* shorten test time by reducing dci QP resources */
    iface->tx.dcis[0].txqp.available = 8;
    do {
        status = uct_ep_am_short(m_e1->ep(1), 0, 0, NULL, 0);
    } while (status == UCS_OK);

    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);

    /* flush another ep. Flush fails because there is no free dci */
    status = uct_ep_flush(m_e1->ep(0), 0, NULL);
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);

    /* put flush op on pending */
    preq.is_done = 0;
    preq.e = m_e1;
    preq.uct_req.func = uct_pending_flush;
    status = uct_ep_pending_add(m_e1->ep(0), &preq.uct_req, 0);
    EXPECT_UCS_OK(status);

    status = uct_ep_am_short(m_e1->ep(0), 0, 0, NULL, 0);
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);

    /* progress till ep is flushed */
    do {
        progress();
    } while (!preq.is_done);

    /* flush the other active ep */
    flush();

    status = uct_ep_am_short(m_e1->ep(0), 0, 0, NULL, 0);
    EXPECT_EQ(UCS_OK, status);
    flush();

    /* check that ep does not hold dci */
    EXPECT_EQ(0, iface->tx.dci_pool[0].stack_top);
}

/* check that ep does not hold dci after purge
 */
UCS_TEST_P(test_dc, dcs_ep_purge_pending, "DC_NUM_DCI=1") {

    ucs_status_t status;
    uct_dc_mlx5_iface_t *iface;
    uct_dc_mlx5_ep_t *ep;

    m_e1->connect_to_iface(0, *m_e2);

    iface = dc_iface(m_e1);
    ep    = dc_ep(m_e1, 0);
    iface->tx.dcis[0].txqp.available = 8;

    do {
        status = uct_ep_am_short(m_e1->ep(0), 0, 0, NULL, 0);
    } while (status == UCS_OK);

    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);

    status = uct_ep_flush(m_e1->ep(0), 0, NULL);
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);

    /* put flush op on pending */
    preq.is_done = 0;
    preq.e = m_e1;
    preq.uct_req.func = uct_pending_flush;
    status = uct_ep_pending_add(m_e1->ep(0), &preq.uct_req, 0);
    EXPECT_UCS_OK(status);

    uct_ep_pending_purge(m_e1->ep(0), purge_cb, NULL);
    flush();
    EXPECT_EQ(UCT_DC_MLX5_EP_NO_DCI, ep->dci);
}

UCS_TEST_P(test_dc, rand_dci_many_eps) {
    uct_dc_mlx5_ep_t *ep;

    m_am_rx_count  = 0;
    entity *rand_e = create_rand_entity();
    uct_iface_set_am_handler(m_e2->iface(), 0, am_handler, NULL, 0);

    uct_dc_mlx5_iface_t *iface = dc_iface(rand_e);
    int num_eps                = 2 * iface->tx.ndci;

    /* Create more eps than we have dcis, all eps should have a valid dci */
    for (int i = 0; i < num_eps; i++) {
        rand_e->connect_to_iface(i, *m_e2);
        ep = dc_ep(rand_e, i);
        EXPECT_NE(UCT_DC_MLX5_EP_NO_DCI, ep->dci);
    }

    /* Try to send on all eps (taking into account available resources) */
    uint32_t num_sends = num_eps;

    for (unsigned i = 0; i < num_sends; i++) {
        ucs_status_t status = uct_ep_am_short(rand_e->ep(i), 0, 0, NULL, 0);
        EXPECT_UCS_OK(status);
    }
    wait_for_value(&m_am_rx_count, num_sends, true);

    EXPECT_EQ(m_am_rx_count, num_sends);

    flush();
}

UCS_TEST_P(test_dc, rand_dci_pending_purge) {
    entity *rand_e             = create_rand_entity();
    uct_dc_mlx5_iface_t *iface = dc_iface(rand_e);
    int num_eps                = RUNNING_ON_VALGRIND ? 3 : 5;
    int ndci                   = iface->tx.ndci;
    int num_reqs               = 10;
    int idx                    = 0;
    uct_pending_req_t preq[num_eps * num_reqs * ndci];
    int dci_id;
    uct_dc_mlx5_ep_t *ep;

    for (dci_id = 0; dci_id < ndci; ++dci_id) {
        for (int i = 0; i < num_eps; i++) {
            int ep_id = i + dci_id*ndci;
            rand_e->connect_to_iface(ep_id, *m_e2);
            ep = dc_ep(rand_e, ep_id);
            EXPECT_NE(UCT_DC_MLX5_EP_NO_DCI, ep->dci);
            int available = iface->tx.dcis[ep->dci].txqp.available;
            iface->tx.dcis[ep->dci].txqp.available = 0;
            for (int j = 0; j < num_reqs; ++j, ++idx) {
                preq[idx].func = NULL;
                ASSERT_UCS_OK(uct_ep_pending_add(rand_e->ep(ep_id),
                                                 &preq[idx], 0));
            }
            iface->tx.dcis[ep->dci].txqp.available = available;
        }
    }

    for (dci_id = 0; dci_id < ndci; ++dci_id) {
        for (int i = 0; i < num_eps; i++) {
            m_purge_count = 0;
            uct_ep_pending_purge(rand_e->ep(i + dci_id*ndci),
                                 purge_count_cb, NULL);
            EXPECT_EQ(num_reqs, m_purge_count);
        }
    }

    flush();
}

UCS_TEST_SKIP_COND_P(test_dc, stress_iface_ops,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY), "DC_NUM_DCI=1") {

    test_iface_ops(dc_iface(m_e1)->tx.dcis[0].txqp.available);
}

UCT_DC_INSTANTIATE_TEST_CASE(test_dc)


class test_dc_flow_control : public test_rc_flow_control {
public:

    /* virtual */
    uct_rc_fc_t* get_fc_ptr(entity *e, int ep_idx = 0) {
        return &ucs_derived_of(e->ep(ep_idx), uct_dc_mlx5_ep_t)->fc;
    }

    void set_fc_wnd(entity *e, int ep_idx = 0, int16_t fc_wnd = 0)
    {
        uct_dc_mlx5_iface_t *iface = ucs_derived_of(e->ep(ep_idx)->iface,
                                                    uct_dc_mlx5_iface_t);
        uct_dc_mlx5_ep_t *ep       = ucs_derived_of(e->ep(ep_idx),
                                                    uct_dc_mlx5_ep_t);

        get_fc_ptr(e, ep_idx)->fc_wnd = fc_wnd;
        ucs_status_t status           = uct_dc_mlx5_ep_check_fc(iface, ep);
        ASSERT_TRUE((status == UCS_OK) || (status == UCS_ERR_NO_RESOURCE));
    }

    virtual void disable_entity(entity *e) {
        uct_dc_mlx5_iface_t *iface = ucs_derived_of(e->iface(),
                                                    uct_dc_mlx5_iface_t);

        for (int i = 0; i < iface->tx.ndci; ++i) {
            uct_rc_txqp_available_set(&iface->tx.dcis[i].txqp, 0);
        }
        iface->tx.dci_pool[0].stack_top = iface->tx.ndci;
    }

    virtual void enable_entity(entity *e, unsigned cq_num = 128) {
        uct_dc_mlx5_iface_t *iface = ucs_derived_of(e->iface(),
                                                    uct_dc_mlx5_iface_t);

        for (int i = 0; i < iface->tx.ndci; ++i) {
            uct_rc_txqp_available_set(&iface->tx.dcis[i].txqp,
                                      iface->tx.dcis[i].txwq.bb_max);
        }
        iface->tx.dci_pool[0].stack_top = 0;

        uint8_t pool_index;
        for (pool_index = 0; pool_index < iface->tx.num_dci_pools;
             ++pool_index) {
            uct_dc_mlx5_iface_progress_pending(iface, pool_index);
        }
    }

    virtual void test_pending_grant(int wnd, uint64_t *wait_fc_seq = NULL)
    {
        test_rc_flow_control::test_pending_grant(wnd, wait_fc_seq);
        flush();
    }
};

UCS_TEST_P(test_dc_flow_control, general_enabled)
{
    /* Do not set FC hard thresh bigger than 1, because DC decreases
     * the window by one when it sends fc grant request. So some checks
     * may fail if threshold is bigger than 1. */
    test_general(8, 4, 1, true);
}

UCS_TEST_P(test_dc_flow_control, general_disabled)
{
    test_general(8, 4, 1, false);
}

UCS_TEST_P(test_dc_flow_control, pending_grant)
{
    test_pending_grant(5);
}

UCS_TEST_P(test_dc_flow_control, pending_grant_and_resend_hard_req,
           "DC_FC_HARD_REQ_TIMEOUT=0.1s")
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(m_e1->iface(),
                                                uct_dc_mlx5_iface_t);
    test_pending_grant(5, &iface->tx.fc_seq);
}

UCS_TEST_P(test_dc_flow_control, fc_disabled_flush)
{
    test_flush_fc_disabled();
}

UCS_TEST_P(test_dc_flow_control, fc_disabled_pending_no_dci) {

    pending_send_request_t pending_req;
    pending_req.uct.func = pending_cb;
    pending_req.cb_count = 0;

    set_fc_disabled(m_e1);

    /* Send on new endpoints until out of DCIs */
    for (int ep_index = 0; ep_index < 20; ++ep_index) {
        m_e1->connect(ep_index, *m_e2, ep_index);

        ucs_status_t status = uct_ep_am_short(m_e1->ep(ep_index), 0, 0, NULL, 0);
        if (status == UCS_ERR_NO_RESOURCE) {
            /* if FC is disabled, it should be OK to set fc_wnd to 0 */
            set_fc_wnd(m_e1, ep_index);

            /* Add to pending */
            status = uct_ep_pending_add(m_e1->ep(ep_index), &pending_req.uct, 0);
            ASSERT_UCS_OK(status);

            wait_for_flag(&pending_req.cb_count);
            EXPECT_EQ(1, pending_req.cb_count);
            break;
        }

        ASSERT_UCS_OK(status);
    }
}

/* Check that soft request is not handled by DC */
UCS_TEST_P(test_dc_flow_control, soft_request)
{
    int wnd = 8;
    int s_thresh = 4;
    int h_thresh = 1;

    set_fc_attributes(m_e1, true, wnd, s_thresh, h_thresh);

    send_am_and_flush(m_e1, wnd - (s_thresh - 1));

    set_tx_moderation(m_e2, 0);
    send_am_and_flush(m_e2, 1);

    /* Check that window is not updated */
    EXPECT_EQ(get_fc_ptr(m_e1)->fc_wnd, s_thresh - 1);
}

/* Check that:
 * 1) flush returns UCS_OK even if there is an outgoing grant request
 * 2) No crash when grant for destroyed ep arrives */
UCS_TEST_P(test_dc_flow_control, flush_destroy)
{
    int wnd = 5;
    ucs_status_t status;

    disable_entity(m_e2);

    set_fc_attributes(m_e1, true, wnd,
                      ucs_max((int)(wnd*0.5), 1),
                      ucs_max((int)(wnd*0.25), 1));

    send_am_and_flush(m_e1, wnd);

    /* At this point m_e1 sent grant request to m_e2, m_e2 received all
     * messages and added grant to m_e1 to pending queue
     * (because it does not have tx resources yet) */

    /* Invoke flush in a loop, because some send completions may not be polled yet */
    ucs_time_t timeout = ucs_get_time() + ucs_time_from_sec(DEFAULT_TIMEOUT_SEC);
    do {
        short_progress_loop();
        status = uct_ep_flush(m_e1->ep(0), 0, NULL);
    } while (((status == UCS_ERR_NO_RESOURCE) || (status == UCS_INPROGRESS)) &&
             (ucs_get_time() < timeout));
    ASSERT_UCS_OK(status);

    m_e1->destroy_eps();

    /* Enable send capabilities of m_e2 and send AM message to force pending queue
     * dispatch. Thus, pending grant will be sent to m_e1. There should not be
     * any warning/error and/or crash. */
    enable_entity(m_e2);
    set_tx_moderation(m_e2, 0);
    send_am_and_flush(m_e2, 1);
}

/* Check that there is no dci leak when just one (out of several) ep gets
 * grant. The leak can happen if some other ep has not got grant yet, but
 * is scheduled for dci allocation. */
UCS_TEST_P(test_dc_flow_control, dci_leak)
{
    disable_entity(m_e2);
    int wnd = 5;
    set_fc_attributes(m_e1, true, wnd,
                      ucs_max((int)(wnd*0.5), 1),
                      ucs_max((int)(wnd*0.25), 1));
    send_am_messages(m_e1, wnd, UCS_OK);
    send_am_messages(m_e1, 1, UCS_ERR_NO_RESOURCE);
    uct_pending_req_t req;
    req.func = reinterpret_cast<ucs_status_t (*)(uct_pending_req*)>
                               (ucs_empty_function_return_no_resource);
    EXPECT_UCS_OK(uct_ep_pending_add(m_e1->ep(0), &req, 0));

    /* Make sure that ep does not hold dci when sends completed */
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(m_e1->iface(), uct_dc_mlx5_iface_t);
    ucs_time_t deadline        = ucs::get_deadline();
    while (iface->tx.dci_pool[0].stack_top && (ucs_get_time() < deadline)) {
        progress();
    }
    EXPECT_EQ(0, iface->tx.dci_pool[0].stack_top);

    /* Clean up FC and pending to avoid assertions during tear down */
    uct_ep_pending_purge(m_e1->ep(0),
           reinterpret_cast<void (*)(uct_pending_req*, void*)> (ucs_empty_function),
           NULL);
    enable_entity(m_e2);
    set_tx_moderation(m_e2, 0);
    send_am_messages(m_e2, 1, UCS_OK);
    validate_grant(m_e1);
}

UCT_DC_INSTANTIATE_TEST_CASE(test_dc_flow_control)

class test_dc_iface_attrs : public test_rc_iface_attrs {
public:
    attr_map_t get_num_iov() {
        return get_num_iov_mlx5_common(UCT_IB_MLX5_AV_FULL_SIZE);
    }
};

UCS_TEST_P(test_dc_iface_attrs, iface_attrs)
{
    basic_iov_test();
}

UCT_DC_INSTANTIATE_TEST_CASE(test_dc_iface_attrs)

class test_dc_fc_deadlock : public test_dc_flow_control {
public:
    test_dc_fc_deadlock() {
        modify_config("IB_TX_QUEUE_LEN", "8");
        modify_config("RC_FC_WND_SIZE", "128");
        modify_config("DC_TX_POLICY", "rand");
    }

protected:
    struct dc_pending {
        uct_pending_req_t uct_req;
        entity *e;
    };

    static ucs_status_t am_pending(uct_pending_req_t *req) {
        struct dc_pending *pr = reinterpret_cast<struct dc_pending *>(req);
        return uct_ep_am_short(pr->e->ep(0), 0, 0, NULL, 0);
    }
};

UCS_TEST_P(test_dc_fc_deadlock, basic, "DC_NUM_DCI=1")
{
    // Send to m_e2 until dci resources are exhausted.
    // Also set FC window to 0 emulating lack of all TX resources
    ucs_status_t status;
    do {
        status = uct_ep_am_short(m_e1->ep(0), 0, 0, NULL, 0);
    } while (status == UCS_OK);
    send_am_messages(m_e1, 1, UCS_ERR_NO_RESOURCE);
    set_fc_wnd(m_e1);

    // Add am send to pending
    struct dc_pending preq;
    preq.e            = m_e1;
    preq.uct_req.func = am_pending;
    EXPECT_UCS_OK(uct_ep_pending_add(m_e1->ep(0), &preq.uct_req, 0));

    // Send whole FC window to m_e1, which will force sending grant request.
    // This grant request will be added to pending on m_e1, because it has no
    // resources.
    int wnd = 5;
    set_fc_attributes(m_e2, true, wnd,
                      ucs_max((int)(wnd*0.5), 1),
                      ucs_max((int)(wnd*0.25), 1));
    send_am_and_flush(m_e2, wnd);

    // Now, make sure that m_e1 will send grant to m_e2 even though FC window
    // is still empty (dci resources will be restored during progression).
    // If grant was not sent, this would be a deadlock situation due to lack
    // of FC resources.
    validate_grant(m_e2);

    // Restore m_e1 for proper cleanup
    uct_ep_pending_purge(m_e1->ep(0), NULL, NULL);
}

UCT_DC_INSTANTIATE_TEST_CASE(test_dc_fc_deadlock)


#ifdef ENABLE_STATS

class test_dc_flow_control_stats : public test_rc_flow_control_stats {
public:
    /* virtual */
    void init() {
        if (UCS_OK != uct_config_modify(m_iface_config, "RC_FC_ENABLE", "y")) {
            UCS_TEST_ABORT("Error: cannot enable flow control");
        }
        test_rc_flow_control_stats::init();
    }

    uct_rc_fc_t* get_fc_ptr(entity *e, int ep_idx = 0) {
        return &ucs_derived_of(e->ep(ep_idx), uct_dc_mlx5_ep_t)->fc;
    }

    uct_rc_fc_t* fake_ep_fc_ptr(entity *e) {
        return &ucs_derived_of(e->iface(), uct_dc_mlx5_iface_t)->tx.fc_ep->fc;
    }
};

UCS_TEST_P(test_dc_flow_control_stats, general)
{
    test_general(5, 2, 1);
}

UCS_TEST_P(test_dc_flow_control_stats, fc_ep)
{
    uint64_t v;
    int wnd = 5;

    set_fc_attributes(m_e1, true, wnd,
                      ucs_max((int)(wnd*0.5), 1),
                      ucs_max((int)(wnd*0.25), 1));

    send_am_messages(m_e1, wnd, UCS_OK);
    validate_grant(m_e1);

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_TX_HARD_REQ);
    EXPECT_EQ(1ul, v);
    v = UCS_STATS_GET_COUNTER(fake_ep_fc_ptr(m_e2)->stats, UCT_RC_FC_STAT_RX_HARD_REQ);
    EXPECT_EQ(1ul, v);

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_RX_PURE_GRANT);
    EXPECT_EQ(1ul, v);
    v = UCS_STATS_GET_COUNTER(fake_ep_fc_ptr(m_e2)->stats, UCT_RC_FC_STAT_TX_PURE_GRANT);
    EXPECT_EQ(1ul, v);
    flush();
}


UCT_DC_INSTANTIATE_TEST_CASE(test_dc_flow_control_stats)

#endif
