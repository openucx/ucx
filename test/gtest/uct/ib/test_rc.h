/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RC_TEST_H_
#define UCT_RC_TEST_H_

#include <common/test.h>
#include <uct/uct_test.h>

extern "C" {
#include <uct/api/uct.h>
#include <uct/ib/rc/base/rc_ep.h>
#include <uct/ib/rc/base/rc_iface.h>
}


class test_rc : public uct_test {
public:
    virtual void init();
    void connect();

    uct_rc_iface_t* rc_iface(entity *e) {
        return ucs_derived_of(e->iface(), uct_rc_iface_t);
    }

    uct_rc_ep_t* rc_ep(entity *e, int ep_idx = 0) {
        return ucs_derived_of(e->ep(ep_idx), uct_rc_ep_t);
    }

    void send_am_messages(entity *e, int wnd, ucs_status_t expected,
                          uint8_t am_id = 0, int ep_idx = 0) {
        for (int i = 0; i < wnd; i++) {
            EXPECT_EQ(expected, uct_ep_am_short(e->ep(ep_idx), am_id, 0, NULL, 0));
        }
    }

    void progress_loop(double delta_ms=10.0) {
        uct_test::short_progress_loop(delta_ms);
    }

    static ucs_status_t am_dummy_handler(void *arg, void *data, size_t length,
                                         unsigned flags) {
        return UCS_OK;
    }

protected:
    entity *m_e1, *m_e2;

};

class test_rc_flow_control : public test_rc {
public:
    typedef struct pending_send_request {
        uct_pending_req_t uct;
        int               cb_count;
        int               purge_count;
    } pending_send_request_t;

    void init();
    void cleanup();

    virtual uct_rc_fc_t* get_fc_ptr(entity *e, int ep_idx = 0) {
        return &rc_ep(e, ep_idx)->fc;
    }

    virtual void disable_entity(entity *e) {
        rc_iface(e)->tx.cq_available = 0;
    }

    virtual void enable_entity(entity *e, unsigned cq_num = 128) {
        rc_iface(e)->tx.cq_available = cq_num;
    }

    virtual void set_tx_moderation(entity *e, int val) {
        rc_iface(e)->config.tx_moderation = val;
    }

    void set_fc_attributes(entity *e, bool enabled, int wnd, int s_thresh, int h_thresh) {
        rc_iface(e)->config.fc_enabled     = enabled;
        rc_iface(e)->config.fc_wnd_size    = get_fc_ptr(e)->fc_wnd = wnd;
        rc_iface(e)->config.fc_soft_thresh = s_thresh;
        rc_iface(e)->config.fc_hard_thresh = h_thresh;

    }

    void set_fc_disabled(entity *e) {
        /* same as default settings in rc_iface_init */
        set_fc_attributes(e, false, std::numeric_limits<int16_t>::max(), 0, 0);
    }

    void send_am_and_flush(entity *e, int num_msg);

    void progress_loop(double delta_ms=10.0) {
        uct_test::short_progress_loop(delta_ms);
    }

    static ucs_status_t am_dummy_handler(void *arg, void *data, size_t length, void *desc) {
        return UCS_OK;
    }

    static ucs_status_t am_handler(void *arg, void *data, size_t length,
                                   unsigned flags)
    {
        ++m_am_rx_count;
        return UCS_OK;
    }

    static void purge_cb(uct_pending_req_t *self, void *arg) {
        pending_send_request_t *req = ucs_container_of(self,
                                                       pending_send_request_t,
                                                       uct);
        ++req->purge_count;
    }

    static ucs_status_t pending_cb(uct_pending_req_t *self) {
        pending_send_request_t *req = ucs_container_of(self,
                                                       pending_send_request_t,
                                                       uct);
        ++req->cb_count;
        return UCS_OK;
    }

    void validate_grant(entity *e);

    void test_general(int wnd, int s_thresh, int h_thresh, bool is_fc_enabled);

    void test_pending_grant(int wnd);

    void test_pending_purge(int wnd, int num_pend_sends);

    void test_flush_fc_disabled();

protected:
    static const uint8_t FLUSH_AM_ID = 1;
    static uint32_t m_am_rx_count;
};


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

    void test_general(int wnd, int s_thresh, int h_thresh);
};
#endif

#endif
