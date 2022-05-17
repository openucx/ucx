/**
* Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "uct_test.h"
#include <common/test.h>
extern "C" {
#include <uct/api/uct.h>
#include <uct/sm/mm/base/mm_ep.h>
#include <uct/sm/scopy/cma/cma_ep.h>
}

#include <vector>


class test_uct_peer_failure : public uct_test {
public:
    typedef struct {
        uct_pending_req_t uct;
        uct_ep_h          ep;
    } pending_send_request_t;

    test_uct_peer_failure();
    virtual void init();

    inline uct_iface_params_t entity_params() {
        static uct_iface_params_t params;
        params.field_mask = UCT_IFACE_PARAM_FIELD_ERR_HANDLER     |
                            UCT_IFACE_PARAM_FIELD_ERR_HANDLER_ARG |
                            UCT_IFACE_PARAM_FIELD_ERR_HANDLER_FLAGS;
        params.err_handler       = get_err_handler();
        params.err_handler_arg   = reinterpret_cast<void*>(this);
        params.err_handler_flags = 0;
        return params;
    }

    virtual uct_error_handler_t get_err_handler() const {
        return err_cb;
    }

    uct_ep_h ep0() {
        return m_sender->ep(0);
    }

    static ucs_status_t am_dummy_handler(void *arg, void *data, size_t length,
                                         unsigned flags);
    static ucs_status_t pending_cb(uct_pending_req_t *self);
    static void purge_cb(uct_pending_req_t *self, void *arg);
    static ucs_status_t err_cb(void *arg, uct_ep_h ep, ucs_status_t status);
    void inject_error(unsigned idx = 0);
    void kill_receiver(unsigned idx = 0);
    void new_receiver();
    void set_am_handlers();
    ucs_status_t send_am(int index);
    void send_recv_am(int index, ucs_status_t exp_status = UCS_OK);
    ucs_status_t flush_ep(size_t index, ucs_time_t deadline = ULONG_MAX);
    ucs_status_t add_pending(uct_ep_h ep, pending_send_request_t &req);
    void fill_resources(bool expect_error, ucs_time_t loop_end_limit);

protected:
    entity                           *m_sender;
    std::vector<entity *>            m_receivers;
    std::map<uct_ep_h, ucs_status_t> m_failed_eps;
    size_t                           m_nreceivers;
    size_t                           m_tx_window;
    size_t                           m_err_count;
    std::vector<size_t>              m_am_count;
    size_t                           m_req_purge_count;
    size_t                           m_req_pending_count;
    static const uint64_t            m_required_caps;
};

