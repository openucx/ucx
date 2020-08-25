
/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.All rights reserved.
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
}
#include <common/test.h>
#include "uct_test.h"

#include <vector>


class test_uct_peer_failure : public uct_test {
private:
    struct am_handler_setter
    {
        am_handler_setter(test_uct_peer_failure *test) : m_test(test) {}

        void operator() (test_uct_peer_failure::entity *e) {
            uct_iface_set_am_handler(e->iface(), 0,
                                     am_dummy_handler,
                                     reinterpret_cast<void*>(m_test), 0);
        }

        test_uct_peer_failure* m_test;
    };

public:

    test_uct_peer_failure() : m_sender(NULL),  m_nreceivers(2),
                              m_tx_window(100), m_err_count(0), m_am_count(0) {}

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

    static ucs_status_t am_dummy_handler(void *arg, void *data, size_t length,
                                         unsigned flags) {
        reinterpret_cast<test_uct_peer_failure*>(arg)->m_am_count++;
        return UCS_OK;
    }

    typedef struct {
        uct_pending_req_t    uct;
        uct_ep_h             ep;
    } pending_send_request_t;

    static ucs_status_t pending_cb(uct_pending_req_t *self)
    {
        const uint64_t send_data    = 0;
        pending_send_request_t *req = ucs_container_of(self,
                                                       pending_send_request_t,
                                                       uct);

        ucs_status_t status;
        do {
            /* Block in the pending handler (sending AM Short to fill UCT
             * resources) to keep the pending requests in pending queue
             * to purge them */
            status = uct_ep_am_short(req->ep, 0, 0, &send_data,
                                     sizeof(send_data));
        } while (status == UCS_OK);

        return status;
    }

    static void purge_cb(uct_pending_req_t *self, void *arg)
    {
        m_req_purge_count++;
    }

    static ucs_status_t err_cb(void *arg, uct_ep_h ep, ucs_status_t status)
    {
        EXPECT_EQ(UCS_ERR_ENDPOINT_TIMEOUT, status);
        reinterpret_cast<test_uct_peer_failure*>(arg)->m_err_count++;
        return UCS_OK;
    }

    void kill_receiver()
    {
        ucs_assert(!m_receivers.empty());
        m_entities.remove(m_receivers.front());
        ucs_assert(m_entities.size() == m_receivers.size());
        m_receivers.erase(m_receivers.begin());
    }

    void new_receiver()
    {
        uct_iface_params_t p = entity_params();
        p.field_mask |= UCT_IFACE_PARAM_FIELD_OPEN_MODE;
        p.open_mode   = UCT_IFACE_OPEN_MODE_DEVICE;
        m_receivers.push_back(uct_test::create_entity(p));
        m_entities.push_back(m_receivers.back());
        m_sender->connect(m_receivers.size() - 1, *m_receivers.back(), 0);

        am_handler_setter(this)(m_receivers.back());
        /* Make sure that TL is up and has resources */
        send_recv_am(m_receivers.size() - 1);
    }

    void set_am_handlers()
    {
        check_caps_skip(UCT_IFACE_FLAG_CB_SYNC);
        std::for_each(m_receivers.begin(), m_receivers.end(),
                      am_handler_setter(this));
    }

    ucs_status_t send_am(int index)
    {
        ucs_status_t status;
        while ((status = uct_ep_am_short(m_sender->ep(index), 0, 0, NULL, 0)) ==
               UCS_ERR_NO_RESOURCE) {
            progress();
        };
        return status;
    }

    void send_recv_am(int index, ucs_status_t exp_status = UCS_OK)
    {
        m_am_count = 0;

        ucs_status_t status = send_am(index);
        EXPECT_EQ(exp_status, status);

        if (exp_status == UCS_OK) {
            wait_for_flag(&m_am_count);
            EXPECT_EQ(m_am_count, 1ul);
        }
    }

    uct_ep_h ep0() {
        return m_sender->ep(0);
    }

    ucs_status_t flush_ep(size_t index, ucs_time_t deadline = ULONG_MAX) {
        uct_completion_t    comp;
        ucs_status_t        status;
        int                 is_time_out;

        comp.count  = 2;
        comp.status = UCS_OK;
        comp.func   = NULL;
        do {
            progress();
            status = uct_ep_flush(m_sender->ep(index), 0, &comp);
            is_time_out = (ucs_get_time() > deadline);
        } while ((status == UCS_ERR_NO_RESOURCE) && !is_time_out);

        if (!is_time_out) {
            ASSERT_UCS_OK_OR_INPROGRESS(status);
        }

        if (status == UCS_OK) {
            return UCS_OK;
        } else if (is_time_out) {
            return UCS_ERR_TIMED_OUT;
        }

        /* coverity[loop_condition] */
        while ((comp.count == 2) && !is_time_out) {
            progress();
            is_time_out = (ucs_get_time() > deadline);
        }

        return (comp.count == 1) ? UCS_OK :
               (is_time_out ? UCS_ERR_TIMED_OUT : UCS_ERR_OUT_OF_RANGE);
    }

    ucs_status_t add_pending(uct_ep_h ep, pending_send_request_t &req) {
        req.ep       = ep;
        req.uct.func = pending_cb;
        return uct_ep_pending_add(ep, &req.uct, 0);
    }

    void fill_resources(bool expect_error, ucs_time_t loop_end_limit) {
        const uint64_t send_data = 0;
        ucs_status_t status;
        do {
            status = uct_ep_am_short(ep0(), 0, 0, &send_data,
                                     sizeof(send_data));
        } while ((status == UCS_OK) && (ucs_get_time() < loop_end_limit));

        if (status == UCS_OK) {
            UCS_TEST_SKIP_R("unable to fill the UCT resources");
        } else if (status != UCS_ERR_NO_RESOURCE) {
            if (expect_error && UCS_IS_ENDPOINT_ERROR(status)) {
                UCS_TEST_SKIP_R("unable to fill the UCT resources, since "
                                "peer failure has been already detected");
            } else {
                UCS_TEST_ABORT("AM Short failed with " <<
                               ucs_status_string(status));
            }
        }
    }

protected:
    entity                *m_sender;
    std::vector<entity *> m_receivers;
    size_t                m_nreceivers;
    size_t                m_tx_window;
    size_t                m_err_count;
    size_t                m_am_count;
    static size_t         m_req_purge_count;
    static const uint64_t m_required_caps;
};

size_t test_uct_peer_failure::m_req_purge_count       = 0ul;
const uint64_t test_uct_peer_failure::m_required_caps = UCT_IFACE_FLAG_AM_SHORT  |
                                                        UCT_IFACE_FLAG_PENDING   |
                                                        UCT_IFACE_FLAG_CB_SYNC   |
                                                        UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;

void test_uct_peer_failure::init()
{
    uct_test::init();

    reduce_tl_send_queues();

    /* To reduce test execution time decrease retransmition timeouts
     * where it is relevant */
    set_config("RC_TIMEOUT?=100us"); /* 100 us should be enough */
    set_config("RC_RETRY_COUNT?=4");
    set_config("UD_TIMEOUT?=3s");

    uct_iface_params_t p = entity_params();
    p.field_mask |= UCT_IFACE_PARAM_FIELD_OPEN_MODE;
    p.open_mode   = UCT_IFACE_OPEN_MODE_DEVICE;
    m_sender = uct_test::create_entity(p);
    m_entities.push_back(m_sender);

    check_skip_test();
    for (size_t i = 0; i < 2; ++i) {
        new_receiver();
    }

    m_err_count       = 0;
    m_req_purge_count = 0;
    m_am_count        = 0;
}

UCS_TEST_SKIP_COND_P(test_uct_peer_failure, peer_failure,
                     !check_caps(UCT_IFACE_FLAG_PUT_SHORT |
                                 m_required_caps))
{
    {
        scoped_log_handler slh(wrap_errors_logger);

        kill_receiver();
        EXPECT_EQ(UCS_OK, uct_ep_put_short(ep0(), NULL, 0, 0, 0));

        flush();
    }

    ASSERT_UCS_OK_OR_INPROGRESS(uct_ep_flush(ep0(), 0, NULL));
    EXPECT_EQ(uct_ep_pending_add(ep0(), NULL, 0), UCS_ERR_BUSY);

    EXPECT_GT(m_err_count, 0ul);
}

UCS_TEST_SKIP_COND_P(test_uct_peer_failure, purge_failed_peer,
                     !check_caps(m_required_caps))
{
    set_am_handlers();

    send_recv_am(0);
    send_recv_am(1);

    const size_t num_pend_sends     = 64ul;
    const ucs_time_t loop_end_limit = ucs::get_deadline();
    std::vector<pending_send_request_t> reqs(num_pend_sends);

    {
        scoped_log_handler slh(wrap_errors_logger);
        ucs_status_t status;

        fill_resources(false, loop_end_limit);
        kill_receiver();

        do {
            status = add_pending(ep0(), reqs[0]);
            if (UCS_OK != status) {
                EXPECT_EQ(UCS_ERR_BUSY, status);
                fill_resources(true, loop_end_limit);
            }
        } while ((status == UCS_ERR_BUSY) && (ucs_get_time() < loop_end_limit));

        if (status == UCS_ERR_BUSY) {
            UCS_TEST_SKIP_R("unable to add pending requests after "
                            "filling UCT resources");
        }

        for (size_t i = 1; i < num_pend_sends; i++) {
            EXPECT_UCS_OK(add_pending(ep0(), reqs[i]));
        }

        flush();
    }

    uct_ep_pending_purge(ep0(), purge_cb, NULL);
    EXPECT_EQ(num_pend_sends, m_req_purge_count);
    EXPECT_GE(m_err_count, 0ul);
}

UCS_TEST_SKIP_COND_P(test_uct_peer_failure, two_pairs_send,
                     !check_caps(m_required_caps))
{
    set_am_handlers();

    /* queue sends on 1st pair */
    for (size_t i = 0; i < m_tx_window; ++i) {
        send_am(0);
    }

    /* kill the 1st receiver while sending on 2nd pair */
    {
        scoped_log_handler slh(wrap_errors_logger);
        kill_receiver();
        send_am(0);
        send_recv_am(1);
        flush();
    }
}

class test_uct_peer_failure_cb : public test_uct_peer_failure {
public:
    virtual uct_error_handler_t get_err_handler() const {
        return err_cb_ep_destroy;
    }

    static ucs_status_t err_cb_ep_destroy(void *arg, uct_ep_h ep, ucs_status_t status) {
        test_uct_peer_failure_cb *self(reinterpret_cast<test_uct_peer_failure_cb*>(arg));
        EXPECT_EQ(self->ep0(), ep);
        self->m_sender->destroy_ep(0);
        return UCS_OK;
    }
};

UCS_TEST_SKIP_COND_P(test_uct_peer_failure_cb, desproy_ep_cb,
                     !check_caps(UCT_IFACE_FLAG_PUT_SHORT |
                                 m_required_caps))
{
    scoped_log_handler slh(wrap_errors_logger);
    kill_receiver();
    EXPECT_EQ(uct_ep_put_short(ep0(), NULL, 0, 0, 0), UCS_OK);
    flush();
}

UCT_INSTANTIATE_TEST_CASE(test_uct_peer_failure_cb)

class test_uct_peer_failure_keepalive : public test_uct_peer_failure
{
};

UCS_TEST_SKIP_COND_P(test_uct_peer_failure_keepalive, killed,
                     !check_caps(UCT_IFACE_FLAG_EP_CHECK))
{
    ucs_status_t status;

    scoped_log_handler slh(wrap_errors_logger);
    flush();
    EXPECT_EQ(0, m_err_count);
    flush();

    status = uct_ep_check(ep0(), 0, NULL);
    ASSERT_UCS_OK(status);
    flush();

    kill_receiver();

    status = uct_ep_check(ep0(), 0, NULL);
    ASSERT_UCS_OK(status);
    flush();

    EXPECT_EQ(1, m_err_count);
}

UCT_INSTANTIATE_NO_SELF_TEST_CASE(test_uct_peer_failure_keepalive)
