
/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.All rights reserved.
* See file LICENSE for terms.
*/

#include "test_peer_failure.h"

#if HAVE_CUDA
extern "C" {
#include <uct/cuda/cuda_ipc/cuda_ipc_ep.h>
}
#endif


const uint64_t test_uct_peer_failure::m_required_caps = UCT_IFACE_FLAG_AM_SHORT  |
                                                        UCT_IFACE_FLAG_PENDING   |
                                                        UCT_IFACE_FLAG_CB_SYNC   |
                                                        UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;

test_uct_peer_failure::test_uct_peer_failure() :
    m_sender(NULL), m_nreceivers(2), m_tx_window(100),
    m_err_count(0), m_am_count(0), m_req_purge_count(0),
    m_req_pending_count(0)
{
}

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
}

ucs_status_t test_uct_peer_failure::am_dummy_handler(void *arg, void *data,
                                                     size_t length,
                                                     unsigned flags)
{
    test_uct_peer_failure* test = reinterpret_cast<test_uct_peer_failure*>(arg);
    uint64_t index              = *reinterpret_cast<uint64_t*>(data);

    test->m_am_count[index]++;
    return UCS_OK;
}

ucs_status_t test_uct_peer_failure::pending_cb(uct_pending_req_t *self)
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

void test_uct_peer_failure::purge_cb(uct_pending_req_t *self, void *arg)
{
    test_uct_peer_failure *test =
            reinterpret_cast<test_uct_peer_failure*>(arg);
    test->m_req_purge_count++;
}

ucs_status_t test_uct_peer_failure::err_cb(void *arg, uct_ep_h ep,
                                           ucs_status_t status)
{
    test_uct_peer_failure *self = reinterpret_cast<test_uct_peer_failure*>(arg);

    self->m_err_count++;
    self->m_failed_eps.insert(std::make_pair(ep, status));

    /* any new op is not determined */
    uct_ep_pending_purge(ep, purge_cb, self);
    EXPECT_EQ(self->m_req_pending_count, self->m_req_purge_count);

    switch (status) {
    case UCS_ERR_ENDPOINT_TIMEOUT:
    case UCS_ERR_CONNECTION_RESET:
    case UCS_ERR_CANCELED: /* goes from ib flushed QP */
        return UCS_OK;
    default:
        EXPECT_TRUE(false) << "unexpected error status: "
                           << ucs_status_string(status);
        return status;
    }
}

void test_uct_peer_failure::inject_error(unsigned idx)
{
    uct_ep_h ep         = m_sender->ep(idx);
    ucs_status_t status = uct_ep_invalidate(ep, 0);

    if (status == UCS_ERR_UNSUPPORTED) {
        kill_receiver(idx);
    } else {
        ASSERT_UCS_OK(status);
    }
}

void test_uct_peer_failure::kill_receiver(unsigned idx)
{
    /* TODO: add mapping sender idx -> receiver */
    ucs_assert(!m_receivers.empty());
    m_entities.remove(m_receivers.front());
    ucs_assert(m_entities.size() == m_receivers.size());
    m_receivers.erase(m_receivers.begin());
}

void test_uct_peer_failure::new_receiver()
{
    uct_iface_params_t p = entity_params();
    p.field_mask |= UCT_IFACE_PARAM_FIELD_OPEN_MODE;
    p.open_mode   = UCT_IFACE_OPEN_MODE_DEVICE;
    m_receivers.push_back(uct_test::create_entity(p));
    m_entities.push_back(m_receivers.back());
    m_sender->connect(m_receivers.size() - 1, *m_receivers.back(), 0);

    if (m_sender->iface_attr().cap.flags & UCT_IFACE_FLAG_AM_SHORT) {
        m_am_count.push_back(0);
        /* Make sure that TL is up and has resources */
        uct_iface_set_am_handler(m_receivers.back()->iface(), 0,
                                 am_dummy_handler, this, 0);
        send_recv_am(m_receivers.size() - 1);
    }
}

void test_uct_peer_failure::set_am_handlers()
{
    check_caps_skip(UCT_IFACE_FLAG_CB_SYNC);
    for (auto e : m_receivers) {
        uct_iface_set_am_handler(e->iface(), 0, am_dummy_handler, this, 0);
    }
}

ucs_status_t test_uct_peer_failure::send_am(int index)
{
    ucs_status_t status;

    do {
        progress();

        /* If the endpoint has failed, return error and avoid calling send */
        std::map<uct_ep_h, ucs_status_t>::iterator it =
                m_failed_eps.find(m_sender->ep(index));
        if (it != m_failed_eps.end()) {
            return it->second;
        }

        status = uct_ep_am_short(m_sender->ep(index), 0, uint64_t(index), NULL,
                                 0);
    } while (status == UCS_ERR_NO_RESOURCE);

    return status;
}

void test_uct_peer_failure::send_recv_am(int index, ucs_status_t exp_status)
{
    m_am_count[index] = 0;

    ucs_status_t status = send_am(index);
    EXPECT_EQ(exp_status, status);

    if (exp_status == UCS_OK) {
        wait_for_flag(&m_am_count[index]);
        EXPECT_EQ(m_am_count[index], 1ul);
    }
}

ucs_status_t test_uct_peer_failure::flush_ep(size_t index,
                                             ucs_time_t deadline)
{
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

ucs_status_t test_uct_peer_failure::add_pending(uct_ep_h ep,
                                                pending_send_request_t &req)
{
    ++m_req_pending_count;
    req.ep       = ep;
    req.uct.func = pending_cb;
    return uct_ep_pending_add(ep, &req.uct, 0);
}

void test_uct_peer_failure::fill_resources(bool expect_error,
                                           ucs_time_t loop_end_limit)
{
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

UCS_TEST_SKIP_COND_P(test_uct_peer_failure, peer_failure,
                     !check_caps(UCT_IFACE_FLAG_PUT_SHORT |
                                 m_required_caps))
{
    {
        scoped_log_handler slh(wrap_errors_logger);

        inject_error();
        EXPECT_EQ(UCS_OK, uct_ep_put_short(ep0(), NULL, 0, 0, 0));

        flush();
    }

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
        inject_error();

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
        inject_error();
        send_am(0);
        send_recv_am(1);
        flush();
    }

    /* test flushing one operations */
    send_recv_am(1, UCS_OK);
    flush();

    /* test flushing many operations */
    for (size_t i = 0; i < (m_tx_window * 10 / ucs::test_time_multiplier()); ++i) {
        send_recv_am(1, UCS_OK);
    }
    flush();
}


UCT_INSTANTIATE_TEST_CASE(test_uct_peer_failure)

class test_uct_peer_failure_multiple : public test_uct_peer_failure
{
public:
    virtual void init();

protected:
    size_t get_tx_queue_len() const;
};

void test_uct_peer_failure_multiple::init()
{
    size_t tx_queue_len = get_tx_queue_len();

    if ((ucs_get_page_size() > 4096) ||
        (ucs_arch_get_cpu_model() == UCS_CPU_MODEL_ARM_AARCH64)) {
        /* NOTE: Too many receivers may cause failure of ibv_open_device */
        m_nreceivers = 10;
    } else {
        m_nreceivers = tx_queue_len;
    }

    /* Do not create more receivers than the number allowed connections or
       number of workers (assuming each worker can open up to 16 files) */
    int max_workers = ucs_sys_max_open_files() / 16;
    m_nreceivers    = std::min(m_nreceivers,
                               std::min<size_t>(max_connections(), max_workers));

    test_uct_peer_failure::m_tx_window  = tx_queue_len / 3;

    test_uct_peer_failure::init();

    m_receivers.reserve(m_nreceivers);
    while (m_receivers.size() < m_nreceivers) {
        new_receiver();
    }
}

size_t test_uct_peer_failure_multiple::get_tx_queue_len() const
{
    bool        set = true;
    std::string name, val;
    size_t      tx_queue_len;

    if (has_rc()) {
        name = "RC_RC_IB_TX_QUEUE_LEN";
    } else if (has_transport("dc_mlx5")) {
        name = "DC_RC_IB_TX_QUEUE_LEN";
    } else if (has_ud()) {
        name = "UD_IB_TX_QUEUE_LEN";
    } else {
        set  = false;
        name = "TX_QUEUE_LEN";
    }

    if (get_config(name, val)) {
        tx_queue_len = ucs::from_string<size_t>(val);
        EXPECT_LT(0ul, tx_queue_len);
    } else {
        tx_queue_len = 256;
        UCS_TEST_MESSAGE << name << " setting not found, "
                         << "taken test default value: " << tx_queue_len;
        if (set) {
            UCS_TEST_ABORT(name + " config name must be found for %s transport" +
                           GetParam()->tl_name);
        }
    }

    return tx_queue_len;
}

/* Skip under valgrind due to brk segment overflow.
 * See https://bugs.kde.org/show_bug.cgi?id=352742 */
UCS_TEST_SKIP_COND_P(test_uct_peer_failure_multiple, test,
                     (RUNNING_ON_VALGRIND ||
                      !check_caps(m_required_caps)),
                     "RC_TM_ENABLE?=n")
{
    /* with DC peer failure may stuck peer DCRs and cause very long DCT close */
    ucs_time_t timeout  = ucs_get_time() +
                          ucs_time_from_sec(300 * ucs::test_time_multiplier());

    {
        scoped_log_handler slh(wrap_errors_logger);
        for (size_t idx = 0; idx < (m_nreceivers - 1); ++idx) {
            for (size_t i = 0; i < m_tx_window; ++i) {
                EXPECT_UCS_OK(send_am(idx));
            }
            inject_error(idx);
        }
        flush(timeout);

        for (size_t idx = 0; idx < (m_nreceivers - 1); ++idx) {
            /* If EP were not failed yet, these ops should trigger that */
            for (size_t i = 0; i < m_tx_window; ++i) {
                if (UCS_STATUS_IS_ERR(send_am(idx))) {
                    break;
                }
            }
        }

        flush(timeout);
    }

    int idx = m_nreceivers - 1;
    m_am_count[idx] = 0;
    EXPECT_UCS_OK(send_am(idx));
    ucs_debug("flushing");
    flush_ep(idx);
    ucs_debug("flushed");
    wait_for_flag(&m_am_count[idx]);
    EXPECT_EQ(m_am_count[idx], 1ul);
}

UCT_INSTANTIATE_TEST_CASE(test_uct_peer_failure_multiple)

class test_uct_keepalive : public uct_test {
public:
    test_uct_keepalive() :
        m_pid(getpid()), m_entity(NULL), m_err_handler_count(0)
    {
    }

    void init()
    {
        m_entity = create_entity(0, err_handler_cb);
        m_entity->connect(0, *m_entity, 0);
        m_entities.push_back(m_entity);
        ASSERT_TRUE(has_mm());
    }

    void cleanup()
    {
        m_entities.clear();
    }

    uct_keepalive_info_t *m_ka()
    {
        uct_mm_ep_t *ep = ucs_derived_of(m_entity->ep(0), uct_mm_ep_t);

        return &ep->keepalive;
    }

    static ucs_status_t
    err_handler_cb(void *arg, uct_ep_h ep, ucs_status_t status)
    {
        test_uct_keepalive *self = reinterpret_cast<test_uct_keepalive*>(arg);
        self->m_err_handler_count++;
        return UCS_OK;
    }

protected:
    void do_keepalive()
    {
        uct_ep_keepalive_check(m_entity->ep(0), m_ka(), m_pid, 0, NULL);
    }

    pid_t                m_pid;
    entity               *m_entity;
    unsigned             m_err_handler_count;
};

UCS_TEST_P(test_uct_keepalive, ep_check)
{
    for (unsigned i = 0; i < 10; ++i) {
        do_keepalive();
    }
    EXPECT_EQ(0u, m_err_handler_count);

    /* change start time saved in KA to force an error from EP check */
    m_ka()->start_time--;

    do_keepalive();
    EXPECT_EQ(0u, m_err_handler_count);

    progress();
    EXPECT_EQ(1u, m_err_handler_count);
}

// Need to instantiate only on one transport
_UCT_INSTANTIATE_TEST_CASE(test_uct_keepalive, posix)


class test_uct_peer_failure_keepalive : public test_uct_peer_failure
{
public:
    test_uct_peer_failure_keepalive()
    {
        m_env.push_back(new ucs::scoped_setenv("UCX_TCP_KEEPIDLE", "inf"));
    }

    void kill_receiver()
    {
        /* Hack: for SHM-based transports we can't really terminate
         * peer EP, but instead we bit change process owner info to force
         * ep_check failure. Simulation of case when peer process is
         * terminated and PID is immediately reused by another process */

        if (has_mm()) {
            uct_mm_ep_t *ep = ucs_derived_of(ep0(), uct_mm_ep_t);
            ep->keepalive.start_time--;
        } else if (has_cuda_ipc()) {
#if HAVE_CUDA
            uct_cuda_ipc_ep_t *ep = ucs_derived_of(ep0(), uct_cuda_ipc_ep_t);
            ep->keepalive.start_time--;
#endif
        } else if (has_cma()) {
            uct_cma_ep_t *ep = ucs_derived_of(ep0(), uct_cma_ep_t);
            ep->keepalive.start_time--;
        }

        /* kill real receiver since DC uses shared KA DCI */
        test_uct_peer_failure::kill_receiver();
    }

protected:
    ucs::ptr_vector<ucs::scoped_setenv> m_env;
};

UCS_TEST_SKIP_COND_P(test_uct_peer_failure_keepalive, killed,
                     !check_caps(UCT_IFACE_FLAG_EP_CHECK))
{
    ucs_status_t status;

    scoped_log_handler slh(wrap_errors_logger);
    flush();
    EXPECT_EQ(0, m_err_count);

    status = uct_ep_check(ep0(), 0, NULL);
    ASSERT_UCS_OK(status);
    flush();

    /* allow keepalive requests to complete */
    short_progress_loop();

    /* we are still alive */
    EXPECT_EQ(0, m_err_count);

    kill_receiver();

    status = uct_ep_check(ep0(), 0, NULL);
    ASSERT_UCS_OK(status);
    flush();

    wait_for_flag(&m_err_count);
    EXPECT_EQ(1, m_err_count);
}

UCT_INSTANTIATE_NO_SELF_TEST_CASE(test_uct_peer_failure_keepalive)
_UCT_INSTANTIATE_TEST_CASE(test_uct_peer_failure_keepalive, cuda_ipc);


class test_uct_peer_failure_rma_zcopy : public test_uct_peer_failure
{
public:
    static const uint64_t INVALID_ADDRESS = UINT64_MAX;

    test_uct_peer_failure_rma_zcopy()
    {
        m_dummy_comp.func   = NULL;
        m_dummy_comp.count  = INT_MAX;
        m_dummy_comp.status = UCS_OK;
    }

    void test_rma_zcopy_peer_failure(bool is_put_op)
    {
        {
            scoped_log_handler slh(wrap_errors_logger);
            const size_t size = 128;
            mapped_buffer sendbuf(size, 0, *m_sender);
            ucs_status_t status;

            // Pretend peer failure by using invalid address
            if (is_put_op) {
                status = uct_ep_put_zcopy(m_sender->ep(0),
                                          sendbuf.iov(), 1,
                                          INVALID_ADDRESS, 0ul, &m_dummy_comp);
            } else {
                status = uct_ep_get_zcopy(m_sender->ep(0),
                                          sendbuf.iov(), 1,
                                          INVALID_ADDRESS, 0ul, &m_dummy_comp);
            }
            EXPECT_FALSE(UCS_STATUS_IS_ERR(status))
                << ucs_status_string(status);

            flush();
        }

        EXPECT_GT(m_err_count, 0ul);
    }

    uct_completion_t m_dummy_comp;
};

UCS_TEST_SKIP_COND_P(test_uct_peer_failure_rma_zcopy, put,
                     !check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE |
                                 UCT_IFACE_FLAG_PUT_ZCOPY))
{
    test_rma_zcopy_peer_failure(true);
}

UCS_TEST_SKIP_COND_P(test_uct_peer_failure_rma_zcopy, get,
                     !check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE |
                                 UCT_IFACE_FLAG_GET_ZCOPY))
{
    test_rma_zcopy_peer_failure(false);
}

_UCT_INSTANTIATE_TEST_CASE(test_uct_peer_failure_rma_zcopy, cma)

