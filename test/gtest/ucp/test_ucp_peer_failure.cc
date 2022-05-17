/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>

#include "ucp_test.h"
#include "ucp_datatype.h"

extern "C" {
#include <ucp/core/ucp_ep.inl>    /* for testing EP RNDV configuration */
#include <ucp/core/ucp_request.h> /* for debug */
#include <ucp/core/ucp_worker.h>  /* for testing memory consumption */
}

class test_ucp_peer_failure : public ucp_test {
public:
    test_ucp_peer_failure();

    static void get_test_variants(std::vector<ucp_test_variant>& variants);

    ucp_ep_params_t get_ep_params();

protected:
    static const int AM_ID = 0;

    enum {
        TEST_AM  = UCS_BIT(0),
        TEST_RMA = UCS_BIT(1),
        FAIL_IMM = UCS_BIT(2),
        WAKEUP   = UCS_BIT(3)
    };

    enum {
        STABLE_EP_INDEX,
        FAILING_EP_INDEX
    };

    typedef ucs::handle<ucp_mem_h, ucp_context_h> mem_handle_t;

    void set_am_handler(entity &e);
    static ucs_status_t
    am_callback(void *arg, const void *header, size_t header_length, void *data,
                size_t length, const ucp_am_recv_param_t *param);
    static void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status);
    ucp_ep_h stable_sender();
    ucp_ep_h failing_sender();
    entity& stable_receiver();
    entity& failing_receiver();
    void *send_nb(ucp_ep_h ep, ucp_rkey_h rkey);
    static ucs_log_func_rc_t
    warn_unreleased_rdesc_handler(const char *file, unsigned line,
                                  const char *function,
                                  ucs_log_level_t level,
                                  const ucs_log_component_config_t *comp_conf,
                                  const char *message, va_list ap);
    void fail_receiver();
    void smoke_test(bool stable_pair);
    static void unmap_memh(ucp_mem_h memh, ucp_context_h context);
    void get_rkey(ucp_ep_h ep, entity& dst, mem_handle_t& memh,
                  ucs::handle<ucp_rkey_h>& rkey);
    void set_rkeys();
    static void send_cb(void *request, ucs_status_t status, void *user_data);

    virtual void cleanup();

    void do_test(size_t msg_size, int pre_msg_count, bool force_close,
                 bool request_must_fail);

    size_t                              m_am_rx_count;
    size_t                              m_err_count;
    ucs_status_t                        m_err_status;
    std::string                         m_sbuf, m_rbuf;
    mem_handle_t                        m_stable_memh, m_failing_memh;
    ucs::handle<ucp_rkey_h>             m_stable_rkey, m_failing_rkey;
};

UCP_INSTANTIATE_TEST_CASE(test_ucp_peer_failure)
// DC without UD auxiliary
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_peer_failure, dc_mlx5, "dc_mlx5")


test_ucp_peer_failure::test_ucp_peer_failure() :
    m_am_rx_count(0), m_err_count(0), m_err_status(UCS_OK)
{
    ucs::fill_random(m_sbuf);
    configure_peer_failure_settings();
}

void test_ucp_peer_failure::get_test_variants(
        std::vector<ucp_test_variant> &variants)
{
    add_variant_with_value(variants, UCP_FEATURE_AM, TEST_AM, "am");
    add_variant_with_value(variants, UCP_FEATURE_RMA, TEST_RMA, "rma");
    add_variant_with_value(variants, UCP_FEATURE_AM, TEST_AM | FAIL_IMM,
                           "am_fail_imm");
    add_variant_with_value(variants, UCP_FEATURE_RMA, TEST_RMA | FAIL_IMM,
                           "rma_fail_imm");
}

ucp_ep_params_t test_ucp_peer_failure::get_ep_params() {
    ucp_ep_params_t params;
    memset(&params, 0, sizeof(params));
    params.field_mask      = UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                             UCP_EP_PARAM_FIELD_ERR_HANDLER;
    params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
    params.err_handler.cb  = err_cb;
    params.err_handler.arg = reinterpret_cast<void*>(this);
    return params;
}

void test_ucp_peer_failure::set_am_handler(entity &e)
{
    if (!(get_variant_value() & TEST_AM)) {
        return;
    }

    ucp_am_handler_param_t param;
    param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                       UCP_AM_HANDLER_PARAM_FIELD_CB |
                       UCP_AM_HANDLER_PARAM_FIELD_ARG;
    param.cb         = am_callback;
    param.arg        = this;
    param.id         = AM_ID;

    ucs_status_t status = ucp_worker_set_am_recv_handler(e.worker(), &param);
    ASSERT_UCS_OK(status);
}

ucs_status_t
test_ucp_peer_failure::am_callback(void *arg, const void *header,
                                   size_t header_length, void *data,
                                   size_t length,
                                   const ucp_am_recv_param_t *param)
{
    test_ucp_peer_failure *self = reinterpret_cast<test_ucp_peer_failure*>(arg);
    ++self->m_am_rx_count;
    return UCS_OK;
}

void test_ucp_peer_failure::err_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
    test_ucp_peer_failure *self = reinterpret_cast<test_ucp_peer_failure*>(arg);
    EXPECT_TRUE((UCS_ERR_CONNECTION_RESET == status) ||
                (UCS_ERR_ENDPOINT_TIMEOUT == status));
    self->m_err_status = status;
    ++self->m_err_count;
}

/* stable pair: sender = ep(0), receiver: entity(size - 1)
 * failing pair: sender = ep(1), receiver: entity(size - 2)*/
ucp_ep_h test_ucp_peer_failure::stable_sender() {
    return sender().ep(0, STABLE_EP_INDEX);
}

ucp_ep_h test_ucp_peer_failure::failing_sender() {
    return sender().ep(0, FAILING_EP_INDEX);
}

ucp_test::entity& test_ucp_peer_failure::stable_receiver() {
    return m_entities.at(m_entities.size() - 1 - STABLE_EP_INDEX);
}

ucp_test::entity& test_ucp_peer_failure::failing_receiver() {
    return m_entities.at(m_entities.size() - 1 - FAILING_EP_INDEX);
}

void *test_ucp_peer_failure::send_nb(ucp_ep_h ep, ucp_rkey_h rkey)
{
    ucp_request_param_t param;
    param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE |
                         UCP_OP_ATTR_FIELD_CALLBACK;
    param.datatype     = DATATYPE;
    param.cb.send      = send_cb;
    if (get_variant_value() & TEST_AM) {
        return ucp_am_send_nbx(ep, AM_ID, NULL, 0, &m_sbuf[0], m_sbuf.size(),
                               &param);
    } else if (get_variant_value() & TEST_RMA) {
        return ucp_put_nbx(ep, &m_sbuf[0], m_sbuf.size(), (uintptr_t)&m_rbuf[0],
                           rkey, &param);
    } else {
        ucs_fatal("invalid test case");
    }
}

ucs_log_func_rc_t
test_ucp_peer_failure::warn_unreleased_rdesc_handler(const char *file, unsigned line,
                                                     const char *function,
                                                     ucs_log_level_t level,
                                                     const ucs_log_component_config_t *comp_conf,
                                                     const char *message, va_list ap)
{
    if (level == UCS_LOG_LEVEL_WARN) {
        std::string err_str = format_message(message, ap);

        if (err_str.find("unexpected tag-receive descriptor") != std::string::npos) {
            return UCS_LOG_FUNC_RC_STOP;
        }
    }

    return UCS_LOG_FUNC_RC_CONTINUE;
}

void test_ucp_peer_failure::fail_receiver() {
    /* TODO: need to handle non-empty TX window in UD EP destructor",
     *       see debug message (ud_ep.c:220)
     *       ucs_debug("ep=%p id=%d conn_id=%d has %d unacked packets",
     *                 self, self->ep_id, self->conn_id,
     *                 (int)ucs_queue_length(&self->tx.window));
     */
    // TODO use force-close to close connections
    flush_worker(failing_receiver());
    m_failing_memh.reset();
    {
        /* transform warning messages about unreleased TM rdescs to test
         * message that are expected here, since we closed receiver w/o
         * reading the messages that were potentially received */
        scoped_log_handler slh(warn_unreleased_rdesc_handler);
        failing_receiver().cleanup();
    }
}

void test_ucp_peer_failure::smoke_test(bool stable_pair)
{
    ucp_ep_h send_ep = stable_pair ? stable_sender() : failing_sender();
    size_t am_count  = m_am_rx_count;

    // Send and wait for completion
    void *sreq = send_nb(send_ep, stable_pair ? m_stable_rkey : m_failing_rkey);
    request_wait(sreq);

    if (get_variant_value() & TEST_AM) {
        // Wait for active message to be received
        while (m_am_rx_count < am_count) {
            progress();
        }
    } else if (get_variant_value() & TEST_RMA) {
        // Flush the sender and expect data to arrive on receiver
        void *freq = ucp_ep_flush_nb(send_ep, 0,
                                     (ucp_send_callback_t)ucs_empty_function);
        request_wait(freq);
        EXPECT_EQ(m_sbuf, m_rbuf);
    }
}

void test_ucp_peer_failure::unmap_memh(ucp_mem_h memh, ucp_context_h context)
{
    ucs_status_t status = ucp_mem_unmap(context, memh);
    if (status != UCS_OK) {
        ucs_warn("failed to unmap memory: %s", ucs_status_string(status));
    }
}

void test_ucp_peer_failure::get_rkey(ucp_ep_h ep, entity& dst, mem_handle_t& memh,
                                     ucs::handle<ucp_rkey_h>& rkey) {
    ucp_mem_map_params_t params;

    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    params.address    = &m_rbuf[0];
    params.length     = m_rbuf.size();

    ucp_mem_h ucp_memh;
    ucs_status_t status = ucp_mem_map(dst.ucph(), &params, &ucp_memh);
    ASSERT_UCS_OK(status);
    memh.reset(ucp_memh, unmap_memh, dst.ucph());

    void *rkey_buffer;
    size_t rkey_buffer_size;
    status = ucp_rkey_pack(dst.ucph(), memh, &rkey_buffer, &rkey_buffer_size);
    ASSERT_UCS_OK(status);

    ucp_rkey_h ucp_rkey;
    status = ucp_ep_rkey_unpack(ep, rkey_buffer, &ucp_rkey);
    ASSERT_UCS_OK(status);
    rkey.reset(ucp_rkey, ucp_rkey_destroy);

    ucp_rkey_buffer_release(rkey_buffer);
}

void test_ucp_peer_failure::set_rkeys() {

    if (get_variant_value() & TEST_RMA) {
        get_rkey(failing_sender(), failing_receiver(), m_failing_memh,
                 m_failing_rkey);
        get_rkey(stable_sender(), stable_receiver(), m_stable_memh,
                 m_stable_rkey);
    }
}

void test_ucp_peer_failure::send_cb(void *request, ucs_status_t status,
                                    void *user_data)
{
}

void test_ucp_peer_failure::cleanup() {
    m_failing_rkey.reset();
    m_stable_rkey.reset();
    m_failing_memh.reset();
    m_stable_memh.reset();
    ucp_test::cleanup();
}

void test_ucp_peer_failure::do_test(size_t msg_size, int pre_msg_count,
                                    bool force_close, bool request_must_fail)
{
    skip_loopback();

    m_sbuf.resize(msg_size);
    m_rbuf.resize(msg_size);

    /* connect 2 ep's from sender() to 2 receiver entities */
    create_entity();
    sender().connect(&stable_receiver(),  get_ep_params(), STABLE_EP_INDEX);
    sender().connect(&failing_receiver(), get_ep_params(), FAILING_EP_INDEX);
    set_am_handler(stable_receiver());
    set_am_handler(failing_receiver());

    set_rkeys();

    /* Since we don't want to test peer failure on a stable pair
     * and don't expect EP timeout error on those EPs,
     * run traffic on a stable pair to connect it */
    smoke_test(true);

    if (!(get_variant_value() & FAIL_IMM)) {
        /* if not fail immediately, run traffic on failing pair to connect it */
        smoke_test(false);
    }

    /* put some sends on the failing pair */
    std::vector<void*> sreqs_pre;
    for (int i = 0; i < pre_msg_count; ++i) {
        progress();
        void *req = send_nb(failing_sender(), m_failing_rkey);
        ASSERT_FALSE(UCS_PTR_IS_ERR(req));
        if (UCS_PTR_IS_PTR(req)) {
            sreqs_pre.push_back(req);
        }
    }

    flush_ep(sender(), 0, FAILING_EP_INDEX);
    EXPECT_EQ(UCS_OK, m_err_status);

    /* Since UCT/UD EP has a SW implementation of reliablity on which peer
     * failure mechanism is based, we should set small UCT/UD EP timeout
     * for UCT/UD EPs for sender's UCP EP to reduce testing time */
    double prev_ib_ud_peer_timeout = sender().set_ib_ud_peer_timeout(3.);

    {
        scoped_log_handler slh(wrap_errors_logger);

        fail_receiver();

        void *sreq = send_nb(failing_sender(), m_failing_rkey);
        flush_ep(sender(), 0, FAILING_EP_INDEX);
        while (!m_err_count) {
            progress();
        }
        EXPECT_NE(UCS_OK, m_err_status);

        if (UCS_PTR_IS_PTR(sreq)) {
            ucs_status_t status;
            /* If rendezvous protocol is used, the m_err_count is increased
             * on the receiver side, so the send request may not complete
             * immediately */
            status = request_wait(sreq);
            if (request_must_fail) {
                EXPECT_TRUE((m_err_status == status) ||
                            (UCS_ERR_CANCELED == status));
            } else {
                EXPECT_TRUE((m_err_status == status) || (UCS_OK == status));
            }
        }

        /* Additional sends must fail */
        void *sreq2         = send_nb(failing_sender(), m_failing_rkey);
        ucs_status_t status = request_wait(sreq2);
        EXPECT_TRUE(UCS_STATUS_IS_ERR(status));
        EXPECT_EQ(m_err_status, status);

        if (force_close) {
            unsigned allocd_eps_before =
                    ucs_strided_alloc_inuse_count(&sender().worker()->ep_alloc);

            ucp_ep_h ep = sender().revoke_ep(0, FAILING_EP_INDEX);

            m_failing_rkey.reset();

            void *creq = ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FORCE);
            request_wait(creq);
            short_progress_loop(); /* allow discard lanes & complete destroy EP */

            unsigned allocd_eps_after =
                    ucs_strided_alloc_inuse_count(&sender().worker()->ep_alloc);

            if (!(get_variant_value() & FAIL_IMM)) {
                EXPECT_LT(allocd_eps_after, allocd_eps_before);
            }
        }

        /* release requests */
        while (!sreqs_pre.empty()) {
            void *req = sreqs_pre.back();
            sreqs_pre.pop_back();
            EXPECT_NE(UCS_INPROGRESS, ucp_request_test(req, NULL));
            ucp_request_release(req);
        }
    }

    /* Since we won't test peer failure anymore, reset UCT/UD EP timeout to the
     * default value to avoid possible UD EP timeout errors under high load */
    sender().set_ib_ud_peer_timeout(prev_ib_ud_peer_timeout);

    /* Check workability of stable pair */
    smoke_test(true);

    /* Check that TX polling is working well */
    while (sender().progress());

    /* Destroy rkey for failing pair */
    m_failing_rkey.reset();
}

UCS_TEST_P(test_ucp_peer_failure, basic) {
    do_test(UCS_KBYTE, /* msg_size */
            0, /* pre_msg_cnt */
            false, /* force_close */
            false /* must_fail */);
}

UCS_TEST_P(test_ucp_peer_failure, zcopy, "ZCOPY_THRESH=1023",
           /* to catch failure with TCP during progressing multi AM Zcopy,
            * since `must_fail=true` */
           "TCP_SNDBUF?=1k", "TCP_RCVBUF?=128",
           "TCP_RX_SEG_SIZE?=512", "TCP_TX_SEG_SIZE?=256") {
    do_test(UCS_KBYTE, /* msg_size */
            0, /* pre_msg_cnt */
            false, /* force_close */
            true /* must_fail */);
}

UCS_TEST_P(test_ucp_peer_failure, bcopy_multi, "SEG_SIZE?=512", "RC_TM_ENABLE?=n") {
    do_test(UCS_KBYTE, /* msg_size */
            0, /* pre_msg_cnt */
            false, /* force_close */
            false /* must_fail */);
}

UCS_TEST_P(test_ucp_peer_failure, force_close, "RC_FC_ENABLE?=n",
           /* To catch unexpected descriptors leak, for multi-fragment protocol
              with TCP */
           "TCP_RX_SEG_SIZE?=1024", "TCP_TX_SEG_SIZE?=1024")
{
    do_test(16000, /* msg_size */
            1000, /* pre_msg_cnt */
            true, /* force_close */
            false /* must_fail */);
}

class test_ucp_peer_failure_keepalive : public test_ucp_peer_failure
{
public:
    test_ucp_peer_failure_keepalive() {
        m_sbuf.resize(1 * UCS_MBYTE);
        m_rbuf.resize(1 * UCS_MBYTE);

        m_env.push_back(new ucs::scoped_setenv("UCX_TCP_KEEPIDLE", "inf"));
        m_env.push_back(new ucs::scoped_setenv("UCX_UD_TIMEOUT", "3s"));
    }

    void init() {
        test_ucp_peer_failure::init();
        create_entity();
        sender().connect(&stable_receiver(), get_ep_params(), STABLE_EP_INDEX);
        sender().connect(&failing_receiver(), get_ep_params(), FAILING_EP_INDEX);
        stable_receiver().connect(&sender(), get_ep_params());
        failing_receiver().connect(&sender(), get_ep_params());
        set_am_handler(failing_receiver());
        set_am_handler(stable_receiver());
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant_with_value(variants, UCP_FEATURE_AM, TEST_AM, "am");
        add_variant_with_value(variants, UCP_FEATURE_AM | UCP_FEATURE_WAKEUP,
                               TEST_AM | WAKEUP, "am_wakeup");
    }

    void wakeup_drain_check_no_events(const std::vector<entity*> &entities)
    {
        ucs_time_t deadline = ucs::get_deadline();
        int ret;

        /* Read all possible wakeup events to make sure that no more events
         * arrive */
        do {
            progress(entities);
            ret = wait_for_wakeup(entities, 0);
        } while ((ret > 0) && (ucs_get_time() < deadline));

        EXPECT_EQ(ret, 0);
    }
};

UCS_TEST_P(test_ucp_peer_failure_keepalive, kill_receiver,
           "KEEPALIVE_INTERVAL=0.3", "KEEPALIVE_NUM_EPS=inf") {
    /* TODO: wireup is not tested yet */

    scoped_log_handler err_handler(wrap_errors_logger);
    scoped_log_handler warn_handler(hide_warns_logger);

    /* initiate p2p pairing */
    ucp_ep_resolve_remote_id(failing_sender(), 0);
    smoke_test(true); /* allow wireup to complete */
    smoke_test(false);

    if (ucp_ep_config(stable_sender())->key.keepalive_lane == UCP_NULL_LANE) {
        UCS_TEST_SKIP_R("Unsupported");
    }

    /* ensure both pair have ep_check map */
    ASSERT_NE(UCP_NULL_LANE,
              ucp_ep_config(failing_sender())->key.keepalive_lane);

    /* aux (ud) transport doesn't support keepalive feature and
     * we are assuming that wireup/connect procedure is done */

    EXPECT_EQ(0, m_err_count); /* ensure no errors are detected */

    /* flush all outstanding ops to allow keepalive to run */
    flush_worker(sender());
    if (get_variant_value() & WAKEUP) {
        check_events({ &sender(), &failing_receiver() }, true);

        /* make sure no remaining events are returned from poll() */
        wakeup_drain_check_no_events({ &sender(), &failing_receiver() });
    }

    /* kill EPs & ifaces */
    failing_receiver().close_all_eps(*this, 0, UCP_EP_CLOSE_MODE_FORCE);
    if (get_variant_value() & WAKEUP) {
        wakeup_drain_check_no_events({ &sender() });
    }
    wait_for_flag(&m_err_count);

    /* dump warnings */
    int warn_count = m_warnings.size();
    for (int i = 0; i < warn_count; ++i) {
        UCS_TEST_MESSAGE << "< " << m_warnings[i] << " >";
    }

    EXPECT_NE(0, m_err_count);

    ucp_ep_h ep = sender().revoke_ep(0, FAILING_EP_INDEX);
    void *creq = ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FORCE);
    request_wait(creq);

    /* make sure no remaining events are returned from poll() */
    if (get_variant_value() & WAKEUP) {
        wakeup_drain_check_no_events({ &sender() });
    }

    /* check if stable receiver is still works */
    m_err_count = 0;
    smoke_test(true);

    EXPECT_EQ(0, m_err_count); /* ensure no errors are detected */
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_peer_failure_keepalive)
