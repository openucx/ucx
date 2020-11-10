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
    enum {
        TEST_TAG = UCS_BIT(0),
        TEST_RMA = UCS_BIT(1),
        FAIL_IMM = UCS_BIT(2)
    };

    enum {
        STABLE_EP_INDEX,
        FAILING_EP_INDEX
    };

    typedef ucs::handle<ucp_mem_h, ucp_context_h> mem_handle_t;

    void set_timeouts();
    static void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status);
    ucp_ep_h stable_sender();
    ucp_ep_h failing_sender();
    entity& stable_receiver();
    entity& failing_receiver();
    void *send_nb(ucp_ep_h ep, ucp_rkey_h rkey);
    void *recv_nb(entity& e);
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
    static void send_cb(void *request, ucs_status_t status);
    static void recv_cb(void *request, ucs_status_t status,
                        ucp_tag_recv_info_t *info);

    virtual void cleanup();

    void do_test(size_t msg_size, int pre_msg_count, bool force_close,
                 bool request_must_fail);

    size_t                              m_err_count;
    ucs_status_t                        m_err_status;
    std::string                         m_sbuf, m_rbuf;
    mem_handle_t                        m_stable_memh, m_failing_memh;
    ucs::handle<ucp_rkey_h>             m_stable_rkey, m_failing_rkey;
    ucs::ptr_vector<ucs::scoped_setenv> m_env;
};

UCP_INSTANTIATE_TEST_CASE(test_ucp_peer_failure)


test_ucp_peer_failure::test_ucp_peer_failure() : m_err_count(0), m_err_status(UCS_OK) {
    ucs::fill_random(m_sbuf);
    set_timeouts();
}

void test_ucp_peer_failure::get_test_variants(std::vector<ucp_test_variant>& variants) {
    add_variant_with_value(variants, UCP_FEATURE_TAG, TEST_TAG, "tag");
    add_variant_with_value(variants, UCP_FEATURE_RMA, TEST_RMA, "rma");
    add_variant_with_value(variants, UCP_FEATURE_TAG, TEST_TAG | FAIL_IMM,
                           "tag_fail_imm");
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

void test_ucp_peer_failure::set_timeouts() {
    /* Set small TL timeouts to reduce testing time */
    m_env.push_back(new ucs::scoped_setenv("UCX_RC_TIMEOUT",     "10ms"));
    m_env.push_back(new ucs::scoped_setenv("UCX_RC_RNR_TIMEOUT", "10ms"));
    m_env.push_back(new ucs::scoped_setenv("UCX_RC_RETRY_COUNT", "2"));
}

void test_ucp_peer_failure::err_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
    test_ucp_peer_failure *self = reinterpret_cast<test_ucp_peer_failure*>(arg);
    EXPECT_EQ(UCS_ERR_ENDPOINT_TIMEOUT, status);
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

void *test_ucp_peer_failure::send_nb(ucp_ep_h ep, ucp_rkey_h rkey) {
    if (get_variant_value() & TEST_TAG) {
        return ucp_tag_send_nb(ep, &m_sbuf[0], m_sbuf.size(), DATATYPE, 0,
                               send_cb);
    } else if (get_variant_value() & TEST_RMA) {
        return ucp_put_nb(ep, &m_sbuf[0], m_sbuf.size(), (uintptr_t)&m_rbuf[0],
                          rkey, send_cb);
    } else {
        ucs_fatal("invalid test case");
    }
}

void *test_ucp_peer_failure::recv_nb(entity& e) {
    ucs_assert(m_rbuf.size() >= m_sbuf.size());
    if (get_variant_value() & TEST_TAG) {
        return ucp_tag_recv_nb(e.worker(), &m_rbuf[0], m_rbuf.size(), DATATYPE, 0,
                               0, recv_cb);
    } else if (get_variant_value() & TEST_RMA) {
        return NULL;
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

void test_ucp_peer_failure::smoke_test(bool stable_pair) {
    void *rreq = recv_nb(stable_pair ? stable_receiver() : failing_receiver());
    void *sreq = send_nb(stable_pair ? stable_sender()   : failing_sender(),
                         stable_pair ? m_stable_rkey     : m_failing_rkey);
    request_wait(sreq);
    request_wait(rreq);
    EXPECT_EQ(m_sbuf, m_rbuf);
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

void test_ucp_peer_failure::send_cb(void *request, ucs_status_t status)
{
}

void test_ucp_peer_failure::recv_cb(void *request, ucs_status_t status,
                                    ucp_tag_recv_info_t *info)
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

    EXPECT_EQ(UCS_OK, m_err_status);
    
    /* Since UCT/UD EP has a SW implementation of reliablity on which peer
     * failure mechanism is based, we should set small UCT/UD EP timeout
     * for UCT/UD EPs for sender's UCP EP to reduce testing time */
    double prev_ib_ud_timeout = sender().set_ib_ud_timeout(3.);

    {
        scoped_log_handler slh(wrap_errors_logger);

        fail_receiver();

        void *sreq = send_nb(failing_sender(), m_failing_rkey);

        while (!m_err_count) {
            progress();
        }
        EXPECT_NE(UCS_OK, m_err_status);

        if (UCS_PTR_IS_PTR(sreq)) {
            /* The request may either succeed or fail, even though the data is
             * not * delivered - depends on when the error is detected on sender
             * side and if zcopy/bcopy protocol is used. In any case, the
             * request must complete, and all resources have to be released.
             */
            ucs_status_t status = ucp_request_check_status(sreq);
            EXPECT_NE(UCS_INPROGRESS, status);
            if (request_must_fail) {
                EXPECT_EQ(m_err_status, status);
            } else {
                EXPECT_TRUE((m_err_status == status) || (UCS_OK == status));
            }
            ucp_request_release(sreq);
        }

        /* Additional sends must fail */
        void *sreq2 = send_nb(failing_sender(), m_failing_rkey);
        EXPECT_FALSE(UCS_PTR_IS_PTR(sreq2));
        EXPECT_EQ(m_err_status, UCS_PTR_STATUS(sreq2));

        if (force_close) {
            unsigned allocd_eps_before =
                    ucs_strided_alloc_inuse_count(&sender().worker()->ep_alloc);

            ucp_ep_h ep = sender().revoke_ep(0, FAILING_EP_INDEX);

            m_failing_rkey.reset();

            void *creq = ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FORCE);
            request_wait(creq);

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
    sender().set_ib_ud_timeout(prev_ib_ud_timeout);

    /* Check workability of stable pair */
    smoke_test(true);

    /* Check that TX polling is working well */
    while (sender().progress());

    /* Destroy rkeys before destroying the worker (which also destroys the
     * endpoints) */
    m_failing_rkey.reset();
    m_stable_rkey.reset();

    /* When all requests on sender are done we need to prevent LOCAL_FLUSH
     * in test teardown. Receiver is killed and doesn't respond on FC requests
     */
    sender().destroy_worker();
}

UCS_TEST_P(test_ucp_peer_failure, basic) {
    do_test(UCS_KBYTE, /* msg_size */
            0, /* pre_msg_cnt */
            false, /* force_close */
            false /* must_fail */);
}

UCS_TEST_P(test_ucp_peer_failure, rndv_disable) {
    const size_t size_max = std::numeric_limits<size_t>::max();

    sender().connect(&receiver(), get_ep_params(), STABLE_EP_INDEX);
    EXPECT_EQ(size_max, ucp_ep_config(sender().ep())->tag.rndv.am_thresh.remote);
    EXPECT_EQ(size_max, ucp_ep_config(sender().ep())->tag.rndv.rma_thresh.remote);
    EXPECT_EQ(size_max, ucp_ep_config(sender().ep())->tag.rndv.am_thresh.local);
    EXPECT_EQ(size_max, ucp_ep_config(sender().ep())->tag.rndv.rma_thresh.local);
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

UCS_TEST_P(test_ucp_peer_failure, force_close, "RC_FC_ENABLE?=n") {
    do_test(16000, /* msg_size */
            1000, /* pre_msg_cnt */
            true, /* force_close */
            false /* must_fail */);
}

UCS_TEST_SKIP_COND_P(test_ucp_peer_failure, disable_sync_send,
                     !(get_variant_value() & TEST_TAG)) {
    const size_t        max_size = UCS_MBYTE;
    std::vector<char>   buf(max_size, 0);
    void                *req;

    sender().connect(&receiver(), get_ep_params());

    /* Make sure API is disabled for any size and data type */
    for (size_t size = 1; size <= max_size; size *= 2) {
        req = ucp_tag_send_sync_nb(sender().ep(), buf.data(), size, DATATYPE,
                                   0x111337, NULL);
        EXPECT_FALSE(UCS_PTR_IS_PTR(req));
        EXPECT_EQ(UCS_ERR_UNSUPPORTED, UCS_PTR_STATUS(req));

        ucp::data_type_desc_t dt_desc(DATATYPE_IOV, buf.data(), size);
        req = ucp_tag_send_sync_nb(sender().ep(), dt_desc.buf(), dt_desc.count(),
                                   dt_desc.dt(), 0x111337, NULL);
        EXPECT_FALSE(UCS_PTR_IS_PTR(req));
        EXPECT_EQ(UCS_ERR_UNSUPPORTED, UCS_PTR_STATUS(req));
    }
}

class test_ucp_peer_failure_keepalive : public test_ucp_peer_failure
{
public:
    test_ucp_peer_failure_keepalive() {
        m_sbuf.resize(1 * UCS_MBYTE);
        m_rbuf.resize(1 * UCS_MBYTE);
    }

    void init() {
        test_ucp_peer_failure::init();
        create_entity();
        sender().connect(&stable_receiver(), get_ep_params(), STABLE_EP_INDEX);
        sender().connect(&failing_receiver(), get_ep_params(), FAILING_EP_INDEX);
        stable_receiver().connect(&sender(), get_ep_params());
        failing_receiver().connect(&sender(), get_ep_params());
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant_with_value(variants, UCP_FEATURE_TAG, TEST_TAG, "tag");
    }
};

UCS_TEST_P(test_ucp_peer_failure_keepalive, kill_receiver,
           "KEEPALIVE_TIMEOUT=0.3", "KEEPALIVE_NUM_EPS=inf") {
    /* TODO: wireup is not tested yet */

    scoped_log_handler err_handler(wrap_errors_logger);
    scoped_log_handler warn_handler(hide_warns_logger);

    smoke_test(true); /* allow wireup to complete */
    smoke_test(false);

    if (ucp_ep_config(stable_sender())->key.ep_check_map == 0) {
        UCS_TEST_SKIP_R("Unsupported");
    }

    /* ensure both pair have ep_check map */
    ASSERT_NE(0, ucp_ep_config(failing_sender())->key.ep_check_map);

    /* aux (ud) transport doesn't support keepalive feature and
     * we are assuming that wireup/connect procedure is done */

    EXPECT_EQ(0, m_err_count); /* ensure no errors are detected */

    /* flush all outstanding ops to allow keepalive to run */
    flush_worker(sender());

    /* kill EPs & ifaces */
    failing_receiver().close_all_eps(*this, 0, UCP_EP_CLOSE_MODE_FORCE);
    failing_receiver().destroy_worker(0);
    wait_for_flag(&m_err_count);

    /* dump warnings */
    int warn_count = m_warnings.size();
    for (int i = 0; i < warn_count; ++i) {
        UCS_TEST_MESSAGE << "< " << m_warnings[i] << " >";
    }

    EXPECT_NE(0, m_err_count);

    /* check if stable receiver is still works */
    m_err_count = 0;
    smoke_test(true);

    EXPECT_EQ(0, m_err_count); /* ensure no errors are detected */
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_peer_failure_keepalive)
