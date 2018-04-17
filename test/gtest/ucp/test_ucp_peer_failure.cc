/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"
#include "ucp_datatype.h"

extern "C" {
#include <ucp/core/ucp_worker.h>
}


class test_ucp_peer_failure_base {
protected:
    enum {
        FAIL_AFTER_WIREUP = ucp_test::DEFAULT_PARAM_VARIANT,
        FAIL_IMMEDIATELY
    };

    test_ucp_peer_failure_base() {
        /* Set small TL timeouts to reduce testing time */
        m_env.push_back(new ucs::scoped_setenv("UCX_RC_TIMEOUT",     "10us"));
        m_env.push_back(new ucs::scoped_setenv("UCX_RC_RETRY_COUNT", "2"));
        std::string ud_timeout = ucs::to_string<int>(1 * ucs::test_time_multiplier()) + "s";
        m_env.push_back(new ucs::scoped_setenv("UCX_UD_TIMEOUT", ud_timeout.c_str()));
    }

    virtual ucp_ep_params_t get_ep_params() {
        ucp_ep_params_t params;
        memset(&params, 0, sizeof(params));
        params.field_mask      = UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLER;
        params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
        params.err_handler.cb  = err_cb;
        params.err_handler.arg = NULL;
        return params;
    }

    void init() {
        m_err_cntr   = 0;
        m_err_status = UCS_OK;
    }

    static void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
        EXPECT_EQ(UCS_ERR_ENDPOINT_TIMEOUT, status);
        m_err_status = status;
        ++m_err_cntr;
    }

protected:
    static size_t                       m_err_cntr;
    static ucs_status_t                 m_err_status;
    ucs::ptr_vector<ucs::scoped_setenv> m_env;
};

size_t       test_ucp_peer_failure_base::m_err_cntr   = 0;
ucs_status_t test_ucp_peer_failure_base::m_err_status = UCS_OK;


class test_ucp_peer_failure :
                    public test_ucp_tag,
                    protected test_ucp_peer_failure_base {
public:
    test_ucp_peer_failure() : m_msg_size(1024) {
    }

    static std::vector<ucp_test_param>
    enum_test_params(const ucp_params_t& ctx_params,
                     const std::string& name,
                     const std::string& test_case_name,
                     const std::string& tls)
    {
        std::vector<ucp_test_param> result =
            test_ucp_tag::enum_test_params(ctx_params, name, test_case_name, tls);

        generate_test_params_variant(ctx_params, name, test_case_name, tls,
                                     FAIL_AFTER_WIREUP, result);
        generate_test_params_variant(ctx_params, name, test_case_name, tls,
                                     FAIL_IMMEDIATELY, result);
        return result;
    }

    virtual void init();
    virtual void cleanup();

    void test_status_after(bool request_must_fail);
    void test_force_close();

protected:
    virtual ucp_ep_params_t get_ep_params() {
        return test_ucp_peer_failure_base::get_ep_params();
    }

    void fail_receiver() {
        /* TODO: need to handle non-empty TX window in UD EP destructor",
         *       see debug message (ud_ep.c:220)
         *       ucs_debug("ep=%p id=%d conn_id=%d has %d unacked packets",
         *                 self, self->ep_id, self->conn_id,
         *                 (int)ucs_queue_length(&self->tx.window));
         */
        flush_worker(receiver());
        m_entities.remove(&receiver());
    }

    void smoke_test() {
        long buf = 0;
        request *req = recv_nb(&buf, sizeof(buf), DATATYPE, 0, 0);
        send_b(&buf, sizeof(buf), DATATYPE, 0, 0);
        wait_and_validate(req);
    }

    void wait_err() {
        while (!m_err_cntr) {
            progress();
        }
    }

    static void err_cb_mod(void *arg, ucp_ep_h ep, ucs_status_t status) {
        EXPECT_EQ(uintptr_t(ucp::MAGIC), uintptr_t(arg));
        err_cb(arg, ep, status);
        m_err_cb_mod = true;
    }

protected:
    const size_t m_msg_size;
    static bool  m_err_cb_mod;
};

bool test_ucp_peer_failure::m_err_cb_mod = false;

void test_ucp_peer_failure::init() {
    m_err_cb_mod = false;

    test_ucp_peer_failure_base::init();
    test_ucp_tag::init();
    if (GetParam().variant != FAIL_IMMEDIATELY) {
        smoke_test();
    }

    /* Make second pair */
    create_entity(true);
    create_entity(false);
    sender().connect(&receiver(), get_ep_params());
    if (GetParam().variant != FAIL_IMMEDIATELY) {
        smoke_test();
    }
    wrap_errors();

    ucp_ep_params_t ep_params_mod = {0};
    ep_params_mod.field_mask = UCP_EP_PARAM_FIELD_ERR_HANDLER;
    ep_params_mod.err_handler.cb = err_cb_mod;
    /* NOTE: using of ucp_ep_params_t::user_data field is more preferable but
     *       need to test err_handler.arg as well */
    ep_params_mod.err_handler.arg = reinterpret_cast<void *>(uintptr_t(ucp::MAGIC));

    for (size_t i = 0; i < m_entities.size(); ++i) {
        for (int widx = 0; widx < e(i).get_num_workers(); ++widx) {
            for (int epidx = 0; epidx < e(i).get_num_eps(widx); ++epidx) {
                void *req = e(i).modify_ep(ep_params_mod, widx, epidx);
                ucp_test::wait(req, widx);
            }
        }
    }
}

void test_ucp_peer_failure::cleanup() {
    restore_errors();
    test_ucp_tag::cleanup();
}

void test_ucp_peer_failure::test_status_after(bool request_must_fail)
{
    fail_receiver();

    std::vector<uint8_t> buf(m_msg_size, 0);
    request *req = send_nb(buf.data(), buf.size(), DATATYPE, 0x111337);
    wait_err();
    EXPECT_NE(UCS_OK, m_err_status);
    EXPECT_TRUE(m_err_cb_mod);

    if (UCS_PTR_IS_PTR(req)) {
        /* The request may either succeed or fail, even though the data is not
         * delivered - depends on when the error is detected on sender side and
         * if zcopy/bcopy protocol is used. In any case, the request must
         * complete, and all resources have to be released.
         */
        EXPECT_TRUE(req->completed);
        if (request_must_fail) {
            EXPECT_EQ(m_err_status, req->status);
        } else {
            EXPECT_TRUE((m_err_status == req->status) || (UCS_OK == req->status));
        }
        request_release(req);
    }

    ucs_status_ptr_t status_ptr = ucp_tag_send_nb(sender().ep(), NULL, 0, DATATYPE,
                                                  0x111337, NULL);
    EXPECT_FALSE(UCS_PTR_IS_PTR(status_ptr));
    EXPECT_EQ(m_err_status, UCS_PTR_STATUS(status_ptr));

    /* Destroy failed sender */
    sender().destroy_worker();
    m_entities.remove(&sender());

    /* Check workability of second pair */
    smoke_test();
}

void test_ucp_peer_failure::test_force_close()
{
    const size_t            msg_size = 16000;
    const size_t            iter     = 1000;
    uint8_t                 *buf     = (uint8_t *)calloc(msg_size, iter);
    size_t                  allocd_eps_before, allocd_eps_after;
    std::vector<request *>  reqs;

    reqs.reserve(iter);
    for (size_t i = 0; i < iter; ++i) {
        request *sreq = send_nb(&buf[i * msg_size], msg_size, DATATYPE, 17);

        if (UCS_PTR_IS_PTR(sreq)) {
            reqs.push_back(sreq);
        } else if (UCS_PTR_IS_ERR(sreq)) {
            EXPECT_EQ(UCS_ERR_NO_RESOURCE, UCS_PTR_STATUS(sreq));
            break;
        }
    }

    fail_receiver();

    allocd_eps_before = ucs_strided_alloc_inuse_count(&sender().worker()->ep_alloc);

    request *close_req = (request *)ucp_ep_close_nb(sender().ep(),
                                                    UCP_EP_CLOSE_MODE_FORCE);
    if (UCS_PTR_IS_PTR(close_req)) {
        wait(close_req);
        ucp_request_release(close_req);
    } else {
        EXPECT_FALSE(UCS_PTR_IS_ERR(close_req));
    }

    allocd_eps_after = ucs_strided_alloc_inuse_count(&sender().worker()->ep_alloc);

    if (GetParam().variant != FAIL_IMMEDIATELY) {
        EXPECT_LT(allocd_eps_after, allocd_eps_before);
    }

    /* The EP can't be used now */
    sender().revoke_ep();

    while (!reqs.empty()) {
        EXPECT_NE(UCS_INPROGRESS, ucp_request_test(reqs.back(), NULL));
        ucp_request_release(reqs.back());
        reqs.pop_back();
    }

    /* Check that TX polling is working well */
    while (sender().progress());

    /* When all requests on sender are done we need to prevent LOCAL_FLUSH
     * in test teardown. Receiver is killed and doesn't respond on FC requests
     */
    sender().destroy_worker();
    free(buf);
}

UCS_TEST_P(test_ucp_peer_failure, disable_sync_send) {
    /* 1GB memory markup takes too long time with valgrind, reduce to 1MB */
    const size_t        max_size = RUNNING_ON_VALGRIND ? (1024 * 1024) :
                                   (1024 * 1024 * 1024);
    std::vector<char>   buf(max_size, 0);
    request             *req;

    /* Make sure API is disabled for any size and data type */
    for (size_t size = 1; size <= max_size; size *= 2) {
        req = send_sync_nb(buf.data(), size, DATATYPE, 0x111337);
        EXPECT_FALSE(UCS_PTR_IS_PTR(req));
        EXPECT_EQ(UCS_ERR_UNSUPPORTED, UCS_PTR_STATUS(req));

        ucp::data_type_desc_t dt_desc(DATATYPE_IOV, buf.data(), size);
        req = send_sync_nb(dt_desc.buf(), dt_desc.count(), dt_desc.dt(), 0x111337);
        EXPECT_FALSE(UCS_PTR_IS_PTR(req));
        EXPECT_EQ(UCS_ERR_UNSUPPORTED, UCS_PTR_STATUS(req));
    }
}

UCS_TEST_P(test_ucp_peer_failure, status_after_error) {
    test_status_after(false);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_peer_failure)


class test_ucp_peer_failure_zcopy : public test_ucp_peer_failure
{
public:
    virtual void init() {
        modify_config("ZCOPY_THRESH", ucs::to_string(m_msg_size - 1));
        test_ucp_peer_failure::init();
    }
};

UCS_TEST_P(test_ucp_peer_failure_zcopy, status_after_error) {
    test_status_after(true);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_peer_failure_zcopy)


class test_ucp_peer_failure_zcopy_multi : public test_ucp_peer_failure_zcopy
{
public:
    virtual void init() {
        /* MAX BCOPY is internally used as fragment size */
        m_env.push_back(new ucs::scoped_setenv("UCX_MAX_BCOPY",
                                               (ucs::to_string(m_msg_size/2) + "b").c_str()));
        /* HW TM does not support multiprotocols and eager protocol for messages
         * bigger than UCT segment size */
        m_env.push_back(new ucs::scoped_setenv("UCX_RC_TM_ENABLE", "n"));
        test_ucp_peer_failure_zcopy::init();
    }
};

UCS_TEST_P(test_ucp_peer_failure_zcopy_multi, status_after_error) {
    test_status_after(true);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_peer_failure_zcopy_multi)


class test_ucp_peer_failure_with_rma : public test_ucp_peer_failure {
public:
    enum {
        FAIL_ON_RMA = FAIL_IMMEDIATELY + 1
    };

    static std::vector<ucp_test_param>
    enum_test_params(const ucp_params_t& ctx_params,
                     const std::string& name,
                     const std::string& test_case_name,
                     const std::string& tls)
    {
        std::vector<ucp_test_param> result =
            test_ucp_peer_failure::enum_test_params(ctx_params, name,
                                                    test_case_name, tls);

        generate_test_params_variant(ctx_params, name, test_case_name, tls,
                                     FAIL_ON_RMA, result);
        return result;
    }

    static ucp_params_t get_ctx_params() {
        ucp_params_t params = test_ucp_tag::get_ctx_params();
        params.features |= UCP_FEATURE_RMA;
        return params;
    }
};

UCS_TEST_P(test_ucp_peer_failure_with_rma, status_after_error) {
    unsigned buf = 0;
    ucp_mem_map_params_t params;
    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    params.address = &buf;
    params.length = sizeof(buf);

    ucp_mem_h memh;
    ucs_status_t status = ucp_mem_map(receiver().ucph(), &params, &memh);
    ASSERT_UCS_OK(status);
    ucp_mem_attr_t mem_attr;
    mem_attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS;
    status = ucp_mem_query(memh, &mem_attr);
    ASSERT_UCS_OK(status);

    void *rkey_buffer;
    size_t rkey_buffer_size;
    status = ucp_rkey_pack(receiver().ucph(), memh, &rkey_buffer, &rkey_buffer_size);
    ASSERT_UCS_OK(status);
    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack(sender().ep(), rkey_buffer, &rkey);
    ASSERT_UCS_OK(status);
    ucp_rkey_buffer_release(rkey_buffer);
    status = ucp_mem_unmap(receiver().ucph(), memh);
    ASSERT_UCS_OK(status);

    fail_receiver();
    if (GetParam().variant == FAIL_ON_RMA) {
        ucp_get_nbi(sender().ep(), mem_attr.address, 1, (uintptr_t)&buf, rkey);
    } else {
        request *req = send_nb(NULL, 0, DATATYPE, 0x111337);
        if (UCS_PTR_IS_PTR(req)) {
            request_release(req);
        }
    }

    ucp_ep_flush(sender().ep());
    wait_err();

    EXPECT_NE(UCS_OK, m_err_status);

    ucs_status_ptr_t status_ptr = ucp_tag_send_nb(sender().ep(), NULL, 0, DATATYPE,
                                           0x111337, NULL);
    EXPECT_FALSE(UCS_PTR_IS_PTR(status_ptr));
    EXPECT_EQ(m_err_status, UCS_PTR_STATUS(status_ptr));

    status = ucp_put(sender().ep(), mem_attr.address, 1, (uintptr_t)&buf, rkey);
    EXPECT_FALSE(UCS_PTR_IS_PTR(status));
    EXPECT_EQ(m_err_status, status);
    ucp_rkey_destroy(rkey);

    /* Destroy failed sender */
    sender().destroy_worker();
    m_entities.remove(&sender());

    /* Check workability of second pair */
    smoke_test();
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_peer_failure_with_rma)

class test_ucp_peer_failure_2pairs :
                    public ucp_test,
                    protected test_ucp_peer_failure_base
{
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features     = UCP_FEATURE_TAG;
        return params;
    }

protected:
    virtual void init();
    virtual void cleanup();

    static void err_cb(void *arg, ucp_ep_h ep, ucs_status_t);
    ucp_worker_h rworker(int i);
    ucp_worker_h sworker();
    void progress();
    void wait_err();
    ucs_status_t wait_req(void *req);
    static void rcomplete_cb(void *req, ucs_status_t status,
                             ucp_tag_recv_info_t *info);
    static void scomplete_cb(void *req, ucs_status_t status);
    void smoke_test(size_t idx);

    static void ep_destructor(ucp_ep_h ep, test_ucp_peer_failure_2pairs* test) {
        test->wait_req(ucp_disconnect_nb(ep));
    }

    virtual ucp_ep_params_t get_ep_params() {
        return test_ucp_peer_failure_base::get_ep_params();
    }

    ucs::handle<ucp_context_h>               m_ucph;
    std::vector<ucs::handle<ucp_worker_h> >  m_workers;
    std::vector<ucs::handle<ucp_ep_h, test_ucp_peer_failure_2pairs*> > m_eps;
    ucs::ptr_vector<ucs::scoped_setenv>      m_env;
};

void test_ucp_peer_failure_2pairs::init()
{
    test_base::init(); /* skip entities creation */
    test_ucp_peer_failure_base::init();

    set_ucp_config(m_ucp_config);
    ucp_params_t cparams = get_ctx_params();
    UCS_TEST_CREATE_HANDLE(ucp_context_h, m_ucph, ucp_cleanup,
                           ucp_init, &cparams, m_ucp_config);

    m_workers.resize(3);
    for (int i = 0; i < 3; ++i) {
        ucp_worker_params_t wparams = get_worker_params();
        UCS_TEST_CREATE_HANDLE(ucp_worker_h, m_workers[i], ucp_worker_destroy,
                               ucp_worker_create, m_ucph, &wparams);
    }

    m_eps.resize(2);
    for (int i = 0; i < 2; ++i) {
        ucp_address_t *address;
        size_t address_length;
        ucs_status_t status;
        ucp_ep_h ep;

        status = ucp_worker_get_address(rworker(i), &address, &address_length);
        ASSERT_UCS_OK(status);

        ucp_ep_params ep_params = get_ep_params();
        ep_params.field_mask |= UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        ep_params.address     = address;

        hide_errors();
        status = ucp_ep_create(sworker(), &ep_params, &ep);
        restore_errors();

        ucp_worker_release_address(rworker(i), address);

        if (status == UCS_ERR_UNREACHABLE) {
            UCS_TEST_SKIP_R(m_errors.empty() ? "" : m_errors.back());
        }

        m_eps[i].reset(ep, ep_destructor, this);
    }

    /* Make sure wire up is done*/
    smoke_test(0);
    smoke_test(1);

    wrap_errors();
}

void test_ucp_peer_failure_2pairs::cleanup()
{
    restore_errors();
    m_eps.clear();
    m_workers.clear();
    test_base::cleanup();
}

void test_ucp_peer_failure_2pairs::err_cb(void *arg, ucp_ep_h ep, ucs_status_t) {
    test_ucp_peer_failure_2pairs *self;
    self = *reinterpret_cast<test_ucp_peer_failure_2pairs**>(arg);
    self->m_err_cntr++;
}

ucp_worker_h test_ucp_peer_failure_2pairs::rworker(int i)
{
    return m_workers[i];
}

ucp_worker_h test_ucp_peer_failure_2pairs::sworker()
{
    return m_workers[2];
}

void test_ucp_peer_failure_2pairs::progress()
{
    for (std::vector<ucs::handle<ucp_worker_h> >::iterator iter = m_workers.begin();
         iter != m_workers.end(); ++iter)
    {
        if (*iter) {
            ucp_worker_progress(*iter);
        }
    }
}

void test_ucp_peer_failure_2pairs::wait_err()
{
    while (!m_err_cntr) {
        progress();
    }
}

ucs_status_t test_ucp_peer_failure_2pairs::wait_req(void *req)
{
    if (req == NULL) {
        return UCS_OK;
    }

    ucs_assert(!!req);
    if (UCS_PTR_IS_ERR(req)) {
        return UCS_PTR_STATUS(req);
    }

    ucs_status_t status;
    do {
        progress();
        status = ucp_request_check_status(req);
    } while (status == UCS_INPROGRESS);
    ucp_request_release(req);
    return status;
}

void test_ucp_peer_failure_2pairs::rcomplete_cb(void *req, ucs_status_t status,
                                                ucp_tag_recv_info_t *info)
{
}

void test_ucp_peer_failure_2pairs::scomplete_cb(void *req, ucs_status_t status)
{
}

void test_ucp_peer_failure_2pairs::smoke_test(size_t idx)
{
    long buf = 0;
    void *rreq = ucp_tag_recv_nb(rworker(idx), &buf, 1,
                                 ucp_dt_make_contig(1), 0, 0,
                                 rcomplete_cb);
    void *sreq = ucp_tag_send_nb(m_eps[idx], &buf, 1,
                                 ucp_dt_make_contig(1), 0,
                                 scomplete_cb);
    wait_req(sreq);
    wait_req(rreq);
}

UCS_TEST_P(test_ucp_peer_failure_2pairs, status_after_error) {

    m_workers[0].reset();

    ucs_status_t status;
    void *sreq;
    unsigned buf = 0;

    do {
        sreq = ucp_tag_send_nb(m_eps[0], &buf, 1, ucp_dt_make_contig(1),
                               0x111337, scomplete_cb);
        status = wait_req(sreq);
    } while ((status == UCS_OK) || !m_err_cntr);

    wait_err();

    EXPECT_NE(UCS_OK, m_err_status);

    sreq = ucp_tag_send_nb(m_eps[0], NULL, 0, ucp_dt_make_contig(1), 0x111337,
                           scomplete_cb);
    EXPECT_FALSE(UCS_PTR_IS_PTR(sreq));
    EXPECT_EQ(m_err_status, UCS_PTR_STATUS(sreq));

    /* Destroy failed sender */
    m_eps[0].reset();

    /* Check workability of second pair */
    smoke_test(1);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_peer_failure_2pairs)

class test_ucp_ep_force_disconnect : public test_ucp_peer_failure
{
public:
    virtual void init() {
        m_env.clear(); /* restore default timeouts. */
        test_ucp_peer_failure::init();
    }
};

UCS_TEST_P(test_ucp_ep_force_disconnect, test) {
    test_force_close();
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_ep_force_disconnect)
