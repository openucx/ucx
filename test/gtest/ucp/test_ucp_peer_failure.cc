/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"

class test_ucp_peer_failure_base {
protected:
    test_ucp_peer_failure_base() {
        /* Set small TL timeouts to reduce testing time */
        m_env.push_back(new ucs::scoped_setenv("UCX_RC_TIMEOUT",     "10us"));
        m_env.push_back(new ucs::scoped_setenv("UCX_RC_RETRY_COUNT", "2"));
        std::string ud_timeout = ucs::to_string<int>(1 * ucs::test_time_multiplier()) + "s";
        m_env.push_back(new ucs::scoped_setenv("UCX_UD_TIMEOUT", ud_timeout.c_str()));
    }

    static ucp_ep_params_t get_ep_params() {
        ucp_ep_params_t params = test_ucp_tag::get_ep_params();
        params.field_mask     |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
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

    using test_ucp_peer_failure_base::get_ep_params;

    virtual void init();
    virtual void cleanup();

    void test_status_after();

protected:
    void fail_receiver() {
        /* TODO: need to handle non-empty TX window in UD EP destructor",
         *       see debug message (ud_ep.c:220)
         *       ucs_debug("ep=%p id=%d conn_id=%d has %d unacked packets",
         *                 self, self->ep_id, self->conn_id,
         *                 (int)ucs_queue_length(&self->tx.window));
         */
        receiver().flush_worker();
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

protected:
    const size_t m_msg_size;
};

void test_ucp_peer_failure::init() {
    test_ucp_peer_failure_base::init();
    test_ucp_tag::init();
    smoke_test();

    /* Make second pair */
    create_entity(true);
    create_entity(false);
    sender().connect(&receiver(), &GetParam().ep_params_cmn);
    smoke_test();
    wrap_errors();
}

void test_ucp_peer_failure::cleanup() {
    restore_errors();
    test_ucp_tag::cleanup();
}

void test_ucp_peer_failure::test_status_after()
{
    fail_receiver();

    std::vector<uint8_t> buf(m_msg_size, 0);
    request *req = send_nb(buf.data(), buf.size(), DATATYPE,
                                         0x111337);
    wait_err();
    EXPECT_NE(UCS_OK, m_err_status);
    if (UCS_PTR_IS_PTR(req)) {
        EXPECT_TRUE(req->completed);
        EXPECT_EQ(m_err_status, req->status);
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

        UCS_TEST_GET_BUFFER_DT_IOV(iov_, iov_cnt_, buf.data(), size, 40ul, 0);
        req = send_sync_nb(iov_, iov_cnt_, DATATYPE_IOV, 0x111337);
        EXPECT_FALSE(UCS_PTR_IS_PTR(req));
        EXPECT_EQ(UCS_ERR_UNSUPPORTED, UCS_PTR_STATUS(req));
    }
}

UCS_TEST_P(test_ucp_peer_failure, status_after_error) {
    test_status_after();
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
    test_status_after();
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_peer_failure_zcopy)


class test_ucp_peer_failure_zcopy_multi : public test_ucp_peer_failure_zcopy
{
public:
    virtual void init() {
        /* MAX BCOPY is internally used as fragment size */
        m_env.push_back(new ucs::scoped_setenv("UCX_MAX_BCOPY",
                                               (ucs::to_string(m_msg_size/2) + "b").c_str()));
        test_ucp_peer_failure_zcopy::init();
    }
};

UCS_TEST_P(test_ucp_peer_failure_zcopy_multi, status_after_error) {
    test_status_after();
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_peer_failure_zcopy_multi)


class test_ucp_peer_failure_with_rma : public test_ucp_peer_failure {
public:
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
    ucp_mem_unmap(receiver().ucph(), memh);

    fail_receiver();
    request *req = send_nb(NULL, 0, DATATYPE, 0x111337);
    wait_err();
    if (UCS_PTR_IS_PTR(req)) {
        request_release(req);
    }

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
    using test_ucp_peer_failure_base::get_ep_params;

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

    ucs::handle<ucp_context_h>               m_ucph;
    std::vector<ucs::handle<ucp_worker_h> >  m_workers;
    std::vector<ucs::handle<ucp_ep_h> >      m_eps;
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

        m_eps[i].reset(ep, ucp_ep_destroy);
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

    ucp_tag_recv_info info;
    ucs_status_t status;
    do {
        progress();
        status = ucp_request_test(req, &info);
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
