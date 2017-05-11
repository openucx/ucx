
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

class test_error_handling : public uct_test {
    friend class test_err_handling_destroy_ep_cb; /* Allow access to m_entities */

public:
    virtual void init();
    virtual void init_iface_params(uct_iface_params_t& params) const;

    static ucs_status_t am_dummy_handler(void *arg, void *data, size_t length,
                                         unsigned flags) {
        return UCS_OK;
    }

    static ucs_status_t pending_cb(uct_pending_req_t *self)
    {
        req_count++;
        return UCS_OK;
    }

    static void purge_cb(uct_pending_req_t *self, void *arg)
    {
        req_count++;
    }

    static void err_cb(void *arg, uct_ep_h ep, ucs_status_t status)
    {
        EXPECT_EQ(err_handler_arg,          arg);
        EXPECT_EQ(UCS_ERR_ENDPOINT_TIMEOUT, status);
        err_count++;
    }

    static void connect(entity *e1, entity *e2) {
        e1->connect(0, *e2, 0);
        e2->connect(0, *e1, 0);

        uct_iface_set_am_handler(e1->iface(), 0, am_dummy_handler,
                                 NULL, UCT_AM_CB_FLAG_ASYNC);
        uct_iface_set_am_handler(e2->iface(), 0, am_dummy_handler,
                                 NULL, UCT_AM_CB_FLAG_ASYNC);
    }

    void close_peer() {
        if (is_ep2ep_tl()) {
            m_entities.back()->destroy_ep(0);
        } else {
            m_entities.remove(m_entities.back());
            ucs_assert(m_entities.size() == 1);
        }
    }

    uct_ep_h ep() {
        return m_entities.front()->ep(0);
    }

    void flush() {
        return m_entities.front()->flush();
    }

protected:
    bool is_ep2ep_tl() const {
        return GetParam()->tl_name == "rc" || GetParam()->tl_name == "rc_mlx5";
    }

protected:
    static size_t err_count;
    static size_t req_count;
    static void* err_handler_arg;
};

void  *test_error_handling::err_handler_arg = (void *)0xdeadbeaf;
size_t test_error_handling::req_count       = 0ul;
size_t test_error_handling::err_count       = 0ul;

void test_error_handling::init()
{
    entity *e1, *e2;

    uct_test::init();

    set_config("LOG_LEVEL=fatal");

    /* To reduce test execution time decrease retransmition timeouts
     * where it is relevant */
    if (GetParam()->tl_name == "rc" || GetParam()->tl_name == "rc_mlx5" ||
        GetParam()->tl_name == "dc" || GetParam()->tl_name == "dc_mlx5") {
        set_config("RC_TIMEOUT=0.0001"); /* 100 us should be enough */
        set_config("RC_RETRY_COUNT=2");
    } else if (GetParam()->tl_name == "ud" || GetParam()->tl_name == "ud_mlx5") {
        set_config("UD_TIMEOUT=1s");
    }

    uct_iface_params_t params;
    init_iface_params(params);
    e1 = uct_test::create_entity(params);
    m_entities.push_back(e1);

    e2 = uct_test::create_entity(params);
    m_entities.push_back(e2);

    connect(e1, e2);
}

void test_error_handling::init_iface_params(uct_iface_params_t& params) const
{
    memset(&params, 0, sizeof(params));
    params.err_handler     = err_cb;
    params.err_handler_arg = err_handler_arg;
}

UCS_TEST_P(test_error_handling, peer_failure)
{
    check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE);

    err_count = 0ul;

    wrap_errors();

    close_peer();
    EXPECT_EQ(uct_ep_put_short(ep(), NULL, 0, 0, 0), UCS_OK);

    flush();

    restore_errors();

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, NULL, 0, NULL, 1);

    /* Check that all ep operations return pre-defined error code */
    EXPECT_EQ(uct_ep_am_short(ep(), 0, 0, NULL, 0), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_am_bcopy(ep(), 0, NULL, NULL), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_am_zcopy(ep(), 0, NULL, 0, iov, iovcnt, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_put_short(ep(), NULL, 0, 0, 0), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_put_bcopy(ep(), NULL, NULL, 0, 0), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_put_zcopy(ep(), iov, iovcnt, 0, 0, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_get_bcopy(ep(), NULL, NULL, 0, 0, 0, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_get_zcopy(ep(), iov, iovcnt, 0, 0, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_add64(ep(), 0, 0, 0), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_add32(ep(), 0, 0, 0), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_fadd64(ep(), 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_fadd32(ep(), 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_swap64(ep(), 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_swap32(ep(), 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_cswap64(ep(), 0, 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_atomic_cswap32(ep(), 0, 0, 0, 0, NULL, NULL),
              UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_flush(ep(), 0, NULL), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_get_address(ep(), NULL), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_pending_add(ep(), NULL), UCS_ERR_ENDPOINT_TIMEOUT);
    EXPECT_EQ(uct_ep_connect_to_ep(ep(), NULL, NULL), UCS_ERR_ENDPOINT_TIMEOUT);

    EXPECT_LT(0ul, err_count);
}

UCS_TEST_P(test_error_handling, purge_failed_peer)
{
    check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE);

    ucs_status_t      status;
    const size_t      num_pend_sends = 3ul;
    uct_pending_req_t reqs[num_pend_sends];

    req_count = 0ul;
    err_count = 0ul;

    wrap_errors();

    close_peer();

    do {
          status = uct_ep_put_short(ep(), NULL, 0, 0, 0);
    } while (status == UCS_OK);

    for (size_t i = 0; i < num_pend_sends; i ++) {
        reqs[i].func = pending_cb;
        EXPECT_EQ(uct_ep_pending_add(ep(), &reqs[i]), UCS_OK);
    }

    flush();

    EXPECT_EQ(uct_ep_am_short(ep(), 0, 0, NULL, 0), UCS_ERR_ENDPOINT_TIMEOUT);

    restore_errors();

    uct_ep_pending_purge(ep(), purge_cb, NULL);
    EXPECT_EQ(num_pend_sends, req_count);
    EXPECT_LT(0ul, err_count);
}

UCT_INSTANTIATE_TEST_CASE(test_error_handling)

class test_err_handling_destroy_ep_cb : public test_error_handling
{
public:
    virtual void init_iface_params(uct_iface_params_t& params) const {
        memset(&params, 0, sizeof(params));
        params.err_handler     = err_cb_ep_destroy;
        params.err_handler_arg = const_cast<test_err_handling_destroy_ep_cb*>(this);
    }

    static void err_cb_ep_destroy(void *arg, uct_ep_h ep, ucs_status_t status) {
        test_err_handling_destroy_ep_cb *self;
        self = reinterpret_cast<test_err_handling_destroy_ep_cb*>(arg);
        EXPECT_EQ(self->ep(), ep);
        self->m_entities.front()->destroy_ep(0);
    }
};

UCS_TEST_P(test_err_handling_destroy_ep_cb, desproy_ep_cb)
{
    check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE);

    wrap_errors();
    close_peer();
    EXPECT_EQ(uct_ep_put_short(ep(), NULL, 0, 0, 0), UCS_OK);

    flush();
    restore_errors();
}

UCT_INSTANTIATE_TEST_CASE(test_err_handling_destroy_ep_cb)
