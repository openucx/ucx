/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"

#include <common/test_helpers.h>
extern "C" {
#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_types.h>
}

using namespace ucs; /* For vector<char> serialization */


class test_ucp_tag_match : public test_ucp_tag {
public:
    enum {
        ENABLE_PROTO = UCS_BIT(8)
    };

    test_ucp_tag_match() {
        if (RUNNING_ON_VALGRIND) {
            m_env.push_back(new ucs::scoped_setenv("UCX_RC_TM_SEG_SIZE", "8k"));
            m_env.push_back(new ucs::scoped_setenv("UCX_TCP_RX_SEG_SIZE", "8k"));
        }
    }

    virtual void init()
    {
        modify_config("TM_THRESH", "1");
        if (use_proto()) {
            modify_config("PROTO_ENABLE", "y");
            modify_config("MAX_EAGER_LANES", "2");
        } else {
            // TODO:
            // 1. test offload and offload MP as different variants
            // 2. Enable offload for new protocols as well when it is fully
            //    supported.
            enable_tag_mp_offload();
        }
        test_ucp_tag::init();
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        UCS_STATIC_ASSERT(!(ENABLE_PROTO & RECV_REQ_INTERNAL));
        UCS_STATIC_ASSERT(!(ENABLE_PROTO & RECV_REQ_EXTERNAL));

        add_variant_with_value(variants, get_ctx_params(), RECV_REQ_INTERNAL,
                               "req_int");
        add_variant_with_value(variants, get_ctx_params(), RECV_REQ_EXTERNAL,
                               "req_ext");
        add_variant_with_value(variants, get_ctx_params(),
                               RECV_REQ_INTERNAL | ENABLE_PROTO, "req_int_proto");
    }

    virtual bool is_external_request()
    {
        return get_variant_value() == RECV_REQ_EXTERNAL;
    }

protected:
    void test_iov(const size_t *iov_sizes, size_t iov_count);

    static void recv_callback_release_req(void *request, ucs_status_t status,
                                          ucp_tag_recv_info_t *info)
    {
        ucp_request_free(request);
        m_req_status = status;
    }

    bool use_proto() const
    {
        return get_variant_value() & ENABLE_PROTO;
    }

    static ucs_status_t m_req_status;
};

ucs_status_t test_ucp_tag_match::m_req_status = UCS_OK;


UCS_TEST_P(test_ucp_tag_match, send_recv_unexp) {
    ucp_tag_recv_info_t info;
    ucs_status_t        status;

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x111337);

    short_progress_loop(); /* Receive messages as unexpected */

    status = recv_b(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sizeof(send_data),   info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(send_data, recv_data);
}

UCS_TEST_SKIP_COND_P(test_ucp_tag_match, send_recv_unexp_rqfree,
                     /* request free cannot be used for external requests */
                     (get_variant_value() == RECV_REQ_EXTERNAL)) {
    request *my_recv_req;
    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    my_recv_req = recv_nb(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));

    request_free(my_recv_req);

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x1337);

    wait_for_flag(&recv_data);
    EXPECT_EQ(send_data, recv_data);
}

UCS_TEST_P(test_ucp_tag_match, send_recv_exp_medium) {
    static const size_t size = 50000;
    request *my_recv_req;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf);

    my_recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
    ASSERT_TRUE(my_recv_req != NULL); /* Couldn't be completed because didn't send yet */

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    wait(my_recv_req);

    EXPECT_EQ(sendbuf.size(),      my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);
    request_free(my_recv_req);
}

void test_ucp_tag_match::test_iov(const size_t *iov_sizes, size_t iov_count)
{
    std::stringstream ss;
    for (size_t i = 0; i < iov_count; ++i) {
        ss << iov_sizes[i] << " ";
    }
    UCS_TEST_MESSAGE << "{ " << ss.str() << "}";

    std::vector<std::string> sendbufs(iov_count);
    std::vector<std::string> recvbufs(iov_count);

    std::string send_flat_data;
    ucp_dt_iov_t send_iov[iov_count], recv_iov[iov_count];
    for (size_t i = 0; i < iov_count; ++i) {
        sendbufs[i].resize(iov_sizes[i]);
        ucs::fill_random(sendbufs[i]);
        send_iov[i].buffer = &sendbufs[i][0];
        send_iov[i].length = sendbufs[i].size();
        send_flat_data    += sendbufs[i];

        recvbufs[i].resize(iov_sizes[iov_count - i - 1], 'x');
        recv_iov[i].buffer = &recvbufs[i][0];
        recv_iov[i].length = recvbufs[i].size();
    }

    request *my_recv_req = recv_nb(recv_iov, iov_count, UCP_DATATYPE_IOV,
                                   0x1337, 0xffff);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));

    send_b(send_iov, iov_count, UCP_DATATYPE_IOV, 0x111337);
    wait(my_recv_req);

    EXPECT_EQ(send_flat_data.size(), my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    request_free(my_recv_req);

    /* Compare data */
    std::string recv_flat_data;
    for (size_t i = 0; i < iov_count; ++i) {
        recv_flat_data += recvbufs[i];
    }
    EXPECT_EQ(send_flat_data, recv_flat_data);
}

UCS_TEST_P(test_ucp_tag_match, send_recv_exp_iov, "RNDV_THRESH=inf")
{
    test_iov(NULL, 0);

    static const size_t iov_sizes0[] = {0};
    test_iov(iov_sizes0, ucs_static_array_size(iov_sizes0));

    static const size_t iov_sizes1[] = {1000, 3000, 10000};
    test_iov(iov_sizes1, ucs_static_array_size(iov_sizes1));

    static const size_t iov_sizes2[] = {1000, 0, 10000, 0};
    test_iov(iov_sizes2, ucs_static_array_size(iov_sizes2));

    static const size_t iov_sizes3[] = {0, 3000, 0, 10000, 0, 0, 0};
    test_iov(iov_sizes3, ucs_static_array_size(iov_sizes3));

    static const size_t iov_sizes4[] = {32, 16, 18, 15, 0, 0, 1, 78, 54, 198,
                                        234354, 1, 10, 100000, 0, 6};
    test_iov(iov_sizes4, ucs_static_array_size(iov_sizes4));
}

UCS_TEST_P(test_ucp_tag_match, send2_nb_recv_exp_medium) {
    static const size_t size = 50000;
    request user_data;
    request *ucx_req;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    request_init(&user_data);

    /* 1st send */

    ucx_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff,
                      &user_data);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(ucx_req));
    ASSERT_TRUE(ucx_req != NULL); /* Couldn't be completed because didn't send yet */

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    wait(ucx_req, &user_data);
    request_free(ucx_req);

    /* 2nd send */

    ucs::fill_random(sendbuf);
    request_init(&user_data);

    ucx_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff,
                      &user_data);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(ucx_req));
    ASSERT_TRUE(ucx_req != NULL); /* Couldn't be completed because didn't send yet */

    request *my_send_req;
    my_send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    wait(ucx_req, &user_data);

    EXPECT_EQ(sendbuf.size(),      user_data.info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, user_data.info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);

    short_progress_loop();

    if (my_send_req != NULL) {
        EXPECT_TRUE(my_send_req->completed);
        EXPECT_EQ(UCS_OK, my_send_req->status);
        request_free(my_send_req);
    }
    request_free(ucx_req);
}

UCS_TEST_P(test_ucp_tag_match, send2_nb_recv_medium_wildcard, "RNDV_THRESH=inf") {
    static const size_t size = 3000000;

    entity &sender2 = sender();
    create_entity(true);
    sender().connect(&receiver(), get_ep_params());

    for (int is_exp = 0; is_exp <= 1; ++is_exp) {

        UCS_TEST_MESSAGE << "Testing " << (is_exp ? "" : "un") << "expected mode, size " << size;

        std::vector<char> sendbuf1(size, 0);
        std::vector<char> sendbuf2(size, 0);
        std::vector<char> recvbuf1(size, 0);
        std::vector<char> recvbuf2(size, 0);

        ucs::fill_random(sendbuf1);
        ucs::fill_random(sendbuf2);

        /* Two sends with different tags */

        request *sreq1, *sreq2;
        sreq1 = (request*)ucp_tag_send_nb(sender().ep(), &sendbuf1[0], sendbuf1.size(),
                                          DATATYPE, 1, send_callback);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(sreq1));

        sreq2 = (request*)ucp_tag_send_nb(sender2.ep(), &sendbuf2[0], sendbuf2.size(),
                                          DATATYPE, 2, send_callback);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(sreq2));


        /* In unexpected mode, we progress all to put the messages on the
         *  unexpected queue
         */
        if (!is_exp) {
            short_progress_loop();
        }

        /* Two receives with any tag */

        request *rreq1, *rreq2;

        rreq1 = recv_nb(&recvbuf1[0], recvbuf1.size(), DATATYPE, 0, 0);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(rreq1));

        rreq2 = recv_nb(&recvbuf2[0], recvbuf2.size(), DATATYPE, 0, 0);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(rreq2));


        /* Wait for receives */
        wait(rreq1);
        wait(rreq2);

        short_progress_loop();

        /* Release sends */
        if (sreq1 != NULL) {
            wait(sreq1);
            EXPECT_TRUE(sreq1->completed);
            request_free(sreq1);
        }
        if (sreq2 != NULL) {
            wait(sreq2);
            EXPECT_TRUE(sreq2->completed);
            request_free(sreq2);
        }

        /* Receives should be completed with correct length */
        ASSERT_TRUE(rreq1->completed);
        ASSERT_TRUE(rreq2->completed);

        EXPECT_EQ(size, rreq1->info.length);
        EXPECT_EQ(size, rreq2->info.length);

        /* The order may be any, but the messages have to be received correctly */
        if (rreq1->info.sender_tag == 1u) {
            ASSERT_EQ(2u, rreq2->info.sender_tag);
            EXPECT_EQ(sendbuf1, recvbuf1);
            EXPECT_EQ(sendbuf2, recvbuf2);
        } else {
            ASSERT_EQ(2u, rreq1->info.sender_tag);
            ASSERT_EQ(1u, rreq2->info.sender_tag);
            EXPECT_EQ(sendbuf2, recvbuf1);
            EXPECT_EQ(sendbuf1, recvbuf2);
        }

        request_free(rreq1);
        request_free(rreq2);
    }
}

UCS_TEST_P(test_ucp_tag_match, send_recv_nb_partial_exp_medium) {
    static const size_t size = 50000;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf);

    request *my_recv_req;
    my_recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    usleep(1000);
    progress();

    wait(my_recv_req);

    EXPECT_EQ(sendbuf.size(),      my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);

    request_free(my_recv_req);
}

UCS_TEST_P(test_ucp_tag_match, send_nb_recv_unexp) {
    ucp_tag_recv_info_t info;
    ucs_status_t        status;

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    request *my_send_req;
    my_send_req = send_nb(&send_data, sizeof(send_data), DATATYPE, 0x111337);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    ucp_worker_progress(receiver().worker());

    status = recv_b(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sizeof(send_data),   info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(send_data, recv_data);

    if (my_send_req != NULL) {
        EXPECT_TRUE(my_send_req->completed);
        EXPECT_EQ(UCS_OK, my_send_req->status);
        request_free(my_send_req);
    }
}

UCS_TEST_P(test_ucp_tag_match, send_recv_cb_release) {

    uint64_t send_data = 0xdeadbeefdeadbeef;

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x111337);

    short_progress_loop(); /* Receive messages as unexpected */

    m_req_status = UCS_INPROGRESS;

    uint64_t recv_data;
    request *recv_req = (request*)ucp_tag_recv_nb(receiver().worker(), &recv_data,
                                                  sizeof(recv_data), DATATYPE, 0, 0,
                                                  recv_callback_release_req);
    if (UCS_PTR_IS_ERR(recv_req)) {
        ASSERT_UCS_OK(UCS_PTR_STATUS(recv_req));
    } else if (recv_req == NULL) {
        UCS_TEST_ABORT("ucp_tag_recv_nb returned NULL");
    } else {
        /* request would be completed and released by the callback */
        while (m_req_status == UCS_INPROGRESS) {
            progress();
        }
        ASSERT_UCS_OK(m_req_status);
    }
}

UCS_TEST_P(test_ucp_tag_match, send_recv_truncated) {
    ucp_tag_recv_info_t info;
    ucs_status_t        status;

    uint64_t send_data = 0xdeadbeefdeadbeef;

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x111337);

    short_progress_loop(); /* Receive messages as unexpected */

    status = recv_b(NULL, 0, DATATYPE, 0x1337, 0xffff, &info);
    EXPECT_EQ(UCS_ERR_MESSAGE_TRUNCATED, status);
}

UCS_TEST_P(test_ucp_tag_match, send_recv_nb_exp) {

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    request *my_recv_req;
    my_recv_req = recv_nb(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff);

    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
    ASSERT_TRUE(my_recv_req != NULL); /* Couldn't be completed because didn't send yet */

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x111337);

    wait(my_recv_req);

    EXPECT_TRUE(my_recv_req->completed);
    EXPECT_EQ(UCS_OK,              my_recv_req->status);
    EXPECT_EQ(sizeof(send_data),   my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_EQ(send_data, recv_data);
    request_free(my_recv_req);
}

UCS_TEST_P(test_ucp_tag_match, send_nb_multiple_recv_unexp) {
    const unsigned      num_requests = 1000;
    ucp_tag_recv_info_t info;
    ucs_status_t        status;

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    std::vector<request*> send_reqs(num_requests);

    skip_loopback();

    for (unsigned i = 0; i < num_requests; ++i) {
        send_reqs[i] = send_nb(&send_data, sizeof(send_data), DATATYPE, 0x111337);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(send_reqs[i]));
    }

    ucp_worker_progress(receiver().worker());

    for (unsigned i = 0; i < num_requests; ++i) {
        status = recv_b(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff,
                        &info);
        ASSERT_UCS_OK(status);
        ASSERT_EQ(num_requests, send_reqs.size());

        EXPECT_EQ(sizeof(send_data),   info.length);
        EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
        EXPECT_EQ(send_data, recv_data);
    }

    for (unsigned i = 0; i < num_requests; ++i) {
        if (send_reqs[i] != NULL) {
            EXPECT_TRUE(send_reqs[i]->completed);
            EXPECT_EQ(UCS_OK, send_reqs[i]->status);
            request_free(send_reqs[i]);
        }
    }
}

UCS_TEST_P(test_ucp_tag_match, sync_send_unexp) {
    ucp_tag_recv_info_t info;
    ucs_status_t        status;

    uint64_t send_data = 0x0102030405060708;
    uint64_t recv_data = 0;

    request *my_send_req = send_sync_nb(&send_data, sizeof(send_data), DATATYPE,
                                        0x111337);
    short_progress_loop();

    ASSERT_TRUE(my_send_req != NULL);
    EXPECT_FALSE(my_send_req->completed);

    ucp_worker_progress(receiver().worker());

    status = recv_b(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sizeof(send_data),   info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(send_data, recv_data);

    wait_for_flag(&my_send_req->completed);

    EXPECT_TRUE(my_send_req->completed);
    EXPECT_EQ(UCS_OK, my_send_req->status);
    request_free(my_send_req);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_match)

class test_ucp_tag_match_rndv : public test_ucp_tag_match {
public:
    enum {
        RNDV_SCHEME_AUTO = 0,
        RNDV_SCHEME_PUT_ZCOPY,
        RNDV_SCHEME_GET_ZCOPY,
        RNDV_SCHEME_LAST,
        PUT_ZCOPY_FLUSH = ENABLE_PROTO << 1
    };

    static const std::string rndv_schemes[];

    void init() {
        ASSERT_LE(rndv_scheme(), (int)RNDV_SCHEME_GET_ZCOPY);
        UCS_STATIC_ASSERT(!(ENABLE_PROTO & UCS_MASK(RNDV_SCHEME_LAST)));
        modify_config("RNDV_THRESH", "0");
        modify_config("RNDV_SCHEME", rndv_schemes[rndv_scheme()]);
        modify_config("RNDV_PUT_FORCE_FLUSH", force_flush() ? "y" : "n");
        test_ucp_tag_match::init();
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        for (int rndv_scheme = 0; rndv_scheme < RNDV_SCHEME_LAST; ++rndv_scheme) {
            add_variant_with_value(variants, get_ctx_params(), rndv_scheme,
                                   "rndv_" + rndv_schemes[rndv_scheme]);
            add_variant_with_value(variants, get_ctx_params(),
                                   rndv_scheme | ENABLE_PROTO,
                                   rndv_schemes[rndv_scheme] + ",proto");
        }

        // Add variant with force flush
        add_variant_with_value(variants, get_ctx_params(),
                               RNDV_SCHEME_PUT_ZCOPY | ENABLE_PROTO |
                                       PUT_ZCOPY_FLUSH,
                               "rndv_put_flush,proto");
    }

protected:
    int rndv_scheme() const
    {
        int mask = ucs_roundup_pow2(static_cast<int>(RNDV_SCHEME_LAST) + 1) - 1;
        ucs_assert(!(mask & ENABLE_PROTO));
        return get_variant_value() & mask;
    }

    bool force_flush() const
    {
        return get_variant_value() & PUT_ZCOPY_FLUSH;
    }
};

const std::string test_ucp_tag_match_rndv::rndv_schemes[] = { "auto",
                                                              "put_zcopy",
                                                              "get_zcopy" };

UCS_TEST_P(test_ucp_tag_match_rndv, length0)
{
    request *my_send_req = send_nb((void*)0xdeadbeef, 0, DATATYPE, 1);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    ucp_tag_recv_info_t info;
    ucs_status_t status = recv_b((void*)0xbadc0fee, 0, DATATYPE, 1, 0, &info);
    EXPECT_EQ(UCS_OK, status);

    wait_and_validate(my_send_req);
}

UCS_TEST_P(test_ucp_tag_match_rndv, sync_send_unexp)
{
    static const size_t size = 1148576;
    request             *my_send_req;
    ucp_tag_recv_info_t info;
    ucs_status_t        status;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf);

    /* sender - send the rts*/
    my_send_req = send_sync_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
    /* receiver - get the rts and put in unexpected */
    short_progress_loop();

    ASSERT_TRUE(my_send_req != NULL);
    EXPECT_FALSE(my_send_req->completed);

    /* receiver - issue a recv req, match the rts, perform rndv-get and send ats to sender */
    status = recv_b(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sendbuf.size(), info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);

    /* sender - get the ATS and set send request to completed */
    wait_for_flag(&my_send_req->completed);

    EXPECT_TRUE(my_send_req->completed);
    EXPECT_EQ(UCS_OK, my_send_req->status);
    request_free(my_send_req);
}

UCS_TEST_P(test_ucp_tag_match_rndv, req_exp)
{
    static const size_t size = 1148576;
    request *my_send_req, *my_recv_req;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    skip_loopback();

    ucs::fill_random(sendbuf);

    /* receiver - put the receive request into expected */
    my_recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
    EXPECT_FALSE(my_recv_req->completed);

    /* sender - send the RTS */
    my_send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    /* receiver - match the rts, perform rndv get and send an ack upon finishing */
    short_progress_loop();
    /* for UCTs that cannot perform real rndv and may do eager send-recv bcopy instead */
    wait(my_recv_req);

    EXPECT_EQ(sendbuf.size(),      my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_TRUE(my_recv_req->completed);
    EXPECT_EQ(sendbuf, recvbuf);

    wait_and_validate(my_send_req);
    request_free(my_recv_req);
}

UCS_TEST_P(test_ucp_tag_match_rndv, rts_unexp)
{
    static const size_t size = 1148576;
    request             *my_send_req;
    ucp_tag_recv_info_t info;
    ucs_status_t        status;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    skip_loopback();

    ucs::fill_random(sendbuf);

    /* sender - send the RTS */
    my_send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    /* receiver - get the RTS and put it into unexpected */
    short_progress_loop();

    /* receiver - issue a receive request, match it with the RTS and perform rndv get */
    status = recv_b(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    /* sender - get the ATS and set send request to completed */
    wait_and_validate(my_send_req);

    EXPECT_EQ(sendbuf.size()     , info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);
}

UCS_TEST_P(test_ucp_tag_match_rndv, truncated)
{
    static const size_t size = 1148576;
    request *my_send_req;
    ucp_tag_recv_info_t info;
    ucs_status_t status;

    std::vector<char> sendbuf(size, 0);

    skip_loopback();

    ucs::fill_random(sendbuf);

    /* sender - send the RTS */
    my_send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    /* receiver - get the RTS and put it into unexpected */
    short_progress_loop();

    /* receiver - issue a receive request with zero length,
     * no assertions should occur */
    status = recv_b(NULL, 0, DATATYPE, 0x1337, 0xffff, &info);
    EXPECT_EQ(UCS_ERR_MESSAGE_TRUNCATED, status);

    /* sender - get the ATS and set send request to completed */
    wait_and_validate(my_send_req);
}

UCS_TEST_P(test_ucp_tag_match_rndv, post_larger_recv)
{
    /* small send size should probably be lower than minimum GET Zcopy
     * size supported by IB TLs */
    static const size_t small_send_size = 16;
    static const size_t small_recv_size = small_send_size * 2;
    static const size_t large_send_size = 1148576;
    static const size_t large_recv_size = large_send_size + 1 * UCS_KBYTE;
    /* array of [send][recv] sizes */
    static const size_t sizes[][2] = { { small_send_size, small_recv_size },
                                       { large_send_size, large_recv_size } };
    request *my_send_req, *my_recv_req;

    for (unsigned i = 0; i < ucs_static_array_size(sizes); i++) {
        size_t send_size = sizes[i][0];
        size_t recv_size = sizes[i][1];
        std::vector<char> sendbuf(send_size, 0);
        std::vector<char> recvbuf(recv_size, 0);

        ucs::fill_random(sendbuf);
        ucs::fill_random(recvbuf);

        my_recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
        EXPECT_FALSE(my_recv_req->completed);

        my_send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

        wait(my_recv_req);

        EXPECT_EQ(sendbuf.size(), my_recv_req->info.length);
        EXPECT_EQ(recvbuf.size(), ((ucp_request_t*)my_recv_req - 1)->recv.length);
        EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
        EXPECT_TRUE(my_recv_req->completed);
        EXPECT_NE(sendbuf, recvbuf);
        EXPECT_TRUE(std::equal(sendbuf.begin(), sendbuf.end(), recvbuf.begin()));

        wait_and_validate(my_send_req);
        request_free(my_recv_req);
    }
}

UCS_TEST_P(test_ucp_tag_match_rndv, req_exp_auto_thresh, "RNDV_THRESH=auto") {
    static const size_t size = 1148576;
    request *my_send_req, *my_recv_req;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    skip_loopback();

    ucs::fill_random(sendbuf);

    /* receiver - put the receive request into expected */
    my_recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
    EXPECT_FALSE(my_recv_req->completed);

    /* sender - send the RTS */
    my_send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    /* receiver - match the rts, perform rndv get and send an ack upon finishing */
    short_progress_loop();
    /* for UCTs that cannot perform real rndv and may do eager send-recv bcopy instead */
    wait(my_recv_req);

    EXPECT_EQ(sendbuf.size(),      my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_TRUE(my_recv_req->completed);
    EXPECT_EQ(sendbuf, recvbuf);

    /* sender - get the ATS and set send request to completed */
    wait_and_validate(my_send_req);
    request_free(my_recv_req);
}

UCS_TEST_P(test_ucp_tag_match_rndv, exp_huge_mix) {
    const std::vector<size_t> sizes = {1000, 2000, 8000,
                                       ucs::limit_buffer_size(2500ul *
                                                              UCS_MBYTE),
                                       ucs::limit_buffer_size(UCS_GBYTE + 32)};

    /* small sizes should warm-up tag cache */
    for (auto c_size : sizes) {
        const size_t size = c_size / ucs::test_time_multiplier() /
                            ucs::test_time_multiplier();
        request *my_send_req, *my_recv_req;

        std::vector<char> sendbuf(size, 0);
        std::vector<char> recvbuf(size, 0);

        ucs::fill_random(sendbuf);
        VALGRIND_MAKE_MEM_UNDEFINED(&recvbuf[0], recvbuf.size());

        my_recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
        EXPECT_FALSE(my_recv_req->completed);

        my_send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

        wait(my_recv_req);

        EXPECT_EQ(sendbuf.size(),      my_recv_req->info.length);
        EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
        EXPECT_TRUE(my_recv_req->completed);
        EXPECT_EQ(sendbuf, recvbuf);

        wait_and_validate(my_send_req);
        request_free(my_recv_req);
    }
}

UCS_TEST_P(test_ucp_tag_match_rndv, bidir_multi_exp_post)
{
    const size_t sizes[] = {8 * UCS_KBYTE, 128 * UCS_KBYTE, 512 * UCS_KBYTE,
                            8 * UCS_MBYTE, 128 * UCS_MBYTE, 512 * UCS_MBYTE};
    const size_t max_total_size = limit_buffer_size();

    receiver().connect(&sender(), get_ep_params());

    for (unsigned i = 0; i < ucs_static_array_size(sizes); ++i) {
        const size_t size = sizes[i] /
                            ucs::test_time_multiplier() /
                            ucs::test_time_multiplier();
        const size_t count = ucs_max((size_t)(5000.0 / sqrt(sizes[i]) /
                                              ucs::test_time_multiplier()), 3lu);

        size_t total_size = size * count * 2;
        if (total_size > max_total_size) {
            UCS_TEST_MESSAGE << "Total size (" << total_size
                             << ") exceeds limit (" << max_total_size
                             << "), stopping the test";
            break;
        }

        std::vector<request*> sreqs;
        std::vector<request*> rreqs;
        std::vector<std::vector<char> > sbufs;
        std::vector<std::vector<char> > rbufs;

        sbufs.resize(count * 2);
        rbufs.resize(count * 2);

        for (size_t repeat = 0; repeat < count * 2; ++repeat) {
            entity &send_e = repeat < count ? sender() : receiver();
            entity &recv_e = repeat < count ? receiver() : sender();
            request *my_send_req, *my_recv_req;

            sbufs[repeat].resize(size, 0);
            rbufs[repeat].resize(size, 0);
            ucs::fill_random(sbufs[repeat]);

            my_recv_req = recv(recv_e, RECV_NB,
                               &rbufs[repeat][0], rbufs[repeat].size(),
                               DATATYPE, 0x1337, 0xffff, NULL);
            ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
            EXPECT_FALSE(my_recv_req->completed);

            my_send_req = send(send_e, SEND_NB,
                               &sbufs[repeat][0], sbufs[repeat].size(),
                               DATATYPE, 0x111337);
            ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

            sreqs.push_back(my_send_req);
            rreqs.push_back(my_recv_req);
        }

        for (size_t repeat = 0; repeat < count * 2; ++repeat) {
            request *my_send_req, *my_recv_req;

            my_recv_req = rreqs[repeat];
            my_send_req = sreqs[repeat];

            wait(my_recv_req);

            EXPECT_EQ(sbufs[repeat].size(), my_recv_req->info.length);
            EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
            EXPECT_TRUE(my_recv_req->completed);
            EXPECT_EQ(sbufs[repeat], rbufs[repeat]);

            wait_and_validate(my_send_req);
            request_free(my_recv_req);
        }
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_match_rndv)
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_tag_match_rndv, mm_tcp, "posix,sysv,tcp")
