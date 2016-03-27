/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"

#include <common/test_helpers.h>

using namespace ucs; /* For vector<char> serialization */


class test_ucp_tag_match : public test_ucp_tag {
public:
    using test_ucp_tag::get_ctx_params;
};

UCS_TEST_P(test_ucp_tag_match, send_recv_exp) {
    ucp_tag_recv_info_t info;
    ucs_status_t status;

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x111337);

    /* No progress - goes to expected */

    status = recv_b(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sizeof(send_data),   info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(send_data, recv_data);
}

UCS_TEST_P(test_ucp_tag_match, send_recv_unexp) {
    ucp_tag_recv_info_t info;
    ucs_status_t status;

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

UCS_TEST_P(test_ucp_tag_match, send_recv_exp_medium) {
    static const size_t size = 50000;
    ucs_status_t status;
    ucp_tag_recv_info_t info;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf.begin(), sendbuf.end());

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    status = recv_b(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sendbuf.size(),      info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);
}

UCS_TEST_P(test_ucp_tag_match, send2_nb_recv_exp_medium) {
    static const size_t size = 50000;
    ucp_tag_recv_info_t info;
    ucs_status_t status;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    /* 1st send */

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    status = recv_b(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    /* 2nd send */

    ucs::fill_random(sendbuf.begin(), sendbuf.end());

    request *my_send_req;
    my_send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    status = recv_b(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sendbuf.size(),      info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);

    short_progress_loop();

    if (my_send_req != NULL) {
        EXPECT_TRUE(my_send_req->completed);
        EXPECT_EQ(UCS_OK, my_send_req->status);
        request_release(my_send_req);
    }
}

UCS_TEST_P(test_ucp_tag_match, send2_nb_recv_medium_wildcard) {
    static const size_t size = 3000000;

    entity *sender1 = create_entity();
    sender1->connect(receiver);

    entity *sender2 = create_entity();
    sender2->connect(receiver);

    for (int is_exp = 0; is_exp <= 1; ++is_exp) {

        UCS_TEST_MESSAGE << "Testing " << (is_exp ? "" : "un") << "expected mode, size " << size;

        std::vector<char> sendbuf1(size, 0);
        std::vector<char> sendbuf2(size, 0);
        std::vector<char> recvbuf1(size, 0);
        std::vector<char> recvbuf2(size, 0);

        ucs::fill_random(sendbuf1.begin(), sendbuf1.end());
        ucs::fill_random(sendbuf2.begin(), sendbuf2.end());

        /* Two sends with different tags */

        request *sreq1, *sreq2;
        sreq1 = (request*)ucp_tag_send_nb(sender1->ep(), &sendbuf1[0], sendbuf1.size(),
                                          DATATYPE, 1, send_callback);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(sreq1));

        sreq2 = (request*)ucp_tag_send_nb(sender2->ep(), &sendbuf2[0], sendbuf2.size(),
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
            EXPECT_TRUE(sreq1->completed);
            request_release(sreq1);
        }
        if (sreq2 != NULL) {
            EXPECT_TRUE(sreq2->completed);
            request_release(sreq2);
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

        request_release(rreq1);
        request_release(rreq2);
    }

    sender1->flush_worker();
    sender1->disconnect();

    sender2->flush_worker();
    sender2->disconnect();
}

UCS_TEST_P(test_ucp_tag_match, send_recv_nb_partial_exp_medium) {
    static const size_t size = 50000;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf.begin(), sendbuf.end());

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    usleep(1000);
    progress();

    request *my_recv_req;
    my_recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));

    wait(my_recv_req);

    EXPECT_EQ(sendbuf.size(),      my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);

    request_release(my_recv_req);
}

UCS_TEST_P(test_ucp_tag_match, send_nb_recv_unexp) {
    ucp_tag_recv_info_t info;
    ucs_status_t status;

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    request *my_send_req;
    my_send_req = send_nb(&send_data, sizeof(send_data), DATATYPE, 0x111337);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    ucp_worker_progress(receiver->worker());

    status = recv_b(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sizeof(send_data),   info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(send_data, recv_data);

    if (my_send_req != NULL) {
        EXPECT_TRUE(my_send_req->completed);
        EXPECT_EQ(UCS_OK, my_send_req->status);
        request_release(my_send_req);
    }
}

UCS_TEST_P(test_ucp_tag_match, send_recv_truncated) {
    ucp_tag_recv_info_t info;
    ucs_status_t status;

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
    request_release(my_recv_req);
}

UCS_TEST_P(test_ucp_tag_match, send_nb_multiple_recv_unexp) {
    const unsigned num_requests = 1000;
    ucp_tag_recv_info_t info;
    ucs_status_t status;

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    std::vector<request*> send_reqs(num_requests);

    for (unsigned i = 0; i < num_requests; ++i) {
        send_reqs[i] = send_nb(&send_data, sizeof(send_data), DATATYPE, 0x111337);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(send_reqs[i]));
    }

    ucp_worker_progress(receiver->worker());

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
            request_release(send_reqs[i]);
        }
    }
}

UCS_TEST_P(test_ucp_tag_match, sync_send_unexp) {
    ucp_tag_recv_info_t info;
    ucs_status_t status;

    uint64_t send_data = 0x0102030405060708;
    uint64_t recv_data = 0;

    request *my_send_req = send_sync_nb(&send_data, sizeof(send_data), DATATYPE,
                                        0x111337);
    short_progress_loop();

    ASSERT_TRUE(my_send_req != NULL);
    EXPECT_FALSE(my_send_req->completed);

    ucp_worker_progress(receiver->worker());

    status = recv_b(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sizeof(send_data),   info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(send_data, recv_data);

    short_progress_loop();

    EXPECT_TRUE(my_send_req->completed);
    EXPECT_EQ(UCS_OK, my_send_req->status);
    request_release(my_send_req);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_match)
