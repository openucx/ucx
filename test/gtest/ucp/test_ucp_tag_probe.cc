/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"

#include <common/test_helpers.h>


class test_ucp_tag_probe : public test_ucp_tag {
public:
    using test_ucp_tag::get_ctx_params;
};


UCS_TEST_P(test_ucp_tag_probe, send_probe) {

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;
    ucp_tag_recv_info info;
    ucp_tag_message_h message;

    message = ucp_tag_probe_nb(receiver->worker(), 0x1337, 0xffff, 0, &info);
    EXPECT_TRUE(message == NULL);

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x111337);

    do {
        progress();
        message = ucp_tag_probe_nb(receiver->worker(), 0x1337, 0xffff, 0, &info);
    } while (message == NULL);

    EXPECT_EQ(sizeof(send_data),   info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);

    request *my_recv_req;
    my_recv_req = (request*)ucp_tag_recv_nb(receiver->worker(), &recv_data,
                                            sizeof(recv_data), DATATYPE, 0x1337,
                                            0xffff, recv_callback);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));

    wait(my_recv_req);
    EXPECT_TRUE(my_recv_req->completed);
    EXPECT_EQ(UCS_OK,              my_recv_req->status);
    EXPECT_EQ(sizeof(send_data),   my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_EQ(send_data, recv_data);
    request_release(my_recv_req);
}

UCS_TEST_P(test_ucp_tag_probe, send_medium_msg_probe) {
    static const size_t size = 50000;
    ucp_tag_recv_info info;
    ucp_tag_message_h message;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf.begin(), sendbuf.end());

    message = ucp_tag_probe_nb(receiver->worker(), 0x1337, 0xffff, 1, &info);
    EXPECT_TRUE(message == NULL);

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    short_progress_loop();

    message = ucp_tag_probe_nb(receiver->worker(), 0x1337, 0xffff, 1, &info);
    ASSERT_TRUE(message != NULL);
    EXPECT_EQ(sendbuf.size(),      info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);

    request *my_recv_req;
    my_recv_req = (request*)ucp_tag_msg_recv_nb(receiver->worker(), &recvbuf[0],
                                                recvbuf.size(), DATATYPE, message,
                                                recv_callback);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));

    wait(my_recv_req);
    EXPECT_TRUE(my_recv_req->completed);
    EXPECT_EQ(UCS_OK,              my_recv_req->status);
    EXPECT_EQ(sendbuf.size(),      my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);
    request_release(my_recv_req);
}

UCS_TEST_P(test_ucp_tag_probe, send_medium_msg_probe_truncated) {
    static const size_t size = 50000;
    ucp_tag_recv_info info;
    ucp_tag_message_h message;

    std::vector<char> sendbuf(size, 0);

    ucs::fill_random(sendbuf.begin(), sendbuf.end());

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    short_progress_loop();

    message = ucp_tag_probe_nb(receiver->worker(), 0x1337, 0xffff, 1, &info);
    ASSERT_TRUE(message != NULL);
    EXPECT_EQ(sendbuf.size(),      info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);

    request *my_recv_req;
    my_recv_req = (request*)ucp_tag_msg_recv_nb(receiver->worker(), NULL, 0,
                                                DATATYPE, message, recv_callback);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));

    wait(my_recv_req);

    EXPECT_TRUE(my_recv_req->completed);
    EXPECT_EQ(UCS_ERR_MESSAGE_TRUNCATED, my_recv_req->status);
    request_release(my_recv_req);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_probe)
