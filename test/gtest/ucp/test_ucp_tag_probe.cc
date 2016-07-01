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

    /* The parameters mean the following:
     *  - s_size and r_size: send and recv buffer sizes.
     *    Can be different for checking message trancation error
     *  - is_sync: specifies the type of send function to be used
     *    (sync or not)
     *  - is_recv_msg: specifies whether probe function needs to remove
     *    matched message. If yes, then ucp_tag_msg_recv_nb is used for
     *    recieve
     * */
    void test_send_probe (size_t s_size, size_t r_size, bool is_sync,
                          int is_recv_msg) {
        ucp_tag_recv_info info;
        ucp_tag_message_h message;
        request *send_req = NULL, *recv_req = NULL;

        std::vector<char> sendbuf(s_size, 0);
        std::vector<char> recvbuf(r_size, 0);

        ucs::fill_random(sendbuf.begin(), sendbuf.end());

        message = ucp_tag_probe_nb(receiver->worker(), 0x1337, 0xffff,
                                   is_recv_msg, &info);
        EXPECT_TRUE(message == NULL);

        if (is_sync) {
            send_req = send_sync_nb(&sendbuf[0], sendbuf.size(), DATATYPE,
                                    0x111337);

        } else {
            send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
        }

        do {
            progress();
            message = ucp_tag_probe_nb(receiver->worker(), 0x1337, 0xffff,
                                       is_recv_msg, &info);
        } while (message == NULL);

        EXPECT_EQ(sendbuf.size(),      info.length);
        EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);

        if (is_recv_msg == 0) {
            recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE,
                               0x1337, 0xffff);
        } else {
            recv_req = (request*)ucp_tag_msg_recv_nb(receiver->worker(),
                                                     &recvbuf[0],recvbuf.size(),
                                                     DATATYPE, message, recv_callback);
            ASSERT_TRUE(!UCS_PTR_IS_ERR(recv_req));
        }

        wait(recv_req);
        EXPECT_TRUE(recv_req->completed);
        if (s_size != r_size) {
            /* Test for correct msg trancation handling */
            EXPECT_EQ(UCS_ERR_MESSAGE_TRUNCATED, recv_req->status);
        } else {
            /* Everything should be received correctly */
            EXPECT_EQ(UCS_OK,              recv_req->status);
            EXPECT_EQ(sendbuf.size(),      recv_req->info.length);
            EXPECT_EQ((ucp_tag_t)0x111337, recv_req->info.sender_tag);
            EXPECT_EQ(sendbuf, recvbuf);
        }
        request_release(recv_req);

        if (is_sync) {
            wait(send_req);
            EXPECT_TRUE(send_req->completed);
            EXPECT_EQ(UCS_OK, send_req->status);
            request_release(send_req);
        }
    }

    int probe_all(std::string &recvbuf)
    {
        ucp_tag_message_h message;
        ucp_tag_recv_info_t info;
        request *req;

        int count = 0;
        for (;;) {
             message = ucp_tag_probe_nb(receiver->worker(), 0, 0, 1, &info);
             if (message == NULL) {
                 return count;
             }

             req = (request*)ucp_tag_msg_recv_nb(receiver->worker(),
                                                 &recvbuf[0], recvbuf.size(),
                                                 DATATYPE, message, recv_callback);
             wait(req);
             request_release(req);
             ++count;
        }
    }

};


UCS_TEST_P(test_ucp_tag_probe, send_probe) {
    test_send_probe (8, 8, false, 0);
    test_send_probe (8, 8, true,  0);
}

UCS_TEST_P(test_ucp_tag_probe, send_medium_msg_probe) {
    test_send_probe (50000, 50000, false, 1);
    test_send_probe (50000, 50000, true,  1);
}

UCS_TEST_P(test_ucp_tag_probe, send_medium_msg_probe_truncated) {
    test_send_probe (50000, 0, false, 1);
    test_send_probe (50000, 0, true,  1);
}

UCS_TEST_P(test_ucp_tag_probe, limited_probe_size) {
    static const int COUNT = 1000;
    std::string sendbuf, recvbuf;
    std::vector<request*> reqs;
    ucp_tag_recv_info_t info;
    request *req;
    int recvd;

    sendbuf.resize(100, '1');
    recvbuf.resize(100, '0');

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
    recv_b(&recvbuf[0], recvbuf.size(), DATATYPE, 0x111337, 0xffffff, &info);

    /* send 1000 messages without calling progress */
    for (int i = 0; i < COUNT; ++i) {
        req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
        if (req != NULL) {
            reqs.push_back(req);
        }

        sender->progress(); /* progress only the sender */
    }

    for (int i = 0; i < 1000; ++i) {
        ucs::safe_usleep(1000);
        sender->progress();
    }

    /* progress once */
    ucp_worker_progress(receiver->worker());

    /* probe should not have too many messages here because we poll once */
    recvd = probe_all(recvbuf);
    EXPECT_LE(recvd, 32);

    /* receive all the rest */
    while (recvd < COUNT) {
        progress();
        recvd += probe_all(recvbuf);
    }

    while (!reqs.empty()) {
        wait(reqs.back());
        request_release(reqs.back());
        reqs.pop_back();
    }
}
UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_probe)
