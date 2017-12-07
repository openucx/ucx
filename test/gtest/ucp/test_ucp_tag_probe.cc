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
    /* The parameters mean the following:
     *  - s_size and r_size: send and recv buffer sizes.
     *    Can be different for checking message transaction error
     *  - is_sync: specifies the type of send function to be used
     *    (sync or not)
     *  - is_recv_msg: specifies whether probe function needs to remove
     *    matched message. If yes, then ucp_tag_msg_recv_nb is used for
     *    receive
     * */
    void test_send_probe (size_t s_size, size_t r_size, bool is_sync,
                          int is_recv_msg) {
        ucp_tag_recv_info_t info;
        ucp_tag_message_h   message;
        request             *send_req = NULL;
        request             *recv_req = NULL;

        std::vector<char> sendbuf(s_size, 0);
        std::vector<char> recvbuf(r_size, 0);

        ucs::fill_random(sendbuf);

        message = ucp_tag_probe_nb(receiver().worker(), 0x1337, 0xffff,
                                   is_recv_msg, &info);
        EXPECT_TRUE(message == NULL);

        if (is_sync) {
            send_req = send_sync_nb(&sendbuf[0], sendbuf.size(), DATATYPE,
                                    0x111337);

        } else {
            send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
        }

        do {
            progress();
            message = ucp_tag_probe_nb(receiver().worker(), 0x1337, 0xffff,
                                       is_recv_msg, &info);
        } while (message == NULL);

        EXPECT_EQ(sendbuf.size(),      info.length);
        EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);

        if (is_recv_msg == 0) {
            recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE,
                               0x1337, 0xffff);
        } else {
            recv_req = (request*)ucp_tag_msg_recv_nb(receiver().worker(),
                                                     &recvbuf[0],recvbuf.size(),
                                                     DATATYPE, message, recv_callback);
            ASSERT_TRUE(!UCS_PTR_IS_ERR(recv_req));
        }

        wait(recv_req);
        EXPECT_TRUE(recv_req->completed);
        if (s_size != r_size) {
            /* Test for correct msg transaction handling */
            EXPECT_EQ(UCS_ERR_MESSAGE_TRUNCATED, recv_req->status);
        } else {
            /* Everything should be received correctly */
            EXPECT_EQ(UCS_OK,              recv_req->status);
            EXPECT_EQ(sendbuf.size(),      recv_req->info.length);
            EXPECT_EQ((ucp_tag_t)0x111337, recv_req->info.sender_tag);
            EXPECT_EQ(sendbuf, recvbuf);
        }
        request_release(recv_req);

        if (UCS_PTR_IS_PTR(send_req)) {
            wait(send_req);
            EXPECT_TRUE(send_req->completed);
            EXPECT_EQ(UCS_OK, send_req->status);
            request_release(send_req);
        }
    }

    int probe_all(std::string &recvbuf)
    {
        ucp_tag_message_h   message;
        ucp_tag_recv_info_t info;
        request             *req;

        int count = 0;
        for (;;) {
             message = ucp_tag_probe_nb(receiver().worker(), 0, 0, 1, &info);
             if (message == NULL) {
                 return count;
             }

             req = (request*)ucp_tag_msg_recv_nb(receiver().worker(),
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

UCS_TEST_P(test_ucp_tag_probe, send_medium_msg_probe, "RNDV_THRESH=1048576") {
    test_send_probe (50000, 50000, false, 1);
    test_send_probe (50000, 50000, true,  1);
}

UCS_TEST_P(test_ucp_tag_probe, send_medium_msg_probe_truncated, "RNDV_THRESH=1048576") {
    test_send_probe (50000, 0, false, 1);
    test_send_probe (50000, 0, true,  1);
}

UCS_TEST_P(test_ucp_tag_probe, send_rndv_msg_probe, "RNDV_THRESH=1048576") {
    static const size_t size = 1148576;
    ucp_tag_recv_info_t info;
    ucp_tag_message_h   message;
    request             *my_send_req, *my_recv_req;

    skip_loopback();

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf);

    message = ucp_tag_probe_nb(receiver().worker(), 0x1337, 0xffff, 1, &info);
    EXPECT_TRUE(message == NULL);

    /* sender - send the RTS */
    my_send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    /* receiver - get the RTS and put it into unexpected */
    wait_for_unexpected_msg(receiver().worker(), 10.0);

    /* receiver - match the rts, remove it from unexpected and return it */
    message = ucp_tag_probe_nb(receiver().worker(), 0x1337, 0xffff, 1, &info);
    /* make sure that there was a match (RTS) */
    ASSERT_TRUE(message != NULL);
    EXPECT_EQ(sendbuf.size(),      info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);

    /* receiver - process the rts and schedule a get operation */
    my_recv_req = (request*)ucp_tag_msg_recv_nb(receiver().worker(), &recvbuf[0],
                                                recvbuf.size(), DATATYPE, message,
                                                recv_callback);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));

    /* receiver - perform rndv get and send the ATS */
    wait(my_recv_req);
    EXPECT_TRUE(my_recv_req->completed);

    /* sender - get the ATS and set send request to completed */
    short_progress_loop();

    EXPECT_EQ(UCS_OK,              my_recv_req->status);
    EXPECT_EQ(sendbuf.size(),      my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);

    wait_and_validate(my_send_req);
    request_release(my_recv_req);
}

UCS_TEST_P(test_ucp_tag_probe, send_2_msg_probe, "RNDV_THRESH=inf") {
    const ucp_datatype_t DT_INT = ucp_dt_make_contig(sizeof(int));
    const ucp_tag_t      TAG    = 0xaaa;
    const size_t         COUNT  = 20000;
    std::vector<request*> reqs;

    /*
     * send in order: 1, 2
     */
    std::vector<int> sdata1(COUNT, 1);
    std::vector<int> sdata2(COUNT, 2);
    request *sreq1 = send_nb(&sdata1[0], COUNT, DT_INT, TAG);
    if (sreq1 != NULL) {
        reqs.push_back(sreq1);
    }
    request *sreq2 = send_nb(&sdata2[0], COUNT, DT_INT, TAG);
    if (sreq2 != NULL) {
        reqs.push_back(sreq2);
    }

    /*
     * probe in order: 1, 2
     */
    ucp_tag_message_h   message1, message2;
    ucp_tag_recv_info_t info;
    do {
        progress();
        message1 = ucp_tag_probe_nb(receiver().worker(), TAG, 0xffff, 1, &info);
    } while (message1 == NULL);
    do {
        progress();
        message2 = ucp_tag_probe_nb(receiver().worker(), TAG, 0xffff, 1, &info);
    } while (message2 == NULL);

    /*
     * receive in **reverse** order: 2, 1
     */
    std::vector<int> rdata2(COUNT);
    request *rreq2 = (request*)ucp_tag_msg_recv_nb(receiver().worker(), &rdata2[0],
                                                   COUNT, DT_INT, message2,
                                                   recv_callback);
    reqs.push_back(rreq2);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(rreq2));
    wait(rreq2);

    std::vector<int> rdata1(COUNT);
    request *rreq1 = (request*)ucp_tag_msg_recv_nb(receiver().worker(), &rdata1[0],
                                                   COUNT, DT_INT, message1,
                                                   recv_callback);
    reqs.push_back(rreq1);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(rreq1));
    wait(rreq1);

    if (sreq1 != NULL) {
        wait(sreq1);
    }
    if (sreq2 != NULL) {
        wait(sreq2);
    }

    /*
     * expect data to arrive in probe order (rather than recv order)
     */
    EXPECT_EQ(sdata1, rdata1);
    EXPECT_EQ(sdata2, rdata2);
    while (!reqs.empty()) {
        request *req = reqs.back();
        EXPECT_TRUE(req->completed);
        EXPECT_EQ(UCS_OK, req->status);
        request_release(req);
        reqs.pop_back();
    }
}

UCS_TEST_P(test_ucp_tag_probe, limited_probe_size) {
    static const int COUNT = 1000;
    std::string sendbuf, recvbuf;
    std::vector<request*> reqs;
    ucp_tag_recv_info_t info;
    request *req;
    int recvd;

    skip_loopback();

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

        sender().progress(); /* progress only the sender */
    }

    for (int i = 0; i < 1000; ++i) {
        ucs::safe_usleep(1000);
        sender().progress();
    }

    /* progress once */
    ucp_worker_progress(receiver().worker());

    /* probe should not have too many messages here because we poll once */
    recvd = probe_all(recvbuf);
    EXPECT_LE(recvd, 128);

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
