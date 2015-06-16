/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ucp_test.h"

class test_ucp_tag : public ucp_test {
    public:
        int count;
        entity sender;
        entity receiver;
};

void ucp_gtest_progress(ucp_worker_h worker, void *arg)
{
    static int count;
    test_ucp_tag *test = (test_ucp_tag *)arg;

    if (count >= 1) 
        return;

    count = 1;
    if (worker != test->sender.worker()) {
        ucp_worker_progress(test->sender.worker());
    }
    if (worker != test->receiver.worker()) {
        ucp_worker_progress(test->receiver.worker());
    }
    count = 0;
}

UCS_TEST_F(test_ucp_tag, send_recv_exp) {
    ucs_status_t status;
    ucp_tag_recv_completion comp;

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    ucp_progress_register(sender.worker(), ucp_gtest_progress, this);
    ucp_progress_register(receiver.worker(), ucp_gtest_progress, this);
    sender.connect(receiver);
    receiver.connect(sender);

    status = ucp_tag_send(sender.ep(), &send_data, sizeof(send_data), 0x111337);
    ASSERT_UCS_OK(status);

    status = ucp_tag_recv(receiver.worker(), &recv_data, sizeof(recv_data),
                          0x1337, 0xffff, &comp);
    ASSERT_UCS_OK(status);

    EXPECT_EQ((ucp_tag_t)0x111337, comp.sender_tag);
    EXPECT_EQ(send_data, recv_data);

    sender.disconnect();
    receiver.disconnect();
}


UCS_TEST_F(test_ucp_tag, send_recv_unexp) {
    ucs_status_t status;
    ucp_tag_recv_completion comp;

    ucp_progress_register(sender.worker(), ucp_gtest_progress, this);
    ucp_progress_register(receiver.worker(), ucp_gtest_progress, this);
    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    sender.connect(receiver);
    receiver.connect(sender);

    status = ucp_tag_send(sender.ep(), &send_data, sizeof(send_data), 0x111337);
    ASSERT_UCS_OK(status);

    usleep(10000);
    ucp_worker_progress(receiver.worker());

    status = ucp_tag_recv(receiver.worker(), &recv_data, sizeof(recv_data),
                          0x1337, 0xffff, &comp);
    ASSERT_UCS_OK(status);

    EXPECT_EQ((ucp_tag_t)0x111337, comp.sender_tag);
    EXPECT_EQ(send_data, recv_data);

    sender.disconnect();
    receiver.disconnect();
}
