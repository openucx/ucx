/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ucp_test.h"

class test_ucp_tag : public ucp_test {
protected:
    virtual void init() {
        ucp_test::init();
        sender   = create_entity();
        receiver = create_entity();
        sender->connect(receiver);
        receiver->connect(sender);
    }

    virtual void cleanup() {
        sender->disconnect();
        receiver->disconnect();
        ucp_test::cleanup();
    }

public:
    int    count;
    entity *sender, *receiver;
};

UCS_TEST_F(test_ucp_tag, send_recv_exp) {
    ucs_status_t status;
    ucp_tag_recv_completion comp;

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    status = ucp_tag_send(sender->ep(), &send_data, sizeof(send_data), 0x111337);
    ASSERT_UCS_OK(status);

    status = ucp_tag_recv(receiver->worker(), &recv_data, sizeof(recv_data),
                          0x1337, 0xffff, &comp);
    ASSERT_UCS_OK(status);

    EXPECT_EQ((ucp_tag_t)0x111337, comp.sender_tag);
    EXPECT_EQ(send_data, recv_data);
}


UCS_TEST_F(test_ucp_tag, send_recv_unexp) {
    ucs_status_t status;
    ucp_tag_recv_completion comp;

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    status = ucp_tag_send(sender->ep(), &send_data, sizeof(send_data), 0x111337);
    ASSERT_UCS_OK(status);

    usleep(10000);
    ucp_worker_progress(receiver->worker());

    status = ucp_tag_recv(receiver->worker(), &recv_data, sizeof(recv_data),
                          0x1337, 0xffff, &comp);
    ASSERT_UCS_OK(status);

    EXPECT_EQ((ucp_tag_t)0x111337, comp.sender_tag);
    EXPECT_EQ(send_data, recv_data);

}
