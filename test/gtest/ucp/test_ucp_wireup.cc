/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ucp_test.h"

class test_ucp_wireup : public ucp_test {
protected:
    virtual uint64_t features() const {
        return UCP_FEATURE_TAG;
    }
};

UCS_TEST_F(test_ucp_wireup, one_sided_wireup) {

    const uint64_t TAG = 0xdeadbeef;
    uint64_t send_data = 0x12121212;
    entity *sender   = create_entity();
    entity *receiver = create_entity();
    ucs_status_t status;

    sender->connect(receiver);

    status = ucp_tag_send(sender->ep(), &send_data, sizeof(send_data), TAG);
    ASSERT_UCS_OK(status);

    ucp_tag_recv_completion_t comp;
    uint64_t recv_data;
    status = ucp_tag_recv(receiver->worker(), &recv_data, sizeof(recv_data), TAG,
                          (ucp_tag_t)-1, &comp);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(send_data, recv_data);
    EXPECT_EQ(sizeof(send_data), comp.rcvd_len);
    EXPECT_EQ(TAG, comp.sender_tag);

    ucp_flush(sender->worker());
}

UCS_TEST_F(test_ucp_wireup, two_sided_wireup) {
    entity *sender   = create_entity();
    entity *receiver = create_entity();

    sender->connect(receiver);
    receiver->connect(sender);
}
