/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"

class test_ucp_wireup : public ucp_test {
protected:
    virtual void get_params(ucp_params_t& params) const {
        ucp_test::get_params(params);
        params.features |= UCP_FEATURE_TAG;
    }
};

UCS_TEST_F(test_ucp_wireup, one_sided_wireup) {

    const ucp_datatype_t DATATYPE = ucp_dt_make_contig(1);
    const uint64_t TAG = 0xdeadbeef;
    uint64_t send_data = 0x12121212;
    entity *sender   = create_entity();
    entity *receiver = create_entity();
    ucs_status_t status;

    sender->connect(receiver);

    status = ucp_tag_send(sender->ep(), &send_data, sizeof(send_data), DATATYPE,
                          TAG);
    ASSERT_UCS_OK(status);

    ucp_tag_recv_info_t info;
    uint64_t recv_data;
    status = ucp_tag_recv(receiver->worker(), &recv_data, sizeof(recv_data),
                          DATATYPE, TAG, (ucp_tag_t)-1, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(send_data, recv_data);
    EXPECT_EQ(sizeof(send_data), info.length);
    EXPECT_EQ(TAG, info.sender_tag);

    ucp_worker_flush(sender->worker());
}

UCS_TEST_F(test_ucp_wireup, two_sided_wireup) {
    entity *sender   = create_entity();
    entity *receiver = create_entity();

    sender->connect(receiver);
    receiver->connect(sender);
}
