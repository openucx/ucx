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

    static void send_completion(void *request, ucs_status_t status) {
    }

    static void recv_completion(void *request, ucs_status_t status,
                                ucp_tag_recv_info_t *info) {
    }

    void wait(void *req) {
        ucs_status_t status;
        do {
            progress();
            status = ucp_request_test(req);
        } while (status == UCS_INPROGRESS);
        ucp_request_release(req);
        ASSERT_UCS_OK(status);
    }
};

UCS_TEST_F(test_ucp_wireup, one_sided_wireup) {

    const ucp_datatype_t DATATYPE = ucp_dt_make_contig(1);
    const uint64_t TAG = 0xdeadbeef;
    uint64_t send_data = 0x12121212;
    entity *sender   = create_entity();
    entity *receiver = create_entity();
    void *req;

    sender->connect(receiver);

    req = ucp_tag_send_nb(sender->ep(), &send_data, sizeof(send_data), DATATYPE,
                          TAG, send_completion);
    if (UCS_PTR_IS_PTR(req)) {
        wait(req);
    } else {
        ASSERT_UCS_OK(UCS_PTR_STATUS(req));
    }

    ucp_tag_recv_info_t info;
    uint64_t recv_data;
    req = ucp_tag_recv_nb(receiver->worker(), &recv_data, sizeof(recv_data),
                          DATATYPE, TAG, (ucp_tag_t)-1, recv_completion);
    wait(req);


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
