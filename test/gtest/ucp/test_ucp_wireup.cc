/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"

class test_ucp_wireup : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_TAG;
        return params;
    }

protected:
    static void send_completion(void *request, ucs_status_t status) {
    }

    static void recv_completion(void *request, ucs_status_t status,
                                ucp_tag_recv_info_t *info) {
    }

    void wait(void *req) {
        do {
            progress();
        } while (!ucp_request_is_completed(req));
        ucp_request_release(req);
    }
};

UCS_TEST_P(test_ucp_wireup, one_sided_wireup) {

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

    uint64_t recv_data = 0;
    req = ucp_tag_recv_nb(receiver->worker(), &recv_data, sizeof(recv_data),
                          DATATYPE, TAG, (ucp_tag_t)-1, recv_completion);
    wait(req);

    EXPECT_EQ(send_data, recv_data);

    ucp_worker_flush(sender->worker());
}

UCS_TEST_P(test_ucp_wireup, two_sided_wireup) {
    entity *sender   = create_entity();
    entity *receiver = create_entity();

    sender->connect(receiver);
    receiver->connect(sender);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_wireup)
