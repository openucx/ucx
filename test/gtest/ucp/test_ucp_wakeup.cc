/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"
#include "poll.h"

class test_ucp_wakeup : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP;
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

UCS_TEST_P(test_ucp_wakeup, efd)
{
    int recv_efd;
    struct pollfd polled;
    ucp_worker_h recv_worker;
    const ucp_datatype_t DATATYPE = ucp_dt_make_contig(1);
    const uint64_t TAG = 0xdeadbeef;
    uint64_t send_data = 0x12121212;
    entity *sender   = create_entity();
    entity *receiver = create_entity();
    void *req;

    polled.events = POLLIN;
    sender->connect(receiver);

    recv_worker = receiver->worker();
    ASSERT_UCS_OK(ucp_worker_get_efd(recv_worker, &recv_efd));

    polled.fd = recv_efd;
    EXPECT_EQ(0, poll(&polled, 1, 0));
    ASSERT_UCS_OK(ucp_worker_arm(recv_worker));

    req = ucp_tag_send_nb(sender->ep(), &send_data, sizeof(send_data), DATATYPE,
                          TAG, send_completion);
    if (UCS_PTR_IS_PTR(req)) {
        wait(req);
    } else {
        ASSERT_UCS_OK(UCS_PTR_STATUS(req));
    }

    ucs::safe_usleep(10e3);

    ASSERT_EQ(1, poll(&polled, 1, 1)) << "fd=" << polled.fd;
    ASSERT_UCS_OK(ucp_worker_arm(recv_worker));

    uint64_t recv_data = 0;
    req = ucp_tag_recv_nb(receiver->worker(), &recv_data, sizeof(recv_data),
                          DATATYPE, TAG, (ucp_tag_t)-1, recv_completion);
    wait(req);

    close(recv_efd);
    ucp_worker_flush(sender->worker());
    EXPECT_EQ(send_data, recv_data);
}

UCS_TEST_P(test_ucp_wakeup, signal)
{
    int efd;
    ucp_worker_h worker;
    struct pollfd polled;
    entity *entity   = create_entity();

    polled.events = POLLIN;

    worker = entity->worker();
    ASSERT_UCS_OK(ucp_worker_get_efd(worker, &efd));

    polled.fd = efd;
    EXPECT_EQ(poll(&polled, 1, 0), 0);
    ASSERT_UCS_OK(ucp_worker_arm(worker));
    ASSERT_UCS_OK(ucp_worker_signal(worker));
    EXPECT_EQ(poll(&polled, 1, 0), 1);
    ASSERT_UCS_OK(ucp_worker_arm(worker));
    EXPECT_EQ(poll(&polled, 1, 0), 0);

    ASSERT_UCS_OK(ucp_worker_signal(worker));
    ASSERT_UCS_OK(ucp_worker_signal(worker));
    EXPECT_EQ(poll(&polled, 1, 0), 1);
    ASSERT_UCS_OK(ucp_worker_arm(worker));
    EXPECT_EQ(poll(&polled, 1, 0), 0);

    close(efd);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_wakeup)
