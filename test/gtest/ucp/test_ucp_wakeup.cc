/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <algorithm>
#include <sys/epoll.h>
#include <sys/poll.h>


class test_ucp_wakeup : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP;
        return params;
    }

protected:
    static void send_completion(void *request, ucs_status_t status) {
        ++comp_cntr;
    }

    static void recv_completion(void *request, ucs_status_t status,
                                ucp_tag_recv_info_t *info) {
        ++comp_cntr;
    }

    void wait(void *req) {
        do {
            progress();
        } while (!ucp_request_is_completed(req));
        ucp_request_release(req);
    }

    void arm(ucp_worker_h worker) {
        ucs_status_t status;
        do {
            status = ucp_worker_arm(worker);
        } while (UCS_ERR_BUSY == status);
        ASSERT_EQ(UCS_OK, status);
    }

    static size_t comp_cntr;
};

size_t test_ucp_wakeup::comp_cntr = 0;

UCS_TEST_P(test_ucp_wakeup, efd)
{
    const ucp_datatype_t DATATYPE = ucp_dt_make_contig(1);
    const uint64_t TAG = 0xdeadbeef;
    ucp_worker_h recv_worker;
    int recv_efd;
    void *req;

    sender().connect(&receiver(), get_ep_params());

    recv_worker = receiver().worker();
    ASSERT_UCS_OK(ucp_worker_get_efd(recv_worker, &recv_efd));

    uint64_t send_data = 0x12121212;
    req = ucp_tag_send_nb(sender().ep(), &send_data, sizeof(send_data), DATATYPE,
                          TAG, send_completion);
    if (UCS_PTR_IS_PTR(req)) {
        wait(req);
    } else {
        ASSERT_UCS_OK(UCS_PTR_STATUS(req));
    }

    uint64_t recv_data = 0;
    req = ucp_tag_recv_nb(receiver().worker(), &recv_data, sizeof(recv_data),
                          DATATYPE, TAG, (ucp_tag_t)-1, recv_completion);
    while (!ucp_request_is_completed(req)) {

        if (ucp_worker_progress(recv_worker)) {
            /* Got some receive events, check request */
            continue;
        }

        ucs_status_t status = ucp_worker_arm(recv_worker);
        if (status == UCS_ERR_BUSY) {
            /* Could not arm, poll again */
            ucp_worker_progress(recv_worker);
            continue;
        }
        ASSERT_UCS_OK(status);

        int ret;
        do {
            struct pollfd pollfd;
            pollfd.events = POLLIN;
            pollfd.fd     = recv_efd;
            ret = poll(&pollfd, 1, -1);
        } while ((ret < 0) && (errno == EINTR));
        if (ret < 0) {
            UCS_TEST_MESSAGE << "poll() failed: " << strerror(errno);
        }
        ASSERT_EQ(1, ret);
        EXPECT_EQ(UCS_ERR_BUSY, ucp_worker_arm(recv_worker));
    }

    ucp_request_release(req);

    flush_worker(sender());
    EXPECT_EQ(send_data, recv_data);
}

UCS_TEST_P(test_ucp_wakeup, tx_wait, "ZCOPY_THRESH=10000")
{
    const ucp_datatype_t DATATYPE = ucp_dt_make_contig(1);
    const size_t COUNT            = 20000;
    const uint64_t TAG            = 0xdeadbeef;
    std::string send_data(COUNT, '2'), recv_data(COUNT, '1');
    void *sreq, *rreq;

    sender().connect(&receiver(), get_ep_params());

    rreq = ucp_tag_recv_nb(receiver().worker(), &recv_data[0], COUNT, DATATYPE,
                           TAG, (ucp_tag_t)-1, recv_completion);

    sreq = ucp_tag_send_nb(sender().ep(), &send_data[0], COUNT, DATATYPE, TAG,
                           send_completion);

    if (UCS_PTR_IS_PTR(sreq)) {
        /* wait for send completion */
        do {
            ucp_worker_wait(sender().worker());
            while (progress());
        } while (!ucp_request_is_completed(sreq));
        ucp_request_release(sreq);
    } else {
        ASSERT_UCS_OK(UCS_PTR_STATUS(sreq));
    }

    wait(rreq);

    EXPECT_EQ(send_data, recv_data);
}

UCS_TEST_P(test_ucp_wakeup, signal)
{
    int efd;
    ucp_worker_h worker;
    struct pollfd polled;

    polled.events = POLLIN;

    worker = sender().worker();
    ASSERT_UCS_OK(ucp_worker_get_efd(worker, &efd));

    polled.fd = efd;
    EXPECT_EQ(0, poll(&polled, 1, 0));
    arm(worker);
    ASSERT_UCS_OK(ucp_worker_signal(worker));
    EXPECT_EQ(1, poll(&polled, 1, 0));
    arm(worker);
    EXPECT_EQ(0, poll(&polled, 1, 0));

    ASSERT_UCS_OK(ucp_worker_signal(worker));
    ASSERT_UCS_OK(ucp_worker_signal(worker));
    EXPECT_EQ(1, poll(&polled, 1, 0));
    arm(worker);
    EXPECT_EQ(0, poll(&polled, 1, 0));

    ASSERT_UCS_OK(ucp_worker_signal(worker));
    EXPECT_EQ(UCS_ERR_BUSY, ucp_worker_arm(worker));
    EXPECT_EQ(UCS_OK, ucp_worker_arm(worker));
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_wakeup)

class test_ucp_wakeup_external_epollfd : public test_ucp_wakeup {
public:
    virtual ucp_worker_params_t get_worker_params() {
        ucp_worker_params_t params = test_ucp_wakeup::get_worker_params();
        params.field_mask |= UCP_WORKER_PARAM_FIELD_EVENT_FD |
                             UCP_WORKER_PARAM_FIELD_USER_DATA;
        params.event_fd  = m_epfd;
        params.user_data = USER_DATA;
        return params;
    }

protected:
    static void* const USER_DATA;

    virtual void init() {
        m_epfd = epoll_create(1);
        ASSERT_GE(m_epfd, 0);
        test_ucp_wakeup::init();
    }

    virtual void cleanup() {
        test_ucp_wakeup::cleanup();
        close(m_epfd);
    }

    int m_epfd;
};

void* const test_ucp_wakeup_external_epollfd::USER_DATA = (void*)0x1337abcdef;


UCS_TEST_P(test_ucp_wakeup_external_epollfd, epoll_wait)
{
    const ucp_datatype_t DATATYPE = ucp_dt_make_contig(1);
    const uint64_t TAG = 0xdeadbeef;
    void *req;

    sender().connect(&receiver(), get_ep_params());

    uint64_t send_data = 0x12121212;
    req = ucp_tag_send_nb(sender().ep(), &send_data, sizeof(send_data), DATATYPE,
                          TAG, send_completion);
    if (UCS_PTR_IS_PTR(req)) {
        wait(req);
    } else {
        ASSERT_UCS_OK(UCS_PTR_STATUS(req));
    }

    uint64_t recv_data = 0;
    req = ucp_tag_recv_nb(receiver().worker(), &recv_data, sizeof(recv_data),
                          DATATYPE, TAG, (ucp_tag_t)-1, recv_completion);
    while (!ucp_request_is_completed(req)) {

        ucp_worker_h recv_worker = receiver().worker();

        if (ucp_worker_progress(recv_worker)) {
            /* Got some receive events, check request */
            continue;
        }

        ucs_status_t status = ucp_worker_arm(recv_worker);
        if (status == UCS_ERR_BUSY) {
            /* Could not arm, poll again */
            ucp_worker_progress(recv_worker);
            continue;
        }
        ASSERT_UCS_OK(status);

        struct epoll_event event;
        int ret;
        do {
            ret = epoll_wait(m_epfd, &event, 1, -1);
        } while ((ret < 0) && (errno == EINTR));
        if (ret < 0) {
            UCS_TEST_MESSAGE << "epoll_wait() failed: " << strerror(errno);
        }
        ASSERT_EQ(1, ret);
        EXPECT_EQ(USER_DATA, event.data.ptr);
    }

    ucp_request_release(req);

    flush_worker(sender());
    EXPECT_EQ(send_data, recv_data);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_wakeup_external_epollfd)
