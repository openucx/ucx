/**
* Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <common/test_helpers.h>
#include <ucs/sys/sys.h>
#include <ifaddrs.h>

class test_ucp_sockaddr : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.field_mask  |= UCP_PARAM_FIELD_FEATURES;
        params.features     = UCP_FEATURE_TAG;
        return params;
    }

    void get_listen_addr(struct sockaddr_in *listen_addr) {
        struct ifaddrs* ifaddrs;
        int ret = getifaddrs(&ifaddrs);
        ASSERT_EQ(ret, 0);

        for (struct ifaddrs *ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
            if (ucs_netif_is_active(ifa->ifa_name) &&
                ucs::is_inet_addr(ifa->ifa_addr)   &&
                ucs::is_ib_netdev(ifa->ifa_name))
            {
                *listen_addr = *(struct sockaddr_in*)(void*)ifa->ifa_addr;
                listen_addr->sin_port = ucs::get_port();
                freeifaddrs(ifaddrs);
                return;
            }
        }
        freeifaddrs(ifaddrs);
        UCS_TEST_SKIP_R("No interface for testing");
    }

    static void scomplete_cb(void *req, ucs_status_t status)
    {
        ASSERT_UCS_OK(status);
    }

    static void rcomplete_cb(void *req, ucs_status_t status,
                             ucp_tag_recv_info_t *info)
    {
        ASSERT_UCS_OK(status);
    }

    void tag_send(entity& from, entity& to) {
        uint64_t send_data = ucs_generate_uuid(0);
        void *send_req = ucp_tag_send_nb(from.ep(), &send_data, 1,
                                         ucp_dt_make_contig(sizeof(send_data)),
                                         1, scomplete_cb);
        if (send_req == NULL) {
        } else if (UCS_PTR_IS_ERR(send_req)) {
            ASSERT_UCS_OK(UCS_PTR_STATUS(send_req));
        } else {
            while (!ucp_request_is_completed(send_req)) {
                progress();
            }
            ucp_request_free(send_req);
        }

        uint64_t recv_data = 0;
        void *recv_req = ucp_tag_recv_nb(to.worker(), &recv_data, 1,
                                         ucp_dt_make_contig(sizeof(recv_data)),
                                         1, 0, rcomplete_cb);
        if (UCS_PTR_IS_ERR(recv_req)) {
            ASSERT_UCS_OK(UCS_PTR_STATUS(recv_req));
        } else {
            while (!ucp_request_is_completed(recv_req)) {
                progress();
            }
            ucp_request_free(recv_req);
        }

        EXPECT_EQ(send_data, recv_data);
    }

    static void err_handler_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
        test_ucp_sockaddr *self = reinterpret_cast<test_ucp_sockaddr*>(arg);
        self->err_handler_count++;
    }

protected:
    volatile int err_handler_count;
};

UCS_TEST_P(test_ucp_sockaddr, listen) {

    struct sockaddr_in listen_addr;
    get_listen_addr(&listen_addr);

    ucs_status_t status = receiver().listen((const struct sockaddr*)&listen_addr,
                                            sizeof(listen_addr));
    if (status == UCS_ERR_INVALID_ADDR) {
        UCS_TEST_SKIP_R("cannot listen to " + ucs::sockaddr_to_str(&listen_addr));
    }

    ucp_ep_params_t ep_params = ucp_test::get_ep_params();
    ep_params.field_mask      |= UCP_EP_PARAM_FIELD_FLAGS |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr    = (struct sockaddr*)&listen_addr;
    ep_params.sockaddr.addrlen = sizeof(listen_addr);
    sender().connect(&receiver(), ep_params);

    tag_send(sender(), receiver());

    /* wait for reverse ep to appear */
    while (receiver().get_num_eps() == 0) {
        progress();
    }
    tag_send(receiver(), sender());
}

UCS_TEST_P(test_ucp_sockaddr, err_handle) {

    struct sockaddr_in listen_addr;
    err_handler_count = 0;

    get_listen_addr(&listen_addr);

    ucs_status_t status = receiver().listen((const struct sockaddr*)&listen_addr,
                                            sizeof(listen_addr));
    if (status == UCS_ERR_INVALID_ADDR) {
        UCS_TEST_SKIP_R("cannot listen to " + ucs::sockaddr_to_str(&listen_addr));
    }

    /* make the client try to connect to a non-existing port on the server side */
    listen_addr.sin_port = 1;

    ucp_ep_params_t ep_params = ucp_test::get_ep_params();
    ep_params.field_mask      |= UCP_EP_PARAM_FIELD_FLAGS |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                 UCP_EP_PARAM_FIELD_USER_DATA;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb   = err_handler_cb;
    ep_params.err_handler.arg  = NULL;
    ep_params.user_data        = reinterpret_cast<void*>(this);
    ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr    = (struct sockaddr*)&listen_addr;
    ep_params.sockaddr.addrlen = sizeof(listen_addr);
    wrap_errors();
    sender().connect(&receiver(), ep_params);
    /* allow for the unreachable event to arrive before restoring errors */
    wait_for_flag(&err_handler_count);
    restore_errors();

    EXPECT_EQ(1, err_handler_count);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_sockaddr)
