/**
* Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <common/test_helpers.h>
#include <ucs/sys/sys.h>
#include <ifaddrs.h>
#include <sys/poll.h>

class test_ucp_sockaddr : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.field_mask  |= UCP_PARAM_FIELD_FEATURES;
        params.features     = UCP_FEATURE_TAG |
                              UCP_FEATURE_WAKEUP;
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

    void inaddr_any_addr(struct sockaddr_in *addr, in_port_t port)
    {
        memset(addr, 0, sizeof(struct sockaddr_in));
        addr->sin_family      = AF_INET;
        addr->sin_addr.s_addr = INADDR_ANY;
        addr->sin_port        = port;
    }

    void start_listener(const struct sockaddr* addr)
    {
        ucs_status_t status = receiver().listen(addr, sizeof(*addr));
        if (status == UCS_ERR_UNREACHABLE) {
            UCS_TEST_SKIP_R("cannot listen to " + ucs::sockaddr_to_str(addr));
        }
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

    static void handle_wakeup(ucp_worker_h send_worker, ucp_worker_h recv_worker)
    {
        int ret, send_efd, recv_efd;
        ucs_status_t status;

        ASSERT_UCS_OK(ucp_worker_get_efd(send_worker, &send_efd));
        ASSERT_UCS_OK(ucp_worker_get_efd(recv_worker, &recv_efd));

        status = ucp_worker_arm(recv_worker);
        if (status == UCS_ERR_BUSY) {
            return;
        }
        ASSERT_UCS_OK(status);

        status = ucp_worker_arm(send_worker);
        if (status == UCS_ERR_BUSY) {
            return;
        }
        ASSERT_UCS_OK(status);

        do {
            struct pollfd pfd[2];
            pfd[0].fd     = send_efd;
            pfd[1].fd     = recv_efd;
            pfd[0].events = POLLIN;
            pfd[1].events = POLLIN;
            ret = poll(pfd, 2, -1);
        } while ((ret < 0) && (errno == EINTR));
        if (ret < 0) {
            UCS_TEST_MESSAGE << "poll() failed: " << strerror(errno);
        }

        EXPECT_GE(ret, 1);
    }

    void wait(ucp_worker_h send_worker, ucp_worker_h recv_worker, bool wakeup)
    {
        if (progress()) {
            return;
        }

        if (wakeup) {
            handle_wakeup(send_worker, recv_worker);
        }
    }

    void tag_send_recv(entity& from, entity& to, bool wakeup)
    {
        uint64_t send_data = ucs_generate_uuid(0);
        void *send_req = ucp_tag_send_nb(from.ep(), &send_data, 1,
                                         ucp_dt_make_contig(sizeof(send_data)),
                                         1, scomplete_cb);
        if (send_req == NULL) {
        } else if (UCS_PTR_IS_ERR(send_req)) {
            ASSERT_UCS_OK(UCS_PTR_STATUS(send_req));
        } else {
            while (!ucp_request_is_completed(send_req)) {
                wait(from.worker(), to.worker(), wakeup);
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
                wait(from.worker(), to.worker(), wakeup);
            }
            ucp_request_free(recv_req);
        }

        EXPECT_EQ(send_data, recv_data);
    }

    void connect_and_send_recv(struct sockaddr *connect_addr, bool wakeup)
    {
        ucp_ep_params_t ep_params = ucp_test::get_ep_params();
        ep_params.field_mask      |= UCP_EP_PARAM_FIELD_FLAGS |
                                     UCP_EP_PARAM_FIELD_SOCK_ADDR |
                                     UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
        ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
        ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
        ep_params.sockaddr.addr    = connect_addr;
        ep_params.sockaddr.addrlen = sizeof(*connect_addr);
        sender().connect(&receiver(), ep_params);

        tag_send_recv(sender(), receiver(), wakeup);

        while (receiver().get_num_eps() == 0) {
            wait(sender().worker(), receiver().worker(), wakeup);
        }

        tag_send_recv(receiver(), sender(), wakeup);
    }

    void listen_and_communicate(bool wakeup)
    {
        struct sockaddr_in connect_addr;
        get_listen_addr(&connect_addr);

        UCS_TEST_MESSAGE << "Testing " << ucs::sockaddr_to_str((const struct sockaddr*)&connect_addr);

        start_listener((const struct sockaddr*)&connect_addr);
        connect_and_send_recv((struct sockaddr*)&connect_addr, wakeup);
    }

    static void err_handler_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
        test_ucp_sockaddr *self = reinterpret_cast<test_ucp_sockaddr*>(arg);
        self->err_handler_count++;
    }

protected:
    volatile int err_handler_count;
};

UCS_TEST_P(test_ucp_sockaddr, listen) {
    listen_and_communicate(false);
}

UCS_TEST_P(test_ucp_sockaddr, wakeup) {
    listen_and_communicate(true);
}

UCS_TEST_P(test_ucp_sockaddr, listen_inaddr_any) {

    struct sockaddr_in connect_addr, inaddr_any_listen_addr;
    get_listen_addr(&connect_addr);
    inaddr_any_addr(&inaddr_any_listen_addr, connect_addr.sin_port);

    UCS_TEST_MESSAGE << "Testing " <<
                     ucs::sockaddr_to_str((const struct sockaddr*)&inaddr_any_listen_addr);

    start_listener((const struct sockaddr*)&inaddr_any_listen_addr);
    connect_and_send_recv((struct sockaddr*)&connect_addr, false);
}

UCS_TEST_P(test_ucp_sockaddr, err_handle) {

    struct sockaddr_in listen_addr;
    err_handler_count = 0;

    get_listen_addr(&listen_addr);

    ucs_status_t status = receiver().listen((const struct sockaddr*)&listen_addr,
                                            sizeof(listen_addr));
    if (status == UCS_ERR_UNREACHABLE) {
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
