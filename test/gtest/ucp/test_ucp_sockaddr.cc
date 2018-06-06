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

#define UCP_INSTANTIATE_ALL_TEST_CASE(_test_case) \
        UCP_INSTANTIATE_TEST_CASE (_test_case) \
        UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, shm, "shm") \
        UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, dc_ud, "dc_x,dc,ud,ud_x,mm") \
        UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, no_ud_ud_x, "dc_x,dc,mm") \
        /* dc_ud case is for testing handling of a large worker address on
         * UCT_IFACE_FLAG_CONNECT_TO_IFACE transports (dc, dc_x) */
        /* no_ud_ud_x case is for testing handling a large worker address
         * but with the lack of ud/ud_x transports, which would return an error
         * and skipped */

class test_ucp_sockaddr : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.field_mask  |= UCP_PARAM_FIELD_FEATURES;
        params.features     = UCP_FEATURE_TAG;
        return params;
    }

    enum {
        EP_ADDR = DEFAULT_PARAM_VARIANT + 1
    };

    static std::vector<ucp_test_param>
    enum_test_params(const ucp_params_t& ctx_params,
                     const std::string& name,
                     const std::string& test_case_name,
                     const std::string& tls)
    {
        std::vector<ucp_test_param> result =
            ucp_test::enum_test_params(ctx_params, name, test_case_name, tls);

        generate_test_params_variant(ctx_params, name, test_case_name, tls,
                                     EP_ADDR, result);
        return result;
    }

    void init()
    {
        test_base::init();
        ucp_ep_params_t ep_params = ucp_test::get_ep_params();

        /* create dummy sender and receiver entities */
        create_entity();
        create_entity();

        /* try to connect the dummy entities to check if the tested transport
         * can support the requested features from ucp_params.
         * regular flow is used here (not client-server) */
        wrap_errors();
        sender().connect(&receiver(), ep_params, 0, 0);
        restore_errors();

        /* remove the dummy sender and receiver entities */
        ucp_test::cleanup();
        /* create valid sender and receiver entities to be used in the test */
        ucp_test::init();
    }

    static ucs_log_func_rc_t
    detect_error_logger(const char *file, unsigned line, const char *function,
                                   ucs_log_level_t level, const char *message, va_list ap)
    {
        if (level == UCS_LOG_LEVEL_ERROR) {
            std::string err_str = format_message(message, ap);
            if ((strstr(err_str.c_str(), "no supported transports found for")) ||
                (strstr(err_str.c_str(), "sockaddr aux resources addresses")) ||
                (strstr(err_str.c_str(), "no peer failure handler")) ||
                /* when the "peer failure" error happens, it is followed by: */
                (strstr(err_str.c_str(), "received event RDMA_CM_EVENT_UNREACHABLE"))) {
                UCS_TEST_MESSAGE << err_str;
                return UCS_LOG_FUNC_RC_STOP;
            }
        }
        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    static void detect_error()
    {
        ucs_log_push_handler(detect_error_logger);
    }

    void get_listen_addr(struct sockaddr_in *listen_addr) {
        struct ifaddrs* ifaddrs;
        int ret = getifaddrs(&ifaddrs);
        ASSERT_EQ(ret, 0);

        for (struct ifaddrs *ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
            if (ucs_netif_is_active(ifa->ifa_name) &&
                ucs::is_inet_addr(ifa->ifa_addr)   &&
                ucs::is_rdmacm_netdev(ifa->ifa_name))
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

    void start_listener(ucp_test_base::entity::listen_cb_type_t cb_type,
                        const struct sockaddr* addr)
    {
        ucs_status_t status = receiver().listen(cb_type, addr, sizeof(*addr));
        if (status == UCS_ERR_UNREACHABLE) {
            UCS_TEST_SKIP_R("cannot listen to " + ucs::sockaddr_to_str(addr));
        }
    }

    static void scomplete_cb(void *req, ucs_status_t status)
    {
        if ((status != UCS_OK) && (status != UCS_ERR_UNREACHABLE)) {
            UCS_TEST_ABORT("Error: " << ucs_status_string(status));
        }
    }

    static void rcomplete_cb(void *req, ucs_status_t status,
                             ucp_tag_recv_info_t *info)
    {
        ASSERT_UCS_OK(status);
    }

    static void wait_for_wakeup(ucp_worker_h send_worker, ucp_worker_h recv_worker)
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

    void check_events(ucp_worker_h send_worker, ucp_worker_h recv_worker,
                      bool wakeup, void *req)
    {
        if (progress()) {
            return;
        }

        if ((req != NULL) && (ucp_request_check_status(req) == UCS_ERR_UNREACHABLE)) {
            return;
        }

        if (wakeup) {
            wait_for_wakeup(send_worker, recv_worker);
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
                check_events(from.worker(), to.worker(), wakeup, send_req);
            }
            /* Check if the error was completed due to the error handling flow.
             * If so, skip the test since a valid error occurred - the one expected
             * from the error handling flow - cases of failure to handle long worker
             * address or transport doesn't support the error handling requirement */
            if (ucp_request_check_status(send_req) == UCS_ERR_UNREACHABLE) {
                ucp_request_free(send_req);
                UCS_TEST_SKIP_R("Skipping due an unreachable destination (unsupported "
                                "feature or too long worker address or no "
                                "supported transport to send partial worker "
                                "address)");
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
                check_events(from.worker(), to.worker(), wakeup, recv_req);
            }
            ucp_request_free(recv_req);
        }

        EXPECT_EQ(send_data, recv_data);
    }

    void wait_for_server_ep(bool wakeup)
    {
        ucs_time_t time_limit = ucs_get_time() + ucs_time_from_sec(UCP_TEST_TIMEOUT_IN_SEC);

        while ((receiver().get_num_eps() == 0) && (ucs_get_time() < time_limit)) {
            check_events(sender().worker(), receiver().worker(), wakeup, NULL);
        }
    }

    void client_ep_connect(struct sockaddr *connect_addr)
    {
        ucp_ep_params_t ep_params = ucp_test::get_ep_params();
        ep_params.field_mask      |= UCP_EP_PARAM_FIELD_FLAGS |
                                     UCP_EP_PARAM_FIELD_SOCK_ADDR |
                                     UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                     UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                     UCP_EP_PARAM_FIELD_USER_DATA;
        /* The error handling requirement is needed since we need to take
         * care of a case where the client gets an error. In case ucp needs to
         * handle a large worker address but neither ud nor ud_x are present */
        ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
        ep_params.err_handler.cb   = err_handler_cb;
        ep_params.err_handler.arg  = NULL;
        ep_params.user_data        = reinterpret_cast<void*>(this);
        ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
        ep_params.sockaddr.addr    = connect_addr;
        ep_params.sockaddr.addrlen = sizeof(*connect_addr);
        sender().connect(&receiver(), ep_params);
    }

    void connect_and_send_recv(struct sockaddr *connect_addr, bool wakeup)
    {
        detect_error();
        client_ep_connect(connect_addr);

        tag_send_recv(sender(), receiver(), wakeup);
        restore_errors();

        wait_for_server_ep(wakeup);

        tag_send_recv(receiver(), sender(), wakeup);
    }

    void listen_and_communicate(ucp_test_base::entity::listen_cb_type_t cb_type,
                                bool wakeup)
    {
        struct sockaddr_in connect_addr;
        get_listen_addr(&connect_addr);
        err_handler_count = 0;

        UCS_TEST_MESSAGE << "Testing "
                         << ucs::sockaddr_to_str(
                                (const struct sockaddr*)&connect_addr);

        start_listener(cb_type, (const struct sockaddr*)&connect_addr);
        connect_and_send_recv((struct sockaddr*)&connect_addr, wakeup);
    }

    static void err_handler_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
        test_ucp_sockaddr *self = reinterpret_cast<test_ucp_sockaddr*>(arg);
        self->err_handler_count++;
        /* The current expected errors are only from the err_handle test
         * and from transports where the worker address is too long but ud/ud_x
         * are not present, or ud/ud_x are present but their addresses are too
         * long as well */
        if ((status != UCS_ERR_UNREACHABLE)) {
            UCS_TEST_ABORT("Error: " << ucs_status_string(status));
        }
    }

protected:
    ucp_test_base::entity::listen_cb_type_t cb_type() const {
        return GetParam().variant == EP_ADDR ?
               ucp_test_base::entity::LISTEN_CB_EP_ADDR :
               ucp_test_base::entity::LISTEN_CB_EP;
    }

    volatile int err_handler_count;
};

UCS_TEST_P(test_ucp_sockaddr, listen) {
    listen_and_communicate(cb_type(), false);
}

UCS_TEST_P(test_ucp_sockaddr, listen_inaddr_any) {

    struct sockaddr_in connect_addr, inaddr_any_listen_addr;
    get_listen_addr(&connect_addr);
    inaddr_any_addr(&inaddr_any_listen_addr, connect_addr.sin_port);
    err_handler_count = 0;

    UCS_TEST_MESSAGE << "Testing "
                     << ucs::sockaddr_to_str(
                        (const struct sockaddr*)&inaddr_any_listen_addr);

    start_listener(cb_type(), (const struct sockaddr*)&inaddr_any_listen_addr);
    connect_and_send_recv((struct sockaddr*)&connect_addr, false);
}

UCS_TEST_P(test_ucp_sockaddr, err_handle) {

    struct sockaddr_in listen_addr;
    err_handler_count = 0;

    get_listen_addr(&listen_addr);

    ucs_status_t status = receiver().listen(cb_type(),
                                            (const struct sockaddr*)&listen_addr,
                                            sizeof(listen_addr));
    if (status == UCS_ERR_UNREACHABLE) {
        UCS_TEST_SKIP_R("cannot listen to " + ucs::sockaddr_to_str(&listen_addr));
    }

    /* make the client try to connect to a non-existing port on the server side */
    listen_addr.sin_port = 1;

    wrap_errors();
    client_ep_connect((struct sockaddr*)&listen_addr);
    /* allow for the unreachable event to arrive before restoring errors */
    wait_for_flag(&err_handler_count);
    restore_errors();

    EXPECT_EQ(1, err_handler_count);
}

UCP_INSTANTIATE_ALL_TEST_CASE(test_ucp_sockaddr)


class test_ucp_sockaddr_with_wakeup : public test_ucp_sockaddr {
public:

    static ucp_params_t get_ctx_params() {
        ucp_params_t params = test_ucp_sockaddr::get_ctx_params();
        params.features    |= UCP_FEATURE_WAKEUP;
        return params;
    }
};

UCS_TEST_P(test_ucp_sockaddr_with_wakeup, wakeup) {
    listen_and_communicate(cb_type(), true);
}

UCP_INSTANTIATE_ALL_TEST_CASE(test_ucp_sockaddr_with_wakeup)


class test_ucp_sockaddr_with_rma_atomic : public test_ucp_sockaddr {
public:

    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.field_mask  |= UCP_PARAM_FIELD_FEATURES;
        params.features     = UCP_FEATURE_RMA   |
                              UCP_FEATURE_AMO32 |
                              UCP_FEATURE_AMO64;
        return params;
    }
};

UCS_TEST_P(test_ucp_sockaddr_with_rma_atomic, wireup_for_rma_atomic) {

    /* This test makes sure that the client-server flow works when the required
     * features are RMA/ATOMIC. With these features, need to make sure that
     * there is a lane for ucp-wireup (an am_lane should be created and used) */
    struct sockaddr_in connect_addr;
    get_listen_addr(&connect_addr);
    err_handler_count = 0;

    UCS_TEST_MESSAGE << "Testing " << ucs::sockaddr_to_str((const struct sockaddr*)&connect_addr);

    start_listener(cb_type(), (const struct sockaddr*)&connect_addr);

    wrap_errors();
    client_ep_connect((struct sockaddr*)&connect_addr);

    /* allow the err_handler callback to be invoked if needed */
    short_progress_loop();
    if (err_handler_count == 1) {
        UCS_TEST_SKIP_R("Skipping due to too long worker address error or no "
                        "matching transport");
    }
    EXPECT_EQ(0, err_handler_count);
    restore_errors();

    wait_for_server_ep(false);

    /* allow the connection establishment flow to complete */
    short_progress_loop();
}

UCP_INSTANTIATE_ALL_TEST_CASE(test_ucp_sockaddr_with_rma_atomic)
