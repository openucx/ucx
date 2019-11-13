/**
* Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"
#include "common/test.h"
#include "ucp/ucp_test.h"

#include <common/test_helpers.h>
#include <ucs/sys/sys.h>
#include <ifaddrs.h>
#include <sys/poll.h>

extern "C" {
#include <ucp/core/ucp_listener.h>
}

#define UCP_INSTANTIATE_ALL_TEST_CASE(_test_case) \
        UCP_INSTANTIATE_TEST_CASE (_test_case) \
        UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, shm, "shm") \
        UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, dc_ud, "dc_x,ud_v,ud_x,mm") \
        UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, no_ud_ud_x, "dc_x,mm") \
        /* dc_ud case is for testing handling of a large worker address on
         * UCT_IFACE_FLAG_CONNECT_TO_IFACE transports (dc_x) */
        /* no_ud_ud_x case is for testing handling a large worker address
         * but with the lack of ud/ud_x transports, which would return an error
         * and skipped */

class test_ucp_sockaddr : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.field_mask  |= UCP_PARAM_FIELD_FEATURES;
        params.features     = UCP_FEATURE_TAG | UCP_FEATURE_STREAM;
        return params;
    }

    enum {
        MT_PARAM_VARIANT = DEFAULT_PARAM_VARIANT + 1, /* Enabled worker level
                                                         multi-threading */
        CONN_REQ_TAG,                                 /* Accepting by ucp_conn_request_h,
                                                         send/recv by TAG API */
        CONN_REQ_STREAM                               /* Accepting by ucp_conn_request_h,
                                                         send/recv by STREAM API */
    };

    typedef enum {
        SEND_RECV_TAG,
        SEND_RECV_STREAM
    } send_recv_type_t;

    ucs::sock_addr_storage test_addr;

    void init() {
        get_sockaddr();
        ucp_test::init();
        skip_loopback();
    }

    static std::vector<ucp_test_param>
    enum_test_params(const ucp_params_t& ctx_params,
                     const std::string& name,
                     const std::string& test_case_name,
                     const std::string& tls)
    {
        std::vector<ucp_test_param> result =
            ucp_test::enum_test_params(ctx_params, name, test_case_name, tls);

        generate_test_params_variant(ctx_params, name, test_case_name, tls,
                                     MT_PARAM_VARIANT, result,
                                     MULTI_THREAD_WORKER);
        generate_test_params_variant(ctx_params, name, test_case_name, tls,
                                     CONN_REQ_TAG, result);
        generate_test_params_variant(ctx_params, name, test_case_name, tls,
                                     CONN_REQ_STREAM, result);
        return result;
    }

    static ucs_log_func_rc_t
    detect_error_logger(const char *file, unsigned line, const char *function,
                        ucs_log_level_t level, const char *message, va_list ap)
    {
        if (level == UCS_LOG_LEVEL_ERROR) {
            static std::vector<std::string> stop_list;
            if (stop_list.empty()) {
                stop_list.push_back("no supported sockaddr auxiliary transports found for");
                stop_list.push_back("sockaddr aux resources addresses");
                stop_list.push_back("no peer failure handler");
                stop_list.push_back("connection request failed on listener");
                /* when the "peer failure" error happens, it is followed by: */
                stop_list.push_back("received event RDMA_CM_EVENT_UNREACHABLE");
                stop_list.push_back(ucs_status_string(UCS_ERR_UNREACHABLE));
                stop_list.push_back(ucs_status_string(UCS_ERR_UNSUPPORTED));
            }

            std::string err_str = format_message(message, ap);
            for (size_t i = 0; i < stop_list.size(); ++i) {
                if (err_str.find(stop_list[i]) != std::string::npos) {
                    UCS_TEST_MESSAGE << err_str;
                    return UCS_LOG_FUNC_RC_STOP;
                }
            }
        }
        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    void get_sockaddr()
    {
        struct ifaddrs* ifaddrs;
        ucs_status_t status;
        size_t size;
        int ret = getifaddrs(&ifaddrs);
        ASSERT_EQ(ret, 0);

        for (struct ifaddrs *ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
            if (ucs_netif_flags_is_active(ifa->ifa_flags) &&
                ucs::is_inet_addr(ifa->ifa_addr) &&
                ucs::is_rdmacm_netdev(ifa->ifa_name))
            {
                status = ucs_sockaddr_sizeof(ifa->ifa_addr, &size);
                ASSERT_UCS_OK(status);
                test_addr.set_sock_addr(*ifa->ifa_addr, size);
                test_addr.set_port(ucs::get_port());

                freeifaddrs(ifaddrs);
                return;
            }
        }
        freeifaddrs(ifaddrs);
        UCS_TEST_SKIP_R("No interface for testing");
    }

    void inaddr_any_ipv4(struct sockaddr_in *addr, in_port_t port)
    {
        memset(addr, 0, sizeof(struct sockaddr_in));
        addr->sin_family      = AF_INET;
        addr->sin_addr.s_addr = INADDR_ANY;
        addr->sin_port        = port;
    }

    void inaddr_any_ipv6(struct sockaddr_in6 *addr, in_port_t port)
    {
        memset(addr, 0, sizeof(struct sockaddr_in6));
        addr->sin6_family = AF_INET6;
        addr->sin6_addr   = in6addr_any;
        addr->sin6_port   = port;
    }

    void start_listener(ucp_test_base::entity::listen_cb_type_t cb_type,
                        const struct sockaddr* addr, size_t addrlen)
    {
        ucs_status_t status = receiver().listen(cb_type, addr, addrlen);
        if (status == UCS_ERR_UNREACHABLE) {
            UCS_TEST_SKIP_R("cannot listen to " + ucs::sockaddr_to_str(addr));
        }
    }

    static void scomplete_cb(void *req, ucs_status_t status)
    {
        if ((status == UCS_OK)              ||
            (status == UCS_ERR_UNREACHABLE) ||
            (status == UCS_ERR_REJECTED)) {
            return;
        }
        UCS_TEST_ABORT("Error: " << ucs_status_string(status));
    }

    static void rtag_complete_cb(void *req, ucs_status_t status,
                                 ucp_tag_recv_info_t *info)
    {
        EXPECT_UCS_OK(status);
    }

    static void rstream_complete_cb(void *req, ucs_status_t status,
                                    size_t length)
    {
        EXPECT_UCS_OK(status);
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

    void send_recv(entity& from, entity& to, send_recv_type_t send_recv_type,
                   bool wakeup, ucp_test_base::entity::listen_cb_type_t cb_type)
    {
        const uint64_t send_data = ucs_generate_uuid(0);
        void *send_req = NULL;
        if (send_recv_type == SEND_RECV_TAG) {
            send_req = ucp_tag_send_nb(from.ep(), &send_data, 1,
                                       ucp_dt_make_contig(sizeof(send_data)), 1,
                                       scomplete_cb);
        } else if (send_recv_type == SEND_RECV_STREAM) {
            send_req = ucp_stream_send_nb(from.ep(), &send_data, 1,
                                          ucp_dt_make_contig(sizeof(send_data)),
                                          scomplete_cb, 0);
        } else {
            ASSERT_TRUE(false) << "unsupported communication type";
        }

        ucs_status_t send_status;
        if (send_req == NULL) {
            send_status = UCS_OK;
        } else if (UCS_PTR_IS_ERR(send_req)) {
            send_status = UCS_PTR_STATUS(send_req);
            ASSERT_UCS_OK(send_status);
        } else {
            while (!ucp_request_is_completed(send_req)) {
                check_events(from.worker(), to.worker(), wakeup, send_req);
            }
            send_status = ucp_request_check_status(send_req);
            ucp_request_free(send_req);
        }

        if (send_status == UCS_ERR_UNREACHABLE) {
            /* Check if the error was completed due to the error handling flow.
             * If so, skip the test since a valid error occurred - the one expected
             * from the error handling flow - cases of failure to handle long worker
             * address or transport doesn't support the error handling requirement */
            UCS_TEST_SKIP_R("Skipping due an unreachable destination (unsupported "
                            "feature or too long worker address or no "
                            "supported transport to send partial worker "
                            "address)");
        } else if ((send_status == UCS_ERR_REJECTED) &&
                   (cb_type == ucp_test_base::entity::LISTEN_CB_REJECT)) {
            return;
        } else {
            ASSERT_UCS_OK(send_status);
        }

        uint64_t recv_data = 0;
        void *recv_req;
        if (send_recv_type == SEND_RECV_TAG) {
            recv_req = ucp_tag_recv_nb(to.worker(), &recv_data, 1,
                                       ucp_dt_make_contig(sizeof(recv_data)),
                                       1, 0, rtag_complete_cb);
        } else {
            ASSERT_TRUE(send_recv_type == SEND_RECV_STREAM);
            ucp_stream_poll_ep_t poll_eps;
            ssize_t              ep_count;
            size_t               recv_length;
            do {
                progress();
                ep_count = ucp_stream_worker_poll(to.worker(), &poll_eps, 1, 0);
            } while (ep_count == 0);
            ASSERT_EQ(1,                  ep_count);
            EXPECT_EQ(to.ep(),            poll_eps.ep);
            EXPECT_EQ((void *)0xdeadbeef, poll_eps.user_data);

            recv_req = ucp_stream_recv_nb(to.ep(), &recv_data, 1,
                                          ucp_dt_make_contig(sizeof(recv_data)),
                                          rstream_complete_cb, &recv_length,
                                          UCP_STREAM_RECV_FLAG_WAITALL);
        }

        if (recv_req != NULL) {
            ASSERT_TRUE(UCS_PTR_IS_PTR(recv_req));
            while (!ucp_request_is_completed(recv_req)) {
                check_events(from.worker(), to.worker(), wakeup, recv_req);
            }
            ucp_request_free(recv_req);
        }

        EXPECT_EQ(send_data, recv_data);
    }

    bool wait_for_server_ep(bool wakeup)
    {
        ucs_time_t deadline = ucs::get_deadline();

        while ((receiver().get_num_eps() == 0) && (m_err_handler_count == 0) &&
               (ucs_get_time() < deadline)) {
            check_events(sender().worker(), receiver().worker(), wakeup, NULL);
        }
        return (m_err_handler_count == 0) && (receiver().get_num_eps() > 0);
    }

    void wait_for_reject(entity &e, bool wakeup)
    {
        ucs_time_t deadline = ucs::get_deadline();

        while ((e.get_rejected_cntr() == 0) &&
               (ucs_get_time() < deadline)) {
            check_events(sender().worker(), receiver().worker(), wakeup, NULL);
        }
        EXPECT_GT(deadline, ucs_get_time());
        EXPECT_EQ(1ul, e.get_rejected_cntr());
    }

    virtual ucp_ep_params_t get_ep_params()
    {
        ucp_ep_params_t ep_params = ucp_test::get_ep_params();
        ep_params.field_mask      |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                     UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                     UCP_EP_PARAM_FIELD_USER_DATA;
        /* The error handling requirement is needed since we need to take
         * care of a case where the client gets an error. In case ucp needs to
         * handle a large worker address but neither ud nor ud_x are present */
        ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
        ep_params.err_handler.cb   = err_handler_cb;
        ep_params.err_handler.arg  = NULL;
        ep_params.user_data        = reinterpret_cast<void*>(this);
        return ep_params;
    }

    void client_ep_connect(const struct sockaddr *connect_addr)
    {
        ucp_ep_params_t ep_params = get_ep_params();
        ep_params.field_mask      |= UCP_EP_PARAM_FIELD_FLAGS |
                                     UCP_EP_PARAM_FIELD_SOCK_ADDR;
        ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
        ep_params.sockaddr.addr    = connect_addr;
        ep_params.sockaddr.addrlen = sizeof(*connect_addr);
        sender().connect(&receiver(), ep_params);
    }

    void connect_and_send_recv(const struct sockaddr *connect_addr, bool wakeup)
    {
        {
            scoped_log_handler slh(detect_error_logger);
            client_ep_connect(connect_addr);
            if (!wait_for_server_ep(wakeup)) {
                UCS_TEST_SKIP_R("cannot connect to server");
            }
        }

        send_recv(sender(), receiver(),
                  (GetParam().variant == CONN_REQ_STREAM) ? SEND_RECV_STREAM :
                  SEND_RECV_TAG, wakeup, cb_type());
    }

    void connect_and_reject(const struct sockaddr *connect_addr, bool wakeup)
    {
        {
            scoped_log_handler slh(detect_error_logger);
            client_ep_connect(connect_addr);
            /* Check reachability with tagged send */
            send_recv(sender(), receiver(), SEND_RECV_TAG, wakeup,
                      ucp_test_base::entity::LISTEN_CB_REJECT);
        }
        wait_for_reject(receiver(), wakeup);
        wait_for_reject(sender(),   wakeup);
    }

    void listen_and_communicate(ucp_test_base::entity::listen_cb_type_t cb_type,
                                bool wakeup)
    {
        UCS_TEST_MESSAGE << "Testing "
                         << ucs::sockaddr_to_str(test_addr.get_sock_addr_ptr());

        start_listener(cb_type, test_addr.get_sock_addr_ptr(), test_addr.get_addr_size());
        connect_and_send_recv(test_addr.get_sock_addr_ptr(), wakeup);
    }

    void listen_and_reject(ucp_test_base::entity::listen_cb_type_t cb_type,
                           bool wakeup)
    {
        UCS_TEST_MESSAGE << "Testing "
                         << ucs::sockaddr_to_str(test_addr.get_sock_addr_ptr());

        start_listener(cb_type, test_addr.get_sock_addr_ptr(), test_addr.get_addr_size());
        connect_and_reject(test_addr.get_sock_addr_ptr(), wakeup);
    }


    static void err_handler_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
        test_ucp_sockaddr *self = reinterpret_cast<test_ucp_sockaddr*>(arg);
        ucp_test::err_handler_cb(static_cast<ucp_test *>(self), ep, status);

        if (status == UCS_ERR_REJECTED) {
            entity *e = self->get_entity_by_ep(ep);
            if (e != NULL) {
                e->inc_rejected_cntr();
                return;
            }
        }

        /* The current expected errors are only from the err_handle test
         * and from transports where the worker address is too long but ud/ud_x
         * are not present, or ud/ud_x are present but their addresses are too
         * long as well */
        if (status != UCS_ERR_UNREACHABLE) {
            UCS_TEST_ABORT("Error: " << ucs_status_string(status));
        }
    }

protected:
    ucp_test_base::entity::listen_cb_type_t cb_type() const {
        if ((GetParam().variant == CONN_REQ_TAG) ||
            (GetParam().variant == CONN_REQ_STREAM)) {
            return ucp_test_base::entity::LISTEN_CB_CONN;
        }
        return ucp_test_base::entity::LISTEN_CB_EP;
    }
};

UCS_TEST_P(test_ucp_sockaddr, listen) {
    listen_and_communicate(cb_type(), false);
}

UCS_TEST_P(test_ucp_sockaddr, listen_inaddr_any) {

    ucs::sock_addr_storage inaddr_any_listen_addr;
    size_t size;

    if (test_addr.get_sock_addr_ptr()->sa_family == AF_INET) {
        struct sockaddr_in sin;

        inaddr_any_ipv4(&sin, test_addr.get_port());
        size = sizeof(struct sockaddr_in);
        inaddr_any_listen_addr.set_sock_addr(*(struct sockaddr*)&sin, size);
    } else {
        EXPECT_EQ(test_addr.get_sock_addr_ptr()->sa_family, AF_INET6);
        struct sockaddr_in6 sin;

        inaddr_any_ipv6(&sin, test_addr.get_port());
        size = sizeof(struct sockaddr_in6);
        inaddr_any_listen_addr.set_sock_addr(*(struct sockaddr*)&sin, size);
    }

    UCS_TEST_MESSAGE << "Testing "
                     << ucs::sockaddr_to_str(inaddr_any_listen_addr.get_sock_addr_ptr());

    start_listener(cb_type(), (const struct sockaddr*)&inaddr_any_listen_addr,
                   size);
    connect_and_send_recv(test_addr.get_sock_addr_ptr(), false);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr, reject,
                     (GetParam().variant > 0)) {
    listen_and_reject(ucp_test_base::entity::LISTEN_CB_REJECT, false);
}

UCS_TEST_P(test_ucp_sockaddr, listener_query) {
    ucp_listener_attr_t listener_attr;
    ucs_status_t status;

    listener_attr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;

    UCS_TEST_MESSAGE << "Testing "
                     << ucs::sockaddr_to_str(test_addr.get_sock_addr_ptr());

    start_listener(cb_type(), test_addr.get_sock_addr_ptr(), test_addr.get_addr_size());
    status = ucp_listener_query(receiver().listenerh(), &listener_attr);
    EXPECT_UCS_OK(status);

    EXPECT_EQ(ucs_sockaddr_cmp(test_addr.get_sock_addr_ptr(),
                               (const struct sockaddr*)&listener_attr.sockaddr,
                               &status), 0);
    EXPECT_UCS_OK(status);
}

UCS_TEST_P(test_ucp_sockaddr, err_handle) {

    ucs::sock_addr_storage listen_addr(test_addr.to_ucs_sock_addr());
    ucs_status_t status = receiver().listen(cb_type(),
                                            listen_addr.get_sock_addr_ptr(),
                                            listen_addr.get_addr_size());
    if (status == UCS_ERR_UNREACHABLE) {
        UCS_TEST_SKIP_R("cannot listen to " + ucs::sockaddr_to_str(&listen_addr));
    }

    /* make the client try to connect to a non-existing port on the server side */
    listen_addr.set_port(1);

    {
        scoped_log_handler slh(wrap_errors_logger);
        client_ep_connect(listen_addr.get_sock_addr_ptr());
        /* allow for the unreachable event to arrive before restoring errors */
        wait_for_flag(&m_err_handler_count);
    }

    EXPECT_EQ(1, m_err_handler_count);
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

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_with_wakeup, reject,
                     (GetParam().variant > 0)) {
    listen_and_reject(ucp_test_base::entity::LISTEN_CB_REJECT, true);
}

UCP_INSTANTIATE_ALL_TEST_CASE(test_ucp_sockaddr_with_wakeup)


class test_ucp_sockaddr_with_rma_atomic : public test_ucp_sockaddr {
public:

    static ucp_params_t get_ctx_params() {
        ucp_params_t params = test_ucp_sockaddr::get_ctx_params();
        params.field_mask  |= UCP_PARAM_FIELD_FEATURES;
        params.features    |= UCP_FEATURE_RMA   |
                              UCP_FEATURE_AMO32 |
                              UCP_FEATURE_AMO64;
        return params;
    }
};

UCS_TEST_P(test_ucp_sockaddr_with_rma_atomic, wireup) {

    /* This test makes sure that the client-server flow works when the required
     * features are RMA/ATOMIC. With these features, need to make sure that
     * there is a lane for ucp-wireup (an am_lane should be created and used) */
    UCS_TEST_MESSAGE << "Testing " <<
        ucs::sockaddr_to_str(test_addr.get_sock_addr_ptr());

    start_listener(cb_type(), test_addr.get_sock_addr_ptr(), test_addr.get_addr_size());

    {
        scoped_log_handler slh(wrap_errors_logger);

        client_ep_connect(test_addr.get_sock_addr_ptr());

        /* allow the err_handler callback to be invoked if needed */
        if (!wait_for_server_ep(false)) {
            EXPECT_EQ(1, m_err_handler_count);
            UCS_TEST_SKIP_R("cannot connect to server");
        }

        EXPECT_EQ(0, m_err_handler_count);
        /* even if server EP is created, in case of long address, wireup will be
         * done later, need to communicate */
        send_recv(sender(), receiver(), (GetParam().variant == CONN_REQ_STREAM) ?
                  SEND_RECV_STREAM : SEND_RECV_TAG, false, cb_type());
    }
}

UCP_INSTANTIATE_ALL_TEST_CASE(test_ucp_sockaddr_with_rma_atomic)
