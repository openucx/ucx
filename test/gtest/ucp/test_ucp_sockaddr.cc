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
        CONN_REQ_TAG = DEFAULT_PARAM_VARIANT + 1,     /* Accepting by ucp_conn_request_h,
                                                         send/recv by TAG API */
        CONN_REQ_STREAM                               /* Accepting by ucp_conn_request_h,
                                                         send/recv by STREAM API */
    };

    enum {
        TEST_MODIFIER_MASK      = UCS_MASK(16),
        TEST_MODIFIER_MT        = UCS_BIT(16),
        TEST_MODIFIER_CM        = UCS_BIT(17)
    };

    enum {
        SEND_DIRECTION_C2S  = UCS_BIT(0), /* send data from client to server */
        SEND_DIRECTION_S2C  = UCS_BIT(1), /* send data from server to client */
        SEND_DIRECTION_BIDI = SEND_DIRECTION_C2S | SEND_DIRECTION_S2C /* bidirectional send */
    };

    typedef enum {
        SEND_RECV_TAG,
        SEND_RECV_STREAM
    } send_recv_type_t;

    ucs::sock_addr_storage m_test_addr;

    void init() {
        if (GetParam().variant & TEST_MODIFIER_CM) {
            modify_config("SOCKADDR_CM_ENABLE", "yes");
        }
        get_sockaddr();
        ucp_test::init();
        skip_loopback();
    }

    static void
    enum_test_params_with_modifier(const ucp_params_t& ctx_params,
                                      const std::string& name,
                                      const std::string& test_case_name,
                                      const std::string& tls,
                                      std::vector<ucp_test_param> &result,
                                      unsigned modifier)
    {
        generate_test_params_variant(ctx_params, name, test_case_name, tls,
                                     modifier, result, SINGLE_THREAD);
        generate_test_params_variant(ctx_params, name, test_case_name, tls,
                                     modifier | TEST_MODIFIER_MT, result,
                                     MULTI_THREAD_WORKER);
    }

    static std::vector<ucp_test_param>
    enum_test_params(const ucp_params_t& ctx_params,
                     const std::string& name,
                     const std::string& test_case_name,
                     const std::string& tls)
    {
        std::vector<ucp_test_param> result =
            ucp_test::enum_test_params(ctx_params, name, test_case_name, tls);

        enum_test_params_with_modifier(ctx_params, name, test_case_name, tls,
                                       result, CONN_REQ_TAG);
        enum_test_params_with_modifier(ctx_params, name, test_case_name, tls,
                                       result, CONN_REQ_TAG | TEST_MODIFIER_CM);
        enum_test_params_with_modifier(ctx_params, name, test_case_name, tls,
                                       result, CONN_REQ_STREAM);
        enum_test_params_with_modifier(ctx_params, name, test_case_name, tls,
                                       result, CONN_REQ_STREAM | TEST_MODIFIER_CM);
        return result;
    }

    static ucs_log_func_rc_t
    detect_error_logger(const char *file, unsigned line, const char *function,
                        ucs_log_level_t level,
                        const ucs_log_component_config_t *comp_conf,
                        const char *message, va_list ap)
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

    void get_sockaddr() {
        std::vector<ucs::sock_addr_storage> saddrs;
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
                saddrs.push_back(ucs::sock_addr_storage());
                status = ucs_sockaddr_sizeof(ifa->ifa_addr, &size);
                ASSERT_UCS_OK(status);
                saddrs.back().set_sock_addr(*ifa->ifa_addr, size);
                saddrs.back().set_port(0); /* listen on any port then update */
            }
        }

        freeifaddrs(ifaddrs);

        if (saddrs.empty()) {
            UCS_TEST_SKIP_R("No interface for testing");
        }

        static const std::string dc_tls[] = { "dc", "dc_x", "ib" };

        bool has_dc = has_any_transport(
            std::vector<std::string>(dc_tls, dc_tls + ucs_array_size(dc_tls)));

        /* FIXME: select random interface, except for DC transport, which do not
                  yet support having different gid_index for different UCT
                  endpoints on same iface */
        int saddr_idx = has_dc ? 0 : (ucs::rand() % saddrs.size());
        m_test_addr   = saddrs[saddr_idx];
    }

    void start_listener(ucp_test_base::entity::listen_cb_type_t cb_type)
    {
        ucs_time_t deadline = ucs::get_deadline();
        ucs_status_t status;

        do {
            status = receiver().listen(cb_type, m_test_addr.get_sock_addr_ptr(),
                                       m_test_addr.get_addr_size(),
                                       get_server_ep_params());
        } while ((status == UCS_ERR_BUSY) && (ucs_get_time() < deadline));

        if (status == UCS_ERR_UNREACHABLE) {
            UCS_TEST_SKIP_R("cannot listen to " + m_test_addr.to_str());
        }

        ASSERT_UCS_OK(status);
        ucp_listener_attr_t attr;
        uint16_t            port;

        attr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
        ASSERT_UCS_OK(ucp_listener_query(receiver().listenerh(), &attr));
        ASSERT_UCS_OK(ucs_sockaddr_get_port(
                        (const struct sockaddr *)&attr.sockaddr, &port));
        m_test_addr.set_port(port);
        UCS_TEST_MESSAGE << "server listening on " << m_test_addr.to_str();
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
            ASSERT_EQ(1,       ep_count);
            EXPECT_EQ(to.ep(), poll_eps.ep);
            EXPECT_EQ(&to,     poll_eps.user_data);

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

        while ((receiver().get_num_eps() == 0) &&
               (sender().get_err_num() == 0) && (ucs_get_time() < deadline)) {
            check_events(sender().worker(), receiver().worker(), wakeup, NULL);
        }

        return (sender().get_err_num() == 0) && (receiver().get_num_eps() > 0);
    }

    void wait_for_reject(entity &e, bool wakeup)
    {
        ucs_time_t deadline = ucs::get_deadline();

        while ((e.get_err_num_rejected() == 0) && (ucs_get_time() < deadline)) {
            check_events(sender().worker(), receiver().worker(), wakeup, NULL);
        }

        EXPECT_GT(deadline, ucs_get_time());
        EXPECT_EQ(1ul, e.get_err_num_rejected());
    }

    virtual ucp_ep_params_t get_ep_params()
    {
        ucp_ep_params_t ep_params = ucp_test::get_ep_params();
        ep_params.field_mask      |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                     UCP_EP_PARAM_FIELD_ERR_HANDLER;
        /* The error handling requirement is needed since we need to take
         * care of a case where the client gets an error. In case ucp needs to
         * handle a large worker address but neither ud nor ud_x are present */
        ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
        ep_params.err_handler.cb   = err_handler_cb;
        ep_params.err_handler.arg  = NULL;
        return ep_params;
    }

    virtual ucp_ep_params_t get_server_ep_params() {
        return get_ep_params();
    }

    void client_ep_connect()
    {
        ucp_ep_params_t ep_params = get_ep_params();
        ep_params.field_mask      |= UCP_EP_PARAM_FIELD_FLAGS |
                                     UCP_EP_PARAM_FIELD_SOCK_ADDR |
                                     UCP_EP_PARAM_FIELD_USER_DATA;
        ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
        ep_params.sockaddr.addr    = m_test_addr.get_sock_addr_ptr();
        ep_params.sockaddr.addrlen = m_test_addr.get_addr_size();
        ep_params.user_data        = &sender();
        sender().connect(&receiver(), ep_params);
    }

    void connect_and_send_recv(bool wakeup, uint64_t flags)
    {
        {
            scoped_log_handler slh(detect_error_logger);
            client_ep_connect();
            if (!wait_for_server_ep(wakeup)) {
                UCS_TEST_SKIP_R("cannot connect to server");
            }
        }

        if (flags & SEND_DIRECTION_C2S) {
            send_recv(sender(), receiver(), send_recv_type(), wakeup,
                      cb_type());
        }

        if (flags & SEND_DIRECTION_S2C) {
            send_recv(receiver(), sender(), send_recv_type(), wakeup,
                      cb_type());
        }
    }

    void connect_and_reject(bool wakeup)
    {
        {
            scoped_log_handler slh(detect_error_logger);
            client_ep_connect();
            /* Check reachability with tagged send */
            send_recv(sender(), receiver(), SEND_RECV_TAG, wakeup,
                      ucp_test_base::entity::LISTEN_CB_REJECT);
        }
        wait_for_reject(receiver(), wakeup);
        wait_for_reject(sender(),   wakeup);
    }

    void listen_and_communicate(bool wakeup, uint64_t flags)
    {
        UCS_TEST_MESSAGE << "Testing " << m_test_addr.to_str();

        start_listener(cb_type());
        connect_and_send_recv(wakeup, flags);
    }

    void listen_and_reject(bool wakeup)
    {
        UCS_TEST_MESSAGE << "Testing " << m_test_addr.to_str();

        start_listener(ucp_test_base::entity::LISTEN_CB_REJECT);
        connect_and_reject(wakeup);
    }

    void one_sided_disconnect(entity &e, enum ucp_ep_close_mode mode) {
        void *req           = e.disconnect_nb(0, 0, mode);
        ucs_time_t deadline = ucs_time_from_sec(10.0) + ucs_get_time();
        while (!is_request_completed(req) && (ucs_get_time() < deadline)) {
            /* TODO: replace the progress() with e().progress() when
                     async progress is implemented. */
            progress();
        };

        e.close_ep_req_free(req);
    }

    void concurrent_disconnect(enum ucp_ep_close_mode mode) {
        ASSERT_EQ(2ul, entities().size());
        ASSERT_EQ(1, sender().get_num_workers());
        ASSERT_EQ(1, sender().get_num_eps());
        ASSERT_EQ(1, receiver().get_num_workers());
        ASSERT_EQ(1, receiver().get_num_eps());

        void *sender_ep_close_req   = sender().disconnect_nb(0, 0, mode);
        void *receiver_ep_close_req = receiver().disconnect_nb(0, 0, mode);

        ucs_time_t deadline = ucs::get_deadline();
        while ((!is_request_completed(sender_ep_close_req) ||
                !is_request_completed(receiver_ep_close_req)) &&
               (ucs_get_time() < deadline)) {
            progress();
        }

        sender().close_ep_req_free(sender_ep_close_req);
        receiver().close_ep_req_free(receiver_ep_close_req);
    }

    static void err_handler_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
        ucp_test::err_handler_cb(arg, ep, status);

        /* The current expected errors are only from the err_handle test
         * and from transports where the worker address is too long but ud/ud_x
         * are not present, or ud/ud_x are present but their addresses are too
         * long as well, in addition we can get disconnect events during test
         * teardown.
         */
        switch (status) {
        case UCS_ERR_REJECTED:
        case UCS_ERR_UNREACHABLE:
        case UCS_ERR_CONNECTION_RESET:
            UCS_TEST_MESSAGE << "ignoring error " <<ucs_status_string(status)
                             << " on endpoint " << ep;
            return;
        default:
            UCS_TEST_ABORT("Error: " << ucs_status_string(status));
        }
    }

protected:
    ucp_test_base::entity::listen_cb_type_t cb_type() const {
        const int variant = (GetParam().variant & TEST_MODIFIER_MASK);
        if ((variant == CONN_REQ_TAG) || (variant == CONN_REQ_STREAM)) {
            return ucp_test_base::entity::LISTEN_CB_CONN;
        }
        return ucp_test_base::entity::LISTEN_CB_EP;
    }

    send_recv_type_t send_recv_type() const {
        switch (GetParam().variant & TEST_MODIFIER_MASK) {
        case CONN_REQ_STREAM:
            return SEND_RECV_STREAM;
        case CONN_REQ_TAG:
            /* fallthrough */
        default:
            return SEND_RECV_TAG;
        }
    }

    bool nonparameterized_test() const {
        return (GetParam().variant != DEFAULT_PARAM_VARIANT) &&
               (GetParam().variant != (CONN_REQ_TAG | TEST_MODIFIER_CM));
    }

    bool no_close_protocol() const {
        return !(GetParam().variant & TEST_MODIFIER_CM);
    }
};

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr, listen, no_close_protocol()) {
    listen_and_communicate(false, 0);
}

UCS_TEST_P(test_ucp_sockaddr, listen_c2s) {
    listen_and_communicate(false, SEND_DIRECTION_C2S);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr, listen_s2c, no_close_protocol()) {
    listen_and_communicate(false, SEND_DIRECTION_S2C);
}

UCS_TEST_P(test_ucp_sockaddr, listen_bidi) {
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr, onesided_disconnect,
                     no_close_protocol()) {
    listen_and_communicate(false, 0);
    one_sided_disconnect(sender(), UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr, onesided_disconnect_c2s) {
    listen_and_communicate(false, SEND_DIRECTION_C2S);
    one_sided_disconnect(sender(), UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr, onesided_disconnect_s2c,
                     no_close_protocol()) {
    listen_and_communicate(false, SEND_DIRECTION_S2C);
    one_sided_disconnect(sender(), UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr, onesided_disconnect_bidi) {
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
    one_sided_disconnect(sender(), UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr, concurrent_disconnect,
                     no_close_protocol()) {
    listen_and_communicate(false, 0);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr, concurrent_disconnect_c2s,
                     no_close_protocol()) {
    listen_and_communicate(false, SEND_DIRECTION_C2S);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr, concurrent_disconnect_s2c,
                     no_close_protocol()) {
    listen_and_communicate(false, SEND_DIRECTION_S2C);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr, concurrent_disconnect_bidi) {
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr, concurrent_disconnect_force,
                     no_close_protocol()) {
    listen_and_communicate(false, 0);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FORCE);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr, concurrent_disconnect_force_c2s,
                     no_close_protocol()) {
    listen_and_communicate(false, SEND_DIRECTION_C2S);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FORCE);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr, concurrent_disconnect_force_s2c,
                     no_close_protocol()) {
    listen_and_communicate(false, SEND_DIRECTION_S2C);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FORCE);
}

UCS_TEST_P(test_ucp_sockaddr, concurrent_disconnect_force_bidi) {
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FORCE);
}

UCS_TEST_P(test_ucp_sockaddr, listen_inaddr_any) {
    /* save testing address */
    ucs::sock_addr_storage test_addr(m_test_addr);
    m_test_addr.reset_to_any();

    UCS_TEST_MESSAGE << "Testing " << m_test_addr.to_str();

    start_listener(cb_type());
    /* get the actual port which was selected by listener */
    test_addr.set_port(m_test_addr.get_port());
    /* restore address */
    m_test_addr = test_addr;
    connect_and_send_recv(false, SEND_DIRECTION_C2S);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr, reject, nonparameterized_test()) {
    listen_and_reject(false);
}

UCS_TEST_P(test_ucp_sockaddr, listener_query) {
    ucp_listener_attr_t listener_attr;
    ucs_status_t status;

    listener_attr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;

    UCS_TEST_MESSAGE << "Testing " << m_test_addr.to_str();

    start_listener(cb_type());
    status = ucp_listener_query(receiver().listenerh(), &listener_attr);
    EXPECT_UCS_OK(status);

    EXPECT_EQ(m_test_addr, listener_attr.sockaddr);
}

UCS_TEST_P(test_ucp_sockaddr, err_handle) {

    ucs::sock_addr_storage listen_addr(m_test_addr.to_ucs_sock_addr());
    ucs_status_t status = receiver().listen(cb_type(),
                                            m_test_addr.get_sock_addr_ptr(),
                                            m_test_addr.get_addr_size(),
                                            get_server_ep_params());
    if (status == UCS_ERR_UNREACHABLE) {
        UCS_TEST_SKIP_R("cannot listen to " + m_test_addr.to_str());
    }

    /* make the client try to connect to a non-existing port on the server side */
    m_test_addr.set_port(1);

    {
        scoped_log_handler slh(wrap_errors_logger);
        client_ep_connect();
        /* allow for the unreachable event to arrive before restoring errors */
        wait_for_flag(&sender().get_err_num());
    }

    EXPECT_EQ(1u, sender().get_err_num());
}

UCP_INSTANTIATE_ALL_TEST_CASE(test_ucp_sockaddr)

class test_ucp_sockaddr_destroy_ep_on_err : public test_ucp_sockaddr {
public:
    test_ucp_sockaddr_destroy_ep_on_err() {
        /* Set small TL timeouts to reduce testing time */
        m_env.push_back(new ucs::scoped_setenv("UCX_RC_TIMEOUT",     "10ms"));
        m_env.push_back(new ucs::scoped_setenv("UCX_RC_RNR_TIMEOUT", "10ms"));
        m_env.push_back(new ucs::scoped_setenv("UCX_RC_RETRY_COUNT", "2"));
    }

    virtual ucp_ep_params_t get_server_ep_params() {
        ucp_ep_params_t params = test_ucp_sockaddr::get_server_ep_params();

        params.field_mask      |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLER        |
                                 UCP_EP_PARAM_FIELD_USER_DATA;
        params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
        params.err_handler.cb   = err_handler_cb;
        params.err_handler.arg  = NULL;
        params.user_data        = &receiver();
        return params;
    }

    static void err_handler_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
        test_ucp_sockaddr::err_handler_cb(arg, ep, status);
        entity *e = reinterpret_cast<entity *>(arg);
        e->disconnect_nb(0, 0, UCP_EP_CLOSE_MODE_FORCE);
    }

private:
    ucs::ptr_vector<ucs::scoped_setenv> m_env;
};

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_destroy_ep_on_err, empty,
                     no_close_protocol()) {
    listen_and_communicate(false, 0);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_destroy_ep_on_err, s2c,
                     no_close_protocol()) {
    listen_and_communicate(false, SEND_DIRECTION_S2C);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_destroy_ep_on_err, c2s,
                     no_close_protocol()) {
    listen_and_communicate(false, SEND_DIRECTION_C2S);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_destroy_ep_on_err, bidi,
                     no_close_protocol()) {
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_destroy_ep_on_err, onesided_client_cforce,
                     no_close_protocol()) {
    listen_and_communicate(false, 0);
    scoped_log_handler slh(wrap_errors_logger);
    one_sided_disconnect(sender(),   UCP_EP_CLOSE_MODE_FORCE);
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_destroy_ep_on_err, onesided_c2s_cforce,
                     no_close_protocol()) {
    listen_and_communicate(false, SEND_DIRECTION_C2S);
    scoped_log_handler slh(wrap_errors_logger);
    one_sided_disconnect(sender(),   UCP_EP_CLOSE_MODE_FORCE);
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_destroy_ep_on_err, onesided_s2c_cforce,
                     no_close_protocol()) {
    listen_and_communicate(false, SEND_DIRECTION_S2C);
    scoped_log_handler slh(wrap_errors_logger);
    one_sided_disconnect(sender(),   UCP_EP_CLOSE_MODE_FORCE);
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_destroy_ep_on_err, onesided_bidi_cforce,
                     no_close_protocol()) {
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
    scoped_log_handler slh(wrap_errors_logger);
    one_sided_disconnect(sender(),   UCP_EP_CLOSE_MODE_FORCE);
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_destroy_ep_on_err, onesided_client_sforce,
                     no_close_protocol()) {
    listen_and_communicate(false, 0);
    scoped_log_handler slh(wrap_errors_logger);
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FORCE);
    one_sided_disconnect(sender(),   UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_destroy_ep_on_err, onesided_c2s_sforce,
                     no_close_protocol()) {
    listen_and_communicate(false, SEND_DIRECTION_C2S);
    scoped_log_handler slh(wrap_errors_logger);
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FORCE);
    one_sided_disconnect(sender(),   UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_destroy_ep_on_err, onesided_s2c_sforce,
                     no_close_protocol()) {
    listen_and_communicate(false, SEND_DIRECTION_S2C);
    scoped_log_handler slh(wrap_errors_logger);
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FORCE);
    one_sided_disconnect(sender(),   UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_destroy_ep_on_err, onesided_bidi_sforce,
                     no_close_protocol()) {
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
    scoped_log_handler slh(wrap_errors_logger);
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FORCE);
    one_sided_disconnect(sender(),   UCP_EP_CLOSE_MODE_FLUSH);
}

UCP_INSTANTIATE_ALL_TEST_CASE(test_ucp_sockaddr_destroy_ep_on_err)

class test_ucp_sockaddr_with_wakeup : public test_ucp_sockaddr {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = test_ucp_sockaddr::get_ctx_params();
        params.features    |= UCP_FEATURE_WAKEUP;
        return params;
    }
};

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_with_wakeup, wakeup,
                     no_close_protocol()) {
    listen_and_communicate(true, 0);
}

UCS_TEST_P(test_ucp_sockaddr_with_wakeup, wakeup_c2s) {
    listen_and_communicate(true, SEND_DIRECTION_C2S);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_with_wakeup, wakeup_s2c,
                     no_close_protocol()) {
    listen_and_communicate(true, SEND_DIRECTION_S2C);
}

UCS_TEST_P(test_ucp_sockaddr_with_wakeup, wakeup_bidi) {
    listen_and_communicate(true, SEND_DIRECTION_BIDI);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_with_wakeup, reject,
                     nonparameterized_test()) {
    listen_and_reject(true);
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
    UCS_TEST_MESSAGE << "Testing " << m_test_addr.to_str();

    start_listener(cb_type());
    {
        scoped_log_handler slh(wrap_errors_logger);

        client_ep_connect();

        /* allow the err_handler callback to be invoked if needed */
        if (!wait_for_server_ep(false)) {
            EXPECT_EQ(1ul, sender().get_err_num());
            UCS_TEST_SKIP_R("cannot connect to server");
        }

        EXPECT_EQ(0ul, sender().get_err_num());
        /* even if server EP is created, in case of long address, wireup will be
         * done later, need to communicate */
        send_recv(sender(), receiver(), send_recv_type(), false, cb_type());
    }
}

UCP_INSTANTIATE_ALL_TEST_CASE(test_ucp_sockaddr_with_rma_atomic)


class test_ucp_sockaddr_protocols : public test_ucp_sockaddr {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = test_ucp_sockaddr::get_ctx_params();
        params.field_mask  |= UCP_PARAM_FIELD_FEATURES;
        params.features    |= UCP_FEATURE_RMA | UCP_FEATURE_AM;
        /* Atomics not supported for now because need to emulate the case
         * of using different device than the one selected by default on the
         * worker for atomic operations */
        return params;
    }

    static std::vector<ucp_test_param>
    enum_test_params(const ucp_params_t& ctx_params,
                     const std::string& name,
                     const std::string& test_case_name,
                     const std::string& tls)
    {
        std::vector<ucp_test_param> result;
        enum_test_params_with_modifier(ctx_params, name, test_case_name, tls,
                                       result, TEST_MODIFIER_CM);
        return result;
    }

    virtual void init() {
        test_ucp_sockaddr::init();
        start_listener(cb_type());
        client_ep_connect();
    }

    void get_nb(std::string& send_buf, std::string& recv_buf, ucp_rkey_h rkey,
                std::vector<void*>& reqs)
    {
         reqs.push_back(ucp_get_nb(sender().ep(), &send_buf[0], send_buf.size(),
                                   (uintptr_t)&recv_buf[0], rkey, scomplete_cb));
    }

    void put_nb(std::string& send_buf, std::string& recv_buf, ucp_rkey_h rkey,
                std::vector<void*>& reqs)
    {
        reqs.push_back(ucp_put_nb(sender().ep(), &send_buf[0], send_buf.size(),
                                  (uintptr_t)&recv_buf[0], rkey, scomplete_cb));
        reqs.push_back(ucp_ep_flush_nb(sender().ep(), 0, scomplete_cb));
    }

protected:
    typedef void (test_ucp_sockaddr_protocols::*rma_nb_func_t)(
                    std::string&, std::string&, ucp_rkey_h, std::vector<void*>&);

    void compare_buffers(std::string& send_buf, std::string& recv_buf)
    {
        EXPECT_TRUE(send_buf == recv_buf)
            << "send_buf: '" << ucs::compact_string(send_buf, 20) << "', "
            << "recv_buf: '" << ucs::compact_string(recv_buf, 20) << "'";
    }

    void test_tag_send_recv(size_t size, bool is_exp, bool is_sync = false)
    {
        std::string send_buf(size, 'x');
        std::string recv_buf(size, 'y');

        void *rreq = NULL, *sreq = NULL;

        if (is_exp) {
            rreq = ucp_tag_recv_nb(receiver().worker(), &recv_buf[0], size,
                                   ucp_dt_make_contig(1), 0, 0, rtag_complete_cb);
        }

        if (is_sync) {
            sreq = ucp_tag_send_sync_nb(sender().ep(), &send_buf[0], size,
                                        ucp_dt_make_contig(1), 0, scomplete_cb);
        } else {
            sreq = ucp_tag_send_nb(sender().ep(), &send_buf[0], size,
                                   ucp_dt_make_contig(1), 0, scomplete_cb);
        }

        if (!is_exp) {
            short_progress_loop();
            rreq = ucp_tag_recv_nb(receiver().worker(), &recv_buf[0], size,
                                   ucp_dt_make_contig(1), 0, 0, rtag_complete_cb);
        }

        wait(sreq);
        wait(rreq);

        compare_buffers(send_buf, recv_buf);
    }

    void wait_for_server_ep()
    {
        if (!test_ucp_sockaddr::wait_for_server_ep(false)) {
            UCS_TEST_ABORT("server endpoint is not created");
        }
    }

    void test_stream_send_recv(size_t size, bool is_exp)
    {
        std::string send_buf(size, 'x');
        std::string recv_buf(size, 'y');
        size_t recv_length;
        void *rreq, *sreq;

        if (is_exp) {
            wait_for_server_ep();
            rreq = ucp_stream_recv_nb(receiver().ep(), &recv_buf[0], size,
                                      ucp_dt_make_contig(1), rstream_complete_cb,
                                      &recv_length, UCP_STREAM_RECV_FLAG_WAITALL);
            sreq = ucp_stream_send_nb(sender().ep(), &send_buf[0], size,
                                      ucp_dt_make_contig(1), scomplete_cb, 0);
        } else {
            sreq = ucp_stream_send_nb(sender().ep(), &send_buf[0], size,
                                   ucp_dt_make_contig(1), scomplete_cb, 0);
            short_progress_loop();
            wait_for_server_ep();
            rreq = ucp_stream_recv_nb(receiver().ep(), &recv_buf[0], size,
                                      ucp_dt_make_contig(1), rstream_complete_cb,
                                      &recv_length, UCP_STREAM_RECV_FLAG_WAITALL);
        }

        wait(sreq);
        wait(rreq);

        compare_buffers(send_buf, recv_buf);
    }

    void register_mem(entity* initiator, entity* target, void *buffer,
                      size_t length, ucp_mem_h *memh_p, ucp_rkey_h *rkey_p)
    {
        ucp_mem_map_params_t params = {0};
        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        params.address    = buffer;
        params.length     = length;

        ucs_status_t status = ucp_mem_map(target->ucph(), &params, memh_p);
        ASSERT_UCS_OK(status);

        void *rkey_buffer;
        size_t rkey_buffer_size;
        status = ucp_rkey_pack(target->ucph(), *memh_p, &rkey_buffer,
                               &rkey_buffer_size);
        ASSERT_UCS_OK(status);

        status = ucp_ep_rkey_unpack(initiator->ep(), rkey_buffer, rkey_p);
        ASSERT_UCS_OK(status);

        ucp_rkey_buffer_release(rkey_buffer);
    }

    void test_rma(size_t size, rma_nb_func_t rma_func)
    {
        std::string send_buf(size, 'x');
        std::string recv_buf(size, 'y');

        ucp_mem_h memh;
        ucp_rkey_h rkey;

        register_mem(&sender(), &receiver(), &recv_buf[0], size, &memh, &rkey);

        std::vector<void*> reqs;
        (this->*rma_func)(send_buf, recv_buf, rkey, reqs);

        while (!reqs.empty()) {
            wait(reqs.back());
            reqs.pop_back();
        }

        compare_buffers(send_buf, recv_buf);

        ucp_rkey_destroy(rkey);
        ucs_status_t status = ucp_mem_unmap(receiver().ucph(), memh);
        ASSERT_UCS_OK(status);
    }

    void test_am_send_recv(size_t size)
    {
        std::string sb(size, 'x');

        bool am_received = false;
        ucp_worker_set_am_handler(receiver().worker(), 0,
                                  rx_am_msg_cb, &am_received, 0);

        ucs_status_ptr_t sreq = ucp_am_send_nb(sender().ep(), 0, &sb[0], size,
                                               ucp_dt_make_contig(1),
                                               scomplete_cb, 0);
        wait(sreq);
        wait_for_flag(&am_received);
        EXPECT_TRUE(am_received);

        ucp_worker_set_am_handler(receiver().worker(), 0, NULL, NULL, 0);
    }

private:
    static ucs_status_t rx_am_msg_cb(void *arg, void *data, size_t length,
                                     ucp_ep_h reply_ep, unsigned flags) {
        volatile bool *am_rx = reinterpret_cast<volatile bool*>(arg);
        EXPECT_FALSE(*am_rx);
        *am_rx = true;
        return UCS_OK;
    }
};

UCS_TEST_P(test_ucp_sockaddr_protocols, tag_zcopy_4k_exp,
           "ZCOPY_THRESH=2k", "RNDV_THRESH=inf")
{
    test_tag_send_recv(4 * UCS_KBYTE, true);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, tag_zcopy_64k_exp,
           "ZCOPY_THRESH=2k", "RNDV_THRESH=inf")
{
    test_tag_send_recv(64 * UCS_KBYTE, true);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, tag_zcopy_4k_exp_sync,
           "ZCOPY_THRESH=2k", "RNDV_THRESH=inf")
{
    test_tag_send_recv(4 * UCS_KBYTE, true, true);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, tag_zcopy_64k_exp_sync,
           "ZCOPY_THRESH=2k", "RNDV_THRESH=inf")
{
    test_tag_send_recv(64 * UCS_KBYTE, true, true);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, tag_rndv_exp, "RNDV_THRESH=10k")
{
    test_tag_send_recv(64 * UCS_KBYTE, true);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, tag_zcopy_4k_unexp,
           "ZCOPY_THRESH=2k", "RNDV_THRESH=inf")
{
    test_tag_send_recv(4 * UCS_KBYTE, false);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, tag_zcopy_64k_unexp,
           "ZCOPY_THRESH=2k", "RNDV_THRESH=inf")
{
    test_tag_send_recv(64 * UCS_KBYTE, false);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, tag_zcopy_4k_unexp_sync,
           "ZCOPY_THRESH=2k", "RNDV_THRESH=inf")
{
    test_tag_send_recv(4 * UCS_KBYTE, false, true);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, tag_zcopy_64k_unexp_sync,
           "ZCOPY_THRESH=2k", "RNDV_THRESH=inf")
{
    test_tag_send_recv(64 * UCS_KBYTE, false, true);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, tag_rndv_unexp, "RNDV_THRESH=10k")
{
    test_tag_send_recv(64 * UCS_KBYTE, false);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, stream_bcopy_4k_exp, "ZCOPY_THRESH=inf")
{
    test_stream_send_recv(4 * UCS_KBYTE, true);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, stream_bcopy_4k_unexp,
           "ZCOPY_THRESH=inf")
{
    test_stream_send_recv(4 * UCS_KBYTE, false);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, stream_bcopy_64k_exp, "ZCOPY_THRESH=inf")
{
    test_stream_send_recv(64 * UCS_KBYTE, true);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, stream_bcopy_64k_unexp,
           "ZCOPY_THRESH=inf")
{
    test_stream_send_recv(64 * UCS_KBYTE, false);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, stream_zcopy_64k_exp, "ZCOPY_THRESH=2k")
{
    test_stream_send_recv(64 * UCS_KBYTE, true);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, stream_zcopy_64k_unexp,
           "ZCOPY_THRESH=2k")
{
    test_stream_send_recv(64 * UCS_KBYTE, false);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, get_bcopy_small)
{
    test_rma(8, &test_ucp_sockaddr_protocols::get_nb);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, get_bcopy, "ZCOPY_THRESH=inf")
{
    test_rma(64 * UCS_KBYTE, &test_ucp_sockaddr_protocols::get_nb);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, get_zcopy, "ZCOPY_THRESH=10k")
{
    test_rma(64 * UCS_KBYTE, &test_ucp_sockaddr_protocols::get_nb);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, put_bcopy_small)
{
    test_rma(8, &test_ucp_sockaddr_protocols::put_nb);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, put_bcopy, "ZCOPY_THRESH=inf")
{
    test_rma(64 * UCS_KBYTE, &test_ucp_sockaddr_protocols::put_nb);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, put_zcopy, "ZCOPY_THRESH=10k")
{
    test_rma(64 * UCS_KBYTE, &test_ucp_sockaddr_protocols::put_nb);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, am_short)
{
    test_am_send_recv(1);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, am_bcopy_1k, "ZCOPY_THRESH=inf")
{
    test_am_send_recv(1 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, am_bcopy_64k, "ZCOPY_THRESH=inf")
{
    test_am_send_recv(64 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, am_zcopy_1k, "ZCOPY_THRESH=512")
{
    test_am_send_recv(1 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, am_zcopy_64k, "ZCOPY_THRESH=512")
{
    test_am_send_recv(64 * UCS_KBYTE);
}


/* Only IB transports support CM for now
 * For DC case, allow fallback to UD if DC is not supported
 */
#define UCP_INSTANTIATE_CM_TEST_CASE(_test_case) \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, dcudx, "dc_x,ud") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, ud,    "ud_v") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, udx,   "ud_x") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, rc,    "rc_v") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, rcx,   "rc_x") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, ib,    "ib")

UCP_INSTANTIATE_CM_TEST_CASE(test_ucp_sockaddr_protocols)
