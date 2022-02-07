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
#include <atomic>
#include <memory>

extern "C" {
#include <uct/base/uct_worker.h>
#include <ucp/core/ucp_listener.h>
#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_request.inl>
#include <ucp/core/ucp_worker.h>
#include <ucp/wireup/wireup_cm.h>
}

#define UCP_INSTANTIATE_ALL_TEST_CASE(_test_case) \
        UCP_INSTANTIATE_TEST_CASE (_test_case) \
        UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, all, "all") \
        UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, shm, "shm") \
        UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, dc_ud, "dc_x,ud_v,ud_x,mm") \
        UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, dc_no_ud_ud_x, "dc_mlx5,mm") \
        /* dc_ud case is for testing handling of a large worker address on
         * UCT_IFACE_FLAG_CONNECT_TO_IFACE transports (dc_x) */
        /* dc_no_ud_ud_x case is for testing handling a large worker address
         * but with the lack of ud/ud_x transports, which would return an error
         * and skipped */

class test_ucp_sockaddr : public ucp_test {
public:
    enum {
        CONN_REQ_TAG  =  1, /* Accepting by ucp_conn_request_h,
                               send/recv by TAG API */
        CONN_REQ_STREAM     /* Accepting by ucp_conn_request_h,
                               send/recv by STREAM API */
    };

    enum {
        TEST_MODIFIER_MASK               = UCS_MASK(16),
        TEST_MODIFIER_MT                 = UCS_BIT(16),
        TEST_MODIFIER_CM_USE_ALL_DEVICES = UCS_BIT(17),
        TEST_MODIFIER_SA_DATA_V2         = UCS_BIT(18)
    };

    enum {
        SEND_DIRECTION_C2S  = UCS_BIT(0), /* send data from client to server */
        SEND_DIRECTION_S2C  = UCS_BIT(1), /* send data from server to client */
        SEND_DIRECTION_BIDI = SEND_DIRECTION_C2S | SEND_DIRECTION_S2C /* bidirectional send */
    };

    typedef enum {
        SEND_RECV_TAG,
        SEND_RECV_STREAM,
        SEND_RECV_AM
    } send_recv_type_t;

    ucs::sock_addr_storage m_test_addr;

    void init() {
        m_err_count = 0;
        modify_config("KEEPALIVE_INTERVAL", "10s");
        modify_config("CM_USE_ALL_DEVICES", cm_use_all_devices() ? "y" : "n");
        modify_config("SA_DATA_VERSION", sa_data_version_v2() ? "v2" : "v1");

        get_sockaddr();
        ucp_test::init();
        skip_loopback();
    }

    static void
    get_test_variants_mt(std::vector<ucp_test_variant>& variants, uint64_t features,
                         int modifier, const std::string& name) {
        add_variant_with_value(variants, features, modifier, name);
        add_variant_with_value(variants, features, modifier | TEST_MODIFIER_MT,
                               name + ",mt", MULTI_THREAD_WORKER);
    }

    static void
    get_test_variants_cm_mode(std::vector<ucp_test_variant>& variants, uint64_t features,
                              int modifier, const std::string& name)
    {
        get_test_variants_mt(variants, features,
                             modifier | TEST_MODIFIER_CM_USE_ALL_DEVICES, name);
        get_test_variants_mt(variants, features,
                             modifier | TEST_MODIFIER_CM_USE_ALL_DEVICES |
                             TEST_MODIFIER_SA_DATA_V2, name + ",sa_data_v2");
        get_test_variants_mt(variants, features, modifier, name + ",not_all_devs");
    }

    static void
    get_test_variants(std::vector<ucp_test_variant>& variants,
                      uint64_t features = UCP_FEATURE_TAG | UCP_FEATURE_STREAM) {
        get_test_variants_cm_mode(variants, features, CONN_REQ_TAG, "tag");
        get_test_variants_cm_mode(variants, features, CONN_REQ_STREAM, "stream");
    }

    static ucs_log_func_rc_t
    detect_warn_logger(const char *file, unsigned line, const char *function,
                       ucs_log_level_t level,
                       const ucs_log_component_config_t *comp_conf,
                       const char *message, va_list ap)
    {
        if (level == UCS_LOG_LEVEL_WARN) {
            std::string err_str = format_message(message, ap);
            if (err_str.find("failed to connect CM lane on device") !=
                std::string::npos) {
                UCS_TEST_MESSAGE << err_str;
                return UCS_LOG_FUNC_RC_STOP;
            }
        }
        return UCS_LOG_FUNC_RC_CONTINUE;
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
                stop_list.push_back("Connection reset by remote peer");
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

    int is_skip_interface(struct ifaddrs *ifa) {
        int skip = 0;

        if (!has_transport("tcp") && !has_transport("all") &&
            !ucs::is_rdmacm_netdev(ifa->ifa_name)) {
            /* IB transports require an IPoIB/RoCE interface since they
             * use rdmacm for connection establishment, which supports
             * only IPoIB IP addresses. therefore, if the interface
             * isn't as such, we continue to the next one. */
            skip = 1;
        } else if ((has_transport("tcp") || has_transport("all")) &&
                   (ifa->ifa_addr->sa_family == AF_INET6)) {
            /* the tcp transport (and 'all' which may fallback to tcp_sockcm)
             * can run either on an rdma-enabled interface (IPoIB/RoCE)
             * or any interface with IPv4 address because IPv6 isn't supported
             * by the tcp transport yet */
            skip = 1;
        }

        return skip;
    }

    void get_sockaddr() {
        std::vector<ucs::sock_addr_storage> saddrs;
        struct ifaddrs* ifaddrs;
        ucs_status_t status;
        size_t size;
        int ret = getifaddrs(&ifaddrs);
        ASSERT_EQ(ret, 0);

        for (struct ifaddrs *ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
            if (is_skip_interface(ifa) || !ucs::is_interface_usable(ifa)) {
                continue;
            }

            saddrs.push_back(ucs::sock_addr_storage());
            status = ucs_sockaddr_sizeof(ifa->ifa_addr, &size);
            ASSERT_UCS_OK(status);
            saddrs.back().set_sock_addr(*ifa->ifa_addr, size,
                                        ucs::is_rdmacm_netdev(ifa->ifa_name));
            saddrs.back().set_port(0); /* listen on any port then update */
        }

        freeifaddrs(ifaddrs);

        if (saddrs.empty()) {
            UCS_TEST_SKIP_R("No interface for testing");
        }

        static const std::string dc_tls[] = { "dc", "dc_x", "dc_mlx5", "ib" };

        bool has_dc = has_any_transport(
            std::vector<std::string>(dc_tls,
                                     dc_tls + ucs_static_array_size(dc_tls)));

        /* FIXME: select random interface, except for DC transport, which do not
                  yet support having different gid_index for different UCT
                  endpoints on same iface */
        int saddr_idx = has_dc ? 0 : (ucs::rand() % saddrs.size());
        m_test_addr   = saddrs[saddr_idx];
    }

    void start_listener(ucp_test_base::entity::listen_cb_type_t cb_type)
    {
        start_listener(cb_type, NULL);
    }

    void start_listener(ucp_test_base::entity::listen_cb_type_t cb_type,
                        ucp_listener_conn_handler_t *custom_cb)
    {
        ucs_time_t deadline = ucs::get_deadline();
        ucs_status_t status;

        do {
            status = receiver().listen(cb_type, m_test_addr.get_sock_addr_ptr(),
                                       m_test_addr.get_addr_size(),
                                       get_server_ep_params(), custom_cb, 0);
            if (m_test_addr.get_port() == 0) {
                /* any port can't be busy */
                break;
            }
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

    ucs_status_t create_listener_wrap_err(const ucp_listener_params_t &params,
                                          ucp_listener_h &listener)
    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        return ucp_listener_create(receiver().worker(), &params, &listener);
    }

    static void complete_err_handling_status_verify(ucs_status_t status)
    {
        EXPECT_TRUE(/* was successful */
                    (status == UCS_OK)                   ||
                    /* completed from error handling for EP */
                    (status == UCS_ERR_ENDPOINT_TIMEOUT) ||
                    (status == UCS_ERR_CONNECTION_RESET) ||
                    (status == UCS_ERR_CANCELED));
    }

    static void scomplete_cb(void *req, ucs_status_t status)
    {
        if ((status == UCS_OK) ||
            (status == UCS_ERR_UNREACHABLE) ||
            (status == UCS_ERR_REJECTED) ||
            (status == UCS_ERR_CANCELED) ||
            (status == UCS_ERR_CONNECTION_RESET)) {
            return;
        }
        UCS_TEST_ABORT("Error: " << ucs_status_string(status));
    }

    static void scomplete_cbx(void *req, ucs_status_t status, void *user_data)
    {
        ASSERT_EQ(NULL, user_data);
        scomplete_cb(req, status);
    }

    static void scomplete_always_ok_cbx(void *req, ucs_status_t status, void *user_data)
    {
        ASSERT_EQ(NULL, user_data);
        EXPECT_UCS_OK(status);
    }

    static void scomplete_reset_data_cbx(void *req, ucs_status_t status,
                                         void *user_data)
    {
        mem_buffer *send_buffer = reinterpret_cast<mem_buffer*>(user_data);
        send_buffer->pattern_fill(0, send_buffer->size());
    }

    static void scomplete_err_handling_cb(void *req, ucs_status_t status)
    {
        complete_err_handling_status_verify(status);
    }

    static void rtag_complete_cb(void *req, ucs_status_t status,
                                 ucp_tag_recv_info_t *info)
    {
        EXPECT_TRUE((status == UCS_OK) || (status == UCS_ERR_CANCELED) ||
                    (status == UCS_ERR_CONNECTION_RESET));
    }

    static void rtag_complete_cbx(void *req, ucs_status_t status,
                                  const ucp_tag_recv_info_t *info,
                                  void *user_data)
    {
        ASSERT_EQ(NULL, user_data);
        rtag_complete_cb(req, status, const_cast<ucp_tag_recv_info_t*>(info));
    }

    static void rtag_complete_always_ok_cbx(void *req, ucs_status_t status,
                                            const ucp_tag_recv_info_t *info,
                                            void *user_data)
    {
        ASSERT_EQ(NULL, user_data);
        EXPECT_UCS_OK(status);
    }

    static void rtag_complete_check_data_cbx(void *req, ucs_status_t status,
                                             const ucp_tag_recv_info_t *tag_info,
                                             void *user_data)
    {
        mem_buffer UCS_V_UNUSED *recv_buffer =
                reinterpret_cast<mem_buffer*>(user_data);

        if (status == UCS_OK) {
            recv_buffer->pattern_check(1, recv_buffer->size());
        }
    }

    static void rtag_complete_err_handling_cb(void *req, ucs_status_t status,
                                              ucp_tag_recv_info_t *info)
    {
        complete_err_handling_status_verify(status);
    }

    static void rstream_complete_cb(void *req, ucs_status_t status,
                                    size_t length)
    {
        EXPECT_TRUE((status == UCS_OK) || (status == UCS_ERR_CANCELED));
    }

    static void rstream_complete_cbx(void *req, ucs_status_t status,
                                     size_t length, void *user_data)
    {
        ASSERT_EQ(NULL, user_data);
        rstream_complete_cb(req, status, length);
    }

    bool check_send_status(ucs_status_t send_status, entity &receiver,
                           void* recv_req,
                           ucp_test_base::entity::listen_cb_type_t cb_type)
    {
        if (send_status == UCS_ERR_UNREACHABLE) {
            request_cancel(receiver, recv_req);
            /* Check if the error was completed due to the error handling flow.
             * If so, skip the test since a valid error occurred - the one expected
             * from the error handling flow - cases of failure to handle long worker
             * address or transport doesn't support the error handling requirement */
            UCS_TEST_SKIP_R("Skipping due to an unreachable destination"
                            " (unsupported feature or too long worker address or"
                            " no supported transport to send partial worker"
                            " address)");
        } else if ((send_status == UCS_ERR_REJECTED) &&
                   (cb_type == ucp_test_base::entity::LISTEN_CB_REJECT)) {
            request_cancel(receiver, recv_req);
            return false;
        } else {
            EXPECT_UCS_OK(send_status);
        }

        return true;
    }

    void* send(entity& from, const void *contig_buffer, size_t length,
               send_recv_type_t send_type, ucp_send_nbx_callback_t cb,
               void *user_data, size_t ep_index = 0)
    {
        ucp_request_param_t params;

        params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                              UCP_OP_ATTR_FIELD_USER_DATA;
        params.cb.send      = cb;
        params.user_data    = user_data;

        ucp_ep_h ep = from.ep(0, ep_index);
        if (send_type == SEND_RECV_TAG) {
            return ucp_tag_send_nbx(ep, contig_buffer, length, 1, &params);
        } else if (send_type == SEND_RECV_STREAM) {
            return ucp_stream_send_nbx(ep, contig_buffer, length, &params);
        } else if (send_type == SEND_RECV_AM) {
            return ucp_am_send_nbx(ep, 0, NULL, 0, contig_buffer,
                                   length, &params);
        }

        UCS_TEST_ABORT("unsupported communication type " << send_type);
    }

    void* recv(entity& to, void *contig_buffer, size_t length,
               ucp_tag_recv_nbx_callback_t cb, void *user_data)
    {
        ucp_request_param_t params = {};

        params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                              UCP_OP_ATTR_FIELD_USER_DATA;
        params.user_data    = user_data;
        params.cb.recv      = cb;
        return ucp_tag_recv_nbx(to.worker(), contig_buffer, length, 1, 0,
                                &params);
    }

    void* recv(entity& to, void *contig_buffer, size_t length,
               ucp_tag_message_h message, ucp_tag_recv_nbx_callback_t cb,
               void *user_data)
    {
        ucp_request_param_t params = {};

        params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                              UCP_OP_ATTR_FIELD_USER_DATA;
        params.user_data    = user_data;
        params.cb.recv      = cb;
        return ucp_tag_msg_recv_nbx(to.worker(), contig_buffer, length,
                                    message, &params);
    }

    void* recv(entity& to, void *contig_buffer, size_t length,
               ucp_stream_recv_nbx_callback_t cb, void *user_data)
    {
        ucp_request_param_t params;

        params.op_attr_mask   = UCP_OP_ATTR_FIELD_CALLBACK |
                                UCP_OP_ATTR_FIELD_USER_DATA |
                                UCP_OP_ATTR_FIELD_FLAGS;
        params.flags          = UCP_STREAM_RECV_FLAG_WAITALL;
        params.user_data      = user_data;
        params.cb.recv_stream = cb;

        ucs_time_t deadline = ucs::get_deadline();
        ucp_stream_poll_ep_t poll_eps;
        ssize_t ep_count;
        do {
            progress();
            ep_count = ucp_stream_worker_poll(to.worker(), &poll_eps, 1, 0);
        } while ((ep_count == 0) && (ucs_get_time() < deadline));
        EXPECT_EQ(1, ep_count);
        EXPECT_EQ(to.ep(), poll_eps.ep);
        EXPECT_EQ(&to, poll_eps.user_data);

        size_t recv_length;
        return ucp_stream_recv_nbx(to.ep(), contig_buffer, length,
                                   &recv_length, &params);
    }

    struct rx_am_msg_arg {
        entity &receiver;
        bool received;
        void *hdr;
        void *buf;
        void *rreq;

        rx_am_msg_arg(entity &_receiver, void *_hdr, void *_buf) :
                receiver(_receiver), received(false), hdr(_hdr), buf(_buf),
                rreq(NULL) { }
    };

    static void rx_am_msg_data_recv_cb(void *request, ucs_status_t status,
                                       size_t length, void *user_data)
    {
        EXPECT_UCS_OK(status);
        volatile rx_am_msg_arg *rx_arg =
                reinterpret_cast<volatile rx_am_msg_arg*>(user_data);
        rx_arg->received = true;
    }

    static ucs_status_t rx_am_msg_cb(void *arg, const void *header,
                                     size_t header_length, void *data,
                                     size_t length,
                                     const ucp_am_recv_param_t *param)
    {
        volatile rx_am_msg_arg *rx_arg =
                reinterpret_cast<volatile rx_am_msg_arg*>(arg);
        EXPECT_FALSE(rx_arg->received);

        memcpy(rx_arg->hdr, header, header_length);
        if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) {
            ucp_request_param_t recv_param;
            recv_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                      UCP_OP_ATTR_FIELD_USER_DATA;
            recv_param.cb.recv_am   = rx_am_msg_data_recv_cb;
            recv_param.user_data    = const_cast<rx_am_msg_arg*>(rx_arg);

            void *rreq = ucp_am_recv_data_nbx(rx_arg->receiver.worker(), data,
                                              rx_arg->buf, length, &recv_param);
            if (UCS_PTR_IS_PTR(rreq)) {
                rx_arg->rreq = rreq;
                return UCS_OK;
            }
        } else {
            memcpy(rx_arg->buf, data, length);
        }

        rx_arg->received = true;
        return UCS_OK;
    }

    void set_am_data_handler(entity &e, uint16_t am_id,
                             ucp_am_recv_callback_t cb, void *arg)
    {
        ucp_am_handler_param_t param;

        /* Initialize Active Message data handler */
        param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                           UCP_AM_HANDLER_PARAM_FIELD_CB |
                           UCP_AM_HANDLER_PARAM_FIELD_ARG;
        param.id         = am_id;
        param.cb         = cb;
        param.arg        = arg;
        ASSERT_UCS_OK(ucp_worker_set_am_recv_handler(e.worker(), &param));
    }

    void send_recv(entity& from, entity& to, send_recv_type_t send_recv_type,
                   bool wakeup,
                   ucp_test_base::entity::listen_cb_type_t cb_type,
                   size_t ep_index = 0)
    {
        const uint64_t send_data = ucs_generate_uuid(0);
        uint64_t recv_data       = 0;
        rx_am_msg_arg am_rx_arg(to, NULL, &recv_data);
        ucs_status_t send_status;

        if (send_recv_type == SEND_RECV_AM) {
            set_am_data_handler(to, 0, rx_am_msg_cb, &am_rx_arg);
        }

        void *send_req = send(from, &send_data, sizeof(send_data),
                              send_recv_type, scomplete_cbx, NULL, ep_index);

        void *recv_req = NULL; // to suppress compiler warning
        if (send_recv_type == SEND_RECV_TAG) {
            recv_req = recv(to, &recv_data, sizeof(recv_data),
                            rtag_complete_cbx, NULL);
        } else if (send_recv_type == SEND_RECV_STREAM) {
            recv_req = recv(to, &recv_data, sizeof(recv_data),
                            rstream_complete_cbx, NULL);
        } else if (send_recv_type != SEND_RECV_AM) {
            UCS_TEST_ABORT("unsupported communication type " +
                           std::to_string(send_recv_type));
        }

        {
            // Suppress possible reject/unreachable errors
            scoped_log_handler slh(wrap_errors_logger);
            send_status = request_wait(send_req, 0, wakeup);
            if (!check_send_status(send_status, to, recv_req, cb_type)) {
                return;
            }
        }

        if (send_recv_type == SEND_RECV_AM) {
            request_wait(am_rx_arg.rreq);
            wait_for_flag(&am_rx_arg.received);
            set_am_data_handler(to, 0, NULL, NULL);
        } else {
            request_wait(recv_req, 0, wakeup);
        }
        EXPECT_EQ(send_data, recv_data);
    }

    bool wait_for_server_ep(bool wakeup)
    {
        ucs_time_t deadline = ucs::get_deadline();

        while ((receiver().get_num_eps() == 0) &&
               (sender().get_err_num() == 0) && (ucs_get_time() < deadline)) {
            check_events({ &sender(), &receiver() }, wakeup);
        }

        return (sender().get_err_num() == 0) && (receiver().get_num_eps() > 0);
    }

    void wait_for_reject(entity &e, bool wakeup)
    {
        ucs_time_t deadline = ucs::get_deadline();

        while ((e.get_err_num_rejected() == 0) && (ucs_get_time() < deadline)) {
            check_events({ &sender(), &receiver() }, wakeup);
        }

        EXPECT_GT(deadline, ucs_get_time());
        EXPECT_EQ(1ul, e.get_err_num_rejected());
    }

    virtual ucp_ep_params_t get_ep_params()
    {
        ucp_ep_params_t ep_params = ucp_test::get_ep_params();
        ep_params.field_mask     |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                    UCP_EP_PARAM_FIELD_ERR_HANDLER;
        /* The error handling requirement is needed since we need to take
         * care of a case where the client gets an error. In case ucp needs to
         * handle a large worker address but neither ud nor ud_x are present */
        ep_params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
        ep_params.err_handler.cb  = err_handler_cb;
        ep_params.err_handler.arg = this;
        return ep_params;
    }

    virtual ucp_ep_params_t get_server_ep_params() {
        return get_ep_params();
    }

    void client_ep_connect_basic(const ucp_ep_params_t &base_ep_params,
                                 size_t ep_index = 0,
                                 bool specify_src_addr = false)
    {
        ucp_ep_params_t ep_params = base_ep_params;
        ucs::sock_addr_storage src_addr(m_test_addr.to_ucs_sock_addr());
        src_addr.set_port(0);

        ep_params.field_mask      |= UCP_EP_PARAM_FIELD_FLAGS |
                                     UCP_EP_PARAM_FIELD_SOCK_ADDR |
                                     UCP_EP_PARAM_FIELD_USER_DATA;
        ep_params.flags           |= UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
        ep_params.sockaddr.addr    = m_test_addr.get_sock_addr_ptr();
        ep_params.sockaddr.addrlen = m_test_addr.get_addr_size();
        ep_params.user_data        = &sender();

        if (specify_src_addr) {
            ep_params.field_mask            |= UCP_EP_PARAM_FIELD_LOCAL_SOCK_ADDR;
            ep_params.local_sockaddr.addr    = src_addr.get_sock_addr_ptr();
            ep_params.local_sockaddr.addrlen = src_addr.get_addr_size();
        }

        sender().connect(&receiver(), ep_params, ep_index);
    }

    void client_ep_connect(size_t ep_index = 0, bool specify_src_addr = false)
    {
        client_ep_connect_basic(get_ep_params(), ep_index, specify_src_addr);
    }

    void connect_and_send_recv(bool wakeup, uint64_t flags,
                               bool specify_src_addr = false)
    {
        {
            scoped_log_handler slh(detect_error_logger);
            client_ep_connect(specify_src_addr);
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

    void listen(ucp_test_base::entity::listen_cb_type_t cb_type)
    {
        UCS_TEST_MESSAGE << "Testing " << m_test_addr.to_str();
        start_listener(cb_type);
    }

    void listen_and_communicate(bool wakeup, uint64_t flags,
                                bool specify_src_addr = false)
    {
        listen(cb_type());
        connect_and_send_recv(wakeup, flags, specify_src_addr);
    }

    void listen_and_reject(bool wakeup)
    {
        listen(ucp_test_base::entity::LISTEN_CB_REJECT);
        connect_and_reject(wakeup);
    }

    void ep_query()
    {
        ucp_ep_attr_t attr;

        attr.field_mask = UCP_EP_ATTR_FIELD_LOCAL_SOCKADDR |
                          UCP_EP_ATTR_FIELD_REMOTE_SOCKADDR;
        ucs_status_t status = ucp_ep_query(receiver().ep(), &attr);
        ASSERT_UCS_OK(status);

        EXPECT_EQ(m_test_addr, attr.local_sockaddr);

        /* The ports are expected to be different. Ignore them. */
        ucs_sockaddr_set_port((struct sockaddr*)&attr.remote_sockaddr,
                              m_test_addr.get_port());
        EXPECT_EQ(m_test_addr, attr.remote_sockaddr);

        memset(&attr, 0, sizeof(attr));
        attr.field_mask = UCP_EP_ATTR_FIELD_LOCAL_SOCKADDR |
                          UCP_EP_ATTR_FIELD_REMOTE_SOCKADDR;
        status = ucp_ep_query(sender().ep(), &attr);
        ASSERT_UCS_OK(status);

        EXPECT_EQ(m_test_addr, attr.remote_sockaddr);

        /* The ports are expected to be different. Ignore them.*/
        ucs_sockaddr_set_port((struct sockaddr*)&attr.local_sockaddr,
                              m_test_addr.get_port());
        EXPECT_EQ(m_test_addr, attr.local_sockaddr);
    }

    void one_sided_disconnect(entity &e, enum ucp_ep_close_mode mode) {
        void *req           = e.disconnect_nb(0, 0, mode);
        ucs_time_t deadline = ucs::get_deadline();
        scoped_log_handler slh(detect_error_logger);
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
        scoped_log_handler slh(detect_error_logger);
        while ((!is_request_completed(sender_ep_close_req) ||
                !is_request_completed(receiver_ep_close_req)) &&
               (ucs_get_time() < deadline)) {
            progress();
        }

        sender().close_ep_req_free(sender_ep_close_req);
        receiver().close_ep_req_free(receiver_ep_close_req);
    }

    void setup_unreachable_listener()
    {
        ucs::sock_addr_storage listen_addr(m_test_addr.to_ucs_sock_addr());
        ucs_status_t status = receiver().listen(cb_type(),
                                                m_test_addr.get_sock_addr_ptr(),
                                                m_test_addr.get_addr_size(),
                                                get_server_ep_params());
        if (status == UCS_ERR_UNREACHABLE) {
            UCS_TEST_SKIP_R("cannot listen to " + m_test_addr.to_str());
        }

        /* make the client try to connect to a non-existing port on the server
         * side */
        m_test_addr.set_port(1);
    }

    static ucs_log_func_rc_t
    detect_fail_no_err_cb(const char *file, unsigned line, const char *function,
                          ucs_log_level_t level,
                          const ucs_log_component_config_t *comp_conf,
                          const char *message, va_list ap)
    {
        if (level == UCS_LOG_LEVEL_ERROR) {
            std::string err_str = format_message(message, ap);

            if (err_str.find("on CM lane will not be handled since no error"
                             " callback is installed") != std::string::npos) {
                UCS_TEST_MESSAGE << "< " << err_str << " >";
                ++m_err_count;
                return UCS_LOG_FUNC_RC_STOP;
            }
        }

        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    static void close_completion(void *request, ucs_status_t status,
                                 void *user_data) {
        *reinterpret_cast<bool*>(user_data) = true;
    }

    static void err_handler_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
        ucp_test::err_handler_cb(arg, ep, status);

        ++m_err_count;

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
        case UCS_ERR_NOT_CONNECTED:
        case UCS_ERR_ENDPOINT_TIMEOUT:
            UCS_TEST_MESSAGE << "ignoring error " << ucs_status_string(status)
                             << " on endpoint " << ep;
            return;
        default:
            UCS_TEST_ABORT("Error: " << ucs_status_string(status));
        }
    }

protected:
    ucp_test_base::entity::listen_cb_type_t cb_type() const {
        const int variant = (get_variant_value() & TEST_MODIFIER_MASK);
        if ((variant == CONN_REQ_TAG) || (variant == CONN_REQ_STREAM)) {
            return ucp_test_base::entity::LISTEN_CB_CONN;
        }
        return ucp_test_base::entity::LISTEN_CB_EP;
    }

    send_recv_type_t send_recv_type() const {
        switch (get_variant_value() & TEST_MODIFIER_MASK) {
        case CONN_REQ_STREAM:
            return SEND_RECV_STREAM;
        case CONN_REQ_TAG:
            /* fallthrough */
        default:
            return SEND_RECV_TAG;
        }
    }

    bool nonparameterized_test() const {
        return (get_variant_value() != DEFAULT_PARAM_VARIANT) &&
               (get_variant_value() != CONN_REQ_TAG);
    }

    bool cm_use_all_devices() const {
        return get_variant_value() & TEST_MODIFIER_CM_USE_ALL_DEVICES;
    }

    bool sa_data_version_v2() const {
        return get_variant_value() & TEST_MODIFIER_SA_DATA_V2;
    }

    bool has_rndv_lanes(ucp_ep_h ep)
    {
        for (ucp_lane_index_t lane_idx = 0;
             lane_idx < ucp_ep_num_lanes(ep); ++lane_idx) {
            if ((lane_idx != ucp_ep_get_cm_lane(ep)) &&
                (ucp_ep_get_iface_attr(ep, lane_idx)->cap.flags &
                 (UCT_IFACE_FLAG_GET_ZCOPY | UCT_IFACE_FLAG_PUT_ZCOPY)) &&
                /* RNDV lanes should be selected if transport supports GET/PUT
                 * Zcopy and: */
                (/* - either memory invalidation can be done on its MD */
                 (ucp_ep_md_attr(ep, lane_idx)->cap.flags &
                  UCT_MD_FLAG_INVALIDATE) ||
                 /* - or CONNECT_TO_EP connection establishment mode is used */
                 (ucp_ep_is_lane_p2p(ep, lane_idx)))) {
                EXPECT_NE(UCP_NULL_LANE, ucp_ep_config(ep)->key.rma_bw_lanes[0])
                        << "RNDV lanes should be selected";
                return true;
            }
        }

        return false;
    }

    static ucs_status_t ep_pending_add(uct_ep_h ep, uct_pending_req_t *req,
                                       unsigned flags)
    {
        if (req->func == ucp_worker_discard_uct_ep_pending_cb) {
            return UCS_ERR_BUSY;
        }

        auto ops = m_sender_uct_ops.find(ep->iface);
        return ops->second.ep_pending_add(ep, req, flags);
    }

    void do_force_close_during_rndv(bool fail_send_ep)
    {
        constexpr size_t length = 4 * UCS_KBYTE;

        listen_and_communicate(false, SEND_DIRECTION_BIDI);

        mem_buffer send_buffer(length, UCS_MEMORY_TYPE_HOST);
        send_buffer.pattern_fill(1, length);
        void *sreq = send(sender(), send_buffer.ptr(), length,
                          send_recv_type(), scomplete_reset_data_cbx,
                          reinterpret_cast<void*>(&send_buffer));

        ucp_ep_h ep = sender().revoke_ep();

        // Wait for the TAG RNDV/RTS packet sent and the request scheduled to
        // be tracked until RNDV/ATS packet is not received from a peer
        ucp_tag_message_h message;
        ucp_tag_recv_info_t info;
        message = message_wait(receiver(), 0, 0, &info);
        ASSERT_NE((void*)NULL, message);
        ASSERT_EQ(UCS_INPROGRESS, ucp_request_check_status(sreq));

        // Prevent destroying UCT endpoints from discarding to not detect error
        // by the receiver earlier than data could invalidate by the sender
        for (auto lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
            if (lane == ucp_ep_get_cm_lane(ep)) {
                continue;
            }

            uct_iface_h uct_iface = ep->uct_eps[lane]->iface;
            auto res              = m_sender_uct_ops.emplace(uct_iface,
                                                             uct_iface->ops);
            if (res.second) {
                uct_iface->ops.ep_flush       =
                        reinterpret_cast<uct_ep_flush_func_t>(
                                ucs_empty_function_return_no_resource);
                uct_iface->ops.ep_pending_add = ep_pending_add;
            }
        }

        if (fail_send_ep) {
            UCS_ASYNC_BLOCK(&ep->worker->async);
            ucp_ep_set_failed(ep, UCP_NULL_LANE, UCS_ERR_CONNECTION_RESET);
            UCS_ASYNC_UNBLOCK(&ep->worker->async);
        }
        void *close_req = ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FORCE);

        // Do some progress of sender's worker to check that it doesn't
        // complete UCP requests prior closing UCT endpoint from discarding
        request_progress(sreq, { &sender() }, 0.5);

        // Restore UCT endpoint's flush function to the original one to allow
        // completion of discarding
        for (auto &elem : m_sender_uct_ops) {
            elem.first->ops.ep_flush       = elem.second.ep_flush;
            elem.first->ops.ep_pending_add = elem.second.ep_pending_add;
        }

        m_sender_uct_ops.clear();

        mem_buffer recv_buffer(length, UCS_MEMORY_TYPE_HOST);
        recv_buffer.pattern_fill(2, length);
        void *rreq = recv(receiver(), recv_buffer.ptr(), length,
                          message, rtag_complete_check_data_cbx,
                          reinterpret_cast<void*>(&recv_buffer));

        if (fail_send_ep) {
            // Progress the receiver to try receiving the data sent by sender
            ucs_status_t status = request_progress(rreq, { &receiver() });
            ASSERT_NE(UCS_INPROGRESS, status);
        } else {
            request_progress(sreq, { &sender() });
            request_progress(rreq, { &receiver() });
        }

        {
            scoped_log_handler slh(wrap_errors_logger);
            std::vector<void*> reqs = { sreq, rreq, close_req };
            requests_wait(reqs);
        }
    }

protected:
    static unsigned m_err_count;
    static std::map<uct_iface_h, uct_iface_ops_t> m_sender_uct_ops;
};

unsigned test_ucp_sockaddr::m_err_count                                    = 0;
std::map<uct_iface_h, uct_iface_ops_t> test_ucp_sockaddr::m_sender_uct_ops = {};


UCS_TEST_P(test_ucp_sockaddr, listen) {
    listen_and_communicate(false, 0);
}

UCS_TEST_P(test_ucp_sockaddr, listen_c2s) {
    listen_and_communicate(false, SEND_DIRECTION_C2S);
}

UCS_TEST_P(test_ucp_sockaddr, listen_s2c) {
    listen_and_communicate(false, SEND_DIRECTION_S2C);
}

UCS_TEST_P(test_ucp_sockaddr, listen_bidi) {
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
}

UCS_TEST_P(test_ucp_sockaddr, ep_query) {
    listen_and_communicate(false, 0);
    ep_query();
}

UCS_TEST_P(test_ucp_sockaddr, set_local_sockaddr)
{
    listen_and_communicate(false, 0, true);
    ep_query();
}

UCS_TEST_P(test_ucp_sockaddr, onesided_disconnect) {
    listen_and_communicate(false, 0);
    one_sided_disconnect(sender(), UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr, onesided_disconnect_c2s) {
    listen_and_communicate(false, SEND_DIRECTION_C2S);
    one_sided_disconnect(sender(), UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr, onesided_disconnect_s2c) {
    listen_and_communicate(false, SEND_DIRECTION_S2C);
    one_sided_disconnect(sender(), UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr, onesided_disconnect_bidi) {
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
    one_sided_disconnect(sender(), UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr, close_callback) {
    listen_and_communicate(false, SEND_DIRECTION_BIDI);

    request_wait(receiver().flush_ep_nb());
    request_wait(sender().flush_ep_nb());
    ucp_ep_h ep = receiver().revoke_ep();

    bool user_data = false;

    ucp_request_param_t param = {0};
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK  |
                         UCP_OP_ATTR_FIELD_USER_DATA |
                         UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    param.cb.send      = close_completion;
    param.user_data    = &user_data;

    ucs_status_ptr_t request = ucp_ep_close_nbx(ep, &param);

    bool is_pointer = UCS_PTR_IS_PTR(request);
    request_wait(request);

    if (is_pointer) {
        ASSERT_TRUE(user_data);
    }
}

UCS_TEST_P(test_ucp_sockaddr, onesided_disconnect_bidi_wait_err_cb) {
    listen_and_communicate(false, SEND_DIRECTION_BIDI);

    one_sided_disconnect(sender(), UCP_EP_CLOSE_MODE_FLUSH);
    wait_for_flag(&m_err_count);
    EXPECT_EQ(1u, m_err_count);
}

UCS_TEST_P(test_ucp_sockaddr, concurrent_disconnect) {
    listen_and_communicate(false, 0);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr, concurrent_disconnect_c2s) {
    listen_and_communicate(false, SEND_DIRECTION_C2S);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr, concurrent_disconnect_s2c) {
    listen_and_communicate(false, SEND_DIRECTION_S2C);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr, concurrent_disconnect_bidi) {
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr, concurrent_disconnect_force) {
    listen_and_communicate(false, 0);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FORCE);
}

UCS_TEST_P(test_ucp_sockaddr, concurrent_disconnect_force_c2s) {
    listen_and_communicate(false, SEND_DIRECTION_C2S);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FORCE);
}

UCS_TEST_P(test_ucp_sockaddr, concurrent_disconnect_force_s2c) {
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

UCS_TEST_P(test_ucp_sockaddr, err_handle)
{
    setup_unreachable_listener();

    {
        scoped_log_handler slh(wrap_errors_logger);
        client_ep_connect();
        /* allow for the unreachable event to arrive before restoring errors */
        wait_for_flag(&sender().get_err_num());
    }

    EXPECT_EQ(1u, sender().get_err_num());
}

UCS_TEST_P(test_ucp_sockaddr, err_handle_without_err_cb)
{
    setup_unreachable_listener();

    {
        scoped_log_handler slh(detect_fail_no_err_cb);
        ucp_ep_params_t ep_params = ucp_test::get_ep_params();

        ep_params.field_mask |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
        ep_params.err_mode    = UCP_ERR_HANDLING_MODE_PEER;

        client_ep_connect_basic(ep_params);

        /* allow for the unreachable event to arrive before restoring errors */
        wait_for_flag(&m_err_count);
        if (m_err_count > 0) {
            sender().add_err(UCS_ERR_CONNECTION_RESET);
        }
    }

    EXPECT_EQ(1u, sender().get_err_num());
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr, force_close_during_rndv,
                     (send_recv_type() != SEND_RECV_TAG), "RNDV_THRESH=0")
{
    do_force_close_during_rndv(false);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr, fail_and_force_close_during_rndv,
                     (send_recv_type() != SEND_RECV_TAG), "RNDV_THRESH=0")
{
    do_force_close_during_rndv(true);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr, listener_invalid_params,
                     nonparameterized_test(), "CM_REUSEADDR?=y")
{
    ucp_listener_params_t params;
    ucp_listener_h listener;
    ucs_status_t status;

    params.field_mask = 0;
    /* address and conn/accept handlers are not specified */
    status            = create_listener_wrap_err(params, listener);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    /* add listen address, use ANY addr/port to avoid BUSY error in the end */
    m_test_addr.reset_to_any();
    m_test_addr.set_port(0);
    params.field_mask       = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR;
    params.sockaddr.addr    = m_test_addr.get_sock_addr_ptr();
    params.sockaddr.addrlen = m_test_addr.get_addr_size();
    /* accept handlers aren't set */
    status                  = create_listener_wrap_err(params, listener);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    /* define conn handler flag but set to NULL */
    params.field_mask       = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                              UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    params.conn_handler.cb  = NULL;
    params.conn_handler.arg = NULL;
    status                  = create_listener_wrap_err(params, listener);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    /* define both conn and accept handlers to NULL */
    params.field_mask         = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                UCP_LISTENER_PARAM_FIELD_CONN_HANDLER |
                                UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER;
    params.accept_handler.cb  = NULL;
    params.accept_handler.arg = NULL;
    status                    = create_listener_wrap_err(params, listener);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    /* define both conn and accept handlers to valid callbacks
     * (should be only 1) */
    params.field_mask        = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                               UCP_LISTENER_PARAM_FIELD_CONN_HANDLER |
                               UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER;
    params.conn_handler.cb   =
            (ucp_listener_conn_callback_t)ucs_empty_function;
    params.accept_handler.cb =
            (ucp_listener_accept_callback_t)ucs_empty_function;
    status                   = create_listener_wrap_err(params, listener);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    /* sockaddr and valid conn handler is OK */
    params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                        UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    status            = create_listener_wrap_err(params, listener);
    ASSERT_UCS_OK(status);
    ucp_listener_destroy(listener);

    /* sockaddr and valid accept handler is OK */
    params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                        UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER;
    status            = create_listener_wrap_err(params, listener);
    ASSERT_UCS_OK(status);
    ucp_listener_destroy(listener);
}

UCP_INSTANTIATE_ALL_TEST_CASE(test_ucp_sockaddr)

class test_ucp_sockaddr_conn_request : public test_ucp_sockaddr {
public:
    virtual ucp_worker_params_t get_worker_params() {
        ucp_worker_params_t params = test_ucp_sockaddr::get_worker_params();
        params.field_mask         |= UCP_WORKER_PARAM_FIELD_CLIENT_ID;
        params.client_id           = reinterpret_cast<uint64_t>(this);
        return params;
    }

    ucp_ep_params_t get_client_ep_params() {
        ucp_ep_params_t ep_params = test_ucp_sockaddr::get_ep_params();
        ep_params.field_mask     |= UCP_EP_PARAM_FIELD_FLAGS;
        ep_params.flags          |= UCP_EP_PARAMS_FLAGS_SEND_CLIENT_ID;
        return ep_params;
    }

    static void conn_handler_cb(ucp_conn_request_h conn_request, void *arg)
    {
        ucp_conn_request_attr_t attr;
        ucs_status_t status;

        attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ID;
        status          = ucp_conn_request_query(conn_request, &attr);
        EXPECT_EQ(UCS_OK, status);
        EXPECT_EQ(reinterpret_cast<uint64_t>(arg), attr.client_id);

        test_ucp_sockaddr_conn_request *self =
            reinterpret_cast<test_ucp_sockaddr_conn_request*>(arg);
        ucp_listener_reject(self->receiver().listenerh(), conn_request);
    }
};

UCS_TEST_P(test_ucp_sockaddr_conn_request, conn_request_query_worker_id)
{
    ucp_listener_conn_handler_t conn_handler;

    conn_handler.cb  = test_ucp_sockaddr_conn_request::conn_handler_cb;
    conn_handler.arg = reinterpret_cast<void*>(this);
    start_listener(ucp_test_base::entity::LISTEN_CB_CUSTOM, &conn_handler);
    {
        scoped_log_handler slh(detect_error_logger);
        client_ep_connect_basic(get_client_ep_params());
        send_recv(sender(), receiver(), SEND_RECV_TAG, false,
                  ucp_test_base::entity::LISTEN_CB_REJECT);
    }
}

UCP_INSTANTIATE_ALL_TEST_CASE(test_ucp_sockaddr_conn_request)

class test_ucp_sockaddr_wireup : public test_ucp_sockaddr {
public:
    static void
    get_test_variants(std::vector<ucp_test_variant>& variants,
                      uint64_t features = UCP_FEATURE_TAG) {
        /* It is enough to check TAG-only, since we are interested in WIREUP
         * testing only */
        get_test_variants_cm_mode(variants, features, CONN_REQ_TAG, "tag");
    }

protected:
    static void cmp_cfg_lanes(ucp_ep_config_key_t *key1, ucp_lane_index_t lane1,
                              ucp_ep_config_key_t *key2, ucp_lane_index_t lane2) {
        EXPECT_TRUE(((lane1 == UCP_NULL_LANE) && (lane2 == UCP_NULL_LANE)) ||
                    ((lane1 != UCP_NULL_LANE) && (lane2 != UCP_NULL_LANE) &&
                     ucp_ep_config_lane_is_peer_match(key1, lane1, key2, lane2)));
    }
};


UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_wireup, compare_cm_and_wireup_configs,
                     !cm_use_all_devices()) {
    ucp_worker_cfg_index_t cm_ep_cfg_index, wireup_ep_cfg_index;
    ucp_ep_config_key_t *cm_ep_cfg_key, *wireup_ep_cfg_key;
    bool should_check_rndv_lanes;

    /* get configuration index for EP created through CM */
    listen_and_communicate(false, SEND_DIRECTION_C2S);
    cm_ep_cfg_index         = sender().ep()->cfg_index;
    cm_ep_cfg_key           = &ucp_ep_config(sender().ep())->key;
    /* Don't check RNDV lanes, because CM prefers p2p connection mode for RNDV
     * lanes and they don't support memory invalidation on MD */
    should_check_rndv_lanes = !has_rndv_lanes(sender().ep());
    EXPECT_NE(UCP_NULL_LANE, ucp_ep_get_cm_lane(sender().ep()));
    disconnect(sender());
    disconnect(receiver());

    /* get configuration index for EP created through WIREUP */
    sender().connect(&receiver(), get_ep_params());
    ucp_ep_params_t params = get_ep_params();
    /* initialize user data for STREAM API testing */
    params.field_mask     |= UCP_EP_PARAM_FIELD_USER_DATA;
    params.user_data       = &receiver();
    receiver().connect(&sender(), params);
    send_recv(sender(), receiver(), send_recv_type(), 0, cb_type());
    wireup_ep_cfg_index = sender().ep()->cfg_index;
    wireup_ep_cfg_key   = &ucp_ep_config(sender().ep())->key;
    EXPECT_EQ(UCP_NULL_LANE, ucp_ep_get_cm_lane(sender().ep()));

    /* EP config indexes must be different because one has CM lane and
     * the other doesn't */
    EXPECT_NE(cm_ep_cfg_index, wireup_ep_cfg_index);

    /* compare AM lanes */
    cmp_cfg_lanes(cm_ep_cfg_key, cm_ep_cfg_key->am_lane,
                  wireup_ep_cfg_key, wireup_ep_cfg_key->am_lane);

    /* compare TAG lanes */
    cmp_cfg_lanes(cm_ep_cfg_key, cm_ep_cfg_key->tag_lane,
                  wireup_ep_cfg_key, wireup_ep_cfg_key->tag_lane);

    /* compare RMA lanes */
    for (ucp_lane_index_t lane = 0;
         cm_ep_cfg_key->rma_lanes[lane] != UCP_NULL_LANE; ++lane) {
        cmp_cfg_lanes(cm_ep_cfg_key, cm_ep_cfg_key->rma_lanes[lane],
                      wireup_ep_cfg_key, wireup_ep_cfg_key->rma_lanes[lane]);
    }

    /* compare RMA BW lanes */
    if (should_check_rndv_lanes) {
        /* EP config RMA BW must be equal */
        EXPECT_EQ(cm_ep_cfg_key->rma_bw_md_map,
                  wireup_ep_cfg_key->rma_bw_md_map);

        for (ucp_lane_index_t lane = 0;
             cm_ep_cfg_key->rma_bw_lanes[lane] != UCP_NULL_LANE; ++lane) {
            cmp_cfg_lanes(cm_ep_cfg_key, cm_ep_cfg_key->rma_bw_lanes[lane],
                          wireup_ep_cfg_key,
                          wireup_ep_cfg_key->rma_bw_lanes[lane]);
        }
    }

    /* compare RKEY PTR lanes */
    cmp_cfg_lanes(cm_ep_cfg_key, cm_ep_cfg_key->rkey_ptr_lane,
                  wireup_ep_cfg_key, wireup_ep_cfg_key->rkey_ptr_lane);

    /* compare AMO lanes */
    for (ucp_lane_index_t lane = 0;
         cm_ep_cfg_key->amo_lanes[lane] != UCP_NULL_LANE; ++lane) {
        cmp_cfg_lanes(cm_ep_cfg_key, cm_ep_cfg_key->amo_lanes[lane],
                      wireup_ep_cfg_key, wireup_ep_cfg_key->amo_lanes[lane]);
    }

    /* compare AM BW lanes */
    for (ucp_lane_index_t lane = 0;
         cm_ep_cfg_key->am_bw_lanes[lane] != UCP_NULL_LANE; ++lane) {
        cmp_cfg_lanes(cm_ep_cfg_key, cm_ep_cfg_key->am_bw_lanes[lane],
                      wireup_ep_cfg_key, wireup_ep_cfg_key->am_bw_lanes[lane]);
    }
}

UCP_INSTANTIATE_ALL_TEST_CASE(test_ucp_sockaddr_wireup)


class test_ucp_sockaddr_wireup_fail : public test_ucp_sockaddr_wireup {
protected:
    typedef enum {
        FAIL_WIREUP_MSG_SEND,
        FAIL_WIREUP_MSG_ADDR_PACK,
        FAIL_WIREUP_CONNECT_TO_EP,
        FAIL_WIREUP_SET_EP_FAILED
    } fail_wireup_t;

    test_ucp_sockaddr_wireup_fail() {
        m_num_fail_injections = 0;
    }

    virtual bool test_all_ep_flags(entity &e, uint32_t flags) const
    {
        UCS_ASYNC_BLOCK(&e.worker()->async);
        bool result = ucs_test_all_flags(e.ep()->flags, flags);
        UCS_ASYNC_UNBLOCK(&e.worker()->async);

        return result;
    }

    bool test_any_ep_flag(entity &e, uint32_t flags) const
    {
        UCS_ASYNC_BLOCK(&e.worker()->async);
        bool result = e.ep()->flags & flags;
        UCS_ASYNC_UNBLOCK(&e.worker()->async);

        return result;
    }

    static ssize_t fail_injection()
    {
        ++m_num_fail_injections;
        return UCS_ERR_ENDPOINT_TIMEOUT;
    }

    void set_iface_failure(uct_iface_h iface, fail_wireup_t fail_wireup_type)
    {
        if (fail_wireup_type == FAIL_WIREUP_MSG_SEND) {
            /* Emulate failure of WIREUP MSG sending by setting the AM Bcopy
             * function which always return EP_TIMEOUT error */
            iface->ops.ep_am_bcopy =
                    reinterpret_cast<uct_ep_am_bcopy_func_t>(fail_injection);
        } else if (fail_wireup_type == FAIL_WIREUP_CONNECT_TO_EP) {
            /* Emulate failure of connecting p2p lanes of peers by setting the
             * connect_to_ep method to the function that always returns error
             */
            iface->ops.ep_connect_to_ep =
                    reinterpret_cast<uct_ep_connect_to_ep_func_t>(
                            fail_injection);
        } else if (fail_wireup_type == FAIL_WIREUP_MSG_ADDR_PACK) {
            /* Emulate failure of preparation of WIREUP MSG sending by setting
             * the device address getter to the function that always returns
             * error */
            iface->ops.iface_get_device_address =
                    reinterpret_cast<uct_iface_get_device_address_func_t>(
                            fail_injection);
        }
    }

    void emulate_failure(entity &e, fail_wireup_t fail_wireup_type)
    {
        ucp_worker_h worker = e.worker();

        UCS_ASYNC_BLOCK(&worker->async);
        if (fail_wireup_type == FAIL_WIREUP_SET_EP_FAILED) {
            /* Emulate failure of the endpoint by invoking error handling
             * procedure */
            ++m_num_fail_injections;
            ucp_ep_set_failed(e.ep(), UCP_NULL_LANE, UCS_ERR_ENDPOINT_TIMEOUT);
        } else {
            /* Make sure that stub WIREUP_EP is updated */
            for (auto lane = 0; lane < ucp_ep_num_lanes(e.ep()); ++lane) {
                set_iface_failure(e.ep()->uct_eps[lane]->iface,
                                  fail_wireup_type);
            }
            for (auto iface_id = 0; iface_id < worker->num_ifaces;
                 ++iface_id) {
                set_iface_failure(worker->ifaces[iface_id]->iface,
                                  fail_wireup_type);
            }
        }
        UCS_ASYNC_UNBLOCK(&worker->async);
    }

    void wait_ep_err_or_wireup_msg_done(entity &e)
    {
        ucs_time_t deadline = ucs::get_deadline();

        while (!test_any_ep_flag(e, UCP_EP_FLAG_CONNECT_ACK_SENT |
                                    UCP_EP_FLAG_CONNECT_REP_SENT) &&
               (m_err_count == 0)) {
            ASSERT_LT(ucs_get_time(), deadline);
            progress();
        }

        if (m_num_fail_injections == 0) {
            EXPECT_EQ(0, m_err_count);
            UCS_TEST_MESSAGE << "failure injection was not done";
        } else {
            EXPECT_GT(m_err_count, 0);
        }
    }

    void connect_and_fail_wireup(entity &e, fail_wireup_t fail_wireup_type,
                                 uint32_t wait_ep_flags,
                                 bool wait_cm_failure = false)
    {
        start_listener(cb_type());

        scoped_log_handler slh(wrap_errors_logger);
        client_ep_connect();
        if (!wait_cm_failure && !wait_for_server_ep(false)) {
            UCS_TEST_SKIP_R("cannot connect to server");
        }

        ucs_time_t deadline = ucs::get_deadline();
        while (!test_all_ep_flags(e, wait_ep_flags) &&
               (m_err_count == 0)) {
            ASSERT_LT(ucs_get_time(), deadline);
            progress();
        }

        emulate_failure(e, fail_wireup_type);
        wait_ep_err_or_wireup_msg_done(e);

        if (wait_cm_failure) {
            one_sided_disconnect(e, UCP_EP_CLOSE_MODE_FORCE);
        } else {
            concurrent_disconnect(UCP_EP_CLOSE_MODE_FORCE);
        }
    }

public:
    static std::atomic<unsigned> m_num_fail_injections;
};


std::atomic<unsigned> test_ucp_sockaddr_wireup_fail::m_num_fail_injections{0};


UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_wireup_fail,
                     connect_and_fail_wireup_msg_send_on_client,
                     !cm_use_all_devices())
{
    connect_and_fail_wireup(sender(), FAIL_WIREUP_MSG_SEND,
                            /* WIREUP_MSGs are sent after the client
                             * is fully connected */
                            UCP_EP_FLAG_CLIENT_CONNECT_CB);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_wireup_fail,
                     connect_and_fail_wireup_msg_send_on_server,
                     !cm_use_all_devices())
{
    connect_and_fail_wireup(receiver(), FAIL_WIREUP_MSG_SEND,
                            /* WIREUP_MSGs are sent after the server
                             * is fully connected */
                            UCP_EP_FLAG_SERVER_NOTIFY_CB);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_wireup_fail,
                     connect_and_fail_wireup_msg_pack_addr_on_client,
                     !cm_use_all_devices())
{
    connect_and_fail_wireup(sender(), FAIL_WIREUP_MSG_ADDR_PACK,
                            /* WIREUP_MSGs are sent after the client
                             * is fully connected and it packs
                             * addresses when sending WIREUP_MSGs */
                            UCP_EP_FLAG_CLIENT_CONNECT_CB);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_wireup_fail,
                     connect_and_fail_wireup_msg_pack_addr_on_server,
                     !cm_use_all_devices())
{
    connect_and_fail_wireup(receiver(), FAIL_WIREUP_MSG_ADDR_PACK,
                            /* WIREUP_MSGs are sent after the server
                             * is fully connected and it packs
                             * addresses when sending WIREUP_MSGs */
                            UCP_EP_FLAG_SERVER_NOTIFY_CB);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_wireup_fail,
                     connect_and_fail_wireup_connect_to_ep_on_client,
                     !cm_use_all_devices())
{
    connect_and_fail_wireup(sender(), FAIL_WIREUP_CONNECT_TO_EP,
                            /* UCT EPs are connected after the client is fully
                             * connected through CM and waiting for
                             * WIREUP_MSG/PRE_REQ */
                            UCP_EP_FLAG_CONNECT_WAIT_PRE_REQ);
}

UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_wireup_fail,
                     connect_and_fail_wireup_connect_to_ep_on_server,
                     !cm_use_all_devices())
{
    connect_and_fail_wireup(receiver(), FAIL_WIREUP_CONNECT_TO_EP,
                            /* UCT EPs are connected after the server is fully
                             * connected through CM and WIREUP_MSG/PRE_REQ is
                             * sent */
                            UCP_EP_FLAG_CONNECT_PRE_REQ_SENT);
}

UCS_TEST_P(test_ucp_sockaddr_wireup_fail,
           connect_and_fail_wireup_set_ep_failed_on_client)
{
    connect_and_fail_wireup(sender(), FAIL_WIREUP_SET_EP_FAILED,
                            UCP_EP_FLAG_CLIENT_CONNECT_CB);
}

UCS_TEST_P(test_ucp_sockaddr_wireup_fail,
           connect_and_fail_wireup_set_ep_failed_on_server)
{
    connect_and_fail_wireup(receiver(), FAIL_WIREUP_SET_EP_FAILED,
                            UCP_EP_FLAG_SERVER_NOTIFY_CB);
}

UCP_INSTANTIATE_ALL_TEST_CASE(test_ucp_sockaddr_wireup_fail)


class test_ucp_sockaddr_wireup_fail_try_next_cm :
        public test_ucp_sockaddr_wireup_fail  {
private:
    typedef struct {
        entity &e;
        bool   found;
    } find_try_next_cm_arg_t;

    static int find_try_next_cm_cb(const ucs_callbackq_elem_t *elem, void *arg)
    {
        find_try_next_cm_arg_t *find_try_next_cm_arg =
                reinterpret_cast<find_try_next_cm_arg_t*>(arg);

        if ((elem->cb == ucp_cm_client_try_next_cm_progress) &&
            (elem->arg == find_try_next_cm_arg->e.ep())) {
            find_try_next_cm_arg->found = true;
        }

        return 0;
    }

    virtual bool test_all_ep_flags(entity &e, uint32_t flags) const
    {
        /* The test expects that only CLIENT_CONNECT_CB flag will be tested
         * here. So, skip the test if CLIENT_CONNECT_CB flag set on the
         * endpoint, since it means that trying next CM won't be done */
        if (!ucs_test_all_flags(flags, UCP_EP_FLAG_CLIENT_CONNECT_CB)) {
            UCS_TEST_ABORT("Error: expect that only " << std::hex <<
                           UCP_EP_FLAG_CLIENT_CONNECT_CB << " flag is set,"
                           " but " << flags << " flags are given");
        }

        if (test_ucp_sockaddr_wireup_fail::test_all_ep_flags(e, flags)) {
            UCS_TEST_SKIP_R("trying the next CM calback wasn't scheduled");
        }

        /* Waiting for ucp_cm_client_try_next_cm_progress() callback being
         * scheduled on a progress. When it is found, the test emulates failure
         * to check that the callback is removed from the callback queue on
         * the worker */
        find_try_next_cm_arg_t find_try_next_cm_arg = { e, false };

        UCS_ASYNC_BLOCK(&e.worker()->async);
        uct_priv_worker_t *worker = ucs_derived_of(e.worker()->uct,
                                                   uct_priv_worker_t);
        ucs_callbackq_remove_if(&worker->super.progress_q,
                                find_try_next_cm_cb, &find_try_next_cm_arg);
        UCS_ASYNC_UNBLOCK(&e.worker()->async);

        return find_try_next_cm_arg.found;
    }
};


UCS_TEST_P(test_ucp_sockaddr_wireup_fail_try_next_cm,
           connect_and_fail_wireup_next_cm_tcp2rdmacm_set_ep_failed_on_client,
           "SOCKADDR_TLS_PRIORITY=tcp,rdmacm")
{
    connect_and_fail_wireup(sender(), FAIL_WIREUP_SET_EP_FAILED,
                            UCP_EP_FLAG_CLIENT_CONNECT_CB, true);
}

UCS_TEST_P(test_ucp_sockaddr_wireup_fail_try_next_cm,
           connect_and_fail_wireup_next_cm_rdmacm2tcp_set_ep_failed_on_client,
           "SOCKADDR_TLS_PRIORITY=rdmacm,tcp")
{
    connect_and_fail_wireup(sender(), FAIL_WIREUP_SET_EP_FAILED,
                            UCP_EP_FLAG_CLIENT_CONNECT_CB, true);
}

UCP_INSTANTIATE_ALL_TEST_CASE(test_ucp_sockaddr_wireup_fail_try_next_cm)


class test_ucp_sockaddr_different_tl_rsc : public test_ucp_sockaddr
{
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        uint64_t features = UCP_FEATURE_STREAM | UCP_FEATURE_TAG;
        test_ucp_sockaddr::get_test_variants_cm_mode(variants, features,
                                                     UNSET_SELF_DEVICES,
                                                     "unset_self_devices");
        test_ucp_sockaddr::get_test_variants_cm_mode(variants, features,
                                                     UNSET_SHM_DEVICES,
                                                     "unset_shm_devices");
        test_ucp_sockaddr::get_test_variants_cm_mode(variants, features,
                                                     UNSET_SELF_DEVICES |
                                                     UNSET_SHM_DEVICES,
                                                     "unset_self_shm_devices");
    }

protected:
    enum {
        UNSET_SELF_DEVICES = UCS_BIT(0),
        UNSET_SHM_DEVICES  = UCS_BIT(1)
    };

    void init()
    {
        m_err_count = 0;
        get_sockaddr();
        test_base::init();
        // entities will be created in a test
    }
};


UCS_TEST_P(test_ucp_sockaddr_different_tl_rsc, unset_devices_and_communicate)
{
    int variants = get_variant_value();

    // create entities with different set of MDs and TL resources on a client
    // and on a server to test non-homogeneous setups
    if (variants & UNSET_SELF_DEVICES) {
        if (is_self()) {
            UCS_TEST_SKIP_R("unable to run test for self transport with unset"
                            " self devices");
        }

        modify_config("SELF_DEVICES", "");
    }
    if (variants & UNSET_SHM_DEVICES) {
        modify_config("SHM_DEVICES", "");
    }
    push_config();

    // create a client with restrictions
    create_entity();

    pop_config();

    // create a server without restrictions
    if (!is_self()) {
        create_entity();
    }

    skip_loopback();
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_sockaddr_different_tl_rsc, all, "all")

class test_ucp_sockaddr_cm_switch : public test_ucp_sockaddr {
protected:
    ucp_rsc_index_t get_num_cms()
    {
        const ucp_worker_h worker    = sender().worker();
        ucp_rsc_index_t num_cm_cmpts = ucp_worker_num_cm_cmpts(worker);
        ucp_rsc_index_t num_cms      = 0;

        for (ucp_rsc_index_t cm_idx = 0; cm_idx < num_cm_cmpts; ++cm_idx) {
            if (worker->cms[cm_idx].cm != NULL) {
                num_cms++;
            }
        }

        return num_cms;
    }

    void check_cm_fallback()
    {
        if (get_num_cms() < 2) {
            UCS_TEST_SKIP_R("No CM for fallback to");
        }
    }
};

UCS_TEST_P(test_ucp_sockaddr_cm_switch,
           rereg_memory_on_cm_switch,
           "ZCOPY_THRESH=0", "TLS=rc",
           "SOCKADDR_TLS_PRIORITY=rdmacm,tcp")
{
    check_cm_fallback();
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_sockaddr_cm_switch, all, "all")

class test_ucp_sockaddr_cm_private_data : public test_ucp_sockaddr {
protected:
    ucp_rsc_index_t get_num_cms()
    {
        const ucp_worker_h worker    = sender().worker();
        ucp_rsc_index_t num_cm_cmpts = ucp_worker_num_cm_cmpts(worker);
        ucp_rsc_index_t num_cms      = 0;

        for (ucp_rsc_index_t cm_idx = 0; cm_idx < num_cm_cmpts; ++cm_idx) {
            if (worker->cms[cm_idx].cm != NULL) {
                num_cms++;
            }
        }

        return num_cms;
    }

    void check_cm_fallback()
    {
        if (get_num_cms() < 2) {
            UCS_TEST_SKIP_R("No CM for fallback to");
        }

        if (!m_test_addr.is_rdmacm_netdev()) {
            UCS_TEST_SKIP_R("RDMACM isn't allowed to be used on " +
                            m_test_addr.to_str());
        }
    }

    void check_rdmacm()
    {
        ucp_rsc_index_t num_cm_cmpts = receiver().ucph()->config.num_cm_cmpts;
        ucp_rsc_index_t cm_idx;

        if (!m_test_addr.is_rdmacm_netdev()) {
            UCS_TEST_SKIP_R("RDMACM isn't allowed to be used on " +
                            m_test_addr.to_str());
        }

        for (cm_idx = 0; cm_idx < num_cm_cmpts; ++cm_idx) {
            if (sender().worker()->cms[cm_idx].cm == NULL) {
                continue;
            }

            std::string cm_name = ucp_context_cm_name(sender().ucph(), cm_idx);
            if (cm_name.compare("rdmacm") == 0) {
                break;
            }
        }

        if (cm_idx == num_cm_cmpts) {
            UCS_TEST_SKIP_R("No RDMACM to check address packing");
        }
    }
};

UCS_TEST_P(test_ucp_sockaddr_cm_private_data,
           short_cm_private_data_fallback_to_next_cm,
           "TCP_CM_PRIV_DATA_LEN?=16", "SOCKADDR_TLS_PRIORITY=tcp,rdmacm")
{
    check_cm_fallback();
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr_cm_private_data,
           create_multiple_lanes_no_fallback_to_next_cm, "TLS=ud,rc,sm",
           "NUM_EPS=128", "SOCKADDR_TLS_PRIORITY=rdmacm")
{
    check_rdmacm();
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr_cm_private_data,
           create_multiple_lanes_have_fallback_to_next_cm, "TLS=ud,rc,sm,tcp",
           "NUM_EPS=128", "SOCKADDR_TLS_PRIORITY=rdmacm,tcp")
{
    check_cm_fallback();
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
    concurrent_disconnect(UCP_EP_CLOSE_MODE_FLUSH);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_sockaddr_cm_private_data, all, "all")


class test_ucp_sockaddr_check_lanes : public test_ucp_sockaddr {
};


UCS_TEST_SKIP_COND_P(test_ucp_sockaddr_check_lanes, check_rndv_lanes,
                     !cm_use_all_devices())
{
    listen_and_communicate(false, SEND_DIRECTION_BIDI);

    EXPECT_EQ(has_rndv_lanes(sender().ep()),
              has_rndv_lanes(receiver().ep()));

    concurrent_disconnect(UCP_EP_CLOSE_MODE_FLUSH);
}

UCP_INSTANTIATE_ALL_TEST_CASE(test_ucp_sockaddr_check_lanes)


class test_ucp_sockaddr_destroy_ep_on_err : public test_ucp_sockaddr {
public:
    test_ucp_sockaddr_destroy_ep_on_err() {
        configure_peer_failure_settings();
    }

    virtual ucp_ep_params_t get_server_ep_params() {
        ucp_ep_params_t params = test_ucp_sockaddr::get_server_ep_params();

        params.field_mask      |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                  UCP_EP_PARAM_FIELD_ERR_HANDLER       |
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
};

UCS_TEST_P(test_ucp_sockaddr_destroy_ep_on_err, empty) {
    listen_and_communicate(false, 0);
}

UCS_TEST_P(test_ucp_sockaddr_destroy_ep_on_err, s2c) {
    listen_and_communicate(false, SEND_DIRECTION_S2C);
}

UCS_TEST_P(test_ucp_sockaddr_destroy_ep_on_err, c2s) {
    listen_and_communicate(false, SEND_DIRECTION_C2S);
}

UCS_TEST_P(test_ucp_sockaddr_destroy_ep_on_err, bidi) {
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
}

UCS_TEST_P(test_ucp_sockaddr_destroy_ep_on_err, onesided_client_cforce) {
    listen_and_communicate(false, 0);
    scoped_log_handler slh(wrap_errors_logger);
    one_sided_disconnect(sender(),   UCP_EP_CLOSE_MODE_FORCE);
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr_destroy_ep_on_err, onesided_c2s_cforce) {
    listen_and_communicate(false, SEND_DIRECTION_C2S);
    scoped_log_handler slh(wrap_errors_logger);
    one_sided_disconnect(sender(),   UCP_EP_CLOSE_MODE_FORCE);
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr_destroy_ep_on_err, onesided_s2c_cforce) {
    listen_and_communicate(false, SEND_DIRECTION_S2C);
    scoped_log_handler slh(wrap_errors_logger);
    one_sided_disconnect(sender(),   UCP_EP_CLOSE_MODE_FORCE);
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr_destroy_ep_on_err, onesided_bidi_cforce) {
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
    scoped_log_handler slh(wrap_errors_logger);
    one_sided_disconnect(sender(),   UCP_EP_CLOSE_MODE_FORCE);
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr_destroy_ep_on_err, onesided_client_sforce) {
    listen_and_communicate(false, 0);
    scoped_log_handler slh(wrap_errors_logger);
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FORCE);
    one_sided_disconnect(sender(),   UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr_destroy_ep_on_err, onesided_c2s_sforce) {
    listen_and_communicate(false, SEND_DIRECTION_C2S);
    scoped_log_handler slh(wrap_errors_logger);
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FORCE);
    one_sided_disconnect(sender(),   UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr_destroy_ep_on_err, onesided_s2c_sforce) {
    listen_and_communicate(false, SEND_DIRECTION_S2C);
    scoped_log_handler slh(wrap_errors_logger);
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FORCE);
    one_sided_disconnect(sender(),   UCP_EP_CLOSE_MODE_FLUSH);
}

UCS_TEST_P(test_ucp_sockaddr_destroy_ep_on_err, onesided_bidi_sforce) {
    listen_and_communicate(false, SEND_DIRECTION_BIDI);
    scoped_log_handler slh(wrap_errors_logger);
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FORCE);
    one_sided_disconnect(sender(),   UCP_EP_CLOSE_MODE_FLUSH);
}

/* The test check that a client disconnection works fine when a server received
 * a conenction request, but a conenction wasn't fully established */
UCS_TEST_P(test_ucp_sockaddr_destroy_ep_on_err, create_and_destroy_immediately)
{
    ucp_test_base::entity::listen_cb_type_t listen_cb_type = cb_type();

    listen(listen_cb_type);

    {
        scoped_log_handler warn_slh(detect_warn_logger);
        scoped_log_handler error_slh(detect_error_logger);
        client_ep_connect();

        if (listen_cb_type == ucp_test_base::entity::LISTEN_CB_CONN) {
            /* Wait for either connection to a peer failed (e.g. no TL to create
             * after CM created a connection) or connection request is provided
             * by UCP */
            while ((m_err_count == 0) &&
                   receiver().is_conn_reqs_queue_empty()) {
                progress();
            }
        } else {
            /* Wait for EP being created on a server side */
            ASSERT_EQ(ucp_test_base::entity::LISTEN_CB_EP, listen_cb_type);
            if (!wait_for_server_ep(false)) {
                UCS_TEST_SKIP_R("cannot connect to server");
            }
        }

        /* Disconnect from a peer while conenction is not fully established with
         * a peer */
        one_sided_disconnect(sender(), UCP_EP_CLOSE_MODE_FORCE);

        /* Wait until either accepting a connection fails on a server side or
         * disconnection is detected by a server in case of a connection was
         * established successfully */
        ucs_time_t loop_end_limit = ucs_get_time() + ucs_time_from_sec(10.0);
        while ((ucs_get_time() < loop_end_limit) &&
               (m_err_count == 0) && (receiver().get_accept_err_num() == 0)) {
            progress();
        }

        EXPECT_TRUE((m_err_count != 0) ||
                    (receiver().get_accept_err_num() != 0));
    }

    /* Disconnect from a client if a connection was established */
    one_sided_disconnect(receiver(), UCP_EP_CLOSE_MODE_FORCE);
}

UCP_INSTANTIATE_ALL_TEST_CASE(test_ucp_sockaddr_destroy_ep_on_err)

class test_ucp_sockaddr_with_wakeup : public test_ucp_sockaddr {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        test_ucp_sockaddr::get_test_variants(variants, UCP_FEATURE_TAG |
                                             UCP_FEATURE_STREAM |
                                             UCP_FEATURE_WAKEUP);
    }
};

UCS_TEST_P(test_ucp_sockaddr_with_wakeup, wakeup) {
    listen_and_communicate(true, 0);
}

UCS_TEST_P(test_ucp_sockaddr_with_wakeup, wakeup_c2s) {
    listen_and_communicate(true, SEND_DIRECTION_C2S);
}

UCS_TEST_P(test_ucp_sockaddr_with_wakeup, wakeup_s2c) {
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
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        uint64_t features = UCP_FEATURE_TAG | UCP_FEATURE_STREAM |
                            UCP_FEATURE_RMA | UCP_FEATURE_AMO32 |
                            UCP_FEATURE_AMO64;
        test_ucp_sockaddr::get_test_variants(variants, features);
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
    virtual ~test_ucp_sockaddr_protocols() { }

    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        /* Atomics not supported for now because need to emulate the case
         * of using different device than the one selected by default on the
         * worker for atomic operations */
        uint64_t features = UCP_FEATURE_TAG | UCP_FEATURE_STREAM |
                            UCP_FEATURE_RMA | UCP_FEATURE_AM;

        add_variant_with_value(variants, features, TEST_MODIFIER_MT,
                               "mt", MULTI_THREAD_WORKER);
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

    typedef void (*stop_cb_t)(void *arg);

    void *do_unexp_recv(std::string &recv_buf, size_t size, void *sreq,
                        bool send_stop, bool recv_stop)
    {
        ucp_tag_recv_info_t recv_info = {};
        bool err_handling             = send_stop || recv_stop;
        ucp_tag_message_h message;

        do {
            short_progress_loop();
            message = ucp_tag_probe_nb(receiver().worker(),
                                       0, 0, 1, &recv_info);
        } while (message == NULL);

        EXPECT_EQ(size, recv_info.length);
        EXPECT_EQ(0,    recv_info.sender_tag);

        if (recv_stop) {
            disconnect(*this, receiver());
        }

        if (send_stop) {
            disconnect(*this, sender());
        }

        ucp_request_param_t recv_param = {};
        recv_param.op_attr_mask        = UCP_OP_ATTR_FIELD_CALLBACK;
        /* TODO: remove casting when changed to using NBX API */
        recv_param.cb.recv             = reinterpret_cast
                                         <ucp_tag_recv_nbx_callback_t>(
                                             !err_handling ? rtag_complete_cb :
                                             rtag_complete_err_handling_cb);
        return ucp_tag_msg_recv_nbx(receiver().worker(), &recv_buf[0], size,
                                    message, &recv_param);
    }

    void sreq_release(void *sreq) {
        if ((sreq == NULL) || !UCS_PTR_IS_PTR(sreq)) {
            return;
        }

        if (ucp_request_check_status(sreq) == UCS_INPROGRESS) {
            ucp_request_t *req = (ucp_request_t*)sreq - 1;
            req->flags        |= UCP_REQUEST_FLAG_COMPLETED;

            ucp_request_t *req_from_id;
            ucs_status_t status = ucp_send_request_get_by_id(
                    sender().worker(), req->id,&req_from_id, 1);
            if (status == UCS_OK) {
                EXPECT_EQ(req, req_from_id);
            }
        }

        ucp_request_release(sreq);
    }

    void extra_send_before_disconnect(entity &e, const std::string &send_buf,
                                      const ucp_request_param_t &send_param)
    {
        void *sreq = ucp_tag_send_nbx(e.ep(), &send_buf[0], send_buf.size(), 0,
                                      &send_param);
        request_wait(sreq);

        e.disconnect_nb(0, 0, UCP_EP_CLOSE_MODE_FORCE);
    }

    void test_tag_send_recv(size_t size, bool is_exp, bool is_sync = false,
                            bool send_stop = false, bool recv_stop = false)
    {
        bool err_handling_test = send_stop || recv_stop;
        unsigned num_iters     = err_handling_test ? 1 : m_num_iters;

        /* send multiple messages to test the protocol both before and after
         * connection establishment */
        for (int i = 0; i < num_iters; i++) {
            std::string send_buf(size, 'x');
            std::string recv_buf(size, 'y');

            void *rreq = NULL, *sreq = NULL;
            std::vector<void*> reqs;

            ucs::auto_ptr<scoped_log_handler> slh;
            if (err_handling_test) {
                slh.reset(new scoped_log_handler(wrap_errors_logger));
            }

            if (is_exp) {
                rreq = ucp_tag_recv_nb(receiver().worker(), &recv_buf[0], size,
                                       ucp_dt_make_contig(1), 0, 0,
                                       rtag_complete_cb);
                reqs.push_back(rreq);
            }

            ucp_request_param_t send_param = {};
            send_param.op_attr_mask        = UCP_OP_ATTR_FIELD_CALLBACK;
            /* TODO: remove casting when changed to using NBX API */
            send_param.cb.send             = reinterpret_cast
                                             <ucp_send_nbx_callback_t>(
                                                 !err_handling_test ? scomplete_cb :
                                                 scomplete_err_handling_cb);
            if (is_sync) {
                sreq = ucp_tag_send_sync_nbx(sender().ep(), &send_buf[0], size, 0,
                                             &send_param);
            } else {
                sreq = ucp_tag_send_nbx(sender().ep(), &send_buf[0], size, 0,
                                        &send_param);
            }
            reqs.push_back(sreq);

            if (!is_exp) {
                rreq = do_unexp_recv(recv_buf, size, sreq, send_stop,
                                     recv_stop);
                reqs.push_back(rreq);
            }

            /* Wait for completions of send and receive requests.
             * The requests could be completed with the following statuses:
             * - UCS_OK, when it was successfully sent before a peer failure was
             *   detected
             * - UCS_ERR_CANCELED, when it was purged from an UCP EP list of
             *   tracked requests
             * - UCS_ERR_* (e.g. UCS_ERR_ENDPOINT_TIMEOUT), when it was
             *   completed from an UCT transport with an error */
            requests_wait(reqs);

            if (!err_handling_test) {
                compare_buffers(send_buf, recv_buf);
            } else {
                wait_for_flag(&m_err_count);

                if (send_stop == false) {
                    extra_send_before_disconnect(sender(), send_buf, send_param);
                } else if (recv_stop == false) {
                    extra_send_before_disconnect(receiver(), send_buf, send_param);
                }
            }
        }
    }

    void wait_for_server_ep()
    {
        if (!test_ucp_sockaddr::wait_for_server_ep(false)) {
            UCS_TEST_ABORT("server endpoint is not created");
        }
    }

    void test_stream_send_recv(size_t size, bool is_exp)
    {
        /* send multiple messages to test the protocol both before and after
         * connection establishment */
        for (int i = 0; i < m_num_iters; i++) {
            std::string send_buf(size, 'x');
            std::string recv_buf(size, 'y');
            size_t recv_length;
            void *rreq, *sreq;

            if (is_exp) {
                wait_for_server_ep();
                rreq = ucp_stream_recv_nb(receiver().ep(), &recv_buf[0], size,
                                          ucp_dt_make_contig(1),
                                          rstream_complete_cb, &recv_length,
                                          UCP_STREAM_RECV_FLAG_WAITALL);
                sreq = ucp_stream_send_nb(sender().ep(), &send_buf[0], size,
                                          ucp_dt_make_contig(1), scomplete_cb,
                                          0);
            } else {
                sreq = ucp_stream_send_nb(sender().ep(), &send_buf[0], size,
                                          ucp_dt_make_contig(1), scomplete_cb,
                                          0);
                short_progress_loop();
                wait_for_server_ep();
                rreq = ucp_stream_recv_nb(receiver().ep(), &recv_buf[0], size,
                                          ucp_dt_make_contig(1),
                                          rstream_complete_cb, &recv_length,
                                          UCP_STREAM_RECV_FLAG_WAITALL);
            }

            request_wait(sreq);
            request_wait(rreq);

            compare_buffers(send_buf, recv_buf);
        }
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
        /* send multiple messages to test the protocol both before and after
         * connection establishment */
        for (int i = 0; i < m_num_iters; i++) {
            std::string send_buf(size, 'x');
            std::string recv_buf(size, 'y');

            ucp_mem_h memh;
            ucp_rkey_h rkey;

            register_mem(&sender(), &receiver(), &recv_buf[0], size, &memh,
                         &rkey);

            std::vector<void*> reqs;
            (this->*rma_func)(send_buf, recv_buf, rkey, reqs);

            while (!reqs.empty()) {
                request_wait(reqs.back());
                reqs.pop_back();
            }

            compare_buffers(send_buf, recv_buf);

            ucp_rkey_destroy(rkey);
            ucs_status_t status = ucp_mem_unmap(receiver().ucph(), memh);
            ASSERT_UCS_OK(status);
        }
    }

    void test_am_send_recv(size_t size, size_t hdr_size = 0ul)
    {
        /* send multiple messages to test the protocol both before and after
         * connection establishment */
        for (int i = 0; i < m_num_iters; i++) {
            std::string sb(size, 'x');
            std::string rb(size, 'y');
            std::string shdr(hdr_size, 'x');
            std::string rhdr(hdr_size, 'y');

            rx_am_msg_arg arg(receiver(), &rhdr[0], &rb[0]);
            set_am_data_handler(receiver(), 0, rx_am_msg_cb, &arg);

            ucp_request_param_t param = {};
            ucs_status_ptr_t sreq     = ucp_am_send_nbx(sender().ep(), 0,
                                                        &shdr[0], hdr_size,
                                                        &sb[0], size, &param);
            request_wait(sreq);
            wait_for_flag(&arg.received);
            // wait for receive request completion after 'received' flag set to
            // make sure AM receive handler was invoked and 'rreq' was posted
            request_wait(arg.rreq);
            EXPECT_TRUE(arg.received);

            compare_buffers(sb, rb);
            compare_buffers(shdr, rhdr);

            set_am_data_handler(receiver(), 0, NULL, NULL);
        }
    }

protected:
    enum {
        SEND_STOP = UCS_BIT(0),
        RECV_STOP = UCS_BIT(1)
    };

    static void disconnect(test_ucp_sockaddr_protocols &test, entity &e) {
        test.one_sided_disconnect(e, UCP_EP_CLOSE_MODE_FORCE);
        while (m_err_count == 0) {
            test.short_progress_loop();
        }
    }

private:
    static const unsigned m_num_iters;
};


const unsigned test_ucp_sockaddr_protocols::m_num_iters = 10;


UCS_TEST_P(test_ucp_sockaddr_protocols, stream_short_exp)
{
    test_stream_send_recv(1, true);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, stream_short_unexp)
{
    test_stream_send_recv(1, false);
}

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

UCS_TEST_P(test_ucp_sockaddr_protocols, am_header_only)
{
    ucp_worker_attr_t attr;
    attr.field_mask = UCP_WORKER_ATTR_FIELD_MAX_AM_HEADER;

    ASSERT_UCS_OK(ucp_worker_query(sender().worker(), &attr));
    test_am_send_recv(0, attr.max_am_header);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, am_bcopy_1k,
           "ZCOPY_THRESH=inf", "RNDV_THRESH=inf")
{
    test_am_send_recv(1 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, am_bcopy_64k,
           "ZCOPY_THRESH=inf", "RNDV_THRESH=inf")
{
    test_am_send_recv(64 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, am_zcopy_1k,
           "ZCOPY_THRESH=512", "RNDV_THRESH=inf")
{
    test_am_send_recv(1 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, am_zcopy_64k,
           "ZCOPY_THRESH=512", "RNDV_THRESH=inf")
{
    test_am_send_recv(64 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_sockaddr_protocols, am_rndv_64k, "RNDV_THRESH=0")
{
    test_am_send_recv(64 * UCS_KBYTE);
}


/* For DC case, allow fallback to UD if DC is not supported */
#define UCP_INSTANTIATE_CM_TEST_CASE(_test_case) \
    UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, dcudx, "dc_x,ud") \
    UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, ud, "ud_v") \
    UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, udx, "ud_x") \
    UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, rc, "rc_v") \
    UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, rcx, "rc_x") \
    UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, ib, "ib") \
    UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, tcp, "tcp") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, all, "all")

UCP_INSTANTIATE_CM_TEST_CASE(test_ucp_sockaddr_protocols)


class test_ucp_sockaddr_protocols_diff_config : public test_ucp_sockaddr_protocols
{
public:
    void init() {
        if (is_self()) {
            UCS_TEST_SKIP_R("self - same config");
        }

        m_err_count = 0;
        modify_config("CM_USE_ALL_DEVICES", cm_use_all_devices() ? "y" : "n");

        get_sockaddr();
        test_base::init();
    }

    static void
    get_test_variants(std::vector<ucp_test_variant>& variants) {
        uint64_t features = UCP_FEATURE_TAG | UCP_FEATURE_STREAM |
                            UCP_FEATURE_RMA | UCP_FEATURE_AM;

        add_variant_with_value(variants, features,
                               TEST_MODIFIER_CM_USE_ALL_DEVICES, "all_devs");
        add_variant_with_value(variants, features, 0, "not_all_devs");
    }

    void init_entity(const char *num_paths) {
        /* coverity[tainted_string_argument] */
        ucs::scoped_setenv num_paths_env("UCX_IB_NUM_PATHS", num_paths);
        create_entity();
    }

    void create_entities_and_connect(bool server_less_num_paths) {
        /* coverity[tainted_string_argument] */
        ucs::scoped_setenv max_eager_lanes_env("UCX_MAX_EAGER_LANES", "2");

        if (server_less_num_paths) {
            // create the client
            init_entity("2");
            // create the server
            init_entity("1");
        } else {
            // create the client
            init_entity("1");
            // create the server
            init_entity("2");
        }

        start_listener(cb_type());
        client_ep_connect();
    }
};


UCS_TEST_P(test_ucp_sockaddr_protocols_diff_config,
           diff_num_paths_small_msg_server_less_lanes)
{
    create_entities_and_connect(true);
    test_tag_send_recv(4 * UCS_KBYTE, false, false);
}

UCS_TEST_P(test_ucp_sockaddr_protocols_diff_config,
           diff_num_paths_large_msg_server_less_lanes)
{
    create_entities_and_connect(true);
    test_tag_send_recv(4 * UCS_MBYTE, false, false);
}

UCS_TEST_P(test_ucp_sockaddr_protocols_diff_config,
           diff_num_paths_small_msg_server_more_lanes)
{
    create_entities_and_connect(false);
    test_tag_send_recv(4 * UCS_KBYTE, false, false);
}

UCS_TEST_P(test_ucp_sockaddr_protocols_diff_config,
           diff_num_paths_large_msg_server_more_lanes)
{
    create_entities_and_connect(false);
    test_tag_send_recv(4 * UCS_MBYTE, false, false);
}

UCP_INSTANTIATE_CM_TEST_CASE(test_ucp_sockaddr_protocols_diff_config)


class test_ucp_sockaddr_protocols_err : public test_ucp_sockaddr_protocols {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        uint64_t features = UCP_FEATURE_TAG;
        test_ucp_sockaddr::get_test_variants_cm_mode(variants, features,
                                                     SEND_STOP, "send_stop");
        test_ucp_sockaddr::get_test_variants_cm_mode(variants, features,
                                                     RECV_STOP, "recv_stop");
        test_ucp_sockaddr::get_test_variants_cm_mode(variants, features,
                                                     SEND_STOP | RECV_STOP,
                                                     "bidi_stop");
    }

protected:
    test_ucp_sockaddr_protocols_err() {
        configure_peer_failure_settings();
    }

    void test_tag_send_recv(size_t size, bool is_exp,
                            bool is_sync = false) {
        /* warmup */
        test_ucp_sockaddr_protocols::test_tag_send_recv(size, is_exp, is_sync);

        /* run error-handling test */
        int variants = get_variant_value();
        test_ucp_sockaddr_protocols::test_tag_send_recv(size, is_exp, is_sync,
                                                        variants & SEND_STOP,
                                                        variants & RECV_STOP);
    }
};


UCS_TEST_P(test_ucp_sockaddr_protocols_err, tag_eager_32_unexp,
           "ZCOPY_THRESH=inf", "RNDV_THRESH=inf")
{
    test_tag_send_recv(32, false, false);
}

UCS_TEST_P(test_ucp_sockaddr_protocols_err, tag_zcopy_4k_unexp,
           "ZCOPY_THRESH=2k", "RNDV_THRESH=inf")
{
    test_tag_send_recv(4 * UCS_KBYTE, false, false);
}

UCS_TEST_P(test_ucp_sockaddr_protocols_err, tag_zcopy_64k_unexp,
           "ZCOPY_THRESH=2k", "RNDV_THRESH=inf")
{
    test_tag_send_recv(64 * UCS_KBYTE, false, false);
}

UCS_TEST_P(test_ucp_sockaddr_protocols_err, tag_eager_32_unexp_sync,
           "ZCOPY_THRESH=inf", "RNDV_THRESH=inf")
{
    test_tag_send_recv(32, false, true);
}

UCS_TEST_P(test_ucp_sockaddr_protocols_err, tag_zcopy_4k_unexp_sync,
           "ZCOPY_THRESH=2k", "RNDV_THRESH=inf")
{
    test_tag_send_recv(4 * UCS_KBYTE, false, true);
}

UCS_TEST_P(test_ucp_sockaddr_protocols_err, tag_zcopy_64k_unexp_sync,
           "ZCOPY_THRESH=2k", "RNDV_THRESH=inf")
{
    test_tag_send_recv(64 * UCS_KBYTE, false, true);
}

UCS_TEST_P(test_ucp_sockaddr_protocols_err, tag_rndv_unexp,
           "RNDV_THRESH=0", "RNDV_SCHEME=auto")
{
    test_tag_send_recv(64 * UCS_KBYTE, false, false);
}

UCS_TEST_P(test_ucp_sockaddr_protocols_err, tag_rndv_unexp_get_scheme,
           "RNDV_THRESH=0", "RNDV_SCHEME=get_zcopy")
{
    test_tag_send_recv(64 * UCS_KBYTE, false, false);
}

UCS_TEST_P(test_ucp_sockaddr_protocols_err, tag_rndv_unexp_put_scheme,
           "RNDV_THRESH=0", "RNDV_SCHEME=put_zcopy")
{
    test_tag_send_recv(64 * UCS_KBYTE, false, false);
}

UCP_INSTANTIATE_CM_TEST_CASE(test_ucp_sockaddr_protocols_err)
UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_sockaddr_protocols_err,
                                        rc_no_ud, "rc_mlx5,rc_verbs")


class test_ucp_sockaddr_protocols_err_sender
      : public test_ucp_sockaddr_protocols {
protected:
    virtual void init() {
        m_err_count = 0;
        modify_config("CM_USE_ALL_DEVICES", cm_use_all_devices() ? "y" : "n");
        /* receiver should try to read wrong data, instead of detecting error
           in keepalive process and closing the connection */
        disable_keepalive();
        get_sockaddr();
        ucp_test::init();
        skip_loopback();
        start_listener(cb_type());
        client_ep_connect();
    }

    test_ucp_sockaddr_protocols_err_sender() {
        configure_peer_failure_settings();
        m_env.push_back(new ucs::scoped_setenv("UCX_IB_REG_METHODS",
                                               "rcache,odp,direct"));
    }

    void entity_disconnect(entity &e)
    {
        void *close_req = e.disconnect_nb(0, 0, UCP_EP_CLOSE_MODE_FORCE);
        if (UCS_PTR_IS_PTR(close_req)) {
            ucs_status_t status = request_progress(close_req, { &e });
            ASSERT_EQ(UCS_ERR_CANCELED, status);
        }
    }

    /* This test is quite tricky: it checks for incorrect behavior on RNDV send
     * on CONNECT_TO_IFACE transports with memory invalidation support: in case
     * if sender EP was killed right after sent RTS then receiver may get
     * incorrect/corrupted data */
    void do_tag_rndv_killed_sender_test(size_t num_senders,
                                        size_t size = 64 * UCS_KBYTE,
                                        size_t num_sends = 1)
    {
        std::vector<ucp_tag_message_h> messages;
        std::vector<void*> reqs;
        ucs_status_t status;

        /* If the sumber of senders greater than 1, send the same buffer on
         * multiple connections to delay the completion of md_invalidate on
         * the closed connection */
        mem_buffer send_buf(size, UCS_MEMORY_TYPE_HOST);
        send_buf.pattern_fill(1, size);
        for (size_t sender_idx = 0; sender_idx < num_senders; ++sender_idx) {
            ucp_send_nbx_callback_t send_cb;

            if (sender_idx > 0) {
                send_cb = scomplete_always_ok_cbx;
                client_ep_connect(sender_idx);
            } else {
                send_cb = scomplete_cbx;
            }

            /* Warmup */
            send_recv(sender(), receiver(), send_recv_type(), false, cb_type(),
                      sender_idx);

            for (size_t i = 0; i < num_sends; ++i) {
                void *sreq = send(sender(), send_buf.ptr(), size,
                                  SEND_RECV_TAG, send_cb, NULL, sender_idx);
                ASSERT_TRUE(UCS_PTR_IS_PTR(sreq));
                ASSERT_EQ(UCS_INPROGRESS, ucp_request_check_status(sreq));
                reqs.push_back(sreq);

                /* Allow receiver to get RTS notification, but do not receive message
                 * body */
                ucp_tag_recv_info_t info;
                ucp_tag_message_h message = message_wait(receiver(), 0, 0, &info);
                ASSERT_NE((void*)NULL, message);
                ASSERT_EQ(UCS_INPROGRESS, ucp_request_check_status(sreq));

                messages.emplace_back(message);
            }
        }

        /* Ignore all errors - it is expected */
        scoped_log_handler slh(hide_errors_logger);

        /* Close the first sender's EP to force send operation to be completed
         * with CANCEL status */
        entity_disconnect(sender());

        mem_buffer extra_recv_buf(size, UCS_MEMORY_TYPE_HOST);
        extra_recv_buf.pattern_fill(2, size);
        for (size_t i = num_sends; i < messages.size(); ++i) {
            void *rreq = recv(receiver(), extra_recv_buf.ptr(), size,
                              messages[i], rtag_complete_always_ok_cbx, NULL);
            reqs.push_back(rreq);
        }

        status = requests_wait(reqs);
        ASSERT_EQ(UCS_ERR_CANCELED, status);
        ASSERT_TRUE(reqs.empty());

        /* Update send buffer by new data - emulation of free(buffer) */
        send_buf.pattern_fill(3, size);

        /* Complete receive operations */
        ucs::ptr_vector<mem_buffer> recv_bufs;
        for (size_t i = 0; i < num_sends; ++i) {
            mem_buffer *recv_buf = new mem_buffer(size, UCS_MEMORY_TYPE_HOST);
            recv_buf->pattern_fill(2, size);
            recv_bufs.push_back(recv_buf);

            void *rreq = recv(receiver(), recv_buf->ptr(), size, messages[i],
                              rtag_complete_check_data_cbx,
                              reinterpret_cast<void*>(recv_buf));
            reqs.push_back(rreq);
        }
        requests_wait(reqs);
    }

private:
    ucs::ptr_vector<ucs::scoped_setenv> m_env;
};


UCS_TEST_P(test_ucp_sockaddr_protocols_err_sender, tag_rndv_killed_sender,
           "RNDV_THRESH=0", "RNDV_SCHEME=get_zcopy")
{
    do_tag_rndv_killed_sender_test(1);
}

UCS_TEST_P(test_ucp_sockaddr_protocols_err_sender,
           tag_rndv_killed_sender_4_extra_senders, "RNDV_THRESH=0",
           "RNDV_SCHEME=get_zcopy")
{
    do_tag_rndv_killed_sender_test(5);
}

UCS_TEST_P(test_ucp_sockaddr_protocols_err_sender,
           tag_rndv_killed_sender_multiple_sends, "RNDV_THRESH=0",
           "RNDV_SCHEME=get_zcopy")
{
    size_t num_sends = ucs_max(100, 100000 / ucs::test_time_multiplier() /
                                    ucs::test_time_multiplier());
    do_tag_rndv_killed_sender_test(1, 128, num_sends);
}

UCS_TEST_P(test_ucp_sockaddr_protocols_err_sender,
           tag_rndv_killed_sender_4_extra_senders_multiple_sends,
           "RNDV_THRESH=0", "RNDV_SCHEME=get_zcopy")
{
    size_t num_sends = ucs_max(100, 100000 / ucs::test_time_multiplier() /
                                    ucs::test_time_multiplier());
    do_tag_rndv_killed_sender_test(4, 128, num_sends);
}

UCP_INSTANTIATE_CM_TEST_CASE(test_ucp_sockaddr_protocols_err_sender)
UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_sockaddr_protocols_err_sender,
                                        rc_no_ud, "rc_mlx5,rc_verbs")
