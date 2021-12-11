/**
* Copyright (C) Mellanox Technologies Ltd. 2017-2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
#include <uct/uct_test.h>

extern "C" {
#include <uct/api/uct.h>
#include <uct/api/v2/uct_v2.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/string.h>
#include <ucs/type/param.h>
#include <ucs/arch/atomic.h>
}

#include <queue>


class test_uct_sockaddr : public uct_test {
    friend class uct_test::entity;
protected:

    class client_user_data;
    typedef std::map<uct_ep_h, client_user_data*> ep_client_data_map_t;

    enum {
        TEST_STATE_CONNECT_REQUESTED             = UCS_BIT(0),
        TEST_STATE_CLIENT_CONNECTED              = UCS_BIT(1),
        TEST_STATE_SERVER_CONNECTED              = UCS_BIT(2),
        TEST_STATE_CLIENT_DISCONNECTED           = UCS_BIT(3),
        TEST_STATE_SERVER_DISCONNECTED           = UCS_BIT(4),
        TEST_STATE_SERVER_REJECTED               = UCS_BIT(5),
        TEST_STATE_CLIENT_GOT_REJECT             = UCS_BIT(6),
        TEST_STATE_CLIENT_GOT_ERROR              = UCS_BIT(7),
        TEST_STATE_CLIENT_GOT_SERVER_UNAVAILABLE = UCS_BIT(8)
    };

    enum {
        TEST_EP_FLAG_DISCONNECT_INITIATOR  = UCS_BIT(0),
        TEST_EP_FLAG_DISCONNECT_CB_INVOKED = UCS_BIT(1)
    };

    class client_user_data {
    public:
        client_user_data(test_uct_sockaddr &test, entity &entity,
                         unsigned ep_index) :
            m_test(test), m_entity(entity), m_ep_index(ep_index)
        {
        }

        test_uct_sockaddr* get_test() const
        {
            return &m_test;
        }

        entity* get_entity() const
        {
            return &m_entity;
        }

        unsigned get_ep_index() const
        {
            return m_ep_index;
        }

        uct_ep_h get_ep() const
        {
            return get_entity()->ep(get_ep_index());
        }

    private:
        test_uct_sockaddr &m_test;
        entity            &m_entity;
        unsigned          m_ep_index;
    };

public:
    test_uct_sockaddr() : m_state(0), m_server(NULL), m_client(NULL),
                          m_server_recv_req_cnt(0), m_client_connect_cb_cnt(0),
                          m_server_connect_cb_cnt(0),
                          m_server_disconnect_cnt(0), m_client_disconnect_cnt(0),
                          m_reject_conn_request(false),
                          m_server_start_disconnect(false),
                          m_delay_conn_reply(false), m_short_priv_data_len(0),
                          m_long_priv_data_len(0) {
        pthread_mutex_init(&m_ep_client_data_lock, NULL);
    }

    ~test_uct_sockaddr() {
        pthread_mutex_destroy(&m_ep_client_data_lock);
    }

    static std::vector<const resource*>
    enum_cm_resources(const std::string &cm_cmpt_name)
    {
        static std::vector<resource> all_resources;

        if (all_resources.empty()) {
            set_cm_resources(all_resources);
        }

        return filter_resources(all_resources,
                                resource::is_equal_component_name,
                                cm_cmpt_name);
    }

    void init() {
        struct {
            bool is_set;
            char cstr[UCS_SOCKADDR_STRING_LEN];
        } src_addr = {
            .is_set = false,
            .cstr   = {0}
        };

        uct_test::init();

        /* This address is accessible, as it was tested at the resource creation */
        m_listen_addr  = GetParam()->listen_sock_addr;
        m_connect_addr = GetParam()->connect_sock_addr;

        const ucs::sock_addr_storage &src_sock_addr =
                                                GetParam()->source_sock_addr;
        if (src_sock_addr.get_sock_addr_ptr() != NULL) {
            int sa_family   = src_sock_addr.get_sock_addr_ptr()->sa_family;
            const char *ret = inet_ntop(sa_family,
                                        src_sock_addr.get_sock_addr_in_buf(),
                                        src_addr.cstr, UCS_SOCKADDR_STRING_LEN);
            EXPECT_EQ(src_addr.cstr, ret);
            set_config((std::string("RDMA_CM_SOURCE_ADDRESS?=") +
                        src_addr.cstr).c_str());
            src_addr.is_set = true;
        }

        uint16_t port = ucs::get_port();
        m_listen_addr.set_port(port);
        m_connect_addr.set_port(port);

        m_server = uct_test::create_entity();
        m_entities.push_back(m_server);
        m_client = uct_test::create_entity();
        m_entities.push_back(m_client);

        m_client->max_conn_priv = m_client->cm_attr().max_conn_priv;
        m_server->max_conn_priv = m_server->cm_attr().max_conn_priv;

        m_short_priv_data_len = 20;
        m_long_priv_data_len  = 420 * UCS_KBYTE;

        m_short_priv_data.resize(m_short_priv_data_len);
        ucs::fill_random(m_short_priv_data);

        m_long_priv_data.resize(m_long_priv_data_len);
        ucs::fill_random(m_long_priv_data);

        UCS_TEST_MESSAGE << "Testing " << GetParam()->component_name << " on "
                         << m_listen_addr << " interface "
                         << GetParam()->dev_name
                         << (src_addr.is_set ?
                             (std::string(" with RDMA_CM_SOURCE_ADDRESS=") +
                              src_addr.cstr) : "");
    }

    void cleanup() {
        {
            ucs::scoped_mutex_lock lock(m_ep_client_data_lock);
            ASSERT_TRUE(m_ep_client_data.empty());
        }

        uct_test::cleanup();
    }

protected:

    void start_listen(uct_cm_listener_conn_request_callback_t server_conn_req_cb) {
        uct_listener_params_t params;

        params.field_mask      = UCT_LISTENER_PARAM_FIELD_CONN_REQUEST_CB |
                                 UCT_LISTENER_PARAM_FIELD_USER_DATA;
        params.conn_request_cb = server_conn_req_cb;
        params.user_data       = static_cast<test_uct_sockaddr *>(this);

        ucs_time_t deadline = ucs::get_deadline();
        ucs_status_t status;
        do {
            status = m_server->listen(m_listen_addr, params);
            if (status == UCS_ERR_BUSY) {
                m_listen_addr.set_port(ucs::get_port());
            } else {
                break;
            }
        } while (ucs_get_time() < deadline);

        ASSERT_EQ(UCS_OK, status);
        m_connect_addr.set_port(m_listen_addr.get_port());
    }

    void handle_client_connecting_status(ucs_status_t status) {
        switch (status) {
        case UCS_OK:
            return;
        case UCS_ERR_REJECTED:
            m_state |= TEST_STATE_CLIENT_GOT_REJECT;
            return;
        case UCS_ERR_UNREACHABLE:
        case UCS_ERR_NOT_CONNECTED:
        case UCS_ERR_CONNECTION_RESET:
            m_state |= TEST_STATE_CLIENT_GOT_SERVER_UNAVAILABLE;
            return;
        default:
            m_state |= TEST_STATE_CLIENT_GOT_ERROR;
            return;
        }
    }

    void connect(uct_test::entity &e, unsigned index,
                 uct_ep_disconnect_cb_t disconnect_cb,
                 const ucs::sock_addr_storage *local_addr = NULL) {
        client_user_data *user_data = new client_user_data(*this, e, index);
        ucs::scoped_async_lock lock(e.async());
        e.connect_to_sockaddr(index, m_connect_addr, local_addr,
                              client_resolve_cb, client_connect_cb,
                              disconnect_cb, user_data);
        add_user_data(user_data);
    }

    void connect(unsigned index = 0) {
        connect(*m_client, index, client_disconnect_cb);
    }

    void listen_and_connect(bool specify_src_addr = false) {
        start_listen(test_uct_sockaddr::conn_request_cb);
        connect(*m_client, 0, client_disconnect_cb,
                specify_src_addr ? &GetParam()->connect_sock_addr : NULL);

        wait_for_bits(&m_state, TEST_STATE_CONNECT_REQUESTED);
        EXPECT_TRUE(m_state & TEST_STATE_CONNECT_REQUESTED);
    }

    void ep_query(unsigned index = 0) {
        uct_ep_attr_t attr;
        ucs_status_t status;

        attr.field_mask = UCT_EP_ATTR_FIELD_LOCAL_SOCKADDR |
                          UCT_EP_ATTR_FIELD_REMOTE_SOCKADDR;
        status = uct_ep_query(m_server->ep(index), &attr);
        ASSERT_UCS_OK(status);

        EXPECT_EQ(m_connect_addr, attr.local_address);

        /* The ports are expected to be different. Ignore them. */
        ucs_sockaddr_set_port((struct sockaddr*)&attr.remote_address,
                              m_connect_addr.get_port());
        EXPECT_EQ(m_connect_addr, attr.remote_address);

        memset(&attr, 0, sizeof(attr));
        attr.field_mask = UCT_EP_ATTR_FIELD_LOCAL_SOCKADDR |
                          UCT_EP_ATTR_FIELD_REMOTE_SOCKADDR;
        status = uct_ep_query(m_client->ep(index), &attr);
        ASSERT_UCS_OK(status);

        /* The ports are expected to be different. Ignore them. */
        ucs_sockaddr_set_port((struct sockaddr*)&attr.local_address,
                              m_connect_addr.get_port());
        EXPECT_EQ(m_connect_addr, attr.local_address);

        EXPECT_EQ(m_connect_addr, attr.remote_address);
    }

    size_t priv_data_do_pack(size_t pack_limit, void *priv_data) {
        if (pack_limit < m_long_priv_data_len) {
            /* small private data length */
            memcpy(priv_data, m_short_priv_data.data(), m_short_priv_data_len);
            return m_short_priv_data_len;
        } else {
            /* large private data length (tcp_sockcm) */
            memcpy(priv_data, m_long_priv_data.data(), m_long_priv_data_len);
            return m_long_priv_data_len;
        }
    }

    ssize_t common_priv_data_cb(size_t pack_limit, void *priv_data) {
        size_t priv_data_len = priv_data_do_pack(pack_limit, priv_data);

        EXPECT_LE(priv_data_len, pack_limit);
        return priv_data_len;
    }

    static ucs_status_t
    client_resolve_cb(void *user_data, const uct_cm_ep_resolve_args_t *args)
    {
        client_user_data *sa_user_data =
                reinterpret_cast<client_user_data*>(user_data);
        test_uct_sockaddr *self = sa_user_data->get_test();
        std::vector<char> priv_data_buf(self->m_client->max_conn_priv);
        ucs_status_t status;
        ssize_t packed;

        status = UCS_PARAM_VALUE(UCT_CM_EP_RESOLVE_ARGS_FIELD, args, status,
                                 STATUS, UCS_OK);

        self->handle_client_connecting_status(status);
        if (status != UCS_OK) {
            goto err;
        }

        packed = self->common_priv_data_cb(priv_data_buf.size(),
                                           priv_data_buf.data());
        if (packed < 0) {
            status = ucs_status_t(packed);
            goto err;
        }

        uct_ep_connect_params_t params;
        params.field_mask          = UCT_EP_CONNECT_PARAM_FIELD_PRIVATE_DATA |
                                     UCT_EP_CONNECT_PARAM_FIELD_PRIVATE_DATA_LENGTH;
        params.private_data        = priv_data_buf.data();
        params.private_data_length = packed;
        return uct_ep_connect(sa_user_data->get_ep(), &params);

    err:
        self->del_user_data(sa_user_data);
        return status;
    }

    static void check_connection_status(ucs_status_t status, bool can_fail)
    {
        if (can_fail) {
            ASSERT_TRUE((status == UCS_OK) ||
                        (status == UCS_ERR_CONNECTION_RESET) ||
                        (status == UCS_ERR_NOT_CONNECTED));
        } else {
            ASSERT_UCS_OK(status);
        }
    }

    virtual void accept(uct_cm_h cm, uct_conn_request_h conn_request,
                        uct_cm_ep_server_conn_notify_callback_t notify_cb,
                        uct_ep_disconnect_cb_t disconnect_cb,
                        void *user_data, bool can_fail)
    {
        std::vector<char> priv_data_buf(m_server->max_conn_priv);
        uct_ep_params_t ep_params;
        ucs_status_t status;
        uct_ep_h ep;

        ASSERT_TRUE(m_server->listener());

        ssize_t packed = common_priv_data_cb(priv_data_buf.size(),
                                             priv_data_buf.data());
        ASSERT_GT(packed, 0);

        m_server->reserve_ep(m_server->num_eps());

        ep_params.field_mask = UCT_EP_PARAM_FIELD_CM                        |
                               UCT_EP_PARAM_FIELD_CONN_REQUEST              |
                               UCT_EP_PARAM_FIELD_USER_DATA                 |
                               UCT_EP_PARAM_FIELD_SOCKADDR_NOTIFY_CB_SERVER |
                               UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB    |
                               UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS         |
                               UCT_EP_PARAM_FIELD_PRIV_DATA                 |
                               UCT_EP_PARAM_FIELD_PRIV_DATA_LENGTH;

        ep_params.cm                  = cm;
        ep_params.conn_request        = conn_request;
        ep_params.sockaddr_cb_flags   = UCT_CB_FLAG_ASYNC;
        ep_params.sockaddr_cb_server  = notify_cb;
        ep_params.disconnect_cb       = disconnect_cb;
        ep_params.user_data           = user_data;
        ep_params.private_data        = priv_data_buf.data();
        ep_params.private_data_length = packed;

        status = uct_ep_create(&ep_params, &ep);
        check_connection_status(status, can_fail);
        if (status == UCS_OK) {
            m_server->eps().back().reset(ep, uct_ep_destroy);
        }
    }

    virtual void server_accept(entity *server, uct_conn_request_h conn_request,
                               uct_cm_ep_server_conn_notify_callback_t notify_cb,
                               uct_ep_disconnect_cb_t disconnect_cb,
                               void *user_data, bool can_fail)
    {
        ucs::scoped_async_lock listen_lock(m_server->async());
        ucs::scoped_async_lock accept_lock(server->async());
        accept(server->cm(), conn_request, notify_cb, disconnect_cb,
               user_data, can_fail);
    }

    void verify_remote_data(const void *remote_data, size_t remote_length)
    {
        std::vector<char> r_data((char*)(remote_data),
                                 (char*)(remote_data) + remote_length);

        if (remote_length == m_short_priv_data_len) {
            EXPECT_EQ(m_short_priv_data, r_data);
        } else if (remote_length == m_long_priv_data_len) {
            EXPECT_EQ(m_long_priv_data, r_data);
        } else {
            UCS_TEST_ABORT("wrong data length received " << remote_length);
        }
    }

    /*
     * Common section for the server's handling of a connection request.
     * Process the connection request and check if the server's accept is
     * required for the calling test.
     *
     * return true if the server should accept the connection request and
     * false if not.
     */
    static bool common_conn_request(uct_listener_h listener, void *arg,
                                    const uct_cm_listener_conn_request_args_t
                                    *conn_req_args) {
        test_uct_sockaddr *self = reinterpret_cast<test_uct_sockaddr *>(arg);
        ucs_sock_addr_t m_connect_addr_sock_addr =
                        self->m_connect_addr.to_ucs_sock_addr();
        uct_conn_request_h conn_request;
        const uct_cm_remote_data_t *remote_data;
        uint16_t client_port;
        ucs_status_t status;

        EXPECT_TRUE(ucs_test_all_flags(conn_req_args->field_mask,
                                       (UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_CONN_REQUEST |
                                        UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_REMOTE_DATA  |
                                        UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_CLIENT_ADDR)));

        conn_request = conn_req_args->conn_request;
        remote_data  = conn_req_args->remote_data;

        /* check the address of the remote client */
        EXPECT_EQ(0, ucs_sockaddr_ip_cmp(m_connect_addr_sock_addr.addr,
                                         conn_req_args->client_address.addr));

        status = ucs_sockaddr_get_port(conn_req_args->client_address.addr, &client_port);
        ASSERT_UCS_OK(status);
        EXPECT_GT(client_port, 0);

        self->verify_remote_data(remote_data->conn_priv_data, remote_data->conn_priv_data_length);

        self->m_state |= TEST_STATE_CONNECT_REQUESTED;

        if (self->m_delay_conn_reply) {
            self->m_delayed_conn_reqs.push(conn_request);
        } else if (self->m_reject_conn_request) {
            status = uct_listener_reject(listener, conn_request);
            ASSERT_UCS_OK(status);
            self->m_state |= TEST_STATE_SERVER_REJECTED;
        } else {
            /* do regular server accept */
            return true;
        }

        return false;
    }

    static void
    conn_request_cb(uct_listener_h listener, void *arg,
                    const uct_cm_listener_conn_request_args_t *conn_req_args) {
        test_uct_sockaddr *self = reinterpret_cast<test_uct_sockaddr *>(arg);

        if (self->common_conn_request(listener, arg, conn_req_args)) {
            EXPECT_TRUE(conn_req_args->field_mask &
                        UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_CONN_REQUEST);
            self->server_accept(self->m_server, conn_req_args->conn_request,
                                server_connect_cb, server_disconnect_cb, self,
                                false);
        }

        ucs_memory_cpu_store_fence();
        self->m_server_recv_req_cnt++;
    }


    static void
    server_connect_cb(uct_ep_h ep, void *arg,
                      const uct_cm_ep_server_conn_notify_args_t *notify_args) {
        test_uct_sockaddr *self = reinterpret_cast<test_uct_sockaddr *>(arg);

        self->m_server_connect_cb_cnt++;
        if ((notify_args->field_mask &
             UCT_CM_EP_SERVER_CONN_NOTIFY_ARGS_FIELD_STATUS) &&
            (notify_args->status != UCS_OK)) {
            return;
        }

        self->m_state |= TEST_STATE_SERVER_CONNECTED;
    }

    static void
    client_connect_cb(uct_ep_h ep, void *arg,
                      const uct_cm_ep_client_connect_args_t *connect_args) {
        client_user_data *sa_user_data =
                reinterpret_cast<client_user_data*>(arg);
        test_uct_sockaddr *self = sa_user_data->get_test();

        const uct_cm_remote_data_t *remote_data;
        ucs_status_t status;

        EXPECT_TRUE(ucs_test_all_flags(connect_args->field_mask,
                                       (UCT_CM_EP_CLIENT_CONNECT_ARGS_FIELD_REMOTE_DATA |
                                        UCT_CM_EP_CLIENT_CONNECT_ARGS_FIELD_STATUS)));

        remote_data = connect_args->remote_data;
        status      = connect_args->status;

        self->handle_client_connecting_status(status);
        if (status == UCS_OK) {
            EXPECT_TRUE(ucs_test_all_flags(remote_data->field_mask,
                                           (UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA_LENGTH |
                                            UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA)));

            self->verify_remote_data(remote_data->conn_priv_data,
                                     remote_data->conn_priv_data_length);

            ASSERT_EQ(sa_user_data->get_ep(), ep);
            status = uct_cm_client_ep_conn_notify(ep);
            ASSERT_UCS_OK(status);

            self->m_state |= TEST_STATE_CLIENT_CONNECTED;
            self->m_client_connect_cb_cnt++;
        } else {
            self->del_user_data(sa_user_data);
        }
    }

    static void
    server_disconnect_cb(uct_ep_h ep, void *arg) {
        test_uct_sockaddr *self = reinterpret_cast<test_uct_sockaddr *>(arg);

        if (!(self->m_server_start_disconnect)) {
            self->m_server->disconnect(ep);
        }

        self->m_state |= TEST_STATE_SERVER_DISCONNECTED;
        self->m_server_disconnect_cnt++;
    }

    static void client_disconnect_cb(uct_ep_h ep, void *arg) {
        client_user_data *sa_user_data =
                reinterpret_cast<client_user_data*>(arg);
        test_uct_sockaddr *self = sa_user_data->get_test();

        if (self->m_server_start_disconnect) {
            /* if the server was the one who initiated the disconnect flow,
             * the client should also disconnect its ep from the server in
             * its disconnect cb */
            self->m_client->disconnect(ep);
        }

        self->m_state |= TEST_STATE_CLIENT_DISCONNECTED;
        self->m_client_disconnect_cnt++;
        self->del_user_data(sa_user_data);
    }

    void cm_disconnect(entity *ent) {
        size_t i;

        /* Disconnect all the existing endpoints */
        for (i = 0; i < ent->num_eps(); ++i) {
            ent->disconnect(ent->ep(i));
        }

        wait_for_bits(&m_state, TEST_STATE_CLIENT_DISCONNECTED |
                                TEST_STATE_SERVER_DISCONNECTED);
        EXPECT_TRUE(ucs_test_all_flags(m_state, (TEST_STATE_SERVER_DISCONNECTED |
                                                 TEST_STATE_CLIENT_DISCONNECTED)));
    }

    void wait_for_client_server_counters(volatile int *server_cnt,
                                         volatile int *client_cnt, int val,
                                         double timeout = 10 * DEFAULT_TIMEOUT_SEC) {
        ucs_time_t deadline;

        deadline = ucs_get_time() + ucs_time_from_sec(timeout) *
                                    ucs::test_time_multiplier();

        while (((*server_cnt < val) || (*client_cnt < val)) &&
               (ucs_get_time() < deadline)) {
            progress();
        }
    }

    void test_delayed_server_response(bool reject, bool early_destroy)
    {
        ucs_status_t status;
        ucs_time_t deadline;

        m_delay_conn_reply = true;

        listen_and_connect();

        EXPECT_FALSE(m_state &
                     (TEST_STATE_SERVER_CONNECTED  | TEST_STATE_CLIENT_CONNECTED |
                      TEST_STATE_CLIENT_GOT_REJECT | TEST_STATE_CLIENT_GOT_ERROR |
                      TEST_STATE_CLIENT_GOT_SERVER_UNAVAILABLE));

        if (early_destroy) {
            {
                ucs::scoped_mutex_lock lock(m_ep_client_data_lock);
                for (int i = 0; i < m_client->num_eps(); ++i) {
                    del_user_data_no_lock(m_client->ep(i));
                }
            }

            m_client->destroy_eps();
        }

        deadline = ucs_get_time() + ucs_time_from_sec(DEFAULT_TIMEOUT_SEC) *
                                    ucs::test_time_multiplier();

        while ((m_server_recv_req_cnt == 0) && (ucs_get_time() < deadline)) {
            progress();
        }
        ASSERT_EQ(1, m_server_recv_req_cnt);
        ucs_memory_cpu_load_fence();

        if (reject) {
            /* wrap errors since a reject is expected */
            scoped_log_handler slh(detect_reject_error_logger);

            status = uct_listener_reject(m_server->listener(),
                                         m_delayed_conn_reqs.front());
            check_connection_status(status, early_destroy);
            if (!early_destroy) {
                wait_for_bits(&m_state, TEST_STATE_CLIENT_GOT_REJECT);
                EXPECT_TRUE(m_state & TEST_STATE_CLIENT_GOT_REJECT);
            }
        } else {
            server_accept(m_server, m_delayed_conn_reqs.front(),
                          server_connect_cb, server_disconnect_cb, this,
                          early_destroy);
            if (!early_destroy) {
                wait_for_bits(&m_state, TEST_STATE_SERVER_CONNECTED |
                                        TEST_STATE_CLIENT_CONNECTED);
                EXPECT_TRUE(ucs_test_all_flags(m_state, TEST_STATE_SERVER_CONNECTED |
                                                        TEST_STATE_CLIENT_CONNECTED));
                cm_disconnect(m_client);
            }
        }

        m_delayed_conn_reqs.pop();
    }

    static ucs_log_func_rc_t
    detect_addr_route_error_logger(const char *file, unsigned line, const char *function,
                                   ucs_log_level_t level,
                                   const ucs_log_component_config_t *comp_conf,
                                   const char *message, va_list ap)
    {
        if (level == UCS_LOG_LEVEL_ERROR) {
            std::string err_str = format_message(message, ap);
            if ((strstr(err_str.c_str(), "client: got error event RDMA_CM_EVENT_ADDR_ERROR"))  ||
                (strstr(err_str.c_str(), "client: got error event RDMA_CM_EVENT_ROUTE_ERROR")) ||
                (strstr(err_str.c_str(), "rdma_resolve_route(to addr=240.0.0.0")) ||
                (strstr(err_str.c_str(), "error event on client ep"))) {
                UCS_TEST_MESSAGE << err_str;
                return UCS_LOG_FUNC_RC_STOP;
            }
        }
        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    static ucs_log_func_rc_t
    detect_reject_error_logger(const char *file, unsigned line, const char *function,
                               ucs_log_level_t level,
                               const ucs_log_component_config_t *comp_conf,
                               const char *message, va_list ap)
    {
        if (level == UCS_LOG_LEVEL_ERROR) {
            std::string err_str = format_message(message, ap);
            if (strstr(err_str.c_str(), "client: got error event RDMA_CM_EVENT_REJECTED")) {
                UCS_TEST_MESSAGE << err_str;
                return UCS_LOG_FUNC_RC_STOP;
            }
        }
        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    static ucs_log_func_rc_t
    detect_double_disconnect_error_logger(const char *file, unsigned line,
                                          const char *function, ucs_log_level_t level,
                                          const ucs_log_component_config_t *comp_conf,
                                          const char *message, va_list ap)
    {
        if (level == UCS_LOG_LEVEL_ERROR) {
            std::string err_str = format_message(message, ap);
            if (err_str.find("duplicate call of uct_ep_disconnect") !=
                std::string::npos) {
                UCS_TEST_MESSAGE << err_str;
                return UCS_LOG_FUNC_RC_STOP;
            }
        }
        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    void basic_listen_connect_disconnect() {
        listen_and_connect();

        wait_for_bits(&m_state, TEST_STATE_SERVER_CONNECTED |
                                TEST_STATE_CLIENT_CONNECTED);
        EXPECT_TRUE(ucs_test_all_flags(m_state, (TEST_STATE_SERVER_CONNECTED |
                                                 TEST_STATE_CLIENT_CONNECTED)));

        cm_disconnect(m_client);
    }

    void add_user_data(client_user_data *user_data) {
        ucs::scoped_mutex_lock lock(m_ep_client_data_lock);

        EXPECT_NE(uct_ep_h(NULL), user_data->get_ep());
        EXPECT_TRUE(m_ep_client_data.insert(std::make_pair(user_data->get_ep(),
                                                           user_data)).second);
    }

    void del_user_data(client_user_data *user_data) {
        ucs::scoped_mutex_lock lock(m_ep_client_data_lock);
        del_user_data_no_lock(user_data->get_ep());
    }

    void release_user_data() {
        ucs::scoped_mutex_lock lock(m_ep_client_data_lock);
        while (!m_ep_client_data.empty()) {
            del_user_data_no_lock(m_ep_client_data.begin()->first);
        }
    }

private:
    void del_user_data_no_lock(uct_ep_h ep) {
        ep_client_data_map_t::iterator it = m_ep_client_data.find(ep);

        EXPECT_NE(m_ep_client_data.end(), it) << "ep: " << ep;
        delete it->second;
        m_ep_client_data.erase(it);
    }

protected:
    ucs::sock_addr_storage m_listen_addr, m_connect_addr;
    uint64_t               m_state;
    entity                 *m_server;
    entity                 *m_client;
    volatile int           m_server_recv_req_cnt, m_client_connect_cb_cnt,
                           m_server_connect_cb_cnt;
    volatile int           m_server_disconnect_cnt, m_client_disconnect_cnt;
    bool                   m_reject_conn_request;
    bool                   m_server_start_disconnect;
    bool                   m_delay_conn_reply;
    std::queue<uct_conn_request_h> m_delayed_conn_reqs;
    size_t                 m_short_priv_data_len, m_long_priv_data_len;
    std::vector<char>      m_short_priv_data;
    std::vector<char>      m_long_priv_data;
    pthread_mutex_t        m_ep_client_data_lock;
    ep_client_data_map_t   m_ep_client_data;
};


UCS_TEST_P(test_uct_sockaddr, cm_query)
{
    ucs_status_t status;
    size_t i;

    for (i = 0; i < m_entities.size(); ++i) {
        uct_cm_attr_t attr;
        attr.field_mask = UCT_CM_ATTR_FIELD_MAX_CONN_PRIV;
        status = uct_cm_query(m_entities.at(i).cm(), &attr);
        ASSERT_UCS_OK(status);
        EXPECT_LT(0ul, attr.max_conn_priv);
    }
}

UCS_TEST_P(test_uct_sockaddr, listener_query)
{
    uct_listener_attr_t attr;
    ucs_status_t status;
    uint16_t port;
    char m_listener_ip_port_str[UCS_SOCKADDR_STRING_LEN];
    char attr_addr_ip_port_str[UCS_SOCKADDR_STRING_LEN];

    start_listen(test_uct_sockaddr::conn_request_cb);

    attr.field_mask = UCT_LISTENER_ATTR_FIELD_SOCKADDR;
    status = uct_listener_query(m_server->listener(), &attr);
    ASSERT_UCS_OK(status);

    ucs_sockaddr_str(m_listen_addr.get_sock_addr_ptr(), m_listener_ip_port_str,
                     UCS_SOCKADDR_STRING_LEN);
    ucs_sockaddr_str((struct sockaddr*)&attr.sockaddr, attr_addr_ip_port_str,
                     UCS_SOCKADDR_STRING_LEN);
    EXPECT_EQ(strcmp(m_listener_ip_port_str, attr_addr_ip_port_str), 0);

    status = ucs_sockaddr_get_port((struct sockaddr*)&attr.sockaddr, &port);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(m_listen_addr.get_port(), port);
}

UCS_TEST_P(test_uct_sockaddr, ep_query)
{
    listen_and_connect();

    wait_for_bits(&m_state, TEST_STATE_SERVER_CONNECTED |
                            TEST_STATE_CLIENT_CONNECTED);
    EXPECT_TRUE(ucs_test_all_flags(m_state, (TEST_STATE_SERVER_CONNECTED |
                                             TEST_STATE_CLIENT_CONNECTED)));

    ep_query();

    cm_disconnect(m_client);
}

UCS_TEST_P(test_uct_sockaddr, set_local_sockaddr)
{
    listen_and_connect(true);
    wait_for_bits(&m_state, TEST_STATE_SERVER_CONNECTED |
                            TEST_STATE_CLIENT_CONNECTED);
    EXPECT_TRUE(ucs_test_all_flags(m_state, TEST_STATE_SERVER_CONNECTED |
                                            TEST_STATE_CLIENT_CONNECTED));
    ep_query();
    cm_disconnect(m_client);
}

UCS_TEST_P(test_uct_sockaddr, cm_open_listen_close)
{
    basic_listen_connect_disconnect();
}

UCS_TEST_P(test_uct_sockaddr, cm_open_listen_close_large_priv_data)
{
    m_entities.clear();

    /* Set the values for max send/recv socket buffers (for tcp_sockcm) to
     * small enough values to have the send/recv of a large data buffer in
     * batches, and not all data at once.
     * Set the value of the transport's private data length to a large enough
     * value to be able to send/recv the batches.
     * A transport for which these values are not configurable, like rdmacm,
     * these operations will fail and have no effect. */
    if (m_cm_config) {
        /* coverity[check_return] */
        uct_config_modify(m_cm_config, "PRIV_DATA_LEN", "900KB");
        uct_config_modify(m_cm_config, "SNDBUF", "100KB");
        uct_config_modify(m_cm_config, "RCVBUF", "100KB");
    }

    /* recreate m_server and m_client with the above env parameters changed */
    init();
    basic_listen_connect_disconnect();
}

UCS_TEST_P(test_uct_sockaddr, cm_open_listen_kill_server)
{
    listen_and_connect();

    wait_for_bits(&m_state, TEST_STATE_SERVER_CONNECTED |
                            TEST_STATE_CLIENT_CONNECTED);
    EXPECT_TRUE(ucs_test_all_flags(m_state, (TEST_STATE_SERVER_CONNECTED |
                                             TEST_STATE_CLIENT_CONNECTED)));

    EXPECT_EQ(1ul, m_entities.remove(m_server));
    m_server = NULL;

    wait_for_bits(&m_state, TEST_STATE_CLIENT_DISCONNECTED);
    EXPECT_TRUE(m_state & TEST_STATE_CLIENT_DISCONNECTED);
}

UCS_TEST_P(test_uct_sockaddr, cm_server_reject)
{
    m_reject_conn_request = true;

    /* wrap errors since a reject is expected */
    scoped_log_handler slh(detect_reject_error_logger);

    listen_and_connect();

    wait_for_bits(&m_state, TEST_STATE_SERVER_REJECTED |
                            TEST_STATE_CLIENT_GOT_REJECT);
    EXPECT_TRUE(ucs_test_all_flags(m_state, (TEST_STATE_SERVER_REJECTED |
                                             TEST_STATE_CLIENT_GOT_REJECT)));

    EXPECT_FALSE((m_state &
                 (TEST_STATE_SERVER_CONNECTED | TEST_STATE_CLIENT_CONNECTED)));
}

UCS_TEST_P(test_uct_sockaddr, many_conns_on_client)
{
    int num_conns_on_client = ucs_max(2, 100 / ucs::test_time_multiplier());

    m_server_start_disconnect = true;

    /* Listen */
    start_listen(conn_request_cb);

    /* Connect */
    /* multiple clients, on the same cm, connecting to the same server */
    for (int i = 0; i < num_conns_on_client; ++i) {
        connect(i);
    }

    /* wait for the server to connect to all the endpoints on the cm */
    wait_for_client_server_counters(&m_server_connect_cb_cnt,
                                    &m_client_connect_cb_cnt,
                                    num_conns_on_client);

    EXPECT_EQ(num_conns_on_client, m_server_recv_req_cnt);
    EXPECT_EQ(num_conns_on_client, m_client_connect_cb_cnt);
    EXPECT_EQ(num_conns_on_client, m_server_connect_cb_cnt);
    EXPECT_EQ(num_conns_on_client, (int)m_client->num_eps());
    EXPECT_EQ(num_conns_on_client, (int)m_server->num_eps());

    /* Disconnect */
    cm_disconnect(m_server);

    /* wait for disconnect to complete */
    wait_for_client_server_counters(&m_server_disconnect_cnt,
                                    &m_client_disconnect_cnt,
                                    num_conns_on_client);

    EXPECT_EQ(num_conns_on_client, m_server_disconnect_cnt);
    EXPECT_EQ(num_conns_on_client, m_client_disconnect_cnt);
}

UCS_TEST_P(test_uct_sockaddr, err_handle)
{
    /* client - try to connect to a server that isn't listening */
    connect();
    EXPECT_FALSE(m_state & TEST_STATE_CONNECT_REQUESTED);

    /* with the TCP port space (which is currently tested with rdmacm),
     * a REJECT event will be generated on the client side and since it's a
     * reject from the network, it would be passed to upper layer as
     * UCS_ERR_UNREACHABLE.
     * with tcp_sockcm, an EPOLLERR event will be generated and transformed
     * to an error code. */
    wait_for_bits(&m_state, TEST_STATE_CLIENT_GOT_SERVER_UNAVAILABLE);
    EXPECT_TRUE(ucs_test_all_flags(m_state, TEST_STATE_CLIENT_GOT_SERVER_UNAVAILABLE));
}

UCS_TEST_P(test_uct_sockaddr, conn_to_non_exist_server_port)
{
    /* Listen */
    start_listen(test_uct_sockaddr::conn_request_cb);

    m_connect_addr.set_port(htons(1));

    /* wrap errors since a reject is expected */
    scoped_log_handler slh(detect_reject_error_logger);

    /* client - try to connect to a non-existing port on the server side. */
    connect();

    /* with the TCP port space (which is currently tested with rdmacm),
     * a REJECT event will be generated on the client side and since it's a
     * reject from the network, it would be passed to upper layer as
     * UCS_ERR_UNREACHABLE.
     * with tcp_sockcm, an EPOLLERR event will be generated and transformed
     * to an error code. */
    wait_for_bits(&m_state, TEST_STATE_CLIENT_GOT_SERVER_UNAVAILABLE);
    EXPECT_TRUE(ucs_test_all_flags(m_state, TEST_STATE_CLIENT_GOT_SERVER_UNAVAILABLE));
}

UCS_TEST_P(test_uct_sockaddr, connect_client_to_server_with_delay)
{
    test_delayed_server_response(false, false);
}

UCS_TEST_P(test_uct_sockaddr, destroy_client_before_accept)
{
    test_delayed_server_response(false, true);
}

UCS_TEST_P(test_uct_sockaddr, connect_client_to_server_reject_with_delay)
{
    test_delayed_server_response(true, false);
}

UCS_TEST_P(test_uct_sockaddr, destroy_client_before_reject)
{
    test_delayed_server_response(true, true);
}

UCS_TEST_P(test_uct_sockaddr, ep_disconnect_err_codes)
{
    bool disconnecting = false;

    listen_and_connect();

    {
        ucs::scoped_async_lock lock(m_client->async());
        if (m_state & TEST_STATE_CLIENT_CONNECTED) {
            UCS_TEST_MESSAGE << "EXP: " << ucs_status_string(UCS_OK);
            EXPECT_EQ(UCS_OK, uct_ep_disconnect(m_client->ep(0), 0));
            disconnecting = true;
        } else {
            UCS_TEST_MESSAGE << "EXP: " << ucs_status_string(UCS_ERR_BUSY);
            EXPECT_EQ(UCS_ERR_BUSY, uct_ep_disconnect(m_client->ep(0), 0));
        }
    }

    wait_for_bits(&m_state, TEST_STATE_SERVER_CONNECTED |
                            TEST_STATE_CLIENT_CONNECTED);
    EXPECT_TRUE(ucs_test_all_flags(m_state, (TEST_STATE_SERVER_CONNECTED |
                                             TEST_STATE_CLIENT_CONNECTED)));

    {
        ucs::scoped_async_lock lock(m_client->async());
        if (disconnecting) {
            scoped_log_handler slh(detect_double_disconnect_error_logger);
            if (m_state & TEST_STATE_CLIENT_DISCONNECTED) {
                UCS_TEST_MESSAGE << "EXP: "
                                 << ucs_status_string(UCS_ERR_NOT_CONNECTED);
                EXPECT_EQ(UCS_ERR_NOT_CONNECTED,
                          uct_ep_disconnect(m_client->ep(0), 0));
            } else {
                UCS_TEST_MESSAGE << "EXP: "
                                 << ucs_status_string(UCS_INPROGRESS);
                EXPECT_EQ(UCS_INPROGRESS,
                          uct_ep_disconnect(m_client->ep(0), 0));
            }
        } else {
            UCS_TEST_MESSAGE << "EXP: " << ucs_status_string(UCS_OK);
            ASSERT_UCS_OK(uct_ep_disconnect(m_client->ep(0), 0));
            disconnecting = true;
        }
    }

    ASSERT_TRUE(disconnecting);
    wait_for_bits(&m_state, TEST_STATE_CLIENT_DISCONNECTED);
    EXPECT_TRUE(m_state & TEST_STATE_CLIENT_DISCONNECTED);

    /* wrap errors since the client will call uct_ep_disconnect the second time
     * on the same endpoint. this ep may not be disconnected yet */
    {
        scoped_log_handler slh(detect_double_disconnect_error_logger);
        UCS_TEST_MESSAGE << "EXP: " << ucs_status_string(UCS_ERR_NOT_CONNECTED);
        EXPECT_EQ(UCS_ERR_NOT_CONNECTED, uct_ep_disconnect(m_client->ep(0), 0));
    }
}

UCT_INSTANTIATE_SOCKADDR_TEST_CASE(test_uct_sockaddr)


class test_uct_sockaddr_err_handle_non_exist_ip : public test_uct_sockaddr {
public:
    void init() {
        /* tcp_sockcm requires setting this parameter to shorten the time of waiting
         * for the connect() to fail when connecting to a non-existing ip.
         * A transport for which this value is not configurable, like rdmacm,
         * will have no effect. */
        modify_config("SYN_CNT", "1", SETENV_IF_NOT_EXIST);

        test_uct_sockaddr::init();
    }
};

UCS_TEST_P(test_uct_sockaddr_err_handle_non_exist_ip, conn_to_non_exist_ip)
{
    struct sockaddr_in addr;
    ucs_status_t status;
    size_t size;

    /* Listen */
    start_listen(test_uct_sockaddr::conn_request_cb);

    /* 240.0.0.0/4 - This block, formerly known as the Class E address
       space, is reserved for future use; see [RFC1112], Section 4.
       therefore, this value can be used as a non-existing IP for this test */
    memset(&addr, 0, sizeof(struct sockaddr_in));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = inet_addr("240.0.0.0");
    addr.sin_port        = m_listen_addr.get_port();

    status = ucs_sockaddr_sizeof((struct sockaddr*)&addr, &size);
    ASSERT_UCS_OK(status);

    m_connect_addr.set_sock_addr(*(struct sockaddr*)&addr, size);

    /* wrap errors now since the client will try to connect to a non existing IP */
    {
        scoped_log_handler slh(detect_addr_route_error_logger);
        /* client - try to connect to a non-existing IP */
        connect();
        wait_for_bits(&m_state, TEST_STATE_CLIENT_GOT_SERVER_UNAVAILABLE, 300);
        EXPECT_TRUE(m_state & TEST_STATE_CLIENT_GOT_SERVER_UNAVAILABLE);

        EXPECT_FALSE(m_state & TEST_STATE_CONNECT_REQUESTED);
        EXPECT_FALSE(m_state &
                    (TEST_STATE_SERVER_CONNECTED | TEST_STATE_CLIENT_CONNECTED));
    }
}

UCT_INSTANTIATE_SOCKADDR_TEST_CASE(test_uct_sockaddr_err_handle_non_exist_ip)


class test_uct_sockaddr_stress : public test_uct_sockaddr {
public:
    test_uct_sockaddr_stress() : m_clients_num(0), m_ep_init_disconnect_cnt(0) {
    }

    typedef struct {
        uct_ep_h         ep;
        volatile uint8_t state;
    } ep_state_t;

    void init() {
        test_uct_sockaddr::init();

        m_clients_num = ucs_max(2, 100 / ucs::test_time_multiplier());
        pthread_mutex_init(&m_lock, NULL);
    }

    void cleanup() {
        pthread_mutex_destroy(&m_lock);
        test_uct_sockaddr::cleanup();
    }

    int get_ep_index(uct_ep_h ep) {
        for (int i = 0; i < (2 * m_clients_num); i++) {
            if (m_all_eps[i].ep == ep) {
                return i;
            }
        }

        return -1;
    }

    void common_test_disconnect(uct_ep_h ep) {
        int index;

        index = get_ep_index(ep);
        ASSERT_TRUE(index >= 0);
        EXPECT_LT(index, (2 * m_clients_num));

        pthread_mutex_lock(&m_lock);
        m_all_eps[index].state |= TEST_EP_FLAG_DISCONNECT_CB_INVOKED;

        if (m_all_eps[index].state & TEST_EP_FLAG_DISCONNECT_INITIATOR) {
            m_ep_init_disconnect_cnt--;
            pthread_mutex_unlock(&m_lock);
        } else {
            pthread_mutex_unlock(&m_lock);
            ASSERT_UCS_OK(uct_ep_disconnect(ep, 0));
        }
    }

    void disconnect_cnt_increment(volatile int *cnt) {
        ucs::scoped_mutex_lock lock(m_lock);
        (*cnt)++;
    }

    static void server_disconnect_cb(uct_ep_h ep, void *arg) {
        test_uct_sockaddr_stress *self =
                        reinterpret_cast<test_uct_sockaddr_stress *>(arg);

        self->common_test_disconnect(ep);
        self->disconnect_cnt_increment(&self->m_server_disconnect_cnt);
    }

    static void client_disconnect_cb(uct_ep_h ep, void *arg) {
        client_user_data *sa_user_data =
                reinterpret_cast<client_user_data*>(arg);
        test_uct_sockaddr_stress *self =
                static_cast<test_uct_sockaddr_stress*>(sa_user_data->get_test());

        EXPECT_EQ(sa_user_data->get_ep(), ep);
        self->common_test_disconnect(ep);
        self->disconnect_cnt_increment(&self->m_client_disconnect_cnt);
        self->del_user_data(sa_user_data);
    }

    void server_accept(entity *server, uct_conn_request_h conn_request,
                       uct_cm_ep_server_conn_notify_callback_t notify_cb,
                       uct_ep_disconnect_cb_t disconnect_cb,
                       void *user_data, bool can_fail) {
        ucs::scoped_async_lock listen_lock(m_server->async());
        ucs::scoped_async_lock accept_lock(server->async());
        test_uct_sockaddr::accept(server->cm(), conn_request, notify_cb,
                                  disconnect_cb, user_data, can_fail);
    }

    static void
    conn_request_cb(uct_listener_h listener, void *arg,
                    const uct_cm_listener_conn_request_args_t *conn_req_args) {
        test_uct_sockaddr_stress *self =
                        reinterpret_cast<test_uct_sockaddr_stress *>(arg);

        if (test_uct_sockaddr::common_conn_request(listener, arg,
                                                   conn_req_args)) {
            EXPECT_TRUE(conn_req_args->field_mask &
                        UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_CONN_REQUEST);
            self->server_accept(self->m_server, conn_req_args->conn_request,
                                server_connect_cb, server_disconnect_cb, self,
                                false);
        }

        ucs_memory_cpu_store_fence();
        self->m_server_recv_req_cnt++;
    }

    static void
    server_connect_cb(uct_ep_h ep, void *arg,
                      const uct_cm_ep_server_conn_notify_args_t *notify_args) {
        test_uct_sockaddr_stress *self =
                reinterpret_cast<test_uct_sockaddr_stress*>(arg);

        test_uct_sockaddr::server_connect_cb(ep, arg, notify_args);
        EXPECT_TRUE(self->m_state & TEST_STATE_SERVER_CONNECTED);
    }

protected:
    int                     m_clients_num;
    std::vector<ep_state_t> m_all_eps;
    int                     m_ep_init_disconnect_cnt;
    pthread_mutex_t         m_lock;
};

UCS_TEST_P(test_uct_sockaddr_stress, many_clients_to_one_server)
{
    int i, disconnected_eps_on_each_side, no_disconnect_eps_cnt = 0;
    entity *client_test;
    time_t seed = time(0);
    ucs_time_t deadline;

    /* Listen */
    start_listen(test_uct_sockaddr_stress::conn_request_cb);

    /* Connect */
    /* multiple clients, each on a cm of its own, connecting to the same server */
    for (i = 0; i < m_clients_num; ++i) {
        client_test = uct_test::create_entity();
        m_entities.push_back(client_test);

        client_test->max_conn_priv  = client_test->cm_attr().max_conn_priv;
        connect(*client_test, 0, client_disconnect_cb);
    }

    /* wait for the server to connect to all the clients */
    wait_for_client_server_counters(&m_server_connect_cb_cnt,
                                    &m_client_connect_cb_cnt, m_clients_num);

    EXPECT_EQ(m_clients_num, m_server_recv_req_cnt);
    EXPECT_EQ(m_clients_num, m_client_connect_cb_cnt);
    EXPECT_EQ(m_clients_num, m_server_connect_cb_cnt);
    EXPECT_EQ(m_clients_num, (int)m_server->num_eps());

    /* Disconnect */
    srand(seed);
    UCS_TEST_MESSAGE << "Using random seed: " << seed;

    m_all_eps.resize(2 * m_clients_num);

    /* save all the clients' and server's eps in the m_all_eps array */
    for (i = 0; i < m_clients_num; ++i) {
        /* first 2 entities are m_server and m_client */
        m_all_eps[i].ep                    = m_entities.at(2 + i).ep(0);
        m_all_eps[i].state                 = 0;
        m_all_eps[m_clients_num + i].ep    = m_server->ep(i);
        m_all_eps[m_clients_num + i].state = 0;
    }

    /* Disconnect */
    /* go over the eps array and for each ep - use rand() to decide whether or
     * not it should initiate a disconnect */
    for (i = 0; i < (2 * m_clients_num); ++i) {
        if ((ucs::rand() % 2) == 0) {
            continue;
        }

        /* don't start a disconnect on an ep that was already disconnected */
        pthread_mutex_lock(&m_lock);
        if (!(m_all_eps[i].state & TEST_EP_FLAG_DISCONNECT_CB_INVOKED)) {
            m_all_eps[i].state |= TEST_EP_FLAG_DISCONNECT_INITIATOR;
            pthread_mutex_unlock(&m_lock);
            /* uct_ep_disconnect cannot be called when m_lock is taken
             * in order to prevent abba deadlock since uct will try taking
             * the async lock inside this function */
            ASSERT_UCS_OK(uct_ep_disconnect(m_all_eps[i].ep, 0));
            /* count the number of eps that initiated a disconnect */
            pthread_mutex_lock(&m_lock);
            m_ep_init_disconnect_cnt++;
        }
        pthread_mutex_unlock(&m_lock);
    }

    /* wait for all the disconnect flows that began, to complete.
     * if an ep initiated a disconnect, its disconnect callback should have been
     * called, and so is the disconnect callback of its remote peer ep.
     * every ep that initiated a disconnect is counted. this counter is
     * decremented in its disconnect cb, therefore once all eps that initiated
     * a disconnect are disconnected, this counter should be equal to zero */
    deadline = ucs_get_time() + ucs_time_from_sec(10 * DEFAULT_TIMEOUT_SEC) *
                                ucs::test_time_multiplier();

    while ((m_ep_init_disconnect_cnt != 0) && (ucs_get_time() < deadline)) {
        progress();
    }
    EXPECT_EQ(0, m_ep_init_disconnect_cnt);

    /* count and print the number of eps that were not disconnected */
    for (i = 0; i < (2 * m_clients_num); i++) {
        if (m_all_eps[i].state == 0) {
            no_disconnect_eps_cnt++;
        } else {
            EXPECT_TRUE((m_all_eps[i].state & ~TEST_EP_FLAG_DISCONNECT_INITIATOR) ==
                        TEST_EP_FLAG_DISCONNECT_CB_INVOKED);
        }
    }

    UCS_TEST_MESSAGE << no_disconnect_eps_cnt <<
                     " (out of " << (2 * m_clients_num) << ") "
                     "eps were not disconnected during the test.";

    disconnected_eps_on_each_side = ((2 * m_clients_num) - no_disconnect_eps_cnt) / 2;
    wait_for_client_server_counters(&m_server_disconnect_cnt,
                                    &m_client_disconnect_cnt,
                                    disconnected_eps_on_each_side);

    EXPECT_EQ(disconnected_eps_on_each_side, m_server_disconnect_cnt);
    EXPECT_EQ(disconnected_eps_on_each_side, m_client_disconnect_cnt);

    /* destroy all the eps here (and not in the test's destruction flow) so that
     * no disconnect callbacks are invoked after the test ends */
    m_entities.clear();

    /* destroyed EPs don't invoke CBs, need to clean up user data manually */
    release_user_data();
}

UCT_INSTANTIATE_SOCKADDR_TEST_CASE(test_uct_sockaddr_stress)


class test_uct_sockaddr_multiple_cms : public test_uct_sockaddr {
public:
    void init() {
        ucs_status_t status;

        test_uct_sockaddr::init();

        status = ucs_async_context_create(UCS_ASYNC_MODE_THREAD_SPINLOCK,
                                          &m_test_async);
        ASSERT_UCS_OK(status);

        status = uct_cm_config_read(GetParam()->component, NULL, NULL, &m_test_config);
        ASSERT_UCS_OK(status);

        UCS_TEST_CREATE_HANDLE(uct_worker_h, m_test_worker, uct_worker_destroy,
                               uct_worker_create, m_test_async,
                               UCS_THREAD_MODE_SINGLE)

        UCS_TEST_CREATE_HANDLE(uct_cm_h, m_test_cm, uct_cm_close,
                               uct_cm_open, GetParam()->component,
                               m_test_worker, m_test_config);
    }

    void cleanup() {
        m_test_cm.reset();
        uct_config_release(m_test_config);
        m_test_worker.reset();
        ucs_async_context_destroy(m_test_async);
        test_uct_sockaddr::cleanup();
    }

    void server_accept(entity *server, uct_conn_request_h conn_request,
                       uct_cm_ep_server_conn_notify_callback_t notify_cb,
                       uct_ep_disconnect_cb_t disconnect_cb,
                       void *user_data, bool can_fail)
    {
        ucs::scoped_async_lock listen_lock(m_server->async());
        ucs::scoped_async_lock accept_lock(*m_test_async);
        accept(m_test_cm, conn_request, notify_cb, disconnect_cb, user_data,
               can_fail);
    }

protected:
    ucs::handle<uct_worker_h> m_test_worker;
    ucs::handle<uct_cm_h>     m_test_cm;
    ucs_async_context_t       *m_test_async;
    uct_cm_config_t           *m_test_config;
};

UCS_TEST_P(test_uct_sockaddr_multiple_cms, server_switch_cm)
{
    listen_and_connect();

    wait_for_bits(&m_state, TEST_STATE_SERVER_CONNECTED |
                            TEST_STATE_CLIENT_CONNECTED);
    EXPECT_TRUE(ucs_test_all_flags(m_state, (TEST_STATE_SERVER_CONNECTED |
                                             TEST_STATE_CLIENT_CONNECTED)));

    cm_disconnect(m_client);

    /* destroy the server's ep here so that it would be destroyed before the cm
     * it is using */
    m_server->destroy_ep(0);
}

UCT_INSTANTIATE_SOCKADDR_TEST_CASE(test_uct_sockaddr_multiple_cms)

/**
 * This class tests "legacy" API @ref uct_ep_params::sockaddr_pack_cb which can
 * be replaced with more flexible API:
 *  - @ref uct_ep_params::cm_resolve_cb + @ref uct_ep_connect on client side
 *  - @ref uct_ep_params::private_data + @ref uct_ep_params::private_data_length
 *    on server side
 * how this is implemented in @ref test_uct_sockaddr.
 */
class test_uct_sockaddr_legacy : public test_uct_sockaddr
{
public:
    static ssize_t client_priv_data_cb(void *arg,
                                       const uct_cm_ep_priv_data_pack_args_t
                                       *pack_args, void *priv_data)
    {
        client_user_data *sa_user_data =
                reinterpret_cast<client_user_data*>(arg);
        test_uct_sockaddr_legacy *self =
                static_cast<test_uct_sockaddr_legacy*>(sa_user_data->get_test());

        return self->common_priv_data_cb(self->m_client->max_conn_priv,
                                         priv_data);
    }

    static ssize_t server_priv_data_cb(void *arg,
                                       const uct_cm_ep_priv_data_pack_args_t
                                       *pack_args, void *priv_data)
    {
        test_uct_sockaddr_legacy *self =
                reinterpret_cast<test_uct_sockaddr_legacy*>(arg);

        return self->common_priv_data_cb(self->m_server->max_conn_priv,
                                         priv_data);
    }

    virtual void accept(uct_cm_h cm, uct_conn_request_h conn_request,
                        uct_cm_ep_server_conn_notify_callback_t notify_cb,
                        uct_ep_disconnect_cb_t disconnect_cb,
                        void *user_data, bool can_fail)
    {
        uct_ep_params_t ep_params;
        ucs_status_t status;
        uct_ep_h ep;

        ASSERT_FALSE(can_fail);
        ASSERT_TRUE(m_server->listener());
        m_server->reserve_ep(m_server->num_eps());

        ep_params.field_mask = UCT_EP_PARAM_FIELD_CM                        |
                               UCT_EP_PARAM_FIELD_CONN_REQUEST              |
                               UCT_EP_PARAM_FIELD_USER_DATA                 |
                               UCT_EP_PARAM_FIELD_SOCKADDR_NOTIFY_CB_SERVER |
                               UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB    |
                               UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS         |
                               UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB;

        ep_params.cm                 = cm;
        ep_params.conn_request       = conn_request;
        ep_params.sockaddr_cb_flags  = UCT_CB_FLAG_ASYNC;
        ep_params.sockaddr_pack_cb   = server_priv_data_cb;
        ep_params.sockaddr_cb_server = notify_cb;
        ep_params.disconnect_cb      = disconnect_cb;
        ep_params.user_data          = user_data;

        status = uct_ep_create(&ep_params, &ep);
        check_connection_status(status, can_fail);
        m_server->eps().back().reset(ep, uct_ep_destroy);
    }
};

UCS_TEST_P(test_uct_sockaddr_legacy, cm_open_listen_close)
{
    start_listen(conn_request_cb);

    ucs_sock_addr_t ucs_remote_addr = m_connect_addr.to_ucs_sock_addr();

    m_client->reserve_ep(0);
    ASSERT_EQ(NULL, m_client->ep(0));

    client_user_data *user_data = new client_user_data(*this, *m_client, 0);

    /* Connect to the server */
    uct_ep_h ep;
    uct_ep_params_t params;
    params.field_mask         = UCT_EP_PARAM_FIELD_CM                         |
                                UCT_EP_PARAM_FIELD_SOCKADDR_CONNECT_CB_CLIENT |
                                UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB     |
                                UCT_EP_PARAM_FIELD_USER_DATA                  |
                                UCT_EP_PARAM_FIELD_SOCKADDR                   |
                                UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS          |
                                UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB;
    params.cm                 = m_client->cm();
    params.sockaddr_cb_client = client_connect_cb;
    params.disconnect_cb      = client_disconnect_cb;
    params.user_data          = user_data;
    params.sockaddr           = &ucs_remote_addr;
    params.sockaddr_cb_flags  = UCT_CB_FLAG_ASYNC;
    params.sockaddr_pack_cb   = client_priv_data_cb;

    ucs_status_t status       = uct_ep_create(&params, &ep);
    ASSERT_UCS_OK(status);
    m_client->eps().at(0).reset(ep, uct_ep_destroy);
    add_user_data(user_data);
    wait_for_bits(&m_state, TEST_STATE_SERVER_CONNECTED |
                            TEST_STATE_CLIENT_CONNECTED);
    EXPECT_TRUE(ucs_test_all_flags(m_state, (TEST_STATE_SERVER_CONNECTED |
                                             TEST_STATE_CLIENT_CONNECTED)));
    cm_disconnect(m_client);
}

UCT_INSTANTIATE_SOCKADDR_TEST_CASE(test_uct_sockaddr_legacy)
