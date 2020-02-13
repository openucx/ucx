/**
* Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
#include <uct/uct_test.h>

extern "C" {
#include <uct/api/uct.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/string.h>
#include <ucs/arch/atomic.h>
}

#include <queue>

class test_uct_sockaddr : public uct_test {
public:
    struct completion : public uct_completion_t {
        volatile bool m_flag;

        completion() : m_flag(false), m_status(UCS_INPROGRESS) {
            count = 1;
            func  = completion_cb;
        }

        ucs_status_t status() const {
            return m_status;
        }
    private:
        static void completion_cb(uct_completion_t *self, ucs_status_t status)
        {
            completion *c = static_cast<completion*>(self);
            c->m_status   = status;
            c->m_flag     = true;
        }

        ucs_status_t m_status;
    };

    test_uct_sockaddr() : server(NULL), client(NULL), err_count(0),
                          server_recv_req(0), delay_conn_reply(false) {
    }

    void check_md_usability() {
        uct_md_attr_t md_attr;
        uct_md_config_t *md_config;
        ucs_status_t status;
        uct_md_h md;

        status = uct_md_config_read(GetParam()->component, NULL, NULL, &md_config);
        EXPECT_TRUE(status == UCS_OK);

        status = uct_md_open(GetParam()->component, GetParam()->md_name.c_str(),
                             md_config, &md);
        EXPECT_TRUE(status == UCS_OK);
        uct_config_release(md_config);

        status = uct_md_query(md, &md_attr);
        ASSERT_UCS_OK(status);

        uct_md_close(md);

        if (!(md_attr.cap.flags & UCT_MD_FLAG_SOCKADDR)) {
            UCS_TEST_SKIP_R(GetParam()->md_name.c_str() +
                            std::string(" does not support client-server "
                                        "connection establishment via sockaddr "
                                        "without a cm"));
        }
    }

    void init() {
        check_md_usability();

        uct_iface_params_t server_params, client_params;
        uint16_t port;

        uct_test::init();

        /* This address is accessible, as it was tested at the resource creation */
        m_listen_addr  = GetParam()->listen_sock_addr;
        m_connect_addr = GetParam()->connect_sock_addr;

        port = ucs::get_port();
        m_listen_addr.set_port(port);
        m_connect_addr.set_port(port);

        /* open iface for the server side */
        server_params.field_mask                     = UCT_IFACE_PARAM_FIELD_OPEN_MODE         |
                                                       UCT_IFACE_PARAM_FIELD_ERR_HANDLER       |
                                                       UCT_IFACE_PARAM_FIELD_ERR_HANDLER_ARG   |
                                                       UCT_IFACE_PARAM_FIELD_ERR_HANDLER_FLAGS |
                                                       UCT_IFACE_PARAM_FIELD_SOCKADDR;
        server_params.open_mode                      = UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER;
        server_params.err_handler                    = err_handler;
        server_params.err_handler_arg                = reinterpret_cast<void*>(this);
        server_params.err_handler_flags              = 0;
        server_params.mode.sockaddr.listen_sockaddr  = m_listen_addr.to_ucs_sock_addr();
        server_params.mode.sockaddr.cb_flags         = UCT_CB_FLAG_ASYNC;
        server_params.mode.sockaddr.conn_request_cb  = conn_request_cb;
        server_params.mode.sockaddr.conn_request_arg = reinterpret_cast<void*>(this);

        /* if origin port is busy, create_entity will retry with another one */
        server = uct_test::create_entity(server_params);
        m_entities.push_back(server);

        check_skip_test();

        port = ucs::sock_addr_storage(server->iface_params().mode.sockaddr
                                                            .listen_sockaddr)
                                      .get_port();
        m_listen_addr.set_port(port);
        m_connect_addr.set_port(port);

        /* open iface for the client side */
        client_params.field_mask                     = UCT_IFACE_PARAM_FIELD_OPEN_MODE       |
                                                       UCT_IFACE_PARAM_FIELD_ERR_HANDLER     |
                                                       UCT_IFACE_PARAM_FIELD_ERR_HANDLER_ARG |
                                                       UCT_IFACE_PARAM_FIELD_ERR_HANDLER_FLAGS;
        client_params.open_mode                      = UCT_IFACE_OPEN_MODE_SOCKADDR_CLIENT;
        client_params.err_handler                    = err_handler;
        client_params.err_handler_arg                = reinterpret_cast<void*>(this);
        client_params.err_handler_flags              = 0;

        client = uct_test::create_entity(client_params);
        m_entities.push_back(client);

        /* initiate the client's private data callback argument */
        client->max_conn_priv = server->iface_attr().max_conn_priv;

        UCS_TEST_MESSAGE << "Testing " << m_listen_addr
                         << " Interface: " << GetParam()->dev_name;
    }

    static ssize_t client_iface_priv_data_cb(void *arg, const char *dev_name,
                                             void *priv_data)
    {
        size_t *max_conn_priv = (size_t*)arg;
        size_t priv_data_len;

        priv_data_len = uct_test::entity::priv_data_do_pack(priv_data);
        EXPECT_LE(priv_data_len, (*max_conn_priv));

        return priv_data_len;
    }

    static void conn_request_cb(uct_iface_h iface, void *arg,
                                uct_conn_request_h conn_request,
                                const void *conn_priv_data, size_t length)
    {
        test_uct_sockaddr *self = reinterpret_cast<test_uct_sockaddr*>(arg);

        EXPECT_EQ(std::string(reinterpret_cast<const char *>
                              (uct_test::entity::client_priv_data.c_str())),
                  std::string(reinterpret_cast<const char *>(conn_priv_data)));

        EXPECT_EQ(1 + uct_test::entity::client_priv_data.length(), length);
        if (self->delay_conn_reply) {
            self->delayed_conn_reqs.push(conn_request);
        } else {
            uct_iface_accept(iface, conn_request);
        }
        ucs_memory_cpu_store_fence();
        self->server_recv_req++;
    }

    static ucs_status_t err_handler(void *arg, uct_ep_h ep, ucs_status_t status)
    {
        test_uct_sockaddr *self = reinterpret_cast<test_uct_sockaddr*>(arg);
        ucs_atomic_add32(&self->err_count, 1);
        return UCS_OK;
    }

protected:
    entity *server, *client;
    ucs::sock_addr_storage m_listen_addr, m_connect_addr;
    volatile uint32_t err_count;
    volatile int server_recv_req;
    std::queue<uct_conn_request_h> delayed_conn_reqs;
    bool delay_conn_reply;
};

UCS_TEST_P(test_uct_sockaddr, connect_client_to_server)
{
    client->connect(0, *server, 0, m_connect_addr, client_iface_priv_data_cb,
                    NULL, NULL, &client->max_conn_priv);


    /* wait for the server to connect */
    while (server_recv_req == 0) {
        progress();
    }
    ASSERT_TRUE(server_recv_req == 1);
    /* since the transport may support a graceful exit in case of an error,
     * make sure that the error handling flow wasn't invoked (there were no
     * errors) */
    EXPECT_EQ(0ul, err_count);
    /* the test may end before the client's ep got connected.
     * it should also pass in this case as well - the client's
     * ep shouldn't be accessed (for connection reply from the server) after the
     * test ends and the client's ep was destroyed */
}

UCS_TEST_P(test_uct_sockaddr, connect_client_to_server_with_delay)
{
    delay_conn_reply = true;
    client->connect(0, *server, 0, m_connect_addr, client_iface_priv_data_cb,
                    NULL, NULL, &client->max_conn_priv);

    /* wait for the server to connect */
    while (server_recv_req == 0) {
        progress();
    }
    ASSERT_EQ(1,   server_recv_req);
    ucs_memory_cpu_load_fence();
    ASSERT_EQ(1ul, delayed_conn_reqs.size());
    EXPECT_EQ(0ul, err_count);
    while (!delayed_conn_reqs.empty()) {
        uct_iface_accept(server->iface(), delayed_conn_reqs.front());
        delayed_conn_reqs.pop();
    }

    completion comp;
    ucs_status_t status = uct_ep_flush(client->ep(0), 0, &comp);
    if (status == UCS_INPROGRESS) {
        wait_for_flag(&comp.m_flag);
        EXPECT_EQ(UCS_OK, comp.status());
    } else {
        EXPECT_EQ(UCS_OK, status);
    }
    EXPECT_EQ(0ul, err_count);
}

UCS_TEST_P(test_uct_sockaddr, connect_client_to_server_reject_with_delay)
{
    delay_conn_reply = true;
    client->connect(0, *server, 0, m_connect_addr, client_iface_priv_data_cb,
                    NULL, NULL, &client->max_conn_priv);

    /* wait for the server to connect */
    while (server_recv_req == 0) {
        progress();
    }
    ASSERT_EQ(1, server_recv_req);
    ucs_memory_cpu_load_fence();
    ASSERT_EQ(1ul, delayed_conn_reqs.size());
    EXPECT_EQ(0ul, err_count);
    while (!delayed_conn_reqs.empty()) {
        uct_iface_reject(server->iface(), delayed_conn_reqs.front());
        delayed_conn_reqs.pop();
    }
    while (err_count == 0) {
        progress();
    }
    EXPECT_EQ(1ul, err_count);
}

UCS_TEST_P(test_uct_sockaddr, many_clients_to_one_server)
{
    int num_clients = ucs_max(2, 100 / ucs::test_time_multiplier());
    uct_iface_params_t client_params;
    entity *client_test;

    /* multiple clients, each on an iface of its own, connecting to the same server */
    for (int i = 0; i < num_clients; ++i) {
        /* open iface for the client side */
        client_params.field_mask        = UCT_IFACE_PARAM_FIELD_OPEN_MODE       |
                                          UCT_IFACE_PARAM_FIELD_ERR_HANDLER     |
                                          UCT_IFACE_PARAM_FIELD_ERR_HANDLER_ARG |
                                          UCT_IFACE_PARAM_FIELD_ERR_HANDLER_FLAGS;
        client_params.open_mode         = UCT_IFACE_OPEN_MODE_SOCKADDR_CLIENT;
        client_params.err_handler       = err_handler;
        client_params.err_handler_arg   = reinterpret_cast<void*>(this);
        client_params.err_handler_flags = 0;

        client_test = uct_test::create_entity(client_params);
        m_entities.push_back(client_test);

        client_test->max_conn_priv = server->iface_attr().max_conn_priv;
        client_test->connect(i, *server, 0, m_connect_addr,
                             client_iface_priv_data_cb, NULL, NULL,
                             &client_test->max_conn_priv);
    }

    while (server_recv_req < num_clients){
        progress();
    }
    ASSERT_TRUE(server_recv_req == num_clients);
    EXPECT_EQ(0ul, err_count);
}

UCS_TEST_P(test_uct_sockaddr, many_conns_on_client)
{
    int num_conns_on_client = ucs_max(2, 100 / ucs::test_time_multiplier());

    /* multiple clients, on the same iface, connecting to the same server */
    for (int i = 0; i < num_conns_on_client; ++i) {
        client->connect(i, *server, 0, m_connect_addr, client_iface_priv_data_cb,
                        NULL, NULL, &client->max_conn_priv);
    }

    while (server_recv_req < num_conns_on_client) {
        progress();
    }
    ASSERT_TRUE(server_recv_req == num_conns_on_client);
    EXPECT_EQ(0ul, err_count);
}

UCS_TEST_SKIP_COND_P(test_uct_sockaddr, err_handle,
                     !check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE))
{
    client->connect(0, *server, 0, m_connect_addr, client_iface_priv_data_cb,
                    NULL, NULL, &client->max_conn_priv);

    scoped_log_handler slh(wrap_errors_logger);
    /* kill the server */
    m_entities.remove(server);

    /* If the server didn't receive a connection request from the client yet,
     * test error handling */
    if (server_recv_req == 0) {
        wait_for_flag(&err_count);
        /* Double check for server_recv_req if it's not delivered from NIC to
         * host memory under hight load */
        EXPECT_TRUE((err_count == 1) || (server_recv_req == 1));
    }
}

UCS_TEST_SKIP_COND_P(test_uct_sockaddr, conn_to_non_exist_server,
                     !check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE))
{
    m_connect_addr.set_port(htons(1));
    err_count = 0;

    /* wrap errors now since the client will try to connect to a non existing port */
    {
        scoped_log_handler slh(wrap_errors_logger);
        /* client - try to connect to a non-existing port on the server side */
        client->connect(0, *server, 0, m_connect_addr, client_iface_priv_data_cb,
                        NULL, NULL, &client->max_conn_priv);
        completion comp;
        ucs_status_t status = uct_ep_flush(client->ep(0), 0, &comp);
        if (status == UCS_INPROGRESS) {
            wait_for_flag(&comp.m_flag);
            EXPECT_EQ(UCS_ERR_UNREACHABLE, comp.status());
        } else {
            EXPECT_EQ(UCS_ERR_UNREACHABLE, status);
        }
        /* destroy the client's ep. this ep shouldn't be accessed anymore */
        client->destroy_ep(0);
    }
}

UCT_INSTANTIATE_SOCKADDR_TEST_CASE(test_uct_sockaddr)

class test_uct_cm_sockaddr : public uct_test {
    friend class uct_test::entity;
protected:
    enum {
        TEST_CM_STATE_CONNECT_REQUESTED   = UCS_BIT(0),
        TEST_CM_STATE_CLIENT_CONNECTED    = UCS_BIT(1),
        TEST_CM_STATE_SERVER_CONNECTED    = UCS_BIT(2),
        TEST_CM_STATE_CLIENT_DISCONNECTED = UCS_BIT(3),
        TEST_CM_STATE_SERVER_DISCONNECTED = UCS_BIT(4),
        TEST_CM_STATE_SERVER_REJECTED     = UCS_BIT(5),
        TEST_CM_STATE_CLIENT_GOT_REJECT   = UCS_BIT(6),
        TEST_CM_STATE_CLIENT_GOT_ERROR    = UCS_BIT(7)
    };

public:
    test_uct_cm_sockaddr() : m_cm_state(0), m_server(NULL), m_client(NULL),
                             m_server_recv_req_cnt(0), m_client_connect_cb_cnt(0),
                             m_server_connect_cb_cnt(0),
                             m_server_disconnect_cnt(0), m_client_disconnect_cnt(0),
                             m_reject_conn_request(false),
                             m_server_start_disconnect(false),
                             m_delay_conn_reply(false) {
    }

    void init() {
        uct_test::init();

        /* This address is accessible, as it was tested at the resource creation */
        m_listen_addr  = GetParam()->listen_sock_addr;
        m_connect_addr = GetParam()->connect_sock_addr;

        uint16_t port = ucs::get_port();
        m_listen_addr.set_port(port);
        m_connect_addr.set_port(port);

        m_server = uct_test::create_entity();
        m_entities.push_back(m_server);
        m_client = uct_test::create_entity();
        m_entities.push_back(m_client);

        /* initiate the client's private data callback argument */
        m_client->max_conn_priv = m_client->cm_attr().max_conn_priv;

        UCS_TEST_MESSAGE << "Testing " << m_listen_addr
                         << " Interface: " << GetParam()->dev_name;
    }

protected:

    void skip_tcp_sockcm() {
        uct_component_attr_t cmpt_attr = {0};
        ucs_status_t status;

        cmpt_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_NAME;
        /* coverity[var_deref_model] */
        status = uct_component_query(GetParam()->component, &cmpt_attr);
        ASSERT_UCS_OK(status);

        if (!strcmp(cmpt_attr.name, "tcp")) {
            UCS_TEST_SKIP_R("tcp cm is not fully implemented");
        }
    }

    void cm_start_listen() {
        uct_listener_params_t params;

        params.field_mask      = UCT_LISTENER_PARAM_FIELD_CONN_REQUEST_CB |
                                 UCT_LISTENER_PARAM_FIELD_USER_DATA;
        params.conn_request_cb = cm_conn_request_cb;
        params.user_data       = static_cast<test_uct_cm_sockaddr *>(this);
        /* if origin port set in init() is busy, listen() will retry with another one */
        m_server->listen(m_listen_addr, params);

        /* the listen function may have changed the initial port on the listener's
         * address. update this port for the address to connect to */
        m_connect_addr.set_port(m_listen_addr.get_port());
    }

    static ssize_t client_cm_priv_data_cb(void *arg, const char *dev_name,
                                          void *priv_data)
    {
        test_uct_cm_sockaddr *self = reinterpret_cast<test_uct_cm_sockaddr *>(arg);
        size_t priv_data_len;

        priv_data_len = uct_test::entity::priv_data_do_pack(priv_data);
        EXPECT_LE(priv_data_len, self->m_client->max_conn_priv);
        return priv_data_len;
    }

    void cm_listen_and_connect() {
        skip_tcp_sockcm();
        cm_start_listen();
        m_client->connect(0, *m_server, 0, m_connect_addr, client_cm_priv_data_cb,
                          client_connect_cb, client_disconnect_cb, this);

        wait_for_bits(&m_cm_state, TEST_CM_STATE_CONNECT_REQUESTED);
        EXPECT_TRUE(m_cm_state & TEST_CM_STATE_CONNECT_REQUESTED);
    }

    virtual void server_accept(entity *server, uct_conn_request_h conn_request,
                               uct_ep_server_connect_cb_t connect_cb,
                               uct_ep_disconnect_cb_t disconnect_cb,
                               void *user_data)
    {
        server->accept(server->cm(), conn_request, connect_cb, disconnect_cb,
                       user_data);
    }

    static void
    cm_conn_request_cb(uct_listener_h listener, void *arg,
                       const char *local_dev_name,
                       uct_conn_request_h conn_request,
                       const uct_cm_remote_data_t *remote_data) {
        test_uct_cm_sockaddr *self = reinterpret_cast<test_uct_cm_sockaddr *>(arg);
        ucs_status_t status;

        EXPECT_EQ(entity::client_priv_data.length() + 1, remote_data->conn_priv_data_length);
        EXPECT_EQ(entity::client_priv_data,
                  std::string(static_cast<const char *>(remote_data->conn_priv_data)));

        self->m_cm_state |= TEST_CM_STATE_CONNECT_REQUESTED;

        if (self->m_delay_conn_reply) {
            self->m_delayed_conn_reqs.push(conn_request);
        } else if (self->m_reject_conn_request) {
            status = uct_listener_reject(listener, conn_request);
            ASSERT_UCS_OK(status);
            self->m_cm_state |= TEST_CM_STATE_SERVER_REJECTED;
        } else {
            self->server_accept(self->m_server, conn_request,
                                server_connect_cb, server_disconnect_cb, self);
        }

        ucs_memory_cpu_store_fence();
        self->m_server_recv_req_cnt++;
    }

    static void
    server_connect_cb(uct_ep_h ep, void *arg, ucs_status_t status) {
        test_uct_cm_sockaddr *self = reinterpret_cast<test_uct_cm_sockaddr *>(arg);

        EXPECT_EQ(UCS_OK, status);
        self->m_cm_state |= TEST_CM_STATE_SERVER_CONNECTED;
        self->m_server_connect_cb_cnt++;
    }

    static void
    server_disconnect_cb(uct_ep_h ep, void *arg) {
        test_uct_cm_sockaddr *self = reinterpret_cast<test_uct_cm_sockaddr *>(arg);

        if (!(self->m_server_start_disconnect)) {
            self->m_server->disconnect(ep);
        }
        self->m_cm_state |= TEST_CM_STATE_SERVER_DISCONNECTED;
        self->m_server_disconnect_cnt++;
    }

    static void
    client_connect_cb(uct_ep_h ep, void *arg,
                      const uct_cm_remote_data_t *remote_data,
                      ucs_status_t status) {
        test_uct_cm_sockaddr *self = reinterpret_cast<test_uct_cm_sockaddr *>(arg);

        if (status == UCS_ERR_REJECTED) {
            self->m_cm_state |= TEST_CM_STATE_CLIENT_GOT_REJECT;
        } else if (status != UCS_OK) {
            self->m_cm_state |= TEST_CM_STATE_CLIENT_GOT_ERROR;
        } else {
            EXPECT_TRUE(ucs_test_all_flags(remote_data->field_mask,
                                           (UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA_LENGTH |
                                            UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA)));
            EXPECT_EQ(entity::server_priv_data.length() + 1, remote_data->conn_priv_data_length);
            EXPECT_EQ(entity::server_priv_data,
                      std::string(static_cast<const char *>(remote_data->conn_priv_data)));
            self->m_cm_state |= TEST_CM_STATE_CLIENT_CONNECTED;
            self->m_client_connect_cb_cnt++;
        }
    }

    static void
    client_disconnect_cb(uct_ep_h ep, void *arg) {
        test_uct_cm_sockaddr *self;

        self = reinterpret_cast<test_uct_cm_sockaddr *>(arg);

        if (self->m_server_start_disconnect) {
            /* if the server was the one who initiated the disconnect flow,
             * the client should also disconnect its ep from the server in
             * its disconnect cb */
            self->m_client->disconnect(ep);
        }

        self->m_cm_state |= TEST_CM_STATE_CLIENT_DISCONNECTED;
        self->m_client_disconnect_cnt++;
    }

    void cm_disconnect(entity *ent) {
        size_t i;

        /* Disconnect all the existing endpoints */
        for (i = 0; i < ent->num_eps(); ++i) {
            ent->disconnect(ent->ep(i));
        }

        wait_for_bits(&m_cm_state, TEST_CM_STATE_CLIENT_DISCONNECTED |
                                   TEST_CM_STATE_SERVER_DISCONNECTED);
        EXPECT_TRUE(ucs_test_all_flags(m_cm_state, (TEST_CM_STATE_SERVER_DISCONNECTED |
                                                    TEST_CM_STATE_CLIENT_DISCONNECTED)));
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

    void test_delayed_server_response(bool reject)
    {
        ucs_status_t status;
        ucs_time_t deadline;

        m_delay_conn_reply = true;

        cm_listen_and_connect();

        EXPECT_FALSE(m_cm_state &
                     (TEST_CM_STATE_SERVER_CONNECTED  | TEST_CM_STATE_CLIENT_CONNECTED |
                      TEST_CM_STATE_CLIENT_GOT_REJECT | TEST_CM_STATE_CLIENT_GOT_ERROR));

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
            ASSERT_UCS_OK(status);

            wait_for_bits(&m_cm_state, TEST_CM_STATE_CLIENT_GOT_REJECT);
            EXPECT_TRUE(m_cm_state & TEST_CM_STATE_CLIENT_GOT_REJECT);
        } else {
            server_accept(m_server, m_delayed_conn_reqs.front(),
                          server_connect_cb, server_disconnect_cb, this);

            wait_for_bits(&m_cm_state, TEST_CM_STATE_SERVER_CONNECTED |
                                       TEST_CM_STATE_CLIENT_CONNECTED);
            EXPECT_TRUE(ucs_test_all_flags(m_cm_state, TEST_CM_STATE_SERVER_CONNECTED |
                                                       TEST_CM_STATE_CLIENT_CONNECTED));
        }

        m_delayed_conn_reqs.pop();
    }

    static ucs_log_func_rc_t
    detect_addr_route_error_logger(const char *file, unsigned line, const char *function,
                                   ucs_log_level_t level, const char *message, va_list ap)
    {
        if (level == UCS_LOG_LEVEL_ERROR) {
            std::string err_str = format_message(message, ap);
            if ((strstr(err_str.c_str(), "client: got error event RDMA_CM_EVENT_ADDR_ERROR"))  ||
                (strstr(err_str.c_str(), "client: got error event RDMA_CM_EVENT_ROUTE_ERROR")) ||
                (strstr(err_str.c_str(), "rdma_resolve_route(to addr=240.0.0.0"))) {
                UCS_TEST_MESSAGE << err_str;
                return UCS_LOG_FUNC_RC_STOP;
            }
        }
        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    static ucs_log_func_rc_t
    detect_reject_error_logger(const char *file, unsigned line, const char *function,
                               ucs_log_level_t level, const char *message, va_list ap)
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

protected:
    ucs::sock_addr_storage m_listen_addr, m_connect_addr;
    uint64_t               m_cm_state;
    entity                 *m_server;
    entity                 *m_client;
    volatile int           m_server_recv_req_cnt, m_client_connect_cb_cnt,
                           m_server_connect_cb_cnt;
    volatile int           m_server_disconnect_cnt, m_client_disconnect_cnt;
    bool                   m_reject_conn_request;
    bool                   m_server_start_disconnect;
    bool                   m_delay_conn_reply;
    std::queue<uct_conn_request_h> m_delayed_conn_reqs;
};

UCS_TEST_P(test_uct_cm_sockaddr, cm_query)
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

UCS_TEST_P(test_uct_cm_sockaddr, listener_query)
{
    uct_listener_attr_t attr;
    ucs_status_t status;
    uint16_t port;
    char m_listener_ip_port_str[UCS_SOCKADDR_STRING_LEN];
    char attr_addr_ip_port_str[UCS_SOCKADDR_STRING_LEN];

    cm_start_listen();

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

UCS_TEST_P(test_uct_cm_sockaddr, cm_open_listen_close)
{
    cm_listen_and_connect();

    wait_for_bits(&m_cm_state, TEST_CM_STATE_SERVER_CONNECTED |
                               TEST_CM_STATE_CLIENT_CONNECTED);
    EXPECT_TRUE(ucs_test_all_flags(m_cm_state, (TEST_CM_STATE_SERVER_CONNECTED |
                                                TEST_CM_STATE_CLIENT_CONNECTED)));

    cm_disconnect(m_client);
}

UCS_TEST_P(test_uct_cm_sockaddr, cm_open_listen_kill_server)
{
    cm_listen_and_connect();

    wait_for_bits(&m_cm_state, TEST_CM_STATE_SERVER_CONNECTED |
                               TEST_CM_STATE_CLIENT_CONNECTED);
    EXPECT_TRUE(ucs_test_all_flags(m_cm_state, (TEST_CM_STATE_SERVER_CONNECTED |
                                                TEST_CM_STATE_CLIENT_CONNECTED)));

    EXPECT_EQ(1ul, m_entities.remove(m_server));
    m_server = NULL;

    wait_for_bits(&m_cm_state, TEST_CM_STATE_CLIENT_DISCONNECTED);
    EXPECT_TRUE(m_cm_state & TEST_CM_STATE_CLIENT_DISCONNECTED);
}

UCS_TEST_P(test_uct_cm_sockaddr, cm_server_reject)
{
    m_reject_conn_request = true;

    /* wrap errors since a reject is expected */
    scoped_log_handler slh(detect_reject_error_logger);
    cm_listen_and_connect();

    wait_for_bits(&m_cm_state, TEST_CM_STATE_SERVER_REJECTED |
                               TEST_CM_STATE_CLIENT_GOT_REJECT);
    EXPECT_TRUE(ucs_test_all_flags(m_cm_state, (TEST_CM_STATE_SERVER_REJECTED |
                                                TEST_CM_STATE_CLIENT_GOT_REJECT)));

    EXPECT_FALSE((m_cm_state &
                (TEST_CM_STATE_SERVER_CONNECTED | TEST_CM_STATE_CLIENT_CONNECTED)));
}

UCS_TEST_P(test_uct_cm_sockaddr, many_clients_to_one_server)
{
    int num_clients = ucs_max(2, 100 / ucs::test_time_multiplier());;
    entity *client_test;

    skip_tcp_sockcm();
    /* Listen */
    cm_start_listen();

    /* Connect */
    /* multiple clients, each on a cm of its own, connecting to the same server */
    for (int i = 0; i < num_clients; ++i) {
        client_test = uct_test::create_entity();
        m_entities.push_back(client_test);
        client_test->max_conn_priv = client_test->cm_attr().max_conn_priv;
        client_test->connect(0, *m_server, 0, m_connect_addr, client_cm_priv_data_cb,
                             client_connect_cb, client_disconnect_cb, this);
    }

    /* wait for the server to connect to all the clients */
    wait_for_client_server_counters(&m_server_connect_cb_cnt,
                                    &m_client_connect_cb_cnt, num_clients);

    EXPECT_EQ(num_clients, m_server_recv_req_cnt);
    EXPECT_EQ(num_clients, m_client_connect_cb_cnt);
    EXPECT_EQ(num_clients, m_server_connect_cb_cnt);
    EXPECT_EQ(num_clients, (int)m_server->num_eps());

    /* Disconnect */
    for (int i = 0; i < num_clients; ++i) {
        /* first 2 entities are m_server and m_client */
        client_test = &m_entities.at(2 + i);
        ASSERT_TRUE(client_test != m_client);

        cm_disconnect(client_test);
    }

    /* don't remove the ep, i.e. don't call uct_ep_destroy on the client's ep,
     * before the client finished disconnecting so that a disconnect event won't
     * arrive on a destroyed endpoint on the client side */

    wait_for_client_server_counters(&m_server_disconnect_cnt,
                                    &m_client_disconnect_cnt, num_clients);

    EXPECT_EQ(num_clients, m_server_disconnect_cnt);
    EXPECT_EQ(num_clients, m_client_disconnect_cnt);

    for (int i = 0; i < num_clients; ++i) {
        client_test = m_entities.back();
        m_entities.remove(client_test);
    }
}

UCS_TEST_P(test_uct_cm_sockaddr, many_conns_on_client)
{
    int num_conns_on_client = ucs_max(2, 100 / ucs::test_time_multiplier());

    m_server_start_disconnect = true;

    skip_tcp_sockcm();
    /* Listen */
    cm_start_listen();

    /* Connect */
    /* multiple clients, on the same cm, connecting to the same server */
    for (int i = 0; i < num_conns_on_client; ++i) {
        m_client->connect(i, *m_server, 0, m_connect_addr, client_cm_priv_data_cb,
                          client_connect_cb, client_disconnect_cb, this);
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

UCS_TEST_P(test_uct_cm_sockaddr, err_handle)
{
    skip_tcp_sockcm();

    /* wrap errors since a reject is expected */
    scoped_log_handler slh(detect_reject_error_logger);

    /* client - try to connect to a server that isn't listening */
    m_client->connect(0, *m_server, 0, m_connect_addr, client_cm_priv_data_cb,
                      client_connect_cb, client_disconnect_cb, this);

    EXPECT_FALSE(m_cm_state & TEST_CM_STATE_CONNECT_REQUESTED);

    /* with the TCP port space (which is currently tested with rdmacm),
     * in this case, a REJECT event will be generated on the client side */
    wait_for_bits(&m_cm_state, TEST_CM_STATE_CLIENT_GOT_REJECT);
    EXPECT_TRUE(ucs_test_all_flags(m_cm_state, TEST_CM_STATE_CLIENT_GOT_REJECT));
}

UCS_TEST_P(test_uct_cm_sockaddr, conn_to_non_exist_server_port)
{
    skip_tcp_sockcm();
    /* Listen */
    cm_start_listen();

    m_connect_addr.set_port(htons(1));

    /* wrap errors since a reject is expected */
    scoped_log_handler slh(detect_reject_error_logger);

    /* client - try to connect to a non-existing port on the server side. */
    m_client->connect(0, *m_server, 0, m_connect_addr, client_cm_priv_data_cb,
                      client_connect_cb, client_disconnect_cb, this);

    /* with the TCP port space (which is currently tested with rdmacm),
     * in this case, a REJECT event will be generated on the client side */
    wait_for_bits(&m_cm_state, TEST_CM_STATE_CLIENT_GOT_REJECT);
    EXPECT_TRUE(ucs_test_all_flags(m_cm_state, TEST_CM_STATE_CLIENT_GOT_REJECT));
}

UCS_TEST_P(test_uct_cm_sockaddr, conn_to_non_exist_ip)
{
    struct sockaddr_in addr;
    ucs_status_t status;
    size_t size;

    skip_tcp_sockcm();
    /* Listen */
    cm_start_listen();

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
        m_client->connect(0, *m_server, 0, m_connect_addr, client_cm_priv_data_cb,
                          client_connect_cb, client_disconnect_cb, this);

        wait_for_bits(&m_cm_state, TEST_CM_STATE_CLIENT_GOT_ERROR);
        EXPECT_TRUE(ucs_test_all_flags(m_cm_state, TEST_CM_STATE_CLIENT_GOT_ERROR));

        EXPECT_FALSE(m_cm_state & TEST_CM_STATE_CONNECT_REQUESTED);
        EXPECT_FALSE(m_cm_state &
                    (TEST_CM_STATE_SERVER_CONNECTED | TEST_CM_STATE_CLIENT_CONNECTED));
    }
}

UCS_TEST_P(test_uct_cm_sockaddr, connect_client_to_server_with_delay)
{
    test_delayed_server_response(false);

    cm_disconnect(m_client);
}

UCS_TEST_P(test_uct_cm_sockaddr, connect_client_to_server_reject_with_delay)
{
    test_delayed_server_response(true);
}

UCS_TEST_P(test_uct_cm_sockaddr, ep_disconnect_err_codes)
{
    bool disconnecting = false;

    cm_listen_and_connect();

    {
        entity::scoped_async_lock lock(*m_client);
        if (m_cm_state & TEST_CM_STATE_CLIENT_CONNECTED) {
            UCS_TEST_MESSAGE << "EXP: " << ucs_status_string(UCS_OK);
            EXPECT_EQ(UCS_OK, uct_ep_disconnect(m_client->ep(0), 0));
            disconnecting = true;
        } else {
            UCS_TEST_MESSAGE << "EXP: " << ucs_status_string(UCS_ERR_BUSY);
            EXPECT_EQ(UCS_ERR_BUSY, uct_ep_disconnect(m_client->ep(0), 0));
        }
    }

    wait_for_bits(&m_cm_state, TEST_CM_STATE_SERVER_CONNECTED |
                               TEST_CM_STATE_CLIENT_CONNECTED);
    EXPECT_TRUE(ucs_test_all_flags(m_cm_state, (TEST_CM_STATE_SERVER_CONNECTED |
                                                TEST_CM_STATE_CLIENT_CONNECTED)));

    {
        entity::scoped_async_lock lock(*m_client);
        if (disconnecting) {
            scoped_log_handler slh(detect_double_disconnect_error_logger);
            if (m_cm_state & TEST_CM_STATE_CLIENT_DISCONNECTED) {
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
    wait_for_bits(&m_cm_state, TEST_CM_STATE_CLIENT_DISCONNECTED);
    EXPECT_TRUE(m_cm_state & TEST_CM_STATE_CLIENT_DISCONNECTED);

    /* wrap errors since the client will call uct_ep_disconnect the second time
     * on the same endpoint. this ep may not be disconnected yet */
    {
        scoped_log_handler slh(detect_double_disconnect_error_logger);
        UCS_TEST_MESSAGE << "EXP: " << ucs_status_string(UCS_ERR_NOT_CONNECTED);
        EXPECT_EQ(UCS_ERR_NOT_CONNECTED, uct_ep_disconnect(m_client->ep(0), 0));
    }
}

UCT_INSTANTIATE_SOCKADDR_TEST_CASE(test_uct_cm_sockaddr)


class test_uct_cm_sockaddr_multiple_cms : public test_uct_cm_sockaddr {
public:
    void init() {
        ucs_status_t status;

        test_uct_cm_sockaddr::init();

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
        test_uct_cm_sockaddr::cleanup();
    }

    void server_accept(entity *server, uct_conn_request_h conn_request,
                       uct_ep_server_connect_cb_t connect_cb,
                       uct_ep_disconnect_cb_t disconnect_cb,
                       void *user_data)
    {
        server->accept(m_test_cm, conn_request, connect_cb, disconnect_cb,
                       user_data);
    }

protected:
    ucs::handle<uct_worker_h> m_test_worker;
    ucs::handle<uct_cm_h>     m_test_cm;
    ucs_async_context_t       *m_test_async;
    uct_cm_config_t           *m_test_config;
};

UCS_TEST_P(test_uct_cm_sockaddr_multiple_cms, server_switch_cm)
{
    cm_listen_and_connect();

    wait_for_bits(&m_cm_state, TEST_CM_STATE_SERVER_CONNECTED |
                               TEST_CM_STATE_CLIENT_CONNECTED);
    EXPECT_TRUE(ucs_test_all_flags(m_cm_state, (TEST_CM_STATE_SERVER_CONNECTED |
                                                TEST_CM_STATE_CLIENT_CONNECTED)));

    cm_disconnect(m_client);

    /* destroy the server's ep here so that it would be destroyed before the cm
     * it is using */
    m_server->destroy_ep(0);
}

UCT_INSTANTIATE_SOCKADDR_TEST_CASE(test_uct_cm_sockaddr_multiple_cms)
