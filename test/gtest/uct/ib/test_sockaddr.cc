/**
* Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
#include <uct/uct_test.h>

extern "C" {
#include <uct/api/uct.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/string.h>
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

    void init() {
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

        server = uct_test::create_entity(server_params);
        m_entities.push_back(server);

        /* if origin port is busy create_entity will retry with other one */
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
        client->client_cb_arg = server->iface_attr().max_conn_priv;
    }

    static void conn_request_cb(uct_iface_h iface, void *arg,
                                uct_conn_request_h conn_request,
                                const void *conn_priv_data, size_t length)
    {
        test_uct_sockaddr *self = reinterpret_cast<test_uct_sockaddr*>(arg);

        EXPECT_EQ(std::string(reinterpret_cast<const char *>
                              (uct_test::entity::client_priv_data.c_str())),
                  std::string(reinterpret_cast<const char *>(conn_priv_data)));
//        std::vector<char> tmp(length);
//        memcpy(&tmp[0], conn_priv_data, length);
//        EXPECT_EQ(uct_test::entity::client_priv_data, tmp);

        EXPECT_EQ(1 + uct_test::entity::client_priv_data.length(), length);
//        EXPECT_EQ(uct_test::entity::client_priv_data.size(), length);
        if (self->delay_conn_reply) {
            self->delayed_conn_reqs.push(conn_request);
        } else {
            uct_iface_accept(iface, conn_request);
        }
        self->server_recv_req++;
    }

    static ucs_status_t err_handler(void *arg, uct_ep_h ep, ucs_status_t status)
    {
        test_uct_sockaddr *self = reinterpret_cast<test_uct_sockaddr*>(arg);
        self->err_count++;
        return UCS_OK;
    }

protected:
    entity *server, *client;
    ucs::sock_addr_storage m_listen_addr, m_connect_addr;
    volatile int err_count, server_recv_req;
    std::queue<uct_conn_request_h> delayed_conn_reqs;
    bool delay_conn_reply;
};

UCS_TEST_P(test_uct_sockaddr, connect_client_to_server)
{
    UCS_TEST_MESSAGE << "Testing "     << m_listen_addr
                     << " Interface: " << GetParam()->dev_name;

    client->connect(0, *server, 0, m_connect_addr, NULL, NULL,
                    &client->client_cb_arg);

    /* wait for the server to connect */
    while (server_recv_req == 0) {
        progress();
    }
    ASSERT_TRUE(server_recv_req == 1);
    /* since the transport may support a graceful exit in case of an error,
     * make sure that the error handling flow wasn't invoked (there were no
     * errors) */
    EXPECT_EQ(0, err_count);
    /* the test may end before the client's ep got connected.
     * it should also pass in this case as well - the client's
     * ep shouldn't be accessed (for connection reply from the server) after the
     * test ends and the client's ep was destroyed */
}

UCS_TEST_P(test_uct_sockaddr, connect_client_to_server_with_delay)
{
    UCS_TEST_MESSAGE << "Testing "     << m_listen_addr
                     << " Interface: " << GetParam()->dev_name;
    delay_conn_reply = true;
    client->connect(0, *server, 0, m_connect_addr, NULL, NULL,
                    &client->client_cb_arg);

    /* wait for the server to connect */
    while (server_recv_req == 0) {
        progress();
    }
    ASSERT_EQ(1,   server_recv_req);
    ASSERT_EQ(1ul, delayed_conn_reqs.size());
    EXPECT_EQ(0,   err_count);
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
    EXPECT_EQ(0, err_count);
}

UCS_TEST_P(test_uct_sockaddr, connect_client_to_server_reject_with_delay)
{
    UCS_TEST_MESSAGE << "Testing "     << m_listen_addr
                     << " Interface: " << GetParam()->dev_name;
    delay_conn_reply = true;
    client->connect(0, *server, 0, m_connect_addr, NULL, NULL,
                    &client->client_cb_arg);

    /* wait for the server to connect */
    while (server_recv_req == 0) {
        progress();
    }
    ASSERT_EQ(1, server_recv_req);
    ASSERT_EQ(1ul, delayed_conn_reqs.size());
    EXPECT_EQ(0, err_count);
    while (!delayed_conn_reqs.empty()) {
        uct_iface_reject(server->iface(), delayed_conn_reqs.front());
        delayed_conn_reqs.pop();
    }
    while (err_count == 0) {
        progress();
    }
    EXPECT_EQ(1, err_count);
}

UCS_TEST_P(test_uct_sockaddr, many_clients_to_one_server)
{
    UCS_TEST_MESSAGE << "Testing "     << m_listen_addr
                     << " Interface: " << GetParam()->dev_name;

    uct_iface_params_t client_params;
    entity *client_test;
    int i, num_clients = 100;

    /* multiple clients, each on an iface of its own, connecting to the same server */
    for (i = 0; i < num_clients; ++i) {
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

        client_test->client_cb_arg = server->iface_attr().max_conn_priv;
        client_test->connect(i, *server, 0, m_connect_addr, NULL, NULL,
                             &client_test->client_cb_arg);
    }

    while (server_recv_req < num_clients){
        progress();
    }
    ASSERT_TRUE(server_recv_req == num_clients);
    EXPECT_EQ(0, err_count);
}

UCS_TEST_P(test_uct_sockaddr, many_conns_on_client)
{
    UCS_TEST_MESSAGE << "Testing "     << m_listen_addr
                     << " Interface: " << GetParam()->dev_name;

    int i, num_conns_on_client = 100;

    /* multiple clients, on the same iface, connecting to the same server */
    for (i = 0; i < num_conns_on_client; ++i) {
        client->connect(i, *server, 0, m_connect_addr, NULL, NULL,
                        &client->client_cb_arg);
    }

    while (server_recv_req < num_conns_on_client) {
        progress();
    }
    ASSERT_TRUE(server_recv_req == num_conns_on_client);
    EXPECT_EQ(0, err_count);
}

UCS_TEST_P(test_uct_sockaddr, err_handle)
{
    check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE);
    UCS_TEST_MESSAGE << "Testing "     << m_listen_addr
                     << " Interface: " << GetParam()->dev_name;

    client->connect(0, *server, 0, m_connect_addr, NULL, NULL,
                    &client->client_cb_arg);

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

UCS_TEST_P(test_uct_sockaddr, conn_to_non_exist_server)
{
    check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE);

    UCS_TEST_MESSAGE << "Testing "     << m_listen_addr
                     << " Interface: " << GetParam()->dev_name;

    m_connect_addr.set_port(1);
    err_count = 0;

    /* wrap errors now since the client will try to connect to a non existing port */
    {
        scoped_log_handler slh(wrap_errors_logger);
        /* client - try to connect to a non-existing port on the server side */
        client->connect(0, *server, 0, m_connect_addr, NULL, NULL,
                        &client->client_cb_arg);
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
        TEST_CM_STATE_CLIENT_GOT_REJECT  = UCS_BIT(6),
        TEST_CM_STATE_NEVER               = UCS_BIT(63) /* for debugging */
    };

public:
    test_uct_cm_sockaddr() : m_cm_state(0), m_server(NULL), m_client(NULL),
                             server_recv_req_cnt(0), client_connect_cb_cnt(0),
                             server_connect_cb_cnt(0),
                             server_disconnect_cnt(0), client_disconnect_cnt(0),
                             reject_conn_request(false) {
    }

    void init() {
        ucs_status_t status;
        uint16_t port;
        size_t i;

        uct_test::init();

        /* This address is accessible, as it was tested at the resource creation */
        m_listen_addr  = GetParam()->listen_sock_addr;
        m_connect_addr = GetParam()->connect_sock_addr;

        port = ucs::get_port();
        m_listen_addr.set_port(port);
        m_connect_addr.set_port(port);

        m_server = uct_test::create_entity();
        m_entities.push_back(m_server);
        m_client = uct_test::create_entity();
        m_entities.push_back(m_client);

        for (i = 0; i < m_entities.size(); ++i) {
            uct_cm_attr_t attr;
            status = uct_cm_query(m_entities.at(i).cm(), &attr);
            ASSERT_UCS_OK(status);
            EXPECT_LE(0ul, attr.max_conn_priv);
        }

        /* initiate the client's private data callback argument */
        m_client->client_cb_arg = m_client->cm_attr().max_conn_priv;
    }
protected:

    void cm_start_listen() {
        uct_listener_params_t params;

        params.field_mask      = UCT_LISTENER_PARAM_FIELD_CONN_REQUEST_CB |
                                 UCT_LISTENER_PARAM_FIELD_USER_DATA;
        params.conn_request_cb = cm_conn_request_cb;
        params.user_data       = static_cast<test_uct_cm_sockaddr *>(this);
        m_server->listen(m_listen_addr, params);
    }

    void cm_listen_and_connect() {
        cm_start_listen();
        m_client->connect(0, *m_server, 0, m_connect_addr,
                          client_connect_cb, client_disconnect_cb, this);

        wait_for_bits(&m_cm_state, TEST_CM_STATE_CONNECT_REQUESTED);
        EXPECT_TRUE(m_cm_state & TEST_CM_STATE_CONNECT_REQUESTED);
    }

    void cm_disconnect(entity *client) {
        size_t i;

        /* Disconnect all the existing endpoints */
        for (i = 0; i < client->num_eps(); ++i) {
            client->disconnect(client->ep(i));
        }

        wait_for_bits(&m_cm_state, TEST_CM_STATE_CLIENT_DISCONNECTED |
                                   TEST_CM_STATE_SERVER_DISCONNECTED);
        EXPECT_TRUE(ucs_test_all_flags(m_cm_state, (TEST_CM_STATE_SERVER_DISCONNECTED |
                                                    TEST_CM_STATE_CLIENT_DISCONNECTED)));
    }

    static void
    cm_conn_request_cb(uct_listener_h listener, void *arg,
                       const char *local_dev_name,
                       uct_conn_request_h conn_request,
                       const uct_cm_remote_data_t *remote_data) {
        test_uct_cm_sockaddr *self;
        ucs_status_t status;

        self = reinterpret_cast<test_uct_cm_sockaddr *>(arg);

        EXPECT_EQ(entity::client_priv_data.length() + 1, remote_data->conn_priv_data_length);
        EXPECT_EQ(entity::client_priv_data,
                  std::string(static_cast<const char *>(remote_data->conn_priv_data)));

//        EXPECT_EQ(entity::client_priv_data.size(), length);
//        std::vector<char> tmp(length);
//        memcpy(&tmp[0], conn_priv_data, length);
//        EXPECT_EQ(entity::client_priv_data, tmp);

        self->server_recv_req_cnt++;
        self->m_cm_state |= TEST_CM_STATE_CONNECT_REQUESTED;

        if (!self->reject_conn_request) {
            self->m_server->accept(conn_request, server_connect_cb,
                                   server_disconnect_cb, self);
        } else {
            status = uct_listener_reject(listener, conn_request);
            ASSERT_UCS_OK(status);
            self->m_cm_state |= TEST_CM_STATE_SERVER_REJECTED;
        }
    }

    static void
    server_connect_cb(uct_ep_h ep, void *arg, ucs_status_t status) {
        test_uct_cm_sockaddr *self;

        self = reinterpret_cast<test_uct_cm_sockaddr *>(arg);
        self->m_cm_state |= TEST_CM_STATE_SERVER_CONNECTED;
        self->server_connect_cb_cnt++;
    }

    static void
    server_disconnect_cb(uct_ep_h ep, void *arg) {
        test_uct_cm_sockaddr *self;

        self = reinterpret_cast<test_uct_cm_sockaddr *>(arg);
        self->m_server->disconnect(ep);
        self->m_cm_state |= TEST_CM_STATE_SERVER_DISCONNECTED;
        self->server_disconnect_cnt++;
    }

    static void
    client_connect_cb(uct_ep_h ep, void *arg,
                      const uct_cm_remote_data_t *remote_data,
                      ucs_status_t status) {
        test_uct_cm_sockaddr *self = reinterpret_cast<test_uct_cm_sockaddr *>(arg);

        if (status == UCS_ERR_REJECTED) {
            self->m_cm_state |= TEST_CM_STATE_CLIENT_GOT_REJECT;
        } else {
            ASSERT_UCS_OK(status);
            EXPECT_EQ(entity::server_priv_data.length() + 1, remote_data->conn_priv_data_length);
            EXPECT_EQ(entity::server_priv_data,
                      std::string(static_cast<const char *>(remote_data->conn_priv_data)));
            self->client_connect_cb_cnt++;
            self->m_cm_state |= TEST_CM_STATE_CLIENT_CONNECTED;
        }

    }

    static void
    client_disconnect_cb(uct_ep_h ep, void *arg) {
        test_uct_cm_sockaddr *self;

        self = reinterpret_cast<test_uct_cm_sockaddr *>(arg);
        self->m_cm_state |= TEST_CM_STATE_CLIENT_DISCONNECTED;
        self->client_disconnect_cnt++;
    }

protected:
    ucs::sock_addr_storage m_listen_addr, m_connect_addr;
    uint64_t        m_cm_state;
    entity          *m_server;
    entity          *m_client;
    volatile int    server_recv_req_cnt, client_connect_cb_cnt, server_connect_cb_cnt;
    volatile int    server_disconnect_cnt, client_disconnect_cnt;
    bool            reject_conn_request;
};

UCS_TEST_P(test_uct_cm_sockaddr, cm_open_listen_close)
{
    UCS_TEST_MESSAGE << "Testing " << m_listen_addr
                     << " Interface: " << GetParam()->dev_name;

    cm_listen_and_connect();

    wait_for_bits(&m_cm_state, TEST_CM_STATE_SERVER_CONNECTED |
                               TEST_CM_STATE_CLIENT_CONNECTED);
    EXPECT_TRUE(ucs_test_all_flags(m_cm_state, (TEST_CM_STATE_SERVER_CONNECTED |
                                                TEST_CM_STATE_CLIENT_CONNECTED)));

    cm_disconnect(m_client);

//    wait_for_bits(&m_cm_state, TEST_CM_STATE_NEVER);
}

UCS_TEST_P(test_uct_cm_sockaddr, cm_server_reject)
{
    UCS_TEST_MESSAGE << "Testing "     << m_listen_addr
                     << " Interface: " << GetParam()->dev_name;

    reject_conn_request = true;

    cm_listen_and_connect();

    wait_for_bits(&m_cm_state, TEST_CM_STATE_SERVER_REJECTED |
                               TEST_CM_STATE_CLIENT_GOT_REJECT);
    EXPECT_TRUE(ucs_test_all_flags(m_cm_state, (TEST_CM_STATE_SERVER_REJECTED |
                                                TEST_CM_STATE_CLIENT_GOT_REJECT)));
}

UCS_TEST_P(test_uct_cm_sockaddr, many_clients_to_one_server)
{
    int i, num_clients = 100;
    entity *client_test;

    UCS_TEST_MESSAGE << "Testing "     << m_listen_addr
                     << " Interface: " << GetParam()->dev_name;

    /* Listen */
    cm_start_listen();

    /* Connect */
    /* multiple clients, each on a cm of its own, connecting to the same server */
    for (i = 0; i < num_clients; ++i) {
        client_test = uct_test::create_entity();
        m_entities.push_back(client_test);
        client_test->client_cb_arg = client_test->cm_attr().max_conn_priv;
        client_test->connect(0, *m_server, 0, m_connect_addr,
                             client_connect_cb, client_disconnect_cb, this);
    }

    /* wait for the server to connect to all the clients */
    while ((client_connect_cb_cnt < num_clients) ||
           (server_connect_cb_cnt < num_clients )) {
        progress();
    }
    EXPECT_EQ(num_clients, server_recv_req_cnt);
    EXPECT_EQ(num_clients, client_connect_cb_cnt);
    EXPECT_EQ(num_clients, server_connect_cb_cnt);
    EXPECT_EQ(num_clients, m_server->num_eps());

    /* Disconnect */
    for (i = 0; i < num_clients; ++i) {
        client_test = m_entities.back();
        ASSERT_TRUE(client_test != m_client);
        cm_disconnect(client_test);

        /* don't remove the ep, i.e. don't call uct_ep_destroy before the client
         * finished disconnecting so that a Disconnect event won't arrive on a
         * destroyed endpoint on the client side */
        while (client_disconnect_cnt < (i + 1)) {
            progress();
        }

        m_entities.remove(client_test);
    }

    while ((server_disconnect_cnt < num_clients) ||
           (client_disconnect_cnt < num_clients)) {
        progress();
    }
    EXPECT_EQ(num_clients, server_disconnect_cnt);
    EXPECT_EQ(num_clients, client_disconnect_cnt);
}

UCS_TEST_P(test_uct_cm_sockaddr, many_conns_on_client)
{
    int i, num_conns_on_client = 100;

    UCS_TEST_MESSAGE << "Testing "     << m_listen_addr
                     << " Interface: " << GetParam()->dev_name;

    /* Listen */
    cm_start_listen();

    /* Connect */
    /* multiple clients, on the same cm, connecting to the same server */
    for (i = 0; i < num_conns_on_client; ++i) {
        m_client->connect(i, *m_server, 0, m_connect_addr,
                          client_connect_cb, client_disconnect_cb, this);
    }

    /* wait for the server to connect to all the endpoints on the cm */
    while ((client_connect_cb_cnt < num_conns_on_client) ||
           (server_connect_cb_cnt < num_conns_on_client )) {
        progress();
    }
    EXPECT_EQ(num_conns_on_client, server_recv_req_cnt);
    EXPECT_EQ(num_conns_on_client, client_connect_cb_cnt);
    EXPECT_EQ(num_conns_on_client, server_connect_cb_cnt);
    EXPECT_EQ(num_conns_on_client, m_client->num_eps());
    EXPECT_EQ(num_conns_on_client, m_server->num_eps());

    /* Disconnect */
    cm_disconnect(m_client);

    /* wait for disconnect to complete */
    while ((server_disconnect_cnt < num_conns_on_client) ||
           (client_disconnect_cnt < num_conns_on_client)) {
        progress();
    }
    EXPECT_EQ(num_conns_on_client, server_disconnect_cnt);
    EXPECT_EQ(num_conns_on_client, client_disconnect_cnt);
}

UCT_INSTANTIATE_SOCKADDR_TEST_CASE(test_uct_cm_sockaddr)
