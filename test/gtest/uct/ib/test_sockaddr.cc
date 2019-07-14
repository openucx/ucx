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

        check_skip_test();

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

        EXPECT_EQ(1 + uct_test::entity::client_priv_data.length(), length);
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

    client->connect(0, *server, 0, m_connect_addr);

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
    client->connect(0, *server, 0, m_connect_addr);

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
    client->connect(0, *server, 0, m_connect_addr);

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
        client_test->connect(i, *server, 0, m_connect_addr);
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
        client->connect(i, *server, 0, m_connect_addr);
    }

    while (server_recv_req < num_conns_on_client) {
        progress();
    }
    ASSERT_TRUE(server_recv_req == num_conns_on_client);
    EXPECT_EQ(0, err_count);
}

UCS_TEST_SKIP_COND_P(test_uct_sockaddr, err_handle,
                     !check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE))
{
    UCS_TEST_MESSAGE << "Testing "     << m_listen_addr
                     << " Interface: " << GetParam()->dev_name;

    client->connect(0, *server, 0, m_connect_addr);

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
    UCS_TEST_MESSAGE << "Testing "     << m_listen_addr
                     << " Interface: " << GetParam()->dev_name;

    m_connect_addr.set_port(1);
    err_count = 0;

    /* wrap errors now since the client will try to connect to a non existing port */
    {
        scoped_log_handler slh(wrap_errors_logger);
        /* client - try to connect to a non-existing port on the server side */
        client->connect(0, *server, 0, m_connect_addr);
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
