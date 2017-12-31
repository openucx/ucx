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

class test_uct_sockaddr : public uct_test {
public:
    void init() {
        uct_test::init();

        uct_iface_params server_params, client_params;
        struct sockaddr_in *addr_in;

        /* If we reached here, the interface is active, as it was tested at the
         * resource creation */
        if (!ucs::is_inet_addr((struct sockaddr *)&(GetParam()->if_addr))) {
            UCS_TEST_SKIP_R("There is no IP on the interface");
        }

        /* If rdmacm is tested, make sure that this is an IPoIB or RoCE interface */
        if (!strcmp(GetParam()->md_name.c_str(), "rdmacm") &&
            (!ucs::is_ib_netdev(GetParam()->dev_name.c_str()))) {
            UCS_TEST_SKIP_R("rdmacm - not an IPoIB or RoCE interface");
        }

        /* This address is accessible, as it was tested at the resource creation */
        sock_addr.addr = (struct sockaddr *)&(GetParam()->if_addr);
        ASSERT_TRUE(sock_addr.addr != NULL);

        addr_in = (struct sockaddr_in *) (sock_addr.addr);

        /* Get a usable port on the host */
        addr_in->sin_port = ucs::get_port();

        /* open iface for the server side */
        memset(&server_params, 0, sizeof(server_params));
        server_params.open_mode                      = UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER;
        server_params.err_handler                    = err_handler;
        server_params.err_handler_arg                = reinterpret_cast<void*>(this);
        server_params.mode.sockaddr.listen_sockaddr  = sock_addr;
        server_params.mode.sockaddr.cb_flags         = UCT_CB_FLAG_ASYNC;
        server_params.mode.sockaddr.conn_request_cb  = conn_request_cb;
        server_params.mode.sockaddr.conn_request_arg = reinterpret_cast<void*>(this);

        server = uct_test::create_entity(server_params);
        m_entities.push_back(server);

        /* open iface for the client side */
        memset(&client_params, 0, sizeof(client_params));
        client_params.open_mode                      = UCT_IFACE_OPEN_MODE_SOCKADDR_CLIENT;
        client_params.err_handler                    = err_handler;
        client_params.err_handler_arg                = reinterpret_cast<void*>(this);

        client = uct_test::create_entity(client_params);
        m_entities.push_back(client);
    }

    static ssize_t conn_request_cb(void *arg, const void *conn_priv_data,
                                   size_t length, void *reply_priv_data)
    {
        test_uct_sockaddr *self = reinterpret_cast<test_uct_sockaddr*>(arg);

        EXPECT_EQ(sizeof(uint64_t), length);
        EXPECT_EQ(0xdeadbeef, *(uint64_t*)conn_priv_data);

        memcpy(reply_priv_data, uct_test::entity::server_priv_data.c_str(),
               uct_test::entity::server_priv_data.length() + 1);

        self->server_recv_req = 1;
        return uct_test::entity::server_priv_data.length() + 1;
    }

    static void err_handler(void *arg, uct_ep_h ep, ucs_status_t status)
    {
        test_uct_sockaddr *self = reinterpret_cast<test_uct_sockaddr*>(arg);
        self->err_count++;
    }

protected:
    entity *server, *client;
    ucs_sock_addr_t sock_addr;
    volatile int err_count, server_recv_req;
};

UCS_TEST_P(test_uct_sockaddr, connect_client_to_server)
{
    UCS_TEST_MESSAGE << "Testing " << ucs::sockaddr_to_str(sock_addr.addr);

    uct_test::entity::client_connected = 0;
    err_count = 0;
    client->connect(0, *server, 0);

    while (uct_test::entity::client_connected == 0);
    ASSERT_TRUE(uct_test::entity::client_connected == 1);

    /* since the transport may support a graceful exit in case of an error,
     * make sure that the error handling flow wasn't invoked (there were no
     * errors) */
    EXPECT_EQ(0, err_count);
}

UCS_TEST_P(test_uct_sockaddr, err_handle)
{
    check_caps(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE);
    UCS_TEST_MESSAGE << "Testing " << ucs::sockaddr_to_str(sock_addr.addr);

    uct_test::entity::client_connected = 0;
    server_recv_req = 0;
    err_count = 0;

    client->connect(0, *server, 0);

    /* kill the server */
    wrap_errors();
    m_entities.remove(server);

    /* If the server didn't receive a connection request from the client yet,
     * test error handling */
    if (server_recv_req == 0) {
        wait_for_flag(&err_count);
        EXPECT_EQ(1, err_count);
        ASSERT_TRUE(uct_test::entity::client_connected == 0);
    }
    restore_errors();
}

UCT_INSTANTIATE_SOCKADDR_TEST_CASE(test_uct_sockaddr)
