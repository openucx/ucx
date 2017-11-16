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
        addr_in->sin_port = 0;   /* Use a random port */

        /* open iface for the server side */
        memset(&server_params, 0, sizeof(server_params));
        server_params.open_mode                      = UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER;
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
        client_params.err_handler_arg                = reinterpret_cast<void*>(this);

        client = uct_test::create_entity(client_params);
        m_entities.push_back(client);
    }

    static ssize_t conn_request_cb(void *arg, const void *conn_priv_data,
                                   size_t length, void *reply_priv_data)
    {
        return 0;
    }

protected:
    entity *server, *client;
    ucs_sock_addr_t sock_addr;;
};

UCS_TEST_P(test_uct_sockaddr, iface_open)
{   /* A temporary empty test to test the initialization of the interface */
    UCS_TEST_MESSAGE << "Opening an iface for " << ucs::get_iface_ip(sock_addr.addr);
}

UCT_INSTANTIATE_SOCKADDR_TEST_CASE(test_uct_sockaddr)
