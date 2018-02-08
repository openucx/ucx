/**
* Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

/*
 * UCP client - server example utility
 * -----------------------------------------------
 *
 * Server side:
 *
 *    UCX_NET_DEVICES=mlx5_0:1 UCX_TLS=rc ./ucp_client_server
 *
 * Client side:
 *
 *    UCX_NET_DEVICES=mlx5_0:1 UCX_TLS=rc ./ucp_client_server <server-ip>
 *
 * Notes:
 *
 *    - The server will listen to incoming connection requests on INADDR_ANY.
 *    - The client needs to pass the IP address of the server side to connect to,
 *      as the first and only argument to the test.
 *    - Currently, the passed IP needs to be an IPoIB address.
 *    - The amount of used resources (HCA's and transports) needs to be limited
 *      for this test (for example: UCX_NET_DEVICES=mlx5_0:1 UCX_TLS=rc_x).
 *      This is currently required since the UCP layer has a limitation on
 *      the size of the transfered transports addresses that are being passed
 *      to the remote peer.
 *
 */

#include "ucx_hello_world.h"

#include <ucp/api/ucp.h>

#include <arpa/inet.h> /*inet_addr */
#include <unistd.h>  /* getopt */
#include <ctype.h>   /* isprint */
#include <pthread.h> /* pthread_self */

const char test_message[] = "UCX Client-Server Hello World";
static uint16_t server_port = 13337;

#define TEST_STRING_LEN sizeof(test_message)

typedef struct ucx_server_ctx {
    ucp_ep_h     ep;
} ucx_server_ctx_t;

typedef struct test_req {
    int complete;
} test_req_t;


/**
 * The callback on the receiving side, which is invoked upon receiving the
 * stream message.
 */
static void stream_recv_cb(void *request, ucs_status_t status, size_t length)
{
    test_req_t *req = request;

    req->complete = 1;

    printf("[0x%x] stream_recv_cb called with status %d (%s), length: %lu\n",
           (unsigned int)pthread_self(), status, ucs_status_string(status),
           length);
}

/**
 * The callback on the sending side, which is invoked after finishing sending
 * the stream message.
 */
static void stream_send_cb(void *request, ucs_status_t status)
{
    test_req_t *req = request;

    req->complete = 1;

    printf("[0x%x] stream_send_cb called with status %d (%s)\n",
           (unsigned int)pthread_self(), status, ucs_status_string(status));
}

/**
 * The callback on the server side which is invoked upon receiving a connection
 * request from the client.
 */
static void server_accept_cb(ucp_ep_h ep, void *arg)
{
    ucx_server_ctx_t *context = arg;

    /* Save the server's endpoint in the user's context, for future usage */
    context->ep = ep;
}

/**
 * Create a listener on the server side to listen on the given address.
 */
ucs_status_t server_listen(ucp_worker_h ucp_worker, const struct sockaddr* addr,
                           socklen_t addrlen, ucp_listener_h *listener,
                           ucx_server_ctx_t *context)
{
    ucp_listener_params_t params;
    ucs_status_t status;

    params.field_mask         = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER;
    params.sockaddr.addr      = addr;
    params.sockaddr.addrlen   = addrlen;
    params.accept_handler.cb  = server_accept_cb;
    params.accept_handler.arg = context;

    status = ucp_listener_create(ucp_worker, &params, listener);
    if (status == UCS_OK) {
        printf("Waiting for connection...\n");
    } else {
        fprintf(stderr, "failed to listen\n");
    }

    return status;
}

/**
 * Set an address for the server to listen on - INADDR_ANY on a well known port.
 */
void set_listen_addr(struct sockaddr_in *listen_addr)
{
    /* The server will listen on INADDR_ANY */
    memset(listen_addr, 0, sizeof(struct sockaddr_in));
    listen_addr->sin_family      = AF_INET;
    listen_addr->sin_addr.s_addr = INADDR_ANY;
    listen_addr->sin_port        = server_port;
}

/**
 * Set an address to connect to. A given IP address on a well known port.
 */
void set_connect_addr(const char *ip, struct sockaddr_in *connect_addr)
{
    memset(connect_addr, 0, sizeof(struct sockaddr_in));
    connect_addr->sin_family      = AF_INET;
    connect_addr->sin_addr.s_addr = inet_addr(ip);
    connect_addr->sin_port        = server_port;
}

/**
 * Initialize the server side. The server starts listening on the set address
 * and waits for its connected endpoint to be created.
 */
static int start_server(ucp_worker_h ucp_worker, ucx_server_ctx_t *context,
                        ucp_listener_h *listener)
{
    struct sockaddr_in listen_addr;
    ucs_status_t status;

    context->ep = NULL;

    set_listen_addr(&listen_addr);
    status = server_listen(ucp_worker, (const struct sockaddr*)&listen_addr,
                           sizeof(listen_addr), listener,
                           context);

    if (status == UCS_OK) {
        while (context->ep == NULL) {
            ucp_worker_progress(ucp_worker);
        }
    }

    return status;
}

/**
 * Initialize the client side. Create an endpoint from the client side to be
 * connected to the remote server (to the given IP).
 */
static ucs_status_t start_client(ucp_worker_h ucp_worker, const char *ip,
                                 ucp_ep_h *client_ep)
{
    ucp_ep_params_t ep_params;
    struct sockaddr_in connect_addr;
    ucs_status_t status;

    set_connect_addr(ip, &connect_addr);

    /*
     * Endpoint field mask bits:
     * UCP_EP_PARAM_FIELD_FLAGS             - Use the value of the 'flags' field.
     * UCP_EP_PARAM_FIELD_SOCK_ADDR         - Use a remote sockaddr to connect
     *                                        to the remote peer.
     * UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE - Error handling mode - this flag
     *                                        is temporarily required since the
     *                                        endpoint will be closed with
     *                                        UCP_EP_CLOSE_MODE_FORCE which
     *                                        requires this mode.
     *                                        Once UCP_EP_CLOSE_MODE_FORCE is
     *                                        removed, the error handling mode
     *                                        will be removed.
     */
    ep_params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS     |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr    = (struct sockaddr*)&connect_addr;
    ep_params.sockaddr.addrlen = sizeof(connect_addr);

    status = ucp_ep_create(ucp_worker, &ep_params, client_ep);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to connect to %s\n", ip);
    }

    return status;
}

/**
 * Verify the received message on the server side and print the result.
 */
static int verify_result(int is_server, char *recv_message)
{
    int ret;

    if (is_server) {
        if (!strcmp(recv_message, test_message)) {
            printf("\n\n----- UCP TEST SUCCESS -------\n\n");
            printf("%s", recv_message);
            printf("\n\n------------------------------\n\n");

            ret = 0;
        } else {
            printf("\n\n----- UCP TEST FAILURE -------\n\n");
            printf("Client sent message:     %s\nServer received message: %s\n",
                   test_message, recv_message);
            printf("\n\n------------------------------\n\n");

            ret = -1;
        }
    } else {
        printf("\n\n-----------------------------------------\n\n");
        printf("Client sent message: \n%s.\nlength: %ld\n",
               test_message, TEST_STRING_LEN);
        printf("\n-----------------------------------------\n\n");
        ret = 0;
    }

    return ret;
}

/**
 * Send and receive a message using the Stream API.
 * The client sends a message to the server and waits until the send it completed.
 * The server receives a message from the client and waits for its completion.
 */
static int send_recv_stream(ucp_worker_h ucp_worker, ucp_ep_h ep, int is_server)
{
    char recv_message[TEST_STRING_LEN]= "";
    test_req_t *request;
    size_t length;
    int ret;

    if (!is_server) {
        /* Client sends a message to the server using the stream API */
        request = ucp_stream_send_nb(ep, test_message, 1,
                                     ucp_dt_make_contig(TEST_STRING_LEN),
                                     stream_send_cb, 0);
        if (UCS_PTR_IS_ERR(request)) {
            fprintf(stderr, "unable to send UCX message\n");
            ret = -1;
            goto out;
        } else if (UCS_PTR_STATUS(request) != UCS_OK) {
            while (request->complete == 0) {
                ucp_worker_progress(ucp_worker);
            }
            ucp_request_free(request);
        }
    } else {
        /* Server receives a message from the client using the stream API */
        request = ucp_stream_recv_nb(ep, &recv_message, 1,
                                     ucp_dt_make_contig(TEST_STRING_LEN),
                                     stream_recv_cb, &length , 0);
        if (UCS_PTR_IS_ERR(request)) {
            fprintf(stderr, "unable to receive UCX message (%u)\n",
                    UCS_PTR_STATUS(request));
            ret = -1;
            goto out;
        } else {
            while (request->complete == 0) {
                ucp_worker_progress(ucp_worker);
            }
            ucp_request_free(request);
            printf("UCX data message was received\n");
        }
    }

    ret = verify_result(is_server, recv_message);

out:
    return ret;
}

/**
 * Close the given endpoint.
 * Currently closing the endpoint with UCP_EP_CLOSE_MODE_FORCE since we currently
 * cannot rely on both client and server to be present during the closing process.
 */
static void ep_close(ucp_worker_h ucp_worker, ucp_ep_h ep)
{
    ucs_status_t status;
    void *close_req;

    close_req = ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FORCE);
    if (UCS_PTR_IS_PTR(close_req)) {
        do {
            ucp_worker_progress(ucp_worker);
            status = ucp_request_check_status(close_req);
        } while (status == UCS_INPROGRESS);

        ucp_request_free(close_req);
    } else if (UCS_PTR_STATUS(close_req) != UCS_OK) {
        fprintf(stderr, "failed to close ep %p\n", (void*)ep);
    }
}

/**
 * A callback to be invoked by UCX in order to initialize the user's request.
 */
static void request_init(void *request)
{
    test_req_t *req = request;
    req->complete = 0;
}

/**
 * Check if we need to print a 'help' message to the user.
 */
static int check_if_print_help(int argc, char * const argv[])
{
    int c = 0;
    opterr = 0;

    while ((c = getopt(argc, argv, "h")) != -1) {
        switch (c) {
        case '?':
            if (isprint (optopt)) {
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            } else {
                fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
            }
        case 'h':
        default:
            fprintf(stderr, "Usage: ucp_client_server [parameters]\n");
            fprintf(stderr, "UCP client-server example utility\n");
            fprintf(stderr, "\nParameters are:\n");
            fprintf(stderr, " IP address of the server "
                    "(required for client and should be ignored for server)\n");
            fprintf(stderr, "\n");
            return -1;
        }
    }

    return 0;
}

int main(int argc, char **argv)
{
    ucx_server_ctx_t context;
    size_t length;
    char recv_message[TEST_STRING_LEN]= "";
    int is_server, ret;

    /* UCP objects */
    ucp_worker_params_t worker_params;
    ucp_context_h ucp_context;
    ucp_listener_h listener;
    ucp_worker_h ucp_worker;
    ucp_params_t ucp_params;
    ucp_config_t *config;
    ucs_status_t status;
    ucp_ep_h ep;

    memset(&ucp_params, 0, sizeof(ucp_params));
    memset(&worker_params, 0, sizeof(worker_params));

    ret = check_if_print_help(argc, argv);
    if (ret != 0) {
        goto err;
    }

    /* UCP initialization */
    status = ucp_config_read(NULL, NULL, &config);
    CHKERR_JUMP(status != UCS_OK, "ucp_config_read\n", err);

    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES     |
                            UCP_PARAM_FIELD_REQUEST_SIZE |
                            UCP_PARAM_FIELD_REQUEST_INIT;
    ucp_params.features   = UCP_FEATURE_STREAM;

    ucp_params.request_size    = sizeof(test_req_t);
    ucp_params.request_init    = request_init;

    status = ucp_init(&ucp_params, config, &ucp_context);
    ucp_config_release(config);
    CHKERR_JUMP(status != UCS_OK, "ucp_init\n", err);

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    status = ucp_worker_create(ucp_context, &worker_params, &ucp_worker);
    CHKERR_JUMP(status != UCS_OK, "ucp_worker_create\n", err_cleanup);

    /* Client-Server initialization */
    if (argc == 1) {
        /* Server side */
        is_server = 1;
        status = start_server(ucp_worker, &context, &listener);
        CHKERR_JUMP(status != UCS_OK, "start server\n", err_worker);
        ep = context.ep;
    } else {
        CHKERR_JUMP(argc != 2, "start the client.\n "
                    "usage: ucp_client_server <server-ip-address>",
                    err_worker);

        /* Client side */
        is_server = 0;
        status = start_client(ucp_worker, argv[1], &ep);
        CHKERR_JUMP(status != UCS_OK, "start client\n", err_worker);
    }

    /* Client-Server communication via Stream API */
    ret = send_recv_stream(ucp_worker, ep, is_server);


    /* Finalization or error flow */
err_listener:
    if (is_server) {
        ucp_listener_destroy(listener);
    }

err_ep:
    ep_close(ucp_worker, ep);

err_worker:
    ucp_worker_destroy(ucp_worker);

err_cleanup:
    ucp_cleanup(ucp_context);

err:
    return ret;
}
