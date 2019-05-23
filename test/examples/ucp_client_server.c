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
 *    ./ucp_client_server
 *
 * Client side:
 *
 *    ./ucp_client_server -a <server-ip>
 *
 * Notes:
 *
 *    - The server will listen to incoming connection requests on INADDR_ANY.
 *    - The client needs to pass the IP address of the server side to connect to
 *      as an argument to the test.
 *    - Currently, the passed IP needs to be an IPoIB or a RoCE address.
 *    - The port which the server side would listen on can be modified with the
 *      '-p' option and should be used on both sides. The default port to use is
 *      13337.
 */

#include <ucp/api/ucp.h>

#include <string.h>    /* memset */
#include <arpa/inet.h> /* inet_addr */
#include <unistd.h>    /* getopt */
#include <stdlib.h>    /* atoi */


const char test_message[] = "UCX Client-Server Hello World";
static uint16_t server_port = 13337;

#define TEST_STRING_LEN sizeof(test_message)


/**
 * Server context to be used in the user's accept callback.
 * It holds the server's endpoint which will be created upon accepting a
 * connection request from the client.
 */
typedef struct ucx_server_ctx {
    ucp_ep_h     ep;
} ucx_server_ctx_t;


/**
 * Stream request context. Holds a value to indicate whether or not the
 * request is completed.
 */
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

    printf("stream_recv_cb returned with status %d (%s), length: %lu\n",
           status, ucs_status_string(status), length);
}

/**
 * The callback on the sending side, which is invoked after finishing sending
 * the stream message.
 */
static void stream_send_cb(void *request, ucs_status_t status)
{
    test_req_t *req = request;

    req->complete = 1;

    printf("stream_send_cb returned with status %d (%s)\n",
           status, ucs_status_string(status));
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
 * Set an address for the server to listen on - INADDR_ANY on a well known port.
 */
void set_listen_addr(struct sockaddr_in *listen_addr)
{
    /* The server will listen on INADDR_ANY */
    memset(listen_addr, 0, sizeof(struct sockaddr_in));
    listen_addr->sin_family      = AF_INET;
    listen_addr->sin_addr.s_addr = INADDR_ANY;
    listen_addr->sin_port        = htons(server_port);
}

/**
 * Set an address to connect to. A given IP address on a well known port.
 */
void set_connect_addr(const char *address_str, struct sockaddr_in *connect_addr)
{
    memset(connect_addr, 0, sizeof(struct sockaddr_in));
    connect_addr->sin_family      = AF_INET;
    connect_addr->sin_addr.s_addr = inet_addr(address_str);
    connect_addr->sin_port        = htons(server_port);
}

/**
 * Initialize the server side. The server starts listening on the set address
 * and waits for its connected endpoint to be created.
 */
static int start_server(ucp_worker_h ucp_worker, ucx_server_ctx_t *context,
                        ucp_listener_h *listener)
{
    struct sockaddr_in listen_addr;
    ucp_listener_params_t params;
    ucs_status_t status;

    set_listen_addr(&listen_addr);

    params.field_mask         = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER;
    params.sockaddr.addr      = (const struct sockaddr*)&listen_addr;
    params.sockaddr.addrlen   = sizeof(listen_addr);
    params.accept_handler.cb  = server_accept_cb;
    params.accept_handler.arg = context;

    /* Create a listener on the server side to listen on the given address.*/
    status = ucp_listener_create(ucp_worker, &params, listener);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to listen (%s)\n", ucs_status_string(status));
    }

    return status;
}

/**
 * Initialize the client side. Create an endpoint from the client side to be
 * connected to the remote server (to the given IP).
 */
static int start_client(ucp_worker_h ucp_worker, const char *ip,
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
        fprintf(stderr, "failed to connect to %s (%s)\n", ip, ucs_status_string(status));
    }

    return status;
}

/**
 * Print the received message on the server side or the sent data on the client
 * side.
 */
static void print_result(int is_server, char *recv_message)
{
    if (is_server) {
        printf("\n\n----- UCP TEST SUCCESS -------\n\n");
        printf("%s", recv_message);
        printf("\n\n------------------------------\n\n");
    } else {
        printf("\n\n-----------------------------------------\n\n");
        printf("Client sent message: \n%s.\nlength: %ld\n",
               test_message, TEST_STRING_LEN);
        printf("\n-----------------------------------------\n\n");
    }
}

static void request_wait(ucp_worker_h ucp_worker, test_req_t *request)
{
    while (request->complete == 0) {
        ucp_worker_progress(ucp_worker);
    }

    /* This request may be reused so initialize it for next time */
    request->complete = 0;
    ucp_request_free(request);
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
    int ret = 0;

    if (!is_server) {
        /* Client sends a message to the server using the stream API */
        request = ucp_stream_send_nb(ep, test_message, 1,
                                     ucp_dt_make_contig(TEST_STRING_LEN),
                                     stream_send_cb, 0);
        if (UCS_PTR_IS_ERR(request)) {
            fprintf(stderr, "unable to send UCX message (%s)\n",
                    ucs_status_string(UCS_PTR_STATUS(request)));
            ret = -1;
            goto out;
        } else if (UCS_PTR_STATUS(request) != UCS_OK) {
            request_wait(ucp_worker, request);
        }
    } else {
        /* Server receives a message from the client using the stream API */
        request = ucp_stream_recv_nb(ep, &recv_message, 1,
                                     ucp_dt_make_contig(TEST_STRING_LEN),
                                     stream_recv_cb, &length , 0);
        if (UCS_PTR_IS_ERR(request)) {
            fprintf(stderr, "unable to receive UCX message (%s)\n",
                    ucs_status_string(UCS_PTR_STATUS(request)));
            ret = -1;
            goto out;
        } else {
            request_wait(ucp_worker, request);
            printf("UCX data message was received\n");
        }
    }

    print_result(is_server, recv_message);

out:
    return ret;
}

/**
 * Close the given endpoint.
 * Currently closing the endpoint with UCP_EP_CLOSE_MODE_FORCE since we currently
 * cannot rely on the client side to be present during the server's endpoint
 * closing process.
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
 * Print this application's usage help message.
 */
static void usage()
{
    fprintf(stderr, "Usage: ucp_client_server [parameters]\n");
    fprintf(stderr, "UCP client-server example utility\n");
    fprintf(stderr, "\nParameters are:\n");
    fprintf(stderr, " -a Set IP address of the server "
                    "(required for client and should not be specified "
                    "for the server)\n");
    fprintf(stderr, " -p Set alternative server port (default:13337)\n");
    fprintf(stderr, "\n");
}

/**
 * Parse the command line arguments.
 */
static int parse_cmd(int argc, char *const argv[], char **server_addr)
{
    int c = 0;
    int port;

    opterr = 0;

    while ((c = getopt(argc, argv, "a:p:")) != -1) {
        switch (c) {
        case 'a':
            *server_addr = optarg;
            break;
        case 'p':
            port = atoi(optarg);
            if ((port < 0) || (port > UINT16_MAX)) {
                fprintf(stderr, "Wrong server port number %d\n", server_port);
                return -1;
            }
            server_port = port;
            break;
        default:
            usage();
            return -1;
        }
    }

    return 0;
}

/**
 * Initialize the UCP context and worker.
 */
static int init_context(ucp_context_h *ucp_context, ucp_worker_h *ucp_worker)
{
    /* UCP objects */
    ucp_worker_params_t worker_params;
    ucp_params_t ucp_params;
    ucs_status_t status;
    int ret = 0;

    memset(&ucp_params, 0, sizeof(ucp_params));
    memset(&worker_params, 0, sizeof(worker_params));

    /* UCP initialization */
    ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES     |
                              UCP_PARAM_FIELD_REQUEST_SIZE |
                              UCP_PARAM_FIELD_REQUEST_INIT;
    ucp_params.features     = UCP_FEATURE_STREAM;

    ucp_params.request_size = sizeof(test_req_t);
    ucp_params.request_init = request_init;

    status = ucp_init(&ucp_params, NULL, ucp_context);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_init (%s)\n", ucs_status_string(status));
        ret = -1;
        goto err;
    }

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    status = ucp_worker_create(*ucp_context, &worker_params, ucp_worker);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_worker_create (%s)\n", ucs_status_string(status));
        ret = -1;
        goto err_cleanup;
    }

    return ret;

err_cleanup:
    ucp_cleanup(*ucp_context);

err:
    return ret;
}


int main(int argc, char **argv)
{
    ucx_server_ctx_t context;
    int is_server, ret;
    char *server_addr = NULL;

    /* UCP objects */
    ucp_context_h ucp_context;
    ucp_listener_h listener;
    ucp_worker_h ucp_worker;
    ucs_status_t status;
    ucp_ep_h ep;

    ret = parse_cmd(argc, argv, &server_addr);
    if (ret != 0) {
        goto err;
    }

    /* Initialize the UCX required objects */
    ret = init_context(&ucp_context, &ucp_worker);
    if (ret != 0) {
        goto err;
    }

    /* Client-Server initialization */
    if (server_addr == NULL) {
        /* Server side */
        is_server = 1;

        /* Initialize the server's endpoint to NULL. Once the server's endpoint
         * is created, this field will have a valid value. */
        context.ep = NULL;

        status = start_server(ucp_worker, &context, &listener);
        if (status != UCS_OK) {
            fprintf(stderr, "failed to start server\n");
            goto err_worker;
        }

        /* Server is always up */
        printf("Waiting for connection...\n");
        while (1) {
            /* Wait for the server's callback to set the context->ep field, thus
             * indicating that the server's endpoint was created and is ready to
             * be used. The client side should initiate the connection, leading
             * to this ep's creation */
            if (context.ep == NULL) {
                ucp_worker_progress(ucp_worker);
            } else {
                /* Client-Server communication via Stream API */
                send_recv_stream(ucp_worker, context.ep, is_server);

                /* Close the endpoint to the client */
                ep_close(ucp_worker, context.ep);

                /* Initialize server's endpoint for the next connection with a new
                 * client */
                context.ep = NULL;
                printf("Waiting for connection...\n");
            };
        }
    } else {
        /* Client side */
        is_server = 0;
        status = start_client(ucp_worker, server_addr, &ep);
        if (status != UCS_OK) {
            fprintf(stderr, "failed to start client\n");
            goto err_worker;
        }

        /* Client-Server communication via Stream API */
        ret = send_recv_stream(ucp_worker, ep, is_server);

        /* Close the endpoint to the server */
        ep_close(ucp_worker, ep);
    }

err_worker:
    ucp_worker_destroy(ucp_worker);

    ucp_cleanup(ucp_context);

err:
    return ret;
}
