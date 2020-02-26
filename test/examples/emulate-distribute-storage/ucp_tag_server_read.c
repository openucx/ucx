#include <ucp/api/ucp.h>

#include <string.h>    /* memset */
#include <arpa/inet.h> /* inet_addr */
#include <unistd.h>    /* getopt */
#include <stdlib.h>    /* atoi */
#include <string.h>
#include <sys/time.h>
#include <pthread.h>

/**
 * Default UCP listen port
 */
#define DEFAULT_PORT       13337
#define IP_STRING_LEN      50
#define PORT_STRING_LEN    8
#define TAG                0xCAFE

static uint16_t server_port = DEFAULT_PORT;

/**
 * Server's application context to be used in the user's connection request
 * callback.
 * It holds the server's listener and the handle to an incoming connection request.
 */
typedef struct ucx_server_ctx {
    volatile ucp_conn_request_h conn_request;
    ucp_listener_h              listener;
    ucp_worker_h                ucp_data_worker;
} ucx_server_ctx_t;

/**
 * Stream request context. Holds a value to indicate whether or not the
 * request is completed.
 */
typedef struct test_req {
    int complete;
} test_req_t;

/**
 * Callback function for tag recv which is called after finishing receiving
 * the message.
 * The recv information will appear in this function.
 */
static void tag_recv_cb(void *request, ucs_status_t status,
                        ucp_tag_recv_info_t *info)
{
    test_req_t *req = request;

    req->complete = 1;
#ifdef UCX_DEBUG
    printf("tag_recv_cb returned with status %d (%s), length: %lu, "
           "sender_tag: 0x%lX\n",
           status, ucs_status_string(status), info->length, info->sender_tag);
#endif
}

/**
 * The callback on the sending side, which is invoked after finishing sending
 * the message.
 */
static void send_cb(void *request, ucs_status_t status)
{
    test_req_t *req = request;

    req->complete = 1;
#ifdef UCX_DEBUG
    printf("send_cb returned with status %d (%s)\n",
           status, ucs_status_string(status));
#endif
}

/**
 * Error handling callback.
 */
static void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    fprintf(stderr, "error handling callback was invoked with status %d (%s)\n",
            status, ucs_status_string(status));
}

/**
 * Set an address for the server to listen on - INADDR_ANY on a well known port.
 */
void set_listen_addr(const char *address_str, struct sockaddr_in *listen_addr)
{
    /*
     * The server will listen on INADDR_ANY
    */
    memset(listen_addr, 0, sizeof(struct sockaddr_in));
    listen_addr->sin_family      = AF_INET;
    listen_addr->sin_addr.s_addr =
        address_str? inet_addr(address_str) : INADDR_ANY;
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
 * Progress the request until it completes. It is a non-blocking function.
 */
static ucs_status_t request_wait(ucp_worker_h ucp_worker, test_req_t *request)
{
    ucs_status_t status;

    /**
     * if operation was completed immediately
    */
    if (request == NULL) {
        return UCS_OK;
    }

    if (UCS_PTR_IS_ERR(request)) {
        return UCS_PTR_STATUS(request);
    }

    while (request->complete == 0) {
        ucp_worker_progress(ucp_worker);
    }
    status = ucp_request_check_status(request);

    /* This request may be reused so initialize it for next time */
    request->complete = 0;
    ucp_request_free(request);

    return status;
}

#define IO_RESPONSE    "ioresponse"

/**
 * Send and receive a message using the Tag-Matching API.
 * The client sends a message to the server and waits until the send it
 * completed.
 * The server receives a message from the client and waits for its
 * completion.
 */
static int send_recv_tag(ucp_worker_h ucp_worker, ucp_ep_h ep)
{
    char recv_message[256 * 1024] = "";
    test_req_t *request;
    ucs_status_t status;
    size_t length;

    /**
     * Server receives a message from the client using the Tag-Matching API
     * The client requests the size of message.
     */
    request = ucp_tag_recv_nb(ucp_worker, &length, sizeof(size_t),
                              ucp_dt_make_contig(1),
                              TAG, 0, tag_recv_cb);

    status = request_wait(ucp_worker, request);
    if (status != UCS_OK){
        fprintf(stderr, "unable to receive UCX message (%s)\n",
                ucs_status_string(status));
        return -1;
    }

    /**
     * The server sends the message to the client after finishing receiving
     * the requested size. The size of the message is equal to the requested
     * size.
     */
    request = ucp_tag_send_nb(ep, recv_message, length,
                              ucp_dt_make_contig(1), TAG,
                              send_cb);

    status = request_wait(ucp_worker, request);
    if (status != UCS_OK){
        fprintf(stderr, "unable to send UCX message (%s)\n",
                ucs_status_string(status));
        return -1;
    }

    /**
     *  After finishing sending message, send ioresponse message to
     * the client to indicate the end of sending
     */
    snprintf(recv_message, sizeof(IO_RESPONSE), IO_RESPONSE);
    request = ucp_tag_send_nb(ep, recv_message, strlen(IO_RESPONSE),
                              ucp_dt_make_contig(1), TAG,
                              send_cb);

    status = request_wait(ucp_worker, request);
    if (status != UCS_OK){
        fprintf(stderr, "unable to send UCX message (%s)\n",
                ucs_status_string(status));
        return -1;
    }

    return 0;
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
    fprintf(stderr, "Usage: ucp_tag_server_read [parameters]\n");
    fprintf(stderr, "UCP server example utility\n");
    fprintf(stderr, "\nParameters are:\n");
    fprintf(stderr, " -l Set IP address where server listens "
                    "(If not specified, server uses INADDR_ANY; "
                    "Irrelevant at client)\n");
    fprintf(stderr, " -p Port number to listen to (default = %d). "
                    "0 on the server side means select a random port and print it\n",
                    DEFAULT_PORT);
    fprintf(stderr, " -f the number of the requests in flight");
    fprintf(stderr, "\n");
}

/**
 * Parse the command line arguments.
 */
static int parse_cmd(int argc, char *const argv[], char **listen_addr,
                     int *flight_requests)
{
    int c = 0;
    int port;

    opterr = 0;

    while ((c = getopt(argc, argv, "l:p:f:")) != -1) {
        switch (c) {
        case 'l':
            *listen_addr = optarg;
            break;
        case 'f':
            *flight_requests = atoi(optarg);

            if ((*flight_requests < 0) || (*flight_requests > 16384))
                return -1;

            break;
        case 'p':
            port = atoi(optarg);
            if ((port < 0) || (port > UINT16_MAX)) {
                fprintf(stderr, "Wrong server port number %d\n", port);
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

static char* sockaddr_get_ip_str(const struct sockaddr_storage *sock_addr,
                                 char *ip_str, size_t max_size)
{
    struct sockaddr_in  addr_in;
    struct sockaddr_in6 addr_in6;

    switch (sock_addr->ss_family) {
    case AF_INET:
        memcpy(&addr_in, sock_addr, sizeof(struct sockaddr_in));
        inet_ntop(AF_INET, &addr_in.sin_addr, ip_str, max_size);
        return ip_str;
    case AF_INET6:
        memcpy(&addr_in6, sock_addr, sizeof(struct sockaddr_in6));
        inet_ntop(AF_INET6, &addr_in6.sin6_addr, ip_str, max_size);
        return ip_str;
    default:
        return "Invalid address family";
    }
}

static char* sockaddr_get_port_str(const struct sockaddr_storage *sock_addr,
                                   char *port_str, size_t max_size)
{
    struct sockaddr_in  addr_in;
    struct sockaddr_in6 addr_in6;

    switch (sock_addr->ss_family) {
    case AF_INET:
        memcpy(&addr_in, sock_addr, sizeof(struct sockaddr_in));
        snprintf(port_str, max_size, "%d", ntohs(addr_in.sin_port));
        return port_str;
    case AF_INET6:
        memcpy(&addr_in6, sock_addr, sizeof(struct sockaddr_in6));
        snprintf(port_str, max_size, "%d", ntohs(addr_in6.sin6_port));
        return port_str;
    default:
        return "Invalid address family";
    }
}

static int client_server_communication(ucp_worker_h worker, ucp_ep_h ep)
{
    /* Client-Server communication via Tag-Matching API */
    return send_recv_tag(worker, ep);
}

/**
 * Create a ucp worker on the given ucp context.
 */
static int init_worker(ucp_context_h ucp_context, ucp_worker_h *ucp_worker)
{
    ucp_worker_params_t worker_params;
    ucs_status_t status;
    int ret = 0;

    memset(&worker_params, 0, sizeof(worker_params));

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    status = ucp_worker_create(ucp_context, &worker_params, ucp_worker);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_worker_create (%s)\n",
                ucs_status_string(status));
        ret = -1;
    }

    return ret;
}

static ucs_status_t server_create_ep(ucp_worker_h data_worker,
                                     ucp_conn_request_h conn_request,
                                     ucp_ep_h *server_ep);

#define    MAX_THREAD_NUM    32

ucx_server_ctx_t g_context[MAX_THREAD_NUM];
int              g_count = 0;
ucp_worker_h     ucp_data_worker[MAX_THREAD_NUM];

/**
 * The thread work function which is called by the callback function
 * server_conn_handle_cb. When a request comes, the callback function
 * will create a thread to handle this request. Then the server continues
 * to listen on the port.
 */
void *handle_client_conn_worker(void *arg)
{
    ucx_server_ctx_t *context = arg;
    ucp_ep_h         server_ep;
    ucs_status_t     status;
    int              ret;

    /**
     * Detach the thread. Then after the thread is complete,
     * all the resource related with the thread will be released
     * to the system.
    */
    ret = pthread_detach(pthread_self());
    if (ret) {
        fprintf(stderr, "line:%d, pthread_detach error:%d\n", __LINE__, ret);
        return NULL;
    }

    status = server_create_ep(context->ucp_data_worker, context->conn_request,
                              &server_ep);
    if (status != UCS_OK) {
        return NULL;
    }

    /* The main function to handle the communications. */
    client_server_communication(context->ucp_data_worker, server_ep);

    /* Close the endpoint to the peer */
    ep_close(context->ucp_data_worker, server_ep);

    return NULL;
}

/**
 * The callback on the server side which is invoked upon receiving a connection
 * request from the client.
 */
static void server_conn_handle_cb(ucp_conn_request_h conn_request, void *arg)
{
    ucx_server_ctx_t *context = arg;
    ucs_status_t status;

    if (context->conn_request == NULL) {
        pthread_t    ntid;
        int          ret;

        context->conn_request = conn_request;
        context->ucp_data_worker = ucp_data_worker[g_count];
        g_context[g_count] = *context;
        ret = pthread_create(&ntid, NULL, handle_client_conn_worker,
                             &g_context[g_count]);
        if (ret != 0)
            fprintf(stderr, "can't create thread: %s\n", strerror(ret));

        context->conn_request = NULL;

        g_count = (g_count + 1) % MAX_THREAD_NUM;
    } else {
        /* The server is already handling a connection request from a client,
         * reject this new one */
        printf("Rejecting a connection request. "
               "Only one client at a time is supported.\n");
        status = ucp_listener_reject(context->listener, conn_request);
        if (status != UCS_OK) {
            fprintf(stderr, "server failed to reject a connection request: (%s)\n",
                    ucs_status_string(status));
        }
    }
}

static ucs_status_t server_create_ep(ucp_worker_h data_worker,
                                     ucp_conn_request_h conn_request,
                                     ucp_ep_h *server_ep)
{
    ucp_ep_params_t ep_params;
    ucs_status_t    status;

    /* Server creates an ep to the client on the data worker.
     * This is not the worker the listener was created on.
     * The client side should have initiated the connection, leading
     * to this ep's creation */
    ep_params.field_mask      = UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                UCP_EP_PARAM_FIELD_CONN_REQUEST;
    ep_params.conn_request    = conn_request;
    ep_params.err_handler.cb  = err_cb;
    ep_params.err_handler.arg = NULL;

    status = ucp_ep_create(data_worker, &ep_params, server_ep);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to create an endpoint on the server: (%s)\n",
                ucs_status_string(status));
    }

    return status;
}

/**
 * Initialize the server side. The server starts listening on the set address.
 */
static ucs_status_t start_server(ucp_worker_h ucp_worker,
                                 ucx_server_ctx_t *context,
                                 ucp_listener_h *listener_p, const char *ip)
{
    struct sockaddr_in listen_addr;
    ucp_listener_params_t params;
    ucp_listener_attr_t attr;
    ucs_status_t status;
    char ip_str[IP_STRING_LEN];
    char port_str[PORT_STRING_LEN];

    set_listen_addr(ip, &listen_addr);

    params.field_mask         = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    params.sockaddr.addr      = (const struct sockaddr*)&listen_addr;
    params.sockaddr.addrlen   = sizeof(listen_addr);
    params.conn_handler.cb    = server_conn_handle_cb;
    params.conn_handler.arg   = context;

    /* Create a listener on the server side to listen on the given address.*/
    status = ucp_listener_create(ucp_worker, &params, listener_p);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to listen (%s)\n", ucs_status_string(status));
        goto out;
    }

    /* Query the created listener to get the port it is listening on. */
    attr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
    status = ucp_listener_query(*listener_p, &attr);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to query the listener (%s)\n",
                ucs_status_string(status));
        ucp_listener_destroy(*listener_p);
        goto out;
    }

    fprintf(stderr, "server is listening on IP %s port %s\n",
            sockaddr_get_ip_str(&attr.sockaddr, ip_str, IP_STRING_LEN),
            sockaddr_get_port_str(&attr.sockaddr, port_str, PORT_STRING_LEN));

    printf("Waiting for connection...\n");

out:
    return status;
}

static int run_server(ucp_context_h ucp_context, ucp_worker_h ucp_worker,
                      char *listen_addr)
{
    ucx_server_ctx_t context;
    ucs_status_t     status;
    int              ret, i, j;

    for (i=0; i<MAX_THREAD_NUM; i++) {
        /* Create a data worker (to be used for data exchange between the server
         * and the client after the connection between them was established) */
        ret = init_worker(ucp_context, &ucp_data_worker[i]);
        if (ret != 0) {
            for (j=0; j<i; j++)
                ucp_worker_destroy(ucp_data_worker[j]);

            goto err;
        }
    }

    /* Initialiaze the server's context. */
    context.conn_request = NULL;

    /* Create a listener on the worker created at first. The 'connection
     * worker' - used for connection establishment between client and server.
     * This listener will stay open for listening to incoming connection
     * requests from the client */
    status = start_server(ucp_worker, &context, &context.listener, listen_addr);
    if (status != UCS_OK) {
        ret = -1;
        goto err_worker;
    }

    /* Server is always up listening */
    while (1) {
        /* Wait for the server to receive a connection request from the client.
         * If there are multiple clients for which the server's connection request
         * callback is involked, i.e. several clients are trying to connect in
         * parallel, the server will handle only the first one and reject the rest */
            ucp_worker_progress(ucp_worker);
    }

    ucp_listener_destroy(context.listener);

err_worker:
    for (j=0; j<MAX_THREAD_NUM; j++)
        ucp_worker_destroy(ucp_data_worker[j]);

err:
    return ret;
}

/**
 * Initialize the UCP context and worker.
 */
static int init_context(ucp_context_h *ucp_context, ucp_worker_h *ucp_worker)
{
    /* UCP objects */
    ucp_params_t ucp_params;
    ucs_status_t status;
    int ret = 0;

    memset(&ucp_params, 0, sizeof(ucp_params));

    /* UCP initialization */
    ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES     |
                              UCP_PARAM_FIELD_REQUEST_SIZE |
                              UCP_PARAM_FIELD_REQUEST_INIT;
    ucp_params.features     = UCP_FEATURE_TAG;
    ucp_params.request_size = sizeof(test_req_t);
    ucp_params.request_init = request_init;

    status = ucp_init(&ucp_params, NULL, ucp_context);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_init (%s)\n", ucs_status_string(status));
        ret = -1;
        goto err;
    }

    ret = init_worker(*ucp_context, ucp_worker);
    if (ret != 0) {
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
    char *listen_addr = NULL;
    int ret;
    int flight_requests = 0;

    /* UCP objects */
    ucp_context_h ucp_context;
    ucp_worker_h  ucp_worker;

    ret = parse_cmd(argc, argv, &listen_addr, &flight_requests);
    if (ret != 0) {
        goto err;
    }

    /* Initialize the UCX required objects */
    ret = init_context(&ucp_context, &ucp_worker);
    if (ret != 0) {
        goto err;
    }

    /* Server initialization */
    ret = run_server(ucp_context, ucp_worker, listen_addr);

    ucp_worker_destroy(ucp_worker);
    ucp_cleanup(ucp_context);

err:
    return ret;
}
