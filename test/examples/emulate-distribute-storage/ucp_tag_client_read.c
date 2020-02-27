#include <ucp/api/ucp.h>

#include <string.h>    /* memset */
#include <arpa/inet.h> /* inet_addr */
#include <unistd.h>    /* getopt */
#include <stdlib.h>    /* atoi */
#include <sys/time.h>

#define DEFAULT_PORT       13337
#define IP_STRING_LEN      50
#define PORT_STRING_LEN    8
#define TAG                0xCAFE

static uint16_t server_port = DEFAULT_PORT;

/**
 * Stream request context. Holds a value to indicate whether or not the
 * request is completed.
 */
typedef struct test_req {
    int complete;
} test_req_t;

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
    printf("error handling callback was invoked with status %d (%s)\n",
           status, ucs_status_string(status));
}

/**
 * Set an address for the server to listen on - INADDR_ANY on a well known port.
 */
void set_listen_addr(const char *address_str, struct sockaddr_in *listen_addr)
{
    /* The server will listen on INADDR_ANY */
    memset(listen_addr, 0, sizeof(struct sockaddr_in));
    listen_addr->sin_family      = AF_INET;
    listen_addr->sin_addr.s_addr = (address_str) ? inet_addr(address_str) : INADDR_ANY;
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
    ep_params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS       |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR   |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb   = err_cb;
    ep_params.err_handler.arg  = NULL;
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
 * Progress the request until it completes.
 */
static ucs_status_t request_wait(ucp_worker_h ucp_worker, test_req_t *request)
{
    ucs_status_t status;

    /*  if operation was completed immediately */
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

#define IO_RESPONSE        "ioresponse"
#define IO_RESPONSE_LEN    sizeof(IO_RESPONSE)

/**
 * Send and receive a message using the Tag-Matching API.
 * The client sends a message to the server and waits until the send it
 * completed. The server receives a message from the client and waits
 * for its completion.
 */
static int send_recv_tag(ucp_worker_h ucp_worker, ucp_ep_h ep)
{
    char *recv_message = NULL;
    test_req_t *request;
    size_t length = 255;
    ucs_status_t status;
#ifdef UCX_DEBUG
    struct timeval tv_begin, tv_end;
    struct timeval tv_send, tv_recv;

    gettimeofday(&tv_begin, NULL);
    gettimeofday(&tv_send, NULL);
#endif

    /**
     * Client sends the length of message to the server using the Tag-Matching
     * API
    */
    request = ucp_tag_send_nb(ep, &length, sizeof(size_t),
                              ucp_dt_make_contig(1), TAG,
                              send_cb);

    status = request_wait(ucp_worker, request);
    if (status != UCS_OK){
        fprintf(stderr, "unable to send UCX message (%s)\n",
                ucs_status_string(status));
        return -1;
    }

    /* recv the message with the request length */
    recv_message = malloc(length + 1);
    if (!recv_message) {
        fprintf(stderr, "line:%d, alloc memory fail\n", __LINE__);
        return -1;
    }
    request = ucp_tag_recv_nb(ucp_worker, recv_message, length,
                              ucp_dt_make_contig(1),
                              TAG, 0, tag_recv_cb);

    status = request_wait(ucp_worker, request);
    if (status != UCS_OK){
        fprintf(stderr, "unable to receive UCX message (%s)\n",
                ucs_status_string(status));
        return -1;
    }

    free(recv_message);

#ifdef UCX_DEBUG
    gettimeofday(&tv_recv, NULL);
    printf("bandwidth:%lu\n", (length * 8 * 1000000) / (tv_recv.tv_sec * 1000000 + tv_recv.tv_usec - tv_send.tv_sec * 1000000 - tv_send.tv_usec));
#endif

    /* recv ioresponse */
    recv_message = malloc(IO_RESPONSE_LEN);
    if (!recv_message) {
        fprintf(stderr, "line:%d, alloc memory fail\n", __LINE__);
        return -1;
    }

    request = ucp_tag_recv_nb(ucp_worker, recv_message, strlen(IO_RESPONSE),
                              ucp_dt_make_contig(1),
                              TAG, 0, tag_recv_cb);

    status = request_wait(ucp_worker, request);
    if (status != UCS_OK){
        fprintf(stderr, "unable to receive UCX message (%s)\n",
                ucs_status_string(status));
        return -1;
    }

#ifdef UCX_DEBUG
    gettimeofday(&tv_end, NULL);
    printf("line:%d, the diff is %lu\n", __LINE__, tv_end.tv_sec * 1000000 + tv_end.tv_usec - tv_begin.tv_sec * 1000000 - tv_begin.tv_usec);
    printf("recv_message:%s\n", recv_message);
#endif

    free(recv_message);
    recv_message = NULL;

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
    fprintf(stderr, "Usage: ucp_tag_client_read [parameters]\n");
    fprintf(stderr, "UCP tag_client_read example utility\n");
    fprintf(stderr, "\nParameters are:\n");
    fprintf(stderr, " -a Set IP address of the server "
                    "(required for client and should not be specified "
                    "for the server)\n");
    fprintf(stderr, " -p Port number to connect to (default = %d). ",
                    DEFAULT_PORT);
    fprintf(stderr, " -f the number of the requests in flight");
    fprintf(stderr, "\n");
}

/**
 * Parse the command line arguments.
 */
static int parse_cmd(int argc, char *const argv[], char **server_addr,
                     int *flight_requests)
{
    int c = 0;
    int port;

    opterr = 0;

    while ((c = getopt(argc, argv, "a:l:p:")) != -1) {
        switch (c) {
        case 'a':
            *server_addr = optarg;
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

static int client_server_communication(ucp_worker_h worker, ucp_ep_h ep)
{
    int ret;

    /* Client-Server communication via Tag-Matching API */
    ret = send_recv_tag(worker, ep);

    /* Close the endpoint to the peer */
    ep_close(worker, ep);

    return ret;
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
        fprintf(stderr, "failed to ucp_worker_create (%s)\n", ucs_status_string(status));
        ret = -1;
    }

    return ret;
}

static int run_client(ucp_context_h ucp_context, ucp_worker_h ucp_worker,
                      char *server_addr)
{
    ucp_ep_h     client_ep;
    ucs_status_t status;
    int          ret;

    status = start_client(ucp_worker, server_addr, &client_ep);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to start client (%s)\n", ucs_status_string(status));
        ret = -1;
        goto out;
    }

    ret = client_server_communication(ucp_worker, client_ep);

out:
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
    char *server_addr = NULL;
    int ret;
    int flight_requests = 0;

    /* UCP objects */
    ucp_context_h ucp_context;
    ucp_worker_h  ucp_worker;

    ret = parse_cmd(argc, argv, &server_addr, &flight_requests);
    if (ret != 0) {
        goto err;
    }

#ifdef UCX_DEBUG
    printf("flight_requests:%d\n", flight_requests);
#endif

    /* Initialize the UCX required objects */
    ret = init_context(&ucp_context, &ucp_worker);
    if (ret != 0) {
        goto err;
    }

    /* Client-Server initialization */
    if (server_addr != NULL) {
       /* Client side */
        ret = run_client(ucp_context, ucp_worker, server_addr);
    }

    ucp_worker_destroy(ucp_worker);
    ucp_cleanup(ucp_context);
err:
    return ret;
}
