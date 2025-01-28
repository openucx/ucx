/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018. ALL RIGHTS RESERVED.
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

#include "hello_world_util.h"
#include "ucp_util.h"

#include <ucp/api/ucp.h>

#include <string.h>    /* memset */
#include <arpa/inet.h> /* inet_addr */
#include <unistd.h>    /* getopt */
#include <stdlib.h>    /* atoi */

#define DEFAULT_PORT           13337
#define IP_STRING_LEN          50
#define PORT_STRING_LEN        8
#define TAG                    0xCAFE
#define COMM_TYPE_DEFAULT      "STREAM"
#define PRINT_INTERVAL         2000
#define DEFAULT_NUM_ITERATIONS 1
#define TEST_AM_ID             0
#define MAX_DEV_COUNT          16


static long test_string_length = 16;
static long iov_cnt            = 1;
static uint16_t server_port    = DEFAULT_PORT;
static sa_family_t ai_family   = AF_INET;
static int num_iterations      = DEFAULT_NUM_ITERATIONS;
static int connection_closed   = 1;


typedef enum {
    CLIENT_SERVER_SEND_RECV_STREAM  = UCS_BIT(0),
    CLIENT_SERVER_SEND_RECV_TAG     = UCS_BIT(1),
    CLIENT_SERVER_SEND_RECV_AM      = UCS_BIT(2),
    CLIENT_SERVER_SEND_RECV_DEFAULT = CLIENT_SERVER_SEND_RECV_STREAM
} send_recv_type_t;


/**
 * Server's application context to be used in the user's connection request
 * callback.
 * It holds the server's listener and the handle to an incoming connection request.
 */
typedef struct ucx_server_ctx {
    volatile ucp_conn_request_h conn_request;
    ucp_listener_h              listener;
    uint64_t                    client_id;
} ucx_server_ctx_t;


/**
 * Stream request context. Holds a value to indicate whether or not the
 * request is completed.
 */
typedef struct test_req {
    int complete;
} test_req_t;


/**
 * Descriptor of the data received with AM API.
 */
static struct {
    volatile int complete;
    int          is_rndv;
    void         *desc;
    void         *recv_buf;
} am_data_desc = {0, 0, NULL, NULL};


/**
 * GPU-specific context
 */
typedef struct dev_ucp_ctx {
    ucp_context_h   ucp_context;
    ucp_worker_h    ucp_context;
    ucp_ep_h        ucp_eps[MAX_DEV_COUNT];
    size_t          ep_count;
} dev_ucp_ctx_t;


/**
 * Print this application's usage help message.
 */
static void usage(void);

static void buffer_free(ucp_dt_iov_t *iov, size_t iov_size)
{
    size_t idx;

    for (idx = 0; idx < iov_size; idx++) {
        mem_type_free(iov[idx].buffer);
    }
}

static int buffer_malloc(ucp_dt_iov_t *iov)
{
    size_t idx;

    for (idx = 0; idx < iov_cnt; idx++) {
        iov[idx].length = test_string_length;
        iov[idx].buffer = mem_type_malloc(iov[idx].length);
        if (iov[idx].buffer == NULL) {
            buffer_free(iov, idx);
            return -1;
        }
    }

    return 0;
}

int fill_buffer(ucp_dt_iov_t *iov)
{
    int ret = 0;
    size_t idx;

    for (idx = 0; idx < iov_cnt; idx++) {
        ret = generate_test_string(iov[idx].buffer, iov[idx].length);
        if (ret != 0) {
            break;
        }
    }
    CHKERR_ACTION(ret != 0, "generate test string", return -1;);
    return 0;
}

static void common_cb(void *user_data, const char *type_str)
{
    test_req_t *ctx;

    if (user_data == NULL) {
        fprintf(stderr, "user_data passed to %s mustn't be NULL\n", type_str);
        return;
    }

    ctx           = user_data;
    ctx->complete = 1;
}

static void tag_recv_cb(void *request, ucs_status_t status,
                        const ucp_tag_recv_info_t *info, void *user_data)
{
    common_cb(user_data, "tag_recv_cb");
}

/**
 * The callback on the receiving side, which is invoked upon receiving the
 * stream message.
 */
static void stream_recv_cb(void *request, ucs_status_t status, size_t length,
                           void *user_data)
{
    common_cb(user_data, "stream_recv_cb");
}

/**
 * The callback on the receiving side, which is invoked upon receiving the
 * active message.
 */
static void am_recv_cb(void *request, ucs_status_t status, size_t length,
                       void *user_data)
{
    common_cb(user_data, "am_recv_cb");
}

/**
 * The callback on the sending side, which is invoked after finishing sending
 * the message.
 */
static void send_cb(void *request, ucs_status_t status, void *user_data)
{
    common_cb(user_data, "send_cb");
}

/**
 * Error handling callback.
 */
static void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    printf("error handling callback was invoked with status %d (%s)\n",
           status, ucs_status_string(status));
    connection_closed = 1;
}

/**
 * Set an address for the server to listen on - INADDR_ANY on a well known port.
 */
void set_sock_addr(const char *address_str, struct sockaddr_storage *saddr)
{
    struct sockaddr_in *sa_in;
    struct sockaddr_in6 *sa_in6;

    /* The server will listen on INADDR_ANY */
    memset(saddr, 0, sizeof(*saddr));

    switch (ai_family) {
    case AF_INET:
        sa_in = (struct sockaddr_in*)saddr;
        if (address_str != NULL) {
            inet_pton(AF_INET, address_str, &sa_in->sin_addr);
        } else {
            sa_in->sin_addr.s_addr = INADDR_ANY;
        }
        sa_in->sin_family = AF_INET;
        sa_in->sin_port   = htons(server_port);
        break;
    case AF_INET6:
        sa_in6 = (struct sockaddr_in6*)saddr;
        if (address_str != NULL) {
            inet_pton(AF_INET6, address_str, &sa_in6->sin6_addr);
        } else {
            sa_in6->sin6_addr = in6addr_any;
        }
        sa_in6->sin6_family = AF_INET6;
        sa_in6->sin6_port   = htons(server_port);
        break;
    default:
        fprintf(stderr, "Invalid address family");
        break;
    }
}

/**
 * Create an endpoint from each client GPU-specific
 * worker to each remote server GPU-specific worker (to the given IP).
 */
static int client_create_eps(dev_ucp_ctx_t *dev_ucp_contexts, int dev_count,
                             const char *address_str)
{
    /* Client must send one separate connection request to the server for each
     * clientGPU-ServerGPU pair of workers. The requests must be sent in the
     * order of the client-server GPU id pairs (see server_create_eps).
     * Note that all the requests are sent to one signle listener on the
     * server. That is, we don't need one listener per worker. */
    ucp_ep_params_t ep_params;
    struct sockaddr_storage connect_addr;
    ucs_status_t status;

    set_sock_addr(address_str, &connect_addr);

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

    for (int cdev = 0; cdev < dev_count; cdev++) {
        for (int sdev = 0; sdev < dev_count; sdev++) {
            status = ucp_ep_create(dev_ucp_contexts[cdev].ucp_worker, &ep_params,
                                   &dev_ucp_contexts[cdev].ucp_eps[sdev]);
            if (status != UCS_OK) {
                fprintf(stderr, "failed to connect to %s (%s)\n", address_str,
                        ucs_status_string(status));
                close_eps(dev_ucp_contexts, dev_count);
                return -1;
            }

            dev_ucp_contexts[cdev].ep_count++;
        }
    }

    return 0;
}

static void print_iov(const ucp_dt_iov_t *iov)
{
    char *msg = alloca(test_string_length);
    size_t idx;

    for (idx = 0; idx < iov_cnt; idx++) {
        /* In case of Non-System memory */
        mem_type_memcpy(msg, iov[idx].buffer, test_string_length);
        printf("%s.\n", msg);
    }
}

/**
 * Print the received message on the server side or the sent data on the client
 * side.
 */
static
void print_result(int is_server, const ucp_dt_iov_t *iov, int current_iter)
{
    if (is_server) {
        printf("Server: iteration #%d\n", (current_iter + 1));
        printf("UCX data message was received\n");
        printf("\n\n----- UCP TEST SUCCESS -------\n\n");
    } else {
        printf("Client: iteration #%d\n", (current_iter + 1));
        printf("\n\n------------------------------\n\n");
    }

    print_iov(iov);

    printf("\n\n------------------------------\n\n");
}

/**
 * Progress the request until it completes.
 */
static ucs_status_t request_wait(ucp_worker_h ucp_worker, void *request,
                                 test_req_t *ctx)
{
    ucs_status_t status;

    /* if operation was completed immediately */
    if (request == NULL) {
        return UCS_OK;
    }

    if (UCS_PTR_IS_ERR(request)) {
        return UCS_PTR_STATUS(request);
    }

    while (ctx->complete == 0) {
        ucp_worker_progress(ucp_worker);
    }
    status = ucp_request_check_status(request);

    ucp_request_free(request);

    return status;
}

static int request_finalize(ucp_worker_h ucp_worker, test_req_t *request,
                            test_req_t *ctx, int is_server, ucp_dt_iov_t *iov,
                            int current_iter)
{
    int ret = 0;
    ucs_status_t status;

    status = request_wait(ucp_worker, request, ctx);
    if (status != UCS_OK) {
        fprintf(stderr, "unable to %s UCX message (%s)\n",
                is_server ? "receive": "send", ucs_status_string(status));
        ret = -1;
        goto release_iov;
    }

    /* Print the output of the first, last and every PRINT_INTERVAL iteration */
    if ((current_iter == 0) || (current_iter == (num_iterations - 1)) ||
        !((current_iter + 1) % (PRINT_INTERVAL))) {
        print_result(is_server, iov, current_iter);
    }

release_iov:
    buffer_free(iov, iov_cnt);
    return ret;
}

static int
fill_request_param(ucp_dt_iov_t *iov, int is_client,
                   void **msg, size_t *msg_length,
                   test_req_t *ctx, ucp_request_param_t *param)
{
    CHKERR_ACTION(buffer_malloc(iov) != 0, "allocate memory", return -1;);

    if (is_client && (fill_buffer(iov) != 0)) {
        buffer_free(iov, iov_cnt);
        return -1;
    }

    *msg        = (iov_cnt == 1) ? iov[0].buffer : iov;
    *msg_length = (iov_cnt == 1) ? iov[0].length : iov_cnt;

    ctx->complete       = 0;
    param->op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                          UCP_OP_ATTR_FIELD_DATATYPE |
                          UCP_OP_ATTR_FIELD_USER_DATA;
    param->datatype     = (iov_cnt == 1) ? ucp_dt_make_contig(1) :
                          UCP_DATATYPE_IOV;
    param->user_data    = ctx;

    return 0;
}

/**
 * Send and receive a message using the Stream API.
 * The client sends a message to the server and waits until the send it completed.
 * The server receives a message from the client and waits for its completion.
 */
static int send_recv_stream(ucp_worker_h ucp_worker, ucp_ep_h ep, int is_server,
                            int current_iter)
{
    ucp_dt_iov_t *iov = alloca(iov_cnt * sizeof(ucp_dt_iov_t));
    ucp_request_param_t param;
    test_req_t *request;
    size_t msg_length;
    void *msg;
    test_req_t ctx;

    memset(iov, 0, iov_cnt * sizeof(*iov));

    if (fill_request_param(iov, !is_server, &msg, &msg_length,
                           &ctx, &param) != 0) {
        return -1;
    }

    if (!is_server) {
        /* Client sends a message to the server using the stream API */
        param.cb.send = send_cb;
        request       = ucp_stream_send_nbx(ep, msg, msg_length, &param);
    } else {
        /* Server receives a message from the client using the stream API */
        param.op_attr_mask  |= UCP_OP_ATTR_FIELD_FLAGS;
        param.flags          = UCP_STREAM_RECV_FLAG_WAITALL;
        param.cb.recv_stream = stream_recv_cb;
        request              = ucp_stream_recv_nbx(ep, msg, msg_length,
                                                   &msg_length, &param);
    }

    return request_finalize(ucp_worker, request, &ctx, is_server, iov,
                            current_iter);
}

/**
 * Send and receive a message using the Tag-Matching API.
 * The client sends a message to the server and waits until the send it completed.
 * The server receives a message from the client and waits for its completion.
 */
static int send_recv_tag(ucp_worker_h ucp_worker, ucp_ep_h ep, int is_server,
                         int current_iter)
{
    ucp_dt_iov_t *iov = alloca(iov_cnt * sizeof(ucp_dt_iov_t));
    ucp_request_param_t param;
    void *request;
    size_t msg_length;
    void *msg;
    test_req_t ctx;

    memset(iov, 0, iov_cnt * sizeof(*iov));

    if (fill_request_param(iov, !is_server, &msg, &msg_length,
                           &ctx, &param) != 0) {
        return -1;
    }

    if (!is_server) {
        /* Client sends a message to the server using the Tag-Matching API */
        param.cb.send = send_cb;
        request       = ucp_tag_send_nbx(ep, msg, msg_length, TAG, &param);
    } else {
        /* Server receives a message from the client using the Tag-Matching API */
        param.cb.recv = tag_recv_cb;
        request       = ucp_tag_recv_nbx(ucp_worker, msg, msg_length, TAG, 0,
                                         &param);
    }

    return request_finalize(ucp_worker, request, &ctx, is_server, iov,
                            current_iter);
}

ucs_status_t ucp_am_data_cb(void *arg, const void *header, size_t header_length,
                            void *data, size_t length,
                            const ucp_am_recv_param_t *param)
{
    ucp_dt_iov_t *iov;
    size_t idx;
    size_t offset;

    if (length != iov_cnt * test_string_length) {
        fprintf(stderr, "received wrong data length %ld (expected %ld)",
                length, iov_cnt * test_string_length);
        return UCS_OK;
    }

    if (header_length != 0) {
        fprintf(stderr, "received unexpected header, length %ld", header_length);
    }

    am_data_desc.complete++;

    if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) {
        /* Rendezvous request arrived, data contains an internal UCX descriptor,
         * which has to be passed to ucp_am_recv_data_nbx function to confirm
         * data transfer.
         */
        am_data_desc.is_rndv = 1;
        am_data_desc.desc    = data;
        return UCS_INPROGRESS;
    }

    /* Message delivered with eager protocol, data should be available
     * immediately
     */
    am_data_desc.is_rndv = 0;

    iov = am_data_desc.recv_buf;
    offset = 0;
    for (idx = 0; idx < iov_cnt; idx++) {
        mem_type_memcpy(iov[idx].buffer, UCS_PTR_BYTE_OFFSET(data, offset),
                        iov[idx].length);
        offset += iov[idx].length;
    }

    return UCS_OK;
}

/**
 * Send and receive a message using Active Message API.
 * The client sends a message to the server and waits until the send is completed.
 * The server gets a message from the client and if it is rendezvous request,
 * initiates receive operation.
 */
static int send_recv_am(ucp_worker_h ucp_worker, ucp_ep_h ep, int is_server,
                        int current_iter)
{
    static int last   = 0;
    ucp_dt_iov_t *iov = alloca(iov_cnt * sizeof(ucp_dt_iov_t));
    test_req_t *request;
    ucp_request_param_t params;
    size_t msg_length;
    void *msg;
    test_req_t ctx;

    memset(iov, 0, iov_cnt * sizeof(*iov));

    if (fill_request_param(iov, !is_server, &msg, &msg_length,
                           &ctx, &params) != 0) {
        return -1;
    }

    if (is_server) {
        am_data_desc.recv_buf = iov;

        /* waiting for AM callback has called */
        while (last == am_data_desc.complete) {
            ucp_worker_progress(ucp_worker);
        }

        last++;

        if (am_data_desc.is_rndv) {
            /* Rendezvous request has arrived, need to invoke receive operation
             * to confirm data transfer from the sender to the "recv_message"
             * buffer. */
            params.op_attr_mask |= UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
            params.cb.recv_am    = am_recv_cb;
            request              = ucp_am_recv_data_nbx(ucp_worker,
                                                        am_data_desc.desc,
                                                        msg, msg_length,
                                                        &params);
        } else {
            /* Data has arrived eagerly and is ready for use, no need to
             * initiate receive operation. */
            request = NULL;
        }
    } else {
        /* Client sends a message to the server using the AM API */
        params.cb.send = (ucp_send_nbx_callback_t)send_cb;
        request        = ucp_am_send_nbx(ep, TEST_AM_ID, NULL, 0ul, msg,
                                         msg_length, &params);
    }

    return request_finalize(ucp_worker, request, &ctx, is_server, iov,
                            current_iter);
}

/**
 * Print this application's usage help message.
 */
static void usage()
{
    fprintf(stderr, "Usage: ucp_client_server [parameters]\n");
    fprintf(stderr, "UCP client-server example utility\n");
    fprintf(stderr, "\nParameters are:\n");
    fprintf(stderr, "  -a Set IP address of the server "
                    "(required for client and should not be specified "
                    "for the server)\n");
    fprintf(stderr, "  -l Set IP address where server listens "
                    "(If not specified, server uses INADDR_ANY; "
                    "Irrelevant at client)\n");
    fprintf(stderr, "  -p Port number to listen/connect to (default = %d). "
                    "0 on the server side means select a random port and print it\n",
                    DEFAULT_PORT);
    fprintf(stderr, "  -c Communication type for the client and server. "
                    "  Valid values are:\n"
                    "      'stream' : Stream API\n"
                    "      'tag'    : Tag API\n"
                    "      'am'     : AM API\n"
                    "     If not specified, %s API will be used.\n", COMM_TYPE_DEFAULT);
    fprintf(stderr, "  -i Number of iterations to run. Client and server must "
                    "have the same value. (default = %d).\n",
                    num_iterations);
    fprintf(stderr, "  -v Number of buffers in a single data "
                    "transfer function call. (default = %ld).\n",
                    iov_cnt);
    print_common_help();
    fprintf(stderr, "\n");
}

/**
 * Parse the command line arguments.
 */
static parse_cmd_status_t parse_cmd(int argc, char *const argv[],
                                    char **server_addr, char **listen_addr,
                                    send_recv_type_t *send_recv_type)
{
    int c = 0;
    int port;

    while ((c = getopt(argc, argv, "a:l:p:c:6i:s:v:m:h")) != -1) {
        switch (c) {
        case 'a':
            *server_addr = optarg;
            break;
        case 'c':
            if (!strcasecmp(optarg, "stream")) {
                *send_recv_type = CLIENT_SERVER_SEND_RECV_STREAM;
            } else if (!strcasecmp(optarg, "tag")) {
                *send_recv_type = CLIENT_SERVER_SEND_RECV_TAG;
            } else if (!strcasecmp(optarg, "am")) {
                *send_recv_type = CLIENT_SERVER_SEND_RECV_AM;
            } else {
                fprintf(stderr, "Wrong communication type %s. "
                        "Using %s as default\n", optarg, COMM_TYPE_DEFAULT);
                *send_recv_type = CLIENT_SERVER_SEND_RECV_DEFAULT;
            }
            break;
        case 'l':
            *listen_addr = optarg;
            break;
        case 'p':
            port = atoi(optarg);
            if ((port < 0) || (port > UINT16_MAX)) {
                fprintf(stderr, "Wrong server port number %d\n", port);
                return PARSE_CMD_STATUS_ERROR;
            }
            server_port = port;
            break;
        case '6':
            ai_family = AF_INET6;
            break;
        case 'i':
            num_iterations = atoi(optarg);
            break;
        case 's':
            test_string_length = atol(optarg);
            if (test_string_length < 0) {
                fprintf(stderr, "Wrong string size %ld\n", test_string_length);
                return PARSE_CMD_STATUS_ERROR;
            }
            break;
        case 'v':
            iov_cnt = atol(optarg);
            if (iov_cnt <= 0) {
                fprintf(stderr, "Wrong iov count %ld\n", iov_cnt);
                return PARSE_CMD_STATUS_ERROR;
            }
            break;
        case 'm':
            test_mem_type = parse_mem_type(optarg);
            if (test_mem_type == UCS_MEMORY_TYPE_LAST) {
                return PARSE_CMD_STATUS_ERROR;
            }
            break;
        case 'h':
            usage();
            return PARSE_CMD_STATUS_PRINT_HELP;
        default:
            usage();
            return PARSE_CMD_STATUS_ERROR;
        }
    }

    return PARSE_CMD_STATUS_OK;
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

static int client_server_communication(ucp_worker_h worker, ucp_ep_h ep,
                                       send_recv_type_t send_recv_type,
                                       int is_server, int current_iter)
{
    int ret;

    switch (send_recv_type) {
    case CLIENT_SERVER_SEND_RECV_STREAM:
        /* Client-Server communication via Stream API */
        ret = send_recv_stream(worker, ep, is_server, current_iter);
        break;
    case CLIENT_SERVER_SEND_RECV_TAG:
        /* Client-Server communication via Tag-Matching API */
        ret = send_recv_tag(worker, ep, is_server, current_iter);
        break;
    case CLIENT_SERVER_SEND_RECV_AM:
        /* Client-Server communication via AM API. */
        ret = send_recv_am(worker, ep, is_server, current_iter);
        break;
    default:
        fprintf(stderr, "unknown send-recv type %d\n", send_recv_type);
        return -1;
    }

    return ret;
}

/**
 * Create a ucp worker on the given ucp context.
 */
static int init_worker(ucp_context_h ucp_context, int client_id,
                       ucp_worker_h *ucp_worker);
{
    ucp_worker_params_t worker_params;
    ucs_status_t status;
    int ret = 0;

    memset(&worker_params, 0, sizeof(worker_params));

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE |
                                UCP_WORKER_PARAM_FIELD_CLIENT_ID;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    worker_params.client_id   = client_id;

    status = ucp_worker_create(ucp_context, &worker_params, ucp_worker);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_worker_create (%s)\n", ucs_status_string(status));
        ret = -1;
    }

    return ret;
}

/**
 * The callback on the server side which is invoked upon receiving a connection
 * request from the client.
 */
static void server_conn_handle_cb(ucp_conn_request_h conn_request, void *arg)
{
    ucx_server_ctx_t *context = arg;
    ucp_conn_request_attr_t attr;
    char ip_str[IP_STRING_LEN];
    char port_str[PORT_STRING_LEN];
    ucs_status_t status;

    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR |
                      UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ID;
    status = ucp_conn_request_query(conn_request, &attr);
    if (status == UCS_OK) {
        printf("Server received a connection request from client at address %s:%s\n",
               sockaddr_get_ip_str(&attr.client_address, ip_str, sizeof(ip_str)),
               sockaddr_get_port_str(&attr.client_address, port_str, sizeof(port_str)));
    } else if (status != UCS_ERR_UNSUPPORTED) {
        fprintf(stderr, "failed to query the connection request (%s)\n",
                ucs_status_string(status));
    }

    /* Accept the request only if we are not processing another client
     * already, or it is coming from the same client as the one we're
     * already processing. Otherwise, reject it. */
    if ((context->client_id == 0) ||
        (context->client_id == conn_attr.client_id)) {
        context->conn_request = conn_request;
        context->client_id    = conn_request.client_id;
    } else {
        printf("Rejecting a connection request. "
               "Only one client at a time is supported.\n");
        status = ucp_listener_reject(context->listener, conn_request);
        if (status != UCS_OK) {
            fprintf(stderr, "server failed to reject a connection request: (%s)\n",
                    ucs_status_string(status));
        }
    }
}

void close_eps(dev_ucp_ctx_t *dev_ucp_contexts, int dev_count)
{
    for (int ldev = 0; ldev < dev_count; ldev++) {
        for (int rdev = 0; rdev < dev_ucp_contexts[ldev].ep_count; rdev++) {
            ep_close(dev_ucp_contexts[ldev].ucp_eps[rdev],
                     UCP_EP_CLOSE_FLAG_FORCE);
        }
    }
}

static ucs_status_t server_create_ep(ucp_worker_h ucp_worker,
                                     ucp_conn_request_h conn_request,
                                     ucp_ep_h *server_ep)
{
    ucp_ep_params_t             ep_params;
    ucs_status_t                status;
    ucp_conn_request_attr_t     conn_attr;
    int                         clinet_dev_id;

    /* Server creates an ep to the client for each of its GPU-specific workers.
     * The client side should have initiated the connection (one for each of
     * its GPU-specific workers), leading to the ep creations here. */
    ep_params.field_mask      = UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                UCP_EP_PARAM_FIELD_CONN_REQUEST;
    ep_params.conn_request    = conn_request;
    ep_params.err_handler.cb  = err_cb;
    ep_params.err_handler.arg = NULL;

    status = ucp_ep_create(ucp_worker, &ep_params, server_ep);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to create an endpoint on the server: (%s)\n",
                ucs_status_string(status));
    }

    return status;
}

static ucs_status_t server_create_eps(ucp_dev_ctx_t *dev_ucp_contexts,
                                      int dev_count)
{
    /* Creating server-side eps. The eps are created upon receiving connection
     * requests initiated by the client. The client must initiate one request
     * from each of its own GPU-specific workers to each of the server's
     * GPU-specific workers. As a result, we'll end up creating one ep for each
     * serverGPU-clientGPU pair. The handles are stored in the ucp_eps array of
     * each GPU-specific context.
     * For each connection request, we need to know:
     *  1. the client-side UCP worker GPU id associated with the request,
     *  2. the server-side UCP worker GPU id that the request wants to target
     * We rely on a contract between the client and server: the client
     * issues the requests in the order of the client-server GPU id pairs.
     * That is, the first request is for client_gpu_0 to server_gpu_0,
     * the second for client_gpu_0 to server_gpu_1, and so on. Thus, we can use
     * a pair of dev_id counters on the server side to map each request to its
     * corresponding client-server GPU ids pair.
     * Note that we assume the client and server use the same number of GPUs.
     * Otherwise, they need to exchange an initial message to let each other
     * know about the number of GPUs they use. */
    for (int cdev = 0; cdev < dev_count; cdev++) { /* server GPUs */
        for (int sdev = 0; sdev < dev_count; sdev++) { /* client GPUs */
            /* Wait for the server to receive a connection request
             * from the client. If there are multiple clients for
             * which the server's connection request callback is invoked,
             * i.e. several clients are trying to connect in parallel, the
             * server will handle only the first one and reject the rest. */
            while (context.conn_request == NULL) {
                ucp_worker_progress(dev_ucp_contexts[0].ucp_worker);
            }

            status = server_create_ep(dev_ucp_contexts[sdev].ucp_worker,
                                      context.conn_request,
                                      &dev_ucp_contexts[sdev].ucp_eps[cdev]);
            if (status != UCS_OK) {
                close_eps(dev_ucp_contexts, dev_count);
                return -1;
            }

            dev_ucp_contexts[sdev].ep_count++;

            /* Now we are ready to accept the next request, but only
             * for the rest of the GPUs from the same client. */
            context.conn_request = NULL;
        }
    }

    return 0;
}

/**
 * Initialize the server side. The server starts listening on the set address.
 */
static ucs_status_t
start_server(ucp_worker_h ucp_worker, ucx_server_ctx_t *context,
             ucp_listener_h *listener_p, const char *address_str)
{
    struct sockaddr_storage listen_addr;
    ucp_listener_params_t params;
    ucp_listener_attr_t attr;
    ucs_status_t status;
    char ip_str[IP_STRING_LEN];
    char port_str[PORT_STRING_LEN];

    set_sock_addr(address_str, &listen_addr);

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

ucs_status_t register_am_recv_callback(ucp_worker_h worker)
{
    ucp_am_handler_param_t param;

    param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                       UCP_AM_HANDLER_PARAM_FIELD_CB |
                       UCP_AM_HANDLER_PARAM_FIELD_ARG;
    param.id         = TEST_AM_ID;
    param.cb         = ucp_am_data_cb;
    param.arg        = worker; /* not used in our callback */

    return ucp_worker_set_am_recv_handler(worker, &param);
}

static int client_server_do_work(ucp_worker_h ucp_worker, ucp_ep_h ep,
                                 send_recv_type_t send_recv_type, int is_server)
{
    int i, ret = 0;
    ucs_status_t status;

    connection_closed = 0;

    for (i = 0; i < num_iterations; i++) {
        ret = client_server_communication(ucp_worker, ep, send_recv_type,
                                          is_server, i);
        if (ret != 0) {
            fprintf(stderr, "%s failed on iteration #%d\n",
                    (is_server ? "server": "client"), i + 1);
            goto out;
        }
    }

    /* Register recv callback on the client side to receive FIN message */
    if (!is_server && (send_recv_type == CLIENT_SERVER_SEND_RECV_AM)) {
        status = register_am_recv_callback(ucp_worker);
        if (status != UCS_OK) {
            ret = -1;
            goto out;
        }
    }

    /* FIN message in reverse direction to acknowledge delivery */
    ret = client_server_communication(ucp_worker, ep, send_recv_type,
                                      !is_server, i + 1);
    if (ret != 0) {
        fprintf(stderr, "%s failed on FIN message\n",
                (is_server ? "server": "client"));
        goto out;
    }

    printf("%s FIN message\n", is_server ? "sent" : "received");

    /* Server waits until the client closed the connection after receiving FIN */
    while (is_server && !connection_closed) {
        ucp_worker_progress(ucp_worker);
    }

out:
    return ret;
}

static int run_server(dev_ucp_ctx_t *dev_ucp_contexts, int dev_count,
                      char *listen_addr, send_recv_type_t send_recv_type)
{
    ucx_server_ctx_t context;
    ucs_status_t     status;
    int              ret;

    if (send_recv_type == CLIENT_SERVER_SEND_RECV_AM) {
        for (int dev_id = 0; dev_id < dev_count; dev_id++) {
            status = register_am_recv_callback(dev_ucp_contexts[dev_id].ucp_worker);
            if (status != UCS_OK) {
                ret = -1;
                goto err;
            }
        }
    }

    /* Initialize the server's context. */
    context.conn_request = NULL;
    context.client_id    = 0;

    /* Create a listener for connection establishment between client and server.
     * This listener will stay open for listening to incoming connection
     * requests from the client.
     * The listener is created on a worker. We create only one listener on one of
     * the workers, and will use it for processing the incoming connection requests
     * from all other workers (that correspond to multiple GPUs). */
    status = start_server(dev_ucp_contexts[0].ucp_worker, &context,
                          &context.listener, listen_addr);
    if (status != UCS_OK) {
        ret = -1;
        goto err;
    }

    /* Server is always up listening */
    while (1) {
        ret = server_create_eps(dev_ucp_contexts, dev_count);

        if (ret != 0) {
            goto err_listener;
        }

        /* The server waits for all the iterations and all GPU pairs to complete
         * before moving on to the next client */
        ret = client_server_do_work(dev_ucp_contexts, dev_count,
                                    send_recv_type, 1);
        if (ret != 0) {
            goto err_ep;
        }

        /* Close all the endpoints to the client */
        close_eps(dev_ucp_contexts, dev_count);

        /* Reinitialize the server's context to be used for the next client */
        context.conn_request = NULL;
        context.client_id    = 0;

        printf("Waiting for connection...\n");
    }

err_ep:
    close_eps(dev_ucp_contexts, dev_count);
err_listener:
    ucp_listener_destroy(context.listener);
err:
    return ret;
}

static int run_client(ucp_worker_h *ucp_workers, int dev_count,
                      char *server_addr, send_recv_type_t send_recv_type)
{
    ucs_status_t status;
    int          ret;

    ret = client_create_eps(dev_ucp_contexts, dev_count, server_addr);
    if (ret != 0) {
        fprintf(stderr, "failed to create client eps(%s)\n",
                ucs_status_string(status));
        goto ep_close;
    }

    ret = client_server_do_work(ucp_worker, client_ep, send_recv_type, 0);

ep_close:
    /* Close all the endpoints to the server */
    close_eps(ucp_workers, client_eps, dev_count);
out:
    return ret;
}

/**
 * Initialize the UCP context and worker.
 */
static int init_context(dev_ucp_ctx_t *dev_ucp_ctx,
                        send_recv_type_t send_recv_type, int dev_id)
{
    /* UCP objects */
    ucp_params_t ucp_params;
    ucs_status_t status;
    uint64_t client_id;
    int ret = 0;

    memset(&ucp_params, 0, sizeof(ucp_params));

    /* UCP initialization */
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_NAME;
    ucp_params.name       = "client_server";

    if (send_recv_type == CLIENT_SERVER_SEND_RECV_STREAM) {
        ucp_params.features = UCP_FEATURE_STREAM;
    } else if (send_recv_type == CLIENT_SERVER_SEND_RECV_TAG) {
        ucp_params.features = UCP_FEATURE_TAG;
    } else {
        ucp_params.features = UCP_FEATURE_AM;
    }

    /* Make sure the GPU context is pushed on the stack before ucp_init */
    cudaSetDevice(dev_id);
    cudaFree(0);

    status = ucp_init(&ucp_params, NULL, &dev_ucp_ctx->ucp_context);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_init for dev %d: (%s)\n",
                dev_id, ucs_status_string(status));
        ret = -1;
        goto err;
    }

    /* Use the same client_id for all the workers. This is needed because with
     * multiple GPUs, the server will receive multiple connection requests per
     * client. Thus, we need a way to distinguish the requests that belong to
     * different clients on the server side. */
    client_id = ucs_generate_uuid((uintptr_t)&ucp_context);
    ret       = init_worker(dev_ucp_ctx->ucp_context, client_id,
                            &dev_ucp_ctx->ucp_worker);
    if (ret != 0) {
        goto err_cleanup;
    }

    dev_ucp_ctx->dev_id   = dev_id;
    dev_ucp_ctx->ep_count = 0;
    return ret;

err_cleanup:
    ucp_cleanup(dev_ucp_ctx->ucp_context);
err:
    return ret;
}

int finalize_context(dev_ucp_ctx_t *dev_ucp_ctx)
{
    ucp_worker_destroy(dev_ucp_ctx->ucp_worker);
    ucp_cleanup(dev_ucp_ctx->ucp_context);
    return 0;
}

int main(int argc, char **argv)
{
    send_recv_type_t send_recv_type = CLIENT_SERVER_SEND_RECV_DEFAULT;
    char *server_addr = NULL;
    char *listen_addr = NULL;
    parse_cmd_status_t parse_cmd_status;
    int ret;
    int dev_id, dev_count;

    /* GPU-specific UCP context */
    dev_ucp_ctx_t dev_ucp_contexts[MAX_DEV_COUNT];

    parse_cmd_status = parse_cmd(argc, argv, &server_addr, &listen_addr,
                                 &send_recv_type);
    if (parse_cmd_status == PARSE_CMD_STATUS_PRINT_HELP) {
        ret = 0;
        goto err;
    } else if (parse_cmd_status == PARSE_CMD_STATUS_ERROR) {
        ret = -1;
        goto err;
    }

    CUDA_FUNC(cudaGetDeviceCount(&dev_count));
    printf("detected %d GPU devices\n", dev_count);

    for (dev_id = 0; dev_id < dev_count; dev_id++) {
        /* Initialize the UCX required objects per GPU */
        ret = init_context(&dev_ucp_contexts[dev_id], send_recv_type, dev_id);
        if (ret != 0) {
            goto err;
        }
    }

    printf("created %d ucp_contexts and ucp_workers\n", dev_id);

    /* Client-Server initialization */
    if (server_addr == NULL) {
        /* Server side */
        ret = run_server(dev_ucp_contexts, dev_count,
                         listen_addr, send_recv_type);
    } else {
        /* Client side */
        ret = run_client(dev_ucp_contexts, dev_count,
                         server_addr, send_recv_type);
    }

    for (int i = 0; i < dev_id; i++) {
        finalize_context(&dev_ucp_contexts[i]);
    }
err:
    return ret;
}
