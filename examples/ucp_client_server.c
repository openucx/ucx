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

#include "hello_world_util.h"

#include <ucp/api/ucp.h>

#include <string.h>    /* memset */
#include <arpa/inet.h> /* inet_addr */
#include <unistd.h>    /* getopt */
#include <stdlib.h>    /* atoi */
#include <errno.h>
#include <limits.h>

#define DEFAULT_PORT           13337
#define IP_STRING_LEN          50
#define PORT_STRING_LEN        8
#define TAG                    0xCAFE
#define COMM_TYPE_DEFAULT      "STREAM"
#define PRINT_INTERVAL         2000
#define DEFAULT_NUM_ITERATIONS 1
#define TEST_AM_ID             0


static long test_string_length = 16;
static uint16_t server_port    = DEFAULT_PORT;
static int num_iterations      = DEFAULT_NUM_ITERATIONS;


typedef enum {
    CLIENT_SERVER_SEND_RECV_STREAM  = UCS_BIT(0),
    CLIENT_SERVER_SEND_RECV_TAG     = UCS_BIT(1),
    CLIENT_SERVER_SEND_RECV_AM      = UCS_BIT(2),
    CLIENT_SERVER_SEND_RECV_DEFAULT = CLIENT_SERVER_SEND_RECV_STREAM
} send_recv_type_t;


typedef struct data_meta {
    int              is_server;
    send_recv_type_t send_recv_type;

    void             *buffer;
    size_t           iov_num;
    size_t           *iov_sizes;
} data_meta_t;


/**
 * Server's application context to be used in the user's connection request
 * callback.
 * It holds the server's listener and the handle to an incoming connection request.
 */
typedef struct ucx_server_ctx {
    volatile ucp_conn_request_h conn_request;
    ucp_listener_h              listener;
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
 * Print this application's usage help message.
 */
static void usage(void);

void buffer_free(data_meta_t *mdata)
{
    size_t       idx;
    ucp_dt_iov_t *iov = mdata->buffer;

    for (idx = 0; idx < mdata->iov_num; idx++) {
        if (iov[idx].buffer == NULL) {
            continue;
        }
        mem_type_free(iov[idx].buffer);
        iov[idx].buffer = NULL;
        iov[idx].length = 0;
    }
    mem_type_free(mdata->buffer);
    mdata->buffer = NULL;
}

int buffer_malloc(data_meta_t *mdata)
{
    size_t       idx;
    ucp_dt_iov_t *iov;

    mdata->buffer = calloc(mdata->iov_num, sizeof(ucp_dt_iov_t));
    iov           = mdata->buffer;
    CHKERR_ACTION(iov == NULL, "allocate memory\n", return -1;);

    for (idx = 0; idx < mdata->iov_num; idx++) {
        iov[idx].length = mdata->iov_sizes[idx];
        iov[idx].buffer = mem_type_malloc(iov[idx].length);
        if (iov[idx].buffer == NULL) {
            buffer_free(mdata);
            return -1;
        }
        mem_type_memset(iov[idx].buffer, 0, iov[idx].length);
    }

    return 0;
}

int fill_buffer(data_meta_t *mdata)
{
    size_t       idx;
    ucp_dt_iov_t *iov = mdata->buffer;
    int          ret = 0;

    for (idx = 0; idx < mdata->iov_num; idx++) {
        ret = generate_test_string(iov[idx].buffer, iov[idx].length);
        if (ret != 0) {
            break;
        }
    }
    CHKERR_ACTION(ret != 0, "generate test string", return -1;);
    return 0;
}

char **copy_buffer(data_meta_t *mdata)
{
    char         **ret;
    size_t       idx;
    ucp_dt_iov_t *iov = mdata->buffer;

    ret = calloc(mdata->iov_num, sizeof(char*));
    CHKERR_ACTION(ret == NULL, "allocate memory\n", return NULL;);

    for (idx = 0; idx < mdata->iov_num; idx++) {
        ret[idx] = calloc(iov[idx].length + 1, sizeof(char));
        if (ret[idx] == NULL) {
            break;
        }
        mem_type_memcpy(ret[idx], iov[idx].buffer, iov[idx].length);
    }
    if (idx == mdata->iov_num) {
        return ret;
    }

    for (idx = 0; idx < mdata->iov_num; idx++) {
        free(ret[idx]);
        ret[idx] = NULL;
    }
    free(ret);
    return NULL;
}

void free_copied_buffer(data_meta_t *mdata, void *msg)
{
    size_t idx;
    char   **pmsg = msg;

    for (idx = 0; idx < mdata->iov_num; idx++) {
        free(pmsg[idx]);
        pmsg[idx] = NULL;
    }
    free(msg);
}

static void tag_recv_cb(void *request, ucs_status_t status,
                        const ucp_tag_recv_info_t *info, void *user_data)
{
    test_req_t *ctx = user_data;

    ctx->complete = 1;
}

/**
 * The callback on the receiving side, which is invoked upon receiving the
 * stream message.
 */
static void
stream_recv_cb(void *request, ucs_status_t status, size_t length,
               void *user_data)
{
    test_req_t *ctx = user_data;

    ctx->complete = 1;
}

/**
 * The callback on the receiving side, which is invoked upon receiving the
 * active message.
 */
static void am_recv_cb(void *request, ucs_status_t status, size_t length,
                       void *user_data)
{
    test_req_t *ctx = user_data;

    ctx->complete = 1;
}

/**
 * The callback on the sending side, which is invoked after finishing sending
 * the message.
 */
static void send_cb(void *request, ucs_status_t status, void *user_data)
{
    test_req_t *ctx = user_data;

    ctx->complete = 1;
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
 * Print the received message on the server side or the sent data on the client
 * side.
 */
static void print_result(data_meta_t *mdata, void *pmsg, int current_iter)
{
    size_t idx;
    char   **msg = pmsg;

    if (mdata->is_server) {
        printf("Server: iteration #%d\n", (current_iter + 1));
        printf("UCX data message was received\n");
        printf("\n\n----- UCP TEST SUCCESS -------\n\n");

        for (idx = 0; idx < mdata->iov_num; idx++) {
            printf("%s\n", msg[idx]);
        }

        printf("\n------------------------------\n\n");
    } else {
        printf("Client: iteration #%d\n", (current_iter + 1));
        printf("\n\n-----------------------------------------\n\n");

        for (idx = 0; idx < mdata->iov_num; idx++) {
            printf("Client sent iov msg: \n%s.\nlength: %ld\n",
                   (strlen(msg[idx]) != 0) ? msg[idx] : "<none>",
                   strlen(msg[idx]) + 1);
        }

        printf("\n-----------------------------------------\n\n");
    }
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
                            test_req_t *ctx, void *msg, int current_iter)
{
    int ret = 0;
    char **msg_str;
    data_meta_t *mdata = msg;
    ucs_status_t status;

    status = request_wait(ucp_worker, request, ctx);
    if (status != UCS_OK) {
        fprintf(stderr, "unable to %s UCX message (%s)\n",
                mdata->is_server ? "receive" : "send",
                ucs_status_string(status));
        ret = -1;
        goto release_msg;
    }

    /* Print the output of the first, last and every PRINT_INTERVAL iteration */
    if ((current_iter == 0) || (current_iter == (num_iterations - 1)) ||
        !((current_iter + 1) % (PRINT_INTERVAL))) {
        msg_str = copy_buffer(mdata);
        if (msg_str == NULL) {
            fprintf(stderr, "memory allocation failed\n");
            ret = -1;
            goto release_msg;
        }
        print_result(mdata, msg_str, current_iter);
        free_copied_buffer(mdata, msg_str);
    }

release_msg:
    buffer_free(mdata);
    return ret;
}

/**
 * Send and receive a message using the Stream API.
 * The client sends a message to the server and waits until the send it completed.
 * The server receives a message from the client and waits for its completion.
 */
static int send_recv_stream(ucp_worker_h ucp_worker, ucp_ep_h ep,
                            data_meta_t *mdata, int current_iter)
{
    ucp_request_param_t param;
    test_req_t *request;
    size_t msg_length;
    void *msg;
    ucp_dt_iov_t *iov;
    test_req_t ctx;
    int ret;

    ret = buffer_malloc(mdata);
    CHKERR_ACTION(ret != 0, "allocate memory\n", return -1;);
    iov = mdata->buffer;

    msg        = mdata->iov_num == 1 ? iov[0].buffer : mdata->buffer;
    msg_length = mdata->iov_num == 1 ? iov[0].length : mdata->iov_num;

    ctx.complete       = 0;
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                         UCP_OP_ATTR_FIELD_DATATYPE |
                         UCP_OP_ATTR_FIELD_USER_DATA;
    param.datatype     = mdata->iov_num == 1 ? ucp_dt_make_contig(1) :
                         UCP_DATATYPE_IOV;
    param.user_data    = &ctx;

    if (!mdata->is_server) {
        ret = fill_buffer(mdata);
        CHKERR_ACTION(ret != 0, "generate test string", return -1;);

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

    return request_finalize(ucp_worker, request, &ctx, mdata, current_iter);
}

/**
 * Send and receive a message using the Tag-Matching API.
 * The client sends a message to the server and waits until the send it completed.
 * The server receives a message from the client and waits for its completion.
 */
static int send_recv_tag(ucp_worker_h ucp_worker, ucp_ep_h ep,
                         data_meta_t *mdata, int current_iter)
{
    ucp_request_param_t param;
    void *request;
    size_t msg_length;
    void *msg;
    ucp_dt_iov_t *iov;
    test_req_t ctx;
    int ret;

    ret = buffer_malloc(mdata);
    CHKERR_ACTION(ret != 0, "allocate memory\n", return -1;);
    iov = mdata->buffer;

    msg        = mdata->iov_num == 1 ? iov[0].buffer : mdata->buffer;
    msg_length = mdata->iov_num == 1 ? iov[0].length : mdata->iov_num;

    ctx.complete       = 0;
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                         UCP_OP_ATTR_FIELD_DATATYPE |
                         UCP_OP_ATTR_FIELD_USER_DATA;
    param.datatype     = mdata->iov_num == 1 ? ucp_dt_make_contig(1) :
                         UCP_DATATYPE_IOV;
    param.user_data    = &ctx;
    if (!mdata->is_server) {
        ret = fill_buffer(mdata);
        CHKERR_ACTION(ret != 0, "generate test string", return -1;);

        /* Client sends a message to the server using the Tag-Matching API */
        param.cb.send = send_cb;
        request       = ucp_tag_send_nbx(ep, msg, msg_length, TAG, &param);
    } else {
        /* Server receives a message from the client using the Tag-Matching API */
        param.cb.recv = tag_recv_cb;
        request       = ucp_tag_recv_nbx(ucp_worker, msg, msg_length, TAG, 0,
                                         &param);
    }

    return request_finalize(ucp_worker, request, &ctx, mdata, current_iter);
}

ucs_status_t ucp_am_data_cb(void *arg, const void *header, size_t header_length,
                            void *data, size_t length,
                            const ucp_am_recv_param_t *param)
{
    data_meta_t  *mdata;
    ucp_dt_iov_t *iov;
    size_t       idx;
    size_t       offset = 0;

    if (length != test_string_length) {
        fprintf(stderr, "received wrong data length %ld (expected %ld)",
                length, test_string_length);
        return UCS_OK;
    }

    if ((header != NULL) || (header_length != 0)) {
        fprintf(stderr, "received unexpected header, length %ld", header_length);
    }

    am_data_desc.complete = 1;

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

    mdata = am_data_desc.recv_buf;
    iov   = mdata->buffer;
    for (idx = 0; idx < mdata->iov_num; idx++) {
        mem_type_memcpy(iov[idx].buffer, (char*)data + offset, iov[idx].length);
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
static int send_recv_am(ucp_worker_h ucp_worker, ucp_ep_h ep,
                        data_meta_t *mdata, int current_iter)
{
    test_req_t *request;
    ucp_request_param_t params;
    size_t msg_length;
    void *msg;
    ucp_dt_iov_t *iov;
    test_req_t ctx;
    int ret;

    ret = buffer_malloc(mdata);
    CHKERR_ACTION(ret != 0, "allocate memory\n", return -1;);
    iov = mdata->buffer;

    msg        = mdata->iov_num == 1 ? iov[0].buffer : mdata->buffer;
    msg_length = mdata->iov_num == 1 ? iov[0].length : mdata->iov_num;

    ctx.complete        = 0;
    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                          UCP_OP_ATTR_FIELD_DATATYPE |
                          UCP_OP_ATTR_FIELD_USER_DATA;
    params.datatype     = mdata->iov_num == 1 ? ucp_dt_make_contig(1) :
                          UCP_DATATYPE_IOV;
    params.user_data    = &ctx;

    if (mdata->is_server) {
        am_data_desc.recv_buf = mdata;

        /* waiting for AM callback has called */
        while (!am_data_desc.complete) {
            ucp_worker_progress(ucp_worker);
        }

        am_data_desc.complete = 0;

        if (am_data_desc.is_rndv) {
            /* Rendezvous request has arrived, need to invoke receive operation
             * to confirm data transfer from the sender to the "recv_message"
             * buffer. */
            params.op_attr_mask |= UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
            params.cb.recv_am    = am_recv_cb,
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
        ret = fill_buffer(mdata);
        CHKERR_ACTION(ret != 0, "generate test string", return -1;);

        /* Client sends a message to the server using the AM API */
        params.cb.send = (ucp_send_nbx_callback_t)send_cb,
        request        = ucp_am_send_nbx(ep, TEST_AM_ID, NULL, 0ul, msg,
                                         msg_length, &params);
    }

    return request_finalize(ucp_worker, request, &ctx, mdata, current_iter);
}

/**
 * Close the given endpoint.
 * Currently closing the endpoint with UCP_EP_CLOSE_MODE_FORCE since we currently
 * cannot rely on the client side to be present during the server's endpoint
 * closing process.
 */
static void ep_close(ucp_worker_h ucp_worker, ucp_ep_h ep)
{
    ucp_request_param_t param;
    ucs_status_t status;
    void *close_req;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    param.flags        = UCP_EP_CLOSE_FLAG_FORCE;
    close_req          = ucp_ep_close_nbx(ep, &param);
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
    print_common_help();
    fprintf(stderr, "\n");
}

static int parse_message_sizes(char *opt_arg, data_meta_t *mdata)
{
    const char delim = ',';
    size_t token_num, token_it;
    char *optarg_ptr, *optarg_ptr2;

    optarg_ptr = opt_arg;
    token_num  = 0;

    while ((optarg_ptr = strchr(optarg_ptr, delim)) != NULL) {
        ++optarg_ptr;
        ++token_num;
    }
    ++token_num;
    mdata->iov_num   = token_num;
    mdata->iov_sizes = calloc(mdata->iov_num, sizeof(mdata->iov_sizes[0]));
    CHKERR_ACTION(mdata->iov_sizes == NULL, "allocate memory\n", return -1;);

    optarg_ptr = opt_arg;
    errno      = 0;

    for (token_it = 0; token_it < mdata->iov_num; ++token_it) {
        mdata->iov_sizes[token_it] = strtoul(optarg_ptr, &optarg_ptr2, 10);
        if ((ERANGE == errno && ULONG_MAX == mdata->iov_sizes[token_it]) ||
            (errno != 0 && mdata->iov_sizes[token_it] == 0) ||
            (optarg_ptr == optarg_ptr2)) {
            free(mdata->iov_sizes);
            mdata->iov_sizes = NULL;
            printf("Invalid message size\n");
            return -1;
        }

        optarg_ptr = optarg_ptr2 + 1;
    }

    if (mdata->iov_num == 1) {
        test_string_length = mdata->iov_sizes[0];
    }

    return 0;
}

/**
 * Parse the command line arguments.
 */
static int parse_cmd(int argc, char *const argv[], char **server_addr,
                     char **listen_addr, data_meta_t *mdata)
{
    int c = 0;
    int port;

    while ((c = getopt(argc, argv, "a:l:p:c:i:s:m:h")) != -1) {
        switch (c) {
        case 'a':
            mdata->is_server = 0;
            *server_addr = optarg;
            break;
        case 'c':
            if (!strcasecmp(optarg, "stream")) {
                mdata->send_recv_type = CLIENT_SERVER_SEND_RECV_STREAM;
            } else if (!strcasecmp(optarg, "tag")) {
                mdata->send_recv_type = CLIENT_SERVER_SEND_RECV_TAG;
            } else if (!strcasecmp(optarg, "am")) {
                mdata->send_recv_type = CLIENT_SERVER_SEND_RECV_AM;
            } else {
                fprintf(stderr, "Wrong communication type %s. "
                        "Using %s as default\n", optarg, COMM_TYPE_DEFAULT);
                mdata->send_recv_type = CLIENT_SERVER_SEND_RECV_DEFAULT;
            }
            break;
        case 'l':
            mdata->is_server = 1;
            *listen_addr = optarg;
            break;
        case 'p':
            port = atoi(optarg);
            if ((port < 0) || (port > UINT16_MAX)) {
                fprintf(stderr, "Wrong server port number %d\n", port);
                return -1;
            }
            server_port = port;
            break;
        case 'i':
            num_iterations = atoi(optarg);
            break;
        case 's':
            if (parse_message_sizes(optarg, mdata) != 0) {
                printf("Wrong string size(s)\n");
                return UCS_ERR_UNSUPPORTED;
            }
            break;
        case 'm':
            test_mem_type = parse_mem_type(optarg);
            if (test_mem_type == UCS_MEMORY_TYPE_LAST) {
                return UCS_ERR_UNSUPPORTED;
            }
            break;
        case 'h':
        default:
            usage();
            return -1;
        }
    }
    if (mdata->iov_num == 0) {
        mdata->iov_num = 1;
        mdata->iov_sizes = calloc(mdata->iov_num, sizeof(mdata->iov_sizes[0]));
        CHKERR_ACTION(mdata->iov_sizes == NULL, "allocate memory\n", return -1;);
        mdata->iov_sizes[0] = test_string_length;
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

static int client_server_communication(ucp_worker_h worker, ucp_ep_h ep,
                                       data_meta_t *mdata, int current_iter)
{
    int ret;

    switch (mdata->send_recv_type) {
    case CLIENT_SERVER_SEND_RECV_STREAM:
        /* Client-Server communication via Stream API */
        ret = send_recv_stream(worker, ep, mdata, current_iter);
        break;
    case CLIENT_SERVER_SEND_RECV_TAG:
        /* Client-Server communication via Tag-Matching API */
        ret = send_recv_tag(worker, ep, mdata, current_iter);
        break;
    case CLIENT_SERVER_SEND_RECV_AM:
        /* Client-Server communication via AM API. */
        ret = send_recv_am(worker, ep, mdata, current_iter);
        break;
    default:
        fprintf(stderr, "unknown send-recv type %d\n", mdata->send_recv_type);
        return -1;
    }

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

    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    status = ucp_conn_request_query(conn_request, &attr);
    if (status == UCS_OK) {
        printf("Server received a connection request from client at address %s:%s\n",
               sockaddr_get_ip_str(&attr.client_address, ip_str, sizeof(ip_str)),
               sockaddr_get_port_str(&attr.client_address, port_str, sizeof(port_str)));
    } else if (status != UCS_ERR_UNSUPPORTED) {
        fprintf(stderr, "failed to query the connection request (%s)\n",
                ucs_status_string(status));
    }

    if (context->conn_request == NULL) {
        context->conn_request = conn_request;
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

static int
client_server_do_work(ucp_worker_h ucp_worker, ucp_ep_h ep, data_meta_t *mdata)
{
    int i, ret = 0;

    for (i = 0; i < num_iterations; i++) {
        ret = client_server_communication(ucp_worker, ep, mdata, i);
        if (ret != 0) {
            fprintf(stderr, "%s failed on iteration #%d\n",
                    (mdata->is_server ? "server" : "client"), i + 1);
            goto out;
        }
    }

out:
    return ret;
}

static int run_server(ucp_context_h ucp_context, ucp_worker_h ucp_worker,
                      char *listen_addr, data_meta_t *mdata)
{
    ucx_server_ctx_t context;
    ucp_worker_h     ucp_data_worker;
    ucp_am_handler_param_t param;
    ucp_ep_h         server_ep;
    ucs_status_t     status;
    int              ret;
    int              idx;

    /* Create a data worker (to be used for data exchange between the server
     * and the client after the connection between them was established) */
    ret = init_worker(ucp_context, &ucp_data_worker);
    if (ret != 0) {
        goto err;
    }

    if (mdata->send_recv_type == CLIENT_SERVER_SEND_RECV_AM) {
        /* Initialize Active Message data handler */
        param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                           UCP_AM_HANDLER_PARAM_FIELD_CB |
                           UCP_AM_HANDLER_PARAM_FIELD_ARG;
        param.id         = TEST_AM_ID;
        param.cb         = ucp_am_data_cb;
        param.arg        = ucp_data_worker; /* not used in our callback */
        status           = ucp_worker_set_am_recv_handler(ucp_data_worker,
                                                          &param);
        if (status != UCS_OK) {
            ret = -1;
            goto err_worker;
        }

        if (mdata->iov_num != 1) {
            test_string_length = 0;
            for (idx = 0; idx < mdata->iov_num; idx++) {
                test_string_length += mdata->iov_sizes[idx];
            }
        }
    }

    /* Initialize the server's context. */
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
         * callback is invoked, i.e. several clients are trying to connect in
         * parallel, the server will handle only the first one and reject the rest */
        while (context.conn_request == NULL) {
            ucp_worker_progress(ucp_worker);
        }

        /* Server creates an ep to the client on the data worker.
         * This is not the worker the listener was created on.
         * The client side should have initiated the connection, leading
         * to this ep's creation */
        status = server_create_ep(ucp_data_worker, context.conn_request,
                                  &server_ep);
        if (status != UCS_OK) {
            ret = -1;
            goto err_listener;
        }

        /* The server waits for all the iterations to complete before moving on
         * to the next client */
        ret = client_server_do_work(ucp_data_worker, server_ep, mdata);
        if (ret != 0) {
            goto err_ep;
        }

        /* Close the endpoint to the client */
        ep_close(ucp_data_worker, server_ep);

        /* Reinitialize the server's context to be used for the next client */
        context.conn_request = NULL;

        printf("Waiting for connection...\n");
    }

err_ep:
    ep_close(ucp_data_worker, server_ep);
err_listener:
    ucp_listener_destroy(context.listener);
err_worker:
    ucp_worker_destroy(ucp_data_worker);
err:
    return ret;
}

static int
run_client(ucp_worker_h ucp_worker, char *server_addr, data_meta_t *mdata)
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

    ret = client_server_do_work(ucp_worker, client_ep, mdata);

    /* Close the endpoint to the server */
    ep_close(ucp_worker, client_ep);

out:
    return ret;
}

/**
 * Initialize the UCP context and worker.
 */
static int init_context(ucp_context_h *ucp_context, ucp_worker_h *ucp_worker,
                        data_meta_t *mdata)
{
    /* UCP objects */
    ucp_params_t ucp_params;
    ucs_status_t status;
    int ret = 0;

    memset(&ucp_params, 0, sizeof(ucp_params));

    /* UCP initialization */
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;

    if (mdata->send_recv_type == CLIENT_SERVER_SEND_RECV_STREAM) {
        ucp_params.features = UCP_FEATURE_STREAM;
    } else if (mdata->send_recv_type == CLIENT_SERVER_SEND_RECV_TAG) {
        ucp_params.features = UCP_FEATURE_TAG;
    } else {
        ucp_params.features = UCP_FEATURE_AM;
    }

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
    char *listen_addr = NULL;
    int               ret;
    data_meta_t       mdata;

    /* UCP objects */
    ucp_context_h ucp_context;
    ucp_worker_h  ucp_worker;

    memset(&mdata, 0, sizeof(mdata));
    mdata.send_recv_type     = CLIENT_SERVER_SEND_RECV_DEFAULT;
    ret = parse_cmd(argc, argv, &server_addr, &listen_addr, &mdata);
    if (ret != 0) {
        goto err;
    }

    /* Initialize the UCX required objects */
    ret = init_context(&ucp_context, &ucp_worker, &mdata);
    if (ret != 0) {
        goto err;
    }

    /* Client-Server initialization */
    if (server_addr == NULL) {
        /* Server side */
        mdata.is_server = 1;
        ret = run_server(ucp_context, ucp_worker, listen_addr, &mdata);
    } else {
        /* Client side */
        mdata.is_server = 0;
        ret = run_client(ucp_worker, server_addr, &mdata);
    }

    ucp_worker_destroy(ucp_worker);
    ucp_cleanup(ucp_context);
err:
    free(mdata.iov_sizes);

    return ret;
}
