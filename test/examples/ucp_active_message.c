/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) Los Alamos National Security, LLC. 2018 ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#define HAVE_CONFIG_H /* Force using config.h, so test would fail if header
                         actually tries to use it */

/*
 * UCP Active Message Matrix  client / server example utility
 * -----------------------------------------------
 *
 * Server:
 *
 *    ./ucp_active_message
 *
 * Client side:
 *
 *    ./ucp_active_message -n <server host name>
 *
 * Notes:
 *
 *    - Client acquires Server UCX address via TCP socket
 *      then Server puts Active Messages onto Client
 *
 * Description:
 *
 *    - Client and Server will connect via a TCP socket.
 *      The Client will then create NUM_MATRICES matrices and set up
 *      an Active Message handler to perform matrix multiplication
 *      dependent on arguments sent from the Server.
 *      The Server will then put the AM's onto the Client and the
 *      Client will execute the active message.
 *
 * Author:
 *
 *    Ilya Nelkenbaum <ilya@nelkenbaum.com>
 *    Sergey Shalnov <sergeysh@mellanox.com> 7-June-2016v
 *    Jack Snyder     <jms285@duke.edu>
 */

#include "ucx_hello_world.h"

#include <ucp/api/ucp.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/epoll.h>
#include <netinet/in.h>
#include <assert.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>  /* getopt */
#include <ctype.h>   /* isprint */
#include <pthread.h> /* pthread_self */
#include <errno.h>   /* errno */
#include <time.h>
#include <signal.h>  /* raise */

#define NUM_MATRICES 50

struct msg {
    uint64_t        data_len;
};

struct ucx_context {
    int             completed;
};

enum ucp_test_mode_t {
    TEST_MODE_PROBE,
    TEST_MODE_WAIT,
    TEST_MODE_EVENTFD
} ucp_test_mode = TEST_MODE_PROBE;

static struct err_handling {
    ucp_err_handling_mode_t ucp_err_mode;
    int                     failure;
} err_handling_opt;

/* arguments that will be set on server and put onto client for AM */
typedef struct am_put_args{
    int array_index;
    int scalar;
} am_put_args_t;

/* argument that will be local to Client and will be passed into
 * every invocation of the am_handler as void *arg
 */
typedef struct am_recv_args{
    int *matrices[NUM_MATRICES];
    int recv_count;
    int matrix_size;
} am_recv_args_t;

static ucs_status_t client_status = UCS_OK;
static uint16_t server_port = 13337;  /* non-privileged port */
static const ucp_tag_t tag  = 0x1337a880u; /* tag that exercises all bits */
static const ucp_tag_t tag_mask = -1;
static ucp_address_t *local_addr;
static ucp_address_t *peer_addr;

static size_t local_addr_len;
static size_t peer_addr_len;

static int parse_cmd(int argc, char * const argv[], char **server_name);

static void request_init(void *request)
{
    struct ucx_context *ctx = (struct ucx_context *) request;
    ctx->completed = 0;
}

static void send_handle(void *request, ucs_status_t status)
{
    struct ucx_context *context = (struct ucx_context *) request;

    context->completed = 1;

    printf("[0x%x] send handler called with status %d (%s)\n",
           (unsigned int)pthread_self(), status, ucs_status_string(status));
}

static void recv_handle(void *request, ucs_status_t status,
                        ucp_tag_recv_info_t *info)
{
    struct ucx_context *context = (struct ucx_context *) request;

    context->completed = 1;

    printf("[0x%x] receive handler called with status %d (%s), length %lu\n",
           (unsigned int)pthread_self(), status, ucs_status_string(status),
           info->length);
}

am_put_args_t *desc;

static ucs_status_t recv_am_handler(void *arg, void *data, 
                                    size_t length, unsigned flags)
{
    int scalar, a_index;
    int i;
    am_recv_args_t *my_args = (am_recv_args_t *) arg;
    am_put_args_t *my_data = (am_put_args_t *) data;

    scalar = my_data->scalar;
    a_index = my_data->array_index;
    
    printf("Matrix : %d being multiplied by %d\n", a_index, scalar);
   
    for(i = 0; i < my_args->matrix_size; i++){
        my_args->matrices[a_index][i] = my_args->matrices[a_index][i] * scalar;
    }
    
    my_args->recv_count++;    
    
    desc = data;

    return UCS_OK;
}

static void failure_handler(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    ucs_status_t *arg_status = (ucs_status_t *) arg;

    printf("[0x%x] failure handler called with status %d (%s)\n",
           (unsigned int)pthread_self(), status, ucs_status_string(status));

    *arg_status = status;
}

static void wait(ucp_worker_h ucp_worker, struct ucx_context *context)
{
    while (context->completed == 0) {
        ucp_worker_progress(ucp_worker);
    }
}

static int run_ucx_client(ucp_worker_h ucp_worker)
{
    ucs_status_t status;
    ucp_ep_h server_ep;
    ucp_ep_params_t ep_params;
    struct msg *msg = NULL;
    struct ucx_context *request = 0;
    size_t msg_len = 0;
    int ret = -1;
    int i,j;

    am_recv_args_t args;

    /* Send client UCX address to server */
    ep_params.field_mask      = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                                UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.address         = peer_addr;
    ep_params.err_mode        = err_handling_opt.ucp_err_mode;

    status = ucp_ep_create(ucp_worker, &ep_params, &server_ep);

    CHKERR_JUMP(status != UCS_OK, "ucp_ep_create\n", err);

    msg_len = sizeof(*msg) + local_addr_len;
    msg = calloc(1, msg_len);
    CHKERR_JUMP(!msg, "allocate memory\n", err_ep);

    msg->data_len = local_addr_len;
    memcpy(msg + 1, local_addr, local_addr_len);

    request = ucp_tag_send_nb(server_ep, msg, msg_len,
                              ucp_dt_make_contig(1), tag,
                              send_handle);
    
    if (UCS_PTR_IS_ERR(request)) {
        fprintf(stderr, "unable to send UCX address message\n");
        free(msg);
        goto err_ep;
    } else if (UCS_PTR_STATUS(request) != UCS_OK) {
        wait(ucp_worker, request);
        request->completed = 0; /* Reset request state before recycling it */
        ucp_request_release(request);
    }

    free (msg);
    
    args.recv_count = 0;
    args.matrix_size = 9;
    for(i = 0; i < NUM_MATRICES; i++){
        args.matrices[i] = calloc(sizeof(int), args.matrix_size);
        printf("Before AM Matrix %d : ", i + 1);
        for(j = 0; j < args.matrix_size; j++){
            args.matrices[i][j] = i + 1;
            if(j % 3 == 0){
                printf("\n");
            }
            printf("%d ", i);
        }
        printf("\n");
    }
    
    ucp_worker_set_am_handler(ucp_worker, 0, recv_am_handler, 
                              &args, UCP_AM_FLAG_WHOLE_MSG);
                          
    while(args.recv_count < NUM_MATRICES){
        ucp_worker_progress(ucp_worker);
    }
  
    for(i = 0; i < NUM_MATRICES; i++){
        printf("After AM Matrix %d : ", i + 1);
        for(j = 0; j < args.matrix_size; j++){
            if(j % 3 == 0){
                printf("\n");
            }
            printf("%d ", args.matrices[i][j]);
        }
        printf("\n");
    } 

    ret = 0;

err_ep:
    ucp_ep_destroy(server_ep);

err:
    return ret;
}

static int run_ucx_server(ucp_worker_h ucp_worker)
{
    char string[1];
    generate_random_string(string, 1);
    ucp_tag_recv_info_t info_tag;
    ucp_tag_message_h msg_tag;
    ucs_status_t status;
    ucp_ep_h client_ep;
    ucp_ep_params_t ep_params;
    struct msg *msg = 0;
    struct ucx_context *request = 0;
    int ret = -1;
    int i;
    am_put_args_t put_args;// = malloc(10000);
    ucs_status_ptr_t status_ptr;
  
    /* Receive client UCX address */
    do {
        /* Progressing before probe to update the state */
        ucp_worker_progress(ucp_worker);

        /* Probing incoming events in non-block mode */
        msg_tag = ucp_tag_probe_nb(ucp_worker, tag, tag_mask, 1, &info_tag);
    } while (msg_tag == NULL);

    msg = malloc(info_tag.length);
    CHKERR_JUMP(!msg, "allocate memory\n", err);
    request = ucp_tag_msg_recv_nb(ucp_worker, msg, info_tag.length,
                                  ucp_dt_make_contig(1), msg_tag, recv_handle);

    if (UCS_PTR_IS_ERR(request)) {
        fprintf(stderr, "unable to receive UCX address message (%s)\n",
                ucs_status_string(UCS_PTR_STATUS(request)));
        free(msg);
        goto err;
    } else {
        wait(ucp_worker, request);
        request->completed = 0;
        ucp_request_release(request);
        printf("UCX address message was received\n");
    }

    peer_addr = malloc(msg->data_len);
    
    if (!peer_addr) {
        fprintf(stderr, "unable to allocate memory for peer address\n");
        free(msg);
        goto err;
    }

    peer_addr_len = msg->data_len;
    memcpy(peer_addr, msg + 1, peer_addr_len);

    free(msg);

    ep_params.field_mask      = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                                UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                UCP_EP_PARAM_FIELD_USER_DATA;
    ep_params.address         = peer_addr;
    ep_params.err_mode        = err_handling_opt.ucp_err_mode;
    ep_params.err_handler.cb  = failure_handler;
    ep_params.err_handler.arg = NULL;
    ep_params.user_data       = &client_status;

    status = ucp_ep_create(ucp_worker, &ep_params, &client_ep);
    CHKERR_JUMP(status != UCS_OK, "ucp_ep_create\n", err);
    
    /* telling client_ep to do matrix multiplication */
    for(i = 0; i < NUM_MATRICES; i++){
        put_args.array_index = i;
        put_args.scalar = i + 1;
        
        status_ptr = ucp_am_send_nb(client_ep, 0, &put_args, 1, 
                                    ucp_dt_make_contig(sizeof(am_put_args_t)), 
                                    send_handle, 0);
        
        if(status_ptr != UCS_OK){
            wait(ucp_worker, status_ptr);
            ((struct ucx_context *) status_ptr)->completed = 0;
            ucp_request_free(status_ptr);
        }
    }
    
    ucp_ep_destroy(client_ep);

err:
    return ret;
}

static int run_test(const char *client_target_name, ucp_worker_h ucp_worker)
{
    if (client_target_name != NULL) {
        return run_ucx_client(ucp_worker);
    } else {
        return run_ucx_server(ucp_worker);
    }
}

int main(int argc, char **argv)
{
    /* UCP temporary vars */
    ucp_params_t ucp_params;
    ucp_worker_params_t worker_params;
    ucp_config_t *config;
    ucs_status_t status;

    /* UCP handler objects */
    ucp_context_h ucp_context;
    ucp_worker_h ucp_worker;

    /* OOB connection vars */
    uint64_t addr_len = 0;
    char *client_target_name = NULL;
    int oob_sock = -1;
    int ret = -1;

    memset(&ucp_params, 0, sizeof(ucp_params));
    memset(&worker_params, 0, sizeof(worker_params));

    /* Parse the command line */
    status = parse_cmd(argc, argv, &client_target_name);
    CHKERR_JUMP(status != UCS_OK, "parse_cmd\n", err);

    /* UCP initialization */
    status = ucp_config_read(NULL, NULL, &config);
    CHKERR_JUMP(status != UCS_OK, "ucp_config_read\n", err);

    ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES |
                              UCP_PARAM_FIELD_REQUEST_SIZE |
                              UCP_PARAM_FIELD_REQUEST_INIT;
    ucp_params.features     = UCP_FEATURE_TAG | UCP_FEATURE_AM;
    if (ucp_test_mode == TEST_MODE_WAIT || ucp_test_mode == TEST_MODE_EVENTFD) {
        ucp_params.features |= UCP_FEATURE_WAKEUP;
    }
    ucp_params.request_size    = sizeof(struct ucx_context);
    ucp_params.request_init    = request_init;

    status = ucp_init(&ucp_params, config, &ucp_context);

    ucp_config_print(config, stdout, NULL, UCS_CONFIG_PRINT_CONFIG);

    ucp_config_release(config);
    CHKERR_JUMP(status != UCS_OK, "ucp_init\n", err);

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    status = ucp_worker_create(ucp_context, &worker_params, &ucp_worker);
    CHKERR_JUMP(status != UCS_OK, "ucp_worker_create\n", err_cleanup);

    status = ucp_worker_get_address(ucp_worker, &local_addr, &local_addr_len);
    CHKERR_JUMP(status != UCS_OK, "ucp_worker_get_address\n", err_worker);

    printf("[0x%x] local address length: %lu\n",
           (unsigned int)pthread_self(), local_addr_len);

    /* OOB connection establishment */
    if (client_target_name) {
        peer_addr_len = local_addr_len;

        oob_sock = client_connect(client_target_name, server_port);
        CHKERR_JUMP(oob_sock < 0, "client_connect\n", err_addr);

        ret = recv(oob_sock, &addr_len, sizeof(addr_len), 0);
        CHKERR_JUMP(ret < 0, "receive address length\n", err_addr);

        peer_addr_len = addr_len;
        peer_addr = malloc(peer_addr_len);
        CHKERR_JUMP(!peer_addr, "allocate memory\n", err_addr);

        ret = recv(oob_sock, peer_addr, peer_addr_len, 0);
        CHKERR_JUMP(ret < 0, "receive address\n", err_peer_addr);
    } else {
        oob_sock = server_connect(server_port);
        CHKERR_JUMP(oob_sock < 0, "server_connect\n", err_peer_addr);

        addr_len = local_addr_len;
        ret = send(oob_sock, &addr_len, sizeof(addr_len), 0);
        CHKERR_JUMP((ret < 0 || ret != sizeof(addr_len)),
                    "send address length\n", err_peer_addr);

        ret = send(oob_sock, local_addr, local_addr_len, 0);
        CHKERR_JUMP((ret < 0 || ret != local_addr_len),
                    "send address\n", err_peer_addr);
    }

    ret = run_test(client_target_name, ucp_worker);

    if (!err_handling_opt.failure) {
        /* Make sure remote is disconnected before destroying local worker */
        barrier(oob_sock);
    }
    close(oob_sock);

err_peer_addr:
    free(peer_addr);

err_addr:
    ucp_worker_release_address(ucp_worker, local_addr);

err_worker:
    ucp_worker_destroy(ucp_worker);

err_cleanup:
    ucp_cleanup(ucp_context);

err:
    return ret;
}

int parse_cmd(int argc, char * const argv[], char **server_name)
{
    int c = 0, index = 0;
    opterr = 0;

    err_handling_opt.ucp_err_mode   = UCP_ERR_HANDLING_MODE_NONE;
    err_handling_opt.failure        = 0;

    while ((c = getopt(argc, argv, "wfben:p:s:h")) != -1) {
        switch (c) {
        case 'n':
            *server_name = optarg;
            break;
        case 'p':
            server_port = atoi(optarg);
            if (server_port <= 0) {
                fprintf(stderr, "Wrong server port number %d\n", server_port);
                return UCS_ERR_UNSUPPORTED;
            }
            break;
        case 'h':
        default:
            fprintf(stderr, "Usage: ucp_active_message [parameters]\n");
            fprintf(stderr, "UCP active message matrix client/server"
                    "example utility\n");
            fprintf(stderr, "\nParameters are:\n");
            fprintf(stderr, "  -n name Set node name or IP address "
                    "of the server (required for client and should be ignored "
                    "for server)\n");
            fprintf(stderr, "  -p port Set alternative server"
                    "port (default:13337)\n");
            fprintf(stderr, "\n");
            return UCS_ERR_UNSUPPORTED;
        }
    }
    fprintf(stderr, "INFO: UCP_ACTIVE_MESSAGE server = %s port = %d\n",
            *server_name, server_port);

    for (index = optind; index < argc; index++) {
        fprintf(stderr, "WARNING: Non-option argument %s\n", argv[index]);
    }
    return UCS_OK;
}

int run_server()
{
    struct sockaddr_in inaddr;
    int lsock  = -1;
    int dsock  = -1;
    int optval = 1;
    int ret;

    lsock = socket(AF_INET, SOCK_STREAM, 0);
    CHKERR_JUMP(lsock < 0, "open server socket\n", err);

    optval = 1;
    ret = setsockopt(lsock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
    CHKERR_JUMP(ret < 0, "setsockopt server\n", err_sock);

    inaddr.sin_family      = AF_INET;
    inaddr.sin_port        = htons(server_port);
    inaddr.sin_addr.s_addr = INADDR_ANY;
    memset(inaddr.sin_zero, 0, sizeof(inaddr.sin_zero));
    ret = bind(lsock, (struct sockaddr*)&inaddr, sizeof(inaddr));
    CHKERR_JUMP(ret < 0, "bind server\n", err_sock);

    ret = listen(lsock, 0);
    CHKERR_JUMP(ret < 0, "listen server\n", err_sock);

    printf("Waiting for connection...\n");

    /* Accept next connection */
    dsock = accept(lsock, NULL, NULL);
    CHKERR_JUMP(dsock < 0, "accept server\n", err_sock);

    close(lsock);

    return dsock;

err_sock:
    close(lsock);

err:
    return -1;
}

int run_client(const char *server)
{
    struct sockaddr_in conn_addr;
    struct hostent *he;
    int connfd;
    int ret;

    connfd = socket(AF_INET, SOCK_STREAM, 0);
    if (connfd < 0) {
        fprintf(stderr, "socket() failed: %s\n", strerror(errno));
        return -1;
    }

    he = gethostbyname(server);
    CHKERR_JUMP((he == NULL || he->h_addr_list == NULL), "found host\n", err_conn);

    conn_addr.sin_family = he->h_addrtype;
    conn_addr.sin_port   = htons(server_port);

    memcpy(&conn_addr.sin_addr, he->h_addr_list[0], he->h_length);
    memset(conn_addr.sin_zero, 0, sizeof(conn_addr.sin_zero));

    ret = connect(connfd, (struct sockaddr*)&conn_addr, sizeof(conn_addr));
    CHKERR_JUMP(ret < 0, "connect client\n", err_conn);

    return connfd;

err_conn:
    close(connfd);

    return -1;
}
