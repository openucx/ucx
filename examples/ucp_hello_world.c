/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) Advanced Micro Devices, Inc. 2018. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef HAVE_CONFIG_H
#  define HAVE_CONFIG_H /* Force using config.h, so test would fail if header
                           actually tries to use it */
#endif

/*
 * UCP hello world client / server example utility
 * -----------------------------------------------
 *
 * Server side:
 *
 *    ./ucp_hello_world
 *
 * Client side:
 *
 *    ./ucp_hello_world -n <server host name>
 *
 * Notes:
 *
 *    - Client acquires Server UCX address via TCP socket
 *
 *
 * Author:
 *
 *    Ilya Nelkenbaum <ilya@nelkenbaum.com>
 *    Sergey Shalnov <sergeysh@mellanox.com> 7-June-2016
 */

#include "hello_world_util.h"

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

static ucs_status_t client_status = UCS_OK;
static uint16_t server_port       = 13337;
static long test_string_length    = 16;
static const ucp_tag_t tag        = 0x1337a880u;
static const ucp_tag_t tag_mask   = UINT64_MAX;
static const char *addr_msg_str   = "UCX address message";
static const char *data_msg_str   = "UCX data message";
static ucp_address_t *local_addr;
static ucp_address_t *peer_addr;

static size_t local_addr_len;
static size_t peer_addr_len;

static ucs_status_t parse_cmd(int argc, char * const argv[], char **server_name);

static void set_msg_data_len(struct msg *msg, uint64_t data_len)
{
    mem_type_memcpy(&msg->data_len, &data_len, sizeof(data_len));
}

static void request_init(void *request)
{
    struct ucx_context *contex = (struct ucx_context *)request;

    contex->completed = 0;
}

static void send_handler(void *request, ucs_status_t status, void *ctx)
{
    struct ucx_context *context = (struct ucx_context *)request;
    const char *str             = (const char *)ctx;

    context->completed = 1;

    printf("[0x%x] send handler called for \"%s\" with status %d (%s)\n",
           (unsigned int)pthread_self(), str, status,
           ucs_status_string(status));
}

static void failure_handler(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    ucs_status_t *arg_status = (ucs_status_t *)arg;

    printf("[0x%x] failure handler called with status %d (%s)\n",
           (unsigned int)pthread_self(), status, ucs_status_string(status));

    *arg_status = status;
}

static void recv_handler(void *request, ucs_status_t status,
                        ucp_tag_recv_info_t *info)
{
    struct ucx_context *context = (struct ucx_context *)request;

    context->completed = 1;

    printf("[0x%x] receive handler called with status %d (%s), length %lu\n",
           (unsigned int)pthread_self(), status, ucs_status_string(status),
           info->length);
}

static ucs_status_t ucx_wait(ucp_worker_h ucp_worker, struct ucx_context *request,
                             const char *op_str, const char *data_str)
{
    ucs_status_t status;

    if (UCS_PTR_IS_ERR(request)) {
        status = UCS_PTR_STATUS(request);
    } else if (UCS_PTR_IS_PTR(request)) {
        while (!request->completed) {
            ucp_worker_progress(ucp_worker);
        }

        request->completed = 0;
        status             = ucp_request_check_status(request);
        ucp_request_release(request);
    } else {
        status = UCS_OK;
    }

    if (status != UCS_OK) {
        fprintf(stderr, "unable to %s %s (%s)\n", op_str, data_str,
                ucs_status_string(status));
    } else {
        printf("finish to %s %s\n", op_str, data_str);
    }

    return status;
}

static ucs_status_t test_poll_wait(ucp_worker_h ucp_worker)
{
    int err            = 0;
    ucs_status_t ret   = UCS_ERR_NO_MESSAGE;
    int epoll_fd_local = 0;
    int epoll_fd       = 0;
    ucs_status_t status;
    struct epoll_event ev;
    ev.data.u64        = 0;

    status = ucp_worker_get_efd(ucp_worker, &epoll_fd);
    CHKERR_JUMP(UCS_OK != status, "ucp_worker_get_efd", err);

    /* It is recommended to copy original fd */
    epoll_fd_local = epoll_create(1);

    ev.data.fd = epoll_fd;
    ev.events = EPOLLIN;
    err = epoll_ctl(epoll_fd_local, EPOLL_CTL_ADD, epoll_fd, &ev);
    CHKERR_JUMP(err < 0, "add original socket to the new epoll\n", err_fd);

    /* Need to prepare ucp_worker before epoll_wait */
    status = ucp_worker_arm(ucp_worker);
    if (status == UCS_ERR_BUSY) { /* some events are arrived already */
        ret = UCS_OK;
        goto err_fd;
    }
    CHKERR_JUMP(status != UCS_OK, "ucp_worker_arm\n", err_fd);

    do {
        err = epoll_wait(epoll_fd_local, &ev, 1, -1);
    } while ((err == -1) && (errno == EINTR));

    ret = UCS_OK;

err_fd:
    close(epoll_fd_local);

err:
    return ret;
}

static int run_ucx_client(ucp_worker_h ucp_worker)
{
    struct msg *msg = NULL;
    size_t msg_len  = 0;
    int ret         = -1;
    ucp_request_param_t send_param;
    ucp_tag_recv_info_t info_tag;
    ucp_tag_message_h msg_tag;
    ucs_status_t status;
    ucp_ep_h server_ep;
    ucp_ep_params_t ep_params;
    struct ucx_context *request;
    char *str;

    /* Send client UCX address to server */
    ep_params.field_mask      = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                                UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.address         = peer_addr;
    ep_params.err_mode        = err_handling_opt.ucp_err_mode;

    status = ucp_ep_create(ucp_worker, &ep_params, &server_ep);
    CHKERR_JUMP(status != UCS_OK, "ucp_ep_create\n", err);

    msg_len = sizeof(*msg) + local_addr_len;
    msg     = malloc(msg_len);
    CHKERR_JUMP(msg == NULL, "allocate memory\n", err_ep);
    memset(msg, 0, msg_len);

    msg->data_len = local_addr_len;
    memcpy(msg + 1, local_addr, local_addr_len);

    send_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                              UCP_OP_ATTR_FIELD_USER_DATA;
    send_param.cb.send      = send_handler;
    send_param.user_data    = (void*)addr_msg_str;
    request                 = ucp_tag_send_nbx(server_ep, msg, msg_len, tag,
                                               &send_param);
    status                  = ucx_wait(ucp_worker, request, "send",
                                       addr_msg_str);
    if (status != UCS_OK) {
        free(msg);
        goto err_ep;
    }

    free(msg);

    if (err_handling_opt.failure) {
        fprintf(stderr, "Emulating unexpected failure on client side\n");
        raise(SIGKILL);
    }

    /* Receive test string from server */
    for (;;) {

        /* Probing incoming events in non-block mode */
        msg_tag = ucp_tag_probe_nb(ucp_worker, tag, tag_mask, 1, &info_tag);
        if (msg_tag != NULL) {
            /* Message arrived */
            break;
        } else if (ucp_worker_progress(ucp_worker)) {
            /* Some events were polled; try again without going to sleep */
            continue;
        }

        /* If we got here, ucp_worker_progress() returned 0, so we can sleep.
         * Following blocked methods used to polling internal file descriptor
         * to make CPU idle and don't spin loop
         */
        if (ucp_test_mode == TEST_MODE_WAIT) {
            /* Polling incoming events*/
            status = ucp_worker_wait(ucp_worker);
            CHKERR_JUMP(status != UCS_OK, "ucp_worker_wait\n", err_ep);
        } else if (ucp_test_mode == TEST_MODE_EVENTFD) {
            status = test_poll_wait(ucp_worker);
            CHKERR_JUMP(status != UCS_OK, "test_poll_wait\n", err_ep);
        }
    }

    msg = mem_type_malloc(info_tag.length);
    CHKERR_JUMP(msg == NULL, "allocate memory\n", err_ep);

    request = ucp_tag_msg_recv_nb(ucp_worker, msg, info_tag.length,
                                  ucp_dt_make_contig(1), msg_tag,
                                  recv_handler);
    status  = ucx_wait(ucp_worker, request, "receive", data_msg_str);
    if (status != UCS_OK) {
        mem_type_free(msg);
        goto err_ep;
    }

    str = calloc(1, test_string_length);
    if (str != NULL) {
        mem_type_memcpy(str, msg + 1, test_string_length);
        printf("\n\n----- UCP TEST SUCCESS ----\n\n");
        printf("%s", str);
        printf("\n\n---------------------------\n\n");
        free(str);
    } else {
        fprintf(stderr, "Memory allocation failed\n");
        mem_type_free(msg);
        goto err_ep;
    }

    mem_type_free(msg);

    ret = 0;

err_ep:
    ucp_ep_destroy(server_ep);

err:
    return ret;
}

static void flush_callback(void *request, ucs_status_t status, void *user_data)
{
}

static ucs_status_t flush_ep(ucp_worker_h worker, ucp_ep_h ep)
{
    ucp_request_param_t param;
    void *request;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
    param.cb.send      = flush_callback;
    request            = ucp_ep_flush_nbx(ep, &param);
    if (request == NULL) {
        return UCS_OK;
    } else if (UCS_PTR_IS_ERR(request)) {
        return UCS_PTR_STATUS(request);
    } else {
        ucs_status_t status;
        do {
            ucp_worker_progress(worker);
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        ucp_request_release(request);
        return status;
    }
}

static int run_ucx_server(ucp_worker_h ucp_worker)
{
    struct msg *msg             = NULL;
    struct ucx_context *request = NULL;
    size_t msg_len              = 0;
    ucp_request_param_t send_param;
    ucp_tag_recv_info_t info_tag;
    ucp_tag_message_h msg_tag;
    ucs_status_t status;
    ucp_ep_h client_ep;
    ucp_ep_params_t ep_params;
    int ret;

    /* Receive client UCX address */
    do {
        /* Progressing before probe to update the state */
        ucp_worker_progress(ucp_worker);

        /* Probing incoming events in non-block mode */
        msg_tag = ucp_tag_probe_nb(ucp_worker, tag, tag_mask, 1, &info_tag);
    } while (msg_tag == NULL);

    msg = malloc(info_tag.length);
    CHKERR_ACTION(msg == NULL, "allocate memory\n", ret = -1; goto err);
    request = ucp_tag_msg_recv_nb(ucp_worker, msg, info_tag.length,
                                  ucp_dt_make_contig(1), msg_tag, recv_handler);
    status  = ucx_wait(ucp_worker, request, "receive", addr_msg_str);
    if (status != UCS_OK) {
        free(msg);
        ret = -1;
        goto err;
    }

    peer_addr_len = msg->data_len;
    peer_addr     = malloc(peer_addr_len);
    if (peer_addr == NULL) {
        fprintf(stderr, "unable to allocate memory for peer address\n");
        free(msg);
        ret = -1;
        goto err;
    }

    memcpy(peer_addr, msg + 1, peer_addr_len);

    free(msg);

    /* Send test string to client */
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
    /* If peer failure testing was requested, it could be possible that UCP EP
     * couldn't be created; in this case set `ret = 0` to report success */
    CHKERR_ACTION(status != UCS_OK, "ucp_ep_create\n",
                  ret = (err_handling_opt.failure) ? 0 : -1; goto err);

    msg_len = sizeof(*msg) + test_string_length;
    msg = mem_type_malloc(msg_len);
    CHKERR_ACTION(msg == NULL, "allocate memory\n", ret = -1; goto err_ep);
    mem_type_memset(msg, 0, msg_len);

    set_msg_data_len(msg, msg_len - sizeof(*msg));
    ret = generate_test_string((char *)(msg + 1), test_string_length);
    CHKERR_JUMP(ret < 0, "generate test string", err_free_mem_type_msg);

    if (err_handling_opt.failure) {
        /* Sleep for small amount of time to ensure that server was killed
         * and peer failure handling is covered */
        sleep(5);
    }

    send_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK  |
                              UCP_OP_ATTR_FIELD_USER_DATA |
                              UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    send_param.cb.send      = send_handler;
    send_param.user_data    = (void*)data_msg_str;
    send_param.memory_type  = test_mem_type;
    request                 = ucp_tag_send_nbx(client_ep, msg, msg_len, tag,
                                               &send_param);
    status                  = ucx_wait(ucp_worker, request, "send",
                                       data_msg_str);
    if (status != UCS_OK) {
        if (!err_handling_opt.failure) {
            ret = -1;
        } else {
            /* If peer failure testing was requested, set `ret = 0` to report
             * success from the application */
            ret = 0;

            /* Make sure that failure_handler was called */
            while (client_status == UCS_OK) {
                ucp_worker_progress(ucp_worker);
            }
        }
        goto err_free_mem_type_msg;
    }

    status = flush_ep(ucp_worker, client_ep);
    printf("flush_ep completed with status %d (%s)\n",
           status, ucs_status_string(status));

    ret = 0;

err_free_mem_type_msg:
    mem_type_free(msg);
err_ep:
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
    ucp_params.features     = UCP_FEATURE_TAG;
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

        ret = recv(oob_sock, &addr_len, sizeof(addr_len), MSG_WAITALL);
        CHKERR_JUMP_RETVAL(ret != (int)sizeof(addr_len),
                           "receive address length\n", err_addr, ret);

        peer_addr_len = addr_len;
        peer_addr = malloc(peer_addr_len);
        CHKERR_JUMP(!peer_addr, "allocate memory\n", err_addr);

        ret = recv(oob_sock, peer_addr, peer_addr_len, MSG_WAITALL);
        CHKERR_JUMP_RETVAL(ret != (int)peer_addr_len,
                           "receive address\n", err_peer_addr, ret);
    } else {
        oob_sock = server_connect(server_port);
        CHKERR_JUMP(oob_sock < 0, "server_connect\n", err_peer_addr);

        addr_len = local_addr_len;
        ret = send(oob_sock, &addr_len, sizeof(addr_len), 0);
        CHKERR_JUMP_RETVAL(ret != (int)sizeof(addr_len),
                           "send address length\n", err_peer_addr, ret);

        ret = send(oob_sock, local_addr, local_addr_len, 0);
        CHKERR_JUMP_RETVAL(ret != (int)local_addr_len, "send address\n",
                           err_peer_addr, ret);
    }

    ret = run_test(client_target_name, ucp_worker);

    if (!ret && !err_handling_opt.failure) {
        /* Make sure remote is disconnected before destroying local worker */
        ret = barrier(oob_sock);
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

ucs_status_t parse_cmd(int argc, char * const argv[], char **server_name)
{
    int c = 0, idx = 0;
    opterr = 0;

    err_handling_opt.ucp_err_mode   = UCP_ERR_HANDLING_MODE_NONE;
    err_handling_opt.failure        = 0;

    while ((c = getopt(argc, argv, "wfben:p:s:m:h")) != -1) {
        switch (c) {
        case 'w':
            ucp_test_mode = TEST_MODE_WAIT;
            break;
        case 'f':
            ucp_test_mode = TEST_MODE_EVENTFD;
            break;
        case 'b':
            ucp_test_mode = TEST_MODE_PROBE;
            break;
        case 'e':
            err_handling_opt.ucp_err_mode   = UCP_ERR_HANDLING_MODE_PEER;
            err_handling_opt.failure        = 1;
            break;
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
        case 's':
            test_string_length = atol(optarg);
            if (test_string_length <= 0) {
                fprintf(stderr, "Wrong string size %ld\n", test_string_length);
                return UCS_ERR_UNSUPPORTED;
            }	
            break;
        case 'm':
            test_mem_type = parse_mem_type(optarg);
            if (test_mem_type == UCS_MEMORY_TYPE_LAST) {
                return UCS_ERR_UNSUPPORTED;
            }
            break;
        case '?':
            if (optopt == 's') {
                fprintf(stderr, "Option -%c requires an argument.\n", optopt);
            } else if (isprint (optopt)) {
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            } else {
                fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
            }
            /* Fall through */
        case 'h':
        default:
            fprintf(stderr, "Usage: ucp_hello_world [parameters]\n");
            fprintf(stderr, "UCP hello world client/server example utility\n");
            fprintf(stderr, "\nParameters are:\n");
            fprintf(stderr, "  -w      Select test mode \"wait\" to test "
                    "ucp_worker_wait function\n");
            fprintf(stderr, "  -f      Select test mode \"event fd\" to test "
                    "ucp_worker_get_efd function with later poll\n");
            fprintf(stderr, "  -b      Select test mode \"busy polling\" to test "
                    "ucp_tag_probe_nb and ucp_worker_progress (default)\n");
            fprintf(stderr, "  -e      Emulate unexpected failure on server side"
                    "and handle an error on client side with enabled "
                    "UCP_ERR_HANDLING_MODE_PEER\n");
            print_common_help();
            fprintf(stderr, "\n");
            return UCS_ERR_UNSUPPORTED;
        }
    }
    fprintf(stderr, "INFO: UCP_HELLO_WORLD mode = %d server = %s port = %d\n",
            ucp_test_mode, *server_name, server_port);

    for (idx = optind; idx < argc; idx++) {
        fprintf(stderr, "WARNING: Non-option argument %s\n", argv[idx]);
    }
    return UCS_OK;
}
