/**
 * Copyright (C) NVIDIA 2024.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "perftest.h"

#include <ucp/api/ucp.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sock.h>
#include <ucs/sys/string.h>
#include <ucs/type/serialize.h>
#include <tools/perf/api/libperf.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <signal.h>

typedef struct ucp_perf_daemon_context_t {
    ucp_context_h                      context;
    ucp_worker_h                       worker;
    ucp_listener_h                     listener;
    uint16_t                           port;
    ucp_ep_h                           host_ep;
    ucp_ep_h                           peer_ep;
    struct sockaddr_storage            peer_address;
    ucp_mem_h                          send_memh;
    ucp_mem_h                          recv_memh;
    void                               *rx_address;
} ucp_perf_daemon_context_t;

static volatile int terminated = 0;

static void
ucp_perf_daemon_ep_close(ucp_perf_daemon_context_t *ctx, ucp_ep_h ep)
{
    ucp_request_param_t param;
    ucs_status_ptr_t close_req;
    ucs_status_t status;

    if (NULL == ep) {
        return;
    }

    param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    param.flags        = UCP_EP_CLOSE_FLAG_FORCE;
    close_req          = ucp_ep_close_nbx(ep, &param);

    if (UCS_PTR_IS_PTR(close_req)) {
        do {
            ucp_worker_progress(ctx->worker);
            status = ucp_request_check_status(close_req);
        } while (status == UCS_INPROGRESS);
        ucp_request_free(close_req);
    } else {
        status = UCS_PTR_STATUS(close_req);
    }

    if (status != UCS_OK) {
        ucs_error("daemon failed to close ep %p", ep);
    }
}

static void ucp_perf_daemon_err_cb(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    terminated = 1;
}

static void
ucp_perf_daemon_server_conn_handle_cb(ucp_conn_request_h conn_request,
                                      void *arg)
{
    ucp_perf_daemon_context_t *ctx = arg;
    ucp_ep_params_t ep_params;
    ucs_status_t status;
    ucp_ep_h ep;

    ep_params.field_mask      = UCP_EP_PARAM_FIELD_ERR_HANDLER  |
                                UCP_EP_PARAM_FIELD_CONN_REQUEST |
                                UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.conn_request    = conn_request;
    ep_params.err_handler.cb  = ucp_perf_daemon_err_cb;
    ep_params.err_handler.arg = ctx;

    status = ucp_ep_create(ctx->worker, &ep_params, &ep);
    if (status != UCS_OK) {
        ucs_error("failed to create an endpoint on the daemon: %s",
                  ucs_status_string(status));
    }
}

static void ucp_perf_daemon_cleanup(ucp_perf_daemon_context_t *ctx)
{
    if (NULL != ctx->send_memh) {
        /* coverity[check_return] */
        ucp_mem_unmap(ctx->context, ctx->send_memh);
    }

    if (NULL != ctx->recv_memh) {
        /* coverity[check_return] */
        ucp_mem_unmap(ctx->context, ctx->recv_memh);
    }

    ucp_perf_daemon_ep_close(ctx, ctx->peer_ep);
    ucp_perf_daemon_ep_close(ctx, ctx->host_ep);
    ucp_listener_destroy(ctx->listener);
    ucp_worker_destroy(ctx->worker);
    ucp_cleanup(ctx->context);
}

static ucs_status_t ucp_perf_daemon_init(ucp_perf_daemon_context_t *ctx)
{
    ucp_listener_params_t listen_params = {};
    ucp_worker_params_t worker_params   = {};
    ucp_params_t ucp_params;
    struct sockaddr_in listen_addr;
    ucp_config_t *config;
    ucs_status_t status;

    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features   = UCP_FEATURE_AM | UCP_FEATURE_EXPORTED_MEMH;

    status = ucp_config_read(NULL, NULL, &config);
    if (status != UCS_OK) {
        ucs_error("daemon failed to read UCP context: %s",
                  ucs_status_string(status));
        goto err;
    }

    status = ucp_init(&ucp_params, config, &ctx->context);
    ucp_config_release(config);
    if (status != UCS_OK) {
        ucs_error("daemon failed to init UCP: %s", ucs_status_string(status));
        goto err;
    }

    status = ucp_worker_create(ctx->context, &worker_params, &ctx->worker);
    if (status != UCS_OK) {
        ucs_error("failed to create worker: %s", ucs_status_string(status));
        goto err_free_ctx;
    }

    listen_addr.sin_family      = AF_INET;
    listen_addr.sin_addr.s_addr = INADDR_ANY;
    listen_addr.sin_port        = htons(ctx->port);

    listen_params.field_mask       = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                     UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    listen_params.sockaddr.addr    = (const struct sockaddr*)&listen_addr;
    listen_params.sockaddr.addrlen = sizeof(listen_addr);
    listen_params.conn_handler.cb  = ucp_perf_daemon_server_conn_handle_cb;
    listen_params.conn_handler.arg = ctx;

    status = ucp_listener_create(ctx->worker, &listen_params, &ctx->listener);
    if (status != UCS_OK) {
        ucs_error("failed to listen: %s", ucs_status_string(status));
        goto err_free_worker;
    }

    return UCS_OK;

err_free_worker:
    ucp_worker_destroy(ctx->worker);
err_free_ctx:
    ucp_cleanup(ctx->context);
err:
    return status;
}

static void ucp_perf_daemon_signal_terminate_handler(int signo)
{
    char msg[64];
    ssize_t ret __attribute__((unused));

    snprintf(msg, sizeof(msg), "Run-time signal handling: %d\n", signo);
    ret = write(STDOUT_FILENO, msg, strlen(msg) + 1);

    terminated = 1;
}

static ucs_status_t ucp_perf_daemon_parse_cmd(ucp_perf_daemon_context_t *ctx,
                                              int argc, char *const argv[])
{
    int c = 0;

    while ((c = getopt(argc, argv, "p:")) != -1) {
        switch (c) {
        case 'p':
            if (UCS_OK != ucs_sock_port_from_string(optarg, &ctx->port)) {
                return UCS_ERR_INVALID_PARAM;
            }
            break;
        default:
            return UCS_ERR_INVALID_PARAM;
        }
    }

    return UCS_OK;
}

static void usage(const ucp_perf_daemon_context_t *ctx, const char *program)
{
    printf("  Usage: %s [ options ]\n", program);
    printf("\n");
    printf("  Common options:\n");
    printf("\n");
    printf("     -p <port>      TCP port to use for data exchange (%d)\n"
           "                    default value: (%d)\n",
           ctx->port, DEFAULT_DAEMON_PORT);
    printf("\n");
}

int main(int argc, char *const argv[])
{
    ucp_perf_daemon_context_t ctx = {};
    struct sigaction new_sigaction;

    ctx.port = DEFAULT_DAEMON_PORT; /* default value */

    if (ucp_perf_daemon_parse_cmd(&ctx, argc, argv) != UCS_OK) {
        usage(&ctx, ucs_basename(argv[0]));
        return EXIT_FAILURE;
    }

    new_sigaction.sa_handler = ucp_perf_daemon_signal_terminate_handler;
    new_sigaction.sa_flags   = 0;
    sigemptyset(&new_sigaction.sa_mask);

    sigaction(SIGINT, &new_sigaction, NULL);
    sigaction(SIGHUP, &new_sigaction, NULL);
    sigaction(SIGTERM, &new_sigaction, NULL);

    if (ucp_perf_daemon_init(&ctx) != UCS_OK) {
        ucs_error("failed to initalize");
        return EXIT_FAILURE;
    }

    while (!terminated) {
        ucp_worker_progress(ctx.worker);
    }

    ucp_perf_daemon_cleanup(&ctx);
    return EXIT_SUCCESS;
}
