/**
 * Copyright (C) NVIDIA 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucp/api/ucp.h>
#include <ucs/debug/log.h>
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
    ucp_ep_h                           client_ep;
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

static void
ucp_perf_daemon_set_am_recv_handler(ucp_worker_h worker, unsigned id,
                                    ucp_am_recv_callback_t cb, void *arg)
{
    ucp_am_handler_param_t param;
    ucs_status_t status;

    param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID    |
                       UCP_AM_HANDLER_PARAM_FIELD_CB    |
                       UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
                       UCP_AM_HANDLER_PARAM_FIELD_ARG;
    param.id         = id;
    param.cb         = cb;
    param.arg        = arg;
    param.flags      = UCP_AM_FLAG_WHOLE_MSG;
    status           = ucp_worker_set_am_recv_handler(worker, &param);

    ucs_assertv_always(status == UCS_OK, "status=%s", ucs_status_string(status));
}

static void ucp_perf_daemon_send_am_eager_msg(ucp_ep_h ep, unsigned am_id)
{
    ucp_request_param_t param = {};
    ucs_status_ptr_t sptr;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    param.flags        = UCP_AM_SEND_FLAG_REPLY | UCP_AM_SEND_FLAG_EAGER;

    sptr = ucp_am_send_nbx(ep, am_id, NULL, 0ul, NULL, 0ul, &param);
    if (UCS_PTR_IS_PTR(sptr)) {
        ucp_request_free(sptr);
    } else if (UCS_PTR_IS_ERR(sptr)) {
        ucs_fatal("failed to send am id %u: %s", am_id,
                  ucs_status_string(UCS_PTR_STATUS(sptr)));
    }
}

static ucp_mem_h
ucp_perf_daemon_memh_import(ucp_perf_daemon_context_t *ctx, void *packed_memh)
{
    ucp_mem_map_params_t params = {};
    ucs_status_t status;
    ucp_mem_h memh;

    params.field_mask           = UCP_MEM_MAP_PARAM_FIELD_EXPORTED_MEMH_BUFFER;
    params.exported_memh_buffer = packed_memh;
    status                      = ucp_mem_map(ctx->context, &params,
                                              &memh);
    if (status != UCS_OK) {
        ucs_error("failed to import memory (%s)", ucs_status_string(status));
        return NULL;
    }

    return memh;
}

static void ucp_perf_daemon_send_cb(void *request, ucs_status_t status,
                                    void *user_data)
{
    ucp_perf_daemon_context_t *ctx = user_data;

    ucp_perf_daemon_send_am_eager_msg(ctx->client_ep,
                                      UCP_PERF_DAEMON_AM_ID_SEND_ACK);
    ucp_request_free(request);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_perf_daemon_handle_cmd_am_send(ucp_perf_daemon_context_t *ctx,
                                   ucp_perf_daemon_req_t *daemon_req)
{
    ucp_request_param_t param = {};
    ucs_status_ptr_t sptr;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH      |
                         UCP_OP_ATTR_FIELD_CALLBACK  |
                         UCP_OP_ATTR_FIELD_USER_DATA |
                         UCP_OP_ATTR_FIELD_FLAGS     |
                         UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    param.memh         = ctx->send_memh;
    param.cb.send      = ucp_perf_daemon_send_cb;
    param.user_data    = ctx;
    param.flags        = UCP_AM_SEND_FLAG_RNDV;

    sptr = ucp_am_send_nbx(ctx->peer_ep, UCP_PERF_DAEMON_AM_ID_OP, NULL, 0ul,
                           (void*)daemon_req->addr, (size_t)daemon_req->length,
                           &param);

    return UCS_PTR_STATUS(sptr);
}

static void ucp_perf_daemon_recv_cb(void *request, ucs_status_t status,
                                    size_t length, void *user_data)
{
    ucp_perf_daemon_context_t *ctx = user_data;

    ucp_perf_daemon_send_am_eager_msg(ctx->client_ep,
                                      UCP_PERF_DAEMON_AM_ID_RECV_ACK);
    ucp_request_free(request);
}

static void ucp_perf_daemon_ep_create(ucp_perf_daemon_context_t *ctx)
{
    ucp_ep_params_t ep_params;
    ucs_status_t status;

    ep_params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS       |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR   |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb   = ucp_perf_daemon_err_cb;
    ep_params.err_handler.arg  = ctx;
    ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr    = (struct sockaddr*)&ctx->peer_address;
    ep_params.sockaddr.addrlen = sizeof(ctx->peer_address);

    status = ucp_ep_create(ctx->worker, &ep_params, &ctx->peer_ep);
    if (status != UCS_OK) {
        ucs_error("daemon failed to create an endpoint: %s",
                  ucs_status_string(status));
    }

    ucp_perf_daemon_send_am_eager_msg(ctx->peer_ep,
                                      UCP_PERF_DAEMON_AM_ID_PEER_INIT);
}

static void ucp_perf_daemon_check_am_msg(const ucp_am_recv_param_t *param,
                                         size_t header_length,
                                         int check_reply_ep)
{
    ucs_assert(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    ucs_assertv(header_length == 0, "header_length %zu", header_length);

    if (check_reply_ep) {
        ucs_assert(param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP);
    }
}

static ucs_status_t
ucp_perf_daemon_init_handler(void *arg, const void *header,
                             size_t header_length, void *data, size_t length,
                             const ucp_am_recv_param_t *param)
{
    ucp_perf_daemon_context_t *ctx = arg;
    void *p                        = data;
    ucs_status_t status;
    uint16_t address_length;
    uint16_t send_memh_buf_size;
    ucp_mem_attr_t attr;

    ucp_perf_daemon_check_am_msg(param, header_length, 1);

    if (ctx->peer_ep != NULL) {
        ucs_fatal("duplicate daemon init req");
        goto out;
    }

    ctx->client_ep = param->reply_ep;

    address_length = *ucs_serialize_next(&p, uint16_t);

    if (address_length != 0) {
        memcpy(&ctx->peer_address,
               ucs_serialize_next_raw(&p, void, address_length),
               address_length);
        ucp_perf_daemon_ep_create(ctx);
    }

    send_memh_buf_size = *ucs_serialize_next(&p, uint16_t);
    ctx->send_memh     = ucp_perf_daemon_memh_import(ctx, p);
    ucs_serialize_next_raw(&p, void, send_memh_buf_size);
    ucs_serialize_next(&p, uint16_t); /*recv_memh_buf_size*/
    ctx->recv_memh     = ucp_perf_daemon_memh_import(ctx, p);

    attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS;
    status          = ucp_mem_query(ctx->recv_memh, &attr);
    if (status != UCS_OK) {
        ucs_fatal("daemon failed to query memh: %s", ucs_status_string(status));
    }

    ctx->rx_address = attr.address;

out:
    return UCS_OK;
}

static ucs_status_t
ucp_perf_daemon_init_peer_handler(void *arg, const void *header,
                                  size_t header_length, void *data,
                                  size_t length,
                                  const ucp_am_recv_param_t *param)
{
    ucp_perf_daemon_context_t *ctx = arg;

    ucp_perf_daemon_check_am_msg(param, header_length, 1);

    ctx->peer_ep = param->reply_ep;

    return UCS_OK;
}

static ucs_status_t
ucp_perf_daemon_req_handler(void *arg, const void *header, size_t header_length,
                            void *data, size_t length,
                            const ucp_am_recv_param_t *param)
{
    ucp_perf_daemon_context_t *ctx = arg;
    ucp_perf_daemon_req_t *dreq    = data;
    ucs_status_t status;

    ucp_perf_daemon_check_am_msg(param, header_length, 0);
    ucs_assertv(length >= sizeof(*dreq), "length=%lu", length);
    ucs_assert(ctx->peer_ep != NULL);

    status = ucp_perf_daemon_handle_cmd_am_send(ctx, dreq);

    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        ucs_error("operation failed: %s", ucs_status_string(status));
    }

    return UCS_OK;
}

static ucs_status_t
ucp_perf_daemon_op_handler(void *arg, const void *header, size_t header_length,
                           void *data, size_t length,
                           const ucp_am_recv_param_t *param)
{
    ucp_perf_daemon_context_t *ctx = arg;
    ucp_request_param_t params;
    ucs_status_ptr_t sptr;

    if (!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)) {
        ucs_error("am message received with unsupported eager protocol");
        return UCS_OK;
    }

    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK  |
                          UCP_OP_ATTR_FIELD_USER_DATA |
                          UCP_OP_ATTR_FIELD_MEMH      |
                          UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    params.user_data    = ctx;
    params.cb.recv_am   = ucp_perf_daemon_recv_cb;
    params.memh         = ctx->recv_memh;

    sptr = ucp_am_recv_data_nbx(ctx->worker, data, ctx->rx_address, length,
                                &params);
    if (UCS_PTR_IS_ERR(sptr)) {
        ucs_error("failed to receive data: %s",
                  ucs_status_string(UCS_PTR_STATUS(sptr)));
        return UCS_OK;
    }

    ucs_assert(UCS_PTR_IS_PTR(sptr));

    return UCS_INPROGRESS;
}

static ucs_status_t
ucp_perf_daemon_fin_handler(void *arg, const void *header, size_t header_length,
                            void *data, size_t length,
                            const ucp_am_recv_param_t *param)
{
    ucp_perf_daemon_check_am_msg(param, header_length, 0);

    terminated = 1;

    return UCS_OK;
}

static void ucp_perf_daemon_cleanup(ucp_perf_daemon_context_t *ctx)
{
    /* coverity[check_return] */
    ucp_mem_unmap(ctx->context, ctx->send_memh);
    /* coverity[check_return] */
    ucp_mem_unmap(ctx->context, ctx->recv_memh);

    ucp_perf_daemon_ep_close(ctx, ctx->peer_ep);
    ucp_perf_daemon_ep_close(ctx, ctx->client_ep);
    ucp_listener_destroy(ctx->listener);
    ucp_worker_destroy(ctx->worker);
    ucp_cleanup(ctx->context);
}

static int ucp_perf_daemon_init(ucp_perf_daemon_context_t *ctx)
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

    ucp_perf_daemon_set_am_recv_handler(ctx->worker, UCP_PERF_DAEMON_AM_ID_INIT,
                                        ucp_perf_daemon_init_handler, ctx);

    ucp_perf_daemon_set_am_recv_handler(ctx->worker,
                                        UCP_PERF_DAEMON_AM_ID_PEER_INIT,
                                        ucp_perf_daemon_init_peer_handler, ctx);

    ucp_perf_daemon_set_am_recv_handler(ctx->worker, UCP_PERF_DAEMON_AM_ID_REQ,
                                        ucp_perf_daemon_req_handler, ctx);

    ucp_perf_daemon_set_am_recv_handler(ctx->worker, UCP_PERF_DAEMON_AM_ID_OP,
                                        ucp_perf_daemon_op_handler, ctx);

    ucp_perf_daemon_set_am_recv_handler(ctx->worker, UCP_PERF_DAEMON_AM_ID_FIN,
                                        ucp_perf_daemon_fin_handler, ctx);

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

    return 0;

err_free_worker:
    ucp_worker_destroy(ctx->worker);
err_free_ctx:
    ucp_cleanup(ctx->context);
err:
    return -1;
}

static void ucp_perf_daemon_signal_terminate_handler(int signo)
{
    char msg[64];
    ssize_t ret __attribute__((unused));

    snprintf(msg, sizeof(msg), "Run-time signal handling: %d\n", signo);
    ret = write(STDOUT_FILENO, msg, strlen(msg) + 1);

    terminated = 1;
}

static int ucp_perf_daemon_parse_cmd(ucp_perf_daemon_context_t *ctx, int argc,
                                     char *const argv[])
{
    int c = 0;

    while ((c = getopt(argc, argv, "p:")) != -1) {
        switch (c) {
        case 'p':
            ctx->port = (uint16_t)atoi(optarg);
            break;
        default:
            return -1;
        }
    }

    return 0;
}

int main(int argc, char *const argv[])
{
    ucp_perf_daemon_context_t ctx = {};
    struct sigaction new_sigaction;

    ctx.port = 1338; /* default value */

    if (ucp_perf_daemon_parse_cmd(&ctx, argc, argv) != 0) {
        ucs_fatal("failed to parse parameters");
    }

    new_sigaction.sa_handler = ucp_perf_daemon_signal_terminate_handler;
    new_sigaction.sa_flags   = 0;
    sigemptyset(&new_sigaction.sa_mask);

    sigaction(SIGINT, &new_sigaction, NULL);
    sigaction(SIGHUP, &new_sigaction, NULL);
    sigaction(SIGTERM, &new_sigaction, NULL);

    if (ucp_perf_daemon_init(&ctx) != 0) {
        ucs_fatal("failed to initalize");
    }

    while (!terminated) {
        ucp_worker_progress(ctx.worker);
    }

    ucp_perf_daemon_cleanup(&ctx);

    return 0;
}

