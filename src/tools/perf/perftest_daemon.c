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
#include <ucp/core/ucp_mm.h>
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
    sa_family_t                        ai_family;
    uint16_t                           port;
    ucp_ep_h                           host_ep;
    ucp_ep_h                           peer_ep;
    struct sockaddr_storage            peer_address;
    ucp_mem_h                          send_memh;
    ucp_mem_h                          recv_memh;
} ucp_perf_daemon_context_t;

static volatile int terminated = 0;

const char *ucp_perf_daemon_am_id_name(ucp_perf_daemon_am_id_t am_id)
{
#define UCP_PERF_DAEMON_NAME_CASE(ID, _) \
    case ID: \
        return #ID;

    switch (am_id) {
        UCP_FOREACH_PERF_DAEMON_AM_ID(UCP_PERF_DAEMON_NAME_CASE);
    default:
        return "UNKNOWN";
    }
}

static void
ucp_perf_daemon_ep_close(ucp_perf_daemon_context_t *ctx, ucp_ep_h ep)
{
    ucp_request_param_t param = {};
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

    ep_params.field_mask      = UCP_EP_PARAM_FIELD_ERR_HANDLER |
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

static ucs_status_t
ucp_perf_daemon_set_am_recv_handler(ucp_perf_daemon_context_t *ctx,
                                    ucp_perf_daemon_am_id_t am_id,
                                    ucp_am_recv_callback_t cb)
{
    ucp_am_handler_param_t param;
    ucs_status_t status;

    param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                       UCP_AM_HANDLER_PARAM_FIELD_CB |
                       UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
                       UCP_AM_HANDLER_PARAM_FIELD_ARG;
    param.id         = am_id;
    param.cb         = cb;
    param.flags      = UCP_AM_FLAG_WHOLE_MSG;
    param.arg        = ctx;

    status = ucp_worker_set_am_recv_handler(ctx->worker, &param);
    if (UCS_OK != status) {
        ucs_error("failed to set am recv handler: %s",
                  ucs_status_string(status));
        return status;
    }

    return UCS_OK;
}

static ucs_status_t
ucp_perf_daemon_send_am_eager_reply(ucp_ep_h ep, ucp_perf_daemon_am_id_t am_id)
{
    ucp_request_param_t param = {};
    ucs_status_ptr_t sreq;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    param.flags        = UCP_AM_SEND_FLAG_REPLY | UCP_AM_SEND_FLAG_EAGER;

    sreq = ucp_am_send_nbx(ep, am_id, NULL, 0ul, NULL, 0ul, &param);
    if (UCS_PTR_IS_PTR(sreq)) {
        ucp_request_free(sreq);
    } else if (UCS_PTR_IS_ERR(sreq)) {
        ucs_error("failed to send am id %u: %s", am_id,
                  ucs_status_string(UCS_PTR_STATUS(sreq)));
        return UCS_PTR_STATUS(sreq);
    }

    return UCS_OK;
}

static void
ucp_perf_daemon_send_cb(void *request, ucs_status_t status, void *user_data)
{
    ucp_perf_daemon_context_t *ctx = user_data;

    ucp_perf_daemon_send_am_eager_reply(ctx->host_ep,
                                        UCP_PERF_DAEMON_AM_ID_SEND_CMPL);
    ucp_request_free(request);
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_perf_daemon_handle_request(
        ucp_perf_daemon_context_t *ctx, ucp_perf_daemon_req_t *req)
{
    ucp_request_param_t param = {};
    ucs_status_ptr_t sptr;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH |
                         UCP_OP_ATTR_FIELD_CALLBACK |
                         UCP_OP_ATTR_FIELD_USER_DATA |
                         UCP_OP_ATTR_FIELD_FLAGS |
                         UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    param.memh         = ctx->send_memh;
    param.cb.send      = ucp_perf_daemon_send_cb;
    param.user_data    = ctx;
    param.flags        = UCP_AM_SEND_FLAG_RNDV;

    sptr = ucp_am_send_nbx(ctx->peer_ep, UCP_PERF_DAEMON_AM_ID_PEER_TX, NULL,
                           0ul, (void*)req->addr, (size_t)req->length, &param);

    return UCS_PTR_STATUS(sptr);
}

static void ucp_perf_daemon_recv_cb(void *request, ucs_status_t status,
                                    size_t length, void *user_data)
{
    ucp_perf_daemon_context_t *ctx = user_data;

    ucp_perf_daemon_send_am_eager_reply(ctx->host_ep,
                                        UCP_PERF_DAEMON_AM_ID_RECV_CMPL);
    ucp_request_free(request);
}

static ucs_status_t
ucp_perf_daemon_create_peer_ep(ucp_perf_daemon_context_t *ctx,
                               const void *address, size_t address_length)
{
    ucp_ep_params_t ep_params;
    ucs_status_t status;

    memcpy(&ctx->peer_address, address, address_length);

    ep_params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR |
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
        return status;
    }

    return ucp_perf_daemon_send_am_eager_reply(ctx->peer_ep,
                                               UCP_PERF_DAEMON_AM_ID_PEER_INIT);
}

static void ucp_perf_daemon_check_am_msg(const ucp_am_recv_param_t *param,
                                         size_t header_length,
                                         int check_reply_ep, int check_rndv,
                                         ucp_perf_daemon_am_id_t am_id)
{
    ucs_trace_data("message received: %s", ucp_perf_daemon_am_id_name(am_id));

    ucs_assertv(check_rndv == !!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV),
                "am message received with unsupported %s protocol",
                (check_rndv ? "eager" : "rndv"));

    ucs_assertv(header_length == 0, "header_length %zu", header_length);

    ucs_assertv(check_reply_ep ==
                        !!(param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP),
                "am message received %s reply EP",
                (check_reply_ep ? "without" : "with"));
}

static ucp_mem_h
ucp_perf_daemon_memh_import(ucp_perf_daemon_context_t *ctx, void *packed_memh)
{
    ucp_mem_map_params_t params = {};
    ucs_status_t status;
    ucp_mem_h memh;

    params.field_mask           = UCP_MEM_MAP_PARAM_FIELD_EXPORTED_MEMH_BUFFER;
    params.exported_memh_buffer = packed_memh;

    status = ucp_mem_map(ctx->context, &params, &memh);
    if (status != UCS_OK) {
        ucs_error("failed to import memory (%s)", ucs_status_string(status));
        return NULL;
    }

    return memh;
}

/*
 * Unpack array length and array contents from a buffer.
 *
 * @param [in]  ptr    Pointer to the buffer.
 * @param [out] length Length of the array.
 * @param [out] value  Pointer to the array.
 */
static void ucp_perf_unpack_array(void **ptr, uint64_t *length, void **value)
{
    *length = *ucs_serialize_next(ptr, uint64_t);
    *value = (length != 0) ? ucs_serialize_next_raw(ptr, void, *length) : NULL;
}

static ucs_status_t
ucp_perf_daemon_init_handler(void *arg, const void *header,
                             size_t header_length, void *data, size_t length,
                             const ucp_am_recv_param_t *param)
{
    ucp_perf_daemon_context_t *ctx = arg;
    void *ptr                      = data;
    void *value;
    uint64_t value_length;
    ucs_status_t status;

    ucp_perf_daemon_check_am_msg(param, header_length, 1, 0,
                                 UCP_PERF_DAEMON_AM_ID_INIT);

    /* Peer EP must not be initialized on receiving HOST_INIT message.
     * Otherwise it means that daemon was already initialized from host. For now
     * we don't support multiple host processes, or connection reestablishment
     */
    if (ctx->peer_ep != NULL) {
        ucs_error("duplicate daemon init req");
        return UCS_ERR_ALREADY_EXISTS;
    }

    ctx->host_ep = param->reply_ep;

    ucp_perf_unpack_array(&ptr, &value_length, &value);
    if (value_length != 0) {
        status = ucp_perf_daemon_create_peer_ep(ctx, value, value_length);
        if (status != UCS_OK) {
            return status;
        }
    }

    ucp_perf_unpack_array(&ptr, &value_length, &value);
    ctx->send_memh = ucp_perf_daemon_memh_import(ctx, value);
    ucp_perf_unpack_array(&ptr, &value_length, &value);
    ctx->recv_memh = ucp_perf_daemon_memh_import(ctx, value);

    ucs_assertv_always(UCS_PTR_BYTE_DIFF(data, ptr) <= length,
                       "data=%p ptr=%p length=%zu", data, ptr, length);
    return UCS_OK;
}

static ucs_status_t
ucp_perf_daemon_send_handler(void *arg, const void *header,
                             size_t header_length, void *data, size_t length,
                             const ucp_am_recv_param_t *param)
{
    ucp_perf_daemon_context_t *ctx = arg;
    ucp_perf_daemon_req_t *req     = data;
    ucs_status_t status;

    ucp_perf_daemon_check_am_msg(param, header_length, 0, 0,
                                 UCP_PERF_DAEMON_AM_ID_SEND_REQ);
    ucs_assertv(length >= sizeof(*req), "length=%lu", length);
    ucs_assert(ctx->peer_ep != NULL);

    status = ucp_perf_daemon_handle_request(ctx, req);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        ucs_error("operation failed: %s", ucs_status_string(status));
    }

    return UCS_OK;
}

static ucs_status_t
ucp_perf_daemon_fin_handler(void *arg, const void *header, size_t header_length,
                            void *data, size_t length,
                            const ucp_am_recv_param_t *param)
{
    ucp_perf_daemon_check_am_msg(param, header_length, 0, 0,
                                 UCP_PERF_DAEMON_AM_ID_FIN);

    terminated = 1;
    return UCS_OK;
}

static ucs_status_t
ucp_perf_daemon_peer_init_handler(void *arg, const void *header,
                                  size_t header_length, void *data,
                                  size_t length,
                                  const ucp_am_recv_param_t *param)
{
    ucp_perf_daemon_context_t *ctx = arg;

    ucp_perf_daemon_check_am_msg(param, header_length, 1, 0,
                                 UCP_PERF_DAEMON_AM_ID_PEER_INIT);

    ctx->peer_ep = param->reply_ep;
    return UCS_OK;
}

static ucs_status_t
ucp_perf_daemon_peer_tx_handler(void *arg, const void *header,
                                size_t header_length, void *data, size_t length,
                                const ucp_am_recv_param_t *param)
{
    ucp_perf_daemon_context_t *ctx = arg;
    ucp_request_param_t params;
    ucs_status_ptr_t sptr;

    ucp_perf_daemon_check_am_msg(param, header_length, 0, 1,
                                 UCP_PERF_DAEMON_AM_ID_PEER_TX);

    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                          UCP_OP_ATTR_FIELD_USER_DATA |
                          UCP_OP_ATTR_FIELD_MEMH |
                          UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    params.user_data    = ctx;
    params.cb.recv_am   = ucp_perf_daemon_recv_cb;
    params.memh         = ctx->recv_memh;

    sptr = ucp_am_recv_data_nbx(ctx->worker, data,
                                ucp_memh_address(ctx->recv_memh), length,
                                &params);
    if (UCS_PTR_IS_ERR(sptr)) {
        ucs_error("failed to receive data: %s",
                  ucs_status_string(UCS_PTR_STATUS(sptr)));
        return UCS_PTR_STATUS(sptr);
    }

    ucs_assert(UCS_PTR_IS_PTR(sptr));
    return UCS_INPROGRESS;
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

static ucs_status_t
ucp_perf_daemon_set_am_handlers(ucp_perf_daemon_context_t *ctx)
{
    struct {
        ucp_perf_daemon_am_id_t id;
        ucp_am_recv_callback_t  cb;
    } handlers[] = {
        {UCP_PERF_DAEMON_AM_ID_INIT, ucp_perf_daemon_init_handler},
        {UCP_PERF_DAEMON_AM_ID_SEND_REQ, ucp_perf_daemon_send_handler},
        {UCP_PERF_DAEMON_AM_ID_FIN, ucp_perf_daemon_fin_handler},
        {UCP_PERF_DAEMON_AM_ID_PEER_INIT, ucp_perf_daemon_peer_init_handler},
        {UCP_PERF_DAEMON_AM_ID_PEER_TX, ucp_perf_daemon_peer_tx_handler},
    };
    size_t i;
    ucs_status_t status;

    for (i = 0; i < ucs_static_array_size(handlers); ++i) {
        status = ucp_perf_daemon_set_am_recv_handler(ctx, handlers[i].id,
                                                     handlers[i].cb);
        if (UCS_OK != status) {
            return status;
        }
    }

    return UCS_OK;
}

static ucs_status_t ucp_perf_daemon_init(ucp_perf_daemon_context_t *ctx)
{
    ucp_listener_params_t listen_params = {};
    ucp_worker_params_t worker_params   = {};
    struct sockaddr_storage listen_addr = {};
    ucp_params_t ucp_params;
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

    status = ucp_perf_daemon_set_am_handlers(ctx);
    if (status != UCS_OK) {
        goto err_free_worker;
    }

    listen_addr.ss_family = ctx->ai_family;
    ucs_sockaddr_set_inaddr_any((struct sockaddr*)&listen_addr, ctx->ai_family);
    ucs_sockaddr_set_port((struct sockaddr*)&listen_addr, ctx->port);

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
        case '6':
            ctx->ai_family = AF_INET6;
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
    printf("     -6             Use IPv6 address for in data exchange\n");
    printf("\n");
}

int main(int argc, char *const argv[])
{
    ucp_perf_daemon_context_t ctx = {};
    struct sigaction new_sigaction;

    ctx.port      = DEFAULT_DAEMON_PORT; /* default value */
    ctx.ai_family = AF_INET;

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
        ucs_error("failed to initialize");
        return EXIT_FAILURE;
    }

    while (!terminated) {
        ucp_worker_progress(ctx.worker);
    }

    ucp_perf_daemon_cleanup(&ctx);
    return EXIT_SUCCESS;
}
