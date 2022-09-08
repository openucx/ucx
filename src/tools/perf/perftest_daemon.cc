/**
 * Copyright (C) NVIDIA 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucp/api/ucp.h>
#include <ucs/debug/log.h>
#include <tools/perf/api/libperf.h>

#include <cstdlib>
#include <cstring>
#include <csignal>
#include <netinet/in.h>
#include <unistd.h>
#include <list>
#include <queue>
#include <set>
#include <algorithm>


typedef struct ucp_perf ucp_perf_t;


typedef std::pair<void*, const void*> am_recv_op_t;


typedef struct ucp_perf_thread_context {
    ucp_perf_t                         *ucp;
    ucp_worker_h                       worker;
    ucp_listener_h                     listener;
    std::set<ucp_ep_h>                 unmatched_eps;
    ucp_ep_h                           ep;
    ucp_ep_h                           daemon_ep;
    struct sockaddr_storage            daemon_addr;
    std::queue<ucp_perf_daemon_req_t*> unhandled_daemon_reqs;
    std::queue<am_recv_op_t>           unhandled_am_recv_ops;
    ucp_perf_daemon_params_t           params;

    ucp_perf_thread_context() : ucp(NULL), worker(NULL), listener(NULL),
                                ep(NULL), daemon_ep(NULL)
    {
        memset(&daemon_addr, 0, sizeof(daemon_addr));
    }
} ucp_perf_thread_context_t;

struct ucp_perf {
    ucp_context_h                      context;
    ucp_perf_thread_context_t          *tctx;
    std::list<ucs_status_ptr_t>        reqs;
    std::queue<ucp_perf_daemon_req_t*> daemon_reqs;
    std::queue<ucp_perf_daemon_ack_t*> daemon_acks;
};


typedef struct {
    int                   completed;
    void                  *buffer;
    struct {
        ucp_perf_daemon_ack_t *daemon_ack;
        ucp_mem_h             memh;
    } perftest_op;
} request_t;


static unsigned thread_count         = 1;
static ucs_thread_mode_t thread_mode = UCS_THREAD_MODE_SINGLE;
static uint16_t port                 = 1338;
static int terminated                = 0;

static void request_init(void *request)
{
    request_t *context = (request_t*)request;

    context->completed              = 0;
    context->buffer                 = NULL;
    context->perftest_op.memh       = NULL;
    context->perftest_op.daemon_ack = NULL;
}

static void workers_destroy(ucp_perf_thread_context_t *tctx, unsigned count)
{
    unsigned i;

    for (i = 0; i < count; i++) {
        ucp_worker_destroy(tctx[i].worker);
    }
}

static void ep_close(ucp_perf_thread_context_t *tctx, ucp_ep_h ep, int force)
{
    unsigned mode = force ? UCP_EP_CLOSE_MODE_FORCE : UCP_EP_CLOSE_MODE_FLUSH;
    ucs_status_ptr_t req;
    std::set<ucp_ep_h>::iterator it;

    if (tctx->ep == ep) {
        tctx->ep = NULL;
        printf("closed ep %p connected to a perftest\n", ep);
    } else if (tctx->daemon_ep == ep) {
        tctx->daemon_ep = NULL;
        printf("closed ep %p connected to a daemon\n", ep);
    } else {
        it = tctx->unmatched_eps.find(ep);
        if (it != tctx->unmatched_eps.end()) {
            printf("closed unmatched ep %p\n", ep);
            tctx->unmatched_eps.erase(it);
        }
    }

    req = ucp_ep_close_nb(ep, mode);
    tctx->ucp->reqs.push_back(req);
}

static void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    ucp_perf_thread_context_t *tctx = (ucp_perf_thread_context_t*)arg;
    ep_close(tctx, ep, 1);
}

static void server_conn_handle_cb(ucp_conn_request_h conn_request, void *arg)
{
    ucp_perf_thread_context_t *tctx = (ucp_perf_thread_context_t*)arg;
    ucp_ep_params_t ep_params;
    ucs_status_t status;
    ucp_ep_h ep;

    ep_params.field_mask      = UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                UCP_EP_PARAM_FIELD_CONN_REQUEST |
                                UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.conn_request    = conn_request;
    ep_params.err_handler.cb  = err_cb;
    ep_params.err_handler.arg = tctx;

    status = ucp_ep_create(tctx->worker, &ep_params, &ep);
    if (status != UCS_OK) {
        ucs_error("failed to create an endpoint on the daemon: %s",
                  ucs_status_string(status));
    }

    tctx->unmatched_eps.insert(ep);
}

static UCS_F_ALWAYS_INLINE ucp_perf_daemon_req_t*
am_perf_daemon_req_alloc(ucp_perf_thread_context_t *tctx, size_t length)
{
    ucp_perf_t *ucp = tctx->ucp;
    ucp_perf_daemon_req_t *daemon_req;
    size_t daemon_req_size;

    if (ucp->daemon_reqs.empty()) {
        goto out;
    }

    daemon_req      = ucp->daemon_reqs.front();
    daemon_req_size =
            sizeof(*daemon_req) + daemon_req->exported_memh_buf_size +
            tctx->params.am_hdr_size;
    ucp->daemon_reqs.pop();

    if (daemon_req_size >= length) {
        return daemon_req;
    }

    delete [] (char*)daemon_req;

out:
    return (ucp_perf_daemon_req_t*)new char[length];
}

static UCS_F_ALWAYS_INLINE void
am_perf_daemon_req_release(ucp_perf_t *ucp, ucp_perf_daemon_req_t *daemon_req)
{
    ucp->daemon_reqs.push(daemon_req);
}

static UCS_F_ALWAYS_INLINE ucp_perf_daemon_ack_t*
am_perf_daemon_ack_alloc(ucp_perf_thread_context_t *tctx, uint8_t type,
                         uint8_t cmd, const void *am_header,
                         size_t am_header_length)
{
    ucp_perf_t *ucp = tctx->ucp;
    ucp_perf_daemon_ack_t *daemon_ack;
    size_t daemon_ack_size;

    ucs_assertv((am_header_length == 0) ||
                (am_header_length == tctx->params.am_hdr_size),
                "am_header_length=%zu params.am_hdr_size=%zu",
                am_header_length, tctx->params.am_hdr_size);

    if (ucp->daemon_acks.empty()) {
        daemon_ack_size = sizeof(*daemon_ack) + tctx->params.am_hdr_size;
        daemon_ack      = (ucp_perf_daemon_ack_t*)new char[daemon_ack_size];
        goto out;
    }

    daemon_ack = ucp->daemon_acks.front();
    ucp->daemon_acks.pop();

out:
    daemon_ack->type = type;
    daemon_ack->cmd  = cmd;
    if (am_header != NULL) {
        memcpy(daemon_ack + 1, am_header, am_header_length);
    }
    return daemon_ack;
}

static UCS_F_ALWAYS_INLINE void
am_perf_daemon_ack_release(ucp_perf_t *ucp, ucp_perf_daemon_ack_t *daemon_ack)
{
    ucp->daemon_acks.push(daemon_ack);
}

static inline void progress_reqs(ucp_perf_t *ucp)
{
    std::list<void*>::iterator itr;
    void *req;

    itr = ucp->reqs.begin();
    while (itr != ucp->reqs.end()) {
        req = *itr;
        if (req == NULL) {
            itr = ucp->reqs.erase(itr);
        } else if (UCS_PTR_IS_PTR(req)) {
            if (ucp_request_check_status(req) != UCS_INPROGRESS) {
                itr = ucp->reqs.erase(itr);
                ucp_request_release(req);
                continue;
            }
        } else if (UCS_PTR_STATUS(req) != UCS_OK) {
            ucs_warn("failed to complete req %p: %s", req,
                     ucs_status_string(UCS_PTR_STATUS(req)));
        }

        ++itr;
    }
}

static void progress(ucp_perf_t *ucp)
{
    unsigned i;

    for (i = 0; i < thread_count; i++) {
        ucp_worker_progress(ucp->tctx[i].worker);
    }

    progress_reqs(ucp);
}

static int set_am_recv_handler(ucp_worker_h worker, unsigned id,
                               ucp_am_recv_callback_t cb, void *arg)
{
    ucp_am_handler_param_t param;
    ucs_status_t status;

    param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                       UCP_AM_HANDLER_PARAM_FIELD_CB |
                       UCP_AM_HANDLER_PARAM_FIELD_ARG;
    param.id         = id;
    param.cb         = cb;
    param.arg        = arg;
    status           = ucp_worker_set_am_recv_handler(worker, &param);
    if (status != UCS_OK) {
        return -1;
    }

    return 0;
}

static void
send_daemon_ack_cb(void *request, ucs_status_t status, void *user_data)
{
    ucp_perf_thread_context_t *tctx   = (ucp_perf_thread_context_t*)user_data;
    request_t *req                    = (request_t*)request;
    ucp_perf_daemon_ack_t *daemon_ack = (ucp_perf_daemon_ack_t*)req->buffer;

    am_perf_daemon_ack_release(tctx->ucp, daemon_ack);
    ucp_request_free(request);
}

static void complete_daemon_req(ucp_perf_thread_context_t *tctx,
                                ucp_perf_daemon_ack_t *daemon_ack)
{
    ucp_request_param_t param = { 0 };
    ucs_status_ptr_t req;
    const void *am_header;
    size_t am_header_length;

    if ((ucp_perf_daemon_type_t)daemon_ack->type == UCP_PERF_DAEMON_RECEIVER) {
        am_header        = (const void*)(daemon_ack + 1);
        am_header_length = tctx->params.am_hdr_size;
    } else {
        am_header        = NULL;
        am_header_length = 0;
    }

    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                         UCP_OP_ATTR_FIELD_USER_DATA;
    param.cb.send      = send_daemon_ack_cb;
    param.user_data    = tctx;

    req = ucp_am_send_nbx(tctx->ep, UCP_PERF_DAEMON_AM_ID_ACK, am_header,
                          am_header_length, daemon_ack, sizeof(*daemon_ack),
                          &param);
    if (!UCS_PTR_IS_PTR(req)) {
        /* coverity[overflow] */
        if (UCS_PTR_STATUS(req) != UCS_OK) {
            ucs_error("AM sending of daemon ACK failed: %s",
                      ucs_status_string(UCS_PTR_STATUS(req)));
        }

        am_perf_daemon_ack_release(tctx->ucp, daemon_ack);
        return;
    }

    ((request_t*)req)->buffer = daemon_ack;
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
am_recv_data(ucp_perf_thread_context_t *tctx, void *desc, void *buffer,
             size_t length, ucp_mem_h memh,
             ucp_am_recv_data_nbx_callback_t recv_cb)
{
    ucp_request_param_t params;
    ucs_status_ptr_t req;

    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                          UCP_OP_ATTR_FIELD_DATATYPE |
                          UCP_OP_ATTR_FIELD_USER_DATA |
                          UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    params.datatype     = ucp_dt_make_contig(1);
    params.user_data    = tctx;
    params.cb.recv_am   = recv_cb;

    if (memh != NULL) {
        params.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
        params.memh          = memh;
    }

    req = ucp_am_recv_data_nbx(tctx->worker, desc, buffer, length, &params);
    ucs_assert(UCS_PTR_IS_PTR(req));

    ((request_t*)req)->buffer = buffer;

    return req;
}

static ucs_status_t
shared_mem_import(ucp_perf_thread_context_t *tctx,
                  ucp_perf_daemon_req_t *daemon_req, void **address_p,
                  ucp_mem_h *memh_p)
{
    ucp_mem_map_params_t params = { 0 };
    ucs_status_t status;
    ucp_mem_h memh;
    ucp_mem_attr_t attr;

    params.field_mask           = UCP_MEM_MAP_PARAM_FIELD_EXPORTED_MEMH_BUFFER;
    params.exported_memh_buffer = (void*)(daemon_req + 1);
    status                      = ucp_mem_map(tctx->ucp->context, &params,
                                              &memh);
    if (status != UCS_OK) {
        ucs_error("failed to import memory (%s)", ucs_status_string(status));
        goto out;
    }

    attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS;
    status          = ucp_mem_query(memh, &attr);
    if (status != UCS_OK) {
        goto out_mem_unmap;
    }

    *address_p = attr.address;
    *memh_p    = memh;
    return UCS_OK;

out_mem_unmap:
    ucp_mem_unmap(tctx->ucp->context, memh);
out:
    return status;
}

static void send_daemon_cb(void *request, ucs_status_t status, void *user_data)
{
    ucp_perf_thread_context_t *tctx = (ucp_perf_thread_context_t*)user_data;
    request_t *req                  = (request_t*)request;

    ucp_mem_unmap(tctx->ucp->context, req->perftest_op.memh);
    complete_daemon_req(tctx, req->perftest_op.daemon_ack);
    ucp_request_free(request);
}

static ucs_status_t
perform_send_cmd_daemon_req(ucp_perf_thread_context_t *tctx,
                            ucp_perf_daemon_req_t *daemon_req, void *address,
                            ucp_mem_h memh)
{
    ucx_perf_cmd_t cmd          = (ucx_perf_cmd_t)daemon_req->cmd;
    ucp_perf_daemon_type_t type = (ucp_perf_daemon_type_t)daemon_req->type;
    ucp_request_param_t param   = { 0 };
    ucp_perf_daemon_ack_t *daemon_ack;
    request_t *req;
    const void *am_header;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH |
                         UCP_OP_ATTR_FIELD_CALLBACK |
                         UCP_OP_ATTR_FIELD_USER_DATA |
                         UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    param.memh         = memh;
    param.cb.send      = send_daemon_cb;
    param.user_data    = tctx;

    daemon_ack = am_perf_daemon_ack_alloc(tctx, type, cmd, NULL, 0);

    switch (cmd) {
    case UCX_PERF_CMD_AM:
        am_header = UCS_PTR_BYTE_OFFSET(daemon_req + 1,
                                        daemon_req->exported_memh_buf_size);
        req       = (request_t*)ucp_am_send_nbx(tctx->daemon_ep,
                                                UCP_PERF_DAEMON_AM_ID_OP,
                                                am_header,
                                                tctx->params.am_hdr_size,
                                                address,
                                                (size_t)daemon_req->length,
                                                &param);
        break;
    default:
        ucs_fatal("unsupported command: %d", cmd);
    }

    if (!UCS_PTR_IS_PTR(req)) {
        complete_daemon_req(tctx, daemon_ack);
        return UCS_PTR_STATUS(req);
    }

    req->perftest_op.memh       = memh;
    req->perftest_op.daemon_ack = daemon_ack;

    return UCS_INPROGRESS;
}

static void am_recv_daemon_cb(void *request, ucs_status_t am_status,
                              size_t length, void *user_data)
{
    ucp_perf_thread_context_t *tctx = (ucp_perf_thread_context_t*)user_data;
    request_t *req                  = (request_t*)request;

    ucp_mem_unmap(tctx->ucp->context, req->perftest_op.memh);
    complete_daemon_req(tctx, req->perftest_op.daemon_ack);
    ucp_request_free(request);
}

static ucs_status_t
perform_recv_cmd_daemon_req(ucp_perf_thread_context_t *tctx,
                            ucp_perf_daemon_req_t *daemon_req,
                            void *address, ucp_mem_h memh, int *release_p)
{
    ucx_perf_cmd_t cmd          = (ucx_perf_cmd_t)daemon_req->cmd;
    ucp_perf_daemon_type_t type = (ucp_perf_daemon_type_t)daemon_req->type;
    const void *am_header       = NULL;
    size_t am_header_length     = 0;
    ucp_perf_daemon_ack_t *daemon_ack;
    request_t *req;
    void *am_recv_desc;

    switch (cmd) {
    case UCX_PERF_CMD_AM:
        if (tctx->unhandled_am_recv_ops.empty()) {
            tctx->unhandled_daemon_reqs.push(daemon_req);
            *release_p = 0;
            return UCS_INPROGRESS;
        }

        am_recv_desc     = tctx->unhandled_am_recv_ops.front().first;
        am_header        = tctx->unhandled_am_recv_ops.front().second;
        am_header_length = tctx->params.am_hdr_size;
        tctx->unhandled_am_recv_ops.pop();

        req = (request_t*)am_recv_data(tctx, am_recv_desc, address,
                                       (size_t)daemon_req->length, memh,
                                       am_recv_daemon_cb);
        break;
    default:
        ucs_fatal("unsupported command: %d", cmd);
    }

    daemon_ack = am_perf_daemon_ack_alloc(tctx, type, cmd, am_header,
                                          am_header_length);

    if (!UCS_PTR_IS_PTR(req)) {
        complete_daemon_req(tctx, daemon_ack);
        return UCS_PTR_STATUS(req);
    }

    req->perftest_op.memh       = memh;
    req->perftest_op.daemon_ack = daemon_ack;

    return UCS_INPROGRESS;
}

static void perform_cmd_daemon_req(ucp_perf_thread_context_t *tctx,
                                   ucp_perf_daemon_req_t *daemon_req,
                                   int *release_p)
{
    ucp_perf_daemon_type_t type = (ucp_perf_daemon_type_t)daemon_req->type;
    ucs_status_t status;
    void *address;
    ucp_mem_h memh;

    status = shared_mem_import(tctx, daemon_req, &address, &memh);
    if (status != UCS_OK) {
        return;
    }

    switch (type) {
    case UCP_PERF_DAEMON_SENDER:
        status = perform_send_cmd_daemon_req(tctx, daemon_req, address, memh);
        break;
    case UCP_PERF_DAEMON_RECEIVER:
        status = perform_recv_cmd_daemon_req(tctx, daemon_req, address, memh,
                                             release_p);
        break;
    default:
        ucs_fatal("unsupported type of command: %d", type);
    }

    if (status != UCS_INPROGRESS) {
        if (UCS_STATUS_IS_ERR(status)) {
            ucs_error("operation (type %u, cmd %u) failed: %s",
                      daemon_req->type, daemon_req->cmd,
                      ucs_status_string(status));
        }
        ucp_mem_unmap(tctx->ucp->context, memh);
    }
}

static void handle_daemon_req(ucp_perf_thread_context_t *tctx,
                              ucp_perf_daemon_req_t *daemon_req,
                              int release)
{
    perform_cmd_daemon_req(tctx, daemon_req, &release);

    if (release) {
        am_perf_daemon_req_release(tctx->ucp, daemon_req);
    }
}

static void am_recv_perf_daemon_request_free(void *request)
{

    delete [] (char*)((request_t*)request)->buffer;
    ucp_request_free(request);
}

static void am_recv_perf_daemon_init_cb(void *request, ucs_status_t am_status,
                                        size_t length, void *user_data)
{
    ucp_perf_thread_context_t *tctx = (ucp_perf_thread_context_t*)user_data;
    ucp_ep_params_t ep_params       = { 0 };
    ucp_request_param_t request_params;
    ucs_status_t status;
    ucs_status_ptr_t req;
    struct sockaddr_storage *daemon_addr;
    ucp_perf_daemon_init_t *daemon_init;

    if (am_status != UCS_OK) {
        ucs_error("failed to receive daemon initialization information from "
                  " a perftest: %s", ucs_status_string(am_status));
        if (request != NULL) {
            am_recv_perf_daemon_request_free(request);
        }
        return;
    }

    if (request != NULL) {
        daemon_init = (ucp_perf_daemon_init_t*)((request_t*)request)->buffer;
        daemon_addr = (struct sockaddr_storage*)(daemon_init + 1);

        if (tctx->daemon_ep != NULL) {
            if (memcmp(daemon_addr, &tctx->daemon_addr, length) == 0) {
                am_recv_perf_daemon_request_free(request);
                return;
            } else {
                ep_close(tctx, tctx->daemon_ep, 0);
            }
        }

        memcpy(&tctx->params, &daemon_init->params,
               sizeof(daemon_init->params));
        memcpy(&tctx->daemon_addr, daemon_addr, length);
        am_recv_perf_daemon_request_free(request);
    }

    ep_params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS       |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR   |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb   = err_cb;
    ep_params.err_handler.arg  = tctx;
    ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr    = (struct sockaddr*)&tctx->daemon_addr;
    ep_params.sockaddr.addrlen = sizeof(tctx->daemon_addr);

    status = ucp_ep_create(tctx->worker, &ep_params, &tctx->daemon_ep);
    if (status != UCS_OK) {
        ucs_error("failed to create an endpoint on the daemon: %s",
                  ucs_status_string(status));
    }

    printf("created ep %p to communicate with daemon\n", tctx->daemon_ep);

    request_params.op_attr_mask = UCP_OP_ATTR_FLAG_NO_IMM_CMPL |
                                  UCP_OP_ATTR_FIELD_FLAGS;
    request_params.flags        = UCP_AM_SEND_FLAG_REPLY;

    req = ucp_am_send_nbx(tctx->daemon_ep, UCP_PERF_DAEMON_AM_ID_PEER_INIT,
                          NULL, 0, NULL, 0, &request_params);
    tctx->ucp->reqs.push_back(req);
}

static ucs_status_t
am_cb_init_ep(ucp_perf_thread_context_t *tctx, ucp_ep_h *ep_storage,
              const char *ep_peer_type, const ucp_am_recv_param_t *param)
{
    std::set<ucp_ep_h>::iterator it;
    ucp_ep_h ep;

    ucs_assert(param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP);

    ep = param->reply_ep;
    it = tctx->unmatched_eps.find(ep);
    if (it == tctx->unmatched_eps.end()) {
        ucs_error("no ep %p contains in the unmatched endpoints", ep);
        goto err_ep_close;
    } else {
        tctx->unmatched_eps.erase(it);
    }

    if (*ep_storage != NULL) {
        ucs_error("ep %p to %s has already been created on the daemon, ep %p "
                  "won't be used", *ep_storage, ep_peer_type, ep);
        goto err_ep_close;
    } else {
        printf("created ep %p to accept connection from %s\n", ep,
               ep_peer_type);
    }

    *ep_storage = ep;
    return UCS_OK;

err_ep_close:
    ep_close(tctx, ep, 1);
    return UCS_ERR_NOT_CONNECTED;
}

static ucs_status_t
am_perf_daemon_init_cb(void *arg, const void *header, size_t header_length,
                       void *data, size_t length,
                       const ucp_am_recv_param_t *param)
{
    ucp_perf_thread_context_t *tctx = (ucp_perf_thread_context_t*)arg;
    ucs_status_t status;
    ucp_perf_daemon_init_t *daemon_init;

    ucs_assertv(header_length == 0, "header_length=%lu", header_length);

    status = am_cb_init_ep(tctx, &tctx->ep, "a perftest client", param);
    if (status != UCS_OK) {
        status = UCS_OK;
        goto out;
    }

    ucs_assertv((length == sizeof(*daemon_init)) ||
                (length == (sizeof(*daemon_init) +
                            sizeof(struct sockaddr_storage))),
                "length=%lu", length);
    if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) {
        daemon_init = (ucp_perf_daemon_init_t*)new char[length];
        if (daemon_init == NULL) {
            status = UCS_ERR_NO_MEMORY;
            goto out;
        }

        am_recv_data(tctx, data, daemon_init, length, NULL,
                     am_recv_perf_daemon_init_cb);
        status = UCS_INPROGRESS;
        goto out;
    }

    daemon_init  = (ucp_perf_daemon_init_t*)data;
    memcpy(&tctx->params, &daemon_init->params, sizeof(daemon_init->params));

    if (daemon_init->daemon_peer_addr_length > 0) {
        if (tctx->daemon_ep != NULL) {
            if (memcmp(daemon_init + 1, &tctx->daemon_addr,
                       daemon_init->daemon_peer_addr_length) == 0) {
                status = UCS_OK;
                goto out;
            } else {
                ep_close(tctx, tctx->daemon_ep, 0);
            }
        }

        memcpy(&tctx->daemon_addr, daemon_init + 1,
               daemon_init->daemon_peer_addr_length);
        am_recv_perf_daemon_init_cb(NULL, UCS_OK, length, tctx);
    }

    status = UCS_OK;

out:
    return status;
}

static ucs_status_t
am_perf_daemon_peer_init_cb(void *arg, const void *header,
                            size_t header_length, void *data, size_t length,
                            const ucp_am_recv_param_t *param)
{
    ucp_perf_thread_context_t *tctx = (ucp_perf_thread_context_t*)arg;
    ucp_perf_daemon_req_t *daemon_req;
    ucs_status_t UCS_V_UNUSED status;
    size_t size, i;

    ucs_assertv(header_length == 0, "header_length=%lu", header_length);
    ucs_assertv(length == 0, "length=%lu", length);

    status = am_cb_init_ep(tctx, &tctx->daemon_ep, "a daemon", param);
    ucs_assertv((status == UCS_OK) || (status == UCS_ERR_NOT_CONNECTED), "%s",
                ucs_status_string(status));

    i    = 0;
    size = tctx->unhandled_daemon_reqs.size();
    while (i++ != size) {
        daemon_req = tctx->unhandled_daemon_reqs.front();
        tctx->unhandled_daemon_reqs.pop();
        handle_daemon_req(tctx, daemon_req, 1);
    }

    return UCS_OK;
}

static void am_recv_perf_daemon_req_cb(void *request, ucs_status_t am_status,
                                       size_t length, void *user_data)
{
    ucp_perf_thread_context_t *tctx = (ucp_perf_thread_context_t*)user_data;
    ucp_perf_daemon_req_t *daemon_req;

    if (am_status != UCS_OK) {
        ucs_error("failed to receive daemon request from a perftest: %s",
                  ucs_status_string(am_status));
        if (request != NULL) {
            am_recv_perf_daemon_request_free(request);
        }
        return;
    }

    daemon_req = (ucp_perf_daemon_req_t*)((request_t*)request)->buffer;
    if (tctx->daemon_ep != NULL) {
        handle_daemon_req(tctx, daemon_req, 1);
    } else {
        tctx->unhandled_daemon_reqs.push(daemon_req);
    }

    ucp_request_free(request);
}

static void am_perf_daemon_req_set(ucp_perf_thread_context_t *tctx,
                                   ucp_perf_daemon_req_t *daemon_req,
                                   size_t length, const void *header,
                                   size_t header_length)
{
    ucs_assertv((header_length == 0) ||
                (header_length == tctx->params.am_hdr_size),
                "header_length=%zu params.am_hdr_size=%zu", header_length,
                tctx->params.am_hdr_size);
    memcpy(UCS_PTR_BYTE_OFFSET(daemon_req, length), header, header_length);
}

static ucs_status_t
am_perf_daemon_req_cb(void *arg, const void *header, size_t header_length,
                      void *data, size_t length,
                      const ucp_am_recv_param_t *param)
{
    ucp_perf_thread_context_t *tctx = (ucp_perf_thread_context_t*)arg;
    ucs_status_t status;
    ucp_perf_daemon_req_t *daemon_req;

    ucs_assertv(length >= sizeof(*daemon_req), "length=%lu", length);

    daemon_req = am_perf_daemon_req_alloc(tctx, length + header_length);
    if (daemon_req == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    am_perf_daemon_req_set(tctx, daemon_req, length, header, header_length);

    if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) {
        am_recv_data(tctx, data, daemon_req, length, NULL,
                     am_recv_perf_daemon_req_cb);
        status = UCS_INPROGRESS;
    } else {
        memcpy(daemon_req, data, length);
        status = UCS_OK;

        if (tctx->daemon_ep != NULL) {
            handle_daemon_req(tctx, daemon_req, 1);
        } else {
            tctx->unhandled_daemon_reqs.push(daemon_req);
        }
    }

out:
    return status;
}

static ucs_status_t
am_perf_daemon_recv_op_cb(void *arg, const void *header, size_t header_length,
                          void *data, size_t length,
                          const ucp_am_recv_param_t *param)
{
    ucp_perf_thread_context_t *tctx = (ucp_perf_thread_context_t*)arg;
    ucp_perf_daemon_req_t *daemon_req;

    if (!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)) {
        ucs_error("unsupported to receive data as not a RNDV");
        return UCS_OK;
    }

    tctx->unhandled_am_recv_ops.push(std::make_pair(data, header));
    if ((tctx->daemon_ep == NULL) ||
        tctx->unhandled_daemon_reqs.empty()) {
        return UCS_INPROGRESS;
    }

    daemon_req = tctx->unhandled_daemon_reqs.front();
    tctx->unhandled_daemon_reqs.pop();

    handle_daemon_req(tctx, daemon_req, 1);

    return UCS_INPROGRESS;
}

static void cleanup(ucp_perf_t *ucp)
{
    unsigned i;
    std::set<ucp_ep_h>::iterator unmatched_ep_it;
    ucp_perf_daemon_req_t *daemon_req;
    ucp_perf_daemon_ack_t *daemon_ack;

    for (i = 0; i < thread_count; i++) {
        ucp_listener_destroy(ucp->tctx[i].listener);

        if (ucp->tctx[i].ep != NULL) {
            ep_close(&ucp->tctx[i], ucp->tctx[i].ep, 0);
        }

        if (ucp->tctx[i].daemon_ep != NULL) {
            ep_close(&ucp->tctx[i], ucp->tctx[i].daemon_ep, 0);
        }

        unmatched_ep_it = ucp->tctx[i].unmatched_eps.begin();
        while (unmatched_ep_it != ucp->tctx[i].unmatched_eps.end()) {
            ucp_ep_h ep = *unmatched_ep_it++;
            ep_close(&ucp->tctx[i], ep, 0);
        }

        while (!ucp->tctx[i].unhandled_daemon_reqs.empty()) {
            ucp_perf_daemon_req_t *daemon_req =
                    ucp->tctx[i].unhandled_daemon_reqs.front();

            ucp->tctx[i].unhandled_daemon_reqs.pop();
            am_perf_daemon_req_release(ucp, daemon_req);
        }

        while (!ucp->tctx[i].unhandled_am_recv_ops.empty()) {
            void *recv_data = ucp->tctx[i].unhandled_am_recv_ops.front().first;

            ucp->tctx[i].unhandled_am_recv_ops.pop();
            ucp_am_data_release(ucp->tctx[i].worker, recv_data);
        }
    }

    while (!ucp->reqs.empty()) {
        progress(ucp);
    }

    for (i = 0; i < thread_count; i++) {
        ucp_worker_destroy(ucp->tctx[i].worker);
    }

    delete [] ucp->tctx;

    while (!ucp->daemon_reqs.empty()) {
        daemon_req = ucp->daemon_reqs.front();
        ucp->daemon_reqs.pop();

        delete [] (char*)daemon_req;
    }

    while (!ucp->daemon_acks.empty()) {
        daemon_ack = ucp->daemon_acks.front();
        ucp->daemon_acks.pop();

        delete [] (char*)daemon_ack;
    }

    ucp_cleanup(ucp->context);
}

static int init(ucp_perf_t *ucp)
{
    ucp_params_t ucp_params;
    ucp_worker_params_t worker_params;
    struct sockaddr_in listen_addr;
    ucp_listener_params_t listen_params;
    ucp_config_t *config;
    ucs_status_t status;
    unsigned i;
    int ret;

    memset(&listen_addr, 0, sizeof(listen_addr));
    listen_addr.sin_family      = AF_INET;
    listen_addr.sin_addr.s_addr = INADDR_ANY;
    listen_addr.sin_port        = htons(port);

    ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES |
                              UCP_PARAM_FIELD_REQUEST_SIZE |
                              UCP_PARAM_FIELD_REQUEST_INIT;
    ucp_params.features     = UCP_FEATURE_AM | UCP_FEATURE_EXPORTED_MEMH;
    ucp_params.request_size = sizeof(request_t);
    ucp_params.request_init = request_init;

    if (thread_count > 1) {
        /* when there is more than one thread, a ucp_worker would be created for
         * each. all of them will share the same ucp_context */
        ucp_params.field_mask       |= UCP_PARAM_FIELD_MT_WORKERS_SHARED;
        ucp_params.mt_workers_shared = 1;
    }

    status = ucp_config_read(NULL, NULL, &config);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucp_init(&ucp_params, config, &ucp->context);
    ucp_config_release(config);
    if (status != UCS_OK) {
        ucs_error("failed to init UCP: %s", ucs_status_string(status));
        goto err;
    }

    ucp->tctx = new ucp_perf_thread_context_t[thread_count];
    if (ucp->tctx == NULL) {
        ucs_error("failed to allocate memory for thread context");
        goto err_cleanup;
    }

    for (i = 0; i < thread_count; i++) {
        ucp->tctx[i].ucp = ucp;
    }

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = thread_mode;

    for (i = 0; i < thread_count; i++) {
        status = ucp_worker_create(ucp->context, &worker_params,
                                   &ucp->tctx[i].worker);
        if (status != UCS_OK) {
            ucs_error("failed to create worker: %s",
                      ucs_status_string(status));
            workers_destroy(ucp->tctx, i);
            goto err_free_tctx;
        }

        ret = set_am_recv_handler(ucp->tctx[i].worker,
                                  UCP_PERF_DAEMON_AM_ID_INIT,
                                  am_perf_daemon_init_cb, &ucp->tctx[i]);
        if (ret != 0) {
            workers_destroy(ucp->tctx, i + 1);
            goto err_free_tctx;
        }

        ret = set_am_recv_handler(ucp->tctx[i].worker,
                                  UCP_PERF_DAEMON_AM_ID_PEER_INIT,
                                  am_perf_daemon_peer_init_cb, &ucp->tctx[i]);
        if (ret != 0) {
            workers_destroy(ucp->tctx, i + 1);
            goto err_free_tctx;
        }

        ret = set_am_recv_handler(ucp->tctx[i].worker,
                                  UCP_PERF_DAEMON_AM_ID_REQ,
                                  am_perf_daemon_req_cb, &ucp->tctx[i]);
        if (ret != 0) {
            workers_destroy(ucp->tctx, i + 1);
            goto err_free_tctx;
        }

        ret = set_am_recv_handler(ucp->tctx[i].worker,
                                  UCP_PERF_DAEMON_AM_ID_OP,
                                  am_perf_daemon_recv_op_cb, &ucp->tctx[i]);
        if (ret != 0) {
            workers_destroy(ucp->tctx, i + 1);
            goto err_free_tctx;
        }
    }

    for (i = 0; i < thread_count; i++) {    
        listen_params.field_mask       = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                         UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
        listen_params.sockaddr.addr    = (const struct sockaddr*)&listen_addr;
        listen_params.sockaddr.addrlen = sizeof(listen_addr);
        listen_params.conn_handler.cb  = server_conn_handle_cb;
        listen_params.conn_handler.arg = &ucp->tctx[i];

        status = ucp_listener_create(ucp->tctx[i].worker, &listen_params,
                                     &ucp->tctx[i].listener);
        if (status != UCS_OK) {
            ucs_error("failed to listen: %s", ucs_status_string(status));
            goto err_workers_destroy;
        }
    }

    return 0;

err_workers_destroy:
    workers_destroy(ucp->tctx, thread_count);
err_free_tctx:
    delete [] ucp->tctx;
err_cleanup:
    ucp_cleanup(ucp->context);
err:
    return -1;
}

static void signal_terminate_handler(int signo)
{
    char msg[64];
    ssize_t ret __attribute__((unused));

    snprintf(msg, sizeof(msg), "Run-time signal handling: %d\n", signo);
    ret = write(STDOUT_FILENO, msg, strlen(msg) + 1);

    terminated = 1;
}

static int parse_cmd(int argc, char *const argv[])
{
    int c = 0;

    while ((c = getopt(argc, argv, "p:")) != -1) {
        switch (c) {
        case 'p':
            port = (uint16_t)atoi(optarg);
            break;
        default:
            return -1;
        }
    }

    return 0;
}

int main(int argc, char *const argv[])
{
    ucp_perf_t ucp;
    struct sigaction new_sigaction;

    new_sigaction.sa_handler = signal_terminate_handler;
    new_sigaction.sa_flags   = 0;
    sigemptyset(&new_sigaction.sa_mask);

    if (parse_cmd(argc, argv) != 0) {
        abort();
    }

    if (sigaction(SIGINT, &new_sigaction, NULL) != 0) {
        ucs_error("failed to set signal handler for SIGINT");
        abort();
    }

    if (init(&ucp) != 0) {
        abort();
    }

    while (!terminated) {
        progress(&ucp);
    }

    cleanup(&ucp);

    return 0;
}
