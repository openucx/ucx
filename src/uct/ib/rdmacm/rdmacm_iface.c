/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rdmacm_iface.h"
#include "rdmacm_ep.h"
#include <uct/base/uct_worker.h>
#include <ucs/sys/string.h>


enum uct_rdmacm_process_event_flags {
    UCT_RDMACM_PROCESS_EVENT_DESTROY_CM_ID_FLAG = UCS_BIT(0),
    UCT_RDMACM_PROCESS_EVENT_ACK_EVENT_FLAG     = UCS_BIT(1)
};

static ucs_config_field_t uct_rdmacm_iface_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_rdmacm_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"BACKLOG", "1024",
     "Maximum number of pending connections for an rdma_cm_id.",
     ucs_offsetof(uct_rdmacm_iface_config_t, backlog), UCS_CONFIG_TYPE_UINT},

    {"CM_ID_QUOTA", "64",
     "How many rdma_cm connections can progress simultaneously.",
     ucs_offsetof(uct_rdmacm_iface_config_t, cm_id_quota), UCS_CONFIG_TYPE_UINT},

    {NULL}
};

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_rdmacm_iface_t, uct_iface_t);

static ucs_status_t uct_rdmacm_iface_query(uct_iface_h tl_iface,
                                           uct_iface_attr_t *iface_attr)
{
    uct_rdmacm_iface_t *rdmacm_iface = ucs_derived_of(tl_iface, uct_rdmacm_iface_t);
    struct sockaddr *addr;
    ucs_status_t status;

    uct_base_iface_query(&rdmacm_iface->super, iface_attr);

    iface_attr->iface_addr_len  = sizeof(ucs_sock_addr_t);
    iface_attr->device_addr_len = 0;
    iface_attr->cap.flags       = UCT_IFACE_FLAG_CONNECT_TO_SOCKADDR |
                                  UCT_IFACE_FLAG_CB_ASYNC            |
                                  UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;
    /* User's private data size is UCT_RDMACM_UDP_PRIV_DATA_LEN minus room for
     * the private_data header (to hold the length of the data) */
    iface_attr->max_conn_priv   = UCT_RDMACM_MAX_CONN_PRIV;

    if (rdmacm_iface->is_server) {
        addr   = rdma_get_local_addr(rdmacm_iface->cm_id);
        status = ucs_sockaddr_copy((struct sockaddr *)&iface_attr->listen_sockaddr,
                                   addr);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

static ucs_status_t uct_rdmacm_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    ucs_sock_addr_t *rdmacm_addr = (ucs_sock_addr_t *)iface_addr;

    rdmacm_addr->addr    = NULL;
    rdmacm_addr->addrlen = 0;
    return UCS_OK;
}

static ucs_status_t uct_rdmacm_accept(struct rdma_cm_id *id)
{
    /* The server will not send any reply data back to the client */
    struct rdma_conn_param conn_param = {0};

    /* Accepting the connection will generate the RDMA_CM_EVENT_ESTABLISHED
     * event on the client side. */
    if (rdma_accept(id, &conn_param)) {
        ucs_error("rdma_accept(to id=%p) failed: %m", id);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static ucs_status_t uct_rdmacm_iface_accept(uct_iface_h tl_iface,
                                            uct_conn_request_h conn_request)
{
    struct rdma_cm_event *event = conn_request;
    ucs_status_t         status;

    ucs_trace("accepting event %p with id %p", event, event->id);
    status = uct_rdmacm_accept(event->id);
    rdma_destroy_id(event->id);
    rdma_ack_cm_event(event);

    return status;
}

static ucs_status_t uct_rdmacm_iface_reject(uct_iface_h tl_iface,
                                            uct_conn_request_h conn_request)
{
    struct rdma_cm_event       *event = conn_request;
    ucs_status_t               status = UCS_OK;
    uct_rdmacm_priv_data_hdr_t hdr    = {
        .length = 0,
        .status = (uint8_t)UCS_ERR_REJECTED
    };

    ucs_trace("rejecting event %p with id %p", event, event->id);
    if (rdma_reject(event->id, &hdr, sizeof(hdr))) {
        ucs_warn("rdma_reject(id=%p) failed: %m", event->id);
        status = UCS_ERR_IO_ERROR;
    }

    rdma_destroy_id(event->id);
    rdma_ack_cm_event(event);
    return status;
}

static ucs_status_t uct_rdmacm_ep_flush(uct_ep_h tl_ep, unsigned flags,
                                        uct_completion_t *comp)
{
    uct_rdmacm_ep_t    *ep = ucs_derived_of(tl_ep, uct_rdmacm_ep_t);
    ucs_status_t       status;
    uct_rdmacm_ep_op_t *op;

    pthread_mutex_lock(&ep->ops_mutex);
    status = ep->status;
    if ((status == UCS_INPROGRESS) && (comp != NULL)) {
        op = ucs_malloc(sizeof(*op), "uct_rdmacm_ep_flush op");
        if (op != NULL) {
            op->user_comp = comp;
            ucs_queue_push(&ep->ops, &op->queue_elem);
        } else {
            status = UCS_ERR_NO_MEMORY;
        }
    }
    pthread_mutex_unlock(&ep->ops_mutex);

    return status;
}

static uct_iface_ops_t uct_rdmacm_iface_ops = {
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_rdmacm_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_ep_t),
    .ep_flush                 = uct_rdmacm_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_pending_purge         = ucs_empty_function,
    .iface_accept             = uct_rdmacm_iface_accept,
    .iface_reject             = uct_rdmacm_iface_reject,
    .iface_progress_enable    = (uct_iface_progress_enable_func_t)ucs_empty_function_return_success,
    .iface_progress_disable   = (uct_iface_progress_disable_func_t)ucs_empty_function_return_success,
    .iface_progress           = ucs_empty_function_return_zero,
    .iface_flush              = uct_base_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_iface_t),
    .iface_query              = uct_rdmacm_iface_query,
    .iface_is_reachable       = (uct_iface_is_reachable_func_t)ucs_empty_function_return_zero,
    .iface_get_device_address = (uct_iface_get_device_address_func_t)ucs_empty_function_return_success,
    .iface_get_address        = uct_rdmacm_iface_get_address
};

ucs_status_t uct_rdmacm_resolve_addr(struct rdma_cm_id *cm_id,
                                     struct sockaddr *addr, int timeout_ms,
                                     ucs_log_level_t log_level)
{
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];

    if (rdma_resolve_addr(cm_id, NULL, addr, timeout_ms)) {
        ucs_log(log_level, "rdma_resolve_addr(addr=%s) failed: %m",
                ucs_sockaddr_str(addr, ip_port_str, UCS_SOCKADDR_STRING_LEN));
        return UCS_ERR_IO_ERROR;
    }
    return UCS_OK;
}

void uct_rdmacm_iface_client_start_next_ep(uct_rdmacm_iface_t *iface)
{
    ucs_status_t status;
    uct_rdmacm_ep_t *ep, *tmp;

    UCS_ASYNC_BLOCK(iface->super.worker->async);

    /* try to start an ep from the pending eps list */
    ucs_list_for_each_safe(ep, tmp, &iface->pending_eps_list, list_elem) {
        status = uct_rdmacm_ep_set_cm_id(iface, ep);
        if (status != UCS_OK) {
            continue;
        }

        ucs_list_del(&ep->list_elem);
        ep->is_on_pending = 0;

        status = uct_rdmacm_ep_resolve_addr(ep);
        if (status == UCS_OK) {
            break;
        }

        uct_rdmacm_ep_set_failed(&iface->super.super, &ep->super.super, status);
    }

    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
}

static void uct_rdmacm_client_handle_failure(uct_rdmacm_iface_t *iface,
                                             uct_rdmacm_ep_t *ep,
                                             ucs_status_t status)
{
    ucs_assert(!iface->is_server);
    if (ep != NULL) {
        pthread_mutex_lock(&ep->ops_mutex);
        uct_rdmacm_ep_set_failed(&iface->super.super, &ep->super.super, status);
        uct_rdmacm_ep_invoke_completions(ep, status);
        pthread_mutex_unlock(&ep->ops_mutex);
    }
}

static void uct_rdmacm_iface_process_conn_req(uct_rdmacm_iface_t *iface,
                                              struct rdma_cm_event *event,
                                              struct sockaddr *remote_addr)
{
    uct_rdmacm_priv_data_hdr_t *hdr;

    hdr = (uct_rdmacm_priv_data_hdr_t*) event->param.ud.private_data;
    ucs_assert(hdr->status == UCS_OK);

    /* TODO check the iface's cb_flags to determine when to invoke this callback.
     * currently only UCT_CB_FLAG_ASYNC is supported so the cb is invoked from here */
    iface->conn_request_cb(&iface->super.super, iface->conn_request_arg,
                           /* connection request*/
                           event,
                           /* private data */
                           UCS_PTR_BYTE_OFFSET(event->param.ud.private_data,
                                               sizeof(uct_rdmacm_priv_data_hdr_t)),
                           /* length */
                           hdr->length);
}

/**
 * Release a cm_id. This function should be called when the async context
 * is locked.
 */
static void uct_rdmacm_iface_release_cm_id(uct_rdmacm_iface_t *iface,
                                           uct_rdmacm_ctx_t **cm_id_ctx_p)
{
    uct_rdmacm_ctx_t *cm_id_ctx = *cm_id_ctx_p;

    ucs_trace("destroying cm_id %p", cm_id_ctx->cm_id);

    ucs_list_del(&cm_id_ctx->list);
    if (cm_id_ctx->ep != NULL) {
        cm_id_ctx->ep->cm_id_ctx = NULL;
    }
    rdma_destroy_id(cm_id_ctx->cm_id);
    ucs_free(cm_id_ctx);
    iface->cm_id_quota++;

    *cm_id_ctx_p = NULL;
}

static unsigned
uct_rdmacm_iface_process_event(uct_rdmacm_iface_t *iface,
                               struct rdma_cm_event *event)
{
    struct sockaddr *remote_addr = rdma_get_peer_addr(event->id);
    uct_rdmacm_md_t *rdmacm_md   = (uct_rdmacm_md_t *)iface->super.md;
    unsigned ret_flags           = UCT_RDMACM_PROCESS_EVENT_ACK_EVENT_FLAG;
    uct_rdmacm_ep_t *ep          = NULL;
    uct_cm_ep_priv_data_pack_args_t pack_args;
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    char dev_name[UCT_DEVICE_NAME_MAX];
    uct_rdmacm_priv_data_hdr_t *hdr;
    struct rdma_conn_param conn_param;
    uct_rdmacm_ctx_t *cm_id_ctx;
    ssize_t priv_data_ret;
    ucs_status_t status;

    if (iface->is_server) {
        ucs_assert((iface->cm_id == event->id) ||
                   ((event->event == RDMA_CM_EVENT_CONNECT_REQUEST) &&
                    (iface->cm_id == event->listen_id)));
    } else {
        cm_id_ctx = event->id->context;
        ep = cm_id_ctx->ep;
    }

    ucs_trace("rdmacm event (fd=%d cm_id %p) on %s (ep=%p): %s. Peer: %s.",
              iface->event_ch->fd, event->id, (iface->is_server ? "server" : "client"),
              ep, rdma_event_str(event->event),
              ucs_sockaddr_str(remote_addr, ip_port_str, UCS_SOCKADDR_STRING_LEN));

    status = UCS_ERR_UNREACHABLE;
    /* The following applies for rdma_cm_id of type RDMA_PS_UDP only */
    switch (event->event) {
    case RDMA_CM_EVENT_ADDR_RESOLVED:
        /* Client - resolve the route to the server */
        if (ep == NULL) {
            /* received an event on an non-existing ep - an already destroyed ep */
            ret_flags |= UCT_RDMACM_PROCESS_EVENT_DESTROY_CM_ID_FLAG;
        } else if (rdma_resolve_route(event->id, UCS_MSEC_PER_SEC *
                                                 rdmacm_md->addr_resolve_timeout)) {
            ucs_error("rdma_resolve_route(to addr=%s) failed: %m",
                      ucs_sockaddr_str(remote_addr, ip_port_str,
                                       UCS_SOCKADDR_STRING_LEN));
            ret_flags |= UCT_RDMACM_PROCESS_EVENT_DESTROY_CM_ID_FLAG;
            uct_rdmacm_client_handle_failure(iface, ep, UCS_ERR_INVALID_ADDR);
        }
        break;

    case RDMA_CM_EVENT_ROUTE_RESOLVED:
        /* Client - send a connection request to the server */
        if (ep == NULL) {
            /* received an event on an non-existing ep - an already destroyed ep */
            ret_flags |= UCT_RDMACM_PROCESS_EVENT_DESTROY_CM_ID_FLAG;
        } else {
            memset(&conn_param, 0, sizeof(conn_param));
            conn_param.private_data = ucs_alloca(UCT_RDMACM_MAX_CONN_PRIV +
                                                 sizeof(uct_rdmacm_priv_data_hdr_t));

            uct_rdmacm_cm_id_to_dev_name(ep->cm_id_ctx->cm_id, dev_name);

            hdr                  = (uct_rdmacm_priv_data_hdr_t*)conn_param.private_data;
            pack_args.field_mask = UCT_CM_EP_PRIV_DATA_PACK_ARGS_FIELD_DEVICE_NAME;
            ucs_strncpy_safe(pack_args.dev_name, dev_name, UCT_DEVICE_NAME_MAX);
            /* TODO check the ep's cb_flags to determine when to invoke this callback.
             * currently only UCT_CB_FLAG_ASYNC is supported so the cb is invoked from here */
            priv_data_ret = ep->pack_cb(ep->pack_cb_arg, &pack_args, hdr + 1);
            if (priv_data_ret < 0) {
                ucs_trace("rdmacm client (iface=%p cm_id=%p fd=%d) failed to fill "
                          "private data. status: %s",
                          iface, event->id, iface->event_ch->fd,
                          ucs_status_string((ucs_status_t)priv_data_ret));
                ret_flags |= UCT_RDMACM_PROCESS_EVENT_DESTROY_CM_ID_FLAG;
                uct_rdmacm_client_handle_failure(iface, ep, (ucs_status_t)priv_data_ret);
                break;
            }

            hdr->length = (uint8_t)priv_data_ret;
            hdr->status = UCS_OK;
            /* The private_data starts with the header of the user's private data
             * and then the private data itself */
            conn_param.private_data_len = sizeof(*hdr) + hdr->length;

            if (rdma_connect(event->id, &conn_param)) {
                ucs_error("rdma_connect(to addr=%s) failed: %m",
                          ucs_sockaddr_str(remote_addr, ip_port_str,
                                           UCS_SOCKADDR_STRING_LEN));
                ret_flags |= UCT_RDMACM_PROCESS_EVENT_DESTROY_CM_ID_FLAG;
                uct_rdmacm_client_handle_failure(iface, ep,
                                                 UCS_ERR_SOME_CONNECTS_FAILED);
            }
        }
        break;

    case RDMA_CM_EVENT_CONNECT_REQUEST:
        /* Server - handle a connection request from the client */
        ucs_assert(iface->is_server);
        uct_rdmacm_iface_process_conn_req(iface, event, remote_addr);
        ret_flags &= ~UCT_RDMACM_PROCESS_EVENT_ACK_EVENT_FLAG;
        break;

    case RDMA_CM_EVENT_REJECTED:
        /* Client - server rejected the connection request */
        ucs_warn("rdmacm connection request to %s rejected, id %p",
                  ucs_sockaddr_str(remote_addr, ip_port_str,
                                   UCS_SOCKADDR_STRING_LEN), event->id);

        ret_flags |= UCT_RDMACM_PROCESS_EVENT_DESTROY_CM_ID_FLAG;
        uct_rdmacm_client_handle_failure(iface, ep, UCS_ERR_REJECTED);
        break;

    case RDMA_CM_EVENT_ESTABLISHED:
        /* Client - connection is ready */
        ucs_assert(!iface->is_server);
        ret_flags |= UCT_RDMACM_PROCESS_EVENT_DESTROY_CM_ID_FLAG;
        if (ep != NULL) {
            pthread_mutex_lock(&ep->ops_mutex);
            ep->status = UCS_OK;
            uct_rdmacm_ep_invoke_completions(ep, UCS_OK);
            pthread_mutex_unlock(&ep->ops_mutex);
        }
        break;

    /* client error events */
    case RDMA_CM_EVENT_UNREACHABLE:
        hdr = (uct_rdmacm_priv_data_hdr_t *)event->param.ud.private_data;
        if ((hdr != NULL) && (event->param.ud.private_data_len > 0) &&
            ((ucs_status_t)hdr->status == UCS_ERR_REJECTED)) {
            ucs_assert(hdr->length == 0);
            ucs_assert(event->param.ud.private_data_len >= sizeof(*hdr));
            ucs_assert(!iface->is_server);
            status = UCS_ERR_REJECTED;
        }
        /* Fall through */
    case RDMA_CM_EVENT_ADDR_ERROR:
    case RDMA_CM_EVENT_ROUTE_ERROR:
    case RDMA_CM_EVENT_CONNECT_RESPONSE:
    /* client and server error events */
    case RDMA_CM_EVENT_CONNECT_ERROR:
    case RDMA_CM_EVENT_DISCONNECTED:
        /* Server/Client - connection was disconnected */
        if (status != UCS_ERR_REJECTED) {
            ucs_error("received event %s. status = %d. Peer: %s.",
                      rdma_event_str(event->event), event->status,
                      ucs_sockaddr_str(remote_addr, ip_port_str, UCS_SOCKADDR_STRING_LEN));
        }

        if (!iface->is_server) {
            ret_flags |= UCT_RDMACM_PROCESS_EVENT_DESTROY_CM_ID_FLAG;
            uct_rdmacm_client_handle_failure(iface, ep, status);
        }
        break;

    default:
        ucs_warn("unexpected RDMACM event: %d", event->event);
        break;
    }

    return ret_flags;
}

static void uct_rdmacm_iface_event_handler(int fd, int events, void *arg)
{
    uct_rdmacm_iface_t             *iface     = arg;
    uct_rdmacm_ctx_t               *cm_id_ctx = NULL;
    struct rdma_cm_event           *event;
    unsigned                       proc_event_flags;
    int                            ret;

    for (;;) {
        /* Fetch an event */
        ret = rdma_get_cm_event(iface->event_ch, &event);
        if (ret) {
            /* EAGAIN (in a non-blocking rdma_get_cm_event) means that
             * there are no more events */
            if (errno != EAGAIN) {
                ucs_warn("rdma_get_cm_event() failed: %m");
            }
            return;
        }

        proc_event_flags = uct_rdmacm_iface_process_event(iface, event);
        if (!iface->is_server) {
            cm_id_ctx = (uct_rdmacm_ctx_t *)event->id->context;
        }

        if (proc_event_flags & UCT_RDMACM_PROCESS_EVENT_ACK_EVENT_FLAG) {
            ret = rdma_ack_cm_event(event);
            if (ret) {
                ucs_warn("rdma_ack_cm_event() failed: %m");
            }
        }

        if ((proc_event_flags & UCT_RDMACM_PROCESS_EVENT_DESTROY_CM_ID_FLAG) &&
            (cm_id_ctx != NULL)) {
            uct_rdmacm_iface_release_cm_id(iface, &cm_id_ctx);
            uct_rdmacm_iface_client_start_next_ep(iface);
        }
    }
}

static UCS_CLASS_INIT_FUNC(uct_rdmacm_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_rdmacm_iface_config_t *config = ucs_derived_of(tl_config, uct_rdmacm_iface_config_t);
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    uct_rdmacm_md_t *rdmacm_md;
    struct sockaddr *listen_addr;
    ucs_status_t status;

    UCT_CHECK_PARAM(params->field_mask & UCT_IFACE_PARAM_FIELD_OPEN_MODE,
                    "UCT_IFACE_PARAM_FIELD_OPEN_MODE is not defined");

    UCT_CHECK_PARAM((params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER) ||
                    (params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_CLIENT),
                    "Invalid open mode %zu", params->open_mode);

    UCT_CHECK_PARAM(!(params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER) ||
                    (params->field_mask & UCT_IFACE_PARAM_FIELD_SOCKADDR),
                    "UCT_IFACE_PARAM_FIELD_SOCKADDR is not defined for UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER");

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_rdmacm_iface_ops, md, worker,
                              params, tl_config
                              UCS_STATS_ARG((params->field_mask & 
                                             UCT_IFACE_PARAM_FIELD_STATS_ROOT) ?
                                            params->stats_root : NULL)
                              UCS_STATS_ARG(UCT_RDMACM_TL_NAME));

    rdmacm_md = ucs_derived_of(self->super.md, uct_rdmacm_md_t);

    if (self->super.worker->async == NULL) {
        ucs_error("rdmacm must have async != NULL");
        return UCS_ERR_INVALID_PARAM;
    }
    if (self->super.worker->async->mode == UCS_ASYNC_MODE_SIGNAL) {
        ucs_warn("rdmacm does not support SIGIO");
    }

    self->config.addr_resolve_timeout = rdmacm_md->addr_resolve_timeout;

    self->event_ch = rdma_create_event_channel();
    if (self->event_ch == NULL) {
        ucs_error("rdma_create_event_channel(open_mode=%zu) failed: %m",
                  params->open_mode);
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    /* Set the event_channel fd to non-blocking mode
     * (so that rdma_get_cm_event won't be blocking) */
    status = ucs_sys_fcntl_modfl(self->event_ch->fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        goto err_destroy_event_channel;
    }

    if (params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER) {
        self->is_server = 1;

        /* Create an id for this interface. Events associated with this id will be
         * reported on the event_channel that was previously created. */
        if (rdma_create_id(self->event_ch, &self->cm_id, NULL, RDMA_PS_UDP)) {
            ucs_error("rdma_create_id() failed: %m");
            status = UCS_ERR_IO_ERROR;
            goto err_destroy_event_channel;
        }

        listen_addr = (struct sockaddr *)params->mode.sockaddr.listen_sockaddr.addr;
        if (rdma_bind_addr(self->cm_id, listen_addr)) {
            status = (errno == EADDRINUSE || errno == EADDRNOTAVAIL) ?
                     UCS_ERR_BUSY : UCS_ERR_IO_ERROR;
            ucs_error("rdma_bind_addr(addr=%s) failed: %m",
                      ucs_sockaddr_str(listen_addr, ip_port_str,
                                       UCS_SOCKADDR_STRING_LEN));
            goto err_destroy_id;
        }

        if (rdma_listen(self->cm_id, config->backlog)) {
            ucs_error("rdma_listen(cm_id:=%p event_channel=%p addr=%s) failed: %m",
                       self->cm_id, self->event_ch,
                       ucs_sockaddr_str(listen_addr, ip_port_str,
                                        UCS_SOCKADDR_STRING_LEN));
            status = UCS_ERR_IO_ERROR;
            goto err_destroy_id;
        }

        ucs_debug("rdma_cm id %p listening on %s:%d", self->cm_id,
                  ucs_sockaddr_str(listen_addr, ip_port_str,
                                   UCS_SOCKADDR_STRING_LEN),
                  ntohs(rdma_get_src_port(self->cm_id)));

        if (!(params->mode.sockaddr.cb_flags & UCT_CB_FLAG_ASYNC)) {
            ucs_fatal("Synchronous callback is not supported");
        }

        self->cb_flags         = params->mode.sockaddr.cb_flags;
        self->conn_request_cb  = params->mode.sockaddr.conn_request_cb;
        self->conn_request_arg = params->mode.sockaddr.conn_request_arg;
    } else {
        self->cm_id            = NULL;
        self->is_server        = 0;
    }

    self->cm_id_quota = config->cm_id_quota;
    ucs_list_head_init(&self->pending_eps_list);
    ucs_list_head_init(&self->used_cm_ids_list);

    /* Server and client register an event handler for incoming messages */
    status = ucs_async_set_event_handler(self->super.worker->async->mode,
                                         self->event_ch->fd, UCS_EVENT_SET_EVREAD,
                                         uct_rdmacm_iface_event_handler,
                                         self, self->super.worker->async);
    if (status != UCS_OK) {
        ucs_error("failed to set event handler");
        goto err_destroy_id;
    }


    ucs_debug("created an RDMACM iface %p. event_channel: %p, fd: %d, cm_id: %p",
              self, self->event_ch, self->event_ch->fd, self->cm_id);
    return UCS_OK;

err_destroy_id:
    if (self->is_server) {
        rdma_destroy_id(self->cm_id);
    }
err_destroy_event_channel:
    rdma_destroy_event_channel(self->event_ch);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_iface_t)
{
    uct_rdmacm_ctx_t *cm_id_ctx, *tmp_cm_id_ctx;

    ucs_async_remove_handler(self->event_ch->fd, 1);
    if (self->is_server) {
        rdma_destroy_id(self->cm_id);
    }

    UCS_ASYNC_BLOCK(self->super.worker->async);

    ucs_list_for_each_safe(cm_id_ctx, tmp_cm_id_ctx,
                           &self->used_cm_ids_list, list) {
        uct_rdmacm_iface_release_cm_id(self, &cm_id_ctx);
    }

    UCS_ASYNC_UNBLOCK(self->super.worker->async);

    rdma_destroy_event_channel(self->event_ch);
}

UCS_CLASS_DEFINE(uct_rdmacm_iface_t, uct_base_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t *,
                                 const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_iface_t, uct_iface_t);

static ucs_status_t
uct_rdmacm_query_tl_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                            unsigned *num_tl_devices_p)
{
    *num_tl_devices_p = 0;
    *tl_devices_p     = NULL;
    return UCS_OK;
}

UCT_TL_DEFINE(&uct_rdmacm_component, rdmacm, uct_rdmacm_query_tl_devices,
              uct_rdmacm_iface_t, "RDMACM_", uct_rdmacm_iface_config_table,
              uct_rdmacm_iface_config_t);
