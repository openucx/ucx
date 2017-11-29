/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "rdmacm_iface.h"
#include "rdmacm_ep.h"
#include <uct/base/uct_worker.h>
#include <ucs/sys/string.h>

static ucs_config_field_t uct_rdmacm_iface_config_table[] = {
    {"BACKLOG", "1024",
     "Maximum number of pending connections for an rdma_cm_id.",
     ucs_offsetof(uct_rdmacm_iface_config_t, backlog), UCS_CONFIG_TYPE_UINT},

    {NULL}
};

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_rdmacm_iface_t, uct_iface_t);

static ucs_status_t uct_rdmacm_iface_query(uct_iface_h tl_iface,
                                           uct_iface_attr_t *iface_attr)
{
    memset(iface_attr, 0, sizeof(uct_iface_attr_t));

    iface_attr->iface_addr_len  = sizeof(ucs_sock_addr_t);
    iface_attr->device_addr_len = 0;
    iface_attr->cap.flags       = UCT_IFACE_FLAG_CONNECT_TO_SOCKADDR |
                                  UCT_IFACE_FLAG_CB_ASYNC;
    /* User's private data size is UCT_RDMACM_UDP_PRIV_DATA_LEN minus room for
     * the private_data header (to hold the length of the data) */
    iface_attr->max_conn_priv   = UCT_RDMACM_UDP_PRIV_DATA_LEN -
                                  sizeof(uct_rdmacm_priv_data_hdr_t);

    return UCS_OK;
}

static int uct_rdmacm_iface_is_reachable(const uct_iface_h tl_iface,
                                         const uct_device_addr_t *dev_addr,
                                         const uct_iface_addr_t *iface_addr)
{
    /* Reachability can be checked with the uct_md_is_sockaddr_accessible API call */
    return 1;
}

static ucs_status_t uct_rdmacm_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    ucs_sock_addr_t *rdmacm_addr = (ucs_sock_addr_t *)iface_addr;

    rdmacm_addr->addr    = NULL;
    rdmacm_addr->addrlen = 0;
    return UCS_OK;
}

static uct_iface_ops_t uct_rdmacm_iface_ops = {
    .ep_create_sockaddr       = UCS_CLASS_NEW_FUNC_NAME(uct_rdmacm_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_ep_t),
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_pending_purge         = ucs_empty_function,
    .iface_progress_enable    = (void*)ucs_empty_function_return_success,
    .iface_progress_disable   = (void*)ucs_empty_function_return_success,
    .iface_progress           = ucs_empty_function_return_zero,
    .iface_flush              = uct_base_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_iface_t),
    .iface_query              = uct_rdmacm_iface_query,
    .iface_is_reachable       = uct_rdmacm_iface_is_reachable,
    .iface_get_device_address = (void*)ucs_empty_function_return_success,
    .iface_get_address        = uct_rdmacm_iface_get_address
};

static UCS_CLASS_INIT_FUNC(uct_rdmacm_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_rdmacm_iface_config_t *config = ucs_derived_of(tl_config, uct_rdmacm_iface_config_t);
    size_t ip_len = ucs_max(INET_ADDRSTRLEN, INET6_ADDRSTRLEN);
    char *ip_str  = ucs_alloca(ip_len);
    uct_rdmacm_md_t *rdmacm_md;
    ucs_status_t status;

    if (!(params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER) &&
        !(params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_CLIENT)) {
        ucs_fatal("Invalid open mode %zu", params->open_mode);
    }

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_rdmacm_iface_ops, md, worker,
                              params, tl_config UCS_STATS_ARG(params->stats_root)
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

    /* Create an id for this interface. Events associated with this id will be
     * reported on the event_channel that was previously created. */
    if (rdma_create_id(self->event_ch, &self->cm_id, NULL, RDMA_PS_UDP)) {
        ucs_error("rdma_create_id() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_destroy_event_channel;
    }

    if (params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER) {
        if (rdma_bind_addr(self->cm_id, (struct sockaddr *)params->mode.sockaddr.listen_sockaddr.addr)) {
            ucs_error("rdma_bind_addr() failed: %m");
            status = UCS_ERR_IO_ERROR;
            goto err_destroy_id;
        }

        if (rdma_listen(self->cm_id, config->backlog)) {
            ucs_error("rdma_listen(cm_id:=%p event_channel=%p addr=%s) failed: %m",
                       self->cm_id, self->event_ch,
                       ucs_sockaddr_str((struct sockaddr *)params->mode.sockaddr.listen_sockaddr.addr,
                                        ip_str, ip_len));
            status = UCS_ERR_IO_ERROR;
            goto err_destroy_id;
        }

        ucs_debug("rdma_cm id %p listening on %s:%d", self->cm_id,
                  ucs_sockaddr_str((struct sockaddr *)params->mode.sockaddr.listen_sockaddr.addr,
                                   ip_str, ip_len),
                  ntohs(rdma_get_src_port(self->cm_id)));

        if (params->mode.sockaddr.cb_flags != UCT_CB_FLAG_ASYNC) {
            ucs_fatal("UCT_CB_FLAG_SYNC is not supported");
        }

        self->cb_flags         = params->mode.sockaddr.cb_flags;
        self->conn_request_cb  = params->mode.sockaddr.conn_request_cb;
        self->conn_request_arg = params->mode.sockaddr.conn_request_arg;
        self->is_server        = 1;
    } else {
        self->is_server        = 0;
    }

    self->ep = NULL;

    ucs_debug("created an RDMACM iface %p. event_channel: %p, fd: %d, cm_id: %p",
              self, self->event_ch, self->event_ch->fd, self->cm_id);
    return UCS_OK;

err_destroy_id:
    rdma_destroy_id(self->cm_id);
err_destroy_event_channel:
    rdma_destroy_event_channel(self->event_ch);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_iface_t)
{
    rdma_destroy_id(self->cm_id);
    rdma_destroy_event_channel(self->event_ch);
}

UCS_CLASS_DEFINE(uct_rdmacm_iface_t, uct_base_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t *,
                                 const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_iface_t, uct_iface_t);

static ucs_status_t uct_rdmacm_query_tl_resources(uct_md_h md,
                                                  uct_tl_resource_desc_t **resource_p,
                                                  unsigned *num_resources_p)
{
    *num_resources_p = 0;
    *resource_p      = NULL;
    return UCS_OK;
}

UCT_TL_COMPONENT_DEFINE(uct_rdmacm_tl,
                        uct_rdmacm_query_tl_resources,
                        uct_rdmacm_iface_t,
                        UCT_RDMACM_TL_NAME,
                        "RDMACM_",
                        uct_rdmacm_iface_config_table,
                        uct_rdmacm_iface_config_t);
UCT_MD_REGISTER_TL(&uct_rdmacm_mdc, &uct_rdmacm_tl);
