/**
* Copyright (C) Mellanox Technologies Ltd. 2019-2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rdmacm_listener.h"

#include <ucs/sys/sock.h>


#define UCS_RDMACM_MAX_BACKLOG_PATH        "/proc/sys/net/rdma_ucm/max_backlog"


static long ucs_rdmacm_max_backlog()
{
    static long max_backlog = 0;

    if ((max_backlog != 0) ||
        (ucs_read_file_number(&max_backlog, 1, UCS_RDMACM_MAX_BACKLOG_PATH) == UCS_OK)) {
        ucs_assert(max_backlog <= INT_MAX);
    } else {
        ucs_diag("unable to read max_backlog value from %s file",
                 UCS_RDMACM_MAX_BACKLOG_PATH);
        max_backlog = 1024;
    }

    return max_backlog;
}

UCS_CLASS_INIT_FUNC(uct_rdmacm_listener_t, uct_cm_h cm,
                    const struct sockaddr *saddr, socklen_t socklen,
                    const uct_listener_params_t *params)
{
    uct_rdmacm_cm_t *rdmacm_cm = ucs_derived_of(cm, uct_rdmacm_cm_t);
    int id_reuse_optval        = 1;
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t status;
    int backlog;

    UCS_CLASS_CALL_SUPER_INIT(uct_listener_t, cm);

    self->conn_request_cb = params->conn_request_cb;
    self->user_data       = (params->field_mask & UCT_LISTENER_PARAM_FIELD_USER_DATA) ?
                            params->user_data : NULL;

    if (rdma_create_id(rdmacm_cm->ev_ch, &self->id, self, RDMA_PS_TCP)) {
        ucs_error("rdma_create_id() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    if (rdmacm_cm->super.config.reuse_addr) {
        if (rdma_set_option(self->id, RDMA_OPTION_ID, RDMA_OPTION_ID_REUSEADDR,
                            &id_reuse_optval, sizeof(id_reuse_optval))) {
            ucs_error("rdma_set_option(REUSEADDR) failed: %m");
            status = UCS_ERR_IO_ERROR;
            goto err_destroy_id;
        }
    }

    if (rdma_bind_addr(self->id, (struct sockaddr*)saddr)) {
        switch (errno) {
        case EADDRINUSE:
        case EADDRNOTAVAIL:
            status = UCS_ERR_BUSY;
            break;
        case ENODEV:
            status = UCS_ERR_NO_DEVICE;
            break;
        default:
            status = UCS_ERR_IO_ERROR;
            break;
        }

        ucs_diag("rdma_bind_addr(addr=%s) failed: %m",
                 ucs_sockaddr_str(saddr, ip_port_str,
                                  UCS_SOCKADDR_STRING_LEN));
        goto err_destroy_id;
    }

    status = uct_listener_backlog_adjust(params, ucs_rdmacm_max_backlog(),
                                         &backlog);
    if (status != UCS_OK) {
        goto err_destroy_id;
    }

    if (rdma_listen(self->id, backlog)) {
        ucs_error("rdma_listen(id:=%p addr=%s backlog=%d) failed: %m",
                  self->id, ucs_sockaddr_str(saddr, ip_port_str,
                                             UCS_SOCKADDR_STRING_LEN),
                  backlog);
        status = UCS_ERR_IO_ERROR;
        goto err_destroy_id;
    }

    ucs_sockaddr_str(rdma_get_local_addr(self->id), ip_port_str,
                     UCS_SOCKADDR_STRING_LEN);
    ucs_debug("listener %p: created on cm %p %s rdma_cm_id %p", self, cm,
              ip_port_str, self->id);
    return UCS_OK;

err_destroy_id:
    uct_rdmacm_cm_destroy_id(self->id);
err:
    return status;
}

ucs_status_t uct_rdmacm_listener_reject(uct_listener_h listener,
                                        uct_conn_request_h conn_request)
{
    uct_rdmacm_listener_t *rdmacm_listener = ucs_derived_of(listener,
                                                            uct_rdmacm_listener_t);
    uct_rdmacm_cm_t *rdmacm_cm             = ucs_derived_of(listener->cm,
                                                            uct_rdmacm_cm_t);
    struct rdma_cm_event *event            = (struct rdma_cm_event*)conn_request;

    ucs_assert_always(rdmacm_listener->id == event->listen_id);

    uct_rdmacm_cm_reject(rdmacm_cm, event->id);
    uct_rdmacm_cm_destroy_id(event->id);
    return uct_rdmacm_cm_ack_event(event);
}

UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_listener_t)
{
    ucs_debug("listener %p: destroying rdma_cm_id %p", self, self->id);
    uct_rdmacm_cm_destroy_id(self->id);
}

ucs_status_t uct_rdmacm_listener_query(uct_listener_h listener,
                                       uct_listener_attr_t *listener_attr)
{
    uct_rdmacm_listener_t *rdmacm_listener = ucs_derived_of(listener,
                                                            uct_rdmacm_listener_t);
    struct sockaddr *addr;
    ucs_status_t status;

    if (listener_attr->field_mask & UCT_LISTENER_ATTR_FIELD_SOCKADDR) {
        addr   = rdma_get_local_addr(rdmacm_listener->id);
        status = ucs_sockaddr_copy((struct sockaddr*)&listener_attr->sockaddr,
                                   addr);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

UCS_CLASS_DEFINE(uct_rdmacm_listener_t, uct_listener_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_listener_t, uct_listener_t,
                          uct_cm_h , const struct sockaddr *, socklen_t ,
                          const uct_listener_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_listener_t, uct_listener_t);
