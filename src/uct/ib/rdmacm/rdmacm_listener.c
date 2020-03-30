/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rdmacm_listener.h"


UCS_CLASS_INIT_FUNC(uct_rdmacm_listener_t, uct_cm_h cm,
                    const struct sockaddr *saddr, socklen_t socklen,
                    const uct_listener_params_t *params)
{
    uct_rdmacm_cm_t *rdmacm_cm  = ucs_derived_of(cm, uct_rdmacm_cm_t);
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

    if (rdma_bind_addr(self->id, (struct sockaddr *)saddr)) {
        status = ((errno == EADDRINUSE) || (errno == EADDRNOTAVAIL)) ?
                 UCS_ERR_BUSY : UCS_ERR_IO_ERROR;
        ucs_error("rdma_bind_addr(addr=%s) failed: %m",
                  ucs_sockaddr_str(saddr, ip_port_str,
                                   UCS_SOCKADDR_STRING_LEN));
        goto err_destroy_id;
    }

    backlog = (params->field_mask & UCT_LISTENER_PARAM_FIELD_BACKLOG) ?
              params->backlog : SOMAXCONN;
    if (rdma_listen(self->id, backlog)) {
        ucs_error("rdma_listen(id:=%p addr=%s backlog=%d) failed: %m",
                  self->id, ucs_sockaddr_str(saddr, ip_port_str,
                                             UCS_SOCKADDR_STRING_LEN),
                  backlog);
        status = UCS_ERR_IO_ERROR;
        goto err_destroy_id;
    }

    ucs_debug("created an RDMACM listener %p on cm %p with cm_id: %p. "
              "listening on %s:%d", self, cm, self->id,
              ucs_sockaddr_str(saddr, ip_port_str, UCS_SOCKADDR_STRING_LEN),
              ntohs(rdma_get_src_port(self->id)));

    return UCS_OK;

err_destroy_id:
    uct_rdmacm_cm_destroy_id(self->id);
err:
    return status;
}

ucs_status_t uct_rdmacm_listener_reject(uct_listener_h listener,
                                        uct_conn_request_h conn_request)
{
    uct_rdmacm_listener_t *rdmacm_listener = ucs_derived_of(listener, uct_rdmacm_listener_t);
    struct rdma_cm_event *event            = (struct rdma_cm_event *)conn_request;

    ucs_assert_always(rdmacm_listener->id == event->listen_id);

    uct_rdmacm_cm_reject(event->id);

    uct_rdmacm_cm_destroy_id(event->id);

    return uct_rdmacm_cm_ack_event(event);
}

UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_listener_t)
{
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
        status = ucs_sockaddr_copy((struct sockaddr *)&listener_attr->sockaddr,
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
