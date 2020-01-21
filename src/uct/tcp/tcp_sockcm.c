/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "tcp_sockcm_ep.h"

#include <ucs/async/async.h>
#include <ucs/sys/sock.h>


ucs_config_field_t uct_tcp_sockcm_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_tcp_sockcm_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_cm_config_table)},

  {"PRIV_DATA_LEN", "2048",
   "TCP CM private data length",
   ucs_offsetof(uct_tcp_sockcm_config_t, priv_data_len), UCS_CONFIG_TYPE_MEMUNITS},

  {NULL}
};

static ucs_status_t uct_tcp_sockcm_query(uct_cm_h cm, uct_cm_attr_t *cm_attr)
{
    uct_tcp_sockcm_t *tcp_sockcm = ucs_derived_of(cm, uct_tcp_sockcm_t);

    if (cm_attr->field_mask & UCT_CM_ATTR_FIELD_MAX_CONN_PRIV) {
        cm_attr->max_conn_priv = tcp_sockcm->priv_data_len;
    }

    return UCS_OK;
}

static uct_cm_ops_t uct_tcp_sockcm_ops = {
    .close            = UCS_CLASS_DELETE_FUNC_NAME(uct_tcp_sockcm_t),
    .cm_query         = uct_tcp_sockcm_query,
    .listener_create  = UCS_CLASS_NEW_FUNC_NAME(uct_tcp_listener_t),
    .listener_reject  = uct_tcp_listener_reject,
    .listener_query   = uct_tcp_listener_query,
    .listener_destroy = UCS_CLASS_DELETE_FUNC_NAME(uct_tcp_listener_t),
    .ep_create        = uct_tcp_sockcm_ep_create
};

static void uct_tcp_close_ep(uct_tcp_sockcm_ep_t *ep)
{
    ucs_list_del(&ep->list);
    ucs_async_remove_handler(ep->fd, 1);
    close(ep->fd);
    ep->fd = -1;
    UCS_CLASS_DELETE(uct_tcp_sockcm_ep_t, ep);
}

void uct_tcp_sa_data_handler(int fd, void *arg)
{
    uct_tcp_sockcm_ep_t *ep = (uct_tcp_sockcm_ep_t*)arg;
    ucs_status_t status;

    ucs_assertv(ep->fd == fd, "ep->fd %d fd %d, ep_state %d", ep->fd, fd, ep->state);

    if (!ucs_socket_is_connected(fd)) {
        ucs_debug("fd %d is not connected. ep state: %d", fd, ep->state);
        /* coverity[check_return] */
        ucs_async_modify_handler(fd, 0);
        return;
    }

    switch (ep->state) {
    case UCT_TCP_SOCKCM_EP_ON_CLIENT:
        /* connect() completed, send data to the server */
        ep->state |= UCT_TCP_SOCKCM_EP_CONNECTED;

        status = uct_tcp_sockcm_ep_send_priv_data(ep);
        if (status != UCS_OK) {
            ucs_async_modify_handler(fd, 0);
        }
        break;
    case UCT_TCP_SOCKCM_EP_ON_CLIENT | UCT_TCP_SOCKCM_EP_CONN_SENDING:
         /* can send, progress the sending */
         status = uct_tcp_sockcm_ep_progress_send(ep);
         if (status != UCS_OK) {
             ucs_async_modify_handler(fd, 0);
         }
         break;
    case UCT_TCP_SOCKCM_EP_ON_CLIENT | UCT_TCP_SOCKCM_EP_CONN_SENT:
         /* finished sending. TODO recv data from the server */
         ucs_async_modify_handler(fd, 0);
         break;
    case UCT_TCP_SOCKCM_EP_ON_SERVER | UCT_TCP_SOCKCM_EP_CONNECTED:
        /* receive data from the client */
        status = uct_tcp_sockcm_ep_recv(ep);
        if (status != UCS_OK) {
            uct_tcp_close_ep(ep);
        }
        break;
    case UCT_TCP_SOCKCM_EP_ON_SERVER | UCT_TCP_SOCKCM_EP_CONN_RECEIVING:
         /* can read, progress the receving */
         status = uct_tcp_sockcm_ep_progress_recv(ep);
         if (status != UCS_OK) {
             uct_tcp_close_ep(ep);
         }
         break;
    case UCT_TCP_SOCKCM_EP_ON_SERVER | UCT_TCP_SOCKCM_EP_CONN_RECEIVED:
         /* finished recv, can send. TODO send data to the client */
         ucs_async_modify_handler(fd, 0);
         break;
    default:
        ucs_error("unexpected event on client ep %p (state=%d)", ep, ep->state);
    }
}

static uct_iface_ops_t uct_tcp_sockcm_iface_ops = {
    .ep_pending_purge         = (uct_ep_pending_purge_func_t)ucs_empty_function,
    .ep_disconnect            = uct_tcp_sockcm_ep_disconnect,
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_tcp_sockcm_ep_t),
    .ep_put_short             = (uct_ep_put_short_func_t)ucs_empty_function_return_unsupported,
    .ep_put_bcopy             = (uct_ep_put_bcopy_func_t)ucs_empty_function_return_unsupported,
    .ep_get_bcopy             = (uct_ep_get_bcopy_func_t)ucs_empty_function_return_unsupported,
    .ep_am_short              = (uct_ep_am_short_func_t)ucs_empty_function_return_unsupported,
    .ep_am_bcopy              = (uct_ep_am_bcopy_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic_cswap64        = (uct_ep_atomic_cswap64_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic64_post         = (uct_ep_atomic64_post_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic64_fetch        = (uct_ep_atomic64_fetch_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic_cswap32        = (uct_ep_atomic_cswap32_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic32_post         = (uct_ep_atomic32_post_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic32_fetch        = (uct_ep_atomic32_fetch_func_t)ucs_empty_function_return_unsupported,
    .ep_pending_add           = (uct_ep_pending_add_func_t)ucs_empty_function_return_unsupported,
    .ep_flush                 = (uct_ep_flush_func_t)ucs_empty_function_return_unsupported,
    .ep_fence                 = (uct_ep_fence_func_t)ucs_empty_function_return_unsupported,
    .ep_check                 = (uct_ep_check_func_t)ucs_empty_function_return_unsupported,
    .ep_create                = (uct_ep_create_func_t)ucs_empty_function_return_unsupported,
    .iface_flush              = (uct_iface_flush_func_t)ucs_empty_function_return_unsupported,
    .iface_fence              = (uct_iface_fence_func_t)ucs_empty_function_return_unsupported,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = (uct_iface_progress_func_t)ucs_empty_function_return_zero,
    .iface_event_fd_get       = (uct_iface_event_fd_get_func_t)ucs_empty_function_return_unsupported,
    .iface_event_arm          = (uct_iface_event_arm_func_t)ucs_empty_function_return_unsupported,
    .iface_close              = ucs_empty_function,
    .iface_query              = (uct_iface_query_func_t)ucs_empty_function_return_unsupported,
    .iface_get_device_address = (uct_iface_get_device_address_func_t)ucs_empty_function_return_unsupported,
    .iface_get_address        = (uct_iface_get_address_func_t)ucs_empty_function_return_unsupported,
    .iface_is_reachable       = (uct_iface_is_reachable_func_t)ucs_empty_function_return_zero
};

UCS_CLASS_INIT_FUNC(uct_tcp_sockcm_t, uct_component_h component,
                    uct_worker_h worker, const uct_cm_config_t *config)
{
    uct_tcp_sockcm_config_t *cm_config = ucs_derived_of(config,
                                                        uct_tcp_sockcm_config_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_cm_t, &uct_tcp_sockcm_ops,
                              &uct_tcp_sockcm_iface_ops, worker, component);

    self->priv_data_len = cm_config->priv_data_len -
                          sizeof(uct_tcp_sockcm_priv_data_hdr_t);

    ucs_list_head_init(&self->ep_list);

    ucs_debug("created tcp_sockcm %p", self);

    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(uct_tcp_sockcm_t)
{
    uct_tcp_sockcm_ep_t *ep, *tmp;

    UCS_ASYNC_BLOCK(self->super.iface.worker->async);

    ucs_list_for_each_safe(ep, tmp, &self->ep_list, list) {
        uct_tcp_close_ep(ep);
    }

    UCS_ASYNC_UNBLOCK(self->super.iface.worker->async);
}

UCS_CLASS_DEFINE(uct_tcp_sockcm_t, uct_cm_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_tcp_sockcm_t, uct_cm_t, uct_component_h,
                          uct_worker_h, const uct_cm_config_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_tcp_sockcm_t, uct_cm_t);
