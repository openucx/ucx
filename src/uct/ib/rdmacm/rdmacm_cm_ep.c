/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rdmacm_cm_ep.h"
#include "rdmacm_cm.h"
#include <ucs/arch/bitops.h>


const char* uct_rdmacm_cm_ep_str(uct_rdmacm_cm_ep_t *cep, char *str,
                                 size_t max_len)
{
    struct sockaddr *local_addr  = rdma_get_local_addr(cep->id);
    struct sockaddr *remote_addr = rdma_get_peer_addr(cep->id);
    char flags_buf[UCT_RDMACM_EP_FLAGS_STRING_LEN];
    char local_ip_port_str[UCS_SOCKADDR_STRING_LEN];
    char remote_ip_port_str[UCS_SOCKADDR_STRING_LEN];

    static const char *ep_flag_to_str[] = {
        [ucs_ilog2(UCT_RDMACM_CM_EP_ON_CLIENT)]                = "client",
        [ucs_ilog2(UCT_RDMACM_CM_EP_ON_SERVER)]                = "server",
        [ucs_ilog2(UCT_RDMACM_CM_EP_CLIENT_CONN_CB_INVOKED)]   = "connect_cb_invoked",
        [ucs_ilog2(UCT_RDMACM_CM_EP_SERVER_NOTIFY_CB_INVOKED)] = "notify_cb_invoked",
        [ucs_ilog2(UCT_RDMACM_CM_EP_GOT_DISCONNECT)]           = "got_disconnect",
        [ucs_ilog2(UCT_RDMACM_CM_EP_DISCONNECTING)]            = "disconnecting",
        [ucs_ilog2(UCT_RDMACM_CM_EP_FAILED)]                   = "failed",
        NULL
    };

    if (ucs_sockaddr_is_known_af(local_addr)) {
        ucs_sockaddr_str(local_addr, local_ip_port_str, UCS_SOCKADDR_STRING_LEN);
    } else {
        ucs_strncpy_safe(local_ip_port_str, "<invalid>", UCS_SOCKADDR_STRING_LEN);
    }

    if (ucs_sockaddr_is_known_af(remote_addr)) {
        ucs_sockaddr_str(remote_addr, remote_ip_port_str, UCS_SOCKADDR_STRING_LEN);
    } else {
        ucs_strncpy_safe(remote_ip_port_str, "<invalid>", UCS_SOCKADDR_STRING_LEN);
    }

    ucs_flags_str(flags_buf, sizeof(flags_buf), cep->flags, ep_flag_to_str);
    ucs_snprintf_safe(str, max_len, "[cep %p %s->%s %s %s]",
                      cep, local_ip_port_str, remote_ip_port_str, flags_buf,
                      ucs_status_string(cep->status));
    return str;
}

int uct_rdmacm_ep_is_connected(uct_rdmacm_cm_ep_t *cep)
{
    return cep->flags & (UCT_RDMACM_CM_EP_CLIENT_CONN_CB_INVOKED |
                         UCT_RDMACM_CM_EP_SERVER_NOTIFY_CB_INVOKED);
}

void uct_rdmacm_cm_ep_client_connect_cb(uct_rdmacm_cm_ep_t *cep,
                                        uct_cm_remote_data_t *remote_data,
                                        ucs_status_t status)
{
    cep->flags |= UCT_RDMACM_CM_EP_CLIENT_CONN_CB_INVOKED;
    uct_cm_ep_client_connect_cb(&cep->super, remote_data, status);
}

void uct_rdmacm_cm_ep_server_conn_notify_cb(uct_rdmacm_cm_ep_t *cep,
                                            ucs_status_t status)
{
    cep->flags |= UCT_RDMACM_CM_EP_SERVER_NOTIFY_CB_INVOKED;
    uct_cm_ep_server_conn_notify_cb(&cep->super, status);
}

void uct_rdmacm_cm_ep_error_cb(uct_rdmacm_cm_ep_t *cep,
                               uct_cm_remote_data_t *remote_data,
                               ucs_status_t status)
{
    if (cep->flags & UCT_RDMACM_CM_EP_FAILED) {
        return;
    }

    ucs_assert(status != UCS_OK);
    cep->status = status;

    if (uct_rdmacm_ep_is_connected(cep)) {
        /* already connected, so call disconnect callback */
        uct_cm_ep_disconnect_cb(&cep->super);
    } else if (cep->flags & UCT_RDMACM_CM_EP_ON_CLIENT) {
        /* not connected yet, so call client side connect callback with err
         * status */
        uct_rdmacm_cm_ep_client_connect_cb(cep, remote_data, status);
    } else {
        ucs_assert(cep->flags & UCT_RDMACM_CM_EP_ON_SERVER);
        /* not connected yet, so call server side notify callback with err
         * status */
        uct_rdmacm_cm_ep_server_conn_notify_cb(cep, status);
    }
}

void uct_rdmacm_cm_ep_set_failed(uct_rdmacm_cm_ep_t *cep,
                                 uct_cm_remote_data_t *remote_data,
                                 ucs_status_t status)
{
    UCS_ASYNC_BLOCK(uct_rdmacm_cm_ep_get_async(cep));
    uct_rdmacm_cm_ep_error_cb(cep, remote_data, status);
    cep->flags |= UCT_RDMACM_CM_EP_FAILED;
    UCS_ASYNC_UNBLOCK(uct_rdmacm_cm_ep_get_async(cep));
}

ucs_status_t uct_rdmacm_cm_ep_conn_notify(uct_ep_h ep)
{
    uct_rdmacm_cm_ep_t *cep                 = ucs_derived_of(ep, uct_rdmacm_cm_ep_t);
    struct sockaddr *remote_addr            = rdma_get_peer_addr(cep->id);
    uct_rdmacm_cm_t UCS_V_UNUSED *rdmacm_cm = uct_rdmacm_cm_ep_get_cm(cep);
    char ep_str[UCT_RDMACM_EP_STRING_LEN];
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];

    ucs_trace("%s rdma_establish on client (cm_id %p, rdmacm %p, event_channel=%p)",
              uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
              cep->id, rdmacm_cm, rdmacm_cm->ev_ch);

    UCS_ASYNC_BLOCK(uct_rdmacm_cm_ep_get_async(cep));
    if (cep->flags & (UCT_RDMACM_CM_EP_FAILED |
                      UCT_RDMACM_CM_EP_GOT_DISCONNECT)) {
        goto ep_failed;
    }

    UCS_ASYNC_UNBLOCK(uct_rdmacm_cm_ep_get_async(cep));

    if (rdma_establish(cep->id)) {
        uct_cm_ep_peer_error(&cep->super,
                             "rdma_establish on ep %p (to server addr=%s) failed: %m",
                             cep, ucs_sockaddr_str(remote_addr, ip_port_str,
                                                   UCS_SOCKADDR_STRING_LEN));
        UCS_ASYNC_BLOCK(uct_rdmacm_cm_ep_get_async(cep));
        cep->status = UCS_ERR_IO_ERROR;
        cep->flags |= UCT_RDMACM_CM_EP_FAILED;
        goto ep_failed;
    }

    return UCS_OK;

ep_failed:
    UCS_ASYNC_UNBLOCK(uct_rdmacm_cm_ep_get_async(cep));
    return cep->status;
}

static void uct_rdmacm_cm_ep_destroy_dummy_cq_qp(uct_rdmacm_cm_ep_t *cep)
{
    int ret;

    if (cep->qp != NULL) {
        ret = ibv_destroy_qp(cep->qp);
        if (ret != 0) {
            ucs_warn("ibv_destroy_qp() returned %d: %m", ret);
        }
    }

    if (cep->cq != NULL) {
        ret = ibv_destroy_cq(cep->cq);
        if (ret != 0) {
            ucs_warn("ibv_destroy_cq() returned %d: %m", ret);
        }
    }

    cep->qp = NULL;
    cep->cq = NULL;
}

static ucs_status_t uct_rdmacm_cm_create_dummy_cq_qp(struct rdma_cm_id *id,
                                                     struct ibv_cq **cq_p,
                                                     struct ibv_qp **qp_p)
{
    struct ibv_qp_init_attr qp_init_attr;
    ucs_status_t status;
    struct ibv_cq *cq;
    struct ibv_qp *qp;

    /* Create a dummy completion queue */
    cq = ibv_create_cq(id->verbs, 1, NULL, NULL, 0);
    if (cq == NULL) {
        ucs_error("ibv_create_cq() failed: %m");
        status =  UCS_ERR_IO_ERROR;
        goto err;
    }

    /* Create a dummy UD qp */
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.send_cq = cq;
    qp_init_attr.recv_cq = cq;
    qp_init_attr.qp_type = IBV_QPT_UD;
    qp_init_attr.cap.max_send_wr  = 2;
    qp_init_attr.cap.max_recv_wr  = 2;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;

    qp = ibv_create_qp(id->pd, &qp_init_attr);
    if (qp == NULL) {
        ucs_error("failed to create a dummy ud qp. %m");
        status = UCS_ERR_IO_ERROR;
        goto err_destroy_cq;
    }

    ucs_debug("created ud QP %p with qp_num: 0x%x and cq %p on rdmacm_id %p",
              qp, qp->qp_num, cq, id);

    *cq_p = cq;
    *qp_p = qp;

    return UCS_OK;

err_destroy_cq:
    ibv_destroy_cq(cq);
err:
    return status;
}

ucs_status_t
uct_rdamcm_cm_ep_set_qp_num(struct rdma_conn_param *conn_param,
                            uct_rdmacm_cm_ep_t *cep)
{
    ucs_status_t status;
    struct ibv_qp *qp;
    struct ibv_cq *cq;

    /* create a dummy qp in order to get a unique qp_num to provide to librdmacm */
    status = uct_rdmacm_cm_create_dummy_cq_qp(cep->id, &cq, &qp);
    if (status != UCS_OK) {
        return status;
    }

    cep->cq             = cq;
    cep->qp             = qp;
    conn_param->qp_num  = qp->qp_num;
    return UCS_OK;
}

ucs_status_t uct_rdmacm_cm_ep_pack_cb(uct_rdmacm_cm_ep_t *cep,
                                      struct rdma_conn_param *conn_param)
{
    uct_rdmacm_priv_data_hdr_t      *hdr;
    ucs_status_t                    status;
    char                            dev_name[UCT_DEVICE_NAME_MAX];
    size_t                          priv_data_ret;
    uct_cm_ep_priv_data_pack_args_t pack_args;

    uct_rdmacm_cm_id_to_dev_name(cep->id, dev_name);

    /* Pack data to send inside rdmacm's conn_param to the remote peer */
    hdr                  = (uct_rdmacm_priv_data_hdr_t*)conn_param->private_data;
    pack_args.field_mask = UCT_CM_EP_PRIV_DATA_PACK_ARGS_FIELD_DEVICE_NAME;
    ucs_strncpy_safe(pack_args.dev_name, dev_name, UCT_DEVICE_NAME_MAX);

    status = uct_cm_ep_pack_cb(&cep->super, cep->super.user_data, &pack_args,
                               hdr + 1, uct_rdmacm_cm_get_max_conn_priv(),
                               &priv_data_ret);

    if (status != UCS_OK) {
        goto err;
    }

    ucs_assert_always(priv_data_ret <= UINT8_MAX);
    hdr->length = (uint8_t)priv_data_ret;
    hdr->status = UCS_OK;

    conn_param->private_data_len = sizeof(*hdr) + hdr->length;

    return UCS_OK;

err:
    return status;
}

static ucs_status_t uct_rdamcm_cm_ep_client_init(uct_rdmacm_cm_ep_t *cep,
                                                 const uct_ep_params_t *params)
{
    uct_rdmacm_cm_t *rdmacm_cm = uct_rdmacm_cm_ep_get_cm(cep);
    uct_cm_base_ep_t *cm_ep    = &cep->super;
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    char ep_str[UCT_RDMACM_EP_STRING_LEN];
    ucs_status_t status;

    cep->flags |= UCT_RDMACM_CM_EP_ON_CLIENT;

    status = UCT_CM_SET_CB(params, UCT_EP_PARAM_FIELD_SOCKADDR_CONNECT_CB_CLIENT,
                           cm_ep->client.connect_cb, params->sockaddr_cb_client,
                           uct_cm_ep_client_connect_callback_t,
                           ucs_empty_function);
    if (status != UCS_OK) {
        goto err;
    }

    if (rdma_create_id(rdmacm_cm->ev_ch, &cep->id, cep, RDMA_PS_TCP)) {
        ucs_error("rdma_create_id() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    ucs_trace("%s rdma_create_id on client (rdmacm %p, event_channel=%p)",
              uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
              rdmacm_cm, rdmacm_cm->ev_ch);

    /* rdma_resolve_addr needs to be called last in the ep_create flow to
     * prevent a race where there are uninitialized fields used when the
     * RDMA_CM_EVENT_ROUTE_RESOLVED event is already received in the the async
     * thread. Therefore, all ep fields have to be initialized before this
     * function is called. */
    ucs_trace("%s: rdma_resolve_addr on cm_id %p",
              uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN), cep->id);
    if (rdma_resolve_addr(cep->id, NULL, (struct sockaddr*)params->sockaddr->addr,
                          1000/* TODO */)) {
        ucs_error("rdma_resolve_addr() to dst addr %s failed: %m",
                  ucs_sockaddr_str((struct sockaddr*)params->sockaddr->addr,
                                   ip_port_str, UCS_SOCKADDR_STRING_LEN));
        status = UCS_ERR_IO_ERROR;
        goto err_destroy_id;
    }

    return UCS_OK;

err_destroy_id:
    uct_rdmacm_cm_destroy_id(cep->id);
err:
    return status;
}

static ucs_status_t uct_rdamcm_cm_ep_server_init(uct_rdmacm_cm_ep_t *cep,
                                                 const uct_ep_params_t *params)
{
    struct rdma_cm_event   *event = (struct rdma_cm_event*)params->conn_request;
    uct_rdmacm_cm_t        *cm    = uct_rdmacm_cm_ep_get_cm(cep);
    uct_cm_base_ep_t       *cm_ep = &cep->super;
    struct rdma_conn_param conn_param;
    ucs_status_t           status;
    char                   ep_str[UCT_RDMACM_EP_STRING_LEN];

    cep->flags |= UCT_RDMACM_CM_EP_ON_SERVER;

    if (event->listen_id->channel != cm->ev_ch) {
        /* the server will open the ep to the client on a different CM.
         * not the one on which its listener is listening on */
        if (rdma_migrate_id(event->id, cm->ev_ch)) {
            ucs_error("failed to migrate id %p to event_channel %p (cm=%p)",
                      event->id, cm->ev_ch, cm);
            status = UCS_ERR_IO_ERROR;
            goto err_reject;
        }

        ucs_debug("%s migrated id %p from event_channel=%p to "
                  "new cm %p (event_channel=%p)",
                  uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
                  event->id, event->listen_id->channel, cm, cm->ev_ch);
    }

    status = UCT_CM_SET_CB(params, UCT_EP_PARAM_FIELD_SOCKADDR_NOTIFY_CB_SERVER,
                           cm_ep->server.notify_cb, params->sockaddr_cb_server,
                           uct_cm_ep_server_conn_notify_callback_t,
                           ucs_empty_function);
    if (status != UCS_OK) {
        goto err;
    }

    cep->id          = event->id;
    cep->id->context = cep;

    memset(&conn_param, 0, sizeof(conn_param));
    conn_param.private_data = ucs_alloca(uct_rdmacm_cm_get_max_conn_priv() +
                                         sizeof(uct_rdmacm_priv_data_hdr_t));

    status = uct_rdmacm_cm_ep_pack_cb(cep, &conn_param);
    if (status != UCS_OK) {
        goto err_reject;
    }

    status = uct_rdamcm_cm_ep_set_qp_num(&conn_param, cep);
    if (status != UCS_OK) {
        goto err_reject;
    }

    ucs_trace("%s: rdma_accept on cm_id %p",
              uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
              event->id);

    if (rdma_accept(event->id, &conn_param)) {
        uct_cm_ep_peer_error(&cep->super, "rdma_accept(on id=%p) failed: %m",
                             event->id);
        uct_rdmacm_cm_ep_destroy_dummy_cq_qp(cep);
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    uct_rdmacm_cm_ack_event(event);
    return UCS_OK;

err_reject:
    uct_rdmacm_cm_reject(event->id);
err:
    UCS_ASYNC_BLOCK(uct_rdmacm_cm_ep_get_async(cep));
    cep->status = status;
    cep->flags |= UCT_RDMACM_CM_EP_FAILED;
    UCS_ASYNC_UNBLOCK(uct_rdmacm_cm_ep_get_async(cep));
    uct_rdmacm_cm_destroy_id(event->id);
    uct_rdmacm_cm_ack_event(event);
    return status;
}

ucs_status_t uct_rdmacm_cm_ep_disconnect(uct_ep_h ep, unsigned flags)
{
    uct_rdmacm_cm_ep_t *cep = ucs_derived_of(ep, uct_rdmacm_cm_ep_t);
    char ep_str[UCT_RDMACM_EP_STRING_LEN];
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t status;

    UCS_ASYNC_BLOCK(uct_rdmacm_cm_ep_get_async(cep));
    if (ucs_unlikely(cep->flags & UCT_RDMACM_CM_EP_FAILED)) {
        uct_cm_ep_peer_error(&cep->super, "%s id=%p to peer %s",
                             uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
                             cep->id, ucs_sockaddr_str(rdma_get_peer_addr(cep->id),
                                                       ip_port_str, UCS_SOCKADDR_STRING_LEN));
        status = cep->status;
        goto out;
    }

    if (ucs_unlikely(cep->flags & UCT_RDMACM_CM_EP_DISCONNECTING)) {
        if (cep->flags & UCT_RDMACM_CM_EP_GOT_DISCONNECT) {
            ucs_error("%s duplicate call of uct_ep_disconnect on a "
                      "disconnected ep (id=%p to peer %s)",
                      uct_rdmacm_cm_ep_str(cep, ep_str,
                                           UCT_RDMACM_EP_STRING_LEN),
                      cep->id, ucs_sockaddr_str(rdma_get_peer_addr(cep->id),
                                                ip_port_str,
                                                UCS_SOCKADDR_STRING_LEN));
            status = UCS_ERR_NOT_CONNECTED;
            goto out;
        }

        ucs_debug("%s: duplicate call of uct_ep_disconnect on an ep "
                  "that was not disconnected yet (id=%p to peer %s).",
                  uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
                  cep->id, ucs_sockaddr_str(rdma_get_peer_addr(cep->id),
                                            ip_port_str,
                                            UCS_SOCKADDR_STRING_LEN));
        status = UCS_INPROGRESS;
        goto out;
    }

    if (!uct_rdmacm_ep_is_connected(cep)) {
        ucs_debug("%s: calling uct_ep_disconnect on an ep that is not "
                  "connected yet (id=%p to peer %s)",
                  uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
                  cep->id, ucs_sockaddr_str(rdma_get_peer_addr(cep->id),
                                            ip_port_str,
                                            UCS_SOCKADDR_STRING_LEN));
        status = UCS_ERR_BUSY;
        goto out;
    }

    cep->flags |= UCT_RDMACM_CM_EP_DISCONNECTING;
    if (rdma_disconnect(cep->id)) {
        ucs_error("%s: (id=%p) failed to disconnect from peer %p",
                  uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
                  cep->id, ucs_sockaddr_str(rdma_get_peer_addr(cep->id), ip_port_str,
                                            UCS_SOCKADDR_STRING_LEN));
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    ucs_debug("%s: (id=%p) disconnecting from peer :%s",
              uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
              cep->id, ucs_sockaddr_str(rdma_get_peer_addr(cep->id), ip_port_str,
                                        UCS_SOCKADDR_STRING_LEN));
    status = UCS_OK;

out:
    UCS_ASYNC_UNBLOCK(uct_rdmacm_cm_ep_get_async(cep));
    return status;
}

UCS_CLASS_INIT_FUNC(uct_rdmacm_cm_ep_t, const uct_ep_params_t *params)
{
    ucs_status_t status;
    char ep_str[UCT_RDMACM_EP_STRING_LEN];

    UCS_CLASS_CALL_SUPER_INIT(uct_cm_base_ep_t, params);

    self->cq     = NULL;
    self->qp     = NULL;
    self->flags  = 0;
    self->status = UCS_OK;

    if (params->field_mask & UCT_EP_PARAM_FIELD_SOCKADDR) {
        status = uct_rdamcm_cm_ep_client_init(self, params);
    } else if (params->field_mask & UCT_EP_PARAM_FIELD_CONN_REQUEST) {
        status = uct_rdamcm_cm_ep_server_init(self, params);
    } else {
        ucs_error("either UCT_EP_PARAM_FIELD_SOCKADDR or UCT_EP_PARAM_FIELD_CONN_REQUEST "
                  "has to be provided");
        status = UCS_ERR_INVALID_PARAM;
    }

    if (status == UCS_OK) {
        ucs_debug("%s created an endpoint on rdmacm %p id: %p",
                  uct_rdmacm_cm_ep_str(self, ep_str, UCT_RDMACM_EP_STRING_LEN),
                  uct_rdmacm_cm_ep_get_cm(self), self->id);
    }

    return status;
}

UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_cm_ep_t)
{
    uct_rdmacm_cm_t *rdmacm_cm     = uct_rdmacm_cm_ep_get_cm(self);
    uct_priv_worker_t *worker_priv = ucs_derived_of(rdmacm_cm->super.iface.worker,
                                                    uct_priv_worker_t);
    char ep_str[UCT_RDMACM_EP_STRING_LEN];

    ucs_trace("%s destroy ep on cm %p (worker_priv=%p)",
              uct_rdmacm_cm_ep_str(self, ep_str, UCT_RDMACM_EP_STRING_LEN),
              rdmacm_cm, worker_priv);

    UCS_ASYNC_BLOCK(worker_priv->async);

    uct_rdmacm_cm_ep_destroy_dummy_cq_qp(self);

    /* rdma_destroy_id() cleans all events not yet reported on progress thread,
     * so no events would be reported to the user after destroying the id */
    uct_rdmacm_cm_destroy_id(self->id);

    UCS_ASYNC_UNBLOCK(worker_priv->async);
}

UCS_CLASS_DEFINE(uct_rdmacm_cm_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_cm_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_cm_ep_t, uct_ep_t);
