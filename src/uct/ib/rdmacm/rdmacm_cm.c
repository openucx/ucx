/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#ifdef HAVE_CONFIG_H
#  include "config.h" /* Defines HAVE_RDMACM_QP_LESS */
#endif

#if HAVE_RDMACM_QP_LESS
//#if 1

#include "rdmacm_cm.h"
#include "rdmacm_iface.h"

#include <uct/ib/base/ib_iface.h>
#include <ucs/async/async.h>

#include <poll.h>
#include <rdma/rdma_cma.h>

UCS_CLASS_INIT_FUNC(uct_rdmacm_listener_t, uct_cm_h cm,
                    const struct sockaddr *saddr, socklen_t socklen,
                    const uct_listener_params_t *params)
{
    uct_rdmacm_cm_t *rdmacm_cm  = ucs_derived_of(cm, uct_rdmacm_cm_t);
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t status;
    int backlog;

    UCS_CLASS_CALL_SUPER_INIT(uct_listener_t, cm);
    self->id              = NULL;
    self->conn_request_cb = params->conn_request_cb;
    self->user_data       = (params->field_mask &
                             UCT_LISTENER_PARAM_FIELD_USER_DATA) ?
                            params->user_data : NULL;

    if (rdma_create_id(rdmacm_cm->ev_ch, &self->id, self, RDMA_PS_TCP)) {
        ucs_error("rdma_create_id() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    if (rdma_bind_addr(self->id, (struct sockaddr *)saddr)) {
        status = (errno == EADDRINUSE || errno == EADDRNOTAVAIL) ?
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
                  self->id, ucs_sockaddr_str(saddr, ip_port_str, UCS_SOCKADDR_STRING_LEN),
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
    rdma_destroy_id(self->id);
err:
    return status;
}

static ucs_status_t uct_rdmacm_listener_reject(uct_listener_h listener,
                                               uct_conn_request_h conn_request)
{
    struct rdma_cm_event *event = (struct rdma_cm_event *)conn_request;
    uct_rdmacm_listener_t *rdmacm_listener;

    rdmacm_listener = ucs_derived_of(listener, uct_rdmacm_listener_t);
    ucs_assert_always(rdmacm_listener->id == event->listen_id);

    if (rdma_reject(event->id, NULL, 0)) {
        ucs_error("rdmacm_listener %p: rdma_reject (id=%p) failed with error: %m",
                  rdmacm_listener, event->id);
    }

    rdma_destroy_id(event->id);

    if (rdma_ack_cm_event(event)) {
        ucs_error("rdmacm_listener %p: rdma_ack_cm_event failed with error: %m",
                  rdmacm_listener);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_listener_t)
{
    rdma_destroy_id(self->id);
}

UCS_CLASS_DEFINE(uct_rdmacm_listener_t, uct_listener_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_listener_t, uct_listener_t,
                          uct_cm_h , const struct sockaddr *,
                          socklen_t ,
                          const uct_listener_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_listener_t, uct_listener_t);

static void uct_rdmacm_cm_cleanup(uct_cm_h cm)
{
    uct_rdmacm_cm_t *rdmacm_cm = ucs_derived_of(cm, uct_rdmacm_cm_t);
    ucs_status_t status;

    status = ucs_async_remove_handler(rdmacm_cm->ev_ch->fd, 1);
    if (status != UCS_OK) {
        ucs_warn("failed to remove event handler for fd %d: %s",
                 rdmacm_cm->ev_ch->fd, ucs_status_string(status));
    }
    rdma_destroy_event_channel(rdmacm_cm->ev_ch);
    ucs_free(cm);
}

static ucs_status_t uct_rdmacm_cm_query(uct_cm_h cm, uct_cm_attr_t *cm_attr)
{
    memset(cm_attr, 0, sizeof(*cm_attr));
    cm_attr->max_conn_priv = UCT_RDMACM_CM_MAX_CONN_PRIV;
    return UCS_OK;
}

UCS_CLASS_DECLARE_DELETE_FUNC(uct_rdmacm_cep_t, uct_ep_t);

ucs_status_t uct_rdmacm_cep_disconnect(uct_ep_h ep, unsigned flags)
{
    uct_rdmacm_cep_t *cep = ucs_derived_of(ep, uct_rdmacm_cep_t);
    struct sockaddr *remote_addr = rdma_get_peer_addr(cep->id);
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];

    if (rdma_disconnect(cep->id)) {
        ucs_error("rdmacm_cm ep %p (id=%p) failed to disconnect from peer %p",
                  cep, cep->id,
                  ucs_sockaddr_str(remote_addr, ip_port_str, UCS_SOCKADDR_STRING_LEN));
        return UCS_ERR_IO_ERROR;
    }

    ucs_debug("rdmacm_cm ep %p (id=%p) disconnecting from peer :%s", cep, cep->id,
              ucs_sockaddr_str(remote_addr, ip_port_str, UCS_SOCKADDR_STRING_LEN));
    return UCS_OK;
}

uct_base_iface_t dummy_iface = {
    .super = {
        .ops = {
            .ep_pending_purge = (void *)ucs_empty_function_return_success,
            .ep_disconnect    = uct_rdmacm_cep_disconnect,
            .ep_destroy       = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_cep_t)
        }
    }
};

static ucs_status_t uct_rdmacm_create_dummy_cq_qp(struct rdma_cm_id *id,
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
        goto out;
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
        goto out_destroy_cq;
    }

    ucs_debug("created ud QP %p with qp_num: 0x%x and cq %p on rdmacm_id %p",
              qp, qp->qp_num, cq, id);

    *cq_p = cq;
    *qp_p = qp;

    return UCS_OK;

out_destroy_cq:
    ibv_destroy_cq(cq);
out:
    return status;
}

static ucs_status_t
uct_rdamcm_cm_fill_other_conn_param(struct rdma_conn_param *conn_param,
                                    const uct_rdmacm_priv_data_hdr_t *hdr,
                                    uct_rdmacm_cep_t *cep)
{
//    static uint32_t qp_num = 0xffffff;
//
//    /* conn_param->private_data should be filled outside this function */
//    conn_param->private_data_len    = sizeof(*hdr) + hdr->length;
//    conn_param->responder_resources = 1;    //remove. all needs to be 0. with memset to 0
//                                            //    before calling this function
//    conn_param->initiator_depth     = 1;    //remove
//    conn_param->retry_count         = 7;    //remove
//    conn_param->rnr_retry_count     = 7;    //remove
//    conn_param->qp_num              = qp_num--;/// create (only create) dummy ud qp (to get a unique qp_num) on the rdmacm_id and destroy this qp when this rdmacm_id is destoryed
//    if (qp_num == 0) {
//        qp_num = 0xffffff;
//    }
//
//    return UCS_OK;

    /////////CREATES a diff of around 100ms in many_conns/clients* gtests on p2p2 interface
    /* create a dummy qp in order to get a unique qp_num to provide to librdmacm */
    ucs_status_t status;
    struct ibv_qp *qp;
    struct ibv_cq *cq;

    status = uct_rdmacm_create_dummy_cq_qp(cep->id, &cq, &qp);
    if (status != UCS_OK) {
        return status;
    }

    cep->cq                      = cq;
    cep->qp                      = qp;
    conn_param->qp_num           = qp->qp_num;
    conn_param->private_data_len = sizeof(*hdr) + hdr->length;

    return UCS_OK;
}

static ucs_status_t uct_rdamcm_cm_init_client_ep(uct_rdmacm_cep_t *cep,
                                                 const uct_ep_params_t *params)
{
    ucs_status_t status;

    cep->wireup.client.connect_cb = params->sockaddr_connect_cb.client;

    if (rdma_create_id(cep->cm->ev_ch, &cep->id, cep, RDMA_PS_TCP)) {
        ucs_error("rdma_create_id() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    if (rdma_resolve_addr(cep->id, NULL,
                          (struct sockaddr *)params->sockaddr->addr,
                          1000/* TODO */)) {
        ucs_error("rdma_resolve_addr() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_destroy_id;
    }

    return UCS_OK;

err_destroy_id:
    rdma_destroy_id(cep->id);
err:
    return status;
}

static ucs_status_t uct_rdamcm_cm_init_server_ep(uct_rdmacm_cep_t *cep,
                                                 const uct_ep_params_t *params)
{
    struct rdma_cm_event *event = (struct rdma_cm_event *)params->conn_request;
    struct rdma_conn_param conn_param;
    uct_rdmacm_priv_data_hdr_t hdr;
    char dev_name[UCT_DEVICE_NAME_MAX];
    ssize_t priv_data_ret;
    ucs_status_t status = UCS_OK;

    cep->wireup.server.connect_cb = params->sockaddr_connect_cb.server;

    /* TODO: migrate id if cm is different */
    ucs_assert(event->listen_id->channel == cep->cm->ev_ch);

    cep->id          = event->id;
    cep->id->context = cep;
    uct_rdmacm_cm_id_to_dev_name(cep->id, dev_name);

    memset(&conn_param, 0, sizeof(conn_param));
    conn_param.private_data = ucs_alloca(UCT_RDMACM_CM_MAX_CONN_PRIV +
                                         sizeof(uct_rdmacm_priv_data_hdr_t));

    priv_data_ret = cep->wireup.priv_pack_cb(cep->user_data, dev_name,
                                            (void*)(conn_param.private_data +
                                             sizeof(uct_rdmacm_priv_data_hdr_t)));
    if ((priv_data_ret < 0) || (priv_data_ret > UCT_RDMACM_CM_MAX_CONN_PRIV)) {
        ucs_error("failed to pack data on the server (ep=%p). packed data size: %zu.",
                  cep, priv_data_ret);
        status = UCS_ERR_INVALID_PARAM;
        goto out;
    }

    hdr.length = (uint8_t)priv_data_ret;
    hdr.status = UCS_OK;

    memcpy((void*)conn_param.private_data, &hdr, sizeof(uct_rdmacm_priv_data_hdr_t));

    status = uct_rdamcm_cm_fill_other_conn_param(&conn_param, &hdr, cep);
    if (status != UCS_OK) {
        goto out;
    }

    if (rdma_accept(event->id, &conn_param)) {
        ucs_error("rdma_accept(on id=%p) failed: %m", event->id);
        status = UCS_ERR_IO_ERROR;
    }
//do we need to destroy event->id here?
    if (rdma_ack_cm_event(event)) {
        status = UCS_ERR_IO_ERROR;
    }

out:
    return status;
}

UCS_CLASS_INIT_FUNC(uct_rdmacm_cep_t, const uct_ep_params_t *params)
{
    ucs_status_t status = UCS_OK;

    if (!(params->field_mask & UCT_EP_PARAM_FIELD_CM)) {
        return UCS_ERR_INVALID_PARAM;
    }

    if (!(params->field_mask & UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS) ||
        !(params->sockaddr_cb_flags & UCT_CB_FLAG_ASYNC)) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (!(params->field_mask & (UCT_EP_PARAM_FIELD_SOCKADDR |
                                UCT_EP_PARAM_FIELD_CONN_REQUEST))) {
        return UCS_ERR_INVALID_PARAM;
    }

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &dummy_iface);

    self->cm                  = ucs_derived_of(params->cm, uct_rdmacm_cm_t);
    self->wireup.priv_pack_cb = (params->field_mask &
                                 UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB) ?
                                params->sockaddr_pack_cb : NULL;
    self->disconnect_cb       = (params->field_mask &
                                 UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB) ?
                                params->disconnect_cb : NULL;
    self->user_data           = (params->field_mask &
                                 UCT_EP_PARAM_FIELD_USER_DATA) ?
                                params->user_data : NULL;

    if (params->field_mask & UCT_EP_PARAM_FIELD_SOCKADDR) {
        status = uct_rdamcm_cm_init_client_ep(self, params);
    } else {
        ucs_assert(params->field_mask & UCT_EP_PARAM_FIELD_CONN_REQUEST);
        status = uct_rdamcm_cm_init_server_ep(self, params);
    }

    if (status == UCS_OK) {
        ucs_debug("created an RDMACM endpoint %p on CM %p. rdmacm_id: %p",
                  self, self->cm, self->id);
    }

    return status;
}

UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_cep_t)
{
    uct_priv_worker_t *worker_priv;
    int ret;

    worker_priv = ucs_derived_of(self->cm->worker, uct_priv_worker_t);

    UCS_ASYNC_BLOCK(worker_priv->async);

    ret = ibv_destroy_qp(self->qp);
    if (ret != 0) {
        ucs_warn("ibv_destroy_qp() returned %d: %m", ret);
    }

    ret = ibv_destroy_cq(self->cq);
    if (ret != 0) {
        ucs_warn("ibv_destroy_cq() returned %d: %m", ret);
    }

    rdma_destroy_id(self->id);
    UCS_ASYNC_UNBLOCK(worker_priv->async);
}

UCS_CLASS_DEFINE(uct_rdmacm_cep_t, uct_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_cep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_cep_t, uct_ep_t);

uct_cm_ops_t uct_rdmacm_cm_ops = {
    .close            = uct_rdmacm_cm_cleanup,
    .cm_query         = uct_rdmacm_cm_query,
    .listener_create  = UCS_CLASS_NEW_FUNC_NAME(uct_rdmacm_listener_t),
    .listener_reject  = uct_rdmacm_listener_reject,
    .listener_destroy = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_listener_t),
    .ep_create        = UCS_CLASS_NEW_FUNC_NAME(uct_rdmacm_cep_t)
};

static ucs_status_t uct_rdmacm_cm_id_to_dev_addr(struct rdma_cm_id *cm_id,
                                                 uct_device_addr_t **dev_addr_p,
                                                 size_t *dev_addr_len_p)
{
    struct ibv_port_attr port_arrt;
//    uct_ib_address_t addr_dummy;
    uct_ib_address_t *dev_addr;
    struct ibv_qp_attr qp_attr;
    size_t addr_length;
    int qp_attr_mask;

    qp_attr.qp_state = IBV_QPS_RTR;
    /* get the qp attributes in order to modify the qp state.
     * the ah_attr fields from them are required to extract the device address
     * of the remote peer.
     */
    if (rdma_init_qp_attr(cm_id, &qp_attr, &qp_attr_mask)) {
        ucs_error("rdma_init_qp_attr failed. id %p, state: %d. %m",
                  cm_id, qp_attr.qp_state);
        return UCS_ERR_IO_ERROR;
    }

    if (ibv_query_port(cm_id->verbs, cm_id->port_num, &port_arrt)) {
        ucs_error("ibv_query_port failed. %m");
        return UCS_ERR_IO_ERROR;
    }

    addr_length = uct_ib_addr_size(&qp_attr.ah_attr.grh.dgid, 0,
                                   port_arrt.link_layer == IBV_LINK_LAYER_ETHERNET);

    dev_addr = ucs_malloc(addr_length, "IB device address");
    if (dev_addr == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

//    uct_ib_set_device_address_flags(&qp_attr.ah_attr.grh.dgid, 0,
//                                    port_arrt.link_layer == IBV_LINK_LAYER_ETHERNET, dev_addr);
////                                    &addr_dummy);

//    *dev_addr = addr_dummy;

    uct_ib_address_pack(&qp_attr.ah_attr.grh.dgid, qp_attr.ah_attr.dlid,
                        port_arrt.link_layer == IBV_LINK_LAYER_ETHERNET, 0,
                        dev_addr);

    *dev_addr_p     = (uct_device_addr_t *)dev_addr;
    *dev_addr_len_p = addr_length;
    return UCS_OK;
}

static void
uct_rdmacm_cm_process_event(uct_rdmacm_cm_t *cm, struct rdma_cm_event *event)
{
    struct sockaddr            *remote_addr = rdma_get_peer_addr(event->id);
    uint8_t                    ack_event = 1;
    struct rdma_conn_param     conn_param;
    uct_rdmacm_priv_data_hdr_t hdr;
    uct_rdmacm_cep_t           *cep;
    uct_rdmacm_listener_t      *listener;
    char                       dev_name[UCT_DEVICE_NAME_MAX];
    char                       ip_port_str[UCS_SOCKADDR_STRING_LEN];
    uct_device_addr_t          *dev_addr;
    size_t                     addr_length;
    uct_cm_remote_data_t       remote_data;
    ucs_status_t               status;
    ssize_t                    priv_data_ret;

    ucs_trace("rdmacm event (fd=%d cm_id %p): %s. Peer: %s.",
              cm->ev_ch->fd, event->id, rdma_event_str(event->event),
              ucs_sockaddr_str(remote_addr, ip_port_str, UCS_SOCKADDR_STRING_LEN));

    /* The following applies for rdma_cm_id of type RDMA_PS_TCP only */
    switch (event->event) {
    case RDMA_CM_EVENT_ADDR_RESOLVED:
        /* Client side event */
        if (rdma_resolve_route(event->id, 1000 /* TODO */)){
            ucs_error("rdma_resolve_route(to addr=%s) failed: %m",
                      ucs_sockaddr_str(remote_addr, ip_port_str,
                                       UCS_SOCKADDR_STRING_LEN));
        }
        cep = (uct_rdmacm_cep_t *)event->id->context;
        ucs_assert(event->id == cep->id);
        break;
    case RDMA_CM_EVENT_ROUTE_RESOLVED:
        /* Client side event */
        cep = (uct_rdmacm_cep_t *)event->id->context;
        ucs_assert(event->id == cep->id);

        uct_rdmacm_cm_id_to_dev_name(cep->id, dev_name);

        memset(&conn_param, 0, sizeof(conn_param));
        conn_param.private_data = ucs_alloca(UCT_RDMACM_CM_MAX_CONN_PRIV +
                                             sizeof(uct_rdmacm_priv_data_hdr_t));
        priv_data_ret = cep->wireup.priv_pack_cb(cep->user_data, dev_name,
                                                 (void*)(conn_param.private_data +
                                                  sizeof(uct_rdmacm_priv_data_hdr_t)));

        if ((priv_data_ret < 0) || (priv_data_ret > UCT_RDMACM_CM_MAX_CONN_PRIV)) {
            ucs_error("failed to pack data on the client (ep=%p). packed data size: %zu.",
                      cep, priv_data_ret);
            break;
        }

        hdr.length = (uint8_t)priv_data_ret;
        hdr.status = UCS_OK;

        memcpy((void*)conn_param.private_data, &hdr, sizeof(uct_rdmacm_priv_data_hdr_t));
        status = uct_rdamcm_cm_fill_other_conn_param(&conn_param, &hdr, cep);
        if (status != UCS_OK) {
            break;
        }

        if (rdma_connect(cep->id, &conn_param)) {
            ucs_error("rdma_connect(to addr=%s) failed: %m",
                      ucs_sockaddr_str(remote_addr, ip_port_str,
                                       UCS_SOCKADDR_STRING_LEN));

        }
        break;
    case RDMA_CM_EVENT_CONNECT_REQUEST:
        /* Server side event */
        listener = event->listen_id->context;
        hdr      = *(uct_rdmacm_priv_data_hdr_t *)event->param.conn.private_data;
        ucs_assert(hdr.status == UCS_OK);

        uct_rdmacm_cm_id_to_dev_name(event->id, dev_name);
        status = uct_rdmacm_cm_id_to_dev_addr(event->id, &dev_addr, &addr_length);
        if (status != UCS_OK) {
            hdr.status = UCS_ERR_REJECTED;
            hdr.length = 0;
            if (rdma_reject(event->id, &hdr, sizeof(hdr))) {
                ucs_error("rdma_reject (id=%p) failed with error: %m", event->id);
            }
            rdma_destroy_id(event->id);
            break;
        }

        remote_data.field_mask = UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR        |
                                 UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR_LENGTH |
                                 UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA  |
                                 UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA_LENGTH;
        remote_data.dev_addr              = dev_addr;
        remote_data.dev_addr_length       = addr_length;
        remote_data.conn_priv_data        = UCS_PTR_BYTE_OFFSET(event->param.conn.private_data,
                                                                sizeof(uct_rdmacm_priv_data_hdr_t));
        remote_data.conn_priv_data_length = hdr.length;

        listener->conn_request_cb(&listener->super, listener->user_data,
                                  dev_name, event, &remote_data);
        ucs_free(dev_addr);
        /* The server will ack the event after accepting the request
         * (in ep_create). */
        ack_event = 0;
        break;
    case RDMA_CM_EVENT_CONNECT_RESPONSE:
        /* Client side event */
        cep = event->id->context;
        ucs_assert(event->id == cep->id);

        status = uct_rdmacm_cm_id_to_dev_addr(event->id, &dev_addr, &addr_length);
        if (status == UCS_OK) {
            hdr = *(uct_rdmacm_priv_data_hdr_t *)event->param.conn.private_data;
            remote_data.field_mask = UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR        |
                                     UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR_LENGTH |
                                     UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA  |
                                     UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA_LENGTH;
            remote_data.dev_addr              = dev_addr;
            remote_data.dev_addr_length       = addr_length;
            remote_data.conn_priv_data        = UCS_PTR_BYTE_OFFSET(event->param.conn.private_data,
                                                                    sizeof(uct_rdmacm_priv_data_hdr_t));
            remote_data.conn_priv_data_length = hdr.length;

            cep->wireup.client.connect_cb(&cep->super.super, cep->user_data,
                                          &remote_data, hdr.status);

            ucs_free(dev_addr);

            if (rdma_establish(event->id)) {
                ucs_error("rdma_establish on ep %p (to server addr=%s) failed: %m",
                          cep, ucs_sockaddr_str(remote_addr, ip_port_str,
                                                UCS_SOCKADDR_STRING_LEN));
            }
        } else {
            ucs_error("client (ep=%p id=%p) failed to process a connect response."
                      "not sending ack back to the server", cep, event->id);
        }
        break;
    case RDMA_CM_EVENT_REJECTED:
        /* Client side event */
        cep = event->id->context;
        ucs_assert(event->id == cep->id);

        /* Network reject or API call */
        remote_data.field_mask = 0;
        if (event->param.conn.private_data != NULL) {
            /* Server side called rdma_reject from this function */
            hdr = *(uct_rdmacm_priv_data_hdr_t *)event->param.conn.private_data;
            if (hdr.length > 0) {
                remote_data.field_mask = UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA |
                                         UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA_LENGTH;
                remote_data.conn_priv_data = UCS_PTR_BYTE_OFFSET(event->param.conn.private_data,
                                                                 sizeof(uct_rdmacm_priv_data_hdr_t));
                remote_data.conn_priv_data_length = hdr.length;
            }
        }

        cep->wireup.client.connect_cb(&cep->super.super, cep->user_data,
                                      &remote_data, UCS_ERR_REJECTED);
        break;
    case RDMA_CM_EVENT_ESTABLISHED:
        /* Server side event */
        cep = event->id->context;
        ucs_assert(event->id == cep->id);
        cep->wireup.server.connect_cb(&cep->super.super, cep->user_data, UCS_OK);
        break;
    case RDMA_CM_EVENT_DISCONNECTED:
        /* Client and Server side event */
        cep = event->id->context;
        cep->disconnect_cb(&cep->super.super, cep->user_data);
        break;
    case RDMA_CM_EVENT_TIMEWAIT_EXIT:
        /* This event is generated when the QP associated with the connection
         * has exited its timewait state and is now ready to be re-used.
         * After a QP has been disconnected, it is maintained in a timewait
         * state to allow any in flight packets to exit the network.
         * After the timewait state has completed, the rdma_cm will report this event.*/
        break;
        /* client error events */
    case RDMA_CM_EVENT_UNREACHABLE:
    case RDMA_CM_EVENT_ADDR_ERROR:
    case RDMA_CM_EVENT_ROUTE_ERROR:
        /* client and server error events */
    case RDMA_CM_EVENT_CONNECT_ERROR:
    case RDMA_CM_EVENT_DEVICE_REMOVAL:
    case RDMA_CM_EVENT_ADDR_CHANGE:
        ucs_error("received an error event %s. status = %d. Peer: %s.",
                  rdma_event_str(event->event), event->status,
                  ucs_sockaddr_str(remote_addr, ip_port_str,
                                   UCS_SOCKADDR_STRING_LEN));
        break;
    default:
        ucs_warn("unexpected RDMACM event: %s", rdma_event_str(event->event));
        break;
    }

    if (ack_event && rdma_ack_cm_event(event)) {
        ucs_warn("rdma_ack_cm_event() failed on event %s: %m",
                 rdma_event_str(event->event));
    }
}

static void uct_rdmacm_cm_event_handler(int fd, void *arg)
{
    uct_rdmacm_cm_t      *cm = (uct_rdmacm_cm_t *)arg;
    struct rdma_cm_event *event;
    int                  ret;

    for (;;) {
        /* Fetch an event */
        ret = rdma_get_cm_event(cm->ev_ch, &event);
        if (ret) {
            /* EAGAIN (in a non-blocking rdma_get_cm_event) means that
             * there are no more events */
            if (errno != EAGAIN) {
                ucs_warn("rdma_get_cm_event() failed: %m");
            }
            return;
        }
        uct_rdmacm_cm_process_event(cm, event);
    }
}

ucs_status_t uct_rdmacm_cm_open(uct_component_h component, uct_worker_h worker,
                                uct_cm_h *uct_cm_p)
{
    uct_priv_worker_t *worker_priv;
    uct_rdmacm_cm_t *rdmacm_cm;
    ucs_status_t status;

    rdmacm_cm = ucs_calloc(1, sizeof(uct_rdmacm_cm_t), "uct_rdmacm_cm_open");
    if (rdmacm_cm == NULL) {
        ucs_error("failed to allocate an rdmacm_cm %m");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    rdmacm_cm->super.ops       = &uct_rdmacm_cm_ops;
    rdmacm_cm->super.component = component;
    rdmacm_cm->worker          = worker;
    rdmacm_cm->ev_ch           = rdma_create_event_channel();
    if (rdmacm_cm->ev_ch == NULL) {
        ucs_error("rdma_create_event_channel failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_free_cm;
    }

    /* Set the event_channel fd to non-blocking mode
     * (so that rdma_get_cm_event won't be blocking) */
    status = ucs_sys_fcntl_modfl(rdmacm_cm->ev_ch->fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        status = UCS_ERR_IO_ERROR;
        goto err_destroy_ev_ch;
    }

    worker_priv = ucs_derived_of(worker, uct_priv_worker_t);
    status = ucs_async_set_event_handler(worker_priv->async->mode,
                                         rdmacm_cm->ev_ch->fd, POLLIN,
                                         uct_rdmacm_cm_event_handler, rdmacm_cm,
                                         worker_priv->async);
    if (status != UCS_OK) {
        goto err_destroy_ev_ch;
    }

    *uct_cm_p = &rdmacm_cm->super;
    return UCS_OK;

err_destroy_ev_ch:
    rdma_destroy_event_channel(rdmacm_cm->ev_ch);
err_free_cm:
    ucs_free(rdmacm_cm);
err:
    return status;
}

#endif /* HAVE_RDMACM_QP_LESS */
