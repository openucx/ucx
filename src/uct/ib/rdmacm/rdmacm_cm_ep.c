/**
* Copyright (C) Mellanox Technologies Ltd. 2019-2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rdmacm_cm_ep.h"
#include "rdmacm_cm.h"
#include <ucs/arch/bitops.h>
#include <ucs/sys/sock.h>
#include <ucs/async/async.h>


const char* uct_rdmacm_cm_ep_str(uct_rdmacm_cm_ep_t *cep, char *str,
                                 size_t max_len)
{
    struct sockaddr *local_addr = cep->id ? rdma_get_local_addr(cep->id) : NULL;
    struct sockaddr *remote_addr = cep->id ? rdma_get_peer_addr(cep->id) : NULL;
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

    if ((local_addr != NULL) && ucs_sockaddr_is_known_af(local_addr)) {
        ucs_sockaddr_str(local_addr, local_ip_port_str, UCS_SOCKADDR_STRING_LEN);
    } else {
        ucs_strncpy_safe(local_ip_port_str, "<invalid>", UCS_SOCKADDR_STRING_LEN);
    }

    if ((remote_addr != NULL) && ucs_sockaddr_is_known_af(remote_addr)) {
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

int uct_rdmacm_ep_client_is_resolved(uct_rdmacm_cm_ep_t *cep)
{
    ucs_assert(cep->flags & UCT_RDMACM_CM_EP_ON_CLIENT);
    return cep->flags & UCT_RDMACM_CM_EP_CLIENT_RESOLVE_CB_INVOKED;
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

static int uct_rdmacm_cm_ep_disconnect_cb(uct_rdmacm_cm_ep_t *cep)
{
    if (uct_rdmacm_ep_is_connected(cep)) {
        /* Already connected, so call disconnect callback */
        uct_cm_ep_disconnect_cb(&cep->super);
        return 1;
    }

    return 0;
}

static void uct_rdmacm_cm_ep_client_error_cb(uct_rdmacm_cm_ep_t *cep,
                                             uct_cm_remote_data_t *remote_data,
                                             ucs_status_t status)
{
    ucs_assert(cep->flags & UCT_RDMACM_CM_EP_ON_CLIENT);

    if (uct_rdmacm_cm_ep_disconnect_cb(cep)) {
        return;
    }

    if (uct_rdmacm_ep_client_is_resolved(cep)) {
        /* Not connected yet, so call client side connect callback with err
         * status. */
        uct_rdmacm_cm_ep_client_connect_cb(cep, remote_data, status);
        return;
    }

    /* Not resolved yet, so call client side resolve callback with err
     * status. */
    uct_rdmacm_cm_ep_resolve_cb(cep, status);
}

static void uct_rdmacm_cm_ep_server_error_cb(uct_rdmacm_cm_ep_t *cep,
                                             ucs_status_t status)
{
    ucs_assert(cep->flags & UCT_RDMACM_CM_EP_ON_SERVER);

    if (uct_rdmacm_cm_ep_disconnect_cb(cep)) {
        return;
    }

    /* Not connected yet, so call server side notify callback with err
     * status */
    uct_rdmacm_cm_ep_server_conn_notify_cb(cep, status);
}

void uct_rdmacm_cm_ep_error_cb(uct_rdmacm_cm_ep_t *cep,
                               uct_cm_remote_data_t *remote_data,
                               ucs_status_t status)
{
    ucs_assertv((status != UCS_OK) && (cep->status == UCS_OK) &&
                !(cep->flags & UCT_RDMACM_CM_EP_FAILED),
                "ep %p: calling err callback with status %s on ep with status "
                "%s flags 0x%"PRIx16, cep, ucs_status_string(status),
                ucs_status_string(cep->status), cep->flags);

    if (cep->flags & UCT_RDMACM_CM_EP_ON_CLIENT) {
        uct_rdmacm_cm_ep_client_error_cb(cep, remote_data, status);
    } else {
        uct_rdmacm_cm_ep_server_error_cb(cep, status);
    }
}

void uct_rdmacm_cm_ep_set_failed(uct_rdmacm_cm_ep_t *cep,
                                 uct_cm_remote_data_t *remote_data,
                                 ucs_status_t status, int invoke_cb)
{
    ucs_assert(status != UCS_OK);
    ucs_assert(ucs_async_is_blocked(uct_rdmacm_cm_ep_get_async(cep)));

    if (cep->flags & UCT_RDMACM_CM_EP_FAILED) {
        return;
    }

    if (invoke_cb) {
        uct_rdmacm_cm_ep_error_cb(cep, remote_data, status);
    }

    /* update the state after err callback to return correct err code from
     * @ref uct_ep_disconnect */
    cep->flags |= UCT_RDMACM_CM_EP_FAILED;
    cep->status = status;
}

ucs_status_t uct_rdmacm_ep_query(uct_ep_h ep, uct_ep_attr_t *ep_attr)
{
    uct_rdmacm_cm_ep_t *rdmacm_ep = ucs_derived_of(ep, uct_rdmacm_cm_ep_t);
    struct sockaddr *local_saddr, *remote_saddr;
    ucs_status_t status;

    if (ep_attr->field_mask & UCT_EP_ATTR_FIELD_LOCAL_SOCKADDR) {
        local_saddr = rdma_get_local_addr(rdmacm_ep->id);
        status = ucs_sockaddr_copy((struct sockaddr*)&ep_attr->local_address,
                                   local_saddr);
        if (status != UCS_OK) {
            return status;
        }
    }

    if (ep_attr->field_mask & UCT_EP_ATTR_FIELD_REMOTE_SOCKADDR) {
        remote_saddr = rdma_get_peer_addr(rdmacm_ep->id);
        status = ucs_sockaddr_copy((struct sockaddr*)&ep_attr->remote_address,
                                   remote_saddr);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

ucs_status_t uct_rdmacm_cm_ep_conn_notify(uct_ep_h ep)
{
    uct_rdmacm_cm_ep_t *cep                 = ucs_derived_of(ep, uct_rdmacm_cm_ep_t);
    struct sockaddr *remote_addr            = rdma_get_peer_addr(cep->id);
    uct_rdmacm_cm_t UCS_V_UNUSED *rdmacm_cm = uct_rdmacm_cm_ep_get_cm(cep);
    char ep_str[UCT_RDMACM_EP_STRING_LEN];
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t status;

    UCS_ASYNC_BLOCK(uct_rdmacm_cm_ep_get_async(cep));

    ucs_trace("%s rdma_establish on client (cm_id %p, rdmacm %p, event_channel=%p)",
              uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
              cep->id, rdmacm_cm, rdmacm_cm->ev_ch);

    if (cep->flags & (UCT_RDMACM_CM_EP_FAILED |
                      UCT_RDMACM_CM_EP_GOT_DISCONNECT)) {
        goto out;
    }

    if (rdma_establish(cep->id)) {
        uct_cm_ep_peer_error(&cep->super,
                             "rdma_establish on ep %p (to server addr=%s) failed: %m",
                             cep, ucs_sockaddr_str(remote_addr, ip_port_str,
                                                   UCS_SOCKADDR_STRING_LEN));
        cep->status = UCS_ERR_CONNECTION_RESET;
        cep->flags |= UCT_RDMACM_CM_EP_FAILED;
        goto out;
    }

out:
    status = cep->status;
    UCS_ASYNC_UNBLOCK(uct_rdmacm_cm_ep_get_async(cep));
    return status;
}

/**
 * Allocate a reserved QPN either from the last FW object allocated,
 * or by allocating a new one. When find a free QPN in an object, it
 * always starts from last allocation position, to make sure the QPN
 * always move forward to prevent stale QPN.
 */
static ucs_status_t
uct_rdamcm_cm_ep_create_reserved_qpn(uct_rdmacm_cm_ep_t *cep,
                                     uct_rdmacm_cm_device_context_t *ctx)
{
    uint32_t qpns_per_obj = UCS_BIT(ctx->log_reserved_qpn_granularity);
    uct_rdmacm_cm_reserved_qpn_blk_t *blk;
    ucs_status_t status;

    ucs_spin_lock(&ctx->lock);

    blk = ucs_list_tail(&ctx->blk_list, uct_rdmacm_cm_reserved_qpn_blk_t, entry);
    if (ucs_list_is_empty(&ctx->blk_list) ||
        (blk->next_avail_qpn_offset >= qpns_per_obj)) {
        status = uct_rdmacm_cm_reserved_qpn_blk_alloc(ctx, cep->id->verbs,
                                                      UCS_LOG_LEVEL_ERROR,
                                                      &blk);
        if (status != UCS_OK) {
            goto out;
        }

        ucs_list_add_tail(&ctx->blk_list, &blk->entry);
    }

    cep->qpn = blk->first_qpn + blk->next_avail_qpn_offset;
    ucs_debug("created reserved qpn 0x%x on rdmacm_id %p",
              cep->qpn, cep->id);
    cep->blk = blk;
    blk->refcount++;
    blk->next_avail_qpn_offset++;
    status = UCS_OK;

out:
    ucs_spin_unlock(&ctx->lock);
    return status;
}

/**
 * Deallocate a reserved QPN. The FW object is destroyed only when all QPNs
 * in this object were used and freed.
 */
static void
uct_rdamcm_cm_ep_destroy_reserved_qpn(uct_rdmacm_cm_device_context_t *ctx,
                                      uct_rdmacm_cm_ep_t *cep)
{
    uint32_t qpns_per_obj;

    ucs_assert(cep->flags & UCT_RDMACM_CM_EP_QPN_CREATED);

    /* Reserved QP number is created on-demand when we send the private data */
    ucs_debug("cm ep destroy reserved qpn 0x%x on rdmacm_id %p",
              cep->qpn, cep->id);

    qpns_per_obj = UCS_BIT(ctx->log_reserved_qpn_granularity);

    ucs_spin_lock(&ctx->lock);

    --cep->blk->refcount;
    if ((cep->blk->next_avail_qpn_offset >= qpns_per_obj) &&
        (cep->blk->refcount == 0)) {
        ucs_list_del(&cep->blk->entry);
        uct_rdmacm_cm_reserved_qpn_blk_release(cep->blk);
    }

    ucs_spin_unlock(&ctx->lock);
    return;
}

static void
uct_rdmacm_cm_ep_destroy_dummy_qp(uct_rdmacm_cm_device_context_t *ctx,
                                  uct_rdmacm_cm_ep_t *cep)
{
    int ret;

    ucs_assert(cep->flags & UCT_RDMACM_CM_EP_QPN_CREATED);
    ucs_assert_always(cep->qp != NULL);

    ucs_debug("cm ep destroy dummy qp_num 0x%x on rdmacm_id %p",
              cep->qp->qp_num, cep->id);

    ret = ibv_destroy_qp(cep->qp);
    if (ret != 0) {
        ucs_warn("ibv_destroy_qp() returned %d: %m", ret);
    }

    cep->qp = NULL;

    ucs_assert(ctx->num_dummy_qps > 0);
    --ctx->num_dummy_qps;
}

static ucs_status_t
uct_rdmacm_cm_create_dummy_qp(uct_rdmacm_cm_device_context_t *ctx,
                              uct_rdmacm_cm_ep_t *cep)
{
    struct ibv_qp_init_attr qp_init_attr = {0};
    struct ibv_qp *qp;

    /* Create a dummy UD qp */
    qp_init_attr.send_cq          = ctx->cq;
    qp_init_attr.recv_cq          = ctx->cq;
    qp_init_attr.qp_type          = IBV_QPT_UD;
    qp_init_attr.cap.max_send_wr  = 2;
    qp_init_attr.cap.max_recv_wr  = 2;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;

    qp = ibv_create_qp(cep->id->pd, &qp_init_attr);
    if (qp == NULL) {
        ucs_error("failed to create a dummy ud qp. %m");
        return UCS_ERR_IO_ERROR;
    }

    ucs_debug("created ud QP %p with qp_num: 0x%x and cq %p on rdmacm_id %p",
              qp, qp->qp_num, ctx->cq, cep->id);

    ++ctx->num_dummy_qps;
    cep->qp = qp;
    return UCS_OK;
}

static ucs_status_t
uct_rdamcm_cm_ep_create_qpn(uct_rdmacm_cm_device_context_t *ctx,
                            uct_rdmacm_cm_ep_t *cep, uint32_t *qpn_p)
{
    ucs_status_t status;

    ucs_assert(ucs_async_is_blocked(uct_rdmacm_cm_ep_get_async(cep)));

    if (ctx->use_reserved_qpn) {
        status = uct_rdamcm_cm_ep_create_reserved_qpn(cep, ctx);
    } else {
        /* create a dummy qp in order to get a unique qp_num to provide to librdmacm */
        status = uct_rdmacm_cm_create_dummy_qp(ctx, cep);
    }
    if (status != UCS_OK) {
        return status;
    }

    cep->flags |= UCT_RDMACM_CM_EP_QPN_CREATED;

    if (ctx->use_reserved_qpn) {
        *qpn_p = cep->qpn;
    } else {
        *qpn_p = cep->qp->qp_num;
    }

    return UCS_OK;
}

static void uct_rdamcm_cm_ep_destroy_qpn(uct_rdmacm_cm_ep_t *cep)
{
    uct_rdmacm_cm_device_context_t *ctx;
    ucs_status_t status;

    ucs_assert(ucs_async_is_blocked(uct_rdmacm_cm_ep_get_async(cep)));

    if (!(cep->flags & UCT_RDMACM_CM_EP_QPN_CREATED)) {
        return;
    }

    status = uct_rdmacm_cm_get_device_context(uct_rdmacm_cm_ep_get_cm(cep),
                                              cep->id->verbs, &ctx);
    if (status != UCS_OK) {
        return;
    }

    if (ctx->use_reserved_qpn) {
        uct_rdamcm_cm_ep_destroy_reserved_qpn(ctx, cep);
    } else {
        uct_rdmacm_cm_ep_destroy_dummy_qp(ctx, cep);
    }

    cep->flags &= ~UCT_RDMACM_CM_EP_QPN_CREATED;
}

static ucs_status_t
uct_rdamcm_cm_ep_set_qp_num(struct rdma_conn_param *conn_param,
                            uct_rdmacm_cm_ep_t *cep)
{
    uct_rdmacm_cm_device_context_t *ctx;
    ucs_status_t status;

    status = uct_rdmacm_cm_get_device_context(uct_rdmacm_cm_ep_get_cm(cep),
                                              cep->id->verbs, &ctx);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_rdamcm_cm_ep_create_qpn(ctx, cep, &conn_param->qp_num);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert(cep->flags & UCT_RDMACM_CM_EP_QPN_CREATED);
    return UCS_OK;
}

ucs_status_t uct_rdmacm_cm_ep_pack_cb(uct_rdmacm_cm_ep_t *cep,
                                      void *private_data,
                                      size_t *priv_data_length_p)
{
    uct_cm_ep_priv_data_pack_args_t pack_args;

    /* Pack data to send inside rdmacm's conn_param to the remote peer */
    pack_args.field_mask = UCT_CM_EP_PRIV_DATA_PACK_ARGS_FIELD_DEVICE_NAME;
    uct_rdmacm_cm_id_to_dev_name(cep->id, pack_args.dev_name);

    return uct_cm_ep_pack_cb(&cep->super, cep->super.user_data, &pack_args,
                             private_data, uct_rdmacm_cm_get_max_conn_priv(),
                             priv_data_length_p);
}

ucs_status_t uct_rdmacm_cm_ep_resolve_cb(uct_rdmacm_cm_ep_t *cep,
                                         ucs_status_t status)
{
    uct_cm_ep_resolve_args_t args;

    args.field_mask = UCT_CM_EP_RESOLVE_ARGS_FIELD_STATUS;
    args.status     = status;
    if (status == UCS_OK) {
        args.field_mask |= UCT_CM_EP_RESOLVE_ARGS_FIELD_DEV_NAME;
        uct_rdmacm_cm_id_to_dev_name(cep->id, args.dev_name);
    }

    cep->flags |= UCT_RDMACM_CM_EP_CLIENT_RESOLVE_CB_INVOKED;
    return uct_cm_ep_resolve_cb(&cep->super, &args);
}

static ucs_status_t uct_rdamcm_cm_ep_client_init(uct_rdmacm_cm_ep_t *cep,
                                                 const uct_ep_params_t *params)
{
    uct_cm_base_ep_t *cm_ep    = &cep->super;
    uct_rdmacm_cm_t *rdmacm_cm = uct_rdmacm_cm_ep_get_cm(cep);
    char src_ip_port_str[UCS_SOCKADDR_STRING_LEN];
    char dst_ip_port_str[UCS_SOCKADDR_STRING_LEN];
    char ep_str[UCT_RDMACM_EP_STRING_LEN];
    ucs_status_t status;
    const struct sockaddr *src_addr;

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
              uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
              cep->id);
    src_addr = UCT_EP_PARAM_VALUE(params, local_sockaddr->addr,
                                  LOCAL_SOCKADDR, rdmacm_cm->config.src_addr);

    if (rdma_resolve_addr(cep->id, (struct sockaddr*)src_addr,
                          (struct sockaddr*)params->sockaddr->addr,
                          uct_rdmacm_cm_get_timeout(rdmacm_cm))) {
        ucs_diag("rdma_resolve_addr(src=%s, dst=%s) failed (%d): %m",
                 ucs_sockaddr_str(src_addr,
                                  src_ip_port_str, UCS_SOCKADDR_STRING_LEN),
                 ucs_sockaddr_str((struct sockaddr*)params->sockaddr->addr,
                                  dst_ip_port_str, UCS_SOCKADDR_STRING_LEN),
                 errno);
        status = UCS_ERR_IO_ERROR;
        goto err_destroy_id;
    }

    return UCS_OK;

err_destroy_id:
    uct_rdmacm_cm_destroy_id(cep->id);
err:
    return status;
}

static ucs_status_t
uct_rdamcm_cm_ep_server_send_priv_data(uct_rdmacm_cm_ep_t *cep,
                                       const uct_ep_params_t *params)
{
    uint8_t pack_priv_data[UCT_RDMACM_TCP_PRIV_DATA_LEN];
    size_t pack_priv_data_length;
    const void *priv_data;
    size_t priv_data_length;
    ucs_status_t status;

    UCS_ASYNC_BLOCK(uct_rdmacm_cm_ep_get_async(cep));

    /* Assume that rdma_ack_cm_event and rdma_migrate_id don't generate events,
     * so the status can't be changed. */
    ucs_assertv(cep->status == UCS_OK, "ep %p: status %s", cep,
                ucs_status_string(cep->status));

    if (ucs_test_all_flags(params->field_mask,
                           UCT_EP_PARAM_FIELD_PRIV_DATA |
                           UCT_EP_PARAM_FIELD_PRIV_DATA_LENGTH)) {
        priv_data        = params->private_data;
        priv_data_length = params->private_data_length;
    } else if (params->field_mask & UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB) {
        status = uct_rdmacm_cm_ep_pack_cb(cep, pack_priv_data,
                                          &pack_priv_data_length);
        if (status != UCS_OK) {
            uct_rdmacm_cm_reject(uct_rdmacm_cm_ep_get_cm(cep), cep->id);
            goto err;
        }

        priv_data        = &pack_priv_data;
        priv_data_length = pack_priv_data_length;
    } else {
        priv_data        = NULL;
        priv_data_length = 0;
    }

    status = uct_rdmacm_cm_ep_send_priv_data(cep, priv_data, priv_data_length);
    if (status == UCS_OK) {
        goto out;
    }

err:
    uct_rdmacm_cm_destroy_id(cep->id);
    cep->id = NULL;
out:
    UCS_ASYNC_UNBLOCK(uct_rdmacm_cm_ep_get_async(cep));
    return status;
}

static ucs_status_t uct_rdamcm_cm_ep_server_init(uct_rdmacm_cm_ep_t *cep,
                                                 const uct_ep_params_t *params)
{
    struct rdma_cm_event *event             = (struct rdma_cm_event*)params->conn_request;
    struct rdma_cm_id *id                   = event->id;
    struct rdma_event_channel *listen_ev_ch = event->listen_id->channel;
    uct_rdmacm_cm_t *cm                     = uct_rdmacm_cm_ep_get_cm(cep);
    ucs_status_t status;
    char ep_str[UCT_RDMACM_EP_STRING_LEN];

    status = uct_rdmacm_cm_ack_event(event);
    if (status != UCS_OK) {
        goto err_destroy_id;
    }

    cep->id          = id;
    cep->id->context = cep;
    cep->flags      |= UCT_RDMACM_CM_EP_ON_SERVER;

    status = UCT_CM_SET_CB(params, UCT_EP_PARAM_FIELD_SOCKADDR_NOTIFY_CB_SERVER,
                           cep->super.server.notify_cb,
                           params->sockaddr_cb_server,
                           uct_cm_ep_server_conn_notify_callback_t,
                           ucs_empty_function);
    if (status != UCS_OK) {
        goto err;
    }

    if (listen_ev_ch != cm->ev_ch) {
        /* The server will open the ep to the client on a different CM.
         * not the one on which its listener is listening on */
        if (rdma_migrate_id(id, cm->ev_ch)) {
            ucs_error("failed to migrate id %p to event_channel %p (cm=%p)",
                      id, cm->ev_ch, cm);
            status = UCS_ERR_IO_ERROR;
            goto err;
        }

        ucs_debug("%s migrated id %p from event_channel=%p to "
                  "new cm %p (event_channel=%p)",
                  uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
                  id, listen_ev_ch, cm, cm->ev_ch);
    }

    return uct_rdamcm_cm_ep_server_send_priv_data(cep, params);

err:
    cep->id = NULL;
err_destroy_id:
    uct_rdmacm_cm_destroy_id(id);
    return status;
}

ucs_status_t
uct_rdmacm_cm_ep_send_priv_data(uct_rdmacm_cm_ep_t *cep, const void *priv_data,
                                size_t priv_data_length)
{
    struct rdma_conn_param conn_param = {0};
    uct_rdmacm_priv_data_hdr_t *hdr;
    char ep_str[UCT_RDMACM_EP_STRING_LEN];
    ucs_status_t status;

    if (priv_data_length > uct_rdmacm_cm_get_max_conn_priv()) {
        status = UCS_ERR_EXCEEDS_LIMIT;
        goto err;
    }

    status = uct_rdamcm_cm_ep_set_qp_num(&conn_param, cep);
    if (status != UCS_OK) {
        goto err;
    }

    conn_param.private_data     = ucs_alloca(UCT_RDMACM_TCP_PRIV_DATA_LEN);
    conn_param.private_data_len = sizeof(*hdr) + priv_data_length;

    hdr         = (uct_rdmacm_priv_data_hdr_t*)conn_param.private_data;
    hdr->status = UCS_OK;
    hdr->length = priv_data_length;
    if (priv_data != NULL) {
        memcpy(hdr + 1, priv_data, priv_data_length);
    }

    if (cep->flags & UCT_RDMACM_CM_EP_ON_CLIENT) {
        ucs_trace("%s rdma_connect on cm_id %p",
                  uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
                  cep->id);
        if (rdma_connect(cep->id, &conn_param)) {
            uct_cm_ep_peer_error(&cep->super,
                                 "rdma_connect(on id=%p) failed: %m", cep->id);
            status = UCS_ERR_IO_ERROR;
            goto err_destroy_qpn;
        }
    } else {
        ucs_assert(cep->flags & UCT_RDMACM_CM_EP_ON_SERVER);
        ucs_trace("%s: rdma_accept on cm_id %p",
                  uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
                  cep->id);
        if (rdma_accept(cep->id, &conn_param)) {
            uct_cm_ep_peer_error(&cep->super,
                                 "rdma_accept(on id=%p) failed: %m", cep->id);
            status = UCS_ERR_CONNECTION_RESET;
            goto err_destroy_qpn;
        }
    }

    return UCS_OK;

err_destroy_qpn:
    uct_rdamcm_cm_ep_destroy_qpn(cep);
err:
    return status;
}

ucs_status_t
uct_rdmacm_cm_ep_connect(uct_ep_h ep, const uct_ep_connect_params_t *params)
{
    uct_rdmacm_cm_ep_t *cep = ucs_derived_of(ep, uct_rdmacm_cm_ep_t);
    const void *priv_data;
    size_t priv_data_length;
    ucs_status_t status;

    uct_ep_connect_params_get(params, &priv_data, &priv_data_length);
    UCS_ASYNC_BLOCK(uct_rdmacm_cm_ep_get_async(cep));
    status = uct_rdmacm_cm_ep_send_priv_data(cep, priv_data, priv_data_length);
    UCS_ASYNC_UNBLOCK(uct_rdmacm_cm_ep_get_async(cep));
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

    ucs_debug("%s: (id=%p) disconnected from peer %s",
              uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
              cep->id,
              ucs_sockaddr_str(rdma_get_peer_addr(cep->id), ip_port_str,
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

    self->qp     = NULL;
    self->qpn    = 0;
    self->blk    = NULL;
    self->flags  = 0;
    self->status = UCS_OK;
    self->id     = NULL;

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

    uct_rdamcm_cm_ep_destroy_qpn(self);

    /* rdma_destroy_id() cleans all events not yet reported on progress thread,
     * so no events would be reported to the user after destroying the id */
    uct_rdmacm_cm_destroy_id(self->id);

    UCS_ASYNC_UNBLOCK(worker_priv->async);
}

UCS_CLASS_DEFINE(uct_rdmacm_cm_ep_t, uct_cm_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_cm_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_cm_ep_t, uct_ep_t);
