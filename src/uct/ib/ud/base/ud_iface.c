/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ud_iface.h"
#include "ud_ep.h"
#include "ud_inl.h"

#include <ucs/arch/cpu.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/type/class.h>
#include <ucs/datastruct/queue.h>
#include <sys/poll.h>


SGLIB_DEFINE_LIST_FUNCTIONS(uct_ud_iface_peer_t, uct_ud_iface_peer_cmp, next)
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(uct_ud_iface_peer_t,
                                        UCT_UD_HASH_SIZE,
                                        uct_ud_iface_peer_hash)

static void uct_ud_iface_reserve_skbs(uct_ud_iface_t *iface, int count);
static void uct_ud_iface_free_res_skbs(uct_ud_iface_t *iface);
static void uct_ud_iface_timer(int timer_id, void *arg);

static void uct_ud_iface_free_pending_rx(uct_ud_iface_t *iface);
static void uct_ud_iface_free_async_comps(uct_ud_iface_t *iface);

static void uct_ud_iface_event(int fd, void *arg);


void uct_ud_iface_cep_init(uct_ud_iface_t *iface)
{
    sglib_hashed_uct_ud_iface_peer_t_init(iface->peers);
}

static void
uct_ud_iface_cep_cleanup_eps(uct_ud_iface_t *iface, uct_ud_iface_peer_t *peer)
{
    uct_ud_ep_t *ep, *tmp;
    uct_iface_t *iface_h = &iface->super.super.super;

    ucs_list_for_each_safe(ep, tmp, &peer->ep_list, cep_list) {
        if (ep->conn_id < peer->conn_id_last) {
            /* active connection should already be cleaned by owner */
            ucs_warn("iface (%p) peer (qpn=%d lid=%d) cleanup with %d endpoints still active",
                     iface, peer->dst_qpn, peer->dlid,
                     (int)ucs_list_length(&peer->ep_list));
            continue;
        }
        ucs_list_del(&ep->cep_list);
        uct_ep_t *ep_h = &ep->super.super;
        ucs_trace("cep:ep_destroy(%p) conn_id %d", ep, ep->conn_id);
        iface_h->ops.ep_destroy(ep_h);
    }
}

void uct_ud_iface_cep_cleanup(uct_ud_iface_t *iface)
{
    uct_ud_iface_peer_t *peer;
    struct sglib_hashed_uct_ud_iface_peer_t_iterator it_peer;

    for (peer = sglib_hashed_uct_ud_iface_peer_t_it_init(&it_peer,
                                                         iface->peers);
         peer != NULL;
         peer = sglib_hashed_uct_ud_iface_peer_t_it_next(&it_peer)) {

        uct_ud_iface_cep_cleanup_eps(iface, peer);
        free(peer);
    }
}

static uct_ud_iface_peer_t *
uct_ud_iface_cep_lookup_addr(uct_ud_iface_t *iface, uint16_t dlid,
                             const union ibv_gid *dgid, uint32_t dest_qpn)
{
    uct_ud_iface_peer_t key;
    key.dlid    = dlid;
    key.dgid    = *dgid;
    key.dst_qpn = dest_qpn;
    return sglib_hashed_uct_ud_iface_peer_t_find_member(iface->peers, &key);
}

static uct_ud_iface_peer_t *
uct_ud_iface_cep_lookup_peer(uct_ud_iface_t *iface,
                             const uct_ib_address_t *src_ib_addr,
                             const uct_ud_iface_addr_t *src_if_addr)
{
    uint32_t dest_qpn = uct_ib_unpack_uint24(src_if_addr->qp_num);
    union ibv_gid dgid;
    uint8_t is_global;
    uint16_t dlid;

    uct_ib_address_unpack(src_ib_addr, &dlid, &is_global, &dgid);
    return uct_ud_iface_cep_lookup_addr(iface, dlid, &dgid, dest_qpn);
}

static uct_ud_ep_t *
uct_ud_iface_cep_lookup_ep(uct_ud_iface_peer_t *peer, uint32_t conn_id)
{
    uint32_t id;
    uct_ud_ep_t *ep;

    if (conn_id != UCT_UD_EP_CONN_ID_MAX) {
        id = conn_id;
    } else {
        id = peer->conn_id_last;
        /* TODO: O(1) lookup in this case (new connection) */
    }
    ucs_list_for_each(ep, &peer->ep_list, cep_list) {
        if (ep->conn_id == id) {
            return ep;
        }
        if (ep->conn_id < id) {
            break;
        }
    }
    return NULL;
}

static uint32_t
uct_ud_iface_cep_getid(uct_ud_iface_peer_t *peer, uint32_t conn_id)
{
    uint32_t new_id;

    if (conn_id != UCT_UD_EP_CONN_ID_MAX) {
        return conn_id;
    }
    new_id = peer->conn_id_last++;
    return new_id;
}

/* insert new ep that is connected to src_if_addr */
ucs_status_t uct_ud_iface_cep_insert(uct_ud_iface_t *iface,
                                     const uct_ib_address_t *src_ib_addr,
                                     const uct_ud_iface_addr_t *src_if_addr,
                                     uct_ud_ep_t *ep, uint32_t conn_id)
{
    uint32_t dest_qpn = uct_ib_unpack_uint24(src_if_addr->qp_num);
    uct_ud_iface_peer_t *peer;
    union ibv_gid dgid;
    uint8_t is_global;
    uct_ud_ep_t *cep;
    uint16_t dlid;

    uct_ib_address_unpack(src_ib_addr, &dlid, &is_global, &dgid);
    peer = uct_ud_iface_cep_lookup_addr(iface, dlid, &dgid, dest_qpn);
    if (peer == NULL) {
        peer = malloc(sizeof *peer);
        if (peer == NULL) {
            return UCS_ERR_NO_MEMORY;
        }

        peer->dlid    = dlid;
        peer->dgid    = dgid;
        peer->dst_qpn = dest_qpn;
        sglib_hashed_uct_ud_iface_peer_t_add(iface->peers, peer);
        ucs_list_head_init(&peer->ep_list);
        peer->conn_id_last = 0;
    }

    ep->conn_id = uct_ud_iface_cep_getid(peer, conn_id);
    if (ep->conn_id == UCT_UD_EP_CONN_ID_MAX) {
        return UCS_ERR_NO_RESOURCE;
    }

    if (ucs_list_is_empty(&peer->ep_list)) {
            ucs_list_add_head(&peer->ep_list, &ep->cep_list);
            return UCS_OK;
    }
    ucs_list_for_each(cep, &peer->ep_list, cep_list) {
        ucs_assert_always(cep->conn_id != ep->conn_id);
        if (cep->conn_id < ep->conn_id) {
            ucs_list_insert_before(&cep->cep_list, &ep->cep_list);
            return UCS_OK;
        }
    }
    return UCS_OK;
}

void uct_ud_iface_cep_remove(uct_ud_ep_t *ep)
{
  if (ucs_list_is_empty(&ep->cep_list)) {
      return;
  }
  ucs_trace("iface(%p) cep_remove:ep(%p)", ep->super.super.iface, ep);
  ucs_list_del(&ep->cep_list);
  ucs_list_head_init(&ep->cep_list);
}

uct_ud_ep_t *uct_ud_iface_cep_lookup(uct_ud_iface_t *iface,
                                     const uct_ib_address_t *src_ib_addr,
                                     const uct_ud_iface_addr_t *src_if_addr,
                                     uint32_t conn_id)
{
    uct_ud_iface_peer_t *peer;
    uct_ud_ep_t *ep;

    peer = uct_ud_iface_cep_lookup_peer(iface, src_ib_addr, src_if_addr);
    if (peer == NULL) {
        return NULL;
    }

    ep = uct_ud_iface_cep_lookup_ep(peer, conn_id);
    if (ep && conn_id == UCT_UD_EP_CONN_ID_MAX) {
        peer->conn_id_last++;
    }
    return ep;
}

void uct_ud_iface_cep_rollback(uct_ud_iface_t *iface,
                               const uct_ib_address_t *src_ib_addr,
                               const uct_ud_iface_addr_t *src_if_addr,
                               uct_ud_ep_t *ep)
{
    uct_ud_iface_peer_t *peer;

    peer = uct_ud_iface_cep_lookup_peer(iface, src_ib_addr, src_if_addr);
    ucs_assert_always(peer != NULL);
    ucs_assert_always(peer->conn_id_last > 0);
    ucs_assert_always(ep->conn_id + 1 == peer->conn_id_last);
    ucs_assert_always(!ucs_list_is_empty(&peer->ep_list));
    ucs_assert_always(!ucs_list_is_empty(&ep->cep_list));

    peer->conn_id_last--;
    uct_ud_iface_cep_remove(ep);
}

static void uct_ud_iface_send_skb_init(uct_iface_h tl_iface, void *obj,
                                       uct_mem_h memh)
{
    uct_ud_send_skb_t *skb = obj;
    uct_ib_mem_t *ib_memh = memh;

    skb->lkey  = ib_memh->lkey;
    skb->flags = 0;
}

static ucs_status_t
uct_ud_iface_create_qp(uct_ud_iface_t *self, const uct_ud_iface_config_t *config)
{
    /* TODO: exp attrs autoconf */
    struct ibv_exp_qp_init_attr qp_init_attr;
    struct ibv_qp_attr qp_attr;
    int ret;

    /* Create QP */
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.qp_context          = NULL;
    qp_init_attr.send_cq             = self->super.send_cq;
    qp_init_attr.recv_cq             = self->super.recv_cq;
    qp_init_attr.srq                 = NULL; /* TODO */
    qp_init_attr.qp_type             = IBV_QPT_UD;
    qp_init_attr.sq_sig_all          = 0;

    /* TODO: cap setting */
    qp_init_attr.cap.max_send_wr     = config->super.tx.queue_len;
    qp_init_attr.cap.max_recv_wr     = config->super.rx.queue_len;
    qp_init_attr.cap.max_send_sge    = 2;
    qp_init_attr.cap.max_recv_sge    = 1;
    qp_init_attr.cap.max_inline_data = ucs_max(config->super.tx.min_inline,
                                               UCT_UD_MIN_INLINE);

#if HAVE_VERBS_EXP_H
    qp_init_attr.pd                  = uct_ib_iface_md(&self->super)->pd;
    qp_init_attr.comp_mask           = IBV_QP_INIT_ATTR_PD;
    /* TODO: inline rcv */
#if 0
    if (mxm_ud_ep_opts(ep)->ud.ib.rx.max_inline > 0) {
        qp_init_attr.comp_mask      |= IBV_EXP_QP_INIT_ATTR_INL_RECV;
        qp_init_attr.max_inl_recv    = mxm_ud_ep_opts(ep)->ud.ib.rx.max_inline;
    }
#endif
    self->qp = ibv_exp_create_qp(uct_ib_iface_device(&self->super)->ibv_context,
                                 &qp_init_attr);
#else
    self->qp = ibv_exp_create_qp(uct_ib_iface_md(&self->super)->pd, &qp_init_attr);
#endif
    if (self->qp == NULL) {
        ucs_error("Failed to create qp: %s [inline: %u rsge: %u ssge: %u rwr: %u swr: %u]",
                  strerror(errno),
                  qp_init_attr.cap.max_inline_data, qp_init_attr.cap.max_recv_sge,
                  qp_init_attr.cap.max_send_sge, qp_init_attr.cap.max_recv_wr,
                  qp_init_attr.cap.max_send_wr);
        goto err;
    }

    self->config.max_inline = qp_init_attr.cap.max_inline_data;
    ucs_assert_always(qp_init_attr.cap.max_inline_data >= UCT_UD_MIN_INLINE);
    uct_ib_iface_set_max_iov(&self->super, qp_init_attr.cap.max_send_sge);

    memset(&qp_attr, 0, sizeof(qp_attr));
    /* Modify QP to INIT state */
    qp_attr.qp_state   = IBV_QPS_INIT;
    qp_attr.pkey_index = self->super.pkey_index;
    qp_attr.port_num   = self->super.config.port_num;
    qp_attr.qkey       = UCT_IB_KEY;
    ret = ibv_modify_qp(self->qp, &qp_attr,
                        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY);
    if (ret) {
        ucs_error("Failed to modify UD QP to INIT: %m");
        goto err_destroy_qp;
    }

    /* Modify to RTR */
    qp_attr.qp_state = IBV_QPS_RTR;
    ret = ibv_modify_qp(self->qp, &qp_attr, IBV_QP_STATE);
    if (ret) {
        ucs_error("Failed to modify UD QP to RTR: %m");
        goto err_destroy_qp;
    }

    /* Modify to RTS */
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn = 0;
    ret = ibv_modify_qp(self->qp, &qp_attr, IBV_QP_STATE | IBV_QP_SQ_PSN);
    if (ret) {
        ucs_error("Failed to modify UD QP to RTS: %m");
        goto err_destroy_qp;
    }

    ucs_debug("iface=%p: created qp 0x%x max_send_wr %u max_recv_wr %u max_inline %u",
            self, self->qp->qp_num,
            qp_init_attr.cap.max_send_wr,
            qp_init_attr.cap.max_recv_wr,
            qp_init_attr.cap.max_inline_data);

    return UCS_OK;
err_destroy_qp:
    ibv_destroy_qp(self->qp);
err:
    return UCS_ERR_INVALID_PARAM;
}

ucs_status_t uct_ud_iface_complete_init(uct_ud_iface_t *iface)
{
    ucs_async_context_t *async = iface->super.super.worker->async;
    ucs_async_mode_t async_mode = async->mode;
    ucs_status_t status;

    uct_ud_iface_reserve_skbs(iface, iface->tx.available);

    /* TODO: make tick configurable */
    iface->async.slow_tick = ucs_time_from_msec(100);
    status = ucs_twheel_init(&iface->async.slow_timer,
                             iface->async.slow_tick / 4,
                             uct_ud_iface_get_async_time(iface));
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_async_add_timer(async_mode, iface->async.slow_tick,
                                 uct_ud_iface_timer, iface, async,
                                 &iface->async.timer_id);
    if (status != UCS_OK) {
        goto err_twheel_cleanup;
    }

    status = ucs_async_set_event_handler(async_mode, iface->super.comp_channel->fd,
                                         POLLIN, uct_ud_iface_event, iface, async);
    if (status != UCS_OK) {
        goto err_remove_timer;
    }

    status = uct_ib_iface_arm_rx_cq(&iface->super, 1);
    if (status != UCS_OK) {
        goto err_unset_event_handler;
    }

    return UCS_OK;

err_unset_event_handler:
    ucs_async_remove_handler(iface->super.comp_channel->fd, 1);
err_remove_timer:
    ucs_async_remove_handler(iface->async.timer_id, 1);
err_twheel_cleanup:
    ucs_twheel_cleanup(&iface->async.slow_timer);
err:
    return status;
}

void uct_ud_iface_remove_async_handlers(uct_ud_iface_t *iface)
{
    ucs_async_remove_handler(iface->super.comp_channel->fd, 1);
    ucs_async_remove_handler(iface->async.timer_id, 1);
}

UCS_CLASS_INIT_FUNC(uct_ud_iface_t, uct_ud_iface_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    unsigned ud_rx_priv_len,
                    const uct_ud_iface_config_t *config)
{
    unsigned rx_priv_len, rx_hdr_len;
    ucs_status_t status;
    size_t data_size;
    int mtu;

    ucs_trace_func("%s: iface=%p ops=%p worker=%p rx_headroom=%zu ud_rx_priv_len=%u",
                   params->dev_name, self, ops, worker,
                   params->rx_headroom, ud_rx_priv_len);

    if (worker->async == NULL) {
        ucs_error("%s ud iface must have valid async context", params->dev_name);
        return UCS_ERR_INVALID_PARAM;
    }

    if (config->super.tx.queue_len <= UCT_UD_TX_MODERATION) {
        ucs_error("%s ud iface tx queue is too short (%d <= %d)",
                  params->dev_name,
                  config->super.tx.queue_len, UCT_UD_TX_MODERATION);
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_ib_device_mtu(params->dev_name, md, &mtu);
    if (status != UCS_OK) {
        return status;
    }

    rx_priv_len = ud_rx_priv_len +
                  sizeof(uct_ud_recv_skb_t) - sizeof(uct_ib_iface_recv_desc_t);
    rx_hdr_len  = UCT_IB_GRH_LEN + sizeof(uct_ud_neth_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_ib_iface_t, &ops->super, md, worker,
                              params, rx_priv_len, rx_hdr_len,
                              config->super.tx.queue_len,
                              config->super.rx.queue_len,
                              mtu, &config->super);

    self->tx.unsignaled          = 0;
    self->tx.available           = config->super.tx.queue_len;

    self->rx.available           = config->super.rx.queue_len;
    self->config.tx_qp_len       = config->super.tx.queue_len;
    self->config.peer_timeout    = ucs_time_from_sec(config->peer_timeout);

    if (config->slow_timer_backoff <= 0.) {
        ucs_error("The slow timer back off should be > 0 (%lf)",
                  config->slow_timer_backoff);
        return UCS_ERR_INVALID_PARAM;
    } else {
        self->config.slow_timer_backoff = config->slow_timer_backoff;
    }

    /* Redefine receive desc release callback */
    self->super.release_desc.cb  = uct_ud_iface_release_desc;

    UCT_UD_IFACE_HOOK_INIT(self);

    if (uct_ud_iface_create_qp(self, config) != UCS_OK) {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_ptr_array_init(&self->eps, 0, "ud_eps");
    uct_ud_iface_cep_init(self);

    status = uct_ib_iface_recv_mpool_init(&self->super, &config->super,
                                          "ud_recv_skb", &self->rx.mp);
    if (status != UCS_OK) {
        goto err_qp;
    }

    data_size = sizeof(uct_ud_ctl_hdr_t) + self->super.addr_size;
    data_size = ucs_max(data_size, self->super.config.seg_size);
    data_size = ucs_max(data_size, sizeof(uct_ud_zcopy_desc_t) + self->config.max_inline);

    status = uct_iface_mpool_init(&self->super.super, &self->tx.mp,
                                  sizeof(uct_ud_send_skb_t) + data_size,
                                  sizeof(uct_ud_send_skb_t),
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &config->super.tx.mp,
                                  self->config.tx_qp_len,
                                  self->config.tx_qp_len,
                                  uct_ud_iface_send_skb_init, "ud_tx_skb");
    if (status != UCS_OK) {
        goto err_mpool;
    }
    self->tx.skb = ucs_mpool_get(&self->tx.mp);
    self->tx.skb_inl.super.len = sizeof(uct_ud_neth_t);
    ucs_queue_head_init(&self->tx.res_skbs);
    ucs_queue_head_init(&self->tx.async_comp_q);

    ucs_arbiter_init(&self->tx.pending_q);
    self->tx.pending_q_len = 0;
    self->tx.in_pending = 0;

    ucs_queue_head_init(&self->rx.pending_q);

    return UCS_OK;

err_mpool:
    ucs_mpool_cleanup(&self->rx.mp, 1);
err_qp:
    ibv_destroy_qp(self->qp);
    ucs_ptr_array_cleanup(&self->eps);
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ud_iface_t)
{
    ucs_trace_func("");

    /* TODO: proper flush and connection termination */
    uct_ud_enter(self);
    ucs_debug("iface(%p): cep cleanup", self);
    uct_ud_iface_cep_cleanup(self);
    uct_ud_iface_free_res_skbs(self);
    uct_ud_iface_free_async_comps(self);
    ucs_mpool_cleanup(&self->tx.mp, 0);
    /* TODO: qp to error state and cleanup all wqes */
    uct_ud_iface_free_pending_rx(self);
    ucs_mpool_cleanup(&self->rx.mp, 0);
    ibv_destroy_qp(self->qp);
    ucs_debug("iface(%p): ptr_array cleanup", self);
    ucs_ptr_array_cleanup(&self->eps);
    ucs_arbiter_cleanup(&self->tx.pending_q);
    ucs_assert(self->tx.pending_q_len == 0);
    uct_ud_leave(self);
}

UCS_CLASS_DEFINE(uct_ud_iface_t, uct_ib_iface_t);

ucs_config_field_t uct_ud_iface_config_table[] = {
    {"IB_", "", NULL,
     ucs_offsetof(uct_ud_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_ib_iface_config_table)},
    {"TIMEOUT", "5.0m", "Transport timeout",
     ucs_offsetof(uct_ud_iface_config_t, peer_timeout), UCS_CONFIG_TYPE_TIME},
    {"SLOW_TIMER_BACKOFF", "2.0", "Timeout multiplier for resending trigger",
     ucs_offsetof(uct_ud_iface_config_t, slow_timer_backoff),
                  UCS_CONFIG_TYPE_DOUBLE},
    {NULL}
};


void uct_ud_iface_query(uct_ud_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    uct_ib_iface_query(&iface->super, UCT_IB_DETH_LEN + sizeof(uct_ud_neth_t),
                       iface_attr);

    iface_attr->cap.flags              = UCT_IFACE_FLAG_AM_SHORT         |
                                         UCT_IFACE_FLAG_AM_BCOPY         |
                                         UCT_IFACE_FLAG_AM_ZCOPY         |
                                         UCT_IFACE_FLAG_CONNECT_TO_EP    |
                                         UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                         UCT_IFACE_FLAG_PENDING          |
                                         UCT_IFACE_FLAG_AM_CB_SYNC       |
                                         UCT_IFACE_FLAG_AM_CB_ASYNC      |
                                         UCT_IFACE_FLAG_WAKEUP           |
                                         UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;

    iface_attr->cap.am.max_short       = iface->config.max_inline - sizeof(uct_ud_neth_t);
    iface_attr->cap.am.max_bcopy       = iface->super.config.seg_size - sizeof(uct_ud_neth_t);
    iface_attr->cap.am.min_zcopy       = 0;
    iface_attr->cap.am.max_zcopy       = iface->super.config.seg_size - sizeof(uct_ud_neth_t);
    iface_attr->cap.am.align_mtu       = uct_ib_mtu_value(uct_ib_iface_port_attr(&iface->super)->active_mtu);
    iface_attr->cap.am.opt_zcopy_align = UCS_SYS_PCI_MAX_PAYLOAD;
    iface_attr->cap.am.max_hdr         = iface->config.max_inline - sizeof(uct_ud_neth_t);
    /* The first iov is reserved for the header */
    iface_attr->cap.am.max_iov         = uct_ib_iface_get_max_iov(&iface->super) - 1;

    iface_attr->cap.put.max_short      = iface->config.max_inline -
                                         sizeof(uct_ud_neth_t) - sizeof(uct_ud_put_hdr_t);

    iface_attr->iface_addr_len         = sizeof(uct_ud_iface_addr_t);
    iface_attr->ep_addr_len            = sizeof(uct_ud_ep_addr_t);

    /* Software overhead */
    iface_attr->overhead               = 80e-9;
}

ucs_status_t
uct_ud_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);
    uct_ud_iface_addr_t *addr = (uct_ud_iface_addr_t *)iface_addr;

    uct_ib_pack_uint24(addr->qp_num, iface->qp->qp_num);

    return UCS_OK;
}

ucs_status_t uct_ud_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                uct_completion_t *comp)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);
    uct_ud_ep_t *ep;
    ucs_status_t status;
    int i, count;

    ucs_trace_func("");

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    uct_ud_enter(iface);

    uct_ud_iface_progress_pending_tx(iface);

    count = 0;
    ucs_ptr_array_for_each(ep, i, &iface->eps) {
        /* ud ep flush returns either ok or in progress */
        status = uct_ud_ep_flush_nolock(iface, ep, NULL);
        if ((status == UCS_INPROGRESS) || (status == UCS_ERR_NO_RESOURCE)) {
            ++count;
        }
    }

    uct_ud_leave(iface);
    if (count != 0) {
        UCT_TL_IFACE_STAT_FLUSH_WAIT(&iface->super.super);
        return UCS_INPROGRESS;
    }
    UCT_TL_IFACE_STAT_FLUSH(&iface->super.super);
    return UCS_OK;
}

void uct_ud_iface_add_ep(uct_ud_iface_t *iface, uct_ud_ep_t *ep)
{
    uint32_t prev_gen;
    ep->ep_id = ucs_ptr_array_insert(&iface->eps, ep, &prev_gen);
}

void uct_ud_iface_remove_ep(uct_ud_iface_t *iface, uct_ud_ep_t *ep)
{
    if (ep->ep_id != UCT_UD_EP_NULL_ID) {
        ucs_trace("iface(%p) remove ep: %p id %d", iface, ep, ep->ep_id);
        ucs_ptr_array_remove(&iface->eps, ep->ep_id, 0);
    }
}

void uct_ud_iface_replace_ep(uct_ud_iface_t *iface,
                             uct_ud_ep_t *old_ep, uct_ud_ep_t *new_ep)
{
    void *p;
    ucs_assert_always(old_ep != new_ep);
    ucs_assert_always(old_ep->ep_id != new_ep->ep_id);
    p = ucs_ptr_array_replace(&iface->eps, old_ep->ep_id, new_ep);
    ucs_assert_always(p == (void *)old_ep);
    ucs_trace("replace_ep: old(%p) id=%d new(%p) id=%d", old_ep, old_ep->ep_id, new_ep, new_ep->ep_id);
    ucs_ptr_array_remove(&iface->eps, new_ep->ep_id, 0);
}


static void uct_ud_iface_reserve_skbs(uct_ud_iface_t *iface, int count)
{
    int i;
    uct_ud_send_skb_t *skb;

    for (i = 0; i < count; i++) {
        skb = ucs_mpool_get(&iface->tx.mp);
        if (skb == NULL) {
            ucs_fatal("failed to reserve %d tx skbs", count);
        }
        ucs_queue_push(&iface->tx.res_skbs, &skb->queue);
    }
}

uct_ud_send_skb_t *uct_ud_iface_res_skb_get(uct_ud_iface_t *iface)
{
    ucs_queue_elem_t *elem;

    elem = ucs_queue_pull(&iface->tx.res_skbs);
    return  ucs_container_of(elem, uct_ud_send_skb_t, queue);
}


static void uct_ud_iface_free_res_skbs(uct_ud_iface_t *iface)
{
    uct_ud_send_skb_t *skb;

    /* Release acknowledged skb's */
    for (skb = uct_ud_iface_res_skb_get(iface);
         skb != NULL;
         skb = uct_ud_iface_res_skb_get(iface)) {
        ucs_mpool_put(skb);
    }
}

void uct_ud_iface_dispatch_async_comps_do(uct_ud_iface_t *iface)
{
    uct_ud_comp_desc_t *cdesc;
    uct_ud_send_skb_t  *skb;

    do {
        skb = ucs_queue_pull_elem_non_empty(&iface->tx.async_comp_q,
                                            uct_ud_send_skb_t, queue);
        cdesc = uct_ud_comp_desc(skb);

        if (ucs_unlikely(skb->flags & UCT_UD_SEND_SKB_FLAG_ERR)) {
            if (skb->flags & UCT_UD_SEND_SKB_FLAG_COMP) {
                uct_invoke_completion(cdesc->comp, skb->status);
            }

            if (!(cdesc->ep->flags & UCT_UD_EP_FLAG_FAILED)) {
                cdesc->ep->flags |= UCT_UD_EP_FLAG_FAILED;
                iface->super.ops->set_ep_failed(&iface->super,
                                                &cdesc->ep->super.super);
            }
        } else {
            uct_invoke_completion(cdesc->comp, UCS_OK);
        }
        cdesc->ep->flags &= ~UCT_UD_EP_FLAG_ASYNC_COMPS;
        skb->flags = 0;
        ucs_mpool_put(skb);
    } while (!ucs_queue_is_empty(&iface->tx.async_comp_q));
}

static void uct_ud_iface_free_async_comps(uct_ud_iface_t *iface)
{
    uct_ud_send_skb_t *skb;

    while (!ucs_queue_is_empty(&iface->tx.async_comp_q)) {
        skb = ucs_queue_pull_elem_non_empty(&iface->tx.async_comp_q,
                                            uct_ud_send_skb_t, queue);
        ucs_mpool_put(skb);
    }
}

ucs_status_t uct_ud_iface_dispatch_pending_rx_do(uct_ud_iface_t *iface)
{
    int count;
    uct_ud_recv_skb_t *skb;
    uct_ud_neth_t *neth;
    unsigned max_poll = iface->super.config.rx_max_poll;

    count = 0;
    do {
        skb = ucs_queue_pull_elem_non_empty(&iface->rx.pending_q, uct_ud_recv_skb_t, u.am.queue);
        neth =  (uct_ud_neth_t *)((char *)uct_ib_iface_recv_desc_hdr(&iface->super,
                                                                     (uct_ib_iface_recv_desc_t *)skb) +
                                  UCT_IB_GRH_LEN);
        uct_ib_iface_invoke_am_desc(&iface->super,
                                    uct_ud_neth_get_am_id(neth),
                                    neth + 1,
                                    skb->u.am.len,
                                    &skb->super);
        count++;
        if (count >= max_poll) {
            return UCS_ERR_NO_RESOURCE;
        }
    } while (!ucs_queue_is_empty(&iface->rx.pending_q));

    return UCS_OK;
}

static void uct_ud_iface_free_pending_rx(uct_ud_iface_t *iface)
{
    uct_ud_recv_skb_t *skb;

    while (!ucs_queue_is_empty(&iface->rx.pending_q)) {
        skb = ucs_queue_pull_elem_non_empty(&iface->rx.pending_q, uct_ud_recv_skb_t, u.am.queue);
        ucs_mpool_put(skb);
    }
}

static inline void uct_ud_iface_async_progress(uct_ud_iface_t *iface)
{
    ucs_derived_of(iface->super.ops, uct_ud_iface_ops_t)->async_progress(iface);
}

static void uct_ud_iface_event(int fd, void *arg)
{
    uct_ud_enter(arg);
    ucs_trace_async("iface(%p) uct_ud_iface_event", arg);
    uct_ud_iface_async_progress(arg);
    uct_ud_leave(arg);
}

static void uct_ud_iface_timer(int timer_id, void *arg)
{
    uct_ud_iface_t *iface = arg;
    ucs_time_t now;

    uct_ud_enter(iface);
    now = uct_ud_iface_get_async_time(iface);
    ucs_trace_async("iface(%p) slow_timer_sweep: now %lu", iface, now);
    ucs_twheel_sweep(&iface->async.slow_timer, now);
    uct_ud_iface_async_progress(iface);
    uct_ud_leave(iface);
}

void uct_ud_iface_release_desc(uct_recv_desc_t *self, void *desc)
{
    uct_ud_iface_t *iface = ucs_container_of(self,
                                             uct_ud_iface_t, super.release_desc);

    uct_ud_enter(iface);
    uct_ib_iface_release_desc(self, desc);
    uct_ud_leave(iface);
}

static void
uct_ud_tx_wnd_purge_outstanding(uct_ud_iface_t *iface, uct_ud_ep_t *ud_ep)
{
    uct_ud_comp_desc_t *cdesc;
    uct_ud_send_skb_t  *skb;

    ucs_queue_for_each_extract(skb, &ud_ep->tx.window, queue, 1) {
        skb->flags |= UCT_UD_SEND_SKB_FLAG_ERR;
        skb->status = UCS_ERR_ENDPOINT_TIMEOUT;
        if (ucs_likely(!(skb->flags & UCT_UD_SEND_SKB_FLAG_COMP))) {
            skb->len = 0;
        }
        cdesc = uct_ud_comp_desc(skb);
        /* don't call user completion from async context. instead, put
         * it on a queue which will be progresed from main thread.
         */
        ucs_queue_push(&iface->tx.async_comp_q, &skb->queue);
        ud_ep->flags |= UCT_UD_EP_FLAG_ASYNC_COMPS;
        cdesc->ep = ud_ep;
    }
}

void uct_ud_iface_handle_failure(uct_ib_iface_t *iface, void *arg)
{
    uct_ud_tx_wnd_purge_outstanding(ucs_derived_of(iface, uct_ud_iface_t),
                                    (uct_ud_ep_t *)arg);
}
