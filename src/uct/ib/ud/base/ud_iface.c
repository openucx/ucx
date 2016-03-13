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

SGLIB_DEFINE_LIST_FUNCTIONS(uct_ud_iface_peer_t, uct_ud_iface_peer_cmp, next)
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(uct_ud_iface_peer_t,
                                        UCT_UD_HASH_SIZE,
                                        uct_ud_iface_peer_hash)

static void uct_ud_iface_reserve_skbs(uct_ud_iface_t *iface, int count);
static void uct_ud_iface_free_res_skbs(uct_ud_iface_t *iface);
static void uct_ud_iface_timer(void *arg);

static void uct_ud_iface_free_pending_rx(uct_ud_iface_t *iface);

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
                     iface, peer->dest_iface.qp_num, peer->dest_iface.lid, 
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
uct_ud_iface_cep_lookup_peer(uct_ud_iface_t *iface, 
                             const uct_sockaddr_ib_t *src_if_addr)
{
    uct_ud_iface_peer_t *peer, key;

    key.dest_iface.qp_num = src_if_addr->qp_num;
    key.dest_iface.lid    = src_if_addr->lid;

    peer = sglib_hashed_uct_ud_iface_peer_t_find_member(iface->peers, &key);
    return peer;
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
                                     const uct_sockaddr_ib_t *src_if_addr,
                                     uct_ud_ep_t *ep, uint32_t conn_id)
{
    uct_ud_iface_peer_t *peer;
    uct_ud_ep_t *cep;

    peer = uct_ud_iface_cep_lookup_peer(iface, src_if_addr);
    if (!peer) {
        peer = (uct_ud_iface_peer_t *)malloc(sizeof(*peer));
        if (!peer) {
            return UCS_ERR_NO_MEMORY;
        }
        peer->dest_iface.qp_num = src_if_addr->qp_num;
        peer->dest_iface.lid    = src_if_addr->lid;
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
                                     const uct_sockaddr_ib_t *src_if_addr,
                                     uint32_t conn_id)
{
    uct_ud_iface_peer_t *peer;
    uct_ud_ep_t *ep;

    peer = uct_ud_iface_cep_lookup_peer(iface, src_if_addr);
    if (!peer) {
        return NULL;
    }

    ep = uct_ud_iface_cep_lookup_ep(peer, conn_id);
    if (ep && conn_id == UCT_UD_EP_CONN_ID_MAX) {
        peer->conn_id_last++;
    }
    return ep;
}

void uct_ud_iface_cep_rollback(uct_ud_iface_t *iface, 
                               const uct_sockaddr_ib_t *src_if_addr,
                               uct_ud_ep_t *ep)
{
    uct_ud_iface_peer_t *peer;

    peer = uct_ud_iface_cep_lookup_peer(iface, src_if_addr);

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
    struct ibv_mr *mr = memh;
    skb->lkey  = mr->lkey;
    skb->flags = 0;
}

static ucs_status_t
uct_ud_iface_create_qp(uct_ud_iface_t *self, uct_ud_iface_config_t *config)
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
    qp_init_attr.pd                  = uct_ib_iface_pd(&self->super)->pd;
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
    self->qp = ibv_exp_create_qp(uct_ib_iface_pd(&self->super)->pd, &qp_init_attr);
#endif
    if (self->qp == NULL) {
        ucs_error("Failed to create qp: %m [inline: %u rsge: %u ssge: %u rwr: %u swr: %u]",
                  qp_init_attr.cap.max_inline_data, qp_init_attr.cap.max_recv_sge,
                  qp_init_attr.cap.max_send_sge, qp_init_attr.cap.max_recv_wr,
                  qp_init_attr.cap.max_send_wr);
        goto err;
    }

    self->config.max_inline = qp_init_attr.cap.max_inline_data;
    ucs_assert_always(qp_init_attr.cap.max_inline_data >= UCT_UD_MIN_INLINE);

    memset(&qp_attr, 0, sizeof(qp_attr));
    /* Modify QP to INIT state */
    qp_attr.qp_state   = IBV_QPS_INIT;
    qp_attr.pkey_index = self->super.pkey_index;
    qp_attr.port_num   = self->super.port_num;
    qp_attr.qkey       = UCT_IB_QKEY;
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

void uct_ud_iface_complete_init(uct_ud_iface_t *iface, ucs_notifier_chain_func_t progress_cb)
{
    ucs_status_t status;

    /* TODO: add progress on first ep creation */
    uct_ud_iface_reserve_skbs(iface, iface->tx.available);

    status = ucs_twheel_init(&iface->async.slow_timer, uct_ud_slow_tick() / 4,
                             uct_ud_iface_get_async_time(iface));
    ucs_assert_always(status == UCS_OK);

    status = ucs_async_add_timer(iface->super.super.worker->async->mode,
                                 uct_ud_slow_tick(), /* TODO: make configurable */
                                 uct_ud_iface_timer, iface,
                                 iface->super.super.worker->async,
                                 &iface->async.timer_id);
    ucs_assertv_always(status == UCS_OK, "status=%s", ucs_status_string(status));

    uct_worker_progress_register(iface->super.super.worker, progress_cb, iface);
}

UCS_CLASS_INIT_FUNC(uct_ud_iface_t, uct_iface_ops_t *ops, uct_pd_h pd,
                    uct_worker_h worker, const char *dev_name, unsigned rx_headroom,
                    unsigned rx_priv_len, uct_ud_iface_config_t *config)
{
    ucs_status_t status;

    ucs_trace_func("%s: iface=%p ops=%p worker=%p rx_headroom=%u rx_priv_len=%u",
                   dev_name, self, ops, worker, rx_headroom, rx_priv_len);

    if (worker->async == NULL) {
        ucs_error("%s ud iface must have valid async context", dev_name);
        return UCS_ERR_INVALID_PARAM;
    }

    if (config->super.tx.queue_len <= UCT_UD_TX_MODERATION) {
        ucs_error("%s ud iface tx queue is too short (%d <= %d)",
                  dev_name,
                  config->super.tx.queue_len, UCT_UD_TX_MODERATION);
        return UCS_ERR_INVALID_PARAM; 
    }

    UCS_CLASS_CALL_SUPER_INIT(uct_ib_iface_t, ops, pd, worker, dev_name, rx_headroom,
                              rx_priv_len + sizeof(uct_ud_recv_skb_t) - sizeof(uct_ib_iface_recv_desc_t), 
                              UCT_IB_GRH_LEN + sizeof(uct_ud_neth_t),
                              config->super.tx.queue_len, 4096, &config->super);
 
    self->tx.unsignaled          = 0;
    self->tx.available           = config->super.tx.queue_len;

    self->rx.available           = config->super.rx.queue_len;
    self->config.tx_qp_len       = config->super.tx.queue_len;

    if (uct_ud_iface_create_qp(self, config) != UCS_OK) {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_ptr_array_init(&self->eps, 0, "ud_eps");
    uct_ud_iface_cep_init(self);    

    /* TODO: correct hdr + payload offset  */
    status = uct_ib_iface_recv_mpool_init(&self->super, &config->super,
                                            "ud_recv_skb", &self->rx.mp); 
    if (status != UCS_OK) {
        goto err_qp;
    }

    status = uct_iface_mpool_init(&self->super.super, &self->tx.mp,
                                sizeof(uct_ud_send_skb_t) + 4096 /* TODO mtu */,
                                sizeof(uct_ud_send_skb_t),
                                UCS_SYS_CACHE_LINE_SIZE,
                                &config->super.tx.mp, self->config.tx_qp_len,
                                uct_ud_iface_send_skb_init, "ud_tx_skb");
    if (status != UCS_OK) {
        goto err_mpool;
    }
    self->tx.skb = ucs_mpool_get(&self->tx.mp);
    self->tx.skb_inl.super.len = sizeof(uct_ud_neth_t);
    ucs_queue_head_init(&self->tx.res_skbs);

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
    ucs_async_remove_timer(self->async.timer_id);
    ucs_twheel_cleanup(&self->async.slow_timer);
    ucs_debug("iface(%p): cep cleanup", self);
    uct_ud_iface_cep_cleanup(self);
    uct_ud_iface_free_res_skbs(self);
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
    {NULL}
};


void uct_ud_iface_query(uct_ud_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    int mtu = 4096; /* TODO: mtu from port header */

    uct_ib_iface_query(&iface->super, UCT_IB_DETH_LEN + sizeof(uct_ud_neth_t),
                       iface_attr);

    iface_attr->cap.flags             = UCT_IFACE_FLAG_AM_SHORT |
                                        UCT_IFACE_FLAG_AM_BCOPY | 
                                        UCT_IFACE_FLAG_CONNECT_TO_EP |
                                        UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                        UCT_IFACE_FLAG_PENDING |
                                        UCT_IFACE_FLAG_AM_CB_SYNC |
                                        UCT_IFACE_FLAG_AM_CB_ASYNC;

    iface_attr->cap.am.max_short      = iface->config.max_inline - sizeof(uct_ud_neth_t);
    iface_attr->cap.am.max_bcopy      = mtu - sizeof(uct_ud_neth_t);
    iface_attr->cap.am.max_zcopy      = 0;

    iface_attr->cap.put.max_short     = iface->config.max_inline - sizeof(uct_ud_neth_t) - sizeof(uct_ud_put_hdr_t);
    iface_attr->cap.put.max_bcopy     = 0;
    iface_attr->cap.put.max_zcopy     = 0;

    iface_attr->iface_addr_len        = sizeof(uct_sockaddr_ib_t);
    iface_attr->ep_addr_len           = sizeof(uct_sockaddr_ib_t);

    /* Software overhead */
    iface_attr->overhead = 80e-9;
}

ucs_status_t
uct_ud_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);
    uct_sockaddr_ib_t *addr = (uct_sockaddr_ib_t *)iface_addr;

    uct_ib_iface_get_address(tl_iface, iface_addr);
    addr->qp_num = iface->qp->qp_num;
    ucs_debug("iface=%p get_address: qpnum=%d lid=%d", iface, addr->qp_num, addr->lid);

    return UCS_OK;
}

ucs_status_t uct_ud_iface_flush(uct_iface_h tl_iface)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);
    uct_ud_ep_t *ep;
    ucs_status_t status;
    int i, count;

    ucs_trace_func("");
    uct_ud_enter(iface);

    count = 0;
    ucs_ptr_array_for_each(ep, i, &iface->eps) {
        /* ud ep flush returns either ok or in progress */
        status = uct_ud_ep_flush_nolock(iface, ep);
        if (status == UCS_INPROGRESS) {
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
        uct_ib_iface_invoke_am(&iface->super, 
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


static void uct_ud_iface_timer(void *arg)
{
    uct_ud_iface_t *iface = (uct_ud_iface_t *)arg;
    ucs_time_t now;

    uct_ud_enter(iface);
    now = uct_ud_iface_get_async_time(iface);
    ucs_trace_async("iface(%p) slow_timer_sweep: now %llu", iface, now);
    ucs_twheel_sweep(&iface->async.slow_timer, now);
    iface->ops.async_progress(arg);
    uct_ud_leave(iface);
}

void uct_ud_iface_release_am_desc(uct_iface_t *tl_iface, void *desc)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);

    uct_ud_enter(iface);
    uct_ib_iface_release_am_desc(tl_iface, desc);
    uct_ud_leave(iface);
}

