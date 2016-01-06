/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ud_ep.h"
#include "ud_iface.h"
#include "ud_inl.h"

#include <uct/ib/base/ib_verbs.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>

static void uct_ud_ep_reset(uct_ud_ep_t *ep)
{
    ep->tx.psn         = 1;
    /* TODO: configurable max window size */
    ep->tx.max_psn     = ep->tx.psn + UCT_UD_MAX_WINDOW;
    ep->tx.acked_psn   = 0;
    ep->tx.rt_psn      = 0;
    ep->tx.pending.ops = UCT_UD_EP_OP_NONE;
    ucs_queue_head_init(&ep->tx.window);
    ep->tx.rt_pos = ucs_queue_iter_begin(&ep->tx.window);

    ep->rx.acked_psn = 0;
    ucs_frag_list_init(ep->tx.psn-1, &ep->rx.ooo_pkts, 0 /*TODO: ooo support */
                       UCS_STATS_ARG(ep->rx.stats));
}

static void uct_ud_ep_slow_timer(ucs_callback_t *self)
{
    uct_ud_ep_t *ep = ucs_container_of(self, uct_ud_ep_t, slow_timer);
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                           uct_ud_iface_t);
    ucs_time_t now;

    UCT_UD_EP_HOOK_CALL_TIMER(ep);
    now = ucs_twheel_get_time(&iface->async.slow_timer);

    if (ucs_queue_is_empty(&ep->tx.window)) {
        return;
    }

    if (now - ep->tx.send_time > 3*uct_ud_slow_tick()) {
        ucs_trace("sceduling resend now: %llu send_time: %llu diff: %llu tick: %llu",
                   now, ep->tx.send_time, now - ep->tx.send_time, uct_ud_slow_tick()); 
        uct_ud_ep_ctl_op_add(iface, ep, UCT_UD_EP_OP_RESEND);
        ep->tx.rt_pos = ucs_queue_iter_begin(&ep->tx.window);
        ep->tx.send_time = ucs_twheel_get_time(&iface->async.slow_timer);
    }

    ucs_wtimer_add(&iface->async.slow_timer, &ep->slow_timer, 
                   uct_ud_slow_tick());
}

UCS_CLASS_INIT_FUNC(uct_ud_ep_t, uct_ud_iface_t *iface)
{
    ucs_trace_func("");

    memset(self, 0, sizeof(*self));
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    self->dest_ep_id = UCT_UD_EP_NULL_ID;
    uct_ud_ep_reset(self);
    ucs_list_head_init(&self->cep_list);
    uct_ud_iface_add_ep(iface, self);
    ucs_wtimer_init(&self->slow_timer, uct_ud_ep_slow_timer);
    ucs_arbiter_group_init(&self->tx.pending.group);
    ucs_arbiter_elem_init(&self->tx.pending.elem);

    UCT_UD_EP_HOOK_INIT(self);
    ucs_debug("NEW EP: iface=%p ep=%p id=%d", iface, self, self->ep_id);
    return UCS_OK;
}

static ucs_arbiter_cb_result_t
uct_ud_ep_pending_cancel_cb(ucs_arbiter_t *arbiter, ucs_arbiter_elem_t *elem,
                        void *arg)
{
    uct_ud_ep_t *ep = ucs_container_of(ucs_arbiter_elem_group(elem), 
                                       uct_ud_ep_t, tx.pending.group);
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    uct_pending_req_t *req;

    /* we may have pending op on ep */
    if (&ep->tx.pending.elem == elem) { 
        /* return ignored by arbiter */
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

    /* uct user should not have anything pending */
    req = ucs_container_of(elem, uct_pending_req_t, priv);
    ucs_warn("ep=%p removing user pending req=%p", ep, req);
    iface->tx.pending_q_len--;
    
    /* return ignored by arbiter */
    return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ud_ep_t)
{
    uct_ud_iface_t *iface = ucs_derived_of(self->super.super.iface, uct_ud_iface_t);

    ucs_trace_func("ep=%p id=%d conn_id=%d", self, self->ep_id, self->conn_id);
    ucs_wtimer_remove(&self->slow_timer);
    uct_ud_iface_remove_ep(iface, self);
    uct_ud_iface_cep_remove(self);
   /* TODO: in disconnect ucs_frag_list_cleanup(&self->rx.ooo_pkts); */

    ucs_arbiter_group_purge(&iface->tx.pending_q, &self->tx.pending.group,
                            uct_ud_ep_pending_cancel_cb, 0);

    if (!ucs_queue_is_empty(&self->tx.window)) {
        ucs_debug("ep=%p id=%d conn_id=%d has %d unacked packets", 
                   self, self->ep_id, self->conn_id, 
                   (int)ucs_queue_length(&self->tx.window));
    }
    ucs_arbiter_group_cleanup(&self->tx.pending.group);
}

UCS_CLASS_DEFINE(uct_ud_ep_t, uct_base_ep_t);

void uct_ud_ep_clone(uct_ud_ep_t *old_ep, uct_ud_ep_t *new_ep)
{
    uct_ep_t *ep_h = &old_ep->super.super;
    uct_iface_t *iface_h = ep_h->iface;

    uct_ud_iface_replace_ep(ucs_derived_of(iface_h, uct_ud_iface_t), old_ep, new_ep);
    memcpy(new_ep, old_ep, sizeof(uct_ud_ep_t)); 
}

ucs_status_t uct_ud_ep_get_address(uct_ep_h tl_ep, struct sockaddr *addr)
{
    uct_ud_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_ep_t);
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    uct_sockaddr_ib_t *ib_addr = (uct_sockaddr_ib_t *)addr;

    uct_ib_iface_get_address(&iface->super.super.super, addr);
    ib_addr->qp_num = iface->qp->qp_num;
    ib_addr->id     = ep->ep_id;
    return UCS_OK;
}

static ucs_status_t uct_ud_ep_connect_to_iface(uct_ud_ep_t *ep,
                                               const uct_sockaddr_ib_t *if_addr)
{   
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super);

    uct_ud_ep_reset(ep);

    ucs_debug("%s:%d slid=%d qpn=%d ep_id=%u ep=%p connected to IFACE dlid=%d qpn=%d", 
              ibv_get_device_name(dev->ibv_context->device),
              iface->super.port_num,
              dev->port_attr[iface->super.port_num-dev->first_port].lid,
              iface->qp->qp_num,
              ep->ep_id, ep, 
              if_addr->lid, if_addr->qp_num);

    return UCS_OK;
}

static ucs_status_t uct_ud_ep_disconnect_from_iface(uct_ep_h tl_ep)
{
    uct_ud_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_ep_t);

    uct_ud_ep_reset(ep);
    ep->dest_ep_id = UCT_UD_EP_NULL_ID;

    return UCS_OK;
}

ucs_status_t uct_ud_ep_create_connected_common(uct_ud_iface_t *iface, 
                                               const uct_sockaddr_ib_t *addr, 
                                               uct_ud_ep_t **new_ep_p, 
                                               uct_ud_send_skb_t **skb_p)
{
    ucs_status_t status;
    uct_ud_ep_t *ep;
    uct_ep_h new_ep_h;

    ep = uct_ud_iface_cep_lookup(iface, addr, UCT_UD_EP_CONN_ID_MAX);
    if (ep) {
        *new_ep_p = ep;
        *skb_p    = NULL;
        return UCS_OK;
    }

    status = uct_ep_create(&iface->super.super.super, &new_ep_h);
    if (status != UCS_OK) {
        return status;
    }
    ep = ucs_derived_of(new_ep_h, uct_ud_ep_t);

    status = uct_ud_ep_connect_to_iface(ep, addr);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ud_iface_cep_insert(iface, addr, ep, UCT_UD_EP_CONN_ID_MAX);
    if (status != UCS_OK) {
        goto err_cep_insert;
    }

    *skb_p = uct_ud_ep_prepare_creq(ep);
    if (!*skb_p) {
        status = UCS_ERR_NO_RESOURCE;
        goto err_creq;
    }

    *new_ep_p = ep;
    return UCS_OK;

err_creq:
    uct_ud_iface_cep_rollback(iface, addr, ep);
err_cep_insert:
    uct_ud_ep_disconnect_from_iface(&ep->super.super);
    return status;
}

void uct_ud_ep_destroy_connected(uct_ud_ep_t *ep, const uct_sockaddr_ib_t *addr)
{
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    uct_ud_iface_cep_rollback(iface, addr, ep);
    uct_ud_ep_disconnect_from_iface(&ep->super.super);
}

ucs_status_t uct_ud_ep_connect_to_ep(uct_ud_ep_t *ep,
                                     const struct sockaddr *addr)
{
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super);
    const uct_sockaddr_ib_t *ib_addr = (uct_sockaddr_ib_t *)addr;

    ucs_assert_always(ep->dest_ep_id == UCT_UD_EP_NULL_ID);
    ucs_trace_func("");

    ep->dest_ep_id = ib_addr->id;
    uct_ud_ep_reset(ep);

    ucs_debug("%s:%d slid=%d qpn=%d ep=%u connected to dlid=%d qpn=%d ep=%u", 
              ibv_get_device_name(dev->ibv_context->device),
              iface->super.port_num,
              dev->port_attr[iface->super.port_num-dev->first_port].lid,
              iface->qp->qp_num,
              ep->ep_id, 
              ib_addr->lid, ib_addr->qp_num, ep->dest_ep_id);

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void 
uct_ud_ep_process_ack(uct_ud_iface_t *iface, uct_ud_ep_t *ep, 
                      uct_ud_psn_t ack_psn)
{
    uct_ud_send_skb_t *skb;

    if (ucs_unlikely(UCT_UD_PSN_COMPARE(ack_psn, <=, ep->tx.acked_psn))) {
        return;
    }

    ep->tx.acked_psn = ack_psn;
    
    /* Release acknowledged skb's */
    ucs_queue_for_each_extract(skb, &ep->tx.window, queue,
                               UCT_UD_PSN_COMPARE(skb->neth[0].psn, <=, ack_psn)) {
        ucs_mpool_put(skb);
    }

    /* update window */
    ep->tx.max_psn = ep->tx.acked_psn + UCT_UD_MAX_WINDOW;

    ucs_arbiter_group_schedule(&iface->tx.pending_q, &ep->tx.pending.group);

    ep->tx.send_time = uct_ud_iface_get_async_time(iface);
}

static inline void uct_ud_ep_rx_put(uct_ud_neth_t *neth, unsigned byte_len)
{
    uct_ud_put_hdr_t *put_hdr;

    put_hdr = (uct_ud_put_hdr_t *)(neth+1);

    memcpy((void *)put_hdr->rva, put_hdr+1, 
            byte_len - sizeof(*neth) - sizeof(*put_hdr));
}

static uct_ud_ep_t *uct_ud_ep_create_passive(uct_ud_iface_t *iface, uct_ud_ctl_hdr_t *ctl)
{
    uct_ud_ep_t *ep;
    ucs_status_t status;
    uct_ep_t *ep_h;
    uct_iface_t *iface_h =  &iface->super.super.super;
    /* create new endpoint */
    status = iface_h->ops.ep_create(iface_h, &ep_h);
    ucs_assert_always(status == UCS_OK);
    ep = ucs_derived_of(ep_h, uct_ud_ep_t);

    status = iface_h->ops.ep_connect_to_ep(ep_h, 
            (struct sockaddr *)&ctl->conn_req.ib_addr);
    ucs_assert_always(status == UCS_OK);

    status = uct_ud_iface_cep_insert(iface, &ctl->conn_req.ib_addr, ep, ctl->conn_req.conn_id);
    ucs_assert_always(status == UCS_OK);
    return ep;
}

static void uct_ud_ep_rx_creq(uct_ud_iface_t *iface, uct_ud_neth_t *neth)
{
    uct_ud_ep_t *ep;
    uct_ud_ctl_hdr_t *ctl = (uct_ud_ctl_hdr_t *)(neth + 1);

    ucs_assert_always(ctl->type == UCT_UD_PACKET_CREQ);

    ep = uct_ud_iface_cep_lookup(iface, &ctl->conn_req.ib_addr, ctl->conn_req.conn_id);
    if (!ep) {
        ep = uct_ud_ep_create_passive(iface, ctl);
        ucs_assert_always(ep != NULL);
        ep->rx.ooo_pkts.head_sn = neth->psn;
    } else {
        if (ep->dest_ep_id == UCT_UD_EP_NULL_ID) {
            /* simultaniuos CREQ */
            ep->dest_ep_id = ctl->conn_req.ib_addr.id;
            ep->rx.ooo_pkts.head_sn = neth->psn;
            ucs_debug("created ep=%p (iface=%p conn_id=%d ep_id=%d, dest_ep_id=%d rx_psn=%u)", ep, iface, ep->conn_id, ep->ep_id, ep->dest_ep_id, ep->rx.ooo_pkts.head_sn);
        }
    }

    ucs_assert_always(ctl->conn_req.conn_id == ep->conn_id);
    ucs_assert_always(ctl->conn_req.ib_addr.id == ep->dest_ep_id);
    /* creq must always have same psn */
    ucs_assert_always(ep->rx.ooo_pkts.head_sn == neth->psn);
    /* scedule connection reply op */
    UCT_UD_EP_HOOK_CALL_RX(ep, neth, sizeof(*neth) + sizeof(*ctl));
    uct_ud_ep_ctl_op_add(iface, ep, UCT_UD_EP_OP_CREP);
}

static void uct_ud_ep_rx_ctl(uct_ud_iface_t *iface, uct_ud_ep_t *ep, uct_ud_ctl_hdr_t *ctl)
{
    ucs_trace_func("");
    ucs_assert_always(ctl->type == UCT_UD_PACKET_CREP);
    /* note that duplicate creps are discared earlier */
    ucs_assert_always(ep->dest_ep_id == UCT_UD_EP_NULL_ID || 
                      ep->dest_ep_id == ctl->conn_rep.src_ep_id);
    ep->dest_ep_id = ctl->conn_rep.src_ep_id;
    ucs_arbiter_group_schedule(&iface->tx.pending_q, &ep->tx.pending.group);
}

uct_ud_send_skb_t *uct_ud_ep_prepare_creq(uct_ud_ep_t *ep)
{
    uct_ud_send_skb_t *skb;
    uct_ud_neth_t *neth;
    uct_ud_ctl_hdr_t *creq;
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    uct_sockaddr_ib_t iface_addr;
    ucs_status_t status;

    ucs_assert_always(ep->dest_ep_id == UCT_UD_EP_NULL_ID);
    ucs_assert_always(ep->ep_id != UCT_UD_EP_NULL_ID);

    memset(&iface_addr, 0, sizeof(iface_addr)); /* make coverity happy */
    status = uct_ud_iface_get_address(&iface->super.super.super, (struct sockaddr *)&iface_addr);
    if (status != UCS_OK) {
        return NULL;
    }

    skb = uct_ud_iface_get_tx_skb(iface, ep);
    if (!skb) {
        return NULL;
    }

    neth = skb->neth;
    uct_ud_neth_init_data(ep, neth);

    neth->packet_type  = UCT_UD_EP_NULL_ID;
    neth->packet_type |= UCT_UD_PACKET_FLAG_CTL;

    creq = (uct_ud_ctl_hdr_t *)(neth + 1);

    creq->type                    = UCT_UD_PACKET_CREQ;
    creq->conn_req.ib_addr.qp_num = iface_addr.qp_num;
    creq->conn_req.ib_addr.lid    = iface_addr.lid;
    creq->conn_req.ib_addr.id     = ep->ep_id;
    creq->conn_req.conn_id        = ep->conn_id;

    skb->len = sizeof(*neth) + sizeof(*creq);
    return skb;
}

void uct_ud_ep_process_rx(uct_ud_iface_t *iface, uct_ud_neth_t *neth, unsigned byte_len,
                          uct_ud_recv_skb_t *skb)
{
    uint32_t dest_id;
    uint32_t is_am, am_id;
    uct_ud_ep_t *ep = 0; /* todo: check why gcc complaints about uninitialized var */
    ucs_frag_list_ooo_type_t ooo_type;

    dest_id = uct_ud_neth_get_dest_id(neth);
    am_id   = uct_ud_neth_get_am_id(neth);
    is_am   = neth->packet_type & UCT_UD_PACKET_FLAG_AM;

    if (ucs_unlikely(dest_id == UCT_UD_EP_NULL_ID)) {
        /* must be connection request packet */
        uct_ud_iface_log_rx(iface, NULL, neth, byte_len);
        uct_ud_ep_rx_creq(iface, neth);
        goto out;
    }
    else if (ucs_unlikely(!ucs_ptr_array_lookup(&iface->eps, dest_id, ep) ||
                     ep->ep_id != dest_id)) {
        uct_ud_iface_log_rxdrop(iface, ep, neth, byte_len);
        /* Drop the packet because it is
         * allowed to do disconnect without flush/barrier. So it
         * is possible to get packet for the ep that has been destroyed 
         */
        ucs_debug("Failed to find ep(%d)", dest_id);
        goto out;
    } 

    ucs_assert(ep->ep_id != UCT_UD_EP_NULL_ID);
    UCT_UD_EP_HOOK_CALL_RX(ep, neth, byte_len);
    uct_ud_iface_log_rx(iface, ep, neth, byte_len);
    
    uct_ud_ep_process_ack(iface, ep, neth->ack_psn);

    if (ucs_unlikely(neth->packet_type & UCT_UD_PACKET_FLAG_ACK_REQ)) {
        uct_ud_ep_ctl_op_add(iface, ep, UCT_UD_EP_OP_ACK);
    }

    if (ucs_unlikely(!is_am)) {
        if ((byte_len == sizeof(*neth))) {
            goto out;
        }
        if (neth->packet_type & UCT_UD_PACKET_FLAG_CTL) {
            uct_ud_ep_rx_ctl(iface, ep, (uct_ud_ctl_hdr_t *)(neth + 1));
            goto out;
        }
    }

    ooo_type = ucs_frag_list_insert(&ep->rx.ooo_pkts, &skb->ooo_elem, neth->psn);
    if (ucs_unlikely(ooo_type != UCS_FRAG_LIST_INSERT_FAST)) {
        if (ooo_type != UCS_FRAG_LIST_INSERT_DUP &&
            ooo_type != UCS_FRAG_LIST_INSERT_FAIL) {
            ucs_fatal("Out of order is not implemented: got %d", ooo_type);
        }
        ucs_trace_data("DUP/OOB - schedule ack, head_sn=%d sn=%d", 
                       ep->rx.ooo_pkts.head_sn, neth->psn);
        uct_ud_ep_ctl_op_add(iface, ep, UCT_UD_EP_OP_ACK);
        goto out;
    }
    

    if (ucs_unlikely(!is_am && (neth->packet_type & UCT_UD_PACKET_FLAG_PUT))) {
        /* TODO: remove once ucp implements put */
        uct_ud_ep_rx_put(neth, byte_len);
        goto out;
    }

    uct_ib_iface_invoke_am(&iface->super, am_id, neth + 1,
                           byte_len - sizeof(*neth), &skb->super);
    return;

out:
    ucs_mpool_put(skb);
}

ucs_status_t uct_ud_ep_flush_nolock(uct_ud_iface_t *iface, uct_ud_ep_t *ep)
{
    uct_ud_send_skb_t *skb;

    if (ucs_unlikely(!uct_ud_ep_is_connected(ep))) {
        return UCS_OK;
    }

    if (ucs_queue_is_empty(&ep->tx.window)) {
        uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_ACK_REQ);
        return UCS_OK;
    }
    
    skb = ucs_queue_tail_elem_non_empty(&ep->tx.window, uct_ud_send_skb_t, queue);
    if (skb->flags & UCT_UD_SEND_SKB_FLAG_ACK_REQ) {
        /* last packet was already sent with ack request. 
         * either by flush or 
         * Do not send more, let reqular retransmission 
         * mechanism do the work
         */
        return UCS_INPROGRESS;
    }

    skb->flags |= UCT_UD_SEND_SKB_FLAG_ACK_REQ;
    uct_ud_ep_ctl_op_add(iface, ep, UCT_UD_EP_OP_ACK_REQ);
    return UCS_INPROGRESS;
}

ucs_status_t uct_ud_ep_flush(uct_ep_h ep_h)
{
    ucs_status_t status;
    uct_ud_ep_t *ep = ucs_derived_of(ep_h, uct_ud_ep_t);
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, 
                                           uct_ud_iface_t);
    uct_ud_enter(iface);
    status = uct_ud_ep_flush_nolock(iface, ep);
    uct_ud_leave(iface);
    return status;
}

static uct_ud_send_skb_t *uct_ud_ep_prepare_crep(uct_ud_ep_t *ep)
{
    uct_ud_send_skb_t *skb;
    uct_ud_neth_t *neth;
    uct_ud_ctl_hdr_t *crep;
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);

    ucs_assert_always(ep->dest_ep_id != UCT_UD_EP_NULL_ID);
    ucs_assert_always(ep->ep_id != UCT_UD_EP_NULL_ID);

    skb = uct_ud_iface_res_skb_get(iface);
    ucs_assert_always(skb != NULL);

    neth = skb->neth;
    uct_ud_neth_init_data(ep, neth);

    neth->packet_type  = ep->dest_ep_id;
    neth->packet_type |= (UCT_UD_PACKET_FLAG_ACK_REQ|UCT_UD_PACKET_FLAG_CTL);

    crep = (uct_ud_ctl_hdr_t *)(neth + 1);

    crep->type               = UCT_UD_PACKET_CREP;
    crep->conn_rep.src_ep_id = ep->ep_id;

    skb->len = sizeof(*neth) + sizeof(*crep);
    uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_CREP);
    return skb;
}

static uct_ud_send_skb_t *uct_ud_ep_resend(uct_ud_ep_t *ep)
{
    uct_ud_send_skb_t *skb, *sent_skb;
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
     /* We had new ACKs which freed the skb's rt_pos was pointing to */
    if (UCT_UD_PSN_COMPARE(ep->tx.rt_psn, <=, ep->tx.acked_psn)) {
        ep->tx.rt_pos = ucs_queue_iter_begin(&ep->tx.window);
        ucs_debug("ep(%p): ack received during resend rt_psn=%d acked_psn=%d", 
                  ep, ep->tx.rt_psn, ep->tx.acked_psn);
    }

    if (ucs_queue_iter_end(&ep->tx.window, ep->tx.rt_pos)) {
        ucs_debug("ep(%p): resending completed", ep);
        uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_RESEND);
        return NULL;
    }
    skb = uct_ud_iface_res_skb_get(iface);
    ucs_assert_always(skb != NULL);

    /* TODO: variable window */
    sent_skb = ucs_queue_iter_elem(sent_skb, ep->tx.rt_pos, queue);
    ep->tx.rt_pos = ucs_queue_iter_next(ep->tx.rt_pos);
    ep->tx.rt_psn = sent_skb->neth->psn;
    memcpy(skb->neth, sent_skb->neth, sent_skb->len);
    skb->neth->ack_psn = ep->rx.acked_psn;
    skb->len = sent_skb->len;
    /* force ack request on every resend */
    skb->neth->packet_type |= UCT_UD_PACKET_FLAG_ACK_REQ;
    ucs_debug("ep(%p): resending rt_psn %u acked_psn %u", ep, ep->tx.rt_psn, ep->tx.acked_psn);

    if (ucs_queue_iter_end(&ep->tx.window, ep->tx.rt_pos)) {
        uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_RESEND);
        ucs_debug("ep(%p): resending completed", ep);
    }
    return skb;
}


static void 
uct_ud_ep_do_pending_ctl(uct_ud_ep_t *ep, uct_ud_iface_t *iface)
{
    uct_ud_send_skb_t *skb;

    if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_CREP)) {
        skb = uct_ud_ep_prepare_crep(ep);
    } else if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_RESEND)) {
        skb =  uct_ud_ep_resend(ep);
    } else if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_ACK)) {
        skb = &iface->tx.skb_inl.super;
        uct_ud_neth_ctl_ack(ep, skb->neth);
        uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_ACK);
    } else if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_ACK_REQ)) {
        skb = &iface->tx.skb_inl.super;
        uct_ud_neth_ctl_ack_req(ep, skb->neth);
        uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_ACK_REQ);
    } else if (uct_ud_ep_ctl_op_isany(ep)) {
        ucs_fatal("unsupported pending op mask: %x", ep->tx.pending.ops);
    } else {
        skb = 0;
    }

    if (!skb) {
        /* no pending - nothing to do */
        return;
    }

    VALGRIND_MAKE_MEM_DEFINED(skb, sizeof *skb);
    iface->ops.tx_skb(iface, ep, skb);
    uct_ud_ep_log_tx(iface, ep, skb);
    uct_ud_iface_res_skb_put(iface, skb);
}

static inline ucs_arbiter_cb_result_t
uct_ud_ep_ctl_op_next(uct_ud_ep_t *ep)
{
    if (uct_ud_ep_ctl_op_isany(ep)) {
        /* can send more control - come here later */
        return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
    } 
    /* no more control - nothing to do in
     * this dispatch cycle. */
    return UCS_ARBITER_CB_RESULT_RESCHED_GROUP;
}

/**
 * pending operations are processed according to priority:
 * - high prio control:
 *   - crep reply 
 *   - resends   
 * - pending uct requests 
 * - low prio control: ack reply/ack requests 
 *
 * Low priority control can be send along with user data, so
 * there is a good chance that processing pending uct reqs will 
 * also deal with the low prio control.
 * However we can not let pending uct req block control forever.
 */
ucs_arbiter_cb_result_t 
uct_ud_ep_do_pending(ucs_arbiter_t *arbiter, ucs_arbiter_elem_t *elem,
                     void *arg)
{
    uct_ud_ep_t *ep = ucs_container_of(ucs_arbiter_elem_group(elem), 
                                       uct_ud_ep_t, tx.pending.group);
    uct_ud_iface_t *iface = ucs_container_of(arbiter, uct_ud_iface_t,
                                             tx.pending_q);
    uintptr_t in_async_progress = (uintptr_t)arg;

    /* check if we have global resources
     * - tx_wqe 
     * - skb 
     * control messages does not need skb.
     */
    if (!uct_ud_iface_can_tx(iface)) {
        return UCS_ARBITER_CB_RESULT_STOP;
    }

    /* here we rely on the fact that arbiter
     * will start next dispatch cycle from the
     * next group. 
     * So it is ok to stop if there is no ctl.
     * However in worst case only one ctl per
     * dispatch cycle will be send.
     */
    if (!uct_ud_iface_has_skbs(iface) && !uct_ud_ep_ctl_op_isany(ep)) {
        return UCS_ARBITER_CB_RESULT_STOP;
    }

    /* we can desched group: iff
     * - no control
     * - no ep resources (connect or window)
     **/
    
    if (!uct_ud_ep_ctl_op_isany(ep) &&  
       (!uct_ud_ep_is_connected(ep) ||
         uct_ud_ep_no_window(ep))) {
        return UCS_ARBITER_CB_RESULT_DESCHED_GROUP;
    }

    if (&ep->tx.pending.elem == elem) { 
        uct_ud_ep_do_pending_ctl(ep, iface);
        if (uct_ud_ep_ctl_op_isany(ep)) {
            /* there is still some ctl left. go to next group */
            return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
        } else {
            /* no more ctl - dummy elem can be removed */
            return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
        }
    } 

    /* user pending can be send iff
     * - not in async progress
     * - there are only low priority ctl pending or not ctl at all
     */
    if (!in_async_progress &&
            (uct_ud_ep_ctl_op_check_ex(ep, UCT_UD_EP_OP_CTL_LOW_PRIO) || 
             !uct_ud_ep_ctl_op_isany(ep))) {
        uct_pending_req_t *req;
        ucs_status_t status;

        req = ucs_container_of(elem, uct_pending_req_t, priv);
        status = req->func(req);

        if (status != UCS_OK) {
            /* avoid deadlock: send low priority ctl if user cb failed
             * no need to check for low prio here because we 
             * already checked above. 
             */
            uct_ud_ep_do_pending_ctl(ep, iface);
            return uct_ud_ep_ctl_op_next(ep);
        } 
        iface->tx.pending_q_len--;
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }
    /* try to send ctl messages */
    uct_ud_ep_do_pending_ctl(ep, iface);
    return uct_ud_ep_ctl_op_next(ep);
}

ucs_status_t uct_ud_ep_pending_add(uct_ep_h ep_h, uct_pending_req_t *req)
{
    uct_ud_ep_t *ep = ucs_derived_of(ep_h, uct_ud_ep_t);
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, 
                                           uct_ud_iface_t);

    uct_ud_enter(iface);

    /* try to flush pending queue first */
    uct_ud_iface_progress_pending(iface, 0);

    if (uct_ud_iface_can_tx(iface) &&
        uct_ud_iface_has_skbs(iface) &&
        uct_ud_ep_is_connected(ep) &&
        !uct_ud_ep_no_window(ep)) {

        uct_ud_leave(iface);
        ucs_debug("can not add pending because ep=%p is able to transmit", ep);
        return UCS_ERR_BUSY;
    }

    ucs_arbiter_elem_init((ucs_arbiter_elem_t *)req->priv);
    ucs_arbiter_group_push_elem(&ep->tx.pending.group, 
                                (ucs_arbiter_elem_t *)req->priv);
    ucs_arbiter_group_schedule(&iface->tx.pending_q, &ep->tx.pending.group);
    iface->tx.pending_q_len++;
    uct_ud_leave(iface);
    return UCS_OK;
}

static ucs_arbiter_cb_result_t
uct_ud_ep_pending_purge_cb(ucs_arbiter_t *arbiter, ucs_arbiter_elem_t *elem,
                        void *arg)
{
    uct_pending_callback_t cb = (uct_pending_callback_t)arg;
    uct_ud_ep_t *ep = ucs_container_of(ucs_arbiter_elem_group(elem), 
                                       uct_ud_ep_t, tx.pending.group);
    uct_pending_req_t *req;
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);

    if (&ep->tx.pending.elem == elem) { 
        /* return ignored by arbiter */
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }
    req = ucs_container_of(elem, uct_pending_req_t, priv);
    if (cb) {
        cb(req);
    } else {
        ucs_warn("ep=%p cancelling user pending request %p", ep, req);
    }
    iface->tx.pending_q_len--;
    
    /* return ignored by arbiter */
    return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
}


void uct_ud_ep_pending_purge(uct_ep_h ep_h, uct_pending_callback_t cb)
{
    uct_ud_ep_t *ep = ucs_derived_of(ep_h, uct_ud_ep_t);
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, 
                                           uct_ud_iface_t);

    uct_ud_enter(iface);
    ucs_arbiter_group_purge(&iface->tx.pending_q, &ep->tx.pending.group,
                            uct_ud_ep_pending_purge_cb, cb);
    if (uct_ud_ep_ctl_op_isany(ep)) {
        ucs_arbiter_group_push_elem(&ep->tx.pending.group,
                                    &ep->tx.pending.elem);
        ucs_arbiter_group_schedule(&iface->tx.pending_q, &ep->tx.pending.group);
    } 
    uct_ud_leave(iface);
}

void  uct_ud_ep_disconnect(uct_ep_h ep)
{
    /*
     * At the moment scedule flush and keep ep
     * until interface is destroyed. User should not send any
     * new data
     * In the future consider doin full fledged disconnect
     * protocol. Kind of TCP (FIN/ACK). Doing this will save memory
     * on the other hand active ep will need more memory to keep its state
     * and such protocol will add extra complexity
     */

    ucs_trace_func("");
    /* cancel user pending */
    uct_ud_ep_pending_purge(ep, 0);

    /* scedule flush */
    uct_ud_ep_flush(ep);

    /* TODO: at leat in debug mode keep and check ep state  */
}
