/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 *
 * $COPYRIGHT$
 * $HEADER$
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
    ep->tx.pending.ops = UCT_UD_EP_OP_NONE;
    ucs_queue_head_init(&ep->tx.window);

    ep->rx.acked_psn = 0;
    ucs_frag_list_init(ep->tx.psn-1, &ep->rx.ooo_pkts, 0 /*TODO: ooo support */
                       UCS_STATS_ARG(ep->rx.stats));
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
    UCT_UD_EP_HOOK_INIT(self);
    ucs_debug("NEW EP: iface=%p ep=%p id=%d", iface, self, self->ep_id);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ud_ep_t)
{
    uct_ud_iface_t *iface = ucs_derived_of(self->super.super.iface, uct_ud_iface_t);

    ucs_trace_func("ep=%p id=%d conn_id=%d", self, self->ep_id, self->conn_id);
    uct_ud_iface_remove_ep(iface, self);
    uct_ud_iface_cep_remove(self);
   /* TODO: in disconnect ucs_frag_list_cleanup(&self->rx.ooo_pkts); */
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

ucs_status_t uct_ud_ep_connect_to_iface(uct_ud_ep_t *ep,
                                        const struct sockaddr *addr)
{   
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super);
    uct_sockaddr_ib_t *if_addr = (uct_sockaddr_ib_t *)addr;

    ep->dest_qpn = if_addr->qp_num;
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

ucs_status_t uct_ud_ep_disconnect_from_iface(uct_ep_h tl_ep)
{
    uct_ud_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_ep_t);

    uct_ud_ep_reset(ep);
    ep->dest_ep_id = UCT_UD_EP_NULL_ID;

    return UCS_OK;
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
    ep->dest_qpn   = ib_addr->qp_num;

    uct_ud_ep_reset(ep);

    ucs_debug("%s:%d slid=%d qpn=%d ep=%u connected to dlid=%d qpn=%d ep=%u", 
              ibv_get_device_name(dev->ibv_context->device),
              iface->super.port_num,
              dev->port_attr[iface->super.port_num-dev->first_port].lid,
              iface->qp->qp_num,
              ep->ep_id, 
              ib_addr->lid, ep->dest_qpn, ep->dest_ep_id);

    return UCS_OK;
}

static inline void uct_ud_ep_process_ack(uct_ud_ep_t *ep, uct_ud_psn_t ack_psn)
{
    uct_ud_send_skb_t *skb;

    if (ucs_unlikely(UCT_UD_PSN_COMPARE(ack_psn, <=, ep->tx.acked_psn))) {
        return;
    }

    ep->tx.acked_psn = ack_psn;
    
    /* Release acknowledged skb's */
    ucs_queue_for_each_extract(skb, &ep->tx.window, queue,
                               UCT_UD_PSN_COMPARE(skb->neth[0].psn, <=, ack_psn)) {
        /* TODO call zcopy completion */
        ucs_mpool_put(skb);
    }

    /* update window */
    ep->tx.max_psn =  ep->tx.acked_psn + UCT_UD_MAX_WINDOW;
}

static inline void uct_ud_ep_rx_put(uct_ud_neth_t *neth, unsigned byte_len)
{
    uct_ud_put_hdr_t *put_hdr;

    put_hdr = (uct_ud_put_hdr_t *)neth+1;

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

    /* connection request */
    ucs_trace_data("CREQ RX (qp=%x lid=%d ep_id=%d conn_id=%d)",
                   ctl->conn_req.ib_addr.qp_num,
                   ctl->conn_req.ib_addr.lid,
                   ctl->conn_req.ib_addr.id,
                   ctl->conn_req.conn_id);
    
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
            ucs_debug("created ep=%p (conn_id=%d ep_id=%d, dest_ep_id=%d rx_psn=%u)", ep, ep->conn_id, ep->ep_id, ep->dest_ep_id, ep->rx.ooo_pkts.head_sn);
        }
    }

    ucs_assert_always(ctl->conn_req.conn_id == ep->conn_id);
    ucs_assert_always(ctl->conn_req.ib_addr.id == ep->dest_ep_id);
    /* creq must always have same psn */
    ucs_assert_always(ep->rx.ooo_pkts.head_sn == neth->psn);
    /* scedule connection reply op */
    UCT_UD_EP_HOOK_CALL_RX(ep, neth);
    uct_ud_iface_queue_pending(iface, ep, UCT_UD_EP_OP_CREP);
}

static void uct_ud_ep_rx_ctl(uct_ud_iface_t *iface, uct_ud_ep_t *ep, uct_ud_ctl_hdr_t *ctl)
{
    ucs_trace_func("");
    ucs_assert_always(ctl->type == UCT_UD_PACKET_CREP);
    /* note that duplicate creps are discared earlier */
    ucs_assert_always(ep->dest_ep_id == UCT_UD_EP_NULL_ID || 
                      ep->dest_ep_id == ctl->conn_rep.src_ep_id);
    ep->dest_ep_id = ctl->conn_rep.src_ep_id;
    ucs_trace_data("CREP RX: ep=%p (conn_id=%d ep_id=%d, dest_ep_id=%d)", ep, ep->conn_id, ep->ep_id, ep->dest_ep_id);
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

    ucs_trace_data("CREQ (qp=%x lid=%d ep_id=%d conn_id=%d)",
                   creq->conn_req.ib_addr.qp_num,
                   creq->conn_req.ib_addr.lid,
                   creq->conn_req.ib_addr.id,
                   creq->conn_req.conn_id);

    skb->len = sizeof(*neth) + sizeof(*creq);
    /* TODO: add to the list of ceps */
    UCT_UD_EP_HOOK_CALL_TX(ep, skb->neth);
    uct_ud_iface_complete_tx_skb(iface, ep, skb);
    return skb;
}

uct_ud_send_skb_t *uct_ud_ep_prepare_crep(uct_ud_ep_t *ep)
{
    uct_ud_send_skb_t *skb;
    uct_ud_neth_t *neth;
    uct_ud_ctl_hdr_t *crep;
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);

    ucs_assert_always(ep->dest_ep_id != UCT_UD_EP_NULL_ID);
    ucs_assert_always(ep->ep_id != UCT_UD_EP_NULL_ID);

    skb = uct_ud_iface_get_tx_skb(iface, ep);
    if (!skb) {
        return NULL;
    }

    neth = skb->neth;
    uct_ud_neth_init_data(ep, neth);

    neth->packet_type  = ep->dest_ep_id;
    neth->packet_type |= (UCT_UD_PACKET_FLAG_ACK_REQ|UCT_UD_PACKET_FLAG_CTL);

    crep = (uct_ud_ctl_hdr_t *)(neth + 1);

    crep->type               = UCT_UD_PACKET_CREP;
    crep->conn_rep.src_ep_id = ep->ep_id;

    ucs_trace_data("CREP ep=%p (ep_id=%d, dest_ep_id=%d) packet_type=0x%0x", ep, ep->ep_id, ep->dest_ep_id, neth->packet_type);
    skb->len = sizeof(*neth) + sizeof(*crep);
    UCT_UD_EP_HOOK_CALL_TX(ep, skb->neth);
    uct_ud_iface_complete_tx_skb(iface, ep, skb);
    return skb;
}

void uct_ud_ep_process_rx(uct_ud_iface_t *iface, uct_ud_neth_t *neth, unsigned byte_len, uct_ud_recv_skb_t *skb)
{
    uint32_t dest_id;
    uint32_t is_am, am_id;
    uct_ud_ep_t *ep = 0; /* todo: check why gcc complaints about uninitialized var */
    ucs_frag_list_ooo_type_t ooo_type;

    dest_id = uct_ud_neth_get_dest_id(neth);
    am_id   = uct_ud_neth_get_am_id(neth);
    is_am   = neth->packet_type & UCT_UD_PACKET_FLAG_AM;

    ucs_trace_data("src_ep= dest_ep=%x psn=%d ack_psn=%d am_id=%d is_am=%d len=%d packet_type=%08x",
                   dest_id, (int)neth->psn, (int)neth->ack_psn, (int)am_id, (int)is_am, byte_len, neth->packet_type);
    if (ucs_unlikely(dest_id == UCT_UD_EP_NULL_ID)) {
        /* must be connection request packet */
        uct_ud_ep_rx_creq(iface, neth);
        goto out;
    }
    else if (ucs_unlikely(!ucs_ptr_array_lookup(&iface->eps, dest_id, ep) ||
                     ep->ep_id != dest_id)) {
        /* TODO: in the future just drop the packet */
        ucs_fatal("Faied to find ep(%d)", dest_id);
        goto out;
    } 
    ucs_trace_data("ep: %p (conn_id=%d ep_id=%d dest_ep_id=%d rx_psn=%u)",
                   ep, ep->conn_id, ep->ep_id, ep->dest_ep_id,
                   ep->rx.ooo_pkts.head_sn);

    ucs_assert(ep->ep_id != UCT_UD_EP_NULL_ID);
    UCT_UD_EP_HOOK_CALL_RX(ep, neth);
    
    uct_ud_ep_process_ack(ep, neth->ack_psn);

    if (ucs_unlikely(neth->packet_type & UCT_UD_PACKET_FLAG_ACK_REQ)) {
        uct_ud_iface_queue_pending(iface, ep, UCT_UD_EP_OP_ACK);
    }

    if (byte_len == sizeof(*neth)) {
        goto out;
    }

    ooo_type = ucs_frag_list_insert(&ep->rx.ooo_pkts, &skb->ooo_elem, neth->psn);
    if (ucs_unlikely(ooo_type != UCS_FRAG_LIST_INSERT_FAST)) {
        ucs_warn("src_ep= dest_ep=%u rx_psn=%hu psn=%hu ack_psn=%hu am_id=%d is_am=%d len=%d",
                 dest_id, ep->rx.ooo_pkts.head_sn, neth->psn, neth->ack_psn, (int)am_id, (int)is_am, byte_len);
        ucs_fatal("Out of order is not implemented: got %d", ooo_type);
        goto out;
    }
    
    if (ucs_unlikely(!is_am && neth->packet_type & UCT_UD_PACKET_FLAG_CTL)) {
        uct_ud_ep_rx_ctl(iface, ep, (uct_ud_ctl_hdr_t *)(neth + 1));
        goto out;
    }

    if (!is_am && (neth->packet_type & UCT_UD_PACKET_FLAG_PUT)) {
        uct_ud_ep_rx_put(neth, byte_len);
        goto out;
    }

    uct_ib_iface_invoke_am(&iface->super, am_id, neth + 1,
                           byte_len - sizeof(*neth), &skb->super);
    return;

out:
    ucs_mpool_put(skb);
}
