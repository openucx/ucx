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


UCS_CLASS_INIT_FUNC(uct_ud_ep_t, uct_ud_iface_t *iface)
{
    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    self->dest_ep_id = UCT_UD_EP_NULL_ID;
    uct_ud_iface_add_ep(iface, self);
    UCT_UD_EP_HOOK_INIT(self);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ud_ep_t)
{
    uct_ud_iface_t *iface = ucs_derived_of(self->super.super.iface, uct_ud_iface_t);

    ucs_trace_func("");

    uct_ud_iface_remove_ep(iface, self);
   /* TODO: in disconnect ucs_frag_list_cleanup(&self->rx.ooo_pkts); */
}

UCS_CLASS_DEFINE(uct_ud_ep_t, uct_base_ep_t);


ucs_status_t uct_ud_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *ep_addr)
{
    uct_ud_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_ep_t);

    ((uct_ud_ep_addr_t*)ep_addr)->ep_id = ep->ep_id;
    ucs_debug("ep_addr=%d", ep->ep_id);
    return UCS_OK;
}

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


ucs_status_t uct_ud_ep_connect_to_ep(uct_ep_h tl_ep,
                                     const uct_iface_addr_t *tl_iface_addr,
                                     const uct_ep_addr_t *tl_ep_addr)
{
    uct_ud_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_ep_t);
    uct_ud_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ud_iface_t);
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super);
    uct_ud_iface_addr_t *if_addr = ucs_derived_of(tl_iface_addr, uct_ud_iface_addr_t);
    uct_ud_ep_addr_t *ep_addr = ucs_derived_of(tl_ep_addr, uct_ud_ep_addr_t);

    ucs_assert_always(ep->dest_ep_id == UCT_UD_EP_NULL_ID);
    ucs_trace_func("");

    ep->dest_ep_id = ep_addr->ep_id;
    ep->dest_qpn = if_addr->qp_num;

    uct_ud_ep_reset(ep);

    ucs_debug("%s:%d slid=%d qpn=%d ep=%u connected to dlid=%d qpn=%d ep=%u", 
              ibv_get_device_name(dev->ibv_context->device),
              iface->super.port_num,
              dev->port_attr[iface->super.port_num-dev->first_port].lid,
              iface->qp->qp_num,
              ep->ep_id, 
              if_addr->lid, if_addr->qp_num, ep->dest_ep_id);

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

void uct_ud_ep_process_rx(uct_ud_iface_t *iface, uct_ud_neth_t *neth, unsigned byte_len, uct_ud_recv_skb_t *skb)
{
    uint32_t dest_id;
    uint32_t is_am, is_put, am_id;
    uct_ud_ep_t *ep = 0; /* todo: check why gcc complaints about uninitialized var */
    ucs_frag_list_ooo_type_t ooo_type;

    dest_id = uct_ud_neth_get_dest_id(neth);
    am_id   = uct_ud_neth_get_am_id(neth);
    is_am   = neth->packet_type & UCT_UD_PACKET_FLAG_AM;
    is_put  = !is_am && (neth->packet_type & UCT_UD_PACKET_FLAG_PUT);

    ucs_trace_data("src_ep= dest_ep=%d psn=%d ack_psn=%d am_id=%d is_am=%d len=%d packet_type=%08x",
                   dest_id, (int)neth->psn, (int)neth->ack_psn, (int)am_id, (int)is_am, byte_len, neth->packet_type);
    if (ucs_unlikely(!ucs_ptr_array_lookup(&iface->eps, dest_id, ep) ||
                     ep->ep_id != dest_id)) {
        /* TODO: in the future just drop the packet */
        ucs_fatal("Failed to find ep(%d)", dest_id);
        return;
    } 
    ucs_assert(ep->ep_id != UCT_UD_EP_NULL_ID);
    UCT_UD_EP_HOOK_CALL_RX(ep, neth);
    
    uct_ud_ep_process_ack(ep, neth->ack_psn);

    if (ucs_unlikely(neth->packet_type & UCT_UD_PACKET_FLAG_ACK_REQ)) {
        uct_ud_iface_queue_pending(iface, ep, UCT_UD_EP_OP_ACK);
    }

    if (!is_am && !is_put) {
        ucs_mpool_put(skb);
        return;
    }

    ooo_type = ucs_frag_list_insert(&ep->rx.ooo_pkts, &skb->ooo_elem, neth->psn);
    if (ucs_unlikely(ooo_type != UCS_FRAG_LIST_INSERT_FAST)) {
        ucs_warn("src_ep= dest_ep=%u rx_psn=%hu psn=%hu ack_psn=%hu am_id=%d is_am=%d len=%d",
                 dest_id, ep->rx.ooo_pkts.head_sn, neth->psn, neth->ack_psn, (int)am_id, (int)is_am, byte_len);
        ucs_fatal("Out of order is not implemented: got %d", ooo_type);
        return;
    }

    if (is_put) {
        uct_ud_put_hdr_t *put_hdr;

        put_hdr = (uct_ud_put_hdr_t *)neth+1;

        memcpy((void *)put_hdr->rva, put_hdr+1, 
                byte_len - sizeof(*neth) - sizeof(*put_hdr));
        ucs_mpool_put(skb);
        return;
    }

    uct_ib_iface_invoke_am(&iface->super, am_id, neth + 1,
                           byte_len - sizeof(*neth), &skb->super);
}
