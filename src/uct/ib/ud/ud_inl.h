
static inline void uct_ud_iface_queue_pending(uct_ud_iface_t *iface,
                                              uct_ud_ep_t *ep, int op)
{
    if (ep->tx.pending.ops == UCT_UD_EP_OP_NONE) {
        ucs_queue_push(&iface->tx.pending_ops, &ep->tx.pending.queue);
        ep->tx.pending.ops |= UCT_UD_EP_OP_INPROGRESS;
    }
    ep->tx.pending.ops |= op;
}


static inline ucs_status_t uct_ud_iface_get_next_pending(uct_ud_iface_t *iface, uct_ud_ep_t **r_ep,
                                                         uct_ud_neth_t *neth,
                                                         uct_ud_send_skb_t **skb)
{
    uct_ud_ep_t *ep;
    ucs_queue_elem_t *elem;

    if (!uct_ud_iface_can_tx(iface)) {
        return UCS_ERR_NO_RESOURCE;
    }

    /* TODO: notify ucp that it can push more data */
    elem = ucs_queue_pull_non_empty(&iface->tx.pending_ops);
    ep = ucs_container_of(elem, uct_ud_ep_t, tx.pending.queue);
    if (ucs_likely(ep->tx.pending.ops & UCT_UD_EP_OP_ACK)) {
        *r_ep = ep;
        uct_ud_neth_ctl_ack(ep, neth);
         --iface->tx.available;
    } else if (ep->tx.pending.ops & UCT_UD_EP_OP_CREP) {
        *skb = uct_ud_ep_prepare_crep(ep);
        if (!*skb) {
            uct_ud_iface_queue_pending(iface, ep, UCT_UD_EP_OP_CREP);
            return UCS_ERR_NO_RESOURCE;
        }
        *r_ep = ep;
    } else if (ep->tx.pending.ops == UCT_UD_EP_OP_INPROGRESS) {
        /* someone already cleared this */
        return UCS_INPROGRESS;
    } else {
        /* TODO: support other ops */
        ucs_fatal("unsupported pending op mask: %x", ep->tx.pending.ops);
    }
    ep->tx.pending.ops = UCT_UD_EP_OP_NONE;
    return UCS_OK;
}

static inline uct_ud_send_skb_t *uct_ud_iface_get_tx_skb(uct_ud_iface_t *iface,
                                                         uct_ud_ep_t *ep)
{
    uct_ud_send_skb_t *skb;

    if (!uct_ud_iface_can_tx(iface)) {
        return NULL;
    }

    if (ep->tx.psn == ep->tx.max_psn) {
        ucs_trace_data("iface=%p ep=%p (%d->%d) tx window full (max_psn=%u)",
                       iface, ep, ep->ep_id, ep->dest_ep_id, (unsigned)ep->tx.max_psn);
        return NULL;
    }

    skb = ucs_mpool_get(iface->tx.mp);
    if (!skb) {
        ucs_trace_data("iface=%p out of tx skbs", iface);
        return NULL;
    }
    return skb;
}

static inline void uct_ud_iface_complete_tx(uct_ud_iface_t *iface,
                                           uct_ud_ep_t *ep,
                                           uct_ud_send_skb_t *skb,
                                           void *data, const void *buffer, unsigned length)
{
    ep->tx.psn++;
    --iface->tx.available;
    skb->len = length;
    memcpy(data, buffer, length);
    ucs_queue_push(&ep->tx.window, &skb->queue);
}


static inline void uct_ud_iface_complete_tx_skb(uct_ud_iface_t *iface,
                                                uct_ud_ep_t *ep,
                                                uct_ud_send_skb_t *skb)
{
    ep->tx.psn++;
    --iface->tx.available;
    ucs_queue_push(&ep->tx.window, &skb->queue);
}
