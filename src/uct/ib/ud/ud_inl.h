#include "ud_log.h"

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
    if (ep->tx.pending.ops & UCT_UD_EP_OP_CREP) {
        *skb = uct_ud_ep_prepare_crep(ep);
        if (!*skb) {
            uct_ud_iface_queue_pending(iface, ep, UCT_UD_EP_OP_CREP);
            return UCS_ERR_NO_RESOURCE;
        }
        *r_ep = ep;
    } else if (ucs_likely(ep->tx.pending.ops & UCT_UD_EP_OP_ACK)) {
        *r_ep = ep;
        uct_ud_neth_ctl_ack(ep, neth);
         --iface->tx.available;
         *skb = NULL;
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

/* TODO: relay on window check instead. max_psn = psn  */
static inline int uct_ud_ep_is_connected(uct_ud_ep_t *ep)
{
    if (ucs_unlikely(ep->dest_ep_id == UCT_UD_EP_NULL_ID)) {
        return 0;
    }
    return 1;
}

static inline uct_ud_send_skb_t *uct_ud_iface_get_tx_skb(uct_ud_iface_t *iface,
                                                         uct_ud_ep_t *ep)
{
    uct_ud_send_skb_t *skb;

    if (!uct_ud_iface_can_tx(iface)) {
        return NULL;
    }

    if (ucs_unlikely(ep->tx.psn == ep->tx.max_psn)) {
        ucs_trace_data("iface=%p ep=%p (%d->%d) tx window full (max_psn=%u)",
                       iface, ep, ep->ep_id, ep->dest_ep_id, (unsigned)ep->tx.max_psn);
        return NULL;
    }

    skb = iface->tx.skb;
    if (ucs_unlikely(!skb)) {
        ucs_trace_data("iface=%p out of tx skbs", iface);
        return NULL;
    }
    ucs_prefetch(skb->neth);
    return skb;
}

/* same as function above but also check that iface is connected */
static inline uct_ud_send_skb_t *uct_ud_iface_get_tx_skb2(uct_ud_iface_t *iface,
                                                          uct_ud_ep_t *ep)
{
    if (ucs_unlikely(!uct_ud_ep_is_connected(ep))) {
        return NULL;
    }

    return uct_ud_iface_get_tx_skb(iface, ep);
}

static inline void uct_ud_iface_complete_tx_inl_nolog(uct_ud_iface_t *iface,
                                                      uct_ud_ep_t *ep,
                                                      uct_ud_send_skb_t *skb,
                                                      void *data, const void *buffer, unsigned length)
{
    iface->tx.skb = ucs_mpool_get(iface->tx.mp);
    ep->tx.psn++;
    --iface->tx.available;
    skb->len += length;
    memcpy(data, buffer, length);
    ucs_queue_push(&ep->tx.window, &skb->queue);
}

#define uct_ud_iface_complete_tx_inl(iface, ep, skb, data, buffer, length) \
    uct_ud_iface_complete_tx_inl_nolog(iface, ep, skb, data, buffer, length); \
    uct_ud_ep_log_tx(ep, skb);

static inline void uct_ud_iface_complete_tx_skb_nolog(uct_ud_iface_t *iface,
                                                      uct_ud_ep_t *ep,
                                                      uct_ud_send_skb_t *skb)
{
    iface->tx.skb = ucs_mpool_get(iface->tx.mp);
    ep->tx.psn++;
    --iface->tx.available;
    ucs_queue_push(&ep->tx.window, &skb->queue);
}

#define uct_ud_iface_complete_tx_skb(iface, ep, skb) \
    uct_ud_iface_complete_tx_skb_nolog(iface, ep, skb); \
    uct_ud_ep_log_tx(ep, skb);


static inline void uct_ud_am_neth(uct_ud_neth_t *neth, uct_ud_ep_t *ep, uint8_t id)
{
    uct_ud_neth_init_data(ep, neth);
    uct_ud_neth_set_type_am(ep, neth, id);
    uct_ud_neth_ack_req(ep, neth);
}

static inline ucs_status_t uct_ud_am_common(uct_ud_iface_t *iface,
                                            uct_ud_ep_t *ep,
                                            uint8_t id,
                                            uct_ud_send_skb_t **skb_p)
{
    uct_ud_send_skb_t *skb;

    UCT_CHECK_AM_ID(id);

    skb = uct_ud_iface_get_tx_skb2(iface, ep);
    if (!skb) {
        return UCS_ERR_NO_RESOURCE;
    }
    VALGRIND_MAKE_MEM_DEFINED(skb, sizeof *skb);

    uct_ud_am_neth(skb->neth, ep, id);

    *skb_p = skb;
    return UCS_OK;
}

static inline void uct_ud_skb_bcopy(uct_ud_send_skb_t *skb, 
                                    uct_pack_callback_t pack_cb, 
                                    void *arg, size_t length)
{
    pack_cb((char *)(skb->neth+1), arg, length);
    skb->len = length + sizeof(uct_ud_neth_t);
}

