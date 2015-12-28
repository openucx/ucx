#include "ud_log.h"

/**
 * scedule control operation. 
 */
static UCS_F_ALWAYS_INLINE void
uct_ud_ep_ctl_op_add(uct_ud_iface_t *iface, uct_ud_ep_t *ep, int op)
{
    ucs_arbiter_group_push_elem(&ep->tx.pending.group, 
                                &ep->tx.pending.elem);
    ep->tx.pending.ops |= op;
    ucs_arbiter_group_schedule(&iface->tx.pending_q, &ep->tx.pending.group);
}

/* check iface resources:tx_queue and skbs and return skb */
static UCS_F_ALWAYS_INLINE 
uct_ud_send_skb_t *uct_ud_iface_get_tx_skb(uct_ud_iface_t *iface,
                                           uct_ud_ep_t *ep)
{
    uct_ud_send_skb_t *skb;

    if (ucs_unlikely(!uct_ud_iface_can_tx(iface))) {
        UCT_TL_IFACE_STAT_TX_NO_RES(&iface->super.super);
        return NULL;
    }

    skb = iface->tx.skb;
    if (ucs_unlikely(skb == NULL)) {
        skb = ucs_mpool_get(&iface->tx.mp);
        if (skb == NULL) {
            ucs_trace_data("iface=%p out of tx skbs", iface);
            UCT_TL_IFACE_STAT_TX_NO_RES(&iface->super.super);
            return NULL;
        }
    }
    VALGRIND_MAKE_MEM_DEFINED(skb, sizeof *skb);
    ucs_prefetch(skb->neth);
    return skb;
}

/* same as above but also check ep resources: window&connection state */
static UCS_F_ALWAYS_INLINE uct_ud_send_skb_t *
uct_ud_ep_get_tx_skb(uct_ud_iface_t *iface, uct_ud_ep_t *ep)
{
    if (ucs_unlikely(!uct_ud_ep_is_connected(ep) || 
                     uct_ud_ep_no_window(ep))) {
        ucs_trace_data("iface=%p ep=%p (%d->%d) no ep resources (psn=%u max_psn=%u)",
                       iface, ep, ep->ep_id, ep->dest_ep_id,
                       (unsigned)ep->tx.psn,
                       (unsigned)ep->tx.max_psn);
        UCT_TL_IFACE_STAT_TX_NO_RES(&iface->super.super);
        return NULL;
    }

    return uct_ud_iface_get_tx_skb(iface, ep);
}

static inline ucs_time_t uct_ud_slow_tick()
{
    return ucs_time_from_msec(100);
}

static inline ucs_time_t uct_ud_fast_tick()
{
    return ucs_time_from_usec(1024);
}


static UCS_F_ALWAYS_INLINE void
uct_ud_iface_complete_tx_inl_nolog(uct_ud_iface_t *iface, uct_ud_ep_t *ep,
                                   uct_ud_send_skb_t *skb, void *data,
                                   const void *buffer, unsigned length)
{
    iface->tx.skb = ucs_mpool_get(&iface->tx.mp);
    ep->tx.psn++;
    skb->len += length;
    memcpy(data, buffer, length);
    ucs_queue_push(&ep->tx.window, &skb->queue);
    ucs_wtimer_add(&iface->async.slow_timer, &ep->slow_timer,
                   uct_ud_iface_get_async_time(iface) - 
                   ucs_twheel_get_time(&iface->async.slow_timer) +
                   uct_ud_slow_tick());
    ep->tx.send_time = uct_ud_iface_get_async_time(iface);
}

#define uct_ud_iface_complete_tx_inl(iface, ep, skb, data, buffer, length) \
    uct_ud_iface_complete_tx_inl_nolog(iface, ep, skb, data, buffer, length); \
    uct_ud_ep_log_tx(iface, ep, skb);


static UCS_F_ALWAYS_INLINE void 
uct_ud_iface_complete_tx_skb_nolog(uct_ud_iface_t *iface, uct_ud_ep_t *ep,
                                   uct_ud_send_skb_t *skb)
{
    iface->tx.skb = ucs_mpool_get(&iface->tx.mp);
    ep->tx.psn++;
    ucs_queue_push(&ep->tx.window, &skb->queue);
    ucs_wtimer_add(&iface->async.slow_timer, &ep->slow_timer,
                   uct_ud_iface_get_async_time(iface) - 
                   ucs_twheel_get_time(&iface->async.slow_timer) +
                   uct_ud_slow_tick());
    ep->tx.send_time = uct_ud_iface_get_async_time(iface);
}

#define uct_ud_iface_complete_tx_skb(iface, ep, skb) \
    uct_ud_iface_complete_tx_skb_nolog(iface, ep, skb); \
    uct_ud_ep_log_tx(iface, ep, skb);


static UCS_F_ALWAYS_INLINE void
uct_ud_am_set_neth(uct_ud_neth_t *neth, uct_ud_ep_t *ep, uint8_t id)
{
    uct_ud_neth_init_data(ep, neth);
    uct_ud_neth_set_type_am(ep, neth, id);
    uct_ud_neth_ack_req(ep, neth);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_ud_am_common(uct_ud_iface_t *iface, uct_ud_ep_t *ep, uint8_t id,
                 uct_ud_send_skb_t **skb_p)
{
    uct_ud_send_skb_t *skb;

    UCT_CHECK_AM_ID(id);

    skb = uct_ud_ep_get_tx_skb(iface, ep);
    if (!skb) {
        return UCS_ERR_NO_RESOURCE;
    }

    uct_ud_am_set_neth(skb->neth, ep, id);

    *skb_p = skb;
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE size_t
uct_ud_skb_bcopy(uct_ud_send_skb_t *skb, uct_pack_callback_t pack_cb, void *arg)
{
    size_t payload_len;

    payload_len = pack_cb(skb->neth + 1, arg);
    skb->len = sizeof(skb->neth[0]) + payload_len;
    return payload_len;
}

