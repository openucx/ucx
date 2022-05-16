/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

static UCS_F_ALWAYS_INLINE void
uct_ud_ep_ctl_op_schedule(uct_ud_iface_t *iface, uct_ud_ep_t *ep)
{
    ucs_arbiter_group_push_elem(&ep->tx.pending.group,
                                &ep->tx.pending.elem);
    ucs_arbiter_group_schedule(&iface->tx.pending_q, &ep->tx.pending.group);
}

/**
 * schedule control operation.
 */
static UCS_F_ALWAYS_INLINE void
uct_ud_ep_ctl_op_add(uct_ud_iface_t *iface, uct_ud_ep_t *ep, int op)
{
    ep->tx.pending.ops |= op;
    uct_ud_ep_ctl_op_schedule(iface, ep);
}

static UCS_F_ALWAYS_INLINE void
uct_ud_ep_tx_stop(uct_ud_ep_t *ep)
{
    ep->tx.max_psn = ep->tx.psn;
}

/*
 * check iface resources:tx_queue and return
 * prefetched/cached skb
 *
 * NOTE: caller must not return skb to mpool until it is
 * removed from the cache
 * skb is removed from cache by
 *  uct_ud_iface_complete_tx_inl()
 *  uct_ud_iface_complete_tx_skb()
 *
 * In case of error flow caller must do nothing with the skb
 */
static UCS_F_ALWAYS_INLINE
uct_ud_send_skb_t *uct_ud_iface_get_tx_skb(uct_ud_iface_t *iface,
                                           uct_ud_ep_t *ep)
{
    uct_ud_send_skb_t *skb;

    if (ucs_unlikely(!uct_ud_iface_can_tx(iface))) {
        UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
        return NULL;
    }

    skb = iface->tx.skb;
    if (ucs_unlikely(skb == NULL)) {
        skb = ucs_mpool_get(&iface->tx.mp);
        if (skb == NULL) {
            ucs_trace_data("iface=%p out of tx skbs", iface);
            UCT_TL_IFACE_STAT_TX_NO_DESC(&iface->super.super);
            return NULL;
        }
        iface->tx.skb = skb;
    }
    VALGRIND_MAKE_MEM_DEFINED(&skb->lkey, sizeof(skb->lkey));
    skb->flags = 0;
    ucs_prefetch(skb->neth);
    return skb;
}

static UCS_F_ALWAYS_INLINE void
uct_ud_skb_release(uct_ud_send_skb_t *skb, int is_inline)
{
    ucs_assert(!(skb->flags & UCT_UD_SEND_SKB_FLAG_INVALID));
    skb->flags = UCT_UD_SEND_SKB_FLAG_INVALID;
    if (is_inline) {
        ucs_mpool_put_inline(skb);
    } else {
        ucs_mpool_put(skb);
    }
}

#if UCS_ENABLE_ASSERT
static UCS_F_ALWAYS_INLINE int uct_ud_ep_has_pending(uct_ud_ep_t *ep)
{
    return !ucs_arbiter_group_is_empty(&ep->tx.pending.group) &&
           !ucs_arbiter_elem_is_only(&ep->tx.pending.elem);
}
#endif

static UCS_F_ALWAYS_INLINE void uct_ud_ep_set_has_pending_flag(uct_ud_ep_t *ep)
{
    ep->flags |= UCT_UD_EP_FLAG_HAS_PENDING;
}

static UCS_F_ALWAYS_INLINE void uct_ud_ep_remove_has_pending_flag(uct_ud_ep_t *ep)
{
    ucs_assert(ep->flags & UCT_UD_EP_FLAG_HAS_PENDING);
    ep->flags &= ~UCT_UD_EP_FLAG_HAS_PENDING;
}

static UCS_F_ALWAYS_INLINE void uct_ud_ep_set_dest_ep_id(uct_ud_ep_t *ep,
                                                         uint32_t dest_id)
{
    ucs_assert(dest_id != UCT_UD_EP_NULL_ID);
    ep->dest_ep_id = dest_id;
    ep->flags     |= UCT_UD_EP_FLAG_CONNECTED;
}

/* same as above but also check ep resources: window&connection state */
static UCS_F_ALWAYS_INLINE uct_ud_send_skb_t *
uct_ud_ep_get_tx_skb(uct_ud_iface_t *iface, uct_ud_ep_t *ep)
{
    if (ucs_unlikely(!uct_ud_ep_is_connected_and_no_pending(ep) ||
                     uct_ud_ep_no_window(ep) ||
                     uct_ud_iface_has_pending_async_ev(iface))) {
        ucs_trace_poll("iface=%p ep=%p (%d->%d) no ep resources (psn=%u max_psn=%u)",
                       iface, ep, ep->ep_id, ep->dest_ep_id,
                       (unsigned)ep->tx.psn,
                       (unsigned)ep->tx.max_psn);
        UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
        return NULL;
    }

    return uct_ud_iface_get_tx_skb(iface, ep);
}

static UCS_F_ALWAYS_INLINE void
uct_ud_skb_set_zcopy_desc(uct_ud_send_skb_t *skb, const uct_iov_t *iov,
                          size_t iovcnt, uct_completion_t *comp)
{
    uct_ud_zcopy_desc_t *zdesc;
    size_t iov_it_length;
    uct_ud_iov_t *ud_iov;
    size_t iov_it;

    skb->flags        |= UCT_UD_SEND_SKB_FLAG_ZCOPY;
    zdesc              = uct_ud_zcopy_desc(skb);
    zdesc->iovcnt      = 0;
    for (iov_it = 0; iov_it < iovcnt; ++iov_it) {
        iov_it_length = uct_iov_get_length(iov + iov_it);
        if (iov_it_length == 0) {
            continue;
        }

        ucs_assert(iov_it_length <= UINT16_MAX);
        ud_iov         = &zdesc->iov[zdesc->iovcnt++];
        ud_iov->buffer = iov[iov_it].buffer;
        ud_iov->lkey   = uct_ib_memh_get_lkey(iov[iov_it].memh);
        ud_iov->length = iov_it_length;
    }
    if (comp != NULL) {
        skb->flags        |= UCT_UD_SEND_SKB_FLAG_COMP;
        zdesc->super.comp  = comp;
    }
}

static UCS_F_ALWAYS_INLINE void
uct_ud_iface_complete_tx_skb(uct_ud_iface_t *iface, uct_ud_ep_t *ep,
                             uct_ud_send_skb_t *skb)
{
    ucs_time_t now = uct_ud_iface_get_time(iface);
    iface->tx.skb  = ucs_mpool_get(&iface->tx.mp);
    ep->tx.psn++;

    if (ucs_queue_is_empty(&ep->tx.window)) {
        ep->tx.send_time = now;
    }

    ucs_queue_push(&ep->tx.window, &skb->queue);
    ep->tx.tick = iface->tx.tick;

    if (!iface->async.disable) {
        ucs_wtimer_add(&iface->tx.timer, &ep->timer,
                       now - ucs_twheel_get_time(&iface->tx.timer) + ep->tx.tick);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_ud_iface_complete_tx_inl(uct_ud_iface_t *iface, uct_ud_ep_t *ep,
                             uct_ud_send_skb_t *skb, void *data,
                             const void *buffer, unsigned length)
{
    skb->len += length;
    memcpy(data, buffer, length);
    uct_ud_iface_complete_tx_skb(iface, ep, skb);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_ud_am_skb_common(uct_ud_iface_t *iface, uct_ud_ep_t *ep, uint8_t id,
                     uct_ud_send_skb_t **skb_p)
{
    uct_ud_send_skb_t *skb;
    uct_ud_neth_t *neth;

    UCT_CHECK_AM_ID(id);

    skb = uct_ud_ep_get_tx_skb(iface, ep);
    if (!skb) {
        return UCS_ERR_NO_RESOURCE;
    }

    /* either we are executing pending operations, or there are no any pending
     * elements, or the only pending element is for sending control messages
     * (we don't care about reordering with respect to control messages)
     */
    ucs_assertv((ep->flags & UCT_UD_EP_FLAG_IN_PENDING) ||
                !uct_ud_ep_has_pending(ep),
                "out-of-order send detected for ep %p am %d ep_pending %d arbelem %p",
                ep, id, (ep->flags & UCT_UD_EP_FLAG_IN_PENDING),
                &ep->tx.pending.elem);

    neth = skb->neth;
    uct_ud_neth_init_data(ep, neth);
    uct_ud_neth_set_type_am(ep, neth, id);
    uct_ud_neth_ack_req(ep, neth);

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

static UCS_F_ALWAYS_INLINE void
uct_ud_iface_dispatch_comp(uct_ud_iface_t *iface, uct_completion_t *comp)
{
    /* Avoid reordering with pending queue - if we have any pending requests,
     * prevent send operations from the completion callback
     */
    uct_ud_iface_raise_pending_async_ev(iface);
    uct_invoke_completion(comp, UCS_OK);
}

static UCS_F_ALWAYS_INLINE void
uct_ud_iface_add_async_comp(uct_ud_iface_t *iface, uct_ud_ep_t *ep,
                            uct_ud_send_skb_t *skb, ucs_status_t status)
{
    uct_ud_comp_desc_t *cdesc = uct_ud_comp_desc(skb);

    cdesc->ep = ep;
    uct_completion_update_status(cdesc->comp, status);
    ucs_queue_push(&iface->tx.async_comp_q, &skb->queue);
}

static UCS_F_ALWAYS_INLINE void
uct_ud_iov_to_skb(uct_ud_send_skb_t *skb, const uct_iov_t *iov, size_t iovcnt)
{
    ucs_iov_iter_t iov_iter;

    ucs_iov_iter_init(&iov_iter);
    skb->len += uct_iov_to_buffer(iov, iovcnt, &iov_iter, skb->neth + 1,
                                  SIZE_MAX);
}
