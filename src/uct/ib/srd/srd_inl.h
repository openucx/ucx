/**
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */


#include <uct/ib/base/ib_log.h>


static UCS_F_ALWAYS_INLINE void
uct_srd_ep_ctl_op_schedule(uct_srd_iface_t *iface, uct_srd_ep_t *ep)
{
    ucs_arbiter_group_push_elem(&ep->tx.pending.group,
                                &ep->tx.pending.elem);
    ucs_arbiter_group_schedule(&iface->tx.pending_q, &ep->tx.pending.group);
}

/**
 * schedule control operation.
 */
static UCS_F_ALWAYS_INLINE void
uct_srd_ep_ctl_op_add(uct_srd_iface_t *iface, uct_srd_ep_t *ep, int op)
{
    ep->tx.pending.ops |= op;
    uct_srd_ep_ctl_op_schedule(iface, ep);
}

#if UCS_ENABLE_ASSERT
static UCS_F_ALWAYS_INLINE int uct_srd_ep_has_pending(uct_srd_ep_t *ep)
{
    return !ucs_arbiter_group_is_empty(&ep->tx.pending.group) &&
           !ucs_arbiter_elem_is_only(&ep->tx.pending.elem);
}
#endif

static UCS_F_ALWAYS_INLINE void uct_srd_ep_set_has_pending_flag(uct_srd_ep_t *ep)
{
    ep->flags |= UCT_SRD_EP_FLAG_HAS_PENDING;
}

static UCS_F_ALWAYS_INLINE void uct_srd_ep_remove_has_pending_flag(uct_srd_ep_t *ep)
{
    ucs_assert(ep->flags & UCT_SRD_EP_FLAG_HAS_PENDING);
    ep->flags &= ~UCT_SRD_EP_FLAG_HAS_PENDING;
}

static UCS_F_ALWAYS_INLINE void uct_srd_ep_set_dest_ep_id(uct_srd_ep_t *ep,
                                                          uint32_t dest_id)
{
    ucs_assert(dest_id != UCT_SRD_EP_NULL_ID);
    ep->dest_ep_id = dest_id;
    ep->flags     |= UCT_SRD_EP_FLAG_CONNECTED;
}

static UCS_F_ALWAYS_INLINE uct_srd_send_desc_t *
uct_srd_ep_get_send_desc(uct_srd_iface_t *iface, uct_srd_ep_t *ep)
{
    uct_srd_send_desc_t *desc = uct_srd_iface_get_send_desc(iface);

    if (ucs_unlikely(!uct_srd_ep_is_connected_and_no_pending(ep) || !desc)) {
        ucs_trace_poll("iface=%p ep=%p (%d->%d) no ep resources (psn=%u)",
                       iface, ep, ep->ep_id, ep->dest_ep_id, (unsigned)ep->tx.psn);
        UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
        return NULL;
    }

    desc->super.ep = ep;
    return desc;
}

static UCS_F_ALWAYS_INLINE uct_srd_send_op_t *
uct_srd_ep_get_send_op(uct_srd_iface_t *iface, uct_srd_ep_t *ep)
{
    uct_srd_send_op_t *send_op = uct_srd_iface_get_send_op(iface);

    if (ucs_unlikely(!uct_srd_ep_is_connected_and_no_pending(ep) || !send_op)) {
        ucs_trace_poll("iface=%p ep=%p (%d->%d) no ep resources (psn=%u)",
                       iface, ep, ep->ep_id, ep->dest_ep_id, (unsigned)ep->tx.psn);
        UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
        return NULL;
    }

    send_op->ep = ep;
    return send_op;
}

static UCS_F_ALWAYS_INLINE void
uct_srd_zcopy_op_set_comp(uct_srd_send_op_t *send_op, uct_completion_t *comp)
{
    if (comp == NULL) {
        send_op->comp_handler = uct_srd_iface_send_op_release;
    } else {
        send_op->comp_handler = uct_srd_iface_send_op_ucomp_release;
        send_op->user_comp = comp;
    }
}

static UCS_F_ALWAYS_INLINE void
uct_srd_post_send(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                  struct ibv_send_wr *wr, unsigned send_flags,
                  unsigned max_log_sge)
{
    struct ibv_send_wr *bad_wr;
    int ret;

    wr->wr.ud.remote_qpn = ep->peer_address.dest_qpn;
    wr->wr.ud.ah         = ep->peer_address.ah;
    wr->send_flags       = send_flags;

    ret = ibv_post_send(iface->qp, wr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_send() returned %d (%m)", ret);
    }

    iface->tx.available--;
    ep->tx.psn++;

    uct_ib_log_post_send(&iface->super, iface->qp, wr, max_log_sge,
                         uct_srd_dump_packet);
}

static UCS_F_ALWAYS_INLINE void
uct_srd_ep_tx_desc(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                   uct_srd_send_desc_t *desc, unsigned send_flags,
                   unsigned max_log_sge)
{
    iface->tx.sge[0].lkey   = desc->lkey;
    iface->tx.sge[0].length = desc->super.len;
    iface->tx.sge[0].addr   = (uintptr_t)(desc + 1);
    iface->tx.wr_desc.wr_id = (uintptr_t)&desc->super;

    uct_srd_post_send(iface, ep, &iface->tx.wr_desc, send_flags, max_log_sge);
    ucs_queue_push(&ep->tx.outstanding_q, &desc->super.out_queue);
}

static UCS_F_ALWAYS_INLINE void
uct_srd_neth_set_type_am(const uct_srd_ep_t *ep,
                         uct_srd_neth_t *neth, uint8_t id)
{
    neth->packet_type = (id << UCT_SRD_PACKET_AM_ID_SHIFT) |
                        ep->dest_ep_id |
                        UCT_SRD_PACKET_FLAG_AM;
}

static UCS_F_ALWAYS_INLINE void
uct_srd_neth_set_psn(const uct_srd_ep_t *ep, uct_srd_neth_t *neth)
{
    neth->psn = ep->tx.psn;
}

static UCS_F_ALWAYS_INLINE void
uct_srd_am_common(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                  uint8_t id, uct_srd_neth_t *neth)
{
    /* either we are executing pending operations,
     * or there are no any pending elements. */
    UCT_SRD_EP_ASSERT_PENDING(ep);

    uct_srd_neth_set_type_am(ep, neth, id);
    uct_srd_neth_set_psn(ep, neth);
}
