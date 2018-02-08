/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ud_ep.h"
#include "ud_iface.h"
#include "ud_inl.h"
#include "ud_def.h"

#include <uct/ib/base/ib_verbs.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>

static void uct_ud_peer_name(uct_ud_peer_name_t *peer)
{
    gethostname(peer->name, sizeof(peer->name));
    peer->pid = getpid();
}

static void uct_ud_ep_set_state(uct_ud_ep_t *ep, uint32_t state)
{
    ep->flags |= state;
}

#if ENABLE_DEBUG_DATA
static void uct_ud_peer_copy(uct_ud_peer_name_t *dst, uct_ud_peer_name_t *src)
{
    memcpy(dst, src, sizeof(*src));
}

#else
#define  uct_ud_peer_copy(dst, src)
#endif


static void uct_ud_ep_resend_start(uct_ud_iface_t *iface, uct_ud_ep_t *ep)
{
    ep->resend.max_psn   = ep->tx.psn - 1;
    ep->resend.psn       = ep->tx.acked_psn + 1;
    ep->resend.pos       = ucs_queue_iter_begin(&ep->tx.window);
    uct_ud_ep_ctl_op_add(iface, ep, UCT_UD_EP_OP_RESEND);
}


static void uct_ud_ep_resend_ack(uct_ud_iface_t *iface, uct_ud_ep_t *ep)
{
    if (UCT_UD_PSN_COMPARE(ep->tx.acked_psn, <, ep->resend.max_psn)) {
        /* new ack arrived that acked something in our resend window. */
        if (UCT_UD_PSN_COMPARE(ep->resend.psn, <=, ep->tx.acked_psn)) {
            ucs_debug("ep(%p): ack received during resend resend.psn=%d tx.acked_psn=%d",
                      ep, ep->resend.psn, ep->tx.acked_psn);
            ep->resend.pos = ucs_queue_iter_begin(&ep->tx.window);
            ep->resend.psn = ep->tx.acked_psn + 1;
        }
        uct_ud_ep_ctl_op_add(iface, ep, UCT_UD_EP_OP_RESEND);
    } else {
        /* everything in resend window was acked - no need to resend anymore */
        ep->resend.psn = ep->resend.max_psn + 1;
        uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_RESEND);
    }
}


static void uct_ud_ep_ca_drop(uct_ud_ep_t *ep)
{
    ucs_debug("ep: %p ca drop@cwnd = %d in flight: %d",
              ep, ep->ca.cwnd, (int)ep->tx.psn-(int)ep->tx.acked_psn-1);
    ep->ca.cwnd /= UCT_UD_CA_MD_FACTOR;
    if (ep->ca.cwnd < UCT_UD_CA_MIN_WINDOW) {
        ep->ca.cwnd = UCT_UD_CA_MIN_WINDOW;
    }
    ep->tx.max_psn    = ep->tx.acked_psn + ep->ca.cwnd;
    if (UCT_UD_PSN_COMPARE(ep->tx.max_psn, >, ep->tx.psn)) {
        /* do not send more until we get acks going */
        uct_ud_ep_tx_stop(ep);
    }
}

static UCS_F_ALWAYS_INLINE void uct_ud_ep_ca_ack(uct_ud_ep_t *ep)
{
    if (ep->ca.cwnd < UCT_UD_CA_MAX_WINDOW) {
        ep->ca.cwnd += UCT_UD_CA_AI_VALUE;
    }
    ep->tx.max_psn = ep->tx.acked_psn + ep->ca.cwnd;
}


static void uct_ud_ep_reset(uct_ud_ep_t *ep)
{
    ep->tx.psn         = UCT_UD_INITIAL_PSN;
    ep->ca.cwnd        = UCT_UD_CA_MIN_WINDOW;
    ep->tx.max_psn     = ep->tx.psn + ep->ca.cwnd;
    ep->tx.acked_psn   = UCT_UD_INITIAL_PSN - 1;
    ep->tx.pending.ops = UCT_UD_EP_OP_NONE;
    ucs_queue_head_init(&ep->tx.window);

    ep->resend.pos       = ucs_queue_iter_begin(&ep->tx.window);
    ep->resend.psn       = ep->tx.psn;
    ep->resend.max_psn   = ep->tx.acked_psn;

    ep->rx.acked_psn = UCT_UD_INITIAL_PSN - 1;
    ucs_frag_list_init(ep->tx.psn-1, &ep->rx.ooo_pkts, 0 /*TODO: ooo support */
                       UCS_STATS_ARG(ep->super.stats));
}

static void uct_ud_ep_slow_timer(ucs_wtimer_t *self)
{
    uct_ud_ep_t *ep = ucs_container_of(self, uct_ud_ep_t, slow_timer);
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                           uct_ud_iface_t);
    ucs_time_t now;
    ucs_time_t diff;

    UCT_UD_EP_HOOK_CALL_TIMER(ep);
    now = ucs_twheel_get_time(&iface->async.slow_timer);
    diff = now - ep->tx.send_time;

    if (ucs_queue_is_empty(&ep->tx.window)) {
        return;
    }

    if (diff > iface->config.peer_timeout) {
        iface->super.ops->handle_failure(&iface->super, ep,
                                         UCS_ERR_ENDPOINT_TIMEOUT);
        return;
    } else if (diff > 3*iface->async.slow_tick) {
        ucs_trace("scheduling resend now: %lu send_time: %lu diff: %lu tick: %lu",
                  now, ep->tx.send_time, now - ep->tx.send_time,
                  ep->tx.slow_tick);
        uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_ACK_REQ);
        uct_ud_ep_ca_drop(ep);
        uct_ud_ep_resend_start(iface, ep);
    } else if ((diff > iface->async.slow_tick) && uct_ud_ep_is_connected(ep)) {
        /* It is possible that the sender is slow.
         * Try to flush the window twice before going into
         * full resend mode.
         */
        uct_ud_ep_ctl_op_add(iface, ep, UCT_UD_EP_OP_ACK_REQ);
    }

    /* Cool down the timer on rescheduling/resending */
    ep->tx.slow_tick *= iface->config.slow_timer_backoff;
    ep->tx.slow_tick = ucs_min(ep->tx.slow_tick, iface->config.peer_timeout/3);
    ucs_wtimer_add(&iface->async.slow_timer, &ep->slow_timer, ep->tx.slow_tick);
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
    self->tx.slow_tick = iface->async.slow_tick;
    ucs_wtimer_init(&self->slow_timer, uct_ud_ep_slow_timer);
    ucs_arbiter_group_init(&self->tx.pending.group);
    ucs_arbiter_elem_init(&self->tx.pending.elem);

    self->path_bits = iface->super.path_bits[0]; /* TODO multi-rail */

    UCT_UD_EP_HOOK_INIT(self);
    ucs_debug("created ep ep=%p iface=%p id=%d src_path_bits=%d",
              self, iface, self->ep_id, self->path_bits);
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
    ucs_frag_list_cleanup(&self->rx.ooo_pkts);

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

ucs_status_t uct_ud_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr)
{
    uct_ud_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_ep_t);
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    uct_ud_ep_addr_t *ep_addr = (uct_ud_ep_addr_t *)addr;

    uct_ib_pack_uint24(ep_addr->iface_addr.qp_num, iface->qp->qp_num);
    uct_ib_pack_uint24(ep_addr->ep_id, ep->ep_id);
    return UCS_OK;
}

static ucs_status_t uct_ud_ep_connect_to_iface(uct_ud_ep_t *ep,
                                               const uct_ib_address_t *ib_addr,
                                               const uct_ud_iface_addr_t *if_addr)
{
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super);
    char buf[128];

    ucs_frag_list_cleanup(&ep->rx.ooo_pkts);
    uct_ud_ep_reset(ep);

    ucs_debug(UCT_IB_IFACE_FMT" lid %d qpn 0x%x epid %u ep %p connected to "
              "IFACE %s qpn 0x%x", UCT_IB_IFACE_ARG(&iface->super),
              dev->port_attr[iface->super.config.port_num - dev->first_port].lid,
              iface->qp->qp_num, ep->ep_id, ep,
              uct_ib_address_str(ib_addr, buf, sizeof(buf)),
              uct_ib_unpack_uint24(if_addr->qp_num));

    return UCS_OK;
}

static ucs_status_t uct_ud_ep_disconnect_from_iface(uct_ep_h tl_ep)
{
    uct_ud_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_ep_t);

    ucs_frag_list_cleanup(&ep->rx.ooo_pkts);
    uct_ud_ep_reset(ep);
    ep->dest_ep_id = UCT_UD_EP_NULL_ID;

    return UCS_OK;
}

ucs_status_t uct_ud_ep_create_connected_common(uct_ud_iface_t *iface,
                                               const uct_ib_address_t *ib_addr,
                                               const uct_ud_iface_addr_t *if_addr,
                                               uct_ud_ep_t **new_ep_p,
                                               uct_ud_send_skb_t **skb_p)
{
    ucs_status_t status;
    uct_ud_ep_t *ep;
    uct_ep_h new_ep_h;

    ep = uct_ud_iface_cep_lookup(iface, ib_addr, if_addr, UCT_UD_EP_CONN_ID_MAX);
    if (ep) {
        uct_ud_ep_set_state(ep, UCT_UD_EP_FLAG_CREQ_NOTSENT);
        *new_ep_p = ep;
        *skb_p    = NULL;
        return UCS_ERR_ALREADY_EXISTS;
    }

    status = uct_ep_create(&iface->super.super.super, &new_ep_h);
    if (status != UCS_OK) {
        return status;
    }
    ep = ucs_derived_of(new_ep_h, uct_ud_ep_t);

    status = uct_ud_ep_connect_to_iface(ep, ib_addr, if_addr);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ud_iface_cep_insert(iface, ib_addr, if_addr, ep, UCT_UD_EP_CONN_ID_MAX);
    if (status != UCS_OK) {
        goto err_cep_insert;
    }

    *skb_p = uct_ud_ep_prepare_creq(ep);
    if (!*skb_p) {
        status = UCS_ERR_NO_RESOURCE;
        uct_ud_ep_ctl_op_add(iface, ep, UCT_UD_EP_OP_CREQ);
    }

    *new_ep_p = ep;
    return status;

err_cep_insert:
    uct_ud_ep_disconnect_from_iface(&ep->super.super);
    return status;
}

void uct_ud_ep_destroy_connected(uct_ud_ep_t *ep,
                                 const uct_ib_address_t *ib_addr,
                                 const uct_ud_iface_addr_t *if_addr)
{
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    uct_ud_iface_cep_rollback(iface, ib_addr, if_addr, ep);
    uct_ud_ep_disconnect_from_iface(&ep->super.super);
}

ucs_status_t uct_ud_ep_connect_to_ep(uct_ud_ep_t *ep,
                                     const uct_ib_address_t *ib_addr,
                                     const uct_ud_ep_addr_t *ep_addr)
{
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super);
    char buf[128];

    ucs_assert_always(ep->dest_ep_id == UCT_UD_EP_NULL_ID);
    ucs_trace_func("");

    ep->dest_ep_id = uct_ib_unpack_uint24(ep_addr->ep_id);

    ucs_frag_list_cleanup(&ep->rx.ooo_pkts);
    uct_ud_ep_reset(ep);

    ucs_debug(UCT_IB_IFACE_FMT" slid %d qpn 0x%x epid %u connected to %s qpn 0x%x "
              "epid %u", UCT_IB_IFACE_ARG(&iface->super),
              dev->port_attr[iface->super.config.port_num - dev->first_port].lid,
              iface->qp->qp_num, ep->ep_id,
              uct_ib_address_str(ib_addr, buf, sizeof(buf)),
              uct_ib_unpack_uint24(ep_addr->iface_addr.qp_num), ep->dest_ep_id);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
uct_ud_iface_add_async_comp(uct_ud_iface_t *iface, uct_ud_ep_t *ep,
                            uct_ud_send_skb_t *skb, ucs_status_t status)
{
    uct_ud_comp_desc_t *cdesc;

    skb->status = status;
    if (status != UCS_OK) {
        if (!(skb->flags & UCT_UD_SEND_SKB_FLAG_COMP)) {
            skb->len = 0;
        }

        if (status == UCS_ERR_ENDPOINT_TIMEOUT) {
            skb->flags |= UCT_UD_SEND_SKB_FLAG_ERR;
            ++ep->tx.err_skb_count;
        } else if (status == UCS_ERR_CANCELED) {
            skb->flags |= UCT_UD_SEND_SKB_FLAG_CANCEL;
        }
    }

    cdesc = uct_ud_comp_desc(skb);

    /* don't call user completion from async context. instead, put
     * it on a queue which will be progressed from main thread.
     */
    ucs_queue_push(&iface->tx.async_comp_q, &skb->queue);
    cdesc->ep  = ep;
    ep->flags |= UCT_UD_EP_FLAG_ASYNC_COMPS;
}

static UCS_F_ALWAYS_INLINE void
uct_ud_ep_process_ack(uct_ud_iface_t *iface, uct_ud_ep_t *ep,
                      uct_ud_psn_t ack_psn, int is_async)
{
    uct_ud_send_skb_t *skb;
    if (ucs_unlikely(UCT_UD_PSN_COMPARE(ack_psn, <=, ep->tx.acked_psn))) {
        return;
    }

    ep->tx.acked_psn = ack_psn;

    /* Release acknowledged skb's */
    ucs_queue_for_each_extract(skb, &ep->tx.window, queue,
                               UCT_UD_PSN_COMPARE(skb->neth->psn, <=, ack_psn)) {
        if (ucs_unlikely(skb->flags & UCT_UD_SEND_SKB_FLAG_COMP)) {
            if (ucs_unlikely(is_async)) {
                uct_ud_iface_add_async_comp(iface, ep, skb, UCS_OK);
                continue;
            }

            uct_invoke_completion(uct_ud_comp_desc(skb)->comp, UCS_OK);
        }

        skb->flags = 0; /* reset also ACK_REQ flag */
        ucs_mpool_put(skb);
    }

    uct_ud_ep_ca_ack(ep);

    if (ucs_unlikely(UCT_UD_PSN_COMPARE(ep->resend.psn, <=, ep->resend.max_psn))) {
        uct_ud_ep_resend_ack(iface, ep);
    }

    ucs_arbiter_group_schedule(&iface->tx.pending_q, &ep->tx.pending.group);

    ep->tx.slow_tick = iface->async.slow_tick;
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
    status = uct_ep_create(iface_h, &ep_h);
    ucs_assert_always(status == UCS_OK);
    ep = ucs_derived_of(ep_h, uct_ud_ep_t);

    status = uct_ep_connect_to_ep(ep_h, (void*)uct_ud_creq_ib_addr(ctl),
                                  (void*)&ctl->conn_req.ep_addr);
    ucs_assert_always(status == UCS_OK);

    status = uct_ud_iface_cep_insert(iface, uct_ud_creq_ib_addr(ctl),
                                     &ctl->conn_req.ep_addr.iface_addr,
                                     ep, ctl->conn_req.conn_id);
    ucs_assert_always(status == UCS_OK);
    return ep;
}

static void uct_ud_ep_rx_creq(uct_ud_iface_t *iface, uct_ud_neth_t *neth)
{
    uct_ud_ep_t *ep;
    uct_ud_ctl_hdr_t *ctl = (uct_ud_ctl_hdr_t *)(neth + 1);

    ucs_assert_always(ctl->type == UCT_UD_PACKET_CREQ);

    ep = uct_ud_iface_cep_lookup(iface, uct_ud_creq_ib_addr(ctl),
                                 &ctl->conn_req.ep_addr.iface_addr,
                                 ctl->conn_req.conn_id);
    if (!ep) {
        ep = uct_ud_ep_create_passive(iface, ctl);
        ucs_assert_always(ep != NULL);
        ep->rx.ooo_pkts.head_sn = neth->psn;
        uct_ud_peer_copy(&ep->peer, (void*)&ctl->peer);
        uct_ud_ep_ctl_op_add(iface, ep, UCT_UD_EP_OP_CREP);
        uct_ud_ep_set_state(ep, UCT_UD_EP_FLAG_PRIVATE);
    } else {
        if (ep->dest_ep_id == UCT_UD_EP_NULL_ID) {
            /* simultanuous CREQ */
            ep->dest_ep_id = uct_ib_unpack_uint24(ctl->conn_req.ep_addr.ep_id);
            ep->rx.ooo_pkts.head_sn = neth->psn;
            uct_ud_peer_copy(&ep->peer, (void*)&ctl->peer);
            ucs_debug("simultanuous CREQ ep=%p"
                      "(iface=%p conn_id=%d ep_id=%d, dest_ep_id=%d rx_psn=%u)",
                      ep, iface, ep->conn_id, ep->ep_id,
                      ep->dest_ep_id, ep->rx.ooo_pkts.head_sn);
            if (UCT_UD_PSN_COMPARE(ep->tx.psn, >, UCT_UD_INITIAL_PSN)) {
                /* our own creq was sent, treat incoming creq as ack and remove our own
                 * from tx window
                 */
                uct_ud_ep_process_ack(iface, ep, UCT_UD_INITIAL_PSN, 0);
            }
            uct_ud_ep_ctl_op_add(iface, ep, UCT_UD_EP_OP_CREP);
        }
    }

    ucs_assert_always(ctl->conn_req.conn_id == ep->conn_id);
    ucs_assert_always(uct_ib_unpack_uint24(ctl->conn_req.ep_addr.ep_id) == ep->dest_ep_id);
    /* creq must always have same psn */
    ucs_assert_always(ep->rx.ooo_pkts.head_sn == neth->psn);
    /* scedule connection reply op */
    UCT_UD_EP_HOOK_CALL_RX(ep, neth, sizeof(*neth) + sizeof(*ctl));
    if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_CREQ)) {
        uct_ud_ep_set_state(ep, UCT_UD_EP_FLAG_CREQ_NOTSENT);
    }
    uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_CREQ);
    uct_ud_ep_set_state(ep, UCT_UD_EP_FLAG_CREQ_RCVD);
}

static void uct_ud_ep_rx_ctl(uct_ud_iface_t *iface, uct_ud_ep_t *ep,
                             uct_ud_neth_t *neth, uct_ud_recv_skb_t *skb)
{
    uct_ud_ctl_hdr_t *ctl = (uct_ud_ctl_hdr_t*)(neth + 1);

    ucs_trace_func("");
    ucs_assert_always(ctl->type == UCT_UD_PACKET_CREP);
    ucs_assert_always(ep->dest_ep_id == UCT_UD_EP_NULL_ID ||
                      ep->dest_ep_id == ctl->conn_rep.src_ep_id);

    /* Discard duplicate CREP */
    if (UCT_UD_PSN_COMPARE(neth->psn, <, ep->rx.ooo_pkts.head_sn)) {
        return;
    }

    ep->rx.ooo_pkts.head_sn = neth->psn;
    ep->dest_ep_id = ctl->conn_rep.src_ep_id;
    ucs_arbiter_group_schedule(&iface->tx.pending_q, &ep->tx.pending.group);
    uct_ud_peer_copy(&ep->peer, (void*)&ctl->peer);
    uct_ud_ep_set_state(ep, UCT_UD_EP_FLAG_CREP_RCVD);
}

uct_ud_send_skb_t *uct_ud_ep_prepare_creq(uct_ud_ep_t *ep)
{
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    uct_ud_ctl_hdr_t *creq;
    uct_ud_send_skb_t *skb;
    uct_ud_neth_t *neth;
    ucs_status_t status;

    ucs_assert_always(ep->dest_ep_id == UCT_UD_EP_NULL_ID);
    ucs_assert_always(ep->ep_id != UCT_UD_EP_NULL_ID);

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
    creq->conn_req.conn_id        = ep->conn_id;

    status = uct_ud_ep_get_address(&ep->super.super,
                                   (void*)&creq->conn_req.ep_addr);
    if (status != UCS_OK) {
        return NULL;
    }

    status = uct_ib_iface_get_device_address(&iface->super.super.super,
                                             (uct_device_addr_t*)uct_ud_creq_ib_addr(creq));
    if (status != UCS_OK) {
        return NULL;
    }

    uct_ud_peer_name((void*)&creq->peer);

    skb->len = sizeof(*neth) + sizeof(*creq) + iface->super.addr_size;
    return skb;
}

void uct_ud_ep_process_rx(uct_ud_iface_t *iface, uct_ud_neth_t *neth, unsigned byte_len,
                          uct_ud_recv_skb_t *skb, int is_async)
{
    uint32_t dest_id;
    uint32_t is_am, am_id;
    uct_ud_ep_t *ep = 0; /* todo: check why gcc complaints about uninitialized var */
    ucs_frag_list_ooo_type_t ooo_type;

    UCT_UD_IFACE_HOOK_CALL_RX(iface, neth, byte_len);

    dest_id = uct_ud_neth_get_dest_id(neth);
    am_id   = uct_ud_neth_get_am_id(neth);
    is_am   = neth->packet_type & UCT_UD_PACKET_FLAG_AM;

    if (ucs_unlikely(dest_id == UCT_UD_EP_NULL_ID)) {
        /* must be connection request packet */
        uct_ud_ep_rx_creq(iface, neth);
        goto out;
    } else if (ucs_unlikely(!ucs_ptr_array_lookup(&iface->eps, dest_id, ep) ||
               ep->ep_id != dest_id))
    {
        /* Drop the packet because it is
         * allowed to do disconnect without flush/barrier. So it
         * is possible to get packet for the ep that has been destroyed
         */
        ucs_trace("RX: failed to find ep %d, dropping packet", dest_id);
        goto out;
    }

    ucs_assert(ep->ep_id != UCT_UD_EP_NULL_ID);
    UCT_UD_EP_HOOK_CALL_RX(ep, neth, byte_len);

    uct_ud_ep_process_ack(iface, ep, neth->ack_psn, is_async);

    if (ucs_unlikely(neth->packet_type & UCT_UD_PACKET_FLAG_ACK_REQ)) {
        uct_ud_ep_ctl_op_add(iface, ep, UCT_UD_EP_OP_ACK);
        ucs_trace_data("ACK_REQ - schedule ack, head_sn=%d sn=%d",
                       ep->rx.ooo_pkts.head_sn, neth->psn);
    }

    if (ucs_unlikely(!is_am)) {
        if ((size_t)byte_len == sizeof(*neth)) {
            goto out;
        }
        if (neth->packet_type & UCT_UD_PACKET_FLAG_CTL) {
            uct_ud_ep_rx_ctl(iface, ep, neth, skb);
            goto out;
        }
    }

    ooo_type = ucs_frag_list_insert(&ep->rx.ooo_pkts, &skb->u.ooo.elem, neth->psn);
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

    if (ucs_unlikely(is_async &&
                     (iface->super.super.am[am_id].flags & UCT_CB_FLAG_SYNC))) {
        skb->u.am.len = byte_len - sizeof(*neth);
        ucs_queue_push(&iface->rx.pending_q, &skb->u.am.queue);
    } else {
        uct_ib_iface_invoke_am_desc(&iface->super, am_id, neth + 1,
                                    byte_len - sizeof(*neth), &skb->super);
    }
    return;

out:
    ucs_mpool_put(skb);
}

ucs_status_t uct_ud_ep_flush_nolock(uct_ud_iface_t *iface, uct_ud_ep_t *ep,
                                    uct_completion_t *comp)
{
    uct_ud_send_skb_t *skb;
    uct_ud_psn_t psn;

    if (ucs_unlikely(!uct_ud_ep_is_connected(ep))) {
        /* check for CREQ either being scheduled or sent and waiting for CREP ack */
        if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_CREQ) ||
            !ucs_queue_is_empty(&ep->tx.window))
        {
            return UCS_ERR_NO_RESOURCE; /* connection in progress */
        }

        return UCS_OK; /* Nothing was ever sent */
    }

    if (!uct_ud_iface_can_tx(iface) || !uct_ud_iface_has_skbs(iface) ||
        uct_ud_ep_no_window(ep))
    {
        /* iface/ep has no resources, prevent reordering with possible pending
         * operations by not starting the flush.
         */
        return UCS_ERR_NO_RESOURCE;
    }

    if (ucs_queue_is_empty(&ep->tx.window)) {
        uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_ACK_REQ);

        /* Check if have pending async completions for this ep,
         *  if not - all was acknowledged, nothing is pending - return OK
         *  if yes - continue to add
         *  */
        if (!(ep->flags & UCT_UD_EP_FLAG_ASYNC_COMPS)) {
            return UCS_OK;
        }

        /*
         * If we have pending async completion, and the user requested a callback,
         * add a new async completion in the queue.
         */
        if (comp != NULL) {
            skb = ucs_mpool_get(&iface->tx.mp);
            if (skb == NULL) {
                return UCS_ERR_NO_RESOURCE;
            }

            skb->flags                  = UCT_UD_SEND_SKB_FLAG_COMP;
            skb->len                    = 0;
            uct_ud_comp_desc(skb)->comp = comp;
            uct_ud_comp_desc(skb)->ep   = ep;
            ucs_queue_push(&iface->tx.async_comp_q, &skb->queue);
        }
    } else {
        skb = ucs_queue_tail_elem_non_empty(&ep->tx.window, uct_ud_send_skb_t, queue);
        psn = skb->neth->psn;
        if (!(skb->flags & UCT_UD_SEND_SKB_FLAG_ACK_REQ)) {
            /* If we didn't ask for ACK on last skb, schedule an ACK message.
             * Also, prevent from sending more control messages like this after
             * first time by turning on the flag on the last skb.
             */
            uct_ud_ep_ctl_op_add_safe(iface, ep, UCT_UD_SEND_SKB_FLAG_ACK_REQ);
            skb->flags |= UCT_UD_SEND_SKB_FLAG_ACK_REQ;
        }

        /* If the user requested a callback, add a dummy skb to the window which
         * will be released when the current sequence number is acknowledged.
         */
        if (comp != NULL) {
            skb = ucs_mpool_get(&iface->tx.mp);
            if (skb == NULL) {
                return UCS_ERR_NO_RESOURCE;
            }

            /* Add dummy skb to the window, which would call user completion
             * callback when getting ACK.
             */
            skb->flags                  = UCT_UD_SEND_SKB_FLAG_COMP;
            skb->len                    = sizeof(skb->neth[0]);
            skb->neth->packet_type      = 0;
            skb->neth->psn              = psn;
            uct_ud_comp_desc(skb)->comp = comp;
            ucs_assert(psn == (uct_ud_psn_t)(ep->tx.psn - 1));

            uct_ud_neth_set_dest_id(skb->neth, UCT_UD_EP_NULL_ID);
            ucs_queue_push(&ep->tx.window, &skb->queue);
            ucs_trace_data("added dummy flush skb %p psn %d user_comp %p", skb,
                           skb->neth->psn, comp);
        }
    }

    return UCS_INPROGRESS;
}

void uct_ud_tx_wnd_purge_outstanding(uct_ud_iface_t *iface, uct_ud_ep_t *ud_ep,
                                     ucs_status_t status)
{
    uct_ud_send_skb_t  *skb;

    uct_ud_ep_tx_stop(ud_ep);

    ucs_queue_for_each_extract(skb, &ud_ep->tx.window, queue, 1) {
        uct_ud_iface_add_async_comp(iface, ud_ep, skb, status);
    }
}

ucs_status_t uct_ud_ep_flush(uct_ep_h ep_h, unsigned flags,
                             uct_completion_t *comp)
{
    ucs_status_t status;
    uct_ud_ep_t *ep = ucs_derived_of(ep_h, uct_ud_ep_t);
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                           uct_ud_iface_t);

    uct_ud_enter(iface);

    if (ucs_unlikely(flags & UCT_FLUSH_FLAG_CANCEL)) {
        uct_ud_tx_wnd_purge_outstanding(iface, ep, UCS_ERR_CANCELED);
        uct_ud_iface_dispatch_zcopy_comps(iface);
        uct_ep_pending_purge(ep_h, NULL, 0);
        /* Open window after cancellation for next sending */
        uct_ud_ep_ca_ack(ep);

        uct_ud_leave(iface);
        return UCS_OK;
    }

    uct_ud_iface_progress_pending_tx(iface);
    status = uct_ud_ep_flush_nolock(iface, ep, comp);
    if (status == UCS_OK) {
        UCT_TL_EP_STAT_FLUSH(&ep->super);
    } else if (status == UCS_INPROGRESS) {
        UCT_TL_EP_STAT_FLUSH_WAIT(&ep->super);
    }
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

    uct_ud_peer_name((void*)&crep->peer);

    skb->len = sizeof(*neth) + sizeof(*crep);
    uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_CREP);
    return skb;
}

static uct_ud_send_skb_t *uct_ud_ep_resend(uct_ud_ep_t *ep)
{
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    uct_ud_send_skb_t *skb, *sent_skb;
    uct_ud_zcopy_desc_t *zdesc;
    size_t iov_it;

    /* check window */
    sent_skb = ucs_queue_iter_elem(sent_skb, ep->resend.pos, queue);
    if ((sent_skb == NULL) || UCT_UD_PSN_COMPARE(sent_skb->neth->psn, >=, ep->tx.max_psn)) {
        ucs_debug("ep(%p): out of window(psn=%d/max_psn=%d) - can not resend more",
                  ep, sent_skb ? sent_skb->neth->psn : -1, ep->tx.max_psn);
        uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_RESEND);
        return NULL;
    }

    /* skip dummy skb created for non-blocking flush */
    if ((uct_ud_neth_get_dest_id(sent_skb->neth) == UCT_UD_EP_NULL_ID) &&
        !(sent_skb->neth->packet_type & UCT_UD_PACKET_FLAG_CTL))
    {
        ep->resend.pos = ucs_queue_iter_next(ep->resend.pos);
        return NULL;
    }

    /* creq/crep must remove creq packet from window */
    ucs_assertv_always(!(uct_ud_ep_is_connected(ep) &&
                       (uct_ud_neth_get_dest_id(sent_skb->neth) == UCT_UD_EP_NULL_ID) &&
                       !(sent_skb->neth->packet_type & UCT_UD_PACKET_FLAG_AM)),
                       "ep(%p): CREQ resend on endpoint which is already connected", ep);

    skb = uct_ud_iface_resend_skb_get(iface);
    ucs_assert_always(skb != NULL);

    ep->resend.pos = ucs_queue_iter_next(ep->resend.pos);
    ep->resend.psn = sent_skb->neth->psn;
    memcpy(skb->neth, sent_skb->neth, sent_skb->len);
    skb->neth->ack_psn = ep->rx.acked_psn;
    skb->len           = sent_skb->len;
    if (sent_skb->flags & UCT_UD_SEND_SKB_FLAG_ZCOPY) {
        zdesc = uct_ud_zcopy_desc(sent_skb);
        for (iov_it = 0; iov_it < zdesc->iovcnt; ++iov_it) {
            if (zdesc->iov[iov_it].length > 0) {
                memcpy((char *)skb->neth + skb->len, zdesc->iov[iov_it].buffer,
                       zdesc->iov[iov_it].length);
                skb->len += zdesc->iov[iov_it].length;
            }
        }
    }
    /* force ack request on every Nth packet or on first packet in resend window */
    if ((skb->neth->psn % UCT_UD_RESENDS_PER_ACK) == 0 ||
        UCT_UD_PSN_COMPARE(skb->neth->psn, ==, ep->tx.acked_psn+1)) {
        skb->neth->packet_type |= UCT_UD_PACKET_FLAG_ACK_REQ;
    } else {
        skb->neth->packet_type &= ~UCT_UD_PACKET_FLAG_ACK_REQ;
    }

    ucs_debug("ep(%p): resending rt_psn %u rt_max_psn %u acked_psn %u max_psn %u ack_req %d",
              ep, ep->resend.psn, ep->resend.max_psn,
              ep->tx.acked_psn, ep->tx.max_psn,
              skb->neth->packet_type&UCT_UD_PACKET_FLAG_ACK_REQ ? 1 : 0);

    if (UCT_UD_PSN_COMPARE(ep->resend.psn, ==, ep->resend.max_psn)) {
        ucs_debug("ep(%p): resending completed", ep);
        ep->resend.psn = ep->resend.max_psn + 1;
        uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_RESEND);
    }

    return skb;
}

static void uct_ud_ep_do_pending_ctl(uct_ud_ep_t *ep, uct_ud_iface_t *iface)
{
    uct_ud_send_skb_t *skb;
    int flag = 0;

    if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_CREQ)) {
        skb = uct_ud_ep_prepare_creq(ep);
        if (skb) {
            flag = 1;
            uct_ud_ep_set_state(ep, UCT_UD_EP_FLAG_CREQ_SENT);
            uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_CREQ);
        }
    } else if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_CREP)) {
        skb = uct_ud_ep_prepare_crep(ep);
        if (skb) {
            flag = 1;
            uct_ud_ep_set_state(ep, UCT_UD_EP_FLAG_CREP_SENT);
            uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_CREP);
        }
    } else if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_RESEND)) {
        skb =  uct_ud_ep_resend(ep);
    } else if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_ACK)) {
        if (uct_ud_ep_is_connected(ep)) {
            skb = (void*)&iface->tx.skb_inl.super;
            uct_ud_neth_ctl_ack(ep, skb->neth);
        } else {
            /* Do not send ACKs if not connected yet. It may happen if
             * CREQ and CREP from peer are lost. Need to wait for CREP
             * resending by peer. */
            skb = NULL;
        }
        uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_ACK);
    } else if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_ACK_REQ)) {
        skb = (void*)&iface->tx.skb_inl.super;
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
    ucs_derived_of(iface->super.ops, uct_ud_iface_ops_t)->tx_skb(ep, skb, flag);
    if (flag) {
        /* creq and crep allocate real skb, it must be put on window like
         * a regular packet to ensure a retransmission.
         */
        uct_ud_iface_complete_tx_skb(iface, ep, skb);
    } else {
        uct_ud_iface_resend_skb_put(iface, skb);
    }
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
 *   - creq request
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

        if (status == UCS_INPROGRESS) {
            return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
        } else if (status != UCS_OK) {
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
    uct_ud_ep_t *ep = ucs_container_of(ucs_arbiter_elem_group(elem),
                                       uct_ud_ep_t, tx.pending.group);
    uct_pending_req_t *req;
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    uct_purge_cb_args_t *cb_args    = arg;
    uct_pending_purge_callback_t cb = cb_args->cb;

    if (&ep->tx.pending.elem == elem) {
        /* return ignored by arbiter */
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }
    req = ucs_container_of(elem, uct_pending_req_t, priv);
    if (cb) {
        cb(req, cb_args->arg);
    } else {
        ucs_debug("ep=%p cancelling user pending request %p", ep, req);
    }
    iface->tx.pending_q_len--;

    /* return ignored by arbiter */
    return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
}


void uct_ud_ep_pending_purge(uct_ep_h ep_h, uct_pending_purge_callback_t cb,
                             void *arg)
{
    uct_ud_ep_t *ep = ucs_derived_of(ep_h, uct_ud_ep_t);
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                           uct_ud_iface_t);
    uct_purge_cb_args_t args = {cb, arg};

    uct_ud_enter(iface);
    ucs_arbiter_group_purge(&iface->tx.pending_q, &ep->tx.pending.group,
                            uct_ud_ep_pending_purge_cb, &args);
    if (uct_ud_ep_ctl_op_isany(ep)) {
        ucs_arbiter_group_push_elem(&ep->tx.pending.group,
                                    &ep->tx.pending.elem);
        ucs_arbiter_group_schedule(&iface->tx.pending_q, &ep->tx.pending.group);
    }
    uct_ud_leave(iface);
}

void  uct_ud_ep_disconnect(uct_ep_h tl_ep)
{
    uct_ud_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_ep_t);
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
    uct_ud_ep_pending_purge(tl_ep, NULL, NULL);

    /* schedule flush */
    uct_ud_ep_flush(tl_ep, 0, NULL);

    ep->flags |= UCT_UD_EP_FLAG_DISCONNECTED;
    /* TODO: at least in debug mode keep and check tl_ep state  */
}
