/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ud_ep.h"
#include "ud_iface.h"
#include "ud_inl.h"
#include "ud_def.h"

#include <uct/api/uct_def.h>
#include <uct/ib/base/ib_verbs.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/time/time.h>


/* Must be less then peer_timeout to avoid false positive errors taking into
 * account timer resolution and not too small to avoid performance degradation
 */
#define UCT_UD_SLOW_TIMER_MAX_TICK(_iface)  ((_iface)->config.peer_timeout / 3)

static void uct_ud_ep_do_pending_ctl(uct_ud_ep_t *ep, uct_ud_iface_t *iface);

static void uct_ud_peer_name(uct_ud_peer_name_t *peer)
{
    ucs_strncpy_zero(peer->name, ucs_get_host_name(), sizeof(peer->name));
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

static void uct_ud_ep_resend_end(uct_ud_ep_t *ep)
{
    uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_RESEND);
    ep->flags &= ~UCT_UD_EP_FLAG_TX_NACKED;
}

static UCS_F_ALWAYS_INLINE void
uct_ud_ep_resend_ack(uct_ud_iface_t *iface, uct_ud_ep_t *ep)
{
    if (ucs_likely(UCT_UD_PSN_COMPARE(ep->resend.psn, >, ep->resend.max_psn))) {
        return;
    }

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
        uct_ud_ep_resend_end(ep);
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
    if (ep->ca.cwnd < ep->ca.wmax) {
        ep->ca.cwnd += UCT_UD_CA_AI_VALUE;
    }
    ep->tx.max_psn = ep->tx.acked_psn + ep->ca.cwnd;
}

static void uct_ud_ep_reset_max_psn(uct_ud_ep_t *ep)
{
    ep->tx.max_psn = ep->tx.psn + ep->ca.cwnd;
}

static void uct_ud_ep_reset(uct_ud_ep_t *ep)
{
    ep->tx.psn         = UCT_UD_INITIAL_PSN;
    ep->ca.cwnd        = UCT_UD_CA_MIN_WINDOW;
    ep->ca.wmax        = ucs_derived_of(ep->super.super.iface,
                                        uct_ud_iface_t)->config.max_window;
    ep->tx.acked_psn   = UCT_UD_INITIAL_PSN - 1;
    ep->tx.pending.ops = UCT_UD_EP_OP_NONE;
    uct_ud_ep_reset_max_psn(ep);
    ucs_queue_head_init(&ep->tx.window);

    ep->resend.pos       = ucs_queue_iter_begin(&ep->tx.window);
    ep->resend.psn       = ep->tx.psn;
    ep->resend.max_psn   = ep->tx.acked_psn;
    ep->tx.resend_count  = 0;
    ep->rx_creq_count    = 0;

    ep->rx.acked_psn = UCT_UD_INITIAL_PSN - 1;
    ucs_frag_list_init(ep->tx.psn-1, &ep->rx.ooo_pkts, 0 /*TODO: ooo support */
                       UCS_STATS_ARG(ep->super.stats));
}

static ucs_status_t uct_ud_ep_free_by_timeout(uct_ud_ep_t *ep,
                                              uct_ud_iface_t *iface)
{
    uct_ud_iface_ops_t *ops;
    ucs_time_t         diff;

    diff = ucs_twheel_get_time(&iface->tx.timer) - ep->close_time;
    if (diff > iface->config.peer_timeout) {
        ucs_debug("ud_ep %p is destroyed after %fs with timeout %fs\n",
                  ep, ucs_time_to_sec(diff),
                  ucs_time_to_sec(iface->config.peer_timeout));
        ops = ucs_derived_of(iface->super.ops, uct_ud_iface_ops_t);
        ops->ep_free(&ep->super.super);
        return UCS_OK;
    }
    return UCS_INPROGRESS;
}

static UCS_F_ALWAYS_INLINE int
uct_ud_skb_is_completed(uct_ud_send_skb_t *skb, uct_ud_psn_t ack_psn)
{
    ucs_assert(!(skb->flags & UCT_UD_SEND_SKB_FLAG_INVALID));
    return UCT_UD_PSN_COMPARE(skb->neth->psn, <=, ack_psn) &&
           !(skb->flags & UCT_UD_SEND_SKB_FLAG_RESENDING);
}

static UCS_F_ALWAYS_INLINE void
uct_ud_ep_window_release_inline(uct_ud_iface_t *iface, uct_ud_ep_t *ep,
                                uct_ud_psn_t ack_psn, ucs_status_t status,
                                int is_async, int invalidate_resend)
{
    uct_ud_send_skb_t *skb;

    ucs_queue_for_each_extract(skb, &ep->tx.window, queue,
                               uct_ud_skb_is_completed(skb, ack_psn)) {
        if (invalidate_resend && (ep->resend.pos == &skb->queue.next)) {
            ep->resend.pos = ucs_queue_iter_begin(&ep->tx.window);
            ep->resend.psn = ep->tx.acked_psn + 1;
        }
        if (ucs_likely(!(skb->flags & UCT_UD_SEND_SKB_FLAG_COMP))) {
            /* fast path case: skb without completion callback */
            uct_ud_skb_release(skb, 1);
        } else if (ucs_likely(!is_async)) {
            /* dispatch user completion immediately */
            uct_ud_iface_dispatch_comp(iface, uct_ud_comp_desc(skb)->comp,
                                       status);
            uct_ud_skb_release(skb, 1);
        } else {
            /* Don't call user completion from async context. Instead, put
             * it on a queue which will be progressed from main thread.
             */
            uct_ud_iface_add_async_comp(iface, skb, status);
        }
    }
}

static UCS_F_NOINLINE void
uct_ud_ep_window_release(uct_ud_ep_t *ep, ucs_status_t status, int is_async)
{
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);

    uct_ud_ep_window_release_inline(iface, ep, ep->tx.acked_psn, status, is_async, 0);
}

void uct_ud_ep_window_release_completed(uct_ud_ep_t *ep, int is_async)
{
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);

    uct_ud_ep_window_release_inline(iface, ep, ep->tx.acked_psn, UCS_OK, is_async, 1);
}

static void uct_ud_ep_purge_outstanding(uct_ud_ep_t *ep)
{
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    uct_ud_ctl_desc_t *cdesc;
    ucs_queue_iter_t iter;

    ucs_queue_for_each_safe(cdesc, iter, &iface->tx.outstanding_q, queue) {
        if (cdesc->ep == ep) {
            ucs_queue_del_iter(&iface->tx.outstanding_q, iter);
            uct_ud_iface_ctl_skb_complete(iface, cdesc, 0);
        }
    }

    ucs_assert_always(ep->tx.resend_count == 0);
}

static void uct_ud_ep_purge(uct_ud_ep_t *ep, ucs_status_t status)
{
    /* reset the maximal TX psn value to the default, since we should be able
     * to do TX operation after purging of the EP and uct_ep_flush(LOCAL)
     * operation has to return UCS_OK */
    uct_ud_ep_reset_max_psn(ep);
    uct_ud_ep_purge_outstanding(ep);
    ep->tx.acked_psn = (uct_ud_psn_t)(ep->tx.psn - 1);
    uct_ud_ep_window_release(ep, status, 0);
    ucs_assert(ucs_queue_is_empty(&ep->tx.window));
}

static unsigned uct_ud_ep_deferred_timeout_handler(void *arg)
{
    uct_ud_ep_t *ep       = arg;
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    ucs_status_t status;

    if (ep->flags & UCT_UD_EP_FLAG_DISCONNECTED) {
        uct_ud_ep_purge(ep, UCS_ERR_ENDPOINT_TIMEOUT);
        return 0;
    }

    if (ep->flags & UCT_UD_EP_FLAG_PRIVATE) {
        ucs_assert(ucs_queue_is_empty(&ep->tx.window));
        uct_ep_destroy(&ep->super.super);
        return 0;
    }

    uct_ud_ep_purge(ep, UCS_ERR_ENDPOINT_TIMEOUT);

    status = uct_iface_handle_ep_err(&iface->super.super.super,
                                     &ep->super.super,
                                     UCS_ERR_ENDPOINT_TIMEOUT);
    if (status != UCS_OK) {
        ucs_fatal("UD endpoint %p to "UCT_UD_EP_PEER_NAME_FMT": "
                  "unhandled timeout error",
                  ep, UCT_UD_EP_PEER_NAME_ARG(ep));
    }

    return 1;
}

static void uct_ud_ep_timer_backoff(uct_ud_ep_t *ep)
{
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);

    ep->tx.tick = ucs_min(ep->tx.tick * iface->tx.timer_backoff,
                          UCT_UD_SLOW_TIMER_MAX_TICK(iface));
    ucs_wtimer_add(&iface->tx.timer, &ep->timer, ep->tx.tick);
}

static UCS_F_ALWAYS_INLINE int uct_ud_ep_is_last_ack_received(uct_ud_ep_t *ep)
{
    return UCT_UD_PSN_COMPARE(ep->tx.acked_psn, ==, ep->tx.psn - 1);
}

static void uct_ud_ep_timer(ucs_wtimer_t *self)
{
    uct_ud_ep_t    *ep    = ucs_container_of(self, uct_ud_ep_t, timer);
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    ucs_time_t now, last_send, diff;
    ucs_status_t status;

    UCT_UD_EP_HOOK_CALL_TIMER(ep);

    if (uct_ud_ep_is_last_ack_received(ep)) {
        /* Do not free the EP until all scheduled communications are done. */
        if (ep->flags & UCT_UD_EP_FLAG_DISCONNECTED) {
            status = uct_ud_ep_free_by_timeout(ep, iface);
            if (status == UCS_INPROGRESS) {
                uct_ud_ep_timer_backoff(ep);
            }
        }
        return;
    }

    ucs_assert(!ucs_queue_is_empty(&ep->tx.window));

    now  = ucs_twheel_get_time(&iface->tx.timer);
    diff = now - ep->tx.send_time;
    if (diff > iface->config.peer_timeout) {
        ucs_debug("ep %p: timeout of %.2f sec, config::peer_timeout - %.2f sec",
                  ep, ucs_time_to_sec(diff),
                  ucs_time_to_sec(iface->config.peer_timeout));
        ucs_callbackq_add_safe(&iface->super.super.worker->super.progress_q,
                               uct_ud_ep_deferred_timeout_handler, ep,
                               UCS_CALLBACKQ_FLAG_ONESHOT);
        return;
    }

    /* If we are already resending, do not consider this timeout as packet drop.
     * It just means the sender is slow.
     */
    if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_ACK_REQ|UCT_UD_EP_OP_RESEND) ||
        (ep->tx.resend_count > 0)) {
        ucs_trace("ep %p: resend still in progress, ops 0x%x tx_count %d",
                  ep, ep->tx.pending.ops, ep->tx.resend_count);
        uct_ud_ep_timer_backoff(ep);
        return;
    }

    last_send = ucs_max(ep->tx.send_time, ep->tx.resend_time);
    diff      = now - last_send;
    if (diff > iface->tx.tick) {
        if (diff > 3 * iface->tx.tick) {
            ucs_trace("scheduling resend now: %lu last_send: %lu diff: %lu tick: %lu",
                      now, last_send, diff, ep->tx.tick);
            uct_ud_ep_ca_drop(ep);
            uct_ud_ep_resend_start(iface, ep);
        }

        if (uct_ud_ep_is_connected(ep)) {
            /* Try to request ACK/NACK twice before going into full resend mode */
            uct_ud_ep_ctl_op_add(iface, ep, UCT_UD_EP_OP_ACK_REQ);
        }
    }

    uct_ud_ep_timer_backoff(ep);
}

UCS_CLASS_INIT_FUNC(uct_ud_ep_t, uct_ud_iface_t *iface,
                    const uct_ep_params_t* params)
{
    ucs_trace_func("");

    memset(self, 0, sizeof(*self));
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    uct_ud_enter(iface);

    self->dest_ep_id = UCT_UD_EP_NULL_ID;
    self->path_index = UCT_EP_PARAMS_GET_PATH_INDEX(params);
    uct_ud_ep_reset(self);
    uct_ud_iface_add_ep(iface, self);
    self->tx.tick = iface->tx.tick;
    ucs_wtimer_init(&self->timer, uct_ud_ep_timer);
    ucs_arbiter_group_init(&self->tx.pending.group);
    ucs_arbiter_elem_init(&self->tx.pending.elem);

    UCT_UD_EP_HOOK_INIT(self);
    ucs_debug("created ep ep=%p iface=%p id=%d", self, iface, self->ep_id);

    uct_ud_leave(iface);

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE int
uct_ud_ep_is_last_pending_elem(uct_ud_ep_t *ep, ucs_arbiter_elem_t *elem)
{
    return (/* this is the only one pending element in the group */
            (ucs_arbiter_elem_is_only(elem)) ||
            (/* the next element in the group is control operation */
             (elem->next == &ep->tx.pending.elem) &&
             /* only two elements are in the group (the 1st element is the
              * current one, the 2nd (or the last) element is the control one) */
             (ucs_arbiter_group_tail(&ep->tx.pending.group) == &ep->tx.pending.elem)));
            
}

static ucs_arbiter_cb_result_t
uct_ud_ep_pending_cancel_cb(ucs_arbiter_t *arbiter, ucs_arbiter_group_t *group,
                            ucs_arbiter_elem_t *elem, void *arg)
{
    uct_ud_ep_t *ep = ucs_container_of(group, uct_ud_ep_t, tx.pending.group);
    uct_pending_req_t *req;

    /* we may have pending op on ep */
    if (&ep->tx.pending.elem == elem) {
        /* return ignored by arbiter */
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

    /* uct user should not have anything pending */
    req = ucs_container_of(elem, uct_pending_req_t, priv);
    ucs_warn("ep=%p removing user pending req=%p", ep, req);

    if (uct_ud_ep_is_last_pending_elem(ep, elem)) {
        uct_ud_ep_remove_has_pending_flag(ep);
    }

    /* return ignored by arbiter */
    return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
}

static int uct_ud_ep_remove_timeout_filter(const ucs_callbackq_elem_t *elem,
                                           void *arg)
{
    return (elem->cb == uct_ud_ep_deferred_timeout_handler) && (elem->arg == arg);
}

static UCS_CLASS_CLEANUP_FUNC(uct_ud_ep_t)
{
    uct_ud_iface_t *iface = ucs_derived_of(self->super.super.iface, uct_ud_iface_t);

    ucs_trace_func("ep=%p id=%d conn_sn=%d", self, self->ep_id, self->conn_sn);

    uct_ud_enter(iface);

    ucs_callbackq_remove_if(&iface->super.super.worker->super.progress_q,
                            uct_ud_ep_remove_timeout_filter, self);
    uct_ud_ep_purge(self, UCS_ERR_CANCELED);

    ucs_wtimer_remove(&iface->tx.timer, &self->timer);
    uct_ud_iface_remove_ep(iface, self);
    uct_ud_iface_cep_remove_ep(iface, self);
    ucs_frag_list_cleanup(&self->rx.ooo_pkts);

    ucs_arbiter_group_purge(&iface->tx.pending_q, &self->tx.pending.group,
                            uct_ud_ep_pending_cancel_cb, 0);

    if (!ucs_queue_is_empty(&self->tx.window)) {
        ucs_debug("ep=%p id=%d conn_sn=%d has %d unacked packets",
                   self, self->ep_id, self->conn_sn,
                   (int)ucs_queue_length(&self->tx.window));
    }
    ucs_arbiter_group_cleanup(&self->tx.pending.group);
    uct_ud_leave(iface);
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
    uct_ib_device_t UCS_V_UNUSED *dev = uct_ib_iface_device(&iface->super);
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
    ep->flags     &= ~UCT_UD_EP_FLAG_CONNECTED;

    return UCS_OK;
}

ucs_status_t uct_ud_ep_create_connected_common(const uct_ep_params_t *ep_params,
                                               uct_ep_h *new_ep_p)
{
    uct_ud_iface_t *iface              = ucs_derived_of(ep_params->iface,
                                                        uct_ud_iface_t);
    const uct_ib_address_t *ib_addr    = (const uct_ib_address_t*)
                                         ep_params->dev_addr;
    const uct_ud_iface_addr_t *if_addr = (const uct_ud_iface_addr_t*)
                                         ep_params->iface_addr;
    int path_index                     = UCT_EP_PARAMS_GET_PATH_INDEX(ep_params);
    void *peer_address;
    uct_ud_send_skb_t *skb;
    uct_ud_ep_conn_sn_t conn_sn;
    uct_ep_params_t params;
    ucs_status_t status;
    uct_ud_ep_t *ep;
    uct_ep_h new_ep_h;

    uct_ud_enter(iface);

    *new_ep_p = NULL;

    conn_sn = uct_ud_iface_cep_get_conn_sn(iface, ib_addr, if_addr, path_index);
    ep      = uct_ud_iface_cep_get_ep(iface, ib_addr, if_addr, path_index,
                                      conn_sn, 1);
    if (ep != NULL) {
        uct_ud_ep_set_state(ep, UCT_UD_EP_FLAG_CREQ_NOTSENT);
        ep->flags &= ~UCT_UD_EP_FLAG_PRIVATE;
        status     = UCS_OK;
        uct_ud_iface_cep_insert_ep(iface, ib_addr, if_addr, path_index,
                                   conn_sn, ep);
        goto out_set_ep;
    }

    params.field_mask = UCT_EP_PARAM_FIELD_IFACE |
                        UCT_EP_PARAM_FIELD_PATH_INDEX;
    params.iface      = &iface->super.super.super;
    params.path_index = path_index;

    status = uct_ep_create(&params, &new_ep_h);
    if (status != UCS_OK) {
        goto out;
    }

    ep          = ucs_derived_of(new_ep_h, uct_ud_ep_t);
    ep->conn_sn = conn_sn;

    status = uct_ud_ep_connect_to_iface(ep, ib_addr, if_addr);
    if (status != UCS_OK) {
        goto out;
    }

    uct_ud_iface_cep_insert_ep(iface, ib_addr, if_addr, path_index, conn_sn, ep);
    peer_address = uct_iface_invoke_ops_func(&iface->super, uct_ud_iface_ops_t,
                                             ep_get_peer_address, ep);

    status = uct_ud_iface_unpack_peer_address(iface, ib_addr, if_addr,
                                              ep->path_index, peer_address);
    if (status != UCS_OK) {
        uct_ud_ep_disconnect_from_iface(&ep->super.super);
        goto out;
    }

    skb = uct_ud_ep_prepare_creq(ep);
    if (skb != NULL) {
        uct_ud_iface_send_ctl(iface, ep, skb, NULL, 0,
                              UCT_UD_IFACE_SEND_CTL_FLAG_SOLICITED, 1);
        uct_ud_iface_complete_tx_skb(iface, ep, skb);
        uct_ud_ep_set_state(ep, UCT_UD_EP_FLAG_CREQ_SENT);
    } else {
        uct_ud_ep_ctl_op_add(iface, ep, UCT_UD_EP_OP_CREQ);
    }

out_set_ep:
    /* cppcheck-suppress autoVariables */
    *new_ep_p = &ep->super.super;
out:
    uct_ud_leave(iface);
    return status;
}

ucs_status_t uct_ud_ep_connect_to_ep(uct_ep_h tl_ep,
                                     const uct_device_addr_t *dev_addr,
                                     const uct_ep_addr_t *uct_ep_addr)
{
    uct_ud_ep_t *ep                   = ucs_derived_of(tl_ep, uct_ud_ep_t);
    uct_ud_iface_t *iface             = ucs_derived_of(ep->super.super.iface,
                                                       uct_ud_iface_t);
    const uct_ib_address_t *ib_addr   = (const uct_ib_address_t*)dev_addr;
    const uct_ud_ep_addr_t *ep_addr   = (const uct_ud_ep_addr_t*)uct_ep_addr;
    uct_ib_device_t UCS_V_UNUSED *dev = uct_ib_iface_device(&iface->super);
    void *peer_address;
    char buf[128];

    ucs_assert_always(ep->dest_ep_id == UCT_UD_EP_NULL_ID);
    ucs_trace_func("");

    uct_ud_ep_set_dest_ep_id(ep, uct_ib_unpack_uint24(ep_addr->ep_id));

    ucs_frag_list_cleanup(&ep->rx.ooo_pkts);
    uct_ud_ep_reset(ep);

    ucs_debug(UCT_IB_IFACE_FMT" slid %d qpn 0x%x epid %u connected to %s "
              "qpn 0x%x epid %u", UCT_IB_IFACE_ARG(&iface->super),
              dev->port_attr[iface->super.config.port_num - dev->first_port].lid,
              iface->qp->qp_num, ep->ep_id,
              uct_ib_address_str(ib_addr, buf, sizeof(buf)),
              uct_ib_unpack_uint24(ep_addr->iface_addr.qp_num),
              ep->dest_ep_id);

    peer_address = uct_iface_invoke_ops_func(&iface->super, uct_ud_iface_ops_t,
                                             ep_get_peer_address, ep);
    return uct_ud_iface_unpack_peer_address(iface, ib_addr,
                                            &ep_addr->iface_addr,
                                            ep->path_index, peer_address);
}

static UCS_F_ALWAYS_INLINE void
uct_ud_ep_process_ack(uct_ud_iface_t *iface, uct_ud_ep_t *ep,
                      uct_ud_psn_t ack_psn, int is_async)
{
    /* Ignore duplicate ACK */
    if (ucs_unlikely(UCT_UD_PSN_COMPARE(ack_psn, <=, ep->tx.acked_psn))) {
        return;
    }

    ep->tx.acked_psn = ack_psn;

    uct_ud_ep_window_release_inline(iface, ep, ack_psn, UCS_OK, is_async, 0);
    uct_ud_ep_ca_ack(ep);
    uct_ud_ep_resend_ack(iface, ep);

    ucs_arbiter_group_schedule(&iface->tx.pending_q, &ep->tx.pending.group);

    ep->tx.tick      = iface->tx.tick;
    ep->tx.send_time = uct_ud_iface_get_time(iface);
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
    uct_ep_params_t params;
    uct_ud_ep_t *ep;
    ucs_status_t status;
    uct_ep_t *ep_h;

    /* create new endpoint */
    params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
    params.iface      = &iface->super.super.super;
    status = uct_ep_create(&params, &ep_h);
    ucs_assert_always(status == UCS_OK);
    ep = ucs_derived_of(ep_h, uct_ud_ep_t);

    status = uct_ep_connect_to_ep(ep_h, (void*)uct_ud_creq_ib_addr(ctl),
                                  (void*)&ctl->conn_req.ep_addr);
    ucs_assert_always(status == UCS_OK);

    ep->path_index = ctl->conn_req.path_index;

    uct_ud_ep_set_state(ep, UCT_UD_EP_FLAG_PRIVATE);

    ep->conn_sn = ctl->conn_req.conn_sn;
    uct_ud_iface_cep_insert_ep(iface, uct_ud_creq_ib_addr(ctl),
                               &ctl->conn_req.ep_addr.iface_addr,
                               ep->path_index, ctl->conn_req.conn_sn, ep);
    return ep;
}

static void uct_ud_ep_rx_creq(uct_ud_iface_t *iface, uct_ud_neth_t *neth)
{
    uct_ud_ctl_hdr_t *ctl = (uct_ud_ctl_hdr_t *)(neth + 1);
    uct_ud_ep_t *ep;

    ucs_assert_always(ctl->type == UCT_UD_PACKET_CREQ);

    ep = uct_ud_iface_cep_get_ep(iface, uct_ud_creq_ib_addr(ctl),
                                 &ctl->conn_req.ep_addr.iface_addr,
                                 ctl->conn_req.path_index,
                                 ctl->conn_req.conn_sn, 0);
    if (ep == NULL) {
        ep = uct_ud_ep_create_passive(iface, ctl);
        ucs_assert_always(ep != NULL);
        ep->rx.ooo_pkts.head_sn = neth->psn;
        uct_ud_peer_copy(&ep->peer, ucs_unaligned_ptr(&ctl->peer));
        uct_ud_ep_ctl_op_add(iface, ep, UCT_UD_EP_OP_CREP);
    } else {
        if (ep->dest_ep_id == UCT_UD_EP_NULL_ID) {
            /* simultaneuous CREQ */
            uct_ud_ep_set_dest_ep_id(ep, uct_ib_unpack_uint24(ctl->conn_req.ep_addr.ep_id));
            ep->rx.ooo_pkts.head_sn = neth->psn;
            uct_ud_peer_copy(&ep->peer, ucs_unaligned_ptr(&ctl->peer));
            ucs_debug("simultaneuous CREQ ep=%p"
                      "(iface=%p conn_sn=%d ep_id=%d, dest_ep_id=%d rx_psn=%u)",
                      ep, iface, ep->conn_sn, ep->ep_id,
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

    ++ep->rx_creq_count;

    ucs_assertv_always(ctl->conn_req.conn_sn == ep->conn_sn,
                       "creq->conn_sn=%d ep->conn_sn=%d",
                       ctl->conn_req.conn_sn, ep->conn_sn);

    ucs_assertv_always(ctl->conn_req.path_index == ep->path_index,
                       "creq->path_index=%d ep->path_index=%d",
                       ctl->conn_req.path_index, ep->path_index);

    ucs_assertv_always(uct_ib_unpack_uint24(ctl->conn_req.ep_addr.ep_id) ==
                       ep->dest_ep_id,
                       "creq->ep_addr.ep_id=%d ep->dest_ep_id=%d",
                       uct_ib_unpack_uint24(ctl->conn_req.ep_addr.ep_id),
                       ep->dest_ep_id);

    /* creq must always have same psn */
    ucs_assertv_always(ep->rx.ooo_pkts.head_sn == neth->psn,
                       "iface=%p ep=%p conn_sn=%d ep_id=%d, dest_ep_id=%d rx_psn=%u "
                       "neth_psn=%u ep_flags=0x%x ctl_ops=0x%x rx_creq_count=%d",
                       iface, ep, ep->conn_sn, ep->ep_id, ep->dest_ep_id,
                       ep->rx.ooo_pkts.head_sn, neth->psn, ep->flags,
                       ep->tx.pending.ops, ep->rx_creq_count);
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

    if (uct_ud_ep_is_connected(ep)) {
        ucs_assertv_always(ep->dest_ep_id == ctl->conn_rep.src_ep_id,
                           "ep=%p [id=%d dest_ep_id=%d flags=0x%x] "
                           "crep [neth->dest=%d dst_ep_id=%d src_ep_id=%d]",
                           ep, ep->ep_id, ep->dest_ep_id, ep->path_index, ep->flags,
                           uct_ud_neth_get_dest_id(neth), ctl->conn_rep.src_ep_id);
    }

    /* Discard duplicate CREP */
    if (UCT_UD_PSN_COMPARE(neth->psn, <, ep->rx.ooo_pkts.head_sn)) {
        return;
    }

    ep->rx.ooo_pkts.head_sn = neth->psn;
    uct_ud_ep_set_dest_ep_id(ep, ctl->conn_rep.src_ep_id);
    ucs_arbiter_group_schedule(&iface->tx.pending_q, &ep->tx.pending.group);
    uct_ud_peer_copy(&ep->peer, ucs_unaligned_ptr(&ctl->peer));
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

    /* CREQ should not be sent if CREP for the counter CREQ is scheduled
     * (or sent already) */
    ucs_assertv_always(!uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_CREP) &&
                       !(ep->flags & UCT_UD_EP_FLAG_CREP_SENT),
                       "iface=%p ep=%p conn_sn=%d rx_psn=%u ep_flags=0x%x "
                       "ctl_ops=0x%x rx_creq_count=%d",
                       iface, ep, ep->conn_sn, ep->rx.ooo_pkts.head_sn,
                       ep->flags, ep->tx.pending.ops, ep->rx_creq_count);

    skb = uct_ud_iface_get_tx_skb(iface, ep);
    if (!skb) {
        return NULL;
    }

    neth = skb->neth;
    uct_ud_neth_init_data(ep, neth);

    neth->packet_type  = UCT_UD_EP_NULL_ID;
    neth->packet_type |= UCT_UD_PACKET_FLAG_CTL;

    creq = (uct_ud_ctl_hdr_t *)(neth + 1);

    creq->type                = UCT_UD_PACKET_CREQ;
    creq->conn_req.conn_sn    = ep->conn_sn;
    creq->conn_req.path_index = ep->path_index;

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

    uct_ud_peer_name(ucs_unaligned_ptr(&creq->peer));

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
                            (ep->ep_id != dest_id)))
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

    if (ucs_unlikely(UCT_UD_PSN_COMPARE(neth->psn, >, ep->rx.ooo_pkts.head_sn + 1))) {
        uct_ud_ep_ctl_op_add(iface, ep, UCT_UD_EP_OP_NACK);
    }

    if (ucs_unlikely(!is_am)) {
        if (neth->packet_type & UCT_UD_PACKET_FLAG_NAK) {
            uct_ud_ep_set_state(ep, UCT_UD_EP_FLAG_TX_NACKED);
            goto out;
        }

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
        if ((ooo_type != UCS_FRAG_LIST_INSERT_DUP) &&
            (ooo_type != UCS_FRAG_LIST_INSERT_FAIL)) {
            ucs_fatal("Out of order is not implemented: got %d", ooo_type);
        }
        ucs_trace_data("DUP/OOB - schedule ack, head_sn=%d sn=%d",
                       ep->rx.ooo_pkts.head_sn, neth->psn);
        goto out;
    }

    if (ucs_unlikely(!is_am && (neth->packet_type & UCT_UD_PACKET_FLAG_PUT))) {
        /* TODO: remove once ucp implements put */
        uct_ud_ep_rx_put(neth, byte_len);
        goto out;
    }

    if (ucs_unlikely(is_async &&
                     !(iface->super.super.am[am_id].flags & UCT_CB_FLAG_ASYNC))) {
        skb->u.am.len = byte_len - sizeof(*neth);
        ucs_queue_push(&iface->rx.pending_q, &skb->u.am.queue);
    } else {
        /* Avoid reordering with respect to pending operations, if user AM handler
         * initiates sends from any endpoint created on the iface.
         * This flag would be cleared after all incoming messages
         * are processed. */
        uct_ud_iface_raise_pending_async_ev(iface);

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

    if (ucs_queue_is_empty(&ep->tx.window) &&
        ucs_queue_is_empty(&iface->tx.async_comp_q)) {
        /* No outstanding operations */
        ucs_assert(ep->tx.resend_count == 0);
        return UCS_OK;
    }

    /* Expedite acknowledgment on the last skb in the window */
    if (uct_ud_ep_is_last_ack_received(ep)) {
        uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_ACK_REQ);
    } else {
        ucs_assert(!ucs_queue_is_empty(&ep->tx.window));
        skb = ucs_queue_tail_elem_non_empty(&ep->tx.window, uct_ud_send_skb_t,
                                            queue);
        if (!(skb->flags & UCT_UD_SEND_SKB_FLAG_ACK_REQ)) {
            /* If we didn't ask for ACK on last skb, send an ACK_REQ message.
             * It will speed up the flush because we will not have to wait until
             * retransmit is triggered.
             * Also, prevent from sending more control messages like this after
             * first time by turning on the flag on the last skb.
             */

            /* Since the function can be called from the arbiter context it is
             * impossible to schedule a control operation. So just raise a
             * flag and if there is no other control send ACK_REQ directly.
             *
             * If there is other control arbiter will take care of it.
             */
            ep->tx.pending.ops |= UCT_UD_EP_OP_ACK_REQ;
            if (uct_ud_ep_ctl_op_check_ex(ep, UCT_UD_EP_OP_ACK_REQ)) {
                uct_ud_ep_do_pending_ctl(ep, iface);
            }

            skb->flags |= UCT_UD_SEND_SKB_FLAG_ACK_REQ;
        }
    }

    /* If the user requested a callback, allocate a dummy skb which will be
     * released when the current sequence number is completed.
     */
    if (comp != NULL) {
        ucs_assert(comp->count > 0);

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
        skb->neth->psn              = (uct_ud_psn_t)(ep->tx.psn - 1);
        uct_ud_neth_set_dest_id(skb->neth, UCT_UD_EP_NULL_ID);
        uct_ud_comp_desc(skb)->comp = comp;

        if (!ucs_queue_is_empty(&ep->tx.window)) {
            /* If window non-empty: add to window */
            ucs_queue_push(&ep->tx.window, &skb->queue);
        } else {
            /* Otherwise, add the skb after async completions */
            ucs_assert(ep->tx.resend_count == 0);
            uct_ud_iface_add_async_comp(iface, skb, UCS_OK);
        }

        ucs_trace_data("added dummy flush skb %p psn %d user_comp %p", skb,
                       skb->neth->psn, comp);
    }

    return UCS_INPROGRESS;
}

ucs_status_t uct_ud_ep_flush(uct_ep_h ep_h, unsigned flags,
                             uct_completion_t *comp)
{
    uct_ud_ep_t *ep = ucs_derived_of(ep_h, uct_ud_ep_t);
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                           uct_ud_iface_t);
    ucs_status_t status;

    uct_ud_enter(iface);

    if (ucs_unlikely(flags & UCT_FLUSH_FLAG_CANCEL)) {
        uct_ep_pending_purge(ep_h, NULL, 0);
        uct_ud_iface_dispatch_async_comps(iface);
        uct_ud_ep_purge(ep, UCS_ERR_CANCELED);
        /* FIXME make flush(CANCEL) operation truly non-blocking and wait until
         * all of the outstanding sends are completed. Without this, zero-copy
         * sends which are still on the QP could be reported as completed which
         * can lead to sending corrupt data, or local access error. */
        status = UCS_OK;
        goto out;
    }

    if (ucs_unlikely(uct_ud_iface_has_pending_async_ev(iface))) {
        status = UCS_ERR_NO_RESOURCE;
        goto out;
    }

    status = uct_ud_ep_flush_nolock(iface, ep, comp);
    if (status == UCS_OK) {
        UCT_TL_EP_STAT_FLUSH(&ep->super);
    } else if (status == UCS_INPROGRESS) {
        UCT_TL_EP_STAT_FLUSH_WAIT(&ep->super);
    }

out:
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

    /* Check that CREQ is neither sheduled nor waiting for CREP ack */
    ucs_assertv_always(!uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_CREQ) &&
                       uct_ud_ep_is_last_ack_received(ep),
                       "iface=%p ep=%p conn_sn=%d ep_id=%d, dest_ep_id=%d rx_psn=%u "
                       "ep_flags=0x%x ctl_ops=0x%x rx_creq_count=%d",
                       iface, ep, ep->conn_sn, ep->ep_id, ep->dest_ep_id,
                       ep->rx.ooo_pkts.head_sn, ep->flags, ep->tx.pending.ops,
                       ep->rx_creq_count);

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

    uct_ud_peer_name(ucs_unaligned_ptr(&crep->peer));

    skb->len = sizeof(*neth) + sizeof(*crep);
    uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_CREP);
    return skb;
}

static void uct_ud_ep_send_creq_crep(uct_ud_iface_t *iface, uct_ud_ep_t *ep,
                                     uct_ud_send_skb_t *skb)
{
    uct_ud_iface_send_ctl(iface, ep, skb, NULL, 0,
                          UCT_UD_IFACE_SEND_CTL_FLAG_SOLICITED, 1);
    uct_ud_iface_complete_tx_skb(iface, ep, skb);
}

static void uct_ud_ep_resend(uct_ud_ep_t *ep)
{
    uct_ud_iface_t *iface       = ucs_derived_of(ep->super.super.iface,
                                                 uct_ud_iface_t);
    size_t max_len_without_nack = sizeof(uct_ud_neth_t) +
                                  sizeof(uct_ud_ctl_hdr_t) +
                                  iface->super.addr_size;
    uct_ud_send_skb_t *skb, *sent_skb;
    ucs_queue_iter_t resend_pos;
    uct_ud_zcopy_desc_t *zdesc;
    uct_ud_iov_t skb_iov, *iov;
    uct_ud_ctl_desc_t *cdesc;
    int max_log_sge;
    uint16_t iovcnt;

    /* check if the resend window was acknowledged */
    if (UCT_UD_PSN_COMPARE(ep->resend.max_psn, <=, ep->tx.acked_psn)) {
        uct_ud_ep_resend_end(ep);
        return;
    }

    /* check window */
    resend_pos = ep->resend.pos;
    if (ucs_queue_iter_end(&ep->tx.window, resend_pos)) {
        uct_ud_ep_resend_end(ep);
        return;
    }

    sent_skb = ucs_queue_iter_elem(sent_skb, resend_pos, queue);

    ucs_assert(((uintptr_t)sent_skb % UCT_UD_SKB_ALIGN) == 0);
    if (UCT_UD_PSN_COMPARE(sent_skb->neth->psn, >=, ep->tx.max_psn)) {
        ucs_debug("ep(%p): out of window(psn=%d/max_psn=%d) - can not resend more",
                  ep, sent_skb ? sent_skb->neth->psn : -1, ep->tx.max_psn);
        uct_ud_ep_resend_end(ep);
        return;
    }

    /* stop resend if packet is larger than CREQ and there wasn't NACK from
     * other side */
    if (!(ep->flags & UCT_UD_EP_FLAG_TX_NACKED) &&
        (sent_skb->len > max_len_without_nack)) {
        uct_ud_ep_resend_end(ep);
        return;
    }

    /* Update resend position */
    ep->resend.pos = ucs_queue_iter_next(resend_pos);

    /* skip skb which was already resent but didn't get send completion yet */
    if (sent_skb->flags & UCT_UD_SEND_SKB_FLAG_RESENDING) {
        ucs_debug("ep(%p): skb %p already being resent", ep, sent_skb);
        return;
    }

    /* skip dummy skb created for non-blocking flush */
    if ((uct_ud_neth_get_dest_id(sent_skb->neth) == UCT_UD_EP_NULL_ID) &&
        !(sent_skb->neth->packet_type & UCT_UD_PACKET_FLAG_CTL)) {
        return;
    }

    /* creq/crep must remove creq packet from window */
    ucs_assertv_always(!(uct_ud_ep_is_connected(ep) &&
                         (uct_ud_neth_get_dest_id(sent_skb->neth) == UCT_UD_EP_NULL_ID) &&
                         !(sent_skb->neth->packet_type & UCT_UD_PACKET_FLAG_AM)),
                       "ep(%p): CREQ resend on endpoint which is already connected", ep);

    /* Allocate a control skb which would refer to the original skb.
     *
     * If we didn't resend an skb, it would be released after remote ACK: we can
     * assume that if it was received by remote side, it has been fully sent by
     * local side. However, if we started resend, all bets are off: we can get
     * an ACK while there is still a resend-skb in the QP. In this case, we must
     * wait for send completion on that resend-skb before signaling completion
     * to the user. If the resend-skb got send completion, assume the original
     * skb was sent as well.
     */
    skb                = uct_ud_iface_ctl_skb_get(iface);
    skb->flags         = UCT_UD_SEND_SKB_FLAG_CTL_RESEND;
    sent_skb->flags   |= UCT_UD_SEND_SKB_FLAG_RESENDING;
    ep->resend.psn     = sent_skb->neth->psn;
    ep->tx.resend_time = uct_ud_iface_get_time(iface);

    if (sent_skb->flags & UCT_UD_SEND_SKB_FLAG_ZCOPY) {
        /* copy neth + am header part */
        skb->len = sent_skb->len;

        /* set iov pointer to payload */
        zdesc       = uct_ud_zcopy_desc(sent_skb);
        iov         = zdesc->iov;
        iovcnt      = zdesc->iovcnt;
        max_log_sge = UCT_IB_MAX_ZCOPY_LOG_SGE(&iface->super);
    } else {
        /* copy neth part only, since we may not have enough room in the control
         * skb for the whole payload + ctl desc, and we also prefer to avoid
         * memcpy() overhead. */
        ucs_assert(sent_skb->len >= sizeof(uct_ud_neth_t));
        skb->len       = sizeof(uct_ud_neth_t);

        /* set iov to skb payload */
        skb_iov.buffer = UCS_PTR_BYTE_OFFSET(sent_skb->neth, sizeof(uct_ud_neth_t));
        skb_iov.length = sent_skb->len - sizeof(uct_ud_neth_t);
        skb_iov.lkey   = sent_skb->lkey;
        iov            = &skb_iov;
        iovcnt         = 1;
        max_log_sge    = 2;
    }

    memcpy(skb->neth, sent_skb->neth, skb->len);
    skb->neth->ack_psn = ep->rx.acked_psn;
    cdesc              = uct_ud_ctl_desc(skb);
    cdesc->self_skb    = skb;
    cdesc->resent_skb  = sent_skb;
    cdesc->ep          = ep;

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
        uct_ud_ep_resend_end(ep);
    }

    /* Send control message and save operation on queue. Use signaled-send to
     * make sure user completion will not be delayed indefinitely */
    cdesc->sn = uct_ud_iface_send_ctl(iface, ep, skb, iov, iovcnt,
                                      UCT_UD_IFACE_SEND_CTL_FLAG_SIGNALED |
                                      UCT_UD_IFACE_SEND_CTL_FLAG_SOLICITED,
                                      max_log_sge);
    uct_ud_iface_add_ctl_desc(iface, cdesc);
    ++ep->tx.resend_count;
}

static void uct_ud_ep_send_ack(uct_ud_iface_t *iface, uct_ud_ep_t *ep)
{
    int ctl_flags = 0;
    uct_ud_ctl_desc_t *cdesc;
    uct_ud_send_skb_t *skb;

    /* Do not send ACKs if not connected yet. It may happen if CREQ and CREP
     * from peer are lost. Need to wait for CREP resend from peer.
     */
    if (!uct_ud_ep_is_connected(ep)) {
        goto out;
    }

    if (sizeof(uct_ud_neth_t) <= iface->config.max_inline) {
        skb        = ucs_alloca(sizeof(*skb) + sizeof(uct_ud_neth_t));
        skb->flags = 0;
#if UCS_ENABLE_ASSERT
        skb->lkey  = 0;
#endif
        ctl_flags |= UCT_UD_IFACE_SEND_CTL_FLAG_INLINE;
    } else {
        skb        = uct_ud_iface_ctl_skb_get(iface);
    }

    uct_ud_neth_init_data(ep, skb->neth);
    skb->flags             = UCT_UD_SEND_SKB_FLAG_CTL_ACK;
    skb->len               = sizeof(uct_ud_neth_t);
    skb->neth->packet_type = ep->dest_ep_id;
    if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_ACK_REQ)) {
        skb->neth->packet_type |= UCT_UD_PACKET_FLAG_ACK_REQ;
        ctl_flags              |= UCT_UD_IFACE_SEND_CTL_FLAG_SOLICITED;
    }

    if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_NACK)) {
        skb->neth->packet_type |= UCT_UD_PACKET_FLAG_NAK;
    }

    if (ctl_flags & UCT_UD_IFACE_SEND_CTL_FLAG_INLINE) {
        uct_ud_iface_send_ctl(iface, ep, skb, NULL, 0, ctl_flags, 1);
    } else {
        /* if skb is taken from memory pool, release it in send completion */
        cdesc             = uct_ud_ctl_desc(skb);
        cdesc->sn         = uct_ud_iface_send_ctl(iface, ep, skb, NULL, 0,
                                                  ctl_flags, 1);
        cdesc->self_skb   = skb;
        cdesc->resent_skb = NULL;
        cdesc->ep         = NULL;
        uct_ud_iface_add_ctl_desc(iface, cdesc);
    }

out:
    uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_CTL_ACK);
}

static void uct_ud_ep_do_pending_ctl(uct_ud_ep_t *ep, uct_ud_iface_t *iface)
{
    uct_ud_send_skb_t *skb;

    if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_CREQ)) {
        skb = uct_ud_ep_prepare_creq(ep);
        if (skb) {
            uct_ud_ep_set_state(ep, UCT_UD_EP_FLAG_CREQ_SENT);
            uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_CREQ);
            uct_ud_ep_send_creq_crep(iface, ep, skb);
        }
    } else if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_CREP)) {
        skb = uct_ud_ep_prepare_crep(ep);
        if (skb) {
            uct_ud_ep_set_state(ep, UCT_UD_EP_FLAG_CREP_SENT);
            uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_CREP);
            uct_ud_ep_send_creq_crep(iface, ep, skb);
        }
    } else if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_RESEND)) {
        uct_ud_ep_resend(ep);
    } else if (uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_CTL_ACK)) {
        uct_ud_ep_send_ack(iface, ep);
    } else {
        ucs_assertv(!uct_ud_ep_ctl_op_isany(ep),
                    "unsupported pending op mask: %x", ep->tx.pending.ops);
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
uct_ud_ep_do_pending(ucs_arbiter_t *arbiter, ucs_arbiter_group_t *group,
                     ucs_arbiter_elem_t *elem,
                     void *arg)
{
    uct_ud_ep_t *ep             = ucs_container_of(group, uct_ud_ep_t,
                                                   tx.pending.group);
    uct_ud_iface_t *iface       = ucs_container_of(arbiter, uct_ud_iface_t,
                                                   tx.pending_q);
    uintptr_t in_async_progress = (uintptr_t)arg;
    uct_pending_req_t *req;
    int allow_callback;
    int async_before_pending;
    ucs_status_t status;
    int is_last_pending_elem;

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
     */
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
     * - there are no high priority pending control messages
     */
    req            = ucs_container_of(elem, uct_pending_req_t, priv);
    allow_callback = !in_async_progress ||
                     (uct_ud_pending_req_priv(req)->flags & UCT_CB_FLAG_ASYNC);
    if (allow_callback && !uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_CTL_HI_PRIO)) {
        ucs_assert(!(ep->flags & UCT_UD_EP_FLAG_IN_PENDING));
        ep->flags |= UCT_UD_EP_FLAG_IN_PENDING;
        async_before_pending = iface->tx.async_before_pending;
        if (uct_ud_pending_req_priv(req)->flags & UCT_CB_FLAG_ASYNC) {
            /* temporary reset the flag to unblock sends from async context */
            iface->tx.async_before_pending = 0;
        }
        /* temporary reset `UCT_UD_EP_HAS_PENDING` flag to unblock sends */
        uct_ud_ep_remove_has_pending_flag(ep);

        is_last_pending_elem = uct_ud_ep_is_last_pending_elem(ep, elem);

        status = req->func(req);
#if UCS_ENABLE_ASSERT
        /* do not touch the request (or the arbiter element) after
         * calling the callback if UCS_OK is returned from the callback */
        if (status == UCS_OK) {
            req  = NULL;
            elem = NULL;
        }
#endif

        uct_ud_ep_set_has_pending_flag(ep);
        iface->tx.async_before_pending = async_before_pending;
        ep->flags &= ~UCT_UD_EP_FLAG_IN_PENDING;

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

        if (is_last_pending_elem) {
            uct_ud_ep_remove_has_pending_flag(ep);
        }

        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

    /* try to send ctl messages */
    uct_ud_ep_do_pending_ctl(ep, iface);
    if (in_async_progress) {
        return uct_ud_ep_ctl_op_next(ep);
    } else {
        /* we still didn't process the current pending request because of hi-prio
         * control messages, so cannot stop sending yet. If we stop, not all
         * resources will be exhausted and out-of-order with pending can occur.
         * (pending control ops may be cleared by uct_ud_ep_do_pending_ctl)
         */
        return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
    }
}

ucs_status_t uct_ud_ep_pending_add(uct_ep_h ep_h, uct_pending_req_t *req,
                                   unsigned flags)
{
    uct_ud_ep_t *ep       = ucs_derived_of(ep_h, uct_ud_ep_t);
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                           uct_ud_iface_t);

    uct_ud_enter(iface);

    /* if there was an async progress all 'send' ops return
     * UCS_ERR_NO_RESOURCE. If we return UCS_ERR_BUSY there will
     * be a deadlock.
     * So we must skip a resource check and add a pending op in order to
     * avoid a deadlock.
     */
    if (ucs_unlikely(uct_ud_iface_has_pending_async_ev(iface))) {
        goto add_req;
    }

    if (uct_ud_iface_can_tx(iface) &&
        uct_ud_iface_has_skbs(iface) &&
        uct_ud_ep_is_connected_and_no_pending(ep) &&
        !uct_ud_ep_no_window(ep)) {

        uct_ud_leave(iface);
        return UCS_ERR_BUSY;
    }

add_req:
    UCS_STATIC_ASSERT(sizeof(uct_ud_pending_req_priv_t) <=
                      UCT_PENDING_REQ_PRIV_LEN);
    uct_ud_pending_req_priv(req)->flags = flags;
    uct_ud_ep_set_has_pending_flag(ep);
    uct_pending_req_arb_group_push(&ep->tx.pending.group, req);
    ucs_arbiter_group_schedule(&iface->tx.pending_q, &ep->tx.pending.group);
    ucs_trace_data("ud ep %p: added pending req %p tx_psn %d acked_psn %d cwnd %d",
                   ep, req, ep->tx.psn, ep->tx.acked_psn, ep->ca.cwnd);
    UCT_TL_EP_STAT_PEND(&ep->super);

    uct_ud_leave(iface);
    return UCS_OK;
}

static ucs_arbiter_cb_result_t
uct_ud_ep_pending_purge_cb(ucs_arbiter_t *arbiter, ucs_arbiter_group_t *group,
                           ucs_arbiter_elem_t *elem, void *arg)
{
    uct_ud_ep_t *ep                 = ucs_container_of(group, uct_ud_ep_t,
                                                       tx.pending.group);
    uct_purge_cb_args_t *cb_args    = arg;
    uct_pending_purge_callback_t cb = cb_args->cb;
    uct_pending_req_t *req;
    int is_last_pending_elem;

    if (&ep->tx.pending.elem == elem) {
        /* return ignored by arbiter */
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

    is_last_pending_elem = uct_ud_ep_is_last_pending_elem(ep, elem);

    req = ucs_container_of(elem, uct_pending_req_t, priv);
    if (cb) {
        cb(req, cb_args->arg);
    } else {
        ucs_debug("ep=%p cancelling user pending request %p", ep, req);
    }

    if (is_last_pending_elem) {
        uct_ud_ep_remove_has_pending_flag(ep);
    }

    /* return ignored by arbiter */
    return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
}


void uct_ud_ep_pending_purge(uct_ep_h ep_h, uct_pending_purge_callback_t cb,
                             void *arg)
{
    uct_ud_ep_t *ep          = ucs_derived_of(ep_h, uct_ud_ep_t);
    uct_ud_iface_t *iface    = ucs_derived_of(ep->super.super.iface,
                                              uct_ud_iface_t);
    uct_purge_cb_args_t args = {cb, arg};

    uct_ud_enter(iface);
    ucs_arbiter_group_purge(&iface->tx.pending_q, &ep->tx.pending.group,
                            uct_ud_ep_pending_purge_cb, &args);
    if (uct_ud_ep_ctl_op_isany(ep)) {
        uct_ud_ep_ctl_op_schedule(iface, ep);
    }
    uct_ud_leave(iface);
}

void uct_ud_ep_disconnect(uct_ep_h tl_ep)
{
    uct_ud_ep_t    *ep    = ucs_derived_of(tl_ep, uct_ud_ep_t);
    uct_ud_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ud_iface_t);

    ucs_debug("ep %p: disconnect", ep);

    uct_ud_enter(iface);

    /* cancel user pending */
    uct_ud_ep_pending_purge(tl_ep, NULL, NULL);

    /* schedule flush */
    uct_ud_ep_flush(tl_ep, 0, NULL);

    /* the EP will be destroyed by interface destroy or timeout in
     * uct_ud_ep_timer
     */
    ep->close_time = ucs_twheel_get_time(&iface->tx.timer);
    ep->flags |= UCT_UD_EP_FLAG_DISCONNECTED;
    ucs_wtimer_add(&iface->tx.timer, &ep->timer,
                   UCT_UD_SLOW_TIMER_MAX_TICK(iface));

    uct_ud_leave(iface);
}
