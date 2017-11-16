/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "dc_ep.h"
#include "dc_iface.h"


UCS_CLASS_INIT_FUNC(uct_dc_ep_t, uct_dc_iface_t *iface, const uct_dc_iface_addr_t *if_addr)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super.super);

    self->atomic_mr_offset = uct_ib_md_atomic_offset(if_addr->atomic_mr_id);

    return uct_dc_ep_basic_init(iface, self);
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_ep_t)
{
    uct_dc_iface_t *iface = ucs_derived_of(self->super.super.iface, uct_dc_iface_t);

    uct_dc_ep_pending_purge(&self->super.super, NULL, NULL);
    ucs_arbiter_group_cleanup(&self->arb_group);
    uct_rc_fc_cleanup(&self->fc);

    ucs_assert_always(self->flags & UCT_DC_EP_FLAG_VALID);

    if (self->dci == UCT_DC_EP_NO_DCI) {
        return;
    }

    /* TODO: this is good for dcs policy only.
     * Need to change if eps share dci
     */
    ucs_assertv_always(uct_dc_iface_dci_has_outstanding(iface, self->dci),
                       "iface (%p) ep (%p) dci leak detected: dci=%d", iface,
                       self, self->dci);

    /* we can handle it but well behaving app should not do this */
    ucs_debug("ep (%p) is destroyed with %d outstanding ops",
              self, (int16_t)iface->super.config.tx_qp_len -
              uct_rc_txqp_available(&iface->tx.dcis[self->dci].txqp));
    uct_rc_txqp_purge_outstanding(&iface->tx.dcis[self->dci].txqp, UCS_ERR_CANCELED, 1);
    iface->tx.dcis[self->dci].ep     = NULL;
#if ENABLE_ASSERT
    iface->tx.dcis[self->dci].flags |= UCT_DC_DCI_FLAG_EP_DESTROYED;
#endif
}

void uct_dc_ep_cleanup(uct_ep_h tl_ep, ucs_class_t *cls)
{
    uct_dc_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_ep_t);
    uct_dc_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_dc_iface_t);

    UCS_CLASS_CLEANUP_CALL(cls, ep);

    if (uct_dc_ep_fc_wait_for_grant(ep)) {
        ucs_trace("not releasing dc_ep %p - waiting for grant", ep);
        ep->flags &= ~UCT_DC_EP_FLAG_VALID;
        ucs_list_add_tail(&iface->tx.gc_list, &ep->list);
    } else {
        ucs_free(ep);
    }
}

void uct_dc_ep_release(uct_dc_ep_t *ep)
{
    ucs_assert_always(!(ep->flags & UCT_DC_EP_FLAG_VALID));
    ucs_debug("release dc_ep %p", ep);
    ucs_list_del(&ep->list);
    ucs_free(ep);
}

/* TODO:
   currently pending code supports only dcs policy
   support hash/random policies
 */
ucs_status_t uct_dc_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *r)
{
    uct_dc_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_iface_t);
    uct_dc_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_ep_t);

    /* ep can tx iff
     * - iface has resources: cqe and tx skb
     * - dci is either assigned or can be assigned
     * - dci has resources
     */
    if (uct_rc_iface_has_tx_resources(&iface->super)) {
        if (ep->dci == UCT_DC_EP_NO_DCI) {
            if (uct_dc_iface_dci_can_alloc(iface) && (ep->fc.fc_wnd > 0)) {
                return UCS_ERR_BUSY;
            }
        } else {
            if (uct_dc_iface_dci_ep_can_send(ep)) {
                return UCS_ERR_BUSY;
            }
        }
    }

    UCS_STATIC_ASSERT(sizeof(ucs_arbiter_elem_t) <= UCT_PENDING_REQ_PRIV_LEN);
    ucs_arbiter_elem_init((ucs_arbiter_elem_t *)r->priv);

    /* no dci:
     *  Do not grab dci here. Instead put the group on dci allocation arbiter.
     *  This way we can assure fairness between all eps waiting for
     *  dci allocation.
     */
    if (ep->dci == UCT_DC_EP_NO_DCI) {
        ucs_arbiter_group_push_elem(&ep->arb_group, (ucs_arbiter_elem_t*)r->priv);
        uct_dc_iface_schedule_dci_alloc(iface, ep);
        return UCS_OK;
    }

    ucs_arbiter_group_push_elem(&ep->arb_group, (ucs_arbiter_elem_t*)r->priv);
    uct_dc_iface_dci_sched_tx(iface, ep);
    return UCS_OK;
}

/**
 * dispatch requests waiting for dci allocation
 */
ucs_arbiter_cb_result_t
uct_dc_iface_dci_do_pending_wait(ucs_arbiter_t *arbiter,
                                 ucs_arbiter_elem_t *elem,
                                 void *arg)
{
    uct_dc_ep_t *ep = ucs_container_of(ucs_arbiter_elem_group(elem), uct_dc_ep_t, arb_group);
    uct_dc_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_dc_iface_t);

    /**
     * stop if dci can not be allocated
     * else move group to the dci arbiter
     */
    ucs_assert_always(ep->dci == UCT_DC_EP_NO_DCI);

    if (!uct_dc_iface_dci_can_alloc(iface)) {
        return UCS_ARBITER_CB_RESULT_STOP;
    }
    uct_dc_iface_dci_alloc(iface, ep);
    ucs_assert_always(ep->dci != UCT_DC_EP_NO_DCI);
    uct_dc_iface_dci_sched_tx(iface, ep);
    return UCS_ARBITER_CB_RESULT_DESCHED_GROUP;
}

/**
 * dispatch requests waiting for tx resources
 */
ucs_arbiter_cb_result_t
uct_dc_iface_dci_do_pending_tx(ucs_arbiter_t *arbiter,
                               ucs_arbiter_elem_t *elem,
                               void *arg)
{

    uct_dc_ep_t *ep = ucs_container_of(ucs_arbiter_elem_group(elem), uct_dc_ep_t, arb_group);
    uct_dc_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_dc_iface_t);
    uct_pending_req_t *req = ucs_container_of(elem, uct_pending_req_t, priv);
    ucs_status_t status;

    if (!uct_rc_iface_has_tx_resources(&iface->super)) {
        return UCS_ARBITER_CB_RESULT_STOP;
    }

    status = req->func(req);
    ucs_trace_data("progress pending request %p returned: %s", req,
                   ucs_status_string(status));
    if (status == UCS_OK) {
        /* Release dci if this is the last elem in the group and the dci has no
         * outstanding operations. For example pending callback did not send
         * anything. (uct_ep_flush or just return ok)
         */
        if (ucs_arbiter_elem_is_last(&ep->arb_group, elem)) {
            uct_dc_iface_dci_free(iface, ep);
        }
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }
    if (status == UCS_INPROGRESS) {
        return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
    }
    if (!uct_dc_iface_dci_ep_can_send(ep) ||
        uct_dc_ep_fc_wait_for_grant(ep)) {
        /* Deschedule the group even if FC is the only resource, which
         * is missing. It will be scheduled again when credits arrive. */

        /* TODO: only good for dcs policy */
        return UCS_ARBITER_CB_RESULT_DESCHED_GROUP;
    }

    ucs_assertv(!uct_rc_iface_has_tx_resources(&iface->super),
                "pending callback returned error but send resources are available");
    return UCS_ARBITER_CB_RESULT_STOP;
}


static ucs_arbiter_cb_result_t uct_dc_ep_abriter_purge_cb(ucs_arbiter_t *arbiter,
                                                          ucs_arbiter_elem_t *elem,
                                                          void *arg)
{
    uct_purge_cb_args_t  *cb_args   = arg;
    uct_pending_purge_callback_t cb = cb_args->cb;
    uct_pending_req_t *req          = ucs_container_of(elem, uct_pending_req_t, priv);
    uct_rc_fc_request_t *freq       = ucs_derived_of(req, uct_rc_fc_request_t);
    uct_dc_ep_t *ep                 = ucs_container_of(ucs_arbiter_elem_group(elem),
                                                       uct_dc_ep_t, arb_group);

    if (ucs_likely(req->func != uct_dc_iface_fc_grant)){
        if (cb != NULL) {
            cb(req, cb_args->arg);
        } else {
            ucs_debug("ep=%p cancelling user pending request %p", ep, req);
        }
    } else {
        /* User callback should not be called for FC messages.
         * Just return pending request memory to the pool */
        ucs_mpool_put(freq);
    }

    return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
}

void uct_dc_ep_pending_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb, void *arg)
{
    uct_dc_iface_t *iface    = ucs_derived_of(tl_ep->iface, uct_dc_iface_t);
    uct_dc_ep_t *ep          = ucs_derived_of(tl_ep, uct_dc_ep_t);
    uct_purge_cb_args_t args = {cb, arg};

    if (ep->dci == UCT_DC_EP_NO_DCI) {
        ucs_arbiter_group_purge(uct_dc_iface_dci_waitq(iface), &ep->arb_group,
                                uct_dc_ep_abriter_purge_cb, &args);
    } else {
        ucs_arbiter_group_purge(uct_dc_iface_tx_waitq(iface), &ep->arb_group,
                                uct_dc_ep_abriter_purge_cb, &args);
        uct_dc_iface_dci_free(iface, ep);
    }
}

ucs_status_t uct_dc_ep_flush(uct_ep_h tl_ep, unsigned flags, uct_completion_t *comp)
{
    uct_dc_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_iface_t);
    uct_dc_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_ep_t);
    ucs_status_t status;

    if (ucs_unlikely(flags & UCT_FLUSH_FLAG_CANCEL)) {
        if (ep->dci != UCT_DC_EP_NO_DCI) {
            uct_rc_txqp_purge_outstanding(&iface->tx.dcis[ep->dci].txqp,
                                          UCS_ERR_CANCELED, 0);
#if ENABLE_ASSERT
            iface->tx.dcis[ep->dci].flags |= UCT_DC_DCI_FLAG_EP_CANCELED;
#endif
        }

        uct_ep_pending_purge(tl_ep, NULL, 0);
        return UCS_OK;
    }

    /* If waiting for FC grant, return NO_RESOURCE to prevent ep destruction.
     * Otherwise grant for destroyed ep will arrive and there will be a
     * segfault when we will try to access the ep by address from the grant
     * message. */
    if (!uct_rc_iface_has_tx_resources(&iface->super)) {
        return UCS_ERR_NO_RESOURCE;
    }

    if (ep->dci == UCT_DC_EP_NO_DCI) {
        if (!uct_dc_iface_dci_can_alloc(iface)) {
            return UCS_ERR_NO_RESOURCE; /* waiting for dci */
        } else {
            UCT_TL_EP_STAT_FLUSH(&ep->super); /* no sends */
            return UCS_OK;
        }
    }

    if (!uct_dc_iface_dci_ep_can_send(ep)) {
        return UCS_ERR_NO_RESOURCE; /* cannot send */
    }

    status = uct_dc_iface_flush_dci(iface, ep->dci);
    if (status == UCS_OK) {
        UCT_TL_EP_STAT_FLUSH(&ep->super);
        return UCS_OK; /* all sends completed */
    }

    ucs_assert(status == UCS_INPROGRESS);
    UCT_TL_EP_STAT_FLUSH_WAIT(&ep->super);
    return UCS_INPROGRESS;
}

ucs_status_t uct_dc_ep_check_fc(uct_dc_iface_t *iface, uct_dc_ep_t *ep)
{
    ucs_status_t status;

    if (iface->super.config.fc_enabled) {
        UCT_RC_CHECK_FC_WND(&ep->fc, ep->super.stats);
        if ((ep->fc.fc_wnd == iface->super.config.fc_hard_thresh) &&
            !(ep->fc.flags & UCT_DC_EP_FC_FLAG_WAIT_FOR_GRANT)) {
            status = uct_rc_fc_ctrl(&ep->super.super,
                                    UCT_RC_EP_FC_FLAG_HARD_REQ,
                                    NULL);
            if (status != UCS_OK) {
                return status;
            }
            ep->fc.flags |= UCT_DC_EP_FC_FLAG_WAIT_FOR_GRANT;
        }
    } else {
        /* Set fc_wnd to max, to send as much as possible without checks */
        ep->fc.fc_wnd = INT16_MAX;
    }
    return UCS_OK;
}

UCS_CLASS_DEFINE(uct_dc_ep_t, uct_base_ep_t);
