/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "dc_ep.h"
#include "dc_iface.h"


UCS_CLASS_INIT_FUNC(uct_dc_ep_t, uct_dc_iface_t *iface, const uct_dc_iface_addr_t *if_addr)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super.super);
    ucs_arbiter_group_init(&self->arb_group);
    self->dci              = UCT_DC_EP_NO_DCI;
    self->state            = UCT_DC_EP_TX_OK;
    self->atomic_mr_offset = uct_ib_md_atomic_offset(if_addr->atomic_mr_id);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_ep_t)
{
    uct_dc_iface_t *iface = ucs_derived_of(self->super.super.iface, uct_dc_iface_t);

    uct_dc_ep_pending_purge(&self->super.super, NULL, NULL);
    ucs_arbiter_group_cleanup(&self->arb_group);

    if (self->dci == UCT_DC_EP_NO_DCI) {
        return;
    }

    /* TODO: this is good for dcs policy only.
     * Need to change if eps share dci
     */

    if (!uct_dc_iface_dci_has_outstanding(iface, self->dci)) {
        ucs_fatal("ifface (%p) ep (%p) dci leak detected: dci=%d",
                  iface, self, self->dci);
    }

    /* we can handle it but well behaving app should not do this */
    ucs_warn("ep (%p) is destroyed with %d outstanding ops",
             self, (int16_t)iface->super.config.tx_qp_len - 
             uct_rc_txqp_available(&iface->tx.dcis[self->dci].txqp));
    uct_rc_txqp_purge_outstanding(&iface->tx.dcis[self->dci].txqp, UCS_ERR_CANCELED, 1);
    iface->tx.dcis[self->dci].ep = NULL;
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
            if (uct_dc_iface_dci_can_alloc(iface)) {
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
        ucs_arbiter_group_schedule(uct_dc_iface_dci_waitq(iface), &ep->arb_group);
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
         *
         * note: arbiter removes elem _after_ dispatch, so a little hack is
         * used
         */ 
        if (elem == ep->arb_group.tail) {
            uct_dc_iface_dci_free(iface, ep);
        }
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }
    if (status == UCS_INPROGRESS) {
        return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
    }
    if (!uct_dc_iface_dci_has_tx_resources(iface, ep->dci)) {
        /* TODO: only good for dcs policy */
        return UCS_ARBITER_CB_RESULT_DESCHED_GROUP;
    }
    if (ep->state == UCT_DC_EP_TX_WAIT) {
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

    cb(req, cb_args->arg);
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

UCS_CLASS_DEFINE(uct_dc_ep_t, uct_base_ep_t);
