/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "srd_ep.h"
#include "srd_log.h"
#include "srd_iface.h"

#include <uct/ib/base/ib_log.h>


int uct_srd_ep_is_connected(const uct_ep_h tl_ep,
                            const uct_ep_is_connected_params_t *params)
{
    uct_srd_ep_t *ep                    = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_ib_iface_t *ib_iface            = ucs_derived_of(tl_ep->iface,
                                                         uct_ib_iface_t);
    const uct_ib_address_t *ib_addr     = (const uct_ib_address_t*)
                                                  params->device_addr;
    const uct_srd_iface_addr_t *if_addr = (const uct_srd_iface_addr_t*)
                                                  params->iface_addr;

    UCT_EP_IS_CONNECTED_CHECK_DEV_IFACE_ADDRS(params);

    if (ep->dest_qpn != uct_ib_unpack_uint24(if_addr->qp_num)) {
        return 0;
    }

    return uct_ib_iface_is_connected(ib_iface, ib_addr, ep->path_index, ep->ah);
}

static UCS_CLASS_INIT_FUNC(uct_srd_ep_t, const uct_ep_params_t *params)
{
    uct_srd_iface_t *iface              = ucs_derived_of(params->iface,
                                                         uct_srd_iface_t);
    const uct_ib_address_t *ib_addr     = (const uct_ib_address_t*)
                                                  params->dev_addr;
    const uct_srd_iface_addr_t *if_addr = (const uct_srd_iface_addr_t*)
                                                  params->iface_addr;
    char str[128], buf[128];
    ucs_status_t status;
    enum ibv_mtu path_mtu;
    struct ibv_ah_attr ah_attr;

    ucs_trace_func("");

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    self->ep_uuid    = ucs_generate_uuid((uintptr_t)self);
    self->path_index = UCT_EP_PARAMS_GET_PATH_INDEX(params);
    self->psn        = UCT_SRD_INITIAL_PSN;
    self->flags      = 0;
    self->dest_qpn   = uct_ib_unpack_uint24(if_addr->qp_num);

    ucs_arbiter_group_init(&self->pending_group);
    ucs_list_head_init(&self->outstanding_list);

    uct_ib_iface_fill_ah_attr_from_addr(&iface->super, ib_addr,
                                        self->path_index, &ah_attr, &path_mtu);
    status = uct_ib_iface_create_ah(&iface->super, &ah_attr, "SRD AH",
                                    &self->ah);
    if (status != UCS_OK) {
        goto err_arb_cleanup;
    }

    status = uct_srd_iface_add_ep(iface, self);
    if (status != UCS_OK) {
        goto err_arb_cleanup;
    }

    status = uct_srd_iface_ctl_add(iface, UCT_SRD_CTL_ID_REQ, self->ep_uuid,
                                   self->ah, self->dest_qpn);
    if (status != UCS_OK) {
        goto err_remove_ep;
    }

    ucs_debug(UCT_IB_IFACE_FMT
              " ep=%p gid=%s qpn=0x%x ep_uuid=0x%"PRIx64" connected "
              "to iface %s qpn=0x%x",
              UCT_IB_IFACE_ARG(&iface->super),
              self, uct_ib_gid_str(&iface->super.gid_info.gid, str, sizeof(str)),
              iface->qp->qp_num, self->ep_uuid,
              uct_ib_address_str(ib_addr, buf, sizeof(buf)),
              uct_ib_unpack_uint24(if_addr->qp_num));
    return UCS_OK;

err_remove_ep:
    uct_srd_iface_remove_ep(iface, self);
err_arb_cleanup:
    ucs_arbiter_group_cleanup(&self->pending_group);
    return status;
}

static void uct_srd_ep_send_op_user_completion(uct_srd_send_op_t *send_op,
                                               ucs_status_t status)
{
    if (send_op->user_comp != NULL) {
        uct_invoke_completion(send_op->user_comp, status);
    }
}

static void uct_srd_ep_send_op_flush_completion(uct_srd_send_op_t *send_op,
                                                ucs_status_t status)
{
    uct_srd_ep_send_op_user_completion(send_op, status);
}

void uct_srd_ep_send_op_completion(uct_srd_send_op_t *send_op)
{
    uct_srd_ep_t *ep = send_op->ep;
    uct_srd_iface_t *iface;
    uct_srd_send_op_t *flush_op;
    ucs_status_t comp_status;

    if (ep == NULL) {
        ucs_list_del(&send_op->list);
        ucs_mpool_put(send_op);
        return;
    }

    iface       = ucs_derived_of(ep->super.super.iface, uct_srd_iface_t);
    comp_status = (ep->flags & UCT_SRD_EP_FLAG_CANCELED)?
                  UCS_ERR_CANCELED : UCS_OK;

    /* Call user completion before changing any transmit state */
    send_op->comp_cb(send_op, comp_status);

    /*
     * If any flush_op, it holds TX resource, so this release below cannot
     * change transmit state.
     */
    ucs_list_del(&send_op->list);
    iface->tx.ep_outstanding--;

    /* Trigger all front line flush completions */
    while (!ucs_list_is_empty(&ep->outstanding_list)) {
        flush_op = ucs_list_head(&ep->outstanding_list, uct_srd_send_op_t,
                                 list);

        ucs_assertv(ep == flush_op->ep, "ep=%p send_op=%p flush_op_ep=%p",
                    ep, send_op, flush_op->ep);

        if (flush_op->comp_cb != uct_srd_ep_send_op_flush_completion) {
            break; /* Not a flush operation */
        }

        /*
         * Flush(CANCEL) with see itself reported as CANCEL. Flush()
         * followed immediately by Flush(CANCEL) will both be reported as
         * CANCEL.
         *
         * State update in iface and ep must be done after user completion.
         */
        flush_op->comp_cb(flush_op, comp_status);
        ucs_list_del(&flush_op->list);
        iface->tx.ep_outstanding--;
        ucs_mpool_put(flush_op);
    }

    /*
     * In case of any fence unblocking, try running pending before any
     * future user completion under the same progress.
     */
    if (((ep->flags & UCT_SRD_EP_FLAG_FENCE) &&
         ucs_list_is_empty(&ep->outstanding_list)) ||
        ((iface->flags & UCT_SRD_IFACE_FENCE) &&
         (iface->tx.ep_outstanding == 0))) {
        uct_srd_iface_pending_ctl_progress(iface);
    }

    ucs_mpool_put(send_op);
}

void uct_srd_ep_send_op_purge(uct_srd_ep_t *ep)
{
    uct_srd_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_srd_iface_t);
    uct_srd_send_op_t *send_op, *next;

    ucs_list_for_each_safe(send_op, next, &ep->outstanding_list, list) {
        ucs_assertv(send_op->ep == ep, "send_op_ep=%p ep=%p", send_op->ep, ep);

        /*
         * Make ep invalid, as ibv_poll_cq() will return this
         * send_op after it has been released.
         */
        send_op->comp_cb(send_op, UCS_ERR_CANCELED);

        /*
         * Destroy flush operation right away as ibv_poll_cq() will not return
         * a reference to the corresponding send_op.
         */
        if (send_op->comp_cb == uct_srd_ep_send_op_flush_completion) {
            ucs_list_del(&send_op->list);
            ucs_mpool_put(send_op);
        } else {
            send_op->ep = NULL;
        }

        iface->tx.ep_outstanding--;
    }

    ucs_list_splice_tail(&iface->tx.outstanding_list, &ep->outstanding_list);
    ucs_list_head_init(&ep->outstanding_list);
}

static UCS_F_ALWAYS_INLINE int
uct_srd_ep_can_tx(uct_srd_ep_t *ep, uct_srd_iface_t *iface)
{
    return /* AH adding ACK received if we have to wait for it */
           (ucs_likely(ep->flags & UCT_SRD_EP_FLAG_AH_ADDED) ||
            !(ep->flags & UCT_SRD_EP_FLAG_RMA)) &&
           /* TX queue available to post to QP */
           uct_srd_iface_can_tx(iface) &&
           /* Interface level fencing */
           (!(iface->flags & UCT_SRD_IFACE_FENCE) ||
            (iface->tx.ep_outstanding == 0)) &&
           /* Endpoint fencing */
           (!(ep->flags & UCT_SRD_EP_FLAG_FENCE) ||
             ucs_list_is_empty(&ep->outstanding_list));
}

ucs_status_t
uct_srd_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *req, unsigned flags)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);

    if (uct_srd_ep_can_tx(ep, iface)) {
        return UCS_ERR_BUSY;
    }

    /*
     * To avoid following case:
     *  - AM short without resource available, pending_add but not scheduled
     *    as there is no AH added ACK, resources become available, and later
     *    AM short erroneously succeeds.
     */
    ep->flags |= UCT_SRD_EP_FLAG_RMA;

    uct_pending_req_arb_group_push(&ep->pending_group, req);
    if (ucs_likely(ep->flags & UCT_SRD_EP_FLAG_AH_ADDED)) {
        ucs_arbiter_group_schedule(&iface->tx.pending_q, &ep->pending_group);
    }

    ucs_trace_data("iface=%p ep=%p: added pending req=%p psn=%u ah_added=%d",
                   iface, ep, req, ep->psn,
                   !!(ep->flags & UCT_SRD_EP_FLAG_AH_ADDED));
    UCT_TL_EP_STAT_PEND(&ep->super);
    return UCS_OK;
}

ucs_arbiter_cb_result_t
uct_srd_ep_do_pending(ucs_arbiter_t *arbiter, ucs_arbiter_group_t *group,
                      ucs_arbiter_elem_t *elem, void *arg)
{
    uct_srd_ep_t *ep       = ucs_container_of(group, uct_srd_ep_t,
                                              pending_group);
    uct_srd_iface_t *iface = ucs_container_of(arbiter, uct_srd_iface_t,
                                              tx.pending_q);
    uct_pending_req_t *req = ucs_container_of(elem, uct_pending_req_t, priv);
    ucs_status_t status;

    if (!uct_srd_ep_can_tx(ep, iface)) {
        if (ucs_unlikely(!(ep->flags & UCT_SRD_EP_FLAG_AH_ADDED))) {
            return UCS_ARBITER_CB_RESULT_DESCHED_GROUP;
        }

        return UCS_ARBITER_CB_RESULT_STOP;
    }

    ucs_trace_data("iface=%p ep=%p progressing pending request %p", iface, ep,
                   req);
#if UCS_ENABLE_ASSERT
    iface->tx.in_pending = 1;
#endif
    status               = req->func(req);
#if UCS_ENABLE_ASSERT
    iface->tx.in_pending = 0;
#endif
    ucs_trace_data("iface=%p ep=%p status returned from progress pending: %s",
                   iface, ep, ucs_status_string(status));

    if (status == UCS_OK) {
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    } else if (status == UCS_INPROGRESS) {
        return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
    }

    return UCS_ARBITER_CB_RESULT_RESCHED_GROUP;
}

static ucs_arbiter_cb_result_t
uct_srd_ep_pending_purge_cb(ucs_arbiter_t *arbiter, ucs_arbiter_group_t *group,
                            ucs_arbiter_elem_t *elem, void *priv)
{
    uct_purge_cb_args_t *cb_args = priv;
    uct_pending_req_t *req       = ucs_container_of(elem, uct_pending_req_t,
                                                    priv);

    if (cb_args->cb != NULL) {
        cb_args->cb(req, cb_args->arg);
    }

    return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
}

void uct_srd_ep_pending_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb,
                              void *cb_arg)
{
    uct_srd_ep_t *ep         = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface   = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_purge_cb_args_t args = {cb, cb_arg};

    ucs_trace_func("ep=%p cb=%p cb_arg=%p", ep, cb, cb_arg);

    ucs_arbiter_group_purge(&iface->tx.pending_q, &ep->pending_group,
                            uct_srd_ep_pending_purge_cb, &args);
}

static UCS_CLASS_CLEANUP_FUNC(uct_srd_ep_t)
{
    uct_srd_iface_t *iface = ucs_derived_of(self->super.super.iface,
                                            uct_srd_iface_t);

    ucs_trace_func("");

    uct_srd_ep_send_op_purge(self);
    uct_srd_ep_pending_purge(&self->super.super, NULL, NULL);
    uct_srd_iface_remove_ep(iface, self);
    ucs_arbiter_group_cleanup(&self->pending_group);
}

UCS_CLASS_DEFINE(uct_srd_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_srd_ep_t, uct_ep_t, const uct_ep_params_t*);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_srd_ep_t, uct_ep_t);

static UCS_F_ALWAYS_INLINE void uct_srd_ep_posted(uct_srd_iface_t *iface,
                                                  uct_srd_ep_t *ep,
                                                  uct_srd_send_op_t *send_op)
{
    ucs_assertv((!(iface->flags & UCT_SRD_IFACE_FENCE) ||
                 (iface->tx.ep_outstanding == 0)) &&
                (!(ep->flags & UCT_SRD_EP_FLAG_FENCE) ||
                 ucs_list_is_empty(&ep->outstanding_list)),
                "iface=%p ep=%p iface_flags=%x ep_flags=%x ep_outstanding=%d "
                "outstanding_is_empty=%d",
                iface, ep, iface->flags, ep->flags, iface->tx.ep_outstanding,
                ucs_list_is_empty(&ep->outstanding_list));

    ucs_list_add_tail(&ep->outstanding_list, &send_op->list);
    iface->flags &= ~UCT_SRD_IFACE_FENCE;
    ep->flags    &= ~UCT_SRD_EP_FLAG_FENCE;
    iface->tx.ep_outstanding++;
    iface->tx.available--;
}

static UCS_F_ALWAYS_INLINE void
uct_srd_ep_hdr_set(const uct_srd_ep_t *ep, uct_srd_hdr_t *neth, uint8_t id)
{
    neth->ep_uuid = ep->ep_uuid;
    neth->psn     = ep->psn;
    neth->id      = id;
}

#define UCT_SRD_CHECK_RES_OR_RETURN(_ep, _iface) \
    if (!uct_srd_ep_can_tx(_ep, _iface)) { \
        UCS_STATS_UPDATE_COUNTER((_ep)->super.stats, UCT_EP_STAT_NO_RES, 1); \
        return UCS_ERR_NO_RESOURCE; \
    } \
    uct_srd_iface_check_pending(_iface, &(_ep)->pending_group);

#define UCT_SRD_CHECK_RMA_RES_OR_RETURN(_ep, _iface) \
    (_ep)->flags |= UCT_SRD_EP_FLAG_RMA; \
    UCT_SRD_CHECK_RES_OR_RETURN(_ep, _iface)

static UCS_F_ALWAYS_INLINE ucs_status_t uct_srd_ep_am_short_prepare(
        uct_srd_iface_t *iface, uct_srd_ep_t *ep, uint8_t id, size_t am_length)
{
    uct_srd_am_short_hdr_t *am = &iface->tx.am_inl_hdr;
    uct_srd_send_op_t *send_op;

    UCT_SRD_CHECK_RES_OR_RETURN(ep, iface);

    /* Use an internal send_op for am_short to track completion ordering */
    send_op = ucs_mpool_get(&iface->tx.send_op_mp);
    if (send_op == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    uct_srd_ep_hdr_set(ep, &am->srd_hdr, id);
    send_op->comp_cb        = (uct_srd_send_op_comp_t)ucs_empty_function;
    send_op->ep             = ep;
    iface->tx.sge[0].addr   = (uintptr_t)am;
    iface->tx.sge[0].length = am_length;
    iface->tx.wr_inl.wr_id  = (uintptr_t)send_op;

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
uct_srd_ep_post_send(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                     struct ibv_send_wr *wr, unsigned send_flags)
{
    uct_srd_send_op_t *send_op = (uct_srd_send_op_t*)wr->wr_id;

    uct_srd_iface_post_send(iface, ep->ah, ep->dest_qpn, wr, send_flags);
    ep->psn++;
    uct_srd_ep_posted(iface, ep, send_op);
}

static UCS_F_ALWAYS_INLINE void uct_srd_ep_am_short_post(uct_srd_iface_t *iface,
                                                         uct_srd_ep_t *ep,
                                                         size_t length)
{
    uct_srd_ep_post_send(iface, ep, &iface->tx.wr_inl, IBV_SEND_INLINE);
    UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, length);
}

ucs_status_t uct_srd_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                 const void *buffer, unsigned length)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    ucs_status_t status;

    UCT_SRD_CHECK_AM_SHORT(iface, id, sizeof(hdr), length);

    status = uct_srd_ep_am_short_prepare(iface, ep, id,
                                         sizeof(uct_srd_am_short_hdr_t));
    if (status != UCS_OK) {
        return status;
    }

    iface->tx.am_inl_hdr.am_hdr = hdr;

    iface->tx.sge[1].addr    = (uintptr_t)buffer;
    iface->tx.sge[1].length  = length;
    iface->tx.wr_inl.num_sge = 2;

    uct_srd_ep_am_short_post(iface, ep, sizeof(hdr) + length);
    return UCS_OK;
}

ucs_status_t uct_srd_ep_am_short_iov(uct_ep_h tl_ep, uint8_t id,
                                     const uct_iov_t *iov, size_t iovcnt)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    size_t length          = uct_iov_total_length(iov, iovcnt);
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt,
                       ucs_min(UCT_IB_MAX_IOV, iface->config.max_send_sge) - 1,
                       "uct_srd_ep_am_short_iov");
    UCT_SRD_CHECK_AM_SHORT(iface, id, 0, length);

    status = uct_srd_ep_am_short_prepare(iface, ep, id, sizeof(uct_srd_hdr_t));
    if (status != UCS_OK) {
        return status;
    }

    iface->tx.wr_inl.num_sge = 1 + uct_ib_verbs_sge_fill_iov(iface->tx.sge + 1,
                                                             iov, iovcnt);

    uct_srd_ep_am_short_post(iface, ep, length);
    return UCS_OK;
}

static void uct_srd_ep_send_desc_tx(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                                    uct_srd_send_desc_t *desc, size_t length)
{
    iface->tx.sge[0].addr   = (uintptr_t)(desc + 1);
    iface->tx.sge[0].length = sizeof(uct_srd_hdr_t) + length;
    iface->tx.sge[0].lkey   = desc->lkey;
    iface->tx.wr_desc.wr_id = (uintptr_t)&desc->super;

    uct_srd_ep_post_send(iface, ep, &iface->tx.wr_desc, 0);
}

ucs_status_t uct_srd_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                 unsigned header_length, const uct_iov_t *iov,
                                 size_t iovcnt, unsigned flags,
                                 uct_completion_t *comp)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_srd_hdr_t *hdr;
    size_t length;
    uct_srd_send_desc_t *desc;

    UCT_CHECK_IOV_SIZE(iovcnt,
                       ucs_min(UCT_IB_MAX_IOV, iface->config.max_send_sge) - 1,
                       "uct_srd_ep_am_zcopy");
    length = uct_iov_total_length(iov, iovcnt);

    UCT_SRD_CHECK_AM_ZCOPY(iface, id, header_length, length);
    UCT_SRD_CHECK_RES_OR_RETURN(ep, iface);

    desc = uct_srd_iface_get_send_desc(iface);
    if (desc == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    desc->super.user_comp = comp;
    desc->super.comp_cb   = uct_srd_ep_send_op_user_completion;
    desc->super.ep        = ep;

    hdr = (uct_srd_hdr_t*)(desc + 1);
    uct_srd_ep_hdr_set(ep, hdr, id);
    memcpy(hdr + 1, header, header_length);
    iface->tx.wr_desc.num_sge = 1 + uct_ib_verbs_sge_fill_iov(iface->tx.sge + 1,
                                                              iov, iovcnt);
    uct_srd_ep_send_desc_tx(iface, ep, desc, header_length);

    UCT_TL_EP_STAT_OP(&ep->super, AM, ZCOPY, header_length + length);
    return UCS_INPROGRESS;
}

ssize_t uct_srd_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                            uct_pack_callback_t pack_cb, void *arg,
                            unsigned flags)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_srd_send_desc_t *desc;
    size_t length;
    uct_srd_hdr_t *hdr;

    UCT_CHECK_AM_ID(id);
    UCT_SRD_CHECK_RES_OR_RETURN(ep, iface);

    desc = uct_srd_iface_get_send_desc(iface);
    if (desc == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    hdr    = (uct_srd_hdr_t*)(desc + 1);
    length = pack_cb(hdr + 1, arg);
    ucs_assertv((sizeof(uct_srd_hdr_t) + length) <=
                    iface->super.config.seg_size,
                "iface=%p ep=%p am_bcopy_length=%zu seg_size=%d", iface, ep,
                sizeof(uct_srd_hdr_t) + length, iface->super.config.seg_size);

    desc->super.comp_cb = (uct_srd_send_op_comp_t)ucs_empty_function;
    desc->super.ep      = ep;

    uct_srd_ep_hdr_set(ep, hdr, id);
    iface->tx.wr_desc.num_sge = 1;
    uct_srd_ep_send_desc_tx(iface, ep, desc, length);

    UCT_TL_EP_STAT_OP(&ep->super, AM, BCOPY, length);
    return length;
}

#define UCT_SRD_EP_LOG_RMA(_iface, _ep, _send_op, _sge, _num_sge, \
                           _remote_addr, _rkey) \
    if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        struct ibv_send_wr __wr = { \
            .wr_id               = (uintptr_t)(_send_op), \
            .send_flags          = 0, \
            .opcode              = IBV_WR_RDMA_READ, \
            .wr.rdma.remote_addr = (_remote_addr), \
            .wr.rdma.rkey        = (_rkey), \
            .sg_list             = (struct ibv_sge*)(_sge), \
            .num_sge             = (_num_sge), \
            .next                = NULL \
        }; \
        __uct_ib_log_post_send_one(__FILE__, __LINE__, __func__, \
                                   &(_iface)->super, (_iface)->qp, &__wr, \
                                   (_ep)->ah, (_ep)->dest_qpn, 2, NULL); \
    }

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_srd_ep_post_rma(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                    uct_srd_send_op_t *send_op, const struct ibv_sge *sge,
                    size_t num_sge, uint64_t remote_addr, uct_rkey_t rkey)
{
#ifdef HAVE_DECL_EFADV_DEVICE_ATTR_CAPS_RDMA_READ
    struct ibv_qp_ex *qp_ex = iface->qp_ex;

    UCT_SRD_EP_LOG_RMA(iface, ep, send_op, sge, num_sge, remote_addr, rkey);

    ibv_wr_start(qp_ex);
    qp_ex->wr_id = (uintptr_t)send_op;
    ibv_wr_rdma_read(qp_ex, rkey, remote_addr);
    ibv_wr_set_sge_list(qp_ex, num_sge, sge);
    ibv_wr_set_ud_addr(qp_ex, ep->ah, ep->dest_qpn, UCT_IB_KEY);
    if (ibv_wr_complete(qp_ex)) {
        ucs_fatal("ibv_wr_complete failed %m");
        return UCS_ERR_IO_ERROR;
    }

    uct_srd_ep_posted(iface, ep, send_op);
    return UCS_INPROGRESS;
#else
    ucs_mpool_put(send_op);
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_srd_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                  size_t iovcnt, uint64_t remote_addr,
                                  uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    size_t length          = uct_iov_total_length(iov, iovcnt);
    size_t num_sge;
    uct_srd_send_op_t *send_op;
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt, iface->config.max_recv_sge,
                       "uct_srd_ep_get_zcopy");

    UCT_CHECK_LENGTH(length, iface->super.config.max_inl_cqe[UCT_IB_DIR_TX] + 1,
                     iface->config.max_get_zcopy, "get_zcopy");
    UCT_SRD_CHECK_RMA_RES_OR_RETURN(ep, iface);

    send_op = ucs_mpool_get(&iface->tx.send_op_mp);
    if (send_op == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    send_op->user_comp = comp;
    send_op->comp_cb   = uct_srd_ep_send_op_user_completion;
    send_op->ep        = ep;

    num_sge = uct_ib_verbs_sge_fill_iov(iface->tx.sge, iov, iovcnt);
    status  = uct_srd_ep_post_rma(iface, ep, send_op, iface->tx.sge, num_sge,
                                  remote_addr, uct_ib_md_direct_rkey(rkey));
    if (!UCS_STATUS_IS_ERR(status)) {
        UCT_TL_EP_STAT_OP(&ep->super, GET, ZCOPY, length);
    }

    return status;
}

static void uct_srd_send_op_get_bcopy_completion(uct_srd_send_op_t *send_op,
                                                 ucs_status_t status)
{
    uct_srd_send_desc_t *desc = ucs_derived_of(send_op, uct_srd_send_desc_t);

    if (status == UCS_OK) {
        desc->unpack_cb(desc->unpack_arg, desc + 1, desc->length);
    }

    uct_srd_ep_send_op_user_completion(send_op, status);
}

ucs_status_t uct_srd_ep_get_bcopy(uct_ep_h tl_ep,
                                  uct_unpack_callback_t unpack_cb, void *arg,
                                  size_t length, uint64_t remote_addr,
                                  uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_srd_send_desc_t *desc;
    ucs_status_t status;

    UCT_CHECK_LENGTH(length, 0, iface->config.max_get_bcopy, "get_bcopy");
    UCT_SRD_CHECK_RMA_RES_OR_RETURN(ep, iface);

    desc = uct_srd_iface_get_send_desc(iface);
    if (desc == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    desc->unpack_arg      = arg;
    desc->unpack_cb       = unpack_cb;
    desc->length          = length;
    desc->super.user_comp = comp;
    desc->super.comp_cb   = uct_srd_send_op_get_bcopy_completion;
    desc->super.ep        = ep;

    iface->tx.sge[0].lkey   = desc->lkey;
    iface->tx.sge[0].length = length;
    iface->tx.sge[0].addr   = (uintptr_t)(desc + 1);

    status = uct_srd_ep_post_rma(iface, ep, &desc->super, iface->tx.sge, 1,
                                 remote_addr, uct_ib_md_direct_rkey(rkey));
    if (!UCS_STATUS_IS_ERR(status)) {
        UCT_TL_EP_STAT_OP(&ep->super, GET, BCOPY, length);
    }

    return status;
}

ucs_status_t uct_srd_ep_fence(uct_ep_h tl_ep, unsigned flags)
{
    uct_srd_ep_t *ep = ucs_derived_of(tl_ep, uct_srd_ep_t);

    ep->flags |= UCT_SRD_EP_FLAG_FENCE;
    UCT_TL_EP_STAT_FENCE(&ep->super);
    return UCS_OK;
}

ucs_status_t
uct_srd_ep_flush(uct_ep_h tl_ep, unsigned flags, uct_completion_t *comp)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_srd_iface_t);
    uct_srd_send_op_t *send_op;

    if (ucs_unlikely(flags & UCT_FLUSH_FLAG_CANCEL)) {
        /* Empty the pending */
        uct_srd_ep_pending_purge(tl_ep, NULL, 0);

        /*
         * Cancel the endpoint, but flush send_op will still wait for existing
         * outstanding operations to complete, as the device could be still be
         * using user resources (zcopy send or receive).
         */
        ep->flags |= UCT_SRD_EP_FLAG_CANCELED;
    }

    if (ucs_list_is_empty(&ep->outstanding_list)) {
        UCT_TL_EP_STAT_FLUSH(&ep->super);
        return UCS_OK;
    }

    if (comp == NULL) {
        goto out;
    }

    send_op = ucs_mpool_get(&iface->tx.send_op_mp);
    if (send_op == NULL) {
        return UCS_ERR_NO_RESOURCE;
    }

    send_op->ep        = ep;
    send_op->user_comp = comp;
    send_op->comp_cb   = uct_srd_ep_send_op_flush_completion;
    ucs_list_add_tail(&ep->outstanding_list, &send_op->list);
    iface->flags &= ~UCT_SRD_IFACE_FENCE;
    ep->flags    &= ~UCT_SRD_EP_FLAG_FENCE;
    iface->tx.ep_outstanding++;
    ucs_trace_data("ep=%p added flush psn=%u send_op=%p with user_comp=%p", ep,
                   ep->psn, send_op, send_op->user_comp);

out:
    UCT_TL_EP_STAT_FLUSH_WAIT(&ep->super);
    return UCS_INPROGRESS;
}
