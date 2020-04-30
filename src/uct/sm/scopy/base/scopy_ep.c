/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "scopy_iface.h"
#include "scopy_ep.h"

#include <uct/base/uct_iov.inl>


const char* uct_scopy_tx_op_str[] = {
    [UCT_SCOPY_TX_PUT_ZCOPY] = "uct_scopy_ep_put_zcopy",
    [UCT_SCOPY_TX_GET_ZCOPY] = "uct_scopy_ep_get_zcopy"
};

UCS_CLASS_INIT_FUNC(uct_scopy_ep_t, const uct_ep_params_t *params)
{
    uct_scopy_iface_t *iface = ucs_derived_of(params->iface, uct_scopy_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    ucs_arbiter_group_init(&self->arb_group);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_scopy_ep_t)
{
    ucs_arbiter_group_cleanup(&self->arb_group);
}

UCS_CLASS_DEFINE(uct_scopy_ep_t, uct_base_ep_t)

static UCS_F_ALWAYS_INLINE void
uct_scopy_ep_tx_init_common(uct_scopy_tx_t *tx, uct_scopy_tx_op_t tx_op,
                            uct_completion_t *comp)
{
    tx->comp = comp;
    tx->op   = tx_op;
    ucs_arbiter_elem_init(&tx->arb_elem);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_scopy_ep_tx_init(uct_ep_h tl_ep, const uct_iov_t *iov,
                     size_t iov_cnt, uint64_t remote_addr,
                     uct_rkey_t rkey, uct_completion_t *comp,
                     uct_scopy_tx_op_t tx_op)
{
    uct_scopy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_scopy_iface_t);
    uct_scopy_ep_t *ep       = ucs_derived_of(tl_ep, uct_scopy_ep_t);
    uct_scopy_tx_t *tx;
    size_t iov_it;

    ucs_assert((tx_op == UCT_SCOPY_TX_PUT_ZCOPY) ||
               (tx_op == UCT_SCOPY_TX_GET_ZCOPY));

    UCT_CHECK_IOV_SIZE(iov_cnt, iface->config.max_iov, uct_scopy_tx_op_str[tx_op]);

    tx = ucs_mpool_get_inline(&iface->tx_mpool);
    if (ucs_unlikely(tx == NULL)) {
        return UCS_ERR_NO_MEMORY;
    }

    uct_scopy_ep_tx_init_common(tx, tx_op, comp);
    tx->rkey        = rkey;
    tx->remote_addr = remote_addr;
    tx->iov_cnt     = 0;
    ucs_iov_iter_init(&tx->iov_iter);
    for (iov_it = 0; iov_it < iov_cnt; iov_it++) {
        if (uct_iov_get_length(&iov[iov_it]) == 0) {
            /* Avoid zero-length IOV elements */
            continue;
        }

        tx->iov[tx->iov_cnt] = iov[iov_it];
        tx->iov_cnt++;
    }

    if (tx_op == UCT_SCOPY_TX_PUT_ZCOPY) {
        UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                          uct_iov_total_length(tx->iov, tx->iov_cnt));
    } else {
        UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                          uct_iov_total_length(tx->iov, tx->iov_cnt));
    }

    if (tx->iov_cnt == 0) {
        uct_scopy_trace_data(tx);
        ucs_mpool_put_inline(tx);
        return UCS_OK;
    }

    if (ucs_unlikely(ucs_arbiter_is_empty(&iface->arbiter))) {
        uct_worker_progress_register_safe(&iface->super.super.worker->super,
                                          (ucs_callback_t)
                                          iface->super.super.super.ops.iface_progress,
                                          iface, UCS_CALLBACKQ_FLAG_FAST,
                                          &iface->super.super.prog.id);
    }

    ucs_arbiter_group_push_elem(&ep->arb_group, &tx->arb_elem);
    ucs_arbiter_group_schedule(&iface->arbiter, &ep->arb_group);

    return UCS_INPROGRESS;
}

ucs_status_t uct_scopy_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                    size_t iov_cnt, uint64_t remote_addr,
                                    uct_rkey_t rkey, uct_completion_t *comp)
{
    return uct_scopy_ep_tx_init(tl_ep, iov, iov_cnt, remote_addr,
                                rkey, comp, UCT_SCOPY_TX_PUT_ZCOPY);
}

ucs_status_t uct_scopy_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                    size_t iov_cnt, uint64_t remote_addr,
                                    uct_rkey_t rkey, uct_completion_t *comp)
{
    return uct_scopy_ep_tx_init(tl_ep, iov, iov_cnt, remote_addr,
                                rkey, comp, UCT_SCOPY_TX_GET_ZCOPY);
}

ucs_arbiter_cb_result_t uct_scopy_ep_progress_tx(ucs_arbiter_t *arbiter,
                                                 ucs_arbiter_group_t *group,
                                                 ucs_arbiter_elem_t *elem,
                                                 void *arg)
{
    uct_scopy_iface_t *iface = ucs_container_of(arbiter, uct_scopy_iface_t,
                                                arbiter);
    uct_scopy_ep_t *ep       = ucs_container_of(group, uct_scopy_ep_t,
                                                arb_group);
    uct_scopy_tx_t *tx       = ucs_container_of(elem, uct_scopy_tx_t,
                                                arb_elem);
    unsigned *count          = (unsigned*)arg;
    ucs_status_t status      = UCS_OK;
    size_t seg_size;

    if (*count == iface->config.tx_quota) {
        return UCS_ARBITER_CB_RESULT_STOP;
    }

    if (tx->op != UCT_SCOPY_TX_FLUSH_COMP) {
        ucs_assert((tx->op == UCT_SCOPY_TX_GET_ZCOPY) ||
                   (tx->op == UCT_SCOPY_TX_PUT_ZCOPY));
        seg_size = iface->config.seg_size;
        status   = iface->tx(&ep->super.super, tx->iov, tx->iov_cnt,
                             &tx->iov_iter, &seg_size, tx->remote_addr,
                             tx->rkey, tx->op);
        if (!UCS_STATUS_IS_ERR(status)) {
            (*count)++;
            ucs_assertv(*count <= iface->config.tx_quota,
                        "count=%u vs quota=%u",
                        *count, iface->config.tx_quota);

            tx->remote_addr += seg_size;
            uct_scopy_trace_data(tx);

            if (tx->iov_iter.iov_index < tx->iov_cnt) {
                return UCS_ARBITER_CB_RESULT_RESCHED_GROUP;
            }
        }
    }

    ucs_assert((tx->comp != NULL) ||
               (tx->op != UCT_SCOPY_TX_FLUSH_COMP));
    if (tx->comp != NULL) {
        uct_invoke_completion(tx->comp, status);
    }

    ucs_mpool_put_inline(tx);

    return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
}

ucs_status_t uct_scopy_ep_flush(uct_ep_h tl_ep, unsigned flags,
                                uct_completion_t *comp)
{
    uct_scopy_ep_t *ep       = ucs_derived_of(tl_ep, uct_scopy_ep_t);
    uct_scopy_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                              uct_scopy_iface_t);
    uct_scopy_tx_t *flush_comp;

    if (ucs_arbiter_group_is_empty(&ep->arb_group)) {
        UCT_TL_EP_STAT_FLUSH(&ep->super);
        return UCS_OK;
    }

    if (comp != NULL) {
        flush_comp = ucs_mpool_get_inline(&iface->tx_mpool);
        if (ucs_unlikely(flush_comp == NULL)) {
            return UCS_ERR_NO_MEMORY;
        }

        uct_scopy_ep_tx_init_common(flush_comp, UCT_SCOPY_TX_FLUSH_COMP, comp);
        ucs_arbiter_group_push_elem(&ep->arb_group, &flush_comp->arb_elem);
    }

    UCT_TL_EP_STAT_FLUSH_WAIT(&ep->super);
    return UCS_INPROGRESS;
}
