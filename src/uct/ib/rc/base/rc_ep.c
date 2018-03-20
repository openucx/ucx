/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rc_ep.h"
#include "rc_iface.h"

#include <uct/ib/base/ib_verbs.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/type/class.h>
#include <endian.h>

#if ENABLE_STATS
static ucs_stats_class_t uct_rc_fc_stats_class = {
    .name = "rc_fc",
    .num_counters = UCT_RC_FC_STAT_LAST,
    .counter_names = {
        [UCT_RC_FC_STAT_NO_CRED]            = "no_cred",
        [UCT_RC_FC_STAT_TX_GRANT]           = "tx_grant",
        [UCT_RC_FC_STAT_TX_PURE_GRANT]      = "tx_pure_grant",
        [UCT_RC_FC_STAT_TX_SOFT_REQ]        = "tx_soft_req",
        [UCT_RC_FC_STAT_TX_HARD_REQ]        = "tx_hard_req",
        [UCT_RC_FC_STAT_RX_GRANT]           = "rx_grant",
        [UCT_RC_FC_STAT_RX_PURE_GRANT]      = "rx_pure_grant",
        [UCT_RC_FC_STAT_RX_SOFT_REQ]        = "rx_soft_req",
        [UCT_RC_FC_STAT_RX_HARD_REQ]        = "rx_hard_req",
        [UCT_RC_FC_STAT_FC_WND]             = "fc_wnd"
    }
};

static ucs_stats_class_t uct_rc_txqp_stats_class = {
    .name = "rc_txqp",
    .num_counters = UCT_RC_TXQP_STAT_LAST,
    .counter_names = {
        [UCT_RC_TXQP_STAT_QP_FULL]          = "qp_full",
        [UCT_RC_TXQP_STAT_SIGNAL]           = "signal"
    }
};
#endif

ucs_status_t uct_rc_txqp_init(uct_rc_txqp_t *txqp, uct_rc_iface_t *iface,
                              int qp_type, struct ibv_qp_cap *cap
                              UCS_STATS_ARG(ucs_stats_node_t* stats_parent))
{
    ucs_status_t status;

    txqp->unsignaled = 0;
    txqp->unsignaled_store = 0;
    txqp->unsignaled_store_count = 0;
    txqp->available  = 0;
    ucs_queue_head_init(&txqp->outstanding);

    status = uct_rc_iface_qp_create(iface, qp_type, &txqp->qp, cap,
                                    iface->config.tx_qp_len);
    if (status != UCS_OK) {
        goto err;
    }

    status = UCS_STATS_NODE_ALLOC(&txqp->stats, &uct_rc_txqp_stats_class,
                                  stats_parent, "-0x%x", txqp->qp->qp_num);
    if (status != UCS_OK) {
        goto err_destroy_qp;
    }

    return UCS_OK;

err_destroy_qp:
    ibv_destroy_qp(txqp->qp);
err:
    return status;
}

void uct_rc_txqp_cleanup(uct_rc_txqp_t *txqp)
{
    int ret;

    ret = ibv_destroy_qp(txqp->qp);
    if (ret != 0) {
        ucs_warn("ibv_destroy_qp() returned %d: %m", ret);
    }

    uct_rc_txqp_purge_outstanding(txqp, UCS_ERR_CANCELED, 1);
    UCS_STATS_NODE_FREE(txqp->stats);
}

ucs_status_t uct_rc_fc_init(uct_rc_fc_t *fc, int16_t winsize
                            UCS_STATS_ARG(ucs_stats_node_t* stats_parent))
{
    ucs_status_t status;

    fc->fc_wnd     = winsize;
    fc->flags      = 0;

    status = UCS_STATS_NODE_ALLOC(&fc->stats, &uct_rc_fc_stats_class,
                                  stats_parent);
    if (status != UCS_OK) {
       return status;
    }

    UCS_STATS_SET_COUNTER(fc->stats, UCT_RC_FC_STAT_FC_WND, fc->fc_wnd);

    return UCS_OK;
}

void uct_rc_fc_cleanup(uct_rc_fc_t *fc)
{
    UCS_STATS_NODE_FREE(fc->stats);
}

static void uct_rc_ep_tag_qp_destroy(uct_rc_ep_t *ep)
{
#if IBV_EXP_HW_TM
    uct_rc_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                           uct_rc_iface_t);
    if (UCT_RC_IFACE_TM_ENABLED(iface)) {
        uct_rc_iface_remove_qp(iface, ep->tm_qp->qp_num);
        if (ibv_destroy_qp(ep->tm_qp)) {
            ucs_warn("failed to destroy TM RNDV QP: %m");
        }
    }
#endif
}

static ucs_status_t uct_rc_ep_tag_qp_create(uct_rc_iface_t *iface,
                                            uct_rc_ep_t *ep)
{
#if IBV_EXP_HW_TM
    struct ibv_qp_cap cap;
    ucs_status_t status;
    int ret;

    if (UCT_RC_IFACE_TM_ENABLED(iface)) {
        /* Send queue of this QP will be used by FW for HW RNDV. Driver requires
         * such a QP to be initialized with zero send queue length. */
        status = uct_rc_iface_qp_create(iface, IBV_QPT_RC, &ep->tm_qp, &cap, 0);
        if (status != UCS_OK) {
            return status;
        }

        status = uct_rc_iface_qp_init(iface, ep->tm_qp);
        if (status != UCS_OK) {
            ret = ibv_destroy_qp(ep->tm_qp);
            if (ret) {
                ucs_warn("ibv_destroy_qp() returned %d: %m", ret);
            }
            return status;
        }
        uct_rc_iface_add_qp(iface, ep, ep->tm_qp->qp_num);
    }
#endif
    return UCS_OK;
}

UCS_CLASS_INIT_FUNC(uct_rc_ep_t, uct_rc_iface_t *iface)
{
    struct ibv_qp_cap cap;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    status = uct_rc_txqp_init(&self->txqp, iface, IBV_QPT_RC, &cap
                              UCS_STATS_ARG(self->super.stats));
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_rc_iface_qp_init(iface, self->txqp.qp);
    if (status != UCS_OK) {
        goto err_txqp_cleanup;
    }

    status = uct_rc_fc_init(&self->fc, iface->config.fc_wnd_size
                            UCS_STATS_ARG(self->super.stats));
    if (status != UCS_OK) {
        goto err_txqp_cleanup;
    }

    status = uct_rc_ep_tag_qp_create(iface, self);
    if (status != UCS_OK) {
        goto err_fc_cleanup;
    }

    self->sl                = iface->super.config.sl;    /* TODO multi-rail */
    self->path_bits         = iface->super.path_bits[0]; /* TODO multi-rail */

    /* Check that FC protocol fits AM id
     * (just in case AM id space gets extended) */
    UCS_STATIC_ASSERT(UCT_RC_EP_FC_MASK < UINT8_MAX);

    ucs_arbiter_group_init(&self->arb_group);

    uct_rc_iface_add_qp(iface, self, self->txqp.qp->qp_num);
    ucs_list_add_head(&iface->ep_list, &self->list);
    return UCS_OK;

err_fc_cleanup:
    uct_rc_fc_cleanup(&self->fc);
err_txqp_cleanup:
    uct_rc_txqp_cleanup(&self->txqp);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_ep_t)
{
    uct_rc_iface_t *iface = ucs_derived_of(self->super.super.iface,
                                           uct_rc_iface_t);
    ucs_debug("destroy rc ep %p", self);

    ucs_list_del(&self->list);
    uct_rc_iface_remove_qp(iface, self->txqp.qp->qp_num);
    uct_rc_ep_tag_qp_destroy(self);
    uct_rc_ep_pending_purge(&self->super.super, NULL, NULL);
    uct_rc_fc_cleanup(&self->fc);
    uct_rc_txqp_cleanup(&self->txqp);
}

UCS_CLASS_DEFINE(uct_rc_ep_t, uct_base_ep_t)

ucs_status_t uct_rc_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr)
{
    uct_rc_ep_t *ep              = ucs_derived_of(tl_ep, uct_rc_ep_t);
    uct_rc_ep_address_t *rc_addr = (uct_rc_ep_address_t*)addr;
    uct_rc_iface_t *iface        = ucs_derived_of(tl_ep->iface, uct_rc_iface_t);

    uct_ib_pack_uint24(rc_addr->qp_num, ep->txqp.qp->qp_num);
    rc_addr->atomic_mr_id = uct_ib_iface_get_atomic_mr_id(&iface->super);

#if IBV_EXP_HW_TM
    if (UCT_RC_IFACE_TM_ENABLED(iface)) {
        uct_ib_pack_uint24(rc_addr->tm_qp_num, ep->tm_qp->qp_num);
    }
#endif

    return UCS_OK;
}

ucs_status_t uct_rc_ep_connect_to_ep(uct_ep_h tl_ep, const uct_device_addr_t *dev_addr,
                                     const uct_ep_addr_t *ep_addr)
{
    uct_rc_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_ep_t);
    uct_rc_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_rc_iface_t);
    const uct_ib_address_t *ib_addr = (const uct_ib_address_t *)dev_addr;
    const uct_rc_ep_address_t *rc_addr = (const uct_rc_ep_address_t*)ep_addr;
    uint32_t qp_num;
    struct ibv_ah_attr ah_attr;
    ucs_status_t status;

    uct_ib_iface_fill_ah_attr_from_addr(&iface->super, ib_addr, ep->path_bits, &ah_attr);

#if IBV_EXP_HW_TM
    if (UCT_RC_IFACE_TM_ENABLED(iface)) {
        /* For HW TM we need 2 QPs, one of which will be used by the device for
         * RNDV offload (for issuing RDMA reads and sending RNDV ACK). No WQEs
         * should be posted to the send side of the QP which is owned by device. */
        status = uct_rc_iface_qp_connect(iface, ep->tm_qp,
                                         uct_ib_unpack_uint24(rc_addr->qp_num),
                                         &ah_attr);
        if (status != UCS_OK) {
            return status;
        }

        /* Need to connect local ep QP to the one owned by device
         * (and bound to XRQ) on the peer. */
        qp_num = uct_ib_unpack_uint24(rc_addr->tm_qp_num);
    } else
#endif
    {
        qp_num = uct_ib_unpack_uint24(rc_addr->qp_num);
    }

    status = uct_rc_iface_qp_connect(iface, ep->txqp.qp, qp_num, &ah_attr);
    if (status != UCS_OK) {
        return status;
    }

    ep->atomic_mr_offset = uct_ib_md_atomic_offset(rc_addr->atomic_mr_id);

    return UCS_OK;
}

ucs_status_t uct_rc_modify_qp(uct_rc_txqp_t *txqp, enum ibv_qp_state state)
{
    struct ibv_qp_attr qp_attr;

    ucs_debug("modify QP 0x%x to state %d", txqp->qp->qp_num, state);
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = state;
    if (ibv_modify_qp(txqp->qp, &qp_attr, IBV_QP_STATE)) {
        ucs_warn("modify qp 0x%x to state %d failed: %m", state,
                 txqp->qp->qp_num);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

void uct_rc_ep_am_packet_dump(uct_base_iface_t *iface, uct_am_trace_type_t type,
                              void *data, size_t length, size_t valid_length,
                              char *buffer, size_t max)
{
    uct_rc_hdr_t *rch = data;
    uint8_t fc_hdr    = uct_rc_fc_get_fc_hdr(rch->am_id);
    uint8_t am_wo_fc;

    /* Do not invoke AM tracer for auxiliary pure FC_GRANT message */
    if (fc_hdr != UCT_RC_EP_FC_PURE_GRANT) {
        am_wo_fc = rch->am_id & ~UCT_RC_EP_FC_MASK; /* mask out FC bits*/
        snprintf(buffer, max, " %c%c am %d "
#if IBV_EXP_HW_TM
                 "tm_op %d "
#endif
                 ,
                 fc_hdr & UCT_RC_EP_FC_FLAG_SOFT_REQ ? 's' :
                 fc_hdr & UCT_RC_EP_FC_FLAG_HARD_REQ ? 'h' : '-',
                 fc_hdr & UCT_RC_EP_FC_FLAG_GRANT    ? 'g' : '-',
                 am_wo_fc
#if IBV_EXP_HW_TM
                 , rch->tmh_opcode
#endif
                 );
        uct_iface_dump_am(iface, type, am_wo_fc, rch + 1, length - sizeof(*rch),
                          buffer + strlen(buffer), max - strlen(buffer));
    } else {
        snprintf(buffer, max, " FC pure grant am ");
    }
}

void uct_rc_ep_get_bcopy_handler(uct_rc_iface_send_op_t *op, const void *resp)
{
    uct_rc_iface_send_desc_t *desc = ucs_derived_of(op, uct_rc_iface_send_desc_t);

    VALGRIND_MAKE_MEM_DEFINED(resp, desc->super.length);

    desc->unpack_cb(desc->super.unpack_arg, resp, desc->super.length);

    uct_invoke_completion(desc->super.user_comp, UCS_OK);

    ucs_mpool_put(desc);
}

void uct_rc_ep_get_bcopy_handler_no_completion(uct_rc_iface_send_op_t *op,
                                               const void *resp)
{
    uct_rc_iface_send_desc_t *desc = ucs_derived_of(op, uct_rc_iface_send_desc_t);

    VALGRIND_MAKE_MEM_DEFINED(resp, desc->super.length);

    desc->unpack_cb(desc->super.unpack_arg, resp, desc->super.length);

    ucs_mpool_put(desc);
}

void uct_rc_ep_send_op_completion_handler(uct_rc_iface_send_op_t *op,
                                          const void *resp)
{
    uct_invoke_completion(op->user_comp, UCS_OK);
    uct_rc_iface_put_send_op(op);
}

ucs_status_t uct_rc_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *n)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_iface_t);
    uct_rc_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_ep_t);

    if (uct_rc_ep_has_tx_resources(ep) &&
        uct_rc_iface_has_tx_resources(iface)) {
        return UCS_ERR_BUSY;
    }

    UCS_STATIC_ASSERT(sizeof(ucs_arbiter_elem_t) <= UCT_PENDING_REQ_PRIV_LEN);
    ucs_arbiter_elem_init((ucs_arbiter_elem_t *)n->priv);
    ucs_arbiter_group_push_elem(&ep->arb_group, (ucs_arbiter_elem_t*)n->priv);

    if (uct_rc_ep_has_tx_resources(ep)) {
        /* If we have ep (but not iface) resources, we need to schedule the ep */
        ucs_arbiter_group_schedule(&iface->tx.arbiter, &ep->arb_group);
    }

    return UCS_OK;
}

ucs_arbiter_cb_result_t uct_rc_ep_process_pending(ucs_arbiter_t *arbiter,
                                                  ucs_arbiter_elem_t *elem,
                                                  void *arg)
{
    uct_pending_req_t *req = ucs_container_of(elem, uct_pending_req_t, priv);
    uct_rc_iface_t *iface UCS_V_UNUSED;
    ucs_status_t status;
    uct_rc_ep_t *ep;

    status = req->func(req);
    ucs_trace_data("progress pending request %p returned: %s", req,
                   ucs_status_string(status));

    if (status == UCS_OK) {
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    } else if (status == UCS_INPROGRESS) {
        return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
    } else {
        ep = ucs_container_of(ucs_arbiter_elem_group(elem), uct_rc_ep_t, arb_group);
        if (! uct_rc_ep_has_tx_resources(ep)) {
            /* No ep resources */
            return UCS_ARBITER_CB_RESULT_DESCHED_GROUP;
        } else {
            /* No iface resources */
            iface = ucs_derived_of(ep->super.super.iface, uct_rc_iface_t);
            ucs_assertv(!uct_rc_iface_has_tx_resources(iface),
                        "pending callback returned error but send resources are available");
            return UCS_ARBITER_CB_RESULT_STOP;
        }
    }
}

static ucs_arbiter_cb_result_t uct_rc_ep_abriter_purge_cb(ucs_arbiter_t *arbiter,
                                                          ucs_arbiter_elem_t *elem,
                                                          void *arg)
{
    uct_rc_fc_request_t *freq;
    uct_purge_cb_args_t *cb_args    = arg;
    uct_pending_purge_callback_t cb = cb_args->cb;
    uct_pending_req_t *req    = ucs_container_of(elem, uct_pending_req_t, priv);
    uct_rc_ep_t       *ep     = ucs_container_of(ucs_arbiter_elem_group(elem),
                                                 uct_rc_ep_t, arb_group);

    /* Invoke user's callback only if it is not internal FC message */
    if (ucs_likely(req->func != uct_rc_ep_fc_grant)){
        if (cb != NULL) {
            cb(req, cb_args->arg);
        } else {
            ucs_debug("ep=%p cancelling user pending request %p", ep, req);
        }
    } else {
        freq = ucs_derived_of(req, uct_rc_fc_request_t);
        ucs_mpool_put(freq);
    }
    return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
}

void uct_rc_ep_pending_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb,
                             void *arg)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_iface_t);
    uct_rc_ep_t *ep       = ucs_derived_of(tl_ep, uct_rc_ep_t);
    uct_purge_cb_args_t  args = {cb, arg};

    ucs_arbiter_group_purge(&iface->tx.arbiter, &ep->arb_group,
                            uct_rc_ep_abriter_purge_cb, &args);
}

ucs_status_t uct_rc_ep_fc_grant(uct_pending_req_t *self)
{
    ucs_status_t status;
    uct_rc_fc_request_t *freq = ucs_derived_of(self, uct_rc_fc_request_t);
    uct_rc_ep_t *ep           = ucs_derived_of(freq->ep, uct_rc_ep_t);
    uct_rc_iface_t *iface     = ucs_derived_of(ep->super.super.iface,
                                               uct_rc_iface_t);

    ucs_assert_always(iface->config.fc_enabled);
    status = uct_rc_fc_ctrl(&ep->super.super, UCT_RC_EP_FC_PURE_GRANT, NULL);
    if (status == UCS_OK) {
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_TX_PURE_GRANT, 1);
        ucs_mpool_put(freq);
    }
    return status;
}

void uct_rc_txqp_purge_outstanding(uct_rc_txqp_t *txqp, ucs_status_t status,
                                   int is_log)
{
    uct_rc_iface_send_op_t *op;
    uct_rc_iface_send_desc_t *desc;

    ucs_queue_for_each_extract(op, &txqp->outstanding, queue, 1) {
        if (op->handler != (uct_rc_send_handler_t)ucs_mpool_put) {
            if (is_log != 0) {
                ucs_warn("destroying rc ep %p with uncompleted operation %p",
                         txqp, op);
            }

            if (op->user_comp != NULL) {
                /* This must be uct_rc_ep_get_bcopy_handler,
                 * uct_rc_ep_send_completion_proxy_handler,
                 * one of the atomic handlers,
                 * so invoke user completion */
                uct_invoke_completion(op->user_comp, status);
            }
        }
        op->flags &= ~UCT_RC_IFACE_SEND_OP_FLAG_INUSE;
        if (op->handler == uct_rc_ep_send_op_completion_handler) {
            uct_rc_iface_put_send_op(op);
        } else {
            desc = ucs_derived_of(op, uct_rc_iface_send_desc_t);
            ucs_mpool_put(desc);
        }
    }
}

ucs_status_t uct_rc_ep_flush(uct_rc_ep_t *ep, int16_t max_available,
                             unsigned flags)
{
    uct_rc_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_rc_iface_t);

    if (ucs_unlikely(flags & UCT_FLUSH_FLAG_CANCEL)) {
        uct_rc_txqp_purge_outstanding(&ep->txqp, UCS_ERR_CANCELED, 0);
        uct_ep_pending_purge(&ep->super.super, NULL, 0);
        return UCS_OK;
    }

    if (!uct_rc_iface_has_tx_resources(iface) || !uct_rc_ep_has_tx_resources(ep)) {
        return UCS_ERR_NO_RESOURCE;
    }

    if (uct_rc_txqp_available(&ep->txqp) == max_available) {
        UCT_TL_EP_STAT_FLUSH(&ep->super);
        return UCS_OK;
    }

    return UCS_INPROGRESS;
}

#if IBV_EXP_HW_TM
ucs_status_t uct_rc_ep_tag_rndv_cancel(uct_ep_h tl_ep, void *op)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_iface_t);

    uint32_t op_index = (uint32_t)((uint64_t)op);
    ucs_ptr_array_remove(&iface->tm.rndv_comps, op_index, 0);
    return UCS_OK;
}
#endif

#define UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC(_num_bits, _is_be) \
    void UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC_NAME(_num_bits, _is_be) \
            (uct_rc_iface_send_op_t *op, const void *resp) \
    { \
        uct_rc_iface_send_desc_t *desc = \
            ucs_derived_of(op, uct_rc_iface_send_desc_t); \
        const uint##_num_bits##_t *value = resp; \
        uint##_num_bits##_t *dest = desc->super.buffer; \
        \
        VALGRIND_MAKE_MEM_DEFINED(value, sizeof(*value)); \
        if (_is_be && (_num_bits == 32)) { \
            *dest = ntohl(*value); /* TODO swap in-place */ \
        } else if (_is_be && (_num_bits == 64)) { \
            *dest = be64toh(*value); \
        } else { \
            *dest = *value; \
        } \
        \
        uct_invoke_completion(desc->super.user_comp, UCS_OK); \
        ucs_mpool_put(desc); \
  }

UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC(32, 0);
UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC(32, 1);
UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC(64, 0);
UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC(64, 1);

void uct_rc_ep_am_zcopy_handler(uct_rc_iface_send_op_t *op, const void *resp)
{
    uct_rc_iface_send_desc_t *desc = ucs_derived_of(op, uct_rc_iface_send_desc_t);
    uct_invoke_completion(desc->super.user_comp, UCS_OK);
    ucs_mpool_put(desc);
}
