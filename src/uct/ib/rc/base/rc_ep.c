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
#include <infiniband/arch.h>


#if ENABLE_STATS
static ucs_stats_class_t uct_rc_ep_stats_class = {
    .name = "rc_ep",
    .num_counters = UCT_RC_EP_STAT_LAST,
    .counter_names = {
        [UCT_RC_EP_STAT_QP_FULL]          = "qp_full",
        [UCT_RC_EP_STAT_SINGAL]           = "signal"
    }
};
#endif

UCS_CLASS_INIT_FUNC(uct_rc_ep_t, uct_rc_iface_t *iface)
{
    struct ibv_qp_cap cap;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    status = uct_rc_iface_qp_create(iface, &self->qp, &cap);
    if (status != UCS_OK) {
        goto err;
    }

    status = UCS_STATS_NODE_ALLOC(&self->stats, &uct_rc_ep_stats_class,
                                  self->super.stats);
    if (status != UCS_OK) {
        goto err_qp_destroy;
    }

    self->unsignaled = 0;
    self->sl         = iface->config.sl;          /* TODO multi-rail */
    self->path_bits  = iface->super.path_bits[0]; /* TODO multi-rail */
    ucs_queue_head_init(&self->outstanding);
    ucs_arbiter_group_init(&self->arb_group);

    uct_rc_iface_add_ep(iface, self);
    return UCS_OK;

err_qp_destroy:
    ibv_destroy_qp(self->qp);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_ep_t)
{
    uct_rc_iface_t *iface = ucs_derived_of(self->super.super.iface, uct_rc_iface_t);
    uct_rc_iface_send_op_t *op;
    int ret;

    ucs_debug("destroy rc ep %p", self);

    uct_rc_iface_remove_ep(iface, self);

    ucs_arbiter_group_cleanup(&self->arb_group);

    UCS_STATS_NODE_FREE(self->stats);
    ret = ibv_destroy_qp(self->qp);
    if (ret != 0) {
        ucs_warn("ibv_destroy_qp() returned %d: %m", ret);
    }

    ucs_queue_for_each_extract(op, &self->outstanding, queue, 1) {
        if (op->handler != (uct_rc_send_handler_t)ucs_mpool_put) {
            ucs_warn("destroying rc ep %p with uncompleted operation %p",
                     self, op);
        }
        op->handler(op);
    }
}

UCS_CLASS_DEFINE(uct_rc_ep_t, uct_base_ep_t)

ucs_status_t uct_rc_ep_get_address(uct_ep_h tl_ep, struct sockaddr *addr)
{
    uct_rc_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_ep_t);
    uct_rc_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_rc_iface_t);
    uct_sockaddr_ib_t *ib_addr = (uct_sockaddr_ib_t*)addr;

    uct_ib_iface_get_address(&iface->super.super.super, addr);
    ib_addr->qp_num = ep->qp->qp_num;
    return UCS_OK;
}

ucs_status_t uct_rc_ep_connect_to_ep(uct_ep_h tl_ep, const struct sockaddr *addr)
{
    uct_rc_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_ep_t);
    uct_rc_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_rc_iface_t);
    const uct_sockaddr_ib_t *ib_addr = (uct_sockaddr_ib_t*)addr;
    struct ibv_qp_attr qp_attr;
    int ret;

    memset(&qp_attr, 0, sizeof(qp_attr));

    qp_attr.qp_state              = IBV_QPS_INIT;
    qp_attr.pkey_index            = iface->super.pkey_index;
    qp_attr.port_num              = iface->super.port_num;
    qp_attr.qp_access_flags       = IBV_ACCESS_LOCAL_WRITE|
                                    IBV_ACCESS_REMOTE_WRITE|
                                    IBV_ACCESS_REMOTE_READ|
                                    IBV_ACCESS_REMOTE_ATOMIC;
    ret = ibv_modify_qp(ep->qp, &qp_attr,
                        IBV_QP_STATE      |
                        IBV_QP_PKEY_INDEX |
                        IBV_QP_PORT       |
                        IBV_QP_ACCESS_FLAGS);
    if (ret) {
         ucs_error("error modifying QP to INIT: %m");
         return UCS_ERR_IO_ERROR;
    }

    ucs_assert((ib_addr->lid & ep->path_bits) == 0);
    qp_attr.qp_state              = IBV_QPS_RTR;
    qp_attr.ah_attr.dlid          = ib_addr->lid | ep->path_bits;
    qp_attr.ah_attr.sl            = iface->super.sl;
    qp_attr.ah_attr.src_path_bits = ep->path_bits;
    qp_attr.ah_attr.static_rate   = 0;
    qp_attr.ah_attr.is_global     = 0; /* TODO RoCE */
    qp_attr.ah_attr.port_num      = iface->super.port_num;
    qp_attr.dest_qp_num           = ib_addr->qp_num;
    qp_attr.rq_psn                = 0;
    qp_attr.path_mtu              = iface->config.path_mtu;
    qp_attr.max_dest_rd_atomic    = iface->config.max_rd_atomic;
    qp_attr.min_rnr_timer         = iface->config.min_rnr_timer;
    ret = ibv_modify_qp(ep->qp, &qp_attr,
                        IBV_QP_STATE              |
                        IBV_QP_AV                 |
                        IBV_QP_PATH_MTU           |
                        IBV_QP_DEST_QPN           |
                        IBV_QP_RQ_PSN             |
                        IBV_QP_MAX_DEST_RD_ATOMIC |
                        IBV_QP_MIN_RNR_TIMER);
    if (ret) {
        ucs_error("error modifying QP to RTR: %m");
        return UCS_ERR_IO_ERROR;
    }

    qp_attr.qp_state              = IBV_QPS_RTS;
    qp_attr.sq_psn                = 0;
    qp_attr.timeout               = iface->config.timeout;
    qp_attr.rnr_retry             = iface->config.rnr_retry;
    qp_attr.retry_cnt             = iface->config.retry_cnt;
    qp_attr.max_rd_atomic         = iface->config.max_rd_atomic;
    ret = ibv_modify_qp(ep->qp, &qp_attr,
                        IBV_QP_STATE              |
                        IBV_QP_TIMEOUT            |
                        IBV_QP_RETRY_CNT          |
                        IBV_QP_RNR_RETRY          |
                        IBV_QP_SQ_PSN             |
                        IBV_QP_MAX_QP_RD_ATOMIC);
    if (ret) {
        ucs_error("error modifying QP to RTS: %m");
        return UCS_ERR_IO_ERROR;
    }

    ucs_debug("connected rc qp 0x%x to lid %d(+%d) sl %d remote_qp 0x%x mtu %zu "
              "timer %dx%d rnr %dx%d rd_atom %d",
              ep->qp->qp_num, qp_attr.ah_attr.dlid, ep->path_bits, qp_attr.ah_attr.sl,
              qp_attr.dest_qp_num, uct_ib_mtu_value(qp_attr.path_mtu),
              qp_attr.timeout, qp_attr.retry_cnt, qp_attr.min_rnr_timer,
              qp_attr.rnr_retry, qp_attr.max_rd_atomic);
    return UCS_OK;
}

void uct_rc_ep_am_packet_dump(uct_base_iface_t *iface, uct_am_trace_type_t type,
                              void *data, size_t length, size_t valid_length,
                              char *buffer, size_t max)
{
    uct_rc_hdr_t *rch = data;

    snprintf(buffer, max, " am %d ", rch->am_id);
    uct_iface_dump_am(iface, type, rch->am_id, rch + 1, length - sizeof(*rch),
                      buffer + strlen(buffer), max - strlen(buffer));
}

void uct_rc_ep_get_bcopy_handler(uct_rc_iface_send_op_t *op)
{
    uct_rc_iface_send_desc_t *desc = ucs_derived_of(op, uct_rc_iface_send_desc_t);

    VALGRIND_MAKE_MEM_DEFINED(desc + 1, desc->super.length);
    desc->unpack_cb(desc->super.unpack_arg, desc + 1, desc->super.length);
    if (desc->super.user_comp) {
        uct_invoke_completion(desc->super.user_comp);
    }
    ucs_mpool_put(desc);
}

void uct_rc_ep_get_bcopy_handler_no_completion(uct_rc_iface_send_op_t *op)
{
    uct_rc_iface_send_desc_t *desc = ucs_derived_of(op, uct_rc_iface_send_desc_t);

    VALGRIND_MAKE_MEM_DEFINED(desc + 1, desc->super.length);
    desc->unpack_cb(desc->super.unpack_arg, desc + 1, desc->super.length);
    ucs_mpool_put(desc);
}

void uct_rc_ep_send_completion_proxy_handler(uct_rc_iface_send_op_t *op)
{
    uct_invoke_completion(op->user_comp);
}

static int uct_rc_iface_has_tx_resources(uct_rc_iface_t *iface)
{
    return uct_rc_iface_have_tx_cqe_avail(iface) &&
           !ucs_mpool_is_empty(&iface->tx.mp);
}

ucs_status_t uct_rc_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *n)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_iface_t);
    uct_rc_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_ep_t);

    if ((ep->available > 0) && uct_rc_iface_has_tx_resources(iface)) {
        return UCS_ERR_BUSY;
    }

    UCS_STATIC_ASSERT(sizeof(ucs_arbiter_elem_t) <= UCT_PENDING_REQ_PRIV_LEN);
    ucs_arbiter_elem_init((ucs_arbiter_elem_t *)n->priv);
    ucs_arbiter_group_push_elem(&ep->arb_group, (ucs_arbiter_elem_t*)n->priv);

    if (ep->available > 0) {
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
        if (ep->available <= 0) {
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
    uct_pending_req_t *req = ucs_container_of(elem, uct_pending_req_t, priv);
    uct_pending_callback_t cb = arg;

    cb(req);
    return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
}

void uct_rc_ep_pending_purge(uct_ep_h tl_ep, uct_pending_callback_t cb)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_iface_t);
    uct_rc_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_ep_t);

    ucs_arbiter_group_purge(&iface->tx.arbiter, &ep->arb_group,
                            uct_rc_ep_abriter_purge_cb, cb);
}

#define UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC(_num_bits, _is_be) \
    void UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC_NAME(_num_bits, _is_be) \
            (uct_rc_iface_send_op_t *op) \
    { \
        uct_rc_iface_send_desc_t *desc = \
            ucs_derived_of(op, uct_rc_iface_send_desc_t); \
        uint##_num_bits##_t *value = (void*)(desc + 1); \
        uint##_num_bits##_t *dest = desc->super.buffer; \
        \
        VALGRIND_MAKE_MEM_DEFINED(value, sizeof(*value)); \
        if (_is_be && (_num_bits == 32)) { \
            *dest = ntohl(*value); /* TODO swap in-place */ \
        } else if (_is_be && (_num_bits == 64)) { \
            *dest = ntohll(*value); \
        } else { \
            *dest = *value; \
        } \
        \
        uct_invoke_completion(desc->super.user_comp); \
        ucs_mpool_put(desc); \
  }

UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC(32, 0);
UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC(32, 1);
UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC(64, 0);
UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC(64, 1);
