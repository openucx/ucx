/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_iface.h"
#include "rc_ep.h"

#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/type/class.h>


ucs_config_field_t uct_rc_iface_config_table[] = {
  {"IB_", "RX_INLINE=64", NULL,
   ucs_offsetof(uct_rc_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_ib_iface_config_table)},

  {NULL}
};

void uct_rc_iface_query(uct_rc_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    memset(&iface_attr->cap, 0, sizeof(iface_attr->cap));
    iface_attr->iface_addr_len      = sizeof(uct_ib_iface_addr_t);
    iface_attr->ep_addr_len         = sizeof(uct_rc_ep_addr_t);
    iface_attr->completion_priv_len = sizeof(ucs_callbackq_elem_t) - sizeof(uct_completion_t);
    iface_attr->cap.flags           = UCT_IFACE_FLAG_AM_SHORT |
                                      UCT_IFACE_FLAG_AM_BCOPY |
                                      UCT_IFACE_FLAG_AM_ZCOPY |
                                      UCT_IFACE_FLAG_PUT_SHORT |
                                      UCT_IFACE_FLAG_PUT_BCOPY |
                                      UCT_IFACE_FLAG_PUT_ZCOPY |
                                      UCT_IFACE_FLAG_GET_BCOPY |
                                      UCT_IFACE_FLAG_GET_ZCOPY;
}

ucs_status_t uct_rc_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);

    *(uct_ib_iface_addr_t*)iface_addr = iface->super.addr;
    return UCS_OK;
}

void uct_rc_iface_add_ep(uct_rc_iface_t *iface, uct_rc_ep_t *ep)
{
    unsigned qp_num = ep->qp->qp_num;
    uct_rc_ep_t ***ptr, **memb;

    ptr = &iface->eps[qp_num >> UCT_RC_QP_TABLE_ORDER];
    if (*ptr == NULL) {
        *ptr = ucs_calloc(UCS_BIT(UCT_RC_QP_TABLE_MEMB_ORDER), sizeof(**ptr),
                           "rc qp table");
    }

    memb = &(*ptr)[qp_num &  UCS_MASK(UCT_RC_QP_TABLE_MEMB_ORDER)];
    ucs_assert(*memb == NULL);
    *memb = ep;
}

void uct_rc_iface_remove_ep(uct_rc_iface_t *iface, uct_rc_ep_t *ep)
{
    unsigned qp_num = ep->qp->qp_num;
    uct_rc_ep_t **memb;

    memb = &iface->eps[qp_num >> UCT_RC_QP_TABLE_ORDER]
                      [qp_num &  UCS_MASK(UCT_RC_QP_TABLE_MEMB_ORDER)];
    ucs_assert(*memb != NULL);
    *memb = NULL;
}

ucs_status_t uct_rc_iface_flush(uct_iface_h tl_iface)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);
    uct_rc_ep_t **memb, *ep;
    unsigned i, j;

    if (iface->tx.outstanding == 0) {
        return UCS_OK;
    }

    /* TODO this iteration is too much.. */
    for (i = 0; i < UCT_RC_QP_TABLE_SIZE; ++i) {
        memb = iface->eps[i];
        if (memb == NULL) {
            continue;
        }

        for (j = 0; j < UCS_BIT(UCT_RC_QP_TABLE_MEMB_ORDER); ++j) {
            ep = memb[j];
            if (ep == NULL) {
                continue;
            }

            uct_ep_flush(&ep->super);
        }
    }
    return UCS_ERR_WOULD_BLOCK;
}

void uct_rc_iface_send_desc_init(uct_iface_h tl_iface, void *obj, uct_lkey_t lkey)
{
    uct_rc_iface_send_desc_t *desc = obj;
    desc->lkey = uct_ib_lkey_mr(lkey)->lkey;
}

static UCS_CLASS_INIT_FUNC(uct_rc_iface_t, uct_iface_ops_t *ops,
                           uct_context_h context, const char *dev_name,
                           size_t rx_headroom, size_t rx_priv_len,
                           uct_rc_iface_config_t *config)
{
    struct ibv_srq_init_attr srq_init_attr;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(ops, context, dev_name, rx_headroom, rx_priv_len,
                              sizeof(uct_rc_hdr_t), config);

    self->tx.outstanding         = 0;
    self->rx.available           = config->super.rx.queue_len;
    self->config.tx_qp_len       = config->super.tx.queue_len;
    self->config.tx_min_sge      = config->super.tx.min_sge;
    self->config.tx_min_inline   = config->super.tx.min_inline;
    self->config.tx_moderation   = ucs_min(ucs_roundup_pow2(config->super.tx.cq_moderation),
                                           ucs_roundup_pow2(config->super.tx.queue_len / 4));
    self->config.rx_max_batch    = ucs_min(config->super.rx.max_batch, config->super.rx.queue_len / 4);
    self->config.rx_inline       = config->super.rx.inl;

    memset(self->eps, 0, sizeof(self->eps));

    /* Create RX buffers mempool */
    status = uct_ib_iface_recv_mpool_create(&self->super, &config->super,
                                            "rc_recv_desc", &self->rx.mp);
    if (status != UCS_OK) {
        goto err;
    }

    /* Create TX buffers mempool */
    status = uct_iface_mpool_create(&self->super.super.super,
                                    sizeof(uct_rc_iface_send_desc_t) + self->super.config.seg_size,
                                    sizeof(uct_rc_iface_send_desc_t),
                                    UCS_SYS_CACHE_LINE_SIZE,
                                    &config->super.tx.mp,
                                    self->config.tx_qp_len,
                                    uct_rc_iface_send_desc_init,
                                    "rc_send_desc", &self->tx.mp);
    if (status != UCS_OK) {
        goto err_destroy_rx_mp;
    }

    /* Create SRQ */
    srq_init_attr.attr.max_sge   = 1;
    srq_init_attr.attr.max_wr    = config->super.rx.queue_len;
    srq_init_attr.attr.srq_limit = 0;
    srq_init_attr.srq_context    = self;
    self->rx.srq = ibv_create_srq(uct_ib_iface_device(&self->super)->pd,
                                  &srq_init_attr);
    if (self->rx.srq == NULL) {
        ucs_error("failed to create SRQ: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_destory_tx_mp;
    }

    return UCS_OK;

err_destory_tx_mp:
    ucs_mpool_destroy(self->tx.mp);
err_destroy_rx_mp:
    ucs_mpool_destroy(self->rx.mp);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_iface_t)
{
    unsigned i;
    int ret;

    /* Release table. TODO release on-demand when removing ep. */
    for (i = 0; i < UCT_RC_QP_TABLE_SIZE; ++i) {
        ucs_free(self->eps[i]);
    }

    /* TODO flush RX buffers */
    ret = ibv_destroy_srq(self->rx.srq);
    if (ret) {
        ucs_warn("failed to destroy SRQ: %m");
    }

    ucs_mpool_destroy(self->tx.mp);
    ucs_mpool_destroy_unchecked(self->rx.mp); /* Cannot flush SRQ */
}

UCS_CLASS_DEFINE(uct_rc_iface_t, uct_ib_iface_t);


ucs_status_t uct_rc_iface_qp_create(uct_rc_iface_t *iface, struct ibv_qp **qp_p,
                                    struct ibv_qp_cap *cap)
{
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super);
    struct ibv_exp_qp_init_attr qp_init_attr;
    struct ibv_qp *qp;

    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.qp_context          = NULL;
    qp_init_attr.send_cq             = iface->super.send_cq;
    qp_init_attr.recv_cq             = iface->super.recv_cq;
    qp_init_attr.srq                 = iface->rx.srq;
    qp_init_attr.cap.max_send_wr     = iface->config.tx_qp_len;
    qp_init_attr.cap.max_recv_wr     = 0;
    qp_init_attr.cap.max_send_sge    = iface->config.tx_min_sge;
    qp_init_attr.cap.max_recv_sge    = 1;
    qp_init_attr.cap.max_inline_data = iface->config.tx_min_inline;
    qp_init_attr.qp_type             = IBV_QPT_RC;
    qp_init_attr.sq_sig_all          = 0;
#if HAVE_DECL_IBV_EXP_CREATE_QP
    qp_init_attr.comp_mask           = IBV_EXP_QP_INIT_ATTR_PD;
    qp_init_attr.pd                  = dev->pd;

#  if HAVE_IB_EXT_ATOMICS
    qp_init_attr.comp_mask          |= IBV_EXP_QP_INIT_ATTR_ATOMICS_ARG;
    qp_init_attr.max_atomic_arg      = UCT_RC_MAX_ATOMIC_SIZE;
#  endif

#  if HAVE_DECL_IBV_EXP_ATOMIC_HCA_REPLY_BE
    if (dev->dev_attr.exp_atomic_cap == IBV_EXP_ATOMIC_HCA_REPLY_BE) {
        qp_init_attr.comp_mask       |= IBV_EXP_QP_INIT_ATTR_CREATE_FLAGS;
        qp_init_attr.exp_create_flags = IBV_EXP_QP_CREATE_ATOMIC_BE_REPLY;
    }
#  endif

#  if HAVE_STRUCT_IBV_EXP_QP_INIT_ATTR_MAX_INL_RECV
    qp_init_attr.comp_mask           |= IBV_EXP_QP_INIT_ATTR_INL_RECV;
    qp_init_attr.max_inl_recv       = iface->config.rx_inline;
#  endif

    qp = ibv_exp_create_qp(dev->ibv_context, &qp_init_attr);
#else
    qp = ibv_create_qp(dev->pd, &qp_init_attr);
#endif
    if (qp == NULL) {
        ucs_error("failed to create qp: %m");
        return UCS_ERR_IO_ERROR;
    }

#if HAVE_STRUCT_IBV_EXP_QP_INIT_ATTR_MAX_INL_RECV
    qp_init_attr.max_inl_recv = qp_init_attr.max_inl_recv / 2; /* Driver bug W/A */
#endif

    *qp_p = qp;
    *cap  = qp_init_attr.cap;
    return UCS_OK;
}
