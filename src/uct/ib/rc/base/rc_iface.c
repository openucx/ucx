/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rc_iface.h"
#include "rc_ep.h"

#include <ucs/arch/cpu.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/type/class.h>


ucs_config_field_t uct_rc_iface_config_table[] = {
  {"IB_", "RX_INLINE=64;RX_QUEUE_LEN=4095", NULL,
   ucs_offsetof(uct_rc_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_ib_iface_config_table)},

  {"PATH_MTU", "default",
   "Path MTU. \"default\" will select the best MTU for the device.",
   ucs_offsetof(uct_rc_iface_config_t, path_mtu), UCS_CONFIG_TYPE_ENUM(uct_ib_mtu_values)},

  {"MAX_RD_ATOMIC", "4",
   "Maximal number of outstanding read or atomic replies",
   ucs_offsetof(uct_rc_iface_config_t, max_rd_atomic), UCS_CONFIG_TYPE_UINT},

  {"TIMEOUT", "1.0s",
   "Transport timeout",
   ucs_offsetof(uct_rc_iface_config_t, tx.timeout), UCS_CONFIG_TYPE_TIME},

  {"RETRY_COUNT", "7",
   "Transport retries",
   ucs_offsetof(uct_rc_iface_config_t, tx.retry_count), UCS_CONFIG_TYPE_UINT},

  {"RNR_TIMEOUT", "100ms",
   "RNR timeout",
   ucs_offsetof(uct_rc_iface_config_t,tx. rnr_timeout), UCS_CONFIG_TYPE_TIME},

  {"RNR_RETRY_COUNT", "7",
   "RNR retries",
   ucs_offsetof(uct_rc_iface_config_t, tx.rnr_retry_count), UCS_CONFIG_TYPE_UINT},

  {"TX_CQ_LEN", "4096",
   "Length of send completion queue. This limits the total number of outstanding signaled sends.",
   ucs_offsetof(uct_rc_iface_config_t, tx.cq_len), UCS_CONFIG_TYPE_UINT},

  {NULL}
};


#if ENABLE_STATS
static ucs_stats_class_t uct_rc_iface_stats_class = {
    .name = "rc_iface",
    .num_counters = UCT_RC_IFACE_STAT_LAST,
    .counter_names = {
        [UCT_RC_IFACE_STAT_RX_COMPLETION] = "rx_completion",
        [UCT_RC_IFACE_STAT_TX_COMPLETION] = "tx_completion",
        [UCT_RC_IFACE_STAT_NO_CQE]        = "no_cqe",
    }
};
#endif

void uct_rc_iface_query(uct_rc_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    memset(&iface_attr->cap, 0, sizeof(iface_attr->cap));
    iface_attr->iface_addr_len      = sizeof(uct_sockaddr_ib_subnet_t);
    iface_attr->ep_addr_len         = sizeof(uct_sockaddr_ib_t);
    iface_attr->cap.flags           = UCT_IFACE_FLAG_AM_SHORT |
                                      UCT_IFACE_FLAG_AM_BCOPY |
                                      UCT_IFACE_FLAG_AM_ZCOPY |
                                      UCT_IFACE_FLAG_PUT_SHORT |
                                      UCT_IFACE_FLAG_PUT_BCOPY |
                                      UCT_IFACE_FLAG_PUT_ZCOPY |
                                      UCT_IFACE_FLAG_GET_BCOPY |
                                      UCT_IFACE_FLAG_GET_ZCOPY |
                                      UCT_IFACE_FLAG_PENDING   |
                                      UCT_IFACE_FLAG_CONNECT_TO_EP |
                                      UCT_IFACE_FLAG_AM_THREAD_SINGLE;
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
    ucs_list_add_head(&iface->ep_list, &ep->list);
}

void uct_rc_iface_remove_ep(uct_rc_iface_t *iface, uct_rc_ep_t *ep)
{
    unsigned qp_num = ep->qp->qp_num;
    uct_rc_ep_t **memb;

    memb = &iface->eps[qp_num >> UCT_RC_QP_TABLE_ORDER]
                      [qp_num &  UCS_MASK(UCT_RC_QP_TABLE_MEMB_ORDER)];
    ucs_assert(*memb != NULL);
    *memb = NULL;
    ucs_list_del(&ep->list);
}

ucs_status_t uct_rc_iface_flush(uct_iface_h tl_iface)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);
    ucs_status_t status;
    unsigned count;
    uct_rc_ep_t *ep;

    count = 0;
    ucs_list_for_each(ep, &iface->ep_list, list) {
        status = uct_ep_flush(&ep->super.super);
        if ((status == UCS_ERR_NO_RESOURCE) || (status == UCS_INPROGRESS)) {
            ++count;
        } else if (status != UCS_OK) {
            return status;
        }
    }

    if (count != 0) {
        return UCS_ERR_NO_RESOURCE;
    }

    UCT_TL_IFACE_STAT_FLUSH(&iface->super.super);
    return UCS_OK;
}

void uct_rc_iface_send_desc_init(uct_iface_h tl_iface, void *obj, uct_mem_h memh)
{
    uct_rc_iface_send_desc_t *desc = obj;
    struct ibv_mr *mr = memh;
    desc->lkey = mr->lkey;
}

static void uct_rc_iface_set_path_mtu(uct_rc_iface_t *iface,
                                      uct_rc_iface_config_t *config)
{
    enum ibv_mtu port_mtu = uct_ib_iface_port_attr(&iface->super)->active_mtu;
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super);

    /* MTU is set by user configuration */
    if (config->path_mtu != UCT_IB_MTU_DEFAULT) {
        iface->config.path_mtu = config->path_mtu + (IBV_MTU_512 - UCT_IB_MTU_512);
    } else if ((port_mtu > IBV_MTU_2048) && (dev->dev_attr.vendor_id == 0x02c9) &&
        ((dev->dev_attr.vendor_part_id == 4099) || (dev->dev_attr.vendor_part_id == 4100) ||
         (dev->dev_attr.vendor_part_id == 4103) || (dev->dev_attr.vendor_part_id == 4104)))
    {
        /* On some devices optimal path_mtu is 2048 */
        iface->config.path_mtu = IBV_MTU_2048;
    } else {
        iface->config.path_mtu = port_mtu;
    }
}

UCS_CLASS_INIT_FUNC(uct_rc_iface_t, uct_iface_ops_t *ops, uct_pd_h pd,
                    uct_worker_h worker, const char *dev_name, unsigned rx_headroom,
                    unsigned rx_priv_len, uct_rc_iface_config_t *config)
{
    struct ibv_srq_init_attr srq_init_attr;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_ib_iface_t, ops, pd, worker, dev_name, rx_headroom,
                              rx_priv_len, sizeof(uct_rc_hdr_t), config->tx.cq_len,
                              &config->super);

    self->tx.cq_available           = config->tx.cq_len - 1; /* Reserve one for error */
    self->tx.next_op                = 0;
    self->rx.available              = 0;
    self->config.tx_qp_len          = config->super.tx.queue_len;
    self->config.tx_min_sge         = config->super.tx.min_sge;
    self->config.tx_min_inline      = config->super.tx.min_inline;
    self->config.tx_moderation      = ucs_min(ucs_roundup_pow2(config->super.tx.cq_moderation),
                                              ucs_roundup_pow2(config->super.tx.queue_len / 4));
    self->config.tx_ops_mask        = ucs_roundup_pow2(config->tx.cq_len) - 1;
    self->config.rx_max_batch       = ucs_min(config->super.rx.max_batch, config->super.rx.queue_len / 4);
    self->config.rx_inline          = config->super.rx.inl;
    self->config.min_rnr_timer      = uct_ib_to_fabric_time(config->tx.rnr_timeout);
    self->config.timeout            = uct_ib_to_fabric_time(config->tx.timeout);
    self->config.rnr_retry          = ucs_min(config->tx.rnr_retry_count,
                                              UCR_RC_QP_MAX_RETRY_COUNT);
    self->config.retry_cnt          = ucs_min(config->tx.retry_count,
                                              UCR_RC_QP_MAX_RETRY_COUNT);
    self->config.max_rd_atomic      = config->max_rd_atomic;

    uct_rc_iface_set_path_mtu(self, config);
    memset(self->eps, 0, sizeof(self->eps));
    ucs_arbiter_init(&self->tx.arbiter);
    ucs_list_head_init(&self->ep_list);

    /* Create RX buffers mempool */
    status = uct_ib_iface_recv_mpool_init(&self->super, &config->super,
                                          "rc_recv_desc", &self->rx.mp);
    if (status != UCS_OK) {
        goto err;
    }

    /* Create TX buffers mempool */
    status = uct_iface_mpool_init(&self->super.super,
                                  &self->tx.mp,
                                  sizeof(uct_rc_iface_send_desc_t) + self->super.config.seg_size,
                                  sizeof(uct_rc_iface_send_desc_t),
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &config->super.tx.mp,
                                  self->config.tx_qp_len,
                                  uct_rc_iface_send_desc_init,
                                  "rc_send_desc");
    if (status != UCS_OK) {
        goto err_destroy_rx_mp;
    }

    /* Allocate tx operations */
    self->tx.ops = ucs_calloc(self->config.tx_ops_mask + 1, sizeof(*self->tx.ops),
                              "rc_tx_ops");
    if (self->tx.ops == NULL) {
        goto err_destroy_tx_mp;
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
        goto err_free_tx_ops;
    }

    self->rx.available           = srq_init_attr.attr.max_wr;

    status = UCS_STATS_NODE_ALLOC(&self->stats, &uct_rc_iface_stats_class,
                                  self->super.super.stats);
    if (status != UCS_OK) {
        goto err_destroy_srq;
    }

    return UCS_OK;

err_destroy_srq:
    ibv_destroy_srq(self->rx.srq);
err_free_tx_ops:
    ucs_free(self->tx.ops);
err_destroy_tx_mp:
    ucs_mpool_cleanup(&self->tx.mp, 1);
err_destroy_rx_mp:
    ucs_mpool_cleanup(&self->rx.mp, 1);
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

    if (!ucs_list_is_empty(&self->ep_list)) {
        ucs_warn("some eps were not destroyed");
    }

    ucs_arbiter_cleanup(&self->tx.arbiter);

    UCS_STATS_NODE_FREE(self->stats);

    /* TODO flush RX buffers */
    ret = ibv_destroy_srq(self->rx.srq);
    if (ret) {
        ucs_warn("failed to destroy SRQ: %m");
    }

    ucs_free(self->tx.ops);
    ucs_mpool_cleanup(&self->tx.mp, 1);
    ucs_mpool_cleanup(&self->rx.mp, 0); /* Cannot flush SRQ */
}

UCS_CLASS_DEFINE(uct_rc_iface_t, uct_ib_iface_t);


ucs_status_t uct_rc_iface_qp_create(uct_rc_iface_t *iface, struct ibv_qp **qp_p,
                                    struct ibv_qp_cap *cap)
{
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super);
    struct ibv_exp_qp_init_attr qp_init_attr;
    struct ibv_qp *qp;
    int inline_recv = 0;

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
    inline_recv = qp_init_attr.max_inl_recv;
#endif

    ucs_debug("created rc qp 0x%x tx %d rx %d tx_inline %d rx_inline %d", qp->qp_num,
              qp_init_attr.cap.max_send_wr, qp_init_attr.cap.max_recv_wr,
              qp_init_attr.cap.max_inline_data, inline_recv);

    *qp_p = qp;
    *cap  = qp_init_attr.cap;
    return UCS_OK;
}
