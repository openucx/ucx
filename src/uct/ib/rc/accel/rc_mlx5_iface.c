/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/ib/mlx5/ib_mlx5_log.h>
#include <uct/ib/mlx5/dv/ib_mlx5_dv.h>
#include <uct/ib/base/ib_device.h>
#include <uct/base/uct_md.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>

#include "rc_mlx5.inl"


/**
 * RC mlx5 interface configuration
 */
typedef struct uct_rc_mlx5_iface_config {
    uct_rc_mlx5_iface_common_config_t super;
    uct_rc_common_config_t            rc_common;
    /* TODO wc_mode, UAR mode SnB W/A... */
} uct_rc_mlx5_iface_config_t;


ucs_config_field_t uct_rc_mlx5_iface_config_table[] = {
  {"RC_", "", NULL,
   ucs_offsetof(uct_rc_mlx5_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_rc_mlx5_common_config_table)},

  {"", "", NULL,
   ucs_offsetof(uct_rc_mlx5_iface_config_t, rc_common),
   UCS_CONFIG_TYPE_TABLE(uct_rc_common_config_table)},

  {NULL}
};


static uct_rc_iface_ops_t uct_rc_mlx5_iface_ops;

#if ENABLE_STATS
ucs_stats_class_t uct_rc_mlx5_iface_stats_class = {
    .name = "mlx5",
    .num_counters = UCT_RC_MLX5_IFACE_STAT_LAST,
    .counter_names = {
     [UCT_RC_MLX5_IFACE_STAT_RX_INL_32] = "rx_inl_32",
     [UCT_RC_MLX5_IFACE_STAT_RX_INL_64] = "rx_inl_64"
    }
};
#endif

void uct_rc_mlx5_iface_check_rx_completion(uct_rc_mlx5_iface_common_t *iface,
                                           struct mlx5_cqe64 *cqe)
{
    uct_ib_mlx5_cq_t *cq      = &iface->cq[UCT_IB_DIR_RX];
    struct mlx5_err_cqe *ecqe = (void*)cqe;
    uct_ib_mlx5_srq_seg_t *seg;
    uint16_t wqe_ctr;

    ucs_memory_cpu_load_fence();

    if (((ecqe->op_own >> 4) == MLX5_CQE_RESP_ERR) &&
        (ecqe->syndrome == MLX5_CQE_SYNDROME_REMOTE_ABORTED_ERR) &&
        (ecqe->vendor_err_synd == UCT_IB_MLX5_CQE_VENDOR_SYND_ODP))
    {
        /* Release the aborted segment */
        wqe_ctr = ntohs(ecqe->wqe_counter);
        seg     = uct_ib_mlx5_srq_get_wqe(&iface->rx.srq, wqe_ctr);
        ++cq->cq_ci;
        uct_rc_mlx5_iface_release_srq_seg(iface, seg, wqe_ctr, UCS_OK,
                                          iface->super.super.config.rx_headroom_offset,
                                          &iface->super.super.release_desc);
    } else {
        ucs_assert((ecqe->op_own >> 4) != MLX5_CQE_INVALID);
        uct_ib_mlx5_check_completion(&iface->super.super, cq, cqe);
    }
}

static UCS_F_ALWAYS_INLINE unsigned
uct_rc_mlx5_iface_poll_tx(uct_rc_mlx5_iface_common_t *iface)
{
    struct mlx5_cqe64 *cqe;
    uct_rc_mlx5_ep_t *ep;
    unsigned qp_num;
    uint16_t hw_ci;

    cqe = uct_ib_mlx5_poll_cq(&iface->super.super, &iface->cq[UCT_IB_DIR_TX]);
    if (cqe == NULL) {
        return 0;
    }

    UCS_STATS_UPDATE_COUNTER(iface->super.stats, UCT_RC_IFACE_STAT_TX_COMPLETION, 1);

    ucs_memory_cpu_load_fence();

    qp_num = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
    ep = ucs_derived_of(uct_rc_iface_lookup_ep(&iface->super, qp_num), uct_rc_mlx5_ep_t);
    ucs_assert(ep != NULL);
    hw_ci = ntohs(cqe->wqe_counter);
    ucs_trace_poll("rc_mlx5 iface %p tx_cqe: ep %p qpn 0x%x hw_ci %d", iface, ep,
                   qp_num, hw_ci);

    uct_rc_mlx5_common_update_tx_res(&iface->super, &ep->tx.wq, &ep->super.txqp,
                                     hw_ci);
    uct_rc_mlx5_txqp_process_tx_cqe(&ep->super.txqp, cqe, hw_ci);

    ucs_arbiter_group_schedule(&iface->super.tx.arbiter, &ep->super.arb_group);
    ucs_arbiter_dispatch(&iface->super.tx.arbiter, 1, uct_rc_ep_process_pending, NULL);

    return 1;
}

unsigned uct_rc_mlx5_iface_progress(void *arg)
{
    uct_rc_mlx5_iface_common_t *iface = arg;
    unsigned count;

    count = uct_rc_mlx5_iface_common_poll_rx(iface, 0);
    if (count > 0) {
        return count;
    }
    return uct_rc_mlx5_iface_poll_tx(iface);
}

static ucs_status_t uct_rc_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(tl_iface, uct_rc_mlx5_iface_common_t);
    uct_rc_iface_t *rc_iface   = &iface->super;
    size_t max_am_inline       = UCT_IB_MLX5_AM_MAX_SHORT(0);
    size_t max_put_inline      = UCT_IB_MLX5_PUT_MAX_SHORT(0);
    ucs_status_t status;

#if HAVE_IBV_DM
    if (iface->dm.dm != NULL) {
        max_am_inline  = ucs_max(iface->dm.dm->seg_len, UCT_IB_MLX5_AM_MAX_SHORT(0));
        max_put_inline = ucs_max(iface->dm.dm->seg_len, UCT_IB_MLX5_PUT_MAX_SHORT(0));
    }
#endif

    status = uct_rc_iface_query(rc_iface, iface_attr,
                                max_put_inline,
                                max_am_inline,
                                UCT_IB_MLX5_AM_ZCOPY_MAX_HDR(0),
                                UCT_IB_MLX5_AM_ZCOPY_MAX_IOV,
                                UCT_RC_MLX5_TM_EAGER_ZCOPY_MAX_IOV(0),
                                sizeof(uct_rc_mlx5_hdr_t));
    if (status != UCS_OK) {
        return status;
    }

    uct_rc_mlx5_iface_common_query(&rc_iface->super, iface_attr, max_am_inline, 0);
    iface_attr->latency.growth += 1e-9; /* 1 ns per each extra QP */
    iface_attr->ep_addr_len     = sizeof(uct_rc_mlx5_ep_address_t);
    return UCS_OK;
}

static ucs_status_t uct_rc_mlx5_iface_arm_cq(uct_ib_iface_t *ib_iface,
                                             uct_ib_dir_t dir,
                                             int solicited_only)
{
    uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(ib_iface, uct_rc_mlx5_iface_common_t);
#if HAVE_DECL_MLX5DV_INIT_OBJ
    return uct_ib_mlx5dv_arm_cq(&iface->cq[dir], solicited_only);
#else
    uct_ib_mlx5_update_cq_ci(iface->super.super.cq[dir],
                             iface->cq[dir].cq_ci);
    return uct_ib_iface_arm_cq(ib_iface, dir, solicited_only);
#endif
}

static void
uct_rc_mlx5_iface_handle_failure(uct_ib_iface_t *ib_iface, void *arg,
                                 ucs_status_t status)
{
    struct mlx5_cqe64  *cqe    = arg;
    uct_rc_iface_t     *iface  = ucs_derived_of(ib_iface, uct_rc_iface_t);
    unsigned           qp_num  = ntohl(cqe->sop_drop_qpn) &
                                 UCS_MASK(UCT_IB_QPN_ORDER);
    uct_rc_mlx5_ep_t   *ep     = ucs_derived_of(uct_rc_iface_lookup_ep(iface,
                                                                       qp_num),
                                                uct_rc_mlx5_ep_t);
    ucs_log_level_t    log_lvl = UCS_LOG_LEVEL_FATAL;
    uct_ib_mlx5_txwq_t txwq_copy;
    size_t             txwq_size;

    if (!ep) {
        return;
    }

    /* Create a copy of RC txwq for completion error reporting, since the QP
     * would be released by set_ep_failed()*/
    txwq_copy = ep->tx.wq;
    txwq_size = ep->tx.wq.qend - ep->tx.wq.qstart;
    txwq_copy.qstart = ucs_malloc(txwq_size, "rc_txwq_copy");
    if (txwq_copy.qstart != NULL) {
        memcpy(txwq_copy.qstart, ep->tx.wq.qstart, txwq_size);
        txwq_copy.qend = txwq_copy.qstart + txwq_size;
    }

    if (uct_rc_mlx5_ep_handle_failure(ep, status) == UCS_OK) {
        log_lvl = ib_iface->super.config.failure_level;
    }

    uct_ib_mlx5_completion_with_err(ib_iface, arg,
                                    txwq_copy.qstart ? &txwq_copy : NULL,
                                    log_lvl);
    ucs_free(txwq_copy.qstart);
}

static void uct_rc_mlx5_iface_progress_enable(uct_iface_h tl_iface, unsigned flags)
{
    uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(tl_iface, uct_rc_mlx5_iface_common_t);

    if (flags & UCT_PROGRESS_RECV) {
        uct_rc_mlx5_iface_common_prepost_recvs(iface);
    }

    uct_base_iface_progress_enable_cb(&iface->super.super.super,
                                      iface->super.progress, flags);
}

ucs_status_t uct_rc_mlx5_iface_create_qp(uct_rc_mlx5_iface_common_t *iface,
                                         uct_ib_mlx5_qp_t *qp,
                                         uct_ib_mlx5_txwq_t *txwq,
                                         uct_ib_qp_attr_t *attr)
{
    uct_ib_iface_t *ib_iface           = &iface->super.super;
    ucs_status_t status;
#if HAVE_DECL_MLX5DV_CREATE_QP
    uct_ib_device_t *dev               = uct_ib_iface_device(ib_iface);
    struct mlx5dv_qp_init_attr dv_attr = {};

    status = uct_ib_mlx5_iface_fill_attr(ib_iface, qp, attr);
    if (status != UCS_OK) {
        return status;
    }

    uct_ib_iface_fill_attr(ib_iface, attr);
#if HAVE_DECL_MLX5DV_QP_CREATE_ALLOW_SCATTER_TO_CQE
    dv_attr.comp_mask    = MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
    dv_attr.create_flags = MLX5DV_QP_CREATE_ALLOW_SCATTER_TO_CQE;
#endif
    qp->verbs.qp = mlx5dv_create_qp(dev->ibv_context, &attr->ibv, &dv_attr);
    if (qp->verbs.qp == NULL) {
        ucs_error("mlx5dv_create_qp("UCT_IB_IFACE_FMT"): failed: %m",
                  UCT_IB_IFACE_ARG(ib_iface));
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    qp->qp_num = qp->verbs.qp->qp_num;
#else
    status = uct_ib_mlx5_iface_create_qp(ib_iface, qp, attr);
    if (status != UCS_OK) {
        goto err;
    }
#endif

    status = uct_rc_iface_qp_init(&iface->super, qp->verbs.qp);
    if (status != UCS_OK) {
        goto err_destory_qp;
    }

    if (attr->cap.max_send_wr) {
        status = uct_ib_mlx5_txwq_init(iface->super.super.super.worker,
                                       iface->tx.mmio_mode, txwq,
                                       qp->verbs.qp);
        if (status != UCS_OK) {
            ucs_error("Failed to get mlx5 QP information");
            goto err_destory_qp;
        }
    }

    return UCS_OK;

err_destory_qp:
    ibv_destroy_qp(qp->verbs.qp);
err:
    return status;
}

#if IBV_HW_TM
static unsigned uct_rc_mlx5_iface_progress_tm(void *arg)
{
    uct_rc_mlx5_iface_common_t *iface = arg;
    unsigned count;

    count = uct_rc_mlx5_iface_common_poll_rx(iface, 1);
    if (count > 0) {
        return count;
    }
    return uct_rc_mlx5_iface_poll_tx(iface);
}

static ucs_status_t uct_rc_mlx5_iface_tag_recv_zcopy(uct_iface_h tl_iface,
                                                     uct_tag_t tag,
                                                     uct_tag_t tag_mask,
                                                     const uct_iov_t *iov,
                                                     size_t iovcnt,
                                                     uct_tag_context_t *ctx)
{
    uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(tl_iface, uct_rc_mlx5_iface_common_t);

    return uct_rc_mlx5_iface_common_tag_recv(iface, tag, tag_mask, iov,
                                             iovcnt, ctx);
}

static ucs_status_t uct_rc_mlx5_iface_tag_recv_cancel(uct_iface_h tl_iface,
                                                      uct_tag_context_t *ctx,
                                                      int force)
{
   uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(tl_iface, uct_rc_mlx5_iface_common_t);

   return uct_rc_mlx5_iface_common_tag_recv_cancel(iface, ctx, force);
}
#endif

static void uct_rc_mlx5_iface_preinit(uct_rc_mlx5_iface_common_t *iface, uct_md_h md,
                                      uct_rc_mlx5_iface_common_config_t *config,
                                      const uct_iface_params_t *params,
                                      uct_ib_iface_init_attr_t *init_attr)
{
#if IBV_HW_TM
    uct_ib_device_t UCS_V_UNUSED *dev = &ucs_derived_of(md, uct_ib_md_t)->dev;
    uint32_t cap_flags                = IBV_DEVICE_TM_FLAGS(dev);
    struct ibv_tmh tmh;

    iface->tm.enabled = config->tm.enable &&
                        (cap_flags & init_attr->tm_cap_bit);

    if (!iface->tm.enabled) {
        goto out_tm_disabled;
    }

    /* Compile-time check that THM and uct_rc_mlx5_hdr_t are wire-compatible for the
     * case of no-tag protocol.
     */
    UCS_STATIC_ASSERT(sizeof(tmh.opcode) == sizeof(((uct_rc_mlx5_hdr_t*)0)->tmh_opcode));
    UCS_STATIC_ASSERT(ucs_offsetof(struct ibv_tmh, opcode) ==
                      ucs_offsetof(uct_rc_mlx5_hdr_t, tmh_opcode));

    UCS_STATIC_ASSERT(sizeof(uct_rc_mlx5_ctx_priv_t) <= UCT_TAG_PRIV_LEN);

    iface->tm.eager_unexp.cb  = (params->field_mask &
                                 UCT_IFACE_PARAM_FIELD_HW_TM_EAGER_CB) ?
                                params->eager_cb : NULL;
    iface->tm.eager_unexp.arg = (params->field_mask &
                                 UCT_IFACE_PARAM_FIELD_HW_TM_EAGER_ARG) ?
                                params->eager_arg : NULL;
    iface->tm.rndv_unexp.cb   = (params->field_mask &
                                 UCT_IFACE_PARAM_FIELD_HW_TM_RNDV_CB) ?
                                params->rndv_cb : NULL;
    iface->tm.rndv_unexp.arg  = (params->field_mask &
                                 UCT_IFACE_PARAM_FIELD_HW_TM_RNDV_ARG) ?
                                params->rndv_arg : NULL;
    iface->tm.unexpected_cnt  = 0;
    iface->tm.num_outstanding = 0;
    iface->tm.num_tags        = ucs_min(IBV_DEVICE_TM_CAPS(dev, max_num_tags),
                                        config->tm.list_size);

    /* There can be:
     * - up to rx.queue_len RX CQEs
     * - up to 3 CQEs for every posted tag: ADD, TM_CONSUMED and MSG_ARRIVED
     * - one SYNC CQE per every IBV_DEVICE_MAX_UNEXP_COUNT unexpected receives */
    UCS_STATIC_ASSERT(IBV_DEVICE_MAX_UNEXP_COUNT);
    init_attr->rx_cq_len = config->super.super.rx.queue_len + iface->tm.num_tags * 3 +
                           config->super.super.rx.queue_len /
                           IBV_DEVICE_MAX_UNEXP_COUNT;
    init_attr->seg_size  = ucs_max(config->tm.seg_size,
                                   config->super.super.seg_size);
    return;

out_tm_disabled:
#else
    iface->tm.enabled    = 0;
#endif
    init_attr->rx_cq_len = config->super.super.rx.queue_len;
    init_attr->seg_size  = config->super.super.seg_size;
}

static ucs_status_t
uct_rc_mlx5_init_rx(uct_rc_iface_t *rc_iface,
                    const uct_rc_iface_config_t *rc_config)
{
    uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(rc_iface, uct_rc_mlx5_iface_common_t);
#if IBV_HW_TM
    uct_rc_mlx5_iface_common_config_t *config = ucs_derived_of(rc_config,
                                                               uct_rc_mlx5_iface_common_config_t);

    if (UCT_RC_MLX5_TM_ENABLED(iface)) {
        struct ibv_exp_create_srq_attr srq_init_attr = {};

        iface->super.progress = uct_rc_mlx5_iface_progress_tm;
        return uct_rc_mlx5_init_rx_tm(iface, config, &srq_init_attr,
                                      sizeof(struct ibv_rvh), 0);
    }
#endif
    iface->super.progress = uct_rc_mlx5_iface_progress;
    return uct_rc_iface_init_rx(rc_iface, rc_config);
}

static void uct_rc_mlx5_iface_event_cq(uct_ib_iface_t *ib_iface,
                                       uct_ib_dir_t dir)
{
    uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(ib_iface, uct_rc_mlx5_iface_common_t);

    iface->cq[dir].cq_sn++;
}

UCS_CLASS_INIT_FUNC(uct_rc_mlx5_iface_common_t,
                    uct_rc_iface_ops_t *ops,
                    uct_md_h md, uct_worker_h worker,
                    const uct_iface_params_t *params,
                    uct_rc_mlx5_iface_common_config_t *config,
                    uct_ib_iface_init_attr_t *init_attr)
{
    uct_ib_device_t *dev;
    ucs_status_t status;

    uct_rc_mlx5_iface_preinit(self, md, config, params, init_attr);

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, ops, md, worker, params,
                              &config->super, init_attr);

    dev                              = uct_ib_iface_device(&self->super.super);
    self->tx.mmio_mode               = config->mlx5_common.mmio_mode;
    self->tx.bb_max                  = ucs_min(config->tx_max_bb, UINT16_MAX);

    status = uct_ib_mlx5_get_cq(self->super.super.cq[UCT_IB_DIR_TX], &self->cq[UCT_IB_DIR_TX]);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_get_cq(self->super.super.cq[UCT_IB_DIR_RX], &self->cq[UCT_IB_DIR_RX]);
    if (status != UCS_OK) {
        return status;
    }

    status = UCS_STATS_NODE_ALLOC(&self->stats, &uct_rc_mlx5_iface_stats_class,
                                  self->super.stats);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_rc_mlx5_iface_common_tag_init(self, config);
    if (status != UCS_OK) {
        goto cleanup_stats;
    }

    status = uct_ib_mlx5_srq_init(&self->rx.srq, self->super.rx.srq.srq,
                                  self->super.super.config.seg_size);
    if (status != UCS_OK) {
        goto cleanup_tm;
    }

    status = uct_rc_mlx5_iface_common_dm_init(self, &self->super, &config->mlx5_common);
    if (status != UCS_OK) {
        goto cleanup_tm;
    }

    self->super.config.fence = uct_ib_device_has_pci_atomics(dev);
    self->super.rx.srq.quota = self->rx.srq.mask + 1;

    /* By default set to something that is always in cache */
    self->rx.pref_ptr = self;

    status = uct_iface_mpool_init(&self->super.super.super,
                                  &self->tx.atomic_desc_mp,
                                  sizeof(uct_rc_iface_send_desc_t) + UCT_IB_MAX_ATOMIC_SIZE,
                                  sizeof(uct_rc_iface_send_desc_t) + UCT_IB_MAX_ATOMIC_SIZE,
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &config->super.super.tx.mp,
                                  self->super.config.tx_qp_len,
                                  uct_rc_iface_send_desc_init,
                                  "rc_mlx5_atomic_desc");
    if (status != UCS_OK) {
        goto cleanup_dm;
    }

    /* For little-endian atomic reply, override the default functions, to still
     * treat the response as big-endian when it arrives in the CQE.
     */
    if (!(uct_ib_iface_device(&self->super.super)->atomic_arg_sizes_be & sizeof(uint64_t))) {
        self->super.config.atomic64_handler     = uct_rc_mlx5_common_atomic64_le_handler;
    }
    if (!(uct_ib_iface_device(&self->super.super)->ext_atomic_arg_sizes_be & sizeof(uint32_t))) {
        self->super.config.atomic32_ext_handler = uct_rc_mlx5_common_atomic32_le_handler;
    }
    if (!(uct_ib_iface_device(&self->super.super)->ext_atomic_arg_sizes_be & sizeof(uint64_t))) {
        self->super.config.atomic64_ext_handler = uct_rc_mlx5_common_atomic64_le_handler;
    }

    return UCS_OK;

cleanup_dm:
    uct_rc_mlx5_iface_common_dm_cleanup(self);
cleanup_tm:
    uct_rc_mlx5_iface_common_tag_cleanup(self);
cleanup_stats:
    UCS_STATS_NODE_FREE(self->stats);
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_iface_common_t)
{
    ucs_mpool_cleanup(&self->tx.atomic_desc_mp, 1);
    uct_rc_mlx5_iface_common_dm_cleanup(self);
    uct_rc_mlx5_iface_common_tag_cleanup(self);
    UCS_STATS_NODE_FREE(self->stats);
}

UCS_CLASS_DEFINE(uct_rc_mlx5_iface_common_t, uct_rc_iface_t);

typedef struct {
    uct_rc_mlx5_iface_common_t  super;
} uct_rc_mlx5_iface_t;

UCS_CLASS_INIT_FUNC(uct_rc_mlx5_iface_t,
                    uct_md_h md, uct_worker_h worker,
                    const uct_iface_params_t *params,
                    const uct_iface_config_t *tl_config)
{
    uct_rc_mlx5_iface_config_t *config = ucs_derived_of(tl_config,
                                                        uct_rc_mlx5_iface_config_t);
    uct_ib_iface_init_attr_t init_attr = {};
    ucs_status_t status;

    init_attr.tm_cap_bit     = IBV_TM_CAP_RC;
    init_attr.fc_req_size    = sizeof(uct_rc_fc_request_t);
    init_attr.flags          = UCT_IB_CQ_IGNORE_OVERRUN;
    init_attr.rx_hdr_len     = sizeof(uct_rc_mlx5_hdr_t);
    init_attr.qp_type        = IBV_QPT_RC;

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_mlx5_iface_common_t, &uct_rc_mlx5_iface_ops,
                              md, worker, params, &config->super, &init_attr);

    self->super.super.config.tx_moderation = ucs_min(config->rc_common.tx_cq_moderation,
                                                     self->super.tx.bb_max / 4);

    status = uct_rc_init_fc_thresh(config->rc_common.soft_thresh,
                                   &config->super.super, &self->super.super);
    if (status != UCS_OK) {
        return status;
    }

    /* Set max_iov for put_zcopy and get_zcopy */
    uct_ib_iface_set_max_iov(&self->super.super.super,
                             (UCT_IB_MLX5_MAX_SEND_WQE_SIZE -
                             sizeof(struct mlx5_wqe_raddr_seg) -
                             sizeof(struct mlx5_wqe_ctrl_seg)) /
                             sizeof(struct mlx5_wqe_data_seg));

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_iface_t)
{
    uct_base_iface_progress_disable(&self->super.super.super.super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
}

UCS_CLASS_DEFINE(uct_rc_mlx5_iface_t, uct_rc_mlx5_iface_common_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_mlx5_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_mlx5_iface_t, uct_iface_t);

static uct_rc_iface_ops_t uct_rc_mlx5_iface_ops = {
    {
    {
    .ep_put_short             = uct_rc_mlx5_ep_put_short,
    .ep_put_bcopy             = uct_rc_mlx5_ep_put_bcopy,
    .ep_put_zcopy             = uct_rc_mlx5_ep_put_zcopy,
    .ep_get_bcopy             = uct_rc_mlx5_ep_get_bcopy,
    .ep_get_zcopy             = uct_rc_mlx5_ep_get_zcopy,
    .ep_am_short              = uct_rc_mlx5_ep_am_short,
    .ep_am_bcopy              = uct_rc_mlx5_ep_am_bcopy,
    .ep_am_zcopy              = uct_rc_mlx5_ep_am_zcopy,
    .ep_atomic_cswap64        = uct_rc_mlx5_ep_atomic_cswap64,
    .ep_atomic_cswap32        = uct_rc_mlx5_ep_atomic_cswap32,
    .ep_atomic64_post         = uct_rc_mlx5_ep_atomic64_post,
    .ep_atomic32_post         = uct_rc_mlx5_ep_atomic32_post,
    .ep_atomic64_fetch        = uct_rc_mlx5_ep_atomic64_fetch,
    .ep_atomic32_fetch        = uct_rc_mlx5_ep_atomic32_fetch,
    .ep_pending_add           = uct_rc_ep_pending_add,
    .ep_pending_purge         = uct_rc_ep_pending_purge,
    .ep_flush                 = uct_rc_mlx5_ep_flush,
    .ep_fence                 = uct_rc_mlx5_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_rc_mlx5_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_mlx5_ep_t),
    .ep_get_address           = uct_rc_mlx5_ep_get_address,
    .ep_connect_to_ep         = uct_rc_mlx5_ep_connect_to_ep,
#if IBV_HW_TM
    .ep_tag_eager_short       = uct_rc_mlx5_ep_tag_eager_short,
    .ep_tag_eager_bcopy       = uct_rc_mlx5_ep_tag_eager_bcopy,
    .ep_tag_eager_zcopy       = uct_rc_mlx5_ep_tag_eager_zcopy,
    .ep_tag_rndv_zcopy        = uct_rc_mlx5_ep_tag_rndv_zcopy,
    .ep_tag_rndv_request      = uct_rc_mlx5_ep_tag_rndv_request,
    .ep_tag_rndv_cancel       = uct_rc_mlx5_ep_tag_rndv_cancel,
    .iface_tag_recv_zcopy     = uct_rc_mlx5_iface_tag_recv_zcopy,
    .iface_tag_recv_cancel    = uct_rc_mlx5_iface_tag_recv_cancel,
#endif
    .iface_flush              = uct_rc_iface_flush,
    .iface_fence              = uct_rc_iface_fence,
    .iface_progress_enable    = uct_rc_mlx5_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_rc_iface_do_progress,
    .iface_event_fd_get       = uct_ib_iface_event_fd_get,
    .iface_event_arm          = uct_rc_iface_event_arm,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_mlx5_iface_t),
    .iface_query              = uct_rc_mlx5_iface_query,
    .iface_get_address        = uct_rc_iface_get_address,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_rc_iface_is_reachable
    },
    .create_cq                = uct_ib_mlx5_create_cq,
    .arm_cq                   = uct_rc_mlx5_iface_arm_cq,
    .event_cq                 = uct_rc_mlx5_iface_event_cq,
    .handle_failure           = uct_rc_mlx5_iface_handle_failure,
    .set_ep_failed            = uct_rc_mlx5_ep_set_failed,
    },
    .init_rx                  = uct_rc_mlx5_init_rx,
    .fc_ctrl                  = uct_rc_mlx5_ep_fc_ctrl,
    .fc_handler               = uct_rc_iface_fc_handler,
};


static ucs_status_t uct_rc_mlx5_query_resources(uct_md_h md,
                                                uct_tl_resource_desc_t **resources_p,
                                                unsigned *num_resources_p)
{
    uct_ib_md_t *ib_md = ucs_derived_of(md, uct_ib_md_t);

    return uct_ib_device_query_tl_resources(&ib_md->dev, "rc_mlx5",
                                            UCT_IB_DEVICE_FLAG_MLX5_PRM |
                                            (ib_md->config.eth_pause ? 0 : UCT_IB_DEVICE_FLAG_LINK_IB),
                                            resources_p, num_resources_p);
}

UCT_TL_COMPONENT_DEFINE(uct_rc_mlx5_tl,
                        uct_rc_mlx5_query_resources,
                        uct_rc_mlx5_iface_t,
                        "rc_mlx5",
                        "RC_MLX5_",
                        uct_rc_mlx5_iface_config_table,
                        uct_rc_mlx5_iface_config_t);
UCT_MD_REGISTER_TL(&uct_ib_mdc, &uct_rc_mlx5_tl);
