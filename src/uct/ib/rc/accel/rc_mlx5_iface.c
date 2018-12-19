/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/ib/mlx5/ib_mlx5_log.h>
#include <uct/ib/mlx5/ib_mlx5_dv.h>
#include <uct/ib/base/ib_device.h>
#include <uct/base/uct_md.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>

#include "rc_mlx5.h"

ucs_config_field_t uct_rc_mlx5_iface_config_table[] = {
  {"RC_", "", NULL,
   ucs_offsetof(uct_rc_mlx5_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_rc_iface_config_table)},

  {"", "", NULL,
   ucs_offsetof(uct_rc_mlx5_iface_config_t, fc),
   UCS_CONFIG_TYPE_TABLE(uct_rc_fc_config_table)},

  {"", "", NULL,
   ucs_offsetof(uct_rc_mlx5_iface_config_t, mlx5_common),
   UCS_CONFIG_TYPE_TABLE(uct_ib_mlx5_iface_config_table)},

  {"TX_MAX_BB", "-1",
   "Limits the number of outstanding WQE building blocks. The actual limit is\n"
   "a minimum between this value and the number of building blocks in the TX QP.\n"
   "-1 means no limit.",
   ucs_offsetof(uct_rc_mlx5_iface_config_t, tx_max_bb), UCS_CONFIG_TYPE_UINT},

  {NULL}
};

static uct_rc_mlx5_iface_ops_t uct_rc_mlx5_iface_ops;

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

#if HAVE_IBV_EXP_DM
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
                                UCT_RC_MLX5_TM_EAGER_ZCOPY_MAX_IOV(0));
    if (status != UCS_OK) {
        return status;
    }

    uct_rc_mlx5_iface_common_query(&rc_iface->super, iface_attr);
    iface_attr->latency.growth += 1e-9; /* 1 ns per each extra QP */
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
    struct mlx5_cqe64 *cqe    = arg;
    uct_rc_iface_t    *iface  = ucs_derived_of(ib_iface, uct_rc_iface_t);
    unsigned          qp_num  = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
    uct_rc_mlx5_ep_t  *ep     = ucs_derived_of(uct_rc_iface_lookup_ep(iface, qp_num),
                                               uct_rc_mlx5_ep_t);
    ucs_log_level_t   log_lvl = UCS_LOG_LEVEL_FATAL;
    ucs_status_t      ep_status;

    if (!ep) {
        return;
    }

    uct_rc_txqp_purge_outstanding(&ep->super.txqp, status, 0);
    /* poll_cqe for mlx5 returns NULL in case of failure and the cq_avaialble
       is not updated for the error cqe and all outstanding wqes*/
    iface->tx.cq_available += ep->tx.wq.bb_max -
                              uct_rc_txqp_available(&ep->super.txqp);
    ep_status = ib_iface->ops->set_ep_failed(ib_iface, &ep->super.super.super,
                                             status);
    if (ep_status == UCS_OK) {
        log_lvl = ib_iface->super.config.failure_level;
    }

    uct_ib_mlx5_completion_with_err(ib_iface, arg, log_lvl);
}

static ucs_status_t uct_rc_mlx5_ep_set_failed(uct_ib_iface_t *iface,
                                              uct_ep_h ep, ucs_status_t status)
{
    return uct_set_ep_failed(&UCS_CLASS_NAME(uct_rc_mlx5_ep_t), ep,
                             &iface->super.super, status);
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

static ucs_status_t uct_rc_mlx5_iface_create_qp(uct_ib_iface_t *iface,
                                                uct_ib_qp_attr_t *attr,
                                                struct ibv_qp **qp_p)
{
#if HAVE_DECL_MLX5DV_CREATE_QP
    uct_ib_device_t *dev               = uct_ib_iface_device(iface);
    struct mlx5dv_qp_init_attr dv_attr = {};
    struct ibv_qp *qp;

    uct_ib_iface_fill_attr(iface, attr);
#if HAVE_DECL_MLX5DV_QP_CREATE_ALLOW_SCATTER_TO_CQE
    dv_attr.comp_mask    = MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
    dv_attr.create_flags = MLX5DV_QP_CREATE_ALLOW_SCATTER_TO_CQE;
#endif
    qp = mlx5dv_create_qp(dev->ibv_context, &attr->ibv, &dv_attr);
    if (qp == NULL) {
        ucs_error("iface=%p: failed to create QP: %m", iface);
        return UCS_ERR_IO_ERROR;
    }

    attr->cap = attr->ibv.cap;
    *qp_p     = qp;

    return UCS_OK;
#else
    return uct_ib_iface_create_qp(iface, attr, qp_p);
#endif
}

#if IBV_EXP_HW_TM
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

static ucs_status_t
uct_rc_mlx5_iface_tag_init(uct_rc_mlx5_iface_common_t *iface,
                           uct_rc_mlx5_iface_config_t *config)
{
#if IBV_EXP_HW_TM
    if (UCT_RC_IFACE_TM_ENABLED(&iface->super)) {
        struct ibv_exp_create_srq_attr srq_init_attr = {};

        iface->super.progress = uct_rc_mlx5_iface_progress_tm;

        return uct_rc_mlx5_iface_common_tag_init(iface, config,
                                                 &srq_init_attr,
                                                 sizeof(struct ibv_exp_tmh_rvh));
    }
#endif
    iface->super.progress = uct_rc_mlx5_iface_progress;
    return UCS_OK;
}

static void uct_rc_mlx5_iface_event_cq(uct_ib_iface_t *ib_iface,
                                       uct_ib_dir_t dir)
{
    uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(ib_iface, uct_rc_mlx5_iface_common_t);

    iface->cq[dir].cq_sn++;
}

UCS_CLASS_INIT_FUNC(uct_rc_mlx5_iface_common_t,
                    uct_rc_mlx5_iface_ops_t *ops,
                    uct_md_h md, uct_worker_h worker,
                    const uct_iface_params_t *params,
                    uct_rc_mlx5_iface_config_t *config,
                    uct_ib_iface_init_attr_t *init_attr)
{
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, &ops->super, md, worker, params,
                              &config->super, init_attr);

    self->tx.mmio_mode               = config->mlx5_common.mmio_mode;
    self->tx.bb_max                  = ucs_min(config->tx_max_bb, UINT16_MAX);
    self->super.config.tx_moderation = ucs_min(self->super.config.tx_moderation,
                                               self->tx.bb_max / 4);

    status = uct_ib_mlx5_get_cq(self->super.super.cq[UCT_IB_DIR_TX], &self->cq[UCT_IB_DIR_TX]);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_get_cq(self->super.super.cq[UCT_IB_DIR_RX], &self->cq[UCT_IB_DIR_RX]);
    if (status != UCS_OK) {
        return status;
    }

    status = ops->iface_tag_init(self, config);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_srq_init(&self->rx.srq, self->super.rx.srq.srq,
                                  self->super.super.config.seg_size);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_rc_mlx5_iface_common_dm_init(self, &self->super, &config->mlx5_common);
    if (status != UCS_OK) {
        return status;
    }

    self->super.rx.srq.quota = self->rx.srq.mask + 1;

    /* By default set to something that is always in cache */
    self->rx.pref_ptr = self;

    status = UCS_STATS_NODE_ALLOC(&self->stats, &uct_rc_mlx5_iface_stats_class,
                                  self->super.stats);
    if (status != UCS_OK) {
        goto cleanup_dm;
    }

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
        UCS_STATS_NODE_FREE(self->stats);
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
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_iface_common_t)
{
    UCS_STATS_NODE_FREE(self->stats);
    ucs_mpool_cleanup(&self->tx.atomic_desc_mp, 1);
    uct_rc_mlx5_iface_common_dm_cleanup(self);
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

    init_attr.res_domain_key = UCT_IB_MLX5_RES_DOMAIN_KEY;
    init_attr.tm_cap_bit     = IBV_EXP_TM_CAP_RC;
    init_attr.fc_req_size    = sizeof(uct_rc_fc_request_t);
    init_attr.flags          = UCT_IB_CQ_IGNORE_OVERRUN;

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_mlx5_iface_common_t, &uct_rc_mlx5_iface_ops,
                              md, worker, params, config, &init_attr);

    status = uct_rc_init_fc_thresh(&config->fc, &config->super, &self->super.super);
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
    uct_rc_mlx5_iface_common_tag_cleanup(&self->super);
}

UCS_CLASS_DEFINE(uct_rc_mlx5_iface_t, uct_rc_mlx5_iface_common_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_mlx5_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_mlx5_iface_t, uct_iface_t);

static uct_rc_mlx5_iface_ops_t uct_rc_mlx5_iface_ops = {
    {
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
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_rc_mlx5_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_mlx5_ep_t),
    .ep_get_address           = uct_rc_ep_get_address,
    .ep_connect_to_ep         = uct_rc_ep_connect_to_ep,
#if IBV_EXP_HW_TM
    .ep_tag_eager_short       = uct_rc_mlx5_ep_tag_eager_short,
    .ep_tag_eager_bcopy       = uct_rc_mlx5_ep_tag_eager_bcopy,
    .ep_tag_eager_zcopy       = uct_rc_mlx5_ep_tag_eager_zcopy,
    .ep_tag_rndv_zcopy        = uct_rc_mlx5_ep_tag_rndv_zcopy,
    .ep_tag_rndv_request      = uct_rc_mlx5_ep_tag_rndv_request,
    .ep_tag_rndv_cancel       = uct_rc_ep_tag_rndv_cancel,
    .iface_tag_recv_zcopy     = uct_rc_mlx5_iface_tag_recv_zcopy,
    .iface_tag_recv_cancel    = uct_rc_mlx5_iface_tag_recv_cancel,
#endif
    .iface_flush              = uct_rc_iface_flush,
    .iface_fence              = uct_base_iface_fence,
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
    .create_qp                = uct_rc_mlx5_iface_create_qp
    },
    .fc_ctrl                  = uct_rc_mlx5_ep_fc_ctrl,
    .fc_handler               = uct_rc_iface_fc_handler,
    },
    .iface_tag_init           = uct_rc_mlx5_iface_tag_init,
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
