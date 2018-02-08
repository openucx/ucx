/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/ib/mlx5/ib_mlx5_log.h>
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

  {"TX_MAX_BB", "-1",
   "Limits the number of outstanding WQE building blocks. The actual limit is\n"
   "a minimum between this value and the number of building blocks in the TX QP.\n"
   "-1 means no limit.",
   ucs_offsetof(uct_rc_mlx5_iface_config_t, tx_max_bb), UCS_CONFIG_TYPE_UINT},

  {NULL}
};

static uct_rc_iface_ops_t uct_rc_mlx5_iface_ops;

static UCS_F_ALWAYS_INLINE unsigned
uct_rc_mlx5_iface_poll_tx(uct_rc_mlx5_iface_t *iface)
{
    struct mlx5_cqe64 *cqe;
    uct_rc_mlx5_ep_t *ep;
    unsigned qp_num;
    uint16_t hw_ci, bb_num;

    cqe = uct_ib_mlx5_poll_cq(&iface->super.super, &iface->mlx5_common.tx.cq);
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

    bb_num = uct_ib_mlx5_txwq_update_bb(&ep->tx.wq, hw_ci) -
             uct_rc_txqp_available(&ep->super.txqp);
    uct_rc_txqp_available_add(&ep->super.txqp, bb_num);
    iface->super.tx.cq_available += bb_num;

    uct_rc_mlx5_txqp_process_tx_cqe(&ep->super.txqp, cqe, hw_ci);

    ucs_arbiter_group_schedule(&iface->super.tx.arbiter, &ep->super.arb_group);
    ucs_arbiter_dispatch(&iface->super.tx.arbiter, 1, uct_rc_ep_process_pending, NULL);

    return 1;
}

unsigned uct_rc_mlx5_iface_progress(void *arg)
{
    uct_rc_mlx5_iface_t *iface = arg;
    unsigned count;

    count = uct_rc_mlx5_iface_common_poll_rx(&iface->mlx5_common,
                                             &iface->super, 0);
    if (count > 0) {
        return count;
    }
    return uct_rc_mlx5_iface_poll_tx(iface);
}

static ucs_status_t uct_rc_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);
    ucs_status_t status;

    status = uct_rc_iface_query(iface, iface_attr,
                                UCT_IB_MLX5_PUT_MAX_SHORT(0),
                                UCT_IB_MLX5_AM_MAX_SHORT(0),
                                UCT_IB_MLX5_AM_ZCOPY_MAX_HDR(0),
                                UCT_IB_MLX5_AM_ZCOPY_MAX_IOV);
    if (status != UCS_OK) {
        return status;
    }

    uct_rc_mlx5_iface_common_query(iface_attr);
    iface_attr->latency.growth += 1e-9; /* 1 ns per each extra QP */
    return UCS_OK;
}

static ucs_status_t uct_rc_mlx5_iface_arm_tx_cq(uct_ib_iface_t *ib_iface)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(ib_iface, uct_rc_mlx5_iface_t);
    uct_ib_mlx5_update_cq_ci(iface->super.super.send_cq, iface->mlx5_common.tx.cq.cq_ci);
    return uct_ib_iface_arm_tx_cq(ib_iface);
}

static ucs_status_t uct_rc_mlx5_iface_arm_rx_cq(uct_ib_iface_t *ib_iface,
                                                int solicited_only)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(ib_iface, uct_rc_mlx5_iface_t);
    uct_ib_mlx5_update_cq_ci(iface->super.super.recv_cq, iface->mlx5_common.rx.cq.cq_ci);
    return uct_ib_iface_arm_rx_cq(ib_iface, solicited_only);
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

    uct_ib_mlx5_completion_with_err(arg, log_lvl);
}

static ucs_status_t uct_rc_mlx5_ep_set_failed(uct_ib_iface_t *iface,
                                              uct_ep_h ep, ucs_status_t status)
{
    return uct_set_ep_failed(&UCS_CLASS_NAME(uct_rc_mlx5_ep_t), ep,
                             &iface->super.super, status);
}

static void uct_rc_mlx5_iface_progress_enable(uct_iface_h tl_iface, unsigned flags)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_mlx5_iface_t);

    if (flags & UCT_PROGRESS_RECV) {
        uct_rc_mlx5_iface_common_prepost_recvs(&iface->super, &iface->mlx5_common);
    }

    uct_base_iface_progress_enable_cb(&iface->super.super.super,
                                      iface->super.progress, flags);
}

#if IBV_EXP_HW_TM
static unsigned uct_rc_mlx5_iface_progress_tm(void *arg)
{
    uct_rc_mlx5_iface_t *iface = arg;
    unsigned count;

    count = uct_rc_mlx5_iface_common_poll_rx(&iface->mlx5_common,
                                             &iface->super, 1);
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
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_mlx5_iface_t);

    return uct_rc_mlx5_iface_common_tag_recv(&iface->mlx5_common, &iface->super,
                                             tag, tag_mask, iov, iovcnt, ctx);
}

static ucs_status_t uct_rc_mlx5_iface_tag_recv_cancel(uct_iface_h tl_iface,
                                                      uct_tag_context_t *ctx,
                                                      int force)
{
   uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_mlx5_iface_t);

   return uct_rc_mlx5_iface_common_tag_recv_cancel(&iface->mlx5_common,
                                                   &iface->super, ctx, force);
}
#endif

static ucs_status_t
uct_rc_mlx5_iface_tag_init(uct_rc_mlx5_iface_t *iface,
                           uct_rc_iface_config_t *rc_config)
{
#if IBV_EXP_HW_TM
    if (UCT_RC_IFACE_TM_ENABLED(&iface->super)) {
        struct ibv_exp_create_srq_attr srq_init_attr = {};

        iface->super.progress = uct_rc_mlx5_iface_progress_tm;

        return uct_rc_mlx5_iface_common_tag_init(&iface->mlx5_common,
                                                 &iface->super, rc_config,
                                                 &srq_init_attr,
                                                 sizeof(struct ibv_exp_tmh_rvh));
    }
#endif
    iface->super.progress = uct_rc_mlx5_iface_progress;
    return UCS_OK;
}


static UCS_CLASS_INIT_FUNC(uct_rc_mlx5_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_rc_mlx5_iface_config_t *config = ucs_derived_of(tl_config,
                                                        uct_rc_mlx5_iface_config_t);
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, &uct_rc_mlx5_iface_ops, md, worker,
                              params, &config->super, 0,
                              sizeof(uct_rc_fc_request_t), IBV_EXP_TM_CAP_RC);


    self->tx.bb_max                  = ucs_min(config->tx_max_bb, UINT16_MAX);
    self->super.config.tx_moderation = ucs_min(self->super.config.tx_moderation,
                                               self->tx.bb_max / 4);

    status = uct_rc_init_fc_thresh(&config->fc, &config->super, &self->super);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_rc_mlx5_iface_tag_init(self, &config->super);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_rc_mlx5_iface_common_init(&self->mlx5_common, &self->super, &config->super);
    if (status != UCS_OK) {
        uct_rc_mlx5_iface_common_tag_cleanup(&self->mlx5_common, &self->super);
        return status;
    }

    /* Set max_iov for put_zcopy and get_zcopy */
    uct_ib_iface_set_max_iov(&self->super.super,
                             ((UCT_IB_MLX5_MAX_BB * MLX5_SEND_WQE_BB) -
                             sizeof(struct mlx5_wqe_raddr_seg) -
                             sizeof(struct mlx5_wqe_ctrl_seg)) /
                             sizeof(struct mlx5_wqe_data_seg));

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_iface_t)
{
    uct_base_iface_progress_disable(&self->super.super.super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    uct_rc_mlx5_iface_common_cleanup(&self->mlx5_common);

    uct_rc_mlx5_iface_common_tag_cleanup(&self->mlx5_common, &self->super);
}


UCS_CLASS_DEFINE(uct_rc_mlx5_iface_t, uct_rc_iface_t);
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
    .ep_atomic_add64          = uct_rc_mlx5_ep_atomic_add64,
    .ep_atomic_fadd64         = uct_rc_mlx5_ep_atomic_fadd64,
    .ep_atomic_swap64         = uct_rc_mlx5_ep_atomic_swap64,
    .ep_atomic_cswap64        = uct_rc_mlx5_ep_atomic_cswap64,
    .ep_atomic_add32          = uct_rc_mlx5_ep_atomic_add32,
    .ep_atomic_fadd32         = uct_rc_mlx5_ep_atomic_fadd32,
    .ep_atomic_swap32         = uct_rc_mlx5_ep_atomic_swap32,
    .ep_atomic_cswap32        = uct_rc_mlx5_ep_atomic_cswap32,
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
    .arm_tx_cq                = uct_rc_mlx5_iface_arm_tx_cq,
    .arm_rx_cq                = uct_rc_mlx5_iface_arm_rx_cq,
    .handle_failure           = uct_rc_mlx5_iface_handle_failure,
    .set_ep_failed            = uct_rc_mlx5_ep_set_failed
    },
    .fc_ctrl                  = uct_rc_mlx5_ep_fc_ctrl,
    .fc_handler               = uct_rc_iface_fc_handler
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
