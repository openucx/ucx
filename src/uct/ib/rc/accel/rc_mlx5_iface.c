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
   ucs_offsetof(uct_rc_mlx5_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_rc_iface_config_table)},

  {"TX_MAX_BB", "-1",
   "Limits the number of outstanding WQE building blocks. The actual limit is\n"
   "a minimum between this value and the number of building blocks in the TX QP.\n"
   "-1 means no limit.",
   ucs_offsetof(uct_rc_mlx5_iface_config_t, tx_max_bb), UCS_CONFIG_TYPE_UINT},

  {NULL}
};

static uct_rc_iface_ops_t uct_rc_mlx5_iface_ops;

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_poll_tx(uct_rc_mlx5_iface_t *iface)
{
    struct mlx5_cqe64 *cqe;
    uct_rc_mlx5_ep_t *ep;
    unsigned qp_num;
    uint16_t hw_ci;

    cqe = uct_ib_mlx5_get_cqe(&iface->super.super, &iface->mlx5_common.tx.cq,
                              UCT_IB_MLX5_CQE64_SIZE_LOG);
    if (cqe == NULL) {
        return;
    }

    UCS_STATS_UPDATE_COUNTER(iface->super.stats, UCT_RC_IFACE_STAT_TX_COMPLETION, 1);

    ucs_memory_cpu_load_fence();

    ucs_assertv(!(cqe->op_own & (MLX5_INLINE_SCATTER_32|MLX5_INLINE_SCATTER_64)),
                "tx inline scatter not supported");

    qp_num = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
    ep = ucs_derived_of(uct_rc_iface_lookup_ep(&iface->super, qp_num), uct_rc_mlx5_ep_t);
    ucs_assert(ep != NULL);

    hw_ci = ntohs(cqe->wqe_counter);
    uct_rc_txqp_available_set(&ep->super.txqp, uct_ib_mlx5_txwq_update_bb(&ep->tx.wq, hw_ci));
    ++iface->super.tx.cq_available;

    uct_rc_ep_process_tx_completion(&iface->super, &ep->super, hw_ci);
    ucs_arbiter_dispatch(&iface->super.tx.arbiter, 1, uct_rc_ep_process_pending, NULL);
}

void uct_rc_mlx5_iface_progress(void *arg)
{
    uct_rc_mlx5_iface_t *iface = arg;
    ucs_status_t status;

    status = uct_rc_mlx5_iface_common_poll_rx(&iface->mlx5_common, &iface->super);
    if (status == UCS_ERR_NO_PROGRESS) {
        uct_rc_mlx5_iface_poll_tx(iface);
    }
}

static ucs_status_t uct_rc_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);

    uct_rc_iface_query(iface, iface_attr);
    uct_rc_mlx5_iface_common_query(iface, iface_attr, IBV_QPT_RC);

    return UCS_OK;
}

static ucs_status_t uct_rc_mlx5_iface_arm_tx_cq(uct_ib_iface_t *ib_iface)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(ib_iface, uct_rc_mlx5_iface_t);
    uct_ib_mlx5_update_cq_ci(iface->super.super.send_cq, iface->mlx5_common.tx.cq.cq_ci);
    return uct_ib_iface_arm_tx_cq(ib_iface);
}

static ucs_status_t uct_rc_mlx5_iface_arm_rx_cq(uct_ib_iface_t *ib_iface,
                                                int solicited)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(ib_iface, uct_rc_mlx5_iface_t);
    uct_ib_mlx5_update_cq_ci(iface->super.super.recv_cq, iface->mlx5_common.rx.cq.cq_ci);
    return uct_ib_iface_arm_rx_cq(ib_iface, solicited);
}

static UCS_F_NOINLINE void uct_rc_mlx5_iface_handle_failure(uct_ib_iface_t *ib_iface,
                                                            void *arg)
{
    struct mlx5_cqe64 *cqe = arg;
    uct_rc_iface_t *iface = ucs_derived_of(ib_iface, uct_rc_iface_t);
    extern ucs_class_t UCS_CLASS_NAME(uct_rc_mlx5_ep_t);
    unsigned  qp_num = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(uct_rc_iface_lookup_ep(iface, qp_num),
                                          uct_rc_mlx5_ep_t);

    if (ep != NULL) {
        uct_ib_mlx5_completion_with_err((void*)cqe,
                                        iface->super.super.config.failure_level);
        uct_rc_txqp_purge_outstanding(&ep->super.txqp, UCS_ERR_ENDPOINT_TIMEOUT, 0);

        uct_set_ep_failed(&UCS_CLASS_NAME(uct_rc_mlx5_ep_t),
                          &ep->super.super.super,
                          &iface->super.super.super);
    }
}

static UCS_CLASS_INIT_FUNC(uct_rc_mlx5_iface_t, uct_md_h md, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    uct_rc_mlx5_iface_config_t *config = ucs_derived_of(tl_config, uct_rc_mlx5_iface_config_t);
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, &uct_rc_mlx5_iface_ops, md, worker,
                              dev_name, rx_headroom, 0, &config->super);

    self->tx.bb_max                  = ucs_min(config->tx_max_bb, UINT16_MAX);
    self->super.config.tx_moderation = ucs_min(self->super.config.tx_moderation,
                                               self->tx.bb_max / 4);

    status = uct_rc_mlx5_iface_common_init(&self->mlx5_common, &self->super, &config->super);
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_iface_t)
{
    uct_rc_mlx5_iface_common_cleanup(&self->mlx5_common);
}


UCS_CLASS_DEFINE(uct_rc_mlx5_iface_t, uct_rc_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_mlx5_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const char*, size_t,
                                 const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_mlx5_iface_t, uct_iface_t);

static uct_rc_iface_ops_t uct_rc_mlx5_iface_ops = {
    {
    {
    .iface_query              = uct_rc_mlx5_iface_query,
    .iface_flush              = uct_rc_iface_flush,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_mlx5_iface_t),
    .iface_release_am_desc    = uct_ib_iface_release_am_desc,
    .iface_wakeup_open        = uct_ib_iface_wakeup_open,
    .iface_wakeup_get_fd      = uct_ib_iface_wakeup_get_fd,
    .iface_wakeup_arm         = uct_ib_iface_wakeup_arm,
    .iface_wakeup_wait        = uct_ib_iface_wakeup_wait,
    .iface_wakeup_signal      = uct_ib_iface_wakeup_signal,
    .iface_wakeup_close       = uct_ib_iface_wakeup_close,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_rc_mlx5_ep_t),
    .ep_get_address           = uct_rc_ep_get_address,
    .ep_connect_to_ep         = uct_rc_ep_connect_to_ep,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_ib_iface_is_reachable,
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_mlx5_ep_t),
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
    .ep_flush                 = uct_rc_mlx5_ep_flush
    },
    .arm_tx_cq                = uct_rc_mlx5_iface_arm_tx_cq,
    .arm_rx_cq                = uct_rc_mlx5_iface_arm_rx_cq,
    .handle_failure           = uct_rc_mlx5_iface_handle_failure
    },
    .fc_ctrl                  = uct_rc_mlx5_ep_fc_ctrl
};


static ucs_status_t uct_rc_mlx5_query_resources(uct_md_h md,
                                                uct_tl_resource_desc_t **resources_p,
                                                unsigned *num_resources_p)
{
    uct_ib_md_t *ib_md = ucs_derived_of(md, uct_ib_md_t);

    return uct_ib_device_query_tl_resources(&ib_md->dev, "rc_mlx5",
                                            UCT_IB_DEVICE_FLAG_MLX5_PRM |
                                            (ib_md->eth_pause ? 0 : UCT_IB_DEVICE_FLAG_LINK_IB),
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
