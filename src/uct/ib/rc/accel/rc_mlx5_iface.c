/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/ib/mlx5/ib_mlx5_log.h>
#include <uct/ib/mlx5/dv/ib_mlx5_dv.h>
#include <uct/ib/mlx5/dv/ib_mlx5_ifc.h>
#include <uct/ib/base/ib_device.h>
#include <uct/base/uct_md.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>

#include "rc_mlx5.inl"


enum {
    UCT_RC_MLX5_IFACE_ADDR_TYPE_BASIC,

    /* Tag Matching address. It additionaly contains QP number which
     * is used for hardware offloads. */
    UCT_RC_MLX5_IFACE_ADDR_TYPE_TM
};


/**
 * RC mlx5 interface configuration
 */
typedef struct uct_rc_mlx5_iface_config {
    uct_rc_iface_config_t             super;
    uct_rc_mlx5_iface_common_config_t rc_mlx5_common;
    /* TODO wc_mode, UAR mode SnB W/A... */
} uct_rc_mlx5_iface_config_t;


ucs_config_field_t uct_rc_mlx5_iface_config_table[] = {
  {"RC_", "", NULL,
   ucs_offsetof(uct_rc_mlx5_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_rc_iface_config_table)},

  {"RC_", "", NULL,
   ucs_offsetof(uct_rc_mlx5_iface_config_t, rc_mlx5_common),
   UCS_CONFIG_TYPE_TABLE(uct_rc_mlx5_common_config_table)},

  {NULL}
};


static uct_rc_iface_ops_t uct_rc_mlx5_iface_ops;

#ifdef ENABLE_STATS
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
        ((ecqe->vendor_err_synd == UCT_IB_MLX5_CQE_VENDOR_SYND_ODP) ||
         (ecqe->vendor_err_synd == UCT_IB_MLX5_CQE_VENDOR_SYND_PSN)))
    {
        /* Release the aborted segment */
        wqe_ctr = ntohs(ecqe->wqe_counter);
        seg     = uct_ib_mlx5_srq_get_wqe(&iface->rx.srq, wqe_ctr);
        ++cq->cq_ci;
        /* TODO: Check if ib_stride_index valid for error CQE */
        uct_rc_mlx5_iface_release_srq_seg(iface, seg, cqe, wqe_ctr, UCS_OK,
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
    ep = ucs_derived_of(uct_rc_iface_lookup_ep(&iface->super, qp_num),
                        uct_rc_mlx5_ep_t);
    ucs_assert(ep != NULL);

    hw_ci = ntohs(cqe->wqe_counter);
    ucs_trace_poll("rc_mlx5 iface %p tx_cqe: ep %p qpn 0x%x hw_ci %d", iface, ep,
                   qp_num, hw_ci);

    uct_rc_mlx5_txqp_process_tx_cqe(&ep->super.txqp, cqe, hw_ci);

    uct_rc_mlx5_common_update_tx_res(&iface->super, &ep->tx.wq, &ep->super.txqp,
                                     hw_ci);

    ucs_arbiter_group_schedule(&iface->super.tx.arbiter, &ep->super.arb_group);
    ucs_arbiter_dispatch(&iface->super.tx.arbiter, 1, uct_rc_ep_process_pending,
                         NULL);

    return 1;
}

unsigned uct_rc_mlx5_iface_progress(void *arg)
{
    uct_rc_mlx5_iface_common_t *iface = arg;
    unsigned count;

    count = uct_rc_mlx5_iface_common_poll_rx(iface, UCT_RC_MLX5_POLL_FLAG_HAS_EP);
    if (!uct_rc_iface_poll_tx(&iface->super, count)) {
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
                                sizeof(uct_rc_mlx5_hdr_t),
                                UCT_RC_MLX5_RMA_MAX_IOV(0));
    if (status != UCS_OK) {
        return status;
    }

    uct_rc_mlx5_iface_common_query(&rc_iface->super, iface_attr, max_am_inline,
                                   UCT_RC_MLX5_TM_EAGER_ZCOPY_MAX_IOV(0));
    iface_attr->cap.flags     |= UCT_IFACE_FLAG_EP_CHECK;
    iface_attr->latency.m     += 1e-9; /* 1 ns per each extra QP */
    iface_attr->ep_addr_len    = sizeof(uct_rc_mlx5_ep_address_t);
    iface_attr->iface_addr_len = sizeof(uint8_t);
    return UCS_OK;
}

static void
uct_rc_mlx5_iface_handle_failure(uct_ib_iface_t *ib_iface, void *arg,
                                 ucs_status_t ep_status)
{
    struct mlx5_cqe64  *cqe    = arg;
    uct_rc_iface_t     *iface  = ucs_derived_of(ib_iface, uct_rc_iface_t);
    unsigned           qp_num  = ntohl(cqe->sop_drop_qpn) &
                                 UCS_MASK(UCT_IB_QPN_ORDER);
    uct_rc_mlx5_ep_t   *ep     = ucs_derived_of(uct_rc_iface_lookup_ep(iface,
                                                                       qp_num),
                                                uct_rc_mlx5_ep_t);
    uint16_t           pi      = ntohs(cqe->wqe_counter);
    ucs_log_level_t    log_lvl;
    ucs_status_t       status;

    ucs_assert(ep != NULL);
    uct_rc_mlx5_common_update_tx_res(iface, &ep->tx.wq, &ep->super.txqp, pi);
    uct_rc_txqp_purge_outstanding(iface, &ep->super.txqp, ep_status, pi, 0);

    if (ep->super.flags & (UCT_RC_EP_FLAG_NO_ERR_HANDLER |
                           UCT_RC_EP_FLAG_FLUSH_CANCEL)) {
        return;
    }

    ep->super.flags |= UCT_RC_EP_FLAG_NO_ERR_HANDLER;

    status  = uct_iface_handle_ep_err(&iface->super.super.super,
                                      &ep->super.super.super, ep_status);
    log_lvl = uct_ib_iface_failure_log_level(ib_iface, status, ep_status);

    uct_ib_mlx5_completion_with_err(ib_iface, arg, &ep->tx.wq, log_lvl);
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
                                         uct_ib_mlx5_qp_attr_t *attr)
{
    uct_ib_iface_t *ib_iface           = &iface->super.super;
    ucs_status_t status;
    uct_ib_mlx5_md_t *md               = ucs_derived_of(ib_iface->super.md,
                                                        uct_ib_mlx5_md_t);
#if HAVE_DECL_MLX5DV_CREATE_QP
    uct_ib_device_t *dev               = &md->super.dev;
    struct mlx5dv_qp_init_attr dv_attr = {};

    if (md->flags & UCT_IB_MLX5_MD_FLAG_DEVX_RC_QP) {
        attr->mmio_mode = iface->tx.mmio_mode;
        status = uct_ib_mlx5_devx_create_qp(ib_iface, qp, txwq, attr);
        if (status != UCS_OK) {
            return status;
        }

        status = uct_rc_mlx5_devx_iface_subscribe_event(iface, qp,
                UCT_IB_MLX5_EVENT_TYPE_SRQ_LAST_WQE,
                IBV_EVENT_QP_LAST_WQE_REACHED, qp->qp_num);
        if (status != UCS_OK) {
            goto err_destory_qp;
        }

        return UCS_OK;
    }

    status = uct_ib_mlx5_iface_fill_attr(ib_iface, qp, attr);
    if (status != UCS_OK) {
        return status;
    }

    uct_ib_iface_fill_attr(ib_iface, &attr->super);
    uct_rc_mlx5_common_fill_dv_qp_attr(iface, &attr->super.ibv, &dv_attr,
                                       UCS_BIT(UCT_IB_DIR_TX) |
                                       UCS_BIT(UCT_IB_DIR_RX));
    qp->verbs.qp = mlx5dv_create_qp(dev->ibv_context, &attr->super.ibv, &dv_attr);
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

    if (attr->super.cap.max_send_wr) {
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
    uct_ib_mlx5_destroy_qp(md, qp);
err:
    return status;
}

static UCS_F_MAYBE_UNUSED unsigned uct_rc_mlx5_iface_progress_tm(void *arg)
{
    uct_rc_mlx5_iface_common_t *iface = arg;
    unsigned count;

    count = uct_rc_mlx5_iface_common_poll_rx(iface,
                                             UCT_RC_MLX5_POLL_FLAG_HAS_EP |
                                             UCT_RC_MLX5_POLL_FLAG_TM);
    if (!uct_rc_iface_poll_tx(&iface->super, count)) {
        return count;
    }
    return uct_rc_mlx5_iface_poll_tx(iface);
}

#if IBV_HW_TM
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

static ucs_status_t uct_rc_mlx5_iface_preinit(uct_rc_mlx5_iface_common_t *iface,
                                              uct_md_h tl_md,
                                              uct_rc_iface_common_config_t *rc_config,
                                              uct_rc_mlx5_iface_common_config_t *mlx5_config,
                                              const uct_iface_params_t *params,
                                              uct_ib_iface_init_attr_t *init_attr)
{
    uct_ib_mlx5_md_t *md = ucs_derived_of(tl_md, uct_ib_mlx5_md_t);
#if IBV_HW_TM
    uct_ib_device_t *dev = &md->super.dev;
    struct ibv_tmh tmh;
    int mtu;
    int tm_params;
    ucs_status_t status;
#endif

    iface->config.cyclic_srq_enable = mlx5_config->cyclic_srq_enable;

#if IBV_HW_TM
    /* Both eager and rndv callbacks should be provided for
     * tag matching support */
    tm_params = ucs_test_all_flags(params->field_mask,
                                   UCT_IFACE_PARAM_FIELD_HW_TM_EAGER_CB |
                                   UCT_IFACE_PARAM_FIELD_HW_TM_RNDV_CB);

    iface->tm.enabled = mlx5_config->tm.enable && tm_params &&
                        (init_attr->flags & UCT_IB_TM_SUPPORTED);
    if (!iface->tm.enabled) {
        goto out_tm_disabled;
    }

    /* Compile-time check that THM and uct_rc_mlx5_hdr_t are wire-compatible
     * for the case of no-tag protocol.
     */
    UCS_STATIC_ASSERT(sizeof(tmh.opcode) ==
                      sizeof(((uct_rc_mlx5_hdr_t*)0)->tmh_opcode));
    UCS_STATIC_ASSERT(ucs_offsetof(struct ibv_tmh, opcode) ==
                      ucs_offsetof(uct_rc_mlx5_hdr_t, tmh_opcode));

    UCS_STATIC_ASSERT(sizeof(uct_rc_mlx5_ctx_priv_t) <= UCT_TAG_PRIV_LEN);

    iface->tm.eager_unexp.cb  = params->eager_cb;
    iface->tm.rndv_unexp.cb   = params->rndv_cb;
    iface->tm.eager_unexp.arg = (params->field_mask &
                                 UCT_IFACE_PARAM_FIELD_HW_TM_EAGER_ARG) ?
                                params->eager_arg : NULL;
    iface->tm.rndv_unexp.arg  = (params->field_mask &
                                 UCT_IFACE_PARAM_FIELD_HW_TM_RNDV_ARG) ?
                                params->rndv_arg : NULL;
    iface->tm.unexpected_cnt  = 0;
    iface->tm.num_outstanding = 0;
    iface->tm.num_tags        = ucs_min(IBV_DEVICE_TM_CAPS(dev, max_num_tags),
                                        mlx5_config->tm.list_size);

    /* There can be:
     * - up to rx.queue_len RX CQEs
     * - up to 3 CQEs for every posted tag: ADD, TM_CONSUMED and MSG_ARRIVED
     * - one SYNC CQE per every IBV_DEVICE_MAX_UNEXP_COUNT unexpected receives */
    UCS_STATIC_ASSERT(IBV_DEVICE_MAX_UNEXP_COUNT);
    init_attr->cq_len[UCT_IB_DIR_RX] = rc_config->super.rx.queue_len +
                                       iface->tm.num_tags * 3 +
                                       rc_config->super.rx.queue_len /
                                       IBV_DEVICE_MAX_UNEXP_COUNT;
    init_attr->seg_size      = ucs_max(mlx5_config->tm.seg_size,
                                       rc_config->super.seg_size);
    iface->tm.mp.num_strides = 1;
    iface->tm.max_bcopy      = init_attr->seg_size;

    if (mlx5_config->tm.mp_enable == UCS_NO) {
        return UCS_OK;
    }

    /* Multi-Packet XRQ initialization */
    if (!ucs_test_all_flags(md->flags, UCT_IB_MLX5_MD_FLAG_MP_RQ       |
                                       UCT_IB_MLX5_MD_FLAG_DEVX_RC_SRQ |
                                       UCT_IB_MLX5_MD_FLAG_DEVX_RC_QP)) {
        goto out_mp_disabled;
    }

    if ((mlx5_config->tm.mp_num_strides != 8) &&
        (mlx5_config->tm.mp_num_strides != 16)) {
        ucs_error("invalid value of TM_MP_NUM_STRIDES: %lu, must be 8 or 16",
                  mlx5_config->tm.mp_num_strides);
        return UCS_ERR_INVALID_PARAM;
    } else {
        iface->tm.mp.num_strides = mlx5_config->tm.mp_num_strides;
    }

    status = uct_ib_device_mtu(params->mode.device.dev_name, tl_md, &mtu);
    if (status != UCS_OK) {
        ucs_error("failed to get port MTU: %s", ucs_status_string(status));
        return UCS_ERR_IO_ERROR;
    }

    init_attr->seg_size = mtu;

    return UCS_OK;

out_tm_disabled:
#else
    iface->tm.enabled                = 0;
#endif
    init_attr->cq_len[UCT_IB_DIR_RX] = rc_config->super.rx.queue_len;
    init_attr->seg_size              = rc_config->super.seg_size;
    iface->tm.mp.num_strides         = 1;

#if IBV_HW_TM
out_mp_disabled:
#endif
    if (mlx5_config->tm.mp_enable == UCS_YES) {
        ucs_error("%s: MP SRQ is requested, but not supported: (md flags 0x%x), "
                  "hardware tag-matching is %s",
                  uct_ib_device_name(&md->super.dev), md->flags,
                  iface->tm.enabled ? "enabled" : "disabled");
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static ucs_status_t
uct_rc_mlx5_iface_init_rx(uct_rc_iface_t *rc_iface,
                          const uct_rc_iface_common_config_t *rc_config)
{
    uct_rc_mlx5_iface_common_t *iface    = ucs_derived_of(rc_iface,
                                                          uct_rc_mlx5_iface_common_t);
    uct_ib_mlx5_md_t *md                 = ucs_derived_of(rc_iface->super.super.md,
                                                          uct_ib_mlx5_md_t);
    struct ibv_srq_init_attr_ex srq_attr = {};
    ucs_status_t status;

    if (UCT_RC_MLX5_TM_ENABLED(iface)) {
        if (md->flags & UCT_IB_MLX5_MD_FLAG_DEVX_RC_SRQ) {
            status = uct_rc_mlx5_devx_init_rx_tm(iface, rc_config, 0,
                                                 UCT_RC_RNDV_HDR_LEN);
        } else {
            status = uct_rc_mlx5_init_rx_tm(iface, rc_config, &srq_attr,
                                            UCT_RC_RNDV_HDR_LEN);
        }

        if (status != UCS_OK) {
            return status;
        }

        iface->super.progress = uct_rc_mlx5_iface_progress_tm;
        return UCS_OK;
    }

    /* MP XRQ is supported with HW TM only */
    ucs_assert(!UCT_RC_MLX5_MP_ENABLED(iface));

    if (ucs_test_all_flags(md->flags, UCT_IB_MLX5_MD_FLAG_RMP |
                                      UCT_IB_MLX5_MD_FLAG_DEVX_RC_SRQ)) {
        status = uct_rc_mlx5_devx_init_rx(iface, rc_config);
    } else {
        status = uct_rc_mlx5_common_iface_init_rx(iface, rc_config);
    }

    if (status != UCS_OK) {
        return status;
    }

    iface->super.progress = uct_rc_mlx5_iface_progress;
    return UCS_OK;
}

static void uct_rc_mlx5_iface_cleanup_rx(uct_rc_iface_t *rc_iface)
{
    uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(rc_iface, uct_rc_mlx5_iface_common_t);
    uct_ib_mlx5_md_t *md              = ucs_derived_of(rc_iface->super.super.md,
                                                       uct_ib_mlx5_md_t);

    uct_rc_mlx5_destroy_srq(md, &iface->rx.srq);
}

static uint8_t uct_rc_mlx5_iface_get_address_type(uct_iface_h tl_iface)
{
    uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(tl_iface,
                                                       uct_rc_mlx5_iface_common_t);

    return UCT_RC_MLX5_TM_ENABLED(iface) ?  UCT_RC_MLX5_IFACE_ADDR_TYPE_TM :
                                            UCT_RC_MLX5_IFACE_ADDR_TYPE_BASIC;
}

static ucs_status_t uct_rc_mlx5_iface_get_address(uct_iface_h tl_iface,
                                                  uct_iface_addr_t *addr)
{
    *(uint8_t*)addr = uct_rc_mlx5_iface_get_address_type(tl_iface);

    return UCS_OK;
}

int uct_rc_mlx5_iface_is_reachable(const uct_iface_h tl_iface,
                                   const uct_device_addr_t *dev_addr,
                                   const uct_iface_addr_t *iface_addr)
{
    uint8_t my_type = uct_rc_mlx5_iface_get_address_type(tl_iface);

    if ((iface_addr != NULL) && (my_type != *(uint8_t*)iface_addr)) {
        return 0;
    }

    return uct_ib_iface_is_reachable(tl_iface, dev_addr, iface_addr);
}

UCS_CLASS_INIT_FUNC(uct_rc_mlx5_iface_common_t,
                    uct_rc_iface_ops_t *ops,
                    uct_md_h tl_md, uct_worker_h worker,
                    const uct_iface_params_t *params,
                    uct_rc_iface_common_config_t *rc_config,
                    uct_rc_mlx5_iface_common_config_t *mlx5_config,
                    uct_ib_iface_init_attr_t *init_attr)
{
    uct_ib_mlx5_md_t *md = ucs_derived_of(tl_md, uct_ib_mlx5_md_t);
    uct_ib_device_t *dev;
    ucs_status_t status;

    status = uct_rc_mlx5_iface_preinit(self, tl_md, rc_config, mlx5_config,
                                       params, init_attr);
    if (status != UCS_OK) {
        return status;
    }

    self->rx.srq.type                = UCT_IB_MLX5_OBJ_TYPE_LAST;
    self->tm.cmd_wq.super.super.type = UCT_IB_MLX5_OBJ_TYPE_LAST;
    init_attr->rx_hdr_len            = UCT_RC_MLX5_MP_ENABLED(self) ?
                                       0 : sizeof(uct_rc_mlx5_hdr_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, ops, tl_md, worker, params,
                              rc_config, init_attr);

    dev                       = uct_ib_iface_device(&self->super.super);
    self->tx.mmio_mode        = mlx5_config->super.mmio_mode;
    self->tx.bb_max           = ucs_min(mlx5_config->tx_max_bb, UINT16_MAX);
    self->tm.am_desc.super.cb = uct_rc_mlx5_release_desc;

    if (!UCT_RC_MLX5_MP_ENABLED(self)) {
        self->tm.am_desc.offset = self->super.super.config.rx_headroom_offset;
    }

    status = uct_ib_mlx5_iface_select_sl(&self->super.super,
                                         &mlx5_config->super,
                                         &rc_config->super);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_get_cq(self->super.super.cq[UCT_IB_DIR_TX],
                                &self->cq[UCT_IB_DIR_TX]);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_get_cq(self->super.super.cq[UCT_IB_DIR_RX],
                                &self->cq[UCT_IB_DIR_RX]);
    if (status != UCS_OK) {
        return status;
    }

    status = UCS_STATS_NODE_ALLOC(&self->stats, &uct_rc_mlx5_iface_stats_class,
                                  self->super.stats);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_rc_mlx5_iface_common_tag_init(self);
    if (status != UCS_OK) {
        goto cleanup_stats;
    }

    status = uct_rc_mlx5_iface_common_dm_init(self, &self->super,
                                              &mlx5_config->super);
    if (status != UCS_OK) {
        goto cleanup_tm;
    }

    self->super.config.fence_mode  = (uct_rc_fence_mode_t)rc_config->fence_mode;
    self->super.rx.srq.quota       = self->rx.srq.mask + 1;
    self->super.config.exp_backoff = mlx5_config->exp_backoff;

    if ((rc_config->fence_mode == UCT_RC_FENCE_MODE_WEAK) ||
        ((rc_config->fence_mode == UCT_RC_FENCE_MODE_AUTO) &&
         (uct_ib_device_has_pci_atomics(dev) || md->super.relaxed_order))) {
        if (uct_ib_device_has_pci_atomics(dev)) {
            self->config.atomic_fence_flag = UCT_IB_MLX5_WQE_CTRL_FLAG_FENCE;
        } else {
            self->config.atomic_fence_flag = 0;
        }
        self->super.config.fence_mode      = UCT_RC_FENCE_MODE_WEAK;
    } else if ((rc_config->fence_mode == UCT_RC_FENCE_MODE_NONE) ||
               ((rc_config->fence_mode == UCT_RC_FENCE_MODE_AUTO) &&
                !uct_ib_device_has_pci_atomics(dev))) {
        self->config.atomic_fence_flag     = 0;
        self->super.config.fence_mode      = UCT_RC_FENCE_MODE_NONE;
    } else {
        ucs_error("incorrect fence value: %d", self->super.config.fence_mode);
        status = UCS_ERR_INVALID_PARAM;
        goto cleanup_tm;
    }

    /* By default set to something that is always in cache */
    self->rx.pref_ptr = self;

    status = uct_iface_mpool_init(&self->super.super.super,
                                  &self->tx.atomic_desc_mp,
                                  sizeof(uct_rc_iface_send_desc_t) + UCT_IB_MAX_ATOMIC_SIZE,
                                  sizeof(uct_rc_iface_send_desc_t) + UCT_IB_MAX_ATOMIC_SIZE,
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &rc_config->super.tx.mp,
                                  self->super.config.tx_qp_len,
                                  uct_rc_iface_send_desc_init,
                                  "rc_mlx5_atomic_desc");
    if (status != UCS_OK) {
        goto cleanup_dm;
    }

#if HAVE_DEVX
    status = uct_rc_mlx5_devx_iface_init_events(self);
    if (status != UCS_OK) {
        goto cleanup_dm;
    }
#endif

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
    uct_rc_iface_cleanup_eps(&self->super);
#if HAVE_DEVX
    uct_rc_mlx5_devx_iface_free_events(self);
#endif
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
                    uct_md_h tl_md, uct_worker_h worker,
                    const uct_iface_params_t *params,
                    const uct_iface_config_t *tl_config)
{
    uct_rc_mlx5_iface_config_t *config = ucs_derived_of(tl_config,
                                                        uct_rc_mlx5_iface_config_t);
    uct_ib_mlx5_md_t UCS_V_UNUSED *md  = ucs_derived_of(tl_md, uct_ib_mlx5_md_t);
    uct_ib_iface_init_attr_t init_attr = {};
    ucs_status_t status;

    init_attr.fc_req_size           = sizeof(uct_rc_pending_req_t);
    init_attr.flags                 = UCT_IB_CQ_IGNORE_OVERRUN;
    init_attr.rx_hdr_len            = sizeof(uct_rc_mlx5_hdr_t);
    init_attr.cq_len[UCT_IB_DIR_TX] = config->super.tx_cq_len;
    init_attr.qp_type               = IBV_QPT_RC;

    if (IBV_DEVICE_TM_FLAGS(&md->super.dev)) {
        init_attr.flags  |= UCT_IB_TM_SUPPORTED;
    }

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_mlx5_iface_common_t, &uct_rc_mlx5_iface_ops,
                              tl_md, worker, params, &config->super.super,
                              &config->rc_mlx5_common, &init_attr);

    self->super.super.config.tx_moderation = ucs_min(config->super.tx_cq_moderation,
                                                     self->super.tx.bb_max / 4);

    status = uct_rc_init_fc_thresh(&config->super, &self->super.super);
    if (status != UCS_OK) {
        return status;
    }

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
    .ep_check                 = uct_rc_ep_check,
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
    .iface_get_address        = uct_rc_mlx5_iface_get_address,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_rc_mlx5_iface_is_reachable
    },
    .create_cq                = uct_ib_mlx5_create_cq,
    .arm_cq                   = uct_rc_mlx5_iface_common_arm_cq,
    .event_cq                 = uct_rc_mlx5_iface_common_event_cq,
    .handle_failure           = uct_rc_mlx5_iface_handle_failure,
    },
    .init_rx                  = uct_rc_mlx5_iface_init_rx,
    .cleanup_rx               = uct_rc_mlx5_iface_cleanup_rx,
    .fc_ctrl                  = uct_rc_mlx5_ep_fc_ctrl,
    .fc_handler               = uct_rc_iface_fc_handler,
    .cleanup_qp               = uct_rc_mlx5_ep_cleanup_qp,
    .ep_post_check            = uct_rc_mlx5_ep_post_check
};

static ucs_status_t
uct_rc_mlx5_query_tl_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                             unsigned *num_tl_devices_p)
{
    uct_ib_md_t *ib_md = ucs_derived_of(md, uct_ib_md_t);
    int flags;

    flags = UCT_IB_DEVICE_FLAG_MLX5_PRM |
            (ib_md->config.eth_pause ? 0 : UCT_IB_DEVICE_FLAG_LINK_IB);
    return uct_ib_device_query_ports(&ib_md->dev, flags, tl_devices_p,
                                     num_tl_devices_p);
}

UCT_TL_DEFINE(&uct_ib_component, rc_mlx5, uct_rc_mlx5_query_tl_devices,
              uct_rc_mlx5_iface_t, "RC_MLX5_", uct_rc_mlx5_iface_config_table,
              uct_rc_mlx5_iface_config_t);
