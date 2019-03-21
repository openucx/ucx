/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "dc_mlx5.h"
#include "dc_mlx5_ep.h"

#include <uct/api/uct.h>
#include <uct/ib/rc/accel/rc_mlx5.inl>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_log.h>
#include <uct/ib/mlx5/ib_mlx5_log.h>
#include <uct/base/uct_md.h>
#include <ucs/arch/bitops.h>
#include <ucs/arch/cpu.h>
#include <ucs/async/async.h>
#include <ucs/debug/log.h>
#include <string.h>


static const char *uct_dc_tx_policy_names[] = {
    [UCT_DC_TX_POLICY_DCS]           = "dcs",
    [UCT_DC_TX_POLICY_DCS_QUOTA]     = "dcs_quota",
    [UCT_DC_TX_POLICY_RAND]          = "rand",
    [UCT_DC_TX_POLICY_LAST]          = NULL
};

/* DC specific parameters, expecting DC_ prefix */
ucs_config_field_t uct_dc_mlx5_iface_config_sub_table[] = {
    {"", "IB_TX_QUEUE_LEN=128;RC_FC_ENABLE=y;", NULL,
     ucs_offsetof(uct_dc_mlx5_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_rc_mlx5_common_config_table)},

    {"", "", NULL,
     ucs_offsetof(uct_dc_mlx5_iface_config_t, ud_common),
     UCS_CONFIG_TYPE_TABLE(uct_ud_iface_common_config_table)},

    {"NUM_DCI", "8",
     "Number of DC initiator QPs (DCI) used by the interface "
     "(up to " UCS_PP_QUOTE(UCT_DC_MLX5_IFACE_MAX_DCIS) ").",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, ndci), UCS_CONFIG_TYPE_UINT},

    {"TX_POLICY", "dcs_quota",
     "Specifies how DC initiator (DCI) is selected by the endpoint. The policies are:\n"
     "\n"
     "dcs        The endpoint either uses already assigned DCI or one is allocated\n"
     "           in a LIFO order, and released once it has no outstanding operations.\n"
     "\n"
     "dcs_quota  Same as \"dcs\" but in addition the DCI is scheduled for release\n"
     "           if it has sent more than quota, and there are endpoints waiting for a DCI.\n"
     "           The dci is released once it completes all outstanding operations.\n"
     "           This policy ensures that there will be no starvation among endpoints.\n"
     "\n"
     "rand       Every endpoint is assigned with a randomly selected DCI.\n"
     "           Multiple endpoints may share the same DCI.",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, tx_policy),
     UCS_CONFIG_TYPE_ENUM(uct_dc_tx_policy_names)},

    {"RAND_DCI_SEED", "0",
     "Seed for DCI allocation when \"rand\" dci policy is used (0 - use default).",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, rand_seed), UCS_CONFIG_TYPE_UINT},

    {"QUOTA", "32",
     "When \"dcs_quota\" policy is selected, how much to send from a DCI when\n"
     "there are other endpoints waiting for it.",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, quota), UCS_CONFIG_TYPE_UINT},

    {NULL}
};

/* Bundle of all parameters */
ucs_config_field_t uct_dc_mlx5_iface_config_table[] = {
    {"DC_", "", NULL, 0,
     UCS_CONFIG_TYPE_TABLE(uct_dc_mlx5_iface_config_sub_table)},

    {"", "", NULL,
     ucs_offsetof(uct_dc_mlx5_iface_config_t, mlx5_ud),
     UCS_CONFIG_TYPE_TABLE(uct_ud_mlx5_iface_common_config_table)},

    {"", "", NULL,
     ucs_offsetof(uct_dc_mlx5_iface_config_t, super.mlx5_common),
     UCS_CONFIG_TYPE_TABLE(uct_ib_mlx5_iface_config_table)},

    {NULL}
};


static ucs_status_t
uct_dc_mlx5_ep_create_connected(const uct_ep_params_t *params, uct_ep_h* ep_p)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(params->iface,
                                                uct_dc_mlx5_iface_t);
    const uct_ib_address_t *ib_addr;
    const uct_dc_mlx5_iface_addr_t *if_addr;
    ucs_status_t status;
    int is_global;
    uct_ib_mlx5_base_av_t av;
    struct mlx5_grh_av grh_av;

    ucs_trace_func("");

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);
    ib_addr = (const uct_ib_address_t *)params->dev_addr;
    if_addr = (const uct_dc_mlx5_iface_addr_t *)params->iface_addr;

    status = uct_ud_mlx5_iface_get_av(&iface->super.super.super, &iface->ud_common,
                                      ib_addr, iface->super.super.super.path_bits[0],
                                      &av, &grh_av, &is_global);
    if (status != UCS_OK) {
        return UCS_ERR_INVALID_ADDR;
    }

    if (is_global) {
        return UCS_CLASS_NEW(uct_dc_mlx5_grh_ep_t, ep_p, iface, if_addr, &av, &grh_av);
    } else {
        return UCS_CLASS_NEW(uct_dc_mlx5_ep_t, ep_p, iface, if_addr, &av);
    }
}

static void uct_dc_mlx5_ep_destroy(uct_ep_h tl_ep)
{
    uct_dc_mlx5_ep_cleanup(tl_ep, &UCS_CLASS_NAME(uct_dc_mlx5_ep_t));
}

static ucs_status_t uct_dc_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    size_t max_am_inline       = UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE);
    size_t max_put_inline      = UCT_IB_MLX5_PUT_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE);
    ucs_status_t status;

#if HAVE_IBV_EXP_DM
    if (iface->super.dm.dm != NULL) {
        max_am_inline  = ucs_max(iface->super.dm.dm->seg_len,
                                 UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE));
        max_put_inline = ucs_max(iface->super.dm.dm->seg_len,
                                 UCT_IB_MLX5_PUT_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE));
    }
#endif

    status = uct_rc_iface_query(&iface->super.super, iface_attr,
                                max_put_inline,
                                max_am_inline,
                                UCT_IB_MLX5_AM_ZCOPY_MAX_HDR(UCT_IB_MLX5_AV_FULL_SIZE),
                                UCT_IB_MLX5_AM_ZCOPY_MAX_IOV,
                                UCT_RC_MLX5_TM_EAGER_ZCOPY_MAX_IOV(UCT_IB_MLX5_AV_FULL_SIZE),
                                sizeof(uct_rc_mlx5_hdr_t));
    if (status != UCS_OK) {
        return status;
    }

    /* fixup flags and address lengths */
    iface_attr->cap.flags &= ~UCT_IFACE_FLAG_CONNECT_TO_EP;
    iface_attr->cap.flags |= UCT_IFACE_FLAG_CONNECT_TO_IFACE;
    iface_attr->ep_addr_len       = 0;
    iface_attr->max_conn_priv     = 0;
    iface_attr->iface_addr_len    = sizeof(uct_dc_mlx5_iface_addr_t);
    iface_attr->latency.overhead += 60e-9; /* connect packet + cqe */

    uct_rc_mlx5_iface_common_query(&iface->super.super.super, iface_attr,
                                   max_am_inline, UCT_IB_MLX5_AV_FULL_SIZE);

    /* Error handling is not supported with random dci policy
     * TODO: Fix */
    if (uct_dc_mlx5_iface_is_dci_rand(iface)) {
        iface_attr->cap.flags &= ~(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE |
                                   UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF    |
                                   UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM);
    }

    return UCS_OK;
}

static void uct_dc_mlx5_iface_progress_enable(uct_iface_h tl_iface, unsigned flags)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);

    uct_base_iface_progress_enable_cb(&iface->super.super, iface->progress, flags);
}

static ucs_status_t uct_dc_mlx5_ep_set_failed(uct_ib_iface_t *ib_iface,
                                              uct_ep_h ep, ucs_status_t status)
{
    return uct_set_ep_failed(&UCS_CLASS_NAME(uct_dc_mlx5_ep_t), ep,
                             &ib_iface->super.super, status);
}

static UCS_F_ALWAYS_INLINE unsigned
uct_dc_mlx5_poll_tx(uct_dc_mlx5_iface_t *iface)
{
    uint8_t dci;
    struct mlx5_cqe64 *cqe;
    uint32_t qp_num;
    uint16_t hw_ci;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    cqe = uct_ib_mlx5_poll_cq(&iface->super.super.super,
                              &iface->super.cq[UCT_IB_DIR_TX]);
    if (cqe == NULL) {
        return 0;
    }
    UCS_STATS_UPDATE_COUNTER(iface->super.super.stats, UCT_RC_IFACE_STAT_TX_COMPLETION, 1);

    ucs_memory_cpu_load_fence();

    qp_num = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
    dci = uct_dc_mlx5_iface_dci_find(iface, qp_num);
    txqp = &iface->tx.dcis[dci].txqp;
    txwq = &iface->tx.dci_wqs[dci];
    hw_ci = ntohs(cqe->wqe_counter);

    ucs_trace_poll("dc iface %p tx_cqe: dci[%d] qpn 0x%x txqp %p hw_ci %d",
                   iface, dci, qp_num, txqp, hw_ci);

    uct_rc_mlx5_common_update_tx_res(&iface->super.super, txwq, txqp, hw_ci);
    uct_dc_mlx5_iface_dci_put(iface, dci);
    uct_rc_mlx5_txqp_process_tx_cqe(txqp, cqe, hw_ci);

    uct_dc_mlx5_iface_progress_pending(iface);
    return 1;
}

static unsigned uct_dc_mlx5_iface_progress(void *arg)
{
    uct_dc_mlx5_iface_t *iface = arg;
    unsigned count;

    count = uct_rc_mlx5_iface_common_poll_rx(&iface->super, 0);
    if (count > 0) {
        return count;
    }
    return uct_dc_mlx5_poll_tx(iface);
}

#if IBV_EXP_HW_TM_DC
static unsigned uct_dc_mlx5_iface_progress_tm(void *arg)
{
    uct_dc_mlx5_iface_t *iface = arg;
    unsigned count;

    count = uct_rc_mlx5_iface_common_poll_rx(&iface->super, 1);
    if (count > 0) {
        return count;
    }
    return uct_dc_mlx5_poll_tx(iface);
}
#endif

static void UCS_CLASS_DELETE_FUNC_NAME(uct_dc_mlx5_iface_t)(uct_iface_t*);

ucs_status_t uct_dc_mlx5_iface_reset_dci(uct_dc_mlx5_iface_t *iface, int dci)
{
    ucs_status_t status;

    ucs_debug("iface %p reset dci[%d]", iface, dci);

    /* Synchronize CQ index with the driver, since it would remove pending
     * completions for this QP (both send and receive) during ibv_destroy_qp().
     */
    uct_rc_mlx5_iface_common_update_cqs_ci(&iface->super,
                                           &iface->super.super.super);
    status = uct_ib_modify_qp(iface->tx.dcis[dci].txqp.qp, IBV_QPS_RESET);
    uct_rc_mlx5_iface_common_sync_cqs_ci(&iface->super,
                                         &iface->super.super.super);

    uct_rc_mlx5_iface_commom_clean(&iface->super.cq[UCT_IB_DIR_TX], NULL,
                                   iface->tx.dcis[dci].txqp.qp->qp_num);

    /* Resume posting from to the beginning of the QP */
    uct_ib_mlx5_txwq_reset(&iface->tx.dci_wqs[dci]);

    return status;
}

static void uct_dc_mlx5_iface_event_cq(uct_ib_iface_t *ib_iface,
                                       uct_ib_dir_t dir)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ib_iface, uct_dc_mlx5_iface_t);

    iface->super.cq[dir].cq_sn++;
}

static ucs_status_t uct_dc_mlx5_iface_create_qp(uct_ib_iface_t *ib_iface,
                                                uct_ib_qp_attr_t *attr,
                                                struct ibv_qp **qp_p)
{
    uct_dc_mlx5_iface_t *iface         = ucs_derived_of(ib_iface, uct_dc_mlx5_iface_t);
#ifdef HAVE_DC_DV
    uct_ib_device_t *dev               = uct_ib_iface_device(ib_iface);
    struct mlx5dv_qp_init_attr dv_attr = {};
    struct ibv_qp *qp;

    uct_ib_iface_fill_attr(ib_iface, attr);
    uct_ib_mlx5_iface_fill_attr(ib_iface, &iface->super.mlx5_common, attr);
    attr->ibv.cap.max_recv_sge          = 0;

    dv_attr.comp_mask                   = MLX5DV_QP_INIT_ATTR_MASK_DC;
    dv_attr.dc_init_attr.dc_type        = MLX5DV_DCTYPE_DCI;
    dv_attr.dc_init_attr.dct_access_key = UCT_IB_KEY;
    qp = mlx5dv_create_qp(dev->ibv_context, &attr->ibv, &dv_attr);
    if (qp == NULL) {
        ucs_error("iface=%p: failed to create DCI: %m", iface);
        return UCS_ERR_IO_ERROR;
    }

    attr->cap = attr->ibv.cap;
    *qp_p     = qp;

    return UCS_OK;
#else
    return uct_ib_mlx5_iface_create_qp(ib_iface, &iface->super.mlx5_common, attr, qp_p);
#endif
}

#ifdef HAVE_DC_DV
ucs_status_t uct_dc_mlx5_iface_dci_connect(uct_dc_mlx5_iface_t *iface,
                                           uct_rc_txqp_t *dci)
{
    struct ibv_qp_attr attr;
    long attr_mask;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state        = IBV_QPS_INIT;
    attr.pkey_index      = iface->super.super.super.pkey_index;
    attr.port_num        = iface->super.super.super.config.port_num;
    attr_mask            = IBV_QP_STATE      |
                           IBV_QP_PKEY_INDEX |
                           IBV_QP_PORT;

    if (ibv_modify_qp(dci->qp, &attr, attr_mask)) {
        ucs_error("error modifying QP to INIT : %m");
        return UCS_ERR_IO_ERROR;
    }

    /* Move QP to the RTR state */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state                   = IBV_QPS_RTR;
    attr.path_mtu                   = iface->super.super.config.path_mtu;
    attr.min_rnr_timer              = iface->super.super.config.min_rnr_timer;
    attr.max_dest_rd_atomic         = 1;
    attr.ah_attr.is_global          = iface->super.super.super.is_global_addr;
    attr.ah_attr.sl                 = iface->super.super.super.config.sl;
    attr_mask                       = IBV_QP_STATE     |
                                      IBV_QP_PATH_MTU;

    if (ibv_modify_qp(dci->qp, &attr, attr_mask)) {
        ucs_error("error modifying DCI QP to RTR: %m");
        return UCS_ERR_IO_ERROR;
    }

    /* Move QP to the RTS state */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state       = IBV_QPS_RTS;
    attr.timeout        = iface->super.super.config.timeout;
    attr.rnr_retry      = iface->super.super.config.rnr_retry;
    attr.retry_cnt      = iface->super.super.config.retry_cnt;
    attr.max_rd_atomic  = iface->super.super.config.max_rd_atomic;
    attr_mask           = IBV_QP_STATE      |
                          IBV_QP_SQ_PSN     |
                          IBV_QP_TIMEOUT    |
                          IBV_QP_RETRY_CNT  |
                          IBV_QP_RNR_RETRY  |
                          IBV_QP_MAX_QP_RD_ATOMIC;

    if (ibv_modify_qp(dci->qp, &attr, attr_mask)) {
        ucs_error("error modifying DCI QP to RTS: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t uct_dc_mlx5_iface_create_dct(uct_dc_mlx5_iface_t *iface)
{
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super.super.super);
    struct mlx5dv_qp_init_attr dv_init_attr = {};
    struct ibv_qp_init_attr_ex init_attr = {};
    struct ibv_qp_attr attr = {};
    int ret;

    init_attr.comp_mask             = IBV_QP_INIT_ATTR_PD;
    init_attr.pd                    = uct_ib_iface_md(&iface->super.super.super)->pd;
    init_attr.recv_cq               = iface->super.super.super.cq[UCT_IB_DIR_RX];
    /* DCT can't send, but send_cq have to point to valid CQ */
    init_attr.send_cq               = iface->super.super.super.cq[UCT_IB_DIR_RX];
    init_attr.srq                   = iface->super.super.rx.srq.srq;
    init_attr.qp_type               = IBV_QPT_DRIVER;
    init_attr.cap.max_inline_data   = iface->super.super.config.rx_inline;

    dv_init_attr.comp_mask                   = MLX5DV_QP_INIT_ATTR_MASK_DC;
    dv_init_attr.dc_init_attr.dc_type        = MLX5DV_DCTYPE_DCT;
    dv_init_attr.dc_init_attr.dct_access_key = UCT_IB_KEY;

    iface->rx_dct = mlx5dv_create_qp(dev->ibv_context,
                                     &init_attr, &dv_init_attr);
    if (iface->rx_dct == NULL) {
        ucs_error("Failed to created DC target %m");
        return UCS_ERR_INVALID_PARAM;
    }

    attr.pkey_index      = iface->super.super.super.pkey_index;
    attr.qp_state        = IBV_QPS_INIT;
    attr.port_num        = iface->super.super.super.config.port_num;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ  |
                           IBV_ACCESS_REMOTE_ATOMIC;

    ret = ibv_modify_qp(iface->rx_dct, &attr, IBV_QP_STATE |
                                              IBV_QP_PKEY_INDEX |
                                              IBV_QP_PORT |
                                              IBV_QP_ACCESS_FLAGS);

    if (ret) {
         ucs_error("error modifying DCT to INIT: %m");
         goto err;
    }

    attr.qp_state                  = IBV_QPS_RTR;
    attr.path_mtu                  = iface->super.super.config.path_mtu;
    attr.min_rnr_timer             = iface->super.super.config.min_rnr_timer;
    attr.ah_attr.grh.hop_limit     = iface->super.super.super.config.hop_limit;
    attr.ah_attr.grh.traffic_class = iface->super.super.super.config.traffic_class;
    attr.ah_attr.grh.sgid_index    = uct_ib_iface_md(&iface->super.super.super)->config.gid_index;
    attr.ah_attr.port_num          = iface->super.super.super.config.port_num;

    ret = ibv_modify_qp(iface->rx_dct, &attr, IBV_QP_STATE |
                                              IBV_QP_MIN_RNR_TIMER |
                                              IBV_QP_AV |
                                              IBV_QP_PATH_MTU);
    if (ret) {
         ucs_error("error modifying DCT to RTR: %m");
         goto err;
    }

    return UCS_OK;

err:
    ibv_destroy_qp(iface->rx_dct);
    return UCS_ERR_IO_ERROR;
}

int uct_dc_mlx5_get_dct_num(uct_dc_mlx5_iface_t *iface)
{
    return iface->rx_dct->qp_num;
}

void uct_dc_mlx5_destroy_dct(uct_dc_mlx5_iface_t *iface)
{
    if (iface->rx_dct != NULL) {
        ibv_destroy_qp(iface->rx_dct);
        iface->rx_dct = NULL;
    }
}
#endif

static ucs_status_t uct_dc_mlx5_iface_init_dcis(uct_dc_mlx5_iface_t *iface,
                                                uct_ib_mlx5_mmio_mode_t mmio_mode)
{
    ucs_status_t status;
    uint16_t bb_max;
    int i;

    bb_max = 0;
    for (i = 0; i < iface->tx.ndci; i++) {
        status = uct_ib_mlx5_txwq_init(iface->super.super.super.super.worker,
                                       mmio_mode, &iface->tx.dci_wqs[i],
                                       iface->tx.dcis[i].txqp.qp);
        if (status != UCS_OK) {
            return status;
        }


        bb_max = iface->tx.dci_wqs[i].bb_max;
        uct_rc_txqp_available_set(&iface->tx.dcis[i].txqp, bb_max);
    }

    iface->super.super.config.tx_qp_len = bb_max;
    return UCS_OK;
}

static void uct_dc_mlx5_iface_cleanup_dcis(uct_dc_mlx5_iface_t *iface)
{
    int i;

    for (i = 0; i < iface->tx.ndci; i++) {
        uct_ib_mlx5_txwq_cleanup(&iface->tx.dci_wqs[i]);
    }
}

static ucs_status_t
uct_dc_mlx5_init_rx(uct_rc_iface_t *rc_iface,
                    const uct_rc_iface_config_t *rc_config)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(rc_iface, uct_dc_mlx5_iface_t);

#if IBV_EXP_HW_TM_DC
    uct_dc_mlx5_iface_config_t *config = ucs_derived_of(rc_config,
                                                        uct_dc_mlx5_iface_config_t);
    if (UCT_RC_MLX5_TM_ENABLED(&iface->super)) {
         struct ibv_exp_create_srq_attr srq_attr      = {};
         struct ibv_exp_srq_dc_offload_params dc_op   = {};

         iface->super.super.progress = uct_dc_mlx5_iface_progress_tm;

         dc_op.timeout    = rc_iface->config.timeout;
         dc_op.path_mtu   = rc_iface->config.path_mtu;
         dc_op.pkey_index = rc_iface->super.pkey_index;
         dc_op.sl         = rc_iface->super.config.sl;
         dc_op.dct_key    = UCT_IB_KEY;

         srq_attr.comp_mask         = IBV_EXP_CREATE_SRQ_DC_OFFLOAD_PARAMS;
         srq_attr.dc_offload_params = &dc_op;

         return uct_rc_mlx5_init_rx_tm(&iface->super, &config->super,
                                       &srq_attr,
                                       sizeof(struct ibv_rvh) +
                                       sizeof(struct ibv_ravh), 0);
     }
#endif

    iface->super.super.progress = uct_dc_mlx5_iface_progress;
    return uct_rc_iface_init_rx(rc_iface, rc_config);
}

#ifdef HAVE_DC_EXP
ucs_status_t uct_dc_mlx5_iface_create_dct(uct_dc_mlx5_iface_t *iface)
{
    struct ibv_exp_dct_init_attr init_attr;

    memset(&init_attr, 0, sizeof(init_attr));

    init_attr.pd               = uct_ib_iface_md(&iface->super.super.super)->pd;
    init_attr.cq               = iface->super.super.super.cq[UCT_IB_DIR_RX];
    init_attr.srq              = iface->super.super.rx.srq.srq;
    init_attr.dc_key           = UCT_IB_KEY;
    init_attr.port             = iface->super.super.super.config.port_num;
    init_attr.mtu              = iface->super.super.config.path_mtu;
    init_attr.access_flags     = IBV_EXP_ACCESS_REMOTE_WRITE |
                                 IBV_EXP_ACCESS_REMOTE_READ |
                                 IBV_EXP_ACCESS_REMOTE_ATOMIC;
    init_attr.min_rnr_timer    = iface->super.super.config.min_rnr_timer;
    init_attr.tclass           = iface->super.super.super.config.traffic_class;
    init_attr.hop_limit        = iface->super.super.super.config.hop_limit;
    init_attr.gid_index        = iface->super.super.super.config.gid_index;
    init_attr.inline_size      = iface->super.super.config.rx_inline;
    init_attr.pkey_index       = iface->super.super.super.pkey_index;

#if HAVE_DECL_IBV_EXP_DCT_OOO_RW_DATA_PLACEMENT
    if (iface->super.super.config.ooo_rw &&
        UCX_IB_DEV_IS_OOO_SUPPORTED(&uct_ib_iface_device(&iface->super.super.super)->dev_attr,
                                    dc)) {
        ucs_debug("creating DC target with out-of-order support dev %s",
                   uct_ib_device_name(uct_ib_iface_device(&iface->super.super.super)));
        init_attr.create_flags |= IBV_EXP_DCT_OOO_RW_DATA_PLACEMENT;
    }
#endif

    iface->rx_dct = ibv_exp_create_dct(uct_ib_iface_device(&iface->super.super.super)->ibv_context,
                                       &init_attr);
    if (iface->rx_dct == NULL) {
        ucs_error("failed to create DC target: %m");
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

/* take dc qp to rts state */
ucs_status_t uct_dc_mlx5_iface_dci_connect(uct_dc_mlx5_iface_t *iface,
                                           uct_rc_txqp_t *dci)
{
    struct ibv_exp_qp_attr attr;
    long attr_mask;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state        = IBV_QPS_INIT;
    attr.pkey_index      = iface->super.super.super.pkey_index;
    attr.port_num        = iface->super.super.super.config.port_num;
    attr.dct_key         = UCT_IB_KEY;
    attr_mask            = IBV_EXP_QP_STATE      |
                           IBV_EXP_QP_PKEY_INDEX |
                           IBV_EXP_QP_PORT       |
                           IBV_EXP_QP_DC_KEY;

    if (ibv_exp_modify_qp(dci->qp, &attr, attr_mask)) {
        ucs_error("error modifying QP to INIT : %m");
        return UCS_ERR_IO_ERROR;
    }

    /* Move QP to the RTR state */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state                   = IBV_QPS_RTR;
    attr.path_mtu                   = iface->super.super.config.path_mtu;
    attr.max_dest_rd_atomic         = 1;
    attr.ah_attr.is_global          = iface->super.super.super.is_global_addr;
    attr.ah_attr.sl                 = iface->super.super.super.config.sl;
    attr_mask                       = IBV_EXP_QP_STATE     |
                                      IBV_EXP_QP_PATH_MTU  |
                                      IBV_EXP_QP_AV;

#if HAVE_DECL_IBV_EXP_QP_OOO_RW_DATA_PLACEMENT
    if (iface->super.super.config.ooo_rw &&
        UCX_IB_DEV_IS_OOO_SUPPORTED(&uct_ib_iface_device(&iface->super.super.super)->dev_attr,
                                    dc)) {
        ucs_debug("enabling out-of-order on DCI QP 0x%x dev %s", dci->qp->qp_num,
                   uct_ib_device_name(uct_ib_iface_device(&iface->super.super.super)));
        attr_mask |= IBV_EXP_QP_OOO_RW_DATA_PLACEMENT;
    }
#endif

    if (ibv_exp_modify_qp(dci->qp, &attr, attr_mask)) {
        ucs_error("error modifying QP to RTR: %m");
        return UCS_ERR_IO_ERROR;
    }

    /* Move QP to the RTS state */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state       = IBV_QPS_RTS;
    attr.timeout        = iface->super.super.config.timeout;
    attr.rnr_retry      = iface->super.super.config.rnr_retry;
    attr.retry_cnt      = iface->super.super.config.retry_cnt;
    attr.max_rd_atomic  = iface->super.super.config.max_rd_atomic;
    attr_mask           = IBV_EXP_QP_STATE      |
                          IBV_EXP_QP_TIMEOUT    |
                          IBV_EXP_QP_RETRY_CNT  |
                          IBV_EXP_QP_RNR_RETRY  |
                          IBV_EXP_QP_MAX_QP_RD_ATOMIC;

    if (ibv_exp_modify_qp(dci->qp, &attr, attr_mask)) {
        ucs_error("error modifying QP to RTS: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

int uct_dc_mlx5_get_dct_num(uct_dc_mlx5_iface_t *iface)
{
    return iface->rx_dct->dct_num;
}

void uct_dc_mlx5_destroy_dct(uct_dc_mlx5_iface_t *iface)
{
    if (iface->rx_dct != NULL) {
        ibv_exp_destroy_dct(iface->rx_dct);
        iface->rx_dct = NULL;
    }
}
#endif

void uct_dc_mlx5_iface_dcis_destroy(uct_dc_mlx5_iface_t *iface, int max)
{
    int i;
    for (i = 0; i < max; i++) {
        uct_rc_txqp_cleanup(&iface->tx.dcis[i].txqp);
    }
}

ucs_status_t uct_dc_mlx5_iface_create_dcis(uct_dc_mlx5_iface_t *iface,
                                           uct_dc_mlx5_iface_config_t *config)
{
    struct ibv_qp_cap cap;
    ucs_status_t status;
    int i;

    ucs_debug("creating %d dci(s)", iface->tx.ndci);

    iface->tx.stack_top = 0;
    for (i = 0; i < iface->tx.ndci; i++) {
        ucs_assert(iface->super.super.super.config.qp_type == UCT_IB_QPT_DCI);
        status = uct_rc_txqp_init(&iface->tx.dcis[i].txqp, &iface->super.super,
                                  &cap UCS_STATS_ARG(iface->super.super.stats));
        if (status != UCS_OK) {
            goto err;
        }

        status = uct_dc_mlx5_iface_dci_connect(iface, &iface->tx.dcis[i].txqp);
        if (status != UCS_OK) {
            uct_rc_txqp_cleanup(&iface->tx.dcis[i].txqp);
            goto err;
        }

        iface->tx.dcis_stack[i] = i;
        iface->tx.dcis[i].ep    = NULL;
#ifdef ENABLE_ASSERT
        iface->tx.dcis[i].flags = 0;
#endif
    }
    uct_ib_iface_set_max_iov(&iface->super.super.super, cap.max_send_sge);
    return UCS_OK;

err:
    uct_dc_mlx5_iface_dcis_destroy(iface, i);
    return status;
}

void uct_dc_mlx5_iface_set_quota(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_iface_config_t *config)
{
    iface->tx.available_quota = iface->super.super.config.tx_qp_len -
                                ucs_min(iface->super.super.config.tx_qp_len, config->quota);
}

void uct_dc_mlx5_iface_init_version(uct_dc_mlx5_iface_t *iface, uct_md_h md)
{
    uct_ib_device_t *dev;
    unsigned         ver;

    dev = &ucs_derived_of(md, uct_ib_md_t)->dev;
    ver = uct_ib_device_spec(dev)->flags & UCT_IB_DEVICE_FLAG_DC;
    ucs_assert(ver != UCT_IB_DEVICE_FLAG_DC);

    iface->version_flag = 0;

    if (ver & UCT_IB_DEVICE_FLAG_DC_V2) {
        iface->version_flag = UCT_DC_MLX5_IFACE_ADDR_DC_V2;
    }

    if (ver & UCT_IB_DEVICE_FLAG_DC_V1) {
        iface->version_flag = UCT_DC_MLX5_IFACE_ADDR_DC_V1;
    }
}

int uct_dc_mlx5_iface_is_reachable(const uct_iface_h tl_iface,
                                   const uct_device_addr_t *dev_addr,
                                   const uct_iface_addr_t *iface_addr)
{
    uct_dc_mlx5_iface_addr_t *addr = (uct_dc_mlx5_iface_addr_t *)iface_addr;
    uct_dc_mlx5_iface_t UCS_V_UNUSED *iface;

    iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    ucs_assert_always(iface_addr != NULL);

    return ((addr->flags & UCT_DC_MLX5_IFACE_ADDR_DC_VERS) == iface->version_flag) &&
           (UCT_DC_MLX5_IFACE_ADDR_TM_ENABLED(addr) ==
            UCT_RC_MLX5_TM_ENABLED(&iface->super)) &&
           uct_ib_iface_is_reachable(tl_iface, dev_addr, iface_addr);
}

ucs_status_t
uct_dc_mlx5_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    uct_dc_mlx5_iface_t      *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_iface_addr_t *addr  = (uct_dc_mlx5_iface_addr_t *)iface_addr;

    uct_ib_pack_uint24(addr->qp_num, uct_dc_mlx5_get_dct_num(iface));
    addr->atomic_mr_id = uct_ib_iface_get_atomic_mr_id(&iface->super.super.super);
    addr->flags        = iface->version_flag;
    if (UCT_RC_MLX5_TM_ENABLED(&iface->super)) {
        addr->flags   |= UCT_DC_MLX5_IFACE_ADDR_HW_TM;
    }

    return UCS_OK;
}

ucs_status_t uct_dc_device_query_tl_resources(uct_ib_device_t *dev,
                                              const char *tl_name, unsigned flags,
                                              uct_tl_resource_desc_t **resources_p,
                                              unsigned *num_resources_p)
{
    return uct_ib_device_query_tl_resources(dev, tl_name,
                                            flags | UCT_IB_DEVICE_FLAG_DC,
                                            resources_p, num_resources_p);
}

static inline ucs_status_t uct_dc_mlx5_iface_flush_dcis(uct_dc_mlx5_iface_t *iface)
{
    int is_flush_done = 1;
    uct_dc_mlx5_ep_t *ep;
    int i;

    for (i = 0; i < iface->tx.ndci; i++) {
        /* TODO: Remove this check - no need to wait for grant, because we
         * use gc_list for removed eps */
        if (!uct_dc_mlx5_iface_is_dci_rand(iface)) {
            ep = uct_dc_mlx5_ep_from_dci(iface, i);
            if ((ep != NULL) && uct_dc_mlx5_ep_fc_wait_for_grant(ep)) {
                return UCS_INPROGRESS;
            }
        }
        if (uct_dc_mlx5_iface_flush_dci(iface, i) != UCS_OK) {
            is_flush_done = 0;
        }
    }
    return is_flush_done ? UCS_OK : UCS_INPROGRESS;
}

ucs_status_t uct_dc_mlx5_iface_flush(uct_iface_h tl_iface, unsigned flags, uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    ucs_status_t status;

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }
    status = uct_dc_mlx5_iface_flush_dcis(iface);
    if (status == UCS_OK) {
        UCT_TL_IFACE_STAT_FLUSH(&iface->super.super.super.super);
    }
    else if (status == UCS_INPROGRESS) {
        UCT_TL_IFACE_STAT_FLUSH_WAIT(&iface->super.super.super.super);
    }
    return status;
}

ucs_status_t uct_dc_mlx5_iface_init_fc_ep(uct_dc_mlx5_iface_t *iface)
{
    ucs_status_t status;
    uct_dc_mlx5_ep_t *ep;

    ep = ucs_malloc(sizeof(uct_dc_mlx5_ep_t), "fc_ep");
    if (ep == NULL) {
        ucs_error("Failed to allocate FC ep");
        status =  UCS_ERR_NO_MEMORY;
        goto err;
    }
    /* We do not have any peer address at this point, so init basic subclasses
     * only (for statistics, iface, etc) */
    status = UCS_CLASS_INIT(uct_base_ep_t, (void*)(&ep->super),
                            &iface->super.super.super.super);
    if (status != UCS_OK) {
        ucs_error("Failed to initialize fake FC ep, status: %s",
                  ucs_status_string(status));
        goto err_free;
    }

    status = uct_dc_mlx5_ep_basic_init(iface, ep);
    if (status != UCS_OK) {
        ucs_error("FC ep init failed %s", ucs_status_string(status));
        goto err_cleanup;
    }

    iface->tx.fc_ep = ep;
    return UCS_OK;

err_cleanup:
    UCS_CLASS_CLEANUP(uct_base_ep_t, &ep->super);
err_free:
    ucs_free(ep);
err:
    return status;
}

void uct_dc_mlx5_iface_cleanup_fc_ep(uct_dc_mlx5_iface_t *iface)
{
    uct_dc_mlx5_ep_pending_purge(&iface->tx.fc_ep->super.super, NULL, NULL);
    ucs_arbiter_group_cleanup(&iface->tx.fc_ep->arb_group);
    uct_rc_fc_cleanup(&iface->tx.fc_ep->fc);
    UCS_CLASS_CLEANUP(uct_base_ep_t, iface->tx.fc_ep);
    ucs_free(iface->tx.fc_ep);
}

ucs_status_t uct_dc_mlx5_iface_fc_grant(uct_pending_req_t *self)
{
    uct_rc_fc_request_t *freq = ucs_derived_of(self, uct_rc_fc_request_t);
    uct_dc_mlx5_ep_t *ep      = ucs_derived_of(freq->ep, uct_dc_mlx5_ep_t);
    uct_rc_iface_t *iface     = ucs_derived_of(ep->super.super.iface,
                                               uct_rc_iface_t);
    ucs_status_t status;

    ucs_assert_always(iface->config.fc_enabled);

    status = uct_rc_fc_ctrl(&ep->super.super, UCT_RC_EP_FC_PURE_GRANT, freq);
    if (status == UCS_OK) {
        ucs_mpool_put(freq);
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_TX_PURE_GRANT, 1);
    }
    return status;
}

ucs_status_t uct_dc_mlx5_iface_fc_handler(uct_rc_iface_t *rc_iface, unsigned qp_num,
                                          uct_rc_hdr_t *hdr, unsigned length,
                                          uint32_t imm_data, uint16_t lid, unsigned flags)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(rc_iface, uct_dc_mlx5_iface_t);
    uint8_t             fc_hdr = uct_rc_fc_get_fc_hdr(hdr->am_id);
    uct_dc_fc_request_t *dc_req;
    int16_t             cur_wnd;
    ucs_status_t        status;
    uct_dc_mlx5_ep_t    *ep;

    ucs_assert(rc_iface->config.fc_enabled);

    if (fc_hdr == UCT_RC_EP_FC_FLAG_HARD_REQ) {
        ep = iface->tx.fc_ep;
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_RX_HARD_REQ, 1);

        dc_req = ucs_mpool_get(&iface->super.super.tx.fc_mp);
        if (ucs_unlikely(dc_req == NULL)) {
            ucs_error("Failed to allocate FC request");
            return UCS_ERR_NO_MEMORY;
        }
        dc_req->super.super.func = uct_dc_mlx5_iface_fc_grant;
        dc_req->super.ep         = &ep->super.super;
        dc_req->dct_num          = imm_data;
        dc_req->lid              = lid;
        dc_req->sender           = *((uct_dc_fc_sender_data_t*)(hdr + 1));

        status = uct_dc_mlx5_iface_fc_grant(&dc_req->super.super);
        if (status == UCS_ERR_NO_RESOURCE){
            status = uct_ep_pending_add(&ep->super.super, &dc_req->super.super,
                                        0);
        }
        ucs_assertv_always(status == UCS_OK, "Failed to send FC grant msg: %s",
                           ucs_status_string(status));
    } else if (fc_hdr == UCT_RC_EP_FC_PURE_GRANT) {
        ep = *((uct_dc_mlx5_ep_t**)(hdr + 1));

        if (!(ep->flags & UCT_DC_MLX5_EP_FLAG_VALID)) {
            uct_dc_mlx5_ep_release(ep);
            return UCS_OK;
        }

        cur_wnd = ep->fc.fc_wnd;

        /* Peer granted resources, so update wnd */
        ep->fc.fc_wnd = rc_iface->config.fc_wnd_size;

        /* Clear the flag for flush to complete  */
        ep->fc.flags &= ~UCT_DC_MLX5_EP_FC_FLAG_WAIT_FOR_GRANT;

        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_RX_PURE_GRANT, 1);
        UCS_STATS_SET_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_FC_WND, ep->fc.fc_wnd);

        /* To preserve ordering we have to dispatch all pending
         * operations if current fc_wnd is <= 0 */
        if (cur_wnd <= 0) {
            if (ep->dci == UCT_DC_MLX5_EP_NO_DCI) {
                ucs_arbiter_group_schedule(uct_dc_mlx5_iface_dci_waitq(iface),
                                           &ep->arb_group);
            } else {
                /* Need to schedule fake ep in TX arbiter, because it
                 * might have been descheduled due to lack of FC window. */
                ucs_arbiter_group_schedule(uct_dc_mlx5_iface_tx_waitq(iface),
                                           uct_dc_mlx5_ep_arb_group(iface, ep));
            }

            uct_dc_mlx5_iface_progress_pending(iface);
        }
    }

    return UCS_OK;
}

void uct_dc_mlx5_iface_set_av_sport(uct_dc_mlx5_iface_t *iface,
                                    uct_ib_mlx5_base_av_t *av,
                                    uint32_t remote_dctn)
{
    uct_ib_mlx5_iface_set_av_sport(&iface->super.super.super, av,
                                   remote_dctn ^ uct_dc_mlx5_get_dct_num(iface));
}

static void uct_dc_mlx5_iface_handle_failure(uct_ib_iface_t *ib_iface,
                                             void *arg, ucs_status_t status)
{
    uct_dc_mlx5_iface_t  *iface  = ucs_derived_of(ib_iface, uct_dc_mlx5_iface_t);
    struct mlx5_cqe64    *cqe    = arg;
    uint32_t             qp_num  = ntohl(cqe->sop_drop_qpn) &
                                   UCS_MASK(UCT_IB_QPN_ORDER);
    uint8_t              dci     = uct_dc_mlx5_iface_dci_find(iface, qp_num);
    uct_rc_txqp_t        *txqp   = &iface->tx.dcis[dci].txqp;
    uct_dc_mlx5_ep_t     *ep;
    ucs_status_t         ep_status;
    int16_t              outstanding;

    if (uct_dc_mlx5_iface_is_dci_rand(iface) ||
        (uct_dc_mlx5_ep_from_dci(iface, dci) == NULL)) {
        uct_ib_mlx5_completion_with_err(ib_iface, arg, &iface->tx.dci_wqs[dci],
                                        ib_iface->super.config.failure_level);
        return;
    }

    ep = uct_dc_mlx5_ep_from_dci(iface, dci);

    uct_rc_txqp_purge_outstanding(txqp, status, 0);

    /* poll_cqe for mlx5 returns NULL in case of failure and the cq_avaialble
       is not updated for the error cqe and all outstanding wqes*/
    outstanding = (int16_t)iface->super.super.config.tx_qp_len -
                  uct_rc_txqp_available(txqp);
    iface->super.super.tx.cq_available += outstanding;
    uct_rc_txqp_available_set(txqp, (int16_t)iface->super.super.config.tx_qp_len);

    /* since we removed all outstanding ops on the dci, it should be released */
    ucs_assert(ep->dci != UCT_DC_MLX5_EP_NO_DCI);
    uct_dc_mlx5_iface_dci_put(iface, dci);
    ucs_assert_always(ep->dci == UCT_DC_MLX5_EP_NO_DCI);

    if (ep == iface->tx.fc_ep) {
        /* Cannot handle errors on flow-control endpoint.
         * Or shall we ignore them?
         */
        ucs_debug("got error on DC flow-control endpoint, iface %p: %s", iface,
                  ucs_status_string(status));
        ep_status = UCS_OK;
    } else {
        ep_status = iface->super.super.super.ops->set_ep_failed(ib_iface,
                                                                &ep->super.super,
                                                                status);
        if (ep_status != UCS_OK) {
            uct_ib_mlx5_completion_with_err(ib_iface, arg,
                                            &iface->tx.dci_wqs[dci],
                                            UCS_LOG_LEVEL_FATAL);
            return;
        }
    }

    uct_ib_mlx5_completion_with_err(ib_iface, arg, &iface->tx.dci_wqs[dci],
                                    ib_iface->super.config.failure_level);

    status = uct_dc_mlx5_iface_reset_dci(iface, dci);
    if (status != UCS_OK) {
        ucs_fatal("iface %p failed to reset dci[%d] qpn 0x%x: %s",
                  iface, dci, txqp->qp->qp_num, ucs_status_string(status));
    }

    status = uct_dc_mlx5_iface_dci_connect(iface, txqp);
    if (status != UCS_OK) {
        ucs_fatal("iface %p failed to connect dci[%d] qpn 0x%x: %s",
                  iface, dci, txqp->qp->qp_num, ucs_status_string(status));
    }
}

static uct_rc_iface_ops_t uct_dc_mlx5_iface_ops = {
    {
    {
    .ep_put_short             = uct_dc_mlx5_ep_put_short,
    .ep_put_bcopy             = uct_dc_mlx5_ep_put_bcopy,
    .ep_put_zcopy             = uct_dc_mlx5_ep_put_zcopy,
    .ep_get_bcopy             = uct_dc_mlx5_ep_get_bcopy,
    .ep_get_zcopy             = uct_dc_mlx5_ep_get_zcopy,
    .ep_am_short              = uct_dc_mlx5_ep_am_short,
    .ep_am_bcopy              = uct_dc_mlx5_ep_am_bcopy,
    .ep_am_zcopy              = uct_dc_mlx5_ep_am_zcopy,
    .ep_atomic_cswap64        = uct_dc_mlx5_ep_atomic_cswap64,
    .ep_atomic_cswap32        = uct_dc_mlx5_ep_atomic_cswap32,
    .ep_atomic64_post         = uct_dc_mlx5_ep_atomic64_post,
    .ep_atomic32_post         = uct_dc_mlx5_ep_atomic32_post,
    .ep_atomic64_fetch        = uct_dc_mlx5_ep_atomic64_fetch,
    .ep_atomic32_fetch        = uct_dc_mlx5_ep_atomic32_fetch,
    .ep_pending_add           = uct_dc_mlx5_ep_pending_add,
    .ep_pending_purge         = uct_dc_mlx5_ep_pending_purge,
    .ep_flush                 = uct_dc_mlx5_ep_flush,
    .ep_fence                 = uct_dc_mlx5_ep_fence,
#if IBV_EXP_HW_TM_DC
    .ep_tag_eager_short       = uct_dc_mlx5_ep_tag_eager_short,
    .ep_tag_eager_bcopy       = uct_dc_mlx5_ep_tag_eager_bcopy,
    .ep_tag_eager_zcopy       = uct_dc_mlx5_ep_tag_eager_zcopy,
    .ep_tag_rndv_zcopy        = uct_dc_mlx5_ep_tag_rndv_zcopy,
    .ep_tag_rndv_request      = uct_dc_mlx5_ep_tag_rndv_request,
    .ep_tag_rndv_cancel       = uct_rc_mlx5_ep_tag_rndv_cancel,
    .iface_tag_recv_zcopy     = uct_dc_mlx5_iface_tag_recv_zcopy,
    .iface_tag_recv_cancel    = uct_dc_mlx5_iface_tag_recv_cancel,
#endif
    .iface_flush              = uct_dc_mlx5_iface_flush,
    .iface_fence              = uct_rc_mlx5_iface_fence,
    .iface_progress_enable    = uct_dc_mlx5_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_rc_iface_do_progress,
    .iface_event_fd_get       = uct_ib_iface_event_fd_get,
    .iface_event_arm          = uct_rc_iface_event_arm,
    .ep_create                = uct_dc_mlx5_ep_create_connected,
    .ep_destroy               = uct_dc_mlx5_ep_destroy,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_dc_mlx5_iface_t),
    .iface_query              = uct_dc_mlx5_iface_query,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_dc_mlx5_iface_is_reachable,
    .iface_get_address        = uct_dc_mlx5_iface_get_address,
    },
    .create_cq                = uct_ib_mlx5_create_cq,
    .arm_cq                   = uct_ib_iface_arm_cq,
    .event_cq                 = uct_dc_mlx5_iface_event_cq,
    .handle_failure           = uct_dc_mlx5_iface_handle_failure,
    .set_ep_failed            = uct_dc_mlx5_ep_set_failed,
    .create_qp                = uct_dc_mlx5_iface_create_qp,
    .init_res_domain          = uct_rc_mlx5_init_res_domain,
    .cleanup_res_domain       = uct_rc_mlx5_cleanup_res_domain,
    },
    .init_rx                  = uct_dc_mlx5_init_rx,
    .fc_ctrl                  = uct_dc_mlx5_ep_fc_ctrl,
    .fc_handler               = uct_dc_mlx5_iface_fc_handler,
};

static UCS_CLASS_INIT_FUNC(uct_dc_mlx5_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_dc_mlx5_iface_config_t *config = ucs_derived_of(tl_config,
                                                        uct_dc_mlx5_iface_config_t);
    uct_ib_iface_init_attr_t init_attr = {};
    ucs_status_t status;
    ucs_trace_func("");

    init_attr.tm_cap_bit     = IBV_EXP_TM_CAP_DC;
    init_attr.qp_type        = UCT_IB_QPT_DCI;
    init_attr.flags          = UCT_IB_CQ_IGNORE_OVERRUN;
    init_attr.fc_req_size    = sizeof(uct_dc_fc_request_t);
    init_attr.rx_hdr_len     = sizeof(uct_rc_mlx5_hdr_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_mlx5_iface_common_t,
                              &uct_dc_mlx5_iface_ops,
                              md, worker, params,
                              &config->super, &init_attr);
    if (config->ndci < 1) {
        ucs_error("dc interface must have at least 1 dci (requested: %d)",
                  config->ndci);
        return UCS_ERR_INVALID_PARAM;
    }

    if (config->ndci > UCT_DC_MLX5_IFACE_MAX_DCIS) {
        ucs_error("dc interface can have at most %d dcis (requested: %d)",
                  UCT_DC_MLX5_IFACE_MAX_DCIS, config->ndci);
        return UCS_ERR_INVALID_PARAM;
    }

    uct_dc_mlx5_iface_init_version(self, md);

    self->tx.ndci                          = config->ndci;
    self->tx.policy                        = config->tx_policy;
    self->super.super.config.tx_moderation = 0; /* disable tx moderation for dcs */
    ucs_list_head_init(&self->tx.gc_list);

    self->tx.rand_seed = config->rand_seed ? config->rand_seed : time(NULL);
    self->tx.pend_cb   = uct_dc_mlx5_iface_is_dci_rand(self) ?
                         uct_dc_mlx5_iface_dci_do_rand_pending_tx :
                         uct_dc_mlx5_iface_dci_do_dcs_pending_tx;

    /* create DC target */
    status = uct_dc_mlx5_iface_create_dct(self);
    if (status != UCS_OK) {
        goto err;
    }

    /* create DC initiators */
    status = uct_dc_mlx5_iface_create_dcis(self, config);
    if (status != UCS_OK) {
        goto err_destroy_dct;
    }

    ucs_debug("dc iface %p: using '%s' policy with %d dcis, dct 0x%x", self,
              uct_dc_tx_policy_names[self->tx.policy], self->tx.ndci,
              UCT_RC_MLX5_TM_ENABLED(&self->super) ?
              0 : uct_dc_mlx5_get_dct_num(self));

    /* Create fake endpoint which will be used for sending FC grants */
    uct_dc_mlx5_iface_init_fc_ep(self);

    ucs_arbiter_init(&self->tx.dci_arbiter);

    /* mlx5 init part */
    status = uct_ud_mlx5_iface_common_init(&self->super.super.super,
                                           &self->ud_common, &config->mlx5_ud);
    if (status != UCS_OK) {
        goto err_destroy_dct;
    }

    status = uct_dc_mlx5_iface_init_dcis(self, self->super.tx.mmio_mode);
    if (status != UCS_OK) {
        goto err_destroy_dct;
    }

    self->tx.available_quota = self->super.super.config.tx_qp_len -
                               ucs_min(self->super.super.config.tx_qp_len, config->quota);
    /* Set max_iov for put_zcopy and get_zcopy */
    uct_ib_iface_set_max_iov(&self->super.super.super,
                             (UCT_IB_MLX5_MAX_SEND_WQE_SIZE -
                             sizeof(struct mlx5_wqe_raddr_seg) -
                             sizeof(struct mlx5_wqe_ctrl_seg) -
                             UCT_IB_MLX5_AV_FULL_SIZE) /
                             sizeof(struct mlx5_wqe_data_seg));

    uct_rc_mlx5_iface_common_prepost_recvs(&self->super);

    ucs_debug("created dc iface %p", self);

    return UCS_OK;

err_destroy_dct:
    uct_dc_mlx5_destroy_dct(self);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_mlx5_iface_t)
{
    uct_dc_mlx5_ep_t *ep, *tmp;

    ucs_trace_func("");
    uct_base_iface_progress_disable(&self->super.super.super.super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    uct_dc_mlx5_iface_cleanup_dcis(self);

    uct_dc_mlx5_destroy_dct(self);

    ucs_list_for_each_safe(ep, tmp, &self->tx.gc_list, list) {
        uct_dc_mlx5_ep_release(ep);
    }
    uct_dc_mlx5_iface_dcis_destroy(self, self->tx.ndci);
    uct_dc_mlx5_iface_cleanup_fc_ep(self);
    ucs_arbiter_cleanup(&self->tx.dci_arbiter);
}

UCS_CLASS_DEFINE(uct_dc_mlx5_iface_t, uct_rc_mlx5_iface_common_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_dc_mlx5_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_dc_mlx5_iface_t, uct_iface_t);

static
ucs_status_t uct_dc_mlx5_query_resources(uct_md_h md,
                                         uct_tl_resource_desc_t **resources_p,
                                         unsigned *num_resources_p)
{
    uct_ib_md_t *ib_md = ucs_derived_of(md, uct_ib_md_t);

    return uct_dc_device_query_tl_resources(&ib_md->dev,"dc_mlx5",
                                            UCT_IB_DEVICE_FLAG_MLX5_PRM |
                                            (ib_md->config.eth_pause ? 0 : UCT_IB_DEVICE_FLAG_LINK_IB),
                                            resources_p, num_resources_p);
}

UCT_TL_COMPONENT_DEFINE(uct_dc_mlx5_tl,
                        uct_dc_mlx5_query_resources,
                        uct_dc_mlx5_iface_t,
                        "dc_mlx5",
                        "DC_MLX5_",
                        uct_dc_mlx5_iface_config_table,
                        uct_dc_mlx5_iface_config_t);
UCT_MD_REGISTER_TL(&uct_ib_mdc, &uct_dc_mlx5_tl);
