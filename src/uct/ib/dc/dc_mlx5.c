/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2020. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "dc_mlx5.inl"

#include <uct/api/uct.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_log.h>
#include <uct/ib/mlx5/ib_mlx5_log.h>
#include <ucs/profile/profile.h>
#include <ucs/vfs/base/vfs_cb.h>
#include <ucs/vfs/base/vfs_obj.h>
#include <uct/base/uct_md.h>
#include <ucs/arch/bitops.h>
#include <ucs/arch/cpu.h>
#include <ucs/async/async.h>
#include <ucs/debug/log.h>
#include <string.h>


#define UCT_DC_MLX5_MAX_TX_CQ_LEN (16 * UCS_MBYTE)


static const char *uct_dc_tx_policy_names[] = {
    [UCT_DC_TX_POLICY_DCS]           = "dcs",
    [UCT_DC_TX_POLICY_DCS_QUOTA]     = "dcs_quota",
    [UCT_DC_TX_POLICY_RAND]          = "rand",
    [UCT_DC_TX_POLICY_HW_DCS]        = "hw_dcs",
    [UCT_DC_TX_POLICY_LAST]          = NULL
};

static const char *uct_dct_affinity_policy_names[] = {
    [UCT_DC_MLX5_DCT_AFFINITY_DEFAULT] = "default",
    [UCT_DC_MLX5_DCT_AFFINITY_RANDOM]  = "random",
    [UCT_DC_MLX5_DCT_AFFINITY_LAST]    = NULL
};

/* DC specific parameters, expecting DC_ prefix */
ucs_config_field_t uct_dc_mlx5_iface_config_sub_table[] = {
    {"RC_", "IB_TX_QUEUE_LEN=128;FC_ENABLE=y;", NULL,
     ucs_offsetof(uct_dc_mlx5_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_rc_iface_common_config_table)},

    /* Since long timeout will block SRQ in case of network failure on single
     * peer default SRQ to list topology. Incur performance degradation. */
    {"RC_", "SRQ_TOPO=list", NULL,
     ucs_offsetof(uct_dc_mlx5_iface_config_t, rc_mlx5_common),
     UCS_CONFIG_TYPE_TABLE(uct_rc_mlx5_common_config_table)},

    {"UD_", "", NULL,
     ucs_offsetof(uct_dc_mlx5_iface_config_t, ud_common),
     UCS_CONFIG_TYPE_TABLE(uct_ud_iface_common_config_table)},

    {"NUM_DCI", "8",
     "Number of DC initiator QPs (DCI) used by the interface. Not relevant for hw_dcs policy.",
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
     "           Multiple endpoints may share the same DCI.\n"
     "\n"
     "hw_dcs     A single DCI that operates as a HW DCS queue. The channels are assigned\n"
     "           in a round-robin fashion.",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, tx_policy),
     UCS_CONFIG_TYPE_ENUM(uct_dc_tx_policy_names)},

    {"LAG_PORT_AFFINITY", "auto",
     "Specifies how DCI select port under RoCE LAG. The values are:\n"
     " auto     Set DCI QP port affinity only if the hardware is configured\n"
     "          to QUEUE_AFFINITY mode.\n"
     " on       Always set DCI QP port affinity.\n"
     " off      Never set DCI QP port affinity.\n",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, tx_port_affinity),
     UCS_CONFIG_TYPE_ON_OFF_AUTO},

    {"DCI_FULL_HANDSHAKE", "no",
     "Force full-handshake protocol for DC initiator. Enabling this mode\n"
     "increases network latency, but is more resilient to packet drops.\n"
     "Setting it to \"auto\" applies full-handshake on AR SLs.",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, dci_full_handshake),
     UCS_CONFIG_TYPE_TERNARY},

    {"DCT_PORT_AFFINITY", "default",
     "Specifies how to set DCT port affinity under queue affinity RoCE LAG. "
     "The values are:\n"
     " default : Set affinity to the first physical port.\n"
     " random  : Use random physical port for each iface.\n"
     " <num>   : Set affinity to this physical port.",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, dct_affinity),
     UCS_CONFIG_TYPE_UINT_ENUM(uct_dct_affinity_policy_names)},

    {"DCI_KA_FULL_HANDSHAKE", "no",
     "Force full-handshake protocol for DC keepalive initiator.",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, dci_ka_full_handshake),
     UCS_CONFIG_TYPE_TERNARY},

    {"DCT_FULL_HANDSHAKE", "no", "Force full-handshake protocol for DC target.",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, dct_full_handshake),
     UCS_CONFIG_TYPE_TERNARY},

    {"RAND_DCI_SEED", "0",
     "Seed for DCI allocation when \"rand\" dci policy is used (0 - use default).",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, rand_seed), UCS_CONFIG_TYPE_UINT},

    {"QUOTA", "32",
     "When \"dcs_quota\" policy is selected, how much to send from a DCI when\n"
     "there are other endpoints waiting for it.",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, quota), UCS_CONFIG_TYPE_UINT},

    {"FC_HARD_REQ_TIMEOUT", "5s",
     "Timeout for re-sending FC_HARD_REQ when FC window is empty.",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, fc_hard_req_timeout),
     UCS_CONFIG_TYPE_TIME_UNITS},

    {"NUM_DCI_CHANNELS", "8",
     "Number of stream channels per DCI to be used. A value "
     "of 1 disables DCI multi-channel support. Relevant only for hw_dcs policy.",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, num_dci_channels),
     UCS_CONFIG_TYPE_UINT},

    {NULL}
};

/* Bundle of all parameters */
ucs_config_field_t uct_dc_mlx5_iface_config_table[] = {
    {"DC_", "", NULL, 0,
     UCS_CONFIG_TYPE_TABLE(uct_dc_mlx5_iface_config_sub_table)},

    {"UD_", "", NULL,
     ucs_offsetof(uct_dc_mlx5_iface_config_t, mlx5_ud),
     UCS_CONFIG_TYPE_TABLE(uct_ud_mlx5_iface_common_config_table)},

    {NULL}
};

static void
uct_dc_mlx5_dci_keepalive_handle_failure(uct_dc_mlx5_iface_t *iface,
                                         struct mlx5_cqe64 *cqe,
                                         uint8_t dci_index,
                                         ucs_status_t ep_status);


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
    unsigned path_index;

    ucs_trace_func("");

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);
    ib_addr    = (const uct_ib_address_t *)params->dev_addr;
    if_addr    = (const uct_dc_mlx5_iface_addr_t *)params->iface_addr;
    path_index = UCT_EP_PARAMS_GET_PATH_INDEX(params);

    status = uct_ud_mlx5_iface_get_av(&iface->super.super.super,
                                      &iface->ud_common, ib_addr, path_index,
                                      "DC ep create", &av, &grh_av, &is_global);
    if (status != UCS_OK) {
        return UCS_ERR_INVALID_ADDR;
    }

    if (is_global) {
        return UCS_CLASS_NEW(uct_dc_mlx5_grh_ep_t, ep_p, iface, if_addr, &av,
                             path_index, &grh_av);
    } else {
        return UCS_CLASS_NEW(uct_dc_mlx5_ep_t, ep_p, iface, if_addr, &av,
                             path_index);
    }
}

static ucs_status_t uct_dc_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    size_t max_am_inline       = UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE);
    size_t max_put_inline      = UCT_IB_MLX5_PUT_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE);
    ucs_status_t status;

#if HAVE_IBV_DM
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
                                sizeof(uct_rc_mlx5_hdr_t),
                                UCT_RC_MLX5_RMA_MAX_IOV(UCT_IB_MLX5_AV_FULL_SIZE));
    if (status != UCS_OK) {
        return status;
    }

    /* fixup flags and address lengths */
    iface_attr->cap.flags     &= ~UCT_IFACE_FLAG_CONNECT_TO_EP;
    iface_attr->cap.flags     |= UCT_IFACE_FLAG_CONNECT_TO_IFACE;
    iface_attr->ep_addr_len    = 0;
    iface_attr->max_conn_priv  = 0;
    iface_attr->iface_addr_len = iface->super.super.config.flush_remote ?
                                 sizeof(uct_dc_mlx5_iface_flush_addr_t) :
                                 sizeof(uct_dc_mlx5_iface_addr_t);
    iface_attr->latency.c     += 60e-9; /* connect packet + cqe */

    uct_rc_mlx5_iface_common_query(&iface->super.super.super, iface_attr,
                                   max_am_inline,
                                   UCT_RC_MLX5_TM_EAGER_ZCOPY_MAX_IOV(UCT_IB_MLX5_AV_FULL_SIZE));

    if (iface->flags & UCT_DC_MLX5_IFACE_FLAG_DISABLE_PUT) {
        iface_attr->cap.flags &= ~(UCT_IFACE_FLAG_PUT_SHORT |
                                   UCT_IFACE_FLAG_PUT_BCOPY |
                                   UCT_IFACE_FLAG_PUT_ZCOPY);
    }

    /* Error handling is not supported with random dci policy
     * TODO: Fix */
    if (uct_dc_mlx5_iface_is_dci_shared(iface)) {
        iface_attr->cap.flags &= ~(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE |
                                   UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF    |
                                   UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM);
    } else {
        iface_attr->cap.flags |= UCT_IFACE_FLAG_EP_CHECK;
    }

    return UCS_OK;
}

static void uct_dc_mlx5_iface_progress_enable(uct_iface_h tl_iface, unsigned flags)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);

    uct_base_iface_progress_enable_cb(&iface->super.super, iface->progress, flags);
}

static UCS_F_ALWAYS_INLINE unsigned
uct_dc_mlx5_poll_tx(uct_dc_mlx5_iface_t *iface, int poll_flags)
{
    uint8_t dci_index;
    struct mlx5_cqe64 *cqe;
    uint16_t hw_ci;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    cqe = uct_ib_mlx5_poll_cq(&iface->super.super.super,
                              &iface->super.cq[UCT_IB_DIR_TX], poll_flags,
                              uct_ib_mlx5_check_completion);
    if (cqe == NULL) {
        return 0;
    }
    UCS_STATS_UPDATE_COUNTER(iface->super.super.super.stats,
                             UCT_IB_IFACE_STAT_TX_COMPLETION, 1);

    ucs_memory_cpu_load_fence();

    dci_index = uct_dc_mlx5_iface_dci_find(iface, cqe);
    txqp      = &iface->tx.dcis[dci_index].txqp;
    txwq      = &iface->tx.dcis[dci_index].txwq;
    hw_ci     = ntohs(cqe->wqe_counter);

    ucs_trace_poll("dc iface %p tx_cqe: dci[%d] txqp %p hw_ci %d",
                   iface, dci_index, txqp, hw_ci);

    uct_rc_mlx5_txqp_process_tx_cqe(txqp, cqe, hw_ci);
    uct_dc_mlx5_update_tx_res(iface, txwq, txqp, hw_ci);

    /**
     * Note: DCI is released after handling completion callbacks,
     *       to avoid OOO sends when this is the only missing resource.
     */
    uct_dc_mlx5_iface_dci_put(iface, dci_index);
    uct_dc_mlx5_iface_progress_pending(iface,
                                       iface->tx.dcis[dci_index].pool_index);
    uct_dc_mlx5_iface_check_tx(iface);
    uct_ib_mlx5_update_db_cq_ci(&iface->super.cq[UCT_IB_DIR_TX]);

    return 1;
}

static UCS_F_ALWAYS_INLINE unsigned
uct_dc_mlx5_iface_progress(void *arg, int flags)
{
    uct_dc_mlx5_iface_t *iface = arg;
    unsigned count;

    count = uct_rc_mlx5_iface_common_poll_rx(&iface->super, flags);
    if (!uct_rc_iface_poll_tx(&iface->super.super, count)) {
        return count;
    }

    return count + uct_dc_mlx5_poll_tx(iface, flags);
}

static unsigned uct_dc_mlx5_iface_progress_cyclic(void *arg)
{
    return uct_dc_mlx5_iface_progress(arg, UCT_IB_MLX5_POLL_FLAG_CQE_ZIP);
}

static unsigned uct_dc_mlx5_iface_progress_ll(void *arg)
{
    return uct_dc_mlx5_iface_progress(arg, UCT_IB_MLX5_POLL_FLAG_LINKED_LIST);
}

static unsigned uct_dc_mlx5_iface_progress_ll_zip(void *arg)
{
    return uct_dc_mlx5_iface_progress(arg, UCT_IB_MLX5_POLL_FLAG_LINKED_LIST |
                                           UCT_IB_MLX5_POLL_FLAG_CQE_ZIP);
}

static unsigned uct_dc_mlx5_iface_progress_tm(void *arg)
{
    return uct_dc_mlx5_iface_progress(arg, UCT_IB_MLX5_POLL_FLAG_TM |
                                           UCT_IB_MLX5_POLL_FLAG_CQE_ZIP);
}

static void UCS_CLASS_DELETE_FUNC_NAME(uct_dc_mlx5_iface_t)(uct_iface_t*);

static void uct_ib_mlx5_dci_qp_update_attr(uct_ib_qp_init_attr_t *qp_attr)
{
    /* DCI doesn't support receiving data, and set minimal possible values for
     * max_send_sge and max_inline_data to minimize WQE length */
    qp_attr->cap.max_recv_sge    = 0;
    qp_attr->cap.max_send_sge    = 1;
    qp_attr->cap.max_inline_data = 0;
}

static void uct_ib_mlx5dv_dci_qp_init_attr(uct_ib_qp_init_attr_t *qp_attr,
                                           struct mlx5dv_qp_init_attr *dv_attr)
{
    uct_ib_mlx5_dci_qp_update_attr(qp_attr);
    uct_ib_mlx5dv_dc_qp_init_attr(dv_attr, MLX5DV_DCTYPE_DCI);
}

static ucs_status_t uct_dc_mlx5_iface_create_dci(uct_dc_mlx5_iface_t *iface,
                                                 uint8_t pool_index,
                                                 uint8_t dci_index,
                                                 uint8_t path_index,
                                                 int full_handshake)
{
    uct_ib_iface_t *ib_iface           = &iface->super.super.super;
    uct_ib_mlx5_qp_attr_t attr         = {};
    ucs_status_t status;
    uct_ib_mlx5_md_t *md               = ucs_derived_of(ib_iface->super.md,
                                                        uct_ib_mlx5_md_t);
    uct_dc_dci_t *dci                  = &iface->tx.dcis[dci_index];
#if HAVE_DC_DV
    uct_ib_device_t *dev               = uct_ib_iface_device(ib_iface);
    struct mlx5dv_qp_init_attr dv_attr = {};
    struct ibv_qp *qp;

    ucs_assert(iface->super.super.super.config.qp_type == UCT_IB_QPT_DCI);

    uct_rc_mlx5_iface_fill_attr(&iface->super, &attr,
                                iface->super.super.config.tx_qp_len,
                                &iface->super.rx.srq);

    if (md->flags & UCT_IB_MLX5_MD_FLAG_DEVX_DCI) {
        attr.super.max_inl_cqe[UCT_IB_DIR_RX] = 0;
        attr.uidx                        = htonl(dci_index) >> UCT_IB_UIDX_SHIFT;
        attr.full_handshake              = full_handshake;
        attr.rdma_wr_disabled            = (iface->flags & UCT_DC_MLX5_IFACE_FLAG_DISABLE_PUT) &&
                                           (md->flags & UCT_IB_MLX5_MD_FLAG_NO_RDMA_WR_OPTIMIZED);
        attr.log_num_dci_stream_channels = ucs_ilog2(iface->tx.num_dci_channels);
        status = uct_ib_mlx5_devx_create_qp(ib_iface,
                                            &iface->super.cq[UCT_IB_DIR_TX],
                                            &iface->super.cq[UCT_IB_DIR_RX],
                                            &dci->txwq.super, &dci->txwq,
                                            &attr);
        if (status != UCS_OK) {
            return status;
        }

        ucs_debug("created DevX DCI 0x%x, rdma_wr_disabled=%d", dci->txwq.super.qp_num,
                  attr.rdma_wr_disabled);
        goto init_qp;
    }

    if (iface->super.cq[UCT_IB_DIR_TX].type != UCT_IB_MLX5_OBJ_TYPE_VERBS) {
        ucs_error("cannot create verbs DCI with DEVX CQ");
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_ib_mlx5_iface_get_res_domain(ib_iface, &dci->txwq.super);
    if (status != UCS_OK) {
        return status;
    }

    uct_ib_mlx5_iface_fill_attr(ib_iface, &dci->txwq.super, &attr);
    uct_ib_iface_fill_attr(ib_iface, &attr.super);
    uct_ib_mlx5dv_dci_qp_init_attr(&attr.super.ibv, &dv_attr);
    uct_rc_mlx5_common_fill_dv_qp_attr(&iface->super, &attr.super.ibv, &dv_attr,
                                       UCS_BIT(UCT_IB_DIR_TX));
    qp = UCS_PROFILE_CALL_ALWAYS(mlx5dv_create_qp, dev->ibv_context,
                                 &attr.super.ibv, &dv_attr);
    if (qp == NULL) {
        uct_ib_check_memlock_limit_msg(UCS_LOG_LEVEL_ERROR,
                                       "%s: mlx5dv_create_qp("UCT_IB_IFACE_FMT", DCI)",
                                       uct_ib_device_name(dev),
                                       UCT_IB_IFACE_ARG(ib_iface));
        status = UCS_ERR_IO_ERROR;
        goto err_put_res_domain;
    }

    dci->txwq.super.verbs.qp = qp;
    dci->txwq.super.qp_num = dci->txwq.super.verbs.qp->qp_num;

init_qp:
#else
    uct_rc_mlx5_iface_fill_attr(&iface->super, &attr,
                                iface->super.super.config.tx_qp_len,
                                &iface->super.rx.srq);
    status = uct_ib_mlx5_iface_create_qp(ib_iface, &dci->txwq.super, &attr);
    if (status != UCS_OK) {
        return status;
    }
#endif
    status = uct_rc_txqp_init(&dci->txqp, &iface->super.super,
                              dci->txwq.super.qp_num
                              UCS_STATS_ARG(iface->super.super.stats));
    if (status != UCS_OK) {
        goto err_qp;
    }

    dci->pool_index = pool_index;
    dci->path_index = path_index;

    status = uct_dc_mlx5_iface_dci_connect(iface, dci);
    if (status != UCS_OK) {
        goto err;
    }

    if (uct_dc_mlx5_iface_is_dci_shared(iface)) {
        ucs_arbiter_group_init(&dci->arb_group);
    } else {
        dci->ep = NULL;
    }

    if (dci->txwq.super.type == UCT_IB_MLX5_OBJ_TYPE_VERBS) {
        status = uct_ib_mlx5_txwq_init(iface->super.super.super.super.worker,
                                       iface->super.tx.mmio_mode, &dci->txwq,
                                       dci->txwq.super.verbs.qp);
        if (status != UCS_OK) {
            goto err;
        }
    }

    uct_rc_txqp_available_set(&dci->txqp, dci->txwq.bb_max);

    return UCS_OK;

err:
    uct_rc_txqp_cleanup(&iface->super.super, &dci->txqp);
err_qp:
    uct_ib_mlx5_destroy_qp(md, &dci->txwq.super);
#if HAVE_DC_DV
err_put_res_domain:
    if (!(md->flags & UCT_IB_MLX5_MD_FLAG_DEVX_DCI)) {
        uct_ib_mlx5_iface_put_res_domain(&dci->txwq.super);
    }
#endif
    return status;
}

#if HAVE_DC_DV
ucs_status_t uct_dc_mlx5_iface_dci_connect(uct_dc_mlx5_iface_t *iface,
                                           uct_dc_dci_t *dci)
{
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.super.super.super.md,
                                          uct_ib_mlx5_md_t);
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super.super.super);
    struct ibv_qp_attr attr;
    long attr_mask;
    ucs_status_t status;

    if (md->flags & UCT_IB_MLX5_MD_FLAG_DEVX) {
        return uct_dc_mlx5_iface_devx_dci_connect(iface, &dci->txwq.super,
                                                  dci->path_index);
    }

    ucs_assert(dci->txwq.super.type == UCT_IB_MLX5_OBJ_TYPE_VERBS);
    memset(&attr, 0, sizeof(attr));
    attr.qp_state        = IBV_QPS_INIT;
    attr.pkey_index      = iface->super.super.super.pkey_index;
    attr.port_num        = iface->super.super.super.config.port_num;
    attr_mask            = IBV_QP_STATE      |
                           IBV_QP_PKEY_INDEX |
                           IBV_QP_PORT;

    if (ibv_modify_qp(dci->txwq.super.verbs.qp, &attr, attr_mask)) {
        ucs_error("ibv_modify_qp(DCI, INIT) failed : %m");
        return UCS_ERR_IO_ERROR;
    }

    status = uct_ib_device_set_ece(dev, dci->txwq.super.verbs.qp,
                                   iface->super.super.config.ece);
    if (status != UCS_OK) {
        return status;
    }

    /* Move QP to the RTR state */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state                   = IBV_QPS_RTR;
    attr.path_mtu                   = iface->super.super.super.config.path_mtu;
    attr.ah_attr.is_global          = iface->super.super.super.config.force_global_addr;
    attr.ah_attr.sl                 = iface->super.super.super.config.sl;
    /* ib_core expects valid ah_attr::port_num when IBV_QP_AV is set */
    attr.ah_attr.port_num           = iface->super.super.super.config.port_num;
    attr_mask                       = IBV_QP_STATE     |
                                      IBV_QP_PATH_MTU  |
                                      IBV_QP_AV;

    if (ibv_modify_qp(dci->txwq.super.verbs.qp, &attr, attr_mask)) {
        ucs_error("ibv_modify_qp(DCI, RTR) failed : %m");
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

    if (ibv_modify_qp(dci->txwq.super.verbs.qp, &attr, attr_mask)) {
        ucs_error("ibv_modify_qp(DCI, RTS) failed : %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t
uct_dc_mlx5_iface_create_dct(uct_dc_mlx5_iface_t *iface,
                             const uct_dc_mlx5_iface_config_t *config)
{
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.super.super.super.md,
                                          uct_ib_mlx5_md_t);
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super.super.super);
    struct mlx5dv_qp_init_attr dv_init_attr = {};
    uct_ib_qp_init_attr_t init_attr         = {};
    struct ibv_qp_attr attr                 = {};
    ucs_status_t status;
    int ret;

    if (md->flags & UCT_IB_MLX5_MD_FLAG_DEVX_DCT) {
        return uct_dc_mlx5_iface_devx_create_dct(iface);
    }

    if (iface->super.cq[UCT_IB_DIR_RX].type != UCT_IB_MLX5_OBJ_TYPE_VERBS) {
        ucs_error("cannot create verbs DCT with DEVX CQ");
        return UCS_ERR_INVALID_PARAM;
    }

    uct_ib_mlx5dv_dct_qp_init_attr(&init_attr, &dv_init_attr, md->super.pd,
                                   iface->super.super.super.cq[UCT_IB_DIR_RX],
                                   iface->super.rx.srq.verbs.srq);
    uct_rc_mlx5_common_fill_dv_qp_attr(&iface->super, &init_attr, &dv_init_attr,
                                       UCS_BIT(UCT_IB_DIR_RX));

    iface->rx.dct.verbs.qp = mlx5dv_create_qp(dev->ibv_context, &init_attr,
                                              &dv_init_attr);
    if (iface->rx.dct.verbs.qp == NULL) {
        uct_ib_check_memlock_limit_msg(UCS_LOG_LEVEL_ERROR,
                                       "%s: mlx5dv_create_qp(DCT)",
                                       uct_ib_device_name(dev));
        return UCS_ERR_INVALID_PARAM;
    }

    attr.pkey_index      = iface->super.super.super.pkey_index;
    attr.qp_state        = IBV_QPS_INIT;
    attr.port_num        = iface->super.super.super.config.port_num;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ  |
                           IBV_ACCESS_REMOTE_ATOMIC;

    ret = ibv_modify_qp(iface->rx.dct.verbs.qp, &attr, IBV_QP_STATE |
                                                       IBV_QP_PKEY_INDEX |
                                                       IBV_QP_PORT |
                                                       IBV_QP_ACCESS_FLAGS);

    if (ret) {
         ucs_error("error modifying DCT to INIT: %m");
         goto err;
    }

    status = uct_ib_device_set_ece(dev, iface->rx.dct.verbs.qp,
                                   iface->super.super.config.ece);
    if (status != UCS_OK) {
        goto err;
    }

    attr.qp_state                  = IBV_QPS_RTR;
    attr.path_mtu                  = iface->super.super.super.config.path_mtu;
    attr.min_rnr_timer             = iface->super.super.config.min_rnr_timer;
    attr.ah_attr.is_global         = iface->super.super.super.config.force_global_addr;
    attr.ah_attr.grh.hop_limit     = iface->super.super.super.config.hop_limit;
    attr.ah_attr.grh.traffic_class = iface->super.super.super.config.traffic_class;
    attr.ah_attr.grh.sgid_index    = iface->super.super.super.gid_info.gid_index;
    attr.ah_attr.port_num          = iface->super.super.super.config.port_num;

    ret = ibv_modify_qp(iface->rx.dct.verbs.qp, &attr, IBV_QP_STATE |
                                                       IBV_QP_MIN_RNR_TIMER |
                                                       IBV_QP_AV |
                                                       IBV_QP_PATH_MTU);
    if (ret) {
         ucs_error("error modifying DCT to RTR: %m");
         goto err;
    }

    iface->rx.dct.type   = UCT_IB_MLX5_OBJ_TYPE_VERBS;
    iface->rx.dct.qp_num = iface->rx.dct.verbs.qp->qp_num;
    return UCS_OK;

err:
    uct_ib_destroy_qp(iface->rx.dct.verbs.qp);
    return UCS_ERR_IO_ERROR;
}

void uct_dc_mlx5_destroy_dct(uct_dc_mlx5_iface_t *iface)
{
    switch (iface->rx.dct.type) {
    case UCT_IB_MLX5_OBJ_TYPE_VERBS:
        uct_ib_destroy_qp(iface->rx.dct.verbs.qp);
        break;
    case UCT_IB_MLX5_OBJ_TYPE_DEVX:
#if HAVE_DEVX
        uct_ib_mlx5_devx_obj_destroy(iface->rx.dct.devx.obj, "DCT");
#endif
        break;
    case UCT_IB_MLX5_OBJ_TYPE_LAST:
        break;
    }
}
#endif

static ucs_status_t
uct_dc_mlx5_init_rx(uct_rc_iface_t *rc_iface,
                    const uct_rc_iface_common_config_t *rc_config)
{
    uct_ib_mlx5_md_t *md                 = ucs_derived_of(rc_iface->super.super.md,
                                                          uct_ib_mlx5_md_t);
    uct_dc_mlx5_iface_config_t *config   = ucs_derived_of(rc_config,
                                                          uct_dc_mlx5_iface_config_t);
    uct_dc_mlx5_iface_t *iface           = ucs_derived_of(rc_iface,
                                                          uct_dc_mlx5_iface_t);
    struct ibv_srq_init_attr_ex srq_attr = {};
    ucs_status_t status;

    if (UCT_RC_MLX5_TM_ENABLED(&iface->super)) {
        if (md->flags & UCT_IB_MLX5_MD_FLAG_DEVX_DC_SRQ) {
            status = uct_rc_mlx5_devx_init_rx_tm(&iface->super, &config->super,
                                                 1, UCT_DC_RNDV_HDR_LEN);
            if (status != UCS_OK) {
                goto err;
            }

            status = uct_dc_mlx5_iface_devx_set_srq_dc_params(iface);
            if (status != UCS_OK) {
                goto err_free_srq;
            }
        } else {
            status = uct_rc_mlx5_init_rx_tm(&iface->super, &config->super,
                                            &srq_attr, UCT_DC_RNDV_HDR_LEN);
            if (status != UCS_OK) {
                goto err;
            }
        }

        iface->super.super.progress = uct_dc_mlx5_iface_progress_tm;
        return status;
    }

    /* MP XRQ is supported with HW TM only */
    ucs_assert(!UCT_RC_MLX5_MP_ENABLED(&iface->super));

    if (ucs_test_all_flags(md->flags, UCT_IB_MLX5_MD_FLAG_RMP |
                                      UCT_IB_MLX5_MD_FLAG_DEVX_DC_SRQ)) {
        status = uct_rc_mlx5_devx_init_rx(&iface->super, &config->super);
    } else {
        status = uct_rc_mlx5_common_iface_init_rx(&iface->super, rc_config);
    }

    if (status != UCS_OK) {
        goto err;
    }

    if (iface->super.config.srq_topo == UCT_RC_MLX5_SRQ_TOPO_LIST) {
        if (iface->super.cq[UCT_IB_DIR_RX].zip ||
            iface->super.cq[UCT_IB_DIR_TX].zip) {
            iface->super.super.progress = uct_dc_mlx5_iface_progress_ll_zip;
        } else {
            iface->super.super.progress = uct_dc_mlx5_iface_progress_ll;
        }
    } else {
        iface->super.super.progress = uct_dc_mlx5_iface_progress_cyclic;
    }
    return UCS_OK;

err_free_srq:
    uct_rc_mlx5_destroy_srq(md, &iface->super.rx.srq);
err:
    return status;
}

void uct_dc_mlx5_cleanup_rx(uct_rc_iface_t *rc_iface)
{
    uct_ib_mlx5_md_t *md       = ucs_derived_of(rc_iface->super.super.md,
                                                uct_ib_mlx5_md_t);
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(rc_iface, uct_dc_mlx5_iface_t);

    uct_rc_mlx5_destroy_srq(md, &iface->super.rx.srq);
}

static void uct_dc_mlx5_iface_dci_pool_destroy(uct_dc_mlx5_dci_pool_t *dci_pool)
{
    ucs_arbiter_cleanup(&dci_pool->arbiter);
    ucs_free(dci_pool->stack);
}

static void uct_dc_mlx5_iface_dcis_destroy(uct_dc_mlx5_iface_t *iface,
                                           uint8_t num_dci_pools,
                                           uint8_t num_dcis)
{
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.super.super.super.md,
                                          uct_ib_mlx5_md_t);
    uint8_t pool_index, dci_index;

    ucs_assert(num_dci_pools <= iface->tx.num_dci_pools);
    ucs_assert(num_dcis <= uct_dc_mlx5_iface_total_ndci(iface));

    for (dci_index = 0; dci_index < num_dcis; dci_index++) {
        uct_rc_txqp_cleanup(&iface->super.super,
                            &iface->tx.dcis[dci_index].txqp);
        uct_ib_mlx5_destroy_qp(md, &iface->tx.dcis[dci_index].txwq.super);

        if (uct_dc_mlx5_iface_is_dci_shared(iface)) {
            ucs_arbiter_group_cleanup(&iface->tx.dcis[dci_index].arb_group);
        }
        uct_ib_mlx5_qp_mmio_cleanup(&iface->tx.dcis[dci_index].txwq.super,
                                    iface->tx.dcis[dci_index].txwq.reg);
    }

    for (pool_index = 0; pool_index < num_dci_pools; pool_index++) {
        uct_dc_mlx5_iface_dci_pool_destroy(&iface->tx.dci_pool[pool_index]);
    }

    ucs_free(iface->tx.dcis);
}

static ucs_status_t
uct_dc_mlx5_iface_dcis_create(uct_dc_mlx5_iface_t *iface,
                              const uct_dc_mlx5_iface_config_t *config)
{
    const uint8_t num_paths = iface->super.super.super.num_paths;
    uint8_t dci_index       = 0;
    uct_dc_mlx5_dci_pool_t *dci_pool;
    uint8_t pool_index, i;
    ucs_status_t status;

    iface->tx.dcis = ucs_calloc((iface->tx.ndci * iface->tx.num_dci_pools) +
                                UCT_DC_MLX5_KEEPALIVE_NUM_DCIS,
                                sizeof(*iface->tx.dcis),
                                "dcis");
    if (iface->tx.dcis == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    for (pool_index = 0; pool_index < iface->tx.num_dci_pools; pool_index++) {
        ucs_debug("creating dci pool %u with %u QPs", pool_index, iface->tx.ndci);

        dci_pool        = &iface->tx.dci_pool[pool_index];
        dci_pool->stack = ucs_calloc(iface->tx.ndci, sizeof(*dci_pool->stack),
                                     "dci pool stack");
        if (dci_pool->stack == NULL) {
            status = UCS_ERR_NO_MEMORY;
            goto err_dcis_destroy;
        }

        ucs_arbiter_init(&dci_pool->arbiter);
        dci_pool->stack_top         = 0;
        dci_pool->release_stack_top = -1;

        for (i = 0; i < iface->tx.ndci; ++i) {
            status = uct_dc_mlx5_iface_create_dci(
                    iface, pool_index, dci_index, pool_index % num_paths,
                    iface->flags & UCT_DC_MLX5_IFACE_FLAG_DCI_FULL_HANDSHAKE);
            if (status != UCS_OK) {
                goto err_dci_pool_destroy;
            }

            dci_pool->stack[i] = dci_index;
            ++dci_index;
        }
    }

    iface->tx.bb_max = iface->tx.dcis[0].txwq.bb_max;
    return UCS_OK;

err_dci_pool_destroy:
    uct_dc_mlx5_iface_dci_pool_destroy(dci_pool);
err_dcis_destroy:
    uct_dc_mlx5_iface_dcis_destroy(iface, pool_index, dci_index);
err:
    return status;
}

void uct_dc_mlx5_iface_set_quota(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_iface_config_t *config)
{
    iface->tx.available_quota = iface->tx.bb_max - ucs_min(iface->tx.bb_max,
                                                           config->quota);
}

static ucs_status_t uct_dc_mlx5_iface_estimate_perf(uct_iface_h tl_iface,
                                                    uct_perf_attr_t *perf_attr)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    ucs_status_t status;

    status = uct_ib_iface_estimate_perf(tl_iface, perf_attr);
    if (status != UCS_OK) {
        return status;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_MAX_INFLIGHT_EPS) {
        perf_attr->max_inflight_eps = iface->tx.ndci;
    }

    return UCS_OK;
}

static void uct_dc_mlx5_iface_vfs_refresh(uct_iface_h tl_iface)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_dci_pool_t *dci_pool;
    int i, pool_index, dci_index;
    uct_dc_dci_t *dci;

    /* Add iface resources */
    uct_rc_iface_vfs_populate(&iface->super.super);

    /* Add objects for DCIs */
    dci_index = 0;
    for (pool_index = 0; pool_index < iface->tx.num_dci_pools; pool_index++) {
        dci_pool = &iface->tx.dci_pool[pool_index];
        ucs_vfs_obj_add_dir(iface, dci_pool, "dci_pool/%d", pool_index);
        for (i = 0; i < iface->tx.ndci; ++i) {
            dci = &iface->tx.dcis[dci_index];
            ucs_vfs_obj_add_dir(dci_pool, dci, "%d", dci_index);
            uct_ib_mlx5_txwq_vfs_populate(&dci->txwq, dci);
            uct_rc_txqp_vfs_populate(&dci->txqp, dci);
            ++dci_index;
        }
    }

    /* Add objects for DCT */
    ucs_vfs_obj_add_dir(iface, &iface->rx.dct, "dct");
    ucs_vfs_obj_add_ro_file(&iface->rx.dct, ucs_vfs_show_primitive,
                            &iface->rx.dct.qp_num, UCS_VFS_TYPE_U32_HEX,
                            "qp_num");
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
    uct_dc_mlx5_iface_t *iface;

    iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    ucs_assert_always(iface_addr != NULL);

    return ((addr->flags & UCT_DC_MLX5_IFACE_ADDR_DC_VERS) ==
            iface->version_flag) &&
           (UCT_DC_MLX5_IFACE_ADDR_TM_ENABLED(addr) ==
            UCT_RC_MLX5_TM_ENABLED(&iface->super)) &&
           uct_ib_iface_is_reachable(tl_iface, dev_addr, iface_addr);
}

ucs_status_t
uct_dc_mlx5_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    uct_dc_mlx5_iface_t *iface           = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_iface_flush_addr_t *addr = (uct_dc_mlx5_iface_flush_addr_t *)iface_addr;
    uct_ib_md_t *md                      = ucs_derived_of(iface->super.super.super.super.md,
                                                          uct_ib_md_t);

    uct_ib_pack_uint24(addr->super.qp_num, iface->rx.dct.qp_num);
    uct_ib_mlx5_md_get_atomic_mr_id(md, &addr->super.atomic_mr_id);
    addr->super.flags = iface->version_flag;
    if (UCT_RC_MLX5_TM_ENABLED(&iface->super)) {
        addr->super.flags |= UCT_DC_MLX5_IFACE_ADDR_HW_TM;
    }

    if (iface->super.super.config.flush_remote) {
        addr->flush_rkey_hi = md->flush_rkey >> 16;
        addr->super.flags  |= UCT_DC_MLX5_IFACE_ADDR_FLUSH_RKEY;
    }

    return UCS_OK;
}

static inline ucs_status_t uct_dc_mlx5_iface_flush_dcis(uct_dc_mlx5_iface_t *iface)
{
    int i;

    if (kh_size(&iface->tx.fc_hash) != 0) {
        /* If some ep is waiting for grant it may have some pending
         * operations, while all QP resources are available. */
        return UCS_INPROGRESS;
    }

    for (i = 0; i < iface->tx.ndci * iface->tx.num_dci_pools; i++) {
        if (uct_dc_mlx5_iface_flush_dci(iface, i) != UCS_OK) {
            return UCS_INPROGRESS;
        }
    }

    return UCS_OK;
}

ucs_status_t uct_dc_mlx5_iface_flush(uct_iface_h tl_iface, unsigned flags, uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    ucs_status_t status;

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = uct_rc_iface_fence_relaxed_order(tl_iface);
    if (status != UCS_OK) {
        return status;
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

    ep->flags = 0;
    status    = uct_dc_mlx5_ep_basic_init(iface, ep);
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

static void uct_dc_mlx5_iface_cleanup_fc_ep(uct_dc_mlx5_iface_t *iface)
{
    uct_dc_mlx5_ep_t *fc_ep = iface->tx.fc_ep;
    uct_rc_iface_send_op_t *op;
    ucs_queue_iter_t iter;
    uct_rc_txqp_t *txqp;

    uct_dc_mlx5_ep_pending_purge(&fc_ep->super.super, NULL, NULL);
    ucs_arbiter_group_cleanup(&fc_ep->arb_group);
    uct_rc_fc_cleanup(&fc_ep->fc);

    if (uct_dc_mlx5_iface_is_dci_shared(iface)) {
        txqp = &iface->tx.dcis[fc_ep->dci].txqp;
        ucs_queue_for_each_safe(op, iter, &txqp->outstanding, queue) {
            if (op->handler == uct_dc_mlx5_ep_fc_pure_grant_send_completion) {
                ucs_queue_del_iter(&txqp->outstanding, iter);
                op->handler(op, NULL);
            }
        }
    } else if (fc_ep->dci != UCT_DC_MLX5_EP_NO_DCI) {
        /* All outstanding operations on this DCI are FC_PURE_GRANT packets */
        txqp = &iface->tx.dcis[fc_ep->dci].txqp;
        uct_rc_txqp_purge_outstanding(&iface->super.super, txqp,
                                      /* complete with OK to avoid re-sending */
                                      UCS_OK,
                                      iface->tx.dcis[fc_ep->dci].txwq.sw_pi, 0);
    }

    UCS_CLASS_CLEANUP(uct_base_ep_t, fc_ep);
    ucs_free(fc_ep);
}

ucs_status_t uct_dc_mlx5_iface_fc_grant(uct_pending_req_t *self)
{
    uct_dc_fc_request_t *fc_req = ucs_derived_of(self, uct_dc_fc_request_t);
    uct_dc_mlx5_ep_t *ep        = ucs_derived_of(fc_req->super.ep,
                                                 uct_dc_mlx5_ep_t);
    uct_rc_iface_t *rc_iface    = ucs_derived_of(ep->super.super.iface,
                                                 uct_rc_iface_t);
    uct_rc_iface_send_op_t *send_op;
    ucs_status_t status;

    ucs_assert_always(rc_iface->config.fc_enabled);

    send_op = ucs_mpool_get(&rc_iface->tx.send_op_mp);
    if (ucs_unlikely(send_op == NULL)) {
        ucs_error("failed to allocate FC_PURE_GRANT op");
        return UCS_ERR_NO_MEMORY;
    }

    uct_rc_ep_init_send_op(send_op, 0, NULL,
                           uct_dc_mlx5_ep_fc_pure_grant_send_completion);
    uct_rc_iface_send_op_set_name(send_op, "dc_mlx5_iface_fc_grant");

    send_op->buffer = fc_req;
    status          = uct_dc_mlx5_ep_fc_pure_grant_send(ep, send_op);
    if (status != UCS_OK) {
        ucs_mpool_put(send_op);
    }

    return status;
}

const char *
uct_dc_mlx5_fc_req_str(uct_dc_fc_request_t *fc_req, char *buf, size_t max)
{
    char gid_str[32];

    ucs_snprintf_zero(buf, max,
                      "FC_PURE_GRANT seq %" PRIu64 " dct_num 0x%x"
                      " lid %d gid %s",
                      fc_req->sender.payload.seq, fc_req->dct_num, fc_req->lid,
                      uct_ib_gid_str(ucs_unaligned_ptr(
                                             &fc_req->sender.payload.gid),
                                     gid_str, sizeof(gid_str)));
    return buf;
}

void uct_dc_mlx5_fc_entry_iter_del(uct_dc_mlx5_iface_t *iface, khiter_t it)
{
    kh_del(uct_dc_mlx5_fc_hash, &iface->tx.fc_hash, it);
    if (kh_size(&iface->tx.fc_hash) == 0) {
        uct_worker_progress_unregister_safe(
                &iface->super.super.super.super.worker->super,
                &iface->tx.fc_hard_req_progress_cb_id);
    }
}

static UCS_F_ALWAYS_INLINE int
uct_dc_mlx5_fc_remove_ep(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep,
                         uint64_t seq)
{
    khiter_t it = kh_get(uct_dc_mlx5_fc_hash, &iface->tx.fc_hash,
                         (uint64_t)ep);
    if ((it != kh_end(&iface->tx.fc_hash)) &&
        ((seq == UINT64_MAX) ||
         (kh_value(&iface->tx.fc_hash, it).seq == seq))) {
        uct_dc_mlx5_fc_entry_iter_del(iface, it);
        return 1;
    }

    return 0;
}

static ucs_status_t
uct_dc_mlx5_iface_fc_handler(uct_rc_iface_t *rc_iface, unsigned qp_num,
                             uct_rc_hdr_t *hdr, unsigned length,
                             uint32_t imm_data, uint16_t lid, unsigned flags)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(rc_iface, uct_dc_mlx5_iface_t);
    uint8_t fc_hdr             = uct_rc_fc_get_fc_hdr(hdr->am_id);
    uct_dc_fc_sender_data_t *sender;
    uct_dc_fc_request_t *dc_req;
    int16_t cur_wnd;
    ucs_status_t status;
    uct_dc_mlx5_ep_t *ep;
    ucs_arbiter_t *waitq;
    ucs_arbiter_group_t *group;
    uint8_t pool_index;
    char buf[128];

    ucs_assert(rc_iface->config.fc_enabled);

    if (fc_hdr == UCT_RC_EP_FLAG_FC_HARD_REQ) {
        ep = iface->tx.fc_ep;
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_RX_HARD_REQ, 1);

        dc_req = ucs_mpool_get(&iface->super.super.tx.pending_mp);
        if (ucs_unlikely(dc_req == NULL)) {
            ucs_error("fc_ep=%p: failed to allocate FC request", ep);
            return UCS_ERR_NO_MEMORY;
        }

        dc_req->super.super.func = uct_dc_mlx5_iface_fc_grant;
        dc_req->super.ep         = &ep->super.super;
        dc_req->dct_num          = imm_data;
        dc_req->lid              = lid;
        dc_req->sender           = *((uct_dc_fc_sender_data_t*)(hdr + 1));

        status = uct_dc_mlx5_iface_fc_grant(&dc_req->super.super);
        if (status == UCS_ERR_NO_RESOURCE){
            uct_dc_mlx5_ep_do_pending_fc(ep, dc_req);
        } else if (status != UCS_OK) {
            ucs_diag("fc_ep %p: failed to send %s: %s", ep,
                     uct_dc_mlx5_fc_req_str(dc_req, buf, sizeof(buf)),
                     ucs_status_string(status));
        }
    } else if (fc_hdr == UCT_RC_EP_FC_PURE_GRANT) {
        sender = (uct_dc_fc_sender_data_t*)(hdr + 1);
        ep     = (uct_dc_mlx5_ep_t*)sender->ep;
        if (!uct_dc_mlx5_fc_remove_ep(iface, ep, sender->payload.seq)) {
            /* Don't process FC_PURE_GRANT further, if an endpoint doesn't
             * exist anymore, unexpected grant was received on an endpoint or
             * this is not the same grant sequence number which was expected */
            return UCS_OK;
        }

        cur_wnd = ep->fc.fc_wnd;
        uct_rc_fc_restore_wnd(&iface->super.super, &ep->fc);
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_RX_PURE_GRANT,
                                 1);

        /* To preserve ordering we have to dispatch all pending operations if
         * current fc_wnd is <= 0 */
        if (cur_wnd == 0) {
            uct_dc_mlx5_get_arbiter_params(iface, ep, &waitq, &group,
                                           &pool_index);
            ucs_arbiter_group_schedule(waitq, group);
            uct_dc_mlx5_iface_progress_pending(iface, pool_index);
            uct_dc_mlx5_iface_check_tx(iface);
        }
    }

    return UCS_OK;
}

static void uct_dc_mlx5_dci_handle_failure(uct_dc_mlx5_iface_t *iface,
                                           struct mlx5_cqe64 *cqe,
                                           uint8_t dci_index,
                                           ucs_status_t status)
{
    uct_dc_mlx5_ep_t *ep;
    ucs_log_level_t  level;

    if (uct_dc_mlx5_iface_is_dci_shared(iface)) {
        ep    = NULL;
        level = UCS_LOG_LEVEL_FATAL; /* error handling is not supported with rand dci */
    } else {
        ep    = uct_dc_mlx5_ep_from_dci(iface, dci_index);
        level = iface->super.super.super.super.config.failure_level;
    }

    if (ep == NULL) {
        uct_ib_mlx5_completion_with_err(&iface->super.super.super,
                                        (uct_ib_mlx5_err_cqe_t*)cqe,
                                        &iface->tx.dcis[dci_index].txwq, level);
        return;
    }

    uct_dc_mlx5_ep_handle_failure(ep, cqe, status);
}

static void uct_dc_mlx5_iface_handle_failure(uct_ib_iface_t *ib_iface,
                                             void *arg, ucs_status_t status)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ib_iface, uct_dc_mlx5_iface_t);
    struct mlx5_cqe64 *cqe     = arg;
    uint8_t dci_index          = uct_dc_mlx5_iface_dci_find(iface, cqe);
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_DC_MLX5_IFACE_TXQP_DCI_GET(iface, dci_index, txqp, txwq);
    uct_ib_mlx5_txwq_update_flags(txwq, UCT_IB_MLX5_TXWQ_FLAG_FAILED, 0);

    if (uct_dc_mlx5_iface_is_dci_keepalive(iface, dci_index)) {
        uct_dc_mlx5_dci_keepalive_handle_failure(iface, cqe, dci_index, status);
    } else {
        uct_dc_mlx5_dci_handle_failure(iface, cqe, dci_index, status);
    }
}

static uct_rc_iface_ops_t uct_dc_mlx5_iface_ops = {
    .super = {
        .super = {
            .iface_estimate_perf   = uct_dc_mlx5_iface_estimate_perf,
            .iface_vfs_refresh     = uct_dc_mlx5_iface_vfs_refresh,
            .ep_query              = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
            .ep_invalidate         = uct_dc_mlx5_ep_invalidate,
            .ep_connect_to_ep_v2   = ucs_empty_function_return_unsupported,
            .iface_is_reachable_v2 = uct_ib_iface_is_reachable_v2
        },
        .create_cq      = uct_rc_mlx5_iface_common_create_cq,
        .destroy_cq     = uct_rc_mlx5_iface_common_destroy_cq,
        .event_cq       = uct_rc_mlx5_iface_common_event_cq,
        .handle_failure = uct_dc_mlx5_iface_handle_failure,
    },
    .init_rx    = uct_dc_mlx5_init_rx,
    .cleanup_rx = uct_dc_mlx5_cleanup_rx,
    .fc_ctrl    = (uct_rc_iface_fc_ctrl_func_t)ucs_empty_function_do_assert,
    .fc_handler = uct_dc_mlx5_iface_fc_handler,
};

static uct_iface_ops_t uct_dc_mlx5_iface_tl_ops = {
    .ep_put_short             = uct_dc_mlx5_ep_put_short,
    .ep_put_bcopy             = uct_dc_mlx5_ep_put_bcopy,
    .ep_put_zcopy             = uct_dc_mlx5_ep_put_zcopy,
    .ep_get_bcopy             = uct_dc_mlx5_ep_get_bcopy,
    .ep_get_zcopy             = uct_dc_mlx5_ep_get_zcopy,
    .ep_am_short              = uct_dc_mlx5_ep_am_short,
    .ep_am_short_iov          = uct_dc_mlx5_ep_am_short_iov,
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
    .ep_check                 = uct_dc_mlx5_ep_check,
#if IBV_HW_TM
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
    .iface_fence              = uct_rc_iface_fence,
    .iface_progress_enable    = uct_dc_mlx5_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_rc_iface_do_progress,
    .iface_event_fd_get       = uct_rc_mlx5_iface_event_fd_get,
    .iface_event_arm          = uct_rc_mlx5_iface_devx_arm,
    .ep_create                = uct_dc_mlx5_ep_create_connected,
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_dc_mlx5_ep_t),
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_dc_mlx5_iface_t),
    .iface_query              = uct_dc_mlx5_iface_query,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_dc_mlx5_iface_is_reachable,
    .iface_get_address        = uct_dc_mlx5_iface_get_address,
};

static ucs_status_t uct_dc_mlx5dv_calc_tx_wqe_ratio(uct_ib_mlx5_md_t *md)
{
    uct_ib_device_t *dev               = &md->super.dev;
    uct_ib_qp_init_attr_t qp_init_attr = {};
    struct mlx5dv_qp_init_attr dv_attr = {};
    struct ibv_qp *dci_qp;
    ucs_status_t status;
    uct_ib_mlx5dv_qp_tmp_objs_t qp_tmp_objs;

    if (md->dv_tx_wqe_ratio.dc != 0) {
        return UCS_OK;
    }

    status = uct_ib_mlx5dv_qp_tmp_objs_create(dev, md->super.pd, &qp_tmp_objs, 0);
    if (status != UCS_OK) {
        goto out;
    }

    uct_ib_mlx5dv_qp_init_attr(&qp_init_attr, md->super.pd, &qp_tmp_objs,
                               IBV_QPT_DRIVER, 0);
    uct_ib_mlx5dv_dci_qp_init_attr(&qp_init_attr, &dv_attr);

    dci_qp = UCS_PROFILE_CALL_ALWAYS(mlx5dv_create_qp, dev->ibv_context,
                                     &qp_init_attr, &dv_attr);
    if (dci_qp == NULL) {
        uct_ib_check_memlock_limit_msg(UCS_LOG_LEVEL_ERROR,
                                       "%s: mlx5dv_create_qp(DCI)",
                                       uct_ib_device_name(dev));
        status = UCS_ERR_IO_ERROR;
        goto out_qp_tmp_objs_close;
    }

    status = uct_ib_mlx5dv_calc_tx_wqe_ratio(dci_qp,
                                             qp_init_attr.cap.max_send_wr,
                                             &md->dv_tx_wqe_ratio.dc);
    uct_ib_destroy_qp(dci_qp);

out_qp_tmp_objs_close:
    uct_ib_mlx5dv_qp_tmp_objs_destroy(&qp_tmp_objs);
out:
    return status;
}

static ucs_status_t uct_dc_mlx5_calc_sq_length(uct_ib_mlx5_md_t *md,
                                               unsigned tx_queue_len,
                                               size_t *sq_length_p)
{
    ucs_status_t status;

    if (md->flags & UCT_IB_MLX5_MD_FLAG_DEVX_DCI) {
        *sq_length_p = uct_ib_mlx5_devx_sq_length(tx_queue_len);
    } else {
        status = uct_dc_mlx5dv_calc_tx_wqe_ratio(md);
        if (status != UCS_OK) {
            return status;
        }

        *sq_length_p = ucs_roundup_pow2(tx_queue_len * md->dv_tx_wqe_ratio.dc);
    }

    return UCS_OK;
}

static ucs_status_t
uct_dc_mlx5_iface_init_port_affinity(uct_dc_mlx5_iface_t *iface,
                                     const uct_dc_mlx5_iface_config_t *config)
{
    uct_ib_iface_t *ib_iface = &iface->super.super.super;
    uct_ib_mlx5_md_t *md     = uct_ib_mlx5_iface_md(ib_iface);
    uct_dc_mlx5_dct_affinity_t dct_affinity;
    uint32_t port_affinity;

    iface->tx.port_affinity = 0;
    if (config->tx_port_affinity == UCS_CONFIG_ON) {
        if ((md->port_select_mode == UCT_IB_MLX5_LAG_QUEUE_AFFINITY) ||
            (md->port_select_mode == UCT_IB_MLX5_LAG_PORT_SELECT_FT)) {
            iface->tx.port_affinity = 1;
        } else {
            ucs_warn("Device %s does not support set"
                     "UCX_DC_MLX5_LAG_PORT_AFFINITY=on, port select mode is %d",
                     uct_ib_device_name(&md->super.dev),
                     md->port_select_mode);
        }
    } else if ((config->tx_port_affinity == UCS_CONFIG_AUTO) &&
               (md->port_select_mode == UCT_IB_MLX5_LAG_QUEUE_AFFINITY)) {
        iface->tx.port_affinity = 1;
    }

    dct_affinity = UCS_CONFIG_UINT_ENUM_INDEX(config->dct_affinity);
    if (dct_affinity == UCT_DC_MLX5_DCT_AFFINITY_DEFAULT) {
        iface->rx.port_affinity = ib_iface->config.port_num;
        return UCS_OK;
    }

    if (dct_affinity == UCT_DC_MLX5_DCT_AFFINITY_RANDOM) {
        if (md->super.dev.lag_level <= 1) {
            iface->rx.port_affinity = ib_iface->config.port_num;
        } else {
            ucs_rand_range(1, md->super.dev.lag_level, &port_affinity);
            iface->rx.port_affinity = port_affinity;
        }
        return UCS_OK;
    }

    if ((config->dct_affinity >= 1) &&
        (config->dct_affinity <= md->super.dev.lag_level)) {
        iface->rx.port_affinity = config->dct_affinity;
        return UCS_OK;
    }

    ucs_error("%s: invalid dct_affinity %d, lag_level=%d",
              uct_ib_device_name(&md->super.dev), config->dct_affinity,
              md->super.dev.lag_level);
    return UCS_ERR_INVALID_PARAM;
}

static UCS_CLASS_INIT_FUNC(uct_dc_mlx5_iface_t, uct_md_h tl_md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_dc_mlx5_iface_config_t *config = ucs_derived_of(tl_config,
                                                        uct_dc_mlx5_iface_config_t);
    uct_ib_mlx5_md_t *md               = ucs_derived_of(tl_md,
                                                        uct_ib_mlx5_md_t);
    uct_ib_iface_init_attr_t init_attr = {};
    unsigned tx_queue_len              = config->super.super.tx.queue_len;
    size_t sq_length;
    ucs_status_t status;
    unsigned tx_cq_size;
    unsigned num_dci_channels;
    int max_dcis;
    uint8_t num_cq_dcis;

    ucs_trace_func("");

    self->tx.policy = config->tx_policy;
    self->tx.ndci   = uct_dc_mlx5_iface_is_hw_dcs(self) ? 1 : config->ndci;

    init_attr.qp_type       = UCT_IB_QPT_DCI;
    init_attr.flags         = UCT_IB_CQ_IGNORE_OVERRUN |
                              UCT_IB_TX_OPS_PER_PATH;
    init_attr.fc_req_size   = sizeof(uct_dc_fc_request_t);
    init_attr.max_rd_atomic = md->max_rd_atomic_dc;

    if (md->flags & UCT_IB_MLX5_MD_FLAG_DC_TM) {
        init_attr.flags  |= UCT_IB_TM_SUPPORTED;
    }

    status = uct_dc_mlx5_calc_sq_length(md, tx_queue_len, &sq_length);
    if (status != UCS_OK) {
        return status;
    }

    num_cq_dcis = self->tx.ndci + UCT_DC_MLX5_KEEPALIVE_NUM_DCIS;
    init_attr.cq_len[UCT_IB_DIR_TX] = sq_length * num_cq_dcis;
    uct_ib_mlx5_parse_cqe_zipping(md, &config->rc_mlx5_common.super,
                                  &init_attr);

    /* TODO check caps instead */
    UCS_CLASS_CALL_SUPER_INIT(uct_rc_mlx5_iface_common_t,
                              &uct_dc_mlx5_iface_tl_ops, &uct_dc_mlx5_iface_ops,
                              tl_md, worker, params, &config->super,
                              &config->rc_mlx5_common, &init_attr);

    tx_cq_size = uct_ib_cq_size(&self->super.super.super, &init_attr,
                                UCT_IB_DIR_TX);

    /* driver will round up num cqes to pow of 2 if needed */
    if (ucs_roundup_pow2(tx_cq_size) > UCT_DC_MLX5_MAX_TX_CQ_LEN) {
        ucs_error("Can't allocate TX resources, try to decrease dcis number (%u)"
                  " or tx qp length (%d)", self->tx.ndci, tx_queue_len);
        return UCS_ERR_INVALID_PARAM;
    }

    if (config->fc_hard_req_timeout == UCS_ULUNITS_AUTO) {
        ucs_error("timeout for resending of FC_HARD_REQ shouldn't be set to"
                  " \"auto\"");
        return UCS_ERR_INVALID_PARAM;
    }

    uct_dc_mlx5_iface_init_version(self, tl_md);

    self->tx.fc_seq                        = 0;
    self->tx.fc_hard_req_timeout           = config->fc_hard_req_timeout;
    self->tx.fc_hard_req_resend_time       = ucs_get_time();
    self->tx.fc_hard_req_progress_cb_id    = UCS_CALLBACKQ_ID_NULL;
    self->tx.dci_release_prog_id           = UCS_CALLBACKQ_ID_NULL;
    self->keepalive_dci                    = -1;
    self->tx.num_dci_pools                 = 1;
    self->super.super.config.tx_moderation = 0; /* disable tx moderation for dcs */
    self->flags                            = 0;
    self->tx.av_fl_mlid                    = self->super.super.super.path_bits[0] & 0x7f;

    kh_init_inplace(uct_dc_mlx5_fc_hash, &self->tx.fc_hash);

    self->tx.rand_seed = config->rand_seed ? config->rand_seed : time(NULL);
    self->tx.pend_cb   = uct_dc_mlx5_iface_is_dci_shared(self) ?
                         uct_dc_mlx5_iface_dci_do_rand_pending_tx :
                         uct_dc_mlx5_iface_dci_do_dcs_pending_tx;

    if (config->num_dci_channels == 0) {
        ucs_error("num_dci_channels must be larger than 0");
        return UCS_ERR_INVALID_PARAM;
    }

    if (uct_dc_mlx5_iface_is_hw_dcs(self)) {
        /* Calculate num_dci_channels: select minimum from requested by runtime
         * and supported by HCA, must be power of two */
        num_dci_channels          = ucs_roundup_pow2(config->num_dci_channels);
        self->tx.num_dci_channels = ucs_min(num_dci_channels,
                                            UCS_BIT(md->log_max_dci_stream_channels));
    } else {
        self->tx.num_dci_channels = 1;
    }

    self->tx.dci_pool_release_bitmap = 0;

    if (ucs_test_all_flags(md->flags, UCT_IB_MLX5_MD_FLAG_DEVX_DCI |
                                      UCT_IB_MLX5_MD_FLAG_CQE_V1)) {
        self->flags           |= UCT_DC_MLX5_IFACE_FLAG_UIDX;
        self->tx.num_dci_pools = self->super.super.super.num_paths;
    }

    if ((params->field_mask & UCT_IFACE_PARAM_FIELD_FEATURES) &&
        !(params->features & UCT_IFACE_FEATURE_PUT)) {
        self->flags |= UCT_DC_MLX5_IFACE_FLAG_DISABLE_PUT;
    }

    max_dcis = ucs_min(INT8_MAX, UINT8_MAX / self->tx.num_dci_pools);
    if ((self->tx.ndci < 1) || (self->tx.ndci > max_dcis)) {
        ucs_error("dc interface must have 1..%d dcis (requested: %u)", max_dcis,
                  self->tx.ndci);
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_dc_mlx5_iface_init_port_affinity(self, config);
    if (status != UCS_OK) {
        goto err;
    };

    UCT_DC_MLX5_CHECK_FORCE_FULL_HANDSHAKE(self, config, dci, DCI, status, err);
    UCT_DC_MLX5_CHECK_FORCE_FULL_HANDSHAKE(self, config, dci_ka, KEEPALIVE,
                                           status, err);
    UCT_DC_MLX5_CHECK_FORCE_FULL_HANDSHAKE(self, config, dct, DCT, status, err);

    ucs_assert(self->tx.num_dci_pools <= UCT_DC_MLX5_IFACE_MAX_DCI_POOLS);

    /* create DC target */
    status = uct_dc_mlx5_iface_create_dct(self, config);
    if (status != UCS_OK) {
        goto err;
    }

    /* create DC initiators */
    status = uct_dc_mlx5_iface_dcis_create(self, config);
    if (status != UCS_OK) {
        goto err_destroy_dct;
    }

    ucs_assertv(sq_length >= self->tx.bb_max, "sq_length %zu bb_max %u",
                sq_length, self->tx.bb_max);

    ucs_debug("dc iface %p: using '%s' policy with %d dcis and %d cqes, dct 0x%x",
              self, uct_dc_tx_policy_names[self->tx.policy], self->tx.ndci,
              tx_cq_size, UCT_RC_MLX5_TM_ENABLED(&self->super) ?
              0 : self->rx.dct.qp_num);

    /* mlx5 init part */
    status = uct_ud_mlx5_iface_common_init(&self->super.super.super,
                                           &self->ud_common, &config->mlx5_ud);
    if (status != UCS_OK) {
        goto err_destroy_fc_ep_and_dcis;
    }

    /* Create fake endpoint which will be used for sending FC grants */
    uct_dc_mlx5_iface_init_fc_ep(self);

    uct_dc_mlx5_iface_set_quota(self, config);

    uct_rc_mlx5_iface_common_prepost_recvs(&self->super);

    ucs_debug("created dc iface %p", self);

    return UCS_OK;

err_destroy_fc_ep_and_dcis:
    uct_dc_mlx5_iface_dcis_destroy(self, self->tx.num_dci_pools,
                                   uct_dc_mlx5_iface_total_ndci(self));
err_destroy_dct:
    uct_dc_mlx5_destroy_dct(self);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_mlx5_iface_t)
{
    ucs_trace_func("");
    uct_base_iface_progress_disable(&self->super.super.super.super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);

    uct_dc_mlx5_destroy_dct(self);
    kh_destroy_inplace(uct_dc_mlx5_fc_hash, &self->tx.fc_hash);
    uct_dc_mlx5_iface_cleanup_fc_ep(self);
    uct_worker_progress_unregister_safe(
            &self->super.super.super.super.worker->super,
            &self->tx.dci_release_prog_id);
    uct_dc_mlx5_iface_dcis_destroy(self, self->tx.num_dci_pools,
                                   uct_dc_mlx5_iface_total_ndci(self));
}

UCS_CLASS_DEFINE(uct_dc_mlx5_iface_t, uct_rc_mlx5_iface_common_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_dc_mlx5_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_dc_mlx5_iface_t, uct_iface_t);

static ucs_status_t
uct_dc_mlx5_query_tl_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                             unsigned *num_tl_devices_p)
{
    uct_ib_md_t *ib_md = ucs_derived_of(md, uct_ib_md_t);
    int flags;

    if (strcmp(ib_md->name, UCT_IB_MD_NAME(mlx5))) {
        return UCS_ERR_NO_DEVICE;
    }

    flags = UCT_IB_DEVICE_FLAG_MLX5_PRM | UCT_IB_DEVICE_FLAG_DC |
            (ib_md->config.eth_pause ? 0 : UCT_IB_DEVICE_FLAG_LINK_IB);
    return uct_ib_device_query_ports(&ib_md->dev, flags, tl_devices_p,
                                     num_tl_devices_p);
}

UCT_TL_DEFINE_ENTRY(&uct_ib_component, dc_mlx5, uct_dc_mlx5_query_tl_devices,
                    uct_dc_mlx5_iface_t, "DC_MLX5_",
                    uct_dc_mlx5_iface_config_table, uct_dc_mlx5_iface_config_t);

static void
uct_dc_mlx5_dci_keepalive_handle_failure(uct_dc_mlx5_iface_t *iface,
                                         struct mlx5_cqe64 *cqe,
                                         uint8_t dci_index,
                                         ucs_status_t ep_status)
{
    uint16_t hw_ci = ntohs(cqe->wqe_counter);
    uct_dc_mlx5_ep_t *ep;
    uct_rc_iface_send_op_t *op;
    ucs_queue_elem_t *elem;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    ucs_assert(dci_index == iface->keepalive_dci);
    UCT_DC_MLX5_IFACE_TXQP_DCI_GET(iface, dci_index, txqp, txwq);

    elem = ucs_queue_pull(&txqp->outstanding);
    if (elem == NULL) {
        /* Outstanding list is empty, just exit */
        goto reset_dci;
    }

    op = ucs_container_of(elem, uct_rc_iface_send_op_t, queue);
    if (hw_ci != op->sn) {
        goto put_op;
    }

    ep = ucs_derived_of(op->ep, uct_dc_mlx5_ep_t);
    uct_dc_mlx5_iface_set_ep_failed(iface, ep, cqe, txwq, ep_status);

put_op:
    ucs_mpool_put(op);

reset_dci:
    uct_rc_txqp_available_set(txqp, iface->tx.bb_max);
    uct_rc_txqp_purge_outstanding(&iface->super.super, txqp, ep_status,
                                  txwq->sw_pi, 0);
    uct_dc_mlx5_iface_reset_dci(iface, dci_index);
}

ucs_status_t uct_dc_mlx5_iface_keepalive_init(uct_dc_mlx5_iface_t *iface)
{
    int full_handshake = iface->flags &
                         UCT_DC_MLX5_IFACE_FLAG_KEEPALIVE_FULL_HANDSHAKE;
    ucs_status_t status;
    uint8_t dci_index;

    if (ucs_likely(iface->flags & UCT_DC_MLX5_IFACE_FLAG_KEEPALIVE)) {
        return UCS_OK;
    }

    dci_index = uct_dc_mlx5_iface_total_ndci(iface);
    status    = uct_dc_mlx5_iface_create_dci(iface, 0, dci_index, 0,
                                             full_handshake);
    if (status != UCS_OK) {
        return status;
    }

    iface->flags        |= UCT_DC_MLX5_IFACE_FLAG_KEEPALIVE;
    iface->keepalive_dci = dci_index;
    return UCS_OK;
}

void uct_dc_mlx5_iface_reset_dci(uct_dc_mlx5_iface_t *iface, uint8_t dci_index)
{
    uct_ib_mlx5_txwq_t *txwq = &iface->tx.dcis[dci_index].txwq;
    ucs_status_t status;

    ucs_debug("iface %p reset dci[%d] qpn 0x%x", iface, dci_index,
              txwq->super.qp_num);

    ucs_assert(!uct_dc_mlx5_iface_dci_has_outstanding(iface, dci_index));

    status = uct_ib_mlx5_modify_qp_state(&iface->super.super.super,
                                         &txwq->super, IBV_QPS_RESET);

    uct_rc_mlx5_iface_commom_clean(&iface->super.cq[UCT_IB_DIR_TX], NULL,
                                   txwq->super.qp_num);

    /* Resume posting from to the beginning of the QP */
    uct_ib_mlx5_txwq_reset(txwq);
    if (status != UCS_OK) {
        ucs_fatal("iface %p failed to reset dci[%d] qpn 0x%x: %s",
                  iface, dci_index, txwq->super.qp_num,
                  ucs_status_string(status));
    }

    status = uct_dc_mlx5_iface_dci_connect(iface, &iface->tx.dcis[dci_index]);
    if (status != UCS_OK) {
        ucs_fatal("iface %p failed to connect dci[%d] qpn 0x%x: %s",
                  iface, dci_index, txwq->super.qp_num,
                  ucs_status_string(status));
    }

    uct_ib_mlx5_txwq_update_flags(txwq, 0, UCT_IB_MLX5_TXWQ_FLAG_FAILED);
}

void uct_dc_mlx5_iface_set_ep_failed(uct_dc_mlx5_iface_t *iface,
                                     uct_dc_mlx5_ep_t *ep,
                                     struct mlx5_cqe64 *cqe,
                                     uct_ib_mlx5_txwq_t *txwq,
                                     ucs_status_t ep_status)
{
    uct_ib_iface_t *ib_iface = &iface->super.super.super;
    ucs_status_t status;
    ucs_log_level_t log_lvl;

    /* We don't purge an endpoint's pending queue, because only a FC endpoint
     * could have internal TX operations scheduled there which shouldn't be
     * purged - they need to be rescheduled when DCI will be recovered after an
     * error */

    if (ep == iface->tx.fc_ep) {
        /* Do not report errors on flow control endpoint */
        if (!(iface->flags & UCT_DC_MLX5_IFACE_FLAG_FC_EP_FAILED)) {
            ucs_debug("got error on DC flow-control endpoint, iface %p: %s",
                      iface, ucs_status_string(ep_status));
        }

        iface->flags |= UCT_DC_MLX5_IFACE_FLAG_FC_EP_FAILED;
        return;
    }

    if (ep->flags & UCT_DC_MLX5_EP_FLAG_ERR_HANDLER_INVOKED) {
        return;
    }

    ep->flags |= UCT_DC_MLX5_EP_FLAG_ERR_HANDLER_INVOKED;

    uct_dc_mlx5_fc_remove_ep(iface, ep, UINT64_MAX);
    uct_rc_fc_restore_wnd(&iface->super.super, &ep->fc);

    if (ep->flags & UCT_DC_MLX5_EP_FLAG_FLUSH_CANCEL) {
        return;
    }

    status     = uct_iface_handle_ep_err(&ib_iface->super.super,
                                         &ep->super.super, ep_status);
    log_lvl    = uct_base_iface_failure_log_level(&ib_iface->super, status,
                                                  ep_status);
    uct_ib_mlx5_completion_with_err(ib_iface, (uct_ib_mlx5_err_cqe_t*)cqe, txwq,
                                    log_lvl);
}
