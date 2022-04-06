/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "dc_mlx5.h"

#include <uct/api/uct.h>
#include <uct/ib/mlx5/dv/ib_mlx5_ifc.h>
#include <ucs/arch/bitops.h>


ucs_status_t uct_dc_mlx5_iface_devx_create_dct(uct_dc_mlx5_iface_t *iface)
{
    uct_ib_device_t *dev  = uct_ib_iface_device(&iface->super.super.super);
    struct mlx5dv_pd dvpd = {};
    struct mlx5dv_cq dvcq = {};
    struct mlx5dv_obj dv  = {};
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(create_dct_in)]   = {};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_dct_out)] = {};
    int dvflags;
    void *dctc;

    dvflags   = MLX5DV_OBJ_PD | MLX5DV_OBJ_CQ;
    dv.pd.in  = uct_ib_iface_md(&iface->super.super.super)->pd;
    dv.pd.out = &dvpd;
    dv.cq.in  = iface->super.super.super.cq[UCT_IB_DIR_RX];
    dv.cq.out = &dvcq;
    mlx5dv_init_obj(&dv, dvflags);

    UCT_IB_MLX5DV_SET(create_dct_in, in, opcode, UCT_IB_MLX5_CMD_OP_CREATE_DCT);
    dctc = UCT_IB_MLX5DV_ADDR_OF(create_dct_in, in, dct_context_entry);
    UCT_IB_MLX5DV_SET(dctc, dctc, pd, dvpd.pdn);
    ucs_assert(iface->super.rx.srq.srq_num != 0);
    UCT_IB_MLX5DV_SET(dctc, dctc, srqn_xrqn, iface->super.rx.srq.srq_num);
    if (UCT_RC_MLX5_TM_ENABLED(&iface->super)) {
        UCT_IB_MLX5DV_SET(dctc, dctc, offload_type, UCT_IB_MLX5_QPC_OFFLOAD_TYPE_RNDV);
    }
    UCT_IB_MLX5DV_SET(dctc, dctc, cqn, dvcq.cqn);
    UCT_IB_MLX5DV_SET64(dctc, dctc, dc_access_key, UCT_IB_KEY);

    UCT_IB_MLX5DV_SET(dctc, dctc, rre, true);
    UCT_IB_MLX5DV_SET(dctc, dctc, rwe, true);
    UCT_IB_MLX5DV_SET(dctc, dctc, rae, true);
    UCT_IB_MLX5DV_SET(dctc, dctc, force_full_handshake,
                      !!(iface->flags &
                         UCT_DC_MLX5_IFACE_FLAG_DCT_FULL_HANDSHAKE));
    UCT_IB_MLX5DV_SET(dctc, dctc, cs_res, uct_ib_mlx5_qpc_cs_res(
                      iface->super.super.super.config.max_inl_cqe[UCT_IB_DIR_RX], 1));
    UCT_IB_MLX5DV_SET(dctc, dctc, atomic_mode, UCT_IB_MLX5_ATOMIC_MODE);
    if (!uct_ib_iface_is_roce(&iface->super.super.super)) {
        UCT_IB_MLX5DV_SET(dctc, dctc, pkey_index,
                          iface->super.super.super.pkey_index);
    }
    UCT_IB_MLX5DV_SET(dctc, dctc, port, iface->super.super.super.config.port_num);

    UCT_IB_MLX5DV_SET(dctc, dctc, min_rnr_nak, iface->super.super.config.min_rnr_timer);

    /* Infiniband and RoCE v1 set traffic class.
     * Also set it for RoCE v2, because some old FW versions rely on tclass
     * even for RoCE v2. */
    UCT_IB_MLX5DV_SET(dctc, dctc, tclass,
                      iface->super.super.super.config.traffic_class);

    if (uct_ib_iface_is_roce_v2(&iface->super.super.super)) {
        /* RoCE V2 sets DSCP */
        UCT_IB_MLX5DV_SET(dctc, dctc, dscp,
                          uct_ib_iface_roce_dscp(&iface->super.super.super));
    }

    UCT_IB_MLX5DV_SET(dctc, dctc, mtu, iface->super.super.super.config.path_mtu);
    UCT_IB_MLX5DV_SET(dctc, dctc, my_addr_index, iface->super.super.super.gid_info.gid_index);
    UCT_IB_MLX5DV_SET(dctc, dctc, hop_limit, iface->super.super.super.config.hop_limit);

    iface->rx.dct.devx.obj = mlx5dv_devx_obj_create(dev->ibv_context, in, sizeof(in),
                                                    out, sizeof(out));
    if (iface->rx.dct.devx.obj == NULL) {
        ucs_error("mlx5dv_devx_obj_create(DCT) failed, syndrome %x: %m",
                  UCT_IB_MLX5DV_GET(create_dct_out, out, syndrome));
        return UCS_ERR_INVALID_PARAM;
    }

    iface->rx.dct.type   = UCT_IB_MLX5_OBJ_TYPE_DEVX;
    iface->rx.dct.qp_num = UCT_IB_MLX5DV_GET(create_dct_out, out, dctn);
    return UCS_OK;
}

ucs_status_t uct_dc_mlx5_iface_devx_dci_connect(uct_dc_mlx5_iface_t *iface,
                                                uct_ib_mlx5_qp_t *qp,
                                                uint8_t path_index)
{
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.super.super.super.md,
                                          uct_ib_mlx5_md_t);
    char in_2init[UCT_IB_MLX5DV_ST_SZ_BYTES(rst2init_qp_in)]   = {};
    char out_2init[UCT_IB_MLX5DV_ST_SZ_BYTES(rst2init_qp_out)] = {};
    char in_2rtr[UCT_IB_MLX5DV_ST_SZ_BYTES(init2rtr_qp_in)]    = {};
    char out_2rtr[UCT_IB_MLX5DV_ST_SZ_BYTES(init2rtr_qp_out)]  = {};
    char in_2rts[UCT_IB_MLX5DV_ST_SZ_BYTES(rtr2rts_qp_in)]     = {};
    char out_2rts[UCT_IB_MLX5DV_ST_SZ_BYTES(rtr2rts_qp_out)]   = {};
    uint32_t opt_param_mask = UCT_IB_MLX5_QP_OPTPAR_RAE;
    ucs_status_t status;
    void *qpc;

    UCT_IB_MLX5DV_SET(rst2init_qp_in, in_2init, opcode, UCT_IB_MLX5_CMD_OP_RST2INIT_QP);
    UCT_IB_MLX5DV_SET(rst2init_qp_in, in_2init, qpn, qp->qp_num);

    qpc = UCT_IB_MLX5DV_ADDR_OF(rst2init_qp_in, in_2init, qpc);
    UCT_IB_MLX5DV_SET(qpc, qpc, pm_state, UCT_IB_MLX5_QPC_PM_STATE_MIGRATED);
    UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.vhca_port_num, iface->super.super.super.config.port_num);
    if (!uct_ib_iface_is_roce(&iface->super.super.super)) {
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.pkey_index,
                          iface->super.super.super.pkey_index);
    }

    status = uct_ib_mlx5_devx_modify_qp(qp, in_2init, sizeof(in_2init),
                                        out_2init, sizeof(out_2init));
    if (status != UCS_OK) {
        return status;
    }

    UCT_IB_MLX5DV_SET(init2rtr_qp_in, in_2rtr, opcode, UCT_IB_MLX5_CMD_OP_INIT2RTR_QP);
    UCT_IB_MLX5DV_SET(init2rtr_qp_in, in_2rtr, qpn, qp->qp_num);

    qpc = UCT_IB_MLX5DV_ADDR_OF(init2rtr_qp_in, in_2rtr, qpc);
    UCT_IB_MLX5DV_SET(qpc, qpc, pm_state, UCT_IB_MLX5_QPC_PM_STATE_MIGRATED);
    UCT_IB_MLX5DV_SET(qpc, qpc, mtu, iface->super.super.super.config.path_mtu);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_msg_max, UCT_IB_MLX5_LOG_MAX_MSG_SIZE);
    UCT_IB_MLX5DV_SET(qpc, qpc, atomic_mode, UCT_IB_MLX5_ATOMIC_MODE);
    UCT_IB_MLX5DV_SET(qpc, qpc, rae, true);
    if (uct_ib_iface_is_roce(&iface->super.super.super)) {
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.eth_prio,
                          iface->super.super.super.config.sl);
        uct_ib_mlx5_devx_set_qpc_port_affinity(md, path_index, qpc,
                                               &opt_param_mask);
    } else {
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.sl,
                          iface->super.super.super.config.sl);
    }

    UCT_IB_MLX5DV_SET(init2rtr_qp_in, in_2rtr, opt_param_mask, opt_param_mask);
    status = uct_ib_mlx5_devx_modify_qp(qp, in_2rtr, sizeof(in_2rtr),
                                        out_2rtr, sizeof(out_2rtr));
    if (status != UCS_OK) {
        return status;
    }

    UCT_IB_MLX5DV_SET(rtr2rts_qp_in, in_2rts, opcode, UCT_IB_MLX5_CMD_OP_RTR2RTS_QP);
    UCT_IB_MLX5DV_SET(rtr2rts_qp_in, in_2rts, qpn, qp->qp_num);

    qpc = UCT_IB_MLX5DV_ADDR_OF(rtr2rts_qp_in, in_2rts, qpc);
    UCT_IB_MLX5DV_SET(qpc, qpc, pm_state, UCT_IB_MLX5_QPC_PM_STATE_MIGRATED);
    /* cppcheck-suppress internalAstError */
    UCT_IB_MLX5DV_SET(qpc, qpc, log_sra_max, ucs_ilog2_or0(iface->super.super.config.max_rd_atomic));
    UCT_IB_MLX5DV_SET(qpc, qpc, retry_count, iface->super.super.config.retry_cnt);
    UCT_IB_MLX5DV_SET(qpc, qpc, rnr_retry, iface->super.super.config.rnr_retry);
    UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.ack_timeout, iface->super.super.config.timeout);
    UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.log_rtm, iface->super.super.config.exp_backoff);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_ack_req_freq,
                      iface->super.config.log_ack_req_freq);

    return uct_ib_mlx5_devx_modify_qp(qp, in_2rts, sizeof(in_2rts),
                                      out_2rts, sizeof(out_2rts));
}

ucs_status_t uct_dc_mlx5_iface_devx_set_srq_dc_params(uct_dc_mlx5_iface_t *iface)
{
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(set_xrq_dc_params_entry_in)]   = {};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(set_xrq_dc_params_entry_out)] = {};
    int ret;

    if (!uct_ib_iface_is_roce(&iface->super.super.super)) {
        UCT_IB_MLX5DV_SET(set_xrq_dc_params_entry_in, in, pkey_table_index,
                          iface->super.super.super.pkey_index);
    }
    UCT_IB_MLX5DV_SET(set_xrq_dc_params_entry_in, in, mtu, iface->super.super.super.config.path_mtu);
    UCT_IB_MLX5DV_SET(set_xrq_dc_params_entry_in, in, sl, iface->super.super.super.config.sl);
    UCT_IB_MLX5DV_SET(set_xrq_dc_params_entry_in, in, reverse_sl, iface->super.super.super.config.sl);
    UCT_IB_MLX5DV_SET(set_xrq_dc_params_entry_in, in, cnak_reverse_sl, iface->super.super.super.config.sl);
    UCT_IB_MLX5DV_SET(set_xrq_dc_params_entry_in, in, ack_timeout, iface->super.super.config.timeout);
    UCT_IB_MLX5DV_SET64(set_xrq_dc_params_entry_in, in, dc_access_key, UCT_IB_KEY);
    ucs_assert(iface->super.rx.srq.srq_num != 0);
    UCT_IB_MLX5DV_SET(set_xrq_dc_params_entry_in, in, xrqn, iface->super.rx.srq.srq_num);
    UCT_IB_MLX5DV_SET(set_xrq_dc_params_entry_in, in, opcode,
                      UCT_IB_MLX5_CMD_OP_SET_XRQ_DC_PARAMS_ENTRY);

    ret = mlx5dv_devx_obj_modify(iface->super.rx.srq.devx.obj, in, sizeof(in), out, sizeof(out));
    if (ret) {
        ucs_error("mlx5dv_devx_obj_modify(SET_XRQ_DC_PARAMS) failed, syndrome %x: %m",
                  UCT_IB_MLX5DV_GET(set_xrq_dc_params_entry_out, out, syndrome));
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

