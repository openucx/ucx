/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "dc_mlx5.h"

#include <uct/api/uct.h>
#include <uct/ib/mlx5/dv/ib_mlx5_ifc.h>
#include <ucs/arch/bitops.h>


ucs_status_t uct_dc_mlx5_iface_devx_create_dct(uct_dc_mlx5_iface_t *iface)
{
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super.super.super);
    uint32_t in[UCT_IB_MLX5DV_ST_SZ_DW(create_dct_in)]   = {};
    uint32_t out[UCT_IB_MLX5DV_ST_SZ_DW(create_dct_out)] = {};
    struct mlx5dv_pd dvpd   = {};
    struct mlx5dv_cq dvcq   = {};
    struct mlx5dv_srq dvsrq = {};
    struct mlx5dv_obj dv    = {};
    int dvflags;
    void *dctc;

    dvflags             = MLX5DV_OBJ_PD | MLX5DV_OBJ_CQ;
    dv.pd.in            = uct_ib_iface_md(&iface->super.super.super)->pd;
    dv.pd.out           = &dvpd;
    dv.cq.in            = iface->super.super.super.cq[UCT_IB_DIR_RX];
    dv.cq.out           = &dvcq;

    if (!UCT_RC_MLX5_TM_ENABLED(&iface->super)) {
        dvflags        |= MLX5DV_OBJ_SRQ;
        dv.srq.in       = iface->super.rx.srq.verbs.srq;
        dv.srq.out      = &dvsrq;
        dvsrq.comp_mask = MLX5DV_SRQ_MASK_SRQN;
    }

    mlx5dv_init_obj(&dv, dvflags);

    UCT_IB_MLX5DV_SET(create_dct_in, in, opcode, UCT_IB_MLX5_CMD_OP_CREATE_DCT);
    dctc = UCT_IB_MLX5DV_ADDR_OF(create_dct_in, in, dct_context_entry);
    UCT_IB_MLX5DV_SET(dctc, dctc, pd, dvpd.pdn);
    if (UCT_RC_MLX5_TM_ENABLED(&iface->super)) {
        UCT_IB_MLX5DV_SET(dctc, dctc, srqn_xrqn, iface->super.rx.srq.srq_num);
        UCT_IB_MLX5DV_SET(dctc, dctc, offload_type, UCT_IB_MLX5_QPC_OFFLOAD_TYPE_RNDV);
    } else {
        UCT_IB_MLX5DV_SET(dctc, dctc, srqn_xrqn, dvsrq.srqn);
    }
    UCT_IB_MLX5DV_SET(dctc, dctc, cqn, dvcq.cqn);
    UCT_IB_MLX5DV_SET64(dctc, dctc, dc_access_key, UCT_IB_KEY);

    UCT_IB_MLX5DV_SET(dctc, dctc, rre, 1);
    UCT_IB_MLX5DV_SET(dctc, dctc, rwe, 1);
    UCT_IB_MLX5DV_SET(dctc, dctc, rae, 1);
    UCT_IB_MLX5DV_SET(dctc, dctc, cs_res, uct_ib_mlx5_qpc_cs_res(
                iface->super.super.super.config.max_inl_resp));
    UCT_IB_MLX5DV_SET(dctc, dctc, atomic_mode, 3);
    UCT_IB_MLX5DV_SET(dctc, dctc, pkey_index, iface->super.super.super.pkey_index);
    UCT_IB_MLX5DV_SET(dctc, dctc, port, iface->super.super.super.config.port_num);

    UCT_IB_MLX5DV_SET(dctc, dctc, min_rnr_nak, iface->super.super.config.min_rnr_timer);
    UCT_IB_MLX5DV_SET(dctc, dctc, tclass, iface->super.super.super.config.traffic_class);
    UCT_IB_MLX5DV_SET(dctc, dctc, mtu, iface->super.super.config.path_mtu);
    UCT_IB_MLX5DV_SET(dctc, dctc, my_addr_index, iface->super.super.super.config.gid_index);
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

ucs_status_t uct_dc_mlx5_iface_devx_set_srq_dc_params(uct_dc_mlx5_iface_t *iface)
{
    uint32_t in[UCT_IB_MLX5DV_ST_SZ_DW(set_xrq_dc_params_entry_in)]   = {};
    uint32_t out[UCT_IB_MLX5DV_ST_SZ_DW(set_xrq_dc_params_entry_out)] = {};
    int ret;

    UCT_IB_MLX5DV_SET(set_xrq_dc_params_entry_in, in, pkey_table_index, iface->super.super.super.pkey_index);
    UCT_IB_MLX5DV_SET(set_xrq_dc_params_entry_in, in, mtu, iface->super.super.config.path_mtu);
    UCT_IB_MLX5DV_SET(set_xrq_dc_params_entry_in, in, sl, iface->super.super.super.config.sl);
    UCT_IB_MLX5DV_SET(set_xrq_dc_params_entry_in, in, reverse_sl, iface->super.super.super.config.sl);
    UCT_IB_MLX5DV_SET(set_xrq_dc_params_entry_in, in, cnak_reverse_sl, iface->super.super.super.config.sl);
    UCT_IB_MLX5DV_SET(set_xrq_dc_params_entry_in, in, ack_timeout, iface->super.super.config.timeout);
    UCT_IB_MLX5DV_SET64(set_xrq_dc_params_entry_in, in, dc_access_key, UCT_IB_KEY);
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

