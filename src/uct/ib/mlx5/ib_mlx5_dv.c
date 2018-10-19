/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ib_mlx5.h"
#include "ib_mlx5_log.h"
#include "ib_mlx5_ifc.h"

#if HAVE_DECL_MLX5DV_INIT_OBJ
ucs_status_t uct_ib_mlx5dv_init_obj(uct_ib_mlx5dv_t *obj, uint64_t type)
{
    int ret;

    ret = mlx5dv_init_obj(&obj->dv, type);
    if (ret != 0) {
        ucs_error("DV failed to get mlx5 information. Type %lx.", type);
        return UCS_ERR_NO_DEVICE;
    }

    return UCS_OK;
}
#endif

#if HAVE_DC_DV
static ucs_status_t uct_ib_mlx5_check_dc(uct_ib_device_t *dev)
{
    struct ibv_context *ctx = dev->ibv_context;
    struct ibv_qp_init_attr_ex qp_attr = {};
    struct mlx5dv_qp_init_attr dv_attr = {};
    ucs_status_t status = UCS_OK;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;

    pd = ibv_alloc_pd(ctx);
    if (pd == NULL) {
        ucs_error("ibv_alloc_pd() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    cq = ibv_create_cq(ctx, 1, NULL, NULL, 0);
    if (cq == NULL) {
        ucs_error("ibv_create_cq() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_cq;
    }

    qp_attr.send_cq              = cq;
    qp_attr.recv_cq              = cq;
    qp_attr.cap.max_send_wr      = 1;
    qp_attr.cap.max_send_sge     = 1;
    qp_attr.qp_type              = IBV_QPT_DRIVER;
    qp_attr.comp_mask            = IBV_QP_INIT_ATTR_PD;
    qp_attr.pd                   = pd;

    dv_attr.comp_mask            = MLX5DV_QP_INIT_ATTR_MASK_DC;
    dv_attr.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCI;

    /* create DCI qp successful means DC is supported */
    qp = mlx5dv_create_qp(ctx, &qp_attr, &dv_attr);
    if (qp) {
        ibv_destroy_qp(qp);
        dev->flags |= UCT_IB_DEVICE_FLAG_DC;
    }

    ibv_destroy_cq(cq);
err_cq:
    ibv_dealloc_pd(pd);
    return status;
}

static ucs_status_t uct_ib_mlx5_device_init(uct_ib_device_t *dev)
{
    ucs_status_t status = UCS_OK;
    int ret = -1;

#if HAVE_DECL_MLX5DV_DEVX_GENERAL_CMD
    uint32_t out[UCT_ST_SZ_DW(query_hca_cap_out)] = {0};
    uint32_t in[UCT_ST_SZ_DW(query_hca_cap_in)] = {0};
    struct ibv_context *ctx = dev->ibv_context;

    UCT_SET(query_hca_cap_in, in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    UCT_SET(query_hca_cap_in, in, op_mod, HCA_CAP_OPMOD_GET_MAX |
                                          (MLX5_CAP_GENERAL << 1));
    ret = mlx5dv_devx_general_cmd(ctx, in, sizeof(in), out, sizeof(out));
    if (ret == 0) {
        if (UCT_GET(query_hca_cap_out, out, capability.cmd_hca_cap.dct)) {
            dev->flags |= UCT_IB_DEVICE_FLAG_DC;
        }
    }
#endif
    if (ret != 0) {
        status = uct_ib_mlx5_check_dc(dev);
    }

    return status;
}

UCT_IB_DEVICE_INIT(uct_ib_mlx5_device_init);

#endif

int uct_ib_mlx5dv_arm_cq(uct_ib_mlx5_cq_t *cq, int solicited)
{
    uint64_t doorbell, sn_ci_cmd;
    uint32_t sn, ci, cmd;

    sn  = cq->cq_sn & 3;
    ci  = cq->cq_ci & 0xffffff;
    cmd = solicited ? MLX5_CQ_DB_REQ_NOT_SOL : MLX5_CQ_DB_REQ_NOT;
    sn_ci_cmd = (sn << 28) | cmd | ci;

    cq->dbrec[UCT_IB_MLX5_CQ_ARM_DB] = htobe32(sn_ci_cmd);

    ucs_memory_cpu_fence();

    doorbell = (sn_ci_cmd << 32) | cq->cq_num;

    *(uint64_t *)((uint8_t *)cq->uar + MLX5_CQ_DOORBELL) = htobe64(doorbell);

    ucs_memory_bus_store_fence();

    return 0;
}

#if HAVE_DECL_MLX5DV_OBJ_AH
void uct_ib_mlx5_get_av(struct ibv_ah *ah, struct mlx5_wqe_av *av)
{
    struct mlx5dv_obj  dv;
    struct mlx5dv_ah   dah;

    dv.ah.in = ah;
    dv.ah.out = &dah;
    mlx5dv_init_obj(&dv, MLX5DV_OBJ_AH);

    *av = *(dah.av);
    av->dqp_dct |= UCT_IB_MLX5_EXTENDED_UD_AV;
}
#elif !HAVE_INFINIBAND_MLX5_HW_H
void uct_ib_mlx5_get_av(struct ibv_ah *ah, struct mlx5_wqe_av *av)
{
    ucs_bug("MLX5DV_OBJ_AH not supported");
}
#endif

