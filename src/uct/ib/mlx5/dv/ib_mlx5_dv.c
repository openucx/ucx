/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ib_mlx5_ifc.h"

#include <uct/ib/mlx5/ib_mlx5.h>
#include <ucs/arch/bitops.h>

#if HAVE_DECL_MLX5DV_INIT_OBJ
ucs_status_t uct_ib_mlx5dv_init_obj(uct_ib_mlx5dv_t *obj, uint64_t type)
{
    int ret;

    ret = mlx5dv_init_obj(&obj->dv, type);
#if HAVE_IBV_EXP_DM
    if (!ret && (type & MLX5DV_OBJ_DM)) {
        ret = uct_ib_mlx5_get_dm_info(obj->dv_dm.in, obj->dv_dm.out);
    }
#endif
    if (ret != 0) {
        ucs_error("DV failed to get mlx5 information. Type %lx.", type);
        return UCS_ERR_NO_DEVICE;
    }

    return UCS_OK;
}
#endif

ucs_status_t uct_ib_mlx5_devx_create_qp(uct_ib_iface_t *iface,
                                        uct_ib_mlx5_qp_t *qp,
                                        uct_ib_mlx5_txwq_t *tx,
                                        uct_ib_qp_attr_t *attr)
{
#if HAVE_DEVX
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.md, uct_ib_mlx5_md_t);
    uct_ib_device_t *dev = &md->super.dev;
    uint32_t in[UCT_IB_MLX5DV_ST_SZ_DW(create_qp_in)] = {};
    uint32_t out[UCT_IB_MLX5DV_ST_SZ_DW(create_qp_out)] = {};
    uint32_t in_2init[UCT_IB_MLX5DV_ST_SZ_DW(rst2init_qp_in)] = {};
    uint32_t out_2init[UCT_IB_MLX5DV_ST_SZ_DW(rst2init_qp_out)] = {};
    ucs_status_t status = UCS_ERR_NO_MEMORY;
    struct mlx5dv_pd dvpd = {};
    struct mlx5dv_cq dvscq = {};
    struct mlx5dv_cq dvrcq = {};
    struct mlx5dv_srq dvsrq = {};
    struct mlx5dv_obj dv = {};
    uct_ib_mlx5_devx_uar_t *uar;
    int max_tx, max_rx, len_tx, len;
    int dvflags;
    void *qpc;
    int ret;

    uct_ib_iface_fill_attr(iface, attr);

    uar = uct_worker_tl_data_get(iface->super.worker,
                                 UCT_IB_MLX5_DEVX_UAR_KEY,
                                 uct_ib_mlx5_devx_uar_t,
                                 uct_ib_mlx5_devx_uar_cmp,
                                 uct_ib_mlx5_devx_uar_init,
                                 md, UCT_IB_MLX5_MMIO_MODE_BF_POST);
    if (UCS_PTR_IS_ERR(uar)) {
        status = UCS_PTR_STATUS(uar);
        goto err;
    }

    max_tx = ucs_roundup_pow2_or0(attr->cap.max_send_wr);
    max_rx = ucs_roundup_pow2_or0(attr->cap.max_recv_wr);
    len_tx = max_tx * UCT_IB_MLX5_MAX_BB * UCT_IB_MLX5_WQE_SEG_SIZE;
    len    = len_tx + max_rx * UCT_IB_MLX5_MAX_BB * UCT_IB_MLX5_WQE_SEG_SIZE;

    if (tx) {
        qp->devx.wq_buf = ucs_memalign(ucs_get_page_size(), len, "qp umem");
        if (qp->devx.wq_buf == NULL) {
            ucs_error("failed to allocate QP buffer of %d bytes: %m", len);
            goto err_uar;
        }

        qp->devx.mem = mlx5dv_devx_umem_reg(dev->ibv_context, qp->devx.wq_buf, len, 0);
        if (!qp->devx.mem) {
            ucs_error("mlx5dv_devx_umem_reg() failed: %m");
            goto err_free_buf;
        }
    } else {
        qp->devx.wq_buf = qp->devx.mem = NULL;
    }

    qp->devx.dbrec = ucs_mpool_get_inline(&md->dbrec_pool);
    if (!qp->devx.dbrec) {
        goto err_free_mem;
    }

    dv.pd.in            = attr->ibv.pd;
    dv.pd.out           = &dvpd;
    dv.cq.in            = attr->ibv.send_cq;
    dv.cq.out           = &dvscq;
    dvflags             = MLX5DV_OBJ_PD | MLX5DV_OBJ_CQ;

    if (attr->srq) {
        dv.srq.in       = attr->srq;
        dvflags        |= MLX5DV_OBJ_SRQ;
        dv.srq.out      = &dvsrq;
        dvsrq.comp_mask = MLX5DV_SRQ_MASK_SRQN;
    } else {
        dvsrq.srqn      = attr->srq_num;
    }

    mlx5dv_init_obj(&dv, dvflags);
    dv.cq.in            = attr->ibv.recv_cq;
    dv.cq.out           = &dvrcq;
    mlx5dv_init_obj(&dv, MLX5DV_OBJ_CQ);

    UCT_IB_MLX5DV_SET(create_qp_in, in, opcode, UCT_IB_MLX5_CMD_OP_CREATE_QP);
    qpc = UCT_IB_MLX5DV_ADDR_OF(create_qp_in, in, qpc);
    UCT_IB_MLX5DV_SET(qpc, qpc, st, UCT_IB_MLX5_QPC_ST_RC);
    UCT_IB_MLX5DV_SET(qpc, qpc, pm_state, UCT_IB_MLX5_QPC_PM_STATE_MIGRATED);
    UCT_IB_MLX5DV_SET(qpc, qpc, pd, dvpd.pdn);
    UCT_IB_MLX5DV_SET(qpc, qpc, uar_page, uar->uar->page_id);
    UCT_IB_MLX5DV_SET(qpc, qpc, rq_type, !!dvsrq.srqn);
    UCT_IB_MLX5DV_SET(qpc, qpc, srqn_rmpn_xrqn, dvsrq.srqn);
    UCT_IB_MLX5DV_SET(qpc, qpc, cqn_snd, dvscq.cqn);
    UCT_IB_MLX5DV_SET(qpc, qpc, cqn_rcv, dvrcq.cqn);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_sq_size, ucs_ilog2_or0(max_tx));
    UCT_IB_MLX5DV_SET(qpc, qpc, log_rq_size, ucs_ilog2_or0(max_rx));
    UCT_IB_MLX5DV_SET(qpc, qpc, cs_req, UCT_IB_MLX5_QPC_CS_REQ_UP_TO_64B);
    UCT_IB_MLX5DV_SET(qpc, qpc, cs_res, UCT_IB_MLX5_QPC_CS_RES_UP_TO_64B);
    UCT_IB_MLX5DV_SET64(qpc, qpc, dbr_addr, qp->devx.dbrec->offset);
    UCT_IB_MLX5DV_SET(qpc, qpc, dbr_umem_id, qp->devx.dbrec->mem_id);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_ack_req_freq, 8);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_sra_max, 2);
    UCT_IB_MLX5DV_SET(qpc, qpc, retry_count, 7);
    UCT_IB_MLX5DV_SET(qpc, qpc, rnr_retry, 7);

    if (qp->devx.mem == NULL) {
        UCT_IB_MLX5DV_SET(qpc, qpc, no_sq, 1);
        UCT_IB_MLX5DV_SET(qpc, qpc, offload_type, 1);
    } else {
        UCT_IB_MLX5DV_SET(create_qp_in, in, wq_umem_id, qp->devx.mem->umem_id);
    }

    status = UCS_ERR_IO_ERROR;

    qp->devx.obj = mlx5dv_devx_obj_create(dev->ibv_context, in, sizeof(in),
                                          out, sizeof(out));
    if (!qp->devx.obj) {
        ucs_error("mlx5dv_devx_obj_create(QP) failed, syndrome %x: %m",
                  UCT_IB_MLX5DV_GET(create_qp_out, out, syndrome));
        goto err_free_db;
    }

    qp->qp_num = UCT_IB_MLX5DV_GET(create_qp_out, out, qpn);

    qpc = UCT_IB_MLX5DV_ADDR_OF(rst2init_qp_in, in_2init, qpc);
    UCT_IB_MLX5DV_SET(rst2init_qp_in, in_2init, opcode, UCT_IB_MLX5_CMD_OP_RST2INIT_QP);
    UCT_IB_MLX5DV_SET(rst2init_qp_in, in_2init, qpn, qp->qp_num);
    UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.vhca_port_num, attr->port);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_ack_req_freq, 8);
    UCT_IB_MLX5DV_SET(qpc, qpc, rwe, 1);

    ret = mlx5dv_devx_obj_modify(qp->devx.obj, in_2init, sizeof(in_2init),
                                 out_2init, sizeof(out_2init));
    if (ret) {
        ucs_error("mlx5dv_devx_obj_modify(2INIT_QP) failed, syndrome %x: %m",
                  UCT_IB_MLX5DV_GET(rst2init_qp_out, out_2init, syndrome));
        goto err_free;
    }

    qp->type = UCT_IB_MLX5_QP_TYPE_DEVX;

    attr->cap.max_send_wr = max_tx;
    attr->cap.max_recv_wr = max_rx;

    if (tx) {
        tx->reg    = &uar->super;
        tx->qstart = qp->devx.wq_buf;
        tx->qend   = qp->devx.wq_buf + len_tx;
        tx->dbrec  = &qp->devx.dbrec->db[MLX5_SND_DBR];
        tx->bb_max = max_tx - 2 * UCT_IB_MLX5_MAX_BB;
        uct_ib_mlx5_txwq_reset(tx);
    }

    return UCS_OK;

err_free:
    mlx5dv_devx_obj_destroy(qp->devx.obj);
err_free_db:
    ucs_mpool_put_inline(qp->devx.dbrec);
err_free_mem:
    if (qp->devx.mem != NULL) {
        mlx5dv_devx_umem_dereg(qp->devx.mem);
    }
err_free_buf:
    ucs_free(qp->devx.wq_buf);
err_uar:
    uct_worker_tl_data_put(uar, uct_ib_mlx5_devx_uar_cleanup);
err:
    return status;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_ib_mlx5_devx_connect_qp(uct_ib_iface_t *iface,
                                         uct_ib_mlx5_qp_t *qp,
                                         uint32_t dest_qp_num,
                                         struct ibv_ah_attr *ah_attr)
{
#if HAVE_DEVX
    uint32_t in_2rtr[UCT_IB_MLX5DV_ST_SZ_DW(init2rtr_qp_in)] = {};
    uint32_t out_2rtr[UCT_IB_MLX5DV_ST_SZ_DW(init2rtr_qp_out)] = {};
    uint32_t in_2rts[UCT_IB_MLX5DV_ST_SZ_DW(rtr2rts_qp_in)] = {};
    uint32_t out_2rts[UCT_IB_MLX5DV_ST_SZ_DW(rtr2rts_qp_out)] = {};
    uint8_t mac[6], *gid = ah_attr->grh.dgid.raw;
    void *qpc;
    int ret;

    ucs_assert_always(qp->type == UCT_IB_MLX5_QP_TYPE_DEVX);
    UCT_IB_MLX5DV_SET(init2rtr_qp_in, in_2rtr, opcode, UCT_IB_MLX5_CMD_OP_INIT2RTR_QP);
    UCT_IB_MLX5DV_SET(init2rtr_qp_in, in_2rtr, qpn, qp->qp_num);
    UCT_IB_MLX5DV_SET(init2rtr_qp_in, in_2rtr, opt_param_mask, 14);

    qpc = UCT_IB_MLX5DV_ADDR_OF(init2rtr_qp_in, in_2rtr, qpc);
    UCT_IB_MLX5DV_SET(qpc, qpc, mtu, 5);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_msg_max, 30);
    UCT_IB_MLX5DV_SET(qpc, qpc, remote_qpn, dest_qp_num);
    if (uct_ib_iface_is_roce(iface)) {
        mac[0] = gid[8] ^ 0x02;
        memcpy(mac + 1, gid + 9, 2);
        memcpy(mac + 3, gid + 13, 3);
        memcpy(UCT_IB_MLX5DV_ADDR_OF(qpc, qpc, primary_address_path.rmac_47_32), mac, 6);
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.hop_limit, 1);
    } else {
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.grh, ah_attr->is_global);
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.rlid, ah_attr->dlid);
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.mlid, ah_attr->src_path_bits & 0x7f);
    }

    memcpy(UCT_IB_MLX5DV_ADDR_OF(qpc, qpc, primary_address_path.rgid_rip),
            &ah_attr->grh.dgid,
            UCT_IB_MLX5DV_FLD_SZ_BYTES(qpc, primary_address_path.rgid_rip));
    UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.vhca_port_num, ah_attr->port_num);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_ack_req_freq, 8);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_rra_max, 2);
    UCT_IB_MLX5DV_SET(qpc, qpc, atomic_mode, 5);
    UCT_IB_MLX5DV_SET(qpc, qpc, rre, 1);
    UCT_IB_MLX5DV_SET(qpc, qpc, rwe, 1);
    UCT_IB_MLX5DV_SET(qpc, qpc, rae, 1);
    UCT_IB_MLX5DV_SET(qpc, qpc, min_rnr_nak, 13);

    ret = mlx5dv_devx_obj_modify(qp->devx.obj, in_2rtr, sizeof(in_2rtr),
                                 out_2rtr, sizeof(out_2rtr));
    if (ret) {
        ucs_error("mlx5dv_devx_obj_modify(2RTR) failed, syndrome %x: %m",
                  UCT_IB_MLX5DV_GET(init2rtr_qp_out, out_2rtr, syndrome));
        return UCS_ERR_IO_ERROR;
    }

    UCT_IB_MLX5DV_SET(rtr2rts_qp_in, in_2rts, opcode, UCT_IB_MLX5_CMD_OP_RTR2RTS_QP);
    UCT_IB_MLX5DV_SET(rtr2rts_qp_in, in_2rts, qpn, qp->qp_num);

    qpc = UCT_IB_MLX5DV_ADDR_OF(rst2init_qp_in, in_2rts, qpc);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_ack_req_freq, 8);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_sra_max, 2);
    UCT_IB_MLX5DV_SET(qpc, qpc, retry_count, 7);
    UCT_IB_MLX5DV_SET(qpc, qpc, rnr_retry, 7);
    UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.ack_timeout, 18);

    ret = mlx5dv_devx_obj_modify(qp->devx.obj, in_2rts, sizeof(in_2rts),
                                 out_2rts, sizeof(out_2rts));
    if (ret) {
        ucs_error("mlx5dv_devx_obj_modify(2RTS) failed, syndrome %x: %m",
                  UCT_IB_MLX5DV_GET(rtr2rts_qp_out, out_2rts, syndrome));
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

void uct_ib_mlx5_devx_destroy_qp(uct_ib_mlx5_qp_t *qp)
{
#if HAVE_DEVX
    int ret = mlx5dv_devx_obj_destroy(qp->devx.obj);
    if (ret) {
        ucs_error("mlx5dv_devx_obj_destroy(QP) failed: %m");
    }
    ucs_mpool_put_inline(qp->devx.dbrec);
    if (qp->devx.mem != NULL) {
        mlx5dv_devx_umem_dereg(qp->devx.mem);
    }
    ucs_free(qp->devx.wq_buf);
#endif
}

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

#if HAVE_DEVX
ucs_status_t uct_ib_mlx5_get_compact_av(uct_ib_iface_t *iface, int *compact_av)
{
    *compact_av = !!(uct_ib_iface_device(iface)->flags & UCT_IB_DEVICE_FLAG_AV);
    return UCS_OK;
}
#endif

