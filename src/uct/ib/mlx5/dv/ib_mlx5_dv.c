/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/ib/mlx5/ib_mlx5.inl>
#include <ucs/arch/bitops.h>

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

void uct_ib_mlx5dv_dc_qp_init_attr(struct mlx5dv_qp_init_attr *dv_attr,
                                   enum mlx5dv_dc_type dc_type)
{
    dv_attr->comp_mask                   = MLX5DV_QP_INIT_ATTR_MASK_DC;
    dv_attr->dc_init_attr.dc_type        = dc_type;
    dv_attr->dc_init_attr.dct_access_key = UCT_IB_KEY;
}

void uct_ib_mlx5dv_dct_qp_init_attr(uct_ib_qp_init_attr_t *qp_attr,
                                    struct mlx5dv_qp_init_attr *dv_attr,
                                    struct ibv_pd *pd, struct ibv_cq *cq,
                                    struct ibv_srq *srq)
{
    qp_attr->comp_mask = IBV_QP_INIT_ATTR_PD;
    qp_attr->pd        = pd;
    qp_attr->recv_cq   = cq;
    /* DCT can't send, but send_cq have to point to valid CQ */
    qp_attr->send_cq   = cq;
    qp_attr->srq       = srq;
    qp_attr->qp_type   = IBV_QPT_DRIVER;
    uct_ib_mlx5dv_dc_qp_init_attr(dv_attr, MLX5DV_DCTYPE_DCT);
}

ucs_status_t
uct_ib_mlx5dv_qp_tmp_objs_create(struct ibv_pd *pd,
                                 uct_ib_mlx5dv_qp_tmp_objs_t *qp_tmp_objs,
                                 int silent)
{
    struct ibv_srq_init_attr srq_attr = {};
    ucs_log_level_t level             = silent ? UCS_LOG_LEVEL_DEBUG :
                                                 UCS_LOG_LEVEL_ERROR;

    qp_tmp_objs->cq = ibv_create_cq(pd->context, 1, NULL, NULL, 0);
    if (qp_tmp_objs->cq == NULL) {
        uct_ib_check_memlock_limit_msg(pd->context, level, "ibv_create_cq()");
        goto out;
    }

    srq_attr.attr.max_sge = 1;
    srq_attr.attr.max_wr  = 1;
    qp_tmp_objs->srq      = ibv_create_srq(pd, &srq_attr);
    if (qp_tmp_objs->srq == NULL) {
        uct_ib_check_memlock_limit_msg(pd->context, level, "ibv_create_srq()");
        goto out_destroy_cq;
    }

    return UCS_OK;

out_destroy_cq:
    ibv_destroy_cq(qp_tmp_objs->cq);
out:
    return UCS_ERR_IO_ERROR;
}

void uct_ib_mlx5dv_qp_tmp_objs_destroy(uct_ib_mlx5dv_qp_tmp_objs_t *qp_tmp_objs)
{
    uct_ib_destroy_srq(qp_tmp_objs->srq);
    ibv_destroy_cq(qp_tmp_objs->cq);
}

size_t uct_ib_mlx5dv_calc_tx_wqe_ratio(struct ibv_qp *qp, uint32_t max_send_wr,
                                       size_t *tx_wqe_ratio_p)
{
    uct_ib_mlx5dv_qp_t qp_info = {};
    uct_ib_mlx5dv_t obj        = {};
    ucs_status_t status;

    obj.dv.qp.in  = qp;
    obj.dv.qp.out = &qp_info.dv;

    status = uct_ib_mlx5dv_init_obj(&obj, MLX5DV_OBJ_QP);
    if (status != UCS_OK) {
        return status;
    }

    *tx_wqe_ratio_p = qp_info.dv.sq.wqe_cnt / max_send_wr;
    return UCS_OK;
}

void uct_ib_mlx5dv_qp_init_attr(uct_ib_qp_init_attr_t *qp_init_attr,
                                struct ibv_pd *pd,
                                const uct_ib_mlx5dv_qp_tmp_objs_t *qp_tmp_objs,
                                enum ibv_qp_type qp_type, uint32_t max_recv_wr)
{
    qp_init_attr->send_cq         = qp_tmp_objs->cq;
    qp_init_attr->recv_cq         = qp_tmp_objs->cq;
    qp_init_attr->srq             = qp_tmp_objs->srq;
    qp_init_attr->qp_type         = qp_type;
    qp_init_attr->sq_sig_all      = 0;
#if HAVE_DECL_IBV_CREATE_QP_EX
    qp_init_attr->comp_mask       = IBV_QP_INIT_ATTR_PD;
    qp_init_attr->pd              = pd;
#endif
    qp_init_attr->cap.max_send_wr = 128;
    qp_init_attr->cap.max_recv_wr = max_recv_wr;
}

#if HAVE_DEVX
ucs_status_t uct_ib_mlx5_devx_create_qp(uct_ib_iface_t *iface,
                                        const uct_ib_mlx5_cq_t *send_cq,
                                        const uct_ib_mlx5_cq_t *recv_cq,
                                        uct_ib_mlx5_qp_t *qp,
                                        uct_ib_mlx5_txwq_t *tx,
                                        uct_ib_mlx5_qp_attr_t *attr)
{
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.md, uct_ib_mlx5_md_t);
    uct_ib_device_t *dev = &md->super.dev;
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(create_qp_in)]           = {};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_qp_out)]         = {};
    char in_2init[UCT_IB_MLX5DV_ST_SZ_BYTES(rst2init_qp_in)]   = {};
    char out_2init[UCT_IB_MLX5DV_ST_SZ_BYTES(rst2init_qp_out)] = {};
    uct_ib_mlx5_mmio_mode_t mmio_mode;
    int max_tx, max_rx, len_tx, len;
    uct_ib_mlx5_devx_uar_t *uar;
    ucs_status_t status;
    void *qpc;

    uct_ib_iface_fill_attr(iface, &attr->super);

    status = uct_ib_mlx5_get_mmio_mode(iface->super.worker, attr->mmio_mode, 0,
                                       UCT_IB_MLX5_BF_REG_SIZE, &mmio_mode);
    if (status != UCS_OK) {
        goto err;
    }

    uar = uct_worker_tl_data_get(iface->super.worker,
                                 UCT_IB_MLX5_DEVX_UAR_KEY,
                                 uct_ib_mlx5_devx_uar_t,
                                 uct_ib_mlx5_devx_uar_cmp,
                                 uct_ib_mlx5_devx_uar_init,
                                 md, mmio_mode);
    if (UCS_PTR_IS_ERR(uar)) {
        status = UCS_PTR_STATUS(uar);
        goto err;
    }

    max_tx = uct_ib_mlx5_devx_sq_length(attr->super.cap.max_send_wr);
    len_tx = max_tx * MLX5_SEND_WQE_BB;
    max_rx = ucs_roundup_pow2_or0(attr->super.cap.max_recv_wr);
    len    = len_tx + max_rx * UCT_IB_MLX5_MAX_BB * UCT_IB_MLX5_WQE_SEG_SIZE;

    if (tx != NULL) {
        status = uct_ib_mlx5_md_buf_alloc(md, len, 0, &qp->devx.wq_buf,
                                          &qp->devx.mem, 0, "qp umem");
        if (status != UCS_OK) {
            goto err_uar;
        }
    } else {
        qp->devx.wq_buf = NULL;
    }

    qp->devx.dbrec = uct_ib_mlx5_get_dbrec(md);
    if (!qp->devx.dbrec) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_mem;
    }

    UCT_IB_MLX5DV_SET(create_qp_in, in, opcode, UCT_IB_MLX5_CMD_OP_CREATE_QP);
    qpc = UCT_IB_MLX5DV_ADDR_OF(create_qp_in, in, qpc);
    if (attr->super.qp_type == UCT_IB_QPT_DCI) {
        UCT_IB_MLX5DV_SET(qpc, qpc, st, UCT_IB_MLX5_QPC_ST_DCI);
        UCT_IB_MLX5DV_SET(qpc, qpc, full_handshake, !!attr->full_handshake);
        UCT_IB_MLX5DV_SET(qpc, qpc, log_num_dci_stream_channels,
                          attr->log_num_dci_stream_channels);
    } else if (attr->super.qp_type == IBV_QPT_RC) {
        UCT_IB_MLX5DV_SET(qpc, qpc, st, UCT_IB_MLX5_QPC_ST_RC);
    } else {
        ucs_error("create qp failed: unknown type %d", attr->super.qp_type);
        status = UCS_ERR_UNSUPPORTED;
        goto err_free_db;
    }
    UCT_IB_MLX5DV_SET(qpc, qpc, pm_state, UCT_IB_MLX5_QPC_PM_STATE_MIGRATED);
    UCT_IB_MLX5DV_SET(qpc, qpc, rdma_wr_disabled, !!attr->rdma_wr_disabled);
    UCT_IB_MLX5DV_SET(qpc, qpc, pd, uct_ib_mlx5_devx_md_get_pdn(md));
    UCT_IB_MLX5DV_SET(qpc, qpc, uar_page, uar->uar->page_id);
    ucs_assert((attr->super.srq == NULL) || (attr->super.srq_num != 0));
    UCT_IB_MLX5DV_SET(qpc, qpc, rq_type, attr->super.srq_num ? 1 /* SRQ */ :
                                                               3 /* no RQ */);
    UCT_IB_MLX5DV_SET(qpc, qpc, srqn_rmpn_xrqn, attr->super.srq_num);
    UCT_IB_MLX5DV_SET(qpc, qpc, cqn_snd, send_cq->cq_num);
    UCT_IB_MLX5DV_SET(qpc, qpc, cqn_rcv, recv_cq->cq_num);
    /* cppcheck-suppress internalAstError */
    UCT_IB_MLX5DV_SET(qpc, qpc, log_sq_size, ucs_ilog2_or0(max_tx));
    UCT_IB_MLX5DV_SET(qpc, qpc, log_rq_size, ucs_ilog2_or0(max_rx));
    UCT_IB_MLX5DV_SET(qpc, qpc, cs_req,
            uct_ib_mlx5_qpc_cs_req(attr->super.max_inl_cqe[UCT_IB_DIR_TX]));
    UCT_IB_MLX5DV_SET(qpc, qpc, cs_res,
            uct_ib_mlx5_qpc_cs_res(attr->super.max_inl_cqe[UCT_IB_DIR_RX], 0));
    UCT_IB_MLX5DV_SET64(qpc, qpc, dbr_addr, qp->devx.dbrec->offset);
    UCT_IB_MLX5DV_SET(qpc, qpc, dbr_umem_id, qp->devx.dbrec->mem_id);
    UCT_IB_MLX5DV_SET(qpc, qpc, user_index, attr->uidx);
    UCT_IB_MLX5DV_SET(qpc, qpc, ts_format, UCT_IB_MLX5_QPC_TS_FORMAT_DEFAULT);

    if (qp->devx.wq_buf == NULL) {
        UCT_IB_MLX5DV_SET(qpc, qpc, no_sq, true);
        UCT_IB_MLX5DV_SET(qpc, qpc, offload_type, true);
        UCT_IB_MLX5DV_SET(create_qp_in, in, wq_umem_id, md->zero_mem.mem->umem_id);
    } else {
        UCT_IB_MLX5DV_SET(create_qp_in, in, wq_umem_id, qp->devx.mem.mem->umem_id);
    }

    if (md->super.ece_enable) {
        UCT_IB_MLX5DV_SET(create_qp_in, in, ece,
                          UCT_IB_MLX5_DEVX_ECE_TRIG_RESP);
    }

    qp->devx.obj = uct_ib_mlx5_devx_obj_create(dev->ibv_context, in,
                                               sizeof(in), out, sizeof(out),
                                               "QP", UCS_LOG_LEVEL_ERROR);
    if (!qp->devx.obj) {
        status = UCS_ERR_IO_ERROR;
        goto err_free_db;
    }

    qp->qp_num = UCT_IB_MLX5DV_GET(create_qp_out, out, qpn);

    if (attr->super.qp_type == IBV_QPT_RC) {
        qpc = UCT_IB_MLX5DV_ADDR_OF(rst2init_qp_in, in_2init, qpc);
        UCT_IB_MLX5DV_SET(rst2init_qp_in, in_2init, opcode, UCT_IB_MLX5_CMD_OP_RST2INIT_QP);
        UCT_IB_MLX5DV_SET(rst2init_qp_in, in_2init, qpn, qp->qp_num);
        UCT_IB_MLX5DV_SET(qpc, qpc, pm_state, UCT_IB_MLX5_QPC_PM_STATE_MIGRATED);
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.vhca_port_num, attr->super.port);
        if (!uct_ib_iface_is_roce(iface)) {
            UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.pkey_index,
                              iface->pkey_index);
        }
        UCT_IB_MLX5DV_SET(qpc, qpc, counter_set_id,
                          uct_ib_mlx5_iface_get_counter_set_id(iface));
        UCT_IB_MLX5DV_SET(qpc, qpc, rwe, true);

        status = uct_ib_mlx5_devx_obj_modify(qp->devx.obj, in_2init,
                                             sizeof(in_2init), out_2init,
                                             sizeof(out_2init), "2INIT_QP");
        if (status != UCS_OK) {
            goto err_free;
        }
    }

    qp->type = UCT_IB_MLX5_OBJ_TYPE_DEVX;

    attr->super.cap.max_send_wr = max_tx;
    attr->super.cap.max_recv_wr = max_rx;

    if (tx != NULL) {
        ucs_assert(qp->devx.wq_buf != NULL);
        tx->reg    = &uar->super;
        tx->qstart = qp->devx.wq_buf;
        tx->qend   = UCS_PTR_BYTE_OFFSET(qp->devx.wq_buf, len_tx);
        tx->dbrec  = &qp->devx.dbrec->db[MLX5_SND_DBR];
        tx->bb_max = max_tx - 2 * UCT_IB_MLX5_MAX_BB;
        ucs_assert(*tx->dbrec == 0);
        uct_ib_mlx5_txwq_reset(tx);
    } else {
        ucs_assert(qp->devx.wq_buf == NULL);
        uct_worker_tl_data_put(uar, uct_ib_mlx5_devx_uar_cleanup);
    }

    return UCS_OK;

err_free:
    uct_ib_mlx5_devx_obj_destroy(qp->devx.obj, "QP");
err_free_db:
    uct_ib_mlx5_put_dbrec(qp->devx.dbrec);
err_free_mem:
    uct_ib_mlx5_md_buf_free(md, qp->devx.wq_buf, &qp->devx.mem);
err_uar:
    uct_worker_tl_data_put(uar, uct_ib_mlx5_devx_uar_cleanup);
err:
    return status;
}

ucs_status_t uct_ib_mlx5_devx_modify_qp(uct_ib_mlx5_qp_t *qp,
                                        const void *in, size_t inlen,
                                        void *out, size_t outlen)
{
    int ret;
    char opcode_str[16];

    ucs_snprintf_zero(opcode_str, sizeof(opcode_str), "opcode=0x%x",
                      UCT_IB_MLX5DV_GET(modify_qp_in, in, opcode));

    switch (qp->type) {
    case UCT_IB_MLX5_OBJ_TYPE_VERBS:
        ret = mlx5dv_devx_qp_modify(qp->verbs.qp, in, inlen, out, outlen);
        if (ret) {
            ucs_error("mlx5dv_devx_qp_modify(%s) failed, syndrome 0x%x: %m",
                      opcode_str,
                      UCT_IB_MLX5DV_GET(modify_qp_out, out, syndrome));
            return UCS_ERR_IO_ERROR;
        }
        break;
    case UCT_IB_MLX5_OBJ_TYPE_DEVX:
        return uct_ib_mlx5_devx_obj_modify(qp->devx.obj, in, inlen, out, outlen,
                                           opcode_str);
    case UCT_IB_MLX5_OBJ_TYPE_NULL:
        return UCS_ERR_INVALID_PARAM;
    case UCT_IB_MLX5_OBJ_TYPE_LAST:
        return UCS_ERR_UNSUPPORTED;
    }

    return UCS_OK;
}

static ucs_status_t
uct_ib_mlx5_devx_query_qp(uct_ib_mlx5_qp_t *qp, void *in, size_t inlen,
                          void *out, size_t outlen)
{
    int ret;

    UCT_IB_MLX5DV_SET(query_qp_in, in, opcode, UCT_IB_MLX5_CMD_OP_QUERY_QP);
    UCT_IB_MLX5DV_SET(query_qp_in, in, qpn, qp->qp_num);

    switch (qp->type) {
    case UCT_IB_MLX5_OBJ_TYPE_VERBS:
        ret = mlx5dv_devx_qp_query(qp->verbs.qp, in, inlen, out, outlen);
        if (ret) {
            ucs_error("mlx5dv_devx_qp_query(QP) failed, syndrome 0x%x: %m",
                      UCT_IB_MLX5DV_GET(query_qp_out, out, syndrome));
            return UCS_ERR_IO_ERROR;
        }
        break;
    case UCT_IB_MLX5_OBJ_TYPE_DEVX:
        ret = mlx5dv_devx_obj_query(qp->devx.obj, in, inlen, out, outlen);
        if (ret) {
            ucs_error("mlx5dv_devx_obj_query(QP) failed, syndrome 0x%x: %m",
                      UCT_IB_MLX5DV_GET(query_qp_out, out, syndrome));
            return UCS_ERR_IO_ERROR;
        }
        break;
    case UCT_IB_MLX5_OBJ_TYPE_NULL:
        return UCS_ERR_INVALID_PARAM;
    case UCT_IB_MLX5_OBJ_TYPE_LAST:
        return UCS_ERR_UNSUPPORTED;
    }

    return UCS_OK;
}

ucs_status_t uct_ib_mlx5_devx_modify_qp_state(uct_ib_mlx5_qp_t *qp,
                                              enum ibv_qp_state state)
{
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(modify_qp_in)]   = {};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(modify_qp_out)] = {};

    switch (state) {
    case IBV_QPS_ERR:
        UCT_IB_MLX5DV_SET(modify_qp_in, in, opcode, UCT_IB_MLX5_CMD_OP_2ERR_QP);
        break;
    case IBV_QPS_RESET:
        UCT_IB_MLX5DV_SET(modify_qp_in, in, opcode, UCT_IB_MLX5_CMD_OP_2RST_QP);
        break;
    default:
        return UCS_ERR_UNSUPPORTED;
    }

    UCT_IB_MLX5DV_SET(modify_qp_in, in, qpn, qp->qp_num);
    return uct_ib_mlx5_devx_modify_qp(qp, in, sizeof(in), out, sizeof(out));
}

void uct_ib_mlx5_devx_destroy_qp(uct_ib_mlx5_md_t *md, uct_ib_mlx5_qp_t *qp)
{
    uct_ib_mlx5_devx_obj_destroy(qp->devx.obj, "QP");
    uct_ib_mlx5_put_dbrec(qp->devx.dbrec);
    uct_ib_mlx5_md_buf_free(md, qp->devx.wq_buf, &qp->devx.mem);
}

ucs_status_t uct_ib_mlx5_devx_obj_modify(struct mlx5dv_devx_obj *obj,
                                         const void *in, size_t inlen,
                                         void *out, size_t outlen,
                                         char *msg_arg)
{
    int ret;
    unsigned syndrome;

    ret = mlx5dv_devx_obj_modify(obj, in, inlen, out, outlen);
    if (ret != 0) {
        syndrome = UCT_IB_MLX5DV_GET(general_obj_out_cmd_hdr, out, syndrome);
        ucs_error("mlx5dv_devx_obj_modify(%s) failed, syndrome 0x%x: %m",
                  msg_arg, syndrome);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

struct mlx5dv_devx_obj *
uct_ib_mlx5_devx_obj_create(struct ibv_context *context, const void *in,
                            size_t inlen, void *out, size_t outlen,
                            char *msg_arg, ucs_log_level_t log_level)
{
    struct mlx5dv_devx_obj *obj;
    unsigned syndrome;

    obj = mlx5dv_devx_obj_create(context, in, inlen, out, outlen);
    if (obj == NULL) {
        syndrome = UCT_IB_MLX5DV_GET(general_obj_out_cmd_hdr, out, syndrome);
        ucs_log(log_level,
                "mlx5dv_devx_obj_create(%s) failed on %s, syndrome 0x%x: %m",
                msg_arg, ibv_get_device_name(context->device), syndrome);
    }

    return obj;
}

ucs_status_t uct_ib_mlx5_devx_query_ooo_sl_mask(uct_ib_mlx5_md_t *md,
                                                uint8_t port_num,
                                                uint16_t *ooo_sl_mask_p)
{
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(query_hca_vport_context_in)]   = {};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(query_hca_vport_context_out)] = {};
    void *ctx;
    ucs_status_t status;

    if (!(md->flags & UCT_IB_MLX5_MD_FLAG_OOO_SL_MASK)) {
        return UCS_ERR_UNSUPPORTED;
    }

    UCT_IB_MLX5DV_SET(query_hca_vport_context_in, in, opcode,
                      UCT_IB_MLX5_CMD_OP_QUERY_HCA_VPORT_CONTEXT);
    UCT_IB_MLX5DV_SET(query_hca_vport_context_in, in, port_num, port_num);

    status = uct_ib_mlx5_devx_general_cmd(md->super.dev.ibv_context, in,
                                          sizeof(in), out, sizeof(out),
                                          "QUERY_HCA_VPORT_CONTEXT", 0);
    if (status != UCS_OK) {
        return status;
    }

    ctx = UCT_IB_MLX5DV_ADDR_OF(query_hca_vport_context_out, out,
                                hca_vport_context);

    *ooo_sl_mask_p = UCT_IB_MLX5DV_GET(hca_vport_context, ctx, ooo_sl_mask);

    return UCS_OK;
}

void uct_ib_mlx5_devx_set_qpc_dp_ordering(
        void *qpc, ucs_ternary_auto_value_t dp_ordering_ooo)
{
    UCT_IB_MLX5DV_SET(qpc, qpc, dp_ordering_0,
                      ucs_ternary_auto_value_is_yes_or_try(dp_ordering_ooo));
    UCT_IB_MLX5DV_SET(qpc, qpc, dp_ordering_1, 0);
    UCT_IB_MLX5DV_SET(qpc, qpc, dp_ordering_force,
                      ucs_ternary_auto_value_is_yes_or_no(dp_ordering_ooo));
}

void uct_ib_mlx5_devx_set_qpc_port_affinity(uct_ib_mlx5_md_t *md,
                                            uint8_t path_index, void *qpc,
                                            uint32_t *opt_param_mask)
{
    uct_ib_device_t *dev = &md->super.dev;
    uint8_t tx_port      = dev->first_port;

    if (!(md->flags & UCT_IB_MLX5_MD_FLAG_LAG)) {
        return;
    }

    *opt_param_mask |= UCT_IB_MLX5_QP_OPTPAR_LAG_TX_AFF;
    if (dev->lag_level > 0) {
        tx_port += path_index % dev->lag_level;
    }
    UCT_IB_MLX5DV_SET(qpc, qpc, lag_tx_port_affinity, tx_port);
}

ucs_status_t
uct_ib_mlx5_devx_query_qp_peer_info(uct_ib_iface_t *iface, uct_ib_mlx5_qp_t *qp,
                                    struct ibv_ah_attr *ah_attr,
                                    uint32_t *dest_qpn)
{
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(query_qp_in)]   = {};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(query_qp_out)] = {};
    void *ctx;
    ucs_status_t status;

    status = uct_ib_mlx5_devx_query_qp(qp, in, sizeof(in), out, sizeof(out));
    if (status != UCS_OK) {
        return UCS_ERR_IO_ERROR;
    }

    ctx                        = UCT_IB_MLX5DV_ADDR_OF(query_qp_out, out, qpc);
    *dest_qpn                  = UCT_IB_MLX5DV_GET(qpc, ctx, remote_qpn);
    ah_attr->dlid              = UCT_IB_MLX5DV_GET(qpc, ctx,
                                                   primary_address_path.rlid);
    ah_attr->sl                = UCT_IB_MLX5DV_GET(qpc, ctx,
                                                   primary_address_path.sl);
    ah_attr->port_num          = UCT_IB_MLX5DV_GET(qpc, ctx,
                                            primary_address_path.vhca_port_num);
    ah_attr->static_rate       = UCT_IB_MLX5DV_GET(qpc, ctx,
                                                primary_address_path.stat_rate);
    ah_attr->src_path_bits     = UCT_IB_MLX5DV_GET(qpc, ctx,
                                                     primary_address_path.mlid);
    ah_attr->is_global         = UCT_IB_MLX5DV_GET(qpc, ctx,
                                                   primary_address_path.grh) ||
                                                   uct_ib_iface_is_roce(iface);
    ah_attr->grh.sgid_index    = UCT_IB_MLX5DV_GET(qpc, ctx,
                                           primary_address_path.src_addr_index);
    ah_attr->grh.traffic_class = UCT_IB_MLX5DV_GET(qpc, ctx,
                                                   primary_address_path.tclass);
    ah_attr->grh.flow_label    = UCT_IB_MLX5DV_GET(qpc, ctx,
                                               primary_address_path.flow_label);
    ah_attr->grh.hop_limit     = UCT_IB_MLX5DV_GET(qpc, ctx,
                                                primary_address_path.hop_limit);

    if (ah_attr->is_global) {
        memcpy(ah_attr->grh.dgid.raw,
               UCT_IB_MLX5DV_ADDR_OF(qpc, ctx, primary_address_path.rgid_rip),
               sizeof(ah_attr->grh.dgid.raw));
    }

    return UCS_OK;
}

ucs_status_t
uct_ib_mlx5_devx_create_cq(uct_ib_iface_t *iface, uct_ib_dir_t dir,
                           const uct_ib_iface_init_attr_t *init_attr,
                           uct_ib_mlx5_cq_t *cq, int preferred_cpu, size_t inl)
{
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(create_cq_in)]   = {0};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_cq_out)] = {0};
    void *cqctx = UCT_IB_MLX5DV_ADDR_OF(create_cq_in, in, cqc);
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.md, uct_ib_mlx5_md_t);
    uct_ib_device_t *dev = uct_ib_iface_device(iface);
    unsigned cq_size = ucs_roundup_pow2(uct_ib_cq_size(iface, init_attr, dir));
    int log_cq_size  = ucs_ilog2(cq_size);
    int cqe_size     = uct_ib_get_cqe_size(inl > 32 ? 128 : 64);
    int num_comp_vectors = dev->ibv_context->num_comp_vectors;
    size_t umem_len      = cqe_size * cq_size;
    ucs_status_t status;
    uint32_t eqn;

    UCT_IB_MLX5DV_SET(create_cq_in, in, opcode, UCT_IB_MLX5_CMD_OP_CREATE_CQ);

    /* Set DB record umem related bits */
    cq->devx.dbrec = uct_ib_mlx5_get_dbrec(md);
    if (cq->devx.dbrec == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }
    UCT_IB_MLX5DV_SET(cqc, cqctx, dbr_umem_id, cq->devx.dbrec->mem_id);
    UCT_IB_MLX5DV_SET64(cqc, cqctx, dbr_addr, cq->devx.dbrec->offset);

    /* Set EQN related bits */
    if (mlx5dv_devx_query_eqn(dev->ibv_context,
                              preferred_cpu % num_comp_vectors, &eqn) != 0) {
        status = UCS_ERR_IO_ERROR;
        goto err_free_db;
    }

    UCT_IB_MLX5DV_SET(cqc, cqctx, c_eqn, eqn);

    /* Set UAR related bits */
    cq->devx.uar = uct_worker_tl_data_get(iface->super.worker,
                                          UCT_IB_MLX5_DEVX_UAR_KEY,
                                          uct_ib_mlx5_devx_uar_t,
                                          uct_ib_mlx5_devx_uar_cmp,
                                          uct_ib_mlx5_devx_uar_init, md,
                                          UCT_IB_MLX5_MMIO_MODE_DB);
    if (UCS_PTR_IS_ERR(cq->devx.uar)) {
        status = UCS_PTR_STATUS(cq->devx.uar);
        goto err_free_db;
    }
    UCT_IB_MLX5DV_SET(cqc, cqctx, uar_page, cq->devx.uar->uar->page_id);

    /* Set CQ umem related bits */
    status = uct_ib_mlx5_md_buf_alloc(md, umem_len, 0, &cq->devx.cq_buf,
                                      &cq->devx.mem, IBV_ACCESS_LOCAL_WRITE,
                                      "cq umem");
    if (status != UCS_OK) {
        goto err_uar;
    }
    memset(cq->devx.cq_buf, 0, umem_len);

    UCT_IB_MLX5DV_SET(create_cq_in, in, cq_umem_id, cq->devx.mem.mem->umem_id);
    UCT_IB_MLX5DV_SET64(create_cq_in, in, cq_umem_offset, 0);

    UCT_IB_MLX5DV_SET(cqc, cqctx, log_cq_size, log_cq_size);
    UCT_IB_MLX5DV_SET(cqc, cqctx, cqe_sz, (cqe_size == 128) ? 1 : 0);

    if (init_attr->cqe_zip_sizes[dir] & cqe_size) {
        UCT_IB_MLX5DV_SET(cqc, cqctx, cqe_comp_en, 1);
        UCT_IB_MLX5DV_SET(cqc, cqctx, cqe_comp_layout, 1);
    }

    if (!UCS_ENABLE_ASSERT && (init_attr->flags & UCT_IB_CQ_IGNORE_OVERRUN)) {
        UCT_IB_MLX5DV_SET(cqc, cqctx, oi, 1);
    }

    cq->devx.obj = uct_ib_mlx5_devx_obj_create(dev->ibv_context, in,
                                               sizeof(in), out, sizeof(out),
                                               "CQ", UCS_LOG_LEVEL_ERROR);
    if (cq->devx.obj == NULL) {
        status = UCS_ERR_IO_ERROR;
        goto err_free_mem;
    }

    uct_ib_mlx5_fill_cq_common(cq, cq_size, cqe_size,
                               UCT_IB_MLX5DV_GET(create_cq_out, out, cqn),
                               cq->devx.cq_buf, cq->devx.uar->uar->base_addr,
                               cq->devx.dbrec->db,
                               !!(init_attr->cqe_zip_sizes[dir] & cqe_size));

    iface->config.max_inl_cqe[dir] = uct_ib_mlx5_inl_cqe(inl, cqe_size);
    iface->cq[dir]                 = NULL;
    cq->type                       = UCT_IB_MLX5_OBJ_TYPE_DEVX;
    return UCS_OK;

err_free_mem:
    uct_ib_mlx5_md_buf_free(md, cq->devx.cq_buf, &cq->devx.mem);
err_uar:
    uct_worker_tl_data_put(cq->devx.uar, uct_ib_mlx5_devx_uar_cleanup);
err_free_db:
    uct_ib_mlx5_put_dbrec(cq->devx.dbrec);
err:
    return status;
}

void uct_ib_mlx5_devx_destroy_cq(uct_ib_mlx5_md_t *md, uct_ib_mlx5_cq_t *cq)
{
    uct_ib_mlx5_devx_obj_destroy(cq->devx.obj, "CQ");
    uct_ib_mlx5_put_dbrec(cq->devx.dbrec);
    uct_worker_tl_data_put(cq->devx.uar, uct_ib_mlx5_devx_uar_cleanup);
    uct_ib_mlx5_md_buf_free(md, cq->devx.cq_buf, &cq->devx.mem);
}
#endif

void uct_ib_mlx5dv_arm_cq(uct_ib_mlx5_cq_t *cq, int solicited)
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
#elif !defined (HAVE_INFINIBAND_MLX5_HW_H)
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
