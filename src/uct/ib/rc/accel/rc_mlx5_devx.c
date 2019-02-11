/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "rc_mlx5.inl"

#include <uct/api/uct.h>
#include <ucs/arch/bitops.h>
#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/mlx5/dv/ib_mlx5_ifc.h>


ucs_status_t
uct_rc_mlx5_devx_init_rx_tm(uct_rc_mlx5_iface_common_t *iface,
                            const uct_rc_iface_common_config_t *config,
                            int dc, unsigned rndv_hdr_len)
{
    uct_ib_mlx5_md_t *md = ucs_derived_of(uct_ib_iface_md(&iface->super.super), uct_ib_mlx5_md_t);
    uct_ib_device_t *dev = &md->super.dev;
    uint32_t in[UCT_IB_MLX5DV_ST_SZ_DW(create_xrq_in)] = {};
    uint32_t out[UCT_IB_MLX5DV_ST_SZ_DW(create_xrq_out)] = {};
    ucs_status_t status = UCS_ERR_NO_MEMORY;
    struct mlx5dv_pd dvpd = {};
    struct mlx5dv_cq dvcq = {};
    struct mlx5dv_obj dv = {};
    void *xrqc, *wq;
    int len, ret, max;

    uct_rc_mlx5_init_rx_tm_common(iface, rndv_hdr_len);

    max = ucs_max(config->super.rx.queue_len, UCT_IB_MLX5_XRQ_MIN_UWQ_POST);
    max = ucs_roundup_pow2(max);
    len = max * UCT_IB_MLX5_SRQ_STRIDE;
    ret = posix_memalign(&iface->rx.srq.buf, ucs_get_page_size(), len);
    if (ret) {
        return status;
    }

    iface->rx.srq.devx.mem = mlx5dv_devx_umem_reg(dev->ibv_context,
                                                  iface->rx.srq.buf, len,
                                                  IBV_ACCESS_LOCAL_WRITE);
    if (!iface->rx.srq.devx.mem) {
        goto err_free_buf;
    }

    iface->rx.srq.devx.dbrec = ucs_mpool_get_inline(&md->dbrec_pool);
    if (!iface->rx.srq.devx.dbrec) {
        goto err_free_mem;
    }

    iface->rx.srq.db = &iface->rx.srq.devx.dbrec->db[MLX5_RCV_DBR];
    dv.pd.in         = uct_ib_iface_md(&iface->super.super)->pd;
    dv.cq.in         = iface->super.super.cq[UCT_IB_DIR_RX];
    dv.pd.out        = &dvpd;
    dv.cq.out        = &dvcq;
    mlx5dv_init_obj(&dv, MLX5DV_OBJ_PD | MLX5DV_OBJ_CQ);

    UCT_IB_MLX5DV_SET(create_xrq_in, in, opcode, UCT_IB_MLX5_CMD_OP_CREATE_XRQ);
    xrqc = UCT_IB_MLX5DV_ADDR_OF(create_xrq_in, in, xrq_context);

    UCT_IB_MLX5DV_SET(xrqc, xrqc, topology, UCT_IB_MLX5_XRQC_TOPOLOGY_TAG_MATCHING);
    UCT_IB_MLX5DV_SET(xrqc, xrqc, offload,  UCT_IB_MLX5_XRQC_OFFLOAD_RNDV);
    UCT_IB_MLX5DV_SET(xrqc, xrqc, tag_matching_topology_context.log_matching_list_sz,
                                  ucs_ilog2(iface->tm.num_tags));
    UCT_IB_MLX5DV_SET(xrqc, xrqc, dc,       dc);
    UCT_IB_MLX5DV_SET(xrqc, xrqc, cqn,      dvcq.cqn);

    wq = UCT_IB_MLX5DV_ADDR_OF(xrqc, xrqc, wq);
    UCT_IB_MLX5DV_SET(wq, wq, log_wq_sz,     ucs_ilog2(max));
    UCT_IB_MLX5DV_SET(wq, wq, log_wq_stride, ucs_ilog2(UCT_IB_MLX5_SRQ_STRIDE));
    UCT_IB_MLX5DV_SET(wq, wq, pd,            dvpd.pdn);
    UCT_IB_MLX5DV_SET(wq, wq, dbr_umem_id,   iface->rx.srq.devx.dbrec->mem_id);
    UCT_IB_MLX5DV_SET64(wq, wq, dbr_addr,    iface->rx.srq.devx.dbrec->offset);
    UCT_IB_MLX5DV_SET(wq, wq, wq_umem_id,    iface->rx.srq.devx.mem->umem_id);

    iface->rx.srq.devx.obj = mlx5dv_devx_obj_create(dev->ibv_context, in, sizeof(in),
                                                    out, sizeof(out));
    if (iface->rx.srq.devx.obj == NULL) {
        ucs_error("mlx5dv_devx_obj_create(SRQ) failed, syndrome %x: %m",
                  UCT_IB_MLX5DV_GET(create_xrq_out, out, syndrome));
        status = UCS_ERR_IO_ERROR;
        goto err_free;
    }

    iface->rx.srq.type        = UCT_IB_MLX5_OBJ_TYPE_DEVX;
    iface->rx.srq.srq_num     = UCT_IB_MLX5DV_GET(create_xrq_out, out, xrqn);
    uct_rc_mlx5_iface_tm_set_cmd_qp_len(iface);
    uct_ib_mlx5_srq_buff_init(&iface->rx.srq, 0, max - 1,
                              iface->super.super.config.seg_size);
    iface->super.rx.srq.quota = max - 1;
    return UCS_OK;

err_free:
    ucs_mpool_put_inline(iface->rx.srq.devx.dbrec);
err_free_mem:
    mlx5dv_devx_umem_dereg(iface->rx.srq.devx.mem);
err_free_buf:
    ucs_free(iface->rx.srq.buf);
    return status;
}

ucs_status_t
uct_rc_mlx5_iface_common_devx_connect_qp(uct_rc_mlx5_iface_common_t *iface,
                                         uct_ib_mlx5_qp_t *qp,
                                         uint32_t dest_qp_num,
                                         const struct ibv_ah_attr *ah_attr)
{
    uint32_t in_2rtr[UCT_IB_MLX5DV_ST_SZ_DW(init2rtr_qp_in)] = {};
    uint32_t out_2rtr[UCT_IB_MLX5DV_ST_SZ_DW(init2rtr_qp_out)] = {};
    uint32_t in_2rts[UCT_IB_MLX5DV_ST_SZ_DW(rtr2rts_qp_in)] = {};
    uint32_t out_2rts[UCT_IB_MLX5DV_ST_SZ_DW(rtr2rts_qp_out)] = {};
    const uint8_t *gid = ah_attr->grh.dgid.raw;
    uint8_t mac[6];
    void *qpc;
    int ret;

    ucs_assert_always(qp->type == UCT_IB_MLX5_OBJ_TYPE_DEVX);
    UCT_IB_MLX5DV_SET(init2rtr_qp_in, in_2rtr, opcode, UCT_IB_MLX5_CMD_OP_INIT2RTR_QP);
    UCT_IB_MLX5DV_SET(init2rtr_qp_in, in_2rtr, qpn, qp->qp_num);
    UCT_IB_MLX5DV_SET(init2rtr_qp_in, in_2rtr, opt_param_mask, 14);

    qpc = UCT_IB_MLX5DV_ADDR_OF(init2rtr_qp_in, in_2rtr, qpc);
    UCT_IB_MLX5DV_SET(qpc, qpc, mtu, 5);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_msg_max, 30);
    UCT_IB_MLX5DV_SET(qpc, qpc, remote_qpn, dest_qp_num);
    if (uct_ib_iface_is_roce(&iface->super.super)) {
        /* TODO resolve L3->L2 address */
        mac[0] = gid[8] ^ 0x02;
        memcpy(mac + 1, gid + 9, 2);
        memcpy(mac + 3, gid + 13, 3);
        memcpy(UCT_IB_MLX5DV_ADDR_OF(qpc, qpc, primary_address_path.rmac_47_32), mac, 6);
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.hop_limit, 1);
    } else {
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.grh, ah_attr->is_global);
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.rlid, ah_attr->dlid);
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.mlid, ah_attr->src_path_bits & 0x7f);
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.hop_limit, ah_attr->grh.hop_limit);
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
    UCT_IB_MLX5DV_SET(qpc, qpc, min_rnr_nak, iface->super.config.min_rnr_timer);

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
    UCT_IB_MLX5DV_SET(qpc, qpc, log_sra_max, ucs_ilog2_or0(iface->super.config.max_rd_atomic));
    UCT_IB_MLX5DV_SET(qpc, qpc, retry_count, iface->super.config.retry_cnt);
    UCT_IB_MLX5DV_SET(qpc, qpc, rnr_retry, iface->super.config.rnr_retry);
    UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.ack_timeout, iface->super.config.timeout);
    UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.log_rtm, iface->super.config.exp_backoff);

    ret = mlx5dv_devx_obj_modify(qp->devx.obj, in_2rts, sizeof(in_2rts),
                                 out_2rts, sizeof(out_2rts));
    if (ret) {
        ucs_error("mlx5dv_devx_obj_modify(2RTS) failed, syndrome %x: %m",
                  UCT_IB_MLX5DV_GET(rtr2rts_qp_out, out_2rts, syndrome));
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

