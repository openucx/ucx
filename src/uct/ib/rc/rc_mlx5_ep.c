/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_mlx5.h"

#include <arpa/inet.h> /* For htonl */


/*
 * Generic inline posting function.
 * Parameters which are not relevant to the opcode are ignored.
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_ep_inline_post(uct_ep_h tl_ep, unsigned opcode,
                           void *buffer, unsigned length,
                           /* SEND */ uint8_t am_id, uint64_t am_hdr,
                           /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey
                           )
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    struct mlx5_wqe_ctrl_seg     *ctrl;
    struct mlx5_wqe_raddr_seg    *raddr;
    struct mlx5_wqe_inl_data_seg *inl;
    uct_rc_mlx5_iface_t *iface;
    unsigned wqe_size, hdr_len;
    uint16_t sw_pi, sw_pi_n;
    uint64_t *src, *dst;
    unsigned i;
    int force_sig;

    sw_pi = ep->tx.sw_pi;
    if (UCS_CIRCULAR_COMPARE16(sw_pi, >=, ep->tx.max_pi)) {
        return UCS_ERR_WOULD_BLOCK;
    }

    switch (opcode) {
    case MLX5_OPCODE_SEND:
        hdr_len     = sizeof(uct_rc_hdr_t) + sizeof(uint64_t);
        wqe_size    = (sizeof(*ctrl) + sizeof(*inl) + hdr_len + length + 15) / 16;
        break;
    case MLX5_OPCODE_RDMA_WRITE:
        hdr_len     = 0;
        wqe_size    = (sizeof(*ctrl) + sizeof(*raddr) + sizeof(*inl) + length + 15) / 16;
        if (length == 0) {
            /* Should not have data segments when data is empty */
            wqe_size    = (sizeof(*ctrl) + sizeof(*raddr) + 15) / 16;
        }
        break;
    case MLX5_OPCODE_NOP:
        hdr_len     = 0;
        wqe_size    = (sizeof(*ctrl) + 15) / 16;
        break;
    default:
        hdr_len     = 0;
        wqe_size    = 0;
        break;
    }
    ucs_assert(wqe_size <= MLX5_SEND_WQE_BB);

    ctrl    = ep->tx.seg;
    sw_pi_n = htons(sw_pi);

    /* Build WQE */
    ctrl->opmod_idx_opcode   = (sw_pi_n << 8) | (opcode << 24);
    ctrl->qpn_ds             = htonl(wqe_size) | ep->qpn_ds;
    switch (opcode) {
    case MLX5_OPCODE_SEND:
        iface = ucs_derived_of(ep->super.super.iface, uct_rc_mlx5_iface_t);
        force_sig            = 0;
        if (ep->super.tx.unsignaled >= iface->super.config.tx_moderation) {
            /* TODO put this check logic in rc_ep */
            ctrl->fm_ce_se   = MLX5_WQE_CTRL_CQ_UPDATE;
        } else {
            ctrl->fm_ce_se   = 0;
        }
        inl = (void*)(ctrl + 1);
        uct_rc_am_short_pack(inl + 1, am_id, am_hdr); /* AM id + header */
        break;
    case MLX5_OPCODE_RDMA_WRITE:
        force_sig            = 1;
        ctrl->fm_ce_se       = MLX5_WQE_CTRL_CQ_UPDATE;
        raddr = (struct mlx5_wqe_raddr_seg*)(ctrl + 1);
        raddr->raddr         = htonll(rdma_raddr);
        raddr->rkey          = (uint32_t)rdma_rkey;
        inl = (void*)(raddr + 1);
        break;
    case MLX5_OPCODE_NOP:
        force_sig            = 1;
        ctrl->fm_ce_se       = MLX5_WQE_CTRL_CQ_UPDATE;
        inl = (void*)(ctrl + 1);
        break;
    default:
        inl = (void*)(ctrl + 1);
        break;
    }

    /* Payload */
    inl->byte_count = htonl((length + hdr_len) | MLX5_INLINE_SEG);
    memcpy((void*)(inl + 1) + hdr_len, buffer, length);

    /* Write doorbell record */
    ucs_compiler_fence(); /* Put memory store fence here too, to prevent WC being flushed after DBrec */
    *ep->tx.dbrec = sw_pi_n << 16;

    /* Make sure that doorbell record is written before ringing the doorbell */
    ucs_memory_bus_store_fence();

    /* Set up copy pointers */
    dst = ep->tx.bf_reg;
    src = (void*)ctrl;

    /* BF copy */
    for (i = 0; i < MLX5_SEND_WQE_BB / sizeof(*dst); ++i) {
        *dst++ = *src++;
    }

    /* We don't want the compiler to reorder instructions and hurt our latency */
    ucs_compiler_fence();

    /* Flip BF register */
    ep->tx.bf_reg = (void*) ((uintptr_t) ep->tx.bf_reg ^ ep->tx.bf_size);

    /* Advance queue pointer */
    if (ucs_unlikely((ep->tx.seg += MLX5_SEND_WQE_BB) >= ep->tx.qend)) {
        ep->tx.seg = ep->tx.qstart;
    }

    /* Completion counters */
    iface = ucs_derived_of(ep->super.super.iface, uct_rc_mlx5_iface_t);
    ++ep->tx.sw_pi;
    ++iface->super.tx.outstanding;
    if (force_sig || (ctrl->fm_ce_se & MLX5_WQE_CTRL_CQ_UPDATE)) {
        ep->super.tx.unsignaled = 0;
        ++iface->super.tx.sig_outstanding;
    } else {
        ++ep->super.tx.unsignaled;
    }
    return UCS_OK;
}

ucs_status_t uct_rc_mlx5_ep_put_short(uct_ep_h tl_ep, void *buffer, unsigned length,
                                      uint64_t remote_addr, uct_rkey_t rkey)
{
    return uct_rc_mlx5_ep_inline_post(tl_ep, MLX5_OPCODE_RDMA_WRITE, buffer,
                                      length, 0, 0, remote_addr, rkey);
}

ucs_status_t uct_rc_mlx5_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                     void *buffer, unsigned length)
{
    return uct_rc_mlx5_ep_inline_post(tl_ep, MLX5_OPCODE_SEND, buffer, length,
                                      id, hdr, 0, 0);
}

ucs_status_t uct_rc_mlx5_ep_flush(uct_ep_h tl_ep)
{
    return uct_rc_mlx5_ep_inline_post(tl_ep, MLX5_OPCODE_NOP, NULL, 0, 0, 0, 0, 0);
}

static UCS_CLASS_INIT_FUNC(uct_rc_mlx5_ep_t, uct_iface_h tl_iface)
{
    uct_ib_mlx5_qp_info_t qp_info;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(tl_iface);

    status = uct_ib_mlx5_get_qp_info(self->super.qp, &qp_info);
    if (status != UCS_OK) {
        ucs_error("Failed to get mlx5 QP information");
        return status;
    }

    if ((qp_info.bf.size == 0) || !ucs_is_pow2(qp_info.bf.size) ||
        (qp_info.sq.stride != MLX5_SEND_WQE_BB) ||
        !ucs_is_pow2(qp_info.sq.wqe_cnt))
    {
        ucs_error("mlx5 device parameters not suitable for transport");
        return UCS_ERR_IO_ERROR;
    }

    self->qpn_ds     = htonl(self->super.qp->qp_num << 8);
    self->tx.qstart  = qp_info.sq.buf;
    self->tx.qend    = qp_info.sq.buf + (MLX5_SEND_WQE_BB *  qp_info.sq.wqe_cnt);
    self->tx.seg     = self->tx.qstart;
    self->tx.sw_pi   = 0;
    self->tx.wqe_cnt = qp_info.sq.wqe_cnt;
    self->tx.max_pi  = self->tx.wqe_cnt;
    self->tx.bf_reg  = qp_info.bf.reg;
    self->tx.bf_size = qp_info.bf.size;
    self->tx.dbrec   = &qp_info.dbrec[MLX5_SND_DBR];

    memset(self->tx.qstart, 0, self->tx.qend - self->tx.qstart);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_ep_t)
{
}

UCS_CLASS_DEFINE(uct_rc_mlx5_ep_t, uct_rc_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_mlx5_ep_t, uct_ep_t, uct_iface_h);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_mlx5_ep_t, uct_ep_t);

