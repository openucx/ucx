/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_mlx5.h"

#include <arpa/inet.h> /* For htonl */


ucs_status_t uct_rc_mlx5_ep_put_short(uct_ep_h tl_ep, void *buffer, unsigned length,
                                      uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    uct_rc_mlx5_wqe_rdma_inl_seg_t *seg = ep->tx.seg;
    unsigned wqe_size;
    uint32_t sw_pi_16_n;
    uint64_t *src, *dst;
    unsigned sw_pi, i;

    sw_pi = ep->tx.sw_pi;
    if (UCS_CIRCULAR_COMPARE32(sw_pi, >=, ep->tx.max_pi)) {
        return UCS_ERR_WOULD_BLOCK;
    }

    sw_pi_16_n = htonl(sw_pi & 0xffff);
    wqe_size   = (length + sizeof(*seg) + 15) / 16;
    ucs_assert(wqe_size <= MLX5_SEND_WQE_BB);

    /* Build WQE */
    seg->ctrl.opmod_idx_opcode   = (sw_pi_16_n >> 8) | (MLX5_OPCODE_RDMA_WRITE << 24);
    seg->ctrl.qpn_ds             = htonl(wqe_size) | ep->qpn_ds;
    seg->ctrl.fm_ce_se           = MLX5_WQE_CTRL_CQ_UPDATE; /* Ask for completion */
    seg->raddr.raddr             = htonll(remote_addr);
    seg->raddr.rkey              = (uint32_t)rkey;

    /* Data */
    UCS_STATIC_ASSERT(seg + 1 == ((void*)seg + 32 + 4));
    seg->inl.byte_count          = htonl(length | MLX5_INLINE_SEG);
    memcpy(seg + 1, buffer, length);

    /* Write doorbell record */
    ucs_compiler_fence(); /* Put memory store fence here too, to prevent WC being flushed after DBrec */
    *ep->tx.dbrec = sw_pi_16_n;

    /* Make sure that doorbell record is written before ringing the doorbell */
    ucs_memory_bus_store_fence();

    /* Set up copy pointers */
    dst = ep->tx.bf_reg;
    src = (void*)seg;

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
    ++ep->tx.sw_pi;
    ++ucs_derived_of(ep->super.super.iface, uct_rc_iface_t)->tx.outstanding;

    return UCS_OK;
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
    self->tx.max_pi  = qp_info.sq.wqe_cnt;
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

