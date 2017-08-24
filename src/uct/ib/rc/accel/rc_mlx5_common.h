/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RC_MLX5_COMMON_H
#define UCT_RC_MLX5_COMMON_H

#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/rc/base/rc_ep.h>
#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/ib/mlx5/ib_mlx5.inl>
#include <uct/ib/mlx5/ib_mlx5_log.h>


#define UCT_RC_MLX5_OPCODE_FLAG_RAW   0x100
#define UCT_RC_MLX5_OPCODE_MASK       0xff

#define UCT_RC_MLX5_PUT_MAX_SHORT(_av_size) \
    UCT_IB_MLX5_MAX_BB * MLX5_SEND_WQE_BB - \
    (sizeof(struct mlx5_wqe_ctrl_seg) + \
     (_av_size) + \
     sizeof(struct mlx5_wqe_inl_data_seg) + \
     sizeof(struct mlx5_wqe_raddr_seg))

#define UCT_RC_MLX5_AM_MAX_SHORT(_av_size) \
    UCT_IB_MLX5_MAX_BB * MLX5_SEND_WQE_BB - \
    (sizeof(struct mlx5_wqe_ctrl_seg) + \
     (_av_size) + \
     sizeof(struct mlx5_wqe_inl_data_seg))

#define UCT_RC_MLX5_CHECK_AM_ZCOPY(_id, _header_length, _length, _seg_size, _av_size) \
    UCT_RC_CHECK_AM_ZCOPY(_id, _header_length, _length, \
                          UCT_IB_MLX5_AM_MAX_HDR(_av_size), _seg_size)

#define UCT_RC_MLX5_CHECK_AM_SHORT(_id, _length, _av_size) \
    UCT_RC_CHECK_AM_SHORT(_id, _length, UCT_RC_MLX5_AM_MAX_SHORT(_av_size))


/* there is no need to do a special check for length == 0 because in that
 * case wqe size is valid: inl + raddr + dgram + ctrl fit in 2 WQ BB
 */
#define UCT_RC_MLX5_CHECK_PUT_SHORT(_length, _av_size) \
    UCT_CHECK_LENGTH(_length, 0, UCT_RC_MLX5_PUT_MAX_SHORT(_av_size), "put_short")


enum {
    UCT_RC_MLX5_IFACE_STAT_RX_INL_32,
    UCT_RC_MLX5_IFACE_STAT_RX_INL_64,
    UCT_RC_MLX5_IFACE_STAT_LAST
};

typedef struct uct_rc_mlx5_iface_common {
    struct {
        uct_ib_mlx5_cq_t   cq;
        ucs_mpool_t        atomic_desc_mp;
    } tx;
    struct {
        uct_ib_mlx5_cq_t   cq;
        uct_ib_mlx5_srq_t  srq;
    } rx;
    UCS_STATS_NODE_DECLARE(stats);
} uct_rc_mlx5_iface_common_t;


unsigned uct_rc_mlx5_iface_srq_post_recv(uct_rc_iface_t *iface, uct_ib_mlx5_srq_t *srq);

ucs_status_t uct_rc_mlx5_iface_common_init(uct_rc_mlx5_iface_common_t *iface,
                                           uct_rc_iface_t *rc_iface,
                                           uct_rc_iface_config_t *config);

void uct_rc_mlx5_iface_common_cleanup(uct_rc_mlx5_iface_common_t *iface);

void uct_rc_mlx5_iface_common_query(uct_rc_iface_t *iface,
                                    uct_iface_attr_t *iface_attr, size_t av_size);

void uct_rc_mlx5_iface_common_update_cqs_ci(uct_rc_mlx5_iface_common_t *iface,
                                            uct_ib_iface_t *ib_iface);

void uct_rc_mlx5_iface_common_sync_cqs_ci(uct_rc_mlx5_iface_common_t *iface,
                                          uct_ib_iface_t *ib_iface);

void uct_rc_mlx5_iface_commom_clean_srq(uct_rc_mlx5_iface_common_t *mlx5_common_iface,
                                        uct_rc_iface_t *rc_iface, uint32_t qpn);

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_txqp_process_tx_cqe(uct_rc_txqp_t *txqp, struct mlx5_cqe64 *cqe,
                                uint16_t hw_ci)
{
    if (cqe->op_own & MLX5_INLINE_SCATTER_32) {
        uct_rc_txqp_completion_inl_resp(txqp, cqe, hw_ci);
    } else if (cqe->op_own & MLX5_INLINE_SCATTER_64) {
        uct_rc_txqp_completion_inl_resp(txqp, cqe - 1, hw_ci);
    } else {
        uct_rc_txqp_completion_desc(txqp, hw_ci);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_common_rx_inline(uct_rc_mlx5_iface_common_t *mlx5_iface,
                                   uct_rc_iface_t *rc_iface,
                                   uct_ib_iface_recv_desc_t *desc,
                                   int stats_counter, unsigned byte_len)
{
    UCS_STATS_UPDATE_COUNTER(mlx5_iface->stats, stats_counter, 1);
    VALGRIND_MAKE_MEM_UNDEFINED(uct_ib_iface_recv_desc_hdr(&rc_iface->super, desc),
                                byte_len);
}

static UCS_F_ALWAYS_INLINE unsigned
uct_rc_mlx5_iface_common_poll_rx(uct_rc_mlx5_iface_common_t *mlx5_common_iface,
                                 uct_rc_iface_t *rc_iface)
{
    uct_ib_mlx5_srq_seg_t *seg;
    uct_ib_iface_recv_desc_t *desc;
    uct_rc_iface_ops_t *rc_ops;
    uct_rc_hdr_t *hdr;
    struct mlx5_cqe64 *cqe;
    unsigned byte_len;
    unsigned qp_num;
    uint16_t wqe_ctr;
    uint16_t max_batch;
    ucs_status_t status;
    unsigned count;
    void *udesc;
    unsigned flags;

    ucs_assert(uct_ib_mlx5_srq_get_wqe(&mlx5_common_iface->rx.srq,
                                       mlx5_common_iface->rx.srq.mask)->srq.next_wqe_index == 0);

    cqe = uct_ib_mlx5_poll_cq(&rc_iface->super, &mlx5_common_iface->rx.cq);
    if (cqe == NULL) {
        /* If no CQE - post receives */
        count = 0;
        goto done;
    }

    ucs_memory_cpu_load_fence();
    UCS_STATS_UPDATE_COUNTER(rc_iface->stats, UCT_RC_IFACE_STAT_RX_COMPLETION, 1);

    byte_len = ntohl(cqe->byte_cnt);
    wqe_ctr  = ntohs(cqe->wqe_counter);
    seg      = uct_ib_mlx5_srq_get_wqe(&mlx5_common_iface->rx.srq, wqe_ctr);
    desc     = seg->srq.desc;

    /* Get a pointer to AM header (after which comes the payload)
     * Support cases of inline scatter by pointing directly to CQE.
     */
    if (cqe->op_own & MLX5_INLINE_SCATTER_32) {
        hdr = (uct_rc_hdr_t*)(cqe);
        uct_rc_mlx5_iface_common_rx_inline(mlx5_common_iface, rc_iface, desc,
                                           UCT_RC_MLX5_IFACE_STAT_RX_INL_32, byte_len);
        flags = 0;
    } else if (cqe->op_own & MLX5_INLINE_SCATTER_64) {
        hdr = (uct_rc_hdr_t*)(cqe - 1);
        uct_rc_mlx5_iface_common_rx_inline(mlx5_common_iface, rc_iface, desc,
                                           UCT_RC_MLX5_IFACE_STAT_RX_INL_64, byte_len);
        flags = 0;
    } else {
        hdr = uct_ib_iface_recv_desc_hdr(&rc_iface->super, desc);
        VALGRIND_MAKE_MEM_DEFINED(hdr, byte_len);
        flags = UCT_CB_PARAM_FLAG_DESC;
    }

    uct_ib_mlx5_log_rx(&rc_iface->super, IBV_QPT_RC, cqe, hdr,
                       uct_rc_ep_am_packet_dump);

    if (ucs_unlikely(hdr->am_id & UCT_RC_EP_FC_MASK)) {
        qp_num = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
        rc_ops = ucs_derived_of(rc_iface->super.ops, uct_rc_iface_ops_t);

        status = rc_ops->fc_handler(rc_iface, qp_num, hdr, byte_len - sizeof(*hdr),
                                    cqe->imm_inval_pkey, cqe->slid, flags);
    } else {
        status = uct_iface_invoke_am(&rc_iface->super.super, hdr->am_id,
                                     hdr + 1, byte_len - sizeof(*hdr),
                                     flags);
    }

    if ((status == UCS_OK) &&
        (wqe_ctr == ((mlx5_common_iface->rx.srq.ready_idx + 1) &
                      mlx5_common_iface->rx.srq.mask))) {
        /* If the descriptor was not used - if there are no "holes", we can just
         * reuse it on the receive queue. Otherwise, ready pointer will stay behind
         * until post_recv allocated more descriptors from the memory pool, fills
         * the holes, and moves it forward.
         */
        ucs_assert(wqe_ctr == ((mlx5_common_iface->rx.srq.free_idx + 1) &
                                mlx5_common_iface->rx.srq.mask));
        ++mlx5_common_iface->rx.srq.ready_idx;
        ++mlx5_common_iface->rx.srq.free_idx;
   } else {
        if (status != UCS_OK) {
            udesc = (char*)desc + rc_iface->super.config.rx_headroom_offset;
            uct_recv_desc(udesc) = &rc_iface->super.release_desc;
            seg->srq.desc        = NULL;
        }
        if (wqe_ctr == ((mlx5_common_iface->rx.srq.free_idx + 1) & mlx5_common_iface->rx.srq.mask)) {
            ++mlx5_common_iface->rx.srq.free_idx;
        } else {
            /* Mark the segment as out-of-order, post_recv will advance free */
            seg->srq.free = 1;
        }
    }

    ++rc_iface->rx.srq.available;
    count = 1;

done:
    max_batch = rc_iface->super.config.rx_max_batch;
    if (rc_iface->rx.srq.available >= max_batch) {
        uct_rc_mlx5_iface_srq_post_recv(rc_iface, &mlx5_common_iface->rx.srq);
    }
    return count;
}


static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_common_post_send(uct_rc_iface_t *iface, enum ibv_qp_type qp_type,
                             uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                             uint8_t opcode, uint8_t opmod, uint8_t fm_ce_se,
                             size_t wqe_size, uct_ib_mlx5_base_av_t *av,
                             uint32_t imm)
{
    struct mlx5_wqe_ctrl_seg *ctrl;
    uint16_t posted;

    ctrl = txwq->curr;

    if (opcode == MLX5_OPCODE_SEND_IMM) {
        uct_ib_mlx5_set_ctrl_seg_with_imm(ctrl, txwq->sw_pi, opcode, opmod,
                                          txqp->qp->qp_num, fm_ce_se, wqe_size,
                                          imm);
    } else {
        uct_ib_mlx5_set_ctrl_seg(ctrl, txwq->sw_pi, opcode, opmod,
                                 txqp->qp->qp_num, fm_ce_se, wqe_size);
    }

    if (qp_type == IBV_EXP_QPT_DC_INI) {
        uct_ib_mlx5_set_dgram_seg((void*)(ctrl + 1), av, NULL, qp_type);
    }

    uct_ib_mlx5_log_tx(&iface->super, qp_type, ctrl, txwq->qstart, txwq->qend,
                       ((opcode == MLX5_OPCODE_SEND) || (opcode == MLX5_OPCODE_SEND_IMM)) ?
                       uct_rc_ep_am_packet_dump : NULL);

    posted = uct_ib_mlx5_post_send(txwq, ctrl, wqe_size);
    if (fm_ce_se & MLX5_WQE_CTRL_CQ_UPDATE) {
        txwq->sig_pi = txwq->sw_pi - posted;
    }

    uct_rc_txqp_posted(txqp, iface, posted, fm_ce_se & MLX5_WQE_CTRL_CQ_UPDATE);
}


/*
 * Generic function that setups and posts WQE with inline segment
 * Parameters which are not relevant to the opcode are ignored.
 *
 *            +--------+-----+-------+--------+------------
 * SEND       | CTRL   | INL | am_id | am_hdr | payload ...
 *            +--------+-----+---+---+-+-------+-----------
 * RDMA_WRITE | CTRL   | RADDR   | INL | payload ...
 *            +--------+---------+-----+-------------------
 *
 * CTRL is mlx5_wqe_ctrl_seg for RC and
 *         mlx5_wqe_ctrl_seg + mlx5_wqe_datagram_seg for DC
 *
 * NOTE: switch is optimized away during inlining because opcode
 * is a compile time constant
 */
static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_txqp_inline_post(uct_rc_iface_t *iface, enum ibv_qp_type qp_type,
                             uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                             unsigned opcode, const void *buffer, unsigned length,
                  /* SEND */ uint8_t am_id, uint64_t am_hdr, uint32_t imm_val_be,
                  /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey,
                  /* AV   */ uct_ib_mlx5_base_av_t *av, size_t av_size,
                             unsigned fm_ce_se)
{
    struct mlx5_wqe_ctrl_seg     *ctrl;
    struct mlx5_wqe_raddr_seg    *raddr;
    struct mlx5_wqe_inl_data_seg *inl;
    uct_rc_am_short_hdr_t        *am;
    uct_rc_hdr_t                 *rc_hdr;
    size_t wqe_size, ctrl_av_size;
    void *next_seg;

    ctrl         = txwq->curr;
    ctrl_av_size = sizeof(*ctrl) + av_size;
    next_seg     = uct_ib_mlx5_txwq_wrap_exact(txwq, (void*)ctrl + ctrl_av_size);

    switch (opcode) {
    case MLX5_OPCODE_SEND_IMM:
        /* Fall through to MLX5_OPCODE_SEND handler */
    case MLX5_OPCODE_SEND:
        /* Set inline segment which has AM id, AM header, and AM payload */
        wqe_size         = ctrl_av_size + sizeof(*inl) + sizeof(*am) + length;
        inl              = next_seg;
        inl->byte_count  = htonl((length + sizeof(*am)) | MLX5_INLINE_SEG);
        am               = (void*)(inl + 1);
        am->rc_hdr.am_id = am_id;
        am->am_hdr       = am_hdr;
        uct_ib_mlx5_inline_copy(am + 1, buffer, length, txwq);
        fm_ce_se        |= uct_rc_iface_tx_moderation(iface, txqp, MLX5_WQE_CTRL_CQ_UPDATE);
        break;

    case MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_RAW:
        /* Send empty AM with just AM id (used by FC) */
        wqe_size         = ctrl_av_size + sizeof(*inl) + sizeof(*rc_hdr);
        inl              = next_seg;
        inl->byte_count  = htonl(sizeof(*rc_hdr) | MLX5_INLINE_SEG);
        rc_hdr           = (void*)(inl + 1);
        rc_hdr->am_id    = am_id;
        fm_ce_se        |= uct_rc_iface_tx_moderation(iface, txqp, MLX5_WQE_CTRL_CQ_UPDATE);
        break;

    case MLX5_OPCODE_RDMA_WRITE:
        /* Set RDMA segment */
        if (length == 0) {
            wqe_size     = ctrl_av_size + sizeof(*raddr);
        } else {
            wqe_size     = ctrl_av_size + sizeof(*raddr) + sizeof(*inl) + length;
        }
        raddr            = next_seg;
        uct_ib_mlx5_ep_set_rdma_seg(raddr, rdma_raddr, rdma_rkey);
        inl              = uct_ib_mlx5_txwq_wrap_none(txwq, raddr + 1);
        inl->byte_count  = htonl(length | MLX5_INLINE_SEG);
        uct_ib_mlx5_inline_copy(inl + 1, buffer, length, txwq);
        fm_ce_se        |= MLX5_WQE_CTRL_CQ_UPDATE;
        break;

    case MLX5_OPCODE_NOP:
        /* Empty inline segment */
        wqe_size         = sizeof(*ctrl) + av_size;
        inl              = next_seg;
        inl->byte_count  = htonl(MLX5_INLINE_SEG);
        fm_ce_se        |= MLX5_WQE_CTRL_CQ_UPDATE | MLX5_WQE_CTRL_FENCE;
        break;

    default:
        ucs_fatal("invalid send opcode");
    }

    uct_rc_mlx5_common_post_send(iface, qp_type, txqp, txwq, opcode, 0, fm_ce_se,
                                 wqe_size, av, imm_val_be);
}


/*
 * Generic data-pointer posting function.
 * Parameters which are not relevant to the opcode are ignored.
 *
 *            +--------+-----+-------+--------+-------+
 * SEND       | CTRL   | INL | am_id | am_hdr | DPSEG |
 *            +--------+-----+---+---+----+----+------+
 * RDMA_WRITE | CTRL   | RADDR   | DPSEG  |
 *            +--------+---------+--------+-------+
 * ATOMIC     | CTRL   | RADDR   | ATOMIC | DPSEG |
 *            +--------+---------+--------+-------+
 *
 * CTRL is mlx5_wqe_ctrl_seg for RC and
 *         mlx5_wqe_ctrl_seg + mlx5_wqe_datagram_seg for DC
 *
 * NOTE: switch is optimized away during inlining because opcode_flags
 * is a compile time constant
 */
static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_txqp_dptr_post(uct_rc_iface_t *iface, enum ibv_qp_type qp_type,
                           uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                           unsigned opcode_flags, const void *buffer,
                           unsigned length, uint32_t *lkey_p,
                           /* SEND */ uint8_t am_id, const void *am_hdr, unsigned am_hdr_len,
                           /* RDMA/ATOMIC */ uint64_t remote_addr, uct_rkey_t rkey,
                           /* ATOMIC */ uint64_t compare_mask, uint64_t compare, uint64_t swap_add,
                           /* AV   */ uct_ib_mlx5_base_av_t *av, size_t av_size,
                           uint8_t fm_ce_se)
{
    struct mlx5_wqe_ctrl_seg                     *ctrl;
    struct mlx5_wqe_raddr_seg                    *raddr;
    struct mlx5_wqe_atomic_seg                   *atomic;
    struct mlx5_wqe_data_seg                     *dptr;
    struct mlx5_wqe_inl_data_seg                 *inl;
    struct uct_ib_mlx5_atomic_masked_cswap32_seg *masked_cswap32;
    struct uct_ib_mlx5_atomic_masked_fadd32_seg  *masked_fadd32;
    struct uct_ib_mlx5_atomic_masked_cswap64_seg *masked_cswap64;
    uct_rc_hdr_t                                 *rch;
    size_t  wqe_size, inl_seg_size, ctrl_av_size;
    uint8_t opmod;
    void *next_seg;

    if (!(fm_ce_se & MLX5_WQE_CTRL_CQ_UPDATE)) {
        fm_ce_se |= uct_rc_iface_tx_moderation(iface, txqp, MLX5_WQE_CTRL_CQ_UPDATE);
    }

    opmod         = 0;
    ctrl         = txwq->curr;
    ctrl_av_size = sizeof(*ctrl) + av_size;
    next_seg     = uct_ib_mlx5_txwq_wrap_exact(txwq, (void*)ctrl + ctrl_av_size);

    switch (opcode_flags) {
    case MLX5_OPCODE_SEND:
        inl_seg_size     = ucs_align_up_pow2(sizeof(*inl) + sizeof(*rch) + am_hdr_len,
                                             UCT_IB_MLX5_WQE_SEG_SIZE);

        ucs_assert(ctrl_av_size + inl_seg_size + sizeof(*dptr) <=
                   UCT_IB_MLX5_MAX_BB * MLX5_SEND_WQE_BB);
        ucs_assert(length + sizeof(*rch) + am_hdr_len <=
                   iface->super.config.seg_size);

        /* Inline segment with AM ID and header */
        inl              = next_seg;
        inl->byte_count  = htonl((sizeof(*rch) + am_hdr_len) | MLX5_INLINE_SEG);
        rch              = (uct_rc_hdr_t *)(inl + 1);
        rch->am_id       = am_id;

        uct_ib_mlx5_inline_copy(rch + 1, am_hdr, am_hdr_len, txwq);

        /* Data segment with payload */
        if (length == 0) {
            wqe_size     = ctrl_av_size + inl_seg_size;
        } else {
            wqe_size     = ctrl_av_size + inl_seg_size + sizeof(*dptr);
            dptr         = uct_ib_mlx5_txwq_wrap_any(txwq, (void*)inl + inl_seg_size);
            uct_ib_mlx5_set_data_seg(dptr, buffer, length, *lkey_p);
        }
        break;

    case MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_RAW:
        /* Data segment only */
        ucs_assert(length < (2ul << 30));
        ucs_assert(length <= iface->super.config.seg_size);

        wqe_size = ctrl_av_size + sizeof(struct mlx5_wqe_data_seg);
        uct_ib_mlx5_set_data_seg(next_seg, buffer, length, *lkey_p);
        break;

    case MLX5_OPCODE_RDMA_READ:
    case MLX5_OPCODE_RDMA_WRITE:
        /* Set RDMA segment */
        ucs_assert(length <= UCT_IB_MAX_MESSAGE_SIZE);

        raddr = next_seg;
        uct_ib_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        /* Data segment */
        if (length == 0) {
            wqe_size     = ctrl_av_size + sizeof(*raddr);
        } else {
            /* dptr cannot wrap, because ctrl+av could be either 2 or 4 segs */
            dptr         = uct_ib_mlx5_txwq_wrap_none(txwq, raddr + 1);
            wqe_size     = ctrl_av_size + sizeof(*raddr) + sizeof(*dptr);
            uct_ib_mlx5_set_data_seg(dptr, buffer, length, *lkey_p);
        }
        break;

    case MLX5_OPCODE_ATOMIC_FA:
    case MLX5_OPCODE_ATOMIC_CS:
        ucs_assert(length == sizeof(uint64_t));
        raddr = next_seg;
        uct_ib_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        /* atomic cannot wrap, because ctrl+av could be either 2 or 4 segs */
        atomic = uct_ib_mlx5_txwq_wrap_none(txwq, raddr + 1);
        if (opcode_flags == MLX5_OPCODE_ATOMIC_CS) {
            atomic->compare = compare;
        }
        atomic->swap_add    = swap_add;

        dptr                = uct_ib_mlx5_txwq_wrap_exact(txwq, atomic + 1);
        uct_ib_mlx5_set_data_seg(dptr, buffer, length, *lkey_p);
        wqe_size            = ctrl_av_size + sizeof(*raddr) + sizeof(*atomic) +
                              sizeof(*dptr);
        break;

    case MLX5_OPCODE_ATOMIC_MASKED_CS:
        raddr = next_seg;
        uct_ib_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        switch (length) {
        case sizeof(uint32_t):
            opmod                        = UCT_IB_MLX5_OPMOD_EXT_ATOMIC(2);
            masked_cswap32               = uct_ib_mlx5_txwq_wrap_none(txwq, raddr + 1);
            masked_cswap32->swap         = swap_add;
            masked_cswap32->compare      = compare;
            masked_cswap32->swap_mask    = (uint32_t)-1;
            masked_cswap32->compare_mask = compare_mask;
            dptr                         = uct_ib_mlx5_txwq_wrap_exact(txwq, masked_cswap32 + 1);
            wqe_size                     = ctrl_av_size + sizeof(*raddr) +
                                           sizeof(*masked_cswap32) + sizeof(*dptr);
            break;
        case sizeof(uint64_t):
            opmod                        = UCT_IB_MLX5_OPMOD_EXT_ATOMIC(3); /* Ext. atomic, size 2**3 */
            masked_cswap64               = uct_ib_mlx5_txwq_wrap_none(txwq, raddr + 1);
            masked_cswap64->swap         = swap_add;
            masked_cswap64->compare      = compare;

            /* 2nd half of masked_cswap64 can wrap */
            masked_cswap64               = uct_ib_mlx5_txwq_wrap_exact(txwq, masked_cswap64 + 1);
            masked_cswap64->swap         = (uint64_t)-1;
            masked_cswap64->compare      = compare_mask;

            dptr                         = uct_ib_mlx5_txwq_wrap_exact(txwq, masked_cswap64 + 1);
            wqe_size                     = ctrl_av_size + sizeof(*raddr) +
                                           2 * sizeof(*masked_cswap64) + sizeof(*dptr);
            break;
        default:
            ucs_fatal("invalid atomic type length %d", length);
        }
        uct_ib_mlx5_set_data_seg(dptr, buffer, length, *lkey_p);
        break;

     case MLX5_OPCODE_ATOMIC_MASKED_FA:
        ucs_assert(length == sizeof(uint32_t));
        raddr = uct_ib_mlx5_txwq_wrap_exact(txwq, (void*)ctrl + ctrl_av_size);
        uct_ib_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        opmod                         = UCT_IB_MLX5_OPMOD_EXT_ATOMIC(2);
        masked_fadd32                 = uct_ib_mlx5_txwq_wrap_none(txwq, raddr + 1);
        masked_fadd32->add            = swap_add;
        masked_fadd32->filed_boundary = 0;

        dptr                          = uct_ib_mlx5_txwq_wrap_exact(txwq, masked_fadd32 + 1);
        wqe_size                      = ctrl_av_size + sizeof(*raddr) +
                                        sizeof(*masked_fadd32) + sizeof(*dptr);
        uct_ib_mlx5_set_data_seg(dptr, buffer, length, *lkey_p);
        break;

    default:
        ucs_fatal("invalid send opcode");
    }

    uct_rc_mlx5_common_post_send(iface, qp_type, txqp, txwq,
                                 (opcode_flags & UCT_RC_MLX5_OPCODE_MASK),
                                 opmod, fm_ce_se, wqe_size, av, 0);
}

static UCS_F_ALWAYS_INLINE
void uct_rc_mlx5_txqp_dptr_post_iov(uct_rc_iface_t *iface, enum ibv_qp_type qp_type,
                                    uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                                    unsigned opcode_flags,
                         /* IOV  */ const uct_iov_t *iov, size_t iovcnt,
                         /* SEND */ uint8_t am_id, const void *am_hdr, unsigned am_hdr_len,
                         /* RDMA */ uint64_t remote_addr, uct_rkey_t rkey,
                         /* AV   */ uct_ib_mlx5_base_av_t *av, size_t av_size,
                                    uint8_t fm_ce_se)
{
    struct mlx5_wqe_ctrl_seg     *ctrl;
    struct mlx5_wqe_raddr_seg    *raddr;
    struct mlx5_wqe_data_seg     *dptr;
    struct mlx5_wqe_inl_data_seg *inl;
    uct_rc_hdr_t                 *rch;
    unsigned                      wqe_size, inl_seg_size, ctrl_av_size;
    void                         *next_seg;

    if (!(fm_ce_se & MLX5_WQE_CTRL_CQ_UPDATE)) {
        fm_ce_se |= uct_rc_iface_tx_moderation(iface, txqp, MLX5_WQE_CTRL_CQ_UPDATE);
    }

    ctrl         = txwq->curr;
    ctrl_av_size = sizeof(*ctrl) + av_size;
    next_seg     = uct_ib_mlx5_txwq_wrap_exact(txwq, (void*)ctrl + ctrl_av_size);

    switch (opcode_flags) {
    case MLX5_OPCODE_SEND:
        inl_seg_size     = ucs_align_up_pow2(sizeof(*inl) + sizeof(*rch) + am_hdr_len,
                                             UCT_IB_MLX5_WQE_SEG_SIZE);

        ucs_assert(uct_iov_total_length(iov, iovcnt) + sizeof(*rch) + am_hdr_len <=
                   iface->super.config.seg_size);

        /* Inline segment with AM ID and header */
        inl              = next_seg;
        inl->byte_count  = htonl((sizeof(*rch) + am_hdr_len) | MLX5_INLINE_SEG);
        rch              = (uct_rc_hdr_t *)(inl + 1);
        rch->am_id       = am_id;

        uct_ib_mlx5_inline_copy(rch + 1, am_hdr, am_hdr_len, txwq);

        /* Data segment with payload */
        dptr             = (struct mlx5_wqe_data_seg *)((char *)inl + inl_seg_size);
        wqe_size         = ctrl_av_size + inl_seg_size +
                           uct_ib_mlx5_set_data_seg_iov(txwq, dptr, iov, iovcnt);

        ucs_assert(wqe_size <= (UCT_IB_MLX5_MAX_BB * MLX5_SEND_WQE_BB));
        break;

    case MLX5_OPCODE_RDMA_READ:
    case MLX5_OPCODE_RDMA_WRITE:
        /* Set RDMA segment */
        ucs_assert(uct_iov_total_length(iov, iovcnt) <= UCT_IB_MAX_MESSAGE_SIZE);

        raddr            = next_seg;
        uct_ib_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        /* Data segment */
        wqe_size         = ctrl_av_size + sizeof(*raddr) +
                           uct_ib_mlx5_set_data_seg_iov(txwq, (void*)(raddr + 1),
                                                        iov, iovcnt);
        break;

    default:
        ucs_fatal("invalid send opcode");
    }

    uct_rc_mlx5_common_post_send(iface, qp_type, txqp, txwq,
                                 opcode_flags & UCT_RC_MLX5_OPCODE_MASK,
                                 0, fm_ce_se, wqe_size, av, 0);
}

#endif
