/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rc_mlx5.h"
#include "rc_mlx5_common.h"

#include <uct/ib/mlx5/ib_mlx5.inl>
#include <uct/ib/mlx5/ib_mlx5_log.h>

#define UCT_RC_MLX5_EP_DECL(_tl_ep, _iface, _ep) \
    uct_rc_mlx5_ep_t *_ep = ucs_derived_of(_tl_ep, uct_rc_mlx5_ep_t); \
    uct_rc_mlx5_iface_common_t *_iface = ucs_derived_of(_tl_ep->iface, \
                                                        uct_rc_mlx5_iface_common_t)

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_common_update_tx_res(uct_rc_iface_t *rc_iface, uct_ib_mlx5_txwq_t *txwq,
                                 uct_rc_txqp_t *txqp, uint16_t hw_ci)
{
    uint16_t bb_num;

    bb_num = uct_ib_mlx5_txwq_update_bb(txwq, hw_ci) - uct_rc_txqp_available(txqp);

    /* Must always have positive number of released resources. The first completion
     * will report bb_num=1 (because prev_sw_pi is initialized to -1) and all the rest
     * report the amount of BBs the previous WQE has consumed.
     */
    ucs_assertv(bb_num > 0, "hw_ci=%d prev_sw_pi=%d available=%d bb_num=%d",
                hw_ci, txwq->prev_sw_pi, txqp->available, bb_num);

    uct_rc_txqp_available_add(txqp, bb_num);
    ucs_assert(uct_rc_txqp_available(txqp) <= txwq->bb_max);

    rc_iface->tx.cq_available += bb_num;
    ucs_assertv(rc_iface->tx.cq_available <= rc_iface->config.tx_cq_len,
                "cq_available=%d tx_cq_len=%d bb_num=%d txwq=%p txqp=%p",
                rc_iface->tx.cq_available, rc_iface->config.tx_cq_len, bb_num,
                txwq, txqp);
}

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
uct_rc_mlx5_iface_common_rx_inline(uct_rc_mlx5_iface_common_t *iface,
                                   uct_ib_iface_recv_desc_t *desc,
                                   int stats_counter, unsigned byte_len)
{
    UCS_STATS_UPDATE_COUNTER(iface->stats, stats_counter, 1);
    VALGRIND_MAKE_MEM_UNDEFINED(uct_ib_iface_recv_desc_hdr(&iface->super.super, desc),
                                byte_len);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_srq_prefetch_setup(uct_rc_mlx5_iface_common_t *iface)
{
    unsigned wqe_ctr = (iface->rx.srq.free_idx + 2) & iface->rx.srq.mask;
    uct_ib_mlx5_srq_seg_t *seg;

    seg = uct_ib_mlx5_srq_get_wqe(&iface->rx.srq, wqe_ctr);
    iface->rx.pref_ptr =
            uct_ib_iface_recv_desc_hdr(&iface->super.super, seg->srq.desc);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_release_srq_seg(uct_rc_mlx5_iface_common_t *iface,
                                  uct_ib_mlx5_srq_seg_t *seg, uint16_t wqe_ctr,
                                  ucs_status_t status, unsigned offset,
                                  uct_recv_desc_t *release_desc)
{
    void *udesc;

    if (ucs_likely((status == UCS_OK) &&
         (wqe_ctr == ((iface->rx.srq.ready_idx + 1) &
                       iface->rx.srq.mask)))) {
         /* If the descriptor was not used - if there are no "holes", we can just
          * reuse it on the receive queue. Otherwise, ready pointer will stay behind
          * until post_recv allocated more descriptors from the memory pool, fills
          * the holes, and moves it forward.
          */
         ucs_assert(wqe_ctr == ((iface->rx.srq.free_idx + 1) &
                                 iface->rx.srq.mask));
         ++iface->rx.srq.ready_idx;
         ++iface->rx.srq.free_idx;
    } else {
         if (status != UCS_OK) {
             udesc                = (char*)seg->srq.desc + offset;
             uct_recv_desc(udesc) = release_desc;
             seg->srq.desc        = NULL;
         }
         if (wqe_ctr == ((iface->rx.srq.free_idx + 1) & iface->rx.srq.mask)) {
             ++iface->rx.srq.free_idx;
         } else {
             /* Mark the segment as out-of-order, post_recv will advance free */
             seg->srq.free = 1;
         }
    }

    ++iface->super.rx.srq.available;
}

static UCS_F_ALWAYS_INLINE struct mlx5_cqe64*
uct_rc_mlx5_iface_poll_rx_cq(uct_rc_mlx5_iface_common_t *iface)
{
    uct_ib_mlx5_cq_t *cq = &iface->cq[UCT_IB_DIR_RX];
    struct mlx5_cqe64 *cqe;
    unsigned index;
    uint8_t op_own;

    /* Prefetch the descriptor if it was scheduled */
    ucs_prefetch(iface->rx.pref_ptr);

    index  = cq->cq_ci;
    cqe    = uct_ib_mlx5_get_cqe(cq, index);
    op_own = cqe->op_own;

    if (ucs_unlikely(uct_ib_mlx5_cqe_is_hw_owned(op_own, index, cq->cq_length))) {
        return NULL;
    } else if (ucs_unlikely(op_own & UCT_IB_MLX5_CQE_OP_OWN_ERR_MASK)) {
        uct_rc_mlx5_iface_check_rx_completion(iface, cqe);
        return NULL;
    }

    cq->cq_ci = index + 1;
    return cqe; /* TODO optimize - let complier know cqe is not null */
}

static UCS_F_ALWAYS_INLINE void*
uct_rc_mlx5_iface_common_data(uct_rc_mlx5_iface_common_t *iface,
                              struct mlx5_cqe64 *cqe,
                              unsigned byte_len, unsigned *flags)
{
    uct_ib_mlx5_srq_seg_t *seg;
    uct_ib_iface_recv_desc_t *desc;
    void *hdr;

    seg  = uct_ib_mlx5_srq_get_wqe(&iface->rx.srq, ntohs(cqe->wqe_counter));
    desc = seg->srq.desc;

    /* Get a pointer to AM or Tag header (after which comes the payload)
     * Support cases of inline scatter by pointing directly to CQE.
     */
    if (cqe->op_own & MLX5_INLINE_SCATTER_32) {
        hdr = cqe;
        uct_rc_mlx5_iface_common_rx_inline(iface, desc,
                                           UCT_RC_MLX5_IFACE_STAT_RX_INL_32,
                                           byte_len);
        *flags = 0;
    } else if (cqe->op_own & MLX5_INLINE_SCATTER_64) {
        hdr = cqe - 1;
        uct_rc_mlx5_iface_common_rx_inline(iface, desc,
                                           UCT_RC_MLX5_IFACE_STAT_RX_INL_64,
                                           byte_len);
        *flags = 0;
    } else {
        hdr = uct_ib_iface_recv_desc_hdr(&iface->super.super, desc);
        VALGRIND_MAKE_MEM_DEFINED(hdr, byte_len);
        *flags = UCT_CB_PARAM_FLAG_DESC;
        /* Assuming that next packet likely will be non-inline,
         * setup the next prefetch pointer
         */
        uct_rc_mlx5_srq_prefetch_setup(iface);
    }

    return hdr;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_common_am_handler(uct_rc_mlx5_iface_common_t *iface,
                                    struct mlx5_cqe64 *cqe,
                                    uct_rc_mlx5_hdr_t *hdr,
                                    unsigned flags, unsigned byte_len)
{
    uint16_t wqe_ctr;
    uct_rc_iface_ops_t *rc_ops;
    uct_ib_mlx5_srq_seg_t *seg;
    uint32_t qp_num;
    ucs_status_t status;

    wqe_ctr = ntohs(cqe->wqe_counter);
    seg     = uct_ib_mlx5_srq_get_wqe(&iface->rx.srq, wqe_ctr);

    uct_ib_mlx5_log_rx(&iface->super.super, cqe, hdr,
                       uct_rc_mlx5_common_packet_dump);

    if (ucs_unlikely(hdr->rc_hdr.am_id & UCT_RC_EP_FC_MASK)) {
        qp_num = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
        rc_ops = ucs_derived_of(iface->super.super.ops, uct_rc_iface_ops_t);

        /* coverity[overrun-buffer-val] */
        status = rc_ops->fc_handler(&iface->super, qp_num, &hdr->rc_hdr,
                                    byte_len - sizeof(*hdr),
                                    cqe->imm_inval_pkey, cqe->slid, flags);
    } else {
        status = uct_iface_invoke_am(&iface->super.super.super, hdr->rc_hdr.am_id,
                                     hdr + 1, byte_len - sizeof(*hdr),
                                     flags);
    }

    uct_rc_mlx5_iface_release_srq_seg(iface, seg, wqe_ctr, status,
                                      iface->super.super.config.rx_headroom_offset,
                                      &iface->super.super.release_desc);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_add_fence(uct_ib_md_t *md, uct_ib_mlx5_txwq_t *wq)
{
    if (md->dev.pci_fadd_arg_sizes || md->dev.pci_cswap_arg_sizes) {
        wq->next_fm = UCT_IB_MLX5_WQE_CTRL_FENCE_ATOMIC;
    }
}

static UCS_F_ALWAYS_INLINE uint8_t
uct_rc_mlx5_ep_fm(uct_rc_mlx5_iface_common_t *iface, uct_ib_mlx5_txwq_t *txwq)
{
    uint8_t fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;

    fm_ce_se     |= txwq->next_fm;
    txwq->next_fm = 0;

    /* a call to iface_fence increases beat, so if endpoint beat is not in
     * sync with iface beat it means the endpoint did not post any WQE with
     * fence flag yet */
    if (txwq->fence_beat != iface->tx.fence_beat) {
        txwq->fence_beat = iface->tx.fence_beat;
        fm_ce_se        |= iface->tx.next_fm;
    }

    return fm_ce_se;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_common_post_send(uct_rc_mlx5_iface_common_t *iface, int qp_type,
                             uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                             uint8_t opcode, uint8_t opmod, uint8_t fm_ce_se,
                             size_t wqe_size, uct_ib_mlx5_base_av_t *av,
                             struct mlx5_grh_av *grh_av, uint32_t imm, int max_log_sge,
                             uct_ib_log_sge_t *log_sge)
{
    struct mlx5_wqe_ctrl_seg *ctrl;
    uint16_t res_count;

    ctrl = txwq->curr;

    if (opcode == MLX5_OPCODE_SEND_IMM) {
        uct_ib_mlx5_set_ctrl_seg_with_imm(ctrl, txwq->sw_pi, opcode, opmod,
                                          txqp->qp->qp_num, fm_ce_se, wqe_size,
                                          imm);
    } else {
        uct_ib_mlx5_set_ctrl_seg(ctrl, txwq->sw_pi, opcode, opmod,
                                 txqp->qp->qp_num, fm_ce_se, wqe_size);
    }

    ucs_assert(qp_type == iface->super.super.config.qp_type);

#if HAVE_TL_DC
    if (qp_type == UCT_IB_QPT_DCI) {
        uct_ib_mlx5_set_dgram_seg((void*)(ctrl + 1), av, grh_av, qp_type);
    }
#endif

    uct_ib_mlx5_log_tx(&iface->super.super, ctrl, txwq->qstart, txwq->qend,
                       max_log_sge, log_sge,
                       ((opcode == MLX5_OPCODE_SEND) || (opcode == MLX5_OPCODE_SEND_IMM)) ?
                       uct_rc_mlx5_common_packet_dump : NULL);

    res_count = uct_ib_mlx5_post_send(txwq, ctrl, wqe_size);
    if (fm_ce_se & MLX5_WQE_CTRL_CQ_UPDATE) {
        txwq->sig_pi = txwq->prev_sw_pi;
    }
    uct_rc_txqp_posted(txqp, &iface->super, res_count, fm_ce_se & MLX5_WQE_CTRL_CQ_UPDATE);
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
uct_rc_mlx5_txqp_inline_post(uct_rc_mlx5_iface_common_t *iface, int qp_type,
                             uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                             unsigned opcode, const void *buffer, unsigned length,
                  /* SEND */ uint8_t am_id, uint64_t am_hdr, uint32_t imm_val_be,
                  /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey,
                  /* AV   */ uct_ib_mlx5_base_av_t *av, struct mlx5_grh_av *grh_av,
                             size_t av_size, unsigned fm_ce_se, int max_log_sge)
{
    struct mlx5_wqe_ctrl_seg     *ctrl;
    struct mlx5_wqe_raddr_seg    *raddr;
    struct mlx5_wqe_inl_data_seg *inl;
    uct_rc_mlx5_am_short_hdr_t   *am;
    uct_rc_mlx5_hdr_t            *rc_hdr;
    size_t wqe_size, ctrl_av_size;
    void *next_seg;

    ctrl         = txwq->curr;
    ctrl_av_size = sizeof(*ctrl) + av_size;
    next_seg     = uct_ib_mlx5_txwq_wrap_exact(txwq, (char*)ctrl + ctrl_av_size);

    switch (opcode) {
    case MLX5_OPCODE_SEND_IMM:
        /* Fall through to MLX5_OPCODE_SEND handler */
    case MLX5_OPCODE_SEND:
        /* Set inline segment which has AM id, AM header, and AM payload */
        wqe_size         = ctrl_av_size + sizeof(*inl) + sizeof(*am) + length;
        inl              = next_seg;
        inl->byte_count  = htonl((length + sizeof(*am)) | MLX5_INLINE_SEG);
        am               = (void*)(inl + 1);
        am->am_hdr       = am_hdr;
        uct_rc_mlx5_am_hdr_fill(&am->rc_hdr, am_id);
        uct_ib_mlx5_inline_copy(am + 1, buffer, length, txwq);
        fm_ce_se        |= uct_rc_iface_tx_moderation(&iface->super, txqp, MLX5_WQE_CTRL_CQ_UPDATE);
        break;

    case MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_RAW:
        /* Send empty AM with just AM id (used by FC) */
        wqe_size         = ctrl_av_size + sizeof(*inl) + sizeof(*rc_hdr);
        inl              = next_seg;
        inl->byte_count  = htonl(sizeof(*rc_hdr) | MLX5_INLINE_SEG);
        rc_hdr           = (void*)(inl + 1);
        uct_rc_mlx5_am_hdr_fill(rc_hdr, am_id);
        fm_ce_se        |= uct_rc_iface_tx_moderation(&iface->super, txqp, MLX5_WQE_CTRL_CQ_UPDATE);
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
        fm_ce_se        |= uct_rc_iface_tx_moderation(&iface->super, txqp, MLX5_WQE_CTRL_CQ_UPDATE);
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
                                 wqe_size, av, grh_av, imm_val_be, max_log_sge, NULL);
}

/*
 * Generic data-pointer posting function.
 * Parameters which are not relevant to the opcode are ignored.
 *
 *            +--------+-----+-------+
 * SEND       | CTRL   | INL | DPSEG |
 *            +--------+-----+---+---+----+
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
uct_rc_mlx5_txqp_dptr_post(uct_rc_mlx5_iface_common_t *iface, int qp_type,
                           uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                           unsigned opcode_flags, const void *buffer,
                           unsigned length, uint32_t *lkey_p,
         /* RDMA/ATOMIC */ uint64_t remote_addr, uct_rkey_t rkey,
         /* ATOMIC      */ uint64_t compare_mask, uint64_t compare,
         /* ATOMIC      */ uint64_t swap_mask, uint64_t swap_add,
         /* AV          */ uct_ib_mlx5_base_av_t *av, struct mlx5_grh_av *grh_av,
                           size_t av_size, uint8_t fm_ce_se, uint32_t imm_val_be,
                           int max_log_sge, uct_ib_log_sge_t *log_sge)
{
    struct mlx5_wqe_ctrl_seg                     *ctrl;
    struct mlx5_wqe_raddr_seg                    *raddr;
    struct mlx5_wqe_atomic_seg                   *atomic;
    struct mlx5_wqe_data_seg                     *dptr;
    struct uct_ib_mlx5_atomic_masked_cswap32_seg *masked_cswap32;
    struct uct_ib_mlx5_atomic_masked_fadd32_seg  *masked_fadd32;
    struct uct_ib_mlx5_atomic_masked_cswap64_seg *masked_cswap64;
    struct uct_ib_mlx5_atomic_masked_fadd64_seg  *masked_fadd64;
    size_t  wqe_size, ctrl_av_size;
    uint8_t opmod;
    void *next_seg;

    if (!(fm_ce_se & MLX5_WQE_CTRL_CQ_UPDATE)) {
        fm_ce_se |= uct_rc_iface_tx_moderation(&iface->super, txqp, MLX5_WQE_CTRL_CQ_UPDATE);
    }

    opmod         = 0;
    ctrl         = txwq->curr;
    ctrl_av_size = sizeof(*ctrl) + av_size;
    next_seg     = uct_ib_mlx5_txwq_wrap_exact(txwq, (char*)ctrl + ctrl_av_size);

    switch (opcode_flags) {
    case MLX5_OPCODE_SEND_IMM: /* Used by tag offload */
    case MLX5_OPCODE_SEND:
        /* Data segment only */
        ucs_assert(length < (2ul << 30));
        ucs_assert(length <= iface->super.super.config.seg_size);

        wqe_size = ctrl_av_size + sizeof(struct mlx5_wqe_data_seg);
        uct_ib_mlx5_set_data_seg(next_seg, buffer, length, *lkey_p);
        break;

    case MLX5_OPCODE_RDMA_READ:
        fm_ce_se |= uct_rc_mlx5_ep_fm(iface, txwq);
        /* Fall through */
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
        fm_ce_se |= uct_rc_mlx5_ep_fm(iface, txwq);
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
        fm_ce_se |= uct_rc_mlx5_ep_fm(iface, txwq);
        raddr = next_seg;
        uct_ib_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        switch (length) {
        case sizeof(uint32_t):
            opmod                        = UCT_IB_MLX5_OPMOD_EXT_ATOMIC(2);
            masked_cswap32               = uct_ib_mlx5_txwq_wrap_none(txwq, raddr + 1);
            masked_cswap32->swap         = swap_add;
            masked_cswap32->compare      = compare;
            masked_cswap32->swap_mask    = swap_mask;
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
            masked_cswap64->swap         = swap_mask;
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
        fm_ce_se |= uct_rc_mlx5_ep_fm(iface, txwq);
        raddr = next_seg;
        uct_ib_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        switch (length) {
        case sizeof(uint32_t):
            opmod                         = UCT_IB_MLX5_OPMOD_EXT_ATOMIC(2);
            masked_fadd32                 = uct_ib_mlx5_txwq_wrap_none(txwq, raddr + 1);
            masked_fadd32->add            = swap_add;
            masked_fadd32->filed_boundary = compare;

            dptr                          = uct_ib_mlx5_txwq_wrap_exact(txwq, masked_fadd32 + 1);
            wqe_size                      = ctrl_av_size + sizeof(*raddr) +
                                            sizeof(*masked_fadd32) + sizeof(*dptr);
            break;
        case sizeof(uint64_t):
            opmod                         = UCT_IB_MLX5_OPMOD_EXT_ATOMIC(3); /* Ext. atomic, size 2**3 */
            masked_fadd64                 = uct_ib_mlx5_txwq_wrap_none(txwq, raddr + 1);
            masked_fadd64->add            = swap_add;
            masked_fadd64->filed_boundary = compare;

            dptr                          = uct_ib_mlx5_txwq_wrap_exact(txwq, masked_fadd64 + 1);
            wqe_size                      = ctrl_av_size + sizeof(*raddr) +
                                            sizeof(*masked_fadd64) + sizeof(*dptr);
            break;
        default:
            ucs_fatal("invalid atomic type length %d", length);
        }
        uct_ib_mlx5_set_data_seg(dptr, buffer, length, *lkey_p);
        break;

    default:
        ucs_fatal("invalid send opcode");
    }

    uct_rc_mlx5_common_post_send(iface, qp_type, txqp, txwq,
                                 (opcode_flags & UCT_RC_MLX5_OPCODE_MASK), opmod,
                                 fm_ce_se, wqe_size, av, grh_av, imm_val_be,
                                 max_log_sge, log_sge);
}

static UCS_F_ALWAYS_INLINE
void uct_rc_mlx5_txqp_dptr_post_iov(uct_rc_mlx5_iface_common_t *iface, int qp_type,
                                    uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                                    unsigned opcode_flags,
                         /* IOV  */ const uct_iov_t *iov, size_t iovcnt,
                         /* SEND */ uint8_t am_id, const void *am_hdr, unsigned am_hdr_len,
                         /* RDMA */ uint64_t remote_addr, uct_rkey_t rkey,
                         /* TAG  */ uct_tag_t tag, uint32_t app_ctx, uint32_t ib_imm_be,
                         /* AV   */ uct_ib_mlx5_base_av_t *av, struct mlx5_grh_av *grh_av,
                                    size_t av_size, uint8_t fm_ce_se, int max_log_sge)
{
    struct mlx5_wqe_ctrl_seg     *ctrl;
    struct mlx5_wqe_raddr_seg    *raddr;
    struct mlx5_wqe_data_seg     *dptr;
    struct mlx5_wqe_inl_data_seg *inl;
    uct_rc_mlx5_hdr_t            *rch;
    unsigned                      wqe_size, inl_seg_size, ctrl_av_size;
    void                         *next_seg;

    if (!(fm_ce_se & MLX5_WQE_CTRL_CQ_UPDATE)) {
        fm_ce_se |= uct_rc_iface_tx_moderation(&iface->super, txqp, MLX5_WQE_CTRL_CQ_UPDATE);
    }

    ctrl         = txwq->curr;
    ctrl_av_size = sizeof(*ctrl) + av_size;
    next_seg     = uct_ib_mlx5_txwq_wrap_exact(txwq, (char*)ctrl + ctrl_av_size);

    switch (opcode_flags) {
    case MLX5_OPCODE_SEND:
        inl_seg_size     = ucs_align_up_pow2(sizeof(*inl) + sizeof(*rch) + am_hdr_len,
                                             UCT_IB_MLX5_WQE_SEG_SIZE);

        ucs_assert(uct_iov_total_length(iov, iovcnt) + sizeof(*rch) + am_hdr_len <=
                   iface->super.super.config.seg_size);

        /* Inline segment with AM ID and header */
        inl              = next_seg;
        inl->byte_count  = htonl((sizeof(*rch) + am_hdr_len) | MLX5_INLINE_SEG);
        rch              = (uct_rc_mlx5_hdr_t *)(inl + 1);

        uct_rc_mlx5_am_hdr_fill(rch, am_id);
        uct_ib_mlx5_inline_copy(rch + 1, am_hdr, am_hdr_len, txwq);

        /* Data segment with payload */
        dptr             = (struct mlx5_wqe_data_seg *)((char *)inl + inl_seg_size);
        wqe_size         = ctrl_av_size + inl_seg_size +
                           uct_ib_mlx5_set_data_seg_iov(txwq, dptr, iov, iovcnt);

        ucs_assert(wqe_size <= UCT_IB_MLX5_MAX_SEND_WQE_SIZE);
        break;

#if IBV_HW_TM
    case MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_TM:
    case MLX5_OPCODE_SEND_IMM|UCT_RC_MLX5_OPCODE_FLAG_TM:
        inl_seg_size     = ucs_align_up_pow2(sizeof(*inl) + sizeof(struct ibv_tmh),
                                             UCT_IB_MLX5_WQE_SEG_SIZE);
        inl              = next_seg;
        inl->byte_count  = htonl(sizeof(struct ibv_tmh) | MLX5_INLINE_SEG);
        dptr             = uct_ib_mlx5_txwq_wrap_exact(txwq, (char *)inl + inl_seg_size);
        wqe_size         = ctrl_av_size + inl_seg_size +
                           uct_ib_mlx5_set_data_seg_iov(txwq, dptr, iov, iovcnt);

        uct_rc_mlx5_fill_tmh((struct ibv_tmh*)(inl + 1), tag, app_ctx,
                             IBV_TMH_EAGER);
        ucs_assert(wqe_size <= UCT_IB_MLX5_MAX_SEND_WQE_SIZE);
        break;
#endif

    case MLX5_OPCODE_RDMA_READ:
        fm_ce_se |= uct_rc_mlx5_ep_fm(iface, txwq);
        /* Fall through */
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
                                 0, fm_ce_se, wqe_size, av, grh_av, ib_imm_be,
                                 max_log_sge, NULL);
}

#if IBV_HW_TM
static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_set_tm_seg(uct_ib_mlx5_txwq_t *txwq,
                       uct_rc_mlx5_wqe_tm_seg_t *tmseg, int op, int index,
                       uint32_t unexp_cnt, uint64_t tag, uint64_t mask,
                       unsigned tm_flags)
{
    tmseg->sw_cnt = htons(unexp_cnt);
    tmseg->opcode = op << 4;
    tmseg->flags  = tm_flags;

    if (op == UCT_RC_MLX5_TM_OPCODE_NOP) {
        return;
    }

    tmseg->index = htons(index);

    if (op == UCT_RC_MLX5_TM_OPCODE_REMOVE) {
        return;
    }

    tmseg->append_tag  = tag;
    tmseg->append_mask = mask;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_release_tag_entry(uct_rc_mlx5_iface_common_t *iface,
                              uct_rc_mlx5_tag_entry_t *tag)
{
    if (!--tag->num_cqes) {
        tag->next            = NULL;
        iface->tm.tail->next = tag;
        iface->tm.tail       = tag;
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_add_cmd_qp_op(uct_rc_mlx5_iface_common_t *iface,
                          uct_rc_mlx5_tag_entry_t *tag)
{
    uct_rc_mlx5_srq_op_t *op;

    op      = iface->tm.cmd_wq.ops +
              (iface->tm.cmd_wq.ops_tail++ & iface->tm.cmd_wq.ops_mask);
    op->tag = tag;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_txqp_tag_inline_post(uct_rc_mlx5_iface_common_t *iface, int qp_type,
                                 uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                                 unsigned opcode, const void *buffer, unsigned length,
                                 const uct_iov_t *iov, /* relevant for RNDV */
                                 uct_tag_t tag, uint32_t app_ctx, int tm_op,
                                 uint32_t imm_val_be, uct_ib_mlx5_base_av_t *av,
                                 struct mlx5_grh_av *grh_av, size_t av_size,
                                 void *ravh, size_t ravh_len, unsigned fm_ce_se)
{
    struct mlx5_wqe_ctrl_seg     *ctrl;
    struct mlx5_wqe_inl_data_seg *inl;
    size_t wqe_size, ctrl_av_size;
    struct ibv_tmh *tmh;
    struct ibv_rvh rvh;
    unsigned tmh_data_len;
    size_t tm_hdr_len;
    void UCS_V_UNUSED *ravh_ptr;
    void *data;

    ctrl         = txwq->curr;
    ctrl_av_size = sizeof(*ctrl) + av_size; /* can be 16, 32 or 64 bytes */
    inl          = uct_ib_mlx5_txwq_wrap_exact(txwq, (char*)ctrl + ctrl_av_size);
    tmh          = (struct ibv_tmh*)(inl + 1);

    ucs_assert((opcode == MLX5_OPCODE_SEND_IMM) || (opcode == MLX5_OPCODE_SEND));

    switch (tm_op) {
    case IBV_TMH_EAGER:
        wqe_size         = ctrl_av_size + sizeof(*inl) + sizeof(*tmh) + length;
        inl->byte_count  = htonl((length + sizeof(*tmh)) | MLX5_INLINE_SEG);
        data             = tmh + 1;
        tmh_data_len     = 0;
        break;

    case IBV_TMH_RNDV:
        /* RVH can be wrapped */
        uct_rc_mlx5_fill_rvh(&rvh, iov->buffer,
                              ((uct_ib_mem_t*)iov->memh)->mr->rkey, iov->length);
        uct_ib_mlx5_inline_copy(tmh + 1, &rvh, sizeof(rvh), txwq);

        tm_hdr_len = sizeof(*tmh) + sizeof(rvh);
#if HAVE_TL_DC
        if (qp_type == UCT_IB_QPT_DCI) {
            /* RAVH can be wrapped as well */
            ravh_ptr = uct_ib_mlx5_txwq_wrap_data(txwq, (char*)tmh +
                                                  sizeof(*tmh) + sizeof(rvh));
            uct_ib_mlx5_inline_copy(ravh_ptr, ravh, ravh_len, txwq);
            tm_hdr_len += ravh_len;
        }
#endif

        tmh_data_len    = uct_rc_mlx5_fill_tmh_priv_data(tmh, buffer, length,
                                                         iface->tm.max_rndv_data);
        length         -= tmh_data_len; /* Note: change length func parameter */
        wqe_size        = ctrl_av_size + sizeof(*inl) + tm_hdr_len + length;
        inl->byte_count = htonl((length + tm_hdr_len) | MLX5_INLINE_SEG);
        data            = uct_ib_mlx5_txwq_wrap_data(txwq, (char*)tmh + tm_hdr_len);

        break;

    default:
        ucs_fatal("Invalid tag opcode: %d", tm_op);
        break;
    }

    ucs_assert(wqe_size <= UCT_IB_MLX5_MAX_SEND_WQE_SIZE);

    uct_rc_mlx5_fill_tmh(tmh, tag, app_ctx, tm_op);

    /* In case of RNDV first bytes of data could be stored in TMH */
    uct_ib_mlx5_inline_copy(data, (char*)buffer + tmh_data_len, length, txwq);
    fm_ce_se |= uct_rc_iface_tx_moderation(&iface->super, txqp, MLX5_WQE_CTRL_CQ_UPDATE);

    uct_rc_mlx5_common_post_send(iface, qp_type, txqp, txwq, opcode, 0, fm_ce_se,
                                 wqe_size, av, grh_av, imm_val_be, INT_MAX, NULL);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_common_post_srq_op(uct_rc_mlx5_cmd_wq_t *cmd_wq,
                                     unsigned extra_wqe_size, unsigned op_code,
                                     uint16_t next_idx, unsigned unexp_cnt,
                                     uct_tag_t tag, uct_tag_t tag_mask,
                                     unsigned tm_flags)
{
    uct_ib_mlx5_txwq_t       *txwq = &cmd_wq->super;
    struct mlx5_wqe_ctrl_seg *ctrl = txwq->curr; /* 16 bytes */
    uct_rc_mlx5_wqe_tm_seg_t *tm;                /* 32 bytes */
    unsigned                  wqe_size;

    wqe_size = sizeof(*ctrl) + sizeof(*tm) + extra_wqe_size;

    tm = uct_ib_mlx5_txwq_wrap_none(txwq, ctrl + 1);

    uct_ib_mlx5_set_ctrl_seg(ctrl, txwq->sw_pi, UCT_RC_MLX5_OPCODE_TAG_MATCHING,
                             0, cmd_wq->qp_num, 0, wqe_size);

    uct_rc_mlx5_set_tm_seg(txwq, tm, op_code, next_idx, unexp_cnt,
                           tag, tag_mask, tm_flags);

    uct_ib_mlx5_post_send(txwq, ctrl, wqe_size);
}


static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_iface_common_tag_recv(uct_rc_mlx5_iface_common_t *iface,
                                  uct_tag_t tag,
                                  uct_tag_t tag_mask, const uct_iov_t *iov,
                                  size_t iovcnt, uct_tag_context_t *ctx)
{
    uct_rc_mlx5_ctx_priv_t   *priv = uct_rc_mlx5_ctx_priv(ctx);
    uct_ib_mlx5_txwq_t       *txwq = &iface->tm.cmd_wq.super;
    struct mlx5_wqe_data_seg *dptr; /* 16 bytes */
    uct_rc_mlx5_tag_entry_t  *tag_entry;
    uint16_t                 next_idx;
    unsigned                 ctrl_size;

    UCT_CHECK_IOV_SIZE(iovcnt, 1ul, "uct_rc_mlx5_iface_common_tag_recv");
    UCT_RC_MLX5_CHECK_TAG(iface);

    ctrl_size           = sizeof(struct mlx5_wqe_ctrl_seg) +
                          sizeof(uct_rc_mlx5_wqe_tm_seg_t);
    tag_entry           = iface->tm.head;
    next_idx            = tag_entry->next - iface->tm.list;
    iface->tm.head      = tag_entry->next;
    tag_entry->next     = NULL;
    tag_entry->ctx      = ctx;
    tag_entry->num_cqes = 2; /* ADD and MSG_ARRIVED/CANCELED */

    /* Save aux data (which will be needed in the following ops) in the context */
    priv->tag_handle   = tag_entry - iface->tm.list;
    priv->tag          = tag;
    priv->buffer       = iov->buffer; /* Only one iov is supported so far */
    priv->length       = iov->length;

    uct_rc_mlx5_add_cmd_qp_op(iface, tag_entry);

    dptr = uct_ib_mlx5_txwq_wrap_none(txwq, (char*)txwq->curr + ctrl_size);
    uct_ib_mlx5_set_data_seg(dptr, iov->buffer, iov->length,
                             ((uct_ib_mem_t *)(iov->memh))->lkey);

    uct_rc_mlx5_iface_common_post_srq_op(&iface->tm.cmd_wq, sizeof(*dptr),
                                         UCT_RC_MLX5_TM_OPCODE_APPEND, next_idx,
                                         iface->tm.unexpected_cnt, tag,
                                         tag_mask,
                                         UCT_RC_MLX5_SRQ_FLAG_TM_CQE_REQ |
                                         UCT_RC_MLX5_SRQ_FLAG_TM_SW_CNT);

    UCT_RC_MLX5_TM_STAT(iface, LIST_ADD);

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_iface_common_tag_recv_cancel(uct_rc_mlx5_iface_common_t *iface,
                                         uct_tag_context_t *ctx, int force)
{
    uct_rc_mlx5_ctx_priv_t   *priv = uct_rc_mlx5_ctx_priv(ctx);
    uint16_t                 index = priv->tag_handle;
    uct_rc_mlx5_tag_entry_t  *tag_entry;
    unsigned flags;

    tag_entry = &iface->tm.list[index];

    if (ucs_likely(force)) {
        flags = UCT_RC_MLX5_SRQ_FLAG_TM_SW_CNT;
        uct_rc_mlx5_release_tag_entry(iface, tag_entry);
    } else {
        flags = UCT_RC_MLX5_SRQ_FLAG_TM_CQE_REQ | UCT_RC_MLX5_SRQ_FLAG_TM_SW_CNT;
        uct_rc_mlx5_add_cmd_qp_op(iface, tag_entry);
    }

    uct_rc_mlx5_iface_common_post_srq_op(&iface->tm.cmd_wq, 0,
                                         UCT_RC_MLX5_TM_OPCODE_REMOVE, index,
                                         iface->tm.unexpected_cnt, 0ul, 0ul,
                                         flags);

    UCT_RC_MLX5_TM_STAT(iface, LIST_DEL);

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_handle_tm_list_op(uct_rc_mlx5_iface_common_t *iface, int opcode)
{
    uct_rc_mlx5_cmd_wq_t *cmd_wq;
    uct_rc_mlx5_srq_op_t *op;
    uct_tag_context_t *ctx;
    uct_rc_mlx5_ctx_priv_t *priv;

    cmd_wq = &iface->tm.cmd_wq;
    op     = cmd_wq->ops + (cmd_wq->ops_head++ & cmd_wq->ops_mask);
    uct_rc_mlx5_release_tag_entry(iface, op->tag);

    if (opcode == UCT_RC_MLX5_CQE_APP_OP_TM_REMOVE) {
        ctx  = op->tag->ctx;
        priv = uct_rc_mlx5_ctx_priv(ctx);
        ctx->completed_cb(ctx, priv->tag, 0, priv->length, UCS_ERR_CANCELED);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_tag_consumed(uct_rc_mlx5_iface_common_t *iface,
                               struct mlx5_cqe64 *cqe, int opcode)
{
    struct ibv_tmh *tmh = (struct ibv_tmh*)cqe;
    uct_rc_mlx5_tag_entry_t *tag;
    uct_tag_context_t *ctx;
    uct_rc_mlx5_ctx_priv_t *priv;

    tag = &iface->tm.list[ntohs(cqe->app_info)];
    ctx = tag->ctx;
    ctx->tag_consumed_cb(ctx);

    if (opcode == UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED) {
        /* Need to save TMH info, which will be used when
         * UCT_RC_MLX5_CQE_APP_OP_TM_EXPECTED CQE is received */
        priv           = uct_rc_mlx5_ctx_priv(ctx);
        priv->tag      = tmh->tag;
        priv->app_ctx  = tmh->app_ctx;
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_handle_expected(uct_rc_mlx5_iface_common_t *iface, struct mlx5_cqe64 *cqe,
                                  unsigned byte_len, uint64_t tag, uint32_t app_ctx)
{
    uint64_t imm_data;
    uct_rc_mlx5_tag_entry_t *tag_entry;
    uct_tag_context_t *ctx;
    uct_rc_mlx5_ctx_priv_t *priv;

    tag_entry = &iface->tm.list[ntohs(cqe->app_info)];
    ctx       = tag_entry->ctx;
    priv      = uct_rc_mlx5_ctx_priv(tag_entry->ctx);

    uct_rc_mlx5_release_tag_entry(iface, tag_entry);

    if (cqe->op_own & MLX5_INLINE_SCATTER_64) {
        ucs_assert(byte_len <= priv->length);
        memcpy(priv->buffer, cqe - 1, byte_len);
    } else {
        VALGRIND_MAKE_MEM_DEFINED(priv->buffer, byte_len);
    }

    imm_data = uct_rc_mlx5_tag_imm_data_unpack(cqe->imm_inval_pkey, app_ctx,
                                               (cqe->op_own >> 4) ==
                                               MLX5_CQE_RESP_SEND_IMM);

    if (UCT_RC_MLX5_TM_IS_SW_RNDV(cqe, imm_data)) {
        ctx->rndv_cb(ctx, tag, priv->buffer, byte_len, UCS_OK);
        UCT_RC_MLX5_TM_STAT(iface, RX_RNDV_REQ_EXP);
    } else {
        ctx->completed_cb(ctx, tag, imm_data, byte_len, UCS_OK);
        UCT_RC_MLX5_TM_STAT(iface, RX_EXP);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_unexp_consumed(uct_rc_mlx5_iface_common_t *iface,
                                 uct_rc_mlx5_release_desc_t *release,
                                 ucs_status_t status, uint16_t wqe_ctr)
{
    uct_ib_mlx5_srq_seg_t *seg;

    seg = uct_ib_mlx5_srq_get_wqe(&iface->rx.srq, wqe_ctr);

    uct_rc_mlx5_iface_release_srq_seg(iface, seg, wqe_ctr,
                                      status, release->offset, &release->super);

    if (ucs_unlikely(!(++iface->tm.unexpected_cnt % IBV_DEVICE_MAX_UNEXP_COUNT))) {
        uct_rc_mlx5_iface_common_post_srq_op(&iface->tm.cmd_wq, 0,
                                             UCT_RC_MLX5_TM_OPCODE_NOP, 0,
                                             iface->tm.unexpected_cnt, 0ul, 0ul,
                                             UCT_RC_MLX5_SRQ_FLAG_TM_SW_CNT);

        UCT_RC_MLX5_TM_STAT(iface, LIST_SYNC);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_tag_handle_unexp(uct_rc_mlx5_iface_common_t *iface,
                                   struct mlx5_cqe64 *cqe, unsigned byte_len)
{
    struct ibv_tmh     *tmh;
    uint64_t           imm_data;
    ucs_status_t       status;
    unsigned           flags;

    tmh = uct_rc_mlx5_iface_common_data(iface, cqe,
                                        byte_len, &flags);

    if (ucs_likely((tmh->opcode == IBV_TMH_EAGER) &&
                   !UCT_RC_MLX5_TM_CQE_WITH_IMM(cqe))) {
        status = iface->tm.eager_unexp.cb(iface->tm.eager_unexp.arg,
                                          tmh + 1, byte_len - sizeof(*tmh),
                                          flags, tmh->tag, 0);

        uct_rc_mlx5_iface_unexp_consumed(iface, &iface->tm.eager_desc,
                                         status, ntohs(cqe->wqe_counter));

        UCT_RC_MLX5_TM_STAT(iface, RX_EAGER_UNEXP);

    } else if (tmh->opcode == IBV_TMH_EAGER) {
        imm_data = uct_rc_mlx5_tag_imm_data_unpack(cqe->imm_inval_pkey,
                                                   tmh->app_ctx, 1);

        if (ucs_unlikely(!imm_data)) {
            /* Opcode is WITH_IMM, but imm_data is 0 - this must be SW RNDV */
            status = iface->tm.rndv_unexp.cb(iface->tm.rndv_unexp.arg,
                                             flags, tmh->tag, tmh + 1,
                                             byte_len - sizeof(*tmh),
                                             0ul, 0, NULL);

            UCT_RC_MLX5_TM_STAT(iface, RX_RNDV_REQ_UNEXP);
        } else {
            status = iface->tm.eager_unexp.cb(iface->tm.eager_unexp.arg,
                                              tmh + 1, byte_len - sizeof(*tmh),
                                              flags, tmh->tag, imm_data);

            UCT_RC_MLX5_TM_STAT(iface, RX_EAGER_UNEXP);
        }

        uct_rc_mlx5_iface_unexp_consumed(iface, &iface->tm.eager_desc,
                                         status, ntohs(cqe->wqe_counter));
    } else {
        ucs_assertv_always(tmh->opcode == IBV_TMH_RNDV,
                           "Unsupported packet arrived %d", tmh->opcode);
        status = uct_rc_mlx5_handle_rndv(iface, tmh, tmh->tag, byte_len);

        uct_rc_mlx5_iface_unexp_consumed(iface, &iface->tm.rndv_desc,
                                         status, ntohs(cqe->wqe_counter));

        UCT_RC_MLX5_TM_STAT(iface, RX_RNDV_UNEXP);
    }
}
#endif /* IBV_HW_TM */

static UCS_F_ALWAYS_INLINE unsigned
uct_rc_mlx5_iface_common_poll_rx(uct_rc_mlx5_iface_common_t *iface,
                                 int is_tag_enabled)
{
    uct_ib_mlx5_srq_seg_t UCS_V_UNUSED *seg;
    struct mlx5_cqe64 *cqe;
    unsigned byte_len;
    uint16_t max_batch;
    unsigned count;
    void *rc_hdr;
    unsigned flags;
#if IBV_HW_TM
    struct ibv_tmh *tmh;
    uct_rc_mlx5_tag_entry_t *tag;
    uct_tag_context_t *ctx;
    uct_rc_mlx5_ctx_priv_t *priv;
#endif

    ucs_assert(uct_ib_mlx5_srq_get_wqe(&iface->rx.srq,
                                       iface->rx.srq.mask)->srq.next_wqe_index == 0);

    cqe = uct_rc_mlx5_iface_poll_rx_cq(iface);
    if (cqe == NULL) {
        /* If no CQE - post receives */
        count = 0;
        goto done;
    }

    ucs_memory_cpu_load_fence();
    UCS_STATS_UPDATE_COUNTER(iface->super.stats, UCT_RC_IFACE_STAT_RX_COMPLETION, 1);

    byte_len = ntohl(cqe->byte_cnt);
    count    = 1;

    if (!is_tag_enabled) {
        rc_hdr = uct_rc_mlx5_iface_common_data(iface, cqe,
                                               byte_len, &flags);
        uct_rc_mlx5_iface_common_am_handler(iface, cqe, rc_hdr, flags, byte_len);
        goto done;
    }

#if IBV_HW_TM
    ucs_assert(cqe->app == UCT_RC_MLX5_CQE_APP_TAG_MATCHING);

    /* Should be a fast path, because small (latency-critical) messages
     * are not supposed to be offloaded to the HW.  */
    if (ucs_likely(cqe->app_op == UCT_RC_MLX5_CQE_APP_OP_TM_UNEXPECTED)) {
        uct_rc_mlx5_iface_tag_handle_unexp(iface, cqe,
                                           byte_len);
        goto done;
    }

    switch (cqe->app_op) {
    case UCT_RC_MLX5_CQE_APP_OP_TM_APPEND:
        uct_rc_mlx5_iface_handle_tm_list_op(iface,
                                            UCT_RC_MLX5_CQE_APP_OP_TM_APPEND);
        break;

    case UCT_RC_MLX5_CQE_APP_OP_TM_REMOVE:
        uct_rc_mlx5_iface_handle_tm_list_op(iface,
                                            UCT_RC_MLX5_CQE_APP_OP_TM_REMOVE);
        break;

    case UCT_RC_MLX5_CQE_APP_OP_TM_NO_TAG:
        tmh = uct_rc_mlx5_iface_common_data(iface, cqe, byte_len, &flags);
        if (tmh->opcode == IBV_TMH_NO_TAG) {
            uct_rc_mlx5_iface_common_am_handler(iface, cqe,
                                                (uct_rc_mlx5_hdr_t*)tmh,
                                                flags, byte_len);
        } else {
            ucs_assert(tmh->opcode == IBV_TMH_FIN);
            uct_rc_mlx5_handle_rndv_fin(iface, tmh->app_ctx);
            seg = uct_ib_mlx5_srq_get_wqe(&iface->rx.srq,
                                          ntohs(cqe->wqe_counter));

            uct_rc_mlx5_iface_release_srq_seg(iface, seg,
                                              ntohs(cqe->wqe_counter), UCS_OK,
                                              0, NULL);

            UCT_RC_MLX5_TM_STAT(iface, RX_RNDV_FIN);
        }
        break;

    case UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED:
        uct_rc_mlx5_iface_tag_consumed(iface, cqe,
                                       UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED);
        break;

    case UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED_MSG:
        tmh = (struct ibv_tmh*)cqe;

        uct_rc_mlx5_iface_tag_consumed(iface, cqe,
                                       UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED_MSG);

        uct_rc_mlx5_iface_handle_expected(iface, cqe, byte_len,
                                          tmh->tag, tmh->app_ctx);
        break;

    case UCT_RC_MLX5_CQE_APP_OP_TM_EXPECTED:
        tag  = &iface->tm.list[ntohs(cqe->app_info)];
        ctx  = tag->ctx;
        priv = uct_rc_mlx5_ctx_priv(ctx);
        uct_rc_mlx5_iface_handle_expected(iface, cqe, byte_len,
                                          priv->tag, priv->app_ctx);
        break;

    default:
        ucs_fatal("Unsupported packet arrived %d", cqe->app_op);
        break;
    }
#endif

done:
    max_batch = iface->super.super.config.rx_max_batch;
    if (ucs_unlikely(iface->super.rx.srq.available >= max_batch)) {
        uct_rc_mlx5_iface_srq_post_recv(&iface->super, &iface->rx.srq);
    }
    return count;
}

#if HAVE_IBV_EXP_DM
/* DM memory should be written by 8 bytes to eliminate
 * processor cache issues. To make this used uct_rc_mlx5_dm_copy_data_t
 * datatype where first hdr_len bytes are filled by message header
 * and tail is filled by head of message. */
static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_common_copy_to_dm(uct_rc_mlx5_dm_copy_data_t *cache, size_t hdr_len,
                                    const void *payload, size_t length, void *dm,
                                    uct_ib_log_sge_t *log_sge)
{
    typedef uint64_t misaligned_t UCS_V_ALIGNED(1);

    uint64_t padding = 0; /* init by 0 to suppress valgrind error */
    size_t head      = (cache && hdr_len) ? ucs_min(length, sizeof(*cache) - hdr_len) : 0;
    size_t body      = ucs_align_down(length - head, sizeof(padding));
    size_t tail      = length - (head + body);
    char   *dst      = dm;
    int i            = 0;

    ucs_assert(sizeof(*cache) >= hdr_len);
    ucs_assert(head + body + tail == length);
    ucs_assert(tail < sizeof(padding));

    /* copy head of payload to tail of cache */
    memcpy(cache->bytes + hdr_len, payload, head);

    UCS_STATIC_ASSERT(sizeof(*cache) == sizeof(cache->bytes));
    UCS_STATIC_ASSERT(sizeof(log_sge->sg_list) / sizeof(log_sge->sg_list[0]) >= 2);

    /* condition is static-evaluated */
    if (cache && hdr_len) {
        /* atomically by 8 bytes copy data to DM */
        /* cache buffer must be aligned, so, source data type is aligned */
        UCS_WORD_COPY(volatile uint64_t, dst, uint64_t, cache->bytes, sizeof(cache->bytes));
        dst += sizeof(cache->bytes);
        if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) {
            log_sge->sg_list[0].addr   = (uint64_t)cache;
            log_sge->sg_list[0].length = (uint32_t)hdr_len;
            i++;
        }
    }
    if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) {
        log_sge->sg_list[i].addr   = (uint64_t)payload;
        log_sge->sg_list[i].length = (uint32_t)length;
        i++;
    }
    log_sge->num_sge = i;

    /* copy payload to DM */
    UCS_WORD_COPY(volatile uint64_t, dst, misaligned_t, (char*)payload + head, body);
    if (tail) {
        dst += body;
        memcpy(&padding, (char*)payload + head + body, tail);
        /* use uint64_t for source datatype because it is aligned buffer on stack */
        UCS_WORD_COPY(volatile uint64_t, dst, uint64_t, &padding, sizeof(padding));
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_common_dm_make_data(uct_rc_mlx5_iface_common_t *iface,
                                uct_rc_mlx5_dm_copy_data_t *cache,
                                size_t hdr_len, const void *payload,
                                unsigned length,
                                uct_rc_iface_send_desc_t **desc_p,
                                void **buffer_p, uct_ib_log_sge_t *log_sge)
{
    uct_rc_iface_send_desc_t *desc;
    void *buffer;

    ucs_assert(iface->dm.dm != NULL);
    ucs_assert(log_sge != NULL);

    desc = ucs_mpool_get_inline(&iface->dm.dm->mp);
    if (ucs_unlikely(desc == NULL)) {
        /* in case if no resources available - fallback to bcopy */
        UCT_RC_IFACE_GET_TX_DESC(&iface->super, &iface->super.tx.mp, desc);
        desc->super.handler = (uct_rc_send_handler_t)ucs_mpool_put;
        buffer = desc + 1;

        /* condition is static-evaluated, no performance penalty */
        if (cache && hdr_len) {
            memcpy(buffer, cache->bytes, hdr_len);
        }
        memcpy(UCS_PTR_BYTE_OFFSET(buffer, hdr_len), payload, length);
        log_sge->num_sge = 0;
    } else {
        /* desc must be partially initialized by mpool.
         * hint to valgrind to make it defined */
        VALGRIND_MAKE_MEM_DEFINED(desc, sizeof(*desc));
        ucs_assert(desc->super.buffer != NULL);
        buffer = (void*)((char*)desc->super.buffer - (char*)iface->dm.dm->start_va);

        uct_rc_mlx5_iface_common_copy_to_dm(cache, hdr_len, payload,
                                            length, desc->super.buffer, log_sge);
        if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) {
            log_sge->sg_list[0].lkey = log_sge->sg_list[1].lkey = desc->lkey;
            log_sge->inline_bitmap = 0;
        }
    }

    *desc_p   = desc;
    *buffer_p = buffer;
    return UCS_OK;
}
#endif

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_iface_common_atomic_data(unsigned opcode, unsigned size, uint64_t value,
                                     int *op, uint64_t *compare_mask, uint64_t *compare,
                                     uint64_t *swap_mask, uint64_t *swap, int *ext)
{
    ucs_assert((size == sizeof(uint64_t)) || (size == sizeof(uint32_t)));

    switch (opcode) {
    case UCT_ATOMIC_OP_ADD:
        switch (size) {
        case sizeof(uint64_t):
            *op       = MLX5_OPCODE_ATOMIC_FA;
            *ext      = 0;
            break;
        case sizeof(uint32_t):
            *op       = MLX5_OPCODE_ATOMIC_MASKED_FA;
            *ext      = 1;
            break;
        default:
            ucs_assertv(0, "incorrect atomic size: %d", size);
            return UCS_ERR_INVALID_PARAM;
        }
        *compare_mask = 0;
        *compare      = 0;
        *swap_mask    = 0;
        *swap         = UCT_RC_MLX5_TO_BE(value, size);
        break;
    case UCT_ATOMIC_OP_AND:
        *op           = MLX5_OPCODE_ATOMIC_MASKED_CS;
        *compare_mask = 0;
        *compare      = 0;
        *swap_mask    = UCT_RC_MLX5_TO_BE(~value, size);
        *swap         = UCT_RC_MLX5_TO_BE(value, size);
        *ext          = 1;
        break;
    case UCT_ATOMIC_OP_OR:
        *op           = MLX5_OPCODE_ATOMIC_MASKED_CS;
        *compare_mask = 0;
        *compare      = 0;
        *swap_mask    = UCT_RC_MLX5_TO_BE(value, size);
        *swap         = UCT_RC_MLX5_TO_BE(value, size);
        *ext          = 1;
        break;
    case UCT_ATOMIC_OP_XOR:
        *op           = MLX5_OPCODE_ATOMIC_MASKED_FA;
        *compare_mask = 0;
        *compare      = -1;
        *swap_mask    = 0;
        *swap         = UCT_RC_MLX5_TO_BE(value, size);
        *ext          = 1;
        break;
    case UCT_ATOMIC_OP_SWAP:
        *op           = MLX5_OPCODE_ATOMIC_MASKED_CS;
        *compare_mask = 0;
        *compare      = 0;
        *swap_mask    = -1;
        *swap         = UCT_RC_MLX5_TO_BE(value, size);
        *ext          = 1;
        break;
    default:
        ucs_assertv(0, "incorrect atomic opcode: %d", opcode);
        return UCS_ERR_UNSUPPORTED;
    }
    return UCS_OK;
}


