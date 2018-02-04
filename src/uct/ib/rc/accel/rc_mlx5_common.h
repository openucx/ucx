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
#define UCT_RC_MLX5_OPCODE_FLAG_TM    0x200
#define UCT_RC_MLX5_OPCODE_MASK       0xff

#define UCT_RC_MLX5_CHECK_AM_ZCOPY(_id, _header_length, _length, _seg_size, _av_size) \
    UCT_RC_CHECK_AM_ZCOPY(_id, _header_length, _length, \
                          UCT_IB_MLX5_AM_ZCOPY_MAX_HDR(_av_size), _seg_size)

#define UCT_RC_MLX5_CHECK_AM_SHORT(_id, _length, _av_size) \
    UCT_RC_CHECK_AM_SHORT(_id, _length, UCT_IB_MLX5_AM_MAX_SHORT(_av_size))


/* there is no need to do a special check for length == 0 because in that
 * case wqe size is valid: inl + raddr + dgram + ctrl fit in 2 WQ BB
 */
#define UCT_RC_MLX5_CHECK_PUT_SHORT(_length, _av_size) \
    UCT_CHECK_LENGTH(_length, 0, UCT_IB_MLX5_PUT_MAX_SHORT(_av_size), "put_short")


enum {
    UCT_RC_MLX5_IFACE_STAT_RX_INL_32,
    UCT_RC_MLX5_IFACE_STAT_RX_INL_64,
    UCT_RC_MLX5_IFACE_STAT_LAST
};

#if IBV_EXP_HW_TM

enum {
    UCT_RC_MLX5_TM_OPCODE_NOP              = 0x00,
    UCT_RC_MLX5_TM_OPCODE_APPEND           = 0x01,
    UCT_RC_MLX5_TM_OPCODE_REMOVE           = 0x02
};

/* TODO: Remove/replace this enum when mlx5dv.h is included */
enum {
    UCT_RC_MLX5_OPCODE_TAG_MATCHING        = 0x28,
    UCT_RC_MLX5_CQE_APP_TAG_MATCHING       = 1,

    /* tag segment flags */
    UCT_RC_MLX5_SRQ_FLAG_TM_SW_CNT         = (1 << 6),
    UCT_RC_MLX5_SRQ_FLAG_TM_CQE_REQ        = (1 << 7),

    /* tag CQE codes */
    UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED     = 0x1,
    UCT_RC_MLX5_CQE_APP_OP_TM_EXPECTED     = 0x2,
    UCT_RC_MLX5_CQE_APP_OP_TM_UNEXPECTED   = 0x3,
    UCT_RC_MLX5_CQE_APP_OP_TM_NO_TAG       = 0x4,
    UCT_RC_MLX5_CQE_APP_OP_TM_APPEND       = 0x5,
    UCT_RC_MLX5_CQE_APP_OP_TM_REMOVE       = 0x6,
    UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED_MSG = 0xA
};

#  define UCT_RC_MLX5_TM_IS_SW_RNDV(_cqe64, _imm_data) \
       (ucs_unlikely((((_cqe64)->op_own >> 4) == MLX5_CQE_RESP_SEND_IMM) && \
                     !(_imm_data)))

#  define UCT_RC_MLX5_CHECK_TAG(_mlx5_common_iface) \
       if (ucs_unlikely((_mlx5_common_iface)->tm.head->next == NULL)) {  \
           return UCS_ERR_EXCEEDS_LIMIT; \
       }


/* TODO: Remove this struct when mlx5dv.h is included! */
typedef struct uct_rc_mlx5_wqe_tm_seg {
    uint8_t                       opcode;
    uint8_t                       flags;
    uint16_t                      index;
    uint8_t                       rsvd0[2];
    uint16_t                      sw_cnt;
    uint8_t                       rsvd1[8];
    uint64_t                      append_tag;
    uint64_t                      append_mask;
} uct_rc_mlx5_wqe_tm_seg_t;


/* Tag matching list entry */
typedef struct uct_rc_mlx5_tag_entry {
    struct uct_rc_mlx5_tag_entry  *next;
    uct_tag_context_t             *ctx;     /* the corresponding UCT context */
    unsigned                      num_cqes; /* how many CQEs is expected for this entry */
} uct_rc_mlx5_tag_entry_t;


/* Pending operation on the command QP */
typedef struct uct_rc_mlx5_srq_op {
    uct_rc_mlx5_tag_entry_t       *tag;
} uct_rc_mlx5_srq_op_t;


/* Command QP work-queue. All tag matching list operations are posted on it. */
typedef struct uct_rc_mlx5_cmd_wq {
    uct_ib_mlx5_txwq_t            super;
    uint32_t                      qp_num;   /* command QP num */
    uct_rc_mlx5_srq_op_t          *ops;     /* array of operations on command QP */
    int                           ops_head; /* points to the next operation to be completed */
    int                           ops_tail; /* points to the last adde operation*/
    int                           ops_mask; /* mask which bounds head and tail by
                                               ops array size */
} uct_rc_mlx5_cmd_wq_t;

#endif /* IBV_EXP_HW_TM  */

typedef struct uct_rc_mlx5_iface_common {
    struct {
        uct_ib_mlx5_cq_t          cq;
        ucs_mpool_t               atomic_desc_mp;
    } tx;
    struct {
        uct_ib_mlx5_cq_t          cq;
        uct_ib_mlx5_srq_t         srq;
        void                      *pref_ptr;
    } rx;
#if IBV_EXP_HW_TM
    struct {
        uct_rc_mlx5_cmd_wq_t      cmd_wq;
        uct_rc_mlx5_tag_entry_t   *head;
        uct_rc_mlx5_tag_entry_t   *tail;
        uct_rc_mlx5_tag_entry_t   *list;
    } tm;
#endif
    UCS_STATS_NODE_DECLARE(stats);
} uct_rc_mlx5_iface_common_t;


unsigned uct_rc_mlx5_iface_srq_post_recv(uct_rc_iface_t *iface, uct_ib_mlx5_srq_t *srq);

void uct_rc_mlx5_iface_common_prepost_recvs(uct_rc_iface_t *iface,
                                            uct_rc_mlx5_iface_common_t *mlx5_common);

ucs_status_t uct_rc_mlx5_iface_common_init(uct_rc_mlx5_iface_common_t *iface,
                                           uct_rc_iface_t *rc_iface,
                                           uct_rc_iface_config_t *config);

void uct_rc_mlx5_iface_common_cleanup(uct_rc_mlx5_iface_common_t *iface);

void uct_rc_mlx5_iface_common_query(uct_iface_attr_t *iface_attr);

void uct_rc_mlx5_iface_common_update_cqs_ci(uct_rc_mlx5_iface_common_t *iface,
                                            uct_ib_iface_t *ib_iface);

void uct_rc_mlx5_iface_common_sync_cqs_ci(uct_rc_mlx5_iface_common_t *iface,
                                          uct_ib_iface_t *ib_iface);

void uct_rc_mlx5_iface_commom_clean_srq(uct_rc_mlx5_iface_common_t *mlx5_common_iface,
                                        uct_rc_iface_t *rc_iface, uint32_t qpn);

ucs_status_t
uct_rc_mlx5_iface_common_tag_init(uct_rc_mlx5_iface_common_t *iface,
                                  uct_rc_iface_t *rc_iface,
                                  uct_rc_iface_config_t *rc_config,
                                  struct ibv_exp_create_srq_attr *srq_init_attr,
                                  unsigned rndv_hdr_len);

void uct_rc_mlx5_iface_common_tag_cleanup(uct_rc_mlx5_iface_common_t *iface,
                                          uct_rc_iface_t *rc_iface);

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

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_srq_prefetch_setup(uct_rc_mlx5_iface_common_t *mlx5_common_iface,
                               uct_rc_iface_t *rc_iface)
{
    unsigned wqe_ctr = (mlx5_common_iface->rx.srq.free_idx + 2) & mlx5_common_iface->rx.srq.mask;
    uct_ib_mlx5_srq_seg_t *seg;

    seg = uct_ib_mlx5_srq_get_wqe(&mlx5_common_iface->rx.srq, wqe_ctr);
    mlx5_common_iface->rx.pref_ptr =
            uct_ib_iface_recv_desc_hdr(&rc_iface->super, seg->srq.desc);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_release_srq_seg(uct_rc_mlx5_iface_common_t *mlx5_common_iface,
                                  uct_rc_iface_t *rc_iface,
                                  uct_ib_mlx5_srq_seg_t *seg, uint16_t wqe_ctr,
                                  ucs_status_t status, unsigned offset,
                                  uct_recv_desc_t *release_desc)
{
    void *udesc;

    if (ucs_likely((status == UCS_OK) &&
         (wqe_ctr == ((mlx5_common_iface->rx.srq.ready_idx + 1) &
                       mlx5_common_iface->rx.srq.mask)))) {
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
             udesc                = (char*)seg->srq.desc + offset;
             uct_recv_desc(udesc) = release_desc;
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
}

static UCS_F_NOINLINE void
uct_rc_mlx5_iface_check_rx_completion(uct_rc_mlx5_iface_common_t *mlx5_common_iface,
                                      uct_rc_iface_t *rc_iface,
                                      struct mlx5_cqe64 *cqe)
{
    uct_ib_mlx5_cq_t *cq      = &mlx5_common_iface->rx.cq;
    struct mlx5_err_cqe *ecqe = (void*)cqe;
    uct_ib_mlx5_srq_seg_t *seg;
    uint16_t wqe_ctr;

    ucs_memory_cpu_load_fence();

    if (((ecqe->op_own >> 4) == MLX5_CQE_RESP_ERR) &&
        (ecqe->syndrome == MLX5_CQE_SYNDROME_REMOTE_ABORTED_ERR) &&
        (ecqe->vendor_err_synd == UCT_IB_MLX5_CQE_VENDOR_SYND_ODP))
    {
        /* Release the aborted segment */
        wqe_ctr = ntohs(ecqe->wqe_counter);
        seg     = uct_ib_mlx5_srq_get_wqe(&mlx5_common_iface->rx.srq, wqe_ctr);
        ++cq->cq_ci;
        uct_rc_mlx5_iface_release_srq_seg(mlx5_common_iface, rc_iface, seg,
                                          wqe_ctr, UCS_OK,
                                          rc_iface->super.config.rx_headroom_offset,
                                          &rc_iface->super.release_desc);
    } else if ((ecqe->op_own >> 4) != MLX5_CQE_INVALID) {
        uct_ib_mlx5_check_completion(&rc_iface->super, cq, cqe);
    }
}

static UCS_F_ALWAYS_INLINE struct mlx5_cqe64*
uct_rc_mlx5_iface_poll_rx_cq(uct_rc_mlx5_iface_common_t *mlx5_common_iface,
                             uct_rc_iface_t *rc_iface)
{
    uct_ib_mlx5_cq_t *cq = &mlx5_common_iface->rx.cq;
    struct mlx5_cqe64 *cqe;
    unsigned index;
    uint8_t op_own;

    /* Prefetch the descriptor if it was scheduled */
    ucs_prefetch(mlx5_common_iface->rx.pref_ptr);

    index  = cq->cq_ci;
    cqe    = uct_ib_mlx5_get_cqe(cq, index);
    op_own = cqe->op_own;

    if (ucs_unlikely((op_own & MLX5_CQE_OWNER_MASK) == !(index & cq->cq_length))) {
        return NULL;
    } else if (ucs_unlikely(op_own & UCT_IB_MLX5_CQE_OP_OWN_ERR_MASK)) {
        uct_rc_mlx5_iface_check_rx_completion(mlx5_common_iface, rc_iface, cqe);
        return NULL;
    }

    cq->cq_ci = index + 1;
    return cqe; /* TODO optimize - let complier know cqe is not null */
}

static UCS_F_ALWAYS_INLINE void*
uct_rc_mlx5_iface_common_data(uct_rc_mlx5_iface_common_t *mlx5_iface,
                              uct_rc_iface_t *rc_iface, struct mlx5_cqe64 *cqe,
                              unsigned byte_len, unsigned *flags)
{
    uct_ib_mlx5_srq_seg_t *seg;
    uct_ib_iface_recv_desc_t *desc;
    void *hdr;

    seg  = uct_ib_mlx5_srq_get_wqe(&mlx5_iface->rx.srq, ntohs(cqe->wqe_counter));
    desc = seg->srq.desc;

    /* Get a pointer to AM or Tag header (after which comes the payload)
     * Support cases of inline scatter by pointing directly to CQE.
     */
    if (cqe->op_own & MLX5_INLINE_SCATTER_32) {
        hdr = (uct_rc_hdr_t*)(cqe);
        uct_rc_mlx5_iface_common_rx_inline(mlx5_iface, rc_iface, desc,
                                           UCT_RC_MLX5_IFACE_STAT_RX_INL_32,
                                           byte_len);
        *flags = 0;
    } else if (cqe->op_own & MLX5_INLINE_SCATTER_64) {
        hdr = (uct_rc_hdr_t*)(cqe - 1);
        uct_rc_mlx5_iface_common_rx_inline(mlx5_iface, rc_iface, desc,
                                           UCT_RC_MLX5_IFACE_STAT_RX_INL_64,
                                           byte_len);
        *flags = 0;
    } else {
        hdr = uct_ib_iface_recv_desc_hdr(&rc_iface->super, desc);
        VALGRIND_MAKE_MEM_DEFINED(hdr, byte_len);
        *flags = UCT_CB_PARAM_FLAG_DESC;
        /* Assuming that next packet likely will be non-inline,
         * setup the next prefetch pointer
         */
         uct_rc_mlx5_srq_prefetch_setup(mlx5_iface, rc_iface);
    }

    return hdr;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_common_am_handler(uct_rc_mlx5_iface_common_t *mlx5_iface,
                                    uct_rc_iface_t *rc_iface,
                                    struct mlx5_cqe64 *cqe, uct_rc_hdr_t *hdr,
                                    unsigned flags, unsigned byte_len)
{
    uint16_t wqe_ctr;
    uct_rc_iface_ops_t *rc_ops;
    uct_ib_mlx5_srq_seg_t *seg;
    uint32_t qp_num;
    ucs_status_t status;

    wqe_ctr = ntohs(cqe->wqe_counter);
    seg     = uct_ib_mlx5_srq_get_wqe(&mlx5_iface->rx.srq, wqe_ctr);

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

    uct_rc_mlx5_iface_release_srq_seg(mlx5_iface, rc_iface, seg, wqe_ctr, status,
                                      rc_iface->super.config.rx_headroom_offset,
                                      &rc_iface->super.release_desc);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_common_post_send(uct_rc_iface_t *iface, enum ibv_qp_type qp_type,
                             uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                             uint8_t opcode, uint8_t opmod, uint8_t fm_ce_se,
                             size_t wqe_size, uct_ib_mlx5_base_av_t *av,
                             struct mlx5_grh_av *grh_av, uint32_t imm)
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
        uct_ib_mlx5_set_dgram_seg((void*)(ctrl + 1), av, grh_av, qp_type);
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
                  /* AV   */ uct_ib_mlx5_base_av_t *av, struct mlx5_grh_av *grh_av,
                             size_t av_size, unsigned fm_ce_se)
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
        am->am_hdr       = am_hdr;
        uct_rc_am_hdr_fill(&am->rc_hdr, am_id);
        uct_ib_mlx5_inline_copy(am + 1, buffer, length, txwq);
        fm_ce_se        |= uct_rc_iface_tx_moderation(iface, txqp, MLX5_WQE_CTRL_CQ_UPDATE);
        break;

    case MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_RAW:
        /* Send empty AM with just AM id (used by FC) */
        wqe_size         = ctrl_av_size + sizeof(*inl) + sizeof(*rc_hdr);
        inl              = next_seg;
        inl->byte_count  = htonl(sizeof(*rc_hdr) | MLX5_INLINE_SEG);
        rc_hdr           = (void*)(inl + 1);
        uct_rc_am_hdr_fill(rc_hdr, am_id);
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
                                 wqe_size, av, grh_av, imm_val_be);
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
uct_rc_mlx5_txqp_dptr_post(uct_rc_iface_t *iface, enum ibv_qp_type qp_type,
                           uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                           unsigned opcode_flags, const void *buffer,
                           unsigned length, uint32_t *lkey_p,
         /* RDMA/ATOMIC */ uint64_t remote_addr, uct_rkey_t rkey,
         /* ATOMIC      */ uint64_t compare_mask, uint64_t compare, uint64_t swap_add,
         /* AV          */ uct_ib_mlx5_base_av_t *av, struct mlx5_grh_av *grh_av,
                           size_t av_size, uint8_t fm_ce_se, uint32_t imm_val_be)
{
    struct mlx5_wqe_ctrl_seg                     *ctrl;
    struct mlx5_wqe_raddr_seg                    *raddr;
    struct mlx5_wqe_atomic_seg                   *atomic;
    struct mlx5_wqe_data_seg                     *dptr;
    struct uct_ib_mlx5_atomic_masked_cswap32_seg *masked_cswap32;
    struct uct_ib_mlx5_atomic_masked_fadd32_seg  *masked_fadd32;
    struct uct_ib_mlx5_atomic_masked_cswap64_seg *masked_cswap64;
    size_t  wqe_size, ctrl_av_size;
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
    case MLX5_OPCODE_SEND_IMM: /* Used by tag offload */
    case MLX5_OPCODE_SEND:
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
                                 opmod, fm_ce_se, wqe_size, av, grh_av, imm_val_be);
}

static UCS_F_ALWAYS_INLINE
void uct_rc_mlx5_txqp_dptr_post_iov(uct_rc_iface_t *iface, enum ibv_qp_type qp_type,
                                    uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                                    unsigned opcode_flags,
                         /* IOV  */ const uct_iov_t *iov, size_t iovcnt,
                         /* SEND */ uint8_t am_id, const void *am_hdr, unsigned am_hdr_len,
                         /* RDMA */ uint64_t remote_addr, uct_rkey_t rkey,
                         /* TAG  */ uct_tag_t tag, uint32_t app_ctx, uint32_t ib_imm_be,
                         /* AV   */ uct_ib_mlx5_base_av_t *av, struct mlx5_grh_av *grh_av,
                                    size_t av_size, uint8_t fm_ce_se)
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

        uct_rc_am_hdr_fill(rch, am_id);
        uct_ib_mlx5_inline_copy(rch + 1, am_hdr, am_hdr_len, txwq);

        /* Data segment with payload */
        dptr             = (struct mlx5_wqe_data_seg *)((char *)inl + inl_seg_size);
        wqe_size         = ctrl_av_size + inl_seg_size +
                           uct_ib_mlx5_set_data_seg_iov(txwq, dptr, iov, iovcnt);

        ucs_assert(wqe_size <= (UCT_IB_MLX5_MAX_BB * MLX5_SEND_WQE_BB));
        break;

#if IBV_EXP_HW_TM
    case MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_TM:
    case MLX5_OPCODE_SEND_IMM|UCT_RC_MLX5_OPCODE_FLAG_TM:
        inl_seg_size     = ucs_align_up_pow2(sizeof(*inl) + sizeof(struct ibv_exp_tmh),
                                             UCT_IB_MLX5_WQE_SEG_SIZE);
        inl              = next_seg;
        inl->byte_count  = htonl(sizeof(struct ibv_exp_tmh) | MLX5_INLINE_SEG);
        dptr             = uct_ib_mlx5_txwq_wrap_exact(txwq, (char *)inl + inl_seg_size);
        wqe_size         = ctrl_av_size + inl_seg_size +
                           uct_ib_mlx5_set_data_seg_iov(txwq, dptr, iov, iovcnt);

        uct_rc_iface_fill_tmh((struct ibv_exp_tmh*)(inl + 1), tag, app_ctx,
                              IBV_EXP_TMH_EAGER);
        ucs_assert(wqe_size <= (UCT_IB_MLX5_MAX_BB * MLX5_SEND_WQE_BB));
        break;
#endif

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
                                 0, fm_ce_se, wqe_size, av, grh_av, ib_imm_be);
}

#if IBV_EXP_HW_TM

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

    tmseg->append_tag  = htobe64(tag);
    tmseg->append_mask = htobe64(mask);
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
uct_rc_mlx5_txqp_tag_inline_post(uct_rc_iface_t *iface, enum ibv_qp_type qp_type,
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
    struct ibv_exp_tmh *tmh;
    struct ibv_exp_tmh_rvh rvh;
    unsigned tmh_data_len;
    size_t tm_hdr_len;
    void *ravh_ptr;
    void *data;

    ctrl         = txwq->curr;
    ctrl_av_size = sizeof(*ctrl) + av_size; /* can be 16, 32 or 64 bytes */
    inl          = uct_ib_mlx5_txwq_wrap_exact(txwq, (char*)ctrl + ctrl_av_size);
    tmh          = (struct ibv_exp_tmh*)(inl + 1);

    ucs_assert((opcode == MLX5_OPCODE_SEND_IMM) || (opcode == MLX5_OPCODE_SEND));

    switch (tm_op) {
    case IBV_EXP_TMH_EAGER:
        wqe_size         = ctrl_av_size + sizeof(*inl) + sizeof(*tmh) + length;
        inl->byte_count  = htonl((length + sizeof(*tmh)) | MLX5_INLINE_SEG);
        data             = tmh + 1;
        tmh_data_len     = 0;
        break;

    case IBV_EXP_TMH_RNDV:
        /* RVH can be wrapped */
        uct_rc_iface_fill_rvh(&rvh, iov->buffer,
                              ((uct_ib_mem_t*)iov->memh)->mr->rkey, iov->length);
        uct_ib_mlx5_inline_copy(tmh + 1, &rvh, sizeof(rvh), txwq);

        if (qp_type == IBV_EXP_QPT_DC_INI) {
            /* RAVH can be wrapped as well */
            ravh_ptr = uct_ib_mlx5_txwq_wrap_data(txwq, (char*)tmh +
                                                  sizeof(*tmh) + sizeof(rvh));
            uct_ib_mlx5_inline_copy(ravh_ptr, ravh, ravh_len, txwq);
            tm_hdr_len = sizeof(*tmh) + sizeof(rvh) + ravh_len;

        } else {
            tm_hdr_len = sizeof(*tmh) + sizeof(rvh);
        }

        tmh_data_len    = uct_rc_iface_fill_tmh_priv_data(tmh, buffer, length,
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

    ucs_assert(wqe_size <= (UCT_IB_MLX5_MAX_BB * MLX5_SEND_WQE_BB));

    uct_rc_iface_fill_tmh(tmh, tag, app_ctx, tm_op);

    /* In case of RNDV first bytes of data could be stored in TMH */
    uct_ib_mlx5_inline_copy(data, (char*)buffer + tmh_data_len, length, txwq);
    fm_ce_se |= uct_rc_iface_tx_moderation(iface, txqp, MLX5_WQE_CTRL_CQ_UPDATE);

    uct_rc_mlx5_common_post_send(iface, qp_type, txqp, txwq, opcode, 0, fm_ce_se,
                                 wqe_size, av, grh_av, imm_val_be);
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
                                  uct_rc_iface_t *rc_iface, uct_tag_t tag,
                                  uct_tag_t tag_mask, const uct_iov_t *iov,
                                  size_t iovcnt, uct_tag_context_t *ctx)
{
    uct_rc_iface_ctx_priv_t  *priv = uct_rc_iface_ctx_priv(ctx);
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
                                         rc_iface->tm.unexpected_cnt, tag,
                                         tag_mask,
                                         UCT_RC_MLX5_SRQ_FLAG_TM_CQE_REQ |
                                         UCT_RC_MLX5_SRQ_FLAG_TM_SW_CNT);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_iface_common_tag_recv_cancel(uct_rc_mlx5_iface_common_t *iface,
                                         uct_rc_iface_t *rc_iface,
                                         uct_tag_context_t *ctx, int force)
{
    uct_rc_iface_ctx_priv_t  *priv = uct_rc_iface_ctx_priv(ctx);
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
                                         rc_iface->tm.unexpected_cnt, 0ul, 0ul,
                                         flags);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_handle_tm_list_op(uct_rc_mlx5_iface_common_t *mlx5_common_iface,
                                    int opcode)
{
    uct_rc_mlx5_cmd_wq_t *cmd_wq;
    uct_rc_mlx5_srq_op_t *op;
    uct_tag_context_t *ctx;
    uct_rc_iface_ctx_priv_t *priv;

    cmd_wq = &mlx5_common_iface->tm.cmd_wq;
    op     = cmd_wq->ops + (cmd_wq->ops_head++ & cmd_wq->ops_mask);
    uct_rc_mlx5_release_tag_entry(mlx5_common_iface, op->tag);

    if (opcode == UCT_RC_MLX5_CQE_APP_OP_TM_REMOVE) {
        ctx  = op->tag->ctx;
        priv = uct_rc_iface_ctx_priv(ctx);
        ctx->completed_cb(ctx, priv->tag, 0, priv->length, UCS_ERR_CANCELED);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_tag_consumed(uct_rc_mlx5_iface_common_t *mlx5_common_iface,
                               struct mlx5_cqe64 *cqe, int opcode)
{
    struct ibv_exp_tmh *tmh = (struct ibv_exp_tmh*)cqe;
    uct_rc_mlx5_tag_entry_t *tag;
    uct_tag_context_t *ctx;
    uct_rc_iface_ctx_priv_t *priv;

    tag = &mlx5_common_iface->tm.list[ntohs(cqe->app_info)];
    ctx = tag->ctx;
    ctx->tag_consumed_cb(ctx);

    if (opcode == UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED) {
        /* Need to save TMH info, which will be used when
         * UCT_RC_MLX5_CQE_APP_OP_TM_EXPECTED CQE is received */
        priv           = uct_rc_iface_ctx_priv(ctx);
        priv->tag      = be64toh(tmh->tag);
        priv->app_ctx  = ntohl(tmh->app_ctx);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_handle_expected(uct_rc_mlx5_iface_common_t *mlx5_common_iface,
                                  uct_rc_iface_t *rc_iface, struct mlx5_cqe64 *cqe,
                                  unsigned byte_len, uint64_t tag, uint32_t app_ctx)
{
    uint64_t imm_data;
    uct_rc_mlx5_tag_entry_t *tag_entry;
    uct_tag_context_t *ctx;
    uct_rc_iface_ctx_priv_t *priv;

    tag_entry = &mlx5_common_iface->tm.list[ntohs(cqe->app_info)];
    ctx       = tag_entry->ctx;
    priv      = uct_rc_iface_ctx_priv(tag_entry->ctx);

    uct_rc_mlx5_release_tag_entry(mlx5_common_iface, tag_entry);

    if (cqe->op_own & MLX5_INLINE_SCATTER_64) {
        ucs_assert(byte_len <= priv->length);
        memcpy(priv->buffer, cqe - 1, byte_len);
    } else {
        VALGRIND_MAKE_MEM_DEFINED(priv->buffer, byte_len);
    }

    imm_data = uct_rc_iface_tag_imm_data_unpack(cqe->imm_inval_pkey, app_ctx,
                                                (cqe->op_own >> 4) ==
                                                MLX5_CQE_RESP_SEND_IMM);

    if (UCT_RC_MLX5_TM_IS_SW_RNDV(cqe, imm_data)) {
        ctx->rndv_cb(ctx, tag, priv->buffer, byte_len, UCS_OK);
    } else {
        ctx->completed_cb(ctx, tag, imm_data, byte_len, UCS_OK);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_unexp_consumed(uct_rc_mlx5_iface_common_t *mlx5_common_iface,
                                 uct_rc_iface_t *rc_iface,
                                 uct_rc_iface_release_desc_t *release,
                                 ucs_status_t status, uint16_t wqe_ctr)
{
    uct_ib_mlx5_srq_seg_t *seg;

    seg = uct_ib_mlx5_srq_get_wqe(&mlx5_common_iface->rx.srq, wqe_ctr);

    uct_rc_mlx5_iface_release_srq_seg(mlx5_common_iface, rc_iface, seg, wqe_ctr,
                                      status, release->offset, &release->super);

    if (ucs_unlikely(!(++rc_iface->tm.unexpected_cnt % IBV_DEVICE_MAX_UNEXP_COUNT))) {
        uct_rc_mlx5_iface_common_post_srq_op(&mlx5_common_iface->tm.cmd_wq, 0,
                                             UCT_RC_MLX5_TM_OPCODE_NOP, 0,
                                             rc_iface->tm.unexpected_cnt, 0ul, 0ul,
                                             UCT_RC_MLX5_SRQ_FLAG_TM_SW_CNT);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_tag_handle_unexp(uct_rc_mlx5_iface_common_t *mlx5_common_iface,
                                   uct_rc_iface_t *rc_iface, struct mlx5_cqe64 *cqe,
                                   unsigned byte_len)
{
    struct ibv_exp_tmh *tmh;
    uint64_t           imm_data;
    ucs_status_t       status;
    unsigned           flags;

    tmh = uct_rc_mlx5_iface_common_data(mlx5_common_iface, rc_iface, cqe,
                                        byte_len, &flags);

    switch (tmh->opcode) {
    case IBV_EXP_TMH_EAGER:
        imm_data = uct_rc_iface_tag_imm_data_unpack(cqe->imm_inval_pkey,
                                                    ntohl(tmh->app_ctx),
                                                    (cqe->op_own >> 4) ==
                                                    MLX5_CQE_RESP_SEND_IMM);

        if (UCT_RC_MLX5_TM_IS_SW_RNDV(cqe, cqe->imm_inval_pkey)) {
            status = rc_iface->tm.rndv_unexp.cb(rc_iface->tm.rndv_unexp.arg,
                                                flags, be64toh(tmh->tag), tmh + 1,
                                                byte_len - sizeof(*tmh),
                                                0ul, 0, NULL);
        } else {
            status = rc_iface->tm.eager_unexp.cb(rc_iface->tm.eager_unexp.arg,
                                                 tmh + 1, byte_len - sizeof(*tmh),
                                                 flags, be64toh(tmh->tag),
                                                 imm_data);
        }

        uct_rc_mlx5_iface_unexp_consumed(mlx5_common_iface, rc_iface,
                                         &rc_iface->tm.eager_desc, status,
                                         ntohs(cqe->wqe_counter));
        break;

    case IBV_EXP_TMH_RNDV:
        status = uct_rc_iface_handle_rndv(rc_iface, tmh, byte_len);

        uct_rc_mlx5_iface_unexp_consumed(mlx5_common_iface, rc_iface,
                                         &rc_iface->tm.rndv_desc, status,
                                         ntohs(cqe->wqe_counter));
        break;

    default:
        ucs_fatal("Unsupported packet arrived %d", tmh->opcode);
        break;
    }
}

#endif /* IBV_EXP_HW_TM */

static UCS_F_ALWAYS_INLINE unsigned
uct_rc_mlx5_iface_common_poll_rx(uct_rc_mlx5_iface_common_t *mlx5_common_iface,
                                 uct_rc_iface_t *rc_iface, int is_tag_enabled)
{
    uct_ib_mlx5_srq_seg_t UCS_V_UNUSED *seg;
    struct mlx5_cqe64 *cqe;
    unsigned byte_len;
    uint16_t max_batch;
    unsigned count;
    void *rc_hdr;
    unsigned flags;
#if IBV_EXP_HW_TM
    struct ibv_exp_tmh *tmh;
    uct_rc_mlx5_tag_entry_t *tag;
    uct_tag_context_t *ctx;
    uct_rc_iface_ctx_priv_t *priv;
#endif

    ucs_assert(uct_ib_mlx5_srq_get_wqe(&mlx5_common_iface->rx.srq,
                                       mlx5_common_iface->rx.srq.mask)->srq.next_wqe_index == 0);

    cqe = uct_rc_mlx5_iface_poll_rx_cq(mlx5_common_iface, rc_iface);
    if (cqe == NULL) {
        /* If no CQE - post receives */
        count = 0;
        goto done;
    }

    ucs_memory_cpu_load_fence();
    UCS_STATS_UPDATE_COUNTER(rc_iface->stats, UCT_RC_IFACE_STAT_RX_COMPLETION, 1);

    byte_len = ntohl(cqe->byte_cnt);
    count    = 1;

    if (!is_tag_enabled) {
        rc_hdr = uct_rc_mlx5_iface_common_data(mlx5_common_iface, rc_iface, cqe,
                                               byte_len, &flags);
        uct_rc_mlx5_iface_common_am_handler(mlx5_common_iface, rc_iface, cqe,
                                            rc_hdr, flags, byte_len);
        goto done;
    }

#if IBV_EXP_HW_TM
    ucs_assert(cqe->app == UCT_RC_MLX5_CQE_APP_TAG_MATCHING);

    switch (cqe->app_op) {
    case UCT_RC_MLX5_CQE_APP_OP_TM_APPEND:
        uct_rc_mlx5_iface_handle_tm_list_op(mlx5_common_iface,
                                            UCT_RC_MLX5_CQE_APP_OP_TM_APPEND);
        break;

    case UCT_RC_MLX5_CQE_APP_OP_TM_REMOVE:
        uct_rc_mlx5_iface_handle_tm_list_op(mlx5_common_iface,
                                            UCT_RC_MLX5_CQE_APP_OP_TM_REMOVE);
        break;

    case UCT_RC_MLX5_CQE_APP_OP_TM_NO_TAG:
        tmh = uct_rc_mlx5_iface_common_data(mlx5_common_iface, rc_iface, cqe,
                                            byte_len, &flags);
        if (tmh->opcode == IBV_EXP_TMH_NO_TAG) {
            uct_rc_mlx5_iface_common_am_handler(mlx5_common_iface, rc_iface,
                                                cqe, (uct_rc_hdr_t*)tmh, flags,
                                                byte_len);
        } else {
            ucs_assert(tmh->opcode == IBV_EXP_TMH_FIN);
            uct_rc_iface_handle_rndv_fin(rc_iface, tmh);
            seg = uct_ib_mlx5_srq_get_wqe(&mlx5_common_iface->rx.srq,
                                          ntohs(cqe->wqe_counter));

            uct_rc_mlx5_iface_release_srq_seg(mlx5_common_iface, rc_iface, seg,
                                              ntohs(cqe->wqe_counter), UCS_OK, 0,
                                              NULL);
        }
        break;

    case UCT_RC_MLX5_CQE_APP_OP_TM_UNEXPECTED:
        uct_rc_mlx5_iface_tag_handle_unexp(mlx5_common_iface, rc_iface, cqe,
                                           byte_len);
        break;

    case UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED:
        uct_rc_mlx5_iface_tag_consumed(mlx5_common_iface, cqe,
                                       UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED);
        break;

    case UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED_MSG:
        tmh = (struct ibv_exp_tmh*)cqe;

        uct_rc_mlx5_iface_tag_consumed(mlx5_common_iface, cqe,
                                       UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED_MSG);

        uct_rc_mlx5_iface_handle_expected(mlx5_common_iface, rc_iface, cqe,
                                          byte_len, be64toh(tmh->tag),
                                          ntohl(tmh->app_ctx));
        break;

    case UCT_RC_MLX5_CQE_APP_OP_TM_EXPECTED:
        tag  = &mlx5_common_iface->tm.list[ntohs(cqe->app_info)];
        ctx  = tag->ctx;
        priv = uct_rc_iface_ctx_priv(ctx);
        uct_rc_mlx5_iface_handle_expected(mlx5_common_iface, rc_iface, cqe,
                                          byte_len, priv->tag, priv->app_ctx);
        break;

    default:
        ucs_fatal("Unsupported packet arrived %d", cqe->app_op);
        break;
    }
#endif

done:
    max_batch = rc_iface->super.config.rx_max_batch;
    if (ucs_unlikely(rc_iface->rx.srq.available >= max_batch)) {
        uct_rc_mlx5_iface_srq_post_recv(rc_iface, &mlx5_common_iface->rx.srq);
    }
    return count;
}

#endif
