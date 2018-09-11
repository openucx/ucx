/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ib_mlx5.h"


static UCS_F_ALWAYS_INLINE struct mlx5_cqe64*
uct_ib_mlx5_get_cqe(uct_ib_mlx5_cq_t *cq,  unsigned index)
{
    return cq->cq_buf + ((index & (cq->cq_length - 1)) << cq->cqe_size_log);
}

static UCS_F_ALWAYS_INLINE struct mlx5_cqe64*
uct_ib_mlx5_poll_cq(uct_ib_iface_t *iface, uct_ib_mlx5_cq_t *cq)
{
    struct mlx5_cqe64 *cqe;
    unsigned index;
    uint8_t op_own;

    index  = cq->cq_ci;
    cqe    = uct_ib_mlx5_get_cqe(cq, index);
    op_own = cqe->op_own;

    if (ucs_unlikely((op_own & MLX5_CQE_OWNER_MASK) == !(index & cq->cq_length))) {
        return NULL;
    } else if (ucs_unlikely(op_own & 0x80)) {
        UCS_STATIC_ASSERT(MLX5_CQE_INVALID & (UCT_IB_MLX5_CQE_OP_OWN_ERR_MASK >> 4));
        if (ucs_unlikely((op_own >> 4) != MLX5_CQE_INVALID)) {
            uct_ib_mlx5_check_completion(iface, cq, cqe);
        }
        return NULL; /* No CQE */
    }

    cq->cq_ci = index + 1;
    return cqe; /* TODO optimize - let complier know cqe is not null */
}


static UCS_F_ALWAYS_INLINE uint16_t
uct_ib_mlx5_txwq_update_bb(uct_ib_mlx5_txwq_t *wq, uint16_t hw_ci)
{
#if ENABLE_ASSERT
    wq->hw_ci = hw_ci;
#endif
    return wq->bb_max - (wq->prev_sw_pi - hw_ci);
}


/* check that work queue has enough space for the new work request */
static inline void
uct_ib_mlx5_txwq_validate(uct_ib_mlx5_txwq_t *wq, uint16_t num_bb)
{

#if ENABLE_ASSERT
    uint16_t wqe_s, wqe_e;
    uint16_t hw_ci, sw_pi;
    uint16_t wqe_cnt;
    int is_ok = 1;

    if (wq->hw_ci == 0xFFFF) {
        return;
    }

    wqe_cnt = (wq->qend - wq->qstart) / MLX5_SEND_WQE_BB;
    if (wqe_cnt < wq->bb_max) {
        ucs_fatal("wqe count (%u) < bb_max (%u)", wqe_cnt, wq->bb_max);
    }

    wqe_s = (wq->curr - wq->qstart) / MLX5_SEND_WQE_BB;
    wqe_e = (wqe_s + num_bb) % wqe_cnt;

    sw_pi = wq->prev_sw_pi % wqe_cnt;
    hw_ci = wq->hw_ci % wqe_cnt;

    if (hw_ci <= sw_pi) {
        if (hw_ci <= wqe_s && wqe_s <= sw_pi) {
            is_ok = 0;
        }
        if (hw_ci <= wqe_e && wqe_e <= sw_pi) {
            is_ok = 0;
        }
    }
    else {
        if (!(sw_pi < wqe_s && wqe_s < hw_ci)) {
            is_ok = 0;
        }
        if (!(sw_pi < wqe_e && wqe_e < hw_ci)) {
            is_ok = 0;
        }
    }
    if (!is_ok) {
        ucs_fatal("tx wq overrun: hw_ci: %u sw_pi: %u cur: %u-%u num_bb: %u wqe_cnt: %u",
                hw_ci, sw_pi, wqe_s, wqe_e, num_bb, wqe_cnt);
    }
#endif
}


/**
 * Copy data to inline segment, taking into account QP wrap-around.
 *
 * @param dest    Inline data in the WQE to copy to.
 * @param src     Data to copy.
 * @param length  Data length.
 *
 */
static UCS_F_ALWAYS_INLINE void
uct_ib_mlx5_inline_copy(void *restrict dest, const void *restrict src, unsigned
                        length, uct_ib_mlx5_txwq_t *wq)
{
    ptrdiff_t n;

    if (dest + length <= wq->qend) {
        memcpy(dest, src, length);
    } else {
        n = wq->qend - dest;
        memcpy(dest, src, n);
        memcpy(wq->qstart, src + n, length - n);
    }
}


/* wrapping of 'seg' should not happen */
static UCS_F_ALWAYS_INLINE void*
uct_ib_mlx5_txwq_wrap_none(uct_ib_mlx5_txwq_t *txwq, void *seg)
{
    ucs_assertv(((unsigned long)seg % UCT_IB_MLX5_WQE_SEG_SIZE) == 0, "seg=%p", seg);
    ucs_assertv(seg >= txwq->qstart, "seg=%p qstart=%p", seg, txwq->qstart);
    ucs_assertv(seg <  txwq->qend,   "seg=%p qend=%p",   seg, txwq->qend);
    return seg;
}


/* wrapping of 'seg' could happen, but only on exact 'qend' boundary */
static UCS_F_ALWAYS_INLINE void *
uct_ib_mlx5_txwq_wrap_exact(uct_ib_mlx5_txwq_t *txwq, void *seg)
{
    ucs_assert(seg <= txwq->qend);
    if (ucs_unlikely(seg == txwq->qend)) {
        seg = txwq->qstart;
    }
    return uct_ib_mlx5_txwq_wrap_none(txwq, seg);
}


/* wrapping of 'seg' could happen, even past 'qend' boundary */
static UCS_F_ALWAYS_INLINE void *
uct_ib_mlx5_txwq_wrap_any(uct_ib_mlx5_txwq_t *txwq, void *seg)
{
    if (ucs_unlikely(seg >= txwq->qend)) {
        seg -= (txwq->qend - txwq->qstart);
    }
    return uct_ib_mlx5_txwq_wrap_none(txwq, seg);
}


/* Wrapping of 'data' could happen, even past 'qend' boundary.
 * Do not check for alignment. */
static UCS_F_ALWAYS_INLINE void *
uct_ib_mlx5_txwq_wrap_data(uct_ib_mlx5_txwq_t *txwq, void *data)
{
    if (ucs_unlikely(data >= txwq->qend)) {
        data -= (txwq->qend - txwq->qstart);
    }
    return data;
}


static UCS_F_ALWAYS_INLINE void
uct_ib_mlx5_ep_set_rdma_seg(struct mlx5_wqe_raddr_seg *raddr, uint64_t rdma_raddr,
                            uct_rkey_t rdma_rkey)
{
#if defined(__SSE4_2__)
    *(__m128i*)raddr = _mm_shuffle_epi8(
                _mm_set_epi64x(rdma_rkey, rdma_raddr),
                _mm_set_epi8(0, 0, 0, 0,            /* reserved */
                             8, 9, 10, 11,          /* rkey */
                             0, 1, 2, 3, 4, 5, 6, 7 /* rdma_raddr */
                             ));
#elif defined(__ARM_NEON)
    uint8x16_t table =  {7,  6,  5, 4, 3, 2, 1, 0, /* rdma_raddr */
                         11, 10, 9, 8,             /* rkey */
                         16,16,16,16};             /* reserved (set 0) */
    uint64x2_t data = {rdma_raddr, rdma_rkey};
    *(uint8x16_t *)raddr = vqtbl1q_u8((uint8x16_t)data, table);
#else
    raddr->raddr = htobe64(rdma_raddr);
    raddr->rkey  = htonl(rdma_rkey);
#endif
}


static UCS_F_ALWAYS_INLINE void
uct_ib_mlx5_set_dgram_seg(struct mlx5_wqe_datagram_seg *seg,
                          uct_ib_mlx5_base_av_t *av, struct mlx5_grh_av *grh_av,
                          enum ibv_qp_type qp_type)
{
    if (qp_type == IBV_QPT_UD) {
        mlx5_av_base(&seg->av)->key.qkey.qkey  = htonl(UCT_IB_KEY);
#if HAVE_TL_DC
    } else if (qp_type == IBV_EXP_QPT_DC_INI) {
        mlx5_av_base(&seg->av)->key.dc_key     = htobe64(UCT_IB_KEY);
#endif
    }
    mlx5_av_base(&seg->av)->dqp_dct            = av->dqp_dct;
    mlx5_av_base(&seg->av)->stat_rate_sl       = av->stat_rate_sl;
    mlx5_av_base(&seg->av)->fl_mlid            = av->fl_mlid;
    mlx5_av_base(&seg->av)->rlid               = av->rlid;

    if (grh_av) {
        ucs_assert(av->dqp_dct & UCT_IB_MLX5_EXTENDED_UD_AV);
#if HAVE_STRUCT_MLX5_GRH_AV_RMAC
        memcpy(mlx5_av_grh(&seg->av)->rmac, grh_av->rmac,
               sizeof(mlx5_av_grh(&seg->av)->rmac));
#endif
        mlx5_av_grh(&seg->av)->tclass      = grh_av->tclass;
        mlx5_av_grh(&seg->av)->hop_limit   = grh_av->hop_limit;
        mlx5_av_grh(&seg->av)->grh_gid_fl  = grh_av->grh_gid_fl;
        memcpy(mlx5_av_grh(&seg->av)->rgid, grh_av->rgid,
               sizeof(mlx5_av_grh(&seg->av)->rgid));
    } else if (av->dqp_dct & UCT_IB_MLX5_EXTENDED_UD_AV) {
        mlx5_av_grh(&seg->av)->grh_gid_fl  = 0;
    }
}


static UCS_F_ALWAYS_INLINE void
uct_ib_mlx5_set_ctrl_seg(struct mlx5_wqe_ctrl_seg* ctrl, uint16_t pi,
                         uint8_t opcode, uint8_t opmod, uint32_t qp_num,
                         uint8_t fm_ce_se, unsigned wqe_size)
{
    uint8_t ds;

    ucs_assert(((unsigned long)ctrl % UCT_IB_MLX5_WQE_SEG_SIZE) == 0);
    ds = ucs_div_round_up(wqe_size, UCT_IB_MLX5_WQE_SEG_SIZE);
#if defined(__SSE4_2__)
    *(__m128i *) ctrl = _mm_shuffle_epi8(
                    _mm_set_epi32(qp_num, ds, pi,
                                  (opcode << 16) | (opmod << 8) | fm_ce_se), /* OR of constants */
                    _mm_set_epi8(0, 0, 0, 0, /* immediate */
                                 0,          /* signal/fence_mode */
                                 0, 0,       /* reserved */
                                 0,          /* signature */
                                 8,          /* data size */
                                 12, 13, 14, /* QP num */
                                 2,          /* opcode */
                                 4, 5,       /* sw_pi in BE */
                                 1           /* opmod */
                                 ));
#elif defined(__ARM_NEON)
    uint8x16_t table = {1,               /* opmod */
                        5,  4,           /* sw_pi in BE */
                        2,               /* opcode */
                        14, 13, 12,      /* QP num */
                        8,               /* data size */
                        16,              /* signature (set 0) */
                        16, 16,          /* reserved (set 0) */
                        0,               /* signal/fence_mode */
                        16, 16, 16, 16}; /* immediate (set to 0)*/
    uint32x4_t data = {(opcode << 16) | (opmod << 8) | (uint32_t)fm_ce_se,
                       pi, ds, qp_num};
    *(uint8x16_t *)ctrl = vqtbl1q_u8((uint8x16_t)data, table);
#else
    ctrl->opmod_idx_opcode = (opcode << 24) | (htons(pi) << 8) | opmod;
    ctrl->qpn_ds           = htonl((qp_num << 8) | ds);
    ctrl->fm_ce_se         = fm_ce_se;
#endif
}


static UCS_F_ALWAYS_INLINE void
uct_ib_mlx5_set_ctrl_seg_with_imm(struct mlx5_wqe_ctrl_seg* ctrl, uint16_t pi,
                                  uint8_t opcode, uint8_t opmod, uint32_t qp_num,
                                  uint8_t fm_ce_se, unsigned wqe_size, uint32_t imm)
{
    uint8_t ds;

    ucs_assert(((unsigned long)ctrl % UCT_IB_MLX5_WQE_SEG_SIZE) == 0);
    ds = ucs_div_round_up(wqe_size, UCT_IB_MLX5_WQE_SEG_SIZE);
#if defined(__SSE4_2__)
    *(__m128i *) ctrl = _mm_shuffle_epi8(
                    _mm_set_epi32(qp_num, imm, (ds << 16) | pi,
                                  (opcode << 16) | (opmod << 8) | fm_ce_se), /* OR of constants */
                    _mm_set_epi8(11, 10, 9, 8, /* immediate */
                                 0,            /* signal/fence_mode */
                                 0, 0,         /* reserved */
                                 0,            /* signature */
                                 6,            /* data size */
                                 12, 13, 14,   /* QP num */
                                 2,            /* opcode */
                                 4, 5,         /* sw_pi in BE */
                                 1             /* opmod */
                                 ));
#elif defined(__ARM_NEON)
    uint8x16_t table = {1,               /* opmod */
                        5,  4,           /* sw_pi in BE */
                        2,               /* opcode */
                        14, 13, 12,      /* QP num */
                        6,               /* data size */
                        16,              /* signature (set 0) */
                        16, 16,          /* reserved (set 0) */
                        0,               /* signal/fence_mode */
                        8, 9, 10, 11}; /* immediate (set to 0)*/
    uint32x4_t data = {(opcode << 16) | (opmod << 8) | (uint32_t)fm_ce_se,
                       (ds << 16) | pi, imm,  qp_num};
    *(uint8x16_t *)ctrl = vqtbl1q_u8((uint8x16_t)data, table);
#else
    ctrl->opmod_idx_opcode = (opcode << 24) | (htons(pi) << 8) | opmod;
    ctrl->qpn_ds           = htonl((qp_num << 8) | ds);
    ctrl->fm_ce_se         = fm_ce_se;
    ctrl->imm              = imm;
#endif
}


static UCS_F_ALWAYS_INLINE void
uct_ib_mlx5_set_data_seg(struct mlx5_wqe_data_seg *dptr,
                         const void *address,
                         unsigned length, uint32_t lkey)
{
    ucs_assert(((unsigned long)dptr % UCT_IB_MLX5_WQE_SEG_SIZE) == 0);
    dptr->byte_count = htonl(length);
    dptr->lkey       = htonl(lkey);
    dptr->addr       = htobe64((uintptr_t)address);
}


static UCS_F_ALWAYS_INLINE
unsigned uct_ib_mlx5_set_data_seg_iov(uct_ib_mlx5_txwq_t *txwq,
                                      struct mlx5_wqe_data_seg *dptr,
                                      const uct_iov_t *iov, size_t iovcnt)
{
    unsigned len = 0;
    size_t   iov_it;

    for (iov_it = 0; iov_it < iovcnt; ++iov_it) {
        if (!iov[iov_it].length) { /* Skip zero length WQE*/
            continue;
        }
        ucs_assert(iov[iov_it].memh != UCT_MEM_HANDLE_NULL);

        /* place data into the buffer */
        dptr = uct_ib_mlx5_txwq_wrap_any(txwq, dptr);
        uct_ib_mlx5_set_data_seg(dptr, iov[iov_it].buffer, iov[iov_it].length,
                                 ((uct_ib_mem_t*)iov[iov_it].memh)->lkey);
        len += sizeof(*dptr);
        ++dptr;
    }
    return len;
}


static UCS_F_ALWAYS_INLINE void uct_ib_mlx5_bf_copy_bb(void * restrict dst,
                                                       void * restrict src)
{
#if defined( __SSE4_2__)
    UCS_WORD_COPY(__m128i, dst, __m128i, src, MLX5_SEND_WQE_BB);
#elif defined(__ARM_NEON)
    UCS_WORD_COPY(int16x8_t, dst, int16x8_t, src, MLX5_SEND_WQE_BB);
#else /* NO SIMD support */
    UCS_WORD_COPY(uint64_t, dst, uint64_t, src, MLX5_SEND_WQE_BB);
#endif
}


static UCS_F_ALWAYS_INLINE uint16_t
uct_ib_mlx5_post_send(uct_ib_mlx5_txwq_t *wq,
                      struct mlx5_wqe_ctrl_seg *ctrl, unsigned wqe_size)
{
    uint16_t n, sw_pi, num_bb, res_count;
    void *src, *dst;

    ucs_assert(((unsigned long)ctrl % UCT_IB_MLX5_WQE_SEG_SIZE) == 0);
    num_bb  = ucs_div_round_up(wqe_size, MLX5_SEND_WQE_BB);
    sw_pi   = wq->sw_pi;

    uct_ib_mlx5_txwq_validate(wq, num_bb);
    /* TODO Put memory store fence here too, to prevent WC being flushed after DBrec */
    ucs_memory_cpu_store_fence();

    /* Write doorbell record */
    *wq->dbrec = htonl(sw_pi += num_bb);

    /* Make sure that doorbell record is written before ringing the doorbell */
    ucs_memory_bus_store_fence();

    /* Set up copy pointers */
    dst = wq->bf->reg.ptr;
    src = ctrl;

    ucs_assert(wqe_size <= UCT_IB_MLX5_BF_REG_SIZE);
    ucs_assert(num_bb <= UCT_IB_MLX5_MAX_BB);
    if (ucs_likely(wq->bf->enable_bf)) {
        /* BF copy */
        for (n = 0; n < num_bb; ++n) {
            uct_ib_mlx5_bf_copy_bb(dst, src);
            dst += MLX5_SEND_WQE_BB;
            src += MLX5_SEND_WQE_BB;
            if (ucs_unlikely(src == wq->qend)) {
                src = wq->qstart;
            }
        }
    } else {
        /* DB copy */
        *(volatile uint64_t *)dst = *(volatile uint64_t *)src;
        ucs_memory_bus_store_fence();
        src = uct_ib_mlx5_txwq_wrap_any(wq, src + (num_bb * MLX5_SEND_WQE_BB));
    }

    /* We don't want the compiler to reorder instructions and hurt latency */
    ucs_compiler_fence();

    /*
     * Advance queue pointer.
     * We return the number of BBs the *previous* WQE has consumed, since CQEs
     * are reporting the index of the first BB rather than the last. We have
     * reserved QP space for at least UCT_IB_MLX5_MAX_BB to accommodate.
     * */
    ucs_assert(ctrl == wq->curr);
    res_count       = wq->sw_pi - wq->prev_sw_pi;
    wq->curr        = src;
    wq->prev_sw_pi += res_count;
    ucs_assert(wq->prev_sw_pi == wq->sw_pi);
    wq->sw_pi       = sw_pi;

    /* Flip BF register */
    wq->bf->reg.addr ^= UCT_IB_MLX5_BF_REG_SIZE;
    return res_count;
}


static inline uct_ib_mlx5_srq_seg_t *
uct_ib_mlx5_srq_get_wqe(uct_ib_mlx5_srq_t *srq, uint16_t index)
{
    ucs_assert(index <= srq->mask);
    return srq->buf + index * UCT_IB_MLX5_SRQ_STRIDE;
}
