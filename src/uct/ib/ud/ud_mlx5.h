/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 *
 * $COPYRIGHT$
 * $HEADER$
 */
#ifndef UD_MLX5_H
#define UD_MLX5_H

#include <uct/ib/base/ib_verbs.h>
#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/ib/mlx5/ib_mlx5_log.h>

#include "ud_iface.h"
#include "ud_ep.h"
#include "ud_def.h"

typedef struct uct_ib_mlx5_wq {
    uint16_t       sw_pi;      /* PI for next WQE */
    uint16_t       max_pi;     /* Maximal PI which can start a new WQE, in case of rx treat this as mask */
    uint16_t       prev_sw_pi; /* PI where last WQE *started*  */
    unsigned       bf_size;
    void           *bf_reg;
    uint32_t       *dbrec;
    void           *seg;
    void           *qstart;
    void           *qend;
} uct_ib_mlx5_txwq_t;

typedef uint16_t uct_ib_mlx5_index_t;

typedef struct uct_ib_mlx5_rxwq {
    /* producer index. It updated when new receive wqe is posted */
    uct_ib_mlx5_index_t       rq_wqe_counter; 
    /* consumer index. It is better to track it ourselves than to do ntohs() 
     * on the index in the cqe
     */
    uct_ib_mlx5_index_t       cq_wqe_counter;
    uct_ib_mlx5_index_t       mask;
    uint32_t                 *dbrec;
    struct mlx5_wqe_data_seg *wqes;
} uct_ib_mlx5_rxwq_t;

typedef struct {
    uct_ud_ep_t          super;
    struct mlx5_wqe_av   av;
} uct_ud_mlx5_ep_t;

typedef struct {
    uct_ud_iface_t        super;
    struct {
        uct_ib_mlx5_txwq_t  wq; 
        uct_ib_mlx5_cq_t    cq;
    } tx;
    struct {
        uct_ib_mlx5_rxwq_t  wq; 
        uct_ib_mlx5_cq_t    cq;
    } rx;
} uct_ud_mlx5_iface_t;


#define UCT_UD_MLX5_MAX_BB 4
#define MXM_IB_MLX5_EXTENDED_UD_AV 0x80000000

#if HAVE_STRUCT_MLX5_WQE_AV_BASE
#  define mlx5_av_base(_av)   (&(_av)->base)
#  define mlx5_av_grh(_av)    (&(_av)->grh_sec)
#else 
#  define mlx5_av_base(_av)   (_av)
#  define mlx5_av_grh(_av)    (_av)
#endif

/* TODO: move common code and update rc_mlx5 header */
struct uct_ib_mlx5_ctrl_dgram_seg {
    struct mlx5_wqe_ctrl_seg     ctrl;
    struct mlx5_wqe_datagram_seg dgram;
} UCS_S_PACKED;

/**
 * Copy data to inline segment, taking into account QP wrap-around.
 *
 * @param dest    Inline data in the WQE to copy to.
 * @param src     Data to copy.
 * @param length  Data length.
 *
 * @return If there was a wrap-around, return -qp_size. Otherwise, return 0.
 */
static inline void uct_ib_mlx5_inline_copy(void *dest, const void *src, unsigned length, uct_ib_mlx5_txwq_t *wq)
{
    void *qend = wq->qend;
    ptrdiff_t n;

    if (dest + length <= qend) {
        memcpy(dest, src, length);
    } else {
        n = qend - dest;
        memcpy(dest, src, n);
        memcpy(wq->qstart, src + n, length - n);
    }
}

static inline void *uct_ib_mlx5_get_next_seg(uct_ib_mlx5_txwq_t *wq, char *seg_base, int seg_len)
{
    void *rseg;

    rseg = seg_base + seg_len;
    if (ucs_unlikely(rseg >= wq->qend)) {
        rseg = wq->qstart;
    }
    ucs_assert((unsigned long)rseg % UCT_IB_MLX5_WQE_SEG_SIZE == 0);
    ucs_assert(rseg >= wq->qstart && rseg < wq->qend);
    return rseg;
}

/* TODO: we do not need this function */
static UCS_F_ALWAYS_INLINE uint16_t
uct_ud_mlx5_calc_max_pi(uct_ud_mlx5_iface_t *iface, uint16_t ci)
{
    return ci + iface->super.config.tx_qp_len - UCT_UD_MLX5_MAX_BB + 1;
}

static inline void uct_ib_mlx5_set_dgram_seg(struct mlx5_wqe_datagram_seg *seg,
                                             struct mlx5_wqe_av *av,
                                             uint8_t path_bits)
{

    mlx5_av_base(&seg->av)->key.qkey.qkey  = htonl(UCT_UD_QKEY);
    mlx5_av_base(&seg->av)->stat_rate_sl   = mlx5_av_base(av)->stat_rate_sl;
    mlx5_av_base(&seg->av)->fl_mlid        = mlx5_av_base(av)->fl_mlid | path_bits;
    mlx5_av_base(&seg->av)->rlid           = mlx5_av_base(av)->rlid | (path_bits << 8);
    mlx5_av_base(&seg->av)->dqp_dct        = mlx5_av_base(av)->dqp_dct;
/*  No need to fill grh
    seg->av.grh_sec.tclass      = av->grh_sec.tclass;
    seg->av.grh_sec.hop_limit   = av->grh_sec.hop_limit;
    seg->av.grh_sec.grh_gid_fl  = av->grh_sec.grh_gid_fl;
*/
    mlx5_av_grh(&seg->av)->grh_gid_fl  = 0;
}

static inline void uct_ib_mlx5_set_ctrl_seg(struct mlx5_wqe_ctrl_seg* ctrl, uint16_t pi,
                                            uint8_t opcode, uint8_t opmod, uint32_t qp_num,
                                            uint8_t fm_ce_se, unsigned wqe_size)
{
    uint8_t ds;

    ucs_assert((unsigned long)ctrl % UCT_IB_MLX5_WQE_SEG_SIZE == 0);
    ds = ucs_div_round_up(wqe_size, UCT_IB_MLX5_WQE_SEG_SIZE);
    ucs_trace_data("ds=%d wqe_size=%d", ds, wqe_size);
#ifdef __SSE4_2__
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
#else
    ctrl->opmod_idx_opcode = (opcode << 24) | (htons(pi) << 8) | opmod;
    ctrl->qpn_ds           = htonl((qp_num << 8) | ds);
    ctrl->fm_ce_se         = fm_ce_se;
#endif
}

static inline void uct_ib_mlx5_set_dptr_seg(struct mlx5_wqe_data_seg *dptr, 
                                            const void *address,
                                            unsigned length, uint32_t lkey)
{
    ucs_assert((unsigned long)dptr % UCT_IB_MLX5_WQE_SEG_SIZE == 0);
    dptr->byte_count = htonl(length);
    dptr->lkey       = htonl(lkey);
    dptr->addr       = htonll((uintptr_t)address);
}


static inline void uct_ib_mlx5_bf_copy_bb(void *dst, void *src)
{
#ifdef __SSE4_2__
        UCS_WORD_COPY(dst, src, __m128i, MLX5_SEND_WQE_BB);
#else 
        UCS_WORD_COPY(dst, src, uint64_t, MLX5_SEND_WQE_BB);
#endif
}


static inline void uct_ib_mlx5_post_send(uct_ib_mlx5_txwq_t *wq, struct mlx5_wqe_ctrl_seg *ctrl, unsigned wqe_size)
{
    unsigned n, num_bb;
    void *src, *dst;
    uint16_t sw_pi;

    ucs_assert((unsigned long)ctrl % UCT_IB_MLX5_WQE_SEG_SIZE == 0);
    uct_ib_mlx5_log_tx(IBV_QPT_UD, ctrl, wq->qstart, wq->qend, NULL);
    num_bb  = ucs_div_round_up(wqe_size, MLX5_SEND_WQE_BB);
    sw_pi   = wq->sw_pi;

    /* TODO Put memory store fence here too, to prevent WC being flushed after DBrec */
    ucs_memory_cpu_store_fence();

    /* Write doorbell record */
    wq->prev_sw_pi = sw_pi;
    *wq->dbrec = htonl(sw_pi += num_bb);

    /* Make sure that doorbell record is written before ringing the doorbell */
    ucs_memory_bus_store_fence();

    /* Set up copy pointers */
    dst = wq->bf_reg;
    src = ctrl;

    /* BF copy */
    /* TODO support DB without BF */
    ucs_assert(wqe_size <= wq->bf_size);
    ucs_assert(num_bb <= UCT_UD_MLX5_MAX_BB);
    for (n = 0; n < num_bb; ++n) {
        uct_ib_mlx5_bf_copy_bb(dst, src);
        dst += MLX5_SEND_WQE_BB;
        src += MLX5_SEND_WQE_BB;
        if (ucs_unlikely(src == wq->qend)) {
            src = wq->qstart;
        }
    }

    /* We don't want the compiler to reorder instructions and hurt latency */
    ucs_compiler_fence();

    /* Advance queue pointer */
    ucs_assert(ctrl == wq->seg);
    wq->seg   = src;
    wq->sw_pi = sw_pi;

    /* Flip BF register */
    wq->bf_reg = (void*) ((uintptr_t) wq->bf_reg ^ wq->bf_size);
}


static inline unsigned uct_ud_mlx5_tx_moderation(uct_ud_mlx5_iface_t *iface)
{
    if (iface->super.tx.unsignaled >= UCT_UD_TX_MODERATION) {
        iface->super.tx.unsignaled = 0;
        return MLX5_WQE_CTRL_CQ_UPDATE;
    }
    iface->super.tx.unsignaled++;
    return 0;
}

#endif

