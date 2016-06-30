/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IB_MLX5_H_
#define UCT_IB_MLX5_H_


#include <uct/ib/base/ib_log.h>
#include <uct/ib/base/ib_device.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>
#include <ucs/type/status.h>

#include <infiniband/mlx5_hw.h>
#include <infiniband/arch.h>
#include <netinet/in.h>
#include <string.h>

#define UCT_IB_MLX5_WQE_SEG_SIZE     16 /* Size of a segment in a WQE */
#define UCT_IB_MLX5_CQE64_MAX_INL    32 /* Inline scatter size in 64-byte CQE */
#define UCT_IB_MLX5_CQE128_MAX_INL   64 /* Inline scatter size in 128-byte CQE */
#define UCT_IB_MLX5_CQE64_SIZE_LOG   6
#define UCT_IB_MLX5_CQE128_SIZE_LOG  7
#define UCT_IB_MLX5_MAX_BB           4
#define UCT_IB_MLX5_WORKER_BF_KEY  0x00c1b7e8U
#define UCT_IB_MLX5_EXTENDED_UD_AV   0x80000000U
#define UCT_IB_MLX5_BF_REG_SIZE      256

#define UCT_IB_MLX5_OPMOD_EXT_ATOMIC(_log_arg_size) \
    ((8) | ((_log_arg_size) - 2))

#if HAVE_STRUCT_MLX5_WQE_AV_BASE
#  define mlx5_av_base(_av)   (&(_av)->base)
#  define mlx5_av_grh(_av)    (&(_av)->grh_sec)
#else 
#  define mlx5_av_base(_av)   (_av)
#  define mlx5_av_grh(_av)    (_av)
#endif

#if !(HAVE_MLX5_WQE_CTRL_SOLICITED)
#  define MLX5_WQE_CTRL_SOLICITED  (1<<1)
#endif


typedef struct uct_ib_mlx5_qp_info {
    uint32_t           qpn;           /* QP number */
    volatile uint32_t  *dbrec;        /* QP doorbell record in RAM */

    struct {
            void       *buf;          /* Work queue buffer */
            unsigned   wqe_cnt;       /* Number of WQEs in the work queue */
            unsigned   stride;        /* Size of each WQE */
    } sq, rq;

    struct {
            void       *reg;          /* BlueFlame register */
            unsigned   size;          /* BlueFlame register size (0 - unsupported) */
    } bf;
} uct_ib_mlx5_qp_info_t;


typedef struct uct_ib_mlx5_srq_info {
    void               *buf;          /* SRQ queue buffer */
    volatile uint32_t  *dbrec;        /* SRQ doorbell record in RAM */
    unsigned           stride;        /* Size of each WQE */
    unsigned           head;
    unsigned           tail;
} uct_ib_mlx5_srq_info_t;


typedef struct uct_ib_mlx5_cq {
    void               *cq_buf;
    unsigned           cq_ci;
    unsigned           cq_length;
    unsigned           cqe_size_log;
} uct_ib_mlx5_cq_t;


struct uct_ib_mlx5_atomic_masked_cswap32_seg {
    uint32_t           swap;
    uint32_t           compare;
    uint32_t           swap_mask;
    uint32_t           compare_mask;
} UCS_S_PACKED;


struct uct_ib_mlx5_atomic_masked_fadd32_seg {
    uint32_t           add;
    uint32_t           filed_boundary;
    uint32_t           reserved[2];
} UCS_S_PACKED;


struct uct_ib_mlx5_atomic_masked_cswap64_seg {
    uint64_t           swap;
    uint64_t           compare;
    uint64_t           swap_mask;
    uint64_t           compare_mask;
} UCS_S_PACKED;

/**
 * Get internal QP information.
 */
ucs_status_t 
uct_ib_mlx5_get_qp_info(struct ibv_qp *qp, uct_ib_mlx5_qp_info_t *qp_info);

/**
 * Get internal SRQ information.
 */
ucs_status_t
uct_ib_mlx5_get_srq_info(struct ibv_srq *srq, uct_ib_mlx5_srq_info_t *srq_info);

/**
 * Get internal CQ information.
 */
ucs_status_t uct_ib_mlx5_get_cq(struct ibv_cq *cq, uct_ib_mlx5_cq_t *mlx5_cq);

/**
 * Update CI to support req_notify_cq
 */
void uct_ib_mlx5_update_cq_ci(struct ibv_cq *cq, unsigned cq_ci);

/**
 * Retrieve CI from the driver
 */
unsigned uct_ib_mlx5_get_cq_ci(struct ibv_cq *cq);

/**
 * Get internal AV information.
 */
void uct_ib_mlx5_get_av(struct ibv_ah *ah, struct mlx5_wqe_av *av);


struct mlx5_cqe64* uct_ib_mlx5_check_completion(uct_ib_mlx5_cq_t *cq,
                                                struct mlx5_cqe64 *cqe);


static inline unsigned uct_ib_mlx5_cqe_size(uct_ib_mlx5_cq_t *cq)
{
    return 1<<cq->cqe_size_log;
}

static UCS_F_ALWAYS_INLINE struct mlx5_cqe64*
uct_ib_mlx5_get_cqe(uct_ib_mlx5_cq_t *cq, int cqe_size_log)
{
    struct mlx5_cqe64 *cqe;
    unsigned index;
    uint8_t op_own;

    index  = cq->cq_ci;
    cqe    = cq->cq_buf + ((index & (cq->cq_length - 1)) << cqe_size_log);
    op_own = cqe->op_own;

    if (ucs_unlikely((op_own & MLX5_CQE_OWNER_MASK) == !(index & cq->cq_length))) {
        return NULL;
    } else if (ucs_unlikely(op_own & 0x80)) {
        if (op_own >> 4 == MLX5_CQE_INVALID) {
            return NULL; /* No CQE */
        } else {
            return uct_ib_mlx5_check_completion(cq, cqe);
        }
    }

    cq->cq_ci = index + 1;
    return cqe; /* TODO optimize - let complier know cqe is not null */
}


/* Blue flame register */
typedef struct uct_ib_mlx5_bf {
    uct_worker_tl_data_t        super;
    union {
        void                    *ptr;
        uintptr_t               addr;
    } reg;
} uct_ib_mlx5_bf_t;


/* Send WQ */
typedef struct uct_ib_mlx5_txwq {
    uint16_t                    sw_pi;      /* PI for next WQE */
    uint16_t                    prev_sw_pi; /* PI where last WQE *started*  */
    uct_ib_mlx5_bf_t            *bf;
    void                        *curr;
    volatile uint32_t           *dbrec;
    void                        *qstart;
    void                        *qend;
    uint16_t                    bb_max;
    uint16_t                    sig_pi;     /* PI for last signaled WQE */
#if ENABLE_ASSERT
    uint16_t                    hw_ci;
#endif
} uct_ib_mlx5_txwq_t;


/* receive WQ */
typedef struct uct_ib_mlx5_rxwq {
    /* producer index. It updated when new receive wqe is posted */
    uint16_t                    rq_wqe_counter;
    /* consumer index. It is better to track it ourselves than to do ntohs() 
     * on the index in the cqe
     */
    uint16_t                    cq_wqe_counter;
    uint16_t                    mask;
    volatile uint32_t           *dbrec;
    struct mlx5_wqe_data_seg    *wqes;
} uct_ib_mlx5_rxwq_t;


ucs_status_t uct_ib_mlx5_get_txwq(uct_worker_h worker, struct ibv_qp *qp,
                                  uct_ib_mlx5_txwq_t *wq);
void uct_ib_mlx5_put_txwq(uct_worker_h worker, uct_ib_mlx5_txwq_t *wq);

ucs_status_t uct_ib_mlx5_get_rxwq(struct ibv_qp *qp, uct_ib_mlx5_rxwq_t *wq);

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
uct_ib_mlx5_inline_copy(void *dest, const void *src, unsigned length, 
                        uct_ib_mlx5_txwq_t *wq)
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

static UCS_F_ALWAYS_INLINE void *
uct_ib_mlx5_get_next_seg(uct_ib_mlx5_txwq_t *wq, void *seg_base, size_t seg_len)
{
    void *rseg;

    rseg = seg_base + seg_len;
    if (ucs_unlikely(rseg >= wq->qend)) {
        rseg = wq->qstart;
    }
    ucs_assert(((unsigned long)rseg % UCT_IB_MLX5_WQE_SEG_SIZE) == 0);
    ucs_assert(rseg >= wq->qstart && rseg < wq->qend);
    return rseg;
}

static UCS_F_ALWAYS_INLINE void
uct_ib_mlx5_set_dgram_seg(struct mlx5_wqe_datagram_seg *seg,
                          struct mlx5_wqe_av *av,
                          uint8_t path_bits)
{

    mlx5_av_base(&seg->av)->key.qkey.qkey  = htonl(UCT_IB_QKEY);
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

static UCS_F_ALWAYS_INLINE void 
uct_ib_mlx5_set_ctrl_seg(struct mlx5_wqe_ctrl_seg* ctrl, uint16_t pi,
                         uint8_t opcode, uint8_t opmod, uint32_t qp_num,
                         uint8_t fm_ce_se, unsigned wqe_size)
{
    uint8_t ds;

    ucs_assert(((unsigned long)ctrl % UCT_IB_MLX5_WQE_SEG_SIZE) == 0);
    ds = ucs_div_round_up(wqe_size, UCT_IB_MLX5_WQE_SEG_SIZE);
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

static UCS_F_ALWAYS_INLINE void 
uct_ib_mlx5_set_data_seg(struct mlx5_wqe_data_seg *dptr, 
                         const void *address,
                         unsigned length, uint32_t lkey)
{
    ucs_assert(((unsigned long)dptr % UCT_IB_MLX5_WQE_SEG_SIZE) == 0);
    dptr->byte_count = htonl(length);
    dptr->lkey       = htonl(lkey);
    dptr->addr       = htonll((uintptr_t)address);
}


static UCS_F_ALWAYS_INLINE void uct_ib_mlx5_bf_copy_bb(void *dst, void *src)
{
#ifdef __SSE4_2__
        UCS_WORD_COPY(dst, src, __m128i, MLX5_SEND_WQE_BB);
#else 
        UCS_WORD_COPY(dst, src, uint64_t, MLX5_SEND_WQE_BB);
#endif
}


static UCS_F_ALWAYS_INLINE uint16_t 
uct_ib_mlx5_post_send(uct_ib_mlx5_txwq_t *wq,
                      struct mlx5_wqe_ctrl_seg *ctrl, unsigned wqe_size)
{
    uint16_t n, sw_pi, num_bb;
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

    /* BF copy */
    /* TODO support DB without BF */
    ucs_assert(wqe_size <= UCT_IB_MLX5_BF_REG_SIZE);
    ucs_assert(num_bb <= UCT_IB_MLX5_MAX_BB);
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
    ucs_assert(ctrl == wq->curr);
    wq->curr       = src;
    wq->prev_sw_pi = wq->sw_pi;
    wq->sw_pi      = sw_pi;

    /* Flip BF register */
    wq->bf->reg.addr ^= UCT_IB_MLX5_BF_REG_SIZE;
    return num_bb;
}

#endif
