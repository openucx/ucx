/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2016. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ib_mlx5.h"

#include <ucs/sys/ptr_arith.h>

typedef struct uct_ib_mlx5_wqe_ctrl_seg {
    __be32      opmod_idx_opcode;
    __be32      qpn_ds;
    uint8_t     signature;
    __be16      dci_stream_channel_id;
    uint8_t     fm_ce_se;
    __be32      imm;
} UCS_S_PACKED uct_ib_mlx5_wqe_ctrl_seg_t;

static UCS_F_ALWAYS_INLINE UCS_F_NON_NULL struct mlx5_cqe64*
uct_ib_mlx5_get_cqe(uct_ib_mlx5_cq_t *cq,  unsigned cqe_index)
{
    return UCS_PTR_BYTE_OFFSET(cq->cq_buf,
                               (cqe_index & cq->cq_length_mask) <<
                                cq->cqe_size_log);
}


static UCS_F_ALWAYS_INLINE int
uct_ib_mlx5_cqe_is_hw_owned(uct_ib_mlx5_cq_t *cq, struct mlx5_cqe64 *cqe,
                            unsigned cqe_index, int poll_flags)
{
    uint8_t sw_it_count = cqe_index >> cq->cq_length_log;
    uint8_t hw_it_count;

    if (poll_flags & UCT_IB_MLX5_POLL_FLAG_CQE_ZIP) {
        hw_it_count = ((uint8_t *)cqe)[cq->own_field_offset];
        return (sw_it_count ^ hw_it_count) & cq->own_mask;
    } else {
        return (sw_it_count ^ cqe->op_own) & MLX5_CQE_OWNER_MASK;
    }
}


/**
 * Checks that cqe_format is equal to 3 (cqe is a part of compression block)
 * or opcode contains information about error.
 */
static UCS_F_ALWAYS_INLINE int
uct_ib_mlx5_cqe_is_error_or_zipped(uint8_t op_own)
{
    static const uint8_t mask = UCT_IB_MLX5_CQE_FORMAT_MASK |
                                UCT_IB_MLX5_CQE_OP_OWN_ERR_MASK;
    return (op_own & mask) >= UCT_IB_MLX5_CQE_FORMAT_MASK;
}

static UCS_F_ALWAYS_INLINE int
uct_ib_mlx5_cqe_stride_index(struct mlx5_cqe64* cqe)
{
#ifdef HAVE_STRUCT_MLX5_CQE64_IB_STRIDE_INDEX
    return ntohs(cqe->ib_stride_index);
#else
    uint16_t *stride = (uint16_t*)&cqe->rsvd20[2];
    return ntohs(*stride);
#endif
}

static UCS_F_ALWAYS_INLINE int uct_ib_mlx5_srq_stride(int num_sge)
{
    int stride;

    stride = sizeof(struct mlx5_wqe_srq_next_seg) +
             (num_sge * sizeof(struct mlx5_wqe_data_seg));

    return ucs_roundup_pow2(stride);
}

static UCS_F_ALWAYS_INLINE int
uct_ib_mlx5_srq_max_wrs(int rxq_len, int num_sge)
{
    return ucs_max(rxq_len / num_sge, UCT_IB_MLX5_XRQ_MIN_UWQ_POST);
}

static UCS_F_ALWAYS_INLINE int
uct_ib_mlx5_cqe_is_grh_present(struct mlx5_cqe64* cqe)
{
    return cqe->flags_rqpn & htonl(UCT_IB_MLX5_CQE_FLAG_L3_IN_DATA |
                                   UCT_IB_MLX5_CQE_FLAG_L3_IN_CQE);
}

static UCS_F_ALWAYS_INLINE void*
uct_ib_mlx5_gid_from_cqe(struct mlx5_cqe64* cqe)
{
    ucs_assert(uct_ib_mlx5_cqe_is_grh_present(cqe) ==
               htonl(UCT_IB_MLX5_CQE_FLAG_L3_IN_CQE)); /* GRH is in CQE */
    return UCS_PTR_BYTE_OFFSET(cqe, -UCT_IB_GRH_LEN);
}


static UCS_F_ALWAYS_INLINE void
uct_ib_mlx5_update_db_cq_ci(uct_ib_mlx5_cq_t *cq)
{
#if UCS_ENABLE_ASSERT
    cq->dbrec[UCT_IB_MLX5_CQ_SET_CI] = htobe32(cq->cq_ci & 0xffffff);
#endif
}


static UCS_F_ALWAYS_INLINE int
uct_ib_mlx5_check_and_init_zipped(uct_ib_mlx5_cq_t *cq, struct mlx5_cqe64 *cqe)
{
    if (cq->cq_unzip.current_idx > 0) {
        return 1;
    } else if ((cqe->op_own & UCT_IB_MLX5_CQE_FORMAT_MASK) == UCT_IB_MLX5_CQE_FORMAT_MASK) {
        ucs_assert(cq->cq_ci > 0);
        /* First zipped CQE in the sequence */
        uct_ib_mlx5_iface_cqe_unzip_init(cq);
        return 1;
    }

    cq->cq_unzip.title_cqe_valid = 0;
    return 0;
}


typedef struct mlx5_cqe64 *
(uct_ib_mlx5_check_compl_cb_t)(uct_ib_iface_t *iface, uct_ib_mlx5_cq_t *cq,
                               struct mlx5_cqe64 *cqe, int poll_flags);


static UCS_F_ALWAYS_INLINE struct mlx5_cqe64 *
uct_ib_mlx5_poll_cq(uct_ib_iface_t *iface, uct_ib_mlx5_cq_t *cq, int poll_flags,
                    uct_ib_mlx5_check_compl_cb_t check_cqe_cb)
{
    struct mlx5_cqe64 *cqe;
    unsigned idx;

    idx = cq->cq_ci;
    cqe = uct_ib_mlx5_get_cqe(cq, idx);

    if (ucs_unlikely(uct_ib_mlx5_cqe_is_hw_owned(cq, cqe, idx, poll_flags))) {
        return NULL;
    }

    ucs_memory_cpu_load_fence();

    if (ucs_unlikely(uct_ib_mlx5_cqe_is_error_or_zipped(cqe->op_own))) {
        return check_cqe_cb(iface, cq, cqe, poll_flags);
    }

    if (poll_flags & UCT_IB_MLX5_POLL_FLAG_CQE_ZIP) {
        /* Next zipped CQE should update title CQE */
        cq->cq_unzip.title_cqe_valid = 0;
    }

    cq->cq_ci = idx + 1;
    return cqe; /* TODO optimize - let compiler know cqe is not null */
}


static UCS_F_ALWAYS_INLINE uint16_t
uct_ib_mlx5_txwq_update_bb(uct_ib_mlx5_txwq_t *wq, uint16_t hw_ci)
{
#if UCS_ENABLE_ASSERT
    wq->hw_ci = hw_ci;
#endif
    return wq->bb_max - (wq->prev_sw_pi - hw_ci);
}


static UCS_F_ALWAYS_INLINE void
uct_ib_mlx5_txwq_update_flags(uct_ib_mlx5_txwq_t *txwq, uint32_t flags_add,
                              uint32_t flags_remove)
{
#if UCS_ENABLE_ASSERT
    txwq->flags = (txwq->flags | flags_add) & ~flags_remove;
#endif
}


/**
 * Check the work queue is in a consistent state, and that it has enough space
 * for the new work request.
 *
 * @param wq             Work queue to validate.
 * @param num_bb         How much we are posting now.
 * @param hw_ci_updated  Whether wq->hw_ci field kept up-to-date on this
 *                       workqueue.
 */
static UCS_F_ALWAYS_INLINE void
uct_ib_mlx5_txwq_validate(uct_ib_mlx5_txwq_t *wq, uint16_t num_bb,
                          int hw_ci_updated)
{
#if UCS_ENABLE_ASSERT
    uct_ib_mlx5_txwq_validate_always(wq, num_bb, hw_ci_updated);
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

    ucs_assert(dest != NULL);
    ucs_assert((src != NULL) || (length == 0));

    if (UCS_PTR_BYTE_OFFSET(dest, length) <= wq->qend) {
        /* cppcheck-suppress nullPointer */
        memcpy(dest, src, length);
    } else {
        n = UCS_PTR_BYTE_DIFF(dest, wq->qend);
        memcpy(dest, src, n);
        memcpy(wq->qstart, UCS_PTR_BYTE_OFFSET(src, n), length - n);
    }
}


/**
 * Copy uct_iov_t array to inline segment, taking into account QP wrap-around.
 *
 * @param dest     Inline data in the WQE to copy to.
 * @param iov      A pointer to an array of uct_iov_t elements.
 * @param iov_cnt  A number of elements in iov array.
 * @param length   A total size of data in iov array.
 * @param wq       Send work-queue.
 */
static UCS_F_ALWAYS_INLINE void
uct_ib_mlx5_inline_iov_copy(void *restrict dest, const uct_iov_t *iov,
                            size_t iovcnt, size_t length,
                            uct_ib_mlx5_txwq_t *wq)
{
    ptrdiff_t remainder_val;
    ucs_iov_iter_t iov_iter;

    ucs_assert(dest != NULL);

    ucs_iov_iter_init(&iov_iter);
    remainder_val = UCS_PTR_BYTE_DIFF(dest, wq->qend);
    if (ucs_likely(length <= remainder_val)) {
        uct_iov_to_buffer(iov, iovcnt, &iov_iter, dest, SIZE_MAX);
    } else {
        uct_iov_to_buffer(iov, iovcnt, &iov_iter, dest, remainder_val);
        uct_iov_to_buffer(iov, iovcnt, &iov_iter, wq->qstart, SIZE_MAX);
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
        seg = UCS_PTR_BYTE_OFFSET(seg, -UCS_PTR_BYTE_DIFF(txwq->qstart,
                                                          txwq->qend));
    }
    return uct_ib_mlx5_txwq_wrap_none(txwq, seg);
}


/* Wrapping of 'data' could happen, even past 'qend' boundary.
 * Do not check for alignment. */
static UCS_F_ALWAYS_INLINE void *
uct_ib_mlx5_txwq_wrap_data(uct_ib_mlx5_txwq_t *txwq, void *data)
{
    if (ucs_unlikely(data >= txwq->qend)) {
        data = UCS_PTR_BYTE_OFFSET(data, -UCS_PTR_BYTE_DIFF(txwq->qstart,
                                                            txwq->qend));
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


static UCS_F_ALWAYS_INLINE size_t
uct_ib_mlx5_set_dgram_seg_grh(struct mlx5_wqe_datagram_seg *seg,
                              struct mlx5_grh_av *grh_av)
{
    struct mlx5_base_av *to_av    = mlx5_av_base(&seg->av);
    struct mlx5_grh_av *to_grh_av = mlx5_av_grh(&seg->av);

    if (grh_av != NULL) {
        ucs_assert(to_av->dqp_dct & UCT_IB_MLX5_EXTENDED_UD_AV);
#if HAVE_STRUCT_MLX5_GRH_AV_RMAC
        memcpy(to_grh_av->rmac, grh_av->rmac, sizeof(to_grh_av->rmac));
#endif
        to_grh_av->tclass     = grh_av->tclass;
        to_grh_av->hop_limit  = grh_av->hop_limit;
        to_grh_av->grh_gid_fl = grh_av->grh_gid_fl;
        memcpy(to_grh_av->rgid, grh_av->rgid, sizeof(to_grh_av->rgid));

        return UCT_IB_MLX5_AV_FULL_SIZE;
    }
    
    if (to_av->dqp_dct & UCT_IB_MLX5_EXTENDED_UD_AV) {
        to_grh_av->grh_gid_fl = 0;

        return UCT_IB_MLX5_AV_FULL_SIZE;
    }

    return UCT_IB_MLX5_AV_BASE_SIZE;
}

static UCS_F_ALWAYS_INLINE void
uct_ib_mlx5_set_ctrl_qpn_ds(struct mlx5_wqe_ctrl_seg *ctrl, uint32_t qp_num,
                            uint8_t ds)
{
    ctrl->qpn_ds = htonl((qp_num << 8) | ds);
}

static UCS_F_ALWAYS_INLINE void
uct_ib_mlx5_set_ctrl_seg(struct mlx5_wqe_ctrl_seg* ctrl, uint16_t pi,
                         uint8_t opcode, uint8_t opmod, uint32_t qp_num,
                         uint8_t fm_ce_se, uint16_t dci_channel,
                         unsigned wqe_size)
{
    uint8_t ds = ucs_div_round_up(wqe_size, UCT_IB_MLX5_WQE_SEG_SIZE);
#if defined(__ARM_NEON)
    uint8x16_t table = {1,               /* opmod */
                        5,  4,           /* sw_pi in BE */
                        2,               /* opcode */
                        14, 13, 12,      /* QP num */
                        8,               /* data size */
                        16,              /* signature (set 0) */
                        11, 10,          /* dci stream channel ID in BE */
                        0,               /* signal/fence_mode */
                        16, 16, 16, 16}; /* immediate (set to 0)*/
    uint32x4_t data = {(opcode << 16) | (opmod << 8) | fm_ce_se,
                       pi, (dci_channel << 16) | ds, qp_num};
#elif !defined (__SSE4_2__)
    uct_ib_mlx5_wqe_ctrl_seg_t *uct_ctrl = (uct_ib_mlx5_wqe_ctrl_seg_t*)ctrl;
#endif

    ucs_assert(((unsigned long)ctrl % UCT_IB_MLX5_WQE_SEG_SIZE) == 0);

#if defined(__SSE4_2__)
    *(__m128i *) ctrl = _mm_shuffle_epi8(
                    _mm_set_epi32(qp_num, (dci_channel << 16) | ds, pi,
                                  (ds << 24) | (opcode << 16) | (opmod << 8) | fm_ce_se), /* OR of constants */
                    _mm_set_epi8(0x80, 0x80, 0x80, 0x80, /* immediate (set 0) */
                                 0,                      /* signal/fence_mode */
                                 10, 11,                 /* dci stream channel ID in BE */
                                 0x80,                   /* signature (set 0) */
                                 8,                      /* data size */
                                 12, 13, 14,             /* QP num */
                                 2,                      /* opcode */
                                 4, 5,                   /* sw_pi in BE */
                                 1                       /* opmod */
                                 ));
#elif defined(__ARM_NEON)
    *(uint8x16_t *)ctrl = vqtbl1q_u8((uint8x16_t)data, table);
#else
    uct_ctrl->opmod_idx_opcode      = (opcode << 24) | (htons(pi) << 8) | opmod;
    uct_ib_mlx5_set_ctrl_qpn_ds(ctrl, qp_num, ds);
    uct_ctrl->dci_stream_channel_id = htons(dci_channel);
    uct_ctrl->fm_ce_se              = fm_ce_se;
#endif
}


static UCS_F_ALWAYS_INLINE void
uct_ib_mlx5_set_ctrl_seg_with_imm(struct mlx5_wqe_ctrl_seg* ctrl, uint16_t pi,
                                  uint8_t opcode, uint8_t opmod, uint32_t qp_num,
                                  uint8_t fm_ce_se, uint16_t dci_channel,
                                  unsigned wqe_size, uint32_t imm)
{
    uint8_t ds = ucs_div_round_up(wqe_size, UCT_IB_MLX5_WQE_SEG_SIZE);
#if defined(__ARM_NEON)
    uint8x16_t table = {1,             /* opmod */
                        5,  4,         /* sw_pi in BE */
                        2,             /* opcode */
                        14, 13, 12,    /* QP num */
                        3,             /* data size */
                        16,            /* signature (set 0) */
                        7, 6,          /* dci stream channel ID in BE */
                        0,             /* signal/fence_mode */
                        8, 9, 10, 11}; /* immediate */
    uint32x4_t data = {(ds << 24) | (opcode << 16) | (opmod << 8) | fm_ce_se,
                       (dci_channel << 16) | pi, imm, qp_num};
#elif !defined (__SSE4_2__)
    uct_ib_mlx5_wqe_ctrl_seg_t *uct_ctrl = (uct_ib_mlx5_wqe_ctrl_seg_t*)ctrl;
#endif

    ucs_assert(((unsigned long)ctrl % UCT_IB_MLX5_WQE_SEG_SIZE) == 0);

#if defined(__SSE4_2__)
    *(__m128i *) ctrl = _mm_shuffle_epi8(
                    _mm_set_epi32(qp_num, imm, (dci_channel << 16) | pi,
                                  (ds << 24) | (opcode << 16) | (opmod << 8) | fm_ce_se), /* OR of constants */
                    _mm_set_epi8(11, 10, 9, 8, /* immediate */
                                 0,            /* signal/fence_mode */
                                 6, 7,         /* dci stream channel id in BE */
                                 0x80,         /* signature (set 0) */
                                 3,            /* data size */
                                 12, 13, 14,   /* QP num */
                                 2,            /* opcode */
                                 4, 5,         /* sw_pi in BE */
                                 1             /* opmod */
                                 ));
#elif defined(__ARM_NEON)
    *(uint8x16_t *)ctrl = vqtbl1q_u8((uint8x16_t)data, table);
#else
    uct_ctrl->opmod_idx_opcode      = (opcode << 24) | (htons(pi) << 8) | opmod;
    uct_ib_mlx5_set_ctrl_qpn_ds(ctrl, qp_num, ds);
    uct_ctrl->dci_stream_channel_id = htons(dci_channel);
    uct_ctrl->fm_ce_se              = fm_ce_se;
    uct_ctrl->imm                   = imm;
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
size_t uct_ib_mlx5_set_data_seg_iov(uct_ib_mlx5_txwq_t *txwq,
                                    struct mlx5_wqe_data_seg *dptr,
                                    const uct_iov_t *iov, size_t iovcnt)
{
    size_t wqe_size = 0;
    size_t iov_it;

    for (iov_it = 0; iov_it < iovcnt; ++iov_it) {
        if (!iov[iov_it].length) { /* Skip zero length WQE*/
            continue;
        }
        ucs_assert(iov[iov_it].memh != UCT_MEM_HANDLE_NULL);

        /* place data into the buffer */
        dptr = uct_ib_mlx5_txwq_wrap_any(txwq, dptr);
        uct_ib_mlx5_set_data_seg(dptr, iov[iov_it].buffer,
                                 uct_iov_get_length(iov + iov_it),
                                 uct_ib_memh_get_lkey(iov[iov_it].memh));
        wqe_size += sizeof(*dptr);
        ++dptr;
    }

    return wqe_size;
}


static UCS_F_ALWAYS_INLINE void uct_ib_mlx5_bf_copy_bb(void * restrict dst,
                                                       void * restrict src)
{
#if defined(__ARM_NEON)
    vst4q_u64(dst, vld4q_u64(src));
#else
#if defined(__SSE4_2__)
    typedef __m128i uct_ib_mlx5_send_wqe_bb_block_t;
#else
    typedef uint8_t uct_ib_mlx5_send_wqe_bb_block_t;
#endif
    /* Prevent compiler to replace by memmove() */
    typedef struct {
        uct_ib_mlx5_send_wqe_bb_block_t
                data[MLX5_SEND_WQE_BB / sizeof(uct_ib_mlx5_send_wqe_bb_block_t)];
    } UCS_S_PACKED uct_ib_mlx5_send_wqe_bb_t;

    UCS_WORD_COPY(uct_ib_mlx5_send_wqe_bb_t, dst,
                  uct_ib_mlx5_send_wqe_bb_t, src,
                  MLX5_SEND_WQE_BB);
#endif
}

static UCS_F_ALWAYS_INLINE
void *uct_ib_mlx5_bf_copy(void *dst, void *src, uint16_t num_bb,
                          const uct_ib_mlx5_txwq_t *wq)
{
    uint16_t n;

    for (n = 0; n < num_bb; ++n) {
        uct_ib_mlx5_bf_copy_bb(dst, src);
        dst = UCS_PTR_BYTE_OFFSET(dst, MLX5_SEND_WQE_BB);
        src = UCS_PTR_BYTE_OFFSET(src, MLX5_SEND_WQE_BB);
        if (ucs_unlikely(src == wq->qend)) {
            src = wq->qstart;
        }
    }
    return src;
}

static UCS_F_ALWAYS_INLINE uint16_t
uct_ib_mlx5_post_send(uct_ib_mlx5_txwq_t *wq, struct mlx5_wqe_ctrl_seg *ctrl,
                      unsigned wqe_size, int hw_ci_updated)
{
    uint16_t sw_pi, num_bb, res_count;
    void *src, *dst;

    ucs_assert(((unsigned long)ctrl % UCT_IB_MLX5_WQE_SEG_SIZE) == 0);
    num_bb  = ucs_div_round_up(wqe_size, MLX5_SEND_WQE_BB);
    sw_pi   = wq->sw_pi;

    uct_ib_mlx5_txwq_validate(wq, num_bb, hw_ci_updated);

    /* TODO Put memory store fence here too, to prevent WC being flushed after DBrec */
    ucs_memory_cpu_store_fence();

    /* Write doorbell record */
    *wq->dbrec = htonl(sw_pi += num_bb);

    /* Make sure that doorbell record is written before ringing the doorbell */
    ucs_memory_bus_store_fence();

    /* Set up copy pointers */
    dst = wq->reg->addr.ptr;
    src = ctrl;

    ucs_assert(wqe_size <= UCT_IB_MLX5_BF_REG_SIZE);
    ucs_assert(num_bb <= UCT_IB_MLX5_MAX_BB);
    if (ucs_likely(wq->reg->mode == UCT_IB_MLX5_MMIO_MODE_BF_POST)) {
        src = uct_ib_mlx5_bf_copy(dst, src, num_bb, wq);
        ucs_memory_bus_cacheline_wc_flush();
    } else if (wq->reg->mode == UCT_IB_MLX5_MMIO_MODE_BF_POST_MT) {
        src = uct_ib_mlx5_bf_copy(dst, src, num_bb, wq);
        /* Make sure that HW observes WC writes in order, in case of multiple
         * threads which use the same BF register in a serialized way
         */
        ucs_memory_cpu_wc_fence();
    } else {
        if (wq->reg->mode == UCT_IB_MLX5_MMIO_MODE_DB) {
            *(volatile uint64_t*)dst = *(volatile uint64_t*)src;
        } else {
            ucs_assert(wq->reg->mode == UCT_IB_MLX5_MMIO_MODE_DB_LOCK);
            ucs_spin_lock(&wq->reg->db_lock);
            *(volatile uint64_t*)dst = *(volatile uint64_t*)src;
            ucs_spin_unlock(&wq->reg->db_lock);
        }

        ucs_memory_bus_store_fence();
        src = UCS_PTR_BYTE_OFFSET(src, num_bb * MLX5_SEND_WQE_BB);
        src = uct_ib_mlx5_txwq_wrap_any(wq, src);
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
    wq->reg->addr.uint ^= UCT_IB_MLX5_BF_REG_SIZE;
    return res_count;
}


static inline uct_ib_mlx5_srq_seg_t *
uct_ib_mlx5_srq_get_wqe(uct_ib_mlx5_srq_t *srq, uint16_t wqe_index)
{
    return UCS_PTR_BYTE_OFFSET(srq->buf, (wqe_index & srq->mask) * srq->stride);
}

static UCS_F_MAYBE_UNUSED void
uct_ib_mlx5_iface_fill_attr(uct_ib_iface_t *iface,
                            uct_ib_mlx5_qp_t *qp,
                            uct_ib_mlx5_qp_attr_t *attr)
{
#if HAVE_DECL_IBV_CREATE_QP_EX
    attr->super.ibv.comp_mask       = IBV_QP_INIT_ATTR_PD;
    if (qp->verbs.rd->pd != NULL) {
        attr->super.ibv.pd          = qp->verbs.rd->pd;
    } else {
        attr->super.ibv.pd          = uct_ib_iface_md(iface)->pd;
    }
#endif
}


static void UCS_F_ALWAYS_INLINE
uct_ib_mlx5_update_cqe_zipping_stats(uct_ib_iface_t *iface,
                                     uct_ib_mlx5_cq_t *cq)
{
    if ((cq->cq_unzip.title.op_own >> 4) == MLX5_CQE_REQ) {
        UCS_STATS_UPDATE_COUNTER(iface->stats,
                                 UCT_IB_IFACE_STAT_TX_COMPLETION_ZIPPED, 1);
    } else {
        UCS_STATS_UPDATE_COUNTER(iface->stats,
                                 UCT_IB_IFACE_STAT_RX_COMPLETION_ZIPPED, 1);
    }
}


static int UCS_F_ALWAYS_INLINE
uct_ib_mlx5_get_atomic_mode(uct_ib_iface_t *iface)
{
    return uct_ib_iface_device(iface)->ext_atomic_arg_sizes ?
        UCT_IB_MLX5_ATOMIC_MODE_EXT : UCT_IB_MLX5_ATOMIC_MODE_COMP;
}
