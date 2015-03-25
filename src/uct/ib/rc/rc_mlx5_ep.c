/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_mlx5.h"

#include <uct/ib/mlx5/ib_mlx5_log.h>
#include <ucs/arch/arch.h>
#include <ucs/sys/compiler.h>
#include <arpa/inet.h> /* For htonl */


#define UCT_RC_MLX5_OPCODE_FLAG_RAW   0x100
#define UCT_RC_MLX5_OPCODE_MASK       0xff


static inline void uct_rc_mlx5_bf_copy_bb(void *dst, void *src)
{
#ifdef __SSE4_2__
    UCS_WORD_COPY(dst, src, __m128i, MLX5_SEND_WQE_BB);
#else
    UCS_WORD_COPY(dst, src, uint64_t, MLX5_SEND_WQE_BB);
#endif
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_ep_set_rdma_seg(struct mlx5_wqe_raddr_seg *raddr, uint64_t rdma_raddr,
                            uct_rkey_t rdma_rkey)
{
#ifdef __SSE4_2__
    *(__m128i*)raddr = _mm_shuffle_epi8(
                _mm_set_epi64x(rdma_rkey, rdma_raddr),
                _mm_set_epi8(0, 0, 0, 0,            /* reserved */
                             8, 9, 10, 11,          /* rkey */
                             0, 1, 2, 3, 4, 5, 6, 7 /* rdma_raddr */
                             ));
#else
    raddr->raddr = htonll(rdma_raddr);
    raddr->rkey  = htonl(rdma_rkey);
#endif
}

static UCS_F_ALWAYS_INLINE
void uct_rc_mlx5_ep_set_dptr_seg(struct mlx5_wqe_data_seg *dptr, void *address,
                                 unsigned length, uint32_t lkey)
{
    dptr->byte_count = htonl(length);
    dptr->lkey       = htonl(lkey);
    dptr->addr       = htonll((uintptr_t)address);
}

void uct_rc_mlx5_set_ctrl_seg(struct mlx5_wqe_ctrl_seg* ctrl, uint16_t pi,
                              uint8_t opcode, uint8_t opmod, uint32_t qp_num,
                              uint8_t fm_ce_se, uint8_t ds)
{
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
uct_rc_mlx5_post_send(uct_rc_mlx5_ep_t *ep, struct mlx5_wqe_ctrl_seg *ctrl,
                      uint8_t opcode, uint8_t opmod, unsigned sig_flag, unsigned wqe_size)
{
    unsigned n, num_seg, num_bb;
    void *src, *dst;
    uint16_t sw_pi;

    num_seg = ucs_div_round_up(wqe_size, UCT_IB_MLX5_WQE_SEG_SIZE);
    num_bb  = ucs_div_round_up(wqe_size, MLX5_SEND_WQE_BB);
    sw_pi   = ep->tx.sw_pi;

    uct_rc_mlx5_set_ctrl_seg(ctrl, sw_pi, opcode, opmod, ep->qp_num, sig_flag,
                             num_seg);
    uct_ib_mlx5_log_tx(IBV_QPT_RC, ctrl, ep->tx.qstart, ep->tx.qend,
                       (opcode == MLX5_OPCODE_SEND) ? uct_rc_ep_am_packet_dump : NULL);

    /* TODO Put memory store fence here too, to prevent WC being flushed after DBrec */
    ucs_memory_cpu_store_fence();

    /* Write doorbell record */
    ep->tx.prev_sw_pi = sw_pi;
    *ep->tx.dbrec = htonl(sw_pi += num_bb);

    /* Make sure that doorbell record is written before ringing the doorbell */
    ucs_memory_bus_store_fence();

    /* Set up copy pointers */
    dst = ep->tx.bf_reg;
    src = ctrl;

    /* BF copy */
    /* TODO support DB without BF */
    ucs_assert(wqe_size <= ep->tx.bf_size);
    ucs_assert(num_bb <= UCT_RC_MLX5_MAX_BB);
    for (n = 0; n < num_bb; ++n) {
        uct_rc_mlx5_bf_copy_bb(dst, src);
        dst += MLX5_SEND_WQE_BB;
        src += MLX5_SEND_WQE_BB;
        if (ucs_unlikely(src == ep->tx.qend)) {
            src = ep->tx.qstart;
        }
    }

    /* We don't want the compiler to reorder instructions and hurt latency */
    ucs_compiler_fence();

    /* Advance queue pointer */
    ucs_assert(ctrl == ep->tx.seg);
    ep->tx.seg   = src;
    ep->tx.sw_pi = sw_pi;

    /* Flip BF register */
    ep->tx.bf_reg = (void*) ((uintptr_t) ep->tx.bf_reg ^ ep->tx.bf_size);

    uct_rc_ep_tx_posted(&ep->super, sig_flag & MLX5_WQE_CTRL_CQ_UPDATE);
}

static UCS_F_ALWAYS_INLINE uint8_t
uct_rc_mlx5_iface_tx_moderation(uct_rc_mlx5_ep_t* ep, uct_rc_mlx5_iface_t* iface)
{
    return (ep->super.unsignaled >= iface->super.config.tx_moderation) ?
           MLX5_WQE_CTRL_CQ_UPDATE : 0;
}

/**
 * Copy data to inline segment, taking into account QP wrap-around.
 *
 * @param dest    Inline data in the WQE to copy to.
 * @patam src     Data to copy.
 * @param length  Data length.
 *
 * @return If there was a wrap-around, return -qp_size. Otherwise, return 0.
 */
static UCS_F_ALWAYS_INLINE ptrdiff_t
uct_rc_mlx5_inline_copy(void *dest, void *src, unsigned length, uct_rc_mlx5_ep_t *ep)
{
    void *qend = ep->tx.qend;
    ptrdiff_t n;

    if (dest + length <= qend) {
        memcpy(dest, src, length);
        return 0;
    } else {
        n = qend - dest;
        memcpy(dest, src, n);
        memcpy(ep->tx.qstart, src + n, length - n);
        return ep->tx.qstart - qend;
    }
}

/*
 * Generic inline posting function.
 * Parameters which are not relevant to the opcode are ignored.
 *
 *            +--------+-----+-------+--------+------------
 * SEND       | CTRL   | INL | am_id | am_hdr | payload ...
 *            +--------+-----+---+---+-+-------+-----------
 * RDMA_WRITE | CTRL   | RADDR   | INL | payload ...
 *            +--------+---------+-----+-------------------
 *
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_ep_inline_post(uct_rc_mlx5_ep_t *ep, unsigned opcode,
                           void *buffer, unsigned length,
                           /* SEND */ uint8_t am_id, uint64_t am_hdr,
                           /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey
                           )
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                uct_rc_mlx5_iface_t);

    struct mlx5_wqe_ctrl_seg     *ctrl;
    struct mlx5_wqe_raddr_seg    *raddr;
    struct mlx5_wqe_inl_data_seg *inl;
    uct_rc_am_short_hdr_t        *am;
    unsigned wqe_size;
    unsigned sig_flag;

    ctrl = ep->tx.seg;
    UCT_RC_MLX5_CHECK_RES(iface, ep);

    switch (opcode) {
    case MLX5_OPCODE_SEND:
        /* Set inline segment which has AM id, AM header, and AM payload */
        wqe_size         = sizeof(*ctrl) + sizeof(*inl) + sizeof(*am) + length;
        UCT_CHECK_LENGTH(wqe_size <= UCT_RC_MLX5_MAX_BB * MLX5_SEND_WQE_BB,
                         "am_short");
        inl              = (void*)(ctrl + 1);
        inl->byte_count  = htonl((length + sizeof(*am)) | MLX5_INLINE_SEG);
        am               = (void*)(inl + 1);
        am->rc_hdr.am_id = am_id;
        am->am_hdr       = am_hdr;
        uct_rc_mlx5_inline_copy(am + 1, buffer, length, ep);
        sig_flag         = uct_rc_iface_tx_moderation(&iface->super, &ep->super,
                                                      MLX5_WQE_CTRL_CQ_UPDATE);
        break;

    case MLX5_OPCODE_RDMA_WRITE:
        /* Set RDMA segment */
        if (length == 0) {
            wqe_size     = sizeof(*ctrl) + sizeof(*raddr);
        } else {
            wqe_size     = sizeof(*ctrl) + sizeof(*raddr) + sizeof(*inl) + length;
        }
        UCT_CHECK_LENGTH(wqe_size <= UCT_RC_MLX5_MAX_BB * MLX5_SEND_WQE_BB,
                        "put_short");
        raddr            = (void*)(ctrl + 1);
        uct_rc_mlx5_ep_set_rdma_seg(raddr, rdma_raddr, rdma_rkey);
        inl              = (void*)(raddr + 1);
        inl->byte_count  = htonl(length | MLX5_INLINE_SEG);
        uct_rc_mlx5_inline_copy(inl + 1, buffer, length, ep);
        sig_flag         = MLX5_WQE_CTRL_CQ_UPDATE;
        break;

    case MLX5_OPCODE_NOP:
        /* Empty inline segment */
        wqe_size         = sizeof(*ctrl);
        inl              = (void*)(ctrl + 1);
        inl->byte_count  = htonl(MLX5_INLINE_SEG);
        sig_flag         = MLX5_WQE_CTRL_CQ_UPDATE | MLX5_WQE_CTRL_FENCE;
        break;

    default:
        return UCS_ERR_INVALID_PARAM;
    }

    uct_rc_mlx5_post_send(ep, ctrl, opcode, 0, sig_flag, wqe_size);
    return UCS_OK;
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
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_ep_dptr_post(uct_rc_mlx5_ep_t *ep, unsigned opcode_flags,
                         void *buffer, unsigned length, uint32_t *lkey_p,
                         /* SEND */ uint8_t am_id, void *am_hdr, unsigned am_hdr_len,
                         /* RDMA/ATOMIC */ uint64_t remote_addr, uct_rkey_t rkey,
                         /* ATOMIC */ uint64_t compare_mask, uint64_t compare, uint64_t swap_add,
                         int signal)
{
    struct mlx5_wqe_ctrl_seg                     *ctrl;
    struct mlx5_wqe_raddr_seg                    *raddr;
    struct mlx5_wqe_atomic_seg                   *atomic;
    struct mlx5_wqe_data_seg                     *dptr;
    struct mlx5_wqe_inl_data_seg                 *inl;
    struct uct_ib_mlx5_atomic_masked_cswap32_seg *masked_cswap32;
    struct uct_ib_mlx5_atomic_masked_fadd32_seg  *masked_fadd32;
    struct uct_ib_mlx5_atomic_masked_cswap64_seg *masked_cswap64;

    uct_rc_mlx5_iface_t *iface;
    uct_rc_hdr_t        *rch;
    unsigned            wqe_size, inl_seg_size;
    ptrdiff_t           wraparound;
    uint8_t             opmod;

    iface = ucs_derived_of(ep->super.super.super.iface, uct_rc_mlx5_iface_t);
    if (!signal) {
        signal = uct_rc_iface_tx_moderation(&iface->super, &ep->super,
                                            MLX5_WQE_CTRL_CQ_UPDATE);
    } else {
        ucs_assert(signal == MLX5_WQE_CTRL_CQ_UPDATE);
    }

    opmod = 0;
    ctrl = ep->tx.seg;
    switch (opcode_flags) {
    case MLX5_OPCODE_SEND:
        UCT_CHECK_LENGTH(length + sizeof(*rch) + am_hdr_len <=
                         iface->super.super.config.seg_size, "send");

        inl_seg_size     = ucs_align_up_pow2(sizeof(*inl) + sizeof(*rch) + am_hdr_len,
                                             UCT_IB_MLX5_WQE_SEG_SIZE);

        /* Inline segment with AM ID and header */
        inl              = (void*)(ctrl + 1);
        inl->byte_count  = htonl((sizeof(*rch) + am_hdr_len) | MLX5_INLINE_SEG);
        rch              = (void*)(inl + 1);
        rch->am_id       = am_id;

        wraparound = uct_rc_mlx5_inline_copy(rch + 1, am_hdr, am_hdr_len, ep);

        /* Data segment with payload */
        if (length == 0) {
            wqe_size     = sizeof(*ctrl) + inl_seg_size;
        } else {
            wqe_size     = sizeof(*ctrl) + inl_seg_size + sizeof(*dptr);
            uct_rc_mlx5_ep_set_dptr_seg((void*)(ctrl + 1) + inl_seg_size + wraparound,
                                        buffer, length, *lkey_p);
        }
        break;

    case MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_RAW:
        /* Data segment only */
        UCT_CHECK_LENGTH((length <= iface->super.super.config.seg_size) && (length > 0),
                         "send");
        ucs_assert(length < (2ul << 30));

        wqe_size         = sizeof(*ctrl) + sizeof(*dptr);
        uct_rc_mlx5_ep_set_dptr_seg((void*)(ctrl + 1), buffer, length, *lkey_p);
        break;

    case MLX5_OPCODE_RDMA_READ:
    case MLX5_OPCODE_RDMA_WRITE:
        /* Set RDMA segment */
        UCT_CHECK_LENGTH(length <= UCT_IB_MAX_MESSAGE_SIZE, "put/get");

        raddr            = (void*)(ctrl + 1);
        uct_rc_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        /* Data segment */
        if (length == 0) {
            wqe_size     = sizeof(*ctrl) + sizeof(*raddr);
        } else {
            wqe_size     = sizeof(*ctrl) + sizeof(*raddr) + sizeof(*dptr);
            uct_rc_mlx5_ep_set_dptr_seg((void*)(raddr + 1), buffer, length, *lkey_p);
        }
        break;

    case MLX5_OPCODE_ATOMIC_FA:
    case MLX5_OPCODE_ATOMIC_CS:
        ucs_assert(length == sizeof(uint64_t));
        raddr = (void*)(ctrl + 1);
        uct_rc_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        atomic            = (void*)(raddr + 1);
        if (opcode_flags == MLX5_OPCODE_ATOMIC_CS) {
            atomic->compare = compare;
        }
        atomic->swap_add  = swap_add;

        uct_rc_mlx5_ep_set_dptr_seg((void*)(atomic + 1), buffer, length, *lkey_p);
        wqe_size          = sizeof(*ctrl) + sizeof(*raddr) + sizeof(*atomic) +
                            sizeof(*dptr);
        break;

    case MLX5_OPCODE_ATOMIC_MASKED_CS:
        raddr = (void*)(ctrl + 1);
        uct_rc_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        switch (length) {
        case sizeof(uint32_t):
            opmod                        = UCT_IB_MLX5_OPMOD_EXT_ATOMIC(2);
            masked_cswap32 = (void*)(raddr + 1);
            masked_cswap32->swap         = swap_add;
            masked_cswap32->compare      = compare;
            masked_cswap32->swap_mask    = (uint32_t)-1;
            masked_cswap32->compare_mask = compare_mask;
            dptr                         = (void*)(masked_cswap32 + 1);
            wqe_size                     = sizeof(*ctrl) + sizeof(*raddr) +
                                           sizeof(*masked_cswap32) + sizeof(*dptr);
            break;
        case sizeof(uint64_t):
            opmod                        = UCT_IB_MLX5_OPMOD_EXT_ATOMIC(3); /* Ext. atomic, size 2**3 */
            masked_cswap64 = (void*)(raddr + 1);
            masked_cswap64->swap         = swap_add;
            masked_cswap64->compare      = compare;
            masked_cswap64->swap_mask    = (uint64_t)-1;
            masked_cswap64->compare_mask = compare_mask;
            dptr                         = (void*)(masked_cswap64 + 1);
            wqe_size                     = sizeof(*ctrl) + sizeof(*raddr) +
                                           sizeof(*masked_cswap64) + sizeof(*dptr);

            /* Handle QP wrap-around. It cannot happen in the middle of
             * masked-cswap segment, because it's still in the first BB.
             */
            ucs_assert((void*)dptr <= ep->tx.qend);
            if (dptr == ep->tx.qend) {
                dptr = ep->tx.qstart;
            } else {
                ucs_assert((void*)masked_cswap64 < ep->tx.qend);
            }
            break;
        default:
            ucs_assert(0);
        }

        uct_rc_mlx5_ep_set_dptr_seg(dptr, buffer, length, *lkey_p);
        break;

     case MLX5_OPCODE_ATOMIC_MASKED_FA:
        ucs_assert(length == sizeof(uint32_t));
        raddr = (void*)(ctrl + 1);
        uct_rc_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        opmod                         = UCT_IB_MLX5_OPMOD_EXT_ATOMIC(2);
        masked_fadd32                 = (void*)(raddr + 1);
        masked_fadd32->add            = swap_add;
        masked_fadd32->filed_boundary = 0;

        uct_rc_mlx5_ep_set_dptr_seg((void*)(masked_fadd32 + 1), buffer, length,
                                    *lkey_p);
        wqe_size                      = sizeof(*ctrl) + sizeof(*raddr) +
                                        sizeof(*masked_fadd32) + sizeof(*dptr);
        break;

    default:
        return UCS_ERR_INVALID_PARAM;
    }

    uct_rc_mlx5_post_send(ep, ctrl, (opcode_flags & UCT_RC_MLX5_OPCODE_MASK),
                          opmod, signal, wqe_size);
    return UCS_OK;
}

/*
 *
 * Helper function for buffer-copy post.
 * Adds the descriptor to the callback queue.
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_ep_bcopy_post(uct_rc_mlx5_ep_t *ep, unsigned opcode, unsigned length,
                          /* SEND */ uint8_t am_id, void *am_hdr, unsigned am_hdr_len,
                          /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey,
                          int force_sig, uct_rc_iface_send_desc_t *desc,
                          ucs_status_t success)
{
    ucs_status_t status;

    desc->super.sn = ep->tx.sw_pi;
    status = uct_rc_mlx5_ep_dptr_post(ep, opcode, desc + 1, length, &desc->lkey,
                                      am_id, am_hdr, am_hdr_len, rdma_raddr, rdma_rkey,
                                      0, 0, 0, force_sig);
    if (status != UCS_OK) {
        ucs_mpool_put(desc);
        return status;
    }

    ucs_queue_push(&ep->super.comp, &desc->super.queue);
    return success;
}

/*
 * Helper function for zero-copy post.
 * Adds user completion to the callback queue.
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_ep_zcopy_post(uct_rc_mlx5_ep_t *ep, unsigned opcode, void *buffer,
                          unsigned length, struct ibv_mr *mr,
                          /* SEND */ uint8_t am_id, void *am_hdr, unsigned am_hdr_len,
                          /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey,
                          int force_sig, uct_completion_t *comp)
{
    uct_rc_mlx5_iface_t *iface  = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_rc_mlx5_iface_t);
    ucs_status_t status;
    uint16_t sn;

    UCT_RC_MLX5_CHECK_RES(iface, ep);

    sn = ep->tx.sw_pi;
    status = uct_rc_mlx5_ep_dptr_post(ep, opcode, buffer, length, &mr->lkey,
                                      am_id, am_hdr, am_hdr_len, rdma_raddr, rdma_rkey,
                                      0, 0, 0,
                                      (comp == NULL) ? force_sig : MLX5_WQE_CTRL_CQ_UPDATE);
    if (status != UCS_OK) {
        return status;
    }

    uct_rc_ep_add_user_completion(&ep->super, comp, sn);
    return UCS_INPROGRESS;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_ep_atomic_post(uct_rc_mlx5_ep_t *ep, unsigned opcode,
                           uct_rc_iface_send_desc_t *desc, unsigned length,
                           uint64_t remote_addr, uct_rkey_t rkey,
                           uint64_t compare_mask, uint64_t compare,
                           uint64_t swap_add, int signal, ucs_status_t success)
{
    ucs_status_t status;

    desc->super.sn = ep->tx.sw_pi;
    status = uct_rc_mlx5_ep_dptr_post(ep, opcode, desc + 1, length, &desc->lkey,
                                      0, NULL, 0, remote_addr, rkey, compare_mask,
                                      compare, swap_add, signal);
    if (status != UCS_OK) {
        return status;
    }

    UCT_TL_EP_STAT_ATOMIC(&ep->super.super);
    ucs_queue_push(&ep->super.comp, &desc->super.queue);
    return success;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_ep_atomic(uct_rc_mlx5_ep_t *ep, int opcode, unsigned length,
                      uct_completion_callback_t proxy_cb, uint64_t remote_addr,
                      uct_rkey_t rkey, uint64_t compare_mask, uint64_t compare,
                      uint64_t swap_add, uct_completion_t *comp)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                uct_rc_mlx5_iface_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_RC_MLX5_CHECK_RES(iface, ep);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, iface->tx.atomic_desc_mp, desc);

    desc->super.super.func = proxy_cb;
    desc->comp             = comp;
    return uct_rc_mlx5_ep_atomic_post(ep, opcode, desc, length, remote_addr,
                                      rkey, compare_mask, compare, swap_add,
                                      MLX5_WQE_CTRL_CQ_UPDATE, UCS_INPROGRESS);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_ep_atomic_add(uct_ep_h tl_ep, int opcode, unsigned length,
                          uint64_t add, uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_mlx5_iface_t);
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_RC_MLX5_CHECK_RES(iface, ep);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, iface->tx.atomic_desc_mp, desc);

    desc->super.super.func = (uct_completion_callback_t)ucs_mpool_put;
    return uct_rc_mlx5_ep_atomic_post(ep, opcode, desc, length, remote_addr, rkey,
                                      0, 0, add, 0, UCS_OK);
}

ucs_status_t uct_rc_mlx5_ep_put_short(uct_ep_h tl_ep, void *buffer, unsigned length,
                                      uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    ucs_status_t status;

    status = uct_rc_mlx5_ep_inline_post(ep, MLX5_OPCODE_RDMA_WRITE, buffer,
                                        length, 0, 0, remote_addr, rkey);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, PUT, SHORT, length);
    return status;
}

ucs_status_t uct_rc_mlx5_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                      void *arg, size_t length, uint64_t remote_addr,
                                      uct_rkey_t rkey)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_mlx5_iface_t);
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;
    ucs_status_t status;

    UCT_CHECK_LENGTH(length <= iface->super.super.config.seg_size, "put_bcopy");
    UCT_RC_MLX5_CHECK_RES(iface, ep);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, iface->super.tx.mp, desc);

    desc->super.super.func = (uct_completion_callback_t)ucs_mpool_put;
    pack_cb(desc + 1, arg, length);
    status = uct_rc_mlx5_ep_bcopy_post(ep, MLX5_OPCODE_RDMA_WRITE, length, 0, NULL,
                                       0, remote_addr, rkey, MLX5_WQE_CTRL_CQ_UPDATE,
                                       desc, UCS_OK);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, PUT, BCOPY, length);
    return status;
}

ucs_status_t uct_rc_mlx5_ep_put_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                      uct_mem_h memh, uint64_t remote_addr,
                                      uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    ucs_status_t status;

    status = uct_rc_mlx5_ep_zcopy_post(ep, MLX5_OPCODE_RDMA_WRITE, buffer, length,
                                       memh, 0, NULL, 0, remote_addr, rkey,
                                       MLX5_WQE_CTRL_CQ_UPDATE, comp);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, PUT, ZCOPY, length);
    return status;
}

ucs_status_t uct_rc_mlx5_ep_get_bcopy(uct_ep_h tl_ep, size_t length,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_mlx5_iface_t);
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;
    ucs_status_t status;

    UCT_CHECK_LENGTH(length <= iface->super.super.config.seg_size, "get_bcopy");
    UCT_RC_MLX5_CHECK_RES(iface, ep);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, iface->super.tx.mp, desc);

    desc->super.super.func = uct_rc_ep_get_bcopy_completion;
    desc->comp             = comp;
#if !NVALGRIND
    desc->super.length     = length;
#endif
    status = uct_rc_mlx5_ep_bcopy_post(ep, MLX5_OPCODE_RDMA_READ, length, 0, NULL,
                                       0, remote_addr, rkey, MLX5_WQE_CTRL_CQ_UPDATE,
                                       desc, UCS_INPROGRESS);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, GET, BCOPY, length);
    return status;
}

ucs_status_t uct_rc_mlx5_ep_get_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                      uct_mem_h memh, uint64_t remote_addr,
                                      uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    ucs_status_t status;

    status = uct_rc_mlx5_ep_zcopy_post(ep, MLX5_OPCODE_RDMA_READ, buffer, length,
                                       memh, 0, NULL, 0, remote_addr, rkey,
                                       MLX5_WQE_CTRL_CQ_UPDATE, comp);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, GET, ZCOPY, length);
    return status;
}

ucs_status_t uct_rc_mlx5_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                     void *payload, unsigned length)
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    ucs_status_t status;

    UCT_CHECK_AM_ID(id);
    status = uct_rc_mlx5_ep_inline_post(ep, MLX5_OPCODE_SEND, payload, length,
                                        id, hdr, 0, 0);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, AM, SHORT,
                                 sizeof(hdr) + length);
    return status;
}

ucs_status_t uct_rc_mlx5_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                     uct_pack_callback_t pack_cb, void *arg,
                                     size_t length)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_mlx5_iface_t);
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;
    ucs_status_t status;
    uct_rc_hdr_t *rch;

    UCT_CHECK_AM_ID(id);
    UCT_RC_MLX5_CHECK_RES(iface, ep);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, iface->super.tx.mp, desc);

    desc->super.super.func = (uct_completion_callback_t)ucs_mpool_put;

    ucs_assert(sizeof(*rch) + length <= iface->super.super.config.seg_size);
    rch        = (void*)(desc + 1);
    rch->am_id = id;
    pack_cb(rch + 1, arg, length);
    status = uct_rc_mlx5_ep_bcopy_post(ep, MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_RAW,
                                       sizeof(*rch) + length, 0, NULL, 0, 0, 0, 0, desc,
                                       UCS_OK);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, AM, BCOPY, length);
    return status;
}

ucs_status_t uct_rc_mlx5_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, void *header,
                                     unsigned header_length, void *payload,
                                     size_t length, uct_mem_h memh,
                                     uct_completion_t *comp)
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    ucs_status_t status;

    UCT_CHECK_AM_ID(id);
    status = uct_rc_mlx5_ep_zcopy_post(ep, MLX5_OPCODE_SEND, payload, length, memh,
                                       id, header, header_length, 0, 0, 0, comp);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, AM, ZCOPY,
                                 header_length + length);
    return status;
}

ucs_status_t uct_rc_mlx5_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                         uint64_t remote_addr, uct_rkey_t rkey)
{
    return uct_rc_mlx5_ep_atomic_add(tl_ep, MLX5_OPCODE_ATOMIC_FA, sizeof(uint64_t),
                                     htonll(add), remote_addr, rkey);
}

ucs_status_t uct_rc_mlx5_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uct_completion_t *comp)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_FA, sizeof(uint64_t),
                                 uct_rc_ep_atomic_completion_64_be1, remote_addr,
                                 rkey, 0, 0, htonll(add), comp);
}

ucs_status_t uct_rc_mlx5_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uct_completion_t *comp)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_CS, sizeof(uint64_t),
                                 uct_rc_ep_atomic_completion_64_be1, remote_addr,
                                 rkey, 0, 0, htonll(swap), comp);
}

ucs_status_t uct_rc_mlx5_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_completion_t *comp)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_CS, sizeof(uint64_t),
                                 uct_rc_ep_atomic_completion_64_be1, remote_addr,
                                 rkey, 0, htonll(compare), htonll(swap), comp);
}

ucs_status_t uct_rc_mlx5_ep_atomic_add32(uct_ep_h tl_ep, uint32_t add,
                                         uint64_t remote_addr, uct_rkey_t rkey)
{
    return uct_rc_mlx5_ep_atomic_add(tl_ep, MLX5_OPCODE_ATOMIC_MASKED_FA,
                                     sizeof(uint32_t), htonl(add), remote_addr, rkey);
}

ucs_status_t uct_rc_mlx5_ep_atomic_fadd32(uct_ep_h tl_ep, uint32_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uct_completion_t *comp)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_FA, sizeof(uint32_t),
                                 uct_rc_ep_atomic_completion_32_be1, remote_addr,
                                 rkey, 0, 0, htonl(add), comp);
}

ucs_status_t uct_rc_mlx5_ep_atomic_swap32(uct_ep_h tl_ep, uint32_t swap,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uct_completion_t *comp)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_CS, sizeof(uint32_t),
                                 uct_rc_ep_atomic_completion_32_be1, remote_addr,
                                 rkey, 0, 0, htonl(swap), comp);
}

ucs_status_t uct_rc_mlx5_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_completion_t *comp)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_CS, sizeof(uint32_t),
                                 uct_rc_ep_atomic_completion_32_be1, remote_addr,
                                 rkey, (uint32_t)-1, htonl(compare), htonl(swap),
                                 comp);
}

ucs_status_t uct_rc_mlx5_ep_flush(uct_ep_h tl_ep)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_mlx5_iface_t);
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    ucs_status_t status;
    uint16_t exp_max_pi;

    /*
     * If we got completion for the last posted WQE, max_pi would be advanced
     * to the value calculated from prev_sw_pi - which is the index where the last
     * posted WQE started. See also uct_rc_mlx5_iface_poll_tx().
     */
    exp_max_pi = uct_rc_mlx5_calc_max_pi(iface, ep->tx.prev_sw_pi);
    if (ep->tx.max_pi == exp_max_pi) {
        UCT_TL_EP_STAT_FLUSH(&ep->super.super);
        return UCS_OK;
    }

    if (ep->super.unsignaled != 0) {
        status = uct_rc_mlx5_ep_inline_post(ep, MLX5_OPCODE_NOP, NULL, 0, 0, 0, 0, 0);
        if (status != UCS_OK) {
            UCT_TL_EP_STAT_FLUSH(&ep->super.super);
            return status;
        }
    }

    return UCS_INPROGRESS;
}

static UCS_CLASS_INIT_FUNC(uct_rc_mlx5_ep_t, uct_iface_h tl_iface)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_mlx5_iface_t);
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

    self->qp_num        = self->super.qp->qp_num;
    self->tx.qstart     = qp_info.sq.buf;
    self->tx.qend       = qp_info.sq.buf + (MLX5_SEND_WQE_BB *  qp_info.sq.wqe_cnt);
    self->tx.seg        = self->tx.qstart;
    self->tx.sw_pi      = 0;
    self->tx.prev_sw_pi = -1;
    self->tx.max_pi     = uct_rc_mlx5_calc_max_pi(iface, self->tx.prev_sw_pi);
    self->tx.bf_reg     = qp_info.bf.reg;
    self->tx.bf_size    = qp_info.bf.size;
    self->tx.dbrec      = &qp_info.dbrec[MLX5_SND_DBR];

    memset(self->tx.qstart, 0, self->tx.qend - self->tx.qstart);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_ep_t)
{
}

UCS_CLASS_DEFINE(uct_rc_mlx5_ep_t, uct_rc_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_mlx5_ep_t, uct_ep_t, uct_iface_h);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_mlx5_ep_t, uct_ep_t);

