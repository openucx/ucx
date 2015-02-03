/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_mlx5.h"

#include <arpa/inet.h> /* For htonl */


#define UCT_RC_MLX5_GET_SW_PI(_ep, _sw_pi_n) \
    { \
        uint16_t sw_pi = (_ep)->tx.sw_pi; \
        if (UCS_CIRCULAR_COMPARE16(sw_pi, >=, (_ep)->tx.max_pi)) { \
            return UCS_ERR_WOULD_BLOCK; \
        } \
        _sw_pi_n = htons(sw_pi); \
    }
#define UCT_RC_MLX5_OPCODE_FLAG_RAW   0x100
#define UCT_RC_MLX5_OPCODE_MASK       0xff


static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_post_send(uct_rc_mlx5_ep_t *ep, struct mlx5_wqe_ctrl_seg *ctrl,
                      uint16_t sw_pi_n, unsigned opcode, unsigned sig_flag,
                      unsigned wqe_size)
{
    uct_rc_mlx5_iface_t *iface;
    uint64_t *src, *dst;
    unsigned i;

    /* TODO use SSE to build the WQE */
    ctrl->opmod_idx_opcode   = (sw_pi_n << 8) | (opcode << 24);
    ctrl->qpn_ds             = htonl(wqe_size) | ep->qpn_ds;
    ctrl->fm_ce_se           = sig_flag;

    /* TODO Put memory store fence here too, to prevent WC being flushed after DBrec */
    ucs_memory_cpu_store_fence();

    /* Write doorbell record */
    *ep->tx.dbrec = sw_pi_n << 16;

    /* Make sure that doorbell record is written before ringing the doorbell */
    ucs_memory_bus_store_fence();

    /* Set up copy pointers */
    dst = ep->tx.bf_reg;
    src = (void*)ctrl;

    /* BF copy */
    /* TODO support several WQEBBs */
    /* TODO support DB without BF */
    ucs_assert(wqe_size*16 <= MLX5_SEND_WQE_BB);
    for (i = 0; i < MLX5_SEND_WQE_BB / sizeof(*dst); ++i) {
        *(dst++) = *(src++);
    }

    /* We don't want the compiler to reorder instructions and hurt latency */
    ucs_compiler_fence();

    /* Advance queue pointer */
    ucs_assert(ctrl == ep->tx.seg);
    if (ucs_unlikely((ep->tx.seg = src) >= ep->tx.qend)) {
        ep->tx.seg = ep->tx.qstart;
    }

    /* Flip BF register */
    ep->tx.bf_reg = (void*) ((uintptr_t) ep->tx.bf_reg ^ ep->tx.bf_size);

    /* Count number of posts */
    iface = ucs_derived_of(ep->super.super.iface, uct_rc_mlx5_iface_t);
    ++ep->tx.sw_pi;
    ++iface->super.tx.outstanding;
    if (sig_flag) {
        ep->super.tx.unsignaled = 0;
    } else {
        ++ep->super.tx.unsignaled;
    }
}

static UCS_F_ALWAYS_INLINE unsigned uct_rc_mlx5_ep_wqe_nsegs(unsigned seg_len)
{
    return (sizeof(struct mlx5_wqe_ctrl_seg) + seg_len + 15) / 16;
}

static UCS_F_ALWAYS_INLINE uint8_t
uct_rc_mlx5_iface_tx_moderation(uct_rc_mlx5_ep_t* ep, uct_rc_mlx5_iface_t* iface)
{
    return (ep->super.tx.unsignaled >= iface->super.config.tx_moderation) ?
           MLX5_WQE_CTRL_CQ_UPDATE : 0;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_ep_set_rdma_seg(struct mlx5_wqe_raddr_seg *raddr, uint64_t rdma_raddr,
                            uct_rkey_t rdma_rkey)
{
    /* TODO sse */
    raddr->raddr = htonll(rdma_raddr);
    raddr->rkey  = (uint32_t)rdma_rkey;
}

static UCS_F_ALWAYS_INLINE
void uct_rc_mlx5_ep_set_dptr_seg(struct mlx5_wqe_data_seg *dptr, void *address,
                                 unsigned length, uint32_t lkey)
{
    /* TODO sse */
    dptr->byte_count = htonl(length);
    dptr->lkey       = htonl(lkey);
    dptr->addr       = htonll((uintptr_t)address);
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
    uct_rc_am_short_hdr_t        *am;
    uct_rc_mlx5_iface_t *iface;
    uint16_t sw_pi_n;
    unsigned wqe_size;
    unsigned sig_flag;

    UCT_RC_MLX5_GET_SW_PI(ep, sw_pi_n);

    ctrl = ep->tx.seg;
    switch (opcode) {
    case MLX5_OPCODE_SEND:
        /* Set inline segment which has AM id, AM header, and AM payload */
        wqe_size         = uct_rc_mlx5_ep_wqe_nsegs(sizeof(*inl) + sizeof(*am) + length);
        iface            = ucs_derived_of(ep->super.super.iface, uct_rc_mlx5_iface_t);
        inl              = (void*)(ctrl + 1);
        inl->byte_count  = htonl((length + sizeof(*am)) | MLX5_INLINE_SEG);
        am               = (void*)(inl + 1);
        am->rc_hdr.am_id = am_id;
        am->am_hdr       = am_hdr;
        memcpy(am + 1, buffer, length);
        sig_flag         = uct_rc_iface_tx_moderation(&iface->super, &ep->super,
                                                      MLX5_WQE_CTRL_CQ_UPDATE);
        break;

    case MLX5_OPCODE_RDMA_WRITE:
        /* Set RDMA segment */
        if (length == 0) {
            wqe_size     = uct_rc_mlx5_ep_wqe_nsegs(sizeof(*raddr));
        } else {
            wqe_size     = uct_rc_mlx5_ep_wqe_nsegs(sizeof(*raddr) + sizeof(*inl) + length);
        }
        raddr            = (void*)(ctrl + 1);
        uct_rc_mlx5_ep_set_rdma_seg(raddr, rdma_raddr, rdma_rkey);
        inl              = (void*)(raddr + 1);
        inl->byte_count  = htonl(length | MLX5_INLINE_SEG);
        memcpy(inl + 1, buffer, length);
        sig_flag         = MLX5_WQE_CTRL_CQ_UPDATE;
        break;

    case MLX5_OPCODE_NOP:
        /* Empty inline segment
         * TODO can there be no data segment at all? */
        wqe_size         = uct_rc_mlx5_ep_wqe_nsegs(0);
        inl              = (void*)(ctrl + 1);
        inl->byte_count  = htonl(MLX5_INLINE_SEG);
        sig_flag         = MLX5_WQE_CTRL_CQ_UPDATE;
        break;

    default:
        return UCS_ERR_INVALID_PARAM;
    }

    uct_rc_mlx5_post_send(ep, ctrl, sw_pi_n, opcode, sig_flag, wqe_size);
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
    struct mlx5_wqe_inl_data_seg                 *inl;
    struct uct_ib_mlx5_atomic_masked_cswap32_seg *masked_cswap32;
    struct uct_ib_mlx5_atomic_masked_fadd32_seg  *masked_fadd32;
    struct uct_ib_mlx5_atomic_masked_cswap64_seg *masked_cswap64;

    uct_rc_mlx5_iface_t *iface;
    uct_rc_hdr_t        *rch;
    uint16_t            sw_pi_n;
    unsigned            wqe_size, inl_seg_size, atomic_seg_size;

    UCT_RC_MLX5_GET_SW_PI(ep, sw_pi_n);

    ucs_assert((signal == 0) || (signal == MLX5_WQE_CTRL_CQ_UPDATE));

    ctrl = ep->tx.seg;
    switch (opcode_flags) {
    case MLX5_OPCODE_SEND:
        inl_seg_size     = ucs_align_up_pow2(sizeof(*inl) + sizeof(*rch) + am_hdr_len, 16);

        /* Inline segment with AM ID and header */
        inl              = (void*)(ctrl + 1);
        inl->byte_count  = htonl((sizeof(*rch) + am_hdr_len) | MLX5_INLINE_SEG);
        rch              = (void*)(inl + 1);
        rch->am_id       = am_id;
        memcpy(rch + 1, am_hdr, am_hdr_len);

        /* Data segment with payload */
        if (length != 0) {
            uct_rc_mlx5_ep_set_dptr_seg((void*)(ctrl + 1) + inl_seg_size, buffer,
                                        length, *lkey_p);
            wqe_size     = uct_rc_mlx5_ep_wqe_nsegs(inl_seg_size +
                                                    sizeof(struct mlx5_wqe_data_seg));
        } else {
            wqe_size     = uct_rc_mlx5_ep_wqe_nsegs(inl_seg_size);
        }
        break;

    case MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_RAW:
        /* Data segment only */
        ucs_assert(length < (2ul << 30));
        ucs_assert(length > 0);
        wqe_size         = uct_rc_mlx5_ep_wqe_nsegs(sizeof(struct mlx5_wqe_data_seg));
        uct_rc_mlx5_ep_set_dptr_seg((void*)(ctrl + 1), buffer, length, *lkey_p);
        break;

    case MLX5_OPCODE_RDMA_READ:
    case MLX5_OPCODE_RDMA_WRITE:
        /* Set RDMA segment */
        ucs_assert(length < (2ul << 30));
        raddr            = (void*)(ctrl + 1);
        uct_rc_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        /* Data segment */
        if (length != 0) {
            uct_rc_mlx5_ep_set_dptr_seg((void*)(raddr + 1), buffer, length, *lkey_p);
            wqe_size     = uct_rc_mlx5_ep_wqe_nsegs(sizeof(*raddr) +
                                                    sizeof(struct mlx5_wqe_data_seg));
        } else {
            wqe_size     = uct_rc_mlx5_ep_wqe_nsegs(sizeof(*raddr));
        }
        break;

    case MLX5_OPCODE_ATOMIC_FA:
    case MLX5_OPCODE_ATOMIC_CS:
        ucs_assert(length == sizeof(uint64_t));
        raddr = (void*)(ctrl + 1);
        uct_rc_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        atomic            = (void*)(raddr + 1);
        if (opcode_flags == MLX5_OPCODE_ATOMIC_CS) {
            atomic->compare   = compare;
        }
        atomic->swap_add  = swap_add;

        uct_rc_mlx5_ep_set_dptr_seg((void*)(atomic + 1), buffer, length, *lkey_p);
        wqe_size = uct_rc_mlx5_ep_wqe_nsegs(sizeof(*raddr) + sizeof(*atomic) +
                                            sizeof(struct mlx5_wqe_data_seg));
        break;

    case MLX5_OPCODE_ATOMIC_MASKED_CS:
        raddr = (void*)(ctrl + 1);
        uct_rc_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        switch (length) {
        case sizeof(uint32_t):
            masked_cswap32 = (void*)(raddr + 1);
            masked_cswap32->swap         = swap_add;
            masked_cswap32->compare      = compare;
            masked_cswap32->swap_mask    = (uint32_t)-1;
            masked_cswap32->compare_mask = compare_mask;
            atomic_seg_size              = sizeof(*masked_cswap32);
            break;
        case sizeof(uint64_t):
            masked_cswap64 = (void*)(raddr + 1);
            masked_cswap64->swap         = swap_add;
            masked_cswap64->compare      = compare;
            masked_cswap64->swap_mask    = (uint64_t)-1;
            masked_cswap64->compare_mask = compare_mask;
            atomic_seg_size              = sizeof(*masked_cswap64);
            /* TODO this flow is disabled until we support multiple WQEBB */
            break;
        default:
            return UCS_ERR_INVALID_PARAM;
        }

        uct_rc_mlx5_ep_set_dptr_seg((void*)(raddr + 1) + atomic_seg_size,
                                    buffer, length, *lkey_p);
        wqe_size = uct_rc_mlx5_ep_wqe_nsegs(sizeof(*raddr) + atomic_seg_size +
                                            sizeof(struct mlx5_wqe_data_seg));
        break;

     case MLX5_OPCODE_ATOMIC_MASKED_FA:
        ucs_assert(length == sizeof(uint32_t));
        raddr = (void*)(ctrl + 1);
        uct_rc_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        masked_fadd32                 = (void*)(raddr + 1);
        masked_fadd32->add            = swap_add;
        masked_fadd32->filed_boundary = 0;

        uct_rc_mlx5_ep_set_dptr_seg((void*)(masked_fadd32 + 1), buffer, length,
                                    *lkey_p);
        wqe_size = uct_rc_mlx5_ep_wqe_nsegs(sizeof(*raddr) + sizeof(*masked_fadd32) +
                                            sizeof(struct mlx5_wqe_data_seg));
        break;

    default:
        return UCS_ERR_INVALID_PARAM;
    }

    if (!signal) {
        iface  = ucs_derived_of(ep->super.super.iface, uct_rc_mlx5_iface_t);
        signal = uct_rc_iface_tx_moderation(&iface->super, &ep->super,
                                            MLX5_WQE_CTRL_CQ_UPDATE);
    }

    uct_rc_mlx5_post_send(ep, ctrl, sw_pi_n,
                          (opcode_flags & UCT_RC_MLX5_OPCODE_MASK),
                          signal, wqe_size);
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

    desc->queue.sn = ep->tx.sw_pi;

    status = uct_rc_mlx5_ep_dptr_post(ep, opcode, desc + 1, length, &desc->lkey,
                                      am_id, am_hdr, am_hdr_len, rdma_raddr, rdma_rkey,
                                      0, 0, 0, force_sig);
    if (status != UCS_OK) {
        ucs_mpool_put(desc);
        return status;
    }

    ucs_callbackq_push(&ep->super.tx.comp, &desc->queue);
    return success;
}

/*
 * Helper function for zero-copy post.
 * Adds user completion to the callback queue.
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_ep_zcopy_post(uct_rc_mlx5_ep_t *ep, unsigned opcode, void *buffer,
                          unsigned length, uct_lkey_t lkey,
                          /* SEND */ uint8_t am_id, void *am_hdr, unsigned am_hdr_len,
                          /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey,
                          int force_sig, uct_completion_t *comp)
{
    ucs_status_t status;
    uint16_t sn;

    sn = ep->tx.sw_pi;
    status = uct_rc_mlx5_ep_dptr_post(ep, opcode,
                                      buffer, length, &uct_ib_lkey_mr(lkey)->lkey,
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

    desc->queue.sn = ep->tx.sw_pi;

    status = uct_rc_mlx5_ep_dptr_post(ep, opcode, desc + 1, length, &desc->lkey,
                                      0, NULL, 0, remote_addr, rkey, compare_mask,
                                      compare, swap_add, signal);
    if (status != UCS_OK) {
        ucs_mpool_put(desc);
        return status;
    }

    ucs_callbackq_push(&ep->super.tx.comp, &desc->queue);
    return success;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_ep_atomic(uct_rc_mlx5_ep_t *ep, int opcode, unsigned length,
                      ucs_callback_func_t proxy_cb, uint64_t remote_addr,
                      uct_rkey_t rkey, uint64_t compare_mask, uint64_t compare,
                      uint64_t swap_add, uct_imm_recv_callback_t cb, void *arg)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                                uct_rc_mlx5_iface_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_TL_IFACE_GET_TX_DESC(iface->tx.atomic_desc_mp, desc, UCS_ERR_WOULD_BLOCK);
    desc->queue.super.func = proxy_cb;
    desc->imm_recv.cb      = cb;
    desc->imm_recv.arg     = arg;

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

    UCT_TL_IFACE_GET_TX_DESC(iface->tx.atomic_desc_mp, desc, UCS_ERR_WOULD_BLOCK);
    desc->queue.super.func = (ucs_callback_func_t)ucs_mpool_put;

    return uct_rc_mlx5_ep_atomic_post(ep, opcode, desc, length, remote_addr, rkey,
                                      0, 0, add, 0, UCS_OK);
}

ucs_status_t uct_rc_mlx5_ep_put_short(uct_ep_h tl_ep, void *buffer, unsigned length,
                                      uint64_t remote_addr, uct_rkey_t rkey)
{
    return uct_rc_mlx5_ep_inline_post(tl_ep, MLX5_OPCODE_RDMA_WRITE, buffer,
                                      length, 0, 0, remote_addr, rkey);
}

ucs_status_t uct_rc_mlx5_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                      void *arg, size_t length, uint64_t remote_addr,
                                      uct_rkey_t rkey)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_mlx5_iface_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_TL_IFACE_GET_TX_DESC(iface->super.tx.mp, desc, UCS_ERR_WOULD_BLOCK);
    desc->queue.super.func = (ucs_callback_func_t)ucs_mpool_put;

    pack_cb(desc + 1, arg, length);
    return uct_rc_mlx5_ep_bcopy_post(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                     MLX5_OPCODE_RDMA_WRITE, length,
                                     0, NULL, 0, remote_addr, rkey,
                                     MLX5_WQE_CTRL_CQ_UPDATE, desc, UCS_OK);
}

ucs_status_t uct_rc_mlx5_ep_put_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                      uct_lkey_t lkey, uint64_t remote_addr,
                                      uct_rkey_t rkey, uct_completion_t *comp)
{
    return uct_rc_mlx5_ep_zcopy_post(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                     MLX5_OPCODE_RDMA_WRITE, buffer, length,
                                     lkey, 0, NULL, 0, remote_addr, rkey,
                                     MLX5_WQE_CTRL_CQ_UPDATE, comp);
}

ucs_status_t uct_rc_mlx5_ep_get_bcopy(uct_ep_h tl_ep, size_t length,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_bcopy_recv_callback_t cb, void *arg)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_mlx5_iface_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_TL_IFACE_GET_TX_DESC(iface->super.tx.mp, desc, UCS_ERR_WOULD_BLOCK);
    desc->queue.super.func  = uct_rc_ep_get_bcopy_completion;
    desc->bcopy_recv.cb     = cb;
    desc->bcopy_recv.arg    = arg;
    desc->bcopy_recv.length = length;

    return uct_rc_mlx5_ep_bcopy_post(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                     MLX5_OPCODE_RDMA_READ, length,
                                     0, NULL, 0, remote_addr, rkey,
                                     MLX5_WQE_CTRL_CQ_UPDATE, desc,
                                     UCS_INPROGRESS);
}

ucs_status_t uct_rc_mlx5_ep_get_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                      uct_lkey_t lkey, uint64_t remote_addr,
                                      uct_rkey_t rkey, uct_completion_t *comp)
{
    return uct_rc_mlx5_ep_zcopy_post(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                     MLX5_OPCODE_RDMA_READ, buffer, length,
                                     lkey, 0, NULL, 0, remote_addr, rkey,
                                     MLX5_WQE_CTRL_CQ_UPDATE, comp);
}

ucs_status_t uct_rc_mlx5_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                     void *payload, unsigned length)
{
    return uct_rc_mlx5_ep_inline_post(tl_ep, MLX5_OPCODE_SEND, payload, length,
                                      id, hdr, 0, 0);
}

ucs_status_t uct_rc_mlx5_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                     uct_pack_callback_t pack_cb, void *arg,
                                     size_t length)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_mlx5_iface_t);
    uct_rc_iface_send_desc_t *desc;
    uct_rc_hdr_t *rch;

    UCT_TL_IFACE_GET_TX_DESC(iface->super.tx.mp, desc, UCS_ERR_WOULD_BLOCK);
    desc->queue.super.func = (void*)ucs_mpool_put;

    rch = (void*)(desc + 1);
    rch->am_id = id;
    ucs_assert(sizeof(*rch) + length <= iface->super.super.config.seg_size);
    pack_cb(rch + 1, arg, length);

    return uct_rc_mlx5_ep_bcopy_post(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                     MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_RAW,
                                     sizeof(*rch) + length, 0, NULL, 0, 0, 0, 0, desc,
                                     UCS_OK);
}

ucs_status_t uct_rc_mlx5_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, void *header,
                                     unsigned header_length, void *payload,
                                     size_t length, uct_lkey_t lkey,
                                     uct_completion_t *comp)
{
    return uct_rc_mlx5_ep_zcopy_post(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                     MLX5_OPCODE_SEND, payload, length, lkey,
                                     id, header, header_length, 0, 0, 0, comp);
}

ucs_status_t uct_rc_mlx5_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                         uint64_t remote_addr, uct_rkey_t rkey)
{
    return uct_rc_mlx5_ep_atomic_add(tl_ep, MLX5_OPCODE_ATOMIC_FA, sizeof(uint64_t),
                                     htonll(add), remote_addr, rkey);
}

ucs_status_t uct_rc_mlx5_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uct_imm_recv_callback_t cb, void *arg)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_FA, sizeof(uint64_t),
                                 uct_rc_ep_atomic_completion_64_be1, remote_addr,
                                 rkey, 0, 0, htonll(add), cb, arg);
}

ucs_status_t uct_rc_mlx5_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uct_imm_recv_callback_t cb, void *arg)
{
    /*
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_CS, sizeof(uint64_t),
                                 uct_rc_ep_atomic_completion_64_be1, remote_addr,
                                 rkey, 0, 0, htonll(swap), cb, arg);
     TODO support >1 WQEBB
    */
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_rc_mlx5_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_imm_recv_callback_t cb, void *arg)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_CS, sizeof(uint64_t),
                                 uct_rc_ep_atomic_completion_64_be1, remote_addr,
                                 rkey, 0, htonll(compare), htonll(swap), cb, arg);
}

ucs_status_t uct_rc_mlx5_ep_atomic_add32(uct_ep_h tl_ep, uint32_t add,
                                         uint64_t remote_addr, uct_rkey_t rkey)
{
    return uct_rc_mlx5_ep_atomic_add(tl_ep, MLX5_OPCODE_ATOMIC_MASKED_FA,
                                     sizeof(uint32_t), htonl(add), remote_addr, rkey);
}

ucs_status_t uct_rc_mlx5_ep_atomic_fadd32(uct_ep_h tl_ep, uint32_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uct_imm_recv_callback_t cb, void *arg)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_FA, sizeof(uint32_t),
                                 uct_rc_ep_atomic_completion_32_be1, remote_addr,
                                 rkey, 0, 0, htonl(add), cb, arg);
}

ucs_status_t uct_rc_mlx5_ep_atomic_swap32(uct_ep_h tl_ep, uint32_t swap,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uct_imm_recv_callback_t cb, void *arg)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_CS, sizeof(uint32_t),
                                 uct_rc_ep_atomic_completion_32_be1, remote_addr,
                                 rkey, 0, 0, htonl(swap), cb, arg);
}

ucs_status_t uct_rc_mlx5_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_imm_recv_callback_t cb, void *arg)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_CS, sizeof(uint32_t),
                                 uct_rc_ep_atomic_completion_32_be1, remote_addr,
                                 rkey, (uint32_t)-1, htonl(compare), htonl(swap),
                                 cb, arg);
}

ucs_status_t uct_rc_mlx5_ep_flush(uct_ep_h tl_ep)
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);

    if (ep->super.tx.unsignaled == 0) {
        return UCS_OK;
    }

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

