/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rc_mlx5.h"

#include <uct/ib/mlx5/ib_mlx5_log.h>
#include <ucs/arch/cpu.h>
#include <ucs/sys/compiler.h>
#include <arpa/inet.h> /* For htonl */


#define UCT_RC_MLX5_OPCODE_FLAG_RAW   0x100
#define UCT_RC_MLX5_OPCODE_MASK       0xff


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

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_post_send(uct_rc_mlx5_ep_t *ep, struct mlx5_wqe_ctrl_seg *ctrl,
                      uint8_t opcode, uint8_t opmod, unsigned sig_flag, unsigned wqe_size)
{
    uint16_t posted;

    uct_ib_mlx5_set_ctrl_seg(ctrl, ep->tx.wq.sw_pi, 
                             opcode, opmod, ep->qp_num, sig_flag, wqe_size);

    uct_ib_mlx5_log_tx(ucs_derived_of(ep->super.super.super.iface, uct_ib_iface_t),
                       IBV_QPT_RC, ctrl, ep->tx.wq.qstart, ep->tx.wq.qend,
                       (opcode == MLX5_OPCODE_SEND) ? uct_rc_ep_am_packet_dump : NULL);

    posted = uct_ib_mlx5_post_send(&ep->tx.wq, ctrl, wqe_size);
    ep->super.available -= posted;
    uct_rc_ep_tx_posted(&ep->super, sig_flag & MLX5_WQE_CTRL_CQ_UPDATE);
}

static UCS_F_ALWAYS_INLINE uint8_t
uct_rc_mlx5_iface_tx_moderation(uct_rc_mlx5_ep_t* ep, uct_rc_mlx5_iface_t* iface)
{
    return (ep->super.unsignaled >= iface->super.config.tx_moderation) ?
           MLX5_WQE_CTRL_CQ_UPDATE : 0;
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
                           const void *buffer, unsigned length,
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

    ctrl = ep->tx.wq.curr;
    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    switch (opcode) {
    case MLX5_OPCODE_SEND:
        /* Set inline segment which has AM id, AM header, and AM payload */
        wqe_size         = sizeof(*ctrl) + sizeof(*inl) + sizeof(*am) + length;
        UCT_CHECK_LENGTH(wqe_size, UCT_RC_MLX5_MAX_BB * MLX5_SEND_WQE_BB,
                         "am_short");
        inl              = (void*)(ctrl + 1);
        inl->byte_count  = htonl((length + sizeof(*am)) | MLX5_INLINE_SEG);
        am               = (void*)(inl + 1);
        am->rc_hdr.am_id = am_id;
        am->am_hdr       = am_hdr;
        uct_ib_mlx5_inline_copy(am + 1, buffer, length, &ep->tx.wq);
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
        UCT_CHECK_LENGTH(wqe_size, UCT_RC_MLX5_MAX_BB * MLX5_SEND_WQE_BB,
                        "put_short");
        raddr            = (void*)(ctrl + 1);
        uct_rc_mlx5_ep_set_rdma_seg(raddr, rdma_raddr, rdma_rkey);
        inl              = (void*)(raddr + 1);
        inl->byte_count  = htonl(length | MLX5_INLINE_SEG);
        uct_ib_mlx5_inline_copy(inl + 1, buffer, length, &ep->tx.wq);
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
static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_ep_dptr_post(uct_rc_mlx5_ep_t *ep, unsigned opcode_flags,
                         const void *buffer, unsigned length, uint32_t *lkey_p,
                         /* SEND */ uint8_t am_id, const void *am_hdr, unsigned am_hdr_len,
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
    uint8_t             opmod;

    iface = ucs_derived_of(ep->super.super.super.iface, uct_rc_mlx5_iface_t);
    if (!signal) {
        signal = uct_rc_iface_tx_moderation(&iface->super, &ep->super,
                                            MLX5_WQE_CTRL_CQ_UPDATE);
    } else {
        ucs_assert(signal == MLX5_WQE_CTRL_CQ_UPDATE);
    }

    opmod = 0;
    ctrl = ep->tx.wq.curr;
    switch (opcode_flags) {
    case MLX5_OPCODE_SEND:
        inl_seg_size     = ucs_align_up_pow2(sizeof(*inl) + sizeof(*rch) + am_hdr_len,
                                             UCT_IB_MLX5_WQE_SEG_SIZE);

        ucs_assert(sizeof(*ctrl) + inl_seg_size + sizeof(*dptr) <=
                   UCT_RC_MLX5_MAX_BB * MLX5_SEND_WQE_BB);
        ucs_assert(length + sizeof(*rch) + am_hdr_len <=
                   iface->super.super.config.seg_size);

        /* Inline segment with AM ID and header */
        inl              = (void*)(ctrl + 1);
        inl->byte_count  = htonl((sizeof(*rch) + am_hdr_len) | MLX5_INLINE_SEG);
        rch              = (void*)(inl + 1);
        rch->am_id       = am_id;

        uct_ib_mlx5_inline_copy(rch + 1, am_hdr, am_hdr_len, &ep->tx.wq);

        /* Data segment with payload */
        if (length == 0) {
            wqe_size     = sizeof(*ctrl) + inl_seg_size;
        } else {
            wqe_size     = sizeof(*ctrl) + inl_seg_size + sizeof(*dptr);
            dptr         = (void*)(ctrl + 1) + inl_seg_size;
            if (ucs_unlikely((void*)dptr >= ep->tx.wq.qend)) {
                dptr = (void*)dptr - (ep->tx.wq.qend - ep->tx.wq.qstart);
            }

            ucs_assert((void*)dptr       >= ep->tx.wq.qstart);
            ucs_assert((void*)(dptr + 1) <= ep->tx.wq.qend);
            uct_ib_mlx5_set_data_seg(dptr, buffer, length, *lkey_p);
        }
        break;

    case MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_RAW:
        /* Data segment only */
        ucs_assert(length < (2ul << 30));
        ucs_assert(length <= iface->super.super.config.seg_size);

        wqe_size         = sizeof(*ctrl) + sizeof(*dptr);
        uct_ib_mlx5_set_data_seg((void*)(ctrl + 1), buffer, length, *lkey_p);
        break;

    case MLX5_OPCODE_RDMA_READ:
    case MLX5_OPCODE_RDMA_WRITE:
        /* Set RDMA segment */
        ucs_assert(length <= UCT_IB_MAX_MESSAGE_SIZE);

        raddr            = (void*)(ctrl + 1);
        uct_rc_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        /* Data segment */
        if (length == 0) {
            wqe_size     = sizeof(*ctrl) + sizeof(*raddr);
        } else {
            wqe_size     = sizeof(*ctrl) + sizeof(*raddr) + sizeof(*dptr);
            uct_ib_mlx5_set_data_seg((void*)(raddr + 1), buffer, length, *lkey_p);
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

        uct_ib_mlx5_set_data_seg((void*)(atomic + 1), buffer, length, *lkey_p);
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
            ucs_assert((void*)dptr <= ep->tx.wq.qend);
            if (dptr == ep->tx.wq.qend) {
                dptr = ep->tx.wq.qstart;
            } else {
                ucs_assert((void*)masked_cswap64 < ep->tx.wq.qend);
            }
            break;
        default:
            ucs_assert(0);
        }

        uct_ib_mlx5_set_data_seg(dptr, buffer, length, *lkey_p);
        break;

     case MLX5_OPCODE_ATOMIC_MASKED_FA:
        ucs_assert(length == sizeof(uint32_t));
        raddr = (void*)(ctrl + 1);
        uct_rc_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        opmod                         = UCT_IB_MLX5_OPMOD_EXT_ATOMIC(2);
        masked_fadd32                 = (void*)(raddr + 1);
        masked_fadd32->add            = swap_add;
        masked_fadd32->filed_boundary = 0;

        uct_ib_mlx5_set_data_seg((void*)(masked_fadd32 + 1), buffer, length,
                                 *lkey_p);
        wqe_size                      = sizeof(*ctrl) + sizeof(*raddr) +
                                        sizeof(*masked_fadd32) + sizeof(*dptr);
        break;

    default:
        ucs_fatal("invalid send opcode");
    }

    uct_rc_mlx5_post_send(ep, ctrl, (opcode_flags & UCT_RC_MLX5_OPCODE_MASK),
                          opmod, signal, wqe_size);
}

/*
 *
 * Helper function for buffer-copy post.
 * Adds the descriptor to the callback queue.
 */
static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_ep_bcopy_post(uct_rc_mlx5_ep_t *ep, unsigned opcode, unsigned length,
                          /* SEND */ uint8_t am_id, void *am_hdr, unsigned am_hdr_len,
                          /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey,
                          int force_sig, uct_rc_iface_send_desc_t *desc)
{
    desc->super.sn = ep->tx.wq.sw_pi;
    uct_rc_mlx5_ep_dptr_post(ep, opcode, desc + 1, length, &desc->lkey,
                             am_id, am_hdr, am_hdr_len, rdma_raddr, rdma_rkey,
                             0, 0, 0, force_sig);
    ucs_queue_push(&ep->super.outstanding, &desc->super.queue);
}

/*
 * Helper function for zero-copy post.
 * Adds user completion to the callback queue.
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_ep_zcopy_post(uct_rc_mlx5_ep_t *ep, unsigned opcode, const void *buffer,
                          unsigned length, struct ibv_mr *mr,
                          /* SEND */ uint8_t am_id, const void *am_hdr, unsigned am_hdr_len,
                          /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey,
                          int force_sig, uct_completion_t *comp)
{
    uct_rc_mlx5_iface_t *iface  = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_rc_mlx5_iface_t);
    uint16_t sn;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    sn = ep->tx.wq.sw_pi;
    uct_rc_mlx5_ep_dptr_post(ep, opcode, buffer, length, &mr->lkey,
                             am_id, am_hdr, am_hdr_len, rdma_raddr, rdma_rkey,
                             0, 0, 0,
                             (comp == NULL) ? force_sig : MLX5_WQE_CTRL_CQ_UPDATE);

    uct_rc_ep_add_send_comp(&iface->super, &ep->super, comp, sn);
    return UCS_INPROGRESS;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_ep_atomic_post(uct_rc_mlx5_ep_t *ep, unsigned opcode,
                           uct_rc_iface_send_desc_t *desc, unsigned length,
                           uint64_t remote_addr, uct_rkey_t rkey,
                           uint64_t compare_mask, uint64_t compare,
                           uint64_t swap_add, int signal)
{
    desc->super.sn = ep->tx.wq.sw_pi;
    uct_rc_mlx5_ep_dptr_post(ep, opcode, desc + 1, length, &desc->lkey,
                                      0, NULL, 0, remote_addr, rkey, compare_mask,
                                      compare, swap_add, signal);

    UCT_TL_EP_STAT_ATOMIC(&ep->super.super);
    ucs_queue_push(&ep->super.outstanding, &desc->super.queue);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_ep_atomic(uct_rc_mlx5_ep_t *ep, int opcode, void *result, unsigned length,
                      uct_rc_send_handler_t handler, uint64_t remote_addr,
                      uct_rkey_t rkey, uint64_t compare_mask, uint64_t compare,
                      uint64_t swap_add, uct_completion_t *comp)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                uct_rc_mlx5_iface_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_CHECK_PARAM(comp != NULL, "completion must be non-NULL");
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, &iface->tx.atomic_desc_mp, desc);

    desc->super.handler   = handler;
    desc->super.buffer    = result;
    desc->super.user_comp = comp;
    uct_rc_mlx5_ep_atomic_post(ep, opcode, desc, length, remote_addr, rkey,
                               compare_mask, compare, swap_add,
                               MLX5_WQE_CTRL_CQ_UPDATE);
    return UCS_INPROGRESS;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_ep_atomic_add(uct_ep_h tl_ep, int opcode, unsigned length,
                          uint64_t add, uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_mlx5_iface_t);
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, &iface->tx.atomic_desc_mp, desc);

    desc->super.handler = (uct_rc_send_handler_t)ucs_mpool_put;
    uct_rc_mlx5_ep_atomic_post(ep, opcode, desc, length, remote_addr, rkey, 0,
                               0, add, 0);
    return UCS_OK;
}

ucs_status_t uct_rc_mlx5_ep_put_short(uct_ep_h tl_ep, const void *buffer, unsigned length,
                                      uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    ucs_status_t status;

    status = uct_rc_mlx5_ep_inline_post(ep, MLX5_OPCODE_RDMA_WRITE, buffer,
                                        length, 0, 0, remote_addr, rkey);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, PUT, SHORT, length);
    return status;
}

ssize_t uct_rc_mlx5_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                 void *arg, uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_mlx5_iface_t);
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;
    size_t length;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, &iface->super.tx.mp, desc);

    desc->super.handler = (uct_rc_send_handler_t)ucs_mpool_put;
    length = pack_cb(desc + 1, arg);

    uct_rc_mlx5_ep_bcopy_post(ep, MLX5_OPCODE_RDMA_WRITE, length, 0, NULL, 0,
                              remote_addr, rkey, MLX5_WQE_CTRL_CQ_UPDATE, desc);
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, BCOPY, length);
    return length;
}

ucs_status_t uct_rc_mlx5_ep_put_zcopy(uct_ep_h tl_ep, const void *buffer, size_t length,
                                      uct_mem_h memh, uint64_t remote_addr,
                                      uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    ucs_status_t status;

    UCT_CHECK_LENGTH(length, UCT_IB_MAX_MESSAGE_SIZE, "put_zcopy");
    status = uct_rc_mlx5_ep_zcopy_post(ep, MLX5_OPCODE_RDMA_WRITE, buffer, length,
                                       memh, 0, NULL, 0, remote_addr, rkey,
                                       MLX5_WQE_CTRL_CQ_UPDATE, comp);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, PUT, ZCOPY, length);
    return status;
}

ucs_status_t uct_rc_mlx5_ep_get_bcopy(uct_ep_h tl_ep,
                                      uct_unpack_callback_t unpack_cb,
                                      void *arg, size_t length,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_mlx5_iface_t);
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_CHECK_LENGTH(length, iface->super.super.config.seg_size, "get_bcopy");
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, &iface->super.tx.mp, desc);

    desc->super.handler     = (comp == NULL) ?
                                uct_rc_ep_get_bcopy_handler_no_completion :
                                uct_rc_ep_get_bcopy_handler;
    desc->super.unpack_arg  = arg;
    desc->super.user_comp   = comp;
    desc->super.length      = length;
    desc->unpack_cb         = unpack_cb;

    uct_rc_mlx5_ep_bcopy_post(ep, MLX5_OPCODE_RDMA_READ, length, 0, NULL, 0,
                              remote_addr, rkey, MLX5_WQE_CTRL_CQ_UPDATE, desc);
    UCT_TL_EP_STAT_OP(&ep->super.super, GET, BCOPY, length);
    return UCS_INPROGRESS;
}

ucs_status_t uct_rc_mlx5_ep_get_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                      uct_mem_h memh, uint64_t remote_addr,
                                      uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    ucs_status_t status;

    UCT_CHECK_LENGTH(length, UCT_IB_MAX_MESSAGE_SIZE, "get_zcopy");
    status = uct_rc_mlx5_ep_zcopy_post(ep, MLX5_OPCODE_RDMA_READ, buffer, length,
                                       memh, 0, NULL, 0, remote_addr, rkey,
                                       MLX5_WQE_CTRL_CQ_UPDATE, comp);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, GET, ZCOPY, length);
    return status;
}

ucs_status_t uct_rc_mlx5_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                     const void *payload, unsigned length)
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

ssize_t uct_rc_mlx5_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                uct_pack_callback_t pack_cb, void *arg)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_mlx5_iface_t);
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;
    uct_rc_hdr_t *rch;
    size_t length;

    UCT_CHECK_AM_ID(id);
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, &iface->super.tx.mp, desc);

    desc->super.handler = (uct_rc_send_handler_t)ucs_mpool_put;

    rch        = (void*)(desc + 1);
    rch->am_id = id;
    length = pack_cb(rch + 1, arg);

    uct_rc_mlx5_ep_bcopy_post(ep, MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_RAW,
                              sizeof(*rch) + length, 0, NULL, 0, 0, 0, 0, desc);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, BCOPY, length);
    return length;
}

ucs_status_t uct_rc_mlx5_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                     unsigned header_length, const void *payload,
                                     size_t length, uct_mem_h memh,
                                     uct_completion_t *comp)
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    ucs_status_t status;

    UCT_CHECK_AM_ID(id);

    UCT_CHECK_LENGTH(sizeof(struct mlx5_wqe_ctrl_seg) +
                     sizeof(struct mlx5_wqe_data_seg) +
                     sizeof(struct mlx5_wqe_inl_data_seg) +
                     sizeof(uct_rc_hdr_t) + header_length,
                     UCT_RC_MLX5_MAX_BB * MLX5_SEND_WQE_BB,
                     "am zcopy");
    UCT_CHECK_LENGTH(header_length + length + sizeof(uct_rc_hdr_t),
                     ucs_derived_of(tl_ep->iface, uct_ib_iface_t)->config.seg_size,
                     "am_zcopy");
    UCT_CHECK_LENGTH(header_length + length, UCT_IB_MAX_MESSAGE_SIZE, "am_zcopy");

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
                                          uint64_t *result, uct_completion_t *comp)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_FA, result, sizeof(uint64_t),
                                 uct_rc_ep_atomic_handler_64_be1, remote_addr,
                                 rkey, 0, 0, htonll(add), comp);
}

ucs_status_t uct_rc_mlx5_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uint64_t *result, uct_completion_t *comp)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_CS,result,  sizeof(uint64_t),
                                 uct_rc_ep_atomic_handler_64_be1, remote_addr,
                                 rkey, 0, 0, htonll(swap), comp);
}

ucs_status_t uct_rc_mlx5_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint64_t *result, uct_completion_t *comp)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_CS, result, sizeof(uint64_t),
                                 uct_rc_ep_atomic_handler_64_be1, remote_addr,
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
                                          uint32_t *result, uct_completion_t *comp)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_FA, result, sizeof(uint32_t),
                                 uct_rc_ep_atomic_handler_32_be1, remote_addr,
                                 rkey, 0, 0, htonl(add), comp);
}

ucs_status_t uct_rc_mlx5_ep_atomic_swap32(uct_ep_h tl_ep, uint32_t swap,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uint32_t *result, uct_completion_t *comp)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_CS, result, sizeof(uint32_t),
                                 uct_rc_ep_atomic_handler_32_be1, remote_addr,
                                 rkey, 0, 0, htonl(swap), comp);
}

ucs_status_t uct_rc_mlx5_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint32_t *result, uct_completion_t *comp)
{
    return uct_rc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_CS, result, sizeof(uint32_t),
                                 uct_rc_ep_atomic_handler_32_be1, remote_addr,
                                 rkey, (uint32_t)-1, htonl(compare), htonl(swap),
                                 comp);
}

ucs_status_t uct_rc_mlx5_ep_flush(uct_ep_h tl_ep)
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    ucs_status_t status;

    if (ep->super.available == ep->tx.wq.bb_max) {
        UCT_TL_EP_STAT_FLUSH(&ep->super.super);
        ucs_trace_data("ep %p is flushed", ep);
        return UCS_OK;
    }

    if (ep->super.unsignaled != 0) {
        status = uct_rc_mlx5_ep_inline_post(ep, MLX5_OPCODE_NOP, NULL, 0, 0, 0, 0, 0);
        if (status != UCS_OK) {
            return status;
        }
    }

    UCT_TL_EP_STAT_FLUSH_WAIT(&ep->super.super);
    return UCS_INPROGRESS;
}

static UCS_CLASS_INIT_FUNC(uct_rc_mlx5_ep_t, uct_iface_h tl_iface)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_mlx5_iface_t);
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_ep_t, &iface->super);

    status = uct_ib_mlx5_get_txwq(iface->super.super.super.worker, self->super.qp,
                                  &self->tx.wq);
    if (status != UCS_OK) {
        ucs_error("Failed to get mlx5 QP information");
        return status;
    }

    self->super.available = self->tx.wq.bb_max;
    self->qp_num          = self->super.qp->qp_num;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_ep_t)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(self->super.super.super.iface,
                                                uct_rc_mlx5_iface_t);
    uct_ib_mlx5_put_txwq(iface->super.super.super.worker, &self->tx.wq);
}

UCS_CLASS_DEFINE(uct_rc_mlx5_ep_t, uct_rc_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_mlx5_ep_t, uct_ep_t, uct_iface_h);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_mlx5_ep_t, uct_ep_t);

