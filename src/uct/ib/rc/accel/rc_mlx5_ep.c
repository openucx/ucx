/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rc_mlx5.h"
#if HAVE_DECL_IBV_CMD_MODIFY_QP
#  include <infiniband/driver.h>
#endif

#include <uct/ib/mlx5/ib_mlx5_log.h>
#include <ucs/vfs/base/vfs_cb.h>
#include <ucs/vfs/base/vfs_obj.h>
#include <ucs/arch/cpu.h>
#include <ucs/sys/compiler.h>
#include <ucs/type/serialize.h>
#include <arpa/inet.h> /* For htonl */

#include "rc_mlx5.inl"


/*
 * Helper function for zero-copy post.
 * Adds user completion to the callback queue.
 */
static UCS_F_ALWAYS_INLINE ucs_status_t uct_rc_mlx5_base_ep_zcopy_post(
        uct_rc_mlx5_base_ep_t *ep, unsigned opcode, const uct_iov_t *iov,
        size_t iovcnt, size_t iov_total_length,
        /* SEND */ uint8_t am_id, const void *am_hdr, unsigned am_hdr_len,
        /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey,
        /* TAG  */ uct_tag_t tag, uint32_t app_ctx, uint32_t ib_imm_be,
        int force_sig, uct_rc_send_handler_t handler, uint16_t op_flags,
        uct_completion_t *comp)
{
    uct_rc_mlx5_iface_common_t *iface  = ucs_derived_of(ep->super.super.super.iface,
                                                        uct_rc_mlx5_iface_common_t);
    uint16_t sn;

    sn = ep->tx.wq.sw_pi;
    uct_rc_mlx5_txqp_dptr_post_iov(iface, IBV_QPT_RC,
                                   &ep->super.txqp, &ep->tx.wq,
                                   opcode, iov, iovcnt,
                                   am_id, am_hdr, am_hdr_len,
                                   rdma_raddr, uct_ib_md_direct_rkey(rdma_rkey),
                                   tag, app_ctx, ib_imm_be, 0,
                                   (comp == NULL) ? force_sig : MLX5_WQE_CTRL_CQ_UPDATE,
                                   0,
                                   UCT_IB_MAX_ZCOPY_LOG_SGE(&iface->super.super));

    uct_rc_txqp_add_send_comp(&iface->super, &ep->super.txqp, handler, comp, sn,
                              op_flags | UCT_RC_IFACE_SEND_OP_FLAG_ZCOPY,
                              iov, iovcnt, iov_total_length);

    return UCS_INPROGRESS;
}

static ucs_status_t UCS_F_ALWAYS_INLINE uct_rc_mlx5_base_ep_put_short_inline(
        uct_ep_h tl_ep, const void *buffer, unsigned length,
        uint64_t remote_addr, uct_rkey_t rkey)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    UCT_RC_MLX5_CHECK_PUT_SHORT(length, 0);
    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    uct_rc_mlx5_ep_fence_put(iface, &ep->tx.wq, &rkey, &remote_addr,
                             ep->super.atomic_mr_offset);
    uct_rc_mlx5_txqp_inline_post(iface, IBV_QPT_RC,
                                 &ep->super.txqp, &ep->tx.wq,
                                 MLX5_OPCODE_RDMA_WRITE,
                                 buffer, length, 0, 0, 0, remote_addr, rkey,
                                 0, 0, 0, INT_MAX);
    uct_rc_ep_enable_flush_remote(&ep->super);
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, SHORT, length);
    return UCS_OK;
}

static ucs_status_t UCS_F_ALWAYS_INLINE
uct_rc_mlx5_base_ep_am_short_inline(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                    const void *payload, unsigned length)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    UCT_RC_MLX5_CHECK_AM_SHORT(id, uct_rc_mlx5_am_short_hdr_t, length, 0);
    UCT_RC_CHECK_RES_AND_FC(&iface->super, &ep->super, id);

    uct_rc_mlx5_txqp_inline_post(iface, IBV_QPT_RC,
                                 &ep->super.txqp, &ep->tx.wq,
                                 MLX5_OPCODE_SEND,
                                 payload, length,
                                 id, hdr, 0,
                                 0, 0,
                                 0,
                                 MLX5_WQE_CTRL_SOLICITED,
                                 0, INT_MAX);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, sizeof(hdr) + length);
    UCT_RC_UPDATE_FC(&ep->super, id);
    return UCS_OK;
}

static ucs_status_t UCS_F_ALWAYS_INLINE uct_rc_mlx5_base_ep_am_short_iov_inline(
        uct_ep_h tl_ep, uint8_t id, const uct_iov_t *iov, size_t iovcnt,
        size_t iov_length)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    UCT_RC_MLX5_CHECK_AM_SHORT(id, uct_rc_mlx5_hdr_t, iov_length, 0);
    UCT_RC_CHECK_RES_AND_FC(&iface->super, &ep->super, id);
    uct_rc_mlx5_txqp_inline_iov_post(iface, IBV_QPT_RC, &ep->super.txqp,
                                     &ep->tx.wq, iov, iovcnt, iov_length, id,
                                     0, 0);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, iov_length);
    UCT_RC_UPDATE_FC(&ep->super, id);

    return UCS_OK;
}

ucs_status_t uct_rc_mlx5_base_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                           unsigned length,
                                           uint64_t remote_addr,
                                           uct_rkey_t rkey)
{
#if HAVE_IBV_DM
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    uct_rc_iface_t *rc_iface = &iface->super;
    ucs_status_t status;

    if (ucs_likely((length <= UCT_IB_MLX5_PUT_MAX_SHORT(0)) || !iface->dm.dm)) {
#endif
        return uct_rc_mlx5_base_ep_put_short_inline(tl_ep, buffer, length,
                                                    remote_addr, rkey);
#if HAVE_IBV_DM
    }

    UCT_CHECK_LENGTH(length, 0, iface->dm.seg_len, "put_short");
    UCT_RC_CHECK_RES(rc_iface, &ep->super);
    uct_rc_mlx5_ep_fence_put(iface, &ep->tx.wq, &rkey, &remote_addr,
                             ep->super.atomic_mr_offset);
    status = uct_rc_mlx5_common_ep_short_dm(iface, IBV_QPT_RC, NULL, 0, buffer,
                                            length, MLX5_OPCODE_RDMA_WRITE,
                                            MLX5_WQE_CTRL_CQ_UPDATE, 0,
                                            remote_addr, rkey, &ep->super.txqp,
                                            &ep->tx.wq, 0);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    uct_rc_ep_enable_flush_remote(&ep->super);
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, SHORT, length);
    return UCS_OK;
#endif
}

ssize_t uct_rc_mlx5_base_ep_put_bcopy(uct_ep_h tl_ep,
                                      uct_pack_callback_t pack_cb, void *arg,
                                      uint64_t remote_addr, uct_rkey_t rkey)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    uct_rc_iface_send_desc_t *desc;
    size_t length;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_PUT_BCOPY_DESC(&iface->super, &iface->super.tx.mp,
                                       desc, pack_cb, arg, length);
    uct_rc_mlx5_ep_fence_put(iface, &ep->tx.wq, &rkey, &remote_addr,
                             ep->super.atomic_mr_offset);
    uct_rc_mlx5_common_txqp_bcopy_post(iface, IBV_QPT_RC, &ep->super.txqp,
                                       &ep->tx.wq, MLX5_OPCODE_RDMA_WRITE,
                                       length, remote_addr, rkey, 0,
                                       MLX5_WQE_CTRL_CQ_UPDATE, 0, 0, desc,
                                       desc + 1, NULL);
    uct_rc_ep_enable_flush_remote(&ep->super);
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, BCOPY, length);

    return length;
}

ucs_status_t uct_rc_mlx5_base_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                           size_t iovcnt, uint64_t remote_addr,
                                           uct_rkey_t rkey,
                                           uct_completion_t *comp)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt, UCT_RC_MLX5_RMA_MAX_IOV(0),
                       "uct_rc_mlx5_ep_put_zcopy");
    UCT_CHECK_LENGTH(uct_iov_total_length(iov, iovcnt), 0, UCT_IB_MAX_MESSAGE_SIZE,
                     "put_zcopy");
    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    uct_rc_mlx5_ep_fence_put(iface, &ep->tx.wq, &rkey, &remote_addr,
                             ep->super.atomic_mr_offset);

    status = uct_rc_mlx5_base_ep_zcopy_post(
            ep, MLX5_OPCODE_RDMA_WRITE, iov, iovcnt, 0ul, 0, NULL, 0,
            remote_addr, rkey, 0ul, 0, 0, MLX5_WQE_CTRL_CQ_UPDATE,
            uct_rc_ep_send_op_completion_handler, 0, comp);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, PUT, ZCOPY,
                                 uct_iov_total_length(iov, iovcnt));
    uct_rc_ep_enable_flush_remote(&ep->super);
    return status;
}

ucs_status_t
uct_rc_mlx5_base_ep_get_bcopy(uct_ep_h tl_ep, uct_unpack_callback_t unpack_cb,
                              void *arg, size_t length, uint64_t remote_addr,
                              uct_rkey_t rkey, uct_completion_t *comp)
{
    uint8_t fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    uct_rc_iface_send_desc_t *desc;

    UCT_CHECK_LENGTH(length, 0, iface->super.super.config.seg_size, "get_bcopy");
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_GET_BCOPY_DESC(&iface->super, &iface->super.tx.mp, desc,
                                       unpack_cb, comp, arg, length);

    uct_rc_mlx5_ep_fence_get(iface, &ep->tx.wq, &rkey, &fm_ce_se);
    uct_rc_mlx5_common_txqp_bcopy_post(iface, IBV_QPT_RC, &ep->super.txqp,
                                       &ep->tx.wq, MLX5_OPCODE_RDMA_READ,
                                       length, remote_addr, rkey, 0, fm_ce_se,
                                       0, 0, desc, desc + 1, NULL);
    UCT_TL_EP_STAT_OP(&ep->super.super, GET, BCOPY, length);
    UCT_RC_RDMA_READ_POSTED(&iface->super, length);

    return UCS_INPROGRESS;
}

ucs_status_t uct_rc_mlx5_base_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                           size_t iovcnt, uint64_t remote_addr,
                                           uct_rkey_t rkey,
                                           uct_completion_t *comp)
{
    uint8_t fm_ce_se    = MLX5_WQE_CTRL_CQ_UPDATE;
    size_t total_length = uct_iov_total_length(iov, iovcnt);
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt, UCT_RC_MLX5_RMA_MAX_IOV(0),
                       "uct_rc_mlx5_ep_get_zcopy");
    UCT_CHECK_LENGTH(total_length,
                     iface->super.super.config.max_inl_cqe[UCT_IB_DIR_TX] + 1,
                     iface->super.config.max_get_zcopy, "get_zcopy");
    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    uct_rc_mlx5_ep_fence_get(iface, &ep->tx.wq, &rkey, &fm_ce_se);
    status = uct_rc_mlx5_base_ep_zcopy_post(
            ep, MLX5_OPCODE_RDMA_READ, iov, iovcnt, total_length, 0, NULL, 0,
            remote_addr, rkey, 0ul, 0, 0, fm_ce_se,
            uct_rc_ep_get_zcopy_completion_handler,
            UCT_RC_IFACE_SEND_OP_FLAG_IOV, comp);
    if (!UCS_STATUS_IS_ERR(status)) {
        UCT_TL_EP_STAT_OP(&ep->super.super, GET, ZCOPY, total_length);
        UCT_RC_RDMA_READ_POSTED(&iface->super, total_length);
    }
    return status;
}

ucs_status_t uct_rc_mlx5_base_ep_am_short(uct_ep_h tl_ep, uint8_t id,
                                          uint64_t hdr, const void *payload,
                                          unsigned length)
{
#if HAVE_IBV_DM
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    ucs_status_t status;
    uct_rc_mlx5_dm_copy_data_t cache;

    if (ucs_likely((sizeof(uct_rc_mlx5_am_short_hdr_t) + length <= UCT_IB_MLX5_AM_MAX_SHORT(0)) ||
                   !iface->dm.dm)) {
#endif
        return uct_rc_mlx5_base_ep_am_short_inline(tl_ep, id, hdr, payload,
                                                   length);
#if HAVE_IBV_DM
    }

    UCT_CHECK_LENGTH(length + sizeof(uct_rc_mlx5_am_short_hdr_t), 0,
                     iface->dm.seg_len, "am_short");
    UCT_CHECK_AM_ID(id);
    UCT_RC_CHECK_RES_AND_FC(&iface->super, &ep->super, id);

    uct_rc_mlx5_am_hdr_fill(&cache.am_hdr.rc_hdr, id);
    cache.am_hdr.am_hdr = hdr;

    status = uct_rc_mlx5_common_ep_short_dm(
            iface, IBV_QPT_RC, &cache, sizeof(cache.am_hdr), payload, length,
            MLX5_OPCODE_SEND, MLX5_WQE_CTRL_SOLICITED | MLX5_WQE_CTRL_CQ_UPDATE,
            0, 0, 0, &ep->super.txqp, &ep->tx.wq, 0);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, sizeof(cache.am_hdr) + length);
    UCT_RC_UPDATE_FC(&ep->super, id);
    return UCS_OK;
#endif
}

ucs_status_t uct_rc_mlx5_base_ep_am_short_iov(uct_ep_h tl_ep, uint8_t id,
                                              const uct_iov_t *iov,
                                              size_t iovcnt)
{
    size_t iov_length = uct_iov_total_length(iov, iovcnt);
#if HAVE_IBV_DM
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    ucs_status_t status;

    if (ucs_likely((sizeof(uct_rc_mlx5_hdr_t) + iov_length <=
                    UCT_IB_MLX5_AM_MAX_SHORT(0)) ||
                   !iface->dm.dm)) {
#endif
        return uct_rc_mlx5_base_ep_am_short_iov_inline(tl_ep, id, iov, iovcnt,
                                                       iov_length);
#if HAVE_IBV_DM
    }

    UCT_CHECK_AM_ID(id);
    UCT_RC_CHECK_RES_AND_FC(&iface->super, &ep->super, id);

    status = uct_rc_mlx5_common_ep_am_short_iov_dm(&ep->super.super, id, iface,
                                                   iov, iovcnt, iov_length,
                                                   IBV_QPT_RC, &ep->super.txqp,
                                                   &ep->tx.wq, 0, 0);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        return status;
    }

    UCT_RC_UPDATE_FC(&ep->super, id);

    return UCS_OK;
#endif
}

ssize_t uct_rc_mlx5_base_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                     uct_pack_callback_t pack_cb, void *arg,
                                     unsigned flags)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    uct_rc_iface_send_desc_t *desc;
    size_t length;

    UCT_CHECK_AM_ID(id);
    UCT_RC_CHECK_RES_AND_FC(&iface->super, &ep->super, id);
    UCT_RC_IFACE_GET_TX_AM_BCOPY_DESC(&iface->super, &iface->super.tx.mp, desc,
                                      id, uct_rc_mlx5_am_hdr_fill, uct_rc_mlx5_hdr_t,
                                      pack_cb, arg, &length);

    uct_rc_mlx5_common_txqp_bcopy_post(iface, IBV_QPT_RC, &ep->super.txqp,
                                       &ep->tx.wq, MLX5_OPCODE_SEND,
                                       sizeof(uct_rc_mlx5_hdr_t) + length, 0, 0,
                                       0, MLX5_WQE_CTRL_SOLICITED, 0, 0, desc,
                                       desc + 1, NULL);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, BCOPY, length);
    UCT_RC_UPDATE_FC(&ep->super, id);

    return length;
}

ucs_status_t
uct_rc_mlx5_base_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                             unsigned header_length, const uct_iov_t *iov,
                             size_t iovcnt, unsigned flags,
                             uct_completion_t *comp)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt, UCT_IB_MLX5_AM_ZCOPY_MAX_IOV,
                       "uct_rc_mlx5_ep_am_zcopy");
    UCT_RC_MLX5_CHECK_AM_ZCOPY(id, header_length, uct_iov_total_length(iov, iovcnt),
                               iface->super.super.config.seg_size, 0);
    UCT_RC_CHECK_RES_AND_FC(&iface->super, &ep->super, id);

    status = uct_rc_mlx5_base_ep_zcopy_post(
            ep, MLX5_OPCODE_SEND, iov, iovcnt, 0ul, id, header, header_length,
            0, 0, 0ul, 0, 0, MLX5_WQE_CTRL_SOLICITED,
            uct_rc_ep_send_op_completion_handler, 0, comp);
    if (ucs_likely(status >= 0)) {
        UCT_TL_EP_STAT_OP(&ep->super.super, AM, ZCOPY,
                          header_length + uct_iov_total_length(iov, iovcnt));
        UCT_RC_UPDATE_FC(&ep->super, id);
    }
    return status;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_base_ep_atomic_post(uct_ep_h tl_ep, unsigned opcode,
                                uct_rc_iface_send_desc_t *desc, unsigned length,
                                uint64_t remote_addr, uct_rkey_t rkey,
                                uint64_t compare_mask, uint64_t compare,
                                uint64_t swap_mask, uint64_t swap_add)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    uint32_t ib_rkey = uct_ib_resolve_atomic_rkey(rkey,
                                                  ep->super.atomic_mr_offset,
                                                  &remote_addr);

    desc->super.sn = ep->tx.wq.sw_pi;
    uct_rc_mlx5_txqp_dptr_post(iface, IBV_QPT_RC,
                               &ep->super.txqp, &ep->tx.wq,
                               opcode, desc + 1, length, &desc->lkey,
                               remote_addr, ib_rkey,
                               compare_mask, compare, swap_mask, swap_add,
                               0, MLX5_WQE_CTRL_CQ_UPDATE, 0, 0, INT_MAX, NULL);

    uct_rc_ep_enable_flush_remote(&ep->super);
    UCT_TL_EP_STAT_ATOMIC(&ep->super.super);
    uct_rc_txqp_add_send_op(&ep->super.txqp, &desc->super);
}

static UCS_F_ALWAYS_INLINE ucs_status_t uct_rc_mlx5_base_ep_atomic_fop(
        uct_ep_h tl_ep, int opcode, void *result, int ext, unsigned length,
        uint64_t remote_addr, uct_rkey_t rkey, uint64_t compare_mask,
        uint64_t compare, uint64_t swap_mask, uint64_t swap_add,
        uct_completion_t *comp)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    uct_rc_iface_send_desc_t *desc;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_ATOMIC_FETCH_DESC(&iface->super,
                                          &iface->tx.atomic_desc_mp, desc,
                                          uct_rc_iface_atomic_handler(&iface->super,
                                                                      ext, length),
                                          result, comp);
    uct_rc_mlx5_base_ep_atomic_post(tl_ep, opcode, desc, length, remote_addr,
                                    rkey, compare_mask, compare, swap_mask,
                                    swap_add);
    return UCS_INPROGRESS;
}

static ucs_status_t UCS_F_ALWAYS_INLINE uct_rc_mlx5_base_ep_atomic_op_post(
        uct_ep_h tl_ep, unsigned opcode, unsigned size, uint64_t value,
        uint64_t remote_addr, uct_rkey_t rkey)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    uct_rc_iface_send_desc_t *desc;
    int op;
    uint64_t compare_mask;
    uint64_t compare;
    uint64_t swap_mask;
    uint64_t swap;
    int      ext; /* not used here */
    ucs_status_t status;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_MLX5_CHECK_ATOMIC_OPS(opcode, size, UCT_RC_MLX5_ATOMIC_OPS);

    status = uct_rc_mlx5_iface_common_atomic_data(opcode, size, value, &op,
                                                  &compare_mask, &compare,
                                                  &swap_mask, &swap, &ext);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        return status;
    }

    UCT_RC_IFACE_GET_TX_ATOMIC_DESC(&iface->super, &iface->tx.atomic_desc_mp,
                                    desc);

    uct_rc_mlx5_base_ep_atomic_post(tl_ep, op, desc, size, remote_addr, rkey,
                                    compare_mask, compare, swap_mask, swap);
    uct_rc_ep_enable_flush_remote(&ep->super);
    return UCS_OK;
}

static ucs_status_t UCS_F_ALWAYS_INLINE uct_rc_mlx5_base_ep_atomic_fop_post(
        uct_ep_h tl_ep, unsigned opcode, unsigned size, uint64_t value,
        void *result, uint64_t remote_addr, uct_rkey_t rkey,
        uct_completion_t *comp)
{
    int op;
    uint64_t compare_mask;
    uint64_t compare;
    uint64_t swap_mask;
    uint64_t swap;
    int      ext;
    ucs_status_t status;

    UCT_RC_MLX5_CHECK_ATOMIC_OPS(opcode, size, UCT_RC_MLX5_ATOMIC_FOPS);

    status = uct_rc_mlx5_iface_common_atomic_data(opcode, size, value, &op, &compare_mask,
                                                  &compare, &swap_mask, &swap, &ext);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        return status;
    }

    return uct_rc_mlx5_base_ep_atomic_fop(tl_ep, op, result, ext, size,
                                          remote_addr, rkey, compare_mask,
                                          compare, swap_mask, swap, comp);
}

ucs_status_t
uct_rc_mlx5_base_ep_atomic32_post(uct_ep_h ep, unsigned opcode, uint32_t value,
                                  uint64_t remote_addr, uct_rkey_t rkey)
{
    return uct_rc_mlx5_base_ep_atomic_op_post(ep, opcode, sizeof(value), value,
                                              remote_addr, rkey);
}

ucs_status_t
uct_rc_mlx5_base_ep_atomic64_post(uct_ep_h ep, unsigned opcode, uint64_t value,
                                  uint64_t remote_addr, uct_rkey_t rkey)
{
    return uct_rc_mlx5_base_ep_atomic_op_post(ep, opcode, sizeof(value), value,
                                              remote_addr, rkey);
}

ucs_status_t
uct_rc_mlx5_base_ep_atomic64_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                   uint64_t value, uint64_t *result,
                                   uint64_t remote_addr, uct_rkey_t rkey,
                                   uct_completion_t *comp)
{
    return uct_rc_mlx5_base_ep_atomic_fop_post(ep, opcode, sizeof(value), value,
                                               result, remote_addr, rkey, comp);
}

ucs_status_t
uct_rc_mlx5_base_ep_atomic32_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                   uint32_t value, uint32_t *result,
                                   uint64_t remote_addr, uct_rkey_t rkey,
                                   uct_completion_t *comp)
{
    return uct_rc_mlx5_base_ep_atomic_fop_post(ep, opcode, sizeof(value), value,
                                               result, remote_addr, rkey, comp);
}

ucs_status_t
uct_rc_mlx5_base_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare,
                                   uint64_t swap, uint64_t remote_addr,
                                   uct_rkey_t rkey, uint64_t *result,
                                   uct_completion_t *comp)
{
    return uct_rc_mlx5_base_ep_atomic_fop(tl_ep, MLX5_OPCODE_ATOMIC_CS, result,
                                          0, sizeof(uint64_t), remote_addr,
                                          rkey, 0, htobe64(compare), UINT64_MAX,
                                          htobe64(swap), comp);
}

ucs_status_t
uct_rc_mlx5_base_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare,
                                   uint32_t swap, uint64_t remote_addr,
                                   uct_rkey_t rkey, uint32_t *result,
                                   uct_completion_t *comp)
{
    return uct_rc_mlx5_base_ep_atomic_fop(tl_ep, MLX5_OPCODE_ATOMIC_MASKED_CS,
                                          result, 1, sizeof(uint32_t),
                                          remote_addr, rkey, UCS_MASK(32),
                                          htonl(compare), UINT64_MAX,
                                          htonl(swap), comp);
}

ucs_status_t uct_rc_mlx5_base_ep_fence(uct_ep_h tl_ep, unsigned flags)
{
    uct_rc_mlx5_base_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_base_ep_t);

    return uct_rc_ep_fence(tl_ep, &ep->tx.wq.fi, 1);
}

void uct_rc_mlx5_base_ep_post_check(uct_ep_h tl_ep)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    uint64_t dummy = 0; /* Dummy buffer to suppress compiler warning */

    uct_rc_mlx5_txqp_inline_post(iface, IBV_QPT_RC,
                                 &ep->super.txqp, &ep->tx.wq,
                                 MLX5_OPCODE_RDMA_WRITE, &dummy, 0,
                                 0, 0, 0,
                                 0, 0,
                                 0, 0,
                                 0, INT_MAX);
}

void uct_rc_mlx5_base_ep_vfs_populate(uct_rc_ep_t *rc_ep)
{
    uct_rc_iface_t *rc_iface  = ucs_derived_of(rc_ep->super.super.iface,
                                               uct_rc_iface_t);
    uct_rc_mlx5_base_ep_t *ep = ucs_derived_of(rc_ep, uct_rc_mlx5_base_ep_t);

    ucs_vfs_obj_add_dir(rc_iface, ep, "ep/%p", ep);
    uct_ib_mlx5_txwq_vfs_populate(&ep->tx.wq, ep);
    uct_rc_txqp_vfs_populate(&ep->super.txqp, ep);
}

static ucs_status_t
uct_rc_mlx5_base_ep_flush_remote(uct_ep_h tl_ep, uct_completion_t *comp)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    uct_rc_iface_send_desc_t *desc;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    UCT_RC_IFACE_GET_TX_DESC(iface, &iface->super.tx.mp, desc);
    desc->super.handler   = uct_rc_ep_flush_remote_handler;
    desc->super.user_comp = comp;

    uct_rc_mlx5_common_txqp_bcopy_post(iface, IBV_QPT_RC, &ep->super.txqp,
                                       &ep->tx.wq, MLX5_OPCODE_RDMA_READ,
                                       UCT_IB_MD_FLUSH_REMOTE_LENGTH, 0,
                                       ep->super.flush_rkey, 0,
                                       MLX5_WQE_CTRL_CQ_UPDATE, 0, 0, desc,
                                       desc + 1, NULL);
    ep->super.flags &= ~UCT_RC_EP_FLAG_FLUSH_REMOTE;

    return UCS_INPROGRESS;
}

ucs_status_t uct_rc_mlx5_base_ep_flush(uct_ep_h tl_ep, unsigned flags,
                                       uct_completion_t *comp)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    int already_canceled = ep->super.flags & UCT_RC_EP_FLAG_FLUSH_CANCEL;
    ucs_status_t status;

    UCT_CHECK_PARAM(!ucs_test_all_flags(flags, UCT_FLUSH_FLAG_CANCEL |
                                               UCT_FLUSH_FLAG_REMOTE),
                    "flush flags CANCEL and REMOTE are mutually exclusive");

    if (flags & UCT_FLUSH_FLAG_REMOTE) {
        UCT_RC_IFACE_CHECK_FLUSH_REMOTE(
                uct_ib_md_is_flush_rkey_valid(ep->super.flush_rkey), ep,
                &iface->super, rcx);
        if (ep->super.flags & UCT_RC_EP_FLAG_FLUSH_REMOTE) {
            return uct_rc_mlx5_base_ep_flush_remote(tl_ep, comp);
        }
    }

    status = uct_rc_ep_flush(&ep->super, ep->tx.wq.bb_max, flags);
    if (status != UCS_INPROGRESS) {
        return status;
    }

    if (uct_rc_txqp_unsignaled(&ep->super.txqp) != 0) {
        UCT_RC_CHECK_RES(&iface->super, &ep->super);
        uct_rc_mlx5_txqp_inline_post(iface, IBV_QPT_RC,
                                     &ep->super.txqp, &ep->tx.wq,
                                     MLX5_OPCODE_NOP, NULL, 0,
                                     0, 0, 0,
                                     0, 0,
                                     0, 0,
                                     0, INT_MAX);
    }

    if (ucs_unlikely((flags & UCT_FLUSH_FLAG_CANCEL) && !already_canceled)) {
        status = uct_ib_mlx5_modify_qp_state(&iface->super.super,
                                             &ep->tx.wq.super, IBV_QPS_ERR);
        if (status != UCS_OK) {
            return status;
        }

        uct_ib_mlx5_txwq_update_flags(&ep->tx.wq, UCT_IB_MLX5_TXWQ_FLAG_FAILED,
                                      0);
    }

    return uct_rc_txqp_add_flush_comp(&iface->super, &ep->super.super,
                                      &ep->super.txqp, comp, ep->tx.wq.sig_pi);
}

ucs_status_t uct_rc_mlx5_base_ep_invalidate(uct_ep_h tl_ep, unsigned flags)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);

    return uct_ib_mlx5_modify_qp_state(&iface->super.super, &ep->tx.wq.super,
                                       IBV_QPS_ERR);
}

ucs_status_t uct_rc_mlx5_base_ep_fc_ctrl(uct_ep_t *tl_ep, unsigned op,
                                         uct_rc_pending_req_t *req)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);

    /* In RC only PURE grant is sent as a separate message. Other FC
     * messages are bundled with AM. */
    ucs_assert(op == UCT_RC_EP_FC_PURE_GRANT);

    UCT_RC_CHECK_TX_CQ_RES(&iface->super, &ep->super);
    uct_rc_mlx5_txqp_inline_post(iface, IBV_QPT_RC,
                                 &ep->super.txqp, &ep->tx.wq,
                                 MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_RAW,
                                 NULL, 0,
                                 UCT_RC_EP_FC_PURE_GRANT, 0, 0,
                                 0, 0,
                                 0, 0,
                                 0, INT_MAX);
    return UCS_OK;
}

ucs_status_t uct_rc_mlx5_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr)
{
    UCT_RC_MLX5_EP_DECL(tl_ep, iface, ep);
    uct_rc_mlx5_ep_address_t *rc_addr       = (uct_rc_mlx5_ep_address_t*)addr;
    uct_ib_md_t *md                         = uct_ib_iface_md(ucs_derived_of(
                                              tl_ep->iface, uct_ib_iface_t));
    uct_rc_mlx5_ep_ext_address_t *ext_addr;
    void *ptr;

    uct_ib_pack_uint24(rc_addr->qp_num, ep->super.tx.wq.super.qp_num);
    rc_addr->atomic_mr_id = uct_ib_md_get_atomic_mr_id(md);

    if (UCT_RC_MLX5_TM_ENABLED(iface)) {
        uct_ib_pack_uint24(rc_addr->tm_qp_num, ep->tm_qp.qp_num);
    }

    if (uct_rc_iface_flush_rkey_enabled(&iface->super)) {
        ext_addr                            = ucs_derived_of(rc_addr,
                                                             uct_rc_mlx5_ep_ext_address_t);
        ext_addr->flags                     = UCT_RC_MLX5_EP_ADDR_FLAG_FLUSH_RKEY;
        ptr                                 = ext_addr + 1;
        *ucs_serialize_next(&ptr, uint16_t) = md->flush_rkey >> 16;
    }

    return UCS_OK;
}

void uct_rc_mlx5_common_packet_dump(uct_base_iface_t *iface, uct_am_trace_type_t type,
                                    void *data, size_t length, size_t valid_length,
                                    char *buffer, size_t max)
{
    uct_rc_mlx5_hdr_t *rch = data;

#if IBV_HW_TM
    if (rch->tmh_opcode != IBV_TMH_NO_TAG) {
        struct ibv_tmh *tmh = ucs_unaligned_ptr(rch);
        struct ibv_rvh *rvh = (void*)(tmh + 1);
        uct_tag_t tag;
        uint32_t app_ctx;

        tag     = tmh->tag;
        app_ctx = tmh->app_ctx;

        switch (rch->tmh_opcode) {
        case IBV_TMH_EAGER:
            snprintf(buffer, max, " EAGER tag %lx app_ctx %d", tag, app_ctx);
            return;
        case IBV_TMH_RNDV:
            snprintf(buffer, max, " RNDV tag %lx app_ctx %d va 0x%lx len %d rkey %x",
                     tag, app_ctx, be64toh(rvh->va), ntohl(rvh->len), ntohl(rvh->rkey));
            return;
        case IBV_TMH_FIN:
            snprintf(buffer, max, " FIN tag %lx app_ctx %d", tag, app_ctx);
            return;
        default:
            break;
        }
    }
#endif

    data = &rch->rc_hdr;
    /* coverity[overrun-buffer-val] */
    uct_rc_ep_packet_dump(iface, type, data, length - UCS_PTR_BYTE_DIFF(rch, data),
                          valid_length, buffer, max);
}

ucs_status_t
uct_rc_mlx5_ep_connect_qp(uct_rc_mlx5_iface_common_t *iface,
                          uct_ib_mlx5_qp_t *qp, uint32_t qp_num,
                          struct ibv_ah_attr *ah_attr, enum ibv_mtu path_mtu,
                          uint8_t path_index)
{
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.super.super.md, uct_ib_mlx5_md_t);

    ucs_assert(path_mtu != UCT_IB_ADDRESS_INVALID_PATH_MTU);
    if (md->flags & UCT_IB_MLX5_MD_FLAG_DEVX) {
        return uct_rc_mlx5_iface_common_devx_connect_qp(
                iface, qp, qp_num, ah_attr, path_mtu, path_index,
                iface->super.config.max_rd_atomic);
    } else {
        return uct_rc_iface_qp_connect(&iface->super, qp->verbs.qp, qp_num,
                                       ah_attr, path_mtu);
    }
}

int uct_rc_mlx5_base_ep_is_connected(const uct_ep_h tl_ep,
                                     const uct_ep_is_connected_params_t *params)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    uint32_t addr_qp = 0;
    uct_rc_mlx5_ep_address_t *rc_addr;
    ucs_status_t status;
    struct ibv_ah_attr ah_attr;
    uint32_t qp_num;

    status = uct_ib_mlx5_query_qp_peer_info(&iface->super.super,
                                            &ep->tx.wq.super, &ah_attr,
                                            &qp_num);
    if (status != UCS_OK) {
        return 0;
    }

    if (params->field_mask & UCT_EP_IS_CONNECTED_FIELD_EP_ADDR) {
        rc_addr       = (uct_rc_mlx5_ep_address_t*)params->ep_addr;
        addr_qp       = uct_ib_unpack_uint24(rc_addr->qp_num);
    }

    return uct_rc_ep_is_connected(&ah_attr, params, qp_num, addr_qp);
}

ucs_status_t
uct_rc_mlx5_ep_connect_to_ep_v2(uct_ep_h tl_ep,
                                const uct_device_addr_t *device_addr,
                                const uct_ep_addr_t *ep_addr,
                                const uct_ep_connect_to_ep_params_t *params)
{
    UCT_RC_MLX5_EP_DECL(tl_ep, iface, ep);
    const uct_ib_address_t *ib_addr         = (const uct_ib_address_t *)device_addr;
    const uct_rc_mlx5_ep_address_t *rc_addr = (const uct_rc_mlx5_ep_address_t*)ep_addr;
    const uct_rc_mlx5_ep_ext_address_t *ext_addr;
    size_t addr_length;
    uint32_t qp_num;
    struct ibv_ah_attr ah_attr;
    enum ibv_mtu path_mtu;
    const void *ptr;
    uint32_t flush_rkey_hi;
    ucs_status_t status;

    uct_ib_iface_fill_ah_attr_from_addr(&iface->super.super, ib_addr,
                                        ep->super.super.path_index, &ah_attr,
                                        &path_mtu);
    ucs_assert(path_mtu != UCT_IB_ADDRESS_INVALID_PATH_MTU);

    if (UCT_RC_MLX5_TM_ENABLED(iface)) {
        /* For HW TM we need 2 QPs, one of which will be used by the device for
         * RNDV offload (for issuing RDMA reads and sending RNDV ACK). No WQEs
         * should be posted to the send side of the QP which is owned by device. */
        status = uct_rc_mlx5_ep_connect_qp(
                iface, &ep->tm_qp, uct_ib_unpack_uint24(rc_addr->qp_num),
                &ah_attr, path_mtu, ep->super.super.path_index);
        if (status != UCS_OK) {
            return status;
        }

        /* Need to connect local ep QP to the one owned by device
         * (and bound to XRQ) on the peer. */
        qp_num = uct_ib_unpack_uint24(rc_addr->tm_qp_num);
    } else {
        qp_num = uct_ib_unpack_uint24(rc_addr->qp_num);
    }

    status = uct_rc_mlx5_ep_connect_qp(iface, &ep->super.tx.wq.super, qp_num,
                                       &ah_attr, path_mtu,
                                       ep->super.super.path_index);
    if (status != UCS_OK) {
        return status;
    }

    ep->super.super.atomic_mr_offset = uct_ib_md_atomic_offset(
            rc_addr->atomic_mr_id);
    ep->super.super.flags           |= UCT_RC_EP_FLAG_CONNECTED;

    addr_length = UCS_PARAM_VALUE(UCT_EP_CONNECT_TO_EP_PARAM_FIELD, params,
                                  ep_addr_length, EP_ADDR_LENGTH,
                                  sizeof(uct_rc_mlx5_ep_address_t));
    if (addr_length <= sizeof(uct_rc_mlx5_ep_address_t)) {
        ep->super.super.flush_rkey = UCT_IB_MD_INVALID_FLUSH_RKEY;
        return UCS_OK;
    }

    ext_addr = ucs_derived_of(rc_addr, uct_rc_mlx5_ep_ext_address_t);
    if (ext_addr->flags & UCT_RC_MLX5_EP_ADDR_FLAG_FLUSH_RKEY) {
        ptr                        = ext_addr + 1;
        flush_rkey_hi              = *ucs_serialize_next(&ptr, uint16_t);
        ep->super.super.flush_rkey = (flush_rkey_hi << 16) |
                                     ((uint32_t)rc_addr->atomic_mr_id << 8);
    } else {
        ep->super.super.flush_rkey = UCT_IB_MD_INVALID_FLUSH_RKEY;
    }

    return UCS_OK;
}

#if IBV_HW_TM

ucs_status_t uct_rc_mlx5_ep_tag_rndv_cancel(uct_ep_h tl_ep, void *op)
{
    uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(tl_ep->iface,
                                                       uct_rc_mlx5_iface_common_t);

    uint32_t op_index = (uint32_t)((uint64_t)op);
    ucs_ptr_array_remove(&iface->tm.rndv_comps, op_index);
    return UCS_OK;
}

static ucs_status_t UCS_F_ALWAYS_INLINE
uct_rc_mlx5_ep_tag_eager_short_inline(uct_ep_h tl_ep, uct_tag_t tag,
                                      const void *data, size_t length)
{
    UCT_RC_MLX5_EP_DECL(tl_ep, iface, ep);
    UCT_CHECK_LENGTH(length + sizeof(struct ibv_tmh), 0,
                     UCT_IB_MLX5_AM_MAX_SHORT(0), "tag_short");
    UCT_RC_CHECK_RES(&iface->super, &ep->super.super);

    uct_rc_mlx5_txqp_tag_inline_post(iface, IBV_QPT_RC, &ep->super.super.txqp,
                                     &ep->super.tx.wq, MLX5_OPCODE_SEND, data,
                                     length, NULL, tag, 0, IBV_TMH_EAGER, 0, 0,
                                     NULL, 0, MLX5_WQE_CTRL_SOLICITED, 0);

    UCT_TL_EP_STAT_OP(&ep->super.super.super, TAG, SHORT, length);

    return UCS_OK;
}

ucs_status_t uct_rc_mlx5_ep_tag_eager_short(uct_ep_h tl_ep, uct_tag_t tag,
                                            const void *data, size_t length)
{
#if HAVE_IBV_DM
    UCT_RC_MLX5_EP_DECL(tl_ep, iface, ep);
    uct_rc_mlx5_dm_copy_data_t cache;
    ucs_status_t status;

    if (ucs_likely((sizeof(struct ibv_tmh) + length <= UCT_IB_MLX5_AM_MAX_SHORT(0)) ||
                   !iface->dm.dm)) {
#endif
        return uct_rc_mlx5_ep_tag_eager_short_inline(tl_ep, tag, data, length);
#if HAVE_IBV_DM
    }

    UCT_CHECK_LENGTH(length + sizeof(struct ibv_tmh), 0,
                     iface->dm.seg_len, "tag_short");
    UCT_RC_CHECK_RES(&iface->super, &ep->super.super);

    uct_rc_mlx5_fill_tmh(ucs_unaligned_ptr(&cache.tm_hdr), tag, 0, IBV_TMH_EAGER);

    status = uct_rc_mlx5_common_ep_short_dm(
            iface, IBV_QPT_RC, &cache, sizeof(cache.tm_hdr), data, length,
            MLX5_OPCODE_SEND, MLX5_WQE_CTRL_SOLICITED | MLX5_WQE_CTRL_CQ_UPDATE,
            0, 0, 0, &ep->super.super.txqp, &ep->super.tx.wq, 0);
    if (!UCS_STATUS_IS_ERR(status)) {
        UCT_TL_EP_STAT_OP(&ep->super.super.super, TAG, SHORT, length);
    }

    return status;
#endif
}

ssize_t uct_rc_mlx5_ep_tag_eager_bcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                       uint64_t imm,
                                       uct_pack_callback_t pack_cb,
                                       void *arg, unsigned flags)
{
    UCT_RC_MLX5_EP_DECL(tl_ep, iface, ep);
    uct_rc_iface_send_desc_t *desc;
    uint32_t app_ctx, ib_imm;
    int opcode;
    size_t length;

    UCT_RC_CHECK_RES(&iface->super, &ep->super.super);

    UCT_RC_MLX5_FILL_TM_IMM(imm, app_ctx, ib_imm, opcode, MLX5_OPCODE_SEND,
                             _IMM);

    UCT_RC_MLX5_IFACE_GET_TM_BCOPY_DESC(&iface->super, iface->tm.bcopy_mp,
                                        desc, tag, app_ctx, pack_cb, arg, length);

    uct_rc_mlx5_common_txqp_bcopy_post(iface, IBV_QPT_RC, &ep->super.super.txqp,
                                       &ep->super.tx.wq, opcode,
                                       sizeof(struct ibv_tmh) + length, 0, 0, 0,
                                       MLX5_WQE_CTRL_SOLICITED, 0, ib_imm, desc,
                                       desc + 1, NULL);

    UCT_TL_EP_STAT_OP(&ep->super.super.super, TAG, BCOPY, length);

    return length;
}

ucs_status_t uct_rc_mlx5_ep_tag_eager_zcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                            uint64_t imm, const uct_iov_t *iov,
                                            size_t iovcnt, unsigned flags,
                                            uct_completion_t *comp)
{
    UCT_RC_MLX5_EP_DECL(tl_ep, iface, ep);
    uint32_t app_ctx, ib_imm;
    int opcode;

    UCT_CHECK_IOV_SIZE(iovcnt, UCT_RC_MLX5_TM_EAGER_ZCOPY_MAX_IOV(0),
                       "uct_rc_mlx5_ep_tag_eager_zcopy");
    UCT_RC_CHECK_ZCOPY_DATA(sizeof(struct ibv_tmh),
                            uct_iov_total_length(iov, iovcnt),
                            iface->tm.max_zcopy);
    UCT_RC_CHECK_RES(&iface->super, &ep->super.super);

    UCT_RC_MLX5_FILL_TM_IMM(imm, app_ctx, ib_imm, opcode, MLX5_OPCODE_SEND,
                             _IMM);

    UCT_TL_EP_STAT_OP(&ep->super.super.super, TAG, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));

    return uct_rc_mlx5_base_ep_zcopy_post(
            &ep->super, opcode | UCT_RC_MLX5_OPCODE_FLAG_TM, iov, iovcnt, 0ul,
            0, "", 0, 0, 0, tag, app_ctx, ib_imm, MLX5_WQE_CTRL_SOLICITED,
            uct_rc_ep_send_op_completion_handler, 0, comp);
}

ucs_status_ptr_t uct_rc_mlx5_ep_tag_rndv_zcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                               const void *header,
                                               unsigned header_length,
                                               const uct_iov_t *iov,
                                               size_t iovcnt, unsigned flags,
                                               uct_completion_t *comp)
{
    UCT_RC_MLX5_EP_DECL(tl_ep, iface, ep);
    unsigned tm_hdr_len   = sizeof(struct ibv_tmh) +
                            sizeof(struct ibv_rvh);
    uint32_t op_index;

    UCT_RC_MLX5_CHECK_RNDV_PARAMS(iovcnt, header_length, tm_hdr_len,
                                   UCT_IB_MLX5_AM_MAX_SHORT(0),
                                   iface->tm.max_rndv_data +
                                   UCT_RC_MLX5_TMH_PRIV_LEN);
    UCT_RC_MLX5_CHECK_RES_PTR(iface, &ep->super);

    op_index = uct_rc_mlx5_tag_get_op_id(iface, comp);

    uct_rc_mlx5_txqp_tag_inline_post(iface, IBV_QPT_RC, &ep->super.super.txqp,
                                     &ep->super.tx.wq, MLX5_OPCODE_SEND, header,
                                     header_length, iov, tag, op_index,
                                     IBV_TMH_RNDV, 0, 0, NULL, 0,
                                     MLX5_WQE_CTRL_SOLICITED, 0);

    return (ucs_status_ptr_t)((uint64_t)op_index);
}

ucs_status_t uct_rc_mlx5_ep_tag_rndv_request(uct_ep_h tl_ep, uct_tag_t tag,
                                             const void* header,
                                             unsigned header_length,
                                             unsigned flags)
{
    UCT_RC_MLX5_EP_DECL(tl_ep, iface, ep);
    UCT_CHECK_LENGTH(header_length + sizeof(struct ibv_tmh), 0,
                     UCT_IB_MLX5_AM_MAX_SHORT(0), "tag_rndv_request");
    UCT_RC_CHECK_RES(&iface->super, &ep->super.super);

    uct_rc_mlx5_txqp_tag_inline_post(iface, IBV_QPT_RC, &ep->super.super.txqp,
                                     &ep->super.tx.wq, MLX5_OPCODE_SEND_IMM,
                                     header, header_length, NULL, tag, 0,
                                     IBV_TMH_EAGER, 0, 0, NULL, 0,
                                     MLX5_WQE_CTRL_SOLICITED, 0);
    return UCS_OK;
}
#endif /* IBV_HW_TM */

static UCS_CLASS_INIT_FUNC(uct_rc_mlx5_base_ep_t, const uct_ep_params_t *params)
{
    uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(params->iface,
                                                       uct_rc_mlx5_iface_common_t);
    uct_ib_mlx5_md_t *md              = ucs_derived_of(iface->super.super.super.md,
                                                       uct_ib_mlx5_md_t);
    uct_ib_mlx5_qp_attr_t attr        = {};
    ucs_status_t status;

    /* Need to create QP before super constructor to get QP number */
    uct_rc_mlx5_iface_fill_attr(iface, &attr, iface->super.config.tx_qp_len,
                                &iface->rx.srq);
    status = uct_rc_mlx5_iface_create_qp(iface, &self->tx.wq.super,
                                         &self->tx.wq, &attr);
    if (status != UCS_OK) {
        return status;
    }

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_ep_t, &iface->super,
                              self->tx.wq.super.qp_num, params);

    if (self->tx.wq.super.type == UCT_IB_MLX5_OBJ_TYPE_VERBS) {
        status = uct_rc_iface_qp_init(&iface->super,
                                      self->tx.wq.super.verbs.qp);
        if (status != UCS_OK) {
            goto err_destroy_txwq_qp;
        }
    }

    status = uct_ib_device_async_event_register(&md->super.dev,
                                                IBV_EVENT_QP_LAST_WQE_REACHED,
                                                self->tx.wq.super.qp_num);
    if (status != UCS_OK) {
        goto err_destroy_txwq_qp;
    }

    status = uct_rc_iface_add_qp(&iface->super, &self->super,
                                 self->tx.wq.super.qp_num);
    if (status != UCS_OK) {
        goto err_event_unreg;
    }

    self->tx.wq.bb_max = ucs_min(self->tx.wq.bb_max, iface->tx.bb_max);
    uct_rc_txqp_available_set(&self->super.txqp, self->tx.wq.bb_max);
    uct_rc_mlx5_iface_common_prepost_recvs(iface);
    return UCS_OK;

err_event_unreg:
    uct_ib_device_async_event_unregister(&md->super.dev,
                                         IBV_EVENT_QP_LAST_WQE_REACHED,
                                         self->tx.wq.super.qp_num);
err_destroy_txwq_qp:
    uct_ib_mlx5_destroy_qp(md, &self->tx.wq.super);
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_base_ep_t)
{
    /* No op, cleanup context is implemented in derived class */
}


UCS_CLASS_DEFINE(uct_rc_mlx5_base_ep_t, uct_rc_ep_t);

UCS_CLASS_INIT_FUNC(uct_rc_mlx5_ep_t, const uct_ep_params_t *params)
{
    uct_rc_mlx5_iface_common_t *iface =
            ucs_derived_of(params->iface, uct_rc_mlx5_iface_common_t);
    uct_ib_mlx5_md_t *md       = ucs_derived_of(iface->super.super.super.md,
                                                uct_ib_mlx5_md_t);
    uct_ib_mlx5_qp_attr_t attr = {};
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_mlx5_base_ep_t, params);

    if (UCT_RC_MLX5_TM_ENABLED(iface)) {
        /* Send queue of this QP will be used by FW for HW RNDV. Driver requires
         * such a QP to be initialized with zero send queue length. */
        memset(&attr, 0, sizeof(attr));
        uct_rc_mlx5_iface_fill_attr(iface, &attr, 0, &iface->rx.srq);
        status = uct_rc_mlx5_iface_create_qp(iface, &self->tm_qp, NULL, &attr);
        if (status != UCS_OK) {
            goto err;
        }

        status = uct_rc_iface_add_qp(&iface->super, &self->super.super,
                                     self->tm_qp.qp_num);
        if (status != UCS_OK) {
            goto err_destroy_tm_qp;
        }
    }

    self->mp.free = 1;
    return UCS_OK;

err_destroy_tm_qp:
    uct_ib_mlx5_destroy_qp(md, &self->tm_qp);
err:
    return status;
}

UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_ep_t)
{
    uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(
            self->super.super.super.super.iface, uct_rc_mlx5_iface_common_t);
    uct_rc_mlx5_iface_qp_cleanup_ctx_t *cleanup_ctx;
    uint16_t outstanding, wqe_count;

    cleanup_ctx = ucs_malloc(sizeof(*cleanup_ctx), "mlx5_qp_cleanup_ctx");
    ucs_assert_always(cleanup_ctx != NULL);
    cleanup_ctx->qp    = self->super.tx.wq.super;
    cleanup_ctx->tm_qp = self->tm_qp;
    cleanup_ctx->reg   = self->super.tx.wq.reg;

    uct_rc_txqp_purge_outstanding(&iface->super, &self->super.super.txqp,
                                  UCS_ERR_CANCELED, self->super.tx.wq.sw_pi, 1);
#if IBV_HW_TM
    if (UCT_RC_MLX5_TM_ENABLED(iface)) {
        uct_rc_iface_remove_qp(&iface->super, self->tm_qp.qp_num);
    }
#endif

    ucs_assert(self->mp.free == 1);
    (void)uct_ib_mlx5_modify_qp_state(&iface->super.super,
                                      &self->super.tx.wq.super, IBV_QPS_ERR);

    /* Keep only one unreleased CQ credit per WQE, so we will not have CQ
       overflow. These CQ credits will be released by error CQE handler. */
    outstanding = self->super.tx.wq.bb_max - self->super.super.txqp.available;
    wqe_count   = uct_ib_mlx5_txwq_num_posted_wqes(&self->super.tx.wq,
                                                   outstanding);
    ucs_assert(outstanding >= wqe_count);
    uct_rc_ep_cleanup_qp(&self->super.super, &cleanup_ctx->super,
                         self->super.tx.wq.super.qp_num,
                         outstanding - wqe_count);
}

UCS_CLASS_DEFINE(uct_rc_mlx5_ep_t, uct_rc_mlx5_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_mlx5_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_mlx5_ep_t, uct_ep_t);
