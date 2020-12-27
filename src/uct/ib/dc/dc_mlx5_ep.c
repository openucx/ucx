/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "dc_mlx5.inl"
#include "dc_mlx5_ep.h"
#include "dc_mlx5.h"

#include <uct/ib/mlx5/ib_mlx5_log.h>

#define UCT_DC_MLX5_IFACE_TXQP_GET(_iface, _ep, _txqp, _txwq) \
    UCT_DC_MLX5_IFACE_TXQP_DCI_GET(_iface, (_ep)->dci, _txqp, _txwq)

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_bcopy_post(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep,
                            unsigned opcode, unsigned length,
                            /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey,
                            uct_rc_iface_send_desc_t *desc, uint8_t send_flags,
                            uint32_t imm_val_be, const void *buffer,
                            uct_ib_log_sge_t *log_sge)
{
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);
    desc->super.sn = txwq->sw_pi;
    uct_rc_mlx5_txqp_dptr_post(&iface->super, UCT_IB_QPT_DCI, txqp, txwq,
                               opcode, buffer, length, &desc->lkey,
                               rdma_raddr, rdma_rkey, 0, 0, 0, 0,
                               &ep->av, uct_dc_mlx5_ep_get_grh(ep),
                               uct_ib_mlx5_wqe_av_size(&ep->av),
                               MLX5_WQE_CTRL_CQ_UPDATE | send_flags, imm_val_be, INT_MAX,
                               log_sge);
    uct_rc_txqp_add_send_op(txqp, &desc->super);
}


static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_zcopy_post(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep,
                             unsigned opcode, const uct_iov_t *iov,
                             size_t iovcnt, size_t iov_total_length,
                             /* SEND */ uint8_t am_id, const void *am_hdr, unsigned am_hdr_len,
                             /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey,
                             /* TAG  */ uct_tag_t tag, uint32_t app_ctx, uint32_t ib_imm_be,
                             uct_rc_send_handler_t handler, uint16_t op_flags,
                             uct_completion_t *comp, uint8_t send_flags)
{
    uint16_t sn;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);

    sn = txwq->sw_pi;
    uct_rc_mlx5_txqp_dptr_post_iov(&iface->super, UCT_IB_QPT_DCI, txqp,
                                   txwq, opcode, iov, iovcnt,
                                   am_id, am_hdr, am_hdr_len,
                                   rdma_raddr, rdma_rkey,
                                   tag, app_ctx, ib_imm_be,
                                   &ep->av, uct_dc_mlx5_ep_get_grh(ep),
                                   uct_ib_mlx5_wqe_av_size(&ep->av),
                                   MLX5_WQE_CTRL_CQ_UPDATE | send_flags,
                                   UCT_IB_MAX_ZCOPY_LOG_SGE(&iface->super.super.super));

    uct_rc_txqp_add_send_comp(&iface->super.super, txqp, handler, comp, sn,
                              op_flags | UCT_RC_IFACE_SEND_OP_FLAG_ZCOPY, iov,
                              iovcnt, iov_total_length);
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_atomic_post(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep,
                              unsigned opcode, uct_rc_iface_send_desc_t *desc, unsigned length,
                              uint64_t remote_addr, uct_rkey_t rkey,
                              uint64_t compare_mask, uint64_t compare,
                              uint64_t swap_mask, uint64_t swap_add)
{
    uint32_t ib_rkey = uct_ib_resolve_atomic_rkey(rkey, ep->atomic_mr_offset,
                                                  &remote_addr);

    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);
    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);

    desc->super.sn = txwq->sw_pi;
    uct_rc_mlx5_txqp_dptr_post(&iface->super, UCT_IB_QPT_DCI, txqp, txwq,
                               opcode, desc + 1, length, &desc->lkey,
                               remote_addr, ib_rkey,
                               compare_mask, compare, swap_mask, swap_add,
                               &ep->av, uct_dc_mlx5_ep_get_grh(ep),
                               uct_ib_mlx5_wqe_av_size(&ep->av),
                               MLX5_WQE_CTRL_CQ_UPDATE, 0, INT_MAX, NULL);

    UCT_TL_EP_STAT_ATOMIC(&ep->super);
    uct_rc_txqp_add_send_op(txqp, &desc->super);
}

static ucs_status_t UCS_F_ALWAYS_INLINE
uct_dc_mlx5_ep_atomic_op_post(uct_ep_h tl_ep, unsigned opcode, unsigned size,
                              uint64_t value, uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;
    int op;
    uint64_t compare_mask;
    uint64_t compare;
    uint64_t swap_mask;
    uint64_t swap;
    int      ext; /* not used here */
    ucs_status_t status;

    UCT_DC_MLX5_CHECK_RES(iface, ep);
    UCT_RC_MLX5_CHECK_ATOMIC_OPS(opcode, size, UCT_RC_MLX5_ATOMIC_OPS);

    status = uct_rc_mlx5_iface_common_atomic_data(opcode, size, value, &op, &compare_mask,
                                                  &compare, &swap_mask, &swap, &ext);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        return status;
    }

    UCT_RC_IFACE_GET_TX_ATOMIC_DESC(&iface->super.super, &iface->super.tx.atomic_desc_mp, desc);
    uct_dc_mlx5_iface_atomic_post(iface, ep, op, desc, size, remote_addr, rkey,
                                  compare_mask, compare, swap_mask, swap);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_dc_mlx5_ep_atomic_fop(uct_dc_mlx5_ep_t *ep, int opcode, void *result, int ext,
                          unsigned length, uint64_t remote_addr, uct_rkey_t rkey,
                          uint64_t compare_mask, uint64_t compare,
                          uint64_t swap_mask, uint64_t swap_add, uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_dc_mlx5_iface_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_DC_MLX5_CHECK_RES(iface, ep);
    UCT_RC_IFACE_GET_TX_ATOMIC_FETCH_DESC(&iface->super.super, &iface->super.tx.atomic_desc_mp,
                                          desc, uct_rc_iface_atomic_handler(&iface->super.super,
                                                                            ext, length),
                                          result, comp);
    uct_dc_mlx5_iface_atomic_post(iface, ep, opcode, desc, length, remote_addr, rkey,
                                  compare_mask, compare, swap_mask, swap_add);
    return UCS_INPROGRESS;
}

static ucs_status_t UCS_F_ALWAYS_INLINE
uct_dc_mlx5_ep_atomic_fop_post(uct_ep_h tl_ep, unsigned opcode, unsigned size,
                               uint64_t value, void *result,
                               uint64_t remote_addr, uct_rkey_t rkey,
                               uct_completion_t *comp)
{
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
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

    return uct_dc_mlx5_ep_atomic_fop(ep, op, result, ext, size, remote_addr, rkey,
                                     compare_mask, compare, swap_mask, swap, comp);
}

ucs_status_t uct_dc_mlx5_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint64_t *result, uct_completion_t *comp)
{
    return uct_dc_mlx5_ep_atomic_fop(ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t),
                                     MLX5_OPCODE_ATOMIC_CS, result, 0, sizeof(uint64_t),
                                     remote_addr, rkey, 0, htobe64(compare), UINT64_MAX,
                                     htobe64(swap), comp);
}

ucs_status_t uct_dc_mlx5_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint32_t *result, uct_completion_t *comp)
{
    return uct_dc_mlx5_ep_atomic_fop(ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t),
                                     MLX5_OPCODE_ATOMIC_MASKED_CS, result, 1,
                                     sizeof(uint32_t), remote_addr, rkey, UCS_MASK(32),
                                     htonl(compare), UINT64_MAX, htonl(swap), comp);
}

ucs_status_t uct_dc_mlx5_ep_atomic32_post(uct_ep_h ep, unsigned opcode, uint32_t value,
                                          uint64_t remote_addr, uct_rkey_t rkey)
{
    return uct_dc_mlx5_ep_atomic_op_post(ep, opcode, sizeof(value), value, remote_addr, rkey);
}

ucs_status_t uct_dc_mlx5_ep_atomic64_post(uct_ep_h ep, unsigned opcode, uint64_t value,
                                          uint64_t remote_addr, uct_rkey_t rkey)
{
    return uct_dc_mlx5_ep_atomic_op_post(ep, opcode, sizeof(value), value, remote_addr, rkey);
}

ucs_status_t uct_dc_mlx5_ep_atomic64_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                           uint64_t value, uint64_t *result,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_completion_t *comp)
{
    return uct_dc_mlx5_ep_atomic_fop_post(ep, opcode, sizeof(value), value, result,
                                          remote_addr, rkey, comp);
}

ucs_status_t uct_dc_mlx5_ep_atomic32_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                           uint32_t value, uint32_t *result,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_completion_t *comp)
{
    return uct_dc_mlx5_ep_atomic_fop_post(ep, opcode, sizeof(value), value, result,
                                          remote_addr, rkey, comp);
}

ucs_status_t uct_dc_mlx5_ep_fence(uct_ep_h tl_ep, unsigned flags)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);

    return uct_rc_ep_fence(tl_ep, &iface->tx.dcis[ep->dci].txwq.fi,
                           ep->dci != UCT_DC_MLX5_EP_NO_DCI);
}

static ucs_status_t UCS_F_ALWAYS_INLINE
uct_dc_mlx5_ep_am_short_inline(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                               const void *buffer, unsigned length)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_RC_MLX5_CHECK_AM_SHORT(id, length, UCT_IB_MLX5_AV_FULL_SIZE);
    UCT_DC_CHECK_RES_AND_FC(iface, ep);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);

    uct_rc_mlx5_txqp_inline_post(&iface->super, UCT_IB_QPT_DCI,
                                 txqp, txwq,
                                 MLX5_OPCODE_SEND,
                                 buffer, length, id, hdr, 0,
                                 0, 0,
                                 &ep->av, uct_dc_mlx5_ep_get_grh(ep),
                                 uct_ib_mlx5_wqe_av_size(&ep->av),
                                 MLX5_WQE_CTRL_SOLICITED, INT_MAX);

    UCT_RC_UPDATE_FC_WND(&iface->super.super, &ep->fc);
    UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, sizeof(hdr) + length);
    return UCS_OK;
}

#if HAVE_IBV_DM
static ucs_status_t UCS_F_ALWAYS_INLINE
uct_dc_mlx5_ep_short_dm(uct_dc_mlx5_ep_t *ep, uct_rc_mlx5_dm_copy_data_t *cache,
                        size_t hdr_len, const void *payload, unsigned length,
                        unsigned opcode, uint8_t fm_ce_se,
                        uint64_t rdma_raddr, uct_rkey_t rdma_rkey)
{
    uct_dc_mlx5_iface_t *iface     = ucs_derived_of(ep->super.super.iface, uct_dc_mlx5_iface_t);
    uct_rc_iface_send_desc_t *desc = NULL;
    void *buffer;
    ucs_status_t status;
    uct_ib_log_sge_t log_sge;

    status = uct_rc_mlx5_common_dm_make_data(&iface->super, cache, hdr_len,
                                             payload, length, &desc,
                                             &buffer, &log_sge);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        return status;
    }

    uct_dc_mlx5_iface_bcopy_post(iface, ep, opcode,
                                 hdr_len + length,
                                 rdma_raddr, rdma_rkey,
                                 desc, fm_ce_se, 0, buffer,
                                 log_sge.num_sge ? &log_sge : NULL);
    return UCS_OK;
}
#endif

ucs_status_t uct_dc_mlx5_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                     const void *buffer, unsigned length)
{
#if HAVE_IBV_DM
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    ucs_status_t status;
    uct_rc_mlx5_dm_copy_data_t cache;

    if (ucs_likely((sizeof(uct_rc_mlx5_am_short_hdr_t) + length <=
                    UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE)) ||
                   !iface->super.dm.dm)) {
#endif
        return uct_dc_mlx5_ep_am_short_inline(tl_ep, id, hdr, buffer, length);
#if HAVE_IBV_DM
    }

    UCT_CHECK_AM_ID(id);
    UCT_CHECK_LENGTH(length + sizeof(uct_rc_mlx5_am_short_hdr_t), 0,
                     iface->super.dm.seg_len, "am_short");
    UCT_DC_CHECK_RES_AND_FC(iface, ep);

    uct_rc_mlx5_am_hdr_fill(&cache.am_hdr.rc_hdr, id);
    cache.am_hdr.am_hdr = hdr;

    status = uct_dc_mlx5_ep_short_dm(ep, &cache, sizeof(cache.am_hdr), buffer, length,
                                     MLX5_OPCODE_SEND,
                                     MLX5_WQE_CTRL_SOLICITED | MLX5_WQE_CTRL_CQ_UPDATE,
                                     0, 0);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }
    UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, sizeof(cache.am_hdr) + length);
    UCT_RC_UPDATE_FC_WND(&iface->super.super, &ep->fc);
    return UCS_OK;
#endif
}

ssize_t uct_dc_mlx5_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                uct_pack_callback_t pack_cb, void *arg,
                                unsigned flags)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;
    size_t length;

    UCT_CHECK_AM_ID(id);
    UCT_DC_CHECK_RES_AND_FC(iface, ep);
    UCT_RC_IFACE_GET_TX_AM_BCOPY_DESC(&iface->super.super, &iface->super.super.tx.mp, desc,
                                      id, uct_rc_mlx5_am_hdr_fill, uct_rc_mlx5_hdr_t,
                                      pack_cb, arg, &length);

    uct_dc_mlx5_iface_bcopy_post(iface, ep, MLX5_OPCODE_SEND,
                                 sizeof(uct_rc_mlx5_hdr_t) + length, 0, 0, desc,
                                 MLX5_WQE_CTRL_SOLICITED, 0, desc + 1, NULL);

    UCT_RC_UPDATE_FC_WND(&iface->super.super, &ep->fc);
    UCT_TL_EP_STAT_OP(&ep->super, AM, BCOPY, length);
    return length;
}

ucs_status_t uct_dc_mlx5_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                     unsigned header_length, const uct_iov_t *iov,
                                     size_t iovcnt, unsigned flags,
                                     uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);

    UCT_CHECK_IOV_SIZE(iovcnt, UCT_IB_MLX5_AM_ZCOPY_MAX_IOV,
                       "uct_dc_mlx5_ep_am_zcopy");
    UCT_RC_MLX5_CHECK_AM_ZCOPY(id, header_length, uct_iov_total_length(iov, iovcnt),
                               iface->super.super.super.config.seg_size,
                               UCT_IB_MLX5_AV_FULL_SIZE);
    UCT_DC_CHECK_RES_AND_FC(iface, ep);

    uct_dc_mlx5_iface_zcopy_post(iface, ep, MLX5_OPCODE_SEND, iov, iovcnt, 0ul,
                                 id, header, header_length, 0, 0, 0ul, 0, 0,
                                 uct_rc_ep_send_op_completion_handler, 0,
                                 comp, MLX5_WQE_CTRL_SOLICITED);

    UCT_RC_UPDATE_FC_WND(&iface->super.super, &ep->fc);
    UCT_TL_EP_STAT_OP(&ep->super, AM, ZCOPY, header_length +
                      uct_iov_total_length(iov, iovcnt));

    return UCS_INPROGRESS;
}

static ucs_status_t UCS_F_ALWAYS_INLINE
uct_dc_mlx5_ep_put_short_inline(uct_ep_h tl_ep, const void *buffer,
                                unsigned length, uint64_t remote_addr,
                                uct_rkey_t rkey)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_RC_MLX5_CHECK_PUT_SHORT(length, UCT_IB_MLX5_AV_FULL_SIZE);
    UCT_DC_MLX5_CHECK_RES(iface, ep);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);
    uct_rc_mlx5_ep_fence_put(&iface->super, txwq, &rkey, &remote_addr,
                             ep->atomic_mr_offset);
    uct_rc_mlx5_txqp_inline_post(&iface->super, UCT_IB_QPT_DCI,
                                 txqp, txwq, MLX5_OPCODE_RDMA_WRITE,
                                 buffer, length, 0, 0, 0, remote_addr, rkey,
                                 &ep->av, uct_dc_mlx5_ep_get_grh(ep),
                                 uct_ib_mlx5_wqe_av_size(&ep->av), 0, INT_MAX);

    UCT_TL_EP_STAT_OP(&ep->super, PUT, SHORT, length);

    return UCS_OK;
}

ucs_status_t uct_dc_mlx5_ep_put_short(uct_ep_h tl_ep, const void *payload,
                                      unsigned length, uint64_t remote_addr,
                                      uct_rkey_t rkey)
{
#if HAVE_IBV_DM
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    ucs_status_t status;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    if (ucs_likely((length <= UCT_IB_MLX5_PUT_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE)) ||
                   !iface->super.dm.dm)) {
#endif
        return uct_dc_mlx5_ep_put_short_inline(tl_ep, payload, length, remote_addr, rkey);
#if HAVE_IBV_DM
    }

    UCT_CHECK_LENGTH(length, 0, iface->super.dm.seg_len, "put_short");
    UCT_DC_MLX5_CHECK_RES(iface, ep);
    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);
    uct_rc_mlx5_ep_fence_put(&iface->super, txwq, &rkey, &remote_addr,
                             ep->atomic_mr_offset);
    status = uct_dc_mlx5_ep_short_dm(ep, NULL, 0, payload, length,
                                     MLX5_OPCODE_RDMA_WRITE,
                                     MLX5_WQE_CTRL_CQ_UPDATE,
                                     remote_addr, rkey);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }
    UCT_TL_EP_STAT_OP(&ep->super, PUT, SHORT, length);
    return UCS_OK;
#endif
}

ssize_t uct_dc_mlx5_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                 void *arg, uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;
    size_t length;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_DC_MLX5_CHECK_RES(iface, ep);
    UCT_RC_IFACE_GET_TX_PUT_BCOPY_DESC(&iface->super.super, &iface->super.super.tx.mp,
                                       desc, pack_cb, arg, length);
    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);
    uct_rc_mlx5_ep_fence_put(&iface->super, txwq, &rkey, &remote_addr,
                             ep->atomic_mr_offset);
    uct_dc_mlx5_iface_bcopy_post(iface, ep, MLX5_OPCODE_RDMA_WRITE, length,
                                 remote_addr, rkey, desc, 0, 0, desc + 1, NULL);
    UCT_TL_EP_STAT_OP(&ep->super, PUT, BCOPY, length);
    return length;
}

ucs_status_t uct_dc_mlx5_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_CHECK_IOV_SIZE(iovcnt, UCT_RC_MLX5_RMA_MAX_IOV(UCT_IB_MLX5_AV_FULL_SIZE),
                       "uct_dc_mlx5_ep_put_zcopy");
    UCT_CHECK_LENGTH(uct_iov_total_length(iov, iovcnt), 0, UCT_IB_MAX_MESSAGE_SIZE,
                     "put_zcopy");
    UCT_DC_MLX5_CHECK_RES(iface, ep);
    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);
    uct_rc_mlx5_ep_fence_put(&iface->super, txwq, &rkey, &remote_addr,
                             ep->atomic_mr_offset);

    uct_dc_mlx5_iface_zcopy_post(iface, ep, MLX5_OPCODE_RDMA_WRITE, iov, iovcnt,
                                 0ul, 0, NULL, 0, remote_addr, rkey, 0ul, 0, 0,
                                 uct_rc_ep_send_op_completion_handler, 0, comp,
                                 0);

    UCT_TL_EP_STAT_OP(&ep->super, PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    return UCS_INPROGRESS;
}

ucs_status_t uct_dc_mlx5_ep_get_bcopy(uct_ep_h tl_ep,
                                      uct_unpack_callback_t unpack_cb,
                                      void *arg, size_t length,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uint8_t fm_ce_se           = 0;
    uct_rc_iface_send_desc_t *desc;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_CHECK_LENGTH(length, 0, iface->super.super.super.config.seg_size,
                     "get_bcopy");
    UCT_DC_MLX5_CHECK_RES(iface, ep);
    UCT_RC_IFACE_GET_TX_GET_BCOPY_DESC(&iface->super.super,
                                       &iface->super.super.tx.mp,
                                       desc, unpack_cb, comp, arg, length);
    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);
    uct_rc_mlx5_ep_fence_get(&iface->super, txwq, &rkey, &fm_ce_se);

    uct_dc_mlx5_iface_bcopy_post(iface, ep, MLX5_OPCODE_RDMA_READ, length,
                                 remote_addr, rkey, desc, fm_ce_se, 0,
                                 desc + 1, NULL);

    UCT_RC_RDMA_READ_POSTED(&iface->super.super, length);
    UCT_TL_EP_STAT_OP(&ep->super, GET, BCOPY, length);

    return UCS_INPROGRESS;
}


ucs_status_t uct_dc_mlx5_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                      size_t iovcnt, uint64_t remote_addr,
                                      uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uint8_t fm_ce_se           = 0;
    size_t total_length        = uct_iov_total_length(iov, iovcnt);
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_CHECK_IOV_SIZE(iovcnt, UCT_RC_MLX5_RMA_MAX_IOV(UCT_IB_MLX5_AV_FULL_SIZE),
                       "uct_dc_mlx5_ep_get_zcopy");
    UCT_CHECK_LENGTH(total_length,
                     iface->super.super.super.config.max_inl_cqe[UCT_IB_DIR_TX] + 1,
                     iface->super.super.config.max_get_zcopy, "get_zcopy");
    UCT_DC_MLX5_CHECK_RES(iface, ep);
    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);
    uct_rc_mlx5_ep_fence_get(&iface->super, txwq, &rkey, &fm_ce_se);

    uct_dc_mlx5_iface_zcopy_post(iface, ep, MLX5_OPCODE_RDMA_READ, iov, iovcnt,
                                 total_length, 0, NULL, 0, remote_addr, rkey,
                                 0ul, 0, 0,
                                 uct_rc_ep_get_zcopy_completion_handler,
                                 UCT_RC_IFACE_SEND_OP_FLAG_IOV, comp,
                                 fm_ce_se);

    UCT_RC_RDMA_READ_POSTED(&iface->super.super, total_length);
    UCT_TL_EP_STAT_OP(&ep->super, GET, ZCOPY, total_length);

    return UCS_INPROGRESS;
}

ucs_status_t uct_dc_mlx5_ep_flush(uct_ep_h tl_ep, unsigned flags,
                                  uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t    *ep    = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_ib_mlx5_md_t    *md    = ucs_derived_of(iface->super.super.super.super.md,
                                                uct_ib_mlx5_md_t);
    ucs_status_t        status;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    if (!uct_dc_mlx5_iface_has_tx_resources(iface)) {
        return UCS_ERR_NO_RESOURCE;
    }

    if (ep->dci == UCT_DC_MLX5_EP_NO_DCI) {
        if (!uct_dc_mlx5_iface_dci_can_alloc(iface)) {
            return UCS_ERR_NO_RESOURCE; /* waiting for dci */
        } else {
            UCT_TL_EP_STAT_FLUSH(&ep->super); /* no sends */
            return UCS_OK;
        }
    }

    if (!uct_dc_mlx5_iface_dci_ep_can_send(ep)) {
        return UCS_ERR_NO_RESOURCE; /* cannot send */
    }

    status = uct_dc_mlx5_iface_flush_dci(iface, ep->dci);
    if (status == UCS_OK) {
        UCT_TL_EP_STAT_FLUSH(&ep->super);
        return UCS_OK; /* all sends completed */
    }

    ucs_assert(status == UCS_INPROGRESS);
    ucs_assert(ep->dci != UCT_DC_MLX5_EP_NO_DCI);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);

    if (ucs_unlikely(flags & UCT_FLUSH_FLAG_CANCEL)) {
        ucs_assert(ucs_arbiter_group_is_empty(&ep->arb_group));
        if (uct_dc_mlx5_iface_is_dci_rand(iface)) {
            return UCS_ERR_UNSUPPORTED;
        }

        if (ep->flags & UCT_DC_MLX5_EP_FLAG_FLUSH_CANCEL) {
            goto out;
        }

        status = uct_ib_mlx5_modify_qp_state(md, &txwq->super, IBV_QPS_ERR);
        if (status != UCS_OK) {
            return status;
        }

        ep->flags |= UCT_DC_MLX5_EP_FLAG_FLUSH_CANCEL;
    }

out:
    return uct_rc_txqp_add_flush_comp(&iface->super.super, &ep->super, txqp,
                                      comp, txwq->sig_pi);
}

#if IBV_HW_TM
static ucs_status_t UCS_F_ALWAYS_INLINE
uct_dc_mlx5_ep_tag_eager_short_inline(uct_ep_h tl_ep, uct_tag_t tag,
                                      const void *data, size_t length)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_CHECK_LENGTH(length + sizeof(struct ibv_tmh), 0,
                     UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE),
                     "uct_dc_mlx5_ep_tag_short");
    UCT_DC_MLX5_CHECK_RES(iface, ep);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);

    uct_rc_mlx5_txqp_tag_inline_post(&iface->super, UCT_IB_QPT_DCI,
                                     txqp, txwq, MLX5_OPCODE_SEND, data, length,
                                     NULL, tag, 0, IBV_TMH_EAGER, 0,
                                     &ep->av, uct_dc_mlx5_ep_get_grh(ep),
                                     uct_ib_mlx5_wqe_av_size(&ep->av), NULL, 0,
                                     MLX5_WQE_CTRL_SOLICITED);

    UCT_TL_EP_STAT_OP(&ep->super, TAG, SHORT, length);

    return UCS_OK;
}

ucs_status_t uct_dc_mlx5_ep_tag_eager_short(uct_ep_h tl_ep, uct_tag_t tag,
                                            const void *data, size_t length)
{
#if HAVE_IBV_DM
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_rc_mlx5_dm_copy_data_t cache;
    ucs_status_t status;

    if (ucs_likely((sizeof(struct ibv_tmh) + length <=
                    UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE)) ||
                   !iface->super.dm.dm)) {
#endif
        return uct_dc_mlx5_ep_tag_eager_short_inline(tl_ep, tag, data, length);
#if HAVE_IBV_DM
    }

    UCT_CHECK_LENGTH(length + sizeof(struct ibv_tmh), 0,
                     iface->super.dm.seg_len, "tag_short");
    UCT_DC_MLX5_CHECK_RES(iface, ep);

    uct_rc_mlx5_fill_tmh(ucs_unaligned_ptr(&cache.tm_hdr), tag, 0, IBV_TMH_EAGER);

    status = uct_dc_mlx5_ep_short_dm(ep, &cache, sizeof(cache.tm_hdr), data,
                                     length, MLX5_OPCODE_SEND,
                                     MLX5_WQE_CTRL_SOLICITED | MLX5_WQE_CTRL_CQ_UPDATE,
                                     0, 0);
    if (!UCS_STATUS_IS_ERR(status)) {
        UCT_TL_EP_STAT_OP(&ep->super, TAG, SHORT, length);
    }

    return status;
#endif
}

ssize_t uct_dc_mlx5_ep_tag_eager_bcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                       uint64_t imm,
                                       uct_pack_callback_t pack_cb,
                                       void *arg, unsigned flags)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;
    uint32_t app_ctx, ib_imm;
    int opcode;
    size_t length;

    UCT_DC_MLX5_CHECK_RES(iface, ep);

    UCT_RC_MLX5_FILL_TM_IMM(imm, app_ctx, ib_imm, opcode, MLX5_OPCODE_SEND,
                            _IMM);

    UCT_RC_MLX5_IFACE_GET_TM_BCOPY_DESC(&iface->super.super,
                                        iface->super.tm.bcopy_mp,
                                        desc, tag, app_ctx, pack_cb,
                                        arg, length);

    uct_dc_mlx5_iface_bcopy_post(iface, ep, opcode,
                                 sizeof(struct ibv_tmh) + length,
                                 0, 0, desc, MLX5_WQE_CTRL_SOLICITED, ib_imm,
                                 desc + 1, NULL);

    UCT_TL_EP_STAT_OP(&ep->super, TAG, BCOPY, length);

    return length;
}

ucs_status_t uct_dc_mlx5_ep_tag_eager_zcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                            uint64_t imm, const uct_iov_t *iov,
                                            size_t iovcnt, unsigned flags,
                                            uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uint32_t app_ctx, ib_imm;
    int opcode;

    UCT_CHECK_IOV_SIZE(iovcnt, UCT_RC_MLX5_TM_EAGER_ZCOPY_MAX_IOV(UCT_IB_MLX5_AV_FULL_SIZE),
                       "uct_dc_mlx5_ep_tag_eager_zcopy");
    UCT_RC_CHECK_ZCOPY_DATA(sizeof(struct ibv_tmh),
                            uct_iov_total_length(iov, iovcnt),
                            iface->super.tm.max_zcopy);

    UCT_DC_MLX5_CHECK_RES(iface, ep);

    UCT_RC_MLX5_FILL_TM_IMM(imm, app_ctx, ib_imm, opcode, MLX5_OPCODE_SEND, _IMM);

    uct_dc_mlx5_iface_zcopy_post(iface, ep, opcode|UCT_RC_MLX5_OPCODE_FLAG_TM,
                                 iov, iovcnt, 0ul, 0, "", 0, 0, 0, tag, app_ctx,
                                 ib_imm, uct_rc_ep_send_op_completion_handler,
                                 0, comp, MLX5_WQE_CTRL_SOLICITED);

    UCT_TL_EP_STAT_OP(&ep->super, TAG, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));

    return UCS_INPROGRESS;
}

ucs_status_ptr_t uct_dc_mlx5_ep_tag_rndv_zcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                               const void *header,
                                               unsigned header_length,
                                               const uct_iov_t *iov,
                                               size_t iovcnt, unsigned flags,
                                               uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    unsigned tm_hdr_len        = sizeof(struct ibv_tmh) +
                                 sizeof(struct ibv_rvh) +
                                 sizeof(struct ibv_ravh);
    struct ibv_ravh ravh;
    uint32_t op_index;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_RC_MLX5_CHECK_RNDV_PARAMS(iovcnt, header_length, tm_hdr_len,
                                   UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE),
                                   iface->super.tm.max_rndv_data +
                                   UCT_RC_MLX5_TMH_PRIV_LEN);
    UCT_DC_CHECK_RES_PTR(iface, ep);

    op_index = uct_rc_mlx5_tag_get_op_id(&iface->super, comp);

    uct_dc_mlx5_iface_fill_ravh(&ravh, iface->rx.dct.qp_num);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);

    uct_rc_mlx5_txqp_tag_inline_post(&iface->super, UCT_IB_QPT_DCI,
                                     txqp, txwq, MLX5_OPCODE_SEND, header,
                                     header_length, iov, tag, op_index,
                                     IBV_TMH_RNDV, 0, &ep->av,
                                     uct_dc_mlx5_ep_get_grh(ep),
                                     uct_ib_mlx5_wqe_av_size(&ep->av), &ravh,
                                     sizeof(ravh), MLX5_WQE_CTRL_SOLICITED);

    return (ucs_status_ptr_t)((uint64_t)op_index);
}

ucs_status_t uct_dc_mlx5_ep_tag_rndv_request(uct_ep_h tl_ep, uct_tag_t tag,
                                             const void* header,
                                             unsigned header_length,
                                             unsigned flags)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_CHECK_LENGTH(header_length + sizeof(struct ibv_tmh), 0,
                     UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE),
                     "tag_rndv_request");
    UCT_DC_MLX5_CHECK_RES(iface, ep);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);

    uct_rc_mlx5_txqp_tag_inline_post(&iface->super, UCT_IB_QPT_DCI,
                                     txqp, txwq, MLX5_OPCODE_SEND_IMM, header,
                                     header_length, NULL, tag, 0,
                                     IBV_TMH_EAGER, 0, &ep->av,
                                     uct_dc_mlx5_ep_get_grh(ep),
                                     uct_ib_mlx5_wqe_av_size(&ep->av), NULL, 0,
                                     MLX5_WQE_CTRL_SOLICITED);
    return UCS_OK;
}

ucs_status_t uct_dc_mlx5_iface_tag_recv_zcopy(uct_iface_h tl_iface,
                                              uct_tag_t tag,
                                              uct_tag_t tag_mask,
                                              const uct_iov_t *iov,
                                              size_t iovcnt,
                                              uct_tag_context_t *ctx)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);

    return uct_rc_mlx5_iface_common_tag_recv(&iface->super, tag, tag_mask,
                                             iov, iovcnt, ctx);
}

ucs_status_t uct_dc_mlx5_iface_tag_recv_cancel(uct_iface_h tl_iface,
                                               uct_tag_context_t *ctx,
                                               int force)
{
   uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);

   return uct_rc_mlx5_iface_common_tag_recv_cancel(&iface->super, ctx, force);
}
#endif

ucs_status_t uct_dc_mlx5_ep_fc_ctrl(uct_ep_t *tl_ep, unsigned op,
                                    uct_rc_pending_req_t *req)
{
    uct_dc_mlx5_ep_t *dc_ep    = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_dc_mlx5_iface_t);
    uct_ib_iface_t *ib_iface   = &iface->super.super.super;
    struct ibv_ah_attr ah_attr = {.is_global = 0};
    uct_rc_iface_send_op_t *grant_ep;
    uct_dc_fc_sender_data_t sender;
    uct_dc_fc_request_t *dc_req;
    struct mlx5_wqe_av mlx5_av;
    uct_ib_mlx5_base_av_t av;
    ucs_status_t status;
    uintptr_t sender_ep;
    struct ibv_ah *ah;

    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    ucs_assert((sizeof(uint8_t) + sizeof(sender_ep)) <=
                UCT_IB_MLX5_AV_FULL_SIZE);

    UCT_DC_MLX5_CHECK_DCI_RES(iface, dc_ep);
    UCT_DC_MLX5_IFACE_TXQP_GET(iface, dc_ep, txqp, txwq);

    dc_req = ucs_derived_of(req, uct_dc_fc_request_t);

    if (op == UCT_RC_EP_FC_PURE_GRANT) {
        ucs_assert(req != NULL);

        sender_ep = (uintptr_t)dc_req->sender.ep;

        /* TODO: look at common code with uct_ud_mlx5_iface_get_av */
        if (dc_req->sender.global.is_global) {
            uct_ib_iface_fill_ah_attr_from_gid_lid(ib_iface, dc_req->lid,
                                                   ucs_unaligned_ptr(&dc_req->sender.global.gid),
                                                   iface->super.super.super.gid_info.gid_index,
                                                   0, &ah_attr);

            status = uct_ib_iface_create_ah(ib_iface, &ah_attr, &ah);
            if (status != UCS_OK) {
                return status;
            }

            uct_ib_mlx5_get_av(ah, &mlx5_av);
        }

        /* Note av initialization is copied from exp verbs */
        av.stat_rate_sl = ib_iface->config.sl; /* (attr->static_rate << 4) | attr->sl */
        av.fl_mlid      = ib_iface->path_bits[0] & 0x7f;

        /* lid in dc_req is in BE already  */
        if (uct_ib_iface_is_roce(ib_iface)) {
            av.rlid     = htons(UCT_IB_ROCE_UDP_SRC_PORT_BASE);
        } else {
            av.rlid     = dc_req->lid | htons(ib_iface->path_bits[0]);
        }
        av.dqp_dct      = htonl(dc_req->dct_num);

        if (!iface->ud_common.config.compact_av || ah_attr.is_global) {
            av.dqp_dct |= UCT_IB_MLX5_EXTENDED_UD_AV;
        }

        uct_rc_mlx5_txqp_inline_post(&iface->super, UCT_IB_QPT_DCI,
                                     txqp, txwq, MLX5_OPCODE_SEND,
                                     &av /*dummy*/, 0, op, sender_ep, 0,
                                     0, 0,
                                     &av, ah_attr.is_global ? mlx5_av_grh(&mlx5_av) : NULL,
                                     uct_ib_mlx5_wqe_av_size(&av), 0, INT_MAX);
    } else {
        ucs_assert(op == UCT_RC_EP_FLAG_FC_HARD_REQ);
        sender.ep               = (uint64_t)dc_ep;
        sender.global.gid       = ib_iface->gid_info.gid;
        sender.global.is_global = dc_ep->flags & UCT_DC_MLX5_EP_FLAG_GRH;

        grant_ep = ucs_mpool_get(&iface->super.super.tx.send_op_mp);
        if (ucs_unlikely(grant_ep == NULL)) {
            ucs_error("failed to allocate fc hard req grant op");
            return UCS_ERR_NO_MEMORY;
        }

        UCS_STATS_UPDATE_COUNTER(dc_ep->fc.stats,
                                 UCT_RC_FC_STAT_TX_HARD_REQ, 1);

        uct_rc_ep_init_send_op(grant_ep, UCT_RC_IFACE_SEND_OP_FLAG_FC_GRANT,
                               NULL, (uct_rc_send_handler_t)ucs_mpool_put);
        grant_ep->ep = tl_ep;
        uct_rc_mlx5_txqp_inline_post(&iface->super, UCT_IB_QPT_DCI,
                                     txqp, txwq, MLX5_OPCODE_SEND_IMM,
                                     &sender.global, sizeof(sender.global), op, sender.ep,
                                     iface->rx.dct.qp_num,
                                     0, 0,
                                     &dc_ep->av,
                                     uct_dc_mlx5_ep_get_grh(dc_ep),
                                     uct_ib_mlx5_wqe_av_size(&dc_ep->av),
                                     MLX5_WQE_CTRL_SOLICITED, INT_MAX);
        uct_rc_txqp_add_send_op_sn(txqp, grant_ep, txwq->prev_sw_pi);
    }

    return UCS_OK;
}

static void uct_dc_mlx5_ep_keepalive_cleanup(uct_dc_mlx5_ep_t *ep)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                                uct_dc_mlx5_iface_t);
    uct_rc_iface_send_op_t *op;
    ucs_queue_iter_t iter;
    uct_rc_txqp_t *txqp;

    if (!(ep->flags & UCT_DC_MLX5_EP_FLAG_KEEPALIVE_POSTED)) {
        return;
    }

    /* clean keepalive requests */
    txqp = &iface->tx.dcis[iface->tx.ndci].txqp;
    ucs_queue_for_each_safe(op, iter, &txqp->outstanding, queue) {
        if (op->ep == &ep->super.super) {
            ucs_queue_del_iter(&txqp->outstanding, iter);
            ucs_mpool_put(op);
            break;
        }
    }
}

static void uct_dc_mlx5_txqp_purge_outstanding(uct_dc_mlx5_iface_t *iface,
                                               uct_rc_txqp_t *txqp,
                                               ucs_status_t status,
                                               uint16_t sn, int warn)
{
    uct_rc_iface_send_op_t *op;
    uct_dc_mlx5_ep_t *ep;

    ucs_queue_for_each_extract(op, &txqp->outstanding, queue,
                               UCS_CIRCULAR_COMPARE16(op->sn, <=, sn)) {
        if (op->flags & UCT_RC_IFACE_SEND_OP_FLAG_FC_GRANT) {
            /* grant operation aborted so need to clear "wait-for-grant" flag */
            ep = ucs_derived_of(op->ep, uct_dc_mlx5_ep_t);
            uct_dc_mlx5_ep_clear_fc_grant_flag(iface, ep);
            ucs_mpool_put(op);
        } else {
            uct_rc_txqp_purge_outstanding_op(&iface->super.super, op, status,
                                             warn);
        }
    }
}

UCS_CLASS_INIT_FUNC(uct_dc_mlx5_ep_t, uct_dc_mlx5_iface_t *iface,
                    const uct_dc_mlx5_iface_addr_t *if_addr,
                    uct_ib_mlx5_base_av_t *av)
{
    uint32_t remote_dctn;

    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super.super.super);

    self->atomic_mr_offset = uct_ib_md_atomic_offset(if_addr->atomic_mr_id);
    remote_dctn            = uct_ib_unpack_uint24(if_addr->qp_num);

    memcpy(&self->av, av, sizeof(*av));
    self->av.dqp_dct |= htonl(remote_dctn);

    return uct_dc_mlx5_ep_basic_init(iface, self);
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_mlx5_ep_t)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(self->super.super.iface,
                                                uct_dc_mlx5_iface_t);

    uct_dc_mlx5_ep_pending_purge(&self->super.super, NULL, NULL);
    uct_rc_fc_cleanup(&self->fc);

    ucs_assert_always(self->flags & UCT_DC_MLX5_EP_FLAG_VALID);
    uct_dc_mlx5_ep_keepalive_cleanup(self);

    if ((self->dci == UCT_DC_MLX5_EP_NO_DCI) ||
        uct_dc_mlx5_iface_is_dci_rand(iface)) {
        return;
    }

    /* TODO: this is good for dcs policy only.
     * Need to change if eps share dci
     */
    ucs_arbiter_group_cleanup(uct_dc_mlx5_ep_arb_group(iface, self));
    ucs_assertv_always(uct_dc_mlx5_iface_dci_has_outstanding(iface, self->dci),
                       "iface (%p) ep (%p) dci leak detected: dci=%d", iface,
                       self, self->dci);

    uct_dc_mlx5_txqp_purge_outstanding(iface, &iface->tx.dcis[self->dci].txqp,
                                       UCS_ERR_CANCELED,
                                       iface->tx.dcis[self->dci].txwq.sw_pi, 1);
    ucs_assert(ucs_queue_is_empty(&iface->tx.dcis[self->dci].txqp.outstanding));
    iface->tx.dcis[self->dci].ep = NULL;
}

UCS_CLASS_DEFINE(uct_dc_mlx5_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_dc_mlx5_ep_t, uct_ep_t, uct_dc_mlx5_iface_t *,
                          const uct_dc_mlx5_iface_addr_t *,
                          uct_ib_mlx5_base_av_t *);

UCS_CLASS_INIT_FUNC(uct_dc_mlx5_grh_ep_t, uct_dc_mlx5_iface_t *iface,
                    const uct_dc_mlx5_iface_addr_t *if_addr,
                    uct_ib_mlx5_base_av_t *av,
                    struct mlx5_grh_av *grh_av)
{
    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(uct_dc_mlx5_ep_t, iface, if_addr, av);

    self->super.flags |= UCT_DC_MLX5_EP_FLAG_GRH;
    memcpy(&self->grh_av, grh_av, sizeof(*grh_av));
    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(uct_dc_mlx5_grh_ep_t)
{
    ucs_trace_func("");
}

UCS_CLASS_DEFINE(uct_dc_mlx5_grh_ep_t, uct_dc_mlx5_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_dc_mlx5_grh_ep_t, uct_ep_t, uct_dc_mlx5_iface_t *,
                          const uct_dc_mlx5_iface_addr_t *,
                          uct_ib_mlx5_base_av_t *, struct mlx5_grh_av *);

void uct_dc_mlx5_ep_cleanup(uct_ep_h tl_ep, ucs_class_t *cls)
{
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_dc_mlx5_iface_t);

    UCS_CLASS_CLEANUP_CALL(cls, ep);

    if (uct_dc_mlx5_ep_fc_wait_for_grant(ep)) {
        ucs_trace("not releasing dc_mlx5_ep %p - waiting for grant", ep);
        ep->flags &= ~UCT_DC_MLX5_EP_FLAG_VALID;
        /* No need to wait for grant on this ep anymore */
        uct_dc_mlx5_ep_clear_fc_grant_flag(iface, ep);
        ucs_list_add_tail(&iface->tx.gc_list, &ep->list);
    } else {
        ucs_free(ep);
    }
}

void uct_dc_mlx5_ep_release(uct_dc_mlx5_ep_t *ep)
{
    ucs_assert_always(!(ep->flags & UCT_DC_MLX5_EP_FLAG_VALID));
    ucs_debug("release dc_mlx5_ep %p", ep);
    ucs_list_del(&ep->list);
    ucs_free(ep);
}

void uct_dc_mlx5_ep_pending_common(uct_dc_mlx5_iface_t *iface,
                                   uct_dc_mlx5_ep_t *ep, uct_pending_req_t *r,
                                   unsigned flags, int push_to_head)
{
    int no_dci = (ep->dci == UCT_DC_MLX5_EP_NO_DCI);
    ucs_arbiter_group_t *group;

    UCS_STATIC_ASSERT(sizeof(uct_dc_mlx5_pending_req_priv) <=
                      UCT_PENDING_REQ_PRIV_LEN);

    if (uct_dc_mlx5_iface_is_dci_rand(iface)) {
        uct_dc_mlx5_pending_req_priv(r)->ep = ep;
        group = uct_dc_mlx5_ep_rand_arb_group(iface, ep);
    } else {
        group = &ep->arb_group;
    }

    if (push_to_head) {
        uct_pending_req_arb_group_push_head(no_dci ?
                                            uct_dc_mlx5_iface_dci_waitq(iface) :
                                            uct_dc_mlx5_iface_tx_waitq(iface),
                                            group, r);
    } else {
        uct_pending_req_arb_group_push(group, r);
    }

    if (no_dci) {
        /* no dci:
         *  Do not grab dci here. Instead put the group on dci allocation arbiter.
         *  This way we can assure fairness between all eps waiting for
         *  dci allocation. Relevant for dcs and dcs_quota policies.
         */
        uct_dc_mlx5_iface_schedule_dci_alloc(iface, ep);
    } else {
        uct_dc_mlx5_iface_dci_sched_tx(iface, ep);
    }

    UCT_TL_EP_STAT_PEND(&ep->super);
}


/* TODO:
   currently pending code supports only dcs policy
   support hash/random policies
 */
ucs_status_t uct_dc_mlx5_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *r,
                                        unsigned flags)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);

    /* ep can tx iff
     * - iface has resources: cqe and tx skb
     * - dci is either assigned or can be assigned
     * - dci has resources
     */
    if (uct_dc_mlx5_iface_has_tx_resources(iface)) {
        if (ep->dci == UCT_DC_MLX5_EP_NO_DCI) {
            if (uct_dc_mlx5_iface_dci_can_alloc(iface) && (ep->fc.fc_wnd > 0)) {
                return UCS_ERR_BUSY;
            }
        } else {
            if (uct_dc_mlx5_iface_dci_ep_can_send(ep)) {
                return UCS_ERR_BUSY;
            }
        }
    }

    uct_dc_mlx5_ep_pending_common(iface, ep, r, flags, 0);

    return UCS_OK;
}

/**
 * dispatch requests waiting for dci allocation
 * Relevant for dcs and dcs_quota policies only.
 */
ucs_arbiter_cb_result_t
uct_dc_mlx5_iface_dci_do_pending_wait(ucs_arbiter_t *arbiter,
                                      ucs_arbiter_group_t *group,
                                      ucs_arbiter_elem_t *elem,
                                      void *arg)
{
    uct_dc_mlx5_ep_t *ep = ucs_container_of(group, uct_dc_mlx5_ep_t, arb_group);
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_dc_mlx5_iface_t);

    ucs_assert(!uct_dc_mlx5_iface_is_dci_rand(iface));

    if (ep->dci != UCT_DC_MLX5_EP_NO_DCI) {
        return UCS_ARBITER_CB_RESULT_DESCHED_GROUP;
    }

    if (!uct_dc_mlx5_iface_dci_can_alloc(iface)) {
        return UCS_ARBITER_CB_RESULT_STOP;
    }
    uct_dc_mlx5_iface_dci_alloc(iface, ep);
    ucs_assert_always(ep->dci != UCT_DC_MLX5_EP_NO_DCI);
    uct_dc_mlx5_iface_dci_sched_tx(iface, ep);
    return UCS_ARBITER_CB_RESULT_DESCHED_GROUP;
}

ucs_arbiter_cb_result_t
uct_dc_mlx5_iface_dci_do_common_pending_tx(uct_dc_mlx5_ep_t *ep,
                                           ucs_arbiter_elem_t *elem)
{
    uct_pending_req_t *req     = ucs_container_of(elem, uct_pending_req_t, priv);
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                                uct_dc_mlx5_iface_t);
    ucs_status_t status;

    if (!uct_dc_mlx5_iface_has_tx_resources(iface)) {
        return UCS_ARBITER_CB_RESULT_STOP;
    }

    status = uct_rc_iface_invoke_pending_cb(&iface->super.super, req);
    if (status == UCS_OK) {
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    } else if (status == UCS_INPROGRESS) {
        return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
    }

    if (!uct_dc_mlx5_iface_dci_ep_can_send(ep)) {
        return UCS_ARBITER_CB_RESULT_DESCHED_GROUP;
    }

    ucs_assertv(!uct_dc_mlx5_iface_has_tx_resources(iface),
                "pending callback returned error but send resources are available");
    return UCS_ARBITER_CB_RESULT_STOP;
}

/**
 * dispatch requests waiting for tx resources (dcs* DCI policies)
 */
ucs_arbiter_cb_result_t
uct_dc_mlx5_iface_dci_do_dcs_pending_tx(ucs_arbiter_t *arbiter,
                                        ucs_arbiter_group_t *group,
                                        ucs_arbiter_elem_t *elem,
                                        void *arg)
{

    uct_dc_mlx5_ep_t *ep       = ucs_container_of(group, uct_dc_mlx5_ep_t,
                                                  arb_group);
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                                uct_dc_mlx5_iface_t);
    int is_only                = ucs_arbiter_elem_is_only(elem);
    ucs_arbiter_cb_result_t res;

    res = uct_dc_mlx5_iface_dci_do_common_pending_tx(ep, elem);
    if (res == UCS_ARBITER_CB_RESULT_REMOVE_ELEM) {
        /* For dcs* policies release dci if this is the last elem in the group
         * and the dci has no outstanding operations. For example pending
         * callback did not send anything. (uct_ep_flush or just return ok)
         */
        if (is_only) {
            uct_dc_mlx5_iface_dci_free(iface, ep);
        }
    }

    return res;
}

/**
 * dispatch requests waiting for tx resources (rand DCI policy)
 */
ucs_arbiter_cb_result_t
uct_dc_mlx5_iface_dci_do_rand_pending_tx(ucs_arbiter_t *arbiter,
                                         ucs_arbiter_group_t *group,
                                         ucs_arbiter_elem_t *elem,
                                         void *arg)
{
    uct_pending_req_t *req     = ucs_container_of(elem, uct_pending_req_t, priv);
    uct_dc_mlx5_ep_t *ep       = uct_dc_mlx5_pending_req_priv(req)->ep;
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                                uct_dc_mlx5_iface_t);
    ucs_arbiter_cb_result_t res;

    res = uct_dc_mlx5_iface_dci_do_common_pending_tx(ep, elem);
    if ((res == UCS_ARBITER_CB_RESULT_DESCHED_GROUP) &&
        uct_rc_fc_has_resources(&iface->super.super, &ep->fc)) {
        /* We can't desched group with rand policy if non FC resources are
         * missing, since it's never scheduled again. */
        res = UCS_ARBITER_CB_RESULT_RESCHED_GROUP;
    }

    return res;
}

static ucs_arbiter_cb_result_t
uct_dc_mlx5_ep_arbiter_purge_cb(ucs_arbiter_t *arbiter, ucs_arbiter_group_t *group,
                                ucs_arbiter_elem_t *elem, void *arg)
{
    uct_purge_cb_args_t *cb_args = arg;
    void **priv_args             = cb_args->arg;
    uct_dc_mlx5_ep_t *ep         = priv_args[0];
    uct_dc_mlx5_iface_t *iface   = ucs_derived_of(ep->super.super.iface,
                                                  uct_dc_mlx5_iface_t);
    uct_pending_req_t *req       = ucs_container_of(elem, uct_pending_req_t, priv);
    uct_rc_pending_req_t *freq;

    if (uct_dc_mlx5_iface_is_dci_rand(iface) &&
        (uct_dc_mlx5_pending_req_priv(req)->ep != ep)) {
        /* element belongs to another ep - do not remove it */
        return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
    }

    if (ucs_likely(req->func != uct_dc_mlx5_iface_fc_grant)){
        if (cb_args->cb != NULL) {
            cb_args->cb(req, priv_args[1]);
        } else {
            ucs_debug("ep=%p cancelling user pending request %p", ep, req);
        }
    } else {
        /* User callback should not be called for FC messages.
         * Just return pending request memory to the pool */
        freq = ucs_derived_of(req, uct_rc_pending_req_t);
        ucs_mpool_put(freq);
    }

    return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
}

void uct_dc_mlx5_ep_pending_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb, void *arg)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    void *priv_args[2]         = {ep, arg};
    uct_purge_cb_args_t args   = {cb, priv_args};

    if (uct_dc_mlx5_iface_is_dci_rand(iface)) {
        ucs_arbiter_group_purge(uct_dc_mlx5_iface_tx_waitq(iface),
                                uct_dc_mlx5_ep_rand_arb_group(iface, ep),
                                uct_dc_mlx5_ep_arbiter_purge_cb, &args);
        return;
    }

    if (ep->dci == UCT_DC_MLX5_EP_NO_DCI) {
        ucs_arbiter_group_purge(uct_dc_mlx5_iface_dci_waitq(iface), &ep->arb_group,
                                uct_dc_mlx5_ep_arbiter_purge_cb, &args);
    } else {
        ucs_arbiter_group_purge(uct_dc_mlx5_iface_tx_waitq(iface), &ep->arb_group,
                                uct_dc_mlx5_ep_arbiter_purge_cb, &args);
        uct_dc_mlx5_iface_dci_free(iface, ep);
    }
}

ucs_status_t uct_dc_mlx5_ep_check_fc(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    ucs_status_t status;

    if (iface->super.super.config.fc_enabled) {
        UCT_RC_CHECK_FC_WND(&ep->fc, ep->super.stats);
        if ((ep->fc.fc_wnd == iface->super.super.config.fc_hard_thresh) &&
            !uct_dc_mlx5_ep_fc_wait_for_grant(ep)) {
            status = uct_rc_fc_ctrl(&ep->super.super,
                                    UCT_RC_EP_FLAG_FC_HARD_REQ,
                                    NULL);
            if (status != UCS_OK) {
                return status;
            }
            ep->flags |= UCT_DC_MLX5_EP_FLAG_FC_WAIT_FOR_GRANT;
            ++iface->tx.fc_grants;
        }
    } else {
        /* Set fc_wnd to max, to send as much as possible without checks */
        ep->fc.fc_wnd = INT16_MAX;
    }
    return UCS_OK;
}

void uct_dc_mlx5_ep_handle_failure(uct_dc_mlx5_ep_t *ep, void *arg,
                                   ucs_status_t ep_status)
{
    struct mlx5_cqe64 *cqe     = arg;
    uct_iface_h tl_iface       = ep->super.super.iface;
    uint8_t dci                = ep->dci;
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    uct_rc_txqp_t *txqp        = &iface->tx.dcis[dci].txqp;
    uct_ib_mlx5_txwq_t *txwq   = &iface->tx.dcis[dci].txwq;
    uint16_t pi                = ntohs(cqe->wqe_counter);

    ucs_assert(!uct_dc_mlx5_iface_is_dci_rand(iface));

    uct_dc_mlx5_update_tx_res(iface, txwq, txqp, pi);
    uct_dc_mlx5_txqp_purge_outstanding(iface, txqp, ep_status, pi, 0);

    ucs_assert(ep->dci != UCT_DC_MLX5_EP_NO_DCI);
    uct_dc_mlx5_iface_dci_put(iface, dci);
    uct_dc_mlx5_iface_set_ep_failed(iface, ep, cqe, txwq, ep_status);

    if (ep->dci == UCT_DC_MLX5_EP_NO_DCI) {
        uct_dc_mlx5_iface_reset_dci(iface, dci);
    }
}

static void
uct_dc_mlx5_ep_check_send_completion(uct_rc_iface_send_op_t *op, const void *resp)
{
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(op->ep, uct_dc_mlx5_ep_t);

    ucs_assert(ep->flags & UCT_DC_MLX5_EP_FLAG_KEEPALIVE_POSTED);
    ep->flags &= ~UCT_DC_MLX5_EP_FLAG_KEEPALIVE_POSTED;
    ucs_mpool_put(op);
}

ucs_status_t
uct_dc_mlx5_ep_check(uct_ep_h tl_ep, unsigned flags, uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uint64_t dummy             = 0;
    ucs_status_t status;
    uct_rc_iface_send_op_t *op;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_EP_KEEPALIVE_CHECK_PARAM(flags, comp);

    if ((ep->dci != UCT_DC_MLX5_EP_NO_DCI) ||
        (ep->flags & UCT_DC_MLX5_EP_FLAG_KEEPALIVE_POSTED)) {
        /* in case if EP has DCI and some TX resources are involved in
         * communications, then keepalive operation is not needed */
        return UCS_OK;
    }

    status = uct_dc_mlx5_iface_keepalive_init(iface);
    if (status != UCS_OK) {
        ucs_error("failed to initialize keepalive dci: %s",
                  ucs_status_string(status));
        return status;
    }

    op = ucs_mpool_get(&iface->super.super.tx.send_op_mp);
    if (ucs_unlikely(op == NULL)) {
        ucs_error("failed to allocate keepalive op");
        return UCS_ERR_NO_MEMORY;
    }

    uct_rc_ep_init_send_op(op, 0, NULL, uct_dc_mlx5_ep_check_send_completion);
    op->ep = tl_ep;
    UCT_DC_MLX5_IFACE_TXQP_DCI_GET(iface, iface->tx.ndci, txqp, txwq);
    uct_rc_mlx5_txqp_inline_post(&iface->super, UCT_IB_QPT_DCI,
                                 txqp, txwq, MLX5_OPCODE_RDMA_WRITE,
                                 &dummy, 0, 0, 0, 0, 0, 0,
                                 &ep->av, uct_dc_mlx5_ep_get_grh(ep),
                                 uct_ib_mlx5_wqe_av_size(&ep->av), 0, INT_MAX);
    uct_rc_txqp_add_send_op_sn(txqp, op, txwq->sig_pi);
    ep->flags |= UCT_DC_MLX5_EP_FLAG_KEEPALIVE_POSTED;
    return UCS_OK;
}
