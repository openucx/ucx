/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "dc_mlx5.h"

#include <uct/api/uct.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_log.h>
#include <uct/ib/mlx5/ib_mlx5_log.h>
#include <uct/base/uct_md.h>
#include <ucs/arch/bitops.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>
#include <string.h>


static ucs_config_field_t uct_dc_mlx5_iface_config_table[] = {
  {"DC_", "RC_FC_ENABLE=y", NULL,
   ucs_offsetof(uct_dc_mlx5_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_dc_iface_config_table)},

  {"", "", NULL,
   ucs_offsetof(uct_dc_mlx5_iface_config_t, ud_common),
   UCS_CONFIG_TYPE_TABLE(uct_ud_mlx5_iface_common_config_table)},

  {"", "", NULL,
   ucs_offsetof(uct_dc_mlx5_iface_config_t, mlx5_common),
   UCS_CONFIG_TYPE_TABLE(uct_mlx5_common_config_table)},

  {NULL}
};


static UCS_CLASS_CLEANUP_FUNC(uct_dc_mlx5_ep_t)
{
    ucs_trace_func("");
}

static UCS_CLASS_INIT_FUNC(uct_dc_mlx5_ep_t, uct_dc_mlx5_iface_t *dc_iface,
                           const uct_dc_iface_addr_t *if_addr,
                           const uct_ib_mlx5_base_av_t *av)
{
    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(uct_dc_ep_t, &dc_iface->super, if_addr);

    memcpy(&self->av, av, sizeof(*av));
    self->av.dqp_dct |= htonl(uct_ib_unpack_uint24(if_addr->qp_num));
    return UCS_OK;
}

UCS_CLASS_DEFINE(uct_dc_mlx5_ep_t, uct_dc_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_dc_mlx5_ep_t, uct_ep_t,
                          uct_dc_mlx5_iface_t *,
                          const uct_dc_iface_addr_t *,
                          uct_ib_mlx5_base_av_t *);

static UCS_CLASS_INIT_FUNC(uct_dc_mlx5_grh_ep_t, uct_dc_mlx5_iface_t *dc_iface,
                           const uct_dc_iface_addr_t *if_addr,
                           uct_ib_mlx5_base_av_t *av,
                           struct mlx5_grh_av *grh_av)
{
    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(uct_dc_mlx5_ep_t, dc_iface, if_addr, av);

    self->super.super.flags |= UCT_DC_EP_FLAG_GRH;
    memcpy(&self->grh_av, grh_av, sizeof(*grh_av));
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_mlx5_grh_ep_t)
{
    ucs_trace_func("");
}

UCS_CLASS_DEFINE(uct_dc_mlx5_grh_ep_t, uct_dc_mlx5_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_dc_mlx5_grh_ep_t, uct_ep_t,
                          uct_dc_mlx5_iface_t *,
                          const uct_dc_iface_addr_t *,
                          uct_ib_mlx5_base_av_t *, struct mlx5_grh_av *);

static ucs_status_t
uct_dc_mlx5_ep_create_connected(uct_iface_h tl_iface,
                                const uct_device_addr_t *dev_addr,
                                const uct_iface_addr_t *iface_addr,
                                uct_ep_h* ep_p)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    const uct_ib_address_t *ib_addr = (const uct_ib_address_t *)dev_addr;
    const uct_dc_iface_addr_t *if_addr = (const uct_dc_iface_addr_t *)iface_addr;
    ucs_status_t status;
    int is_global;
    uct_ib_mlx5_base_av_t av;
    struct mlx5_grh_av grh_av;

    ucs_trace_func("");
    status = uct_ud_mlx5_iface_get_av(&iface->super.super.super, &iface->ud_common,
                                      ib_addr, iface->super.super.super.path_bits[0],
                                      &av, &grh_av, &is_global);
    if (status != UCS_OK) {
        return UCS_ERR_INVALID_ADDR;
    }

    if (is_global) {
        return UCS_CLASS_NEW(uct_dc_mlx5_grh_ep_t, ep_p, iface, if_addr, &av, &grh_av);
    } else {
        return UCS_CLASS_NEW(uct_dc_mlx5_ep_t, ep_p, iface, if_addr, &av);
    }
}

static void uct_dc_mlx5_ep_destroy(uct_ep_h tl_ep)
{
    uct_dc_ep_cleanup(tl_ep, &UCS_CLASS_NAME(uct_dc_mlx5_ep_t));
}

static ucs_status_t uct_dc_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    size_t max_am_inline       = UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE);
    size_t max_put_inline      = UCT_IB_MLX5_PUT_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE);
    ucs_status_t status;

#if HAVE_IBV_EXP_DM
    if (iface->mlx5_common.dm.dm != NULL) {
        max_am_inline  = ucs_max(iface->mlx5_common.dm.dm->seg_len,
                                 UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE));
        max_put_inline = ucs_max(iface->mlx5_common.dm.dm->seg_len,
                                 UCT_IB_MLX5_PUT_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE));
    }
#endif

    status = uct_dc_iface_query(&iface->super, iface_attr,
                                max_put_inline,
                                max_am_inline,
                                UCT_IB_MLX5_AM_ZCOPY_MAX_HDR(UCT_IB_MLX5_AV_FULL_SIZE),
                                UCT_IB_MLX5_AM_ZCOPY_MAX_IOV,
                                UCT_RC_MLX5_TM_EAGER_ZCOPY_MAX_IOV(UCT_IB_MLX5_AV_FULL_SIZE));
    if (status != UCS_OK) {
        return status;
    }

    uct_rc_mlx5_iface_common_query(&iface->super.super.super, iface_attr);
    return UCS_OK;
}

#define UCT_DC_MLX5_TXQP_DECL(_txqp, _txwq) \
    uct_rc_txqp_t *_txqp; \
    uct_ib_mlx5_txwq_t *_txwq;

#define UCT_DC_MLX5_IFACE_TXQP_GET(_iface, _ep, _txqp, _txwq) \
{ \
    uint8_t dci; \
    dci = (_ep)->dci; \
    _txqp = &(_iface)->super.tx.dcis[dci].txqp; \
    _txwq = &(_iface)->dci_wqs[dci]; \
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_bcopy_post(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep,
                            unsigned opcode, unsigned length,
                            /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey,
                            uct_rc_iface_send_desc_t *desc, uint8_t send_flags,
                            uint32_t imm_val_be, const void *buffer,
                            uct_ib_log_sge_t *log_sge)
{
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, &ep->super, txqp, txwq);
    desc->super.sn = txwq->sw_pi;
    uct_rc_mlx5_txqp_dptr_post(&iface->super.super, IBV_EXP_QPT_DC_INI, txqp, txwq,
                               opcode, buffer, length, &desc->lkey,
                               rdma_raddr, uct_ib_md_direct_rkey(rdma_rkey),
                               0, 0, 0, 0,
                               &ep->av, uct_dc_mlx5_ep_get_grh(ep),
                               uct_ib_mlx5_wqe_av_size(&ep->av),
                               MLX5_WQE_CTRL_CQ_UPDATE | send_flags, imm_val_be, INT_MAX,
                               log_sge);
    uct_rc_txqp_add_send_op(txqp, &desc->super);
}


static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_zcopy_post(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep,
                             unsigned opcode, const uct_iov_t *iov, size_t iovcnt,
                             /* SEND */ uint8_t am_id, const void *am_hdr, unsigned am_hdr_len,
                             /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey,
                             /* TAG  */ uct_tag_t tag, uint32_t app_ctx, uint32_t ib_imm_be,
                             uct_completion_t *comp, uint8_t send_flags)
{
    uint16_t sn;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, &ep->super, txqp, txwq);

    sn = txwq->sw_pi;
    uct_rc_mlx5_txqp_dptr_post_iov(&iface->super.super, IBV_EXP_QPT_DC_INI, txqp,
                                   txwq, opcode, iov, iovcnt,
                                   am_id, am_hdr, am_hdr_len,
                                   rdma_raddr, uct_ib_md_direct_rkey(rdma_rkey),
                                   tag, app_ctx, ib_imm_be,
                                   &ep->av, uct_dc_mlx5_ep_get_grh(ep),
                                   uct_ib_mlx5_wqe_av_size(&ep->av),
                                   MLX5_WQE_CTRL_CQ_UPDATE | send_flags,
                                   UCT_IB_MAX_ZCOPY_LOG_SGE(&iface->super.super.super));

    uct_rc_txqp_add_send_comp(&iface->super.super, txqp, comp, sn);
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_atomic_post(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep,
                              unsigned opcode, uct_rc_iface_send_desc_t *desc, unsigned length,
                              uint64_t remote_addr, uct_rkey_t rkey,
                              uint64_t compare_mask, uint64_t compare,
                              uint64_t swap_mask, uint64_t swap_add)
{
    uint32_t ib_rkey = uct_ib_resolve_atomic_rkey(rkey, ep->super.atomic_mr_offset,
                                                  &remote_addr);

    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);
    UCT_DC_MLX5_IFACE_TXQP_GET(iface, &ep->super, txqp, txwq);

    desc->super.sn = txwq->sw_pi;
    uct_rc_mlx5_txqp_dptr_post(&iface->super.super, IBV_EXP_QPT_DC_INI, txqp, txwq,
                               opcode, desc + 1, length, &desc->lkey,
                               remote_addr, ib_rkey,
                               compare_mask, compare, swap_mask, swap_add,
                               &ep->av, uct_dc_mlx5_ep_get_grh(ep),
                               uct_ib_mlx5_wqe_av_size(&ep->av),
                               MLX5_WQE_CTRL_CQ_UPDATE, 0, INT_MAX, NULL);

    UCT_TL_EP_STAT_ATOMIC(&ep->super.super);
    uct_rc_txqp_add_send_op(txqp, &desc->super);
}

static inline void uct_dc_mlx5_iface_add_send_comp(uct_dc_mlx5_iface_t *iface,
                                                   uct_dc_mlx5_ep_t *ep,
                                                   uct_completion_t *comp)
{
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, &ep->super, txqp, txwq);
    uct_rc_txqp_add_send_comp(&iface->super.super, txqp, comp, txwq->sig_pi);
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

    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_MLX5_CHECK_ATOMIC_OPS(opcode, size, UCT_RC_MLX5_ATOMIC_OPS);

    status = uct_rc_mlx5_iface_common_atomic_data(opcode, size, value, &op, &compare_mask,
                                                  &compare, &swap_mask, &swap, &ext);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        return status;
    }

    UCT_RC_IFACE_GET_TX_ATOMIC_DESC(&iface->super.super, &iface->mlx5_common.tx.atomic_desc_mp, desc);
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
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.super.iface, uct_dc_mlx5_iface_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_ATOMIC_FETCH_DESC(&iface->super.super, &iface->mlx5_common.tx.atomic_desc_mp,
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
                                     remote_addr, rkey, 0, htobe64(compare), -1,
                                     htobe64(swap), comp);
}

ucs_status_t uct_dc_mlx5_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint32_t *result, uct_completion_t *comp)
{
    return uct_dc_mlx5_ep_atomic_fop(ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t),
                                     MLX5_OPCODE_ATOMIC_MASKED_CS, result, 1,
                                     sizeof(uint32_t), remote_addr, rkey, UCS_MASK(32),
                                     htonl(compare), -1, htonl(swap), comp);
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

static ucs_status_t UCS_F_ALWAYS_INLINE
uct_dc_mlx5_ep_am_short_inline(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                               const void *buffer, unsigned length)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_RC_MLX5_CHECK_AM_SHORT(id, length, UCT_IB_MLX5_AV_FULL_SIZE);
    UCT_DC_CHECK_RES_AND_FC(&iface->super, &ep->super);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, &ep->super, txqp, txwq);

    uct_rc_mlx5_txqp_inline_post(&iface->super.super, IBV_EXP_QPT_DC_INI,
                                 txqp, txwq,
                                 MLX5_OPCODE_SEND,
                                 buffer, length, id, hdr, 0,
                                 0, 0,
                                 &ep->av, uct_dc_mlx5_ep_get_grh(ep),
                                 uct_ib_mlx5_wqe_av_size(&ep->av),
                                 MLX5_WQE_CTRL_SOLICITED, INT_MAX);

    UCT_RC_UPDATE_FC_WND(&iface->super.super, &ep->super.fc);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, sizeof(hdr) + length);
    return UCS_OK;
}

#if HAVE_IBV_EXP_DM
static ucs_status_t UCS_F_ALWAYS_INLINE
uct_dc_mlx5_ep_short_dm(uct_dc_mlx5_ep_t *ep, uct_rc_mlx5_dm_copy_data_t *cache,
                        size_t hdr_len, const void *payload, unsigned length,
                        unsigned opcode, uint8_t fm_ce_se,
                        uint64_t rdma_raddr, uct_rkey_t rdma_rkey)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.super.iface, uct_dc_mlx5_iface_t);
    uct_rc_iface_send_desc_t *desc;
    void *buffer;
    ucs_status_t status;
    uct_ib_log_sge_t log_sge;

    status = uct_rc_mlx5_common_dm_make_data(&iface->mlx5_common, &iface->super.super,
                                             cache, hdr_len, payload, length, &desc,
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
#if HAVE_IBV_EXP_DM
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    ucs_status_t status;
    uct_rc_mlx5_dm_copy_data_t cache;

    if (ucs_likely((sizeof(uct_rc_am_short_hdr_t) + length <=
                    UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE)) ||
                   !iface->mlx5_common.dm.dm)) {
#endif
        return uct_dc_mlx5_ep_am_short_inline(tl_ep, id, hdr, buffer, length);
#if HAVE_IBV_EXP_DM
    }

    UCT_CHECK_LENGTH(length + sizeof(uct_rc_am_short_hdr_t), 0,
                     iface->mlx5_common.dm.seg_len, "am_short");
    UCT_DC_CHECK_RES_AND_FC(&iface->super, &ep->super);

    uct_rc_am_hdr_fill(&cache.am_hdr.rc_hdr, id);
    cache.am_hdr.am_hdr = hdr;

    status = uct_dc_mlx5_ep_short_dm(ep, &cache, sizeof(cache.am_hdr), buffer, length,
                                     MLX5_OPCODE_SEND,
                                     MLX5_WQE_CTRL_SOLICITED | MLX5_WQE_CTRL_CQ_UPDATE,
                                     0, 0);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, sizeof(cache.am_hdr) + length);
    UCT_RC_UPDATE_FC_WND(&iface->super.super, &ep->super.fc);
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

    UCT_DC_CHECK_RES_AND_FC(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_AM_BCOPY_DESC(&iface->super.super, &iface->super.super.tx.mp, desc,
                                      id, pack_cb, arg, &length);

    uct_dc_mlx5_iface_bcopy_post(iface, ep, MLX5_OPCODE_SEND,
                                 sizeof(uct_rc_hdr_t) + length, 0, 0, desc,
                                 MLX5_WQE_CTRL_SOLICITED, 0, desc + 1, NULL);

    UCT_RC_UPDATE_FC_WND(&iface->super.super, &ep->super.fc);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, BCOPY, length);
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
    UCT_DC_CHECK_RES_AND_FC(&iface->super, &ep->super);

    uct_dc_mlx5_iface_zcopy_post(iface, ep, MLX5_OPCODE_SEND, iov, iovcnt,
                                 id, header, header_length, 0, 0, 0ul, 0, 0,
                                 comp, MLX5_WQE_CTRL_SOLICITED);

    UCT_RC_UPDATE_FC_WND(&iface->super.super, &ep->super.fc);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, ZCOPY, header_length +
                      uct_iov_total_length(iov, iovcnt));

    return UCS_INPROGRESS;
}


static ucs_status_t UCS_F_ALWAYS_INLINE
uct_dc_mlx5_ep_put_short_inline(uct_ep_h tl_ep, const void *buffer,
                                unsigned length, uint64_t remote_addr,
                                uct_rkey_t rkey)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_RC_MLX5_CHECK_PUT_SHORT(length, UCT_IB_MLX5_AV_FULL_SIZE);
    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, &ep->super, txqp, txwq);
    uct_rc_mlx5_txqp_inline_post(&iface->super.super, IBV_EXP_QPT_DC_INI,
                                 txqp, txwq,
                                 MLX5_OPCODE_RDMA_WRITE,
                                 buffer, length, 0, 0, 0,
                                 remote_addr, uct_ib_md_direct_rkey(rkey),
                                 &ep->av, uct_dc_mlx5_ep_get_grh(ep),
                                 uct_ib_mlx5_wqe_av_size(&ep->av), 0, INT_MAX);

    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, SHORT, length);

    return UCS_OK;
}

ucs_status_t uct_dc_mlx5_ep_put_short(uct_ep_h tl_ep, const void *payload,
                                      unsigned length, uint64_t remote_addr,
                                      uct_rkey_t rkey)
{
#if HAVE_IBV_EXP_DM
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    ucs_status_t status;

    if (ucs_likely((length <= UCT_IB_MLX5_PUT_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE)) ||
                   !iface->mlx5_common.dm.dm)) {
#endif
        return uct_dc_mlx5_ep_put_short_inline(tl_ep, payload, length, remote_addr, rkey);
#if HAVE_IBV_EXP_DM
    }

    UCT_CHECK_LENGTH(length, 0, iface->mlx5_common.dm.seg_len, "put_short");
    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    status = uct_dc_mlx5_ep_short_dm(ep, NULL, 0, payload, length,
                                     MLX5_OPCODE_RDMA_WRITE,
                                     MLX5_WQE_CTRL_CQ_UPDATE,
                                     remote_addr, rkey);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, SHORT, length);
    return UCS_OK;
#endif
}

ssize_t uct_dc_mlx5_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                 void *arg, uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;
    size_t length;

    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_PUT_BCOPY_DESC(&iface->super.super, &iface->super.super.tx.mp,
                                       desc, pack_cb, arg, length);
    uct_dc_mlx5_iface_bcopy_post(iface, ep, MLX5_OPCODE_RDMA_WRITE, length,
                                 remote_addr, rkey, desc, 0, 0, desc + 1, NULL);
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, BCOPY, length);
    return length;
}

ucs_status_t uct_dc_mlx5_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);

    UCT_CHECK_IOV_SIZE(iovcnt, uct_ib_iface_get_max_iov(&iface->super.super.super),
                       "uct_dc_mlx5_ep_put_zcopy");
    UCT_CHECK_LENGTH(uct_iov_total_length(iov, iovcnt), 0, UCT_IB_MAX_MESSAGE_SIZE,
                     "put_zcopy");
    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    uct_dc_mlx5_iface_zcopy_post(iface, ep, MLX5_OPCODE_RDMA_WRITE, iov, iovcnt,
                                 0, NULL, 0, remote_addr, rkey, 0ul, 0, 0,
                                 comp, 0);

    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, ZCOPY,
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
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_CHECK_LENGTH(length, 0, iface->super.super.super.config.seg_size, "get_bcopy");
    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_GET_BCOPY_DESC(&iface->super.super, &iface->super.super.tx.mp,
                                       desc, unpack_cb, comp, arg, length);
    uct_dc_mlx5_iface_bcopy_post(iface, ep, MLX5_OPCODE_RDMA_READ, length,
                                 remote_addr, rkey, desc, 0, 0, desc + 1, NULL);
    UCT_TL_EP_STAT_OP(&ep->super.super, GET, BCOPY, length);
    return UCS_INPROGRESS;
}


ucs_status_t uct_dc_mlx5_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);

    UCT_CHECK_IOV_SIZE(iovcnt, uct_ib_iface_get_max_iov(&iface->super.super.super),
                       "uct_dc_mlx5_ep_get_zcopy");
    UCT_CHECK_LENGTH(uct_iov_total_length(iov, iovcnt),
                     iface->super.super.super.config.max_inl_resp + 1, UCT_IB_MAX_MESSAGE_SIZE,
                     "get_zcopy");
    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    uct_dc_mlx5_iface_zcopy_post(iface, ep, MLX5_OPCODE_RDMA_READ, iov, iovcnt,
                                 0, NULL, 0, remote_addr, rkey, 0ul, 0, 0,
                                 comp, 0);
    UCT_TL_EP_STAT_OP(&ep->super.super, GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    return UCS_INPROGRESS;
}

ucs_status_t uct_dc_mlx5_ep_flush(uct_ep_h tl_ep, unsigned flags, uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t    *ep    = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    ucs_status_t        status;

    status = uct_dc_ep_flush(tl_ep, flags, comp);
    if (status == UCS_OK) {
        return UCS_OK; /* all sends completed */
    }

    if (status == UCS_INPROGRESS) {
        ucs_assert(ep->super.dci != UCT_DC_EP_NO_DCI);
        uct_dc_mlx5_iface_add_send_comp(iface, ep, comp);
    }
    return status;
}

static UCS_F_ALWAYS_INLINE unsigned
uct_dc_mlx5_poll_tx(uct_dc_mlx5_iface_t *iface)
{
    uint8_t dci;
    struct mlx5_cqe64 *cqe;
    uint32_t qp_num;
    uint16_t hw_ci;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    cqe = uct_ib_mlx5_poll_cq(&iface->super.super.super,
                              &iface->mlx5_common.cq[UCT_IB_DIR_TX]);
    if (cqe == NULL) {
        return 0;
    }
    UCS_STATS_UPDATE_COUNTER(iface->super.super.stats, UCT_RC_IFACE_STAT_TX_COMPLETION, 1);

    ucs_memory_cpu_load_fence();

    qp_num = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
    dci = uct_dc_iface_dci_find(&iface->super, qp_num);
    txqp = &iface->super.tx.dcis[dci].txqp;
    txwq = &iface->dci_wqs[dci];
    hw_ci = ntohs(cqe->wqe_counter);

    ucs_trace_poll("dc_mlx5 iface %p tx_cqe: dci[%d] qpn 0x%x txqp %p hw_ci %d",
                   iface, dci, qp_num, txqp, hw_ci);

    uct_rc_mlx5_common_update_tx_res(&iface->super.super, txwq, txqp, hw_ci);
    uct_dc_iface_dci_put(&iface->super, dci);
    uct_rc_mlx5_txqp_process_tx_cqe(txqp, cqe, hw_ci);

    uct_dc_iface_progress_pending(&iface->super);
    return 1;
}

static unsigned uct_dc_mlx5_iface_progress(void *arg)
{
    uct_dc_mlx5_iface_t *iface = arg;
    unsigned count;

    count = uct_rc_mlx5_iface_common_poll_rx(&iface->mlx5_common,
                                             &iface->super.super, 0);
    if (count > 0) {
        return count;
    }
    return uct_dc_mlx5_poll_tx(iface);
}


#if IBV_EXP_HW_TM_DC
static ucs_status_t UCS_F_ALWAYS_INLINE
uct_dc_mlx5_ep_tag_eager_short_inline(uct_ep_h tl_ep, uct_tag_t tag,
                                      const void *data, size_t length)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_CHECK_LENGTH(length + sizeof(struct ibv_exp_tmh), 0,
                     UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE),
                     "uct_dc_mlx5_ep_tag_short");
    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, &ep->super, txqp, txwq);

    uct_rc_mlx5_txqp_tag_inline_post(&iface->super.super, IBV_EXP_QPT_DC_INI,
                                     txqp, txwq, MLX5_OPCODE_SEND, data, length,
                                     NULL, tag, 0, IBV_EXP_TMH_EAGER, 0,
                                     &ep->av, uct_dc_mlx5_ep_get_grh(ep),
                                     uct_ib_mlx5_wqe_av_size(&ep->av), NULL, 0,
                                     MLX5_WQE_CTRL_SOLICITED);

    return UCS_OK;
}

ucs_status_t uct_dc_mlx5_ep_tag_eager_short(uct_ep_h tl_ep, uct_tag_t tag,
                                            const void *data, size_t length)
{
#if HAVE_IBV_EXP_DM
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_rc_mlx5_dm_copy_data_t cache;

    if (ucs_likely((sizeof(struct ibv_exp_tmh) + length <=
                    UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE)) ||
                   !iface->mlx5_common.dm.dm)) {
#endif
        return uct_dc_mlx5_ep_tag_eager_short_inline(tl_ep, tag, data, length);
#if HAVE_IBV_EXP_DM
    }

    UCT_CHECK_LENGTH(length + sizeof(struct ibv_exp_tmh), 0,
                     iface->mlx5_common.dm.seg_len, "tag_short");
    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    uct_rc_mlx5_fill_tmh(ucs_unaligned_ptr(&cache.tm_hdr), tag, 0, IBV_EXP_TMH_EAGER);

    return uct_dc_mlx5_ep_short_dm(ep, &cache, sizeof(cache.tm_hdr), data, length,
                                   MLX5_OPCODE_SEND,
                                   MLX5_WQE_CTRL_SOLICITED | MLX5_WQE_CTRL_CQ_UPDATE,
                                   0, 0);
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

    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    UCT_RC_IFACE_FILL_TM_IMM(imm, app_ctx, ib_imm, opcode, MLX5_OPCODE_SEND, _IMM);

    UCT_RC_MLX5_IFACE_GET_TM_BCOPY_DESC(&iface->super.super,
                                        &iface->super.super.tx.mp, desc, tag,
                                        app_ctx, pack_cb, arg, length);

    uct_dc_mlx5_iface_bcopy_post(iface, ep, opcode,
                                 sizeof(struct ibv_exp_tmh) + length,
                                 0, 0, desc, MLX5_WQE_CTRL_SOLICITED, ib_imm, desc + 1, NULL);

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
    UCT_RC_CHECK_ZCOPY_DATA(sizeof(struct ibv_exp_tmh),
                            uct_iov_total_length(iov, iovcnt),
                            iface->super.super.super.config.seg_size);
    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    UCT_RC_IFACE_FILL_TM_IMM(imm, app_ctx, ib_imm, opcode, MLX5_OPCODE_SEND, _IMM);

    uct_dc_mlx5_iface_zcopy_post(iface, ep, opcode|UCT_RC_MLX5_OPCODE_FLAG_TM,
                                 iov, iovcnt, 0, "", 0, 0, 0, tag, app_ctx,
                                 ib_imm, comp, MLX5_WQE_CTRL_SOLICITED);

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
    unsigned tm_hdr_len        = sizeof(struct ibv_exp_tmh) +
                                 sizeof(struct ibv_exp_tmh_rvh) +
                                 sizeof(struct ibv_exp_tmh_ravh);
    struct ibv_exp_tmh_ravh ravh;
    uint32_t op_index;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_RC_IFACE_CHECK_RNDV_PARAMS(iovcnt, header_length, tm_hdr_len,
                                   UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE),
                                   iface->super.super.tm.max_rndv_data +
                                   UCT_RC_IFACE_TMH_PRIV_LEN);
    UCT_DC_CHECK_RES_PTR(&iface->super, &ep->super);

    op_index = uct_rc_iface_tag_get_op_id(&iface->super.super, comp);

    uct_dc_iface_fill_ravh(&ravh, iface->super.rx.dct->dct_num);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, &ep->super, txqp, txwq);

    uct_rc_mlx5_txqp_tag_inline_post(&iface->super.super, IBV_EXP_QPT_DC_INI,
                                     txqp, txwq, MLX5_OPCODE_SEND, header,
                                     header_length, iov, tag, op_index,
                                     IBV_EXP_TMH_RNDV, 0, &ep->av,
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

    UCT_CHECK_LENGTH(header_length + sizeof(struct ibv_exp_tmh), 0,
                     UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE),
                     "tag_rndv_request");
    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, &ep->super, txqp, txwq);

    uct_rc_mlx5_txqp_tag_inline_post(&iface->super.super, IBV_EXP_QPT_DC_INI,
                                     txqp, txwq, MLX5_OPCODE_SEND_IMM, header,
                                     header_length, NULL, tag, 0,
                                     IBV_EXP_TMH_EAGER, 0, &ep->av,
                                     uct_dc_mlx5_ep_get_grh(ep),
                                     uct_ib_mlx5_wqe_av_size(&ep->av), NULL, 0,
                                     MLX5_WQE_CTRL_SOLICITED);
    return UCS_OK;
}

static ucs_status_t uct_dc_mlx5_iface_tag_recv_zcopy(uct_iface_h tl_iface,
                                                     uct_tag_t tag,
                                                     uct_tag_t tag_mask,
                                                     const uct_iov_t *iov,
                                                     size_t iovcnt,
                                                     uct_tag_context_t *ctx)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);

    return uct_rc_mlx5_iface_common_tag_recv(&iface->mlx5_common,
                                             &iface->super.super, tag, tag_mask,
                                             iov, iovcnt, ctx);
}

static ucs_status_t uct_dc_mlx5_iface_tag_recv_cancel(uct_iface_h tl_iface,
                                                      uct_tag_context_t *ctx,
                                                      int force)
{
   uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);

   return uct_rc_mlx5_iface_common_tag_recv_cancel(&iface->mlx5_common,
                                                   &iface->super.super, ctx, force);
}

static unsigned uct_dc_mlx5_iface_progress_tm(void *arg)
{
    uct_dc_mlx5_iface_t *iface = arg;
    unsigned count;

    count = uct_rc_mlx5_iface_common_poll_rx(&iface->mlx5_common,
                                             &iface->super.super, 1);
    if (count > 0) {
        return count;
    }
    return uct_dc_mlx5_poll_tx(iface);
}
#endif

static void uct_dc_mlx5_iface_progress_enable(uct_iface_h tl_iface, unsigned flags)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);

    uct_base_iface_progress_enable_cb(&iface->super.super, iface->progress, flags);
}

static void uct_dc_mlx5_iface_handle_failure(uct_ib_iface_t *ib_iface,
                                             void *arg, ucs_status_t status)
{
    struct mlx5_cqe64 *cqe    = arg;
    uint32_t          qp_num  = ntohl(cqe->sop_drop_qpn) &
                                UCS_MASK(UCT_IB_QPN_ORDER);
    ucs_log_level_t   log_lvl = UCS_LOG_LEVEL_FATAL;

    if (uct_dc_handle_failure(ib_iface, qp_num, status) == UCS_OK) {
        log_lvl = ib_iface->super.config.failure_level;
    }

    uct_ib_mlx5_completion_with_err(arg, log_lvl);
}

static ucs_status_t uct_dc_mlx5_ep_set_failed(uct_ib_iface_t *ib_iface,
                                              uct_ep_h ep, ucs_status_t status)
{
    return uct_set_ep_failed(&UCS_CLASS_NAME(uct_dc_mlx5_ep_t), ep,
                             &ib_iface->super.super, status);
}

ucs_status_t uct_dc_mlx5_ep_fc_ctrl(uct_ep_t *tl_ep, unsigned op,
                                    uct_rc_fc_request_t *req)
{
    uintptr_t sender_ep;
    uct_dc_fc_sender_data_t sender;
    uct_ib_mlx5_base_av_t av;
    uct_dc_fc_request_t *dc_req;
    uct_dc_mlx5_ep_t *dc_mlx5_ep;
    uct_dc_ep_t *dc_ep         = ucs_derived_of(tl_ep, uct_dc_ep_t);
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_dc_mlx5_iface_t);
    uct_ib_iface_t *ib_iface = &iface->super.super.super;
    struct ibv_ah_attr ah_attr = {.is_global = 0};
    struct ibv_ah *ah;
    struct mlx5_wqe_av mlx5_av;
    ucs_status_t status;

    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    ucs_assert((sizeof(uint8_t) + sizeof(sender_ep)) <=
                UCT_IB_MLX5_AV_FULL_SIZE);

    UCT_DC_CHECK_RES(&iface->super, dc_ep);
    UCT_DC_MLX5_IFACE_TXQP_GET(iface, dc_ep, txqp, txwq);

    dc_req = ucs_derived_of(req, uct_dc_fc_request_t);

    if (op == UCT_RC_EP_FC_PURE_GRANT) {
        ucs_assert(req != NULL);

        sender_ep = (uintptr_t)dc_req->sender.ep;

        /* TODO: look at common code with uct_ud_mlx5_iface_get_av */
        if (dc_req->sender.global.is_global) {
            uct_ib_iface_fill_ah_attr_from_gid_lid(ib_iface, dc_req->lid,
                                                   ucs_unaligned_ptr(&dc_req->sender.global.gid),
                                                   ib_iface->path_bits[0], &ah_attr);

            status = uct_ib_iface_create_ah(ib_iface, &ah_attr, &ah);
            if (status != UCS_OK) {
                return status;
            }

            uct_ib_mlx5_get_av(ah, &mlx5_av);

            ibv_destroy_ah(ah);
        }

        /* Note av initialization is copied from exp verbs */
        av.stat_rate_sl = ib_iface->config.sl; /* (attr->static_rate << 4) | attr->sl */
        av.fl_mlid      = ib_iface->path_bits[0] & 0x7f;

        /* lid in dc_req is in BE already  */
        av.rlid         = dc_req->lid | htons(ib_iface->path_bits[0]);
        av.dqp_dct      = htonl(dc_req->dct_num);

        if (!iface->ud_common.config.compact_av || ah_attr.is_global) {
            av.dqp_dct |= UCT_IB_MLX5_EXTENDED_UD_AV;
        }

        uct_rc_mlx5_txqp_inline_post(&iface->super.super, IBV_EXP_QPT_DC_INI,
                                     txqp, txwq, MLX5_OPCODE_SEND,
                                     &av /*dummy*/, 0, op, sender_ep, 0,
                                     0, 0,
                                     &av, ah_attr.is_global ? mlx5_av_grh(&mlx5_av) : NULL,
                                     uct_ib_mlx5_wqe_av_size(&av), 0, INT_MAX);
    } else {
        ucs_assert(op == UCT_RC_EP_FC_FLAG_HARD_REQ);
        dc_mlx5_ep              = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
        sender.ep               = (uint64_t)dc_ep;
        sender.global.gid       = ib_iface->gid;
        sender.global.is_global = dc_mlx5_ep->super.flags & UCT_DC_EP_FLAG_GRH;

        UCS_STATS_UPDATE_COUNTER(dc_ep->fc.stats,
                                 UCT_RC_FC_STAT_TX_HARD_REQ, 1);

        uct_rc_mlx5_txqp_inline_post(&iface->super.super, IBV_EXP_QPT_DC_INI,
                                     txqp, txwq, MLX5_OPCODE_SEND_IMM,
                                     &sender.global, sizeof(sender.global), op, sender.ep,
                                     iface->super.rx.dct->dct_num,
                                     0, 0,
                                     &dc_mlx5_ep->av,
                                     uct_dc_mlx5_ep_get_grh(dc_mlx5_ep),
                                     uct_ib_mlx5_wqe_av_size(&dc_mlx5_ep->av),
                                     MLX5_WQE_CTRL_SOLICITED, INT_MAX);
    }

    return UCS_OK;
}


static void UCS_CLASS_DELETE_FUNC_NAME(uct_dc_mlx5_iface_t)(uct_iface_t*);

static ucs_status_t uct_dc_mlx5_iface_reset_dci(uct_dc_iface_t *dc_iface, int dci)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(dc_iface, uct_dc_mlx5_iface_t);
    ucs_status_t status;

    ucs_debug("iface %p reset dci[%d]", iface, dci);

    /* Synchronize CQ index with the driver, since it would remove pending
     * completions for this QP (both send and receive) during ibv_destroy_qp().
     */
    uct_rc_mlx5_iface_common_update_cqs_ci(&iface->mlx5_common,
                                           &iface->super.super.super);
    status = uct_ib_modify_qp(iface->super.tx.dcis[dci].txqp.qp, IBV_QPS_RESET);
    uct_rc_mlx5_iface_common_sync_cqs_ci(&iface->mlx5_common,
                                         &iface->super.super.super);

    uct_rc_mlx5_iface_commom_clean(&iface->mlx5_common.cq[UCT_IB_DIR_TX], NULL,
                                   iface->super.tx.dcis[dci].txqp.qp->qp_num);

    /* Resume posting from to the beginning of the QP */
    uct_ib_mlx5_txwq_reset(&iface->dci_wqs[dci]);

    return status;
}

static void uct_dc_mlx5_iface_event_cq(uct_ib_iface_t *ib_iface,
                                       uct_ib_dir_t dir)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ib_iface, uct_dc_mlx5_iface_t);

    iface->mlx5_common.cq[dir].cq_sn++;
}

static uct_dc_iface_ops_t uct_dc_mlx5_iface_ops = {
    {
    {
    {
    .ep_put_short             = uct_dc_mlx5_ep_put_short,
    .ep_put_bcopy             = uct_dc_mlx5_ep_put_bcopy,
    .ep_put_zcopy             = uct_dc_mlx5_ep_put_zcopy,
    .ep_get_bcopy             = uct_dc_mlx5_ep_get_bcopy,
    .ep_get_zcopy             = uct_dc_mlx5_ep_get_zcopy,
    .ep_am_short              = uct_dc_mlx5_ep_am_short,
    .ep_am_bcopy              = uct_dc_mlx5_ep_am_bcopy,
    .ep_am_zcopy              = uct_dc_mlx5_ep_am_zcopy,
    .ep_atomic_cswap64        = uct_dc_mlx5_ep_atomic_cswap64,
    .ep_atomic_cswap32        = uct_dc_mlx5_ep_atomic_cswap32,
    .ep_atomic64_post         = uct_dc_mlx5_ep_atomic64_post,
    .ep_atomic32_post         = uct_dc_mlx5_ep_atomic32_post,
    .ep_atomic64_fetch        = uct_dc_mlx5_ep_atomic64_fetch,
    .ep_atomic32_fetch        = uct_dc_mlx5_ep_atomic32_fetch,
    .ep_pending_add           = uct_dc_ep_pending_add,
    .ep_pending_purge         = uct_dc_ep_pending_purge,
    .ep_flush                 = uct_dc_mlx5_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
#if IBV_EXP_HW_TM_DC
    .ep_tag_eager_short       = uct_dc_mlx5_ep_tag_eager_short,
    .ep_tag_eager_bcopy       = uct_dc_mlx5_ep_tag_eager_bcopy,
    .ep_tag_eager_zcopy       = uct_dc_mlx5_ep_tag_eager_zcopy,
    .ep_tag_rndv_zcopy        = uct_dc_mlx5_ep_tag_rndv_zcopy,
    .ep_tag_rndv_request      = uct_dc_mlx5_ep_tag_rndv_request,
    .ep_tag_rndv_cancel       = uct_rc_ep_tag_rndv_cancel,
    .iface_tag_recv_zcopy     = uct_dc_mlx5_iface_tag_recv_zcopy,
    .iface_tag_recv_cancel    = uct_dc_mlx5_iface_tag_recv_cancel,
#endif
    .iface_flush              = uct_dc_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_dc_mlx5_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_rc_iface_do_progress,
    .iface_event_fd_get       = uct_ib_iface_event_fd_get,
    .iface_event_arm          = uct_rc_iface_event_arm,
    .ep_create_connected      = uct_dc_mlx5_ep_create_connected,
    .ep_destroy               = uct_dc_mlx5_ep_destroy,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_dc_mlx5_iface_t),
    .iface_query              = uct_dc_mlx5_iface_query,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_dc_iface_is_reachable,
    .iface_get_address        = uct_dc_iface_get_address,
    },
    .arm_cq                   = uct_ib_iface_arm_cq,
    .event_cq                 = uct_dc_mlx5_iface_event_cq,
    .handle_failure           = uct_dc_mlx5_iface_handle_failure,
    .set_ep_failed            = uct_dc_mlx5_ep_set_failed
    },
    .fc_ctrl                  = uct_dc_mlx5_ep_fc_ctrl,
    .fc_handler               = uct_dc_iface_fc_handler,
    },
    .reset_dci                = uct_dc_mlx5_iface_reset_dci
};


static ucs_status_t uct_dc_mlx5_iface_init_dcis(uct_dc_mlx5_iface_t *iface)
{
    ucs_status_t status;
    uint16_t bb_max;
    int i;

    bb_max = 0;
    for (i = 0; i < iface->super.tx.ndci; i++) {
        status = uct_ib_mlx5_txwq_init(iface->super.super.super.super.worker,
                                       &iface->dci_wqs[i],
                                       iface->super.tx.dcis[i].txqp.qp);
        if (status != UCS_OK) {
            return status;
        }


        bb_max = iface->dci_wqs[i].bb_max;
        uct_rc_txqp_available_set(&iface->super.tx.dcis[i].txqp, bb_max);
    }

    iface->super.super.config.tx_qp_len = bb_max;
    return UCS_OK;
}

static void uct_dc_mlx5_iface_cleanup_dcis(uct_dc_mlx5_iface_t *iface)
{
    int i;

    for (i = 0; i < iface->super.tx.ndci; i++) {
        uct_ib_mlx5_txwq_cleanup(&iface->dci_wqs[i]);
    }
}

static ucs_status_t uct_dc_mlx5_iface_tag_init(uct_dc_mlx5_iface_t *iface,
                                               uct_rc_iface_config_t *rc_config)
{
#if IBV_EXP_HW_TM_DC
    if (UCT_RC_IFACE_TM_ENABLED(&iface->super.super)) {
        struct ibv_exp_create_srq_attr srq_init_attr = {};
        struct ibv_exp_srq_dc_offload_params dc_op   = {};
        ucs_status_t status;

        uct_dc_iface_fill_xrq_init_attrs(&iface->super.super, &srq_init_attr, &dc_op);

        status = uct_rc_mlx5_iface_common_tag_init(&iface->mlx5_common,
                                                   &iface->super.super, rc_config,
                                                   &srq_init_attr,
                                                   sizeof(struct ibv_exp_tmh_rvh) +
                                                   sizeof(struct ibv_exp_tmh_ravh));
        if (status != UCS_OK) {
            return status;
        }

        /* TM XRQ is ready, can create DCT now */
        status = uct_dc_iface_create_dct(&iface->super);
        if (status != UCS_OK) {
            uct_rc_mlx5_iface_common_tag_cleanup(&iface->mlx5_common,
                                                 &iface->super.super);
            return status;
        }
        iface->super.super.progress = uct_dc_mlx5_iface_progress_tm;
    } else
#endif
    {
        iface->super.super.progress = uct_dc_mlx5_iface_progress;
    }
    return UCS_OK;
}

static void uct_dc_mlx5_iface_tag_cleanup(uct_dc_mlx5_iface_t *iface)
{
    if (UCT_RC_IFACE_TM_ENABLED(&iface->super.super)) {
        ibv_exp_destroy_dct(iface->super.rx.dct);
        iface->super.rx.dct = NULL;
    }

    uct_rc_mlx5_iface_common_tag_cleanup(&iface->mlx5_common, &iface->super.super);
}

static UCS_CLASS_INIT_FUNC(uct_dc_mlx5_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_dc_mlx5_iface_config_t *config = ucs_derived_of(tl_config,
                                                        uct_dc_mlx5_iface_config_t);
    uct_ib_iface_init_attr_t init_attr = {};
    ucs_status_t status;

    ucs_trace_func("");

    init_attr.res_domain_key = UCT_IB_MLX5_RES_DOMAIN_KEY;
    init_attr.tm_cap_bit     = IBV_EXP_TM_CAP_DC;
    init_attr.flags          = UCT_IB_CQ_IGNORE_OVERRUN;

    UCS_CLASS_CALL_SUPER_INIT(uct_dc_iface_t, &uct_dc_mlx5_iface_ops, md,
                              worker, params, &config->super, &init_attr);

    status = uct_dc_mlx5_iface_tag_init(self, &config->super.super);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_rc_mlx5_iface_common_init(&self->mlx5_common, &self->super.super,
                                           &config->super.super, &config->mlx5_common);
    if (status != UCS_OK) {
        goto err_rc_mlx5_tag_cleanup;
    }

    status = uct_ud_mlx5_iface_common_init(&self->super.super.super,
                                           &self->ud_common, &config->ud_common);
    if (status != UCS_OK) {
        goto err_rc_mlx5_common_cleanup;
    }

    status = uct_dc_mlx5_iface_init_dcis(self);
    if (status != UCS_OK) {
        goto err_rc_mlx5_common_cleanup;
    }

    uct_dc_iface_set_quota(&self->super, &config->super);
    /* Set max_iov for put_zcopy and get_zcopy */
    uct_ib_iface_set_max_iov(&self->super.super.super,
                             (UCT_IB_MLX5_MAX_SEND_WQE_SIZE -
                             sizeof(struct mlx5_wqe_raddr_seg) -
                             sizeof(struct mlx5_wqe_ctrl_seg) -
                             UCT_IB_MLX5_AV_FULL_SIZE) /
                             sizeof(struct mlx5_wqe_data_seg));

    uct_rc_mlx5_iface_common_prepost_recvs(&self->super.super,
                                           &self->mlx5_common);

    ucs_debug("created dc iface %p", self);
    return UCS_OK;

err_rc_mlx5_common_cleanup:
    uct_rc_mlx5_iface_common_cleanup(&self->mlx5_common);
err_rc_mlx5_tag_cleanup:
    uct_dc_mlx5_iface_tag_cleanup(self);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_mlx5_iface_t)
{
    ucs_trace_func("");
    uct_base_iface_progress_disable(&self->super.super.super.super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    uct_dc_mlx5_iface_cleanup_dcis(self);
    uct_rc_mlx5_iface_common_cleanup(&self->mlx5_common);
    uct_dc_mlx5_iface_tag_cleanup(self);
}

UCS_CLASS_DEFINE(uct_dc_mlx5_iface_t, uct_dc_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_dc_mlx5_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_dc_mlx5_iface_t, uct_iface_t);

static
ucs_status_t uct_dc_mlx5_query_resources(uct_md_h md,
                                         uct_tl_resource_desc_t **resources_p,
                                         unsigned *num_resources_p)
{
    uct_ib_md_t *ib_md = ucs_derived_of(md, uct_ib_md_t);

    return uct_dc_device_query_tl_resources(&ib_md->dev,"dc_mlx5",
                                            UCT_IB_DEVICE_FLAG_MLX5_PRM |
                                            (ib_md->config.eth_pause ? 0 : UCT_IB_DEVICE_FLAG_LINK_IB),
                                            resources_p, num_resources_p);
}


UCT_TL_COMPONENT_DEFINE(uct_dc_mlx5_tl,
                        uct_dc_mlx5_query_resources,
                        uct_dc_mlx5_iface_t,
                        "dc_mlx5",
                        "DC_MLX5_",
                        uct_dc_mlx5_iface_config_table,
                        uct_dc_mlx5_iface_config_t);
UCT_MD_REGISTER_TL(&uct_ib_mdc, &uct_dc_mlx5_tl);
