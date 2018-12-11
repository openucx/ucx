/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "dc_mlx5.h"
#include "dc_mlx5_ep.h"

#include <uct/api/uct.h>
#include <uct/ib/rc/accel/rc_mlx5.inl>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_log.h>
#include <uct/ib/mlx5/ib_mlx5_log.h>
#include <uct/base/uct_md.h>
#include <ucs/arch/bitops.h>
#include <ucs/arch/cpu.h>
#include <ucs/async/async.h>
#include <ucs/debug/log.h>
#include <string.h>


ucs_config_field_t uct_dc_mlx5_iface_config_sub_table[] = {
    {"RC_", "IB_TX_QUEUE_LEN=128;FC_ENABLE=y;", NULL,
     ucs_offsetof(uct_dc_mlx5_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_rc_iface_config_table)},

    {"", "", NULL,
     ucs_offsetof(uct_dc_mlx5_iface_config_t, ud_common),
     UCS_CONFIG_TYPE_TABLE(uct_ud_iface_common_config_table)},

    {"NUM_DCI", "8",
     "Number of DC initiator QPs (DCI) used by the interface "
     "(up to " UCS_PP_QUOTE(UCT_DC_MLX5_IFACE_MAX_DCIS) ").",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, ndci), UCS_CONFIG_TYPE_UINT},

    {"TX_POLICY", "dcs_quota",
     "Specifies how DC initiator (DCI) is selected by the endpoint. The policies are:\n"
     "\n"
     "dcs        The endpoint either uses already assigned DCI or one is allocated\n"
     "           in a LIFO order, and released once it has no outstanding operations.\n"
     "\n"
     "dcs_quota  Same as \"dcs\" but in addition the DCI is scheduled for release\n"
     "           if it has sent more than quota, and there are endpoints waiting for a DCI.\n"
     "           The dci is released once it completes all outstanding operations.\n"
     "           This policy ensures that there will be no starvation among endpoints.",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, tx_policy),
     UCS_CONFIG_TYPE_ENUM(uct_dc_tx_policy_names)},

    {"QUOTA", "32",
     "When \"dcs_quota\" policy is selected, how much to send from a DCI when\n"
     "there are other endpoints waiting for it.",
     ucs_offsetof(uct_dc_mlx5_iface_config_t, quota), UCS_CONFIG_TYPE_UINT},

    {NULL}
};

ucs_config_field_t uct_dc_mlx5_iface_config_table[] = {
    {"DC_", "", NULL, 0,
     UCS_CONFIG_TYPE_TABLE(uct_dc_mlx5_iface_config_sub_table)},

    {"", "", NULL,
     ucs_offsetof(uct_dc_mlx5_iface_config_t, mlx5_ud),
     UCS_CONFIG_TYPE_TABLE(uct_ud_mlx5_iface_common_config_table)},

    {"", "", NULL,
     ucs_offsetof(uct_dc_mlx5_iface_config_t, mlx5_common),
     UCS_CONFIG_TYPE_TABLE(uct_ib_mlx5_iface_config_table)},

    {NULL}
};


static ucs_status_t
uct_dc_mlx5_ep_create_connected(uct_iface_h tl_iface,
                                const uct_device_addr_t *dev_addr,
                                const uct_iface_addr_t *iface_addr,
                                uct_ep_h* ep_p)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    const uct_ib_address_t *ib_addr = (const uct_ib_address_t *)dev_addr;
    const uct_dc_mlx5_iface_addr_t *if_addr = (const uct_dc_mlx5_iface_addr_t *)iface_addr;
    ucs_status_t status;
    int is_global;
    uct_ib_mlx5_base_av_t av;
    struct mlx5_grh_av grh_av;

    ucs_trace_func("");
    status = uct_ud_mlx5_iface_get_av(&iface->super.super, &iface->ud_common,
                                      ib_addr, iface->super.super.path_bits[0],
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
    uct_dc_mlx5_ep_cleanup(tl_ep, &UCS_CLASS_NAME(uct_dc_mlx5_ep_t));
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

    status = uct_rc_iface_query(&iface->super, iface_attr,
                                max_put_inline,
                                max_am_inline,
                                UCT_IB_MLX5_AM_ZCOPY_MAX_HDR(UCT_IB_MLX5_AV_FULL_SIZE),
                                UCT_IB_MLX5_AM_ZCOPY_MAX_IOV,
                                UCT_RC_MLX5_TM_EAGER_ZCOPY_MAX_IOV(UCT_IB_MLX5_AV_FULL_SIZE));
    if (status != UCS_OK) {
        return status;
    }

    /* fixup flags and address lengths */
    iface_attr->cap.flags &= ~UCT_IFACE_FLAG_CONNECT_TO_EP;
    iface_attr->cap.flags |= UCT_IFACE_FLAG_CONNECT_TO_IFACE;
    iface_attr->ep_addr_len       = 0;
    iface_attr->max_conn_priv     = 0;
    iface_attr->iface_addr_len    = sizeof(uct_dc_mlx5_iface_addr_t);
    iface_attr->latency.overhead += 60e-9; /* connect packet + cqe */

    uct_rc_mlx5_iface_common_query(&iface->super.super, iface_attr);
    return UCS_OK;
}

#define UCT_DC_MLX5_TXQP_DECL(_txqp, _txwq) \
    uct_rc_txqp_t *_txqp; \
    uct_ib_mlx5_txwq_t *_txwq;

#define UCT_DC_MLX5_IFACE_TXQP_GET(_iface, _ep, _txqp, _txwq) \
{ \
    uint8_t dci; \
    dci = (_ep)->dci; \
    _txqp = &(_iface)->tx.dcis[dci].txqp; \
    _txwq = &(_iface)->tx.dci_wqs[dci]; \
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

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);
    desc->super.sn = txwq->sw_pi;
    uct_rc_mlx5_txqp_dptr_post(&iface->super, UCT_IB_QPT_DCI, txqp, txwq,
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

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);

    sn = txwq->sw_pi;
    uct_rc_mlx5_txqp_dptr_post_iov(&iface->super, UCT_IB_QPT_DCI, txqp,
                                   txwq, opcode, iov, iovcnt,
                                   am_id, am_hdr, am_hdr_len,
                                   rdma_raddr, uct_ib_md_direct_rkey(rdma_rkey),
                                   tag, app_ctx, ib_imm_be,
                                   &ep->av, uct_dc_mlx5_ep_get_grh(ep),
                                   uct_ib_mlx5_wqe_av_size(&ep->av),
                                   MLX5_WQE_CTRL_CQ_UPDATE | send_flags,
                                   UCT_IB_MAX_ZCOPY_LOG_SGE(&iface->super.super));

    uct_rc_txqp_add_send_comp(&iface->super, txqp, comp, sn,
                              UCT_RC_IFACE_SEND_OP_FLAG_ZCOPY);
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

static inline void uct_dc_mlx5_iface_add_send_comp(uct_dc_mlx5_iface_t *iface,
                                                   uct_dc_mlx5_ep_t *ep,
                                                   uct_completion_t *comp)
{
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);
    uct_rc_txqp_add_send_comp(&iface->super, txqp, comp, txwq->sig_pi, 0);
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

    UCT_RC_IFACE_GET_TX_ATOMIC_DESC(&iface->super, &iface->mlx5_common.tx.atomic_desc_mp, desc);
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
    UCT_RC_IFACE_GET_TX_ATOMIC_FETCH_DESC(&iface->super, &iface->mlx5_common.tx.atomic_desc_mp,
                                          desc, uct_rc_iface_atomic_handler(&iface->super,
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

    UCT_RC_UPDATE_FC_WND(&iface->super, &ep->fc);
    UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, sizeof(hdr) + length);
    return UCS_OK;
}

#if HAVE_IBV_EXP_DM
static ucs_status_t UCS_F_ALWAYS_INLINE
uct_dc_mlx5_ep_short_dm(uct_dc_mlx5_ep_t *ep, uct_rc_mlx5_dm_copy_data_t *cache,
                        size_t hdr_len, const void *payload, unsigned length,
                        unsigned opcode, uint8_t fm_ce_se,
                        uint64_t rdma_raddr, uct_rkey_t rdma_rkey)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_dc_mlx5_iface_t);
    uct_rc_iface_send_desc_t *desc;
    void *buffer;
    ucs_status_t status;
    uct_ib_log_sge_t log_sge;

    status = uct_rc_mlx5_common_dm_make_data(&iface->mlx5_common, &iface->super,
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
    UCT_DC_CHECK_RES_AND_FC(iface, ep);

    uct_rc_am_hdr_fill(&cache.am_hdr.rc_hdr, id);
    cache.am_hdr.am_hdr = hdr;

    status = uct_dc_mlx5_ep_short_dm(ep, &cache, sizeof(cache.am_hdr), buffer, length,
                                     MLX5_OPCODE_SEND,
                                     MLX5_WQE_CTRL_SOLICITED | MLX5_WQE_CTRL_CQ_UPDATE,
                                     0, 0);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }
    UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, sizeof(cache.am_hdr) + length);
    UCT_RC_UPDATE_FC_WND(&iface->super, &ep->fc);
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

    UCT_DC_CHECK_RES_AND_FC(iface, ep);
    UCT_RC_IFACE_GET_TX_AM_BCOPY_DESC(&iface->super, &iface->super.tx.mp, desc,
                                      id, pack_cb, arg, &length);

    uct_dc_mlx5_iface_bcopy_post(iface, ep, MLX5_OPCODE_SEND,
                                 sizeof(uct_rc_hdr_t) + length, 0, 0, desc,
                                 MLX5_WQE_CTRL_SOLICITED, 0, desc + 1, NULL);

    UCT_RC_UPDATE_FC_WND(&iface->super, &ep->fc);
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
                               iface->super.super.config.seg_size,
                               UCT_IB_MLX5_AV_FULL_SIZE);
    UCT_DC_CHECK_RES_AND_FC(iface, ep);

    uct_dc_mlx5_iface_zcopy_post(iface, ep, MLX5_OPCODE_SEND, iov, iovcnt,
                                 id, header, header_length, 0, 0, 0ul, 0, 0,
                                 comp, MLX5_WQE_CTRL_SOLICITED);

    UCT_RC_UPDATE_FC_WND(&iface->super, &ep->fc);
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
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_RC_MLX5_CHECK_PUT_SHORT(length, UCT_IB_MLX5_AV_FULL_SIZE);
    UCT_DC_MLX5_CHECK_RES(iface, ep);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);
    uct_rc_mlx5_txqp_inline_post(&iface->super, UCT_IB_QPT_DCI,
                                 txqp, txwq,
                                 MLX5_OPCODE_RDMA_WRITE,
                                 buffer, length, 0, 0, 0,
                                 remote_addr, uct_ib_md_direct_rkey(rkey),
                                 &ep->av, uct_dc_mlx5_ep_get_grh(ep),
                                 uct_ib_mlx5_wqe_av_size(&ep->av), 0, INT_MAX);

    UCT_TL_EP_STAT_OP(&ep->super, PUT, SHORT, length);

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
    UCT_DC_MLX5_CHECK_RES(iface, ep);
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
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;
    size_t length;

    UCT_DC_MLX5_CHECK_RES(iface, ep);
    UCT_RC_IFACE_GET_TX_PUT_BCOPY_DESC(&iface->super, &iface->super.tx.mp,
                                       desc, pack_cb, arg, length);
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
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);

    UCT_CHECK_IOV_SIZE(iovcnt, uct_ib_iface_get_max_iov(&iface->super.super),
                       "uct_dc_mlx5_ep_put_zcopy");
    UCT_CHECK_LENGTH(uct_iov_total_length(iov, iovcnt), 0, UCT_IB_MAX_MESSAGE_SIZE,
                     "put_zcopy");
    UCT_DC_MLX5_CHECK_RES(iface, ep);

    uct_dc_mlx5_iface_zcopy_post(iface, ep, MLX5_OPCODE_RDMA_WRITE, iov, iovcnt,
                                 0, NULL, 0, remote_addr, rkey, 0ul, 0, 0,
                                 comp, 0);

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
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_CHECK_LENGTH(length, 0, iface->super.super.config.seg_size, "get_bcopy");
    UCT_DC_MLX5_CHECK_RES(iface, ep);
    UCT_RC_IFACE_GET_TX_GET_BCOPY_DESC(&iface->super, &iface->super.tx.mp,
                                       desc, unpack_cb, comp, arg, length);
    uct_dc_mlx5_iface_bcopy_post(iface, ep, MLX5_OPCODE_RDMA_READ, length,
                                 remote_addr, rkey, desc, 0, 0, desc + 1, NULL);
    UCT_TL_EP_STAT_OP(&ep->super, GET, BCOPY, length);
    return UCS_INPROGRESS;
}


ucs_status_t uct_dc_mlx5_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);

    UCT_CHECK_IOV_SIZE(iovcnt, uct_ib_iface_get_max_iov(&iface->super.super),
                       "uct_dc_mlx5_ep_get_zcopy");
    UCT_CHECK_LENGTH(uct_iov_total_length(iov, iovcnt),
                     iface->super.super.config.max_inl_resp + 1, UCT_IB_MAX_MESSAGE_SIZE,
                     "get_zcopy");
    UCT_DC_MLX5_CHECK_RES(iface, ep);

    uct_dc_mlx5_iface_zcopy_post(iface, ep, MLX5_OPCODE_RDMA_READ, iov, iovcnt,
                                 0, NULL, 0, remote_addr, rkey, 0ul, 0, 0,
                                 comp, 0);
    UCT_TL_EP_STAT_OP(&ep->super, GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    return UCS_INPROGRESS;
}

ucs_status_t uct_dc_mlx5_ep_flush(uct_ep_h tl_ep, unsigned flags, uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t    *ep    = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    ucs_status_t        status;

    if (ucs_unlikely(flags & UCT_FLUSH_FLAG_CANCEL)) {
        if (ep->dci != UCT_DC_MLX5_EP_NO_DCI) {
            uct_rc_txqp_purge_outstanding(&iface->tx.dcis[ep->dci].txqp,
                                          UCS_ERR_CANCELED, 0);
#if ENABLE_ASSERT
            iface->tx.dcis[ep->dci].flags |= UCT_DC_DCI_FLAG_EP_CANCELED;
#endif
        }

        uct_ep_pending_purge(tl_ep, NULL, 0);
        return UCS_OK;
    }

    if (!uct_rc_iface_has_tx_resources(&iface->super)) {
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
    UCT_TL_EP_STAT_FLUSH_WAIT(&ep->super);

    ucs_assert(ep->dci != UCT_DC_MLX5_EP_NO_DCI);
    uct_dc_mlx5_iface_add_send_comp(iface, ep, comp);

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

    cqe = uct_ib_mlx5_poll_cq(&iface->super.super,
                              &iface->mlx5_common.cq[UCT_IB_DIR_TX]);
    if (cqe == NULL) {
        return 0;
    }
    UCS_STATS_UPDATE_COUNTER(iface->super.stats, UCT_RC_IFACE_STAT_TX_COMPLETION, 1);

    ucs_memory_cpu_load_fence();

    qp_num = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
    dci = uct_dc_mlx5_iface_dci_find(iface, qp_num);
    txqp = &iface->tx.dcis[dci].txqp;
    txwq = &iface->tx.dci_wqs[dci];
    hw_ci = ntohs(cqe->wqe_counter);

    ucs_trace_poll("dc iface %p tx_cqe: dci[%d] qpn 0x%x txqp %p hw_ci %d",
                   iface, dci, qp_num, txqp, hw_ci);

    uct_rc_mlx5_common_update_tx_res(&iface->super, txwq, txqp, hw_ci);
    uct_dc_mlx5_iface_dci_put(iface, dci);
    uct_rc_mlx5_txqp_process_tx_cqe(txqp, cqe, hw_ci);

    uct_dc_mlx5_iface_progress_pending(iface);
    return 1;
}

static unsigned uct_dc_mlx5_iface_progress(void *arg)
{
    uct_dc_mlx5_iface_t *iface = arg;
    unsigned count;

    count = uct_rc_mlx5_iface_common_poll_rx(&iface->mlx5_common,
                                             &iface->super, 0);
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
    UCT_DC_MLX5_CHECK_RES(iface, ep);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);

    uct_rc_mlx5_txqp_tag_inline_post(&iface->super, UCT_IB_QPT_DCI,
                                     txqp, txwq, MLX5_OPCODE_SEND, data, length,
                                     NULL, tag, 0, IBV_EXP_TMH_EAGER, 0,
                                     &ep->av, uct_dc_mlx5_ep_get_grh(ep),
                                     uct_ib_mlx5_wqe_av_size(&ep->av), NULL, 0,
                                     MLX5_WQE_CTRL_SOLICITED);

    UCT_TL_EP_STAT_OP(&ep->super, TAG, SHORT, length);

    return UCS_OK;
}

ucs_status_t uct_dc_mlx5_ep_tag_eager_short(uct_ep_h tl_ep, uct_tag_t tag,
                                            const void *data, size_t length)
{
#if HAVE_IBV_EXP_DM
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_rc_mlx5_dm_copy_data_t cache;
    ucs_status_t status;

    if (ucs_likely((sizeof(struct ibv_exp_tmh) + length <=
                    UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE)) ||
                   !iface->mlx5_common.dm.dm)) {
#endif
        return uct_dc_mlx5_ep_tag_eager_short_inline(tl_ep, tag, data, length);
#if HAVE_IBV_EXP_DM
    }

    UCT_CHECK_LENGTH(length + sizeof(struct ibv_exp_tmh), 0,
                     iface->mlx5_common.dm.seg_len, "tag_short");
    UCT_DC_MLX5_CHECK_RES(iface, ep);

    uct_rc_mlx5_fill_tmh(ucs_unaligned_ptr(&cache.tm_hdr), tag, 0, IBV_EXP_TMH_EAGER);

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

    UCT_RC_IFACE_FILL_TM_IMM(imm, app_ctx, ib_imm, opcode, MLX5_OPCODE_SEND, _IMM);

    UCT_RC_MLX5_IFACE_GET_TM_BCOPY_DESC(&iface->super,
                                        &iface->super.tx.mp, desc, tag,
                                        app_ctx, pack_cb, arg, length);

    uct_dc_mlx5_iface_bcopy_post(iface, ep, opcode,
                                 sizeof(struct ibv_exp_tmh) + length,
                                 0, 0, desc, MLX5_WQE_CTRL_SOLICITED, ib_imm, desc + 1, NULL);

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
    UCT_RC_CHECK_ZCOPY_DATA(sizeof(struct ibv_exp_tmh),
                            uct_iov_total_length(iov, iovcnt),
                            iface->super.super.config.seg_size);
    UCT_DC_MLX5_CHECK_RES(iface, ep);

    UCT_RC_IFACE_FILL_TM_IMM(imm, app_ctx, ib_imm, opcode, MLX5_OPCODE_SEND, _IMM);

    uct_dc_mlx5_iface_zcopy_post(iface, ep, opcode|UCT_RC_MLX5_OPCODE_FLAG_TM,
                                 iov, iovcnt, 0, "", 0, 0, 0, tag, app_ctx,
                                 ib_imm, comp, MLX5_WQE_CTRL_SOLICITED);

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
    unsigned tm_hdr_len        = sizeof(struct ibv_exp_tmh) +
                                 sizeof(struct ibv_exp_tmh_rvh) +
                                 sizeof(struct ibv_exp_tmh_ravh);
    struct ibv_exp_tmh_ravh ravh;
    uint32_t op_index;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_RC_IFACE_CHECK_RNDV_PARAMS(iovcnt, header_length, tm_hdr_len,
                                   UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE),
                                   iface->super.tm.max_rndv_data +
                                   UCT_RC_IFACE_TMH_PRIV_LEN);
    UCT_DC_CHECK_RES_PTR(iface, ep);

    op_index = uct_rc_iface_tag_get_op_id(&iface->super, comp);

    uct_dc_mlx5_iface_fill_ravh(&ravh, uct_dc_mlx5_get_dct_num(iface));

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);

    uct_rc_mlx5_txqp_tag_inline_post(&iface->super, UCT_IB_QPT_DCI,
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
    UCT_DC_MLX5_CHECK_RES(iface, ep);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);

    uct_rc_mlx5_txqp_tag_inline_post(&iface->super, UCT_IB_QPT_DCI,
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
                                             &iface->super, tag, tag_mask,
                                             iov, iovcnt, ctx);
}

static ucs_status_t uct_dc_mlx5_iface_tag_recv_cancel(uct_iface_h tl_iface,
                                                      uct_tag_context_t *ctx,
                                                      int force)
{
   uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);

   return uct_rc_mlx5_iface_common_tag_recv_cancel(&iface->mlx5_common,
                                                   &iface->super, ctx, force);
}

static unsigned uct_dc_mlx5_iface_progress_tm(void *arg)
{
    uct_dc_mlx5_iface_t *iface = arg;
    unsigned count;

    count = uct_rc_mlx5_iface_common_poll_rx(&iface->mlx5_common,
                                             &iface->super, 1);
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

    uct_ib_mlx5_completion_with_err(ib_iface, arg, log_lvl);
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
    uct_dc_mlx5_ep_t *dc_ep    = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_dc_mlx5_iface_t);
    uct_ib_iface_t *ib_iface   = &iface->super.super;
    struct ibv_ah_attr ah_attr = {.is_global = 0};
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

    UCT_DC_MLX5_CHECK_RES(iface, dc_ep);
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

        uct_rc_mlx5_txqp_inline_post(&iface->super, UCT_IB_QPT_DCI,
                                     txqp, txwq, MLX5_OPCODE_SEND,
                                     &av /*dummy*/, 0, op, sender_ep, 0,
                                     0, 0,
                                     &av, ah_attr.is_global ? mlx5_av_grh(&mlx5_av) : NULL,
                                     uct_ib_mlx5_wqe_av_size(&av), 0, INT_MAX);
    } else {
        ucs_assert(op == UCT_RC_EP_FC_FLAG_HARD_REQ);
        sender.ep               = (uint64_t)dc_ep;
        sender.global.gid       = ib_iface->gid;
        sender.global.is_global = dc_ep->flags & UCT_DC_MLX5_EP_FLAG_GRH;

        UCS_STATS_UPDATE_COUNTER(dc_ep->fc.stats,
                                 UCT_RC_FC_STAT_TX_HARD_REQ, 1);

        uct_rc_mlx5_txqp_inline_post(&iface->super, UCT_IB_QPT_DCI,
                                     txqp, txwq, MLX5_OPCODE_SEND_IMM,
                                     &sender.global, sizeof(sender.global), op, sender.ep,
                                     uct_dc_mlx5_get_dct_num(iface),
                                     0, 0,
                                     &dc_ep->av,
                                     uct_dc_mlx5_ep_get_grh(dc_ep),
                                     uct_ib_mlx5_wqe_av_size(&dc_ep->av),
                                     MLX5_WQE_CTRL_SOLICITED, INT_MAX);
    }

    return UCS_OK;
}


static void UCS_CLASS_DELETE_FUNC_NAME(uct_dc_mlx5_iface_t)(uct_iface_t*);

ucs_status_t uct_dc_mlx5_iface_reset_dci(uct_dc_mlx5_iface_t *dc_mlx5_iface, int dci)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(dc_mlx5_iface, uct_dc_mlx5_iface_t);
    ucs_status_t status;

    ucs_debug("iface %p reset dci[%d]", iface, dci);

    /* Synchronize CQ index with the driver, since it would remove pending
     * completions for this QP (both send and receive) during ibv_destroy_qp().
     */
    uct_rc_mlx5_iface_common_update_cqs_ci(&iface->mlx5_common,
                                           &iface->super.super);
    status = uct_ib_modify_qp(iface->tx.dcis[dci].txqp.qp, IBV_QPS_RESET);
    uct_rc_mlx5_iface_common_sync_cqs_ci(&iface->mlx5_common,
                                         &iface->super.super);

    uct_rc_mlx5_iface_commom_clean(&iface->mlx5_common.cq[UCT_IB_DIR_TX], NULL,
                                   iface->tx.dcis[dci].txqp.qp->qp_num);

    /* Resume posting from to the beginning of the QP */
    uct_ib_mlx5_txwq_reset(&iface->tx.dci_wqs[dci]);

    return status;
}

static void uct_dc_mlx5_iface_event_cq(uct_ib_iface_t *ib_iface,
                                       uct_ib_dir_t dir)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ib_iface, uct_dc_mlx5_iface_t);

    iface->mlx5_common.cq[dir].cq_sn++;
}

#if HAVE_DC_DV
static ucs_status_t uct_dc_mlx5_iface_create_qp(uct_ib_iface_t *iface,
                                                uct_ib_qp_attr_t *attr,
                                                struct ibv_qp **qp_p)
{
    uct_ib_device_t *dev               = uct_ib_iface_device(iface);
    struct mlx5dv_qp_init_attr dv_attr = {};
    struct ibv_qp *qp;

    uct_ib_iface_fill_attr(iface, attr);
    attr->ibv.cap.max_recv_sge          = 0;

    dv_attr.comp_mask                   = MLX5DV_QP_INIT_ATTR_MASK_DC;
    dv_attr.dc_init_attr.dc_type        = MLX5DV_DCTYPE_DCI;
    dv_attr.dc_init_attr.dct_access_key = UCT_IB_KEY;
    qp = mlx5dv_create_qp(dev->ibv_context, &attr->ibv, &dv_attr);
    if (qp == NULL) {
        ucs_error("iface=%p: failed to create DCI: %m", iface);
        return UCS_ERR_IO_ERROR;
    }

    attr->cap = attr->ibv.cap;
    *qp_p     = qp;

    return UCS_OK;
}

ucs_status_t uct_dc_mlx5_iface_dci_connect(uct_dc_mlx5_iface_t *iface,
                                           uct_rc_txqp_t *dci)
{
    struct ibv_qp_attr attr;
    long attr_mask;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state        = IBV_QPS_INIT;
    attr.pkey_index      = iface->super.super.pkey_index;
    attr.port_num        = iface->super.super.config.port_num;
    attr_mask            = IBV_QP_STATE      |
                           IBV_QP_PKEY_INDEX |
                           IBV_QP_PORT;

    if (ibv_modify_qp(dci->qp, &attr, attr_mask)) {
        ucs_error("error modifying QP to INIT : %m");
        return UCS_ERR_IO_ERROR;
    }

    /* Move QP to the RTR state */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state                   = IBV_QPS_RTR;
    attr.path_mtu                   = iface->super.config.path_mtu;
    attr.min_rnr_timer              = iface->super.config.min_rnr_timer;
    attr.max_dest_rd_atomic         = 1;
    attr.ah_attr.is_global          = iface->super.super.is_global_addr;
    attr.ah_attr.sl                 = iface->super.super.config.sl;
    attr_mask                       = IBV_QP_STATE     |
                                      IBV_QP_PATH_MTU;

    if (ibv_modify_qp(dci->qp, &attr, attr_mask)) {
        ucs_error("error modifying DCI QP to RTR: %m");
        return UCS_ERR_IO_ERROR;
    }

    /* Move QP to the RTS state */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state       = IBV_QPS_RTS;
    attr.timeout        = iface->super.config.timeout;
    attr.rnr_retry      = iface->super.config.rnr_retry;
    attr.retry_cnt      = iface->super.config.retry_cnt;
    attr.max_rd_atomic  = iface->super.config.max_rd_atomic;
    attr_mask           = IBV_QP_STATE      |
                          IBV_QP_SQ_PSN     |
                          IBV_QP_TIMEOUT    |
                          IBV_QP_RETRY_CNT  |
                          IBV_QP_RNR_RETRY  |
                          IBV_QP_MAX_QP_RD_ATOMIC;

    if (ibv_modify_qp(dci->qp, &attr, attr_mask)) {
        ucs_error("error modifying DCI QP to RTS: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t uct_dc_mlx5_iface_create_dct(uct_dc_mlx5_iface_t *iface)
{
    uct_dc_mlx5_iface_t *mlx5 = ucs_derived_of(iface, uct_dc_mlx5_iface_t);
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super.super);
    struct mlx5dv_qp_init_attr dv_init_attr = {};
    struct ibv_qp_init_attr_ex init_attr = {};
    struct ibv_qp_attr attr = {};
    int ret;

    init_attr.comp_mask             = IBV_QP_INIT_ATTR_PD;
    init_attr.pd                    = uct_ib_iface_md(&iface->super.super)->pd;
    init_attr.recv_cq               = iface->super.super.cq[UCT_IB_DIR_RX];
    /* DCT can't send, but send_cq have to point to valid CQ */
    init_attr.send_cq               = iface->super.super.cq[UCT_IB_DIR_RX];
    init_attr.srq                   = iface->super.rx.srq.srq;
    init_attr.qp_type               = IBV_QPT_DRIVER;
    init_attr.cap.max_inline_data   = iface->super.config.rx_inline;

    dv_init_attr.comp_mask                   = MLX5DV_QP_INIT_ATTR_MASK_DC;
    dv_init_attr.dc_init_attr.dc_type        = MLX5DV_DCTYPE_DCT;
    dv_init_attr.dc_init_attr.dct_access_key = UCT_IB_KEY;

    mlx5->rx_dct = mlx5dv_create_qp(dev->ibv_context,
                                    &init_attr, &dv_init_attr);
    if (mlx5->rx_dct == NULL) {
        ucs_error("Failed to created DC target %m");
        return UCS_ERR_INVALID_PARAM;
    }

    attr.pkey_index      = iface->super.super.pkey_index;
    attr.qp_state        = IBV_QPS_INIT;
    attr.port_num        = iface->super.super.config.port_num;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ  |
                           IBV_ACCESS_REMOTE_ATOMIC;

    ret = ibv_modify_qp(mlx5->rx_dct, &attr, IBV_QP_STATE |
                                             IBV_QP_PKEY_INDEX |
                                             IBV_QP_PORT |
                                             IBV_QP_ACCESS_FLAGS);

    if (ret) {
         ucs_error("error modifying DCT to INIT: %m");
         goto err;
    }

    attr.qp_state                  = IBV_QPS_RTR;
    attr.path_mtu                  = iface->super.config.path_mtu;
    attr.min_rnr_timer             = iface->super.config.min_rnr_timer;
    attr.ah_attr.grh.hop_limit     = iface->super.super.config.hop_limit;
    attr.ah_attr.grh.traffic_class = iface->super.super.config.traffic_class;
    attr.ah_attr.grh.sgid_index    = uct_ib_iface_md(&iface->super.super)->config.gid_index;
    attr.ah_attr.port_num          = iface->super.super.config.port_num;

    ret = ibv_modify_qp(mlx5->rx_dct, &attr, IBV_QP_STATE |
                                             IBV_QP_MIN_RNR_TIMER |
                                             IBV_QP_AV |
                                             IBV_QP_PATH_MTU);
    if (ret) {
         ucs_error("error modifying DCT to RTR: %m");
         goto err;
    }

    return UCS_OK;

err:
    ibv_destroy_qp(mlx5->rx_dct);
    return UCS_ERR_IO_ERROR;
}

int uct_dc_mlx5_get_dct_num(uct_dc_mlx5_iface_t *iface)
{
    uct_dc_mlx5_iface_t *mlx5 = ucs_derived_of(iface, uct_dc_mlx5_iface_t);

    return mlx5->rx_dct->qp_num;
}

void uct_dc_mlx5_destroy_dct(uct_dc_mlx5_iface_t *iface)
{
    uct_dc_mlx5_iface_t *mlx5 = ucs_derived_of(iface, uct_dc_mlx5_iface_t);

    if (mlx5->rx_dct != NULL) {
        ibv_destroy_qp(mlx5->rx_dct);
    }
    mlx5->rx_dct = NULL;
}
#endif

static uct_rc_iface_ops_t uct_dc_mlx5_iface_ops = {
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
    .ep_pending_add           = uct_dc_mlx5_ep_pending_add,
    .ep_pending_purge         = uct_dc_mlx5_ep_pending_purge,
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
    .iface_flush              = uct_dc_mlx5_iface_flush,
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
    .iface_is_reachable       = uct_dc_mlx5_iface_is_reachable,
    .iface_get_address        = uct_dc_mlx5_iface_get_address,
    },
    .create_cq                = uct_ib_mlx5_create_cq,
    .arm_cq                   = uct_ib_iface_arm_cq,
    .event_cq                 = uct_dc_mlx5_iface_event_cq,
    .handle_failure           = uct_dc_mlx5_iface_handle_failure,
    .set_ep_failed            = uct_dc_mlx5_ep_set_failed,
#if HAVE_DC_DV
    .create_qp                = uct_dc_mlx5_iface_create_qp
#else
    .create_qp                = uct_ib_iface_create_qp
#endif
    },
    .fc_ctrl                  = uct_dc_mlx5_ep_fc_ctrl,
    .fc_handler               = uct_dc_mlx5_iface_fc_handler,
};


static ucs_status_t uct_dc_mlx5_iface_init_dcis(uct_dc_mlx5_iface_t *iface,
                                                uct_ib_mlx5_mmio_mode_t mmio_mode)
{
    ucs_status_t status;
    uint16_t bb_max;
    int i;

    bb_max = 0;
    for (i = 0; i < iface->tx.ndci; i++) {
        status = uct_ib_mlx5_txwq_init(iface->super.super.super.worker,
                                       mmio_mode, &iface->tx.dci_wqs[i],
                                       iface->tx.dcis[i].txqp.qp);
        if (status != UCS_OK) {
            return status;
        }


        bb_max = iface->tx.dci_wqs[i].bb_max;
        uct_rc_txqp_available_set(&iface->tx.dcis[i].txqp, bb_max);
    }

    iface->super.config.tx_qp_len = bb_max;
    return UCS_OK;
}

static void uct_dc_mlx5_iface_cleanup_dcis(uct_dc_mlx5_iface_t *iface)
{
    int i;

    for (i = 0; i < iface->tx.ndci; i++) {
        uct_ib_mlx5_txwq_cleanup(&iface->tx.dci_wqs[i]);
    }
}

static ucs_status_t uct_dc_mlx5_iface_tag_init(uct_dc_mlx5_iface_t *iface,
                                               uct_dc_mlx5_iface_config_t *config)
{
#if IBV_EXP_HW_TM_DC
    if (UCT_RC_IFACE_TM_ENABLED(&iface->super)) {
        struct ibv_exp_create_srq_attr srq_init_attr = {};
        struct ibv_exp_srq_dc_offload_params dc_op   = {};
        ucs_status_t status;

        uct_dc_mlx5_iface_fill_xrq_init_attrs(&iface->super, &srq_init_attr, &dc_op);

        status = uct_rc_mlx5_iface_common_tag_init(&iface->mlx5_common,
                                                   &iface->super,
                                                   &config->super,
                                                   &config->mlx5_common,
                                                   &srq_init_attr,
                                                   sizeof(struct ibv_exp_tmh_rvh) +
                                                   sizeof(struct ibv_exp_tmh_ravh));
        if (status != UCS_OK) {
            return status;
        }

        /* TM XRQ is ready, can create DCT now */
        status = uct_dc_mlx5_iface_create_dct(iface);
        if (status != UCS_OK) {
            uct_rc_mlx5_iface_common_tag_cleanup(&iface->mlx5_common,
                                                 &iface->super);
            return status;
        }
        iface->super.progress = uct_dc_mlx5_iface_progress_tm;
    } else
#endif
    {
        iface->super.progress = uct_dc_mlx5_iface_progress;
    }
    return UCS_OK;
}

static void uct_dc_mlx5_iface_tag_cleanup(uct_dc_mlx5_iface_t *iface)
{
    if (UCT_RC_IFACE_TM_ENABLED(&iface->super)) {
        uct_dc_mlx5_destroy_dct(iface);
    }

    uct_rc_mlx5_iface_common_tag_cleanup(&iface->mlx5_common, &iface->super);
}

#if HAVE_DC_EXP
ucs_status_t uct_dc_mlx5_iface_create_dct(uct_dc_mlx5_iface_t *iface)
{
    struct ibv_exp_dct_init_attr init_attr;

    memset(&init_attr, 0, sizeof(init_attr));

    init_attr.pd               = uct_ib_iface_md(&iface->super.super)->pd;
    init_attr.cq               = iface->super.super.cq[UCT_IB_DIR_RX];
    init_attr.srq              = iface->super.rx.srq.srq;
    init_attr.dc_key           = UCT_IB_KEY;
    init_attr.port             = iface->super.super.config.port_num;
    init_attr.mtu              = iface->super.config.path_mtu;
    init_attr.access_flags     = IBV_EXP_ACCESS_REMOTE_WRITE |
                                 IBV_EXP_ACCESS_REMOTE_READ |
                                 IBV_EXP_ACCESS_REMOTE_ATOMIC;
    init_attr.min_rnr_timer    = iface->super.config.min_rnr_timer;
    init_attr.tclass           = iface->super.super.config.traffic_class;
    init_attr.hop_limit        = iface->super.super.config.hop_limit;
    init_attr.gid_index        = iface->super.super.config.gid_index;
    init_attr.inline_size      = iface->super.config.rx_inline;
    init_attr.pkey_index       = iface->super.super.pkey_index;

#if HAVE_DECL_IBV_EXP_DCT_OOO_RW_DATA_PLACEMENT
    if (iface->super.config.ooo_rw &&
        UCX_IB_DEV_IS_OOO_SUPPORTED(&uct_ib_iface_device(&iface->super.super)->dev_attr,
                                    dc)) {
        ucs_debug("creating DC target with out-of-order support dev %s",
                   uct_ib_device_name(uct_ib_iface_device(&iface->super.super)));
        init_attr.create_flags |= IBV_EXP_DCT_OOO_RW_DATA_PLACEMENT;
    }
#endif

    iface->rx_dct = ibv_exp_create_dct(uct_ib_iface_device(&iface->super.super)->ibv_context,
                                       &init_attr);
    if (iface->rx_dct == NULL) {
        ucs_error("Failed to created DC target %m");
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

/* take dc qp to rts state */
ucs_status_t uct_dc_mlx5_iface_dci_connect(uct_dc_mlx5_iface_t *iface,
                                           uct_rc_txqp_t *dci)
{
    struct ibv_exp_qp_attr attr;
    long attr_mask;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state        = IBV_QPS_INIT;
    attr.pkey_index      = iface->super.super.pkey_index;
    attr.port_num        = iface->super.super.config.port_num;
    attr.dct_key         = UCT_IB_KEY;
    attr_mask            = IBV_EXP_QP_STATE      |
                           IBV_EXP_QP_PKEY_INDEX |
                           IBV_EXP_QP_PORT       |
                           IBV_EXP_QP_DC_KEY;

    if (ibv_exp_modify_qp(dci->qp, &attr, attr_mask)) {
        ucs_error("error modifying QP to INIT : %m");
        return UCS_ERR_IO_ERROR;
    }

    /* Move QP to the RTR state */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state                   = IBV_QPS_RTR;
    attr.path_mtu                   = iface->super.config.path_mtu;
    attr.max_dest_rd_atomic         = 1;
    attr.ah_attr.is_global          = iface->super.super.is_global_addr;
    attr.ah_attr.sl                 = iface->super.super.config.sl;
    attr_mask                       = IBV_EXP_QP_STATE     |
                                      IBV_EXP_QP_PATH_MTU  |
                                      IBV_EXP_QP_AV;

#if HAVE_DECL_IBV_EXP_QP_OOO_RW_DATA_PLACEMENT
    if (iface->super.config.ooo_rw &&
        UCX_IB_DEV_IS_OOO_SUPPORTED(&uct_ib_iface_device(&iface->super.super)->dev_attr,
                                    dc)) {
        ucs_debug("enabling out-of-order on DCI QP 0x%x dev %s", dci->qp->qp_num,
                   uct_ib_device_name(uct_ib_iface_device(&iface->super.super)));
        attr_mask |= IBV_EXP_QP_OOO_RW_DATA_PLACEMENT;
    }
#endif

    if (ibv_exp_modify_qp(dci->qp, &attr, attr_mask)) {
        ucs_error("error modifying QP to RTR: %m");
        return UCS_ERR_IO_ERROR;
    }

    /* Move QP to the RTS state */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state       = IBV_QPS_RTS;
    attr.timeout        = iface->super.config.timeout;
    attr.rnr_retry      = iface->super.config.rnr_retry;
    attr.retry_cnt      = iface->super.config.retry_cnt;
    attr.max_rd_atomic  = iface->super.config.max_rd_atomic;
    attr_mask           = IBV_EXP_QP_STATE      |
                          IBV_EXP_QP_TIMEOUT    |
                          IBV_EXP_QP_RETRY_CNT  |
                          IBV_EXP_QP_RNR_RETRY  |
                          IBV_EXP_QP_MAX_QP_RD_ATOMIC;

    if (ibv_exp_modify_qp(dci->qp, &attr, attr_mask)) {
        ucs_error("error modifying QP to RTS: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

int uct_dc_mlx5_get_dct_num(uct_dc_mlx5_iface_t *iface)
{
    return iface->rx_dct->dct_num;
}

void uct_dc_mlx5_destroy_dct(uct_dc_mlx5_iface_t *iface)
{
    if (iface->rx_dct != NULL) {
        ibv_exp_destroy_dct(iface->rx_dct);
    }
    iface->rx_dct = NULL;
}

static ucs_status_t uct_dc_mlx5_device_init(uct_ib_device_t *dev)
{
#if HAVE_DECL_IBV_EXP_DEVICE_DC_TRANSPORT && HAVE_STRUCT_IBV_EXP_DEVICE_ATTR_EXP_DEVICE_CAP_FLAGS
    if (dev->dev_attr.exp_device_cap_flags & IBV_EXP_DEVICE_DC_TRANSPORT) {
        dev->flags |= UCT_IB_DEVICE_FLAG_DC;
    }
#endif
    return UCS_OK;
}

UCT_IB_DEVICE_INIT(uct_dc_mlx5_device_init);

#endif

void uct_dc_mlx5_iface_dcis_destroy(uct_dc_mlx5_iface_t *iface, int max)
{
    int i;
    for (i = 0; i < max; i++) {
        uct_rc_txqp_cleanup(&iface->tx.dcis[i].txqp);
    }
}

ucs_status_t uct_dc_mlx5_iface_create_dcis(uct_dc_mlx5_iface_t *iface,
                                           uct_dc_mlx5_iface_config_t *config)
{
    struct ibv_qp_cap cap;
    ucs_status_t status;
    int i;

    ucs_debug("creating %d dci(s)", iface->tx.ndci);

    iface->tx.stack_top = 0;
    for (i = 0; i < iface->tx.ndci; i++) {
        status = uct_rc_txqp_init(&iface->tx.dcis[i].txqp, &iface->super,
                                  UCT_IB_QPT_DCI, &cap
                                  UCS_STATS_ARG(iface->super.stats));
        if (status != UCS_OK) {
            goto err;
        }

        status = uct_dc_mlx5_iface_dci_connect(iface, &iface->tx.dcis[i].txqp);
        if (status != UCS_OK) {
            uct_rc_txqp_cleanup(&iface->tx.dcis[i].txqp);
            goto err;
        }

        iface->tx.dcis_stack[i] = i;
        iface->tx.dcis[i].ep    = NULL;
#if ENABLE_ASSERT
        iface->tx.dcis[i].flags = 0;
#endif
    }
    uct_ib_iface_set_max_iov(&iface->super.super, cap.max_send_sge);
    return UCS_OK;

err:
    uct_dc_mlx5_iface_dcis_destroy(iface, i);
    return status;
}

void uct_dc_mlx5_iface_set_quota(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_iface_config_t *config)
{
    iface->tx.available_quota = iface->super.config.tx_qp_len -
                                ucs_min(iface->super.config.tx_qp_len, config->quota);
}

void uct_dc_mlx5_iface_init_version(uct_dc_mlx5_iface_t *iface, uct_md_h md)
{
    uct_ib_device_t *dev;
    unsigned         ver;

    dev = &ucs_derived_of(md, uct_ib_md_t)->dev;
    ver = uct_ib_device_spec(dev)->flags & UCT_IB_DEVICE_FLAG_DC;
    ucs_assert(ver != UCT_IB_DEVICE_FLAG_DC);

    iface->version_flag = 0;

    if (ver & UCT_IB_DEVICE_FLAG_DC_V2) {
        iface->version_flag = UCT_DC_MLX5_IFACE_ADDR_DC_V2;
    }

    if (ver & UCT_IB_DEVICE_FLAG_DC_V1) {
        iface->version_flag = UCT_DC_MLX5_IFACE_ADDR_DC_V1;
    }
}

int uct_dc_mlx5_iface_is_reachable(const uct_iface_h tl_iface,
                                   const uct_device_addr_t *dev_addr,
                                   const uct_iface_addr_t *iface_addr)
{
    uct_dc_mlx5_iface_addr_t *addr = (uct_dc_mlx5_iface_addr_t *)iface_addr;
    uct_dc_mlx5_iface_t UCS_V_UNUSED *iface;

    iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    ucs_assert_always(iface_addr != NULL);

    return ((addr->flags & UCT_DC_MLX5_IFACE_ADDR_DC_VERS) == iface->version_flag) &&
           (UCT_DC_MLX5_IFACE_ADDR_TM_ENABLED(addr) ==
            UCT_RC_IFACE_TM_ENABLED(&iface->super)) &&
           uct_ib_iface_is_reachable(tl_iface, dev_addr, iface_addr);
}

ucs_status_t
uct_dc_mlx5_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    uct_dc_mlx5_iface_t      *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_iface_addr_t *addr  = (uct_dc_mlx5_iface_addr_t *)iface_addr;

    uct_ib_pack_uint24(addr->qp_num, uct_dc_mlx5_get_dct_num(iface));
    addr->atomic_mr_id = uct_ib_iface_get_atomic_mr_id(&iface->super.super);
    addr->flags        = iface->version_flag;
    if (UCT_RC_IFACE_TM_ENABLED(&iface->super)) {
        addr->flags   |= UCT_DC_MLX5_IFACE_ADDR_HW_TM;
    }

    return UCS_OK;
}

ucs_status_t uct_dc_device_query_tl_resources(uct_ib_device_t *dev,
                                              const char *tl_name, unsigned flags,
                                              uct_tl_resource_desc_t **resources_p,
                                              unsigned *num_resources_p)
{
    return uct_ib_device_query_tl_resources(dev, tl_name,
                                            flags | UCT_IB_DEVICE_FLAG_DC,
                                            resources_p, num_resources_p);
}

static inline ucs_status_t uct_dc_mlx5_iface_flush_dcis(uct_dc_mlx5_iface_t *iface)
{
    int i;
    int is_flush_done = 1;

    for (i = 0; i < iface->tx.ndci; i++) {
        if ((iface->tx.dcis[i].ep != NULL) &&
            uct_dc_mlx5_ep_fc_wait_for_grant(iface->tx.dcis[i].ep)) {
            return UCS_INPROGRESS;
        }
        if (uct_dc_mlx5_iface_flush_dci(iface, i) != UCS_OK) {
            is_flush_done = 0;
        }
    }
    return is_flush_done ? UCS_OK : UCS_INPROGRESS;
}

ucs_status_t uct_dc_mlx5_iface_flush(uct_iface_h tl_iface, unsigned flags, uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    ucs_status_t status;

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }
    status = uct_dc_mlx5_iface_flush_dcis(iface);
    if (status == UCS_OK) {
        UCT_TL_IFACE_STAT_FLUSH(&iface->super.super.super);
    }
    else if (status == UCS_INPROGRESS) {
        UCT_TL_IFACE_STAT_FLUSH_WAIT(&iface->super.super.super);
    }
    return status;
}

ucs_status_t uct_dc_mlx5_iface_init_fc_ep(uct_dc_mlx5_iface_t *iface)
{
    ucs_status_t status;
    uct_dc_mlx5_ep_t *ep;

    ep = ucs_malloc(sizeof(uct_dc_mlx5_ep_t), "fc_ep");
    if (ep == NULL) {
        ucs_error("Failed to allocate FC ep");
        status =  UCS_ERR_NO_MEMORY;
        goto err;
    }
    /* We do not have any peer address at this point, so init basic subclasses
     * only (for statistics, iface, etc) */
    status = UCS_CLASS_INIT(uct_base_ep_t, (void*)(&ep->super),
                            &iface->super.super.super);
    if (status != UCS_OK) {
        ucs_error("Failed to initialize fake FC ep, status: %s",
                  ucs_status_string(status));
        goto err_free;
    }

    status = uct_dc_mlx5_ep_basic_init(iface, ep);
    if (status != UCS_OK) {
        ucs_error("FC ep init failed %s", ucs_status_string(status));
        goto err_cleanup;
    }

    iface->tx.fc_ep = ep;
    return UCS_OK;

err_cleanup:
    UCS_CLASS_CLEANUP(uct_base_ep_t, &ep->super);
err_free:
    ucs_free(ep);
err:
    return status;
}

void uct_dc_mlx5_iface_cleanup_fc_ep(uct_dc_mlx5_iface_t *iface)
{
    uct_dc_mlx5_ep_pending_purge(&iface->tx.fc_ep->super.super, NULL, NULL);
    ucs_arbiter_group_cleanup(&iface->tx.fc_ep->arb_group);
    uct_rc_fc_cleanup(&iface->tx.fc_ep->fc);
    UCS_CLASS_CLEANUP(uct_base_ep_t, iface->tx.fc_ep);
    ucs_free(iface->tx.fc_ep);
}

ucs_status_t uct_dc_mlx5_iface_fc_grant(uct_pending_req_t *self)
{
    uct_rc_fc_request_t *freq = ucs_derived_of(self, uct_rc_fc_request_t);
    uct_dc_mlx5_ep_t *ep      = ucs_derived_of(freq->ep, uct_dc_mlx5_ep_t);
    uct_rc_iface_t *iface     = ucs_derived_of(ep->super.super.iface,
                                               uct_rc_iface_t);
    ucs_status_t status;

    ucs_assert_always(iface->config.fc_enabled);

    status = uct_rc_fc_ctrl(&ep->super.super, UCT_RC_EP_FC_PURE_GRANT, freq);
    if (status == UCS_OK) {
        ucs_mpool_put(freq);
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_TX_PURE_GRANT, 1);
    }
    return status;
}

ucs_status_t uct_dc_mlx5_iface_fc_handler(uct_rc_iface_t *rc_iface, unsigned qp_num,
                                          uct_rc_hdr_t *hdr, unsigned length,
                                          uint32_t imm_data, uint16_t lid, unsigned flags)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(rc_iface, uct_dc_mlx5_iface_t);
    uint8_t             fc_hdr = uct_rc_fc_get_fc_hdr(hdr->am_id);
    uct_dc_fc_request_t *dc_req;
    int16_t             cur_wnd;
    ucs_status_t        status;
    uct_dc_mlx5_ep_t    *ep;

    ucs_assert(rc_iface->config.fc_enabled);

    if (fc_hdr == UCT_RC_EP_FC_FLAG_HARD_REQ) {
        ep = iface->tx.fc_ep;
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_RX_HARD_REQ, 1);

        dc_req = ucs_mpool_get(&iface->super.tx.fc_mp);
        if (ucs_unlikely(dc_req == NULL)) {
            ucs_error("Failed to allocate FC request");
            return UCS_ERR_NO_MEMORY;
        }
        dc_req->super.super.func = uct_dc_mlx5_iface_fc_grant;
        dc_req->super.ep         = &ep->super.super;
        dc_req->dct_num          = imm_data;
        dc_req->lid              = lid;
        dc_req->sender           = *((uct_dc_fc_sender_data_t*)(hdr + 1));

        status = uct_dc_mlx5_iface_fc_grant(&dc_req->super.super);
        if (status == UCS_ERR_NO_RESOURCE){
            status = uct_ep_pending_add(&ep->super.super, &dc_req->super.super,
                                        0);
        }
        ucs_assertv_always(status == UCS_OK, "Failed to send FC grant msg: %s",
                           ucs_status_string(status));
    } else if (fc_hdr == UCT_RC_EP_FC_PURE_GRANT) {
        ep = *((uct_dc_mlx5_ep_t**)(hdr + 1));

        if (!(ep->flags & UCT_DC_MLX5_EP_FLAG_VALID)) {
            uct_dc_mlx5_ep_release(ep);
            return UCS_OK;
        }

        cur_wnd = ep->fc.fc_wnd;

        /* Peer granted resources, so update wnd */
        ep->fc.fc_wnd = rc_iface->config.fc_wnd_size;

        /* Clear the flag for flush to complete  */
        ep->fc.flags &= ~UCT_DC_MLX5_EP_FC_FLAG_WAIT_FOR_GRANT;

        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_RX_PURE_GRANT, 1);
        UCS_STATS_SET_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_FC_WND, ep->fc.fc_wnd);

        /* To preserve ordering we have to dispatch all pending
         * operations if current fc_wnd is <= 0 */
        if (cur_wnd <= 0) {
            if (ep->dci == UCT_DC_MLX5_EP_NO_DCI) {
                ucs_arbiter_group_schedule(uct_dc_mlx5_iface_dci_waitq(iface),
                                           &ep->arb_group);
            } else {
                /* Need to schedule fake ep in TX arbiter, because it
                 * might have been descheduled due to lack of FC window. */
                ucs_arbiter_group_schedule(uct_dc_mlx5_iface_tx_waitq(iface),
                                           &ep->arb_group);
            }

            uct_dc_mlx5_iface_progress_pending(iface);
        }
    }

    return UCS_OK;
}

ucs_status_t uct_dc_handle_failure(uct_ib_iface_t *ib_iface, uint32_t qp_num,
                                   ucs_status_t status)
{
    uct_dc_mlx5_iface_t  *iface  = ucs_derived_of(ib_iface, uct_dc_mlx5_iface_t);
    uint8_t              dci     = uct_dc_mlx5_iface_dci_find(iface, qp_num);
    uct_rc_txqp_t        *txqp   = &iface->tx.dcis[dci].txqp;
    uct_dc_mlx5_ep_t     *ep     = iface->tx.dcis[dci].ep;
    ucs_status_t         ep_status;
    int16_t              outstanding;

    if (!ep) {
        return UCS_OK;
    }

    uct_rc_txqp_purge_outstanding(txqp, status, 0);

    /* poll_cqe for mlx5 returns NULL in case of failure and the cq_avaialble
       is not updated for the error cqe and all outstanding wqes*/
    outstanding = (int16_t)iface->super.config.tx_qp_len -
                  uct_rc_txqp_available(txqp);
    iface->super.tx.cq_available += outstanding;
    uct_rc_txqp_available_set(txqp, (int16_t)iface->super.config.tx_qp_len);

    /* since we removed all outstanding ops on the dci, it should be released */
    ucs_assert(ep->dci != UCT_DC_MLX5_EP_NO_DCI);
    uct_dc_mlx5_iface_dci_put(iface, dci);
    ucs_assert_always(ep->dci == UCT_DC_MLX5_EP_NO_DCI);

    ep_status = iface->super.super.ops->set_ep_failed(ib_iface,
                                                      &ep->super.super, status);
    if (ep_status == UCS_OK) {
        status = uct_dc_mlx5_iface_reset_dci(iface, dci);
        if (status != UCS_OK) {
            ucs_fatal("iface %p failed to reset dci[%d] qpn 0x%x: %s",
                       iface, dci, txqp->qp->qp_num, ucs_status_string(status));
        }

        status = uct_dc_mlx5_iface_dci_connect(iface, txqp);
        if (status != UCS_OK) {
            ucs_fatal("iface %p failed to connect dci[%d] qpn 0x%x: %s",
                      iface, dci, txqp->qp->qp_num, ucs_status_string(status));
        }
    }

    return ep_status;
}

#if IBV_EXP_HW_TM_DC
void uct_dc_mlx5_iface_fill_xrq_init_attrs(uct_rc_iface_t *rc_iface,
                                           struct ibv_exp_create_srq_attr *srq_attr,
                                           struct ibv_exp_srq_dc_offload_params *dc_op)
{
    dc_op->timeout    = rc_iface->config.timeout;
    dc_op->path_mtu   = rc_iface->config.path_mtu;
    dc_op->pkey_index = rc_iface->super.pkey_index;
    dc_op->sl         = rc_iface->super.config.sl;
    dc_op->dct_key    = UCT_IB_KEY;

    srq_attr->comp_mask         = IBV_EXP_CREATE_SRQ_DC_OFFLOAD_PARAMS;
    srq_attr->dc_offload_params = dc_op;
}
#endif

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
    init_attr.fc_req_size    = sizeof(uct_dc_fc_request_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, &uct_dc_mlx5_iface_ops, md, worker, params,
                              &config->super, &init_attr);
    if (config->ndci < 1) {
        ucs_error("dc interface must have at least 1 dci (requested: %d)",
                  config->ndci);
        return UCS_ERR_INVALID_PARAM;
    }

    if (config->ndci > UCT_DC_MLX5_IFACE_MAX_DCIS) {
        ucs_error("dc interface can have at most %d dcis (requested: %d)",
                  UCT_DC_MLX5_IFACE_MAX_DCIS, config->ndci);
        return UCS_ERR_INVALID_PARAM;
    }

    uct_dc_mlx5_iface_init_version(self, md);

    self->tx.ndci                    = config->ndci;
    self->tx.policy                  = config->tx_policy;
    self->tx.available_quota         = 0; /* overridden by mlx5/verbs */
    self->super.config.tx_moderation = 0; /* disable tx moderation for dcs */
    ucs_list_head_init(&self->tx.gc_list);

    /* create DC target */
    if (!UCT_RC_IFACE_TM_ENABLED(&self->super)) {
        status = uct_dc_mlx5_iface_create_dct(self);
        if (status != UCS_OK) {
            goto err;
        }
    }

    /* create DC initiators */
    status = uct_dc_mlx5_iface_create_dcis(self, config);
    if (status != UCS_OK) {
        goto err_destroy_dct;
    }

    ucs_debug("dc iface %p: using '%s' policy with %d dcis, dct 0x%x", self,
              uct_dc_tx_policy_names[self->tx.policy], self->tx.ndci,
              UCT_RC_IFACE_TM_ENABLED(&self->super) ?
              0 : uct_dc_mlx5_get_dct_num(self));

    /* Create fake endpoint which will be used for sending FC grants */
    uct_dc_mlx5_iface_init_fc_ep(self);

    ucs_arbiter_init(&self->tx.dci_arbiter);

    /* mlx5 init part */
    status = uct_dc_mlx5_iface_tag_init(self, config);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_rc_mlx5_iface_common_init(&self->mlx5_common, &self->super,
                                           &config->super, &config->mlx5_common);
    if (status != UCS_OK) {
        goto err_rc_mlx5_tag_cleanup;
    }

    status = uct_ud_mlx5_iface_common_init(&self->super.super,
                                           &self->ud_common, &config->mlx5_ud);
    if (status != UCS_OK) {
        goto err_rc_mlx5_common_cleanup;
    }

    status = uct_dc_mlx5_iface_init_dcis(self, config->mlx5_common.mmio_mode);
    if (status != UCS_OK) {
        goto err_rc_mlx5_common_cleanup;
    }

    self->tx.available_quota = self->super.config.tx_qp_len -
                               ucs_min(self->super.config.tx_qp_len, config->quota);
    /* Set max_iov for put_zcopy and get_zcopy */
    uct_ib_iface_set_max_iov(&self->super.super,
                             (UCT_IB_MLX5_MAX_SEND_WQE_SIZE -
                             sizeof(struct mlx5_wqe_raddr_seg) -
                             sizeof(struct mlx5_wqe_ctrl_seg) -
                             UCT_IB_MLX5_AV_FULL_SIZE) /
                             sizeof(struct mlx5_wqe_data_seg));

    uct_rc_mlx5_iface_common_prepost_recvs(&self->super,
                                           &self->mlx5_common);

    ucs_debug("created dc iface %p", self);

    return UCS_OK;

err_rc_mlx5_common_cleanup:
    uct_rc_mlx5_iface_common_cleanup(&self->mlx5_common);
err_rc_mlx5_tag_cleanup:
    uct_dc_mlx5_iface_tag_cleanup(self);
err_destroy_dct:
    if (!UCT_RC_IFACE_TM_ENABLED(&self->super)) {
        uct_dc_mlx5_destroy_dct(self);
    }
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_mlx5_iface_t)
{
    uct_dc_mlx5_ep_t *ep, *tmp;

    ucs_trace_func("");
    uct_base_iface_progress_disable(&self->super.super.super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    uct_dc_mlx5_iface_cleanup_dcis(self);
    uct_rc_mlx5_iface_common_cleanup(&self->mlx5_common);
    uct_dc_mlx5_iface_tag_cleanup(self);

    uct_dc_mlx5_destroy_dct(self);
    ucs_list_for_each_safe(ep, tmp, &self->tx.gc_list, list) {
        uct_dc_mlx5_ep_release(ep);
    }
    uct_dc_mlx5_iface_dcis_destroy(self, self->tx.ndci);
    uct_dc_mlx5_iface_cleanup_fc_ep(self);
    ucs_arbiter_cleanup(&self->tx.dci_arbiter);
}

UCS_CLASS_DEFINE(uct_dc_mlx5_iface_t, uct_rc_iface_t);

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
