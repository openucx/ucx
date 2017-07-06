/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "dc_mlx5.h"

#include <uct/api/uct.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_log.h>
#include <uct/ib/base/ib_md.h>
#include <uct/ib/mlx5/ib_mlx5_log.h>
#include <uct/ib/base/ib_md.h>
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

  {NULL}
};

static void uct_dc_mlx5_iface_global_addr_error(uct_dc_mlx5_iface_t *iface,
                                                const uct_ib_address_t *ib_addr)
{
    struct ibv_ah_attr ah_attr;
    char gid_buf[64];

    uct_ib_iface_fill_ah_attr(&iface->super.super.super, ib_addr, 0, &ah_attr);
    inet_ntop(AF_INET6, ah_attr.grh.dgid.raw, gid_buf, sizeof(gid_buf));
    ucs_error("dc_mlx5 does not support global address, cannot connect from %s:%d"
              " to lid %d gid %s",
              uct_ib_device_name(uct_ib_iface_device(&iface->super.super.super)),
              iface->super.super.super.config.port_num, ah_attr.dlid, gid_buf);
}

static UCS_CLASS_INIT_FUNC(uct_dc_mlx5_ep_t, uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    const uct_ib_address_t *ib_addr = (const uct_ib_address_t *)dev_addr;
    const uct_dc_iface_addr_t *if_addr = (const uct_dc_iface_addr_t *)iface_addr;
    struct mlx5_grh_av grh_av;
    ucs_status_t status;
    int is_global;
    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(uct_dc_ep_t, &iface->super, if_addr);

    status = uct_ud_mlx5_iface_get_av(&iface->super.super.super, &iface->ud_common,
                                      ib_addr, iface->super.super.super.path_bits[0],
                                      &self->av, &grh_av, &is_global);
    if (status != UCS_OK) {
        return UCS_ERR_INVALID_ADDR;
    }

    if (is_global) {
        uct_dc_mlx5_iface_global_addr_error(iface, ib_addr);
        return UCS_ERR_UNREACHABLE;
    }

    self->av.dqp_dct |= htonl(uct_ib_unpack_uint24(if_addr->qp_num));

    ucs_debug("created ep %p on iface %p", self, iface);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_mlx5_ep_t)
{
    ucs_trace_func("");
}

UCS_CLASS_DEFINE(uct_dc_mlx5_ep_t, uct_dc_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_dc_mlx5_ep_t, uct_ep_t, uct_iface_h, const uct_device_addr_t *,
                          const uct_iface_addr_t *);

static void uct_dc_mlx5_ep_destroy(uct_ep_h tl_ep)
{
    uct_dc_ep_cleanup(tl_ep, &UCS_CLASS_NAME(uct_dc_mlx5_ep_t));
}

static ucs_status_t uct_dc_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);

    uct_dc_iface_query(&iface->super, iface_attr);
    uct_rc_mlx5_iface_common_query(&iface->super.super, iface_attr,
                                   UCT_IB_MLX5_AV_FULL_SIZE);
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
                            /* SEND */ uint8_t am_id, void *am_hdr, unsigned am_hdr_len,
                            /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey,
                            uct_rc_iface_send_desc_t *desc)
{
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, &ep->super, txqp, txwq);
    desc->super.sn = txwq->sw_pi;
    uct_rc_mlx5_txqp_dptr_post(&iface->super.super, IBV_EXP_QPT_DC_INI, txqp, txwq,
                               opcode, desc + 1, length, &desc->lkey,
                               am_id, am_hdr, am_hdr_len,
                               rdma_raddr, uct_ib_md_direct_rkey(rdma_rkey),
                               0, 0, 0,
                               &ep->av, uct_ib_mlx5_wqe_av_size(&ep->av),
                               MLX5_WQE_CTRL_CQ_UPDATE);
    uct_rc_txqp_add_send_op(txqp, &desc->super);
}


static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_zcopy_post(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep,
                             unsigned opcode, const uct_iov_t *iov, size_t iovcnt,
                             /* SEND */ uint8_t am_id, const void *am_hdr, unsigned am_hdr_len,
                             /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey,
                             uct_completion_t *comp)
{
    uint16_t sn;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, &ep->super, txqp, txwq);

    sn = txwq->sw_pi;
    uct_rc_mlx5_txqp_dptr_post_iov(&iface->super.super, IBV_EXP_QPT_DC_INI, txqp,
                                   txwq, opcode, iov, iovcnt,
                                   am_id, am_hdr, am_hdr_len,
                                   rdma_raddr, uct_ib_md_direct_rkey(rdma_rkey),
                                   &ep->av, uct_ib_mlx5_wqe_av_size(&ep->av),
                                   MLX5_WQE_CTRL_CQ_UPDATE);

    uct_rc_txqp_add_send_comp(&iface->super.super, txqp, comp, sn);
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_atomic_post(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep,
                              unsigned opcode, uct_rc_iface_send_desc_t *desc, unsigned length,
                              uint64_t remote_addr, uct_rkey_t rkey,
                              uint64_t compare_mask, uint64_t compare, uint64_t swap_add)
{
    uint32_t ib_rkey = uct_ib_resolve_atomic_rkey(rkey, ep->super.atomic_mr_offset,
                                                  &remote_addr);

    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);
    UCT_DC_MLX5_IFACE_TXQP_GET(iface, &ep->super, txqp, txwq);

    desc->super.sn = txwq->sw_pi;
    uct_rc_mlx5_txqp_dptr_post(&iface->super.super, IBV_EXP_QPT_DC_INI, txqp, txwq,
                               opcode, desc + 1, length, &desc->lkey,
                               0, desc/*dummy*/, 0, remote_addr, ib_rkey,
                               compare_mask, compare, swap_add,
                               &ep->av, uct_ib_mlx5_wqe_av_size(&ep->av),
                               MLX5_WQE_CTRL_CQ_UPDATE);

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

ucs_status_t uct_dc_mlx5_ep_atomic_add(uct_ep_h tl_ep,
                                         int opcode, unsigned length,
                                         uint64_t add, uint64_t remote_addr, uct_rkey_t rkey)
{

    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_ATOMIC_ADD_DESC(&iface->super.super, &iface->mlx5_common.tx.atomic_desc_mp, desc);
    uct_dc_mlx5_iface_atomic_post(iface, ep, opcode, desc, length,
                                  remote_addr, rkey, 0, 0, add);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_dc_mlx5_ep_atomic(uct_dc_mlx5_ep_t *ep, int opcode, void *result, int ext,
                      unsigned length, uint64_t remote_addr, uct_rkey_t rkey,
                      uint64_t compare_mask, uint64_t compare,
                      uint64_t swap_add, uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.super.iface, uct_dc_mlx5_iface_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_ATOMIC_DESC(&iface->super.super, &iface->mlx5_common.tx.atomic_desc_mp, desc,
                                    uct_rc_iface_atomic_handler(&iface->super.super, ext, length),
                                    result, comp);
    uct_dc_mlx5_iface_atomic_post(iface, ep, opcode, desc, length, remote_addr, rkey,
                                  compare_mask, compare, swap_add);
    return UCS_INPROGRESS;
}

ucs_status_t uct_dc_mlx5_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                         uint64_t remote_addr, uct_rkey_t rkey)
{
    return uct_dc_mlx5_ep_atomic_add(tl_ep, MLX5_OPCODE_ATOMIC_FA, sizeof(uint64_t),
                                     htobe64(add), remote_addr, rkey);
}

ucs_status_t uct_dc_mlx5_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uint64_t *result, uct_completion_t *comp)
{
    return uct_dc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_FA, result, 0, sizeof(uint64_t),
                                 remote_addr, rkey, 0, 0, htobe64(add), comp);
}

ucs_status_t uct_dc_mlx5_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uint64_t *result, uct_completion_t *comp)
{
    return uct_dc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_CS, result, 1,
                                 sizeof(uint64_t), remote_addr, rkey, 0, 0,
                                 htobe64(swap), comp);
}

ucs_status_t uct_dc_mlx5_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint64_t *result, uct_completion_t *comp)
{
    return uct_dc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_CS, result, 0, sizeof(uint64_t),
                                 remote_addr, rkey, 0, htobe64(compare), htobe64(swap),
                                 comp);
}

ucs_status_t uct_dc_mlx5_ep_atomic_add32(uct_ep_h tl_ep, uint32_t add,
                                         uint64_t remote_addr, uct_rkey_t rkey)
{
    return uct_dc_mlx5_ep_atomic_add(tl_ep, MLX5_OPCODE_ATOMIC_MASKED_FA,
                                     sizeof(uint32_t), htonl(add), remote_addr,
                                     rkey);
}

ucs_status_t uct_dc_mlx5_ep_atomic_fadd32(uct_ep_h tl_ep, uint32_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uint32_t *result, uct_completion_t *comp)
{
    return uct_dc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_FA, result, 1,
                                 sizeof(uint32_t), remote_addr, rkey, 0, 0,
                                 htonl(add), comp);
}

ucs_status_t uct_dc_mlx5_ep_atomic_swap32(uct_ep_h tl_ep, uint32_t swap,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uint32_t *result, uct_completion_t *comp)
{
    return uct_dc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_CS, result, 1,
                                 sizeof(uint32_t), remote_addr, rkey, 0, 0,
                                 htonl(swap), comp);
}

ucs_status_t uct_dc_mlx5_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint32_t *result, uct_completion_t *comp)
{
    return uct_dc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_CS, result, 1,
                                 sizeof(uint32_t), remote_addr, rkey, UCS_MASK(32),
                                 htonl(compare), htonl(swap), comp);
}

ucs_status_t uct_dc_mlx5_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
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
                                 &ep->av, uct_ib_mlx5_wqe_av_size(&ep->av));

    UCT_RC_UPDATE_FC_WND(&iface->super.super, &ep->super.fc);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, sizeof(hdr) + length);
    return UCS_OK;
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

    uct_dc_mlx5_iface_bcopy_post(iface, ep,
                                 MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_RAW,
                                 sizeof(uct_rc_hdr_t) + length, 0, NULL, 0, 0, 0, desc);

    UCT_RC_UPDATE_FC_WND(&iface->super.super, &ep->super.fc);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, BCOPY, length);
    return length;
}


ucs_status_t uct_dc_mlx5_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                     unsigned header_length, const uct_iov_t *iov,
                                     size_t iovcnt, uct_completion_t *comp)
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
                                 id, header, header_length, 0, 0, comp);

    UCT_RC_UPDATE_FC_WND(&iface->super.super, &ep->super.fc);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, ZCOPY, header_length +
                      uct_iov_total_length(iov, iovcnt));

    return UCS_INPROGRESS;
}


ucs_status_t uct_dc_mlx5_ep_put_short(uct_ep_h tl_ep, const void *buffer,
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
                                 &ep->av, uct_ib_mlx5_wqe_av_size(&ep->av));

    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, SHORT, length);

    return UCS_OK;
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
    uct_dc_mlx5_iface_bcopy_post(iface, ep,
                                 MLX5_OPCODE_RDMA_WRITE, length, 0, NULL, 0,
                                 remote_addr, rkey, desc);
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
                                 0, NULL, 0, remote_addr, rkey, comp);

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
    uct_dc_mlx5_iface_bcopy_post(iface, ep,
                                 MLX5_OPCODE_RDMA_READ, length, 0, NULL, 0,
                                 remote_addr, rkey, desc);
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
                                 0, NULL, 0, remote_addr, rkey, comp);
    UCT_TL_EP_STAT_OP(&ep->super.super, GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    return UCS_INPROGRESS;
}

ucs_status_t uct_dc_mlx5_ep_flush(uct_ep_h tl_ep, unsigned flags, uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    ucs_status_t status;

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

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_poll_tx(uct_dc_mlx5_iface_t *iface)
{
    uint8_t dci;
    struct mlx5_cqe64 *cqe;
    uint32_t qp_num;
    uint16_t hw_ci;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    cqe = uct_ib_mlx5_poll_cq(&iface->super.super.super, &iface->mlx5_common.tx.cq);
    if (cqe == NULL) {
        return;
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

    uct_rc_txqp_available_set(txqp, uct_ib_mlx5_txwq_update_bb(txwq, hw_ci));
    uct_dc_iface_dci_put(&iface->super, dci);
    uct_rc_mlx5_txqp_process_tx_cqe(txqp, cqe, hw_ci);

    iface->super.super.tx.cq_available++;

    if (uct_dc_iface_dci_can_alloc(&iface->super)) {
        ucs_arbiter_dispatch(uct_dc_iface_dci_waitq(&iface->super), 1,
                             uct_dc_iface_dci_do_pending_wait, NULL);
    }
    ucs_arbiter_dispatch(uct_dc_iface_tx_waitq(&iface->super), 1, 
                         uct_dc_iface_dci_do_pending_tx, NULL);
}

static void uct_dc_mlx5_iface_progress(void *arg)
{
    uct_dc_mlx5_iface_t *iface = arg;
    ucs_status_t status;

    status = uct_rc_mlx5_iface_common_poll_rx(&iface->mlx5_common, &iface->super.super);
    if (status == UCS_ERR_NO_PROGRESS) {
        uct_dc_mlx5_poll_tx(iface);
    }
}

static void uct_dc_mlx5_iface_handle_failure(uct_ib_iface_t *ib_iface, void *arg)
{
    struct mlx5_cqe64 *cqe   = arg;
    uint32_t          qp_num = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);

    uct_ib_mlx5_completion_with_err(arg, UCS_LOG_LEVEL_ERROR);
    uct_dc_handle_failure(ib_iface, qp_num);
}

static void uct_dc_mlx5_ep_set_failed(uct_ib_iface_t *ib_iface, uct_ep_h ep)
{
    uct_set_ep_failed(&UCS_CLASS_NAME(uct_dc_mlx5_ep_t), ep,
                      &ib_iface->super.super);
}

ucs_status_t uct_dc_mlx5_ep_fc_ctrl(uct_ep_t *tl_ep, unsigned op,
                                    uct_rc_fc_request_t *req)
{
    uintptr_t sender_ep;
    uct_ib_iface_t *ib_iface;
    uct_ib_mlx5_base_av_t av;
    uct_dc_fc_request_t *dc_req;
    uct_dc_mlx5_ep_t *dc_mlx5_ep;
    uct_dc_ep_t *dc_ep         = ucs_derived_of(tl_ep, uct_dc_ep_t);
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_dc_mlx5_iface_t);
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    ucs_assert((sizeof(uint8_t) + sizeof(sender_ep)) <=
                UCT_IB_MLX5_AV_FULL_SIZE);

    UCT_DC_CHECK_RES(&iface->super, dc_ep);
    UCT_DC_MLX5_IFACE_TXQP_GET(iface, dc_ep, txqp, txwq);

    if (op == UCT_RC_EP_FC_PURE_GRANT) {
        ucs_assert(req != NULL);
        dc_req    = ucs_derived_of(req, uct_dc_fc_request_t);
        sender_ep = (uintptr_t)dc_req->sender_ep;
        ib_iface  = &iface->super.super.super;

        /* Note av initialization is copied from exp verbs */
        av.stat_rate_sl = ib_iface->config.sl; /* (attr->static_rate << 4) | attr->sl */
        av.fl_mlid      = ib_iface->path_bits[0] & 0x7f;

        /* lid in dc_req is in BE already  */
        av.rlid         = dc_req->lid | htons(ib_iface->path_bits[0]);
        av.dqp_dct      = htonl(dc_req->dct_num);

        if (!iface->ud_common.config.compact_av) {
            av.dqp_dct |= UCT_IB_MLX5_EXTENDED_UD_AV;
        }

        uct_rc_mlx5_txqp_inline_post(&iface->super.super, IBV_EXP_QPT_DC_INI,
                                     txqp, txwq, MLX5_OPCODE_SEND,
                                     &av /*dummy*/, 0, op, sender_ep, 0,
                                     0, 0,
                                     &av, uct_ib_mlx5_wqe_av_size(&av));
    } else {
        ucs_assert(op == UCT_RC_EP_FC_FLAG_HARD_REQ);
        sender_ep    = (uintptr_t)dc_ep;
        dc_mlx5_ep   = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);

        UCS_STATS_UPDATE_COUNTER(dc_ep->fc.stats,
                                 UCT_RC_FC_STAT_TX_HARD_REQ, 1);

        uct_rc_mlx5_txqp_inline_post(&iface->super.super, IBV_EXP_QPT_DC_INI,
                                     txqp, txwq, MLX5_OPCODE_SEND_IMM,
                                     &dc_mlx5_ep->av /*dummy*/, 0, op, sender_ep,
                                     iface->super.rx.dct->dct_num,
                                     0, 0,
                                     &dc_mlx5_ep->av,
                                     uct_ib_mlx5_wqe_av_size(&dc_mlx5_ep->av));
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
    status = uct_rc_modify_qp(&iface->super.tx.dcis[dci].txqp, IBV_QPS_RESET);
    uct_rc_mlx5_iface_common_sync_cqs_ci(&iface->mlx5_common,
                                         &iface->super.super.super);

    /* Resume posting from to the beginning of the QP */
    uct_ib_mlx5_txwq_reset(&iface->dci_wqs[dci]);

    return status;
}

static int uct_dc_mlx5_iface_is_reachable(const uct_iface_h tl_iface,
                                          const uct_device_addr_t *dev_addr,
                                          const uct_iface_addr_t *iface_addr)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    const uct_ib_address_t *ib_addr = (const void*)dev_addr;
    struct ibv_ah_attr ah_attr;

    /* dc_mlx5 does not support global address */
    uct_ib_iface_fill_ah_attr(&iface->super.super.super, ib_addr, 0, &ah_attr);
    if (ah_attr.is_global) {
        return 0;
    }

    return uct_ib_iface_is_reachable(tl_iface, dev_addr, iface_addr);
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
    .ep_atomic_add64          = uct_dc_mlx5_ep_atomic_add64,
    .ep_atomic_fadd64         = uct_dc_mlx5_ep_atomic_fadd64,
    .ep_atomic_swap64         = uct_dc_mlx5_ep_atomic_swap64,
    .ep_atomic_cswap64        = uct_dc_mlx5_ep_atomic_cswap64,
    .ep_atomic_add32          = uct_dc_mlx5_ep_atomic_add32,
    .ep_atomic_fadd32         = uct_dc_mlx5_ep_atomic_fadd32,
    .ep_atomic_swap32         = uct_dc_mlx5_ep_atomic_swap32,
    .ep_atomic_cswap32        = uct_dc_mlx5_ep_atomic_cswap32,
    .ep_pending_add           = uct_dc_ep_pending_add,
    .ep_pending_purge         = uct_dc_ep_pending_purge,
    .ep_flush                 = uct_dc_mlx5_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .iface_flush              = uct_dc_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_dc_iface_progress_enable,
    .iface_progress_disable   = uct_dc_iface_progress_disable,
    .iface_event_fd_get       = uct_ib_iface_event_fd_get,
    .iface_event_arm          = uct_ib_iface_event_arm,
    .ep_create_connected      = UCS_CLASS_NEW_FUNC_NAME(uct_dc_mlx5_ep_t),
    .ep_destroy               = uct_dc_mlx5_ep_destroy,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_dc_mlx5_iface_t),
    .iface_query              = uct_dc_mlx5_iface_query,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_dc_mlx5_iface_is_reachable,
    .iface_get_address        = uct_dc_iface_get_address,
    },
    .arm_tx_cq                = uct_ib_iface_arm_tx_cq,
    .arm_rx_cq                = uct_ib_iface_arm_rx_cq,
    .handle_failure           = uct_dc_mlx5_iface_handle_failure,
    .set_ep_failed            = uct_dc_mlx5_ep_set_failed
    },
    .fc_ctrl                  = uct_dc_mlx5_ep_fc_ctrl,
    .fc_handler               = uct_dc_iface_fc_handler,
    },
    .reset_dci                = uct_dc_mlx5_iface_reset_dci,
    .progress                 = uct_dc_mlx5_iface_progress
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

static UCS_CLASS_INIT_FUNC(uct_dc_mlx5_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_dc_mlx5_iface_config_t *config = ucs_derived_of(tl_config,
                                                        uct_dc_mlx5_iface_config_t);
    ucs_status_t status;

    ucs_trace_func("");
    UCS_CLASS_CALL_SUPER_INIT(uct_dc_iface_t, &uct_dc_mlx5_iface_ops, md,
                              worker, params, 0, &config->super);

    status = uct_rc_mlx5_iface_common_init(&self->mlx5_common, &self->super.super,
                                           &config->super.super);
    if (status != UCS_OK) {
        goto err;
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
                             ((UCT_IB_MLX5_MAX_BB * MLX5_SEND_WQE_BB) -
                             sizeof(struct mlx5_wqe_raddr_seg) -
                             sizeof(struct mlx5_wqe_ctrl_seg) -
                             UCT_IB_MLX5_AV_FULL_SIZE) /
                             sizeof(struct mlx5_wqe_data_seg));

    /* TODO: only register progress when we have a connection */
    uct_dc_iface_progress_enable(&self->super.super.super.super.super,
                                 UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    ucs_debug("created dc iface %p", self);
    return UCS_OK;

err_rc_mlx5_common_cleanup:
    uct_rc_mlx5_iface_common_cleanup(&self->mlx5_common);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_mlx5_iface_t)
{
    ucs_trace_func("");
    uct_dc_iface_progress_disable(&self->super.super.super.super.super,
                                  UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    uct_rc_mlx5_iface_common_cleanup(&self->mlx5_common);
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
