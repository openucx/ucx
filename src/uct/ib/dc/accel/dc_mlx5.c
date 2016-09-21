/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
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

static UCS_CLASS_INIT_FUNC(uct_dc_mlx5_ep_t,
                           uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);
    const uct_ib_address_t *ib_addr = (const uct_ib_address_t *)dev_addr;
    const uct_dc_iface_addr_t *if_addr = (const uct_dc_iface_addr_t *)iface_addr;
    ucs_status_t status;
    struct ibv_ah *ah;

    ucs_trace_func("");
    UCS_CLASS_CALL_SUPER_INIT(uct_dc_ep_t, &iface->super, if_addr);

    status = uct_ib_iface_create_ah(&iface->super.super.super, ib_addr, 0, &ah);
    if (status != UCS_OK) {
        return UCS_ERR_INVALID_ADDR;
    }

    uct_ib_mlx5_get_av(ah, &self->av);
    mlx5_av_base(&self->av)->key.dc_key      = htonll(UCT_IB_DC_KEY);
    mlx5_av_base(&self->av)->dqp_dct         = htonl(uct_ib_unpack_uint24(if_addr->qp_num) |
                                                     UCT_IB_MLX5_EXTENDED_UD_AV);

    ibv_destroy_ah(ah);
    ucs_debug("created ep %p on iface %p", self, iface);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_mlx5_ep_t)
{
    ucs_trace_func("");
}

UCS_CLASS_DEFINE(uct_dc_mlx5_ep_t, uct_dc_ep_t);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_dc_mlx5_ep_t, uct_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_dc_mlx5_ep_t, uct_ep_t, uct_iface_h, const uct_device_addr_t *,
                          const uct_iface_addr_t *);


static ucs_status_t uct_dc_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_mlx5_iface_t);

    uct_dc_iface_query(&iface->super, iface_attr);
    uct_rc_mlx5_iface_common_query(&iface->super.super, iface_attr, IBV_EXP_QPT_DC_INI);

    /*TODO: remove flags once we have a full functionality */
    iface_attr->cap.flags           = UCT_IFACE_FLAG_AM_SHORT|
                                      UCT_IFACE_FLAG_AM_BCOPY|
                                      UCT_IFACE_FLAG_AM_ZCOPY|
                                      UCT_IFACE_FLAG_PUT_SHORT|
                                      UCT_IFACE_FLAG_PUT_BCOPY|
                                      UCT_IFACE_FLAG_PUT_ZCOPY|
                                      UCT_IFACE_FLAG_GET_BCOPY|
                                      UCT_IFACE_FLAG_GET_ZCOPY|
                                      UCT_IFACE_FLAG_ATOMIC_ADD64|
                                      UCT_IFACE_FLAG_ATOMIC_FADD64|
                                      UCT_IFACE_FLAG_ATOMIC_SWAP64|
                                      UCT_IFACE_FLAG_ATOMIC_CSWAP64| 
                                      UCT_IFACE_FLAG_ATOMIC_ADD32|
                                      UCT_IFACE_FLAG_ATOMIC_FADD32|
                                      UCT_IFACE_FLAG_ATOMIC_SWAP32|
                                      UCT_IFACE_FLAG_ATOMIC_CSWAP32|
                                      UCT_IFACE_FLAG_ATOMIC_DEVICE |
                                      UCT_IFACE_FLAG_PENDING|
                                      UCT_IFACE_FLAG_AM_CB_SYNC|UCT_IFACE_FLAG_CONNECT_TO_IFACE;

    return UCS_OK;
}


#define UCT_DC_MLX5_TXQP_DECL(_txqp, _txwq) \
    uct_rc_txqp_t *_txqp; \
    uct_ib_mlx5_txwq_t *_txwq; 

#define UCT_DC_MLX5_IFACE_TXQP_GET(_iface, _ep,  _txqp, _txwq) \
{ \
    uint8_t dci; \
    dci = (_ep)->super.dci; \
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

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);
    desc->super.sn = txwq->sw_pi;
    uct_rc_mlx5_txqp_dptr_post(&iface->super.super, txqp, txwq, opcode, desc + 1, length, &desc->lkey,
                               am_id, am_hdr, am_hdr_len,
                               rdma_raddr, uct_ib_md_direct_rkey(rdma_rkey),
                               0, 0, 0, &ep->av, MLX5_WQE_CTRL_CQ_UPDATE, IBV_EXP_QPT_DC_INI);
    uct_rc_txqp_add_send_op(txqp, &desc->super);
}


static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_zcopy_post(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep,
                             unsigned opcode, const void *buffer,
                             unsigned length, uct_ib_mem_t *memh,
                             /* SEND */ uint8_t am_id, const void *am_hdr, unsigned am_hdr_len,
                             /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey,
                             uct_completion_t *comp)
{
    uint16_t sn;
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);

    sn = txwq->sw_pi;
    uct_rc_mlx5_txqp_dptr_post(&iface->super.super, txqp, txwq, 
                               opcode, buffer, length, &memh->lkey,
                               am_id, am_hdr, am_hdr_len,
                               rdma_raddr, uct_ib_md_direct_rkey(rdma_rkey),
                               0, 0, 0, &ep->av,
                               MLX5_WQE_CTRL_CQ_UPDATE,
                               IBV_EXP_QPT_DC_INI);

    uct_rc_txqp_add_send_comp(&iface->super.super, txqp, comp, sn);
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_atomic_post(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep,
                              unsigned opcode, uct_rc_iface_send_desc_t *desc, unsigned length,
                              uint64_t remote_addr, uct_rkey_t rkey,
                              uint64_t compare_mask, uint64_t compare, uint64_t swap_add)
{
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);

    desc->super.sn = txwq->sw_pi;
    uct_rc_mlx5_txqp_dptr_post(&iface->super.super, txqp, txwq,
                               opcode, desc + 1, length, &desc->lkey,
                               0, NULL, 0, 
                               remote_addr + ep->super.umr_offset,
                               uct_ib_md_umr_rkey(rkey), 
                               compare_mask, compare, swap_add,
                               &ep->av, MLX5_WQE_CTRL_CQ_UPDATE, IBV_EXP_QPT_DC_INI);

    UCT_TL_EP_STAT_ATOMIC(&ep->super.super);
    uct_rc_txqp_add_send_op(txqp, &desc->super);
}

static inline void uct_dc_mlx5_iface_add_send_comp(uct_dc_mlx5_iface_t *iface,
                                                   uct_dc_mlx5_ep_t *ep,
                                                   uct_completion_t *comp)
{
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);
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
                                     htonll(add), remote_addr, rkey);
}

ucs_status_t uct_dc_mlx5_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uint64_t *result, uct_completion_t *comp)
{
    return uct_dc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_FA, result, 0, sizeof(uint64_t),
                                 remote_addr, rkey, 0, 0, htonll(add), comp);
}

ucs_status_t uct_dc_mlx5_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uint64_t *result, uct_completion_t *comp)
{
    return uct_dc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_MASKED_CS, result, 1,
                                 sizeof(uint64_t), remote_addr, rkey, 0, 0,
                                 htonll(swap), comp);
}

ucs_status_t uct_dc_mlx5_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint64_t *result, uct_completion_t *comp)
{
    return uct_dc_mlx5_ep_atomic(ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t),
                                 MLX5_OPCODE_ATOMIC_CS, result, 0, sizeof(uint64_t),
                                 remote_addr, rkey, 0, htonll(compare), htonll(swap),
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

    UCT_RC_MLX5_CHECK_AM_SHORT(id, length, IBV_EXP_QPT_DC_INI);
    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);

    uct_rc_mlx5_txqp_inline_post(&iface->super.super, txqp, txwq,
                                 MLX5_OPCODE_SEND,
                                 buffer, length, id, hdr,
                                 0, 0, &ep->av,
                                 IBV_EXP_QPT_DC_INI);

    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, sizeof(hdr) + length);
    return UCS_OK;
}

ssize_t uct_dc_mlx5_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                 uct_pack_callback_t pack_cb, void *arg)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    uct_rc_iface_send_desc_t *desc;
    size_t length;

    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_AM_BCOPY_DESC(&iface->super.super, &iface->super.super.tx.mp, desc,
                                      id, pack_cb, arg, &length);

    uct_dc_mlx5_iface_bcopy_post(iface, ep,
                                 MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_RAW,
                                 sizeof(uct_rc_hdr_t) + length, 0, NULL, 0, 0, 0, desc);

    UCT_TL_EP_STAT_OP(&ep->super.super, AM, BCOPY, length);
    return length;
}


ucs_status_t uct_dc_mlx5_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                     unsigned header_length, const void *payload,
                                     size_t length, uct_mem_h memh,
                                     uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);

    UCT_RC_MLX5_CHECK_AM_ZCOPY(id, header_length, length, 
                               iface->super.super.super.config.seg_size, IBV_EXP_QPT_DC_INI);

    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    uct_dc_mlx5_iface_zcopy_post(iface, ep, MLX5_OPCODE_SEND, payload, length, memh,
                                 id, header, header_length, 0, 0, comp);

    UCT_TL_EP_STAT_OP(&ep->super.super, AM, ZCOPY, header_length + length);

    return UCS_INPROGRESS;
}


ucs_status_t uct_dc_mlx5_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                      unsigned length, uint64_t remote_addr,
                                      uct_rkey_t rkey)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);
    UCT_DC_MLX5_TXQP_DECL(txqp, txwq);

    UCT_RC_MLX5_CHECK_PUT_SHORT(length, IBV_EXP_QPT_DC_INI);

    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    UCT_DC_MLX5_IFACE_TXQP_GET(iface, ep, txqp, txwq);
    uct_rc_mlx5_txqp_inline_post(&iface->super.super, txqp, txwq,
                                 MLX5_OPCODE_RDMA_WRITE,
                                 buffer, length, 0, 0,
                                 remote_addr, uct_ib_md_direct_rkey(rkey),
                                 &ep->av, IBV_EXP_QPT_DC_INI);

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

ucs_status_t uct_dc_mlx5_ep_put_zcopy(uct_ep_h tl_ep, const void *buffer, size_t length,
                                      uct_mem_h memh, uint64_t remote_addr,
                                      uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);

    UCT_CHECK_LENGTH(length, UCT_IB_MAX_MESSAGE_SIZE, "put_zcopy");
    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    uct_dc_mlx5_iface_zcopy_post(iface, ep, MLX5_OPCODE_RDMA_WRITE, buffer, length,
                                 memh, 0, NULL, 0, remote_addr, rkey, comp);

    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, ZCOPY, length);
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

    UCT_CHECK_LENGTH(length, iface->super.super.super.config.seg_size, "get_bcopy");
    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_GET_BCOPY_DESC(&iface->super.super, &iface->super.super.tx.mp, 
                                       desc, unpack_cb, comp, arg, length);
    uct_dc_mlx5_iface_bcopy_post(iface, ep, 
                                 MLX5_OPCODE_RDMA_READ, length, 0, NULL, 0,
                                 remote_addr, rkey, desc);
    UCT_TL_EP_STAT_OP(&ep->super.super, GET, BCOPY, length);
    return UCS_INPROGRESS;
}


ucs_status_t uct_dc_mlx5_ep_get_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                      uct_mem_h memh, uint64_t remote_addr,
                                      uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_mlx5_iface_t);
    uct_dc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_mlx5_ep_t);

    UCT_CHECK_LENGTH(length, UCT_IB_MAX_MESSAGE_SIZE, "get_zcopy");
    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    uct_dc_mlx5_iface_zcopy_post(iface, ep, MLX5_OPCODE_RDMA_READ, buffer, length,
                                 memh, 0, NULL, 0, remote_addr, rkey, comp); 
    UCT_TL_EP_STAT_OP(&ep->super.super, GET, ZCOPY, length);
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

    cqe = uct_ib_mlx5_get_cqe(&iface->super.super.super, &iface->mlx5_common.tx.cq,
                              UCT_IB_MLX5_CQE64_SIZE_LOG);
    if (cqe == NULL) {
        return;
    }
    UCS_STATS_UPDATE_COUNTER(iface->super.super.stats, UCT_RC_IFACE_STAT_TX_COMPLETION, 1);

    ucs_memory_cpu_load_fence();

    ucs_assertv(!(cqe->op_own & (MLX5_INLINE_SCATTER_32|MLX5_INLINE_SCATTER_64)),
                "tx inline scatter not supported");

    qp_num = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
    dci = uct_dc_iface_dci_find(&iface->super, qp_num);
    txqp = &iface->super.tx.dcis[dci].txqp;
    txwq = &iface->dci_wqs[dci];

    hw_ci = ntohs(cqe->wqe_counter);
    uct_rc_txqp_available_set(txqp, uct_ib_mlx5_txwq_update_bb(txwq, hw_ci));
    uct_rc_txqp_completion(txqp, hw_ci);
    iface->super.super.tx.cq_available++;

    uct_dc_iface_dci_put(&iface->super, dci);
    if (uct_dc_iface_dci_can_alloc(&iface->super)) {
        ucs_arbiter_dispatch(&iface->super.super.tx.arbiter, 1, uct_dc_iface_dci_do_pending_wait, NULL);
    }
    ucs_arbiter_dispatch(&iface->super.tx.dci_arbiter, 1, uct_dc_iface_dci_do_pending_tx, NULL);
}

/* TODO: make a macro that defines progress func */
static void uct_dc_mlx5_iface_progress(void *arg)
{
    uct_dc_mlx5_iface_t *iface = arg;
    ucs_status_t status;

    status = uct_rc_mlx5_iface_common_poll_rx(&iface->mlx5_common, &iface->super.super);
    if (status == UCS_ERR_NO_PROGRESS) {
        uct_dc_mlx5_poll_tx(iface);
    }
}

static UCS_F_NOINLINE void uct_dc_mlx5_iface_handle_failure(uct_ib_iface_t *ib_iface,
                                                            void *arg)
{
    struct mlx5_cqe64 *cqe = arg;
    uct_ib_mlx5_completion_with_err((void*)cqe, 0);
}

static void UCS_CLASS_DELETE_FUNC_NAME(uct_dc_mlx5_iface_t)(uct_iface_t*);

static uct_rc_iface_ops_t uct_dc_mlx5_iface_ops = {
    {
        {
            .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_dc_mlx5_iface_t),
            .iface_query              = uct_dc_mlx5_iface_query,
            .iface_get_device_address = uct_ib_iface_get_device_address,
            .iface_is_reachable       = uct_ib_iface_is_reachable,
            .iface_release_am_desc    = uct_ib_iface_release_am_desc, 
            .iface_get_address        = uct_dc_iface_get_address,

            .iface_flush              = uct_dc_iface_flush,

            .ep_create_connected      = UCS_CLASS_NEW_FUNC_NAME(uct_dc_mlx5_ep_t),
            .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_dc_mlx5_ep_t),

            .ep_am_short              = uct_dc_mlx5_ep_am_short,
            .ep_am_bcopy              = uct_dc_mlx5_ep_am_bcopy,
            .ep_am_zcopy              = uct_dc_mlx5_ep_am_zcopy,

            .ep_put_short             = uct_dc_mlx5_ep_put_short,
            .ep_put_bcopy             = uct_dc_mlx5_ep_put_bcopy,
            .ep_put_zcopy             = uct_dc_mlx5_ep_put_zcopy,

            .ep_get_bcopy             = uct_dc_mlx5_ep_get_bcopy,
            .ep_get_zcopy             = uct_dc_mlx5_ep_get_zcopy,

            .ep_atomic_add64          = uct_dc_mlx5_ep_atomic_add64,
            .ep_atomic_fadd64         = uct_dc_mlx5_ep_atomic_fadd64,
            .ep_atomic_swap64         = uct_dc_mlx5_ep_atomic_swap64,
            .ep_atomic_cswap64        = uct_dc_mlx5_ep_atomic_cswap64,

            .ep_atomic_add32          = uct_dc_mlx5_ep_atomic_add32,
            .ep_atomic_fadd32         = uct_dc_mlx5_ep_atomic_fadd32,
            .ep_atomic_swap32         = uct_dc_mlx5_ep_atomic_swap32,
            .ep_atomic_cswap32        = uct_dc_mlx5_ep_atomic_cswap32,

            .ep_flush                 = uct_dc_mlx5_ep_flush,

            .ep_pending_add           = uct_dc_ep_pending_add,
            .ep_pending_purge         = uct_dc_ep_pending_purge,
        },
        .arm_tx_cq                = uct_ib_iface_arm_tx_cq,
        .arm_rx_cq                = uct_ib_iface_arm_rx_cq,
        .handle_failure           = uct_dc_mlx5_iface_handle_failure
    },
    .fc_ctrl                  = NULL /* TODO: */
};


static ucs_status_t uct_dc_mlx5_iface_init_dcis(uct_dc_mlx5_iface_t *iface)
{
    ucs_status_t status;
    uint16_t bb_max;
    int i;

    bb_max = 0;
    for (i = 0; i < iface->super.tx.ndci; i++) {
        status = uct_ib_mlx5_get_txwq(iface->super.super.super.super.worker,
                                      iface->super.tx.dcis[i].txqp.qp,
                                      &iface->dci_wqs[i]);
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
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    uct_dc_iface_config_t *config = ucs_derived_of(tl_config,
                                                   uct_dc_iface_config_t);
    ucs_status_t status;

    ucs_trace_func("");
    UCS_CLASS_CALL_SUPER_INIT(uct_dc_iface_t, &uct_dc_mlx5_iface_ops, md,
                              worker, dev_name, rx_headroom, 0, config);

    status = uct_rc_mlx5_iface_common_init(&self->mlx5_common, &self->super.super,
                                           &config->super.super);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_dc_mlx5_iface_init_dcis(self);
    if (status != UCS_OK) {
        goto err_common_cleanup;
    }

    /* TODO: only register progress when we have a connection */
    uct_worker_progress_register(worker, uct_dc_mlx5_iface_progress, self);
    ucs_debug("created dc iface %p", self);
    return UCS_OK;

err_common_cleanup:
    uct_rc_mlx5_iface_common_cleanup(&self->mlx5_common);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_mlx5_iface_t)
{
    ucs_trace_func("");
    uct_worker_progress_unregister(self->super.super.super.super.worker,
                                   uct_dc_mlx5_iface_progress, self);
    uct_rc_mlx5_iface_common_cleanup(&self->mlx5_common);
}

UCS_CLASS_DEFINE(uct_dc_mlx5_iface_t, uct_dc_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_dc_mlx5_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const char*, size_t,
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
                                            (ib_md->eth_pause ? 0 : UCT_IB_DEVICE_FLAG_LINK_IB),
                                            resources_p, num_resources_p);
}


UCT_TL_COMPONENT_DEFINE(uct_dc_mlx5_tl,
                        uct_dc_mlx5_query_resources,
                        uct_dc_mlx5_iface_t,
                        "dc_mlx5",
                        "DC_MLX5_",
                        uct_dc_iface_config_table,
                        uct_dc_iface_config_t);
UCT_MD_REGISTER_TL(&uct_ib_mdc, &uct_dc_mlx5_tl);

