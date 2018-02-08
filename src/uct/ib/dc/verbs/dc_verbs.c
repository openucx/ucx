/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "dc_verbs.h"

#include <uct/ib/rc/verbs/rc_verbs_common.h>
#include <uct/api/uct.h>
#include <uct/ib/base/ib_log.h>
#include <uct/base/uct_md.h>
#include <ucs/arch/bitops.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>
#include <string.h>

static ucs_config_field_t uct_dc_verbs_iface_config_table[] = {
  {"DC_", "", NULL,
   ucs_offsetof(uct_dc_verbs_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_dc_iface_config_table)},

  {"", "", NULL,
   ucs_offsetof(uct_dc_verbs_iface_config_t, verbs_common),
   UCS_CONFIG_TYPE_TABLE(uct_rc_verbs_iface_common_config_table)},

  {NULL}
};

static UCS_CLASS_INIT_FUNC(uct_dc_verbs_ep_t,
                           uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    uct_dc_verbs_iface_t      *iface    = ucs_derived_of(tl_iface, uct_dc_verbs_iface_t);
    uct_ib_iface_t            *ib_iface = &iface->super.super.super;
    const uct_ib_address_t    *ib_addr  = (const uct_ib_address_t *)dev_addr;
    const uct_dc_iface_addr_t *if_addr  = (const uct_dc_iface_addr_t *)iface_addr;
    ucs_status_t               status;
    struct ibv_ah_attr         ah_attr;

    ucs_trace_func("");
    UCS_CLASS_CALL_SUPER_INIT(uct_dc_ep_t, &iface->super, if_addr);

    uct_ib_iface_fill_ah_attr_from_addr(ib_iface, ib_addr, ib_iface->path_bits[0], &ah_attr);
    status = uct_ib_iface_create_ah(ib_iface, &ah_attr, &self->ah);
    if (status != UCS_OK) {
        return status;
    }

    self->dest_qpn = uct_ib_unpack_uint24(if_addr->qp_num);
    ucs_debug("created ep %p on iface %p", self, iface);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_verbs_ep_t)
{
    ucs_trace_func("");
    ibv_destroy_ah(self->ah);
}

UCS_CLASS_DEFINE(uct_dc_verbs_ep_t, uct_dc_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_dc_verbs_ep_t, uct_ep_t, uct_iface_h, const uct_device_addr_t *,
                          const uct_iface_addr_t *);

static void uct_dc_verbs_ep_destroy(uct_ep_h tl_ep)
{
    uct_dc_ep_cleanup(tl_ep, &UCS_CLASS_NAME(uct_dc_verbs_ep_t));
}

static ucs_status_t uct_dc_verbs_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_verbs_iface_t);
    uct_rc_verbs_iface_common_t *verbs_common = &iface->verbs_common;
    ucs_status_t status;

    status = uct_dc_iface_query(&iface->super, iface_attr,
                                verbs_common->config.max_inline,
                                verbs_common->config.max_inline,
                                verbs_common->config.short_desc_size,
                                uct_ib_iface_get_max_iov(&iface->super.super.super) - 1);
    if (status != UCS_OK) {
        return status;
    }

    iface_attr->overhead       = 75e-9; /* Software overhead */
    iface_attr->iface_addr_len = sizeof(uct_dc_iface_addr_t);

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
uct_dc_verbs_iface_post_send_to_dci(uct_dc_verbs_iface_t* iface,
                                    struct ibv_exp_send_wr *wr,
                                    uint8_t dci, struct ibv_ah *ah,
                                    uint32_t dct_num, uint64_t send_flags)
{
    struct ibv_exp_send_wr *bad_wr;
    int ret;
    uct_rc_txqp_t *txqp;

    txqp = &iface->super.tx.dcis[dci].txqp;
    /* TODO: check tx moderation */
    send_flags |= IBV_SEND_SIGNALED;

    wr->exp_send_flags    = send_flags;
    wr->wr_id             = uct_rc_txqp_unsignaled(txqp);
    wr->dc.ah             = ah;
    wr->dc.dct_number     = dct_num;
    wr->dc.dct_access_key = UCT_IB_KEY;

    uct_ib_log_exp_post_send(&iface->super.super.super, txqp->qp, wr,
                             (wr->exp_opcode == IBV_EXP_WR_SEND) ?
                             uct_rc_ep_am_packet_dump : NULL);

    ret = ibv_exp_post_send(txqp->qp, wr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_send() returned %d (%m)", ret);
    }

    uct_rc_verbs_txqp_posted(txqp, &iface->dcis_txcnt[dci],
                             &iface->super.super, send_flags & IBV_SEND_SIGNALED);
}

static UCS_F_ALWAYS_INLINE void
uct_dc_verbs_iface_post_send(uct_dc_verbs_iface_t* iface, uct_dc_verbs_ep_t *ep,
                             struct ibv_exp_send_wr *wr, uint64_t send_flags)
{
    uct_dc_verbs_iface_post_send_to_dci(iface, wr, ep->super.dci, ep->ah,
                                        ep->dest_qpn, send_flags);
}

static UCS_F_ALWAYS_INLINE void
uct_dc_verbs_iface_post_send_desc(uct_dc_verbs_iface_t *iface,
                                  uct_dc_verbs_ep_t *ep,
                                  struct ibv_exp_send_wr *wr,
                                  uct_rc_iface_send_desc_t *desc, uint64_t send_flags)
{
    UCT_RC_VERBS_FILL_DESC_WR(wr, desc);
    uct_dc_verbs_iface_post_send(iface, ep, wr, send_flags);
    uct_rc_txqp_add_send_op_sn(&iface->super.tx.dcis[ep->super.dci].txqp, &desc->super,
                               iface->dcis_txcnt[ep->super.dci].pi);
}

static inline void uct_dc_verbs_iface_add_send_comp(uct_dc_verbs_iface_t *iface,
                                                    uct_dc_verbs_ep_t *ep,
                                                    uct_completion_t *comp)
{
    uint8_t dci = ep->super.dci;
    uct_rc_txqp_add_send_comp(&iface->super.super, &iface->super.tx.dcis[dci].txqp,
                              comp, iface->dcis_txcnt[dci].pi);
}

static inline ucs_status_t
uct_dc_verbs_ep_rdma_zcopy(uct_dc_verbs_ep_t *ep, const uct_iov_t *iov,
                           size_t iovcnt, uint64_t remote_addr, uct_rkey_t rkey,
                           uct_completion_t *comp, int opcode)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_dc_verbs_iface_t);
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge[UCT_IB_MAX_IOV];
    size_t sge_cnt;

    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    sge_cnt = uct_ib_verbs_sge_fill_iov(sge, iov, iovcnt);
    UCT_SKIP_ZERO_LENGTH(sge_cnt);
    UCT_RC_VERBS_FILL_RDMA_WR_IOV(wr, wr.exp_opcode, opcode, sge, sge_cnt,
                                  remote_addr, rkey);
    wr.next = NULL;

    uct_dc_verbs_iface_post_send(iface, ep, &wr, IBV_SEND_SIGNALED);
    uct_dc_verbs_iface_add_send_comp(iface, ep, comp);
    return UCS_INPROGRESS;
}


ucs_status_t uct_dc_verbs_ep_get_bcopy(uct_ep_h tl_ep,
                                       uct_unpack_callback_t unpack_cb,
                                       void *arg, size_t length,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge;

    UCT_CHECK_LENGTH(length, 0, iface->super.super.super.config.seg_size, "get_bcopy");
    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_GET_BCOPY_DESC(&iface->super.super, &iface->super.super.tx.mp,
                                       desc, unpack_cb, comp, arg, length);
    UCT_RC_VERBS_FILL_RDMA_WR(wr, wr.exp_opcode, IBV_EXP_WR_RDMA_READ, sge,
                              length, remote_addr, rkey);
    UCT_TL_EP_STAT_OP(&ep->super.super, GET, BCOPY, length);
    uct_dc_verbs_iface_post_send_desc(iface, ep, &wr, desc, IBV_SEND_SIGNALED);

    return UCS_INPROGRESS;
}

ucs_status_t uct_dc_verbs_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    uct_dc_verbs_ep_t *ep  = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_ib_iface_t  *iface = ucs_derived_of(tl_ep->iface, uct_ib_iface_t);
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt, uct_ib_iface_get_max_iov(iface),
                       "uct_dc_verbs_ep_get_zcopy");
    UCT_CHECK_LENGTH(uct_iov_total_length(iov, iovcnt),
                     iface->config.max_inl_resp + 1, UCT_IB_MAX_MESSAGE_SIZE,
                     "get_zcopy");

    status = uct_dc_verbs_ep_rdma_zcopy(ep, iov, iovcnt, remote_addr, rkey, comp,
                                        IBV_WR_RDMA_READ);
    if (status == UCS_INPROGRESS) {
        UCT_TL_EP_STAT_OP(&ep->super.super, GET, ZCOPY,
                          uct_iov_total_length(iov, iovcnt));
    }
    return status;
}

ucs_status_t uct_dc_verbs_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                       unsigned length, uint64_t remote_addr,
                                       uct_rkey_t rkey)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);

    UCT_CHECK_LENGTH(length, 0, iface->verbs_common.config.max_inline, "put_short");

    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_VERBS_FILL_INL_PUT_WR(iface, remote_addr, rkey, buffer, length);
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, SHORT, length);
    uct_dc_verbs_iface_post_send(iface, ep, &iface->inl_rwrite_wr,
                                 IBV_SEND_INLINE|IBV_SEND_SIGNALED);

    return UCS_OK;
}

ssize_t uct_dc_verbs_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                  void *arg, uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge;
    size_t length;

    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_PUT_BCOPY_DESC(&iface->super.super, &iface->super.super.tx.mp,
                                       desc, pack_cb, arg, length);
    UCT_RC_VERBS_FILL_RDMA_WR(wr, wr.exp_opcode, IBV_EXP_WR_RDMA_WRITE, sge,
                              length, remote_addr, rkey);
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, BCOPY, length);
    uct_dc_verbs_iface_post_send_desc(iface, ep, &wr, desc, IBV_SEND_SIGNALED);

    return length;
}

ucs_status_t uct_dc_verbs_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    uct_dc_verbs_ep_t *ep  = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_ib_iface_t  *iface = ucs_derived_of(tl_ep->iface, uct_ib_iface_t);
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt, uct_ib_iface_get_max_iov(iface),
                       "uct_dc_verbs_ep_put_zcopy");
    UCT_CHECK_LENGTH(uct_iov_total_length(iov, iovcnt), 0, UCT_IB_MAX_MESSAGE_SIZE,
                     "put_zcopy");

    status = uct_dc_verbs_ep_rdma_zcopy(ep, iov, iovcnt, remote_addr, rkey, comp,
                                        IBV_WR_RDMA_WRITE);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, PUT, ZCOPY,
                                 uct_iov_total_length(iov, iovcnt));
    return status;
}

ucs_status_t uct_dc_verbs_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                      const void *buffer, unsigned length)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_rc_verbs_iface_common_t *verbs_common = &iface->verbs_common;

    UCT_RC_CHECK_AM_SHORT(id, length, verbs_common->config.max_inline);

    UCT_DC_CHECK_RES_AND_FC(&iface->super, &ep->super);
    uct_rc_verbs_iface_fill_inl_am_sge(verbs_common, id, hdr, buffer, length);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, sizeof(hdr) + length);
    uct_dc_verbs_iface_post_send(iface, ep, &iface->inl_am_wr,
                                 IBV_SEND_INLINE | IBV_SEND_SOLICITED);
    UCT_RC_UPDATE_FC_WND(&iface->super.super, &ep->super.fc);

    return UCS_OK;
}

ssize_t uct_dc_verbs_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                 uct_pack_callback_t pack_cb, void *arg,
                                 unsigned flags)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge;
    size_t length;

    UCT_CHECK_AM_ID(id);

    UCT_DC_CHECK_RES_AND_FC(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_AM_BCOPY_DESC(&iface->super.super,
                                      &iface->super.super.tx.mp, desc, id,
                                      pack_cb, arg, &length);
    UCT_RC_VERBS_FILL_AM_BCOPY_WR(wr, sge, length + sizeof(uct_rc_hdr_t),
                                  wr.exp_opcode);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, BCOPY, length);
    uct_dc_verbs_iface_post_send_desc(iface, ep, &wr, desc, IBV_SEND_SOLICITED);
    UCT_RC_UPDATE_FC_WND(&iface->super.super, &ep->super.fc);

    return length;
}


ucs_status_t uct_dc_verbs_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                      unsigned header_length, const uct_iov_t *iov,
                                      size_t iovcnt, unsigned flags,
                                      uct_completion_t *comp)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_rc_verbs_iface_common_t *verbs_common = &iface->verbs_common;
    uct_rc_iface_send_desc_t *desc;
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge[UCT_IB_MAX_IOV]; /* The first sge is reserved for the header */
    int send_flags;
    size_t sge_cnt;

    UCT_CHECK_IOV_SIZE(iovcnt,
                       uct_ib_iface_get_max_iov(&iface->super.super.super) - 1,
                       "uct_dc_verbs_ep_am_zcopy");
    UCT_RC_CHECK_AM_ZCOPY(id, header_length, uct_iov_total_length(iov, iovcnt),
                          verbs_common->config.short_desc_size,
                          iface->super.super.super.config.seg_size);
    UCT_DC_CHECK_RES_AND_FC(&iface->super, &ep->super);

    UCT_RC_IFACE_GET_TX_AM_ZCOPY_DESC(&iface->super.super,
                                      &verbs_common->short_desc_mp, desc, id,
                                      header, header_length, comp, &send_flags);

    sge[0].length = sizeof(uct_rc_hdr_t) + header_length;
    sge_cnt = uct_ib_verbs_sge_fill_iov(sge + 1, iov, iovcnt);
    UCT_RC_VERBS_FILL_AM_ZCOPY_WR_IOV(wr, sge, sge_cnt + 1, wr.exp_opcode);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, ZCOPY,
                      header_length + uct_iov_total_length(iov, iovcnt));
    uct_dc_verbs_iface_post_send_desc(iface, ep, &wr, desc,
                                      send_flags | IBV_SEND_SOLICITED);
    UCT_RC_UPDATE_FC_WND(&iface->super.super, &ep->super.fc);

    return UCS_INPROGRESS;
}


static UCS_F_ALWAYS_INLINE void
uct_dc_verbs_iface_atomic_post(uct_dc_verbs_iface_t *iface, uct_dc_verbs_ep_t *ep,
                               int opcode, uint64_t compare_add,
                               uint64_t swap, uint64_t remote_addr, uct_rkey_t rkey,
                               uct_rc_iface_send_desc_t *desc, int force_sig)
{
    uint32_t ib_rkey = uct_ib_resolve_atomic_rkey(rkey, ep->super.atomic_mr_offset,
                                                  &remote_addr);
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge;

    UCT_RC_VERBS_FILL_ATOMIC_WR(wr, wr.exp_opcode, sge, opcode, compare_add,
                                swap, remote_addr, ib_rkey);
    UCT_TL_EP_STAT_ATOMIC(&ep->super.super);
    uct_dc_verbs_iface_post_send_desc(iface, ep, &wr, desc, force_sig);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_dc_verbs_ep_atomic(uct_dc_verbs_ep_t *ep, int opcode, void *result,
                       uint64_t compare_add, uint64_t swap, uint64_t remote_addr,
                       uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_dc_verbs_iface_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_ATOMIC_DESC(&iface->super.super, &iface->verbs_common.short_desc_mp, desc,
                                    iface->super.super.config.atomic64_handler,
                                    result, comp);
    uct_dc_verbs_iface_atomic_post(iface, ep, opcode, compare_add, swap, remote_addr,
                                   rkey, desc, IBV_SEND_SIGNALED);
    return UCS_INPROGRESS;
}

#if HAVE_IB_EXT_ATOMICS
static UCS_F_ALWAYS_INLINE void
uct_dc_verbs_iface_ext_atomic_post(uct_dc_verbs_iface_t *iface, uct_dc_verbs_ep_t *ep,
                                   int opcode, uint32_t length, uint32_t compare_mask,
                                   uint64_t compare_add, uint64_t swap, uint64_t remote_addr,
                                   uct_rkey_t rkey, uct_rc_iface_send_desc_t *desc, int force_sig)
{
    struct ibv_sge sge;
    struct ibv_exp_send_wr wr;

    uct_rc_verbs_fill_ext_atomic_wr(&wr, &sge, opcode, length, compare_mask,
                                    compare_add, swap, remote_addr, rkey, ep->super.atomic_mr_offset);
    UCT_TL_EP_STAT_ATOMIC(&ep->super.super);
    uct_dc_verbs_iface_post_send_desc(iface, ep, &wr, desc, force_sig|IBV_EXP_SEND_EXT_ATOMIC_INLINE);
}


static UCS_F_ALWAYS_INLINE ucs_status_t
uct_dc_verbs_ep_ext_atomic(uct_dc_verbs_ep_t *ep, int opcode, void *result,
                           uint32_t length, uint64_t compare_mask,
                           uint64_t compare_add, uint64_t swap, uint64_t remote_addr,
                           uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_dc_verbs_iface_t);
    uct_rc_iface_send_desc_t *desc;
    uct_rc_send_handler_t handler = uct_rc_iface_atomic_handler(&iface->super.super, 1, length);

    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_ATOMIC_DESC(&iface->super.super, &iface->verbs_common.short_desc_mp, desc,
                                    handler, result, comp);
    uct_dc_verbs_iface_ext_atomic_post(iface, ep, opcode, length, compare_mask,
                                       compare_add, swap, remote_addr,
                                       rkey, desc, IBV_SEND_SIGNALED);
    return UCS_INPROGRESS;
}
#endif

ucs_status_t uct_dc_verbs_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey)
{

    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;

    /* TODO don't allocate descriptor - have dummy buffer */
    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_ATOMIC_ADD_DESC(&iface->super.super, &iface->verbs_common.short_desc_mp, desc);

    uct_dc_verbs_iface_atomic_post(iface, ep,
                                   IBV_WR_ATOMIC_FETCH_AND_ADD, add, 0,
                                   remote_addr, rkey, desc,
                                   IBV_SEND_SIGNALED);
    return UCS_OK;
}

ucs_status_t uct_dc_verbs_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint64_t *result, uct_completion_t *comp)
{

    return uct_dc_verbs_ep_atomic(ucs_derived_of(tl_ep, uct_dc_verbs_ep_t),
                                  IBV_WR_ATOMIC_FETCH_AND_ADD, result, add, 0,
                                  remote_addr, rkey, comp);
}


ucs_status_t uct_dc_verbs_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint64_t *result, uct_completion_t *comp)
{
#if HAVE_IB_EXT_ATOMICS
    return uct_dc_verbs_ep_ext_atomic(ucs_derived_of(tl_ep, uct_dc_verbs_ep_t),
                                      IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP,
                                      result, sizeof(uint64_t), 0, 0, swap, remote_addr,
                                      rkey, comp);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_dc_verbs_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uint64_t *result, uct_completion_t *comp)
{
    return uct_dc_verbs_ep_atomic(ucs_derived_of(tl_ep, uct_dc_verbs_ep_t),
                                  IBV_WR_ATOMIC_CMP_AND_SWP, result, compare, swap,
                                  remote_addr, rkey, comp);
}


ucs_status_t uct_dc_verbs_ep_atomic_add32(uct_ep_h tl_ep, uint32_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey)
{
#if HAVE_IB_EXT_ATOMICS
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_ATOMIC_ADD_DESC(&iface->super.super, &iface->verbs_common.short_desc_mp, desc);

    /* TODO don't allocate descriptor - have dummy buffer */
    uct_dc_verbs_iface_ext_atomic_post(iface, ep, IBV_EXP_WR_EXT_MASKED_ATOMIC_FETCH_AND_ADD,
                                       sizeof(uint32_t), 0, add, 0, remote_addr,
                                       rkey, desc, IBV_EXP_SEND_SIGNALED);
    return UCS_OK;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_dc_verbs_ep_atomic_fadd32(uct_ep_h tl_ep, uint32_t add,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint32_t *result, uct_completion_t *comp)
{
#if HAVE_IB_EXT_ATOMICS
    return uct_dc_verbs_ep_ext_atomic(ucs_derived_of(tl_ep, uct_dc_verbs_ep_t),
                                      IBV_EXP_WR_EXT_MASKED_ATOMIC_FETCH_AND_ADD,
                                      result, sizeof(uint32_t), 0, add, 0,
                                      remote_addr, rkey, comp);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_dc_verbs_ep_atomic_swap32(uct_ep_h tl_ep, uint32_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint32_t *result, uct_completion_t *comp)
{
#if HAVE_IB_EXT_ATOMICS
    return uct_dc_verbs_ep_ext_atomic(ucs_derived_of(tl_ep, uct_dc_verbs_ep_t),
                                      IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP,
                                   result, sizeof(uint32_t), 0, 0, swap,
                                   remote_addr, rkey, comp);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_dc_verbs_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uint32_t *result, uct_completion_t *comp)
{
#if HAVE_IB_EXT_ATOMICS
    return uct_dc_verbs_ep_ext_atomic(ucs_derived_of(tl_ep, uct_dc_verbs_ep_t),
                                      IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP,
                                      result, sizeof(uint32_t), (uint32_t)(-1),
                                      compare, swap, remote_addr, rkey, comp);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}


ucs_status_t uct_dc_verbs_ep_flush(uct_ep_h tl_ep, unsigned flags, uct_completion_t *comp)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t    *ep    = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    ucs_status_t         status;

    status = uct_dc_ep_flush(tl_ep, flags, comp);
    if (status == UCS_OK) {
        return UCS_OK; /* all sends completed */
    }

    if (status == UCS_INPROGRESS) {
        ucs_assert(ep->super.dci != UCT_DC_EP_NO_DCI);
        uct_dc_verbs_iface_add_send_comp(iface, ep, comp);
    }
    return status;
}

ucs_status_t uct_dc_verbs_iface_create_ah(uct_dc_iface_t *dc_iface, uint16_t lid,
                                          union ibv_gid *gid, struct ibv_ah **ah_p)
{
    uct_ib_iface_t *iface = &dc_iface->super.super;
    struct ibv_ah_attr ah_attr;
    struct ibv_ah *ah;
    ucs_status_t status;

    uct_ib_iface_fill_ah_attr_from_gid_lid(iface, lid, gid, iface->path_bits[0], &ah_attr);

    status = uct_ib_iface_create_ah(iface, &ah_attr, &ah);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    *ah_p = ah;
    return UCS_OK;
}

static void uct_dc_verbs_handle_failure(uct_ib_iface_t *ib_iface, void *arg,
                                        ucs_status_t status)
{
    struct ibv_wc *wc = arg;

    ucs_log(ib_iface->super.config.failure_level,
            "Send completion with error on qp 0x%x: %s syndrome 0x%x",
            wc->qp_num, ibv_wc_status_str(wc->status), wc->vendor_err);
    uct_dc_handle_failure(ib_iface, wc->qp_num, status);
}

static ucs_status_t uct_dc_verbs_ep_set_failed(uct_ib_iface_t *iface,
                                               uct_ep_h ep, ucs_status_t status)
{
    return uct_set_ep_failed(&UCS_CLASS_NAME(uct_dc_verbs_ep_t), ep,
                             &iface->super.super, status);
}

static ucs_status_t uct_dc_verbs_reset_dci(uct_dc_iface_t *dc_iface, int dci)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(dc_iface, uct_dc_verbs_iface_t);
    uint16_t new_ci;

    new_ci = iface->dcis_txcnt[dci].pi;
    ucs_debug("iface %p resetting dci[%d], ci %d --> %d", iface, dci,
              iface->dcis_txcnt[dci].ci, new_ci);
    iface->dcis_txcnt[dci].ci = new_ci;

    return uct_rc_modify_qp(&dc_iface->tx.dcis[dci].txqp, IBV_QPS_RESET);
}

/* Send either request for grants or grant message. Request includes ep
 * structure address, which will be received back in a grant message.
 * This will help to determine the particular ep targeted by the grant.
 */
ucs_status_t uct_dc_verbs_ep_fc_ctrl(uct_ep_h tl_ep, unsigned op,
                                     uct_rc_fc_request_t *req)
{
    struct ibv_exp_send_wr wr;
    struct ibv_ah *ah;
    ucs_status_t status;
    uct_dc_fc_request_t *dc_req;
    uct_dc_verbs_ep_t *dc_verbs_ep;
    uct_dc_ep_t *dc_ep          = ucs_derived_of(tl_ep, uct_dc_ep_t);
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_dc_verbs_iface_t);
    uct_rc_hdr_t *hdr           = &iface->verbs_common.am_inl_hdr.rc_hdr;
    uct_ib_iface_t *ib_iface    = &iface->super.super.super;
    uct_dc_fc_sender_data_t sender = {
        .ep               = (uintptr_t)dc_ep,
        .global.gid       = ib_iface->gid,
        .global.is_global = ib_iface->addr_type != UCT_IB_ADDRESS_TYPE_LINK_LOCAL};

    ucs_assert(sizeof(*hdr) + sizeof(sender) <=
               iface->verbs_common.config.max_inline);

    UCT_DC_CHECK_RES(&iface->super, dc_ep);

    hdr->am_id                            = op;
    wr.sg_list                            = iface->verbs_common.inl_sge;
    wr.num_sge                            = 2;
    wr.dc.dct_access_key                  = UCT_IB_KEY;
    wr.next                               = NULL;

    iface->verbs_common.inl_sge[0].addr   = (uintptr_t)hdr;
    iface->verbs_common.inl_sge[0].length = sizeof(*hdr);

    dc_req = ucs_derived_of(req, uct_dc_fc_request_t);

    if (op == UCT_RC_EP_FC_PURE_GRANT) {
        ucs_assert(req != NULL);

        status = uct_dc_verbs_iface_create_ah(
                    &iface->super, dc_req->lid,
                    dc_req->sender.global.is_global ? (void*)&dc_req->sender.global.gid : NULL,
                    &ah);
        if (status != UCS_OK) {
            return status;
        }
        wr.exp_opcode                         = IBV_EXP_WR_SEND;
        iface->verbs_common.inl_sge[1].addr   = (uintptr_t)&dc_req->sender.ep;
        iface->verbs_common.inl_sge[1].length = sizeof(dc_req->sender.ep);
        uct_dc_verbs_iface_post_send_to_dci(iface, &wr, dc_ep->dci, ah,
                                            dc_req->dct_num,
                                            IBV_SEND_INLINE | IBV_SEND_SIGNALED);
        ibv_destroy_ah(ah);
    } else {
        ucs_assert(op == UCT_RC_EP_FC_FLAG_HARD_REQ);
        wr.exp_opcode                         = IBV_EXP_WR_SEND_WITH_IMM;
        iface->verbs_common.inl_sge[1].addr   = (uintptr_t)&sender;
        iface->verbs_common.inl_sge[1].length = sizeof(sender);

        /* Send out DCT number to the peer, so it will be able
         * to send grants back */
        wr.ex.imm_data                        = iface->super.rx.dct->dct_num;

        dc_verbs_ep = ucs_derived_of(dc_ep, uct_dc_verbs_ep_t);
        uct_dc_verbs_iface_post_send(iface, dc_verbs_ep, &wr,
                                     IBV_SEND_INLINE | IBV_SEND_SIGNALED |
                                     IBV_SEND_SOLICITED) ;
        UCS_STATS_UPDATE_COUNTER(dc_ep->fc.stats,
                                 UCT_RC_FC_STAT_TX_HARD_REQ, 1);
    }
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE unsigned
uct_dc_verbs_poll_tx(uct_dc_verbs_iface_t *iface)
{
    int i;
    ucs_status_t status;
    unsigned num_wcs = iface->super.super.super.config.rx_max_poll;
    struct ibv_wc wc[num_wcs];
    int count;
    uint8_t dci;

    UCT_RC_VERBS_IFACE_FOREACH_TXWQE(&iface->super.super, i, wc, num_wcs) {
        if (ucs_unlikely(wc[i].status != IBV_WC_SUCCESS)) {
            status = uct_rc_verbs_wc_to_ucs_status(wc[i].status);
            iface->super.super.super.ops->handle_failure(&iface->super.super.super,
                                                         &wc[i], status);
            continue;
        }
        dci = uct_dc_iface_dci_find(&iface->super, wc[i].qp_num);
        count = uct_rc_verbs_txcq_get_comp_count(&wc[i],
                                                 &iface->super.tx.dcis[dci].txqp);
        ucs_assert(count == 1);
        ucs_trace_poll("dc_verbs iface %p tx_wc: dci[%d] qpn 0x%x count %d",
                       iface, dci, wc[i].qp_num, count);

        uct_rc_verbs_txqp_completed(&iface->super.tx.dcis[dci].txqp, &iface->dcis_txcnt[dci], count);
        uct_dc_iface_dci_put(&iface->super, dci);
        uct_rc_txqp_completion_desc(&iface->super.tx.dcis[dci].txqp, iface->dcis_txcnt[dci].ci);
        iface->super.super.tx.cq_available++;
    }

    uct_dc_iface_progress_pending(&iface->super);
    return num_wcs;
}

static unsigned uct_dc_verbs_iface_progress(void *arg)
{
    uct_dc_verbs_iface_t *iface = arg;
    unsigned count;

    count = uct_rc_verbs_iface_poll_rx_common(&iface->super.super);
    if (count > 0) {
        return count;
    }
    return uct_dc_verbs_poll_tx(iface);
}

#if IBV_EXP_HW_TM_DC
ucs_status_t uct_dc_verbs_ep_tag_eager_short(uct_ep_h tl_ep, uct_tag_t tag,
                                             const void *data, size_t length)
{
    uct_dc_verbs_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_dc_verbs_iface_t);
    struct ibv_exp_tmh tmh;

    UCT_CHECK_LENGTH(length + sizeof(struct ibv_exp_tmh), 0,
                     iface->verbs_common.config.max_inline, "tag_short");
    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    uct_rc_iface_fill_tmh(&tmh, tag, 0, IBV_EXP_TMH_EAGER);
    uct_rc_verbs_iface_fill_inl_sge(&iface->verbs_common, &tmh, sizeof(tmh),
                                    data, length);
    uct_dc_verbs_iface_post_send(iface, ep, &iface->inl_am_wr,
                                 IBV_SEND_INLINE | IBV_SEND_SOLICITED);

    return UCS_OK;
}

ssize_t uct_dc_verbs_ep_tag_eager_bcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                        uint64_t imm,
                                        uct_pack_callback_t pack_cb,
                                        void *arg, unsigned flags)
{
    uct_dc_verbs_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_dc_verbs_iface_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge;
    size_t length;
    uint32_t app_ctx;

    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    UCT_RC_IFACE_FILL_TM_IMM(imm, app_ctx, wr.ex.imm_data, wr.exp_opcode,
                             IBV_EXP_WR_SEND, _WITH_IMM);
    UCT_RC_IFACE_GET_TM_BCOPY_DESC(&iface->super.super,
                                   &iface->super.super.tx.mp, desc,
                                   tag, app_ctx, pack_cb, arg, length);
    UCT_RC_VERBS_FILL_SGE(wr, sge, length + sizeof(struct ibv_exp_tmh));
    uct_dc_verbs_iface_post_send_desc(iface, ep, &wr, desc, IBV_SEND_SOLICITED);

    return length;
}

ucs_status_t uct_dc_verbs_ep_tag_eager_zcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                             uint64_t imm, const uct_iov_t *iov,
                                             size_t iovcnt, unsigned flags,
                                             uct_completion_t *comp)
{
    uct_dc_verbs_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_dc_verbs_iface_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_sge sge[UCT_IB_MAX_IOV];
    struct ibv_exp_send_wr wr;
    int send_flags;
    size_t sge_cnt;
    uint32_t app_ctx;

    UCT_CHECK_IOV_SIZE(iovcnt, 1ul, "uct_dc_verbs_ep_tag_eager_zcopy");
    UCT_RC_CHECK_ZCOPY_DATA(sizeof(struct ibv_exp_tmh),
                            uct_iov_total_length(iov, iovcnt),
                            iface->super.super.super.config.seg_size);
    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    sge_cnt = uct_ib_verbs_sge_fill_iov(sge + 1, iov, iovcnt);

    UCT_RC_IFACE_FILL_TM_IMM(imm, app_ctx, wr.ex.imm_data, wr.exp_opcode,
                             IBV_EXP_WR_SEND, _WITH_IMM);
    UCT_RC_VERBS_GET_TM_ZCOPY_DESC(&iface->super.super,
                                   &iface->verbs_common.short_desc_mp,
                                   desc, tag, app_ctx, comp, &send_flags,
                                   sge[0]);
    wr.num_sge = sge_cnt + 1;
    wr.sg_list = sge;
    uct_dc_verbs_iface_post_send_desc(iface, ep, &wr, desc,
                                      send_flags | IBV_SEND_SOLICITED);

    return UCS_INPROGRESS;
}

ucs_status_ptr_t uct_dc_verbs_ep_tag_rndv_zcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                                const void *header,
                                                unsigned header_length,
                                                const uct_iov_t *iov,
                                                size_t iovcnt, unsigned flags,
                                                uct_completion_t *comp)
{
    uct_dc_verbs_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_dc_verbs_iface_t);
    unsigned tmh_len            = sizeof(struct ibv_exp_tmh) +
                                  sizeof(struct ibv_exp_tmh_rvh) +
                                  sizeof(struct ibv_exp_tmh_ravh);
    void *tmh                   = ucs_alloca(tmh_len);
    void *rvh                   = tmh + sizeof(struct ibv_exp_tmh);
    unsigned rndv_idx;

    UCT_RC_IFACE_CHECK_RNDV_PARAMS(iovcnt, header_length, tmh_len,
                                   iface->verbs_common.config.max_inline,
                                   iface->super.super.tm.max_rndv_data +
                                   UCT_RC_IFACE_TMH_PRIV_LEN);
    UCT_DC_CHECK_RES_PTR(&iface->super, &ep->super);

    rndv_idx = uct_rc_verbs_iface_fill_rndv_hdrs(&iface->super.super,
                                                 &iface->verbs_common,
                                                 tmh, header, header_length,
                                                 iface->super.super.tm.max_rndv_data,
                                                 tmh_len, tag, iov, comp);

    uct_dc_iface_fill_ravh(rvh + sizeof(struct ibv_exp_tmh_rvh),
                           iface->super.rx.dct->dct_num);

    uct_dc_verbs_iface_post_send(iface, ep, &iface->inl_am_wr,
                                 IBV_SEND_INLINE | IBV_SEND_SOLICITED);
    return (ucs_status_ptr_t)((uint64_t)rndv_idx);
}

/* For RNDV request send regular eager packet with IBV_SEND_WITH_IMM and
 * imm_value = 0. Receiver will handle such message as rndv request. */
ucs_status_t uct_dc_verbs_ep_tag_rndv_request(uct_ep_h tl_ep, uct_tag_t tag,
                                              const void* header,
                                              unsigned header_length,
                                              unsigned flags)
{
    uct_dc_verbs_ep_t *ep       = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_dc_verbs_iface_t);
    struct ibv_exp_tmh tmh;
    uint32_t app_ctx;
    struct ibv_exp_send_wr wr;

    UCT_CHECK_LENGTH(header_length + sizeof(tmh), 0,
                     iface->verbs_common.config.max_inline, "tag_short");
    UCT_DC_CHECK_RES(&iface->super, &ep->super);

    wr.sg_list    = iface->verbs_common.inl_sge;
    wr.num_sge    = 2;
    wr.exp_opcode = IBV_EXP_WR_SEND_WITH_IMM;
    wr.next       = NULL;

    uct_rc_iface_tag_imm_data_pack(&(wr.ex.imm_data), &app_ctx, 0ul);
    uct_rc_iface_fill_tmh(&tmh, tag, app_ctx, IBV_EXP_TMH_EAGER);
    uct_rc_verbs_iface_fill_inl_sge(&iface->verbs_common, &tmh, sizeof(tmh),
                                    header, header_length);
    uct_dc_verbs_iface_post_send(iface, ep, &wr,
                                 IBV_SEND_INLINE | IBV_SEND_SOLICITED);

    return UCS_OK;
}

static ucs_status_t uct_dc_verbs_iface_tag_recv_zcopy(uct_iface_h tl_iface,
                                                      uct_tag_t tag,
                                                      uct_tag_t tag_mask,
                                                      const uct_iov_t *iov,
                                                      size_t iovcnt,
                                                      uct_tag_context_t *ctx)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_iface,
                                                 uct_dc_verbs_iface_t);
    return uct_rc_verbs_iface_common_tag_recv(&iface->verbs_common,
                                              &iface->super.super, tag,
                                              tag_mask, iov, iovcnt, ctx);
}

static ucs_status_t uct_dc_verbs_iface_tag_recv_cancel(uct_iface_h tl_iface,
                                                       uct_tag_context_t *ctx,
                                                       int force)
{
   uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_iface,
                                                uct_dc_verbs_iface_t);

   return uct_rc_verbs_iface_common_tag_recv_cancel(&iface->verbs_common,
                                                    &iface->super.super,
                                                    ctx, force);
}

static unsigned uct_dc_verbs_iface_progress_tm(void *arg)
{
    uct_dc_verbs_iface_t *iface = arg;
    unsigned count;

    count = uct_rc_verbs_iface_poll_rx_tm(&iface->verbs_common,
                                          &iface->super.super);
    if (count > 0) {
        return count;
    }

    return uct_dc_verbs_poll_tx(iface);
}
#endif /* IBV_EXP_HW_TM_DC */

static ucs_status_t
uct_dc_verbs_iface_tag_init(uct_dc_verbs_iface_t *iface,
                            uct_dc_verbs_iface_config_t *config)
{
#if IBV_EXP_HW_TM_DC

    if (UCT_RC_IFACE_TM_ENABLED(&iface->super.super)) {
        struct ibv_exp_create_srq_attr srq_init_attr = {};
        struct ibv_exp_srq_dc_offload_params dc_op   = {};
        ucs_status_t status;

        uct_dc_iface_fill_xrq_init_attrs(&iface->super.super, &srq_init_attr,
                                         &dc_op);

        status = uct_rc_verbs_iface_common_tag_init(&iface->verbs_common,
                                                    &iface->super.super,
                                                    &config->verbs_common,
                                                    &config->super.super,
                                                    &srq_init_attr,
                                                    sizeof(struct ibv_exp_tmh_rvh) +
                                                    sizeof(struct ibv_exp_tmh_ravh));
        if (status != UCS_OK) {
            return status;
        }

        /* TM XRQ is ready, can create DCT now */
        status = uct_dc_iface_create_dct(&iface->super);
        if (status != UCS_OK) {
            uct_rc_iface_tag_cleanup(&iface->super.super);
            return status;
        }

        iface->super.super.progress = uct_dc_verbs_iface_progress_tm;
    } else
#endif
    {
        iface->super.super.progress = uct_dc_verbs_iface_progress;
    }

    return UCS_OK;
}

static void uct_dc_verbs_iface_tag_cleanup(uct_dc_verbs_iface_t *iface)
{
    if (UCT_RC_IFACE_TM_ENABLED(&iface->super.super)) {
        ibv_exp_destroy_dct(iface->super.rx.dct);
        iface->super.rx.dct = NULL;
    }

    uct_rc_iface_tag_cleanup(&iface->super.super);
}

static ucs_status_t
uct_dc_verbs_iface_event_arm(uct_iface_h tl_iface, unsigned events)
{
    uct_rc_iface_t UCS_V_UNUSED *iface = ucs_derived_of(tl_iface,
                                                        uct_rc_iface_t);

    return uct_rc_iface_common_event_arm(tl_iface, events,
                                         UCT_RC_IFACE_TM_ENABLED(iface));
}

static void UCS_CLASS_DELETE_FUNC_NAME(uct_dc_verbs_iface_t)(uct_iface_t*);

static uct_dc_iface_ops_t uct_dc_verbs_iface_ops = {
    {
    {
    {
    .ep_put_short             = uct_dc_verbs_ep_put_short,
    .ep_put_bcopy             = uct_dc_verbs_ep_put_bcopy,
    .ep_put_zcopy             = uct_dc_verbs_ep_put_zcopy,
    .ep_get_bcopy             = uct_dc_verbs_ep_get_bcopy,
    .ep_get_zcopy             = uct_dc_verbs_ep_get_zcopy,
    .ep_am_short              = uct_dc_verbs_ep_am_short,
    .ep_am_bcopy              = uct_dc_verbs_ep_am_bcopy,
    .ep_am_zcopy              = uct_dc_verbs_ep_am_zcopy,
    .ep_atomic_add64          = uct_dc_verbs_ep_atomic_add64,
    .ep_atomic_fadd64         = uct_dc_verbs_ep_atomic_fadd64,
    .ep_atomic_swap64         = uct_dc_verbs_ep_atomic_swap64,
    .ep_atomic_cswap64        = uct_dc_verbs_ep_atomic_cswap64,
    .ep_atomic_add32          = uct_dc_verbs_ep_atomic_add32,
    .ep_atomic_fadd32         = uct_dc_verbs_ep_atomic_fadd32,
    .ep_atomic_swap32         = uct_dc_verbs_ep_atomic_swap32,
    .ep_atomic_cswap32        = uct_dc_verbs_ep_atomic_cswap32,
    .ep_pending_add           = uct_dc_ep_pending_add,
    .ep_pending_purge         = uct_dc_ep_pending_purge,
    .ep_flush                 = uct_dc_verbs_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create_connected      = UCS_CLASS_NEW_FUNC_NAME(uct_dc_verbs_ep_t),
    .ep_destroy               = uct_dc_verbs_ep_destroy,
#if IBV_EXP_HW_TM_DC
    .ep_tag_eager_short       = uct_dc_verbs_ep_tag_eager_short,
    .ep_tag_eager_bcopy       = uct_dc_verbs_ep_tag_eager_bcopy,
    .ep_tag_eager_zcopy       = uct_dc_verbs_ep_tag_eager_zcopy,
    .ep_tag_rndv_zcopy        = uct_dc_verbs_ep_tag_rndv_zcopy,
    .ep_tag_rndv_cancel       = uct_rc_ep_tag_rndv_cancel,
    .ep_tag_rndv_request      = uct_dc_verbs_ep_tag_rndv_request,
    .iface_tag_recv_zcopy     = uct_dc_verbs_iface_tag_recv_zcopy,
    .iface_tag_recv_cancel    = uct_dc_verbs_iface_tag_recv_cancel,
#endif
    .iface_flush              = uct_dc_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_rc_verbs_iface_common_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_rc_iface_do_progress,
    .iface_event_fd_get       = uct_ib_iface_event_fd_get,
    .iface_event_arm          = uct_dc_verbs_iface_event_arm,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_dc_verbs_iface_t),
    .iface_query              = uct_dc_verbs_iface_query,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_dc_iface_is_reachable,
    .iface_get_address        = uct_dc_iface_get_address
    },
    .arm_tx_cq                = uct_ib_iface_arm_tx_cq,
    .arm_rx_cq                = uct_ib_iface_arm_rx_cq,
    .handle_failure           = uct_dc_verbs_handle_failure,
    .set_ep_failed            = uct_dc_verbs_ep_set_failed
    },
    .fc_ctrl                  = uct_dc_verbs_ep_fc_ctrl,
    .fc_handler               = uct_dc_iface_fc_handler,
    },
    .reset_dci                = uct_dc_verbs_reset_dci
};

void uct_dc_verbs_iface_init_wrs(uct_dc_verbs_iface_t *self)
{

    /* Initialize inline work request */
    memset(&self->inl_am_wr, 0, sizeof(self->inl_am_wr));
    self->inl_am_wr.sg_list                 = self->verbs_common.inl_sge;
    self->inl_am_wr.num_sge                 = 2;
    self->inl_am_wr.exp_opcode              = IBV_EXP_WR_SEND;
    self->inl_am_wr.exp_send_flags          = IBV_SEND_INLINE;
    self->inl_am_wr.dc.dct_access_key       = UCT_IB_KEY;

    memset(&self->inl_rwrite_wr, 0, sizeof(self->inl_rwrite_wr));
    self->inl_rwrite_wr.sg_list             = self->verbs_common.inl_sge;
    self->inl_rwrite_wr.num_sge             = 1;
    self->inl_rwrite_wr.exp_opcode          = IBV_EXP_WR_RDMA_WRITE;
    self->inl_rwrite_wr.exp_send_flags      = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    self->inl_rwrite_wr.dc.dct_access_key   = UCT_IB_KEY;
}

static UCS_CLASS_INIT_FUNC(uct_dc_verbs_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_dc_verbs_iface_config_t *config = ucs_derived_of(tl_config,
                                                         uct_dc_verbs_iface_config_t);
    struct ibv_qp_init_attr dci_init_attr;
    struct ibv_qp_attr dci_attr;
    ucs_status_t status;
    int i, ret;
    unsigned rx_qlen_init;

    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(uct_dc_iface_t, &uct_dc_verbs_iface_ops, md,
                              worker, params, 0, &config->super,
                              IBV_EXP_TM_CAP_DC);

    uct_dc_verbs_iface_init_wrs(self);

    status = uct_dc_verbs_iface_tag_init(self, config);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_rc_verbs_iface_common_init(&self->verbs_common,
                                            &self->super.super,
                                            &config->verbs_common,
                                            &config->super.super);
    if (status != UCS_OK) {
        goto err_tag_cleanup;
    }

    ret = ibv_query_qp(self->super.tx.dcis[0].txqp.qp, &dci_attr, 0,
                       &dci_init_attr);
    if (ret) {
        ucs_error("ibv_query_qp() failed: %m");
        goto err_common_cleanup;
    }
    self->verbs_common.config.max_inline = dci_init_attr.cap.max_inline_data;

    for (i = 0; i < self->super.tx.ndci; i++) {
        uct_rc_verbs_txcnt_init(&self->dcis_txcnt[i]);
        uct_rc_txqp_available_set(&self->super.tx.dcis[i].txqp,
                                  self->super.super.config.tx_qp_len);
    }
    uct_dc_iface_set_quota(&self->super, &config->super);

    rx_qlen_init = config->super.ud_common.rx_queue_len_init;
    ucs_mpool_grow(&self->super.super.rx.mp,
                   ucs_min(rx_qlen_init, self->super.super.rx.srq.quota));

    status = uct_rc_verbs_iface_common_prepost_recvs(&self->super.super,
                                                     rx_qlen_init);
    if (status != UCS_OK) {
        goto err_common_cleanup;
    }

    ucs_debug("created dc iface %p", self);
    return UCS_OK;

err_common_cleanup:
    uct_rc_verbs_iface_common_cleanup(&self->verbs_common);
err_tag_cleanup:
    uct_dc_verbs_iface_tag_cleanup(self);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_verbs_iface_t)
{
    ucs_trace_func("");
    uct_base_iface_progress_disable(&self->super.super.super.super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    uct_rc_verbs_iface_common_cleanup(&self->verbs_common);
    uct_dc_verbs_iface_tag_cleanup(self);
}

UCS_CLASS_DEFINE(uct_dc_verbs_iface_t, uct_dc_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_dc_verbs_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_dc_verbs_iface_t, uct_iface_t);

static
ucs_status_t uct_dc_verbs_query_resources(uct_md_h md,
                                          uct_tl_resource_desc_t **resources_p,
                                          unsigned *num_resources_p)
{
    uct_ib_md_t *ib_md = ucs_derived_of(md, uct_ib_md_t);

    return uct_dc_device_query_tl_resources(&ib_md->dev, "dc",
                                            (ib_md->config.eth_pause ? 0 : UCT_IB_DEVICE_FLAG_LINK_IB),
                                            resources_p, num_resources_p);
}


UCT_TL_COMPONENT_DEFINE(uct_dc_verbs_tl,
                        uct_dc_verbs_query_resources,
                        uct_dc_verbs_iface_t,
                        "dc",
                        "DC_VERBS_",
                        uct_dc_verbs_iface_config_table,
                        uct_dc_verbs_iface_config_t);
UCT_MD_REGISTER_TL(&uct_ib_mdc, &uct_dc_verbs_tl);
