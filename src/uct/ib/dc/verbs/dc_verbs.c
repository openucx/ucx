/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "dc_verbs.h"

#include <uct/ib/rc/verbs/rc_verbs_common.h>
#include <uct/api/uct.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_log.h>
#include <uct/base/uct_md.h>
#include <ucs/arch/bitops.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>
#include <string.h>


static UCS_CLASS_INIT_FUNC(uct_dc_verbs_ep_t,
                           uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_verbs_iface_t);
    const uct_ib_address_t *ib_addr = (const uct_ib_address_t *)dev_addr;
    const uct_dc_iface_addr_t *if_addr = (const uct_dc_iface_addr_t *)iface_addr;
    ucs_status_t status;
    int is_global;

    ucs_trace_func("");
    UCS_CLASS_CALL_SUPER_INIT(uct_dc_ep_t, &iface->super, if_addr);

    status = uct_ib_iface_create_ah(&iface->super.super.super, ib_addr,
                                    iface->super.super.super.path_bits[0], &self->ah,
                                    &is_global);
    if (status != UCS_OK) {
        return UCS_ERR_INVALID_ADDR;
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
UCS_CLASS_DEFINE_DELETE_FUNC(uct_dc_verbs_ep_t, uct_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_dc_verbs_ep_t, uct_ep_t, uct_iface_h, const uct_device_addr_t *,
                          const uct_iface_addr_t *);


static ucs_status_t uct_dc_verbs_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_verbs_iface_t);

    uct_dc_iface_query(&iface->super, iface_attr);
    uct_rc_verbs_iface_common_query(&iface->verbs_common,
                                    &iface->super.super, iface_attr);
    /*TODO: remove flags once we have a full functionality */
    iface_attr->cap.flags           = UCT_IFACE_FLAG_AM_ZCOPY|
                                      UCT_IFACE_FLAG_AM_BCOPY|
                                      UCT_IFACE_FLAG_AM_SHORT|
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

    /*TODO: remove max_iov initialization once we have a full functionality */
    iface_attr->cap.put.max_iov     = 1;
    iface_attr->cap.get.max_iov     = 1;
    iface_attr->cap.am.max_iov      = 1;

    return UCS_OK;
}


static UCS_F_ALWAYS_INLINE void
uct_dc_verbs_iface_post_send(uct_dc_verbs_iface_t* iface, uct_dc_verbs_ep_t *ep,
                             struct ibv_exp_send_wr *wr, uint64_t send_flags)
{
    struct ibv_exp_send_wr *bad_wr;
    int ret;
    uint8_t dci;
    uct_rc_txqp_t *txqp;

    dci  = ep->super.dci;
    txqp = &iface->super.tx.dcis[dci].txqp;
    /* TODO: check tx moderation */
    send_flags |= IBV_SEND_SIGNALED;

    wr->exp_send_flags    = send_flags;
    wr->wr_id             = txqp->unsignaled;
    wr->dc.ah             = ep->ah;
    wr->dc.dct_number     = ep->dest_qpn;
    wr->dc.dct_access_key = UCT_IB_DC_KEY;

    uct_ib_log_exp_post_send(&iface->super.super.super, txqp->qp, wr,
                             (wr->exp_opcode == (int)IBV_WR_SEND) ? uct_rc_ep_am_packet_dump : NULL);

    UCT_IB_INSTRUMENT_RECORD_SEND_EXP_WR_LEN("uct_dc_verbs_ep_post_send", wr);

    ret = ibv_exp_post_send(txqp->qp, wr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_send() returned %d (%m)", ret);
    }

    uct_rc_verbs_txqp_posted(txqp, &iface->dcis_txcnt[dci],
                             &iface->super.super, send_flags & IBV_SEND_SIGNALED);
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
uct_dc_verbs_ep_rdma_zcopy(uct_dc_verbs_ep_t *ep, const void *buffer, size_t length,
                           uct_ib_mem_t *memh, uint64_t remote_addr,
                           uct_rkey_t rkey, uct_completion_t *comp,
                           int opcode)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_dc_verbs_iface_t);
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge;

    UCT_SKIP_ZERO_LENGTH(length);

    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_VERBS_FILL_RDMA_WR(wr, wr.exp_opcode, opcode, sge, length, remote_addr, rkey);
    wr.next = NULL;
    uct_rc_verbs_rdma_zcopy_sge_fill(&sge, buffer, length, memh);

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

    UCT_CHECK_LENGTH(length, iface->super.super.super.config.seg_size, "get_bcopy");
    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_GET_BCOPY_DESC(&iface->super.super, &iface->super.super.tx.mp,
                                       desc, unpack_cb, comp, arg, length);
    UCT_RC_VERBS_FILL_RDMA_WR(wr, wr.exp_opcode, IBV_WR_RDMA_READ, sge, length, remote_addr,
                              rkey);
    UCT_TL_EP_STAT_OP(&ep->super.super, GET, BCOPY, length);
    uct_dc_verbs_iface_post_send_desc(iface, ep, &wr, desc, IBV_SEND_SIGNALED);

    return UCS_INPROGRESS;
}

ucs_status_t uct_dc_verbs_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    ucs_status_t status;

    UCT_CHECK_PARAM_IOV(iov, iovcnt, buffer, length, memh);

    UCT_CHECK_LENGTH(length, UCT_IB_MAX_MESSAGE_SIZE, "get_zcopy");
    status = uct_dc_verbs_ep_rdma_zcopy(ep, buffer, length, memh, remote_addr,
                                        rkey, comp, IBV_WR_RDMA_READ);
    if (status == UCS_INPROGRESS) {
        UCT_TL_EP_STAT_OP(&ep->super.super, GET, ZCOPY, length);
    }
    return status;
}

ucs_status_t uct_dc_verbs_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                       unsigned length, uint64_t remote_addr,
                                       uct_rkey_t rkey)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);

    UCT_CHECK_LENGTH(length, iface->verbs_common.config.max_inline, "put_short");

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
    UCT_RC_VERBS_FILL_RDMA_WR(wr, wr.exp_opcode, IBV_WR_RDMA_WRITE, sge,
                              length, remote_addr, rkey);
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, BCOPY, length);
    uct_dc_verbs_iface_post_send_desc(iface, ep, &wr, desc, IBV_SEND_SIGNALED);

    return length;
}

ucs_status_t uct_dc_verbs_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    ucs_status_t status;

    UCT_CHECK_PARAM_IOV(iov, iovcnt, buffer, length, memh);

    UCT_CHECK_LENGTH(length, UCT_IB_MAX_MESSAGE_SIZE, "put_zcopy");
    status = uct_dc_verbs_ep_rdma_zcopy(ep, buffer, length, memh, remote_addr,
                                        rkey, comp, IBV_WR_RDMA_WRITE);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, PUT, ZCOPY, length);
    return status;
}

ucs_status_t uct_dc_verbs_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                      const void *buffer, unsigned length)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_rc_am_short_hdr_t am;

    UCT_RC_CHECK_AM_SHORT(id, length, iface->verbs_common.config.max_inline);

    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    uct_rc_verbs_iface_fill_inl_am_sge(&iface->verbs_common, &am, id, hdr, buffer, length);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, sizeof(hdr) + length);
    uct_dc_verbs_iface_post_send(iface, ep, &iface->inl_am_wr, IBV_SEND_INLINE);

    return UCS_OK;
}

ssize_t uct_dc_verbs_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                 uct_pack_callback_t pack_cb, void *arg)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge;
    size_t length;

    UCT_CHECK_AM_ID(id);

    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_AM_BCOPY_DESC(&iface->super.super, &iface->super.super.tx.mp, desc,
                                      id, pack_cb, arg, &length);
    UCT_RC_VERBS_FILL_AM_BCOPY_WR(wr, sge, length, wr.exp_opcode);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, BCOPY, length);
    uct_dc_verbs_iface_post_send_desc(iface, ep, &wr, desc, 0);

    return length;
}


ucs_status_t uct_dc_verbs_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                      unsigned header_length, const uct_iov_t *iov,
                                      size_t iovcnt, uct_completion_t *comp)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge[2];
    int send_flags;

    UCT_CHECK_PARAM_IOV(iov, iovcnt, buffer, length, memh);

    UCT_RC_CHECK_AM_ZCOPY(id, header_length, length,
                          iface->verbs_common.config.short_desc_size,
                          iface->super.super.super.config.seg_size);

    UCT_DC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_AM_ZCOPY_DESC(&iface->super.super, &iface->verbs_common.short_desc_mp, desc,
                                      id, header, header_length, comp, &send_flags);
    uct_rc_verbs_am_zcopy_sge_fill(sge, header_length, buffer, length, memh);
    UCT_RC_VERBS_FILL_AM_ZCOPY_WR(wr, sge, wr.exp_opcode);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, ZCOPY, header_length + length);
    uct_dc_verbs_iface_post_send_desc(iface, ep, &wr, desc, send_flags);

    return UCS_INPROGRESS;
}


static UCS_F_ALWAYS_INLINE void
uct_dc_verbs_iface_atomic_post(uct_dc_verbs_iface_t *iface, uct_dc_verbs_ep_t *ep,
                               int opcode, uint64_t compare_add,
                               uint64_t swap, uint64_t remote_addr, uct_rkey_t rkey,
                               uct_rc_iface_send_desc_t *desc, int force_sig)
{
    struct ibv_sge sge;
    struct ibv_exp_send_wr wr;

    UCT_RC_VERBS_FILL_ATOMIC_WR(wr, wr.exp_opcode, sge, opcode,
                                compare_add, swap, remote_addr, rkey, ep->super.umr_offset);
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
                                    compare_add, swap, remote_addr, rkey, ep->super.umr_offset);
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
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    ucs_status_t status;

    status = uct_dc_ep_flush(tl_ep, flags, comp);
    if (status == UCS_OK) {
        return UCS_OK; /* all sends completed */
    }

    if (status == UCS_INPROGRESS) {
        uct_dc_verbs_iface_add_send_comp(iface, ep, comp);
    }
    return status;
}

static UCS_F_ALWAYS_INLINE void
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
            ucs_fatal("iface=%p: send completion %d with error: %s wqe: %p wr_id: %llu",
                      iface, i, ibv_wc_status_str(wc[i].status),
                      &wc[i], (unsigned long long)wc[i].wr_id);
        }
        count = uct_rc_verbs_txcq_get_comp_count(&wc[i]);
        ucs_assert(count == 1);
        dci = uct_dc_iface_dci_find(&iface->super, wc[i].qp_num);
        uct_rc_verbs_txqp_completed(&iface->super.tx.dcis[dci].txqp, &iface->dcis_txcnt[dci], count);
        ucs_trace_poll("dc tx completion on dc %d count %d", dci, count);
        uct_rc_txqp_completion(&iface->super.tx.dcis[dci].txqp, iface->dcis_txcnt[dci].ci);
        uct_dc_iface_dci_put(&iface->super, dci);
    }

    iface->super.super.tx.cq_available += num_wcs;
    if (uct_dc_iface_dci_can_alloc(&iface->super)) {
        ucs_arbiter_dispatch(&iface->super.super.tx.arbiter, 1, uct_dc_iface_dci_do_pending_wait, NULL);
    }
    ucs_arbiter_dispatch(&iface->super.tx.dci_arbiter, 1, uct_dc_iface_dci_do_pending_tx, NULL);
}

/* TODO: make a macro that defines progress func */
static void uct_dc_verbs_iface_progress(void *arg)
{
    uct_dc_verbs_iface_t *iface = arg;
    ucs_status_t status;

    status = uct_rc_verbs_iface_poll_rx_common(&iface->super.super);
    if (status == UCS_ERR_NO_PROGRESS) {
        uct_dc_verbs_poll_tx(iface);
    }
}

static void UCS_CLASS_DELETE_FUNC_NAME(uct_dc_verbs_iface_t)(uct_iface_t*);

static uct_rc_iface_ops_t uct_dc_verbs_iface_ops = {
    {
        {
            .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_dc_verbs_iface_t),
            .iface_query              = uct_dc_verbs_iface_query,
            .iface_get_device_address = uct_ib_iface_get_device_address,
            .iface_is_reachable       = uct_ib_iface_is_reachable,
            .iface_release_am_desc    = uct_ib_iface_release_am_desc,
            .iface_get_address        = uct_dc_iface_get_address,

            .iface_flush              = uct_dc_iface_flush,

            .ep_create_connected      = UCS_CLASS_NEW_FUNC_NAME(uct_dc_verbs_ep_t),
            .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_dc_verbs_ep_t),

            .ep_am_short              = uct_dc_verbs_ep_am_short,
            .ep_am_bcopy              = uct_dc_verbs_ep_am_bcopy,
            .ep_am_zcopy              = uct_dc_verbs_ep_am_zcopy,

            .ep_put_short             = uct_dc_verbs_ep_put_short,
            .ep_put_bcopy             = uct_dc_verbs_ep_put_bcopy,
            .ep_put_zcopy             = uct_dc_verbs_ep_put_zcopy,

            .ep_get_bcopy             = uct_dc_verbs_ep_get_bcopy,
            .ep_get_zcopy             = uct_dc_verbs_ep_get_zcopy,

            .ep_atomic_add64          = uct_dc_verbs_ep_atomic_add64,
            .ep_atomic_fadd64         = uct_dc_verbs_ep_atomic_fadd64,
            .ep_atomic_swap64         = uct_dc_verbs_ep_atomic_swap64,
            .ep_atomic_cswap64        = uct_dc_verbs_ep_atomic_cswap64,

            .ep_atomic_add32          = uct_dc_verbs_ep_atomic_add32,
            .ep_atomic_fadd32         = uct_dc_verbs_ep_atomic_fadd32,
            .ep_atomic_swap32         = uct_dc_verbs_ep_atomic_swap32,
            .ep_atomic_cswap32        = uct_dc_verbs_ep_atomic_cswap32,

            .ep_flush                 = uct_dc_verbs_ep_flush,

            .ep_pending_add           = uct_dc_ep_pending_add,
            .ep_pending_purge         = uct_dc_ep_pending_purge,
        },
        .arm_tx_cq                = uct_ib_iface_arm_tx_cq,
        .arm_rx_cq                = uct_ib_iface_arm_rx_cq,
    },
    .fc_ctrl                  = NULL /* TODO: */
};

void uct_dc_verbs_iface_init_wrs(uct_dc_verbs_iface_t *self)
{

    /* Initialize inline work request */
    memset(&self->inl_am_wr, 0, sizeof(self->inl_am_wr));
    self->inl_am_wr.sg_list                 = self->verbs_common.inl_sge;
    self->inl_am_wr.num_sge                 = 2;
    self->inl_am_wr.exp_opcode              = IBV_WR_SEND;
    self->inl_am_wr.exp_send_flags          = IBV_SEND_INLINE;
    self->inl_am_wr.dc.dct_access_key       = UCT_IB_DC_KEY;

    memset(&self->inl_rwrite_wr, 0, sizeof(self->inl_rwrite_wr));
    self->inl_rwrite_wr.sg_list             = self->verbs_common.inl_sge;
    self->inl_rwrite_wr.num_sge             = 1;
    self->inl_rwrite_wr.exp_opcode          = IBV_WR_RDMA_WRITE;
    self->inl_rwrite_wr.exp_send_flags      = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    self->inl_rwrite_wr.dc.dct_access_key   = UCT_IB_DC_KEY;
}

static UCS_CLASS_INIT_FUNC(uct_dc_verbs_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_dc_iface_config_t *config = ucs_derived_of(tl_config,
                                                   uct_dc_iface_config_t);
    struct ibv_qp_init_attr dci_init_attr;
    struct ibv_qp_attr dci_attr;
    ucs_status_t status;
    int i, ret;

    ucs_trace_func("");
    UCS_CLASS_CALL_SUPER_INIT(uct_dc_iface_t, &uct_dc_verbs_iface_ops, md,
                              worker, params, 0, config);

    uct_dc_verbs_iface_init_wrs(self);

    status = uct_rc_verbs_iface_common_init(&self->verbs_common, &self->super.super,
                                            &config->super);
    if (status != UCS_OK) {
        goto err;
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

    status = uct_rc_verbs_iface_prepost_recvs_common(&self->super.super);
    if (status != UCS_OK) {
        goto err_common_cleanup;
    }

    /* TODO: only register progress when we have a connection */
    uct_worker_progress_register(worker, uct_dc_verbs_iface_progress, self);
    ucs_debug("created dc iface %p", self);
    return UCS_OK;

err_common_cleanup:
    uct_rc_verbs_iface_common_cleanup(&self->verbs_common);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_verbs_iface_t)
{
    ucs_trace_func("");
    uct_worker_progress_unregister(self->super.super.super.super.worker,
                                   uct_dc_verbs_iface_progress, self);
    uct_rc_verbs_iface_common_cleanup(&self->verbs_common);
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
                                            (ib_md->eth_pause ? 0 : UCT_IB_DEVICE_FLAG_LINK_IB),
                                            resources_p, num_resources_p);
}


UCT_TL_COMPONENT_DEFINE(uct_dc_verbs_tl,
                        uct_dc_verbs_query_resources,
                        uct_dc_verbs_iface_t,
                        "dc",
                        "DC_VERBS_",
                        uct_dc_iface_config_table,
                        uct_dc_iface_config_t);
UCT_MD_REGISTER_TL(&uct_ib_mdc, &uct_dc_verbs_tl);

