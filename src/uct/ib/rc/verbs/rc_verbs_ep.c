/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rc_verbs.h"

#include <ucs/arch/bitops.h>
#include <uct/ib/base/ib_log.h>

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_ep_post_send(uct_rc_verbs_iface_t* iface, uct_rc_verbs_ep_t* ep,
                          struct ibv_send_wr *wr, int send_flags)
{
    struct ibv_send_wr *bad_wr;
    int ret;

    uct_rc_txqp_check(&ep->super.txqp);

    if (!(send_flags & IBV_SEND_SIGNALED)) {
        send_flags |= uct_rc_iface_tx_moderation(&iface->super, &ep->super.txqp,
                                                 IBV_SEND_SIGNALED);
    }
    wr->send_flags = send_flags;
    wr->wr_id      = uct_rc_txqp_unsignaled(&ep->super.txqp);

    uct_ib_log_post_send(&iface->super.super, ep->super.txqp.qp, wr,
                         (wr->opcode == IBV_WR_SEND) ? uct_rc_verbs_ep_am_packet_dump : NULL);

    ret = ibv_post_send(ep->super.txqp.qp, wr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_send() returned %d (%m)", ret);
    }

    uct_rc_verbs_txqp_posted(&ep->super.txqp, &ep->txcnt, &iface->super, send_flags & IBV_SEND_SIGNALED);
}

#if HAVE_DECL_IBV_EXP_POST_SEND && (HAVE_DECL_IBV_EXP_WR_NOP || HAVE_IB_EXT_ATOMICS)
static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_exp_post_send(uct_rc_verbs_ep_t *ep, struct ibv_exp_send_wr *wr,
                           uint64_t signal)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_rc_verbs_iface_t);
    uct_rc_txqp_check(&ep->super.txqp);

    struct ibv_exp_send_wr *bad_wr;
    int ret;

    signal |= uct_rc_iface_tx_moderation(&iface->super, &ep->super.txqp,
                                         IBV_EXP_SEND_SIGNALED);
    wr->exp_send_flags = signal;
    wr->wr_id          = uct_rc_txqp_unsignaled(&ep->super.txqp);

    uct_ib_log_exp_post_send(&iface->super.super, ep->super.txqp.qp, wr,
                             (wr->exp_opcode == IBV_EXP_WR_SEND) ?
                             uct_rc_verbs_ep_am_packet_dump : NULL);

    ret = ibv_exp_post_send(ep->super.txqp.qp, wr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_exp_post_send() returned %d (%m)", ret);
    }

    uct_rc_verbs_txqp_posted(&ep->super.txqp, &ep->txcnt, &iface->super, signal);
}
#endif

/*
 * Helper function for posting sends with a descriptor.
 * User needs to fill: wr.opcode, wr.sg_list, wr.num_sge, first sge length, and
 * operation-specific fields (e.g rdma).
 */
static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_ep_post_send_desc(uct_rc_verbs_ep_t* ep, struct ibv_send_wr *wr,
                               uct_rc_iface_send_desc_t *desc, int send_flags)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_rc_verbs_iface_t);
    UCT_RC_VERBS_FILL_DESC_WR(wr, desc);
    uct_rc_verbs_ep_post_send(iface, ep, wr, send_flags);
    uct_rc_txqp_add_send_op_sn(&ep->super.txqp, &desc->super, ep->txcnt.pi);
}

static inline ucs_status_t
uct_rc_verbs_ep_rdma_zcopy(uct_rc_verbs_ep_t *ep, const uct_iov_t *iov,
                           size_t iovcnt, uint64_t remote_addr, uct_rkey_t rkey,
                           uct_completion_t *comp, int opcode)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_rc_verbs_iface_t);
    struct ibv_sge sge[UCT_IB_MAX_IOV];
    struct ibv_send_wr wr;
    size_t sge_cnt;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    sge_cnt = uct_ib_verbs_sge_fill_iov(sge, iov, iovcnt);
    UCT_SKIP_ZERO_LENGTH(sge_cnt);
    UCT_RC_VERBS_FILL_RDMA_WR_IOV(wr, wr.opcode, opcode, sge, sge_cnt, remote_addr, rkey);
    wr.next = NULL;

    uct_rc_verbs_ep_post_send(iface, ep, &wr, IBV_SEND_SIGNALED);
    uct_rc_txqp_add_send_comp(&iface->super, &ep->super.txqp, comp, ep->txcnt.pi);
    return UCS_INPROGRESS;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_ep_atomic_post(uct_rc_verbs_ep_t *ep, int opcode, uint64_t compare_add,
                            uint64_t swap, uint64_t remote_addr, uct_rkey_t rkey,
                            uct_rc_iface_send_desc_t *desc, int force_sig)
{
    uint32_t ib_rkey = uct_ib_resolve_atomic_rkey(rkey, ep->super.atomic_mr_offset,
                                                  &remote_addr);
    struct ibv_send_wr wr;
    struct ibv_sge sge;

    UCT_RC_VERBS_FILL_ATOMIC_WR(wr, wr.opcode, sge, opcode, compare_add, swap,
                                remote_addr, ib_rkey);
    UCT_TL_EP_STAT_ATOMIC(&ep->super.super);
    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, force_sig);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_verbs_ep_atomic(uct_rc_verbs_ep_t *ep, int opcode, void *result,
                       uint64_t compare_add, uint64_t swap, uint64_t remote_addr,
                       uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_rc_verbs_iface_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_ATOMIC_DESC(&iface->super, &iface->verbs_common.short_desc_mp, desc,
                                    iface->super.config.atomic64_handler,
                                    result, comp);
    uct_rc_verbs_ep_atomic_post(ep, opcode, compare_add, swap, remote_addr,
                                rkey, desc, IBV_SEND_SIGNALED);
    return UCS_INPROGRESS;
}

#if HAVE_IB_EXT_ATOMICS
static inline void
uct_rc_verbs_ep_ext_atomic_post(uct_rc_verbs_ep_t *ep, int opcode, uint32_t length,
                               uint64_t compare_mask, uint64_t compare_add,
                               uint64_t swap, uint64_t remote_addr, uct_rkey_t rkey,
                               uct_rc_iface_send_desc_t *desc, uint64_t force_sig)
{
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge;

    uct_rc_verbs_fill_ext_atomic_wr(&wr, &sge, opcode, length, compare_mask,
                                    compare_add, swap, remote_addr, rkey, ep->super.atomic_mr_offset);
    UCT_RC_VERBS_FILL_DESC_WR(&wr, desc);
    UCT_TL_EP_STAT_ATOMIC(&ep->super.super);
    uct_rc_verbs_exp_post_send(ep, &wr, force_sig|IBV_EXP_SEND_EXT_ATOMIC_INLINE);
    uct_rc_txqp_add_send_op_sn(&ep->super.txqp, &desc->super, ep->txcnt.pi);
}

static inline ucs_status_t
uct_rc_verbs_ep_ext_atomic(uct_rc_verbs_ep_t *ep, int opcode, void *result,
                           uint32_t length, uint64_t compare_mask,
                           uint64_t compare_add, uint64_t swap, uint64_t remote_addr,
                           uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_rc_verbs_iface_t);
    uct_rc_send_handler_t handler = uct_rc_iface_atomic_handler(&iface->super, 1,
                                                                length);
    uct_rc_iface_send_desc_t *desc;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_ATOMIC_DESC(&iface->super, &iface->verbs_common.short_desc_mp, desc,
                                    handler, result, comp);
    uct_rc_verbs_ep_ext_atomic_post(ep, opcode, length, compare_mask, compare_add,
                                    swap, remote_addr, rkey, desc,
                                    IBV_EXP_SEND_SIGNALED);
    return UCS_INPROGRESS;
}
#endif

ucs_status_t uct_rc_verbs_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                       unsigned length, uint64_t remote_addr,
                                       uct_rkey_t rkey)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);

    UCT_CHECK_LENGTH(length, 0, iface->verbs_common.config.max_inline, "put_short");

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_VERBS_FILL_INL_PUT_WR(iface, remote_addr, rkey, buffer, length);
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, SHORT, length);
    uct_rc_verbs_ep_post_send(iface, ep, &iface->inl_rwrite_wr,
                              IBV_SEND_INLINE | IBV_SEND_SIGNALED);
    return UCS_OK;
}

ssize_t uct_rc_verbs_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                  void *arg, uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_send_wr wr;
    struct ibv_sge sge;
    size_t length;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_PUT_BCOPY_DESC(&iface->super, &iface->super.tx.mp, desc,
                                       pack_cb, arg, length);
    UCT_RC_VERBS_FILL_RDMA_WR(wr, wr.opcode, IBV_WR_RDMA_WRITE, sge,
                              length, remote_addr, rkey);
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, BCOPY, length);
    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, IBV_SEND_SIGNALED);
    return length;
}

ucs_status_t uct_rc_verbs_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    uct_ib_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ib_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt, uct_ib_iface_get_max_iov(iface),
                       "uct_rc_verbs_ep_put_zcopy");
    status = uct_rc_verbs_ep_rdma_zcopy(ep, iov, iovcnt, remote_addr,
                                        rkey, comp, IBV_WR_RDMA_WRITE);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, PUT, ZCOPY,
                                 uct_iov_total_length(iov, iovcnt));
    return status;
}

ucs_status_t uct_rc_verbs_ep_get_bcopy(uct_ep_h tl_ep,
                                       uct_unpack_callback_t unpack_cb,
                                       void *arg, size_t length,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_send_wr wr;
    struct ibv_sge sge;

    UCT_CHECK_LENGTH(length, 0, iface->super.super.config.seg_size, "get_bcopy");
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_GET_BCOPY_DESC(&iface->super, &iface->super.tx.mp, desc,
                                       unpack_cb, comp, arg, length);

    UCT_RC_VERBS_FILL_RDMA_WR(wr, wr.opcode, IBV_WR_RDMA_READ, sge, length, remote_addr,
                              rkey);

    UCT_TL_EP_STAT_OP(&ep->super.super, GET, BCOPY, length);
    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, IBV_SEND_SIGNALED);
    return UCS_INPROGRESS;
}

ucs_status_t uct_rc_verbs_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    uct_ib_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ib_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt, uct_ib_iface_get_max_iov(iface),
                       "uct_rc_verbs_ep_get_zcopy");
    status = uct_rc_verbs_ep_rdma_zcopy(ep, iov, iovcnt, remote_addr,
                                        rkey, comp, IBV_WR_RDMA_READ);
    if (status == UCS_INPROGRESS) {
        UCT_TL_EP_STAT_OP(&ep->super.super, GET, ZCOPY,
                          uct_iov_total_length(iov, iovcnt));
    }
    return status;
}

ucs_status_t uct_rc_verbs_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                      const void *buffer, unsigned length)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_common_t *verbs_common = &iface->verbs_common;

    UCT_RC_CHECK_AM_SHORT(id, length + verbs_common->config.notag_hdr_size,
                          verbs_common->config.max_inline);
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_CHECK_FC(&iface->super, &ep->super, id);
    uct_rc_verbs_iface_fill_inl_am_sge(verbs_common, id, hdr, buffer, length);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, sizeof(hdr) + length);
    uct_rc_verbs_ep_post_send(iface, ep, &iface->inl_am_wr,
                              IBV_SEND_INLINE | IBV_SEND_SOLICITED);
    UCT_RC_UPDATE_FC(&iface->super, &ep->super, id);

    return UCS_OK;
}

ssize_t uct_rc_verbs_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                 uct_pack_callback_t pack_cb, void *arg,
                                 unsigned flags)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_send_wr wr;
    struct ibv_sge sge;
    size_t length;
    size_t data_length;

    UCT_CHECK_AM_ID(id);

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_CHECK_FC(&iface->super, &ep->super, id);
    UCT_RC_VERBS_GET_TX_AM_BCOPY_DESC(&iface->verbs_common, &iface->super,
                                      &iface->super.tx.mp, desc, id, pack_cb,
                                      arg, length, data_length);
    UCT_RC_VERBS_FILL_AM_BCOPY_WR(wr, sge, length, wr.opcode);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, BCOPY, data_length);
    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, IBV_SEND_SOLICITED);
    UCT_RC_UPDATE_FC(&iface->super, &ep->super, id);

    return data_length;
}

ucs_status_t uct_rc_verbs_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                      unsigned header_length, const uct_iov_t *iov,
                                      size_t iovcnt, unsigned flags,
                                      uct_completion_t *comp)
{
    uct_rc_verbs_iface_t     *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t        *ep    = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc  = NULL;
    uct_rc_verbs_iface_common_t *verbs_common = &iface->verbs_common;
    struct ibv_sge sge[UCT_IB_MAX_IOV]; /* First sge is reserved for the header */
    struct ibv_send_wr wr;
    int send_flags;
    size_t sge_cnt;

    UCT_CHECK_IOV_SIZE(iovcnt, uct_ib_iface_get_max_iov(&iface->super.super) - 1,
                       "uct_rc_verbs_ep_am_zcopy");
    UCT_RC_CHECK_AM_ZCOPY(id,
                          header_length + verbs_common->config.notag_hdr_size,
                          uct_iov_total_length(iov, iovcnt),
                          verbs_common->config.short_desc_size,
                          iface->super.super.config.seg_size);
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_CHECK_FC(&iface->super, &ep->super, id);

    UCT_RC_VERBS_GET_TX_AM_ZCOPY_DESC(verbs_common, &iface->super,
                                      &verbs_common->short_desc_mp, desc, id,
                                      header, header_length, comp, &send_flags,
                                      sge[0]);
    sge_cnt = uct_ib_verbs_sge_fill_iov(sge + 1, iov, iovcnt);
    UCT_RC_VERBS_FILL_AM_ZCOPY_WR_IOV(wr, sge, (sge_cnt + 1), wr.opcode);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, ZCOPY,
                      (header_length + uct_iov_total_length(iov, iovcnt)));

    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, send_flags | IBV_SEND_SOLICITED);
    UCT_RC_UPDATE_FC(&iface->super, &ep->super, id);

    return UCS_INPROGRESS;
}

ucs_status_t uct_rc_verbs_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;

    /* TODO don't allocate descriptor - have dummy buffer */
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_ATOMIC_ADD_DESC(&iface->super, &iface->verbs_common.short_desc_mp, desc);

    uct_rc_verbs_ep_atomic_post(ep,
                                IBV_WR_ATOMIC_FETCH_AND_ADD, add, 0,
                                remote_addr, rkey, desc,
                                IBV_SEND_SIGNALED);
    return UCS_OK;
}

ucs_status_t uct_rc_verbs_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint64_t *result, uct_completion_t *comp)
{
    return uct_rc_verbs_ep_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                  IBV_WR_ATOMIC_FETCH_AND_ADD, result, add, 0,
                                  remote_addr, rkey, comp);
}

ucs_status_t uct_rc_verbs_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint64_t *result, uct_completion_t *comp)
{
#if HAVE_IB_EXT_ATOMICS
    return uct_rc_verbs_ep_ext_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                      IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP,
                                      result, sizeof(uint64_t), 0, 0, swap, remote_addr,
                                      rkey, comp);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_verbs_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uint64_t *result, uct_completion_t *comp)
{
    return uct_rc_verbs_ep_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                  IBV_WR_ATOMIC_CMP_AND_SWP, result, compare, swap,
                                  remote_addr, rkey, comp);
}

ucs_status_t uct_rc_verbs_ep_atomic_add32(uct_ep_h tl_ep, uint32_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey)
{
#if HAVE_IB_EXT_ATOMICS
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_ATOMIC_ADD_DESC(&iface->super, &iface->verbs_common.short_desc_mp, desc);

    /* TODO don't allocate descriptor - have dummy buffer */
    uct_rc_verbs_ep_ext_atomic_post(ep, IBV_EXP_WR_EXT_MASKED_ATOMIC_FETCH_AND_ADD,
                                    sizeof(uint32_t), 0, add, 0, remote_addr,
                                    rkey, desc, IBV_EXP_SEND_SIGNALED);
    return UCS_OK;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_verbs_ep_atomic_fadd32(uct_ep_h tl_ep, uint32_t add,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint32_t *result, uct_completion_t *comp)
{
#if HAVE_IB_EXT_ATOMICS
    return uct_rc_verbs_ep_ext_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                      IBV_EXP_WR_EXT_MASKED_ATOMIC_FETCH_AND_ADD,
                                      result, sizeof(uint32_t), 0, add, 0,
                                      remote_addr, rkey, comp);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_verbs_ep_atomic_swap32(uct_ep_h tl_ep, uint32_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint32_t *result, uct_completion_t *comp)
{
#if HAVE_IB_EXT_ATOMICS
    return uct_rc_verbs_ep_ext_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                      IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP,
                                   result, sizeof(uint32_t), 0, 0, swap,
                                   remote_addr, rkey, comp);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_verbs_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uint32_t *result, uct_completion_t *comp)
{
#if HAVE_IB_EXT_ATOMICS
    return uct_rc_verbs_ep_ext_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                      IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP,
                                      result, sizeof(uint32_t), (uint32_t)(-1),
                                      compare, swap, remote_addr, rkey, comp);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

static ucs_status_t uct_rc_verbs_ep_nop(uct_rc_verbs_ep_t *ep)
{
#if HAVE_DECL_IBV_EXP_WR_NOP
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_rc_verbs_iface_t);
    struct ibv_exp_send_wr wr;

    wr.next           = NULL;
    wr.num_sge        = 0;
    wr.exp_opcode     = IBV_EXP_WR_NOP;
    wr.exp_send_flags = IBV_EXP_SEND_FENCE;
    wr.comp_mask      = 0;
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    uct_rc_verbs_exp_post_send(ep, &wr, IBV_EXP_SEND_SIGNALED);
    return UCS_OK;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_verbs_ep_flush(uct_ep_h tl_ep, unsigned flags,
                                   uct_completion_t *comp)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    ucs_status_t status;

    status = uct_rc_ep_flush(&ep->super, iface->config.tx_max_wr, flags);
    if (status != UCS_INPROGRESS) {
        return status;
    }

    if (uct_rc_txqp_unsignaled(&ep->super.txqp) != 0) {
        if (IBV_DEVICE_HAS_NOP(&uct_ib_iface_device(&iface->super.super)->dev_attr)) {
            status = uct_rc_verbs_ep_nop(ep);
        } else {
            status = uct_rc_verbs_ep_put_short(tl_ep, NULL, 0, 0, 0);
        }
        if (status != UCS_OK) {
            return status;
        }
    }

    uct_rc_txqp_add_send_comp(&iface->super, &ep->super.txqp, comp, ep->txcnt.pi);
    UCT_TL_EP_STAT_FLUSH_WAIT(&ep->super.super);
    return UCS_INPROGRESS;
}

ucs_status_t uct_rc_verbs_ep_fc_ctrl(uct_ep_t *tl_ep, unsigned op,
                                     uct_rc_fc_request_t *req)
{
    struct ibv_send_wr fc_wr;
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    size_t notag_hdr_size = iface->verbs_common.config.notag_hdr_size;
    uct_rc_hdr_t *hdr     = iface->verbs_common.am_inl_hdr + notag_hdr_size;

    /* In RC only PURE grant is sent as a separate message. Other FC
     * messages are bundled with AM. */
    ucs_assert(op == UCT_RC_EP_FC_PURE_GRANT);

    /* Do not check FC WND here to avoid head-to-head deadlock.
     * Credits grant should be sent regardless of FC wnd state. */
    ucs_assert(sizeof(*hdr) + notag_hdr_size <=
               iface->verbs_common.config.max_inline);
    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    hdr->am_id    = UCT_RC_EP_FC_PURE_GRANT;
    fc_wr.sg_list = iface->verbs_common.inl_sge;
    fc_wr.opcode  = IBV_WR_SEND;
    fc_wr.next    = NULL;
    fc_wr.num_sge = 1;

    iface->verbs_common.inl_sge[0].addr    = (uintptr_t)iface->verbs_common.am_inl_hdr;
    iface->verbs_common.inl_sge[0].length  = sizeof(*hdr) + notag_hdr_size;

    uct_rc_verbs_ep_post_send(iface, ep, &fc_wr, IBV_SEND_INLINE);
    return UCS_OK;
}

static ucs_status_t uct_rc_verbs_ep_tag_qp_create(uct_rc_verbs_iface_t *iface,
                                                  uct_rc_verbs_ep_t *ep)
{
#if IBV_EXP_HW_TM
    struct ibv_qp_cap cap;
    ucs_status_t status;
    int ret;

    if (iface->verbs_common.tm.enabled) {
        /* Send queue of this QP will be used by FW for HW RNDV. Driver requires
         * such a QP to be initialized with zero send queue length. */
        status = uct_rc_iface_qp_create(&iface->super, IBV_QPT_RC, &ep->tm_qp,
                                        &cap, 0);
        if (status != UCS_OK) {
            return status;
        }

        status = uct_rc_iface_qp_init(&iface->super, ep->tm_qp);
        if (status != UCS_OK) {
            ret = ibv_destroy_qp(ep->tm_qp);
            if (ret) {
                ucs_warn("ibv_destroy_qp() returned %d: %m", ret);
            }
            return status;
        }
        uct_rc_iface_add_qp(&iface->super, &ep->super, ep->tm_qp->qp_num);
    }
#endif
    return UCS_OK;
}

static void uct_rc_verbs_ep_tag_qp_destroy(uct_rc_verbs_ep_t *ep)
{
#if IBV_EXP_HW_TM
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_rc_verbs_iface_t);
    if (iface->verbs_common.tm.enabled) {
        uct_rc_iface_remove_qp(&iface->super, ep->tm_qp->qp_num);
        if (ibv_destroy_qp(ep->tm_qp)) {
            ucs_warn("failed to destroy TM RNDV QP: %m");
        }
    }
#endif
}

ucs_status_t uct_rc_verbs_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr)
{
    ucs_status_t status;
    uct_rc_verbs_iface_t *iface UCS_V_UNUSED;

    status = uct_rc_ep_get_address(tl_ep, addr);
    if (status != UCS_OK) {
        return status;
    }
#if IBV_EXP_HW_TM
    iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    if (iface->verbs_common.tm.enabled) {
        uct_rc_verbs_ep_tm_address_t *tm_addr = (uct_rc_verbs_ep_tm_address_t*)addr;
        uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);

        uct_ib_pack_uint24(tm_addr->tm_qp_num, ep->tm_qp->qp_num);
    }
#endif
    return UCS_OK;
}

ucs_status_t uct_rc_verbs_ep_connect_to_ep(uct_ep_h tl_ep,
                                           const uct_device_addr_t *dev_addr,
                                           const uct_ep_addr_t *ep_addr)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    const uct_ib_address_t *ib_addr = (const uct_ib_address_t *)dev_addr;
    const uct_rc_ep_address_t *rc_addr = (const uct_rc_ep_address_t*)ep_addr;
    struct ibv_ah_attr ah_attr;
    ucs_status_t status;
    uint32_t qp_num;

    uct_ib_iface_fill_ah_attr(&iface->super.super, ib_addr, ep->super.path_bits,
                              &ah_attr);
#if IBV_EXP_HW_TM
    if (iface->verbs_common.tm.enabled) {
        /* For HW TM we need 2 QPs, one of which will be used by the device for
         * RNDV offload (for issuing RDMA reads and sending RNDV ACK). No WQEs
         * should be posted to the send side of the QP which is owned by device. */
        uct_rc_verbs_ep_tm_address_t *tm_addr = ucs_derived_of(rc_addr,
                                                uct_rc_verbs_ep_tm_address_t);

        status = uct_rc_iface_qp_connect(&iface->super, ep->tm_qp,
                                         uct_ib_unpack_uint24(rc_addr->qp_num),
                                         &ah_attr);
        if (status != UCS_OK) {
            return status;
        }

        /* Need to connect local ep QP to the one owned by device
         * (and bound to XRQ) on the peer. */
        qp_num = uct_ib_unpack_uint24(tm_addr->tm_qp_num);
    } else
#endif
    {
        qp_num = uct_ib_unpack_uint24(rc_addr->qp_num);
    }

    status = uct_rc_iface_qp_connect(&iface->super, ep->super.txqp.qp,
                                     qp_num, &ah_attr);
    if (status != UCS_OK) {
        return status;
    }

    ep->super.atomic_mr_offset = uct_ib_md_atomic_offset(rc_addr->atomic_mr_id);

    return UCS_OK;
}

#if IBV_EXP_HW_TM
ucs_status_t uct_rc_verbs_ep_tag_eager_short(uct_ep_h tl_ep, uct_tag_t tag,
                                             const void *data, size_t length)
{
    uct_rc_verbs_ep_t *ep       = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_rc_verbs_iface_t);
    struct ibv_exp_tmh tmh;

    UCT_CHECK_LENGTH(length + sizeof(struct ibv_exp_tmh), 0,
                     iface->verbs_common.config.max_inline, "tag_short");
    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    uct_rc_verbs_iface_fill_tmh(&tmh, tag, 0, IBV_EXP_TMH_EAGER);
    uct_rc_verbs_iface_fill_inl_sge(&iface->verbs_common, &tmh, sizeof(tmh), data, length);

    uct_rc_verbs_ep_post_send(iface, ep, &iface->inl_am_wr, IBV_SEND_INLINE);
    return UCS_OK;
}

ssize_t uct_rc_verbs_ep_tag_eager_bcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                        uint64_t imm,
                                        uct_pack_callback_t pack_cb,
                                        void *arg)
{
    uct_rc_verbs_ep_t *ep       = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_rc_verbs_iface_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_send_wr wr;
    struct ibv_sge sge;
    size_t length;
    uint32_t app_ctx;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    UCT_RC_VERBS_FILL_TM_IMM(wr, imm, app_ctx, wr.opcode, wr.imm_data, IBV);
    UCT_RC_VERBS_GET_TM_BCOPY_DESC(&iface->super, &iface->super.tx.mp, desc,
                                   tag, app_ctx, pack_cb, arg, length);
    UCT_RC_VERBS_FILL_SGE(wr, sge, length + sizeof(struct ibv_exp_tmh));
    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, 0);
    return length;
}

ucs_status_t uct_rc_verbs_ep_tag_eager_zcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                             uint64_t imm, const uct_iov_t *iov,
                                             size_t iovcnt, uct_completion_t *comp)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep       = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_sge sge[UCT_IB_MAX_IOV];
    struct ibv_send_wr wr;
    int send_flags;
    size_t sge_cnt;
    uint32_t app_ctx;

    UCT_CHECK_IOV_SIZE(iovcnt, 1ul, "uct_rc_verbs_ep_tag_eager_zcopy");
    UCT_RC_CHECK_ZCOPY_DATA(sizeof(struct ibv_exp_tmh),
                            uct_iov_total_length(iov, iovcnt),
                            iface->super.super.config.seg_size);
    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    sge_cnt = uct_ib_verbs_sge_fill_iov(sge + 1, iov, iovcnt);

    UCT_RC_VERBS_FILL_TM_IMM(wr, imm, app_ctx, wr.opcode, wr.imm_data, IBV);
    UCT_RC_VERBS_GET_TM_ZCOPY_DESC(&iface->super,
                                   &iface->verbs_common.short_desc_mp,
                                   desc, tag, app_ctx, comp, &send_flags,
                                   sge[0]);
    wr.num_sge = sge_cnt + 1;
    wr.sg_list = sge;

    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, send_flags);
    return UCS_INPROGRESS;
}

ucs_status_ptr_t uct_rc_verbs_ep_tag_rndv_zcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                                const void *header,
                                                unsigned header_length,
                                                const uct_iov_t *iov,
                                                size_t iovcnt,
                                                uct_completion_t *comp)
{
    uct_rc_verbs_ep_t *ep       = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_rc_verbs_iface_t);
    unsigned tmh_len     = sizeof(struct ibv_exp_tmh) +
                           sizeof(struct ibv_exp_tmh_rvh);
    void *tmh            = ucs_alloca(tmh_len);
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super.super);
    uint32_t op_index;

    UCT_RC_VERBS_CHECK_RNDV_PARAMS(iovcnt, header_length, tmh_len,
                                   iface->verbs_common.config.max_inline,
                                   IBV_DEVICE_TM_CAPS(dev, max_rndv_hdr_size));
    UCT_RC_VERBS_CHECK_RES_PTR(&iface->super, &ep->super);

    op_index = uct_rc_verbs_iface_tag_get_op_id(&iface->verbs_common, comp);
    uct_rc_verbs_iface_fill_tmh(tmh, tag, op_index, IBV_EXP_TMH_RNDV);
    uct_rc_verbs_iface_fill_rvh(tmh + sizeof(struct ibv_exp_tmh), iov->buffer,
                                ((uct_ib_mem_t*)iov->memh)->mr->rkey, iov->length);
    uct_rc_verbs_iface_fill_inl_sge(&iface->verbs_common, tmh, tmh_len,
                                    header, header_length);

    uct_rc_verbs_ep_post_send(iface, ep, &iface->inl_am_wr, IBV_SEND_INLINE);
    return (ucs_status_ptr_t)((uint64_t)op_index);
}

ucs_status_t uct_rc_verbs_ep_tag_rndv_cancel(uct_ep_h tl_ep, void *op)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);

    uint32_t op_index = (uint32_t)((uint64_t)op);
    ucs_ptr_array_remove(&iface->verbs_common.tm.rndv_comps, op_index, 0);
    return UCS_OK;
}

/* For RNDV request send regular eager packet with IBV_SEND_WITH_IMM and
 * imm_value = 0. Receiver will handle such message as rndv request. */
ucs_status_t uct_rc_verbs_ep_tag_rndv_request(uct_ep_h tl_ep, uct_tag_t tag,
                                              const void* header,
                                              unsigned header_length)
{
    uct_rc_verbs_ep_t *ep       = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_rc_verbs_iface_t);
    struct ibv_exp_tmh tmh;
    uint32_t app_ctx;
    struct ibv_send_wr wr;

    UCT_CHECK_LENGTH(header_length + sizeof(tmh), 0,
                     iface->verbs_common.config.max_inline, "tag_short");
    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    wr.sg_list = iface->verbs_common.inl_sge;
    wr.num_sge = 2;
    wr.opcode  = IBV_WR_SEND_WITH_IMM;
    wr.next    = NULL;

    uct_rc_verbs_tag_imm_data_pack(&(wr.imm_data), &app_ctx, 0ul);
    uct_rc_verbs_iface_fill_tmh(&tmh, tag, app_ctx, IBV_EXP_TMH_EAGER);
    uct_rc_verbs_iface_fill_inl_sge(&iface->verbs_common, &tmh, sizeof(tmh),
                                    header, header_length);
    uct_rc_verbs_ep_post_send(iface, ep, &wr, IBV_SEND_INLINE);
    return UCS_OK;
}

#endif /* IBV_EXP_HW_TM */

UCS_CLASS_INIT_FUNC(uct_rc_verbs_ep_t, uct_iface_h tl_iface)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_verbs_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_ep_t, &iface->super);

    uct_rc_txqp_available_set(&self->super.txqp, iface->config.tx_max_wr);
    uct_rc_verbs_txcnt_init(&self->txcnt);

    return uct_rc_verbs_ep_tag_qp_create(iface, self);
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_verbs_ep_t)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(self->super.super.super.iface,
                                                 uct_rc_verbs_iface_t);

    /* NOTE: usually, ci == pi here, but if user calls
     *       flush(UCT_FLUSH_FLAG_CANCEL) then ep_destroy without next progress,
     *       TX-completion handler is not able to return CQ credits because
     *       the EP will not be found (base class destructor deletes itself from
     *       iface->eps). So, lets return credits here since handle_failure
     *       ignores not found EP. */
    ucs_assert(self->txcnt.pi >= self->txcnt.ci);
    iface->super.tx.cq_available += self->txcnt.pi - self->txcnt.ci;
    ucs_assert(iface->super.tx.cq_available < iface->super.config.tx_ops_count);
    uct_rc_verbs_ep_tag_qp_destroy(self);
}

UCS_CLASS_DEFINE(uct_rc_verbs_ep_t, uct_rc_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_verbs_ep_t, uct_ep_t, uct_iface_h);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_verbs_ep_t, uct_ep_t);
