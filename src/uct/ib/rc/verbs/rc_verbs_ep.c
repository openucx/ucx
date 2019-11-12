/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rc_verbs.h"
#include "rc_verbs_impl.h"

#include <ucs/arch/bitops.h>
#include <uct/ib/base/ib_log.h>

void uct_rc_verbs_txcnt_init(uct_rc_verbs_txcnt_t *txcnt)
{
    txcnt->pi = txcnt->ci = 0;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_ep_post_send(uct_rc_verbs_iface_t* iface, uct_rc_verbs_ep_t* ep,
                          struct ibv_send_wr *wr, int send_flags, int max_log_sge)
{
    struct ibv_send_wr *bad_wr;
    int ret;

    ucs_assertv(ep->qp->state == IBV_QPS_RTS, "QP 0x%x state is %d",
                ep->qp->qp_num, ep->qp->state);

    if (!(send_flags & IBV_SEND_SIGNALED)) {
        send_flags |= uct_rc_iface_tx_moderation(&iface->super, &ep->super.txqp,
                                                 IBV_SEND_SIGNALED);
    }
    if (wr->opcode == IBV_WR_RDMA_READ) {
        send_flags |= uct_rc_ep_fm(&iface->super, &ep->fi, IBV_SEND_FENCE);
    }

    wr->send_flags = send_flags;
    wr->wr_id      = uct_rc_txqp_unsignaled(&ep->super.txqp);

    uct_ib_log_post_send(&iface->super.super, ep->qp, wr, max_log_sge,
                         (wr->opcode == IBV_WR_SEND) ? uct_rc_ep_packet_dump : NULL);

    ret = ibv_post_send(ep->qp, wr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_send() returned %d (%m)", ret);
    }

    uct_rc_verbs_txqp_posted(&ep->super.txqp, &ep->txcnt, &iface->super, send_flags & IBV_SEND_SIGNALED);
}

/*
 * Helper function for posting sends with a descriptor.
 * User needs to fill: wr.opcode, wr.sg_list, wr.num_sge, first sge length, and
 * operation-specific fields (e.g rdma).
 */
static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_ep_post_send_desc(uct_rc_verbs_ep_t* ep, struct ibv_send_wr *wr,
                               uct_rc_iface_send_desc_t *desc, int send_flags,
                               int max_log_sge)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_rc_verbs_iface_t);
    UCT_RC_VERBS_FILL_DESC_WR(wr, desc);
    uct_rc_verbs_ep_post_send(iface, ep, wr, send_flags, max_log_sge);
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
    UCT_RC_VERBS_FILL_RDMA_WR_IOV(wr, wr.opcode, (enum ibv_wr_opcode)opcode,
                                  sge, sge_cnt, remote_addr, rkey);
    wr.next = NULL;

    uct_rc_verbs_ep_post_send(iface, ep, &wr, IBV_SEND_SIGNALED, INT_MAX);
    uct_rc_txqp_add_send_comp(&iface->super, &ep->super.txqp, comp, ep->txcnt.pi,
                              UCT_RC_IFACE_SEND_OP_FLAG_ZCOPY);
    return UCS_INPROGRESS;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_ep_atomic_post(uct_rc_verbs_ep_t *ep, int opcode, uint64_t compare_add,
                            uint64_t swap, uint64_t remote_addr, uct_rkey_t rkey,
                            uct_rc_iface_send_desc_t *desc, int force_sig)
{
    struct ibv_send_wr wr;
    struct ibv_sge sge;

    UCT_RC_VERBS_FILL_ATOMIC_WR(wr, wr.opcode, sge, (enum ibv_wr_opcode)opcode,
                                compare_add, swap, remote_addr,
                                uct_ib_md_direct_rkey(rkey));
    UCT_TL_EP_STAT_ATOMIC(&ep->super.super);
    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, force_sig, INT_MAX);
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
    UCT_RC_IFACE_GET_TX_ATOMIC_FETCH_DESC(&iface->super, &iface->short_desc_mp,
                                          desc, iface->super.config.atomic64_handler,
                                          result, comp);
    uct_rc_verbs_ep_atomic_post(ep, opcode, compare_add, swap, remote_addr,
                                rkey, desc, IBV_SEND_SIGNALED |
                                uct_rc_ep_fm(&iface->super, &ep->fi, IBV_SEND_FENCE));
    return UCS_INPROGRESS;
}

ucs_status_t uct_rc_verbs_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                       unsigned length, uint64_t remote_addr,
                                       uct_rkey_t rkey)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);

    UCT_CHECK_LENGTH(length, 0, iface->config.max_inline, "put_short");

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_VERBS_FILL_INL_PUT_WR(iface, remote_addr, rkey, buffer, length);
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, SHORT, length);
    uct_rc_verbs_ep_post_send(iface, ep, &iface->inl_rwrite_wr,
                              IBV_SEND_INLINE | IBV_SEND_SIGNALED, INT_MAX);
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
    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, IBV_SEND_SIGNALED, INT_MAX);
    return length;
}

ucs_status_t uct_rc_verbs_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    uct_ib_iface_t UCS_V_UNUSED *iface = ucs_derived_of(tl_ep->iface,
                                                        uct_ib_iface_t);
    uct_rc_verbs_ep_t *ep              = ucs_derived_of(tl_ep,
                                                        uct_rc_verbs_ep_t);
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
    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, IBV_SEND_SIGNALED, INT_MAX);
    return UCS_INPROGRESS;
}

ucs_status_t uct_rc_verbs_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    uct_ib_iface_t UCS_V_UNUSED *iface = ucs_derived_of(tl_ep->iface,
                                                        uct_ib_iface_t);
    uct_rc_verbs_ep_t *ep              = ucs_derived_of(tl_ep,
                                                        uct_rc_verbs_ep_t);
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

    UCT_RC_CHECK_AM_SHORT(id, length, iface->config.max_inline);
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_CHECK_FC(&iface->super, &ep->super, id);
    uct_rc_verbs_iface_fill_inl_am_sge(iface, id, hdr, buffer, length);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, sizeof(hdr) + length);
    uct_rc_verbs_ep_post_send(iface, ep, &iface->inl_am_wr,
                              IBV_SEND_INLINE | IBV_SEND_SOLICITED, INT_MAX);
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

    UCT_CHECK_AM_ID(id);

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_CHECK_FC(&iface->super, &ep->super, id);
    UCT_RC_IFACE_GET_TX_AM_BCOPY_DESC(&iface->super, &iface->super.tx.mp, desc,
                                      id, uct_rc_am_hdr_fill, uct_rc_hdr_t,
                                      pack_cb, arg, &length);
    UCT_RC_VERBS_FILL_AM_BCOPY_WR(wr, sge, length + sizeof(uct_rc_hdr_t),
                                  wr.opcode);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, BCOPY, length);
    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, IBV_SEND_SOLICITED, INT_MAX);
    UCT_RC_UPDATE_FC(&iface->super, &ep->super, id);

    return length;
}

ucs_status_t uct_rc_verbs_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                      unsigned header_length, const uct_iov_t *iov,
                                      size_t iovcnt, unsigned flags,
                                      uct_completion_t *comp)
{
    uct_rc_verbs_iface_t     *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t        *ep    = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc  = NULL;
    struct ibv_sge sge[UCT_IB_MAX_IOV]; /* First sge is reserved for the header */
    struct ibv_send_wr wr;
    int send_flags;
    size_t sge_cnt;

    UCT_CHECK_IOV_SIZE(iovcnt, uct_ib_iface_get_max_iov(&iface->super.super) - 1,
                       "uct_rc_verbs_ep_am_zcopy");
    UCT_RC_CHECK_AM_ZCOPY(id, header_length, uct_iov_total_length(iov, iovcnt),
                          iface->config.short_desc_size,
                          iface->super.super.config.seg_size);
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_CHECK_FC(&iface->super, &ep->super, id);

    UCT_RC_IFACE_GET_TX_AM_ZCOPY_DESC(&iface->super, &iface->short_desc_mp,
                                      desc, id, header, header_length, comp,
                                      &send_flags);
    sge[0].length = sizeof(uct_rc_hdr_t) + header_length;
    sge_cnt = uct_ib_verbs_sge_fill_iov(sge + 1, iov, iovcnt);
    UCT_RC_VERBS_FILL_AM_ZCOPY_WR_IOV(wr, sge, (sge_cnt + 1), wr.opcode);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, ZCOPY,
                      (header_length + uct_iov_total_length(iov, iovcnt)));

    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, send_flags | IBV_SEND_SOLICITED,
                                   UCT_IB_MAX_ZCOPY_LOG_SGE(&iface->super.super));
    UCT_RC_UPDATE_FC(&iface->super, &ep->super, id);

    return UCS_INPROGRESS;
}

ucs_status_t uct_rc_verbs_ep_atomic64_post(uct_ep_h tl_ep, unsigned opcode, uint64_t value,
                                           uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep       = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;

    if (opcode != UCT_ATOMIC_OP_ADD) {
        return UCS_ERR_UNSUPPORTED;
    }

    /* TODO don't allocate descriptor - have dummy buffer */
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_ATOMIC_DESC(&iface->super, &iface->short_desc_mp, desc);

    uct_rc_verbs_ep_atomic_post(ep,
                                IBV_WR_ATOMIC_FETCH_AND_ADD, value, 0,
                                remote_addr, rkey, desc,
                                IBV_SEND_SIGNALED);
    return UCS_OK;
}

ucs_status_t uct_rc_verbs_ep_atomic64_fetch(uct_ep_h tl_ep, uct_atomic_op_t opcode,
                                            uint64_t value, uint64_t *result,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uct_completion_t *comp)
{
    if (opcode != UCT_ATOMIC_OP_ADD) {
        return UCS_ERR_UNSUPPORTED;
    }

    return uct_rc_verbs_ep_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                  IBV_WR_ATOMIC_FETCH_AND_ADD, result, value, 0,
                                  remote_addr, rkey, comp);
}

ucs_status_t uct_rc_verbs_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uint64_t *result, uct_completion_t *comp)
{
    return uct_rc_verbs_ep_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                  IBV_WR_ATOMIC_CMP_AND_SWP, result, compare, swap,
                                  remote_addr, rkey, comp);
}

ucs_status_t uct_rc_verbs_ep_flush(uct_ep_h tl_ep, unsigned flags,
                                   uct_completion_t *comp)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    ucs_status_t status;

    if (ucs_unlikely(flags & UCT_FLUSH_FLAG_CANCEL)) {
        uct_ep_pending_purge(&ep->super.super.super, NULL, 0);
        uct_rc_verbs_ep_handle_failure(ep, UCS_ERR_CANCELED);
        return UCS_OK;
    }

    status = uct_rc_ep_flush(&ep->super, iface->config.tx_max_wr, flags);
    if (status != UCS_INPROGRESS) {
        return status;
    }

    if (uct_rc_txqp_unsignaled(&ep->super.txqp) != 0) {
        status = uct_rc_verbs_ep_put_short(tl_ep, NULL, 0, 0, 0);
        if (status != UCS_OK) {
            return status;
        }
    }

    return uct_rc_txqp_add_flush_comp(&iface->super, &ep->super.super,
                                      &ep->super.txqp, comp, ep->txcnt.pi);
}

ucs_status_t uct_rc_verbs_ep_fence(uct_ep_h tl_ep, unsigned flags)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);

    return uct_rc_ep_fence(tl_ep, &ep->fi, 1);
}

ucs_status_t uct_rc_verbs_ep_fc_ctrl(uct_ep_t *tl_ep, unsigned op,
                                     uct_rc_fc_request_t *req)
{
    struct ibv_send_wr fc_wr;
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_hdr_t *hdr;
    struct ibv_sge sge;
    int flags;

    if (!iface->fc_desc) {
        hdr                      = &iface->am_inl_hdr.rc_hdr;
        hdr->am_id               = UCT_RC_EP_FC_PURE_GRANT;
        fc_wr.sg_list            = iface->inl_sge;
        iface->inl_sge[0].addr   = (uintptr_t)hdr;
        iface->inl_sge[0].length = sizeof(*hdr);
        flags                    = IBV_SEND_INLINE;
    } else {
        hdr           = (uct_rc_hdr_t*)(iface->fc_desc + 1);
        sge.addr      = (uintptr_t)hdr;
        sge.length    = sizeof(*hdr);
        sge.lkey      = iface->fc_desc->lkey;
        fc_wr.sg_list = &sge;
        flags         = 0;
    }

    /* In RC only PURE grant is sent as a separate message. Other FC
     * messages are bundled with AM. */
    ucs_assert(op == UCT_RC_EP_FC_PURE_GRANT);

    /* Do not check FC WND here to avoid head-to-head deadlock.
     * Credits grant should be sent regardless of FC wnd state. */
    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    fc_wr.opcode  = IBV_WR_SEND;
    fc_wr.next    = NULL;
    fc_wr.num_sge = 1;

    uct_rc_verbs_ep_post_send(iface, ep, &fc_wr, flags, INT_MAX);
    return UCS_OK;
}

ucs_status_t uct_rc_verbs_ep_handle_failure(uct_rc_verbs_ep_t *ep,
                                            ucs_status_t status)
{
    uct_rc_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                           uct_rc_iface_t);

    iface->tx.cq_available += ep->txcnt.pi - ep->txcnt.ci;
    /* Reset CI to prevent cq_available overrun on ep_destroy */
    ep->txcnt.ci = ep->txcnt.pi;
    uct_rc_txqp_purge_outstanding(&ep->super.txqp, status, 0);

    return iface->super.ops->set_ep_failed(&iface->super, &ep->super.super.super,
                                           status);
}

ucs_status_t uct_rc_verbs_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr)
{
    uct_rc_verbs_ep_t *ep        = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_ep_address_t *rc_addr = (uct_rc_ep_address_t*)addr;

    uct_ib_pack_uint24(rc_addr->qp_num, ep->qp->qp_num);
    return UCS_OK;
}

ucs_status_t uct_rc_verbs_ep_connect_to_ep(uct_ep_h tl_ep, const uct_device_addr_t *dev_addr,
                                           const uct_ep_addr_t *ep_addr)
{
    uct_rc_verbs_ep_t *ep              = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_t *iface              = ucs_derived_of(tl_ep->iface, uct_rc_iface_t);
    const uct_ib_address_t *ib_addr    = (const uct_ib_address_t *)dev_addr;
    const uct_rc_ep_address_t *rc_addr = (const uct_rc_ep_address_t*)ep_addr;
    uint32_t qp_num;
    struct ibv_ah_attr ah_attr;

    uct_ib_iface_fill_ah_attr_from_addr(&iface->super, ib_addr, &ah_attr);
    qp_num = uct_ib_unpack_uint24(rc_addr->qp_num);

    return uct_rc_iface_qp_connect(iface, ep->qp, qp_num, &ah_attr);
}

UCS_CLASS_INIT_FUNC(uct_rc_verbs_ep_t, const uct_ep_params_t *params)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(params->iface, uct_rc_verbs_iface_t);
    uct_ib_qp_attr_t attr = {};
    ucs_status_t status;

    status = uct_rc_iface_qp_create(&iface->super, &self->qp, &attr,
                                    iface->super.config.tx_qp_len, iface->srq);
    if (status != UCS_OK) {
        goto err;
    }

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_ep_t, &iface->super, self->qp->qp_num);

    status = uct_rc_iface_qp_init(&iface->super, self->qp);
    if (status != UCS_OK) {
        goto err_qp_cleanup;
    }

    uct_rc_iface_add_qp(&iface->super, &self->super, self->qp->qp_num);

    uct_rc_txqp_available_set(&self->super.txqp, iface->config.tx_max_wr);
    uct_rc_verbs_txcnt_init(&self->txcnt);
    uct_ib_fence_info_init(&self->fi);

    return UCS_OK;

err_qp_cleanup:
    uct_ib_destroy_qp(self->qp);
err:
    return status;
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
    uct_rc_iface_remove_qp(&iface->super, self->qp->qp_num);
    uct_ib_destroy_qp(self->qp);
}

UCS_CLASS_DEFINE(uct_rc_verbs_ep_t, uct_rc_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_verbs_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_verbs_ep_t, uct_ep_t);
