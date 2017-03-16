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
                         (wr->opcode == IBV_WR_SEND) ? uct_rc_ep_am_packet_dump : NULL);

    ret = ibv_post_send(ep->super.txqp.qp, wr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_send() returned %d (%m)", ret);
    }

    uct_rc_verbs_txqp_posted(&ep->super.txqp, &ep->txcnt, &iface->super, send_flags & IBV_SEND_SIGNALED);
}

#if HAVE_DECL_IBV_EXP_POST_SEND
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
                             (wr->exp_opcode == IBV_EXP_WR_SEND) ? uct_rc_ep_am_packet_dump : NULL);

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
    unsigned is_rdma_read       = (IBV_WR_RDMA_READ == opcode);
    unsigned copy_used          = 0;
    void *iov_buf_desc          = NULL;
    void *desc_iov_t            = NULL; /* avoid break strict-aliasing rules */
    uct_rc_iface_send_desc_t *desc_iov;
    struct ibv_sge sge[UCT_IB_MAX_IOV];
    struct ibv_send_wr wr;
    ucs_status_t status;
    size_t sge_cnt;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    status = uct_ib_verbs_sge_fill_iov(sge, iov, iovcnt,
                                       &iface->super.super, &iface->super.tx.mp,
                                       &desc_iov_t, sizeof(*desc_iov),
                                       &iov_buf_desc,
                                       ucs_offsetof(uct_rc_iface_send_desc_t, lkey),
                                       is_rdma_read, &copy_used, &sge_cnt);
    if (UCS_OK != status) {
        return status;
    }
    UCT_SKIP_ZERO_LENGTH(sge_cnt);
    UCT_RC_VERBS_FILL_RDMA_WR_IOV(wr, wr.opcode, opcode, sge, sge_cnt, remote_addr, rkey);
    wr.next = NULL;

    uct_rc_verbs_ep_post_send(iface, ep, &wr, IBV_SEND_SIGNALED);
    uct_rc_txqp_add_send_comp(&iface->super, &ep->super.txqp, comp, ep->txcnt.pi);
    /* TODO if copy_used == sge_cnt the UCS_OK could be returned */
    if (copy_used) {
        ucs_assert(NULL != desc_iov_t);
        desc_iov = desc_iov_t;
        if (is_rdma_read) {
            /* schedule a callback to copy payload to the user buffer at completion */
            desc_iov->super.handler = uct_rc_ep_get_zcopy_handler;
            desc_iov->super.buffer  = iov_buf_desc;
        } else {
            desc_iov->super.handler = (uct_rc_send_handler_t)ucs_mpool_put;
        }
        uct_rc_txqp_add_send_op_sn(&ep->super.txqp, &desc_iov->super, ep->txcnt.pi);
    }
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

    UCT_CHECK_LENGTH(length, iface->verbs_common.config.max_inline, "put_short");

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

    UCT_CHECK_IOV_PARAM(iov, iovcnt, iface, UCT_IB_MAX_MESSAGE_SIZE,
                        iface->config.seg_size, "put_zcopy");

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

    UCT_CHECK_LENGTH(length, iface->super.super.config.seg_size, "get_bcopy");
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

    UCT_CHECK_IOV_PARAM(iov, iovcnt, iface, UCT_IB_MAX_MESSAGE_SIZE,
                        iface->config.seg_size, "get_zcopy");

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
    uct_rc_am_short_hdr_t am;

    UCT_RC_CHECK_AM_SHORT(id, length, iface->verbs_common.config.max_inline);

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_CHECK_FC(&iface->super, &ep->super, id);
    uct_rc_verbs_iface_fill_inl_am_sge(&iface->verbs_common, &am, id, hdr, buffer, length);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, sizeof(hdr) + length);
    uct_rc_verbs_ep_post_send(iface, ep, &iface->inl_am_wr, IBV_SEND_INLINE);
    UCT_RC_UPDATE_FC(&iface->super, &ep->super, id);

    return UCS_OK;
}

ssize_t uct_rc_verbs_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                 uct_pack_callback_t pack_cb, void *arg)
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
                                      id, pack_cb, arg, &length);
    UCT_RC_VERBS_FILL_AM_BCOPY_WR(wr, sge, length, wr.opcode);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, BCOPY, length);
    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, 0);
    UCT_RC_UPDATE_FC(&iface->super, &ep->super, id);

    return length;
}

ucs_status_t uct_rc_verbs_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                      unsigned header_length, const uct_iov_t *iov,
                                      size_t iovcnt, uct_completion_t *comp)
{
    uct_rc_verbs_iface_t     *iface    = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t        *ep       = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc     = NULL;
    unsigned copy_used                 = 0;
    void *desc_iov_t                   = NULL; /* avoid break strict-aliasing rules */
    uct_rc_iface_send_desc_t *desc_iov;
    struct ibv_sge sge[UCT_IB_MAX_IOV]; /* First sge is reserved for the header */
    struct ibv_send_wr wr;
    ucs_status_t status;
    int send_flags;
    size_t sge_cnt;

    UCT_CHECK_IOV_SIZE(iovcnt, uct_ib_iface_get_max_iov(&iface->super.super) - 1,
                       "uct_rc_verbs_ep_am_zcopy");
    UCT_CHECK_LENGTH(uct_iov_total_copy_length(iov, iovcnt),
                     iface->super.super.config.seg_size, "am_zcopy lazy_mem");
    UCT_RC_CHECK_AM_ZCOPY(id, header_length, uct_iov_total_length(iov, iovcnt),
                          iface->verbs_common.config.short_desc_size,
                          iface->super.super.config.seg_size);
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_CHECK_FC(&iface->super, &ep->super, id);
    UCT_RC_IFACE_GET_TX_AM_ZCOPY_DESC(&iface->super, &iface->verbs_common.short_desc_mp,
                                      desc, id, header, header_length, comp, &send_flags);

    sge[0].length = sizeof(uct_rc_hdr_t) + header_length;
    status = uct_ib_verbs_sge_fill_iov(sge + 1, iov, iovcnt,
                                       &iface->super.super, &iface->super.tx.mp,
                                       &desc_iov_t, sizeof(*desc_iov), NULL,
                                       ucs_offsetof(uct_rc_iface_send_desc_t, lkey),
                                       0, &copy_used, &sge_cnt);
    if (UCS_OK != status) {
        return status;
    }
    UCT_RC_VERBS_FILL_AM_ZCOPY_WR_IOV(wr, sge, (sge_cnt + 1), wr.opcode);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, ZCOPY,
                      (header_length + uct_iov_total_length(iov, iovcnt)));

    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, send_flags);
    /* TODO if copy_used == sge_cnt the UCS_OK could be returned */
    if (copy_used) {
        ucs_assert(NULL != desc_iov_t);
        desc_iov = desc_iov_t;
        desc_iov->super.handler = (uct_rc_send_handler_t)ucs_mpool_put;
        uct_rc_txqp_add_send_op_sn(&ep->super.txqp, &desc_iov->super, ep->txcnt.pi);
    }
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

    if (!uct_rc_iface_has_tx_resources(&iface->super)) {
        return UCS_ERR_NO_RESOURCE;
    }

    if (uct_rc_txqp_available(&ep->super.txqp) == iface->config.tx_max_wr) {
        UCT_TL_EP_STAT_FLUSH(&ep->super.super);
        return UCS_OK;
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
    } else if (!uct_rc_ep_has_tx_resources(&ep->super)) {
        return UCS_ERR_NO_RESOURCE;
    }

    uct_rc_txqp_add_send_comp(&iface->super, &ep->super.txqp, comp, ep->txcnt.pi);
    UCT_TL_EP_STAT_FLUSH_WAIT(&ep->super.super);
    return UCS_INPROGRESS;
}

ucs_status_t uct_rc_verbs_ep_fc_ctrl(uct_ep_t *tl_ep, unsigned op,
                                     uct_rc_fc_request_t *req)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_hdr_t hdr;
    struct ibv_send_wr fc_wr;

    /* In RC only PURE grant is sent as a separate message. Other FC
     * messages are bundled with AM. */
    ucs_assert(op == UCT_RC_EP_FC_PURE_GRANT);

    /* Do not check FC WND here to avoid head-to-head deadlock.
     * Credits grant should be sent regardless of FC wnd state. */
    ucs_assert(sizeof(hdr) <= iface->verbs_common.config.max_inline);
    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    hdr.am_id                     = UCT_RC_EP_FC_PURE_GRANT;

    fc_wr.sg_list                 = iface->verbs_common.inl_sge;
    fc_wr.num_sge                 = 1;
    fc_wr.opcode                  = IBV_WR_SEND;
    fc_wr.next                    = NULL;

    iface->verbs_common.inl_sge[0].addr        = (uintptr_t)&hdr;
    iface->verbs_common.inl_sge[0].length      = sizeof(hdr);

    uct_rc_verbs_ep_post_send(iface, ep, &fc_wr, IBV_SEND_INLINE);
    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_rc_verbs_ep_t, uct_iface_h tl_iface)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_verbs_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_ep_t, &iface->super);

    uct_rc_txqp_available_set(&self->super.txqp, iface->config.tx_max_wr);
    uct_rc_verbs_txcnt_init(&self->txcnt);

    uct_worker_progress_register(iface->super.super.super.worker,
                                 uct_rc_verbs_iface_progress, iface);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_verbs_ep_t)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(self->super.super.super.iface,
                                                 uct_rc_verbs_iface_t);
    uct_worker_progress_unregister(iface->super.super.super.worker,
                                   uct_rc_verbs_iface_progress, iface);
}

UCS_CLASS_DEFINE(uct_rc_verbs_ep_t, uct_rc_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_verbs_ep_t, uct_ep_t, uct_iface_h);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_verbs_ep_t, uct_ep_t);

