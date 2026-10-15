/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RC_VERBS_IMPL_H
#define UCT_RC_VERBS_IMPL_H

#include <ucs/arch/bitops.h>

#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/rc/base/rc_ep.h>


static inline void
uct_rc_verbs_txqp_posted(uct_rc_txqp_t *txqp, uct_rc_verbs_txcnt_t *txcnt,
                         uct_rc_iface_t *iface, int signaled)
{
    txcnt->pi++;
    uct_rc_txqp_posted(txqp, iface, 1, signaled);
}

ucs_status_t
uct_rc_verbs_iface_common_prepost_recvs(uct_rc_verbs_iface_t *iface);

void uct_rc_verbs_iface_common_progress_enable(uct_iface_h tl_iface, unsigned flags);

void uct_rc_verbs_iface_gc_drain_check(uct_rc_verbs_iface_t *iface);

void uct_rc_verbs_iface_eps_post_recv(uct_rc_verbs_iface_t *iface);

unsigned uct_rc_verbs_iface_post_recv_always(uct_rc_verbs_iface_t *iface,
                                             uct_rc_verbs_ep_t *ep,
                                             unsigned max);

static inline unsigned
uct_rc_verbs_iface_post_recv_common(uct_rc_verbs_iface_t *iface,
                                    uct_rc_verbs_ep_t *ep, int fill)
{
    unsigned available = (ep == NULL) ? iface->super.rx.srq.available :
                                        ep->rx_available;
    unsigned batch     = iface->super.super.config.rx_max_batch;
    unsigned count, posted;

    if (available < batch) {
        if (ucs_likely(fill == 0)) {
            return 0;
        } else {
            count = available;
        }
    } else {
        count = batch;
    }

    posted = uct_rc_verbs_iface_post_recv_always(iface, ep, count);

    /* A QP with no posted receive WRs cannot generate a receive completion.
     * Retry from progress when the shared receive buffer pool cannot supply a
     * full batch. */
    if ((ep != NULL) && (posted < count) && (ep->rx_available >= batch)) {
        iface->rx_post_recv_pending = 1;
    }

    return posted;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_iface_handle_am(uct_rc_iface_t *iface, uct_rc_hdr_t *hdr,
                             uint64_t wr_id, uint32_t qp_num, uint32_t length,
                             uint32_t imm_data, uint32_t slid)
{
    uct_ib_iface_recv_desc_t *desc;
    uct_rc_iface_ops_t *rc_ops;
    ucs_status_t status;
    void *udesc;

    desc = (uct_ib_iface_recv_desc_t *)wr_id;
    if (ucs_unlikely(hdr->am_id & UCT_RC_EP_FC_MASK)) {
        rc_ops = ucs_derived_of(iface->super.ops, uct_rc_iface_ops_t);
        status = rc_ops->fc_handler(iface, qp_num, hdr, length - sizeof(*hdr),
                                    imm_data, slid, UCT_CB_PARAM_FLAG_DESC);
    } else {
        status = uct_iface_invoke_am(&iface->super.super, hdr->am_id, hdr + 1,
                                     length - sizeof(*hdr), UCT_CB_PARAM_FLAG_DESC);
    }

    if (ucs_likely(status != UCS_INPROGRESS)) {
        ucs_mpool_put_inline(desc);
    } else {
        udesc = (char*)desc + iface->super.config.rx_headroom_offset;
        uct_recv_desc(udesc) = &iface->super.release_desc;
    }
}

/* Account for a receive WR flush completion after its QP was removed from the
 * lookup table. A QP created without an SRQ is kept until every posted receive
 * WR has generated a completion. */
static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_iface_gc_drain_cqe(uct_rc_verbs_iface_t *iface, uint32_t qp_num)
{
    uct_rc_verbs_iface_qp_cleanup_ctx_t *cleanup_ctx;

    ucs_list_for_each(cleanup_ctx, &iface->super.qp_gc_list, super.list) {
        if (cleanup_ctx->super.qp_num == qp_num) {
            ucs_assertv(cleanup_ctx->rx_remaining > 0, "qp_num=0x%x", qp_num);
            cleanup_ctx->rx_remaining--;
            return;
        }
    }
}

/* An endpoint which was cancelled or failed must not receive any more */
static UCS_F_ALWAYS_INLINE int
uct_rc_verbs_ep_can_post_recv(const uct_rc_verbs_ep_t *ep)
{
    return !(ep->super.flags & (UCT_RC_EP_FLAG_FLUSH_CANCEL |
                                UCT_RC_EP_FLAG_ERR_HANDLER_INVOKED));
}

/* Account for a receive completion of a QP without an SRQ. */
static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_iface_rx_complete_nosrq(uct_rc_verbs_iface_t *iface,
                                     uint32_t qp_num, int repost)
{
    uct_rc_ep_t *rc_ep    = uct_rc_iface_lookup_ep(&iface->super, qp_num);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(rc_ep, uct_rc_verbs_ep_t);

    if (ep == NULL) {
        uct_rc_verbs_iface_gc_drain_cqe(iface, qp_num);
        return;
    }

    ep->rx_available++;
    if (repost && uct_rc_verbs_ep_can_post_recv(ep)) {
        uct_rc_verbs_iface_post_recv_common(iface, ep, 0);
    }
}

/* Poll receive completions. When nosrq is set, receive WRs are posted to every
 * QP separately, so the QP number of a completion selects the receive queue. */
static UCS_F_ALWAYS_INLINE unsigned
uct_rc_verbs_iface_poll_rx(uct_rc_verbs_iface_t *iface, int nosrq)
{
    uct_ib_iface_recv_desc_t *desc;
    uct_rc_hdr_t *hdr;
    unsigned i;
    ucs_status_t status;
    unsigned num_wcs = iface->super.super.config.rx_max_poll;
    struct ibv_wc wc[num_wcs];

    status = uct_ib_poll_cq(iface->super.super.cq[UCT_IB_DIR_RX], &num_wcs, wc);
    if (status != UCS_OK) {
        num_wcs = 0;
        goto out;
    }

    for (i = 0; i < num_wcs; i++) {
        desc = (uct_ib_iface_recv_desc_t *)(uintptr_t)wc[i].wr_id;
        hdr  = (uct_rc_hdr_t *)uct_ib_iface_recv_desc_hdr(&iface->super.super, desc);
        if (ucs_unlikely(wc[i].status != IBV_WC_SUCCESS)) {
            /* A failed receive does not deliver an active message, so return
             * its descriptor to the pool. A flushed WR belongs to a QP which is
             * being destroyed, so post nothing to it, while an aborted one
             * leaves the QP usable. */
            if ((wc[i].status == IBV_WC_REM_ABORT_ERR) ||
                (wc[i].status == IBV_WC_WR_FLUSH_ERR)) {
                ucs_mpool_put_inline(desc);
                if (nosrq) {
                    uct_rc_verbs_iface_rx_complete_nosrq(iface, wc[i].qp_num,
                                                         wc[i].status !=
                                                         IBV_WC_WR_FLUSH_ERR);
                }
                continue;
            }
            UCT_IB_IFACE_VERBS_COMPLETION_FATAL("receive", &iface->super.super, i, wc);
        }
        VALGRIND_MAKE_MEM_DEFINED(hdr, wc[i].byte_len);

        uct_ib_log_recv_completion(&iface->super.super, &wc[i], hdr, wc[i].byte_len,
                                   uct_rc_ep_packet_dump);
        uct_rc_verbs_iface_handle_am(&iface->super, hdr, wc[i].wr_id, wc[i].qp_num,
                                     wc[i].byte_len, wc[i].imm_data, wc[i].slid);

        if (nosrq) {
            uct_rc_verbs_iface_rx_complete_nosrq(iface, wc[i].qp_num, 1);
        }
    }
    if (!nosrq) {
        iface->super.rx.srq.available += num_wcs;
    }
    UCS_STATS_UPDATE_COUNTER(iface->super.super.stats,
                             UCT_IB_IFACE_STAT_RX_COMPLETION, num_wcs);

out:
    if (nosrq) {
        if (ucs_unlikely(iface->rx_post_recv_pending)) {
            uct_rc_verbs_iface_eps_post_recv(iface);
        }

        if (ucs_unlikely(!ucs_list_is_empty(&iface->super.qp_gc_list))) {
            uct_rc_verbs_iface_gc_drain_check(iface);
        }
    } else {
        uct_rc_verbs_iface_post_recv_common(iface, NULL, 0);
    }

    return num_wcs;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_iface_fill_inl_sge(uct_rc_verbs_iface_t *iface, const void *addr0,
                                unsigned len0, const void* addr1, unsigned len1)
{
    iface->inl_sge[0].addr      = (uintptr_t)addr0;
    iface->inl_sge[0].length    = len0;
    iface->inl_sge[1].addr      = (uintptr_t)addr1;
    iface->inl_sge[1].length    = len1;
}

static inline void
uct_rc_verbs_iface_fill_inl_am_sge(uct_rc_verbs_iface_t *iface,
                                   uint8_t id, uint64_t hdr,
                                   const void *buffer, unsigned length)
{
    uct_rc_am_short_hdr_t *am = &iface->am_inl_hdr;

    am->rc_hdr.am_id          = id;
    am->am_hdr                = hdr;
    iface->inl_am_wr.num_sge  = 2;
    uct_rc_verbs_iface_fill_inl_sge(iface, am, sizeof(*am), buffer, length);
}

static inline void
uct_rc_verbs_iface_fill_inl_am_sge_iov(uct_rc_verbs_iface_t *iface, uint8_t id,
                                       const uct_iov_t *iov, size_t iovcnt)
{
    uct_rc_hdr_t *rch        = &iface->am_inl_hdr.rc_hdr;

    rch->am_id               = id;
    iface->inl_sge[0].addr   = (uintptr_t)rch;
    iface->inl_sge[0].length = sizeof(*rch);
    iface->inl_am_wr.num_sge = uct_ib_verbs_sge_fill_iov(iface->inl_sge + 1, iov,
                                                         iovcnt) + 1;
}

#define UCT_RC_VERBS_FILL_SGE(_wr, _sge, _length) \
    _wr.sg_list = &_sge; \
    _wr.num_sge = 1; \
    _sge.length = _length;

#define UCT_RC_VERBS_FILL_INL_PUT_WR(_iface, _raddr, _rkey, _buf, _len) \
    _iface->inl_rwrite_wr.wr.rdma.remote_addr = _raddr; \
    _iface->inl_rwrite_wr.wr.rdma.rkey        = _rkey; \
    _iface->inl_sge[0].addr                   = (uintptr_t)_buf; \
    _iface->inl_sge[0].length                 = _len;

#define UCT_RC_VERBS_FILL_AM_BCOPY_WR(_wr, _sge, _length, _wr_opcode) \
    UCT_RC_VERBS_FILL_SGE(_wr, _sge, _length) \
    _wr_opcode = (typeof(_wr_opcode))IBV_WR_SEND;

#define UCT_RC_VERBS_FILL_AM_ZCOPY_WR_IOV(_wr, _sge, _iovlen, _wr_opcode) \
    _wr.sg_list = _sge; \
    _wr.num_sge = _iovlen; \
    _wr_opcode  = (typeof(_wr_opcode))IBV_WR_SEND;

#define UCT_RC_VERBS_FILL_RDMA_WR(_wr, _wr_opcode, _opcode, \
                                  _sge, _length, _raddr, _rkey) \
    UCT_RC_VERBS_FILL_SGE(_wr, _sge, _length) \
    _wr.wr.rdma.remote_addr = _raddr; \
    _wr.wr.rdma.rkey        = _rkey; \
    _wr_opcode              = _opcode; \

#define UCT_RC_VERBS_FILL_RDMA_WR_IOV(_wr, _wr_opcode, _opcode, _sge, _sgelen, \
                                      _raddr, _rkey) \
    _wr.wr.rdma.remote_addr = _raddr; \
    _wr.wr.rdma.rkey        = _rkey; \
    _wr.sg_list             = _sge; \
    _wr.num_sge             = _sgelen; \
    _wr_opcode              = _opcode;

#define UCT_RC_VERBS_FILL_DESC_WR(_wr, _desc) \
    { \
        struct ibv_sge *sge; \
        (_wr)->next    = NULL; \
        sge            = (_wr)->sg_list; \
        sge->addr      = (uintptr_t)((_desc) + 1); \
        sge->lkey      = (_desc)->lkey; \
    }

#define UCT_RC_VERBS_FILL_ATOMIC_WR(_wr, _wr_opcode, _sge, _opcode, \
                                    _compare_add, _swap, _remote_addr, _rkey) \
    UCT_RC_VERBS_FILL_SGE(_wr, _sge, sizeof(uint64_t)) \
    _wr_opcode                = _opcode; \
    _wr.wr.atomic.compare_add = _compare_add; \
    _wr.wr.atomic.swap        = _swap; \
    _wr.wr.atomic.remote_addr = _remote_addr; \
    _wr.wr.atomic.rkey        = _rkey;  \


#endif
