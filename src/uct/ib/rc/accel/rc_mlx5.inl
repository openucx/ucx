/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rc_mlx5.h"
#include "rc_mlx5_common.h"

#include <uct/base/uct_iov.inl>
#include <uct/ib/mlx5/ib_mlx5.inl>
#include <uct/ib/mlx5/ib_mlx5_log.h>

#define UCT_RC_MLX5_EP_DECL(_tl_ep, _iface, _ep) \
    uct_rc_mlx5_ep_t *_ep = ucs_derived_of(_tl_ep, uct_rc_mlx5_ep_t); \
    uct_rc_mlx5_iface_common_t *_iface = ucs_derived_of(_tl_ep->iface, \
                                                        uct_rc_mlx5_iface_common_t)


static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_ep_fence_put(uct_rc_mlx5_iface_common_t *iface, uct_ib_mlx5_txwq_t *txwq,
                         uct_rkey_t *rkey, uint64_t *addr, uint16_t offset)
{
    uct_rc_ep_fence_put(&iface->super, &txwq->fi, rkey, addr, offset);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_ep_fence_get(uct_rc_mlx5_iface_common_t *iface, uct_ib_mlx5_txwq_t *txwq,
                         uct_rkey_t *rkey, uint8_t *fm_ce_se)
{
    *rkey      = uct_ib_md_direct_rkey(*rkey);
    *fm_ce_se |= uct_rc_ep_fm(&iface->super, &txwq->fi, iface->config.atomic_fence_flag);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_txqp_process_tx_cqe(uct_rc_txqp_t *txqp, struct mlx5_cqe64 *cqe,
                                uint16_t hw_ci)
{
    if (cqe->op_own & MLX5_INLINE_SCATTER_32) {
        uct_rc_txqp_completion_inl_resp(txqp, cqe, hw_ci);
    } else if (cqe->op_own & MLX5_INLINE_SCATTER_64) {
        uct_rc_txqp_completion_inl_resp(txqp, cqe - 1, hw_ci);
    } else {
        uct_rc_txqp_completion_desc(txqp, hw_ci);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_common_rx_inline(uct_rc_mlx5_iface_common_t *iface,
                                   uct_ib_iface_recv_desc_t *desc,
                                   int stats_counter, unsigned byte_len)
{
    UCS_STATS_UPDATE_COUNTER(iface->stats, stats_counter, 1);
    VALGRIND_MAKE_MEM_UNDEFINED(uct_ib_iface_recv_desc_hdr(&iface->super.super, desc),
                                byte_len);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_srq_prefetch_setup(uct_rc_mlx5_iface_common_t *iface)
{
    unsigned wqe_ctr = iface->rx.srq.free_idx + 2;
    uct_ib_mlx5_srq_seg_t *seg;

    seg = uct_ib_mlx5_srq_get_wqe(&iface->rx.srq, wqe_ctr);
    iface->rx.pref_ptr =
            uct_ib_iface_recv_desc_hdr(&iface->super.super, seg->srq.desc);
}

static UCS_F_NOINLINE void
uct_rc_mlx5_iface_hold_srq_desc(uct_rc_mlx5_iface_common_t *iface,
                                uct_ib_mlx5_srq_seg_t *seg,
                                struct mlx5_cqe64 *cqe, uint16_t wqe_ctr,
                                ucs_status_t status, unsigned offset,
                                uct_recv_desc_t *release_desc)
{
    void *udesc;
    int stride_idx;
    int desc_offset;

    if (UCT_RC_MLX5_MP_ENABLED(iface)) {
        /* stride_idx is valid in non inline CQEs only.
         * We can assume that stride_idx is correct here, because CQE
         * with data would always force upper layer to save the data and
         * return UCS_OK from the corresponding callback. */
        stride_idx = uct_ib_mlx5_cqe_stride_index(cqe);
        ucs_assert(stride_idx < iface->tm.mp.num_strides);
        ucs_assert(!(cqe->op_own & (MLX5_INLINE_SCATTER_32 |
                                    MLX5_INLINE_SCATTER_64)));

        udesc       = (void*)be64toh(seg->dptr[stride_idx].addr);
        desc_offset = offset - iface->super.super.config.rx_hdr_offset;
        udesc       = UCS_PTR_BYTE_OFFSET(udesc, desc_offset);
        uct_recv_desc(udesc) = release_desc;
        seg->srq.ptr_mask   &= ~UCS_BIT(stride_idx);
    } else {
        udesc                = UCS_PTR_BYTE_OFFSET(seg->srq.desc, offset);
        uct_recv_desc(udesc) = release_desc;
        seg->srq.ptr_mask   &= ~1;
        seg->srq.desc        = NULL;
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_release_srq_seg(uct_rc_mlx5_iface_common_t *iface,
                                  uct_ib_mlx5_srq_seg_t *seg,
                                  struct mlx5_cqe64 *cqe, uint16_t wqe_ctr,
                                  ucs_status_t status, unsigned offset,
                                  uct_recv_desc_t *release_desc, int poll_flags)
{
    uct_ib_mlx5_srq_t *srq = &iface->rx.srq;
    uint16_t wqe_index;
    int seg_free;

    /* Need to wrap wqe_ctr, because in case of cyclic srq topology
     * it is wrapped around 0xFFFF regardless of real SRQ size.
     * But it respects srq size when srq topology is a linked-list. */
    wqe_index = wqe_ctr & srq->mask;

    if (ucs_unlikely(status != UCS_OK)) {
        uct_rc_mlx5_iface_hold_srq_desc(iface, seg, cqe, wqe_ctr, status,
                                        offset, release_desc);
    }

    if (UCT_RC_MLX5_MP_ENABLED(iface)) {
        if (--seg->srq.strides) {
            /* Segment can't be freed until all strides are consumed */
            return;
        }
        seg->srq.strides = iface->tm.mp.num_strides;
    }

    ++iface->super.rx.srq.available;

    if (poll_flags & UCT_RC_MLX5_POLL_FLAG_LINKED_LIST) {
        seg                     = uct_ib_mlx5_srq_get_wqe(srq, srq->free_idx);
        seg->srq.next_wqe_index = htons(wqe_index);
        srq->free_idx           = wqe_index;
        return;
    }

    seg_free = (seg->srq.ptr_mask == UCS_MASK(iface->tm.mp.num_strides));

    if (ucs_likely(seg_free && (wqe_ctr == (srq->ready_idx + 1)))) {
         /* If the descriptor was not used - if there are no "holes", we can just
          * reuse it on the receive queue. Otherwise, ready pointer will stay behind
          * until post_recv allocated more descriptors from the memory pool, fills
          * the holes, and moves it forward.
          */
         ucs_assert(wqe_ctr == (srq->free_idx + 1));
         ++srq->ready_idx;
         ++srq->free_idx;
         return;
    }

    if (wqe_ctr == (srq->free_idx + 1)) {
        ++srq->free_idx;
    } else {
        /* Mark the segment as out-of-order, post_recv will advance free */
        seg->srq.free = 1;
    }
}

#define uct_rc_mlx5_iface_mp_hash_lookup(_h_name, _h_ptr, _key, _last, _flags, \
                                         _iface) \
    ({ \
        uct_rc_mlx5_mp_context_t *ctx; \
        khiter_t h_it; \
        int ret; \
        h_it = kh_get(_h_name, _h_ptr, _key); \
        if (h_it == kh_end(_h_ptr)) { \
            /* No data from this sender - this must be the first fragment */ \
            *(_flags) |= UCT_CB_PARAM_FLAG_FIRST; \
            if (ucs_likely(_last)) { \
                /* fast path - single fragment message */ \
                return &(_iface)->tm.mp.last_frag_ctx; \
            } \
            h_it = kh_put(_h_name, _h_ptr, _key, &ret); \
            ucs_assert(ret != 0); \
            ctx  = &kh_value(_h_ptr, h_it); \
        } else { \
            ctx = &kh_value(_h_ptr, h_it); \
            if (_last) { \
                (_iface)->tm.mp.last_frag_ctx = *ctx; \
                kh_del(_h_name, _h_ptr, h_it); \
                return &(_iface)->tm.mp.last_frag_ctx; \
            } \
        } \
        *(_flags) |= UCT_CB_PARAM_FLAG_MORE; \
        ctx; \
    })

static UCS_F_ALWAYS_INLINE uct_rc_mlx5_mp_context_t*
uct_rc_mlx5_iface_rx_mp_context_from_ep(uct_rc_mlx5_iface_common_t *iface,
                                        struct mlx5_cqe64 *cqe, unsigned *flags)
{
    uint32_t qp_num      = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(uct_rc_iface_lookup_ep(&iface->super,
                                                                 qp_num),
                                          uct_rc_mlx5_ep_t);
    ucs_assert(ep != NULL);
    if (ep->mp.free) {
        *flags     |= UCT_CB_PARAM_FLAG_FIRST;
        ep->mp.free = 0;
    }

    if (cqe->byte_cnt & htonl(UCT_IB_MLX5_MP_RQ_LAST_MSG_FLAG)) {
        ucs_assert(!ep->mp.free);
        ep->mp.free = 1;
    } else {
        *flags |= UCT_CB_PARAM_FLAG_MORE;
    }

    return &ep->mp;
}

static UCS_F_ALWAYS_INLINE uct_rc_mlx5_mp_context_t*
uct_rc_mlx5_iface_rx_mp_context_from_hash(uct_rc_mlx5_iface_common_t *iface,
                                          struct mlx5_cqe64 *cqe,
                                          unsigned *flags)
{
#if UCS_ENABLE_ASSERT
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.super.super.md,
                                          uct_ib_mlx5_md_t);
#endif
    uct_rc_mlx5_mp_context_t *mp_ctx;
    uct_rc_mlx5_mp_hash_key_t key_gid;
    uint64_t key_lid;
    void *gid;
    int last;

    if (ucs_likely(ucs_test_all_flags(cqe->byte_cnt,
                                      htonl(UCT_IB_MLX5_MP_RQ_LAST_MSG_FLAG |
                                            UCT_IB_MLX5_MP_RQ_FIRST_MSG_FLAG)))) {
        ucs_assert(md->flags & UCT_IB_MLX5_MD_FLAG_MP_XRQ_FIRST_MSG);
        *flags |= UCT_CB_PARAM_FLAG_FIRST;
        return &iface->tm.mp.last_frag_ctx;
    }

    last = cqe->byte_cnt & htonl(UCT_IB_MLX5_MP_RQ_LAST_MSG_FLAG);

    if (uct_ib_mlx5_cqe_is_grh_present(cqe)) {
        gid            = uct_ib_mlx5_gid_from_cqe(cqe);
        /* Use guid and QP as a key. No need to fetch just qp
         * and convert to le. */
        key_gid.guid   = *(uint64_t*)UCS_PTR_BYTE_OFFSET(gid, 8);
        key_gid.qp_num = cqe->flags_rqpn;
        mp_ctx         = uct_rc_mlx5_iface_mp_hash_lookup(uct_rc_mlx5_mp_hash_gid,
                                                          &iface->tm.mp.hash_gid,
                                                          key_gid, last, flags,
                                                          iface);
    } else {
        /* Combine QP and SLID as a key. No need to fetch just qp
         * and convert to le. */
        key_lid        = (uint64_t)cqe->flags_rqpn << 32 | cqe->slid;
        mp_ctx         = uct_rc_mlx5_iface_mp_hash_lookup(uct_rc_mlx5_mp_hash_lid,
                                                          &iface->tm.mp.hash_lid,
                                                          key_lid, last, flags,
                                                          iface);
    }

    ucs_assert(mp_ctx != NULL);
    return mp_ctx;
}

static UCS_F_ALWAYS_INLINE struct mlx5_cqe64*
uct_rc_mlx5_iface_poll_rx_cq(uct_rc_mlx5_iface_common_t *iface, int poll_flags)
{
    uct_ib_mlx5_cq_t *cq = &iface->cq[UCT_IB_DIR_RX];
    struct mlx5_cqe64 *cqe;
    unsigned idx;
    uint8_t op_own;

    /* Prefetch the descriptor if it was scheduled */
    ucs_prefetch(iface->rx.pref_ptr);

    idx    = cq->cq_ci;
    cqe    = uct_ib_mlx5_get_cqe(cq, idx);
    op_own = cqe->op_own;

    if (ucs_unlikely(uct_ib_mlx5_cqe_is_hw_owned(op_own, idx, cq->cq_length))) {
        return NULL;
    } else if (ucs_unlikely(uct_ib_mlx5_cqe_is_error_or_zipped(op_own))) {
        return uct_rc_mlx5_iface_check_rx_completion(iface, cqe, poll_flags);
    }

    cq->cq_ci = idx + 1;
    return cqe; /* TODO optimize - let complier know cqe is not null */
}

static UCS_F_ALWAYS_INLINE void*
uct_rc_mlx5_iface_common_data(uct_rc_mlx5_iface_common_t *iface,
                              struct mlx5_cqe64 *cqe,
                              unsigned byte_len, unsigned *flags)
{
    uct_ib_mlx5_srq_seg_t *seg;
    uct_ib_iface_recv_desc_t *desc;
    void *hdr;

    seg  = uct_ib_mlx5_srq_get_wqe(&iface->rx.srq, ntohs(cqe->wqe_counter));
    desc = seg->srq.desc;

    /* Get a pointer to AM or Tag header (after which comes the payload)
     * Support cases of inline scatter by pointing directly to CQE.
     */
    if (cqe->op_own & MLX5_INLINE_SCATTER_32) {
        hdr = cqe;
        uct_rc_mlx5_iface_common_rx_inline(iface, desc,
                                           UCT_RC_MLX5_IFACE_STAT_RX_INL_32,
                                           byte_len);
        *flags = 0;
    } else if (cqe->op_own & MLX5_INLINE_SCATTER_64) {
        hdr = cqe - 1;
        uct_rc_mlx5_iface_common_rx_inline(iface, desc,
                                           UCT_RC_MLX5_IFACE_STAT_RX_INL_64,
                                           byte_len);
        *flags = 0;
    } else {
        hdr = uct_ib_iface_recv_desc_hdr(&iface->super.super, desc);
        VALGRIND_MAKE_MEM_DEFINED(hdr, byte_len);
        *flags = UCT_CB_PARAM_FLAG_DESC;
        /* Assuming that next packet likely will be non-inline,
         * setup the next prefetch pointer
         */
        uct_rc_mlx5_srq_prefetch_setup(iface);
    }

    return hdr;
}

static UCS_F_ALWAYS_INLINE uct_rc_mlx5_mp_context_t*
uct_rc_mlx5_iface_single_frag_context(uct_rc_mlx5_iface_common_t *iface,
                                      unsigned *flags)
{
    *flags |= UCT_CB_PARAM_FLAG_FIRST;
    return &iface->tm.mp.last_frag_ctx;
}

static UCS_F_ALWAYS_INLINE void*
uct_rc_mlx5_iface_tm_common_data(uct_rc_mlx5_iface_common_t *iface,
                                 struct mlx5_cqe64 *cqe, unsigned byte_len,
                                 unsigned *flags, int poll_flags,
                                 uct_rc_mlx5_mp_context_t **context_p)
{
    uct_ib_mlx5_srq_seg_t *seg;
    void *hdr;
    int stride_idx;

    if (!UCT_RC_MLX5_MP_ENABLED(iface)) {
        /* uct_rc_mlx5_iface_common_data will initialize flags value */
        hdr        = uct_rc_mlx5_iface_common_data(iface, cqe, byte_len, flags);
        *context_p = uct_rc_mlx5_iface_single_frag_context(iface, flags);
        return hdr;
    }

    ucs_assert(byte_len <= UCT_IB_MLX5_MP_RQ_BYTE_CNT_MASK);
    *flags = 0;

    if (ucs_test_all_flags(poll_flags, UCT_RC_MLX5_POLL_FLAG_HAS_EP |
                                       UCT_RC_MLX5_POLL_FLAG_TAG_CQE)) {
        *context_p = uct_rc_mlx5_iface_rx_mp_context_from_ep(iface, cqe, flags);
    } else if (poll_flags & UCT_RC_MLX5_POLL_FLAG_TAG_CQE) {
        *context_p = uct_rc_mlx5_iface_rx_mp_context_from_hash(iface, cqe, flags);
    } else {
        /* Non-tagged messages (AM, RNDV Fin) should always arrive in
         * a single fragment */
        *context_p = uct_rc_mlx5_iface_single_frag_context(iface, flags);
    }

    /* Get a pointer to the tag header or the payload (if it is not the first
     * fragment). */
    if (cqe->op_own & MLX5_INLINE_SCATTER_32) {
        hdr = cqe;
        uct_rc_mlx5_iface_common_rx_inline(iface, NULL,
                                           UCT_RC_MLX5_IFACE_STAT_RX_INL_32,
                                           byte_len);
    } else if (cqe->op_own & MLX5_INLINE_SCATTER_64) {
        hdr = cqe - 1;
        uct_rc_mlx5_iface_common_rx_inline(iface, NULL,
                                           UCT_RC_MLX5_IFACE_STAT_RX_INL_64,
                                           byte_len);
    } else {
        *flags    |= UCT_CB_PARAM_FLAG_DESC;
        seg        = uct_ib_mlx5_srq_get_wqe(&iface->rx.srq, ntohs(cqe->wqe_counter));
        stride_idx = uct_ib_mlx5_cqe_stride_index(cqe);
        ucs_assert(stride_idx < iface->tm.mp.num_strides);
        hdr        = (void*)be64toh(seg->dptr[stride_idx].addr);
        VALGRIND_MAKE_MEM_DEFINED(hdr, byte_len);
    }

    return hdr;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_common_am_handler(uct_rc_mlx5_iface_common_t *iface,
                                    struct mlx5_cqe64 *cqe,
                                    uct_rc_mlx5_hdr_t *hdr, unsigned flags,
                                    unsigned byte_len, int poll_flags)
{
    uint16_t wqe_ctr;
    uct_rc_iface_ops_t *rc_ops;
    uct_ib_mlx5_srq_seg_t *seg;
    uint32_t qp_num;
    ucs_status_t status;

    wqe_ctr = ntohs(cqe->wqe_counter);
    seg     = uct_ib_mlx5_srq_get_wqe(&iface->rx.srq, wqe_ctr);

    uct_ib_mlx5_log_rx(&iface->super.super, cqe, hdr,
                       uct_rc_mlx5_common_packet_dump);

    if (ucs_unlikely(hdr->rc_hdr.am_id & UCT_RC_EP_FC_MASK)) {
        qp_num = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
        rc_ops = ucs_derived_of(iface->super.super.ops, uct_rc_iface_ops_t);

        /* coverity[overrun-buffer-val] */
        status = rc_ops->fc_handler(&iface->super, qp_num, &hdr->rc_hdr,
                                    byte_len - sizeof(*hdr),
                                    cqe->imm_inval_pkey, cqe->slid, flags);
    } else {
        status = uct_iface_invoke_am(&iface->super.super.super, hdr->rc_hdr.am_id,
                                     hdr + 1, byte_len - sizeof(*hdr),
                                     flags);
    }

    uct_rc_mlx5_iface_release_srq_seg(iface, seg, cqe, wqe_ctr, status,
                                      iface->tm.am_desc.offset,
                                      &iface->tm.am_desc.super, poll_flags);
}

static UCS_F_ALWAYS_INLINE uint8_t
uct_rc_mlx5_ep_fm_cq_update(uct_rc_mlx5_iface_common_t *iface,
                            uct_ib_mlx5_txwq_t *txwq, int flag)
{
    uint8_t fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;

    fm_ce_se |= uct_rc_ep_fm(&iface->super, &txwq->fi, flag);

    return fm_ce_se;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_common_post_send(uct_rc_mlx5_iface_common_t *iface, int qp_type,
                             uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                             uint8_t opcode, uint8_t opmod, uint8_t fm_ce_se,
                             size_t wqe_size, uct_ib_mlx5_base_av_t *av,
                             struct mlx5_grh_av *grh_av, uint32_t imm, int max_log_sge,
                             uct_ib_log_sge_t *log_sge)
{
    struct mlx5_wqe_ctrl_seg *ctrl;
    uint16_t res_count;

    if (opcode != MLX5_OPCODE_NOP) {
        /* If FAILED, allow only NOP sends to be posted (used by endpoint
         * flush operations) */
        ucs_assert(!(txwq->flags & UCT_IB_MLX5_TXWQ_FLAG_FAILED));
    }

    ctrl = txwq->curr;

    if (opcode == MLX5_OPCODE_SEND_IMM) {
        uct_ib_mlx5_set_ctrl_seg_with_imm(ctrl, txwq->sw_pi, opcode, opmod,
                                          txwq->super.qp_num, fm_ce_se, wqe_size,
                                          imm);
    } else {
        uct_ib_mlx5_set_ctrl_seg(ctrl, txwq->sw_pi, opcode, opmod,
                                 txwq->super.qp_num, fm_ce_se, wqe_size);
    }

    ucs_assert(qp_type == iface->super.super.config.qp_type);

#if HAVE_TL_DC
    if (qp_type == UCT_IB_QPT_DCI) {
        uct_ib_mlx5_set_dgram_seg((void*)(ctrl + 1), av, grh_av, qp_type);
    }
#endif

    uct_ib_mlx5_log_tx(&iface->super.super, ctrl, txwq->qstart, txwq->qend,
                       max_log_sge, log_sge,
                       ((opcode == MLX5_OPCODE_SEND) || (opcode == MLX5_OPCODE_SEND_IMM)) ?
                       uct_rc_mlx5_common_packet_dump : NULL);

    res_count = uct_ib_mlx5_post_send(txwq, ctrl, wqe_size, 1);
    if (fm_ce_se & MLX5_WQE_CTRL_CQ_UPDATE) {
        txwq->sig_pi = txwq->prev_sw_pi;
    }

#if HAVE_TL_DC
    if (qp_type == UCT_IB_QPT_DCI) {
        txqp->available -= res_count;
        return;
    }
#endif

    uct_rc_txqp_posted(txqp, &iface->super, res_count, fm_ce_se & MLX5_WQE_CTRL_CQ_UPDATE);
}

static UCS_F_ALWAYS_INLINE void uct_rc_mlx5_txqp_inline_iov_post(
        uct_rc_mlx5_iface_common_t *iface, int qp_type, uct_rc_txqp_t *txqp,
        uct_ib_mlx5_txwq_t *txwq, const uct_iov_t *iov, size_t iovcnt,
        size_t iov_length, uint8_t am_id, uct_ib_mlx5_base_av_t *av,
        struct mlx5_grh_av *grh_av, size_t av_size)
{
    struct mlx5_wqe_ctrl_seg *ctrl = txwq->curr;
    struct mlx5_wqe_inl_data_seg *inl;
    uct_rc_mlx5_hdr_t *rch;
    size_t wqe_size, ctrl_av_size;
    unsigned fm_ce_se;

    ctrl_av_size    = sizeof(*ctrl) + av_size;
    wqe_size        = ctrl_av_size + sizeof(*inl) + sizeof(*rch) + iov_length;
    inl             = UCS_PTR_BYTE_OFFSET(ctrl, ctrl_av_size);
    inl             = uct_ib_mlx5_txwq_wrap_exact(txwq, inl);
    inl->byte_count = htonl((iov_length + sizeof(*rch)) | MLX5_INLINE_SEG);
    rch             = (uct_rc_mlx5_hdr_t*)(inl + 1);
    fm_ce_se        = MLX5_WQE_CTRL_SOLICITED | uct_rc_iface_tx_moderation(
            &iface->super, txqp, MLX5_WQE_CTRL_CQ_UPDATE);

    uct_rc_mlx5_am_hdr_fill(rch, am_id);
    uct_ib_mlx5_inline_iov_copy(rch + 1, iov, iovcnt, iov_length, txwq);
    uct_rc_mlx5_common_post_send(iface, qp_type, txqp, txwq, MLX5_OPCODE_SEND,
                                 0, fm_ce_se, wqe_size, av, grh_av, 0, INT_MAX,
                                 NULL);
}

/*
 * Generic function that setups and posts WQE with inline segment
 * Parameters which are not relevant to the opcode are ignored.
 *
 *            +--------+-----+-------+--------+------------
 * SEND       | CTRL   | INL | am_id | am_hdr | payload ...
 *            +--------+-----+---+---+-+-------+-----------
 * RDMA_WRITE | CTRL   | RADDR   | INL | payload ...
 *            +--------+---------+-----+-------------------
 *
 * CTRL is mlx5_wqe_ctrl_seg for RC and
 *         mlx5_wqe_ctrl_seg + mlx5_wqe_datagram_seg for DC
 *
 * NOTE: switch is optimized away during inlining because opcode
 * is a compile time constant
 */
static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_txqp_inline_post(uct_rc_mlx5_iface_common_t *iface, int qp_type,
                             uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                             unsigned opcode, const void *buffer, unsigned length,
                  /* SEND */ uint8_t am_id, uint64_t am_hdr, uint32_t imm_val_be,
                  /* RDMA */ uint64_t rdma_raddr, uct_rkey_t rdma_rkey,
                  /* AV   */ uct_ib_mlx5_base_av_t *av, struct mlx5_grh_av *grh_av,
                             size_t av_size, unsigned fm_ce_se, int max_log_sge)
{
    struct mlx5_wqe_ctrl_seg     *ctrl;
    struct mlx5_wqe_raddr_seg    *raddr;
    struct mlx5_wqe_inl_data_seg *inl;
    uct_rc_mlx5_am_short_hdr_t   *am;
    uct_rc_mlx5_hdr_t            *rc_hdr;
    size_t wqe_size, ctrl_av_size;
    void *next_seg;

    ctrl         = txwq->curr;
    ctrl_av_size = sizeof(*ctrl) + av_size;
    next_seg     = UCS_PTR_BYTE_OFFSET(ctrl, ctrl_av_size);
    next_seg     = uct_ib_mlx5_txwq_wrap_exact(txwq, next_seg);

    switch (opcode) {
    case MLX5_OPCODE_SEND_IMM:
        /* Fall through to MLX5_OPCODE_SEND handler */
    case MLX5_OPCODE_SEND:
        /* Set inline segment which has AM id, AM header, and AM payload */
        wqe_size         = ctrl_av_size + sizeof(*inl) + sizeof(*am) + length;
        inl              = next_seg;
        inl->byte_count  = htonl((length + sizeof(*am)) | MLX5_INLINE_SEG);
        am               = (void*)(inl + 1);
        am->am_hdr       = am_hdr;
        uct_rc_mlx5_am_hdr_fill(&am->rc_hdr, am_id);
        uct_ib_mlx5_inline_copy(am + 1, buffer, length, txwq);
        fm_ce_se        |= uct_rc_iface_tx_moderation(&iface->super, txqp, MLX5_WQE_CTRL_CQ_UPDATE);
        break;

    case MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_RAW:
        /* Send empty AM with just AM id (used by FC) */
        wqe_size         = ctrl_av_size + sizeof(*inl) + sizeof(*rc_hdr);
        inl              = next_seg;
        inl->byte_count  = htonl(sizeof(*rc_hdr) | MLX5_INLINE_SEG);
        rc_hdr           = (void*)(inl + 1);
        uct_rc_mlx5_am_hdr_fill(rc_hdr, am_id);
        fm_ce_se        |= uct_rc_iface_tx_moderation(&iface->super, txqp, MLX5_WQE_CTRL_CQ_UPDATE);
        break;

    case MLX5_OPCODE_RDMA_WRITE:
        /* Set RDMA segment */
        if (length == 0) {
            wqe_size     = ctrl_av_size + sizeof(*raddr);
        } else {
            wqe_size     = ctrl_av_size + sizeof(*raddr) + sizeof(*inl) + length;
        }
        raddr            = next_seg;
        uct_ib_mlx5_ep_set_rdma_seg(raddr, rdma_raddr, rdma_rkey);
        inl              = uct_ib_mlx5_txwq_wrap_none(txwq, raddr + 1);
        inl->byte_count  = htonl(length | MLX5_INLINE_SEG);
        uct_ib_mlx5_inline_copy(inl + 1, buffer, length, txwq);
        fm_ce_se        |= uct_rc_iface_tx_moderation(&iface->super, txqp, MLX5_WQE_CTRL_CQ_UPDATE);
        break;

    case MLX5_OPCODE_NOP:
        /* Empty inline segment */
        wqe_size         = sizeof(*ctrl) + av_size;
        inl              = next_seg;
        inl->byte_count  = htonl(MLX5_INLINE_SEG);
        fm_ce_se        |= MLX5_WQE_CTRL_CQ_UPDATE | MLX5_WQE_CTRL_FENCE;
        break;

    default:
        ucs_fatal("invalid send opcode");
    }

    uct_rc_mlx5_common_post_send(iface, qp_type, txqp, txwq, opcode, 0, fm_ce_se,
                                 wqe_size, av, grh_av, imm_val_be, max_log_sge, NULL);
}

/*
 * Generic data-pointer posting function.
 * Parameters which are not relevant to the opcode are ignored.
 *
 *            +--------+-----+-------+
 * SEND       | CTRL   | INL | DPSEG |
 *            +--------+-----+---+---+----+
 * RDMA_WRITE | CTRL   | RADDR   | DPSEG  |
 *            +--------+---------+--------+-------+
 * ATOMIC     | CTRL   | RADDR   | ATOMIC | DPSEG |
 *            +--------+---------+--------+-------+
 *
 * CTRL is mlx5_wqe_ctrl_seg for RC and
 *         mlx5_wqe_ctrl_seg + mlx5_wqe_datagram_seg for DC
 *
 * NOTE: switch is optimized away during inlining because opcode_flags
 * is a compile time constant
 */
static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_txqp_dptr_post(uct_rc_mlx5_iface_common_t *iface, int qp_type,
                           uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                           unsigned opcode_flags, const void *buffer,
                           unsigned length, uint32_t *lkey_p,
         /* RDMA/ATOMIC */ uint64_t remote_addr, uct_rkey_t rkey,
         /* ATOMIC      */ uint64_t compare_mask, uint64_t compare,
         /* ATOMIC      */ uint64_t swap_mask, uint64_t swap_add,
         /* AV          */ uct_ib_mlx5_base_av_t *av, struct mlx5_grh_av *grh_av,
                           size_t av_size, uint8_t fm_ce_se, uint32_t imm_val_be,
                           int max_log_sge, uct_ib_log_sge_t *log_sge)
{
    struct mlx5_wqe_ctrl_seg                     *ctrl;
    struct mlx5_wqe_raddr_seg                    *raddr;
    struct mlx5_wqe_atomic_seg                   *atomic;
    struct mlx5_wqe_data_seg                     *dptr;
    struct uct_ib_mlx5_atomic_masked_cswap32_seg *masked_cswap32;
    struct uct_ib_mlx5_atomic_masked_fadd32_seg  *masked_fadd32;
    struct uct_ib_mlx5_atomic_masked_cswap64_seg *masked_cswap64;
    struct uct_ib_mlx5_atomic_masked_fadd64_seg  *masked_fadd64;
    size_t  wqe_size, ctrl_av_size;
    uint8_t opmod;
    void *next_seg;

    if (!(fm_ce_se & MLX5_WQE_CTRL_CQ_UPDATE)) {
        fm_ce_se |= uct_rc_iface_tx_moderation(&iface->super, txqp, MLX5_WQE_CTRL_CQ_UPDATE);
    }

    opmod        = 0;
    ctrl         = txwq->curr;
    ctrl_av_size = sizeof(*ctrl) + av_size;
    next_seg     = UCS_PTR_BYTE_OFFSET(ctrl, ctrl_av_size);
    next_seg     = uct_ib_mlx5_txwq_wrap_exact(txwq, next_seg);

    switch (opcode_flags) {
    case MLX5_OPCODE_SEND_IMM: /* Used by tag offload */
    case MLX5_OPCODE_SEND:
        /* Data segment only */
        ucs_assert(length < (2ul << 30));

        /* TODO: make proper check for all cases TM, MP, etc
         * ucs_assert(length <= iface->super.super.config.seg_size); */

        wqe_size = ctrl_av_size + sizeof(struct mlx5_wqe_data_seg);
        uct_ib_mlx5_set_data_seg(next_seg, buffer, length, *lkey_p);
        break;

    case MLX5_OPCODE_RDMA_READ:
        fm_ce_se |= MLX5_WQE_CTRL_CQ_UPDATE;
        /* Fall through */
    case MLX5_OPCODE_RDMA_WRITE:
        /* Set RDMA segment */
        ucs_assert(length <= UCT_IB_MAX_MESSAGE_SIZE);

        raddr = next_seg;
        uct_ib_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        /* Data segment */
        if (length == 0) {
            wqe_size     = ctrl_av_size + sizeof(*raddr);
        } else {
            /* dptr cannot wrap, because ctrl+av could be either 2 or 4 segs */
            dptr         = uct_ib_mlx5_txwq_wrap_none(txwq, raddr + 1);
            wqe_size     = ctrl_av_size + sizeof(*raddr) + sizeof(*dptr);
            uct_ib_mlx5_set_data_seg(dptr, buffer, length, *lkey_p);
        }
        break;

    case MLX5_OPCODE_ATOMIC_FA:
    case MLX5_OPCODE_ATOMIC_CS:
        fm_ce_se |= uct_rc_mlx5_ep_fm_cq_update(iface, txwq,
                                                iface->config.atomic_fence_flag);
        ucs_assert(length == sizeof(uint64_t));
        raddr = next_seg;
        uct_ib_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        /* atomic cannot wrap, because ctrl+av could be either 2 or 4 segs */
        atomic = uct_ib_mlx5_txwq_wrap_none(txwq, raddr + 1);
        if (opcode_flags == MLX5_OPCODE_ATOMIC_CS) {
            atomic->compare = compare;
        }
        atomic->swap_add    = swap_add;

        dptr                = uct_ib_mlx5_txwq_wrap_exact(txwq, atomic + 1);
        uct_ib_mlx5_set_data_seg(dptr, buffer, length, *lkey_p);
        wqe_size            = ctrl_av_size + sizeof(*raddr) + sizeof(*atomic) +
                              sizeof(*dptr);
        break;

    case MLX5_OPCODE_ATOMIC_MASKED_CS:
        fm_ce_se |= uct_rc_mlx5_ep_fm_cq_update(iface, txwq,
                                                iface->config.atomic_fence_flag);
        raddr     = next_seg;
        uct_ib_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        switch (length) {
        case sizeof(uint32_t):
            opmod                        = UCT_IB_MLX5_OPMOD_EXT_ATOMIC(2);
            masked_cswap32               = uct_ib_mlx5_txwq_wrap_none(txwq, raddr + 1);
            masked_cswap32->swap         = swap_add;
            masked_cswap32->compare      = compare;
            masked_cswap32->swap_mask    = swap_mask;
            masked_cswap32->compare_mask = compare_mask;
            dptr                         = uct_ib_mlx5_txwq_wrap_exact(txwq, masked_cswap32 + 1);
            wqe_size                     = ctrl_av_size + sizeof(*raddr) +
                                           sizeof(*masked_cswap32) + sizeof(*dptr);
            break;
        case sizeof(uint64_t):
            opmod                        = UCT_IB_MLX5_OPMOD_EXT_ATOMIC(3); /* Ext. atomic, size 2**3 */
            masked_cswap64               = uct_ib_mlx5_txwq_wrap_none(txwq, raddr + 1);
            masked_cswap64->swap         = swap_add;
            masked_cswap64->compare      = compare;

            /* 2nd half of masked_cswap64 can wrap */
            masked_cswap64               = uct_ib_mlx5_txwq_wrap_exact(txwq, masked_cswap64 + 1);
            masked_cswap64->swap         = swap_mask;
            masked_cswap64->compare      = compare_mask;

            dptr                         = uct_ib_mlx5_txwq_wrap_exact(txwq, masked_cswap64 + 1);
            wqe_size                     = ctrl_av_size + sizeof(*raddr) +
                                           2 * sizeof(*masked_cswap64) + sizeof(*dptr);
            break;
        default:
            ucs_fatal("invalid atomic type length %d", length);
        }
        uct_ib_mlx5_set_data_seg(dptr, buffer, length, *lkey_p);
        break;

     case MLX5_OPCODE_ATOMIC_MASKED_FA:
        fm_ce_se |= uct_rc_mlx5_ep_fm_cq_update(iface, txwq, iface->config.atomic_fence_flag);
        raddr     = next_seg;
        uct_ib_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        switch (length) {
        case sizeof(uint32_t):
            opmod                         = UCT_IB_MLX5_OPMOD_EXT_ATOMIC(2);
            masked_fadd32                 = uct_ib_mlx5_txwq_wrap_none(txwq, raddr + 1);
            masked_fadd32->add            = swap_add;
            masked_fadd32->filed_boundary = compare;

            dptr                          = uct_ib_mlx5_txwq_wrap_exact(txwq, masked_fadd32 + 1);
            wqe_size                      = ctrl_av_size + sizeof(*raddr) +
                                            sizeof(*masked_fadd32) + sizeof(*dptr);
            break;
        case sizeof(uint64_t):
            opmod                         = UCT_IB_MLX5_OPMOD_EXT_ATOMIC(3); /* Ext. atomic, size 2**3 */
            masked_fadd64                 = uct_ib_mlx5_txwq_wrap_none(txwq, raddr + 1);
            masked_fadd64->add            = swap_add;
            masked_fadd64->filed_boundary = compare;

            dptr                          = uct_ib_mlx5_txwq_wrap_exact(txwq, masked_fadd64 + 1);
            wqe_size                      = ctrl_av_size + sizeof(*raddr) +
                                            sizeof(*masked_fadd64) + sizeof(*dptr);
            break;
        default:
            ucs_fatal("invalid atomic type length %d", length);
        }
        uct_ib_mlx5_set_data_seg(dptr, buffer, length, *lkey_p);
        break;

    default:
        ucs_fatal("invalid send opcode");
    }

    uct_rc_mlx5_common_post_send(iface, qp_type, txqp, txwq,
                                 (opcode_flags & UCT_RC_MLX5_OPCODE_MASK), opmod,
                                 fm_ce_se, wqe_size, av, grh_av, imm_val_be,
                                 max_log_sge, log_sge);
}

static UCS_F_ALWAYS_INLINE
void uct_rc_mlx5_txqp_dptr_post_iov(uct_rc_mlx5_iface_common_t *iface, int qp_type,
                                    uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                                    unsigned opcode_flags,
                         /* IOV  */ const uct_iov_t *iov, size_t iovcnt,
                         /* SEND */ uint8_t am_id, const void *am_hdr, unsigned am_hdr_len,
                         /* RDMA */ uint64_t remote_addr, uct_rkey_t rkey,
                         /* TAG  */ uct_tag_t tag, uint32_t app_ctx, uint32_t ib_imm_be,
                         /* AV   */ uct_ib_mlx5_base_av_t *av, struct mlx5_grh_av *grh_av,
                                    size_t av_size, uint8_t fm_ce_se, int max_log_sge)
{
    struct mlx5_wqe_ctrl_seg     *ctrl;
    struct mlx5_wqe_raddr_seg    *raddr;
    struct mlx5_wqe_data_seg     *dptr;
    struct mlx5_wqe_inl_data_seg *inl;
    uct_rc_mlx5_hdr_t            *rch;
    unsigned                      wqe_size, inl_seg_size, ctrl_av_size;
    void                         *next_seg;

    if (!(fm_ce_se & MLX5_WQE_CTRL_CQ_UPDATE)) {
        fm_ce_se |= uct_rc_iface_tx_moderation(&iface->super, txqp, MLX5_WQE_CTRL_CQ_UPDATE);
    }

    ctrl         = txwq->curr;
    ctrl_av_size = sizeof(*ctrl) + av_size;
    next_seg     = UCS_PTR_BYTE_OFFSET(ctrl, ctrl_av_size);
    next_seg     = uct_ib_mlx5_txwq_wrap_exact(txwq, next_seg);

    switch (opcode_flags) {
    case MLX5_OPCODE_SEND:
        inl_seg_size     = ucs_align_up_pow2(sizeof(*inl) + sizeof(*rch) + am_hdr_len,
                                             UCT_IB_MLX5_WQE_SEG_SIZE);

        ucs_assert(uct_iov_total_length(iov, iovcnt) + sizeof(*rch) + am_hdr_len <=
                   iface->super.super.config.seg_size);

        /* Inline segment with AM ID and header */
        inl              = next_seg;
        inl->byte_count  = htonl((sizeof(*rch) + am_hdr_len) | MLX5_INLINE_SEG);
        rch              = (uct_rc_mlx5_hdr_t *)(inl + 1);

        uct_rc_mlx5_am_hdr_fill(rch, am_id);
        uct_ib_mlx5_inline_copy(rch + 1, am_hdr, am_hdr_len, txwq);

        /* Data segment with payload */
        dptr             = (struct mlx5_wqe_data_seg *)((char *)inl + inl_seg_size);
        wqe_size         = ctrl_av_size + inl_seg_size +
                           uct_ib_mlx5_set_data_seg_iov(txwq, dptr, iov, iovcnt);

        ucs_assert(wqe_size <= UCT_IB_MLX5_MAX_SEND_WQE_SIZE);
        break;

#if IBV_HW_TM
    case MLX5_OPCODE_SEND|UCT_RC_MLX5_OPCODE_FLAG_TM:
    case MLX5_OPCODE_SEND_IMM|UCT_RC_MLX5_OPCODE_FLAG_TM:
        inl_seg_size     = ucs_align_up_pow2(sizeof(*inl) + sizeof(struct ibv_tmh),
                                             UCT_IB_MLX5_WQE_SEG_SIZE);
        inl              = next_seg;
        inl->byte_count  = htonl(sizeof(struct ibv_tmh) | MLX5_INLINE_SEG);
        dptr             = uct_ib_mlx5_txwq_wrap_exact(txwq, (char *)inl + inl_seg_size);
        wqe_size         = ctrl_av_size + inl_seg_size +
                           uct_ib_mlx5_set_data_seg_iov(txwq, dptr, iov, iovcnt);

        uct_rc_mlx5_fill_tmh((struct ibv_tmh*)(inl + 1), tag, app_ctx,
                             IBV_TMH_EAGER);
        ucs_assert(wqe_size <= UCT_IB_MLX5_MAX_SEND_WQE_SIZE);
        break;
#endif

    case MLX5_OPCODE_RDMA_READ:
        fm_ce_se |= MLX5_WQE_CTRL_CQ_UPDATE;
        /* Fall through */
    case MLX5_OPCODE_RDMA_WRITE:
        /* Set RDMA segment */
        ucs_assert(uct_iov_total_length(iov, iovcnt) <= UCT_IB_MAX_MESSAGE_SIZE);

        raddr            = next_seg;
        uct_ib_mlx5_ep_set_rdma_seg(raddr, remote_addr, rkey);

        /* Data segment */
        wqe_size         = ctrl_av_size + sizeof(*raddr) +
                           uct_ib_mlx5_set_data_seg_iov(txwq, (void*)(raddr + 1),
                                                        iov, iovcnt);
        break;

    default:
        ucs_fatal("invalid send opcode");
    }

    uct_rc_mlx5_common_post_send(iface, qp_type, txqp, txwq,
                                 opcode_flags & UCT_RC_MLX5_OPCODE_MASK,
                                 0, fm_ce_se, wqe_size, av, grh_av, ib_imm_be,
                                 max_log_sge, NULL);
}

/*
 * Helper function for buffer-copy post.
 * Adds the descriptor to the callback queue.
 */
static UCS_F_ALWAYS_INLINE void uct_rc_mlx5_common_txqp_bcopy_post(
        uct_rc_mlx5_iface_common_t *iface, int qp_type, uct_rc_txqp_t *txqp,
        uct_ib_mlx5_txwq_t *txwq, unsigned opcode, unsigned length,
        uint64_t rdma_raddr, uct_rkey_t rdma_rkey, uct_ib_mlx5_base_av_t *av,
        struct mlx5_grh_av *grh_av, size_t av_size, uint8_t fm_ce_se,
        uint32_t imm_val_be, uct_rc_iface_send_desc_t *desc, const void *buffer,
        uct_ib_log_sge_t *log_sge)
{
    desc->super.sn = txwq->sw_pi;
    uct_rc_mlx5_txqp_dptr_post(iface, qp_type, txqp, txwq, opcode, buffer,
                               length, &desc->lkey, rdma_raddr, rdma_rkey, 0, 0,
                               0, 0, av, grh_av, av_size, fm_ce_se, imm_val_be,
                               INT_MAX, log_sge);
    uct_rc_txqp_add_send_op(txqp, &desc->super);
}

#if IBV_HW_TM
static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_set_tm_seg(uct_ib_mlx5_txwq_t *txwq,
                       uct_rc_mlx5_wqe_tm_seg_t *tmseg, int op, int tag_index,
                       uint32_t unexp_cnt, uint64_t tag, uint64_t mask,
                       unsigned tm_flags)
{
    tmseg->sw_cnt = htons(unexp_cnt);
    tmseg->opcode = op << 4;
    tmseg->flags  = tm_flags;

    if (op == UCT_RC_MLX5_TM_OPCODE_NOP) {
        return;
    }

    tmseg->index = htons(tag_index);

    if (op == UCT_RC_MLX5_TM_OPCODE_REMOVE) {
        return;
    }

    tmseg->append_tag  = tag;
    tmseg->append_mask = mask;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_release_tag_entry(uct_rc_mlx5_iface_common_t *iface,
                              uct_rc_mlx5_tag_entry_t *tag)
{
    if (!--tag->num_cqes) {
        tag->next            = NULL;
        iface->tm.tail->next = tag;
        iface->tm.tail       = tag;
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_add_cmd_wq_op(uct_rc_mlx5_iface_common_t *iface,
                          uct_rc_mlx5_tag_entry_t *tag)
{
    uct_rc_mlx5_srq_op_t *op;

    op      = iface->tm.cmd_wq.ops +
              (iface->tm.cmd_wq.ops_tail++ & iface->tm.cmd_wq.ops_mask);
    op->tag = tag;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_txqp_tag_inline_post(uct_rc_mlx5_iface_common_t *iface, int qp_type,
                                 uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
                                 unsigned opcode, const void *buffer, unsigned length,
                                 const uct_iov_t *iov, /* relevant for RNDV */
                                 uct_tag_t tag, uint32_t app_ctx, int tm_op,
                                 uint32_t imm_val_be, uct_ib_mlx5_base_av_t *av,
                                 struct mlx5_grh_av *grh_av, size_t av_size,
                                 void *ravh, size_t ravh_len, unsigned fm_ce_se)
{
    struct mlx5_wqe_ctrl_seg     *ctrl;
    struct mlx5_wqe_inl_data_seg *inl;
    size_t wqe_size, ctrl_av_size;
    struct ibv_tmh *tmh;
    struct ibv_rvh rvh;
    unsigned tmh_data_len;
    size_t tm_hdr_len;
    void UCS_V_UNUSED *ravh_ptr;
    void *data;

    ctrl         = txwq->curr;
    ctrl_av_size = sizeof(*ctrl) + av_size; /* can be 16, 32 or 64 bytes */
    inl          = uct_ib_mlx5_txwq_wrap_exact(txwq, (char*)ctrl + ctrl_av_size);
    tmh          = (struct ibv_tmh*)(inl + 1);

    ucs_assert((opcode == MLX5_OPCODE_SEND_IMM) || (opcode == MLX5_OPCODE_SEND));

    switch (tm_op) {
    case IBV_TMH_EAGER:
        wqe_size         = ctrl_av_size + sizeof(*inl) + sizeof(*tmh) + length;
        inl->byte_count  = htonl((length + sizeof(*tmh)) | MLX5_INLINE_SEG);
        data             = tmh + 1;
        tmh_data_len     = 0;
        break;

    case IBV_TMH_RNDV:
        ucs_assert(iov != NULL);
        /* RVH can be wrapped */
        /* cppcheck-suppress nullPointer */
        uct_rc_mlx5_fill_rvh(&rvh, iov->buffer,
        /* cppcheck-suppress nullPointer */
                              ((uct_ib_mem_t*)iov->memh)->rkey, iov->length);
        uct_ib_mlx5_inline_copy(tmh + 1, &rvh, sizeof(rvh), txwq);

        tm_hdr_len = sizeof(*tmh) + sizeof(rvh);
#if HAVE_TL_DC
        if (qp_type == UCT_IB_QPT_DCI) {
            /* RAVH can be wrapped as well */
            ravh_ptr = uct_ib_mlx5_txwq_wrap_data(txwq, (char*)tmh +
                                                  sizeof(*tmh) + sizeof(rvh));
            uct_ib_mlx5_inline_copy(ravh_ptr, ravh, ravh_len, txwq);
            tm_hdr_len += ravh_len;
        }
#endif

        tmh_data_len    = uct_rc_mlx5_fill_tmh_priv_data(tmh, buffer, length,
                                                         iface->tm.max_rndv_data);
        length         -= tmh_data_len; /* Note: change length func parameter */
        wqe_size        = ctrl_av_size + sizeof(*inl) + tm_hdr_len + length;
        inl->byte_count = htonl((length + tm_hdr_len) | MLX5_INLINE_SEG);
        data            = uct_ib_mlx5_txwq_wrap_data(txwq, (char*)tmh + tm_hdr_len);

        break;

    default:
        ucs_fatal("Invalid tag opcode: %d", tm_op);
        break;
    }

    ucs_assert(wqe_size <= UCT_IB_MLX5_MAX_SEND_WQE_SIZE);

    uct_rc_mlx5_fill_tmh(tmh, tag, app_ctx, tm_op);

    /* In case of RNDV first bytes of data could be stored in TMH */
    uct_ib_mlx5_inline_copy(data, (char*)buffer + tmh_data_len, length, txwq);
    fm_ce_se |= uct_rc_iface_tx_moderation(&iface->super, txqp, MLX5_WQE_CTRL_CQ_UPDATE);

    uct_rc_mlx5_common_post_send(iface, qp_type, txqp, txwq, opcode, 0, fm_ce_se,
                                 wqe_size, av, grh_av, imm_val_be, INT_MAX, NULL);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_common_post_srq_op(uct_rc_mlx5_cmd_wq_t *cmd_wq,
                                     unsigned extra_wqe_size, unsigned op_code,
                                     uint16_t next_idx, unsigned unexp_cnt,
                                     uct_tag_t tag, uct_tag_t tag_mask,
                                     unsigned tm_flags)
{
    uct_ib_mlx5_txwq_t       *txwq = &cmd_wq->super;
    struct mlx5_wqe_ctrl_seg *ctrl = txwq->curr; /* 16 bytes */
    uct_rc_mlx5_wqe_tm_seg_t *tm;                /* 32 bytes */
    unsigned                  wqe_size;

    wqe_size = sizeof(*ctrl) + sizeof(*tm) + extra_wqe_size;

    tm = uct_ib_mlx5_txwq_wrap_none(txwq, ctrl + 1);

    uct_ib_mlx5_set_ctrl_seg(ctrl, txwq->sw_pi, UCT_RC_MLX5_OPCODE_TAG_MATCHING,
                             0, txwq->super.qp_num, 0, wqe_size);

    uct_rc_mlx5_set_tm_seg(txwq, tm, op_code, next_idx, unexp_cnt,
                           tag, tag_mask, tm_flags);

    uct_ib_mlx5_post_send(txwq, ctrl, wqe_size, 0);
}


static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_iface_common_tag_recv(uct_rc_mlx5_iface_common_t *iface,
                                  uct_tag_t tag,
                                  uct_tag_t tag_mask, const uct_iov_t *iov,
                                  size_t iovcnt, uct_tag_context_t *ctx)
{
    uct_rc_mlx5_ctx_priv_t   *priv = uct_rc_mlx5_ctx_priv(ctx);
    uct_ib_mlx5_txwq_t       *txwq = &iface->tm.cmd_wq.super;
    struct mlx5_wqe_data_seg *dptr; /* 16 bytes */
    uct_rc_mlx5_tag_entry_t  *tag_entry;
    uint16_t                 next_idx;
    unsigned                 ctrl_size;
    int                      ret;

    UCT_CHECK_IOV_SIZE(iovcnt, 1ul, "uct_rc_mlx5_iface_common_tag_recv");
    UCT_RC_MLX5_CHECK_TAG(iface);

    kh_put(uct_rc_mlx5_tag_addrs, &iface->tm.tag_addrs, iov->buffer, &ret);
    if (ucs_unlikely(ret == 0)) {
        /* Do not post the same buffer more than once (even with different tags)
         * to avoid memory corruption. */
        return UCS_ERR_ALREADY_EXISTS;
    }
    ucs_assert(ret > 0);

    ctrl_size           = sizeof(struct mlx5_wqe_ctrl_seg) +
                          sizeof(uct_rc_mlx5_wqe_tm_seg_t);
    tag_entry           = iface->tm.head;
    next_idx            = tag_entry->next - iface->tm.list;
    iface->tm.head      = tag_entry->next;
    tag_entry->next     = NULL;
    tag_entry->ctx      = ctx;
    tag_entry->num_cqes = 2; /* ADD and MSG_ARRIVED/CANCELED */

    /* Save aux data (which will be needed in the following ops) in the context */
    priv->tag_handle   = tag_entry - iface->tm.list;
    priv->tag          = tag;
    priv->buffer       = iov->buffer; /* Only one iov is supported so far */
    priv->length       = iov->length;

    uct_rc_mlx5_add_cmd_wq_op(iface, tag_entry);

    dptr = uct_ib_mlx5_txwq_wrap_none(txwq, (char*)txwq->curr + ctrl_size);
    uct_ib_mlx5_set_data_seg(dptr, iov->buffer, iov->length,
                             uct_ib_memh_get_lkey(iov->memh));

    uct_rc_mlx5_iface_common_post_srq_op(&iface->tm.cmd_wq, sizeof(*dptr),
                                         UCT_RC_MLX5_TM_OPCODE_APPEND, next_idx,
                                         iface->tm.unexpected_cnt, tag,
                                         tag_mask,
                                         UCT_RC_MLX5_SRQ_FLAG_TM_CQE_REQ |
                                         UCT_RC_MLX5_SRQ_FLAG_TM_SW_CNT);

    UCT_RC_MLX5_TM_STAT(iface, LIST_ADD);

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_tag_del_from_hash(uct_rc_mlx5_iface_common_t *iface,
                                    void *buffer)
{
    khiter_t iter;

    iter = kh_get(uct_rc_mlx5_tag_addrs, &iface->tm.tag_addrs, buffer);
    ucs_assert(iter != kh_end(&iface->tm.tag_addrs));
    kh_del(uct_rc_mlx5_tag_addrs, &iface->tm.tag_addrs, iter);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_iface_common_tag_recv_cancel(uct_rc_mlx5_iface_common_t *iface,
                                         uct_tag_context_t *ctx, int force)
{
    uct_rc_mlx5_ctx_priv_t   *priv = uct_rc_mlx5_ctx_priv(ctx);
    uint16_t                 idx   = priv->tag_handle;
    uct_rc_mlx5_tag_entry_t  *tag_entry;
    unsigned flags;

    tag_entry = &iface->tm.list[idx];

    if (ucs_likely(force)) {
        flags = UCT_RC_MLX5_SRQ_FLAG_TM_SW_CNT;
        uct_rc_mlx5_release_tag_entry(iface, tag_entry);
        uct_rc_mlx5_iface_tag_del_from_hash(iface, priv->buffer);
    } else {
        flags = UCT_RC_MLX5_SRQ_FLAG_TM_CQE_REQ | UCT_RC_MLX5_SRQ_FLAG_TM_SW_CNT;
        uct_rc_mlx5_add_cmd_wq_op(iface, tag_entry);
    }

    uct_rc_mlx5_iface_common_post_srq_op(&iface->tm.cmd_wq, 0,
                                         UCT_RC_MLX5_TM_OPCODE_REMOVE, idx,
                                         iface->tm.unexpected_cnt, 0ul, 0ul,
                                         flags);

    UCT_RC_MLX5_TM_STAT(iface, LIST_DEL);

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_handle_tm_list_op(uct_rc_mlx5_iface_common_t *iface, int opcode)
{
    uct_rc_mlx5_cmd_wq_t *cmd_wq;
    uct_rc_mlx5_srq_op_t *op;
    uct_tag_context_t *ctx;
    uct_rc_mlx5_ctx_priv_t *priv;

    cmd_wq = &iface->tm.cmd_wq;
    op     = cmd_wq->ops + (cmd_wq->ops_head++ & cmd_wq->ops_mask);
    uct_rc_mlx5_release_tag_entry(iface, op->tag);

    if (opcode == UCT_RC_MLX5_CQE_APP_OP_TM_REMOVE) {
        ctx  = op->tag->ctx;
        priv = uct_rc_mlx5_ctx_priv(ctx);
        uct_rc_mlx5_iface_tag_del_from_hash(iface, priv->buffer);
        ctx->completed_cb(ctx, priv->tag, 0, priv->length, NULL, UCS_ERR_CANCELED);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_tag_consumed(uct_rc_mlx5_iface_common_t *iface,
                               struct mlx5_cqe64 *cqe, int opcode)
{
    struct ibv_tmh *tmh = (struct ibv_tmh*)cqe;
    uct_rc_mlx5_tag_entry_t *tag;
    uct_tag_context_t *ctx;
    uct_rc_mlx5_ctx_priv_t *priv;

    /* coverity[tainted_data] */
    tag = &iface->tm.list[ntohs(cqe->app_info)];
    ctx = tag->ctx;
    ctx->tag_consumed_cb(ctx);

    if (opcode == UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED) {
        /* Need to save TMH info, which will be used when
         * UCT_RC_MLX5_CQE_APP_OP_TM_EXPECTED CQE is received */
        priv           = uct_rc_mlx5_ctx_priv(ctx);
        priv->tag      = tmh->tag;
        priv->app_ctx  = tmh->app_ctx;
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_handle_expected(uct_rc_mlx5_iface_common_t *iface, struct mlx5_cqe64 *cqe,
                                  uint64_t tag, uint32_t app_ctx)
{
    int is_inline = cqe->op_own & MLX5_INLINE_SCATTER_64;
    uint64_t imm_data;
    uct_rc_mlx5_tag_entry_t *tag_entry;
    uct_tag_context_t *ctx;
    uct_rc_mlx5_ctx_priv_t *priv;
    unsigned byte_len;

    /* coverity[tainted_data] */
    tag_entry = &iface->tm.list[ntohs(cqe->app_info)];
    ctx       = tag_entry->ctx;
    priv      = uct_rc_mlx5_ctx_priv(tag_entry->ctx);
    /* Tag expected CQEs use all bits of byte_cnt even if MP XRQ is configured */
    byte_len  = ntohl(cqe->byte_cnt);

    uct_rc_mlx5_release_tag_entry(iface, tag_entry);
    uct_rc_mlx5_iface_tag_del_from_hash(iface, priv->buffer);

    if (is_inline) {
        ucs_assert(byte_len <= priv->length);
    } else {
        VALGRIND_MAKE_MEM_DEFINED(priv->buffer, byte_len);
    }

    imm_data = uct_rc_mlx5_tag_imm_data_unpack(cqe->imm_inval_pkey, app_ctx,
                                               (cqe->op_own >> 4) ==
                                               MLX5_CQE_RESP_SEND_IMM);

    if (UCT_RC_MLX5_TM_IS_SW_RNDV(cqe, imm_data)) {
        ctx->rndv_cb(ctx, tag, is_inline ? (cqe - 1) : priv->buffer, byte_len,
                     UCS_OK, is_inline ? UCT_TAG_RECV_CB_INLINE_DATA : 0);
        UCT_RC_MLX5_TM_STAT(iface, RX_RNDV_REQ_EXP);
    } else {
        ctx->completed_cb(ctx, tag, imm_data, byte_len,
                          (is_inline) ? (cqe - 1) : NULL, UCS_OK);
        UCT_RC_MLX5_TM_STAT(iface, RX_EXP);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_unexp_consumed(uct_rc_mlx5_iface_common_t *iface,
                                 unsigned offset, uct_recv_desc_t *release_desc,
                                 struct mlx5_cqe64 *cqe, ucs_status_t status,
                                 uint16_t wqe_ctr, int poll_flags)
{
    uct_ib_mlx5_srq_seg_t *seg;

    seg = uct_ib_mlx5_srq_get_wqe(&iface->rx.srq, wqe_ctr);

    uct_rc_mlx5_iface_release_srq_seg(iface, seg, cqe, wqe_ctr,
                                      status, offset, release_desc, poll_flags);

    if (ucs_unlikely(!(iface->tm.unexpected_cnt % IBV_DEVICE_MAX_UNEXP_COUNT))) {
        uct_rc_mlx5_iface_common_post_srq_op(&iface->tm.cmd_wq, 0,
                                             UCT_RC_MLX5_TM_OPCODE_NOP, 0,
                                             iface->tm.unexpected_cnt, 0ul, 0ul,
                                             UCT_RC_MLX5_SRQ_FLAG_TM_SW_CNT);

        UCT_RC_MLX5_TM_STAT(iface, LIST_SYNC);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_tag_handle_unexp(uct_rc_mlx5_iface_common_t *iface,
                                   struct mlx5_cqe64 *cqe, unsigned byte_len,
                                   int poll_flags)
{
    struct ibv_tmh           *tmh;
    uint64_t                 imm_data;
    ucs_status_t             status;
    unsigned                 flags;
    uct_rc_mlx5_mp_context_t *msg_ctx;

    tmh = uct_rc_mlx5_iface_tm_common_data(iface, cqe, byte_len, &flags,
                                           poll_flags |
                                           UCT_RC_MLX5_POLL_FLAG_TAG_CQE,
                                           &msg_ctx);

    /* Fast path: single fragment eager message */
    if (ucs_likely(UCT_RC_MLX5_SINGLE_FRAG_MSG(flags) &&
                   (tmh->opcode == IBV_TMH_EAGER) &&
                   !UCT_RC_MLX5_TM_CQE_WITH_IMM(cqe))) {
        status = iface->tm.eager_unexp.cb(iface->tm.eager_unexp.arg, tmh + 1,
                                          byte_len - sizeof(*tmh), flags,
                                          tmh->tag, 0, &msg_ctx->context);

        ++iface->tm.unexpected_cnt;
        uct_rc_mlx5_iface_unexp_consumed(iface, iface->tm.eager_desc.offset,
                                         &iface->tm.eager_desc.super, cqe,
                                         status, ntohs(cqe->wqe_counter),
                                         poll_flags);

        UCT_RC_MLX5_TM_STAT(iface, RX_EAGER_UNEXP);
        return;

    }

    if (ucs_unlikely(!(flags & UCT_CB_PARAM_FLAG_FIRST))) {
        /* Either middle or last fragment. Can pass zero tag, because it was
         * already provided in the first fragment. If it is last fragment and
         * CQE contains immediate value, construct user's immediate data using
         * imm value and TMH->app_ctx (saved in message context when the first
         * message arrived). Note, in case of send with immediate, only last
         * fragment CQE contains immediate data. */
        ucs_assert(!UCT_RC_MLX5_TM_CQE_WITH_IMM(cqe) ||
                   !(flags & UCT_CB_PARAM_FLAG_MORE));
        imm_data = uct_rc_mlx5_tag_imm_data_unpack(cqe->imm_inval_pkey,
                                                   msg_ctx->app_ctx,
                                                   UCT_RC_MLX5_TM_CQE_WITH_IMM(cqe));
        status   = iface->tm.eager_unexp.cb(iface->tm.eager_unexp.arg, tmh,
                                            byte_len, flags, msg_ctx->tag,
                                            imm_data, &msg_ctx->context);

        /* Do not increase unexpected_cnt count here, because it is counter per
         * message rather than per every fragment */
        uct_rc_mlx5_iface_unexp_consumed(iface,
                                         iface->super.super.config.rx_headroom_offset,
                                         &iface->super.super.release_desc,
                                         cqe, status, ntohs(cqe->wqe_counter),
                                         poll_flags);
        return;
    }

    ++iface->tm.unexpected_cnt;

    if (ucs_unlikely(tmh->opcode == IBV_TMH_RNDV)) {
        uct_rc_mlx5_handle_unexp_rndv(iface, tmh, tmh->tag, cqe, flags,
                                      byte_len, poll_flags);
        return;
    }

    ucs_assertv_always(tmh->opcode == IBV_TMH_EAGER,
                       "Unsupported packet arrived %d", tmh->opcode);

    /* Eager sync only, eager sync first or eager first. CQE can contain
       immediate value if it is eager sync only or sw rndv messages */
    imm_data = uct_rc_mlx5_tag_imm_data_unpack(cqe->imm_inval_pkey,
                                               tmh->app_ctx,
                                               UCT_RC_MLX5_TM_CQE_WITH_IMM(cqe));

    if (UCT_RC_MLX5_TM_CQE_WITH_IMM(cqe) && !imm_data) {
        ucs_assert(UCT_RC_MLX5_SINGLE_FRAG_MSG(flags));
        /* Opcode is WITH_IMM, but imm_data is 0 - this must be SW RNDV */
        status = iface->tm.rndv_unexp.cb(iface->tm.rndv_unexp.arg, 0, tmh->tag,
                                         tmh + 1, byte_len - sizeof(*tmh),
                                         0ul, 0, NULL);

        UCT_RC_MLX5_TM_STAT(iface, RX_RNDV_REQ_UNEXP);
    } else {

        /* Save app_context to assemble eager immediate data when the last
           fragment arrives (and contains imm value) */
        msg_ctx->app_ctx = tmh->app_ctx;

        /* Save tag to pass it with non-first fragments */
        msg_ctx->tag     = tmh->tag;

        status = iface->tm.eager_unexp.cb(iface->tm.eager_unexp.arg,
                                          tmh + 1, byte_len - sizeof(*tmh),
                                          flags, tmh->tag, imm_data,
                                          &msg_ctx->context);

        UCT_RC_MLX5_TM_STAT(iface, RX_EAGER_UNEXP);
    }

    uct_rc_mlx5_iface_unexp_consumed(iface, iface->tm.eager_desc.offset,
                                     &iface->tm.eager_desc.super, cqe,
                                     status, ntohs(cqe->wqe_counter),
                                     poll_flags);
}

static UCS_F_NOINLINE void
uct_rc_mlx5_iface_handle_filler_cqe(uct_rc_mlx5_iface_common_t *iface,
                                    struct mlx5_cqe64 *cqe, int poll_flags)
{
    uct_ib_mlx5_srq_seg_t *seg;

    /* filler CQE is relevant for MP XRQ only */
    ucs_assert_always(UCT_RC_MLX5_MP_ENABLED(iface));

    seg = uct_ib_mlx5_srq_get_wqe(&iface->rx.srq, ntohs(cqe->wqe_counter));

    /* at least one stride should be in HW ownership when filler CQE arrives */
    ucs_assert(seg->srq.strides);
    uct_rc_mlx5_iface_release_srq_seg(iface, seg, cqe, ntohs(cqe->wqe_counter),
                                      UCS_OK, 0, NULL, poll_flags);
}
#endif /* IBV_HW_TM */

static UCS_F_ALWAYS_INLINE unsigned
uct_rc_mlx5_iface_common_poll_rx(uct_rc_mlx5_iface_common_t *iface,
                                 int poll_flags)
{
    uct_ib_mlx5_srq_seg_t UCS_V_UNUSED *seg;
    struct mlx5_cqe64 *cqe;
    unsigned byte_len;
    uint16_t max_batch;
    unsigned count;
    void *rc_hdr;
    unsigned flags;
#if IBV_HW_TM
    struct ibv_tmh *tmh;
    uct_rc_mlx5_tag_entry_t *tag;
    uct_tag_context_t *ctx;
    uct_rc_mlx5_ctx_priv_t *priv;
    uct_rc_mlx5_mp_context_t UCS_V_UNUSED *dummy_ctx;
#endif

    ucs_assert((poll_flags & UCT_RC_MLX5_POLL_FLAG_LINKED_LIST) ||
               (uct_ib_mlx5_srq_get_wqe(&iface->rx.srq,
                                        iface->rx.srq.mask)->srq.next_wqe_index == 0));

    cqe = uct_rc_mlx5_iface_poll_rx_cq(iface, poll_flags);
    if (cqe == NULL) {
        /* If no CQE - post receives to prevent hang when:
         * - RX buffers are limited and all are taken by the user.
         * - All cqes are polled by the user, but new WQEs not posted due to
         *   lack of RX buffers.
         * Need to post new WQEs when some RX buffers are freed by the user.
         */
        count = 0;
        goto out;
    }

    ucs_memory_cpu_load_fence();
    UCS_STATS_UPDATE_COUNTER(iface->super.super.stats,
                             UCT_IB_IFACE_STAT_RX_COMPLETION, 1);

    byte_len = ntohl(cqe->byte_cnt) & UCT_IB_MLX5_MP_RQ_BYTE_CNT_MASK;
    count    = 1;

    if (!(poll_flags & UCT_RC_MLX5_POLL_FLAG_TM)) {
        rc_hdr = uct_rc_mlx5_iface_common_data(iface, cqe, byte_len, &flags);
        uct_rc_mlx5_iface_common_am_handler(iface, cqe, rc_hdr, flags,
                                            byte_len, poll_flags);
        goto out_update_db;
    }

#if IBV_HW_TM
    ucs_assert(cqe->app == UCT_RC_MLX5_CQE_APP_TAG_MATCHING);

    if (ucs_unlikely(byte_len & UCT_IB_MLX5_MP_RQ_FILLER_FLAG)) {
        /* TODO: Check if cqe->app_op is valid for filler CQE. Then this check
         * could be done for specific CQE types only. */
        uct_rc_mlx5_iface_handle_filler_cqe(iface, cqe, poll_flags);
        count = 0;
        goto out_update_db;
    }

    /* Should be a fast path, because small (latency-critical) messages
     * are not supposed to be offloaded to the HW.  */
    if (ucs_likely(cqe->app_op == UCT_RC_MLX5_CQE_APP_OP_TM_UNEXPECTED)) {
        uct_rc_mlx5_iface_tag_handle_unexp(iface, cqe, byte_len, poll_flags);
        goto out_update_db;
    }

    switch (cqe->app_op) {
    case UCT_RC_MLX5_CQE_APP_OP_TM_APPEND:
        uct_rc_mlx5_iface_handle_tm_list_op(iface,
                                            UCT_RC_MLX5_CQE_APP_OP_TM_APPEND);
        break;

    case UCT_RC_MLX5_CQE_APP_OP_TM_REMOVE:
        uct_rc_mlx5_iface_handle_tm_list_op(iface,
                                            UCT_RC_MLX5_CQE_APP_OP_TM_REMOVE);
        break;

    case UCT_RC_MLX5_CQE_APP_OP_TM_NO_TAG:
        /* TODO: optimize */
        tmh = uct_rc_mlx5_iface_tm_common_data(iface, cqe, byte_len, &flags,
                                               poll_flags, &dummy_ctx);

        /* With MP XRQ, AM can be single-fragment only */
        ucs_assert(UCT_RC_MLX5_SINGLE_FRAG_MSG(flags));

        if (tmh->opcode == IBV_TMH_NO_TAG) {
            uct_rc_mlx5_iface_common_am_handler(iface, cqe,
                                                (uct_rc_mlx5_hdr_t*)tmh,
                                                flags, byte_len, poll_flags);
        } else {
            ucs_assert(tmh->opcode == IBV_TMH_FIN);
            uct_rc_mlx5_handle_rndv_fin(iface, tmh->app_ctx);
            seg = uct_ib_mlx5_srq_get_wqe(&iface->rx.srq,
                                          ntohs(cqe->wqe_counter));

            uct_rc_mlx5_iface_release_srq_seg(iface, seg, cqe,
                                              ntohs(cqe->wqe_counter), UCS_OK,
                                              0, NULL, poll_flags);

            UCT_RC_MLX5_TM_STAT(iface, RX_RNDV_FIN);
        }
        break;

    case UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED:
        uct_rc_mlx5_iface_tag_consumed(iface, cqe,
                                       UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED);
        break;

    case UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED_MSG:
        tmh = (struct ibv_tmh*)cqe;

        uct_rc_mlx5_iface_tag_consumed(iface, cqe,
                                       UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED_MSG);

        uct_rc_mlx5_iface_handle_expected(iface, cqe, tmh->tag, tmh->app_ctx);
        break;

    case UCT_RC_MLX5_CQE_APP_OP_TM_EXPECTED:
        /* coverity[tainted_data] */
        tag  = &iface->tm.list[ntohs(cqe->app_info)];
        ctx  = tag->ctx;
        priv = uct_rc_mlx5_ctx_priv(ctx);
        uct_rc_mlx5_iface_handle_expected(iface, cqe, priv->tag, priv->app_ctx);
        break;

    default:
        ucs_fatal("Unsupported packet arrived %d", cqe->app_op);
        break;
    }
#endif

out_update_db:
    uct_ib_mlx5_update_db_cq_ci(&iface->cq[UCT_IB_DIR_RX]);
out:
    max_batch = iface->super.super.config.rx_max_batch;
    if (ucs_unlikely(iface->super.rx.srq.available >= max_batch)) {
        if (poll_flags & UCT_RC_MLX5_POLL_FLAG_LINKED_LIST) {
            uct_rc_mlx5_iface_srq_post_recv_ll(iface);
        } else {
            uct_rc_mlx5_iface_srq_post_recv(iface);
        }
    }
    return count;
}

#if HAVE_IBV_DM
/* DM memory should be written by 8 bytes to eliminate
 * processor cache issues. To make this used uct_rc_mlx5_dm_copy_data_t
 * datatype where first hdr_len bytes are filled by message header
 * and tail is filled by head of message. */
static void UCS_F_ALWAYS_INLINE uct_rc_mlx5_iface_common_copy_to_dm(
        uct_rc_mlx5_dm_copy_data_t *cache, size_t hdr_len, const uct_iov_t *iov,
        size_t iovcnt, void *dm, uct_ib_log_sge_t *log_sge)
{
    typedef uint64_t misaligned_t UCS_V_ALIGNED(1);

    size_t head = ((cache != NULL) && (hdr_len > 0)) ?
                  sizeof(cache->bytes) - hdr_len : 0;
    int i       = 0;
    ucs_iov_iter_t iov_iter;
    uint64_t word;
    size_t to_copy;
    void *src;

    ucs_assert(sizeof(cache->bytes) >= hdr_len);
    ucs_iov_iter_init(&iov_iter);

    /* copy head of iov to tail of cache */
    uct_iov_to_buffer(iov, iovcnt, &iov_iter,
                      UCS_PTR_BYTE_OFFSET(cache->bytes, hdr_len), head);

    UCS_STATIC_ASSERT(ucs_static_array_size(log_sge->sg_list) >= 2);

    /* condition is static-evaluated */
    if ((cache != NULL) && (hdr_len > 0)) {
        /* atomically by 8 bytes copy data to DM */
        /* cache buffer must be aligned, so, source data type is aligned */
        UCS_WORD_COPY(volatile uint64_t, dm, uint64_t, cache->bytes,
                      sizeof(cache->bytes));
        dm = UCS_PTR_BYTE_OFFSET(dm, sizeof(cache->bytes));
        if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) {
            log_sge->sg_list[i].addr   = (uint64_t)cache;
            log_sge->sg_list[i].length = (uint32_t)hdr_len;
            ++i;
        }
    }
    if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) {
        log_sge->sg_list[i].addr   = (uint64_t)iov->buffer;
        log_sge->sg_list[i].length = (uint32_t)iov->length;
        ++i;
    }
    log_sge->num_sge = i;

    while (iov_iter.iov_index < iovcnt) {
        to_copy = ucs_align_down(
                iov[iov_iter.iov_index].length - iov_iter.buffer_offset,
                sizeof(word));
        src     = UCS_PTR_BYTE_OFFSET(iov[iov_iter.iov_index].buffer,
                                      iov_iter.buffer_offset);
        UCS_WORD_COPY(volatile uint64_t, dm, misaligned_t, src, to_copy);

        dm                      = UCS_PTR_BYTE_OFFSET(dm, to_copy);
        iov_iter.buffer_offset += to_copy;

        if (iov_iter.buffer_offset < iov[iov_iter.iov_index].length) {
            /* copy remainder of the current iov buffer, and partially next
             * buffer(s) to fill word if there is smth left to copy */
            word = 0;
            uct_iov_to_buffer(iov, iovcnt, &iov_iter, &word, sizeof(word));
            UCS_WORD_COPY(volatile uint64_t, dm, uint64_t, &word, sizeof(word));
            dm = UCS_PTR_BYTE_OFFSET(dm, sizeof(word));
        } else {
            ++iov_iter.iov_index;
            iov_iter.buffer_offset = 0;
        }
    }
}

static ucs_status_t UCS_F_ALWAYS_INLINE uct_rc_mlx5_common_dm_make_data(
        uct_rc_mlx5_iface_common_t *iface, uct_rc_mlx5_dm_copy_data_t *cache,
        size_t hdr_len, const uct_iov_t *iov, size_t iovcnt,
        uct_rc_iface_send_desc_t **desc_p, void **buffer_p,
        uct_ib_log_sge_t *log_sge)
{
    uct_rc_iface_send_desc_t *desc;
    void *buffer;
    ucs_iov_iter_t iov_iter;

    ucs_assert(iface->dm.dm != NULL);
    ucs_assert(log_sge != NULL);

    desc = ucs_mpool_get_inline(&iface->dm.dm->mp);
    if (ucs_unlikely(desc == NULL)) {
        /* in case if no resources available - fallback to bcopy */
        UCT_RC_IFACE_GET_TX_DESC(&iface->super, &iface->super.tx.mp, desc);
        desc->super.handler = (uct_rc_send_handler_t)ucs_mpool_put;
        buffer = desc + 1;

        /* condition is static-evaluated, no performance penalty */
        if (cache && hdr_len) {
            memcpy(buffer, cache->bytes, hdr_len);
        }
        ucs_iov_iter_init(&iov_iter);
        uct_iov_to_buffer(iov, iovcnt, &iov_iter,
                          UCS_PTR_BYTE_OFFSET(buffer, hdr_len), SIZE_MAX);
        log_sge->num_sge = 0;
    } else {
        /* desc must be partially initialized by mpool.
         * hint to valgrind to make it defined */
        VALGRIND_MAKE_MEM_DEFINED(desc, sizeof(*desc));
        ucs_assert(desc->super.buffer != NULL);
        buffer = (void*)UCS_PTR_BYTE_DIFF(iface->dm.dm->start_va,
                                          desc->super.buffer);

        uct_rc_mlx5_iface_common_copy_to_dm(cache, hdr_len, iov, iovcnt,
                                            desc->super.buffer, log_sge);

        if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) {
            log_sge->sg_list[0].lkey = log_sge->sg_list[1].lkey = desc->lkey;
            log_sge->inline_bitmap = 0;
        }
    }

    *desc_p   = desc;
    *buffer_p = buffer;
    return UCS_OK;
}

static ucs_status_t UCS_F_ALWAYS_INLINE uct_rc_mlx5_common_ep_short_iov_dm(
        uct_rc_mlx5_iface_common_t *iface, int qp_type,
        uct_rc_mlx5_dm_copy_data_t *cache, size_t hdr_len, const uct_iov_t *iov,
        size_t iovcnt, size_t iov_length, unsigned opcode, uint8_t fm_ce_se,
        uint64_t rdma_raddr, uct_rkey_t rdma_rkey, uct_rc_txqp_t *txqp,
        uct_ib_mlx5_txwq_t *txwq, uct_ib_mlx5_base_av_t *av,
        struct mlx5_grh_av *grh_av, size_t av_size)
{
    uct_rc_iface_send_desc_t *desc = NULL;
    void *buffer;
    ucs_status_t status;
    uct_ib_log_sge_t log_sge;

    status = uct_rc_mlx5_common_dm_make_data(iface, cache, hdr_len, iov, iovcnt,
                                             &desc, &buffer, &log_sge);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        return status;
    }

    uct_rc_mlx5_common_txqp_bcopy_post(iface, qp_type, txqp, txwq, opcode,
                                       hdr_len + iov_length, rdma_raddr,
                                       rdma_rkey, av, grh_av, av_size, fm_ce_se,
                                       0, desc, buffer,
                                       (log_sge.num_sge > 0) ? &log_sge : NULL);

    return UCS_OK;
}

static ucs_status_t UCS_F_ALWAYS_INLINE uct_rc_mlx5_common_ep_short_dm(
        uct_rc_mlx5_iface_common_t *iface, int qp_type,
        uct_rc_mlx5_dm_copy_data_t *cache, size_t hdr_len, const void *payload,
        unsigned length, unsigned opcode, uint8_t fm_ce_se, uint64_t rdma_raddr,
        uct_rkey_t rdma_rkey, uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
        uct_ib_mlx5_base_av_t *av, struct mlx5_grh_av *grh_av, size_t av_size)
{
    uct_iov_t iov;

    iov.buffer = (void*)payload;
    iov.count  = 1;
    iov.length = length;

    return uct_rc_mlx5_common_ep_short_iov_dm(iface, qp_type, cache, hdr_len,
                                              &iov, 1, length, opcode, fm_ce_se,
                                              rdma_raddr, rdma_rkey, txqp, txwq,
                                              av, grh_av, av_size);
}

static ucs_status_t UCS_F_ALWAYS_INLINE uct_rc_mlx5_common_ep_am_short_iov_dm(
        uct_base_ep_t *ep, uint8_t am_id, uct_rc_mlx5_iface_common_t *iface,
        const uct_iov_t *iov, size_t iovcnt, size_t iov_length, int qp_type,
        uct_rc_txqp_t *txqp, uct_ib_mlx5_txwq_t *txwq,
        uct_ib_mlx5_base_av_t *av, struct mlx5_grh_av *grh_av, size_t av_size)
{
    uct_rc_mlx5_dm_copy_data_t cache;
    ucs_status_t status;

    UCT_CHECK_LENGTH(sizeof(cache.am_hdr.rc_hdr) + iov_length, 0,
                     iface->dm.seg_len, "am_short_iov");

    uct_rc_mlx5_am_hdr_fill(&cache.am_hdr.rc_hdr, am_id);
    status = uct_rc_mlx5_common_ep_short_iov_dm(
            iface, qp_type, &cache, sizeof(cache.am_hdr.rc_hdr), iov, iovcnt,
            iov_length, MLX5_OPCODE_SEND,
            MLX5_WQE_CTRL_SOLICITED | MLX5_WQE_CTRL_CQ_UPDATE, 0, 0, txqp, txwq,
            av, grh_av, av_size);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        return status;
    }

    UCT_TL_EP_STAT_OP(ep, AM, SHORT, sizeof(cache.am_hdr.rc_hdr) + iov_length);

    return UCS_OK;
}
#endif

static ucs_status_t UCS_F_ALWAYS_INLINE
uct_rc_mlx5_iface_common_atomic_data(unsigned opcode, unsigned size, uint64_t value,
                                     int *op, uint64_t *compare_mask, uint64_t *compare,
                                     uint64_t *swap_mask, uint64_t *swap, int *ext)
{
    ucs_assert((size == sizeof(uint64_t)) || (size == sizeof(uint32_t)));

    switch (opcode) {
    case UCT_ATOMIC_OP_ADD:
        switch (size) {
        case sizeof(uint64_t):
            *op       = MLX5_OPCODE_ATOMIC_FA;
            *ext      = 0;
            break;
        case sizeof(uint32_t):
            *op       = MLX5_OPCODE_ATOMIC_MASKED_FA;
            *ext      = 1;
            break;
        default:
            ucs_assertv(0, "incorrect atomic size: %d", size);
            return UCS_ERR_INVALID_PARAM;
        }
        *compare_mask = 0;
        *compare      = 0;
        *swap_mask    = 0;
        *swap         = UCT_RC_MLX5_TO_BE(value, size);
        break;
    case UCT_ATOMIC_OP_AND:
        *op           = MLX5_OPCODE_ATOMIC_MASKED_CS;
        *compare_mask = 0;
        *compare      = 0;
        *swap_mask    = UCT_RC_MLX5_TO_BE(~value, size);
        *swap         = UCT_RC_MLX5_TO_BE(value, size);
        *ext          = 1;
        break;
    case UCT_ATOMIC_OP_OR:
        *op           = MLX5_OPCODE_ATOMIC_MASKED_CS;
        *compare_mask = 0;
        *compare      = 0;
        *swap_mask    = UCT_RC_MLX5_TO_BE(value, size);
        *swap         = UCT_RC_MLX5_TO_BE(value, size);
        *ext          = 1;
        break;
    case UCT_ATOMIC_OP_XOR:
        *op           = MLX5_OPCODE_ATOMIC_MASKED_FA;
        *compare_mask = 0;
        *compare      = UINT64_MAX;
        *swap_mask    = 0;
        *swap         = UCT_RC_MLX5_TO_BE(value, size);
        *ext          = 1;
        break;
    case UCT_ATOMIC_OP_SWAP:
        *op           = MLX5_OPCODE_ATOMIC_MASKED_CS;
        *compare_mask = 0;
        *compare      = 0;
        *swap_mask    = UINT64_MAX;
        *swap         = UCT_RC_MLX5_TO_BE(value, size);
        *ext          = 1;
        break;
    default:
        ucs_assertv(0, "incorrect atomic opcode: %d", opcode);
        return UCS_ERR_UNSUPPORTED;
    }
    return UCS_OK;
}

