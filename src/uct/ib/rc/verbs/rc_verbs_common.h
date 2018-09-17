/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RC_VERBS_COMMON_H
#define UCT_RC_VERBS_COMMON_H

#include <ucs/arch/bitops.h>

#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/rc/base/rc_ep.h>


/* definitions common to rc_verbs and dc_verbs go here */


#define UCT_RC_VERBS_IFACE_FOREACH_TXWQE(_iface, _i, _wc, _num_wcs) \
      status = uct_ib_poll_cq((_iface)->super.cq[UCT_IB_DIR_TX], &_num_wcs, _wc); \
      if (status != UCS_OK) { \
          return 0; \
      } \
      UCS_STATS_UPDATE_COUNTER((_iface)->stats, \
                               UCT_RC_IFACE_STAT_TX_COMPLETION, _num_wcs); \
      for (_i = 0; _i < _num_wcs; ++_i)


typedef struct uct_rc_verbs_txcnt {
    uint16_t       pi;      /* producer (post_send) count */
    uint16_t       ci;      /* consumer (ibv_poll_cq) completion count */
} uct_rc_verbs_txcnt_t;


/**
 * RC/DC verbs interface configuration
 */
typedef struct uct_rc_verbs_iface_common_config {
    size_t                 max_am_hdr;
    unsigned               tx_max_wr;
#if IBV_EXP_HW_TM
    double                 tm_sync_ratio;
#endif
    /* TODO flags for exp APIs */
} uct_rc_verbs_iface_common_config_t;


typedef struct uct_rc_verbs_iface_common {
    struct ibv_sge         inl_sge[2];
    uct_rc_am_short_hdr_t  am_inl_hdr;
    ucs_mpool_t            short_desc_mp;

    struct {
        unsigned           num_canceled;
        unsigned           tag_sync_thresh;
    } tm;

    /* TODO: make a separate datatype */
    struct {
        size_t             short_desc_size;
        size_t             max_inline;
    } config;
} uct_rc_verbs_iface_common_t;


extern ucs_config_field_t uct_rc_verbs_iface_common_config_table[];

void uct_rc_verbs_txcnt_init(uct_rc_verbs_txcnt_t *txcnt);

ucs_status_t uct_rc_verbs_wc_to_ucs_status(enum ibv_wc_status status);

void uct_rc_verbs_common_packet_dump(uct_base_iface_t *iface, uct_am_trace_type_t type,
                                     void *data, size_t length, size_t valid_length,
                                     char *buffer, size_t max);

static inline void
uct_rc_verbs_txqp_posted(uct_rc_txqp_t *txqp, uct_rc_verbs_txcnt_t *txcnt,
                         uct_rc_iface_t *iface, int signaled)
{
    txcnt->pi++;
    uct_rc_txqp_posted(txqp, iface, 1, signaled);
}

static inline void
uct_rc_verbs_txqp_completed(uct_rc_txqp_t *txqp, uct_rc_verbs_txcnt_t *txcnt, uint16_t count)
{
    txcnt->ci += count;
    uct_rc_txqp_available_add(txqp, count);
}

ucs_status_t uct_rc_verbs_iface_common_init(uct_rc_verbs_iface_common_t *iface,
                                            uct_rc_iface_t *rc_iface,
                                            uct_rc_verbs_iface_common_config_t *config,
                                            uct_rc_iface_config_t *rc_config);

void uct_rc_verbs_iface_common_cleanup(uct_rc_verbs_iface_common_t *iface);

ucs_status_t uct_rc_verbs_iface_common_prepost_recvs(uct_rc_iface_t *iface,
                                                     unsigned max);

void uct_rc_verbs_iface_common_progress_enable(uct_iface_h tl_iface, unsigned flags);

unsigned uct_rc_verbs_iface_post_recv_always(uct_rc_iface_t *iface, unsigned max);

static inline unsigned uct_rc_verbs_iface_post_recv_common(uct_rc_iface_t *iface,
                                                           int fill)
{
    unsigned batch = iface->super.config.rx_max_batch;
    unsigned count;

    if (iface->rx.srq.available < batch) {
        if (ucs_likely(fill == 0)) {
            return 0;
        } else {
            count = iface->rx.srq.available;
        }
    } else {
        count = batch;
    }
    return uct_rc_verbs_iface_post_recv_always(iface, count);
}


/* TODO: think of a better name */
static inline int
uct_rc_verbs_txcq_get_comp_count(struct ibv_wc *wc, uct_rc_txqp_t *txqp)
{
    uint16_t count = 1;

    if (ucs_likely(wc->wr_id != RC_UNSIGNALED_INF)) {
        return wc->wr_id + 1;
    }

    ucs_assert(txqp->unsignaled_store != RC_UNSIGNALED_INF);
    ucs_assert(txqp->unsignaled_store_count != 0);

    txqp->unsignaled_store_count--;
    if (txqp->unsignaled_store_count == 0) {
        count += txqp->unsignaled_store;
        txqp->unsignaled_store = 0;
    }

    return count;
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
    if (ucs_likely(status == UCS_OK)) {
        ucs_mpool_put_inline(desc);
    } else {
        udesc = (char*)desc + iface->super.config.rx_headroom_offset;
        uct_recv_desc(udesc) = &iface->super.release_desc;
    }
}

static UCS_F_ALWAYS_INLINE unsigned
uct_rc_verbs_iface_poll_rx_common(uct_rc_iface_t *iface)
{
    uct_rc_hdr_t *hdr;
    unsigned i;
    ucs_status_t status;
    unsigned num_wcs = iface->super.config.rx_max_poll;
    struct ibv_wc wc[num_wcs];

    status = uct_ib_poll_cq(iface->super.cq[UCT_IB_DIR_RX], &num_wcs, wc);
    if (status != UCS_OK) {
        num_wcs = 0;
        goto out;
    }

    UCT_IB_IFACE_VERBS_FOREACH_RXWQE(&iface->super, i, hdr, wc, num_wcs) {
        uct_ib_log_recv_completion(&iface->super, IBV_QPT_RC, &wc[i], hdr,
                                   wc[i].byte_len, uct_rc_verbs_common_packet_dump);
        uct_rc_verbs_iface_handle_am(iface, hdr, wc[i].wr_id, wc[i].qp_num,
                                     wc[i].byte_len, wc[i].imm_data, wc[i].slid);
    }
    iface->rx.srq.available += num_wcs;
    UCS_STATS_UPDATE_COUNTER(iface->stats, UCT_RC_IFACE_STAT_RX_COMPLETION, num_wcs);

out:
    uct_rc_verbs_iface_post_recv_common(iface, 0);
    return num_wcs;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_iface_fill_inl_sge(uct_rc_verbs_iface_common_t *iface, const void *addr0,
                                unsigned len0, const void* addr1, unsigned len1)
{
    iface->inl_sge[0].addr      = (uintptr_t)addr0;
    iface->inl_sge[0].length    = len0;
    iface->inl_sge[1].addr      = (uintptr_t)addr1;
    iface->inl_sge[1].length    = len1;
}

static inline void
uct_rc_verbs_iface_fill_inl_am_sge(uct_rc_verbs_iface_common_t *iface,
                                   uint8_t id, uint64_t hdr,
                                   const void *buffer, unsigned length)
{
    uct_rc_am_short_hdr_t *am = &iface->am_inl_hdr;
    am->rc_hdr.am_id = id;
    am->am_hdr       = hdr;
    uct_rc_verbs_iface_fill_inl_sge(iface, am, sizeof(*am), buffer, length);
}

#if IBV_EXP_HW_TM


/* If message arrived with imm_data = 0 - it is SW RNDV request */
#  define UCT_RC_VERBS_TM_IS_SW_RNDV(_flags, _imm_data) \
       (ucs_unlikely(((_flags) & IBV_EXP_WC_WITH_IMM) && !(_imm_data)))

#  define UCT_RC_VERBS_GET_TM_ZCOPY_DESC(_iface, _mp, _desc, _tag, _app_ctx, \
                                         _comp, _send_flags, _sge) \
       { \
           void *hdr; \
           UCT_RC_IFACE_GET_TX_TM_DESC(_iface, _mp, _desc, _tag, _app_ctx, hdr) \
           uct_rc_zcopy_desc_set_comp(_desc, _comp, _send_flags); \
           _sge.length = sizeof(struct ibv_exp_tmh); \
       }

#  define UCT_RC_VERBS_FILL_TM_ADD_WR(_wr, _tag, _tag_mask, _sge, _sge_cnt, _ctx) \
       { \
           (_wr)->tm.add.tag        = tag; \
           (_wr)->tm.add.mask       = tag_mask; \
           (_wr)->tm.add.sg_list    = _sge; \
           (_wr)->tm.add.num_sge    = _sge_cnt; \
           (_wr)->tm.add.recv_wr_id = (uint64_t)_ctx; \
       }

#  define UCT_RC_VERBS_FILL_TM_OP_WR(_iface, _wr, _opcode, _flags, _wr_id) \
       { \
           (_wr)->tm.unexpected_cnt = (_iface)->tm.unexpected_cnt; \
           (_wr)->wr_id             = _wr_id; \
           (_wr)->opcode            = (enum ibv_exp_ops_wr_opcode)_opcode; \
           (_wr)->flags             = _flags | IBV_EXP_OPS_TM_SYNC; \
           (_wr)->next              = NULL; \
       }

#  define UCT_RC_VERBS_CHECK_TAG(_iface) \
       if (!(_iface)->tm.num_tags) {  \
           return UCS_ERR_EXCEEDS_LIMIT; \
       }

ucs_status_t
uct_rc_verbs_iface_common_tag_init(uct_rc_verbs_iface_common_t *iface,
                                   uct_rc_iface_t *rc_iface,
                                   uct_rc_verbs_iface_common_config_t *config,
                                   uct_rc_iface_config_t *rc_config,
                                   struct ibv_exp_create_srq_attr *srq_init_attr,
                                   size_t rndv_hdr_len);

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_verbs_iface_post_op(uct_rc_iface_t *iface, struct ibv_exp_ops_wr *wr,
                           int op, int flags, uint64_t wr_id)
{
    struct ibv_exp_ops_wr *bad_wr;
    int ret;

    UCT_RC_VERBS_FILL_TM_OP_WR(iface, wr, op, flags, wr_id);

    ret = ibv_exp_post_srq_ops(iface->rx.srq.srq, wr, &bad_wr);
    if (ret) {
        ucs_error("ibv_exp_post_srq_ops(op=%d) failed: %m", op);
        return UCS_ERR_IO_ERROR;
    }
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_verbs_iface_post_signaled_op(uct_rc_verbs_iface_common_t *iface,
                                    uct_rc_iface_t *rc_iface,
                                    struct ibv_exp_ops_wr *wr, int op)
{
    ucs_status_t status;

    status = uct_rc_verbs_iface_post_op(rc_iface, wr, op, IBV_EXP_OPS_SIGNALED,
                                        iface->tm.num_canceled);
    if (status != UCS_OK) {
        return status;
    }

    iface->tm.num_canceled = 0;
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_verbs_iface_common_tag_recv(uct_rc_verbs_iface_common_t *iface,
                                   uct_rc_iface_t *rc_iface,
                                   uct_tag_t tag, uct_tag_t tag_mask,
                                   const uct_iov_t *iov, size_t iovcnt,
                                   uct_tag_context_t *ctx)
{
    uct_rc_iface_ctx_priv_t *priv = uct_rc_iface_ctx_priv(ctx);
    ucs_status_t status;
    struct ibv_sge sge[UCT_IB_MAX_IOV];
    struct ibv_exp_ops_wr wr;
    size_t sge_cnt;

    UCT_CHECK_IOV_SIZE(iovcnt, 1ul, "uct_rc_verbs_iface_common_tag_recv");
    UCT_RC_VERBS_CHECK_TAG(rc_iface);

    sge_cnt = uct_ib_verbs_sge_fill_iov(sge, iov, iovcnt);
    UCT_RC_VERBS_FILL_TM_ADD_WR(&wr, tag, tag_mask, sge, sge_cnt, ctx);

    status = uct_rc_verbs_iface_post_signaled_op(iface, rc_iface, &wr,
                                                 IBV_EXP_WR_TAG_ADD);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    --rc_iface->tm.num_tags;
    ++rc_iface->tm.num_outstanding;

    /* Save tag index in the device tags list returned by ibv_exp_post_srq_ops.
     * It may be needed for cancelling this posted tag. */
    priv->tag_handle = wr.tm.handle;
    priv->tag        = tag;
    priv->buffer     = iov->buffer; /* Only one iov is supported so far */
    priv->length     = uct_iov_total_length(iov, iovcnt);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_verbs_iface_common_tag_recv_cancel(uct_rc_verbs_iface_common_t *iface,
                                          uct_rc_iface_t *rc_iface,
                                          uct_tag_context_t *ctx, int force)
{
   uct_rc_iface_ctx_priv_t *priv = uct_rc_iface_ctx_priv(ctx);
   struct ibv_exp_ops_wr wr;
   ucs_status_t status;

   wr.tm.handle = priv->tag_handle;

   status = uct_rc_verbs_iface_post_op(rc_iface, &wr, IBV_EXP_WR_TAG_DEL,
                                       force ? 0 : IBV_EXP_OPS_SIGNALED,
                                       (uint64_t)ctx);
   if (status != UCS_OK) {
       return status;
   }

   if (force) {
       if (rc_iface->tm.num_outstanding) {
           ++iface->tm.num_canceled;
       } else {
           /* No pending ADD operations, free the tag immediately */
           ++rc_iface->tm.num_tags;
       }
       if (iface->tm.num_canceled > iface->tm.tag_sync_thresh) {
           /* Too many pending cancels. Need to issue a signaled operation
            * to free the canceled tags */
           uct_rc_verbs_iface_post_signaled_op(iface, rc_iface, &wr,
                                               IBV_EXP_WR_TAG_SYNC);
       }
   }

   return UCS_OK;
}

static UCS_F_ALWAYS_INLINE uint32_t
uct_rc_verbs_iface_fill_rndv_hdrs(uct_rc_iface_t *iface,
                                  uct_rc_verbs_iface_common_t *verbs_common,
                                  struct ibv_exp_tmh *tmh, const void *hdr,
                                  unsigned hdr_len, unsigned max_rndv_priv_data,
                                  unsigned tmh_len, uct_tag_t tag,
                                  const uct_iov_t *iov, uct_completion_t *comp)
{
    uint32_t op_index;
    unsigned tmh_data_len;

    op_index = uct_rc_iface_tag_get_op_id(iface, comp);
    uct_rc_iface_fill_tmh(tmh, tag, op_index, IBV_EXP_TMH_RNDV);
    tmh_data_len = uct_rc_iface_fill_tmh_priv_data(tmh, hdr, hdr_len,
                                                   max_rndv_priv_data);
    uct_rc_iface_fill_rvh((struct ibv_exp_tmh_rvh*)(tmh + 1), iov->buffer,
                          ((uct_ib_mem_t*)iov->memh)->mr->rkey, iov->length);
    uct_rc_verbs_iface_fill_inl_sge(verbs_common, tmh, tmh_len,
                                    (char*)hdr + tmh_data_len,
                                    hdr_len - tmh_data_len);

    return op_index;
}

/* This function check whether the error occured due to "MESSAGE_TRUNCATED"
 * error in Tag Matching (i.e. if posted buffer was not enough to fit the
 * incoming message). If this is the case the error should be reported in
 * the corresponding callback and QP should be reset back to normal. Otherwise
 * treat the error as fatal. */
static UCS_F_NOINLINE void
uct_rc_verbs_iface_wc_error(enum ibv_wc_status status)
{
    /* TODO: handle MSG TRUNCATED error */
    ucs_fatal("Receive completion with error on XRQ: %s",
              uct_ib_wc_status_str(status));
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_iface_tag_handle_exp(uct_rc_iface_t *iface, struct ibv_exp_wc *wc)
{
    uct_tag_context_t *ctx        = (uct_tag_context_t*)wc->wr_id;
    uct_rc_iface_ctx_priv_t *priv = uct_rc_iface_ctx_priv(ctx);
    uint64_t imm_data;

    if (wc->exp_wc_flags & IBV_EXP_WC_TM_MATCH) {
        /* Need to keep app_ctx in case DATA will come with immediate */
        priv->app_ctx = wc->tm_info.priv;
        priv->tag     = wc->tm_info.tag;
        ctx->tag_consumed_cb(ctx);
    }

    if (wc->exp_wc_flags & IBV_EXP_WC_TM_DATA_VALID) {
        imm_data = uct_rc_iface_tag_imm_data_unpack(wc->imm_data, priv->app_ctx,
                                                    wc->exp_wc_flags &
                                                    IBV_EXP_WC_WITH_IMM);

        VALGRIND_MAKE_MEM_DEFINED(priv->buffer, wc->byte_len);
        if (UCT_RC_VERBS_TM_IS_SW_RNDV(wc->exp_wc_flags, imm_data)) {
            ctx->rndv_cb(ctx, priv->tag, priv->buffer, wc->byte_len, UCS_OK);
        } else {
            ctx->completed_cb(ctx, priv->tag, imm_data, wc->byte_len, UCS_OK);
        }
        ++iface->tm.num_tags;
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_iface_unexp_consumed(uct_rc_verbs_iface_common_t *iface,
                                  uct_rc_iface_t *rc_iface,
                                  uct_ib_iface_recv_desc_t *ib_desc,
                                  uct_rc_iface_release_desc_t *release,
                                  ucs_status_t comp_status)
{
    struct ibv_exp_ops_wr wr;
    void *udesc;

    if (comp_status == UCS_OK) {
        ucs_mpool_put_inline(ib_desc);
    } else {
        udesc = (char*)ib_desc + release->offset;
        uct_recv_desc(udesc) = &release->super;
    }

    if (ucs_unlikely(!(++rc_iface->tm.unexpected_cnt % IBV_DEVICE_MAX_UNEXP_COUNT))) {
        uct_rc_verbs_iface_post_signaled_op(iface, rc_iface, &wr,
                                            IBV_EXP_WR_TAG_SYNC);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_iface_tag_handle_unexp(uct_rc_verbs_iface_common_t *iface,
                                    uct_rc_iface_t *rc_iface,
                                    struct ibv_exp_wc *wc)
{
    uct_ib_iface_recv_desc_t *ib_desc = (uct_ib_iface_recv_desc_t*)(uintptr_t)wc->wr_id;
    struct ibv_exp_tmh *tmh;
    uct_rc_hdr_t *rc_hdr;
    uint64_t imm_data;
    ucs_status_t status;

    tmh = (struct ibv_exp_tmh*)uct_ib_iface_recv_desc_hdr(&rc_iface->super, ib_desc);
    VALGRIND_MAKE_MEM_DEFINED(tmh, wc->byte_len);

    switch (tmh->opcode) {
    case IBV_EXP_TMH_EAGER:
        imm_data = uct_rc_iface_tag_imm_data_unpack(wc->imm_data,
                                                    ntohl(tmh->app_ctx),
                                                    wc->exp_wc_flags &
                                                    IBV_EXP_WC_WITH_IMM);

        if (UCT_RC_VERBS_TM_IS_SW_RNDV(wc->exp_wc_flags, imm_data)) {
            status = rc_iface->tm.rndv_unexp.cb(rc_iface->tm.rndv_unexp.arg,
                                                UCT_CB_PARAM_FLAG_DESC,
                                                be64toh(tmh->tag), tmh + 1,
                                                wc->byte_len - sizeof(*tmh),
                                                0ul, 0, NULL);
        } else {
            status = rc_iface->tm.eager_unexp.cb(rc_iface->tm.eager_unexp.arg,
                                                 tmh + 1, wc->byte_len - sizeof(*tmh),
                                                 UCT_CB_PARAM_FLAG_DESC,
                                                 be64toh(tmh->tag), imm_data);
        }
        uct_rc_verbs_iface_unexp_consumed(iface, rc_iface, ib_desc,
                                          &rc_iface->tm.eager_desc, status);
        break;

    case IBV_EXP_TMH_NO_TAG:
        rc_hdr = (uct_rc_hdr_t*)tmh;
        uct_ib_log_recv_completion(&rc_iface->super, IBV_QPT_RC, wc, rc_hdr,
                                   wc->byte_len, uct_rc_verbs_common_packet_dump);
        uct_rc_verbs_iface_handle_am(rc_iface, rc_hdr, wc->wr_id, wc->qp_num,
                                     wc->byte_len, wc->imm_data, wc->slid);
        break;

    case IBV_EXP_TMH_RNDV:
        status = uct_rc_iface_handle_rndv(rc_iface, tmh, be64toh(tmh->tag),
                                          wc->byte_len);

        uct_rc_verbs_iface_unexp_consumed(iface, rc_iface, ib_desc,
                                          &rc_iface->tm.rndv_desc, status);
        break;

    case IBV_EXP_TMH_FIN:
        uct_rc_iface_handle_rndv_fin(rc_iface, ntohl(tmh->app_ctx));
        ucs_mpool_put_inline(ib_desc);
        break;

    default:
        ucs_fatal("Unsupported packet arrived %d", tmh->opcode);
        break;
    }

    ++rc_iface->rx.srq.available;
}

static UCS_F_ALWAYS_INLINE unsigned
uct_rc_verbs_iface_poll_rx_tm(uct_rc_verbs_iface_common_t *iface,
                              uct_rc_iface_t *rc_iface)
{
    const unsigned max_wcs = rc_iface->super.config.rx_max_poll;
    struct ibv_exp_wc wc[max_wcs];
    uct_tag_context_t *ctx;
    uct_rc_iface_ctx_priv_t *priv;
    int num_wcs, i;

    num_wcs = ibv_exp_poll_cq(rc_iface->super.cq[UCT_IB_DIR_RX], max_wcs, wc,
                              sizeof(wc[0]));
    if (num_wcs <= 0) {
        if (ucs_unlikely(num_wcs < 0)) {
            ucs_fatal("Failed to poll receive CQ %d", num_wcs);
        }
        goto out;
    }

    for (i = 0; i < num_wcs; ++i) {
        if (ucs_unlikely(wc[i].status != IBV_WC_SUCCESS)) {
            uct_rc_verbs_iface_wc_error(wc[i].status);
            continue;
        }
        switch (wc[i].exp_opcode) {
        case IBV_EXP_WC_TM_NO_TAG:
        case IBV_EXP_WC_RECV:
            uct_rc_verbs_iface_tag_handle_unexp(iface, rc_iface, &wc[i]);
            break;

        case IBV_EXP_WC_TM_RECV:
            if (wc[i].exp_wc_flags &
                (IBV_EXP_WC_TM_MATCH | IBV_EXP_WC_TM_DATA_VALID)) {
                uct_rc_verbs_iface_tag_handle_exp(rc_iface, &wc[i]);
            } else {
                uct_rc_verbs_iface_tag_handle_unexp(iface, rc_iface, &wc[i]);
            }
            break;

        case IBV_EXP_WC_TM_DEL:
            ctx  = (uct_tag_context_t*)wc[i].wr_id;
            priv = uct_rc_iface_ctx_priv(ctx);
            ctx->completed_cb(ctx, priv->tag, 0, priv->length,
                              UCS_ERR_CANCELED);
            ++rc_iface->tm.num_tags;
            break;

        case IBV_EXP_WC_TM_ADD:
            --rc_iface->tm.num_outstanding;
            /* Fall through */
        case IBV_EXP_WC_TM_SYNC:
            rc_iface->tm.num_tags += wc[i].wr_id;
            break;

        default:
            ucs_error("Wrong opcode in CQE %d", wc[i].exp_opcode);
            break;

        }
    }
    /* TODO: Add stat */
out:
    uct_rc_verbs_iface_post_recv_common(rc_iface, 0);

    return num_wcs;
}

#else

#  define UCT_RC_VERBS_TM_ENABLED(_iface) 0

#endif /* IBV_EXP_HW_TM */


#define UCT_RC_VERBS_FILL_SGE(_wr, _sge, _length) \
    _wr.sg_list = &_sge; \
    _wr.num_sge = 1; \
    _sge.length = _length;

#define UCT_RC_VERBS_FILL_INL_PUT_WR(_iface, _raddr, _rkey, _buf, _len) \
    _iface->inl_rwrite_wr.wr.rdma.remote_addr = _raddr; \
    _iface->inl_rwrite_wr.wr.rdma.rkey        = uct_ib_md_direct_rkey(_rkey); \
    _iface->verbs_common.inl_sge[0].addr      = (uintptr_t)_buf; \
    _iface->verbs_common.inl_sge[0].length    = _len;

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
    _wr.wr.rdma.rkey        = uct_ib_md_direct_rkey(_rkey); \
    _wr_opcode              = _opcode; \

#define UCT_RC_VERBS_FILL_RDMA_WR_IOV(_wr, _wr_opcode, _opcode, _sge, _sgelen, \
                                      _raddr, _rkey) \
    _wr.wr.rdma.remote_addr = _raddr; \
    _wr.wr.rdma.rkey        = uct_ib_md_direct_rkey(_rkey); \
    _wr.sg_list             = _sge; \
    _wr.num_sge             = _sgelen; \
    _wr_opcode              = _opcode;

#define UCT_RC_VERBS_FILL_DESC_WR(_wr, _desc) \
    { \
        struct ibv_sge *sge; \
        (_wr)->next    = NULL; \
        sge            = (_wr)->sg_list; \
        sge->addr      = (uintptr_t)(desc + 1); \
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


#if HAVE_IB_EXT_ATOMICS
static inline void
uct_rc_verbs_fill_ext_atomic_wr(struct ibv_exp_send_wr *wr, struct ibv_sge *sge,
                                int opcode, uint32_t length, uint32_t compare_mask,
                                uint64_t compare_add, uint64_t swap, uint64_t remote_addr,
                                uct_rkey_t rkey, size_t atomic_mr_offset)
{
    sge->length        = length;
    wr->sg_list        = sge;
    wr->num_sge        = 1;
    wr->exp_opcode     = (enum ibv_exp_wr_opcode)opcode;
    wr->comp_mask      = 0;

    wr->ext_op.masked_atomics.log_arg_sz  = ucs_ilog2(length);
    wr->ext_op.masked_atomics.rkey        = uct_ib_resolve_atomic_rkey(rkey,
                                                                       atomic_mr_offset,
                                                                       &remote_addr);
    wr->ext_op.masked_atomics.remote_addr = remote_addr;

    switch (opcode) {
    case IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP:
        wr->ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap.compare_mask = compare_mask;
        wr->ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap.compare_val  = compare_add;
        wr->ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap.swap_mask    = (uint64_t)(-1);
        wr->ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap.swap_val     = swap;
        break;
    case IBV_EXP_WR_EXT_MASKED_ATOMIC_FETCH_AND_ADD:
        wr->ext_op.masked_atomics.wr_data.inline_data.op.fetch_add.add_val        = compare_add;
        wr->ext_op.masked_atomics.wr_data.inline_data.op.fetch_add.field_boundary = 0;
        break;
    }
}

static UCS_F_ALWAYS_INLINE
ucs_status_t uct_rc_verbs_ep_atomic32_data(uct_atomic_op_t opcode, uint32_t value,
                                           int *op, uint32_t *add, uint32_t *swap)
{
    switch (opcode) {
    case UCT_ATOMIC_OP_ADD:
        *op   = IBV_EXP_WR_EXT_MASKED_ATOMIC_FETCH_AND_ADD;
        *add  = value;
        *swap = 0;
        return UCS_OK;
    case UCT_ATOMIC_OP_SWAP:
        *op   = IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP;
        *add  = 0;
        *swap = value;
        return UCS_OK;
    default:
        return UCS_ERR_UNSUPPORTED;
    }
}
#endif


#endif
