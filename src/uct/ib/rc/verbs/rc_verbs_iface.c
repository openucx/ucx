/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rc_verbs.h"
#include "rc_verbs_common.h"

#include <uct/api/uct.h>
#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_log.h>
#include <uct/ib/base/ib_md.h>
#include <uct/base/uct_md.h>
#include <ucs/arch/bitops.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>
#include <string.h>

static uct_rc_iface_ops_t uct_rc_verbs_iface_ops;

static ucs_config_field_t uct_rc_verbs_iface_config_table[] = {
  {"RC_", "", NULL,
   ucs_offsetof(uct_rc_verbs_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_rc_iface_config_table)},

  {"", "", NULL,
   ucs_offsetof(uct_rc_verbs_iface_config_t, verbs_common),
   UCS_CONFIG_TYPE_TABLE(uct_rc_verbs_iface_common_config_table)},

  {"", "", NULL,
   ucs_offsetof(uct_rc_verbs_iface_config_t, fc),
   UCS_CONFIG_TYPE_TABLE(uct_rc_fc_config_table)},

#if IBV_EXP_HW_TM
  {"TM_ENABLE", "y",
   "Enable HW tag matching",
   ucs_offsetof(uct_rc_verbs_iface_config_t, tm.enable), UCS_CONFIG_TYPE_BOOL},

  {"TM_LIST_SIZE", "64",
   "Limits the number of tags posted to the HW for matching. The actual limit \n"
   "is a minimum between this value and the maximum value supported by the HW. \n"
   "-1 means no limit.",
   ucs_offsetof(uct_rc_verbs_iface_config_t, tm.list_size), UCS_CONFIG_TYPE_UINT},

  {"TM_RX_RNDV_QUEUE_LEN", "128",
   "Length of receive queue in the QP owned by the device. It is used for receiving \n"
   "RNDV Complete messages sent by the device",
   ucs_offsetof(uct_rc_verbs_iface_config_t, tm.rndv_queue_len), UCS_CONFIG_TYPE_UINT},
#endif

  {NULL}
};

static void uct_rc_verbs_handle_failure(uct_ib_iface_t *ib_iface, void *arg)
{
    uct_rc_verbs_ep_t *ep;
    struct ibv_wc     *wc    = arg;
    uct_rc_iface_t    *iface = ucs_derived_of(ib_iface, uct_rc_iface_t);

    ep = ucs_derived_of(uct_rc_iface_lookup_ep(iface, wc->qp_num),
                        uct_rc_verbs_ep_t);
    if (!ep) {
        return;
    }

    ucs_log(iface->super.super.config.failure_level,
            "Send completion with error: %s",
            ibv_wc_status_str(wc->status));
    printf("error happended on ep=%p WR_id=%lu\n", ep, wc->wr_id);

    uct_rc_ep_failed_purge_outstanding(&ep->super.super.super, ib_iface,
                                       &ep->super.txqp);
}

static void uct_rc_verbs_ep_set_failed(uct_ib_iface_t *iface, uct_ep_h ep)
{
    uct_set_ep_failed(&UCS_CLASS_NAME(uct_rc_verbs_ep_t), ep, &iface->super.super);
}

void uct_rc_verbs_ep_am_packet_dump(uct_base_iface_t *base_iface,
                                    uct_am_trace_type_t type,
                                    void *data, size_t length,
                                    size_t valid_length,
                                    char *buffer, size_t max)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(base_iface,
                                                 uct_rc_verbs_iface_t);
    uct_rc_ep_am_packet_dump(base_iface, type,
                             data + iface->verbs_common.config.notag_hdr_size,
                             length - iface->verbs_common.config.notag_hdr_size,
                             valid_length, buffer, max);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_iface_poll_tx(uct_rc_verbs_iface_t *iface)
{
    uct_rc_verbs_ep_t *ep;
    uint16_t count;
    int i;
    unsigned num_wcs = iface->super.super.config.tx_max_poll;
    struct ibv_wc wc[num_wcs];
    ucs_status_t status;

    UCT_RC_VERBS_IFACE_FOREACH_TXWQE(&iface->super, i, wc, num_wcs) {
        count = uct_rc_verbs_txcq_get_comp_count(&wc[i]);
        ep = ucs_derived_of(uct_rc_iface_lookup_ep(&iface->super, wc[i].qp_num),
                            uct_rc_verbs_ep_t);
        ucs_trace_poll("rc_verbs iface %p tx_wc: ep %p qpn 0x%x count %d",
                       iface, ep, wc[i].qp_num, count);

        if (ucs_unlikely((wc[i].status != IBV_WC_SUCCESS) || (ep == NULL))) {
            iface->super.super.ops->handle_failure(&iface->super.super, &wc[i]);
            continue;
        }
        uct_rc_verbs_txqp_completed(&ep->super.txqp, &ep->txcnt, count);

        uct_rc_txqp_completion_desc(&ep->super.txqp, ep->txcnt.ci);
        ucs_arbiter_group_schedule(&iface->super.tx.arbiter, &ep->super.arb_group);
    }
    iface->super.tx.cq_available += num_wcs;
    ucs_arbiter_dispatch(&iface->super.tx.arbiter, 1, uct_rc_ep_process_pending, NULL);
}

void uct_rc_verbs_iface_progress(void *arg)
{
    uct_rc_verbs_iface_t *iface = arg;
    ucs_status_t status;

    status = uct_rc_verbs_iface_poll_rx_common(&iface->super);
    if (status == UCS_ERR_NO_PROGRESS) {
        uct_rc_verbs_iface_poll_tx(iface);
    }
}

#if IBV_EXP_HW_TM
/* This function check whether the error occured due to "MESSAGE_TRUNCATED"
 * error in Tag Matching (i.e. if posted buffer was not enough to fit the
 * incoming message). If this is the case the error should be reported in
 * the corresponding callback and QP should be reset back to normal. Otherwise
 * treat the error as fatal. */
static UCS_F_NOINLINE void
uct_rc_verbs_iface_wc_error(uct_rc_verbs_iface_t *iface, int status)
{
    /* TODO: handle MSG TRUNCATED error */
    ucs_fatal("Receive completion with error on XRQ: %s",
              ibv_wc_status_str(status));
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_verbs_iface_post_op(uct_rc_verbs_iface_t *iface, struct ibv_exp_ops_wr *wr,
                           struct ibv_sge *sge, size_t sge_cnt, int op,
                           uct_tag_context_t *ctx, int flags)
{
    struct ibv_exp_ops_wr *bad_wr;
    int ret;

    UCT_RC_VERBS_FILL_TM_OP_WR(iface, wr, sge, sge_cnt, op, ctx, flags);

    ret = ibv_exp_post_srq_ops(iface->tm.xrq.srq, wr, &bad_wr);
    if (ret) {
        ucs_error("ibv_exp_post_srq_ops(op=%d) failed: %m (%d)", op, ret);
        return UCS_ERR_IO_ERROR;
    }
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_iface_tag_handle_exp(uct_rc_verbs_iface_t *iface,
                                  struct ibv_exp_wc *wc)
{
    uct_tag_context_t *ctx        = (uct_tag_context_t*)wc->wr_id;
    uct_rc_verbs_ctx_priv_t *priv = uct_rc_verbs_iface_ctx_priv(ctx);

    if (wc->exp_wc_flags & IBV_EXP_WC_TM_MATCH) {
        /* Need to keep app_ctx in case DATA will come with immediate */
        priv->imm_data = wc->tm_info.priv;
        priv->tag      = wc->tm_info.tag;
        ctx->tag_consumed_cb(ctx);
    }

    if (wc->exp_wc_flags & IBV_EXP_WC_TM_DATA_VALID) {
        priv->imm_data = uct_rc_verbs_tag_imm_data_unpack(wc, priv->imm_data);
        if (UCT_RC_VERBS_TM_IS_SW_RNDV(wc->exp_wc_flags, priv->imm_data)) {
            ctx->rndv_cb(ctx, priv->tag, priv->buffer, wc->byte_len, UCS_OK);
        } else {
            ctx->completed_cb(ctx, priv->tag, priv->imm_data,
                              wc->byte_len, UCS_OK);
        }
        ++iface->tm.tag_available;
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_iface_unexp_consumed(uct_rc_verbs_iface_t *iface,
                                  uct_ib_iface_recv_desc_t *ib_desc,
                                  uct_rc_verbs_release_desc_t *release,
                                  ucs_status_t status)
{
    struct ibv_exp_ops_wr wr;
    void *udesc;

    if (status == UCS_OK) {
        ucs_mpool_put_inline(ib_desc);
    } else {
        udesc = (char*)ib_desc + release->offset;
        uct_recv_desc(udesc) = &release->super;
    }

    if (ucs_unlikely(!(++iface->tm.unexpected_cnt % IBV_DEVICE_MAX_UNEXP_COUNT))) {
        uct_rc_verbs_iface_post_op(iface, &wr, NULL, 0, IBV_EXP_WR_TAG_SYNC,
                                   NULL, 0);
    }
    ++iface->tm.xrq.available;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_iface_tag_handle_unexp(uct_rc_verbs_iface_t *iface,
                                    struct ibv_exp_wc *wc)
{
    uct_ib_md_t *ib_md = uct_ib_iface_md(&iface->super.super);
    uct_ib_iface_recv_desc_t *ib_desc = (uct_ib_iface_recv_desc_t*)(uintptr_t)wc->wr_id;
    struct ibv_exp_tmh *tmh = uct_ib_iface_recv_desc_hdr(&iface->super.super, ib_desc);
    uct_rc_hdr_t *rc_hdr;
    uint64_t imm_data;
    ucs_status_t status;
    void *rb;
    int found;
    uct_completion_t *rndv_comp;
    struct ibv_exp_tmh_rvh *rvh;

    VALGRIND_MAKE_MEM_DEFINED(tmh, wc->byte_len);

    switch (tmh->opcode) {
    case IBV_EXP_TMH_EAGER:
        imm_data = uct_rc_verbs_tag_imm_data_unpack(wc, ntohl(tmh->app_ctx));

        if (UCT_RC_VERBS_TM_IS_SW_RNDV(wc->exp_wc_flags, imm_data)) {
            status = iface->tm.rndv_unexp.cb(iface->tm.rndv_unexp.arg,
                                             UCT_CB_FLAG_DESC, be64toh(tmh->tag), tmh + 1,
                                             wc->byte_len - sizeof(*tmh), 0ul, 0,
                                             NULL);
        } else {
            status = iface->tm.eager_unexp.cb(iface->tm.eager_unexp.arg,
                                              tmh + 1, wc->byte_len - sizeof(*tmh),
                                              UCT_CB_FLAG_DESC, be64toh(tmh->tag), imm_data);
        }
        uct_rc_verbs_iface_unexp_consumed(iface, ib_desc,
                                          &iface->tm.eager_desc, status);
        break;

    case IBV_EXP_TMH_NO_TAG:
        rc_hdr = (uct_rc_hdr_t*)((char*)tmh + iface->verbs_common.config.notag_hdr_size);
        uct_ib_log_recv_completion(&iface->super.super, IBV_QPT_RC, wc, rc_hdr,
                                   wc->byte_len - iface->verbs_common.config.notag_hdr_size,
                                   uct_rc_ep_am_packet_dump);
        uct_rc_verbs_iface_handle_am(&iface->super, rc_hdr, wc->wr_id, wc->qp_num,
                                     wc->byte_len - iface->verbs_common.config.notag_hdr_size,
                                     wc->imm_data, wc->slid);
        ++iface->tm.xrq.available;
        break;

    case IBV_EXP_TMH_RNDV:
        rvh = (struct ibv_exp_tmh_rvh*)(tmh + 1);
        /* Create "packed" rkey to pass it in the callback */
        rb = uct_md_fill_md_name(&ib_md->super, (char*)tmh + wc->byte_len);
        uct_ib_md_pack_rkey(ntohl(rvh->rkey), UCT_IB_INVALID_RKEY, rb);

        status = iface->tm.rndv_unexp.cb(iface->tm.rndv_unexp.arg, UCT_CB_FLAG_DESC,
                                         be64toh(tmh->tag), rvh + 1, wc->byte_len -
                                         (sizeof(*tmh) + sizeof(*rvh)), be64toh(rvh->va),
                                         ntohl(rvh->len), (char*)tmh + wc->byte_len);

        uct_rc_verbs_iface_unexp_consumed(iface, ib_desc,
                                          &iface->tm.rndv_desc, status);
        break;

    case IBV_EXP_TMH_FIN:
        found = ucs_ptr_array_lookup(&iface->tm.rndv_comps, ntohl(tmh->app_ctx),
                                     rndv_comp);
        ucs_assert_always(found > 0);
        uct_invoke_completion(rndv_comp, UCS_OK);
        ucs_ptr_array_remove(&iface->tm.rndv_comps, ntohl(tmh->app_ctx), 0);
        ucs_mpool_put_inline(ib_desc);
        ++iface->super.rx.srq.available;
        break;

    default:
        ucs_fatal("Unsupported packet arrived %d", tmh->opcode);
        break;
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_verbs_iface_poll_rx_tm(uct_rc_verbs_iface_t *iface)
{
    unsigned num_wcs    = iface->super.super.config.rx_max_poll;
    ucs_status_t status = UCS_OK;
    struct ibv_exp_wc wc[num_wcs];
    uct_tag_context_t *ctx;
    uct_rc_verbs_ctx_priv_t *priv;
    int i;

    num_wcs = ibv_exp_poll_cq(iface->super.super.recv_cq, num_wcs, wc,
                              sizeof(wc[0]));
    if (num_wcs <= 0) {
        if (ucs_likely(num_wcs == 0)) {
            status = UCS_ERR_NO_PROGRESS;
            goto out;
        }
        ucs_fatal("Failed to poll receive CQ %d", num_wcs);
    }

    for (i = 0; i < num_wcs; ++i) {
        if (ucs_unlikely(wc[i].status != IBV_WC_SUCCESS)) {
            uct_rc_verbs_iface_wc_error(iface, wc[i].status);
            continue;
        }

        switch (wc[i].exp_opcode) {
        case IBV_EXP_WC_TM_NO_TAG:
        case IBV_EXP_WC_RECV:
            uct_rc_verbs_iface_tag_handle_unexp(iface, &wc[i]);
            break;

        case IBV_EXP_WC_TM_RECV:
            if (wc[i].exp_wc_flags & (IBV_EXP_WC_TM_MATCH | IBV_EXP_WC_TM_DATA_VALID)) {
                uct_rc_verbs_iface_tag_handle_exp(iface, &wc[i]);
            } else {
                uct_rc_verbs_iface_tag_handle_unexp(iface, &wc[i]);
            }
            break;

        case IBV_EXP_WC_TM_DEL:
            ctx  = (uct_tag_context_t*)wc[i].wr_id;
            priv = uct_rc_verbs_iface_ctx_priv(ctx);
            ctx->completed_cb(ctx, priv->tag, 0, priv->length,
                              UCS_ERR_CANCELED);
            ++iface->tm.tag_available;
            break;

        case IBV_EXP_WC_TM_ADD:
            ++iface->tm.num_outstanding;
            break;

        default:
            ucs_error("Wrong opcode in CQE %d", wc[i].exp_opcode);
            break;

        }
    }
    /* TODO: Add stat */
out:
    /* All tag unexpected and AM messages arrive to XRQ */
    uct_rc_verbs_iface_post_recv_common(&iface->super, &iface->tm.xrq, 0);

    /* Only RNDV FIN messages arrive to SRQ (sent by FW) */
    uct_rc_verbs_iface_post_recv_common(&iface->super, &iface->super.rx.srq, 0);
    return status;
}

void uct_rc_verbs_iface_progress_tm(void *arg)
{
    uct_rc_verbs_iface_t *iface = arg;
    ucs_status_t status;

    status = uct_rc_verbs_iface_poll_rx_tm(iface);
    if (status == UCS_ERR_NO_PROGRESS) {
        uct_rc_verbs_iface_poll_tx(iface);
    }
}

static ucs_status_t uct_rc_verbs_iface_tag_recv_zcopy(uct_iface_h tl_iface,
                                                      uct_tag_t tag,
                                                      uct_tag_t tag_mask,
                                                      const uct_iov_t *iov,
                                                      size_t iovcnt,
                                                      uct_tag_context_t *ctx)
{
    uct_rc_verbs_ctx_priv_t *priv = (uct_rc_verbs_ctx_priv_t*)ctx->priv;
    uct_rc_verbs_iface_t *iface   = ucs_derived_of(tl_iface,
                                                   uct_rc_verbs_iface_t);
    ucs_status_t status;
    struct ibv_sge sge[UCT_IB_MAX_IOV];
    struct ibv_exp_ops_wr wr;
    size_t sge_cnt;

    UCT_CHECK_IOV_SIZE(iovcnt, 1ul, "uct_rc_verbs_iface_tag_recv_zcopy");
    UCT_RC_VERBS_CHECK_TAG(iface);

    sge_cnt        = uct_ib_verbs_sge_fill_iov(sge, iov, iovcnt);
    wr.tm.add.tag  = tag;
    wr.tm.add.mask = tag_mask;

    status = uct_rc_verbs_iface_post_op(iface, &wr, sge, sge_cnt,
                                        IBV_EXP_WR_TAG_ADD, ctx,
                                        IBV_EXP_OPS_SIGNALED);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    --iface->tm.tag_available;
    --iface->tm.num_outstanding;

    /* Save tag index in the device tags list returned by ibv_exp_post_srq_ops.
     * It may be needed for cancelling this posted tag. */
    priv->tag_handle = wr.tm.handle;
    priv->tag        = tag;
    priv->buffer     = iov->buffer; /* Only one iov is supported so far */
    priv->length     = uct_iov_total_length(iov, iovcnt);
    return UCS_OK;
}

static ucs_status_t uct_rc_verbs_iface_tag_recv_cancel(uct_iface_h tl_iface,
                                                       uct_tag_context_t *ctx,
                                                       int force)
{
   uct_rc_verbs_ctx_priv_t *priv = (uct_rc_verbs_ctx_priv_t*)ctx->priv;
   uct_rc_verbs_iface_t *iface   = ucs_derived_of(tl_iface,
                                                  uct_rc_verbs_iface_t);
   struct ibv_exp_ops_wr wr;
   int flags;

   wr.tm.handle = priv->tag_handle;

   if (force) {
       flags = 0;
       /* We will not get completion, so free one tag immediately */
       ++iface->tm.tag_available;
   } else {
       flags    = IBV_EXP_OPS_SIGNALED;
       wr.wr_id = (uint64_t)ctx; \
   }
   return uct_rc_verbs_iface_post_op(iface, &wr, NULL, 0, IBV_EXP_WR_TAG_DEL,
                                     ctx, flags);
}

static void uct_rc_verbs_iface_release_desc(uct_recv_desc_t *self, void *desc)
{
    uct_rc_verbs_release_desc_t *release = ucs_derived_of(self,
                                                          uct_rc_verbs_release_desc_t);
    void *ib_desc = desc - release->offset;
    ucs_mpool_put_inline(ib_desc);
}
#endif /* IBV_EXP_HW_TM */

static ucs_status_t uct_rc_verbs_iface_tag_init(uct_rc_verbs_iface_t *iface,
                                                uct_rc_verbs_iface_config_t *config)
{
#if IBV_EXP_HW_TM
    struct ibv_exp_create_srq_attr srq_init_attr;
    ucs_status_t status;
    int rx_hdr_len;
    uct_ib_md_t *md = ucs_derived_of(iface->super.super.super.md, uct_ib_md_t);

    if (UCT_RC_VERBS_TM_ENABLED(iface)) {
        /* Create XRQ with TM capability */
        memset(&srq_init_attr, 0, sizeof(srq_init_attr));
        srq_init_attr.base.attr.max_sge   = 1;
        srq_init_attr.base.attr.max_wr    = ucs_max(UCT_RC_VERBS_TAG_MIN_POSTED,
                                                    config->super.super.rx.queue_len);
        srq_init_attr.base.attr.srq_limit = 0;
        srq_init_attr.base.srq_context    = iface;
        srq_init_attr.srq_type            = IBV_EXP_SRQT_TAG_MATCHING;
        srq_init_attr.pd                  = md->pd;
        srq_init_attr.cq                  = iface->super.super.recv_cq;
        srq_init_attr.tm_cap.max_num_tags = iface->tm.tag_available;

        /* 2 ops for each tag (ADD + DEL) and extra tag for SYNC ops.
         * Do not rely on max_tag_ops device capability, so that XRQ creation
         * would fail if such # of ops is not supported (which is not expected).
         * TODO: Check that we have only outstanding sync operation at time.
         * We should stop preposting new unexp buffers if need to post SYNC and
         * the previous one is not completed yet. */
        srq_init_attr.tm_cap.max_ops      = 2 * iface->tm.tag_available + 1;
        srq_init_attr.comp_mask           = IBV_EXP_CREATE_SRQ_CQ |
                                            IBV_EXP_CREATE_SRQ_TM;

        iface->tm.xrq.srq = ibv_exp_create_srq(md->dev.ibv_context, &srq_init_attr);
        if (iface->tm.xrq.srq == NULL) {
            ucs_error("Failed to create TM XRQ: %m");
            return UCS_ERR_IO_ERROR;
        }

        /* Only ADD operations decrease num_outstanding counter. So, limit num_outstanding
         * to the number of tags, DEL and SYNC operations should be always successful. */
        iface->tm.num_outstanding = iface->tm.tag_available;

        iface->tm.xrq.available = srq_init_attr.base.attr.max_wr;
        --iface->tm.tag_available; /* 1 tag should be always available in HW match list */

        /* AM (NO_TAG) and eager messages have different header sizes.
         * Receive descriptor offsets are calculated based on AM hdr length.
         * Need to store headers difference for correct release of descriptors
         * consumed by unexpected eager messages. */
        rx_hdr_len = iface->super.super.config.rx_payload_offset -
                     iface->super.super.config.rx_hdr_offset;
        ucs_assert_always(sizeof(struct ibv_exp_tmh) >= rx_hdr_len);
        iface->tm.eager_desc.super.cb = uct_rc_verbs_iface_release_desc;
        iface->tm.eager_desc.offset   = sizeof(struct ibv_exp_tmh) - rx_hdr_len +
                                        iface->super.super.config.rx_headroom_offset;

        iface->tm.rndv_desc.super.cb = uct_rc_verbs_iface_release_desc;
        iface->tm.rndv_desc.offset   = iface->tm.eager_desc.offset +
                                       sizeof(struct ibv_exp_tmh_rvh);

        status = uct_rc_verbs_iface_prepost_recvs_common(&iface->super, &iface->tm.xrq);
        if (status != UCS_OK) {
            ibv_destroy_srq(iface->tm.xrq.srq);
            return status;
        }

        /* Init ptr array to store completions of RNDV operations. Index in
         * ptr_array is used as operation ID and is passed in "app_context"
         * of TM header. */
        ucs_ptr_array_init(&iface->tm.rndv_comps, 0, "rm_rndv_completions");
    }
#endif
    return UCS_OK;
}

static void uct_rc_verbs_iface_tag_preinit(uct_rc_verbs_iface_t *iface,
                                           uct_md_h md,
                                           uct_rc_verbs_iface_config_t *config,
                                           const uct_iface_params_t *params,
                                           unsigned *rx_cq_len,
                                           unsigned *srq_size,
                                           unsigned *rx_hdr_len,
                                           unsigned *short_mp_size)
{
#if IBV_EXP_HW_TM
    uct_ib_md_t *ib_md   = ucs_derived_of(md, uct_ib_md_t);
    uct_ib_device_t *dev = &ib_md->dev;
    struct ibv_exp_tmh tmh;

    iface->tm.enabled = UCT_RC_VERBS_TM_CONFIG(config, enable);

    if (IBV_DEVICE_TM_CAPS(dev, max_num_tags) &&
        UCT_RC_VERBS_TM_CONFIG(config, enable)) {

        UCS_STATIC_ASSERT(sizeof(uct_rc_verbs_ctx_priv_t) <= UCT_TAG_PRIV_LEN);

        iface->progress            = uct_rc_verbs_iface_progress_tm;
        iface->tm.eager_unexp.cb   = params->eager_cb;
        iface->tm.eager_unexp.arg  = params->eager_arg;
        iface->tm.rndv_unexp.cb    = params->rndv_cb;
        iface->tm.rndv_unexp.arg   = params->rndv_arg;
        iface->tm.unexpected_cnt   = 0;
        iface->tm.tag_available    = ucs_min(IBV_DEVICE_TM_CAPS(dev, max_num_tags),
                                             UCT_RC_VERBS_TM_CONFIG(config, list_size));

        /* There can be up to 3 CQEs for every posted tag: ADD, TM_CONSUMED
         * and MSG_ARRIVED. */
        *rx_cq_len     = config->super.super.rx.queue_len + iface->tm.tag_available * 2;
        *srq_size      = UCT_RC_VERBS_TM_CONFIG(config, rndv_queue_len);
        /* Only opcode (rather than the whole TMH) is sent with NO_TAG protocol */
        *rx_hdr_len    = sizeof(uct_rc_hdr_t) + sizeof(tmh.opcode);
        *short_mp_size = ucs_max(*rx_hdr_len, sizeof(struct ibv_exp_tmh));

        ucs_debug("Tag Matching enabled: tag list size %d", iface->tm.tag_available);
    } else
#endif
    {
        iface->progress = uct_rc_verbs_iface_progress;
        *rx_hdr_len     = *short_mp_size = sizeof(uct_rc_hdr_t);
        *rx_cq_len      = *srq_size = config->super.super.rx.queue_len;
    }
}

static void uct_rc_verbs_iface_tag_cleanup(uct_rc_verbs_iface_t *iface)
{
#if IBV_EXP_HW_TM
    if (UCT_RC_VERBS_TM_ENABLED(iface)) {
        if (ibv_destroy_srq(iface->tm.xrq.srq)) {
            ucs_warn("failed to destroy TM XRQ: %m");
        }
        ucs_ptr_array_cleanup(&iface->tm.rndv_comps);
    }
#endif
}

static void uct_rc_verbs_iface_init_inl_wrs(uct_rc_verbs_iface_t *iface)
{
    iface->verbs_common.config.notag_hdr_size =
        uct_rc_verbs_notag_header_fill(iface, iface->verbs_common.am_inl_hdr);

    memset(&iface->inl_am_wr, 0, sizeof(iface->inl_am_wr));
    iface->inl_am_wr.sg_list        = iface->verbs_common.inl_sge;
    iface->inl_am_wr.num_sge        = 2;
    iface->inl_am_wr.opcode         = IBV_WR_SEND;
    iface->inl_am_wr.send_flags     = IBV_SEND_INLINE;

    memset(&iface->inl_rwrite_wr, 0, sizeof(iface->inl_rwrite_wr));
    iface->inl_rwrite_wr.sg_list    = iface->verbs_common.inl_sge;
    iface->inl_rwrite_wr.num_sge    = 1;
    iface->inl_rwrite_wr.opcode     = IBV_WR_RDMA_WRITE;
    iface->inl_rwrite_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
}

void uct_rc_verbs_iface_tag_query(uct_rc_verbs_iface_t *iface,
                                  uct_iface_attr_t *iface_attr)
{
#if IBV_EXP_HW_TM
    uct_ib_iface_t *ib_iface = &iface->super.super;
    uct_ib_device_t *dev     = uct_ib_iface_device(ib_iface);
    unsigned eager_hdr_size  = sizeof(struct ibv_exp_tmh);

    if (UCT_RC_VERBS_TM_ENABLED(iface)) {
        iface_attr->ep_addr_len = sizeof(uct_rc_verbs_ep_tm_address_t);

        /* Redefine AM caps, because we have to send TMH (with NO_TAG
         * operation) with every AM message. */
        iface_attr->cap.am.max_short -= iface->verbs_common.config.notag_hdr_size;
        if (iface_attr->cap.am.max_short <= 0) {
            iface_attr->cap.am.max_short = 0;
            iface_attr->cap.flags &= ~UCT_IFACE_FLAG_AM_SHORT;
        }

        iface_attr->cap.am.max_bcopy -= iface->verbs_common.config.notag_hdr_size;
        iface_attr->cap.am.max_zcopy -= iface->verbs_common.config.notag_hdr_size;
        iface_attr->cap.am.max_hdr   -= iface->verbs_common.config.notag_hdr_size;

        iface_attr->latency.growth   += 3e-9; /* + 3ns for TM QP */

        iface_attr->cap.flags        |= UCT_IFACE_FLAG_TAG_EAGER_BCOPY |
                                        UCT_IFACE_FLAG_TAG_EAGER_ZCOPY |
                                        UCT_IFACE_FLAG_TAG_RNDV_ZCOPY;

        iface_attr->cap.tag.eager.max_short =
        ucs_max(0, iface->verbs_common.config.max_inline - eager_hdr_size);

        if (iface_attr->cap.tag.eager.max_short > 0 ) {
            iface_attr->cap.flags |= UCT_IFACE_FLAG_TAG_EAGER_SHORT;
        }

        iface_attr->cap.tag.eager.max_bcopy = ib_iface->config.seg_size - eager_hdr_size;
        iface_attr->cap.tag.eager.max_zcopy = ib_iface->config.seg_size - eager_hdr_size;
        iface_attr->cap.tag.rndv.max_zcopy  = uct_ib_iface_port_attr(ib_iface)->max_msg_sz;
        iface_attr->cap.tag.rndv.max_hdr    = IBV_DEVICE_TM_CAPS(dev, max_rndv_hdr_size);

        iface_attr->cap.tag.eager.max_iov   = 1;
        iface_attr->cap.tag.rndv.max_iov    = 1;
        iface_attr->cap.tag.recv.max_iov    = 1;
        iface_attr->cap.tag.recv.min_recv   = 0;
    }
#endif

#if (HAVE_IBV_EXP_QP_CREATE_UMR_CAPS || HAVE_EXP_UMR_NEW_API)
    iface_attr->cap.flags |= UCT_IFACE_FLAG_MEM_NC;
#endif
}

static ucs_status_t uct_rc_verbs_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_verbs_iface_t);

    uct_rc_iface_query(&iface->super, iface_attr);
    uct_rc_verbs_iface_common_query(&iface->verbs_common, &iface->super, iface_attr);
    iface_attr->latency.growth += 3e-9; /* 3ns per each extra QP */

    uct_rc_verbs_iface_tag_query(iface, iface_attr);

    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_rc_verbs_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_rc_verbs_iface_config_t *config =
                    ucs_derived_of(tl_config, uct_rc_verbs_iface_config_t);
    ucs_status_t status;
    struct ibv_qp_cap cap;
    struct ibv_qp *qp;
    unsigned srq_size;
    unsigned rx_hdr_len;
    unsigned short_mp_size;
    unsigned rx_cq_len;
    size_t max_hdr_len;

    uct_rc_verbs_iface_tag_preinit(self, md, config, params, &rx_cq_len, &srq_size,
                                   &rx_hdr_len, &short_mp_size);

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, &uct_rc_verbs_iface_ops, md,
                              worker, params, &config->super, 0, rx_cq_len,
                              rx_hdr_len, srq_size, sizeof(uct_rc_fc_request_t));

    self->config.tx_max_wr           = ucs_min(config->verbs_common.tx_max_wr,
                                               self->super.config.tx_qp_len);
    self->super.config.tx_moderation = ucs_min(self->super.config.tx_moderation,
                                               self->config.tx_max_wr / 4);

    max_hdr_len = ucs_max(config->verbs_common.max_am_hdr, short_mp_size);
    status = uct_rc_verbs_iface_common_init(&self->verbs_common, &self->super,
                                            &config->verbs_common, &config->super,
                                            max_hdr_len);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_rc_verbs_iface_tag_init(self, config);
    if (status != UCS_OK) {
        goto err_common_cleanup;
    }

    uct_rc_verbs_iface_init_inl_wrs(self);

    /* Check FC parameters correctness */
    status = uct_rc_init_fc_thresh(&config->fc, &config->super, &self->super);
    if (status != UCS_OK) {
        goto err_tag_cleanup;
    }

    /* Create a dummy QP in order to find out max_inline */
    status = uct_rc_iface_qp_create(&self->super, IBV_QPT_RC, &qp, &cap,
                                    self->super.rx.srq.srq,
                                    self->super.config.tx_qp_len);
    if (status != UCS_OK) {
        goto err_tag_cleanup;
    }
    ibv_destroy_qp(qp);

    self->verbs_common.config.max_inline   = cap.max_inline_data;
    uct_ib_iface_set_max_iov(&self->super.super, cap.max_send_sge);

    status = uct_rc_verbs_iface_prepost_recvs_common(&self->super,
                                                     &self->super.rx.srq);
    if (status != UCS_OK) {
        goto err_tag_cleanup;
    }
    return UCS_OK;

err_tag_cleanup:
    uct_rc_verbs_iface_tag_cleanup(self);
err_common_cleanup:
    uct_rc_verbs_iface_common_cleanup(&self->verbs_common);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_verbs_iface_t)
{
    uct_rc_verbs_iface_common_cleanup(&self->verbs_common);
    uct_rc_verbs_iface_tag_cleanup(self);
}

UCS_CLASS_DEFINE(uct_rc_verbs_iface_t, uct_rc_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_verbs_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_verbs_iface_t, uct_iface_t);

static uct_rc_iface_ops_t uct_rc_verbs_iface_ops = {
    {
    {
    .ep_am_short              = uct_rc_verbs_ep_am_short,
    .ep_am_bcopy              = uct_rc_verbs_ep_am_bcopy,
    .ep_am_zcopy              = uct_rc_verbs_ep_am_zcopy,
    .ep_put_short             = uct_rc_verbs_ep_put_short,
    .ep_put_bcopy             = uct_rc_verbs_ep_put_bcopy,
    .ep_put_zcopy             = uct_rc_verbs_ep_put_zcopy,
    .ep_get_bcopy             = uct_rc_verbs_ep_get_bcopy,
    .ep_get_zcopy             = uct_rc_verbs_ep_get_zcopy,
    .ep_atomic_add64          = uct_rc_verbs_ep_atomic_add64,
    .ep_atomic_fadd64         = uct_rc_verbs_ep_atomic_fadd64,
    .ep_atomic_swap64         = uct_rc_verbs_ep_atomic_swap64,
    .ep_atomic_cswap64        = uct_rc_verbs_ep_atomic_cswap64,
    .ep_atomic_add32          = uct_rc_verbs_ep_atomic_add32,
    .ep_atomic_fadd32         = uct_rc_verbs_ep_atomic_fadd32,
    .ep_atomic_swap32         = uct_rc_verbs_ep_atomic_swap32,
    .ep_atomic_cswap32        = uct_rc_verbs_ep_atomic_cswap32,
    .ep_pending_add           = uct_rc_ep_pending_add,
    .ep_pending_purge         = uct_rc_ep_pending_purge,
    .ep_flush                 = uct_rc_verbs_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_rc_verbs_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_verbs_ep_t),
    .ep_get_address           = uct_rc_verbs_ep_get_address,
    .ep_connect_to_ep         = uct_rc_verbs_ep_connect_to_ep,
    .ep_mem_reg_nc            = uct_rc_ep_reg_nc,
    .iface_flush              = uct_rc_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
#if IBV_EXP_HW_TM
    .iface_tag_recv_zcopy     = uct_rc_verbs_iface_tag_recv_zcopy,
    .iface_tag_recv_cancel    = uct_rc_verbs_iface_tag_recv_cancel,
    .ep_tag_eager_short       = uct_rc_verbs_ep_tag_eager_short,
    .ep_tag_eager_bcopy       = uct_rc_verbs_ep_tag_eager_bcopy,
    .ep_tag_eager_zcopy       = uct_rc_verbs_ep_tag_eager_zcopy,
    .ep_tag_rndv_zcopy        = uct_rc_verbs_ep_tag_rndv_zcopy,
    .ep_tag_rndv_cancel       = uct_rc_verbs_ep_tag_rndv_cancel,
    .ep_tag_rndv_request      = uct_rc_verbs_ep_tag_rndv_request,
#endif
    .iface_event_fd_get       = uct_ib_iface_event_fd_get,
    .iface_event_arm          = uct_ib_iface_event_arm,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_verbs_iface_t),
    .iface_query              = uct_rc_verbs_iface_query,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_ib_iface_is_reachable
    },
    .arm_tx_cq                = uct_ib_iface_arm_tx_cq,
    .arm_rx_cq                = uct_ib_iface_arm_rx_cq,
    .handle_failure           = uct_rc_verbs_handle_failure,
    .set_ep_failed            = uct_rc_verbs_ep_set_failed
    },
    .fc_ctrl                  = uct_rc_verbs_ep_fc_ctrl,
    .fc_handler               = uct_rc_iface_fc_handler
};

static ucs_status_t uct_rc_verbs_query_resources(uct_md_h md,
                                                 uct_tl_resource_desc_t **resources_p,
                                                 unsigned *num_resources_p)
{
    uct_ib_md_t *ib_md = ucs_derived_of(md, uct_ib_md_t);

    return uct_ib_device_query_tl_resources(&ib_md->dev, "rc",
                                            (ib_md->config.eth_pause ? 0 : UCT_IB_DEVICE_FLAG_LINK_IB),
                                            resources_p, num_resources_p);
}

UCT_TL_COMPONENT_DEFINE(uct_rc_verbs_tl,
                        uct_rc_verbs_query_resources,
                        uct_rc_verbs_iface_t,
                        "rc",
                        "RC_VERBS_",
                        uct_rc_verbs_iface_config_table,
                        uct_rc_verbs_iface_config_t);
UCT_MD_REGISTER_TL(&uct_ib_mdc, &uct_rc_verbs_tl);
