/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "rc_verbs_common.h"

#include <uct/api/uct.h>
#include <ucs/arch/bitops.h>
#include <uct/ib/rc/base/rc_iface.h>


ucs_config_field_t uct_rc_verbs_iface_common_config_table[] = {
  {"MAX_AM_HDR", "128",
   "Buffer size to reserve for active message headers. If set to 0, the transport will\n"
   "not support zero-copy active messages.",
   ucs_offsetof(uct_rc_verbs_iface_common_config_t, max_am_hdr), UCS_CONFIG_TYPE_MEMUNITS},

  {"TX_MAX_WR", "-1",
   "Limits the number of outstanding posted work requests. The actual limit is\n"
   "a minimum between this value and the TX queue length. -1 means no limit.",
   ucs_offsetof(uct_rc_verbs_iface_common_config_t, tx_max_wr), UCS_CONFIG_TYPE_UINT},

#if IBV_EXP_HW_TM
  {"TM_ENABLE", "y",
   "Enable HW tag matching",
   ucs_offsetof(uct_rc_verbs_iface_common_config_t, tm.enable), UCS_CONFIG_TYPE_BOOL},

  {"TM_LIST_SIZE", "1024",
   "Limits the number of tags posted to the HW for matching. The actual limit \n"
   "is a minimum between this value and the maximum value supported by the HW. \n"
   "-1 means no limit.",
   ucs_offsetof(uct_rc_verbs_iface_common_config_t, tm.list_size), UCS_CONFIG_TYPE_UINT},

  {"TM_SYNC_RATIO", "0.5",
   "Maximal portion of the tag matching list which can be canceled without requesting\n"
   "a completion.",
   ucs_offsetof(uct_rc_verbs_iface_common_config_t, tm.sync_ratio), UCS_CONFIG_TYPE_DOUBLE},
#endif

  {NULL}
};

static void uct_rc_verbs_iface_common_tag_query(uct_rc_verbs_iface_common_t *iface,
                                                uct_ib_iface_t *ib_iface,
                                                uct_iface_attr_t *iface_attr)
{
#if IBV_EXP_HW_TM
    uct_ib_device_t *dev     = uct_ib_iface_device(ib_iface);
    unsigned eager_hdr_size  = sizeof(struct ibv_exp_tmh);

    if (!iface->tm.enabled) {
        return;
    }

    /* Redefine AM caps, because we have to send TMH (with NO_TAG
     * operation) with every AM message. */
    iface_attr->cap.am.max_short -= iface->config.notag_hdr_size;
    if (iface_attr->cap.am.max_short <= 0) {
        iface_attr->cap.am.max_short = 0;
        iface_attr->cap.flags &= ~UCT_IFACE_FLAG_AM_SHORT;
    }

    iface_attr->cap.am.max_bcopy -= iface->config.notag_hdr_size;
    iface_attr->cap.am.max_zcopy -= iface->config.notag_hdr_size;
    iface_attr->cap.am.max_hdr   -= iface->config.notag_hdr_size;

    iface_attr->latency.growth   += 3e-9; /* + 3ns for TM QP */

    iface_attr->cap.flags        |= UCT_IFACE_FLAG_TAG_EAGER_BCOPY |
                                    UCT_IFACE_FLAG_TAG_EAGER_ZCOPY |
                                    UCT_IFACE_FLAG_TAG_RNDV_ZCOPY;

    iface_attr->cap.tag.eager.max_short =
        ucs_max(0, iface->config.max_inline - eager_hdr_size);

    if (iface_attr->cap.tag.eager.max_short > 0 ) {
        iface_attr->cap.flags |= UCT_IFACE_FLAG_TAG_EAGER_SHORT;
    }

    iface_attr->cap.tag.eager.max_bcopy = ib_iface->config.seg_size - eager_hdr_size;
    iface_attr->cap.tag.eager.max_zcopy = ib_iface->config.seg_size - eager_hdr_size;
    iface_attr->cap.tag.eager.max_iov   = 1;

    iface_attr->cap.tag.rndv.max_zcopy  = uct_ib_iface_port_attr(ib_iface)->max_msg_sz;
    iface_attr->cap.tag.rndv.max_hdr    = IBV_DEVICE_TM_CAPS(dev, max_rndv_hdr_size);
    iface_attr->cap.tag.rndv.max_iov    = 1;

    iface_attr->cap.tag.recv.max_zcopy  = uct_ib_iface_port_attr(ib_iface)->max_msg_sz;
    iface_attr->cap.tag.recv.max_iov    = 1;
    iface_attr->cap.tag.recv.min_recv   = 0;
#endif
}

void uct_rc_verbs_iface_common_query(uct_rc_verbs_iface_common_t *verbs_iface,
                                     uct_rc_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    /* PUT */
    iface_attr->cap.put.max_short = verbs_iface->config.max_inline;
    iface_attr->cap.put.max_bcopy = iface->super.config.seg_size;
    iface_attr->cap.put.min_zcopy = 0;
    iface_attr->cap.put.max_zcopy = uct_ib_iface_port_attr(&iface->super)->max_msg_sz;
    iface_attr->cap.put.max_iov   = uct_ib_iface_get_max_iov(&iface->super);

    /* GET */
    iface_attr->cap.get.max_bcopy = iface->super.config.seg_size;
    iface_attr->cap.get.min_zcopy = iface->super.config.max_inl_resp + 1;
    iface_attr->cap.get.max_zcopy = uct_ib_iface_port_attr(&iface->super)->max_msg_sz;
    iface_attr->cap.get.max_iov   = uct_ib_iface_get_max_iov(&iface->super);

    /* AM */
    iface_attr->cap.am.max_short  = verbs_iface->config.max_inline - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.max_bcopy  = iface->super.config.seg_size - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.min_zcopy  = 0;
    iface_attr->cap.am.max_zcopy  = iface->super.config.seg_size - sizeof(uct_rc_hdr_t);
    /* The first IOV is reserved for the header */
    iface_attr->cap.am.max_iov    = uct_ib_iface_get_max_iov(&iface->super) - 1;

    /* TODO: may need to change for dc/rc */
    iface_attr->cap.am.max_hdr    = verbs_iface->config.short_desc_size - sizeof(uct_rc_hdr_t);

    iface_attr->cap.flags        |= UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;

    /* Software overhead */
    iface_attr->overhead          = 75e-9;

    /* TAG Offload */
    uct_rc_verbs_iface_common_tag_query(verbs_iface, &iface->super, iface_attr);
}

unsigned uct_rc_verbs_iface_post_recv_always(uct_rc_iface_t *iface, unsigned max)
{
    struct ibv_recv_wr *bad_wr;
    uct_ib_recv_wr_t *wrs;
    unsigned count;
    int ret;

    wrs  = ucs_alloca(sizeof *wrs  * max);

    count = uct_ib_iface_prepare_rx_wrs(&iface->super, &iface->rx.mp,
                                        wrs, max);
    if (ucs_unlikely(count == 0)) {
        return 0;
    }

    ret = ibv_post_srq_recv(iface->rx.srq.srq, &wrs[0].ibwr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_srq_recv() returned %d: %m", ret);
    }
    iface->rx.srq.available -= count;

    return count;
}

ucs_status_t uct_rc_verbs_iface_prepost_recvs_common(uct_rc_iface_t *iface)
{
    while (iface->rx.srq.available > 0) {
        if (uct_rc_verbs_iface_post_recv_common(iface, 1) == 0) {
            ucs_error("failed to post receives");
            return UCS_ERR_NO_MEMORY;
        }
    }
    return UCS_OK;
}

void uct_rc_verbs_iface_common_progress_enable(uct_rc_verbs_iface_common_t *iface,
                                               uct_rc_iface_t *rc_iface,
                                               unsigned flags)
{
    uct_base_iface_progress_enable_cb(&rc_iface->super.super, iface->progress,
                                      flags);
}

#if IBV_EXP_HW_TM

static void uct_rc_verbs_iface_release_desc(uct_recv_desc_t *self, void *desc)
{
    uct_rc_verbs_release_desc_t *release = ucs_derived_of(self,
                                                          uct_rc_verbs_release_desc_t);
    void *ib_desc = desc - release->offset;
    ucs_mpool_put_inline(ib_desc);
}

ucs_status_t
uct_rc_verbs_iface_common_tag_init(uct_rc_verbs_iface_common_t *iface,
                                   uct_rc_iface_t *rc_iface,
                                   uct_rc_verbs_iface_common_config_t *config,
                                   uct_rc_iface_config_t *rc_config,
                                   struct ibv_exp_create_srq_attr *srq_init_attr,
                                   size_t rndv_hdr_len)

{
    int sync_ops_count;
    int rx_hdr_len;
    uct_ib_md_t *md = ucs_derived_of(rc_iface->super.super.md, uct_ib_md_t);

    if (!iface->tm.enabled) {
        goto out;
    }

    /* Create XRQ with TM capability */
    srq_init_attr->base.attr.max_sge   = 1;
    srq_init_attr->base.attr.max_wr    = ucs_max(UCT_RC_VERBS_TAG_MIN_POSTED,
                                                rc_config->super.rx.queue_len);
    srq_init_attr->base.attr.srq_limit = 0;
    srq_init_attr->base.srq_context    = iface;
    srq_init_attr->srq_type            = IBV_EXP_SRQT_TAG_MATCHING;
    srq_init_attr->pd                  = md->pd;
    srq_init_attr->cq                  = rc_iface->super.recv_cq;
    srq_init_attr->tm_cap.max_num_tags = iface->tm.num_tags;

    /* 2 ops for each tag (ADD + DEL) and extra ops for SYNC.
     * There can be up to 1/"tag_sync_ratio" SYNC ops during cancellation.
     * Also we assume that there can be up to two pending SYNC ops during
     * unexpected messages flow. */
    if (config->tm.sync_ratio > 0) {
        sync_ops_count = ceil(1.0 / config->tm.sync_ratio);
    } else {
        sync_ops_count = iface->tm.num_tags;
    }
    srq_init_attr->tm_cap.max_ops = (2 * iface->tm.num_tags) + sync_ops_count + 2;
    srq_init_attr->comp_mask     |= IBV_EXP_CREATE_SRQ_CQ | IBV_EXP_CREATE_SRQ_TM;

    rc_iface->rx.srq.srq = ibv_exp_create_srq(md->dev.ibv_context, srq_init_attr);
    if (rc_iface->rx.srq.srq == NULL) {
        ucs_error("Failed to create TM XRQ: %m");
        return UCS_ERR_IO_ERROR;
    }

    iface->tm.tag_sync_thresh = iface->tm.num_tags * config->tm.sync_ratio;
    rc_iface->rx.srq.available   = srq_init_attr->base.attr.max_wr;

    /* AM (NO_TAG) and eager messages have different header sizes.
     * Receive descriptor offsets are calculated based on AM hdr length.
     * Need to store headers difference for correct release of descriptors
     * consumed by unexpected eager messages. */
    rx_hdr_len = rc_iface->super.config.rx_payload_offset -
                 rc_iface->super.config.rx_hdr_offset;
    ucs_assert_always(sizeof(struct ibv_exp_tmh) >= rx_hdr_len);

    iface->tm.eager_desc.super.cb = uct_rc_verbs_iface_release_desc;
    iface->tm.eager_desc.offset   = sizeof(struct ibv_exp_tmh) - rx_hdr_len +
                                    rc_iface->super.config.rx_headroom_offset;

    iface->tm.rndv_desc.super.cb  = uct_rc_verbs_iface_release_desc;
    iface->tm.rndv_desc.offset    = iface->tm.eager_desc.offset + rndv_hdr_len;

    /* Init ptr array to store completions of RNDV operations. Index in
     * ptr_array is used as operation ID and is passed in "app_context"
     * of TM header. */
    ucs_ptr_array_init(&iface->tm.rndv_comps, 0, "rm_rndv_completions");

    ucs_debug("Tag Matching enabled: tag list size %d", iface->tm.num_tags);

out:
    return UCS_OK;
}

#endif /* IBV_EXP_HW_TM */

void uct_rc_verbs_iface_common_preinit(uct_rc_verbs_iface_common_t *iface,
                                       uct_md_h md,
                                       uct_rc_verbs_iface_common_config_t *config,
                                       uct_rc_iface_config_t *rc_config,
                                       const uct_iface_params_t *params,
                                       int tm_cap_flag, unsigned *rx_hdr_len,
                                       unsigned *rx_cq_len)
{
#if IBV_EXP_HW_TM
    struct ibv_exp_tmh tmh;
    uct_ib_device_t *dev = &ucs_derived_of(md, uct_ib_md_t)->dev;
    uint32_t  cap_flags  = IBV_DEVICE_TM_CAPS(dev, capability_flags);

    iface->tm.enabled = (config->tm.enable && (cap_flags & tm_cap_flag));

    if (iface->tm.enabled) {
        UCS_STATIC_ASSERT(sizeof(uct_rc_verbs_ctx_priv_t) <= UCT_TAG_PRIV_LEN);

        iface->tm.eager_unexp.cb  = params->eager_cb;
        iface->tm.eager_unexp.arg = params->eager_arg;
        iface->tm.rndv_unexp.cb   = params->rndv_cb;
        iface->tm.rndv_unexp.arg  = params->rndv_arg;
        iface->tm.unexpected_cnt  = 0;
        iface->tm.num_outstanding = 0;
        iface->tm.num_canceled    = 0;
        iface->tm.num_tags        = ucs_min(IBV_DEVICE_TM_CAPS(dev, max_num_tags),
                                            config->tm.list_size);

        /* Only opcode (rather than the whole TMH) is sent with NO_TAG protocol */
        *rx_hdr_len    = sizeof(uct_rc_hdr_t) + sizeof(tmh.opcode);

        /* There can be:
         * - up to rx.queue_len RX CQEs
         * - up to 3 CQEs for every posted tag: ADD, TM_CONSUMED and MSG_ARRIVED
         * - one SYNC CQE per every IBV_DEVICE_MAX_UNEXP_COUNT unexpected receives */
        UCS_STATIC_ASSERT(IBV_DEVICE_MAX_UNEXP_COUNT);
        *rx_cq_len     = rc_config->super.rx.queue_len + iface->tm.num_tags * 3  +
                         rc_config->super.rx.queue_len / IBV_DEVICE_MAX_UNEXP_COUNT;
        return;
    }
#endif
    *rx_hdr_len = sizeof(uct_rc_hdr_t);
    *rx_cq_len  = rc_config->super.rx.queue_len;
}

void uct_rc_verbs_iface_common_tag_cleanup(uct_rc_verbs_iface_common_t *iface)
{
#if IBV_EXP_HW_TM
    if (UCT_RC_VERBS_TM_ENABLED(iface)) {
        ucs_ptr_array_cleanup(&iface->tm.rndv_comps);
    }
#endif
}

ucs_status_t uct_rc_verbs_iface_common_init(uct_rc_verbs_iface_common_t *iface,
                                            uct_rc_iface_t *rc_iface,
                                            uct_rc_verbs_iface_common_config_t *config,
                                            uct_rc_iface_config_t *rc_config,
                                            unsigned rc_hdr_len)
{
    ucs_status_t status;

    memset(iface->inl_sge, 0, sizeof(iface->inl_sge));

    /* Configuration */
    iface->config.short_desc_size = ucs_max(rc_hdr_len, config->max_am_hdr);
    iface->config.short_desc_size = ucs_max(UCT_RC_MAX_ATOMIC_SIZE,
                                            iface->config.short_desc_size);

    /* Create AM headers and Atomic mempool */
    status = uct_iface_mpool_init(&rc_iface->super.super,
                                  &iface->short_desc_mp,
                                  sizeof(uct_rc_iface_send_desc_t) +
                                      iface->config.short_desc_size,
                                  sizeof(uct_rc_iface_send_desc_t),
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &rc_config->super.tx.mp,
                                  rc_iface->config.tx_qp_len,
                                  uct_rc_iface_send_desc_init,
                                  "rc_verbs_short_desc");
    if (status != UCS_OK) {
        goto err;
    }

    iface->am_inl_hdr = ucs_mpool_get(&iface->short_desc_mp);
    if (iface->am_inl_hdr == NULL) {
        ucs_error("Failed to allocate AM short header");
        status = UCS_ERR_NO_MEMORY;
        goto err_mpool_cleanup;
    }
    iface->config.notag_hdr_size = uct_rc_verbs_notag_header_fill(iface,
                                                                  iface->am_inl_hdr);

    status = uct_rc_verbs_iface_prepost_recvs_common(rc_iface);
    if (status != UCS_OK) {
        goto err_am_inl_hdr_put;
    }

    return UCS_OK;

err_am_inl_hdr_put:
    ucs_mpool_put(iface->am_inl_hdr);
err_mpool_cleanup:
    ucs_mpool_cleanup(&iface->short_desc_mp, 1);
err:
    return status;
}

void uct_rc_verbs_iface_common_cleanup(uct_rc_verbs_iface_common_t *self)
{
    ucs_mpool_put(self->am_inl_hdr);
    ucs_mpool_cleanup(&self->short_desc_mp, 1);
}

void uct_rc_verbs_txcnt_init(uct_rc_verbs_txcnt_t *txcnt)
{
    txcnt->pi = txcnt->ci = 0;
}

