/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "rc_verbs_common.h"

#include <uct/api/uct.h>
#include <uct/ib/rc/base/rc_iface.h>
#include <ucs/arch/bitops.h>
#include <ucs/arch/cpu.h>


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
  {"TM_SYNC_RATIO", "0.5",
   "Maximal portion of the tag matching list which can be canceled without requesting\n"
   "a completion.",
   ucs_offsetof(uct_rc_verbs_iface_common_config_t, tm_sync_ratio), UCS_CONFIG_TYPE_DOUBLE},
#endif
  {NULL}
};

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

ucs_status_t uct_rc_verbs_iface_common_prepost_recvs(uct_rc_iface_t *iface,
                                                     unsigned max)
{
    unsigned count;

    count = ucs_min(max, iface->rx.srq.quota);
    iface->rx.srq.available += count;
    iface->rx.srq.quota     -= count;
    while (iface->rx.srq.available > 0) {
        if (uct_rc_verbs_iface_post_recv_common(iface, 1) == 0) {
            ucs_error("failed to post receives");
            return UCS_ERR_NO_MEMORY;
        }
    }
    return UCS_OK;
}

void uct_rc_verbs_iface_common_progress_enable(uct_iface_h tl_iface, unsigned flags)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);

    if (flags & UCT_PROGRESS_RECV) {
        /* ignore return value from prepost_recv, since it's not really possible
         * to handle here, and some receives were already pre-posted during iface
         * creation anyway.
         */
        uct_rc_verbs_iface_common_prepost_recvs(iface, UINT_MAX);
    }

    uct_base_iface_progress_enable_cb(&iface->super.super, iface->progress,
                                      flags);
}

#if IBV_EXP_HW_TM

ucs_status_t
uct_rc_verbs_iface_common_tag_init(uct_rc_verbs_iface_common_t *iface,
                                   uct_rc_iface_t *rc_iface,
                                   uct_rc_verbs_iface_common_config_t *config,
                                   uct_rc_iface_config_t *rc_config,
                                   struct ibv_exp_create_srq_attr *srq_init_attr,
                                   size_t rndv_hdr_len)

{
    unsigned sync_ops_count;
    ucs_status_t status;

    if (!UCT_RC_IFACE_TM_ENABLED(rc_iface)) {
        return UCS_OK;
    }

    /* There can be up to 1/"tag_sync_ratio" SYNC ops during cancellation. */
    if (config->tm_sync_ratio > 0) {
        sync_ops_count = ceil(1.0 / config->tm_sync_ratio);
    } else {
        sync_ops_count = rc_iface->tm.num_tags;
    }

    status = uct_rc_iface_tag_init(rc_iface, rc_config, srq_init_attr,
                                   rndv_hdr_len, sync_ops_count);
    if (status != UCS_OK) {
        return status;
    }

    iface->tm.num_canceled    = 0;
    iface->tm.tag_sync_thresh = rc_iface->tm.num_tags * config->tm_sync_ratio;

    return UCS_OK;
}

#endif /* IBV_EXP_HW_TM */

ucs_status_t uct_rc_verbs_iface_common_init(uct_rc_verbs_iface_common_t *iface,
                                            uct_rc_iface_t *rc_iface,
                                            uct_rc_verbs_iface_common_config_t *config,
                                            uct_rc_iface_config_t *rc_config)
{
    ucs_status_t status;

    memset(iface->inl_sge, 0, sizeof(iface->inl_sge));
    uct_rc_am_hdr_fill(&iface->am_inl_hdr.rc_hdr, 0);

    /* Configuration */
    iface->config.short_desc_size = ucs_max(sizeof(uct_rc_hdr_t),
                                            config->max_am_hdr);
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

    return UCS_OK;

err:
    return status;
}

void uct_rc_verbs_iface_common_cleanup(uct_rc_verbs_iface_common_t *self)
{
    ucs_mpool_cleanup(&self->short_desc_mp, 1);
}

void uct_rc_verbs_txcnt_init(uct_rc_verbs_txcnt_t *txcnt)
{
    txcnt->pi = txcnt->ci = 0;
}

ucs_status_t uct_rc_verbs_wc_to_ucs_status(enum ibv_wc_status status)
{
    switch (status)
    {
    case IBV_WC_SUCCESS:
        return UCS_OK;
    case IBV_WC_RETRY_EXC_ERR:
    case IBV_WC_RNR_RETRY_EXC_ERR:
        return UCS_ERR_ENDPOINT_TIMEOUT;
    default:
        return UCS_ERR_IO_ERROR;
    }
}
