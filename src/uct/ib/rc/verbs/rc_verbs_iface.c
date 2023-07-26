/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rc_verbs.h"
#include "rc_verbs_impl.h"

#include <uct/api/uct.h>
#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_log.h>
#include <uct/base/uct_md.h>
#include <ucs/arch/bitops.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>
#include <string.h>

static uct_rc_iface_ops_t uct_rc_verbs_iface_ops;
static uct_iface_ops_t uct_rc_verbs_iface_tl_ops;

static const char *uct_rc_verbs_flush_mode_names[] = {
    [UCT_RC_VERBS_FLUSH_MODE_RDMA_WRITE_0] = "write0",
    [UCT_RC_VERBS_FLUSH_MODE_FLOW_CONTROL] = "fc",
    [UCT_RC_VERBS_FLUSH_MODE_AUTO]         = "auto",
    [UCT_RC_VERBS_FLUSH_MODE_LAST]         = NULL
};

static ucs_config_field_t uct_rc_verbs_iface_config_table[] = {
  {"RC_", "", NULL,
   ucs_offsetof(uct_rc_verbs_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_rc_iface_config_table)},

  {"MAX_AM_HDR", "128",
   "Buffer size to reserve for active message headers. If set to 0, the transport will\n"
   "not support zero-copy active messages.",
   ucs_offsetof(uct_rc_verbs_iface_config_t, max_am_hdr), UCS_CONFIG_TYPE_MEMUNITS},

  {"TX_MAX_WR", "-1",
   "Limits the number of outstanding posted work requests. The actual limit is\n"
   "a minimum between this value and the TX queue length. -1 means no limit.",
   ucs_offsetof(uct_rc_verbs_iface_config_t, tx_max_wr), UCS_CONFIG_TYPE_UINT},

  {"FLUSH_MODE", "auto",
   "Method to use for posting flush operation:\n"
   " - write0 : Post empty RDMA_WRITE\n"
   " - fc     : Send flow control message\n"
   " - auto   : Select automatically based on device support",
   ucs_offsetof(uct_rc_verbs_iface_config_t, flush_mode),
   UCS_CONFIG_TYPE_ENUM(uct_rc_verbs_flush_mode_names)},

  {NULL}
};

static unsigned uct_rc_verbs_get_tx_res_count(uct_rc_verbs_ep_t *ep,
                                              struct ibv_wc *wc)
{
    return wc->wr_id - ep->txcnt.ci;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_update_tx_res(uct_rc_iface_t *iface, uct_rc_verbs_ep_t *ep,
                           unsigned count)
{
    ep->txcnt.ci += count;
    uct_rc_txqp_available_add(&ep->super.txqp, count);
    uct_rc_iface_update_reads(iface);
    uct_rc_iface_add_cq_credits(iface, count);
}

static void uct_rc_verbs_handle_failure(uct_ib_iface_t *ib_iface, void *arg,
                                        ucs_status_t ep_status)
{
    struct ibv_wc *wc       = arg;
    uct_rc_iface_t *iface   = ucs_derived_of(ib_iface, uct_rc_iface_t);
    ucs_log_level_t log_lvl = UCS_LOG_LEVEL_FATAL;
    char peer_info[128]     = {};
    unsigned dest_qpn;
    uct_rc_verbs_ep_t *ep;
    ucs_status_t status;
    unsigned count;
    struct ibv_ah_attr ah_attr;

    ep = ucs_derived_of(uct_rc_iface_lookup_ep(iface, wc->qp_num),
                        uct_rc_verbs_ep_t);
    if (!ep) {
        return;
    }

    count = uct_rc_verbs_get_tx_res_count(ep, wc);
    uct_rc_txqp_purge_outstanding(iface, &ep->super.txqp, ep_status,
                                  ep->txcnt.ci + count, 0);
    ucs_arbiter_group_purge(&iface->tx.arbiter, &ep->super.arb_group,
                            uct_rc_ep_arbiter_purge_internal_cb, NULL);
    uct_rc_verbs_update_tx_res(iface, ep, count);

    if (ep->super.flags & (UCT_RC_EP_FLAG_ERR_HANDLER_INVOKED |
                           UCT_RC_EP_FLAG_FLUSH_CANCEL)) {
        goto out;
    }

    ep->super.flags |= UCT_RC_EP_FLAG_ERR_HANDLER_INVOKED;
    uct_rc_fc_restore_wnd(iface, &ep->super.fc);

    status  = uct_iface_handle_ep_err(&iface->super.super.super,
                                      &ep->super.super.super, ep_status);
    log_lvl = uct_base_iface_failure_log_level(&ib_iface->super, status,
                                               ep_status);
    status  = uct_ib_query_qp_peer_info(ep->qp, &ah_attr, &dest_qpn);
    if (status == UCS_OK) {
        uct_ib_log_dump_qp_peer_info(ib_iface, &ah_attr, dest_qpn, peer_info,
                                     sizeof(peer_info));
    }

    ucs_log(log_lvl,
            "send completion with error: %s [qpn 0x%x wrid 0x%lx"
            "vendor_err 0x%x]\n%s", ibv_wc_status_str(wc->status), wc->qp_num,
            wc->wr_id, wc->vendor_err, peer_info);

out:
    uct_rc_iface_arbiter_dispatch(iface);
}

static ucs_status_t uct_rc_verbs_wc_to_ucs_status(enum ibv_wc_status status)
{
    switch (status)
    {
    case IBV_WC_SUCCESS:
        return UCS_OK;
    case IBV_WC_REM_ACCESS_ERR:
    case IBV_WC_REM_OP_ERR:
        return UCS_ERR_CONNECTION_RESET;
    case IBV_WC_RETRY_EXC_ERR:
    case IBV_WC_RNR_RETRY_EXC_ERR:
    case IBV_WC_REM_ABORT_ERR:
        return UCS_ERR_ENDPOINT_TIMEOUT;
    case IBV_WC_WR_FLUSH_ERR:
        return UCS_ERR_CANCELED;
    default:
        return UCS_ERR_IO_ERROR;
    }
}

static UCS_F_ALWAYS_INLINE unsigned
uct_rc_verbs_iface_poll_tx(uct_rc_verbs_iface_t *iface)
{
    uct_rc_verbs_ep_t *ep;
    uint16_t count;
    int i;
    unsigned num_wcs = iface->super.super.config.tx_max_poll;
    struct ibv_wc wc[num_wcs];
    ucs_status_t status;

    UCT_RC_VERBS_IFACE_FOREACH_TXWQE(&iface->super, i, wc, num_wcs) {
        ep = ucs_derived_of(uct_rc_iface_lookup_ep(&iface->super, wc[i].qp_num),
                            uct_rc_verbs_ep_t);
        if (ucs_unlikely((wc[i].status != IBV_WC_SUCCESS) || (ep == NULL))) {
            status = uct_rc_verbs_wc_to_ucs_status(wc[i].status);
            iface->super.super.ops->handle_failure(&iface->super.super, &wc[i],
                                                   status);
            continue;
        }

        count = uct_rc_verbs_get_tx_res_count(ep, &wc[i]);
        ucs_trace_poll("rc_verbs iface %p tx_wc wrid 0x%lx ep %p qpn 0x%x count %d",
                       iface, wc[i].wr_id, ep, wc[i].qp_num, count);

        uct_rc_txqp_completion_desc(&ep->super.txqp, ep->txcnt.ci + count);
        ucs_arbiter_group_schedule(&iface->super.tx.arbiter,
                                   &ep->super.arb_group);
        uct_rc_verbs_update_tx_res(&iface->super, ep, count);
        ucs_arbiter_dispatch(&iface->super.tx.arbiter, 1, uct_rc_ep_process_pending,
                             NULL);
    }

    return num_wcs;
}

static unsigned uct_rc_verbs_iface_progress(void *arg)
{
    uct_rc_verbs_iface_t *iface = arg;
    unsigned count;

    count = uct_rc_verbs_iface_poll_rx_common(iface);
    if (!uct_rc_iface_poll_tx(&iface->super, count)) {
        return count;
    }

    return uct_rc_verbs_iface_poll_tx(iface);
}

static void uct_rc_verbs_iface_init_inl_wrs(uct_rc_verbs_iface_t *iface)
{
    memset(&iface->inl_am_wr, 0, sizeof(iface->inl_am_wr));
    iface->inl_am_wr.sg_list        = iface->inl_sge;
    iface->inl_am_wr.opcode         = IBV_WR_SEND;
    iface->inl_am_wr.send_flags     = IBV_SEND_INLINE;

    memset(&iface->inl_rwrite_wr, 0, sizeof(iface->inl_rwrite_wr));
    iface->inl_rwrite_wr.sg_list    = iface->inl_sge;
    iface->inl_rwrite_wr.num_sge    = 1;
    iface->inl_rwrite_wr.opcode     = IBV_WR_RDMA_WRITE;
    iface->inl_rwrite_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
}

static ucs_status_t uct_rc_verbs_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_verbs_iface_t);
    uct_ib_md_t *md             = uct_ib_iface_md(ucs_derived_of(iface, uct_ib_iface_t));
    uint8_t mr_id;
    ucs_status_t status;

    status = uct_rc_iface_query(&iface->super, iface_attr,
                                iface->config.max_inline,
                                iface->config.max_inline,
                                iface->config.short_desc_size,
                                iface->config.max_send_sge - 1,
                                sizeof(uct_rc_hdr_t),
                                iface->config.max_send_sge);
    if (status != UCS_OK) {
        return status;
    }

    iface_attr->cap.flags |= UCT_IFACE_FLAG_EP_CHECK;
    iface_attr->latency.m += 1e-9;  /* 1 ns per each extra QP */
    iface_attr->overhead   = 75e-9; /* Software overhead */

    iface_attr->ep_addr_len = (md->ops->get_atomic_mr_id(md, &mr_id) == UCS_OK) ?
                              sizeof(uct_rc_verbs_ep_flush_addr_t) :
                              sizeof(uct_rc_verbs_ep_addr_t);

    return UCS_OK;
}

static ucs_status_t
uct_rc_iface_verbs_init_rx(uct_rc_iface_t *rc_iface,
                           const uct_rc_iface_common_config_t *config)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(rc_iface, uct_rc_verbs_iface_t);

    return uct_rc_iface_init_rx(rc_iface, config, &iface->srq);
}

void uct_rc_iface_verbs_cleanup_rx(uct_rc_iface_t *rc_iface)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(rc_iface, uct_rc_verbs_iface_t);

    /* TODO flush RX buffers */
    uct_ib_destroy_srq(iface->srq);
}

static void
uct_rc_verbs_iface_qp_cleanup(uct_rc_iface_qp_cleanup_ctx_t *rc_cleanup_ctx)
{
    uct_rc_verbs_iface_qp_cleanup_ctx_t *cleanup_ctx =
            ucs_derived_of(rc_cleanup_ctx, uct_rc_verbs_iface_qp_cleanup_ctx_t);
    uct_ib_destroy_qp(cleanup_ctx->qp);
}

static UCS_CLASS_INIT_FUNC(uct_rc_verbs_iface_t, uct_md_h tl_md,
                           uct_worker_h worker, const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_rc_verbs_iface_config_t *config =
                    ucs_derived_of(tl_config, uct_rc_verbs_iface_config_t);
    uct_ib_iface_config_t *ib_config    = &config->super.super.super;
    uct_ib_md_t *ib_md                  = ucs_derived_of(tl_md, uct_ib_md_t);
    uct_ib_iface_init_attr_t init_attr  = {};
    uct_ib_qp_attr_t attr               = {};
    const char *dev_name;
    ucs_status_t status;
    struct ibv_qp *qp;
    uct_rc_hdr_t *hdr;

    init_attr.fc_req_size           = sizeof(uct_rc_pending_req_t);
    init_attr.rx_hdr_len            = sizeof(uct_rc_hdr_t);
    init_attr.qp_type               = IBV_QPT_RC;
    init_attr.cq_len[UCT_IB_DIR_RX] = ib_config->rx.queue_len;
    init_attr.cq_len[UCT_IB_DIR_TX] = config->super.tx_cq_len;
    init_attr.seg_size              = ib_config->seg_size;
    init_attr.max_rd_atomic         = IBV_DEV_ATTR(&ib_md->dev, max_qp_rd_atom);

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, &uct_rc_verbs_iface_tl_ops,
                              &uct_rc_verbs_iface_ops, tl_md, worker, params,
                              &config->super.super, &init_attr);

    self->config.tx_max_wr               = ucs_min(config->tx_max_wr,
                                                   self->super.config.tx_qp_len);
    self->super.config.tx_moderation     = ucs_min(config->super.tx_cq_moderation,
                                                   self->config.tx_max_wr / 4);
    self->super.config.fence_mode        = (uct_rc_fence_mode_t)config->super.super.fence_mode;
    self->super.progress                 = uct_rc_verbs_iface_progress;
    self->super.super.config.sl          = uct_ib_iface_config_select_sl(ib_config);

    if ((config->super.super.fence_mode == UCT_RC_FENCE_MODE_WEAK) ||
        (config->super.super.fence_mode == UCT_RC_FENCE_MODE_AUTO)) {
        self->super.config.fence_mode = UCT_RC_FENCE_MODE_WEAK;
    } else if (config->super.super.fence_mode == UCT_RC_FENCE_MODE_NONE) {
        self->super.config.fence_mode = UCT_RC_FENCE_MODE_NONE;
    } else {
        ucs_error("incorrect fence value: %d", self->super.config.fence_mode);
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    memset(self->inl_sge, 0, sizeof(self->inl_sge));
    uct_rc_am_hdr_fill(&self->am_inl_hdr.rc_hdr, 0);

    /* Configuration */
    self->config.short_desc_size = ucs_max(sizeof(uct_rc_hdr_t),
                                           config->max_am_hdr);
    self->config.short_desc_size = ucs_max(UCT_IB_MAX_ATOMIC_SIZE,
                                           self->config.short_desc_size);

    /* Flush mode */
    if (config->flush_mode == UCT_RC_VERBS_FLUSH_MODE_AUTO) {
        /* Use flow control for flush on older devices */
        dev_name                 = uct_ib_device_name(
                                       uct_ib_iface_device(&self->super.super));
        self->config.flush_by_fc = (strstr(dev_name, "mthca") == dev_name);
    } else {
        self->config.flush_by_fc = (config->flush_mode ==
                                    UCT_RC_VERBS_FLUSH_MODE_FLOW_CONTROL);
    }

    /* Create AM headers and Atomic mempool */
    status = uct_iface_mpool_init(&self->super.super.super,
                                  &self->short_desc_mp,
                                  sizeof(uct_rc_iface_send_desc_t) +
                                      self->config.short_desc_size,
                                  sizeof(uct_rc_iface_send_desc_t),
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &ib_config->tx.mp,
                                  self->super.config.tx_qp_len,
                                  uct_rc_iface_send_desc_init,
                                  "rc_verbs_short_desc");
    if (status != UCS_OK) {
        goto err;
    }

    uct_rc_verbs_iface_init_inl_wrs(self);

    /* Check FC parameters correctness */
    status = uct_rc_init_fc_thresh(&config->super, &self->super);
    if (status != UCS_OK) {
        goto err_common_cleanup;
    }

    /* Create a dummy QP in order to find out max_inline */
    status = uct_rc_iface_qp_create(&self->super, &qp, &attr,
                                    self->super.config.tx_qp_len,
                                    self->srq);
    if (status != UCS_OK) {
        goto err_common_cleanup;
    }
    uct_ib_destroy_qp(qp);

    self->config.max_inline   = attr.cap.max_inline_data;
    self->config.max_send_sge = ucs_min(UCT_IB_MAX_IOV, attr.cap.max_send_sge);
    ucs_assertv_always(self->config.max_send_sge > 1, /* need 1 iov for am header*/
                       "max_send_sge %zu", self->config.max_send_sge);

    if ((self->config.max_inline < sizeof(*hdr)) || self->config.flush_by_fc) {
        self->fc_desc = ucs_mpool_get(&self->short_desc_mp);
        ucs_assert_always(self->fc_desc != NULL);
        hdr        = (uct_rc_hdr_t*)(self->fc_desc + 1);
        hdr->am_id = UCT_RC_EP_FC_PURE_GRANT;
    } else {
        self->fc_desc = NULL;
    }

    return UCS_OK;

err_common_cleanup:
    ucs_mpool_cleanup(&self->short_desc_mp, 1);
err:
    return status;
}

ucs_status_t
uct_rc_verbs_iface_common_prepost_recvs(uct_rc_verbs_iface_t *iface)
{
    if (iface->super.rx.srq.quota == 0) {
        return UCS_OK;
    }

    iface->super.rx.srq.available = iface->super.rx.srq.quota;
    iface->super.rx.srq.quota     = 0;
    while (iface->super.rx.srq.available > 0) {
        if (uct_rc_verbs_iface_post_recv_common(iface, 1) == 0) {
            ucs_error("failed to post receives");
            return UCS_ERR_NO_MEMORY;
        }
    }

    return UCS_OK;
}

void uct_rc_verbs_iface_common_progress_enable(uct_iface_h tl_iface, unsigned flags)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_verbs_iface_t);

    if (flags & UCT_PROGRESS_RECV) {
        /* ignore return value from prepost_recv, since it's not really possible
         * to handle here, and some receives were already pre-posted during iface
         * creation anyway.
         */
        uct_rc_verbs_iface_common_prepost_recvs(iface);
    }

    uct_base_iface_progress_enable_cb(&iface->super.super.super,
                                      iface->super.progress,
                                      flags);
}

unsigned uct_rc_verbs_iface_post_recv_always(uct_rc_verbs_iface_t *iface, unsigned max)
{
    struct ibv_recv_wr *bad_wr;
    uct_ib_recv_wr_t *wrs;
    unsigned count;
    int ret;

    wrs  = ucs_alloca(sizeof *wrs  * max);

    count = uct_ib_iface_prepare_rx_wrs(&iface->super.super, &iface->super.rx.mp,
                                        wrs, max);
    if (ucs_unlikely(count == 0)) {
        return 0;
    }

    ret = ibv_post_srq_recv(iface->srq, &wrs[0].ibwr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_srq_recv() returned %d: %m", ret);
    }
    iface->super.rx.srq.available -= count;

    return count;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_verbs_iface_t)
{
    uct_base_iface_progress_disable(&self->super.super.super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);

    uct_rc_iface_cleanup_qps(&self->super);

    if (self->fc_desc != NULL) {
        ucs_mpool_put(self->fc_desc);
    }
    ucs_mpool_cleanup(&self->short_desc_mp, 1);
}

UCS_CLASS_DEFINE(uct_rc_verbs_iface_t, uct_rc_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_verbs_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_verbs_iface_t, uct_iface_t);

static uct_iface_ops_t uct_rc_verbs_iface_tl_ops = {
    .ep_am_short              = uct_rc_verbs_ep_am_short,
    .ep_am_short_iov          = uct_rc_verbs_ep_am_short_iov,
    .ep_am_bcopy              = uct_rc_verbs_ep_am_bcopy,
    .ep_am_zcopy              = uct_rc_verbs_ep_am_zcopy,
    .ep_put_short             = uct_rc_verbs_ep_put_short,
    .ep_put_bcopy             = uct_rc_verbs_ep_put_bcopy,
    .ep_put_zcopy             = uct_rc_verbs_ep_put_zcopy,
    .ep_get_bcopy             = uct_rc_verbs_ep_get_bcopy,
    .ep_get_zcopy             = uct_rc_verbs_ep_get_zcopy,
    .ep_atomic_cswap64        = uct_rc_verbs_ep_atomic_cswap64,
    .ep_atomic64_post         = uct_rc_verbs_ep_atomic64_post,
    .ep_atomic64_fetch        = uct_rc_verbs_ep_atomic64_fetch,
    .ep_atomic_cswap32        = (uct_ep_atomic_cswap32_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic32_post         = (uct_ep_atomic32_post_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic32_fetch        = (uct_ep_atomic32_fetch_func_t)ucs_empty_function_return_unsupported,
    .ep_pending_add           = uct_rc_ep_pending_add,
    .ep_pending_purge         = uct_rc_ep_pending_purge,
    .ep_flush                 = uct_rc_verbs_ep_flush,
    .ep_fence                 = uct_rc_verbs_ep_fence,
    .ep_check                 = uct_rc_ep_check,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_rc_verbs_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_verbs_ep_t),
    .ep_get_address           = uct_rc_verbs_ep_get_address,
    .ep_connect_to_ep         = uct_base_ep_connect_to_ep,
    .iface_flush              = uct_rc_iface_flush,
    .iface_fence              = uct_rc_iface_fence,
    .iface_progress_enable    = uct_rc_verbs_iface_common_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_rc_iface_do_progress,
    .iface_event_fd_get       = uct_ib_iface_event_fd_get,
    .iface_event_arm          = uct_rc_iface_event_arm,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_verbs_iface_t),
    .iface_query              = uct_rc_verbs_iface_query,
    .iface_get_address        = ucs_empty_function_return_success,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_ib_iface_is_reachable,
};

static uct_rc_iface_ops_t uct_rc_verbs_iface_ops = {
    .super = {
        .super = {
            .iface_estimate_perf   = uct_rc_iface_estimate_perf,
            .iface_vfs_refresh     = uct_rc_iface_vfs_refresh,
            .ep_query              = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
            .ep_invalidate         = (uct_ep_invalidate_func_t)ucs_empty_function_return_unsupported,
            .ep_connect_to_ep_v2   = uct_rc_verbs_ep_connect_to_ep_v2,
            .iface_is_reachable_v2 = uct_ib_iface_is_reachable_v2
        },
        .create_cq      = uct_ib_verbs_create_cq,
        .destroy_cq     = uct_ib_verbs_destroy_cq,
        .event_cq       = (uct_ib_iface_event_cq_func_t)ucs_empty_function,
        .handle_failure = uct_rc_verbs_handle_failure,
    },
    .init_rx         = uct_rc_iface_verbs_init_rx,
    .cleanup_rx      = uct_rc_iface_verbs_cleanup_rx,
    .fc_ctrl         = uct_rc_verbs_ep_fc_ctrl,
    .fc_handler      = uct_rc_iface_fc_handler,
    .cleanup_qp      = uct_rc_verbs_iface_qp_cleanup,
    .ep_post_check   = uct_rc_verbs_ep_post_check,
    .ep_vfs_populate = uct_rc_verbs_ep_vfs_populate
};

static ucs_status_t
uct_rc_verbs_can_create_qp(struct ibv_context *ctx, struct ibv_pd *pd)
{
    struct ibv_qp_init_attr qp_init_attr = {
        .qp_type             = IBV_QPT_RC,
        .sq_sig_all          = 0,
        .cap.max_send_wr     = 1,
        .cap.max_recv_wr     = 1,
        .cap.max_send_sge    = 1,
        .cap.max_recv_sge    = 1,
        .cap.max_inline_data = 0
    };
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    ucs_status_t status;

    cq = ibv_create_cq(ctx, 1, NULL, NULL, 0);
    if (cq == NULL) {
        uct_ib_check_memlock_limit_msg(UCS_LOG_LEVEL_DEBUG, "ibv_create_cq()");
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    qp_init_attr.send_cq = cq;
    qp_init_attr.recv_cq = cq;

    qp = ibv_create_qp(pd, &qp_init_attr);
    if (qp == NULL) {
        uct_ib_check_memlock_limit_msg(UCS_LOG_LEVEL_DEBUG, "ibv_create_qp()");
        status = UCS_ERR_UNSUPPORTED;
        goto err_destroy_cq;
    }

    status = UCS_OK;

    ibv_destroy_qp(qp);
err_destroy_cq:
    ibv_destroy_cq(cq);
err:
    return status;
}

static ucs_status_t
uct_rc_verbs_query_tl_devices(uct_md_h md,
                              uct_tl_device_resource_t **tl_devices_p,
                              unsigned *num_tl_devices_p)
{
    uct_ib_md_t *ib_md = ucs_derived_of(md, uct_ib_md_t);
    ucs_status_t status;

    /* device does not support RC if we cannot create an RC QP */
    status = uct_rc_verbs_can_create_qp(ib_md->dev.ibv_context, ib_md->pd);
    if (status != UCS_OK) {
        return status;
    }

    return uct_ib_device_query_ports(&ib_md->dev, 0, tl_devices_p,
                                     num_tl_devices_p);
}

UCT_TL_DEFINE_ENTRY(&uct_ib_component, rc_verbs, uct_rc_verbs_query_tl_devices,
                    uct_rc_verbs_iface_t, "RC_VERBS_",
                    uct_rc_verbs_iface_config_table,
                    uct_rc_verbs_iface_config_t);
