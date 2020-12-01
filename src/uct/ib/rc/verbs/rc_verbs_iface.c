/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rc_verbs.h"
#include "rc_verbs_impl.h"

#include <uct/api/uct.h>
#include <uct/ib/mlx5/exp/ib_exp.h>
#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_log.h>
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

  {"MAX_AM_HDR", "128",
   "Buffer size to reserve for active message headers. If set to 0, the transport will\n"
   "not support zero-copy active messages.",
   ucs_offsetof(uct_rc_verbs_iface_config_t, max_am_hdr), UCS_CONFIG_TYPE_MEMUNITS},

  {"TX_MAX_WR", "-1",
   "Limits the number of outstanding posted work requests. The actual limit is\n"
   "a minimum between this value and the TX queue length. -1 means no limit.",
   ucs_offsetof(uct_rc_verbs_iface_config_t, tx_max_wr), UCS_CONFIG_TYPE_UINT},

  {NULL}
};

static unsigned uct_rc_verbs_get_tx_res_count(uct_rc_verbs_ep_t *ep,
                                              struct ibv_wc *wc)
{
    return wc->wr_id - ep->txcnt.ci;
}

static void uct_rc_verbs_update_tx_res(uct_rc_iface_t *iface,
                                       uct_rc_verbs_ep_t *ep, unsigned count)
{
    ep->txcnt.ci += count;
    uct_rc_txqp_available_add(&ep->super.txqp, count);
    iface->tx.cq_available += count;
    uct_rc_iface_update_reads(iface);
}

static void uct_rc_verbs_handle_failure(uct_ib_iface_t *ib_iface, void *arg,
                                        ucs_status_t ep_status)
{
    struct ibv_wc     *wc      = arg;
    uct_rc_iface_t    *iface   = ucs_derived_of(ib_iface, uct_rc_iface_t);
    ucs_log_level_t    log_lvl = UCS_LOG_LEVEL_FATAL;
    uct_rc_verbs_ep_t *ep;
    ucs_status_t       status;
    unsigned           count;

    ep = ucs_derived_of(uct_rc_iface_lookup_ep(iface, wc->qp_num),
                        uct_rc_verbs_ep_t);
    if (!ep) {
        return;
    }

    count = uct_rc_verbs_get_tx_res_count(ep, wc);
    uct_rc_txqp_purge_outstanding(iface, &ep->super.txqp, ep_status,
                                  ep->txcnt.ci + count, 0);
    uct_rc_verbs_update_tx_res(iface, ep, count);

    if (ep->super.flags & (UCT_RC_EP_FLAG_NO_ERR_HANDLER |
                           UCT_RC_EP_FLAG_FLUSH_CANCEL)) {
        return;
    }

    ep->super.flags |= UCT_RC_EP_FLAG_NO_ERR_HANDLER;

    status  = uct_iface_handle_ep_err(&iface->super.super.super,
                                      &ep->super.super.super, ep_status);
    log_lvl = uct_ib_iface_failure_log_level(ib_iface, status, ep_status);

    ucs_log(log_lvl,
            "send completion with error: %s qpn 0x%x wrid 0x%lx vendor_err 0x%x",
            ibv_wc_status_str(wc->status), wc->qp_num, wc->wr_id, wc->vendor_err);
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
        uct_rc_verbs_update_tx_res(&iface->super, ep, count);

        ucs_arbiter_group_schedule(&iface->super.tx.arbiter,
                                   &ep->super.arb_group);
        ucs_arbiter_dispatch(&iface->super.tx.arbiter, 1,
                             uct_rc_ep_process_pending, NULL);
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
    iface->inl_am_wr.num_sge        = 2;
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

    iface_attr->ep_addr_len = sizeof(uct_rc_verbs_ep_address_t);
    if (md->ops->get_atomic_mr_id(md, &mr_id) == UCS_OK) {
        iface_attr->ep_addr_len += sizeof(mr_id);
    }

    return UCS_OK;
}

ucs_status_t uct_rc_verbs_iface_flush_mem_create(uct_rc_verbs_iface_t *iface)
{
    uct_ib_md_t *md = uct_ib_iface_md(&iface->super.super);
    ucs_status_t status;
    struct ibv_mr *mr;
    void *mem;

    if (iface->flush_mr != NULL) {
        ucs_assert(iface->flush_mem != NULL);
        return UCS_OK;
    }

    /*
     * Map a whole page for the remote side to issue a dummy RDMA_WRITE on it,
     * to flush its outstanding operations. A whole page is used to prevent any
     * other allocations from using same page, so it would be fork-safe.
     */
    mem = ucs_mmap(NULL, ucs_get_page_size(), PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS, -1, 0, "flush_mem");
    if (mem == MAP_FAILED) {
        ucs_error("failed to allocate page for remote flush: %m");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    status = uct_ib_reg_mr(md->pd, mem, ucs_get_page_size(),
                           UCT_IB_MEM_ACCESS_FLAGS, &mr, 0);
    if (status != UCS_OK) {
        goto err_munmap;
    }

    iface->flush_mem = mem;
    iface->flush_mr  = mr;
    return UCS_OK;

err_munmap:
    ucs_munmap(mem, ucs_get_page_size());
err:
    return status;
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

static UCS_CLASS_INIT_FUNC(uct_rc_verbs_iface_t, uct_md_h tl_md,
                           uct_worker_h worker, const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_rc_verbs_iface_config_t *config =
                    ucs_derived_of(tl_config, uct_rc_verbs_iface_config_t);
    uct_ib_iface_config_t *ib_config    = &config->super.super.super;
    uct_ib_iface_init_attr_t init_attr  = {};
    uct_ib_qp_attr_t attr               = {};
    ucs_status_t status;
    struct ibv_qp *qp;
    uct_rc_hdr_t *hdr;

    init_attr.fc_req_size            = sizeof(uct_rc_pending_req_t);
    init_attr.rx_hdr_len             = sizeof(uct_rc_hdr_t);
    init_attr.qp_type                = IBV_QPT_RC;
    init_attr.cq_len[UCT_IB_DIR_RX]  = ib_config->rx.queue_len;
    init_attr.cq_len[UCT_IB_DIR_TX]  = config->super.tx_cq_len;
    init_attr.seg_size               = ib_config->seg_size;

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, &uct_rc_verbs_iface_ops, tl_md,
                              worker, params, &config->super.super, &init_attr);

    self->config.tx_max_wr           = ucs_min(config->tx_max_wr,
                                               self->super.config.tx_qp_len);
    self->super.config.tx_moderation = ucs_min(config->super.tx_cq_moderation,
                                               self->config.tx_max_wr / 4);
    self->super.config.fence_mode    = (uct_rc_fence_mode_t)config->super.super.fence_mode;
    self->super.progress             = uct_rc_verbs_iface_progress;
    self->super.super.config.sl      = uct_ib_iface_config_select_sl(ib_config);
    self->flush_mem                  = NULL;
    self->flush_mr                   = NULL;

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
    uct_ib_exp_qp_fill_attr(&self->super.super, &attr);
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

    if (self->config.max_inline < sizeof(*hdr)) {
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

ucs_status_t uct_rc_verbs_iface_common_prepost_recvs(uct_rc_verbs_iface_t *iface,
                                                     unsigned max)
{
    unsigned count;

    count = ucs_min(max, iface->super.rx.srq.quota);
    iface->super.rx.srq.available += count;
    iface->super.rx.srq.quota     -= count;
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
        uct_rc_verbs_iface_common_prepost_recvs(iface, UINT_MAX);
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

    uct_rc_iface_cleanup_eps(&self->super);

    if (self->flush_mr != NULL) {
        uct_ib_dereg_mr(self->flush_mr);
        ucs_assert(self->flush_mem != NULL);
        ucs_munmap(self->flush_mem, ucs_get_page_size());
    }
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
    .ep_connect_to_ep         = uct_rc_verbs_ep_connect_to_ep,
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
    },
    .create_cq                = uct_ib_verbs_create_cq,
    .arm_cq                   = uct_ib_iface_arm_cq,
    .event_cq                 = (uct_ib_iface_event_cq_func_t)ucs_empty_function,
    .handle_failure           = uct_rc_verbs_handle_failure,
    },
    .init_rx                  = uct_rc_iface_verbs_init_rx,
    .cleanup_rx               = uct_rc_iface_verbs_cleanup_rx,
    .fc_ctrl                  = uct_rc_verbs_ep_fc_ctrl,
    .fc_handler               = uct_rc_iface_fc_handler,
    .cleanup_qp               = uct_rc_verbs_ep_cleanup_qp,
    .ep_post_check            = uct_rc_verbs_ep_post_check
};

static ucs_status_t
uct_rc_verbs_query_tl_devices(uct_md_h md,
                              uct_tl_device_resource_t **tl_devices_p,
                              unsigned *num_tl_devices_p)
{
    uct_ib_md_t *ib_md = ucs_derived_of(md, uct_ib_md_t);
    int flags;

    flags = ib_md->config.eth_pause ? 0 : UCT_IB_DEVICE_FLAG_LINK_IB;
    return uct_ib_device_query_ports(&ib_md->dev, flags, tl_devices_p,
                                     num_tl_devices_p);
}

UCT_TL_DEFINE(&uct_ib_component, rc_verbs, uct_rc_verbs_query_tl_devices,
              uct_rc_verbs_iface_t, "RC_VERBS_", uct_rc_verbs_iface_config_table,
              uct_rc_verbs_iface_config_t);
