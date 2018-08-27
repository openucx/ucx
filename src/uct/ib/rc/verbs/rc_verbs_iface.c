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

  {NULL}
};

static void uct_rc_verbs_handle_failure(uct_ib_iface_t *ib_iface, void *arg,
                                        ucs_status_t status)
{
    struct ibv_wc     *wc      = arg;
    uct_rc_iface_t    *iface   = ucs_derived_of(ib_iface, uct_rc_iface_t);
    ucs_log_level_t    log_lvl = UCS_LOG_LEVEL_FATAL;
    uct_rc_verbs_ep_t *ep;

    ep = ucs_derived_of(uct_rc_iface_lookup_ep(iface, wc->qp_num),
                        uct_rc_verbs_ep_t);
    if (!ep) {
        return;
    }

    iface->tx.cq_available += ep->txcnt.pi - ep->txcnt.ci;
    /* Reset CI to prevent cq_available overrun on ep_destoroy */
    ep->txcnt.ci = ep->txcnt.pi;
    uct_rc_txqp_purge_outstanding(&ep->super.txqp, status, 0);

    if (ib_iface->ops->set_ep_failed(ib_iface, &ep->super.super.super,
                                     status) == UCS_OK) {
        log_lvl = iface->super.super.config.failure_level;
    }

    ucs_log(log_lvl, "send completion with error: %s",
            ibv_wc_status_str(wc->status));
}

static ucs_status_t uct_rc_verbs_ep_set_failed(uct_ib_iface_t *iface,
                                               uct_ep_h ep, ucs_status_t status)
{
    return uct_set_ep_failed(&UCS_CLASS_NAME(uct_rc_verbs_ep_t), ep,
                             &iface->super.super, status);
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

        count = uct_rc_verbs_txcq_get_comp_count(&wc[i], &ep->super.txqp);
        ucs_trace_poll("rc_verbs iface %p tx_wc: ep %p qpn 0x%x count %d",
                       iface, ep, wc[i].qp_num, count);
        uct_rc_verbs_txqp_completed(&ep->super.txqp, &ep->txcnt, count);
        iface->super.tx.cq_available += count;

        uct_rc_txqp_completion_desc(&ep->super.txqp, ep->txcnt.ci);
        ucs_arbiter_group_schedule(&iface->super.tx.arbiter, &ep->super.arb_group);
    }
    ucs_arbiter_dispatch(&iface->super.tx.arbiter, 1, uct_rc_ep_process_pending, NULL);
    return num_wcs;
}

static unsigned uct_rc_verbs_iface_progress(void *arg)
{
    uct_rc_verbs_iface_t *iface = arg;
    unsigned count;

    count = uct_rc_verbs_iface_poll_rx_common(&iface->super);
    if (count > 0) {
        return count;
    }

    return uct_rc_verbs_iface_poll_tx(iface);
}

#if IBV_EXP_HW_TM
static unsigned uct_rc_verbs_iface_progress_tm(void *arg)
{
    uct_rc_verbs_iface_t *iface = arg;
    unsigned count;

    count = uct_rc_verbs_iface_poll_rx_tm(&iface->verbs_common, &iface->super);
    if (count > 0) {
        return count;
    }
    return uct_rc_verbs_iface_poll_tx(iface);
}

static ucs_status_t uct_rc_verbs_iface_tag_recv_zcopy(uct_iface_h tl_iface,
                                                      uct_tag_t tag,
                                                      uct_tag_t tag_mask,
                                                      const uct_iov_t *iov,
                                                      size_t iovcnt,
                                                      uct_tag_context_t *ctx)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_iface,
                                                 uct_rc_verbs_iface_t);
    return uct_rc_verbs_iface_common_tag_recv(&iface->verbs_common,
                                              &iface->super, tag,
                                              tag_mask, iov, iovcnt, ctx);
}

static ucs_status_t uct_rc_verbs_iface_tag_recv_cancel(uct_iface_h tl_iface,
                                                       uct_tag_context_t *ctx,
                                                       int force)
{
   uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_iface,
                                                uct_rc_verbs_iface_t);

   return uct_rc_verbs_iface_common_tag_recv_cancel(&iface->verbs_common,
                                                    &iface->super, ctx, force);
}
#endif /* IBV_EXP_HW_TM */

static ucs_status_t
uct_rc_verbs_iface_tag_init(uct_rc_verbs_iface_t *iface,
                            uct_rc_verbs_iface_config_t *config)
{
#if IBV_EXP_HW_TM
    if (UCT_RC_IFACE_TM_ENABLED(&iface->super)) {
        struct ibv_exp_create_srq_attr srq_init_attr = {};

        iface->super.progress = uct_rc_verbs_iface_progress_tm;

        return uct_rc_verbs_iface_common_tag_init(&iface->verbs_common,
                                                  &iface->super,
                                                  &config->verbs_common,
                                                  &config->super,
                                                  &srq_init_attr,
                                                  sizeof(struct ibv_exp_tmh_rvh));
    }
#endif
    iface->super.progress = uct_rc_verbs_iface_progress;
    return UCS_OK;
}

static void uct_rc_verbs_iface_init_inl_wrs(uct_rc_verbs_iface_t *iface)
{
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

static ucs_status_t uct_rc_verbs_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_iface_common_t *verbs_common = &iface->verbs_common;
    ucs_status_t status;

    status = uct_rc_iface_query(&iface->super, iface_attr,
                                verbs_common->config.max_inline,
                                verbs_common->config.max_inline,
                                verbs_common->config.short_desc_size,
                                uct_ib_iface_get_max_iov(&iface->super.super) - 1,
                                uct_ib_iface_get_max_iov(&iface->super.super) - 1);
    if (status != UCS_OK) {
        return status;
    }

    iface_attr->latency.growth += 1e-9;            /* 1 ns per each extra QP */
    iface_attr->iface_addr_len  = sizeof(uint8_t); /* overwrite */
    iface_attr->overhead        = 75e-9;           /* Software overhead */

    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_rc_verbs_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_rc_verbs_iface_config_t *config =
                    ucs_derived_of(tl_config, uct_rc_verbs_iface_config_t);
    ucs_status_t status;
    uct_ib_iface_init_attr_t init_attr = {};
    struct ibv_qp_cap cap;
    struct ibv_qp *qp;

    init_attr.res_domain_key = UCT_IB_IFACE_NULL_RES_DOMAIN_KEY;
    init_attr.tm_cap_bit     = IBV_EXP_TM_CAP_RC;
    init_attr.fc_req_size    = sizeof(uct_rc_fc_request_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, &uct_rc_verbs_iface_ops, md,
                              worker, params, &config->super, &init_attr);

    self->config.tx_max_wr           = ucs_min(config->verbs_common.tx_max_wr,
                                               self->super.config.tx_qp_len);
    self->super.config.tx_moderation = ucs_min(self->super.config.tx_moderation,
                                               self->config.tx_max_wr / 4);

    status = uct_rc_verbs_iface_tag_init(self, config);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_rc_verbs_iface_common_init(&self->verbs_common,
                                            &self->super,
                                            &config->verbs_common,
                                            &config->super);
    if (status != UCS_OK) {
        goto err_tag_cleanup;
    }

    uct_rc_verbs_iface_init_inl_wrs(self);

    /* Check FC parameters correctness */
    status = uct_rc_init_fc_thresh(&config->fc, &config->super, &self->super);
    if (status != UCS_OK) {
        goto err_common_cleanup;
    }

    /* Create a dummy QP in order to find out max_inline */
    status = uct_rc_iface_qp_create(&self->super, IBV_QPT_RC, &qp, &cap,
                                    self->super.config.tx_qp_len);
    if (status != UCS_OK) {
        goto err_common_cleanup;
    }
    ibv_destroy_qp(qp);

    self->verbs_common.config.max_inline = cap.max_inline_data;
    uct_ib_iface_set_max_iov(&self->super.super, cap.max_send_sge);


    return UCS_OK;

err_common_cleanup:
    uct_rc_verbs_iface_common_cleanup(&self->verbs_common);
err_tag_cleanup:
    uct_rc_iface_tag_cleanup(&self->super);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_verbs_iface_t)
{
    uct_base_iface_progress_disable(&self->super.super.super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    uct_rc_verbs_iface_common_cleanup(&self->verbs_common);
    uct_rc_iface_tag_cleanup(&self->super);
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
    .ep_atomic_cswap32        = uct_rc_verbs_ep_atomic_cswap32,
    .ep_atomic32_post         = uct_rc_verbs_ep_atomic32_post,
    .ep_atomic32_fetch        = uct_rc_verbs_ep_atomic32_fetch,
    .ep_pending_add           = uct_rc_ep_pending_add,
    .ep_pending_purge         = uct_rc_ep_pending_purge,
    .ep_flush                 = uct_rc_verbs_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_rc_verbs_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_verbs_ep_t),
    .ep_get_address           = uct_rc_ep_get_address,
    .ep_connect_to_ep         = uct_rc_ep_connect_to_ep,
    .iface_flush              = uct_rc_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_rc_verbs_iface_common_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_rc_iface_do_progress,
#if IBV_EXP_HW_TM
    .iface_tag_recv_zcopy     = uct_rc_verbs_iface_tag_recv_zcopy,
    .iface_tag_recv_cancel    = uct_rc_verbs_iface_tag_recv_cancel,
    .ep_tag_eager_short       = uct_rc_verbs_ep_tag_eager_short,
    .ep_tag_eager_bcopy       = uct_rc_verbs_ep_tag_eager_bcopy,
    .ep_tag_eager_zcopy       = uct_rc_verbs_ep_tag_eager_zcopy,
    .ep_tag_rndv_zcopy        = uct_rc_verbs_ep_tag_rndv_zcopy,
    .ep_tag_rndv_cancel       = uct_rc_ep_tag_rndv_cancel,
    .ep_tag_rndv_request      = uct_rc_verbs_ep_tag_rndv_request,
#endif
    .iface_event_fd_get       = uct_ib_iface_event_fd_get,
    .iface_event_arm          = uct_rc_iface_event_arm,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_verbs_iface_t),
    .iface_query              = uct_rc_verbs_iface_query,
    .iface_get_address        = uct_rc_iface_get_address,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_rc_iface_is_reachable,
    },
    .arm_cq                   = uct_ib_iface_arm_cq,
    .event_cq                 = (void*)ucs_empty_function,
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
