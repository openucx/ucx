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

#if HAVE_IBV_EX_HW_TM
  {"TM_ENABLE", "y",
   "Enable HW tag matching",
   ucs_offsetof(uct_rc_verbs_iface_config_t, tm.enable), UCS_CONFIG_TYPE_BOOL},

  {"TM_LIST_SIZE", "64",
   "Limits the number of tags posted to the HW for matching. The actual limit \n"
   "is a minimum between this value and the maximum value supported by the HW. \n"
   "-1 means no limit.",
   ucs_offsetof(uct_rc_verbs_iface_config_t, tm.list_size), UCS_CONFIG_TYPE_UINT},
#endif

  {NULL}
};

static UCS_F_NOINLINE void uct_rc_verbs_handle_failure(uct_ib_iface_t *ib_iface,
                                                       void *arg)
{
    struct ibv_wc *wc = arg;
    uct_rc_verbs_ep_t *ep;
    extern ucs_class_t UCS_CLASS_NAME(uct_rc_verbs_ep_t);
    uct_rc_iface_t *iface = ucs_derived_of(ib_iface, uct_rc_iface_t);
    ep = ucs_derived_of(uct_rc_iface_lookup_ep(iface, wc->qp_num),
                        uct_rc_verbs_ep_t);
    if (ep != NULL) {
        ucs_log(iface->super.super.config.failure_level,
                "Send completion with error: %s",
                ibv_wc_status_str(wc->status));

        uct_rc_txqp_purge_outstanding(&ep->super.txqp, UCS_ERR_ENDPOINT_TIMEOUT, 0);

        uct_set_ep_failed(&UCS_CLASS_NAME(uct_rc_verbs_ep_t),
                          &ep->super.super.super,
                          &iface->super.super.super);
    }
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

#if HAVE_IBV_EX_HW_TM
/* This function check whether the error occured due to "MESSAGE_TRUNCATED"
 * error in Tag Matching (i.e. if posted buffer was not enough to fit the
 * incoming message). If this is the case the error should be reported in
 * the corresponding callback and QP should be reset back to normal. Otherwise
 * treat the error as fatal. */
static UCS_F_NOINLINE void
uct_rc_verbs_iface_wc_error(uct_rc_verbs_iface_t *iface, struct ibv_wc *wc)
{
    /* TODO: handle MSG TRUNCATED error */
    ucs_fatal("Receive completion with error on XRQ: %s",
              ibv_wc_status_str(wc->status));
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_iface_tag_handle_notag(uct_rc_verbs_iface_t *iface,
                                    struct ibv_wc *wc)
{
    struct ibv_tm_info tm_info;
    size_t tm_info_len;
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super.super);
    uct_ib_iface_recv_desc_t *ib_desc = (uct_ib_iface_recv_desc_t*)(uintptr_t)wc->wr_id;
    void *desc = uct_ib_iface_recv_desc_hdr(&iface->super.super, ib_desc);

    VALGRIND_MAKE_MEM_DEFINED(desc, wc->byte_len);

    tm_info_len = ibv_unpack_tm_info(dev->ibv_context, desc, &tm_info);

    if (tm_info.op == IBV_TM_OP_NO_TAG) {
        uct_rc_verbs_iface_handle_am(&iface->super, wc,
                                     (uct_rc_hdr_t*)((char*)desc + tm_info_len),
                                     wc->byte_len - tm_info_len);
    } else {
        ucs_error("Unsupported packet arrived %d", tm_info.op);
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_verbs_iface_poll_rx_tm(uct_rc_verbs_iface_t *iface)
{
    unsigned i;
    ucs_status_t status;
    unsigned num_wcs = iface->super.super.config.rx_max_poll;
    struct ibv_wc wc[num_wcs];


    status = uct_ib_poll_cq(iface->super.super.recv_cq, &num_wcs, wc);
    if (status != UCS_OK) {
        goto out;
    }

    for (i = 0; i < num_wcs; i++) {
        if (ucs_unlikely(wc[i].status != IBV_WC_SUCCESS)) {
            uct_rc_verbs_iface_wc_error(iface, &wc[i]);
            continue;
        }
        switch (wc[i].opcode) {
        case IBV_WC_TM_NO_TAG:
            uct_rc_verbs_iface_tag_handle_notag(iface, &wc[i]);
            break;

        default:
            ucs_error("Wrong opcode in CQE %d", wc[i].opcode);
            break;
        }
    }
    iface->super.rx.srq.available += num_wcs;
    UCS_STATS_UPDATE_COUNTER(iface->super.stats, UCT_RC_IFACE_STAT_RX_COMPLETION, num_wcs);

out:
    /* All tag unexpected and AM messages arrive to XRQ */
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
#endif /* HAVE_IBV_EX_HW_TM */

static ucs_status_t uct_rc_verbs_iface_tag_init(uct_rc_verbs_iface_t *iface,
                                                uct_rc_verbs_iface_config_t *config)
{
#if HAVE_IBV_EX_HW_TM
    struct ibv_srq_init_attr_ex srq_init_attr;
    uct_ib_md_t *md = ucs_derived_of(iface->super.super.super.md, uct_ib_md_t);

    if (UCT_RC_VERBS_TM_ENABLED(iface)) {
        /* Create XRQ with TM capability */
        memset(&srq_init_attr, 0, sizeof(srq_init_attr));
        srq_init_attr.attr.max_sge        = 1;
        srq_init_attr.attr.max_wr         = ucs_max(UCT_RC_VERBS_TAG_MIN_POSTED,
                                                    config->super.super.rx.queue_len);
        srq_init_attr.attr.srq_limit      = 0;
        srq_init_attr.srq_type            = IBV_SRQT_TAG_MATCHING;
        srq_init_attr.srq_context         = iface;
        srq_init_attr.pd                  = md->pd;
        srq_init_attr.cq                  = iface->super.super.recv_cq;
        srq_init_attr.tm_cap.max_num_tags = iface->tm.tag_available;
        srq_init_attr.tm_cap.max_tm_ops   = ucs_min(2*iface->tm.tag_available,
                                            IBV_DEVICE_TM_CAPS(&md->dev, max_tag_ops));
        srq_init_attr.comp_mask           = IBV_SRQ_INIT_ATTR_TYPE |
                                            IBV_SRQ_INIT_ATTR_PD |
                                            IBV_SRQ_INIT_ATTR_CQ |
                                            IBV_SRQ_INIT_ATTR_TAG_MATCHING;

        iface->super.rx.srq.srq = ibv_create_srq_ex(md->dev.ibv_context, &srq_init_attr);
        if (iface->super.rx.srq.srq == NULL) {
            ucs_error("Failed to create TM XRQ: %m");
            return UCS_ERR_IO_ERROR;
        }
        iface->super.rx.srq.available = srq_init_attr.attr.max_wr;

        --iface->tm.tag_available; /* 1 tag should be always available */
    }
#endif
    return UCS_OK;
}

static ucs_status_t uct_rc_verbs_iface_tag_preinit(uct_rc_verbs_iface_t *iface,
                                                   uct_md_h md,
                                                   uct_rc_verbs_iface_config_t *config,
                                                   const uct_iface_params_t *params,
                                                   unsigned *srq_size,
                                                   unsigned *rx_hdr_len)
{
#if HAVE_IBV_EX_HW_TM
    size_t notag_hdr_size;
    struct ibv_tm_info tm_info;
    uct_ib_md_t *ib_md   = ucs_derived_of(md, uct_ib_md_t);
    uct_ib_device_t *dev = &ib_md->dev;
    void *tm_buf         = ucs_alloca(IBV_DEVICE_TM_CAPS(dev, max_header_size));

    iface->tm.enabled = UCT_RC_VERBS_TM_CONFIG(config, enable);

    if (IBV_DEVICE_TM_CAPS(dev, max_num_tags) &&
        UCT_RC_VERBS_TM_CONFIG(config, enable)) {
        iface->progress            = uct_rc_verbs_iface_progress_tm;
        iface->tm.eager_unexp.cb   = params->eager_cb;
        iface->tm.eager_unexp.arg  = params->eager_arg;
        iface->tm.rndv_unexp.cb    = params->rndv_cb;
        iface->tm.rndv_unexp.arg   = params->rndv_arg;
        iface->tm.tag_available    = ucs_min(IBV_DEVICE_TM_CAPS(dev, max_num_tags),
                                             UCT_RC_VERBS_TM_CONFIG(config, list_size));
        /* Get NO_TAG header size */
        tm_info.op     = IBV_TM_OP_NO_TAG;
        notag_hdr_size = ibv_pack_tm_info(dev->ibv_context, tm_buf, &tm_info);

        ucs_debug("Tag Matching enabled: tag list size %d", iface->tm.tag_available);
        *srq_size   = 0;
        *rx_hdr_len = sizeof(uct_rc_hdr_t) + notag_hdr_size;

    } else
#endif
    {
        iface->verbs_common.config.notag_hdr_size = 0;
        iface->progress = uct_rc_verbs_iface_progress;
        *srq_size       = config->super.super.rx.queue_len;
        *rx_hdr_len     = sizeof(uct_rc_hdr_t);
    }

    return UCS_OK;
}

static void uct_rc_verbs_iface_tag_cleanup(uct_rc_verbs_iface_t *iface)
{
    if (UCT_RC_VERBS_TM_ENABLED(iface)) {
        if (ibv_destroy_srq(iface->super.rx.srq.srq)) {
            ucs_warn("failed to destroy TM XRQ: %m");
        }
        iface->super.rx.srq.srq = NULL;
    }
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

void uct_rc_verbs_iface_tm_query(uct_rc_verbs_iface_t *iface,
                                 uct_iface_attr_t *iface_attr)
{
#if HAVE_IBV_EX_HW_TM
    /* Redefine AM caps, because we have to send TMH (with NO_TAG
     * operation) with every AM message. */
    iface_attr->cap.am.max_short -= iface->verbs_common.config.notag_hdr_size;
    if (iface_attr->cap.am.max_short <= 0) {
        iface_attr->cap.flags &= ~UCT_IFACE_FLAG_AM_SHORT;
    }

    iface_attr->cap.am.max_bcopy -= iface->verbs_common.config.notag_hdr_size;
    iface_attr->cap.am.max_zcopy -= iface->verbs_common.config.notag_hdr_size;
    iface_attr->cap.am.max_hdr   -= iface->verbs_common.config.notag_hdr_size;
#endif
}

static ucs_status_t uct_rc_verbs_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_verbs_iface_t);

    uct_rc_iface_query(&iface->super, iface_attr);
    uct_rc_verbs_iface_common_query(&iface->verbs_common, &iface->super, iface_attr);
    iface_attr->latency.growth += 3e-9; /* 3ns per each extra QP */

    if (UCT_RC_VERBS_TM_ENABLED(iface)) {
        uct_rc_verbs_iface_tm_query(iface, iface_attr);
    }
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
    size_t am_hdr_len;

    uct_rc_verbs_iface_tag_preinit(self, md, config, params, &srq_size,
                                   &rx_hdr_len);

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, &uct_rc_verbs_iface_ops,
                              md, worker, params, &config->super, 0,
                              config->super.super.rx.queue_len,
                              rx_hdr_len, srq_size,
                              sizeof(uct_rc_fc_request_t));

    self->config.tx_max_wr           = ucs_min(config->verbs_common.tx_max_wr,
                                               self->super.config.tx_qp_len);
    self->super.config.tx_moderation = ucs_min(self->super.config.tx_moderation,
                                               self->config.tx_max_wr / 4);

    am_hdr_len = ucs_max(config->verbs_common.max_am_hdr, rx_hdr_len);
    status = uct_rc_verbs_iface_common_init(&self->verbs_common, &self->super,
                                            &config->verbs_common, &config->super,
                                            am_hdr_len);
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
    .iface_query              = uct_rc_verbs_iface_query,
    .iface_flush              = uct_rc_iface_flush,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_verbs_iface_t),
    .iface_release_desc       = uct_ib_iface_release_desc,
    .iface_wakeup_open        = uct_ib_iface_wakeup_open,
    .iface_wakeup_get_fd      = uct_ib_iface_wakeup_get_fd,
    .iface_wakeup_arm         = uct_ib_iface_wakeup_arm,
    .iface_wakeup_wait        = uct_ib_iface_wakeup_wait,
    .iface_wakeup_signal      = uct_ib_iface_wakeup_signal,
    .iface_wakeup_close       = uct_ib_iface_wakeup_close,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_rc_verbs_ep_t),
    .ep_get_address           = uct_rc_ep_get_address,
    .ep_connect_to_ep         = uct_rc_ep_connect_to_ep,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_ib_iface_is_reachable,
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_verbs_ep_t),
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
    .ep_flush                 = uct_rc_verbs_ep_flush
    },
    .arm_tx_cq                = uct_ib_iface_arm_tx_cq,
    .arm_rx_cq                = uct_ib_iface_arm_rx_cq,
    .handle_failure           = uct_rc_verbs_handle_failure
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
                                            (ib_md->eth_pause ? 0 : UCT_IB_DEVICE_FLAG_LINK_IB),
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
