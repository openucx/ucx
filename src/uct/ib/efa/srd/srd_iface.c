/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <uct/ib/efa/srd/srd_log.h>
#include <uct/ib/efa/srd/srd_iface.h>
#include <uct/ib/efa/base/ib_efa.h>
#include <uct/ib/base/ib_log.h>
#include <uct/ib/ud/verbs/ud_verbs.h>


static uct_iface_ops_t uct_srd_iface_tl_ops;

ucs_status_t
uct_srd_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    uct_srd_iface_t *iface     = ucs_derived_of(tl_iface, uct_srd_iface_t);
    uct_srd_iface_addr_t *addr = (uct_srd_iface_addr_t*)iface_addr;

    uct_ib_pack_uint24(addr->qp_num, iface->qp->qp_num);

    return UCS_OK;
}

static uct_ib_iface_ops_t uct_srd_iface_ops = {
    .super = {
        .iface_estimate_perf   = uct_base_iface_estimate_perf,
        .iface_vfs_refresh     = (uct_iface_vfs_refresh_func_t)
            ucs_empty_function,
        .ep_query              = (uct_ep_query_func_t)
            ucs_empty_function_return_unsupported,
        .ep_invalidate         = (uct_ep_invalidate_func_t)
            ucs_empty_function_return_unsupported,
        .ep_connect_to_ep_v2   = (uct_ep_connect_to_ep_v2_func_t)
            ucs_empty_function_return_unsupported,
        .iface_is_reachable_v2 = uct_ib_iface_is_reachable_v2,
    },
    .create_cq      = uct_ib_verbs_create_cq,
    .destroy_cq     = uct_ib_verbs_destroy_cq,
    .event_cq       = (uct_ib_iface_event_cq_func_t)ucs_empty_function,
    .handle_failure = (uct_ib_iface_handle_failure_func_t)
            ucs_empty_function_do_assert
};

static ucs_status_t
uct_srd_iface_create_qp(uct_srd_iface_t *iface,
                        const uct_srd_iface_config_t *config,
                        struct efadv_device_attr *efa_attr)
{
#ifdef HAVE_DECL_EFADV_DEVICE_ATTR_CAPS_RDMA_READ
    struct efadv_qp_init_attr efa_qp_init_attr = {0};
    struct ibv_qp_init_attr_ex qp_init_attr    = {0};
#else
    struct ibv_qp_init_attr qp_init_attr       = {0};
#endif
    uct_ib_efadv_md_t *md                      =
        ucs_derived_of(uct_ib_iface_md(&iface->super), uct_ib_efadv_md_t);
    struct ibv_pd *pd                          = md->super.pd;
    struct ibv_qp_attr qp_attr                 = {0};
    int ret;

    qp_init_attr.qp_type             = IBV_QPT_DRIVER;
    qp_init_attr.sq_sig_all          = 1;
    qp_init_attr.send_cq             = iface->super.cq[UCT_IB_DIR_TX];
    qp_init_attr.recv_cq             = iface->super.cq[UCT_IB_DIR_RX];
    qp_init_attr.cap.max_send_wr     = ucs_min(config->super.tx.queue_len,
                                               efa_attr->max_sq_wr);
    qp_init_attr.cap.max_recv_wr     = ucs_min(config->super.rx.queue_len,
                                               efa_attr->max_rq_wr);
    qp_init_attr.cap.max_send_sge    = ucs_min(config->super.tx.min_sge + 1,
                                               IBV_DEV_ATTR(&md->super.dev,
                                                            max_sge));
    qp_init_attr.cap.max_recv_sge    = 1;
    qp_init_attr.cap.max_inline_data = ucs_min(config->super.tx.min_inline,
                                               md->super.dev.max_inline_data);

#ifdef HAVE_DECL_EFADV_DEVICE_ATTR_CAPS_RDMA_READ
    qp_init_attr.pd             = pd;
    qp_init_attr.comp_mask      = IBV_QP_INIT_ATTR_PD |
                                  IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
    qp_init_attr.send_ops_flags = IBV_QP_EX_WITH_SEND;
    if (uct_ib_efadv_has_rdma_read(efa_attr)) {
        qp_init_attr.send_ops_flags |= IBV_QP_EX_WITH_RDMA_READ;
    }

    efa_qp_init_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;

    iface->qp    = efadv_create_qp_ex(pd->context, &qp_init_attr,
                                      &efa_qp_init_attr, sizeof(efa_qp_init_attr));
    iface->qp_ex = ibv_qp_to_qp_ex(iface->qp);
#else
    iface->qp = efadv_create_driver_qp(pd, &qp_init_attr,
                                       EFADV_QP_DRIVER_TYPE_SRD);
#endif

    if (iface->qp == NULL) {
        ucs_error("iface=%p: failed to create SRD QP on " UCT_IB_IFACE_FMT
                  " TX wr:%d sge:%d inl:%d resp:%d RX wr:%d sge:%d resp:%d: %m",
                  iface, UCT_IB_IFACE_ARG(&iface->super),
                  qp_init_attr.cap.max_send_wr, qp_init_attr.cap.max_send_sge,
                  qp_init_attr.cap.max_inline_data,
                  iface->super.config.max_inl_cqe[UCT_IB_DIR_TX],
                  qp_init_attr.cap.max_recv_wr, qp_init_attr.cap.max_recv_sge,
                  iface->super.config.max_inl_cqe[UCT_IB_DIR_RX]);
        return UCS_ERR_IO_ERROR;
    }

    iface->config.max_send_sge  = qp_init_attr.cap.max_send_sge;
    iface->config.max_recv_sge  = qp_init_attr.cap.max_recv_sge;
    iface->config.max_get_zcopy = efa_attr->max_rdma_size;
    iface->config.max_get_bcopy = ucs_min(iface->super.config.seg_size,
                                          efa_attr->max_rdma_size);
    iface->config.max_inline    = qp_init_attr.cap.max_inline_data;
    iface->config.tx_qp_len     = qp_init_attr.cap.max_send_wr;
    iface->tx.available         = qp_init_attr.cap.max_send_wr;
    iface->rx.available         = qp_init_attr.cap.max_recv_wr;

    ucs_debug("iface=%p: created SRD QP 0x%x on " UCT_IB_IFACE_FMT
              " TX wr:%d sge:%d inl:%d resp:%d RX wr:%d sge:%d resp:%d",
              iface, iface->qp->qp_num, UCT_IB_IFACE_ARG(&iface->super),
              qp_init_attr.cap.max_send_wr, qp_init_attr.cap.max_send_sge,
              qp_init_attr.cap.max_inline_data,
              iface->super.config.max_inl_cqe[UCT_IB_DIR_TX],
              qp_init_attr.cap.max_recv_wr, qp_init_attr.cap.max_recv_sge,
              iface->super.config.max_inl_cqe[UCT_IB_DIR_RX]);

    /* Transition the QP to RTS */
    qp_attr.qp_state   = IBV_QPS_INIT;
    qp_attr.pkey_index = iface->super.pkey_index;
    qp_attr.port_num   = iface->super.config.port_num;
    qp_attr.qkey       = UCT_IB_KEY;
    ret                = ibv_modify_qp(iface->qp, &qp_attr,
                                       IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                       IBV_QP_PORT | IBV_QP_QKEY);
    if (ret != 0) {
        ucs_error("failed to modify SRD QP to INIT: %m");
        goto err_destroy_qp;
    }

    qp_attr.qp_state = IBV_QPS_RTR;
    ret              = ibv_modify_qp(iface->qp, &qp_attr, IBV_QP_STATE);
    if (ret != 0) {
        ucs_error("failed to modify SRD QP to RTR: %m");
        goto err_destroy_qp;
    }

    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn   = 0;
    ret = ibv_modify_qp(iface->qp, &qp_attr, IBV_QP_STATE | IBV_QP_SQ_PSN);
    if (ret != 0) {
        ucs_error("failed to modify SRD QP to RTS: %m");
        goto err_destroy_qp;
    }

    return UCS_OK;

err_destroy_qp:
    uct_ib_destroy_qp(iface->qp);
    return UCS_ERR_INVALID_PARAM;
}

static ucs_mpool_ops_t uct_srd_send_op_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};


static void uct_srd_iface_send_op_purge(uct_srd_iface_t *iface)
{
    uct_srd_send_op_t *send_op;

    while (!ucs_list_is_empty(&iface->tx.outstanding_list)) {
        send_op = ucs_list_extract_head(&iface->tx.outstanding_list,
                                        uct_srd_send_op_t, list);
        ucs_mpool_put(send_op);
    }
}

static UCS_F_ALWAYS_INLINE void uct_srd_iface_post_recv(uct_srd_iface_t *iface)
{
    int max = iface->super.config.rx_max_batch;
    struct ibv_recv_wr *bad_wr;
    uct_ib_recv_wr_t *wrs;
    unsigned count;
    int ret;

    wrs   = ucs_alloca(sizeof(*wrs) * max);
    count = uct_ib_iface_prepare_rx_wrs(&iface->super, &iface->rx.mp, wrs, max);
    if (count != 0) {
        ret = ibv_post_recv(iface->qp, &wrs[0].ibwr, &bad_wr);
        if (ret != 0) {
            ucs_fatal("ibv_post_recv() returned %d: %m", ret);
        }

        iface->rx.available -= count;
    }
}

static void
uct_srd_iface_send_desc_init(uct_iface_h tl_iface, void *obj, uct_mem_h memh)
{
    uct_srd_send_desc_t *desc = obj;

    desc->lkey     = uct_ib_memh_get_lkey(memh);
    desc->super.ep = NULL;
}

#ifdef ENABLE_STATS
static ucs_stats_class_t uct_srd_iface_stats_class = {
    .name         = "srd_iface",
    .num_counters = 0,
    .class_id     = UCS_STATS_CLASS_ID_INVALID,
};
#endif

static UCS_CLASS_INIT_FUNC(uct_srd_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_srd_iface_config_t *config     = ucs_derived_of(tl_config,
                                                        uct_srd_iface_config_t);
    uct_ib_md_t *ib_md                 = ucs_derived_of(md, uct_ib_md_t);
    uct_ib_iface_init_attr_t init_attr = {0};
    ucs_mpool_params_t mp_params;
    struct efadv_device_attr efa_attr;
    ucs_status_t status;
    int mtu, ret;

    UCT_CHECK_PARAM(params->field_mask & UCT_IFACE_PARAM_FIELD_OPEN_MODE,
                    "UCT_IFACE_PARAM_FIELD_OPEN_MODE not set");
    if (!(params->open_mode & UCT_IFACE_OPEN_MODE_DEVICE)) {
        ucs_error("only UCT_IFACE_OPEN_MODE_DEVICE is supported");
        return UCS_ERR_UNSUPPORTED;
    }

    status = uct_ib_device_mtu(params->mode.device.dev_name, md, &mtu);
    if (status != UCS_OK) {
        return status;
    }

    init_attr.cq_len[UCT_IB_DIR_TX] = config->super.tx.queue_len;
    init_attr.cq_len[UCT_IB_DIR_RX] = config->super.rx.queue_len;
    init_attr.rx_priv_len           = sizeof(uct_srd_recv_desc_t) -
                                      sizeof(uct_ib_iface_recv_desc_t);
    init_attr.rx_hdr_len            = sizeof(uct_srd_hdr_t);
    init_attr.seg_size              = ucs_min(mtu, config->super.seg_size);
    init_attr.qp_type               = IBV_QPT_DRIVER;

    UCS_CLASS_CALL_SUPER_INIT(uct_ib_iface_t, &uct_srd_iface_tl_ops,
                              &uct_srd_iface_ops, md, worker, params,
                              &config->super, &init_attr);

    status = UCS_STATS_NODE_ALLOC(&self->stats, &uct_srd_iface_stats_class,
                                  self->super.stats, "-%p", self);
    if (status != UCS_OK) {
        return status;
    }

    ret = efadv_query_device(ib_md->pd->context, &efa_attr, sizeof(efa_attr));
    if (ret != 0) {
        status = UCS_ERR_IO_ERROR;
        ucs_debug("efadv_query_device(%s) failed: %d",
                  ibv_get_device_name(ib_md->pd->context->device), ret);
        goto err_cleanup_stats_node;
    }

    ucs_arbiter_init(&self->tx.pending_q);
    ucs_list_head_init(&self->tx.outstanding_list);

    status = uct_srd_iface_create_qp(self, config, &efa_attr);
    if (status != UCS_OK) {
        goto err_cleanup_stats_node;
    }

    ucs_mpool_params_reset(&mp_params);
    mp_params.name            = "srd_send_op";
    mp_params.elem_size       = sizeof(uct_srd_send_op_t);
    mp_params.align_offset    = 0;
    mp_params.alignment       = UCT_SRD_SEND_OP_ALIGN;
    mp_params.elems_per_chunk = 128;
    mp_params.ops             = &uct_srd_send_op_mpool_ops;
    mp_params.max_elems       = self->tx.available;

    status = ucs_mpool_init(&mp_params, &self->tx.send_op_mp);
    if (status != UCS_OK) {
        goto err_cleanup_qp;
    }

    status = uct_iface_mpool_init(&self->super.super, &self->tx.send_desc_mp,
                                  sizeof(uct_srd_send_desc_t) +
                                      self->super.config.seg_size,
                                  sizeof(uct_srd_send_desc_t),
                                  UCT_SRD_SEND_DESC_ALIGN,
                                  &config->super.tx.mp,
                                  self->config.tx_qp_len,
                                  uct_srd_iface_send_desc_init,
                                  "srd_send_desc");
    if (status != UCS_OK) {
        goto err_cleanup_send_op_mp;
    }

    kh_init_inplace(uct_srd_rx_ctx_hash, &self->rx.ctx_hash);

    status = uct_ib_iface_recv_mpool_init(&self->super, &config->super, params,
                                          "srd_recv_desc", &self->rx.mp);
    if (status != UCS_OK) {
        goto err_cleanup_send_desc_mp;
    }

    uct_ud_send_wr_init(&self->tx.wr_inl, self->tx.sge, 1);
    uct_ud_send_wr_init(&self->tx.wr_desc, self->tx.sge, 0);

    self->super.config.sl = uct_ib_iface_config_select_sl(&config->super);

    while (self->rx.available >= self->super.config.rx_max_batch) {
        uct_srd_iface_post_recv(self);
    }

    return UCS_OK;

err_cleanup_send_desc_mp:
    ucs_mpool_cleanup(&self->tx.send_desc_mp, 1);
err_cleanup_send_op_mp:
    ucs_mpool_cleanup(&self->tx.send_op_mp, 1);
err_cleanup_qp:
    uct_ib_destroy_qp(self->qp);
err_cleanup_stats_node:
    UCS_STATS_NODE_FREE(self->stats);
    return status;
}

static void uct_iface_rx_ctx_cleanup(uct_srd_rx_ctx_t *ctx)
{
    ucs_frag_list_elem_t *elem;
    uct_srd_recv_desc_t *desc;

    while ((elem = ucs_frag_list_remove(&ctx->ooo_pkts, 1)) != NULL) {
        desc = ucs_container_of(elem, uct_srd_recv_desc_t, elem);
        ucs_mpool_put(desc);
    }

    ucs_frag_list_cleanup(&ctx->ooo_pkts);
    ucs_free(ctx);
}

static UCS_CLASS_CLEANUP_FUNC(uct_srd_iface_t)
{
    uct_srd_rx_ctx_t *ctx;

    uct_base_iface_progress_disable(&self->super.super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    uct_srd_iface_send_op_purge(self);
    ucs_arbiter_cleanup(&self->tx.pending_q);
    uct_ib_destroy_qp(self->qp);
    kh_foreach_value(&self->rx.ctx_hash, ctx, {
        uct_iface_rx_ctx_cleanup(ctx);
    });
    kh_destroy_inplace(uct_srd_rx_ctx_hash, &self->rx.ctx_hash);
    ucs_mpool_cleanup(&self->rx.mp, 0);
    ucs_mpool_cleanup(&self->tx.send_op_mp, 1);
    ucs_mpool_cleanup(&self->tx.send_desc_mp, 1);
    UCS_STATS_NODE_FREE(self->stats);
}

UCS_CLASS_DEFINE(uct_srd_iface_t, uct_ib_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_srd_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_srd_iface_t, uct_iface_t);

ucs_config_field_t uct_srd_iface_config_table[] = {
    {UCT_IB_CONFIG_PREFIX, "", NULL,
     ucs_offsetof(uct_srd_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_ib_iface_config_table)},

    {"SRD_", "", NULL, ucs_offsetof(uct_srd_iface_config_t, ud_common),
     UCS_CONFIG_TYPE_TABLE(uct_ud_iface_common_config_table)},

    {NULL}
};

static UCS_F_ALWAYS_INLINE unsigned
uct_srd_iface_poll_tx(uct_srd_iface_t *iface)
{
    unsigned num_wcs = iface->super.config.tx_max_poll;
    struct ibv_wc wc[num_wcs];
    ucs_status_t status;
    int i;

    status = uct_ib_poll_cq(iface->super.cq[UCT_IB_DIR_TX], &num_wcs, wc);
    if (status != UCS_OK) {
        return 0;
    }

    for (i = 0; i < num_wcs; i++) {
        if (ucs_unlikely(wc[i].status != IBV_WC_SUCCESS)) {
            UCT_IB_IFACE_VERBS_COMPLETION_ERR("send", &iface->super, i, wc);
            continue;
        }

        uct_srd_ep_send_op_completion((uct_srd_send_op_t*)wc[i].wr_id);
    }

    iface->tx.available += num_wcs;
    return num_wcs;
}

static UCS_F_ALWAYS_INLINE void
uct_srd_iface_process_rx_desc(uct_srd_iface_t *iface, uct_srd_recv_desc_t *desc)
{
    uct_srd_hdr_t *hdr = uct_ib_iface_recv_desc_hdr(&iface->super,
                                                    &desc->super);

    uct_ib_iface_invoke_am_desc(&iface->super, hdr->id, hdr + 1, desc->length,
                                &desc->super);
}

static uct_srd_rx_ctx_t *
uct_srd_rx_ctx_create(uct_srd_iface_t *iface, uint64_t uuid)
{
    uct_srd_rx_ctx_t *ctx;
    int ret;
    ucs_status_t status;
    khiter_t iter;

    ctx = ucs_malloc(sizeof(*ctx), "uct_srd_rx_ctx_t");
    if (ctx == NULL) {
        ucs_error("iface=%p failed to alloc rx ctx ep_uuid=%zx", iface, uuid);
        return NULL;
    }

    ctx->uuid = uuid;
    status    = ucs_frag_list_init(UCT_SRD_INITIAL_PSN - 1, &ctx->ooo_pkts,
                                   -1 UCS_STATS_ARG(iface->stats));
    if (status != UCS_OK) {
        ucs_error("iface=%p failed to initialize defrag rx ctx ep_uuid=%zx "
                  "status=%s", iface, ctx->uuid, ucs_status_string(status));
        goto fail;
    }

    iter = kh_put(uct_srd_rx_ctx_hash, &iface->rx.ctx_hash, ctx->uuid, &ret);
    if (ret == UCS_KH_PUT_FAILED) {
        ucs_error("iface=%p failed to insert in rx ctx hash ep_uuid=%zx",
                  iface, ctx->uuid);
        ucs_frag_list_cleanup(&ctx->ooo_pkts);
        goto fail;
    }

    kh_value(&iface->rx.ctx_hash, iter) = ctx;
    return ctx;

fail:
    ucs_free(ctx);
    return NULL;
}

static void uct_srd_iface_process_rx(uct_srd_iface_t *iface, uct_srd_hdr_t *hdr,
                                     unsigned length, uct_srd_recv_desc_t *desc)
{
    uct_srd_rx_ctx_t *ctx;
    khiter_t iter;
    ucs_frag_list_ooo_type_t ooo_type;
    ucs_frag_list_elem_t *elem;

    ucs_trace_func("");

    /* Get the context of the remote sender */
    iter = kh_get(uct_srd_rx_ctx_hash, &iface->rx.ctx_hash, hdr->ep_uuid);
    if (ucs_unlikely(iter == kh_end(&iface->rx.ctx_hash))) {
        ctx = uct_srd_rx_ctx_create(iface, hdr->ep_uuid);
        if (ctx == NULL) {
            goto err;
        }
    } else {
        ctx = kh_value(&iface->rx.ctx_hash, iter);
    }

    desc->length = length - sizeof(*hdr);

    ooo_type = ucs_frag_list_insert(&ctx->ooo_pkts, &desc->elem, hdr->psn);
    if (ucs_likely(ooo_type == UCS_FRAG_LIST_INSERT_FAST)) {
        /* Not added to the empty fragment list */
        uct_srd_iface_process_rx_desc(iface, desc);
        return;
    } else if (ooo_type == UCS_FRAG_LIST_INSERT_FIRST) {
        /* Not added to the fragment list, more to pull */
        uct_srd_iface_process_rx_desc(iface, desc);
    } else if (ooo_type == UCS_FRAG_LIST_INSERT_SLOW) {
        /* Inserted, but nothing can be pulled from fragment list */
        return;
    } else if (ooo_type != UCS_FRAG_LIST_INSERT_READY) {
        ucs_error("iface=%p failed fragment insert: ooo_type=%d ctx=%p "
                  "ep_uuid=%"PRIx64" head_sn=%u psn=%u",
                  iface, ooo_type, ctx, ctx->uuid, ctx->ooo_pkts.head_sn,
                  hdr->psn);
        goto err;
    }

    while ((elem = ucs_frag_list_pull(&ctx->ooo_pkts)) != NULL) {
        desc = ucs_container_of(elem, uct_srd_recv_desc_t, elem);
        uct_srd_iface_process_rx_desc(iface, desc);
    }

    return;

err:
    ucs_mpool_put(desc);
}

static UCS_F_ALWAYS_INLINE unsigned
uct_srd_iface_poll_rx(uct_srd_iface_t *iface)
{
    unsigned num_wcs = iface->super.config.rx_max_poll;
    struct ibv_wc wc[num_wcs];
    ucs_status_t status;
    void *packet;
    int i;

    status = uct_ib_poll_cq(iface->super.cq[UCT_IB_DIR_RX], &num_wcs, wc);
    if (status != UCS_OK) {
        num_wcs = 0;
        goto out;
    }

    UCT_IB_IFACE_VERBS_FOREACH_RXWQE(&iface->super, i, packet, wc, num_wcs) {
        uct_ib_log_recv_completion(&iface->super, &wc[i], packet,
                                   wc[i].byte_len, uct_srd_dump_packet);
        uct_srd_iface_process_rx(iface, (uct_srd_hdr_t*)packet, wc[i].byte_len,
                                 (uct_srd_recv_desc_t*)wc[i].wr_id);
    }

    iface->rx.available += num_wcs;

out:
    if (iface->rx.available >= iface->super.config.rx_max_batch) {
        uct_srd_iface_post_recv(iface);
    }

    return num_wcs;
}

static unsigned uct_srd_iface_progress(uct_iface_h tl_iface)
{
    uct_srd_iface_t *iface = ucs_derived_of(tl_iface, uct_srd_iface_t);
    unsigned count;

    count = uct_srd_iface_poll_rx(iface);
    if (count == 0) {
        count = uct_srd_iface_poll_tx(iface);
    }

    return count;
}

ucs_status_t
uct_srd_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_srd_iface_t *iface = ucs_derived_of(tl_iface, uct_srd_iface_t);
    size_t active_mtu      = uct_ib_mtu_value(
            uct_ib_iface_port_attr(&iface->super)->active_mtu);
    ucs_status_t status;

    /* Common parameters */
    status = uct_ib_iface_query(&iface->super,
                                UCT_IB_DETH_LEN + sizeof(uct_srd_hdr_t),
                                iface_attr);

    /* General attributes */
    iface_attr->cap.am.align_mtu        = active_mtu;
    iface_attr->cap.get.align_mtu       = active_mtu;
    iface_attr->cap.am.opt_zcopy_align  = UCS_SYS_PCI_MAX_PAYLOAD;
    iface_attr->cap.get.opt_zcopy_align = UCS_SYS_PCI_MAX_PAYLOAD;

    iface_attr->cap.flags = UCT_IFACE_FLAG_AM_BCOPY | UCT_IFACE_FLAG_AM_ZCOPY |
                            UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                            UCT_IFACE_FLAG_PENDING | UCT_IFACE_FLAG_EP_CHECK |
                            UCT_IFACE_FLAG_CB_SYNC |
                            UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;
    iface_attr->iface_addr_len = sizeof(uct_srd_iface_addr_t);
    iface_attr->ep_addr_len    = 0;
    iface_attr->max_conn_priv  = 0;

    iface_attr->latency.c += 30e-9;
    iface_attr->overhead   = 105e-9;

    /* AM */
    iface_attr->cap.am.max_bcopy = iface->super.config.seg_size -
                                   sizeof(uct_srd_hdr_t);
    iface_attr->cap.am.min_zcopy = 0;
    iface_attr->cap.am.max_zcopy = iface->super.config.seg_size -
                                   sizeof(uct_srd_hdr_t);
    iface_attr->cap.am.max_iov   = iface->config.max_send_sge - 1;
    iface_attr->cap.am.max_hdr   = uct_ib_iface_hdr_size(
            iface->super.config.seg_size, sizeof(uct_srd_hdr_t));
    iface_attr->cap.am.max_short = uct_ib_iface_hdr_size(
            iface->config.max_inline, sizeof(uct_srd_hdr_t));

    /* GET */
    iface_attr->cap.get.max_bcopy = iface->config.max_get_bcopy;
    iface_attr->cap.get.max_zcopy = iface->config.max_get_zcopy;
    iface_attr->cap.get.max_iov   = iface->config.max_send_sge;
    iface_attr->cap.get.min_zcopy =
            iface->super.config.max_inl_cqe[UCT_IB_DIR_TX] + 1;

    return status;
}

static ucs_status_t
uct_srd_query_tl_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                         unsigned *num_tl_devices_p)
{
    uct_ib_md_t *ib_md = ucs_derived_of(md, uct_ib_md_t);
    struct ibv_context *ctx;
    struct efadv_device_attr efa_attr;
    int ret;

    ctx = ibv_open_device(ib_md->dev.ibv_context->device);
    if (ctx == NULL) {
        return UCS_ERR_NO_DEVICE;
    }

    ret = efadv_query_device(ctx, &efa_attr, sizeof(efa_attr));
    ibv_close_device(ctx);
    if (ret != 0) {
        return UCS_ERR_NO_DEVICE;
    }

    return uct_ib_device_query_ports(&ib_md->dev, 0, tl_devices_p,
                                     num_tl_devices_p);
}

static uct_iface_ops_t uct_srd_iface_tl_ops = {
    .ep_flush                 = (uct_ep_flush_func_t)
        ucs_empty_function_return_unsupported,
    .ep_fence                 = (uct_ep_fence_func_t)
        ucs_empty_function_return_unsupported,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_srd_ep_t),
    .ep_get_address           = (uct_ep_get_address_func_t)
        ucs_empty_function_return_unsupported,
    .ep_connect_to_ep         = (uct_ep_connect_to_ep_func_t)
        ucs_empty_function_return_unsupported,
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_srd_ep_t),
    .ep_am_bcopy              = uct_srd_ep_am_bcopy,
    .ep_am_zcopy              = uct_srd_ep_am_zcopy,
    .ep_get_zcopy             = uct_srd_ep_get_zcopy,
    .ep_get_bcopy             = uct_srd_ep_get_bcopy,
    .ep_am_short              = uct_srd_ep_am_short,
    .ep_am_short_iov          = uct_srd_ep_am_short_iov,
    .ep_pending_add           = (uct_ep_pending_add_func_t)
        ucs_empty_function_return_unsupported,
    .ep_pending_purge         = (uct_ep_pending_purge_func_t)
        ucs_empty_function_return_unsupported,
    .iface_flush              = uct_base_iface_flush,
    .iface_fence              = (uct_iface_fence_func_t)
        ucs_empty_function_return_unsupported,
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_srd_iface_progress,
    .iface_query              = uct_srd_iface_query,
    .iface_get_address        = uct_srd_iface_get_address,
    .iface_is_reachable       = uct_base_iface_is_reachable,
    .iface_event_fd_get       = (uct_iface_event_fd_get_func_t)
        ucs_empty_function_return_unsupported,
    .iface_event_arm          = (uct_iface_event_arm_func_t)
        ucs_empty_function_return_unsupported,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_srd_iface_t),
    .iface_get_device_address = uct_ib_iface_get_device_address
};

UCT_TL_DEFINE_ENTRY(&uct_ib_component, srd, uct_srd_query_tl_devices,
                    uct_srd_iface_t, "SRD_", uct_srd_iface_config_table,
                    uct_srd_iface_config_t);
