/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

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
#ifdef HAVE_DECL_EFA_DV_RDMA_READ
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
                                               efa_attr->max_sq_sge);
    qp_init_attr.cap.max_recv_sge    = 1;
    qp_init_attr.cap.max_inline_data = ucs_min(config->super.tx.min_inline,
                                               md->super.dev.max_inline_data);

#ifdef HAVE_DECL_EFA_DV_RDMA_READ
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

    iface->config.max_inline = qp_init_attr.cap.max_inline_data;
    iface->config.tx_qp_len  = qp_init_attr.cap.max_send_wr;
    iface->tx.available      = qp_init_attr.cap.max_send_wr;
    iface->rx.available      = qp_init_attr.cap.max_recv_wr;

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

static UCS_F_NOINLINE void
uct_srd_iface_post_recv_always(uct_srd_iface_t *iface, int max)
{
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

static UCS_F_ALWAYS_INLINE void
uct_srd_iface_post_recv(uct_srd_iface_t *iface)
{
    unsigned batch = iface->super.config.rx_max_batch;

    if (iface->rx.available >= batch) {
        uct_srd_iface_post_recv_always(iface, batch);
    }
}

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
    init_attr.rx_hdr_len            = sizeof(uct_srd_hdr_t);
    init_attr.seg_size              = ucs_min(mtu, config->super.seg_size);
    init_attr.qp_type               = IBV_QPT_DRIVER;

    UCS_CLASS_CALL_SUPER_INIT(uct_ib_iface_t, &uct_srd_iface_tl_ops,
                              &uct_srd_iface_ops, md, worker, params,
                              &config->super, &init_attr);

    ret = efadv_query_device(ib_md->pd->context, &efa_attr, sizeof(efa_attr));
    if (ret != 0) {
        ucs_debug("efadv_query_device(%s) failed: %d",
                  ibv_get_device_name(ib_md->pd->context->device), ret);
        return UCS_ERR_IO_ERROR;
    }

    ucs_arbiter_init(&self->tx.pending_q);
    ucs_list_head_init(&self->tx.outstanding_list);

    ucs_mpool_params_reset(&mp_params);
    mp_params.name            = "srd_send_op";
    mp_params.elem_size       = sizeof(uct_srd_send_op_t);
    mp_params.align_offset    = 0;
    mp_params.alignment       = UCT_SRD_SEND_OP_ALIGN;
    mp_params.elems_per_chunk = 256;
    mp_params.ops             = &uct_srd_send_op_mpool_ops;

    status = ucs_mpool_init(&mp_params, &self->tx.send_op_mp);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_iface_recv_mpool_init(&self->super, &config->super, params,
                                          "srd_recv_desc", &self->rx.mp);
    if (status != UCS_OK) {
        goto err_cleanup_send_op_mp;
    }

    status = uct_srd_iface_create_qp(self, config, &efa_attr);
    if (status != UCS_OK) {
        goto err_cleanup_rx_mp;
    }

    uct_ud_send_wr_init(&self->tx.wr_inl, self->tx.sge, 1);
    uct_ud_send_wr_init(&self->tx.wr_desc, self->tx.sge, 0);

    self->super.config.sl      = uct_ib_iface_config_select_sl(&config->super);
    self->config.max_send_sge  = efa_attr.max_sq_sge;
    self->config.max_get_zcopy = efa_attr.max_rdma_size;

    uct_srd_iface_post_recv(self);

    return UCS_OK;

err_cleanup_rx_mp:
    ucs_mpool_cleanup(&self->rx.mp, 1);
err_cleanup_send_op_mp:
    ucs_mpool_cleanup(&self->tx.send_op_mp, 1);
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_srd_iface_t)
{
    uct_base_iface_progress_disable(&self->super.super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    uct_srd_iface_send_op_purge(self);
    ucs_arbiter_cleanup(&self->tx.pending_q);
    uct_ib_destroy_qp(self->qp);
    ucs_mpool_cleanup(&self->rx.mp, 0);
    ucs_mpool_cleanup(&self->tx.send_op_mp, 1);
    ucs_assertv(ucs_list_is_empty(&self->tx.outstanding_list),
                "iface=%p tx outstanding list is not empty", self);
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

static unsigned uct_srd_iface_progress(uct_iface_h tl_iface)
{
    uct_srd_iface_t *iface = ucs_derived_of(tl_iface, uct_srd_iface_t);

    return uct_srd_iface_poll_tx(iface);
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
    iface_attr->cap.get.max_bcopy = iface->super.config.seg_size;
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
    .ep_am_bcopy              = (uct_ep_am_bcopy_func_t)
        ucs_empty_function_return_unsupported,
    .ep_am_zcopy              = (uct_ep_am_zcopy_func_t)
        ucs_empty_function_return_unsupported,
    .ep_get_zcopy             = (uct_ep_get_zcopy_func_t)
        ucs_empty_function_return_unsupported,
    .ep_am_short              = uct_srd_ep_am_short,
    .ep_am_short_iov          = (uct_ep_am_short_iov_func_t)
        ucs_empty_function_return_unsupported,
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
