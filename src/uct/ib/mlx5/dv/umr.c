/**
* Copyright (C) 2023, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "umr.h"

#include <uct/ib/mlx5/ib_mlx5.inl>
#include <uct/ib/mlx5/dv/ib_mlx5_dv.h>
#include <uct/ib/rc/accel/rc_mlx5.h>
#include <uct/base/uct_iface.c>
#include <ucs/debug/log.h>

extern uct_tl_t UCT_TL_NAME(rc_mlx5);

typedef struct uct_ib_mlx5_umr {
    ucs_async_context_t async;
    uct_worker_h worker;
    uct_iface_h iface;
    uct_ep_h ep;
} uct_ib_mlx5_umr_t;

ucs_status_t uct_ib_umr_init(uct_md_t *md, uct_ib_mlx5_umr_h *umr_p)
{
    uct_ib_md_t *ib_md   = ucs_derived_of(md, uct_ib_md_t);
    uct_ib_mlx5_umr_t *umr;
    uct_worker_h worker;
    uct_iface_params_t params = {};
    uct_iface_config_t *config;
    uct_ep_params_t ep_params;
    uct_iface_h iface;
    uct_ep_h ep;
    ucs_status_t status;
    uct_iface_attr_t iface_attr;
    uct_device_addr_t *dev_addr;
    uct_ep_addr_t *ep_addr;
    uct_config_bundle_t *bundle = NULL;
    uct_tl_t *tl = &UCT_TL_NAME(rc_mlx5);
    char dev_name[32];

    umr = ucs_malloc(sizeof(*umr), "umr ctx");
    if (umr == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    status = ucs_async_context_init(&umr->async, UCS_ASYNC_MODE_LAST);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_worker_create(&umr->async, UCS_THREAD_MODE_LAST, &worker);
    if (status != UCS_OK) {
        goto err_async;
    }

    status = uct_config_read(&bundle, &tl->config, NULL);
    if (status != UCS_OK) {
        goto err_worker;
    }

    config = (uct_iface_config_t *)bundle->data;
    uct_config_modify(config, "RC_MMIO_MODE", "db");

    params.open_mode = UCT_IFACE_OPEN_MODE_DEVICE;
    sprintf(dev_name, "%s:%d", uct_ib_device_name(&ib_md->dev), 1);
    params.mode.device.dev_name = (const char *)dev_name;

    status = tl->iface_open(md, worker, &params, config, &iface);
    if (status != UCS_OK) {
        goto err_worker;
    }

    ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
    ep_params.iface      = iface;

    status = iface->ops.ep_create(&ep_params, &ep);
    if (status != UCS_OK) {
        goto err_iface;
    }

    status = iface->ops.iface_query(iface, &iface_attr);
    if (status != UCS_OK) {
        goto err_ep;
    }

    dev_addr = ucs_alloca(iface_attr.device_addr_len);
    ep_addr  = ucs_alloca(iface_attr.ep_addr_len);

    status = iface->ops.iface_get_device_address(iface, dev_addr);
    if (status != UCS_OK) {
        goto err_ep;
    }

    status = iface->ops.ep_get_address(ep, ep_addr);
    if (status != UCS_OK) {
        goto err_ep;
    }

    status = iface->ops.ep_connect_to_ep(ep, dev_addr, ep_addr);
    if (status != UCS_OK) {
        goto err_ep;
    }

    iface->ops.iface_progress_enable(iface, UCT_PROGRESS_SEND |
                                            UCT_PROGRESS_RECV);

    umr->worker = worker;
    umr->iface = iface;
    umr->ep = ep;
    *umr_p = umr;

    return UCS_OK;

err_ep:
    iface->ops.ep_destroy(ep);
err_iface:
    iface->ops.iface_close(iface);
err_config:
    uct_config_release(config);
err_worker:
    uct_worker_destroy(worker);
err_async:
    ucs_async_context_cleanup(&umr->async);
err:
    ucs_free(umr);
    return status;
}

void uct_ib_umr_cleanup(uct_ib_mlx5_umr_h umr)
{
    umr->iface->ops.iface_progress_disable(umr->iface, UCT_PROGRESS_SEND |
                                                       UCT_PROGRESS_RECV);
    umr->iface->ops.ep_destroy(umr->ep);
    umr->iface->ops.iface_close(umr->iface);
    uct_worker_destroy(umr->worker);
    ucs_async_context_cleanup(&umr->async);
}

uct_ib_mlx5_txwq_t *uct_ib_umr_get_txwq(uct_ib_mlx5_umr_h umr)
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(umr->ep, uct_rc_mlx5_ep_t);

    return &ep->tx.wq;
}

ucs_status_t uct_ib_umr_post(uct_ib_mlx5_umr_h umr, void *wqe_end,
                             uint32_t lkey)
{
    uct_rc_mlx5_ep_t *ep           = ucs_derived_of(umr->ep, uct_rc_mlx5_ep_t);
    uct_rc_txqp_t *txqp            = &ep->super.txqp;
    uct_ib_mlx5_txwq_t *txwq       = &ep->tx.wq;
    struct mlx5_wqe_ctrl_seg *ctrl = txwq->curr;
    size_t wqe_size                = uct_ib_mlx5_txwq_diff(txwq, ctrl, wqe_end);

    uct_ib_mlx5_set_ctrl_seg_with_imm(ctrl, txwq->sw_pi, MLX5_OPCODE_UMR, 0,
            txwq->super.qp_num, MLX5_WQE_CTRL_CQ_UPDATE, 0, wqe_size, lkey);

    txqp->available -= uct_ib_mlx5_post_send(txwq, ctrl, wqe_size, 1);
    txwq->sig_pi = txwq->prev_sw_pi;

    while (umr->iface->ops.iface_progress(umr->iface) == 0)
        ;

    return UCS_OK;
}
