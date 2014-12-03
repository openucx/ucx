/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_mlx5.h"

#include <uct/api/uct.h>
#include <uct/ib/base/ib_context.h>
#include <uct/tl/context.h>
#include <ucs/debug/log.h>


ucs_config_field_t uct_rc_mlx5_iface_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_rc_mlx5_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_rc_iface_config_table)},

  {NULL}
};

static void uct_rc_mlx5_iface_progress(void *arg)
{
    uct_rc_mlx5_iface_t *iface = arg;
    struct mlx5_cqe64 *cqe;
    uct_rc_mlx5_ep_t *ep;
    unsigned index, qp_num;

    index = iface->tx.cq_ci;
    cqe   = iface->tx.cq_buf + (index & (iface->tx.cq_length - 1)) * sizeof(struct mlx5_cqe64);
    if (uct_ib_mlx5_cqe_hw_owned(cqe, index, iface->tx.cq_length)) {
        return; /* CQ is empty */
    }

    iface->tx.cq_ci = index + 1;
    --iface->super.tx.outstanding;

    ucs_memory_cpu_load_fence();

    qp_num = ntohl(cqe->sop_drop_qpn) & 0xffffff;
    ep = ucs_derived_of(uct_rc_iface_lookup_ep(&iface->super, qp_num), uct_rc_mlx5_ep_t);
    ucs_assert(ep != NULL);

    ++ep->tx.max_pi;
}

static ucs_status_t uct_rc_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);

    uct_rc_iface_query(iface, iface_attr);
    iface_attr->max_short = MLX5_SEND_WQE_BB - sizeof(uct_rc_mlx5_wqe_rdma_inl_seg_t);  /* TODO */
    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_rc_mlx5_iface_t, uct_context_h context,
                           const char *dev_name, size_t rx_headroom,
                           uct_iface_config_t *tl_config)
{
    uct_rc_mlx5_iface_config_t *config = ucs_derived_of(tl_config, uct_rc_mlx5_iface_config_t);
    uct_ib_mlx5_cq_info_t cq_info;
    ucs_status_t status;
    int ret;

    extern uct_iface_ops_t uct_rc_mlx5_iface_ops;
    UCS_CLASS_CALL_SUPER_INIT(&uct_rc_mlx5_iface_ops, context, dev_name,
                              rx_headroom, &config->super);

    ret = ibv_exp_cq_ignore_overrun(self->super.super.send_cq);
    if (ret != 0) {
        ucs_error("Failed to modify send CQ to ignore overrun: %s", strerror(ret));
        return UCS_ERR_UNSUPPORTED;
    }

    status = uct_ib_mlx5_get_cq_info(self->super.super.send_cq, &cq_info);
    if (status != UCS_OK) {
        ucs_error("Failed to get mlx5 CQ information");
        return status;
    }

    self->tx.cq_buf      = cq_info.buf;
    self->tx.cq_ci       = 0;
    self->tx.cq_length   = cq_info.cqe_cnt;

    ucs_notifier_chain_add(&context->progress_chain, uct_rc_mlx5_iface_progress,
                           self);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_iface_t)
{
    uct_context_h context = self->super.super.super.pd->context;
    ucs_notifier_chain_remove(&context->progress_chain, uct_rc_mlx5_iface_progress, self);
}


UCS_CLASS_DEFINE(uct_rc_mlx5_iface_t, uct_rc_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_mlx5_iface_t, uct_iface_t, uct_context_h,
                                 const char*, size_t, uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_mlx5_iface_t, uct_iface_t);

uct_iface_ops_t uct_rc_mlx5_iface_ops = {
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_mlx5_iface_t),
    .iface_get_address   = uct_rc_iface_get_address,
    .iface_flush         = uct_rc_iface_flush,
    .ep_get_address      = uct_rc_ep_get_address,
    .ep_connect_to_iface = NULL,
    .ep_connect_to_ep    = uct_rc_ep_connect_to_ep,
    .iface_query         = uct_rc_mlx5_iface_query,
    .ep_put_short        = uct_rc_mlx5_ep_put_short,
    .ep_create           = UCS_CLASS_NEW_FUNC_NAME(uct_rc_mlx5_ep_t),
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_mlx5_ep_t),
};


static ucs_status_t uct_rc_mlx5_query_resources(uct_context_h context,
                                                uct_resource_desc_t **resources_p,
                                                unsigned *num_resources_p)
{
    /* TODO take transport overhead into account */
    return uct_ib_query_resources(context, UCT_IB_RESOURCE_FLAG_MLX5_PRM,
                                  resources_p, num_resources_p);
}

static uct_tl_ops_t uct_rc_mlx5_tl_ops = {
    .query_resources     = uct_rc_mlx5_query_resources,
    .iface_open          = UCS_CLASS_NEW_FUNC_NAME(uct_rc_mlx5_iface_t),
    .rkey_unpack         = uct_ib_rkey_unpack,
};

static void uct_rc_mlx5_register(uct_context_t *context)
{
    uct_register_tl(context, "rc_mlx5", uct_rc_mlx5_iface_config_table,
                    sizeof(uct_rc_mlx5_iface_config_t), &uct_rc_mlx5_tl_ops);
}

UCS_COMPONENT_DEFINE(uct_context_t, rc_mlx5, uct_rc_mlx5_register, ucs_empty_function, 0)
