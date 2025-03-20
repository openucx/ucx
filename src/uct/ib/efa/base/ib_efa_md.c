/**
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/ib/efa/base/ib_efa.h>
#include <ucs/type/status.h>
#include <ucs/debug/log.h>


static uct_ib_md_ops_t uct_ib_efa_md_ops;

static ucs_status_t uct_ib_efa_md_open(struct ibv_device *ibv_device,
                                       const uct_ib_md_config_t *md_config,
                                       uct_ib_md_t **p_md)
{
    struct ibv_context *ctx;
    uct_ib_efadv_md_t *md;
    uct_ib_device_t *dev;
    struct efadv_device_attr attr;
    ucs_status_t status;
    int ret;
    uint64_t access_flags;

    ctx = ibv_open_device(ibv_device);
    if (ctx == NULL) {
        ucs_diag("ibv_open_device(%s) failed: %m",
                 ibv_get_device_name(ibv_device));
        return UCS_ERR_IO_ERROR;
    }

    ret = efadv_query_device(ctx, &attr, sizeof(attr));
    if (ret != 0) {
        ucs_debug("efadv_query_device(%s) failed: %d",
                  ibv_get_device_name(ibv_device), ret);
        status = UCS_ERR_UNSUPPORTED;
        goto err_free_context;
    }

    md = ucs_derived_of(uct_ib_md_alloc(sizeof(*md), "ib_efadv_md", ctx),
                        uct_ib_efadv_md_t);
    if (md == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_context;
    }

    md->super.super.ops = &uct_ib_efa_md_ops.super;
    md->super.name      = UCT_IB_MD_NAME(efa);

    status = uct_ib_device_query(&md->super.dev, ibv_device);
    if (status != UCS_OK) {
        goto err_md_free;
    }

    if (uct_ib_efadv_has_rdma_read(&attr)) {
        access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ;
    } else {
        access_flags = IBV_ACCESS_LOCAL_WRITE;
    }

    dev                        = &md->super.dev;
    dev->mr_access_flags       = access_flags;
    dev->max_inline_data       = attr.inline_buf_size;
    dev->ordered_send_comp     = 0;
    /*
     * FIXME: Always disabling channel completion because of leak (gtest):
     * - https://github.com/amzn/amzn-drivers/issues/306
     */
    dev->req_notify_cq_support = 0;

    status = uct_ib_md_open_common(&md->super, ibv_device, md_config);
    if (status != UCS_OK) {
        goto err_md_free;
    }

    *p_md = &md->super;
    return UCS_OK;

err_md_free:
    uct_ib_md_free(&md->super);
err_free_context:
    uct_ib_md_device_context_close(ctx);
    return status;
}

static uct_ib_md_ops_t uct_ib_efa_md_ops = {
    .super = {
        .close              = uct_ib_md_close,
        .query              = uct_ib_md_query,
        .mem_alloc          = (uct_md_mem_alloc_func_t)ucs_empty_function_return_unsupported,
        .mem_free           = (uct_md_mem_free_func_t)ucs_empty_function_return_unsupported,
        .mem_advise         = uct_ib_mem_advise,
        .mem_reg            = uct_ib_verbs_mem_reg,
        .mem_dereg          = uct_ib_verbs_mem_dereg,
        .mem_query          = (uct_md_mem_query_func_t)ucs_empty_function_return_unsupported,
        .mkey_pack          = uct_ib_verbs_mkey_pack,
        .mem_attach         =(uct_md_mem_attach_func_t)ucs_empty_function_return_unsupported,
        .detect_memory_type = (uct_md_detect_memory_type_func_t)ucs_empty_function_return_unsupported,
    },
    .open = uct_ib_efa_md_open,
};

UCT_IB_MD_DEFINE_ENTRY(efa, uct_ib_efa_md_ops);

extern uct_tl_t UCT_TL_NAME(srd);

void UCS_F_CTOR uct_efa_init(void)
{
    ucs_list_add_head(&uct_ib_ops, &UCT_IB_MD_OPS_NAME(efa).list);
    uct_tl_register(&uct_ib_component, &UCT_TL_NAME(srd));
}

void UCS_F_DTOR uct_efa_cleanup(void)
{
    uct_tl_unregister(&UCT_TL_NAME(srd));
    ucs_list_del(&UCT_IB_MD_OPS_NAME(efa).list);
}
