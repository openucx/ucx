/**
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/ib/efa/ib_efa.h>
#include <uct/ib/base/ib_md.h>
#include <ucs/type/status.h>
#include <ucs/debug/log.h>


typedef struct uct_ib_efa_mem {
    uct_ib_mem_t        super;
    uct_ib_mr_t         mrs[];
} uct_ib_efa_mem_t;

static uint64_t
uct_ib_efadv_access_flags(const uct_ib_efadv_t *efadv)
{
    uint64_t access_flags = IBV_ACCESS_LOCAL_WRITE;
#ifdef HAVE_DECL_EFA_DV_RDMA_READ
    if (efadv->efadv_attr.device_caps & EFADV_DEVICE_ATTR_CAPS_RDMA_READ) {
        access_flags |= IBV_ACCESS_REMOTE_READ;
    }
#endif
    return access_flags;
}

static ucs_status_t uct_ib_efa_reg_key(uct_ib_md_t *md, void *address,
                                       size_t length, uint64_t access_flags,
                                       uct_ib_mem_t *ib_memh,
                                       uct_ib_mr_type_t mr_type, int silent)
{
    uct_ib_efa_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_efa_mem_t);

    return uct_ib_reg_key_impl(md, address, length, access_flags,
                               ib_memh, &memh->mrs[mr_type], mr_type, silent);
}

static ucs_status_t uct_ib_efa_dereg_key(uct_ib_md_t *md,
                                         uct_ib_mem_t *ib_memh,
                                         uct_ib_mr_type_t mr_type)
{
    uct_ib_efa_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_efa_mem_t);

    return uct_ib_dereg_mr(memh->mrs[mr_type].ib);
}

static uct_ib_md_ops_t uct_ib_efa_md_ops;

static ucs_status_t uct_ib_efa_md_open(struct ibv_device *ibv_device,
                                       const uct_ib_md_config_t *md_config,
                                       uct_ib_md_t **p_md)
{
    ucs_status_t status;
    uct_ib_device_t *dev;
    uct_ib_efadv_md_t *md;
    int num_mrs;

    md = ucs_calloc(1, sizeof(*md), "ib_efa_md");
    if (md == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    dev = &md->super.dev;
    dev->ibv_context = ibv_open_device(ibv_device);
    if (dev->ibv_context == NULL) {
        ucs_warn("ibv_open_device(%s) failed: %m", ibv_get_device_name(ibv_device));
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    md->super.config = md_config->ext;

    status = uct_ib_device_query(dev, ibv_device);
    if (status != UCS_OK) {
        goto err_free_context;
    }

    dev->flags = uct_ib_device_spec(dev)->flags;

    if (!(dev->flags & UCT_IB_DEVICE_FLAG_EFA)) {
        status = UCS_ERR_UNSUPPORTED;
        goto err_free_context;
    }

    if (UCT_IB_HAVE_ODP_IMPLICIT(&dev->dev_attr)) {
        dev->flags |= UCT_IB_DEVICE_FLAG_ODP_IMPLICIT;
    }

    if (IBV_EXP_HAVE_ATOMIC_HCA(&dev->dev_attr)) {
        dev->atomic_arg_sizes = sizeof(uint64_t);
    }

    md->super.ops = &uct_ib_efa_md_ops;

    uct_ib_md_parse_relaxed_order(&md->super, md_config);
    num_mrs = 1;      /* UCT_IB_MR_DEFAULT */

    if (md->super.relaxed_order) {
        ++num_mrs;    /* UCT_IB_MR_STRICT_ORDER */
    }

    md->super.memh_struct_size = sizeof(uct_ib_efa_mem_t) +
                                 (sizeof(uct_ib_mr_t) * num_mrs);

    status = uct_ib_md_open_common(&md->super, ibv_device, md_config);
    if (status != UCS_OK) {
        goto err_free_context;
    }

    status = uct_ib_efadv_query(dev->ibv_context, &md->efadv.efadv_attr);
    if (status != UCS_OK) {
        goto err_free_context;
    }

    dev->access_flags = uct_ib_efadv_access_flags(&md->efadv);

    *p_md = &md->super;
    return UCS_OK;

err_free_context:
    ibv_close_device(dev->ibv_context);
err:
    ucs_free(md);
    return status;
}

static uct_ib_md_ops_t uct_ib_efa_md_ops = {
    .open                = uct_ib_efa_md_open,
    .cleanup             = (uct_ib_md_cleanup_func_t)ucs_empty_function,
    .reg_key             = uct_ib_efa_reg_key,
    .dereg_key           = uct_ib_efa_dereg_key,
    .reg_atomic_key      = (uct_ib_md_reg_atomic_key_func_t)ucs_empty_function_return_unsupported,
    .dereg_atomic_key    = (uct_ib_md_dereg_atomic_key_func_t)ucs_empty_function_return_success,
    .reg_multithreaded   = (uct_ib_md_reg_multithreaded_func_t)ucs_empty_function_return_unsupported,
    .dereg_multithreaded = (uct_ib_md_dereg_multithreaded_func_t)ucs_empty_function_return_unsupported,
    .mem_prefetch        = (uct_ib_md_mem_prefetch_func_t)ucs_empty_function_return_success,
    .get_atomic_mr_id    = (uct_ib_md_get_atomic_mr_id_func_t)ucs_empty_function_return_unsupported,
};

UCT_IB_MD_OPS(uct_ib_efa_md_ops, 1);
