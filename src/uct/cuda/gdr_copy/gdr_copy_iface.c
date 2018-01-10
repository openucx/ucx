/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "gdr_copy_iface.h"
#include "gdr_copy_md.h"
#include "gdr_copy_ep.h"

#include <ucs/type/class.h>
#include <ucs/sys/string.h>


static ucs_config_field_t uct_gdr_copy_iface_config_table[] = {

    {"", "", NULL,
     ucs_offsetof(uct_gdr_copy_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {NULL}
};

/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_gdr_copy_iface_t)(uct_iface_t*);

static ucs_status_t uct_gdr_copy_iface_get_address(uct_iface_h tl_iface,
                                                   uct_iface_addr_t *iface_addr)
{
    int *gdr_copy_addr = (int*)iface_addr;

    *gdr_copy_addr = 0;
    return UCS_OK;
}

static int uct_gdr_copy_iface_is_reachable(const uct_iface_h iface,
                                           const uct_device_addr_t *dev_addr,
                                           const uct_iface_addr_t *iface_addr)
{
    return 1;
}

static ucs_status_t uct_gdr_copy_iface_query(uct_iface_h iface,
                                             uct_iface_attr_t *iface_attr)
{
    memset(iface_attr, 0, sizeof(uct_iface_attr_t));

    iface_attr->iface_addr_len          = sizeof(int);
    iface_attr->device_addr_len         = 0;
    iface_attr->ep_addr_len             = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                          UCT_IFACE_FLAG_PUT_SHORT;

    iface_attr->cap.put.max_short       = UINT_MAX;
    iface_attr->cap.put.max_bcopy       = 0;
    iface_attr->cap.put.min_zcopy       = 0;
    iface_attr->cap.put.max_zcopy       = 0;
    iface_attr->cap.put.opt_zcopy_align = 1;
    iface_attr->cap.put.align_mtu       = iface_attr->cap.put.opt_zcopy_align;
    iface_attr->cap.put.max_iov         = 1;

    iface_attr->cap.get.max_bcopy       = 0;
    iface_attr->cap.get.min_zcopy       = 0;
    iface_attr->cap.get.max_zcopy       = 0;
    iface_attr->cap.get.opt_zcopy_align = 1;
    iface_attr->cap.get.align_mtu       = iface_attr->cap.get.opt_zcopy_align;
    iface_attr->cap.get.max_iov         = 1;

    iface_attr->cap.am.max_short        = 0;
    iface_attr->cap.am.max_bcopy        = 0;
    iface_attr->cap.am.min_zcopy        = 0;
    iface_attr->cap.am.max_zcopy        = 0;
    iface_attr->cap.am.opt_zcopy_align  = 1;
    iface_attr->cap.am.align_mtu        = iface_attr->cap.am.opt_zcopy_align;
    iface_attr->cap.am.max_hdr          = 0;
    iface_attr->cap.am.max_iov          = 1;

    iface_attr->latency.overhead        = 1e-6; /* 1 us */
    iface_attr->latency.growth          = 0;
    iface_attr->bandwidth               = 6911 * 1024.0 * 1024.0;
    iface_attr->overhead                = 0;
    iface_attr->priority                = 0;

    return UCS_OK;
}

static uct_iface_ops_t uct_gdr_copy_iface_ops = {
    .ep_put_short             = uct_gdr_copy_ep_put_short,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create_connected      = UCS_CLASS_NEW_FUNC_NAME(uct_gdr_copy_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_gdr_copy_ep_t),
    .iface_flush              = uct_base_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = ucs_empty_function_return_zero,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_gdr_copy_iface_t),
    .iface_query              = uct_gdr_copy_iface_query,
    .iface_get_device_address = (void*)ucs_empty_function_return_success,
    .iface_get_address        = uct_gdr_copy_iface_get_address,
    .iface_is_reachable       = uct_gdr_copy_iface_is_reachable,
};

static UCS_CLASS_INIT_FUNC(uct_gdr_copy_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_gdr_copy_iface_ops, md, worker,
                              params, tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(UCT_GDR_COPY_TL_NAME));

    if (strncmp(params->mode.device.dev_name,
                UCT_CUDA_DEV_NAME, strlen(UCT_CUDA_DEV_NAME)) != 0) {
        ucs_error("No device was found: %s", params->mode.device.dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_gdr_copy_iface_t)
{
    /* tasks to tear down the domain */
}

UCS_CLASS_DEFINE(uct_gdr_copy_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_gdr_copy_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                          const uct_iface_params_t*, const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_gdr_copy_iface_t, uct_iface_t);


static ucs_status_t uct_gdr_copy_query_tl_resources(uct_md_h md,
                                                    uct_tl_resource_desc_t **resource_p,
                                                    unsigned *num_resources_p)
{
    uct_tl_resource_desc_t *resource;

    resource = ucs_calloc(1, sizeof(uct_tl_resource_desc_t), "resource desc");
    if (NULL == resource) {
        ucs_error("Failed to allocate memory");
        return UCS_ERR_NO_MEMORY;
    }

    ucs_snprintf_zero(resource->tl_name, sizeof(resource->tl_name), "%s",
                      UCT_GDR_COPY_TL_NAME);
    ucs_snprintf_zero(resource->dev_name, sizeof(resource->dev_name), "%s",
                      UCT_CUDA_DEV_NAME);
    resource->dev_type = UCT_DEVICE_TYPE_ACC;

    *num_resources_p = 1;
    *resource_p      = resource;
    return UCS_OK;
}

UCT_TL_COMPONENT_DEFINE(uct_gdr_copy_tl,
                        uct_gdr_copy_query_tl_resources,
                        uct_gdr_copy_iface_t,
                        UCT_GDR_COPY_TL_NAME,
                        "GDR_COPY_",
                        uct_gdr_copy_iface_config_table,
                        uct_gdr_copy_iface_config_t);
UCT_MD_REGISTER_TL(&uct_gdr_copy_md_component, &uct_gdr_copy_tl);
