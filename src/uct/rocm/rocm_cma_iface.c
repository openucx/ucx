/*
 * Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "rocm_cma_iface.h"
#include "rocm_cma_md.h"
#include "rocm_cma_ep.h"


#include <uct/base/uct_md.h>
#include <ucs/sys/string.h>

/* Note: Treat ROCM memory as the special case of shared memory. */
#include <uct/sm/base/sm_iface.h>

/**
 *  Specify special environment variables to tune ROCm interface.
 *  So far none but keep for future.
 */
static ucs_config_field_t uct_rocm_cma_iface_config_table[] = {

    /** @note: Some super options doesn't make any sense for ROCm
        but keep them to have compatibility with upper logic dependencies
        on uct_iface_config_t */

    {"", "", NULL,
    ucs_offsetof(uct_rocm_cma_iface_config_t, super),
    UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {NULL}
};


UCT_MD_REGISTER_TL(&uct_rocm_cma_md_component, &uct_rocm_cma_tl);

static ucs_status_t uct_rocm_cma_iface_get_address(uct_iface_t *tl_iface,
                                                   uct_iface_addr_t *addr)
{
    /* Use process id as interface address */
    *(pid_t*)addr = getpid();
    return UCS_OK;
}


static ucs_status_t uct_rocm_cma_iface_query(uct_iface_h tl_iface,
                                             uct_iface_attr_t *iface_attr)
{
    memset(iface_attr, 0, sizeof(uct_iface_attr_t));
    ucs_trace("uct_rocm_cma_iface_query");

    /* default values for all shared memory transports */
    iface_attr->cap.put.min_zcopy       = 0;
    iface_attr->cap.put.max_zcopy       = SIZE_MAX;
    iface_attr->cap.put.opt_zcopy_align = sizeof(uint32_t);
    iface_attr->cap.put.align_mtu       = iface_attr->cap.put.opt_zcopy_align;
    iface_attr->cap.put.max_iov         = uct_sm_get_max_iov();

    iface_attr->cap.get.min_zcopy       = 0;
    iface_attr->cap.get.max_zcopy       = SIZE_MAX;
    iface_attr->cap.get.opt_zcopy_align = sizeof(uint32_t);
    iface_attr->cap.get.align_mtu       = iface_attr->cap.get.opt_zcopy_align;
    iface_attr->cap.get.max_iov         = uct_sm_get_max_iov();

    iface_attr->cap.am.max_iov          = 1;
    iface_attr->cap.am.opt_zcopy_align  = 1;
    iface_attr->cap.am.align_mtu        = iface_attr->cap.am.opt_zcopy_align;

    iface_attr->iface_addr_len          = sizeof(pid_t);
    iface_attr->device_addr_len         = UCT_SM_IFACE_DEVICE_ADDR_LEN;
    iface_attr->ep_addr_len             = 0;
    iface_attr->max_conn_priv           = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_GET_ZCOPY |
                                          UCT_IFACE_FLAG_PUT_ZCOPY |
                                          UCT_IFACE_FLAG_PENDING   |
                                          UCT_IFACE_FLAG_CONNECT_TO_IFACE;

    /**
     * @todo: Add logic to query ROCr about latency information.
     *
     * Question: How to handle/choose the correct one in the multi-GPUs case
     *           when latency could depends on source and target locations as
     *           well as on which GPU will be used?
     *
     * Keep value the same as for CMA transport
     *
     */
    iface_attr->latency.overhead        = 80e-9; /* 80 ns */
    iface_attr->latency.growth          = 0;
    iface_attr->bandwidth               = 10240 * 1024.0 * 1024.0; /* 10240 MB*/
    iface_attr->overhead                = 0.4e-6; /* 0.4 us */


    return UCS_OK;
}

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_rocm_cma_iface_t, uct_iface_t);

static uct_iface_ops_t uct_rocm_cma_iface_ops = {
    .ep_put_zcopy             = uct_rocm_cma_ep_put_zcopy,
    .ep_get_zcopy             = uct_rocm_cma_ep_get_zcopy,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_sm_ep_fence,
    .ep_create_connected      = UCS_CLASS_NEW_FUNC_NAME(uct_rocm_cma_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_rocm_cma_ep_t),
    .iface_flush              = uct_base_iface_flush,
    .iface_fence              = uct_sm_iface_fence,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = ucs_empty_function_return_zero,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_rocm_cma_iface_t),
    .iface_query              = uct_rocm_cma_iface_query,
    .iface_get_address        = uct_rocm_cma_iface_get_address,
    .iface_get_device_address = uct_sm_iface_get_device_address,
    .iface_is_reachable       = uct_sm_iface_is_reachable
};


static UCS_CLASS_INIT_FUNC(uct_rocm_cma_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    ucs_assert(params->open_mode & UCT_IFACE_OPEN_MODE_DEVICE);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_rocm_cma_iface_ops, md, worker,
                              params, tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(UCT_ROCM_CMA_TL_NAME));

    self->rocm_md = (uct_rocm_cma_md_t *)md;

    return UCS_OK;
}


static UCS_CLASS_CLEANUP_FUNC(uct_rocm_cma_iface_t)
{
    /* No OP */
}

UCS_CLASS_DEFINE(uct_rocm_cma_iface_t, uct_base_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_rocm_cma_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rocm_cma_iface_t, uct_iface_t);


static ucs_status_t uct_rocm_cma_query_tl_resources(uct_md_h md,
                                                    uct_tl_resource_desc_t **resource_p,
                                                    unsigned *num_resources_p)
{

    /** @note Report the single device due to the complexity to deal with
     *        numerous agents (multi-GPUs cases) especially for
     *        p2p transfer cases.
     */
    uct_tl_resource_desc_t *resource;

    resource = ucs_calloc(1, sizeof(uct_tl_resource_desc_t), "ROCm resource desc");
    if (NULL == resource) {
        ucs_error("Failed to allocate memory");
        return UCS_ERR_NO_MEMORY;
    }

    ucs_snprintf_zero(resource->tl_name, sizeof(resource->tl_name), "%s",
                      UCT_ROCM_CMA_TL_NAME);
    ucs_snprintf_zero(resource->dev_name, sizeof(resource->dev_name), "%s",
                      md->component->name);

    ucs_trace_func("TL %s, DEV %s", resource->tl_name, resource->dev_name);

    resource->dev_type = UCT_DEVICE_TYPE_ACC;  /* Specify device type as
                                                "accelerator" device.*/

    *num_resources_p = 1;
    *resource_p      = resource;

    return UCS_OK;
}


UCT_TL_COMPONENT_DEFINE(uct_rocm_cma_tl,
                        uct_rocm_cma_query_tl_resources,
                        uct_rocm_cma_iface_t,
                        UCT_ROCM_CMA_TL_NAME,
                        "ROCM_TL_",
                        uct_rocm_cma_iface_config_table,
                        uct_rocm_cma_iface_config_t);

