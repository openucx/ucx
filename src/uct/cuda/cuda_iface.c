/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "cuda_iface.h"
#include "cuda_pd.h"
#include "cuda_ep.h"

#include <ucs/type/class.h>


static ucs_config_field_t uct_cuda_iface_config_table[] = {

    {"", "", NULL,
     ucs_offsetof(uct_cuda_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {NULL}
};


/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_iface_t)(uct_iface_t*);

static ucs_status_t uct_cuda_iface_get_address(uct_iface_h tl_iface,
                                               struct sockaddr *iface_addr)
{
    uct_sockaddr_process_t *cuda_addr = (uct_sockaddr_process_t*)iface_addr;

    cuda_addr->sp_family = UCT_AF_PROCESS;
    cuda_addr->node_guid = ucs_machine_guid();
    cuda_addr->id        = 0;
    return UCS_OK;
}

static int uct_cuda_iface_is_reachable(uct_iface_h iface,
                                       const struct sockaddr* addr)
{
    return 0;
}

static ucs_status_t uct_cuda_iface_query(uct_iface_h iface,
                                         uct_iface_attr_t *iface_attr)
{
    memset(iface_attr, 0, sizeof(uct_iface_attr_t));

    /* FIXME all of these values */
    iface_attr->iface_addr_len         = sizeof(uct_sockaddr_process_t);
    iface_attr->ep_addr_len            = 0;
    iface_attr->cap.flags              = 0;

    iface_attr->cap.put.max_short      = 0;
    iface_attr->cap.put.max_bcopy      = 0;
    iface_attr->cap.put.max_zcopy      = 0;

    iface_attr->cap.get.max_bcopy      = 0;
    iface_attr->cap.get.max_zcopy      = 0;

    iface_attr->cap.am.max_short       = 0;
    iface_attr->cap.am.max_bcopy       = 0;
    iface_attr->cap.am.max_zcopy       = 0;
    iface_attr->cap.am.max_hdr         = 0;
    return UCS_OK;
}

static uct_iface_ops_t uct_cuda_iface_ops = {
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_iface_t),
    .iface_get_address   = uct_cuda_iface_get_address,
    .iface_flush         = (void*)ucs_empty_function_return_success,
    .iface_query         = uct_cuda_iface_query,
    .iface_is_reachable  = uct_cuda_iface_is_reachable,
    .ep_create_connected = UCS_CLASS_NEW_FUNC_NAME(uct_cuda_ep_t),
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_ep_t),
    .ep_put_short        = uct_cuda_ep_put_short,
    .ep_am_short         = uct_cuda_ep_am_short,
};

static UCS_CLASS_INIT_FUNC(uct_cuda_iface_t, uct_pd_h pd, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_cuda_iface_ops, pd, worker,
                              tl_config UCS_STATS_ARG(NULL));

    if (strcmp(dev_name, UCT_CUDA_DEV_NAME) != 0) {
        ucs_error("No device was found: %s", dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_iface_t)
{
    /* tasks to tear down the domain */
}

UCS_CLASS_DEFINE(uct_cuda_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_cuda_iface_t, uct_iface_t, uct_pd_h, uct_worker_h,
                          const char*, size_t, const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_cuda_iface_t, uct_iface_t);


static ucs_status_t uct_cuda_query_tl_resources(uct_pd_h pd,
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
                      UCT_CUDA_TL_NAME);
    ucs_snprintf_zero(resource->dev_name, sizeof(resource->dev_name), "%s",
                      UCT_CUDA_DEV_NAME);
    resource->latency    = 1; /* FIXME temp value */
    resource->bandwidth  = (long) (6911 * pow(1024,2)); /* FIXME temp value */

    *num_resources_p = 1;
    *resource_p      = resource;
    return UCS_OK;
}

UCT_TL_COMPONENT_DEFINE(uct_cuda_tl,
                        uct_cuda_query_tl_resources,
                        uct_cuda_iface_t,
                        UCT_CUDA_TL_NAME,
                        "CUDA_",
                        uct_cuda_iface_config_table,
                        uct_cuda_iface_config_t);
UCT_PD_REGISTER_TL(&uct_cuda_pd, &uct_cuda_tl);
