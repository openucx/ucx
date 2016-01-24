/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "cma_pd.h"
#include "cma_iface.h"
#include "cma_ep.h"

#include <uct/api/addr.h>
#include <uct/base/uct_pd.h>

UCT_PD_REGISTER_TL(&uct_cma_pd_component, &uct_cma_tl);

static ucs_config_field_t uct_cma_iface_config_table[] = {
    {"", "ALLOC=huge,mmap,heap", NULL,
    ucs_offsetof(uct_cma_iface_config_t, super),
    UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},
    {NULL}
};

static ucs_status_t uct_cma_iface_get_address(uct_iface_t *tl_iface,
                                             struct sockaddr *addr)
{
    uct_sockaddr_process_t *iface_addr = (void*)addr;
    iface_addr->sp_family = UCT_AF_PROCESS;
    iface_addr->node_guid = ucs_machine_guid();
    iface_addr->id        = getpid();
    return UCS_OK;
}

static int uct_cma_iface_is_reachable(uct_iface_t *tl_iface,
                                     const struct sockaddr *addr)
{
    return (addr->sa_family == UCT_AF_PROCESS) &&
           (((uct_sockaddr_process_t*)addr)->node_guid == ucs_machine_guid());
}

static ucs_status_t uct_cma_iface_query(uct_iface_h tl_iface,
                                       uct_iface_attr_t *iface_attr)
{
    memset(iface_attr, 0, sizeof(uct_iface_attr_t));

    /* default values for all shared memory transports */
    iface_attr->cap.put.max_zcopy      = SIZE_MAX;
    iface_attr->cap.get.max_zcopy      = SIZE_MAX;
    iface_attr->iface_addr_len         = sizeof(uct_sockaddr_process_t);
    iface_attr->ep_addr_len            = 0;
    iface_attr->cap.flags              = UCT_IFACE_FLAG_GET_ZCOPY |
                                         UCT_IFACE_FLAG_PUT_ZCOPY |
                                         UCT_IFACE_FLAG_CONNECT_TO_IFACE;
    return UCS_OK;
}

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_cma_iface_t, uct_iface_t);

static uct_iface_ops_t uct_cma_iface_ops = {
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_cma_iface_t),
    .iface_query         = uct_cma_iface_query,
    .iface_get_address   = uct_cma_iface_get_address,
    .iface_is_reachable  = uct_cma_iface_is_reachable,
    .iface_flush         = (void*)ucs_empty_function_return_success,
    .ep_put_zcopy        = uct_cma_ep_put_zcopy,
    .ep_get_zcopy        = uct_cma_ep_get_zcopy,
    .ep_flush            = (void*)ucs_empty_function_return_success,
    .ep_create_connected = UCS_CLASS_NEW_FUNC_NAME(uct_cma_ep_t),
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_cma_ep_t),
};

static UCS_CLASS_INIT_FUNC(uct_cma_iface_t, uct_pd_h pd, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_cma_iface_ops, pd, worker,
                              tl_config UCS_STATS_ARG(NULL));
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cma_iface_t)
{
}

UCS_CLASS_DEFINE(uct_cma_iface_t, uct_base_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_cma_iface_t, uct_iface_t, uct_pd_h,
                                 uct_worker_h, const char *, size_t,
                                 const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_cma_iface_t, uct_iface_t);

static ucs_status_t uct_cma_query_tl_resources(uct_pd_h pd,
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
                      UCT_CMA_TL_NAME);
    ucs_snprintf_zero(resource->dev_name, sizeof(resource->dev_name), "%s",
                      pd->component->name);
    resource->latency    =  10;
    resource->bandwidth  = (long) (6911 * pow(1024,2)); /* FIXME temp value */
    resource->dev_type   = UCT_DEVICE_TYPE_SHM;

    *num_resources_p = 1;
    *resource_p      = resource;
    return UCS_OK;
}

UCT_TL_COMPONENT_DEFINE(uct_cma_tl,
                        uct_cma_query_tl_resources,
                        uct_cma_iface_t,
                        UCT_CMA_TL_NAME,
                        "",
                        uct_cma_iface_config_table,
                        uct_cma_iface_config_t);
