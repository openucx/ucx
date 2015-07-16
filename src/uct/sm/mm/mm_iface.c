/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "mm_iface.h"
#include "mm_ep.h"

#include <uct/api/addr.h>
#include <uct/tl/context.h>


static ucs_config_field_t uct_mm_iface_config_table[] = {
    {"", "ALLOC=pd", NULL,
    ucs_offsetof(uct_mm_iface_config_t, super),
    UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},
    {NULL}
};

static ucs_status_t uct_mm_iface_get_address(uct_iface_t *tl_iface,
                                             struct sockaddr *addr)
{
    uct_sockaddr_process_t *iface_addr = (void*)addr;
    iface_addr->sp_family = UCT_AF_PROCESS;
    iface_addr->node_guid = ucs_machine_guid();
    /* TODO pack shared memory id */
    return UCS_OK;
}

static int uct_mm_iface_is_reachable(uct_iface_t *tl_iface,
                                     const struct sockaddr *addr)
{
    return (addr->sa_family == UCT_AF_PROCESS) &&
           (((uct_sockaddr_process_t*)addr)->node_guid == ucs_machine_guid());
}

static ucs_status_t uct_mm_flush()
{
    ucs_memory_cpu_store_fence();
    return UCS_OK;
}

static ucs_status_t uct_mm_iface_query(uct_iface_h tl_iface,
                                       uct_iface_attr_t *iface_attr)
{
    memset(iface_attr, 0, sizeof(uct_iface_attr_t));

    /* default values for all shared memory transports */
    iface_attr->cap.put.max_short      = UINT_MAX;
    iface_attr->cap.put.max_bcopy      = SIZE_MAX;
    iface_attr->cap.put.max_zcopy      = SIZE_MAX;
    iface_attr->cap.get.max_bcopy      = SIZE_MAX;
    iface_attr->cap.get.max_zcopy      = SIZE_MAX;
    iface_attr->iface_addr_len         = sizeof(uct_sockaddr_process_t);
    iface_attr->ep_addr_len            = 0;
    iface_attr->cap.flags              = UCT_IFACE_FLAG_PUT_SHORT       |
                                         UCT_IFACE_FLAG_PUT_BCOPY       |
                                         UCT_IFACE_FLAG_ATOMIC_ADD32    |
                                         UCT_IFACE_FLAG_ATOMIC_ADD64    |
                                         UCT_IFACE_FLAG_ATOMIC_FADD64   |
                                         UCT_IFACE_FLAG_ATOMIC_FADD32   |
                                         UCT_IFACE_FLAG_ATOMIC_SWAP64   |
                                         UCT_IFACE_FLAG_ATOMIC_SWAP32   |
                                         UCT_IFACE_FLAG_ATOMIC_CSWAP64  |
                                         UCT_IFACE_FLAG_ATOMIC_CSWAP32  |
                                         UCT_IFACE_FLAG_GET_BCOPY       |
                                         UCT_IFACE_FLAG_CONNECT_TO_IFACE;
    return UCS_OK;
}

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_mm_iface_t, uct_iface_t);

static uct_iface_ops_t uct_mm_iface_ops = {
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_mm_iface_t),
    .iface_query         = uct_mm_iface_query,
    .iface_get_address   = uct_mm_iface_get_address,
    .iface_is_reachable  = uct_mm_iface_is_reachable,
    .iface_flush         = (void*)uct_mm_flush,
    .ep_put_short        = uct_mm_ep_put_short,
    .ep_put_bcopy        = uct_mm_ep_put_bcopy,
    .ep_get_bcopy        = uct_mm_ep_get_bcopy,
    .ep_am_short         = uct_mm_ep_am_short,
    .ep_atomic_add64     = uct_mm_ep_atomic_add64,
    .ep_atomic_fadd64    = uct_mm_ep_atomic_fadd64,
    .ep_atomic_cswap64   = uct_mm_ep_atomic_cswap64,
    .ep_atomic_swap64    = uct_mm_ep_atomic_swap64,
    .ep_atomic_add32     = uct_mm_ep_atomic_add32,
    .ep_atomic_fadd32    = uct_mm_ep_atomic_fadd32,
    .ep_atomic_cswap32   = uct_mm_ep_atomic_cswap32,
    .ep_atomic_swap32    = uct_mm_ep_atomic_swap32,
    .ep_flush            = (void*)uct_mm_flush,
    .ep_create_connected = UCS_CLASS_NEW_FUNC_NAME(uct_mm_ep_t),
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_mm_ep_t),
};

static UCS_CLASS_INIT_FUNC(uct_mm_iface_t, uct_pd_h pd, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_mm_iface_ops, pd, worker,
                              tl_config UCS_STATS_ARG(NULL));

    /* TODO allocate receive FIFO using uct_mm_pd_mapper_ops(pd)->alloc */

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_mm_iface_t)
{
}

UCS_CLASS_DEFINE(uct_mm_iface_t, uct_base_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_mm_iface_t, uct_iface_t, uct_pd_h,
                                 uct_worker_h, const char *, size_t,
                                 const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_mm_iface_t, uct_iface_t);

static ucs_status_t uct_mm_query_tl_resources(uct_pd_h pd,
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
                      UCT_MM_TL_NAME);
    ucs_snprintf_zero(resource->dev_name, sizeof(resource->dev_name), "%s",
                      pd->component->name);
    resource->latency    =  10;
    resource->bandwidth  = (long) (6911 * pow(1024,2)); /* FIXME temp value */

    *num_resources_p = 1;
    *resource_p      = resource;
    return UCS_OK;
}

UCT_TL_COMPONENT_DEFINE(uct_mm_tl,
                        uct_mm_query_tl_resources,
                        uct_mm_iface_t,
                        UCT_MM_TL_NAME,
                        "MM_",
                        uct_mm_iface_config_table,
                        uct_mm_iface_config_t);
