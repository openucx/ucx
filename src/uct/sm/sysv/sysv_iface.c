/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "sysv_pd.h"
#include "sysv_iface.h"


static ucs_config_field_t uct_sysv_iface_config_table[] = {
    {"SM_", "ALLOC=pd", NULL,
    ucs_offsetof(uct_sysv_iface_config_t, super),
    UCS_CONFIG_TYPE_TABLE(uct_mm_iface_config_table)},
    {NULL}
};


#define UCT_SYSV_MAX_SHORT_LENGTH 2048 /* FIXME temp value for now */
#define UCT_SYSV_MAX_BCOPY_LENGTH 40960 /* FIXME temp value for now */
#define UCT_SYSV_MAX_ZCOPY_LENGTH 81920 /* FIXME temp value for now */


ucs_status_t uct_sysv_iface_query(uct_iface_h tl_iface, 
                                  uct_iface_attr_t *iface_attr)
{
    ucs_status_t status;

    /* initialize the defaults from mm */
    status = uct_mm_iface_query(tl_iface, iface_attr);
    if (UCS_OK != status) return status;

    /* set TL specific flags */
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
                                         UCT_IFACE_FLAG_PUT_ZCOPY       |
                                         UCT_IFACE_FLAG_GET_BCOPY       |
                                         UCT_IFACE_FLAG_GET_ZCOPY       |
                                         UCT_IFACE_FLAG_CONNECT_TO_IFACE;

    iface_attr->completion_priv_len    = 0; /* TBD */
    return UCS_OK;
}

static ucs_status_t uct_sysv_iface_get_address(uct_iface_h tl_iface,
                                               struct sockaddr *addr)
{
    uct_sysv_iface_t *iface = ucs_derived_of(tl_iface, uct_sysv_iface_t);
    uct_sockaddr_process_t *iface_addr = (uct_sockaddr_process_t*)addr;

    uct_mm_iface_get_address(&iface->super, iface_addr);
    iface_addr->cookie = 0; /* TODO AM fifo id */
    return UCS_OK;
}

/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_sysv_iface_t)(uct_iface_t*);

/* point as much to mm as possible
 * to override, create a uct_sysv_* function and update the table here
 */
uct_iface_ops_t uct_sysv_iface_ops = {
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_sysv_iface_t),
    .iface_query         = uct_sysv_iface_query,
    .iface_get_address   = uct_sysv_iface_get_address,
    .iface_is_reachable  = uct_mm_iface_is_reachable,
    .iface_flush         = uct_mm_iface_flush,
    .ep_put_short        = uct_mm_ep_put_short,
    .ep_put_bcopy        = uct_mm_ep_put_bcopy,
    .ep_put_zcopy        = uct_mm_ep_put_zcopy,
    .ep_get_bcopy        = uct_mm_ep_get_bcopy,
    .ep_get_zcopy        = uct_mm_ep_get_zcopy,
    .ep_am_short         = uct_mm_ep_am_short,
    .ep_atomic_add64     = uct_mm_ep_atomic_add64,
    .ep_atomic_fadd64    = uct_mm_ep_atomic_fadd64,
    .ep_atomic_cswap64   = uct_mm_ep_atomic_cswap64,
    .ep_atomic_swap64    = uct_mm_ep_atomic_swap64,
    .ep_atomic_add32     = uct_mm_ep_atomic_add32,
    .ep_atomic_fadd32    = uct_mm_ep_atomic_fadd32,
    .ep_atomic_cswap32   = uct_mm_ep_atomic_cswap32,
    .ep_atomic_swap32    = uct_mm_ep_atomic_swap32,
    .ep_create_connected = UCS_CLASS_NEW_FUNC_NAME(uct_sysv_ep_t),
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_sysv_ep_t),
};

static UCS_CLASS_INIT_FUNC(uct_sysv_iface_t, uct_pd_h pd, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    /* initialize with the mm constructor */
    UCS_CLASS_CALL_SUPER_INIT(uct_mm_iface_t, &uct_sysv_iface_ops, pd, worker,
                              tl_config);

    /* can override default max size values 
     * from mm (self->super.config.*) here */
    self->super.config.max_put     = UCT_SYSV_MAX_SHORT_LENGTH;
    self->super.config.max_bcopy   = UCT_SYSV_MAX_BCOPY_LENGTH;
    self->super.config.max_zcopy   = UCT_SYSV_MAX_ZCOPY_LENGTH;

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_sysv_iface_t)
{
}

/* point to mm */
UCS_CLASS_DEFINE(uct_sysv_iface_t, uct_mm_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_sysv_iface_t, uct_iface_t, uct_pd_h, uct_worker_h,
                                 const char*, size_t, const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_sysv_iface_t, uct_iface_t);


static ucs_status_t uct_sysv_query_tl_resources(uct_pd_h pd,
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
                      UCT_SYSV_TL_NAME);
    ucs_snprintf_zero(resource->dev_name, sizeof(resource->dev_name), "%s",
                      UCT_SYSV_TL_NAME);
    resource->latency    = 1; /* FIXME temp value */
    resource->bandwidth  = (long) (6911 * pow(1024,2)); /* FIXME temp value */

    *num_resources_p = 1;
    *resource_p     = resource;
    return UCS_OK;
}

UCT_TL_COMPONENT_DEFINE(&uct_sysv_pd, uct_sysv_tl,
                        uct_sysv_query_tl_resources,
                        uct_sysv_iface_t,
                        UCT_SYSV_TL_NAME,
                        "SYSV_",
                        uct_sysv_iface_config_table,
                        uct_sysv_iface_config_t);
