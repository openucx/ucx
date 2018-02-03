/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "self_iface.h"
#include "self_md.h"
#include "self_ep.h"

#include <uct/sm/base/sm_ep.h>
#include <ucs/type/class.h>
#include <ucs/sys/string.h>


static ucs_config_field_t uct_self_iface_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_self_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

     UCT_IFACE_MPOOL_CONFIG_FIELDS("", 16384, 16, "",
                                   ucs_offsetof(uct_self_iface_config_t, mp), ""),

    {NULL}
};

static ucs_status_t uct_self_iface_query(uct_iface_h iface, uct_iface_attr_t *attr)
{
    uct_self_iface_t *self_iface = ucs_derived_of(iface, uct_self_iface_t);

    ucs_trace_func("iface=%p", iface);
    memset(attr, 0, sizeof(*attr));

    attr->iface_addr_len         = sizeof(uct_self_iface_addr_t);
    attr->device_addr_len        = 0;
    attr->ep_addr_len            = 0; /* No UCT_IFACE_FLAG_CONNECT_TO_EP supported */
    attr->max_conn_priv          = 0;
    attr->cap.flags              = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                   UCT_IFACE_FLAG_AM_SHORT         |
                                   UCT_IFACE_FLAG_AM_BCOPY         |
                                   UCT_IFACE_FLAG_PUT_SHORT        |
                                   UCT_IFACE_FLAG_PUT_BCOPY        |
                                   UCT_IFACE_FLAG_GET_BCOPY        |
                                   UCT_IFACE_FLAG_ATOMIC_ADD32     |
                                   UCT_IFACE_FLAG_ATOMIC_ADD64     |
                                   UCT_IFACE_FLAG_ATOMIC_FADD64    |
                                   UCT_IFACE_FLAG_ATOMIC_FADD32    |
                                   UCT_IFACE_FLAG_ATOMIC_SWAP64    |
                                   UCT_IFACE_FLAG_ATOMIC_SWAP32    |
                                   UCT_IFACE_FLAG_ATOMIC_CSWAP64   |
                                   UCT_IFACE_FLAG_ATOMIC_CSWAP32   |
                                   UCT_IFACE_FLAG_ATOMIC_CPU       |
                                   UCT_IFACE_FLAG_PENDING          |
                                   UCT_IFACE_FLAG_CB_SYNC          |
                                   UCT_IFACE_FLAG_EP_CHECK;

    attr->cap.put.max_short       = UINT_MAX;
    attr->cap.put.max_bcopy       = SIZE_MAX;
    attr->cap.put.min_zcopy       = 0;
    attr->cap.put.max_zcopy       = 0;
    attr->cap.put.opt_zcopy_align = UCS_SYS_CACHE_LINE_SIZE;
    attr->cap.put.align_mtu       = attr->cap.put.opt_zcopy_align;
    attr->cap.put.max_iov         = 1;

    attr->cap.get.max_bcopy       = SIZE_MAX;
    attr->cap.get.min_zcopy       = 0;
    attr->cap.get.max_zcopy       = 0;
    attr->cap.get.opt_zcopy_align = UCS_SYS_CACHE_LINE_SIZE;
    attr->cap.get.align_mtu       = attr->cap.get.opt_zcopy_align;
    attr->cap.get.max_iov         = 1;

    attr->cap.am.max_short        = self_iface->data_length;
    attr->cap.am.max_bcopy        = self_iface->data_length;
    attr->cap.am.min_zcopy        = 0;
    attr->cap.am.max_zcopy        = 0;
    attr->cap.am.opt_zcopy_align  = UCS_SYS_CACHE_LINE_SIZE;
    attr->cap.am.align_mtu        = attr->cap.am.opt_zcopy_align;
    attr->cap.am.max_hdr          = 0;
    attr->cap.am.max_iov          = 1;

    attr->latency.overhead        = 0;
    attr->latency.growth          = 0;
    attr->bandwidth               = 6911 * 1024.0 * 1024.0;
    attr->overhead                = 10e-9;
    attr->priority                = 0;

    return UCS_OK;
}

static ucs_status_t uct_self_iface_get_address(uct_iface_h iface,
                                               uct_iface_addr_t *addr)
{
    const uct_self_iface_t *self_iface = 0;

    ucs_trace_func("iface=%p", iface);
    self_iface = ucs_derived_of(iface, uct_self_iface_t);
    *(uct_self_iface_addr_t*)addr = self_iface->id;
    return UCS_OK;
}

static int uct_self_iface_is_reachable(const uct_iface_h iface, const uct_device_addr_t *dev_addr,
                                       const uct_iface_addr_t *iface_addr)
{
    const uct_self_iface_t *self_iface = NULL;
    const uct_self_iface_addr_t *self_addr = NULL;

    if (NULL == iface_addr) {
        return 0;
    }
    self_iface = ucs_derived_of(iface, uct_self_iface_t);
    self_addr = (const uct_self_iface_addr_t *) iface_addr;
    ucs_trace_func("iface=%p id=%lx addr=%lx", iface, self_iface->id, *self_addr);
    return  self_iface->id == *self_addr;
}

static void uct_self_iface_release_desc(uct_recv_desc_t *self, void *desc)
{
    uct_recv_desc_t *self_desc = (uct_recv_desc_t *)desc - 1;
    ucs_mpool_put(self_desc);
}

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_self_iface_t, uct_iface_t);

static uct_iface_ops_t uct_self_iface_ops = {
    .ep_put_short             = uct_sm_ep_put_short,
    .ep_put_bcopy             = uct_sm_ep_put_bcopy,
    .ep_get_bcopy             = uct_sm_ep_get_bcopy,
    .ep_am_short              = uct_self_ep_am_short,
    .ep_am_bcopy              = uct_self_ep_am_bcopy,
    .ep_atomic_add64          = uct_sm_ep_atomic_add64,
    .ep_atomic_fadd64         = uct_sm_ep_atomic_fadd64,
    .ep_atomic_cswap64        = uct_sm_ep_atomic_cswap64,
    .ep_atomic_swap64         = uct_sm_ep_atomic_swap64,
    .ep_atomic_add32          = uct_sm_ep_atomic_add32,
    .ep_atomic_fadd32         = uct_sm_ep_atomic_fadd32,
    .ep_atomic_cswap32        = uct_sm_ep_atomic_cswap32,
    .ep_atomic_swap32         = uct_sm_ep_atomic_swap32,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_check                 = ucs_empty_function_return_success,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_create_connected      = UCS_CLASS_NEW_FUNC_NAME(uct_self_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_self_ep_t),
    .iface_flush              = uct_base_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = ucs_empty_function_return_zero,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_self_iface_t),
    .iface_query              = uct_self_iface_query,
    .iface_get_device_address = ucs_empty_function_return_success,
    .iface_get_address        = uct_self_iface_get_address,
    .iface_is_reachable       = uct_self_iface_is_reachable
};

static UCS_CLASS_INIT_FUNC(uct_self_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    ucs_status_t status;
    uct_self_iface_config_t *self_config = 0;

    ucs_trace_func("Creating a loop-back transport self=%p rxh=%lu",
                   self, params->rx_headroom);

    ucs_assert(params->open_mode & UCT_IFACE_OPEN_MODE_DEVICE);

    if (strcmp(params->mode.device.dev_name, UCT_SELF_NAME) != 0) {
        ucs_error("No device was found: %s", params->mode.device.dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_self_iface_ops, md, worker,
                              params, tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(UCT_SELF_NAME));

    self_config = ucs_derived_of(tl_config, uct_self_iface_config_t);

    self->id              = ucs_generate_uuid((uintptr_t)self);
    self->rx_headroom     = params->rx_headroom;
    self->data_length     = self_config->super.max_bcopy;
    self->release_desc.cb = uct_self_iface_release_desc;

    /* create a memory pool for data transferred */
    status = uct_iface_mpool_init(&self->super,
                                  &self->msg_desc_mp,
                                  sizeof(uct_recv_desc_t) + self->rx_headroom +
                                                            self->data_length,
                                  sizeof(uct_recv_desc_t) + self->rx_headroom,
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &self_config->mp,
                                  256,
                                  ucs_empty_function,
                                  "self_msg_desc");
    if (UCS_OK != status) {
        ucs_error("Failed to create a memory pool for the loop-back transport");
        return status;
    }

    ucs_debug("Created a loop-back iface. id=0x%lx, len=%u, tx_hdr=%lu",
              self->id, self->data_length, self->rx_headroom);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_self_iface_t)
{
    ucs_trace_func("self=%p", self);

    ucs_mpool_cleanup(&self->msg_desc_mp, 1);
}

UCS_CLASS_DEFINE(uct_self_iface_t, uct_base_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_self_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static ucs_status_t uct_self_query_tl_resources(uct_md_h md,
                                                uct_tl_resource_desc_t **resource_p,
                                                unsigned *num_resources_p)
{
    uct_tl_resource_desc_t *resource = 0;

    ucs_trace_func("md=%p", md);
    resource = ucs_calloc(1, sizeof(*resource), "resource desc");
    if (NULL == resource) {
        ucs_error("Failed to allocate memory");
        return UCS_ERR_NO_MEMORY;
    }

    ucs_snprintf_zero(resource->tl_name, sizeof(resource->tl_name), "%s",
                      UCT_SELF_NAME);
    ucs_snprintf_zero(resource->dev_name, sizeof(resource->dev_name), "%s",
                      UCT_SELF_NAME);
    resource->dev_type = UCT_DEVICE_TYPE_SELF;

    *num_resources_p = 1;
    *resource_p      = resource;
    return UCS_OK;
}

UCT_TL_COMPONENT_DEFINE(uct_self_tl, uct_self_query_tl_resources, uct_self_iface_t,
                        UCT_SELF_NAME, "SELF_", uct_self_iface_config_table, uct_self_iface_config_t);
UCT_MD_REGISTER_TL(&uct_self_md, &uct_self_tl);
