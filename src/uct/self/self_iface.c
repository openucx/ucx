/**
 * Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "self_iface.h"
#include "self_md.h"
#include "self_ep.h"

#include <ucs/type/class.h>

static ucs_config_field_t uct_self_iface_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_self_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {NULL}
};

static ucs_status_t uct_self_iface_query(uct_iface_h iface, uct_iface_attr_t *attr)
{
    ucs_trace_func("iface=%p", iface);
    memset(attr, 0, sizeof(uct_iface_attr_t));

    attr->iface_addr_len         = sizeof(uct_self_iface_addr_t);
    attr->device_addr_len        = sizeof(uct_self_iface_addr_t);
    attr->ep_addr_len            = 0; //No UCT_IFACE_FLAG_CONNECT_TO_EP supported
    attr->cap.flags              = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                   UCT_IFACE_FLAG_AM_SHORT         |
                                   UCT_IFACE_FLAG_AM_CB_SYNC;

    attr->cap.put.max_short      = UINT_MAX;
    attr->cap.put.max_bcopy      = SIZE_MAX;
    attr->cap.put.max_zcopy      = SIZE_MAX;

    attr->cap.get.max_bcopy      = SIZE_MAX;
    attr->cap.get.max_zcopy      = SIZE_MAX;

    attr->cap.am.max_short       = UINT_MAX;
    attr->cap.am.max_bcopy       = SIZE_MAX;
    attr->cap.am.max_zcopy       = SIZE_MAX;
    attr->cap.am.max_hdr         = 0;

    attr->latency                = 10e-9; // 10 ns; TODO fix the score
    attr->bandwidth              = 6911 * 1024.0 * 1024.0;
    attr->overhead               = 3e-9; // 3 ns; TODO fix the score

    return UCS_OK;
}

static ucs_status_t uct_self_iface_get_address(uct_iface_h iface,
                                               uct_iface_addr_t *addr)
{
    ucs_trace_func("iface=%p", iface);
    const uct_self_iface_t *local_iface = ucs_derived_of(iface, uct_self_iface_t);
    *(uct_self_iface_addr_t*)addr = local_iface->id;
    return UCS_OK;
}

static ucs_status_t uct_self_iface_get_device_address(uct_iface_h iface,
                                                      uct_device_addr_t *addr)
{
    ucs_trace_func("iface=%p", iface);
    return uct_self_iface_get_address(iface, (uct_iface_addr_t*) addr);
}

static int uct_self_iface_is_reachable(uct_iface_h iface,
                                       const uct_device_addr_t *addr)
{
    const uct_self_iface_t *local_iface = ucs_derived_of(iface, uct_self_iface_t);
    ucs_trace_func("iface=%p id=%lx addr=%lx",
                   iface, local_iface->id, *(uct_self_iface_addr_t*)addr);
    return  local_iface->id == *(const uct_self_iface_addr_t*)addr;
}

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_self_iface_t, uct_iface_t);

static uct_iface_ops_t uct_self_iface_ops = {
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_self_iface_t),
    .iface_get_device_address = uct_self_iface_get_device_address,
    .iface_get_address        = uct_self_iface_get_address,
    .iface_query              = uct_self_iface_query,
    .iface_is_reachable       = uct_self_iface_is_reachable,
    .ep_create_connected      = UCS_CLASS_NEW_FUNC_NAME(uct_self_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_self_ep_t),
    .ep_am_short              = uct_self_ep_am_short,
};

static UCS_CLASS_INIT_FUNC(uct_self_iface_t, uct_md_h md, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    ucs_trace_func("Creating a loop-back transport self=%p", self);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_self_iface_ops, md, worker,
                              tl_config UCS_STATS_ARG(NULL));

    if (strcmp(dev_name, UCT_SELF_NAME) != 0) {
        ucs_error("No device was found: %s", dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    self->id = getpid(); /* TODO need uniq id for node + process + thread */
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_self_iface_t)
{
    ucs_trace_func("self=%p", self);
}

UCS_CLASS_DEFINE(uct_self_iface_t, uct_base_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_self_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const char *, size_t,
                                 const uct_iface_config_t *);

static ucs_status_t uct_self_query_tl_resources(uct_md_h md,
                                                uct_tl_resource_desc_t **resource_p,
                                                unsigned *num_resources_p)
{
    uct_tl_resource_desc_t *resource;
    ucs_trace_func("md=%p", md);

    resource = ucs_calloc(1, sizeof(uct_tl_resource_desc_t), "resource desc");
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
