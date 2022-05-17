/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cma_md.h"
#include "cma_iface.h"
#include "cma_ep.h"

#include <uct/base/uct_md.h>
#include <ucs/sys/string.h>


typedef struct {
    pid_t                            id;
} ucs_cma_iface_base_device_addr_t;

typedef struct {
    ucs_cma_iface_base_device_addr_t super;
    ucs_sys_ns_t                     pid_ns;
} ucs_cma_iface_ext_device_addr_t;


static ucs_config_field_t uct_cma_iface_config_table[] = {
    {"SCOPY_", "ALLOC=huge,thp,mmap,heap;SM_BW=11145MBs", NULL,
     ucs_offsetof(uct_cma_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_scopy_iface_config_table)},

    {NULL}
};

static ucs_status_t uct_cma_iface_get_address(uct_iface_t *tl_iface,
                                              uct_iface_addr_t *addr)
{
    ucs_cma_iface_ext_device_addr_t *iface_addr = (void*)addr;

    ucs_assert(!(getpid() & UCT_CMA_IFACE_ADDR_FLAG_PID_NS));

    iface_addr->super.id = getpid();
    if (!ucs_sys_ns_is_default(UCS_SYS_NS_TYPE_PID)) {
        iface_addr->super.id |= UCT_CMA_IFACE_ADDR_FLAG_PID_NS;
        iface_addr->pid_ns    = ucs_sys_get_ns(UCS_SYS_NS_TYPE_PID);
    }
    return UCS_OK;
}

static ucs_status_t uct_cma_iface_query(uct_iface_h tl_iface,
                                        uct_iface_attr_t *iface_attr)
{
    uct_cma_iface_t *iface = ucs_derived_of(tl_iface, uct_cma_iface_t);

    uct_scopy_iface_query(&iface->super, iface_attr);

    iface_attr->iface_addr_len      =
            ucs_sys_ns_is_default(UCS_SYS_NS_TYPE_PID) ?
            sizeof(ucs_cma_iface_base_device_addr_t) :
            sizeof(ucs_cma_iface_ext_device_addr_t);
    iface_attr->bandwidth.dedicated = iface->super.super.config.bandwidth;
    iface_attr->bandwidth.shared    = 0;
    iface_attr->cap.flags          |= UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE |
                                      UCT_IFACE_FLAG_EP_CHECK;

    return UCS_OK;
}

static int
uct_cma_iface_is_reachable(const uct_iface_h tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *tl_iface_addr)
{
    ucs_cma_iface_ext_device_addr_t *iface_addr = (void*)tl_iface_addr;

    if (!uct_sm_iface_is_reachable(tl_iface, dev_addr, tl_iface_addr)) {
        return 0;
    }

    if (iface_addr->super.id & UCT_CMA_IFACE_ADDR_FLAG_PID_NS) {
        return ucs_sys_get_ns(UCS_SYS_NS_TYPE_PID) == iface_addr->pid_ns;
    }

    return ucs_sys_ns_is_default(UCS_SYS_NS_TYPE_PID);
}

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_cma_iface_t, uct_iface_t);

static uct_iface_ops_t uct_cma_iface_tl_ops = {
    .ep_put_zcopy             = uct_scopy_ep_put_zcopy,
    .ep_get_zcopy             = uct_scopy_ep_get_zcopy,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_flush                 = uct_scopy_ep_flush,
    .ep_fence                 = uct_sm_ep_fence,
    .ep_check                 = uct_cma_ep_check,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_cma_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_cma_ep_t),
    .iface_flush              = uct_scopy_iface_flush,
    .iface_fence              = uct_sm_iface_fence,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = uct_scopy_iface_progress,
    .iface_event_fd_get       = ucs_empty_function_return_unsupported,
    .iface_event_arm          = uct_scopy_iface_event_arm,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_cma_iface_t),
    .iface_query              = uct_cma_iface_query,
    .iface_get_address        = uct_cma_iface_get_address,
    .iface_get_device_address = uct_sm_iface_get_device_address,
    .iface_is_reachable       = uct_cma_iface_is_reachable,
};

static uct_scopy_iface_ops_t uct_cma_iface_ops = {
    .super = {
        .iface_estimate_perf = uct_base_iface_estimate_perf,
        .iface_vfs_refresh   = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
        .ep_query            = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
        .ep_invalidate       = (uct_ep_invalidate_func_t)ucs_empty_function_return_unsupported,
    },
    .ep_tx = uct_cma_ep_tx,
};

static UCS_CLASS_INIT_FUNC(uct_cma_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_scopy_iface_t, &uct_cma_iface_tl_ops,
                              &uct_cma_iface_ops, md, worker, params,
                              tl_config);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cma_iface_t)
{
    /* No op */
}

UCS_CLASS_DEFINE(uct_cma_iface_t, uct_scopy_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_cma_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_cma_iface_t, uct_iface_t);

UCT_TL_DEFINE_ENTRY(&uct_cma_component, cma, uct_sm_base_query_tl_devices,
                    uct_cma_iface_t, "CMA_", uct_cma_iface_config_table,
                    uct_cma_iface_config_t);

UCT_SINGLE_TL_INIT(&uct_cma_component, cma, ctor,,)
