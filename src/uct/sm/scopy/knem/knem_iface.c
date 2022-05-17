/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "knem_md.h"
#include "knem_iface.h"
#include "knem_ep.h"

#include <uct/base/uct_md.h>
#include <ucs/sys/string.h>


static ucs_config_field_t uct_knem_iface_config_table[] = {
    {"SCOPY_", "SM_BW=13862MBs", NULL,
     ucs_offsetof(uct_knem_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_scopy_iface_config_table)},

    {NULL}
};

static ucs_status_t uct_knem_iface_query(uct_iface_h tl_iface,
                                         uct_iface_attr_t *iface_attr)
{
    uct_knem_iface_t *iface = ucs_derived_of(tl_iface, uct_knem_iface_t);

    uct_scopy_iface_query(&iface->super, iface_attr);

    iface_attr->iface_addr_len      = 0;
    iface_attr->bandwidth.shared    = iface->super.super.config.bandwidth;
    iface_attr->bandwidth.dedicated = 0;

    return UCS_OK;
}

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_knem_iface_t, uct_iface_t);

static uct_iface_ops_t uct_knem_iface_tl_ops = {
    .ep_put_zcopy             = uct_scopy_ep_put_zcopy,
    .ep_get_zcopy             = uct_scopy_ep_get_zcopy,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_flush                 = uct_scopy_ep_flush,
    .ep_fence                 = uct_sm_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_knem_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_knem_ep_t),
    .iface_flush              = uct_scopy_iface_flush,
    .iface_fence              = uct_sm_iface_fence,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = uct_scopy_iface_progress,
    .iface_event_fd_get       = ucs_empty_function_return_unsupported,
    .iface_event_arm          = uct_scopy_iface_event_arm,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_knem_iface_t),
    .iface_query              = uct_knem_iface_query,
    .iface_get_device_address = uct_sm_iface_get_device_address,
    .iface_get_address        = ucs_empty_function_return_success,
    .iface_is_reachable       = uct_sm_iface_is_reachable,
};

static uct_scopy_iface_ops_t uct_knem_iface_ops = {
    .super = {
        .iface_estimate_perf = uct_base_iface_estimate_perf,
        .iface_vfs_refresh   = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
        .ep_query            = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
        .ep_invalidate       = (uct_ep_invalidate_func_t)ucs_empty_function_return_unsupported
    },
    .ep_tx = uct_knem_ep_tx,
};

static UCS_CLASS_INIT_FUNC(uct_knem_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_scopy_iface_t, &uct_knem_iface_tl_ops,
                              &uct_knem_iface_ops, md, worker, params,
                              tl_config);
    self->knem_md = (uct_knem_md_t *)md;

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_knem_iface_t)
{
    /* No OP */
}

UCS_CLASS_DEFINE(uct_knem_iface_t, uct_scopy_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_knem_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_knem_iface_t, uct_iface_t);

UCT_TL_DEFINE_ENTRY(&uct_knem_component, knem, uct_sm_base_query_tl_devices,
                    uct_knem_iface_t, "KNEM_", uct_knem_iface_config_table,
                    uct_knem_iface_config_t);

UCT_SINGLE_TL_INIT(&uct_knem_component, knem, ctor,,)
