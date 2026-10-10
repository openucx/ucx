/*
 * Copyright (C) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "musa_base.h"

#include <ucs/type/class.h>

extern uct_component_t uct_musa_component;

typedef struct {
    uct_base_iface_t super;
} uct_musa_iface_t;

UCS_CLASS_DECLARE_DELETE_FUNC(uct_musa_iface_t, uct_iface_t);

static ucs_status_t
uct_musa_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_musa_iface_t *iface = ucs_derived_of(tl_iface, uct_musa_iface_t);

    uct_base_iface_query(&iface->super, iface_attr);
    iface_attr->cap.flags          = 0;
    iface_attr->bandwidth.dedicated = 0.0001;
    iface_attr->max_num_eps         = 0;
    return UCS_OK;
}

static uct_iface_ops_t uct_musa_iface_ops = {
    .ep_flush               = (uct_ep_flush_func_t)
                              ucs_empty_function_return_success,
    .ep_fence               = (uct_ep_fence_func_t)
                              ucs_empty_function_return_unsupported,
    .ep_destroy             = (uct_ep_destroy_func_t)
                              ucs_empty_function_return_unsupported,
    .iface_flush            = (uct_iface_flush_func_t)
                              ucs_empty_function_return_unsupported,
    .iface_fence            = (uct_iface_fence_func_t)
                              ucs_empty_function_return_unsupported,
    .iface_progress_enable  = (uct_iface_progress_enable_func_t)
                              ucs_empty_function,
    .iface_progress_disable = (uct_iface_progress_disable_func_t)
                              ucs_empty_function,
    .iface_progress         = (uct_iface_progress_func_t)
                              ucs_empty_function_return_zero,
    .iface_close            = UCS_CLASS_DELETE_FUNC_NAME(uct_musa_iface_t),
    .iface_query            = uct_musa_iface_query,
    .iface_get_device_address = (uct_iface_get_device_address_func_t)
                                ucs_empty_function_return_success,
    .iface_is_reachable     = uct_base_iface_is_reachable
};

static uct_iface_internal_ops_t uct_musa_iface_internal_ops = {
    .iface_query_v2 = uct_iface_base_query_v2,
    .iface_estimate_perf = (uct_iface_estimate_perf_func_t)
                           ucs_empty_function_return_unsupported,
    .iface_vfs_refresh = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
    .ep_query = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
    .ep_invalidate = (uct_ep_invalidate_func_t)
                     ucs_empty_function_return_unsupported,
    .ep_connect_to_ep_v2 = (uct_ep_connect_to_ep_v2_func_t)
                           ucs_empty_function_return_unsupported,
    .iface_is_reachable_v2 = (uct_iface_is_reachable_v2_func_t)
                             ucs_empty_function_return_zero,
    .ep_is_connected = (uct_ep_is_connected_func_t)
                       ucs_empty_function_return_zero_int,
    .ep_outstanding_purge = (uct_ep_outstanding_purge_func_t)
                            ucs_empty_function_return_unsupported
};

static UCS_CLASS_INIT_FUNC(uct_musa_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_musa_iface_ops,
                              &uct_musa_iface_internal_ops, md, worker, params,
                              tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG("musa"));
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_musa_iface_t)
{
}

UCS_CLASS_DEFINE(uct_musa_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_musa_iface_t, uct_iface_t, uct_md_h,
                          uct_worker_h, const uct_iface_params_t *,
                          const uct_iface_config_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_musa_iface_t, uct_iface_t);

UCT_TL_DEFINE(&uct_musa_component, musa, uct_musa_base_query_devices,
              uct_musa_iface_t, "MUSA_", uct_iface_config_table,
              uct_iface_config_t);
