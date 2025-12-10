/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019-2023. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gaudi_gdr_iface.h"
#include "gaudi_gdr_md.h"

#include <float.h>
#include <uct/gaudi/base/gaudi_base.h>

static ucs_status_t
uct_gaudi_gdr_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_gaudi_gdr_iface_t *iface = ucs_derived_of(tl_iface,
                                                  uct_gaudi_gdr_iface_t);

    uct_base_iface_query(&iface->super, iface_attr);
    iface_attr->cap.flags = UCT_IFACE_FLAG_CONNECT_TO_IFACE;
    iface_attr->bandwidth.dedicated =
            0.0001; // DBL_MIN will be round down to 0 when packing address;
    iface_attr->bandwidth.shared = 0;
    iface_attr->max_num_eps      = 0;

    return UCS_OK;
}

static uct_iface_ops_t uct_gaudi_gdr_iface_ops = {
    .ep_pending_purge = (uct_ep_pending_purge_func_t)ucs_empty_function,
    .ep_connect    = (uct_ep_connect_func_t)ucs_empty_function_return_success,
    .ep_disconnect = (uct_ep_disconnect_func_t)ucs_empty_function_return_success,
    .cm_ep_conn_notify = (uct_cm_ep_conn_notify_func_t)
            ucs_empty_function_return_unsupported,
    .ep_destroy = (uct_ep_destroy_func_t)ucs_empty_function_return_unsupported,
    .ep_put_short = (uct_ep_put_short_func_t)
            ucs_empty_function_return_unsupported,
    .ep_put_bcopy = (uct_ep_put_bcopy_func_t)
            ucs_empty_function_return_unsupported,
    .ep_get_bcopy = (uct_ep_get_bcopy_func_t)
            ucs_empty_function_return_unsupported,
    .ep_am_short = (uct_ep_am_short_func_t)ucs_empty_function_return_unsupported,
    .ep_am_short_iov = (uct_ep_am_short_iov_func_t)
            ucs_empty_function_return_unsupported,
    .ep_am_bcopy = (uct_ep_am_bcopy_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic_cswap64 = (uct_ep_atomic_cswap64_func_t)
            ucs_empty_function_return_unsupported,
    .ep_atomic64_post  = (uct_ep_atomic64_post_func_t)
            ucs_empty_function_return_unsupported,
    .ep_atomic64_fetch = (uct_ep_atomic64_fetch_func_t)
            ucs_empty_function_return_unsupported,
    .ep_atomic_cswap32 = (uct_ep_atomic_cswap32_func_t)
            ucs_empty_function_return_unsupported,
    .ep_atomic32_post  = (uct_ep_atomic32_post_func_t)
            ucs_empty_function_return_unsupported,
    .ep_atomic32_fetch = (uct_ep_atomic32_fetch_func_t)
            ucs_empty_function_return_unsupported,
    .ep_pending_add    = (uct_ep_pending_add_func_t)
            ucs_empty_function_return_unsupported,
    .ep_flush    = (uct_ep_flush_func_t)ucs_empty_function_return_success,
    .ep_fence    = (uct_ep_fence_func_t)ucs_empty_function_return_unsupported,
    .ep_check    = (uct_ep_check_func_t)ucs_empty_function_return_unsupported,
    .ep_create   = (uct_ep_create_func_t)ucs_empty_function_return_unsupported,
    .iface_flush = (uct_iface_flush_func_t)ucs_empty_function_return_unsupported,
    .iface_fence = (uct_iface_fence_func_t)ucs_empty_function_return_unsupported,
    .iface_progress_enable  = (uct_iface_progress_enable_func_t)
            ucs_empty_function,
    .iface_progress_disable = (uct_iface_progress_disable_func_t)
            ucs_empty_function,
    .iface_progress = (uct_iface_progress_func_t)ucs_empty_function_return_zero,
    .iface_event_fd_get       = (uct_iface_event_fd_get_func_t)
            ucs_empty_function_return_unsupported,
    .iface_event_arm          = (uct_iface_event_arm_func_t)
            ucs_empty_function_return_unsupported,
    .iface_close              = (uct_iface_close_func_t)ucs_empty_function,
    .iface_query              = uct_gaudi_gdr_iface_query,
    .iface_get_device_address = (uct_iface_get_device_address_func_t)
            ucs_empty_function_return_success,
    .iface_get_address        = (uct_iface_get_address_func_t)
            ucs_empty_function_return_success,
    .iface_is_reachable       = uct_base_iface_is_reachable
};

static uct_iface_internal_ops_t uct_gaudi_gdr_iface_internal_ops = {
    .iface_query_v2      = uct_iface_base_query_v2,
    .iface_estimate_perf = (uct_iface_estimate_perf_func_t)
            ucs_empty_function_return_unsupported,
    .iface_vfs_refresh   = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
    .ep_query = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
    .ep_invalidate         = (uct_ep_invalidate_func_t)
            ucs_empty_function_return_unsupported,
    .ep_connect_to_ep_v2   = (uct_ep_connect_to_ep_v2_func_t)
            ucs_empty_function_return_unsupported,
    .iface_is_reachable_v2 = (uct_iface_is_reachable_v2_func_t)
            ucs_empty_function_return_zero,
    .ep_is_connected       = (uct_ep_is_connected_func_t)
            ucs_empty_function_return_zero_int
};

static UCS_CLASS_INIT_FUNC(uct_gaudi_gdr_iface_t, uct_md_h md,
                           uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_gaudi_gdr_iface_ops,
                              &uct_gaudi_gdr_iface_internal_ops, md, worker,
                              params,
                              tl_config UCS_STATS_ARG(params->stats_root)
                                      UCS_STATS_ARG(UCT_GAUDI_GDR_TL_NAME));

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_gaudi_gdr_iface_t)
{
}

UCS_CLASS_DEFINE(uct_gaudi_gdr_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_gaudi_gdr_iface_t, uct_iface_t, uct_md_h,
                          uct_worker_h, const uct_iface_params_t*,
                          const uct_iface_config_t*);

UCT_TL_DEFINE(&uct_gaudi_gdr_component, gaudi_gdr, uct_gaudi_base_query_devices,
              uct_gaudi_gdr_iface_t, "GAUDI_GDR_", uct_iface_config_table,
              uct_iface_config_t);
