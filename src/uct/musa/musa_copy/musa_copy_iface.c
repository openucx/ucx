/*
 * Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "musa_copy_iface.h"
#include "musa_copy_ep.h"

#include <uct/musa/base/musa_base.h>

#include <ucs/sys/sys.h>
#include <ucs/type/class.h>

#include <limits.h>
#include <stdlib.h>
#include <string.h>

extern uct_component_t uct_musa_component;

static void UCS_CLASS_DELETE_FUNC_NAME(uct_musa_copy_iface_t)(uct_iface_t *);

static ucs_config_field_t uct_musa_copy_iface_config_table[] = {
    {"", "", NULL, ucs_offsetof(uct_musa_copy_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"LAT", "1e-7", "Latency",
     ucs_offsetof(uct_musa_copy_iface_config_t, latency), UCS_CONFIG_TYPE_TIME},

    {"BW", "10000MBs", "Effective memory bandwidth",
     ucs_offsetof(uct_musa_copy_iface_config_t, bandwidth),
     UCS_CONFIG_TYPE_BW},

    {NULL}
};

static ucs_status_t
uct_musa_copy_iface_get_address(uct_iface_h tl_iface,
                                uct_iface_addr_t *iface_addr)
{
    uct_musa_copy_iface_t *iface = ucs_derived_of(tl_iface,
                                                  uct_musa_copy_iface_t);

    *(uct_musa_copy_iface_addr_t*)iface_addr = iface->id;
    return UCS_OK;
}

static int
uct_musa_copy_iface_is_reachable_v2(
        const uct_iface_h tl_iface,
        const uct_iface_is_reachable_params_t *params)
{
    uct_musa_copy_iface_t *iface = ucs_derived_of(tl_iface,
                                                  uct_musa_copy_iface_t);
    const uct_musa_copy_iface_addr_t *addr;

    if (!uct_iface_is_reachable_params_addrs_valid(params)) {
        return 0;
    }

    addr = (const uct_musa_copy_iface_addr_t*)params->iface_addr;
    if ((addr == NULL) || (*addr != iface->id)) {
        uct_iface_fill_info_str_buf(params, "iface address mismatch");
        return 0;
    }

    return uct_iface_scope_is_reachable(tl_iface, params);
}

static ucs_status_t
uct_musa_copy_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_musa_copy_iface_t *iface = ucs_derived_of(tl_iface,
                                                  uct_musa_copy_iface_t);

    uct_base_iface_query(&iface->super, iface_attr);

    iface_attr->iface_addr_len          = sizeof(uct_musa_copy_iface_addr_t);
    iface_attr->device_addr_len         = 0;
    iface_attr->ep_addr_len             = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                          UCT_IFACE_FLAG_GET_SHORT |
                                          UCT_IFACE_FLAG_PUT_SHORT |
                                          UCT_IFACE_FLAG_GET_ZCOPY |
                                          UCT_IFACE_FLAG_PUT_ZCOPY |
                                          UCT_IFACE_FLAG_PENDING;

    iface_attr->cap.put.max_short       = UINT_MAX;
    iface_attr->cap.put.max_bcopy       = 0;
    iface_attr->cap.put.min_zcopy       = 0;
    iface_attr->cap.put.max_zcopy       = SIZE_MAX;
    iface_attr->cap.put.opt_zcopy_align = 1;
    iface_attr->cap.put.align_mtu       = 1;
    iface_attr->cap.put.max_iov         = 1;

    iface_attr->cap.get.max_short       = UINT_MAX;
    iface_attr->cap.get.max_bcopy       = 0;
    iface_attr->cap.get.min_zcopy       = 0;
    iface_attr->cap.get.max_zcopy       = SIZE_MAX;
    iface_attr->cap.get.opt_zcopy_align = 1;
    iface_attr->cap.get.align_mtu       = 1;
    iface_attr->cap.get.max_iov         = 1;

    iface_attr->cap.am.max_short        = 0;
    iface_attr->cap.am.max_bcopy        = 0;
    iface_attr->cap.am.min_zcopy        = 0;
    iface_attr->cap.am.max_zcopy        = 0;
    iface_attr->cap.am.opt_zcopy_align  = 1;
    iface_attr->cap.am.align_mtu        = 1;
    iface_attr->cap.am.max_hdr          = 0;
    iface_attr->cap.am.max_iov          = 1;

    iface_attr->latency                 = ucs_linear_func_make(
            iface->config.latency, 0);
    iface_attr->bandwidth.dedicated     = 0;
    iface_attr->bandwidth.shared        = iface->config.bandwidth;
    iface_attr->overhead                = 0;
    iface_attr->priority                = 0;
    return UCS_OK;
}

static unsigned uct_musa_copy_iface_progress(uct_iface_h tl_iface)
{
    (void)tl_iface;
    return 0;
}

static uct_iface_ops_t uct_musa_copy_iface_ops = {
    .ep_get_short             = uct_musa_copy_ep_get_short,
    .ep_put_short             = uct_musa_copy_ep_put_short,
    .ep_get_zcopy             = uct_musa_copy_ep_get_zcopy,
    .ep_put_zcopy             = uct_musa_copy_ep_put_zcopy,
    .ep_pending_add           = (uct_ep_pending_add_func_t)
                                ucs_empty_function_return_busy,
    .ep_pending_purge         = (uct_ep_pending_purge_func_t)
                                ucs_empty_function,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_musa_copy_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_musa_copy_ep_t),
    .iface_flush              = uct_base_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_musa_copy_iface_progress,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(
                                    uct_musa_copy_iface_t),
    .iface_query              = uct_musa_copy_iface_query,
    .iface_get_device_address = (uct_iface_get_device_address_func_t)
                                ucs_empty_function_return_success,
    .iface_get_address        = uct_musa_copy_iface_get_address,
    .iface_is_reachable       = uct_base_iface_is_reachable
};

static uct_iface_internal_ops_t uct_musa_copy_iface_internal_ops = {
    .iface_query_v2        = uct_iface_base_query_v2,
    .iface_estimate_perf   = uct_base_iface_estimate_perf,
    .iface_vfs_refresh     = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
    .ep_query              = (uct_ep_query_func_t)
                             ucs_empty_function_return_unsupported,
    .ep_invalidate         = (uct_ep_invalidate_func_t)
                             ucs_empty_function_return_unsupported,
    .ep_connect_to_ep_v2   = (uct_ep_connect_to_ep_v2_func_t)
                             ucs_empty_function_return_unsupported,
    .iface_is_reachable_v2 = uct_musa_copy_iface_is_reachable_v2,
    .ep_is_connected       = uct_base_ep_is_connected,
    .ep_get_device_ep      = (uct_ep_get_device_ep_func_t)
                             ucs_empty_function_return_unsupported,
    .ep_put_sgl_zcopy      = (uct_ep_put_sgl_zcopy_func_t)
                             ucs_empty_function_return_unsupported,
    .ep_get_sgl_zcopy      = (uct_ep_get_sgl_zcopy_func_t)
                             ucs_empty_function_return_unsupported,
    .ep_outstanding_purge  = (uct_ep_outstanding_purge_func_t)
                             ucs_empty_function_return_unsupported
};

static UCS_CLASS_INIT_FUNC(uct_musa_copy_iface_t, uct_md_h md,
                           uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_musa_copy_iface_config_t *config =
            ucs_derived_of(tl_config, uct_musa_copy_iface_config_t);
    const char *device_name = params->mode.device.dev_name;
    char *end;
    long device;
    unsigned device_count;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_musa_copy_iface_ops,
                              &uct_musa_copy_iface_internal_ops, md, worker,
                              params, tl_config
                              UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(UCT_MUSA_COPY_TL_NAME));

    device = 0;
    if ((device_name != NULL) && (strncmp(device_name, "musa", 4) == 0)) {
        device = strtol(device_name + 4, &end, 10);
        if ((end == (device_name + 4)) || (*end != '\0') || (device < 0)) {
            return UCS_ERR_INVALID_PARAM;
        }
    }

    status = uct_musa_base_get_device_count(&device_count);
    if ((status != UCS_OK) || ((unsigned)device >= device_count)) {
        return UCS_ERR_NO_DEVICE;
    }

    self->id               = ucs_generate_uuid((uintptr_t)self);
    self->config.latency   = config->latency;
    self->config.bandwidth = config->bandwidth;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_musa_copy_iface_t)
{
    uct_base_iface_progress_disable(&self->super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
}

UCS_CLASS_DEFINE(uct_musa_copy_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_musa_copy_iface_t, uct_iface_t, uct_md_h,
                          uct_worker_h, const uct_iface_params_t *,
                          const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_musa_copy_iface_t, uct_iface_t);

static ucs_status_t
uct_musa_copy_query_devices(uct_md_h md,
                            uct_tl_device_resource_t **tl_devices_p,
                            unsigned *num_tl_devices_p)
{
    ucs_status_t status;
    unsigned full_count;

    status = uct_musa_base_query_devices(md, tl_devices_p, num_tl_devices_p);
    if (status != UCS_OK) {
        return status;
    }

    full_count = *num_tl_devices_p;

    /* FIXME: UCX memtype EP asserts num_lanes <= 2. Remove this cap when UCX
     * supports multi-lane memtype endpoints or selects a single device
     * deterministically (same limitation as ze_copy). The copy EP selects the
     * device from the pointer attributes, so a single iface is sufficient. */
    if (full_count > 1) {
        ucs_diag("musa_copy: %u devices found, limiting to 1 "
                 "for memtype ep compatibility",
                 full_count);
        *num_tl_devices_p = 1;
    }

    return UCS_OK;
}

UCT_TL_DEFINE(&uct_musa_component, musa_copy, uct_musa_copy_query_devices,
              uct_musa_copy_iface_t, "MUSA_COPY_",
              uct_musa_copy_iface_config_table, uct_musa_copy_iface_config_t);
