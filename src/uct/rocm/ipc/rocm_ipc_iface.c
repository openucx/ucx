/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019-2023. ALL RIGHTS RESERVED.
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rocm_ipc_iface.h"
#include "rocm_ipc_md.h"
#include "rocm_ipc_ep.h"

#include <uct/rocm/base/rocm_base.h>
#include <uct/rocm/base/rocm_signal.h>

static ucs_config_field_t uct_rocm_ipc_iface_config_table[] = {

    {"", "", NULL, ucs_offsetof(uct_rocm_ipc_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"MIN_ZCOPY", "128", "Minimum data size for ROCm/IPC zcopy protocols",
     ucs_offsetof(uct_rocm_ipc_iface_config_t, min_zcopy),
     UCS_CONFIG_TYPE_MEMUNITS},

    {"LAT", "1e-7", "Latency",
     ucs_offsetof(uct_rocm_ipc_iface_config_t, latency), UCS_CONFIG_TYPE_TIME},

    {NULL}
};

static double uct_rocm_ipc_iface_get_bw()
{
    static double bw = -1.0;
    hsa_amd_link_info_type_t type;

    if (bw < 0.0) {
        uct_rocm_base_get_link_type(&type);
        switch (type) {
        case HSA_AMD_LINK_INFO_TYPE_PCIE:
            bw = 200.0 * UCS_GBYTE;
            break;
        case HSA_AMD_LINK_INFO_TYPE_XGMI:
            bw = 400.0 * UCS_GBYTE;
            break;
        default:
            bw = 100.0 * UCS_GBYTE;
            break;
        }
    }
    return bw;
}

ucs_status_t uct_rocm_ipc_iface_get_device_address(uct_iface_t *tl_iface,
                                                   uct_device_addr_t *addr)
{
    *(uint64_t*)addr = ucs_get_system_id();
    return UCS_OK;
}

static ucs_status_t uct_rocm_ipc_iface_get_address(uct_iface_h tl_iface,
                                                   uct_iface_addr_t *iface_addr)
{
    *(pid_t*)iface_addr = getpid();
    return UCS_OK;
}

static int
uct_rocm_ipc_iface_is_reachable_v2(const uct_iface_h tl_iface,
                                   const uct_iface_is_reachable_params_t *params)
{
    if (!uct_iface_is_reachable_params_addrs_valid(params)) {
        return 0;
    }

    if (getpid() == *(pid_t*)params->iface_addr) {
        uct_iface_fill_info_str_buf(params, "same process");
        return 0;
    }

    if (ucs_get_system_id() != *((const uint64_t*)params->device_addr)) {
        uct_iface_fill_info_str_buf(params,
                                    "the device addr is from another machine");
        return 0;
    }

    return uct_iface_scope_is_reachable(tl_iface, params);
}

static ucs_status_t uct_rocm_ipc_iface_query(uct_iface_h tl_iface,
                                             uct_iface_attr_t *iface_attr)
{
    uct_rocm_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_rocm_ipc_iface_t);

    uct_base_iface_query(&iface->super, iface_attr);

    iface_attr->cap.put.min_zcopy       = iface->config.min_zcopy;
    iface_attr->cap.put.max_zcopy       = SIZE_MAX;
    iface_attr->cap.put.opt_zcopy_align = sizeof(uint32_t);
    iface_attr->cap.put.align_mtu       = iface_attr->cap.put.opt_zcopy_align;
    iface_attr->cap.put.max_iov         = 1;

    iface_attr->cap.get.min_zcopy       = iface->config.min_zcopy;
    iface_attr->cap.get.max_zcopy       = SIZE_MAX;
    iface_attr->cap.get.opt_zcopy_align = sizeof(uint32_t);
    iface_attr->cap.get.align_mtu       = iface_attr->cap.get.opt_zcopy_align;
    iface_attr->cap.get.max_iov         = 1;

    iface_attr->iface_addr_len          = sizeof(pid_t);
    iface_attr->device_addr_len         = sizeof(uint64_t);
    iface_attr->ep_addr_len             = 0;
    iface_attr->max_conn_priv           = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_GET_ZCOPY |
                                          UCT_IFACE_FLAG_PUT_ZCOPY |
                                          UCT_IFACE_FLAG_PENDING   |
                                          UCT_IFACE_FLAG_CONNECT_TO_IFACE;

    iface_attr->latency = ucs_linear_func_make(iface->config.latency, 0);
    iface_attr->bandwidth.dedicated     = 0;
    iface_attr->bandwidth.shared        = uct_rocm_ipc_iface_get_bw();
    iface_attr->overhead                = 0;
    iface_attr->priority                = 0;

    return UCS_OK;
}

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_rocm_ipc_iface_t, uct_iface_t);

static ucs_status_t
uct_rocm_ipc_iface_flush(uct_iface_h tl_iface, unsigned flags,
                         uct_completion_t *comp)
{
    uct_rocm_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_rocm_ipc_iface_t);

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (ucs_queue_is_empty(&iface->signal_queue)) {
        UCT_TL_IFACE_STAT_FLUSH(ucs_derived_of(tl_iface, uct_base_iface_t));
        return UCS_OK;
    }

    UCT_TL_IFACE_STAT_FLUSH_WAIT(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_INPROGRESS;
}

static unsigned uct_rocm_ipc_iface_progress(uct_iface_h tl_iface)
{
    uct_rocm_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_rocm_ipc_iface_t);

    return uct_rocm_base_progress(&iface->signal_queue);
}

static uct_iface_internal_ops_t uct_rocm_ipc_iface_internal_ops = {
    .iface_estimate_perf   = uct_base_iface_estimate_perf,
    .iface_vfs_refresh     = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
    .ep_query              = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
    .ep_invalidate         = (uct_ep_invalidate_func_t)ucs_empty_function_return_unsupported,
    .ep_connect_to_ep_v2   = ucs_empty_function_return_unsupported,
    .iface_is_reachable_v2 = uct_rocm_ipc_iface_is_reachable_v2
};

static uct_iface_ops_t uct_rocm_ipc_iface_ops = {
    .ep_put_zcopy             = uct_rocm_ipc_ep_put_zcopy,
    .ep_get_zcopy             = uct_rocm_ipc_ep_get_zcopy,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_rocm_ipc_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_rocm_ipc_ep_t),
    .iface_flush              = uct_rocm_ipc_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_rocm_ipc_iface_progress,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_rocm_ipc_iface_t),
    .iface_query              = uct_rocm_ipc_iface_query,
    .iface_get_address        = uct_rocm_ipc_iface_get_address,
    .iface_get_device_address = uct_rocm_ipc_iface_get_device_address,
    .iface_is_reachable       = uct_base_iface_is_reachable
};


static UCS_CLASS_INIT_FUNC(uct_rocm_ipc_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_rocm_ipc_iface_config_t *config =
            ucs_derived_of(tl_config, uct_rocm_ipc_iface_config_t);
    ucs_status_t status;
    ucs_mpool_params_t mp_params;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_rocm_ipc_iface_ops,
                              &uct_rocm_ipc_iface_internal_ops,
                              md, worker, params,
                              tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(UCT_ROCM_IPC_TL_NAME));

    self->config.min_zcopy = config->min_zcopy;
    self->config.latency   = config->latency;

    ucs_mpool_params_reset(&mp_params);
    mp_params.elem_size       = sizeof(uct_rocm_base_signal_desc_t);
    mp_params.elems_per_chunk = 128;
    mp_params.max_elems       = 1024;
    mp_params.ops             = &uct_rocm_base_signal_desc_mpool_ops;
    mp_params.name            = "ROCM_IPC signal objects";
    status = ucs_mpool_init(&mp_params, &self->signal_pool);
    if (status != UCS_OK) {
        ucs_error("rocm/ipc signal mpool creation failed");
        return status;
    }

    ucs_queue_head_init(&self->signal_queue);

    return UCS_OK;
}


static UCS_CLASS_CLEANUP_FUNC(uct_rocm_ipc_iface_t)
{
    uct_base_iface_progress_disable(&self->super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    ucs_mpool_cleanup(&self->signal_pool, 1);
}

UCS_CLASS_DEFINE(uct_rocm_ipc_iface_t, uct_base_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_rocm_ipc_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rocm_ipc_iface_t, uct_iface_t);

UCT_TL_DEFINE(&uct_rocm_ipc_component, rocm_ipc, uct_rocm_base_query_devices,
              uct_rocm_ipc_iface_t, "ROCM_IPC_",
              uct_rocm_ipc_iface_config_table, uct_rocm_ipc_iface_config_t);
