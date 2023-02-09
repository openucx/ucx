/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019-2023. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rocm_copy_iface.h"
#include "rocm_copy_md.h"
#include "rocm_copy_ep.h"

#include <uct/rocm/base/rocm_base.h>
#include <uct/rocm/base/rocm_signal.h>
#include <ucs/type/class.h>
#include <ucs/sys/string.h>


static ucs_config_field_t uct_rocm_copy_iface_config_table[] = {

    {"", "", NULL, ucs_offsetof(uct_rocm_copy_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"D2H_THRESH", "1k",
     "Threshold for switching to hsa memcpy for device-to-host copies",
     ucs_offsetof(uct_rocm_copy_iface_config_t, d2h_thresh),
     UCS_CONFIG_TYPE_MEMUNITS},

    {"H2D_THRESH", "1m",
     "Threshold for switching to hsa memcpy for host-to-device copies",
     ucs_offsetof(uct_rocm_copy_iface_config_t, h2d_thresh),
     UCS_CONFIG_TYPE_MEMUNITS},

    {"ENABLE_ASYNC_ZCOPY", "y",
     "Enable asynchronous zcopy operations",
     ucs_offsetof(uct_rocm_copy_iface_config_t, enable_async_zcopy),
     UCS_CONFIG_TYPE_BOOL},

    {NULL}
};

/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_rocm_copy_iface_t)(uct_iface_t*);


static ucs_status_t uct_rocm_copy_iface_get_address(uct_iface_h tl_iface,
                                                    uct_iface_addr_t *iface_addr)
{
    uct_rocm_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_rocm_copy_iface_t);

    *(uct_rocm_copy_iface_addr_t*)iface_addr = iface->id;
    return UCS_OK;
}

static int uct_rocm_copy_iface_is_reachable(const uct_iface_h tl_iface,
                                            const uct_device_addr_t *dev_addr,
                                            const uct_iface_addr_t *iface_addr)
{
    uct_rocm_copy_iface_t  *iface = ucs_derived_of(tl_iface, uct_rocm_copy_iface_t);
    uct_rocm_copy_iface_addr_t *addr = (uct_rocm_copy_iface_addr_t*)iface_addr;

    return (addr != NULL) && (iface->id == *addr);
}

static ucs_status_t uct_rocm_copy_iface_query(uct_iface_h tl_iface,
                                              uct_iface_attr_t *iface_attr)
{
    uct_rocm_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_rocm_copy_iface_t);

    uct_base_iface_query(&iface->super, iface_attr);

    iface_attr->iface_addr_len          = sizeof(uct_rocm_copy_iface_addr_t);
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
    iface_attr->cap.put.align_mtu       = iface_attr->cap.put.opt_zcopy_align;
    iface_attr->cap.put.max_iov         = 1;

    iface_attr->cap.get.max_short       = UINT_MAX;
    iface_attr->cap.get.max_bcopy       = 0;
    iface_attr->cap.get.min_zcopy       = 0;
    iface_attr->cap.get.max_zcopy       = SIZE_MAX;
    iface_attr->cap.get.opt_zcopy_align = 1;
    iface_attr->cap.get.align_mtu       = iface_attr->cap.get.opt_zcopy_align;
    iface_attr->cap.get.max_iov         = 1;

    iface_attr->cap.am.max_short        = 0;
    iface_attr->cap.am.max_bcopy        = 0;
    iface_attr->cap.am.min_zcopy        = 0;
    iface_attr->cap.am.max_zcopy        = 0;
    iface_attr->cap.am.opt_zcopy_align  = 1;
    iface_attr->cap.am.align_mtu        = iface_attr->cap.am.opt_zcopy_align;
    iface_attr->cap.am.max_hdr          = 0;
    iface_attr->cap.am.max_iov          = 1;

    iface_attr->latency                 = ucs_linear_func_make(10e-6, 0);
    iface_attr->bandwidth.dedicated     = 6911.0 * UCS_MBYTE;
    iface_attr->bandwidth.shared        = 0;
    iface_attr->overhead                = 0;
    iface_attr->priority                = 0;

    return UCS_OK;
}

static ucs_status_t
uct_rocm_copy_iface_flush(uct_iface_h tl_iface, unsigned flags,
                          uct_completion_t *comp)
{
    uct_rocm_copy_iface_t *iface = ucs_derived_of(tl_iface,
                                                  uct_rocm_copy_iface_t);

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

static unsigned uct_rocm_copy_iface_progress(uct_iface_h tl_iface)
{
    uct_rocm_copy_iface_t *iface = ucs_derived_of(tl_iface,
                                                  uct_rocm_copy_iface_t);

    return uct_rocm_base_progress (&iface->signal_queue);
}


static uct_iface_ops_t uct_rocm_copy_iface_ops = {
    .ep_get_short             = uct_rocm_copy_ep_get_short,
    .ep_put_short             = uct_rocm_copy_ep_put_short,
    .ep_get_zcopy             = uct_rocm_copy_ep_get_zcopy,
    .ep_put_zcopy             = uct_rocm_copy_ep_put_zcopy,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_rocm_copy_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_rocm_copy_ep_t),
    .iface_flush              = uct_rocm_copy_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_rocm_copy_iface_progress,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_rocm_copy_iface_t),
    .iface_query              = uct_rocm_copy_iface_query,
    .iface_get_device_address = (uct_iface_get_device_address_func_t)ucs_empty_function_return_success,
    .iface_get_address        = uct_rocm_copy_iface_get_address,
    .iface_is_reachable       = uct_rocm_copy_iface_is_reachable,
};


static ucs_status_t
uct_rocm_copy_estimate_perf(uct_iface_h tl_iface, uct_perf_attr_t *perf_attr)
{
    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_BANDWIDTH) {
        perf_attr->bandwidth.dedicated = 0;
        if (!(perf_attr->field_mask & UCT_PERF_ATTR_FIELD_OPERATION)) {
            perf_attr->bandwidth.shared = 0;
        } else {
            switch (perf_attr->operation) {
            case UCT_EP_OP_GET_SHORT:
                perf_attr->bandwidth.shared = 2000.0 * UCS_MBYTE;
                break;
            case UCT_EP_OP_GET_ZCOPY:
                perf_attr->bandwidth.shared = 8000.0 * UCS_MBYTE;
                break;
            case UCT_EP_OP_PUT_SHORT:
                perf_attr->bandwidth.shared = 10500.0 * UCS_MBYTE;
                break;
            case UCT_EP_OP_PUT_ZCOPY:
                perf_attr->bandwidth.shared = 9500.0 * UCS_MBYTE;
                break;
            default:
                perf_attr->bandwidth.shared = 0;
                break;
            }
        }
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD) {
        perf_attr->send_pre_overhead = 0;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD) {
        perf_attr->send_post_overhead = 0;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_RECV_OVERHEAD) {
        perf_attr->recv_overhead = 0;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_LATENCY) {
        perf_attr->latency = ucs_linear_func_make(10e-6, 0);
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_MAX_INFLIGHT_EPS) {
        perf_attr->max_inflight_eps = SIZE_MAX;
    }

    return UCS_OK;
}


static uct_iface_internal_ops_t uct_rocm_copy_iface_internal_ops = {
    .iface_estimate_perf = uct_rocm_copy_estimate_perf,
    .iface_vfs_refresh   = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
    .ep_query            = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
    .ep_invalidate       = (uct_ep_invalidate_func_t)ucs_empty_function_return_unsupported,
    .ep_connect_to_ep_v2 = ucs_empty_function_return_unsupported
};

static UCS_CLASS_INIT_FUNC(uct_rocm_copy_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_rocm_copy_iface_config_t *config = ucs_derived_of(tl_config,
                                                          uct_rocm_copy_iface_config_t);
    ucs_status_t status;
    ucs_mpool_params_t mp_params;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_rocm_copy_iface_ops,
                              &uct_rocm_copy_iface_internal_ops,
                              md, worker, params,
                              tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(UCT_ROCM_COPY_TL_NAME));

    self->id                        = ucs_generate_uuid((uintptr_t)self);
    self->config.d2h_thresh         = config->d2h_thresh;
    self->config.h2d_thresh         = config->h2d_thresh;
    self->config.enable_async_zcopy = config->enable_async_zcopy;

    ucs_mpool_params_reset(&mp_params);
    mp_params.elem_size       = sizeof(uct_rocm_base_signal_desc_t);
    mp_params.elems_per_chunk = 128;
    mp_params.max_elems       = 1024;
    mp_params.ops             = &uct_rocm_base_signal_desc_mpool_ops;
    mp_params.name            = "ROCM_COPY signal objects";
    status                    = ucs_mpool_init(&mp_params, &self->signal_pool);
    if (status != UCS_OK) {
        ucs_error("rocm/copy signal mpool creation failed");
        return status;
    }

    ucs_queue_head_init(&self->signal_queue);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rocm_copy_iface_t)
{
    uct_base_iface_progress_disable(&self->super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    ucs_mpool_cleanup(&self->signal_pool, 1);
}

UCS_CLASS_DEFINE(uct_rocm_copy_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rocm_copy_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                          const uct_iface_params_t*, const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rocm_copy_iface_t, uct_iface_t);

UCT_TL_DEFINE(&uct_rocm_copy_component, rocm_copy,
              uct_rocm_base_query_devices, uct_rocm_copy_iface_t,
              "ROCM_COPY_", uct_rocm_copy_iface_config_table,
              uct_rocm_copy_iface_config_t);
