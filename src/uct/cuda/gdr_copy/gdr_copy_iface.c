/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "gdr_copy_iface.h"
#include "gdr_copy_md.h"
#include "gdr_copy_ep.h"

#include <uct/cuda/base/cuda_iface.h>
#include <ucs/type/class.h>
#include <ucs/sys/string.h>


#define UCT_GDR_COPY_IFACE_DEFAULT_BANDWIDTH (6911.0 * UCS_MBYTE)
#define UCT_GDR_COPY_IFACE_OVERHEAD          0
#define UCT_GDR_COPY_IFACE_LATENCY           ucs_linear_func_make(1e-6, 0)


static ucs_config_field_t uct_gdr_copy_iface_config_table[] = {

    {"", "", NULL,
     ucs_offsetof(uct_gdr_copy_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {NULL}
};

/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_gdr_copy_iface_t)(uct_iface_t*);

static ucs_status_t uct_gdr_copy_iface_get_address(uct_iface_h tl_iface,
                                                   uct_iface_addr_t *iface_addr)
{
    uct_gdr_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_gdr_copy_iface_t);

    *(uct_gdr_copy_iface_addr_t*)iface_addr = iface->id;
    return UCS_OK;
}

static int uct_gdr_copy_iface_is_reachable(const uct_iface_h tl_iface,
                                           const uct_device_addr_t *dev_addr,
                                           const uct_iface_addr_t *iface_addr)
{
    uct_gdr_copy_iface_t  *iface = ucs_derived_of(tl_iface, uct_gdr_copy_iface_t);
    uct_gdr_copy_iface_addr_t *addr = (uct_gdr_copy_iface_addr_t*)iface_addr;

    return (addr != NULL) && (iface->id == *addr);
}

static ucs_status_t
uct_gdr_copy_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_gdr_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_gdr_copy_iface_t);

    uct_base_iface_query(&iface->super, iface_attr);

    iface_attr->iface_addr_len          = sizeof(uct_gdr_copy_iface_addr_t);
    iface_attr->device_addr_len         = 0;
    iface_attr->ep_addr_len             = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                          UCT_IFACE_FLAG_PUT_SHORT |
                                          UCT_IFACE_FLAG_GET_SHORT;

    iface_attr->cap.put.max_short       = UINT_MAX;
    iface_attr->cap.put.max_bcopy       = 0;
    iface_attr->cap.put.min_zcopy       = 0;
    iface_attr->cap.put.max_zcopy       = 0;
    iface_attr->cap.put.opt_zcopy_align = 1;
    iface_attr->cap.put.align_mtu       = iface_attr->cap.put.opt_zcopy_align;
    iface_attr->cap.put.max_iov         = 1;

    iface_attr->cap.get.max_short       = UINT_MAX;
    iface_attr->cap.get.max_bcopy       = 0;
    iface_attr->cap.get.min_zcopy       = 0;
    iface_attr->cap.get.max_zcopy       = 0;
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

    iface_attr->latency                 = UCT_GDR_COPY_IFACE_LATENCY;
    iface_attr->bandwidth.dedicated     = 0;
    iface_attr->bandwidth.shared        = UCT_GDR_COPY_IFACE_DEFAULT_BANDWIDTH;
    iface_attr->overhead                = UCT_GDR_COPY_IFACE_OVERHEAD;
    iface_attr->priority                = 0;

    return UCS_OK;
}

static ucs_status_t
uct_gdr_copy_estimate_perf(uct_iface_h iface, uct_perf_attr_t *perf_attr)
{
    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_BANDWIDTH) {
        perf_attr->bandwidth.dedicated = 0;
        if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_OPERATION) {
            switch (perf_attr->operation) {
            case UCT_EP_OP_GET_SHORT:
            case UCT_EP_OP_GET_ZCOPY:
                perf_attr->bandwidth.shared = 440.0 * UCS_MBYTE;
                break;
            case UCT_EP_OP_PUT_SHORT:
                perf_attr->bandwidth.shared = 10200.0 * UCS_MBYTE;
                break;
            default:
                perf_attr->bandwidth.shared =
                        UCT_GDR_COPY_IFACE_DEFAULT_BANDWIDTH;
            }
        } else {
            perf_attr->bandwidth.shared = UCT_GDR_COPY_IFACE_DEFAULT_BANDWIDTH;
        }
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD) {
        perf_attr->send_pre_overhead = UCT_GDR_COPY_IFACE_OVERHEAD;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD) {
        perf_attr->send_post_overhead = 0;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_RECV_OVERHEAD) {
        perf_attr->recv_overhead = 0;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_LATENCY) {
        perf_attr->latency = UCT_GDR_COPY_IFACE_LATENCY;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_MAX_INFLIGHT_EPS) {
        perf_attr->max_inflight_eps = SIZE_MAX;
    }

    return UCS_OK;
}

static uct_iface_ops_t uct_gdr_copy_iface_ops = {
    .ep_put_short             = uct_gdr_copy_ep_put_short,
    .ep_get_short             = uct_gdr_copy_ep_get_short,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_gdr_copy_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_gdr_copy_ep_t),
    .iface_flush              = uct_base_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = ucs_empty_function_return_zero,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_gdr_copy_iface_t),
    .iface_query              = uct_gdr_copy_iface_query,
    .iface_get_device_address = (uct_iface_get_device_address_func_t)ucs_empty_function_return_success,
    .iface_get_address        = uct_gdr_copy_iface_get_address,
    .iface_is_reachable       = uct_gdr_copy_iface_is_reachable,
};

static uct_iface_internal_ops_t uct_gdr_copy_iface_internal_ops = {
    .iface_estimate_perf = uct_gdr_copy_estimate_perf,
    .iface_vfs_refresh   = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
    .ep_query            = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
    .ep_invalidate       = (uct_ep_invalidate_func_t)ucs_empty_function_return_unsupported
};

static UCS_CLASS_INIT_FUNC(uct_gdr_copy_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_gdr_copy_iface_ops,
                              &uct_gdr_copy_iface_internal_ops, md, worker,
                              params,
                              tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG("gdr_copy"));

    if (strncmp(params->mode.device.dev_name,
                UCT_CUDA_DEV_NAME, strlen(UCT_CUDA_DEV_NAME)) != 0) {
        ucs_error("No device was found: %s", params->mode.device.dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    self->id = ucs_generate_uuid((uintptr_t)self);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_gdr_copy_iface_t)
{
    /* tasks to tear down the domain */
}

UCS_CLASS_DEFINE(uct_gdr_copy_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_gdr_copy_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                          const uct_iface_params_t*, const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_gdr_copy_iface_t, uct_iface_t);

UCT_TL_DEFINE(&uct_gdr_copy_component, gdr_copy, uct_cuda_base_query_devices,
              uct_gdr_copy_iface_t, "GDR_COPY_",
              uct_gdr_copy_iface_config_table, uct_gdr_copy_iface_config_t);
