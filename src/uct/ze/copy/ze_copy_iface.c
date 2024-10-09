/*
 * Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ze_copy_iface.h"
#include "ze_copy_md.h"
#include "ze_copy_ep.h"

#include <uct/ze/base/ze_base.h>
#include <ucs/type/class.h>
#include <ucs/sys/string.h>


/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_ze_copy_iface_t)(uct_iface_t*);


static ucs_status_t uct_ze_copy_iface_get_address(uct_iface_h tl_iface,
                                                  uct_iface_addr_t *iface_addr)
{
    uct_ze_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_ze_copy_iface_t);

    *(uct_ze_copy_iface_addr_t*)iface_addr = iface->id;
    return UCS_OK;
}

static int
uct_ze_copy_iface_is_reachable_v2(const uct_iface_h tl_iface,
                                  const uct_iface_is_reachable_params_t *params)
{
    uct_ze_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_ze_copy_iface_t);
    uct_ze_copy_iface_addr_t *addr;

    if (!uct_iface_is_reachable_params_addrs_valid(params)) {
        return 0;
    }

    addr = (uct_ze_copy_iface_addr_t*)params->iface_addr;
    if (addr == NULL) {
        uct_iface_fill_info_str_buf(params, "device address is empty");
        return 0;
    }

    if (iface->id != *addr) {
        uct_iface_fill_info_str_buf(params,
                                    "different iface id %"PRIx64" vs %"PRIx64"",
                                    iface->id, *addr);
        return 0;
    }

    return uct_iface_scope_is_reachable(tl_iface, params);
}

static ucs_status_t
uct_ze_copy_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_ze_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_ze_copy_iface_t);

    uct_base_iface_query(&iface->super, iface_attr);

    iface_attr->iface_addr_len  = sizeof(uct_ze_copy_iface_addr_t);
    iface_attr->device_addr_len = 0;
    iface_attr->ep_addr_len     = 0;
    iface_attr->cap.flags       = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                  UCT_IFACE_FLAG_GET_SHORT |
                                  UCT_IFACE_FLAG_PUT_SHORT |
                                  UCT_IFACE_FLAG_GET_ZCOPY |
                                  UCT_IFACE_FLAG_PUT_ZCOPY | UCT_IFACE_FLAG_PENDING;

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

    iface_attr->cap.am.max_short       = 0;
    iface_attr->cap.am.max_bcopy       = 0;
    iface_attr->cap.am.min_zcopy       = 0;
    iface_attr->cap.am.max_zcopy       = 0;
    iface_attr->cap.am.opt_zcopy_align = 1;
    iface_attr->cap.am.align_mtu       = iface_attr->cap.am.opt_zcopy_align;
    iface_attr->cap.am.max_hdr         = 0;
    iface_attr->cap.am.max_iov         = 1;

    iface_attr->latency             = ucs_linear_func_make(10e-6, 0);
    iface_attr->bandwidth.dedicated = 50000.0 * UCS_MBYTE;
    iface_attr->bandwidth.shared    = 0;
    iface_attr->overhead            = 0;
    iface_attr->priority            = 0;

    return UCS_OK;
}

static uct_iface_ops_t uct_ze_copy_iface_ops = {
    .ep_get_short             = uct_ze_copy_ep_get_short,
    .ep_put_short             = uct_ze_copy_ep_put_short,
    .ep_get_zcopy             = uct_ze_copy_ep_get_zcopy,
    .ep_put_zcopy             = uct_ze_copy_ep_put_zcopy,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = uct_ep_create,
    .ep_destroy               = uct_ep_destroy,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_ze_copy_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_ze_copy_ep_t),
    .iface_flush              = uct_base_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = ucs_empty_function_return_zero,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_ze_copy_iface_t),
    .iface_query              = uct_ze_copy_iface_query,
    .iface_get_device_address = ucs_empty_function_return_success,
    .iface_get_address        = uct_ze_copy_iface_get_address,
    .iface_is_reachable       = uct_base_iface_is_reachable,
};


static ucs_status_t
uct_ze_copy_estimate_perf(uct_iface_h tl_iface, uct_perf_attr_t *perf_attr)
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


static uct_iface_internal_ops_t uct_ze_copy_iface_internal_ops = {
    .iface_estimate_perf   = uct_ze_copy_estimate_perf,
    .iface_vfs_refresh     = ucs_empty_function,
    .ep_query              = ucs_empty_function_return_unsupported,
    .ep_invalidate         = ucs_empty_function_return_unsupported,
    .ep_connect_to_ep_v2   = ucs_empty_function_return_unsupported,
    .iface_is_reachable_v2 = uct_ze_copy_iface_is_reachable_v2,
    .ep_is_connected       = uct_base_ep_is_connected
};

static UCS_CLASS_INIT_FUNC(uct_ze_copy_iface_t, uct_md_h md,
                           uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_ze_copy_md_t *ze_md         = ucs_derived_of(md, uct_ze_copy_md_t);
    ze_command_queue_desc_t cq_desc = {};
    ze_command_list_desc_t cl_desc  = {};
    ze_device_handle_t device;
    ze_command_queue_handle_t cmdq;
    ze_command_list_handle_t cmdl;
    ze_result_t ret;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_ze_copy_iface_ops,
                              &uct_ze_copy_iface_internal_ops, md, worker,
                              params,
                              tl_config UCS_STATS_ARG(params->stats_root)
                                      UCS_STATS_ARG(UCT_ZE_COPY_TL_NAME));

    /* TODO: choose device based on params */
    device = uct_ze_base_get_device(0);
    if (device == NULL) {
        return UCS_ERR_NO_DEVICE;
    }

    ret = zeCommandQueueCreate(ze_md->ze_context, device, &cq_desc, &cmdq);
    if (ret != ZE_RESULT_SUCCESS) {
        return UCS_ERR_NO_DEVICE;
    }

    ret = zeCommandListCreate(ze_md->ze_context, device, &cl_desc, &cmdl);
    if (ret != ZE_RESULT_SUCCESS) {
        zeCommandQueueDestroy(cmdq);
        return UCS_ERR_NO_DEVICE;
    }

    self->ze_cmdq = cmdq;
    self->ze_cmdl = cmdl;
    self->id      = ucs_generate_uuid((uintptr_t)self);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ze_copy_iface_t)
{
    zeCommandListDestroy(self->ze_cmdl);
    zeCommandQueueDestroy(self->ze_cmdq);
}

UCS_CLASS_DEFINE(uct_ze_copy_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_ze_copy_iface_t, uct_iface_t, uct_md_h,
                          uct_worker_h, const uct_iface_params_t*,
                          const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ze_copy_iface_t, uct_iface_t);

UCT_TL_DEFINE(&uct_ze_copy_component, ze_copy, uct_ze_base_query_devices,
              uct_ze_copy_iface_t, "ZE_COPY_", uct_iface_config_table,
              uct_iface_config_t);
