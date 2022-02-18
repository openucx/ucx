/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rocm_ipc_iface.h"
#include "rocm_ipc_md.h"
#include "rocm_ipc_ep.h"

#include <uct/rocm/base/rocm_base.h>
#include <ucs/arch/cpu.h>
#include <ucs/type/class.h>
#include <ucs/sys/string.h>


static ucs_config_field_t uct_rocm_ipc_iface_config_table[] = {

    {"", "", NULL,
     ucs_offsetof(uct_rocm_ipc_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {NULL}
};

static double uct_rocm_ipc_iface_get_bw()
{
    double bw = 30.0 * UCS_GBYTE;
    hsa_amd_link_info_type_t type;

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

static int uct_rocm_ipc_iface_is_reachable(const uct_iface_h tl_iface,
                                           const uct_device_addr_t *dev_addr,
                                           const uct_iface_addr_t *iface_addr)
{
    return (ucs_get_system_id() == *((const uint64_t*)dev_addr)) &&
           (getpid() != *(pid_t*)iface_addr);
}

static ucs_status_t uct_rocm_ipc_iface_query(uct_iface_h tl_iface,
                                             uct_iface_attr_t *iface_attr)
{
    uct_rocm_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_rocm_ipc_iface_t);

    uct_base_iface_query(&iface->super, iface_attr);

    iface_attr->cap.put.min_zcopy       = 0;
    iface_attr->cap.put.max_zcopy       = SIZE_MAX;
    iface_attr->cap.put.opt_zcopy_align = sizeof(uint32_t);
    iface_attr->cap.put.align_mtu       = iface_attr->cap.put.opt_zcopy_align;
    iface_attr->cap.put.max_iov         = 1;

    iface_attr->cap.get.min_zcopy       = 0;
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

    iface_attr->latency                 = ucs_linear_func_make(1e-9, 0);
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
    static const unsigned max_signals = 16;
    unsigned count = 0;
    uct_rocm_ipc_signal_desc_t *rocm_ipc_signal;
    ucs_queue_iter_t iter;

    ucs_queue_for_each_safe(rocm_ipc_signal, iter, &iface->signal_queue, queue) {
        if (hsa_signal_load_scacquire(rocm_ipc_signal->signal) != 0) {
            continue;
        }

        ucs_queue_del_iter(&iface->signal_queue, iter);
        if (rocm_ipc_signal->comp != NULL) {
            uct_invoke_completion(rocm_ipc_signal->comp, UCS_OK);
        }

        ucs_trace_poll("ROCM_IPC Signal Done :%p", rocm_ipc_signal);
        ucs_mpool_put(rocm_ipc_signal);
        count++;

        if (count >= max_signals) {
            break;
        }
    }

    return count;
}

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
    .iface_is_reachable       = uct_rocm_ipc_iface_is_reachable
};

static void uct_rocm_ipc_signal_desc_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_rocm_ipc_signal_desc_t *base = (uct_rocm_ipc_signal_desc_t *)obj;
    hsa_status_t status;

    memset(base, 0, sizeof(*base));
    status = hsa_signal_create(1, 0, NULL, &base->signal);
    if (status != HSA_STATUS_SUCCESS) {
        ucs_fatal("fail to create signal");
    }
}

static void uct_rocm_ipc_signal_desc_cleanup(ucs_mpool_t *mp, void *obj)
{
    uct_rocm_ipc_signal_desc_t *base = (uct_rocm_ipc_signal_desc_t *)obj;
    hsa_status_t status;

    status = hsa_signal_destroy(base->signal);
    if (status != HSA_STATUS_SUCCESS) {
        ucs_fatal("fail to destroy signal");
    }
}

static ucs_mpool_ops_t uct_rocm_ipc_signal_desc_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = uct_rocm_ipc_signal_desc_init,
    .obj_cleanup   = uct_rocm_ipc_signal_desc_cleanup,
    .obj_str       = NULL
};

static UCS_CLASS_INIT_FUNC(uct_rocm_ipc_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_rocm_ipc_iface_ops, 
                              &uct_base_iface_internal_ops,
			      md, worker, params,
                              tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(UCT_ROCM_IPC_TL_NAME));

    status = ucs_mpool_init(&self->signal_pool,
                            0,
                            sizeof(uct_rocm_ipc_signal_desc_t),
                            0,
                            UCS_SYS_CACHE_LINE_SIZE,
                            128,
                            1024,
                            &uct_rocm_ipc_signal_desc_mpool_ops,
                            "ROCM_IPC signal objects");
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
