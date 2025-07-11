/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018-2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_ipc_iface.h"
#include "cuda_ipc_md.h"
#include "cuda_ipc_ep.h"

#include <uct/cuda/base/cuda_iface.h>
#include <uct/cuda/base/cuda_md.h>
#include <uct/cuda/base/cuda_nvml.h>
#include <ucs/type/class.h>
#include <ucs/sys/string.h>
#include <ucs/debug/assert.h>
#include <ucs/async/eventfd.h>
#include <pthread.h>


typedef enum {
    UCT_CUDA_IPC_DEVICE_ADDR_FLAG_MNNVL = UCS_BIT(0)
} uct_cuda_ipc_device_addr_flags_t;


typedef struct {
    uint64_t     system_uuid;
    uint8_t      flags; /* uct_cuda_ipc_device_addr_flags_t */
} UCS_S_PACKED uct_cuda_ipc_device_addr_t;


static ucs_config_field_t uct_cuda_ipc_iface_config_table[] = {

    {"", "", NULL,
     ucs_offsetof(uct_cuda_ipc_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"MAX_POLL", "16",
     "Max number of event completions to pick during cuda events polling",
      ucs_offsetof(uct_cuda_ipc_iface_config_t, params.max_poll), UCS_CONFIG_TYPE_UINT},

    {"MAX_STREAMS", UCS_PP_MAKE_STRING(UCT_CUDA_IPC_MAX_PEERS),
     "Max number of CUDA streams to make concurrent progress on",
      ucs_offsetof(uct_cuda_ipc_iface_config_t, params.max_streams), UCS_CONFIG_TYPE_UINT},

    {"CACHE", "y",
     "Enable remote endpoint IPC memhandle mapping cache",
     ucs_offsetof(uct_cuda_ipc_iface_config_t, params.enable_cache),
     UCS_CONFIG_TYPE_BOOL},

    {"ENABLE_GET_ZCOPY", "auto",
     "Enable get operations except for platforms known to have slower performance",
     ucs_offsetof(uct_cuda_ipc_iface_config_t, params.enable_get_zcopy),
     UCS_CONFIG_TYPE_ON_OFF_AUTO},

    {"MAX_EVENTS", "inf",
     "Max number of cuda events. -1 is infinite",
     ucs_offsetof(uct_cuda_ipc_iface_config_t, params.max_cuda_ipc_events), UCS_CONFIG_TYPE_UINT},

    {"BW", "auto",
     "Effective p2p memory bandwidth",
     ucs_offsetof(uct_cuda_ipc_iface_config_t, params.bandwidth), UCS_CONFIG_TYPE_BW},

    {"LAT", "1.8us",
     "Estimated latency",
     ucs_offsetof(uct_cuda_ipc_iface_config_t, params.latency), UCS_CONFIG_TYPE_TIME},

    {"OVERHEAD", "4.0us",
     "Estimated CPU overhead for transferring GPU memory",
     ucs_offsetof(uct_cuda_ipc_iface_config_t, params.overhead), UCS_CONFIG_TYPE_TIME},
    {NULL}
};


/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_ipc_iface_t)(uct_iface_t*);


ucs_status_t uct_cuda_ipc_iface_get_device_address(uct_iface_t *tl_iface,
                                                   uct_device_addr_t *addr)
{
    uct_cuda_ipc_device_addr_t *dev_addr = (uct_cuda_ipc_device_addr_t*)addr;
    uct_cuda_ipc_iface_t *iface          = ucs_derived_of(tl_iface,
                                                          uct_cuda_ipc_iface_t);
    uct_cuda_ipc_md_t *md                = ucs_derived_of(iface->super.super.md,
                                                           uct_cuda_ipc_md_t);

    if (md->enable_mnnvl) {
        dev_addr->flags = UCT_CUDA_IPC_DEVICE_ADDR_FLAG_MNNVL;
    }

    dev_addr->system_uuid = ucs_get_system_id();

    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_iface_get_address(uct_iface_h tl_iface,
                                                   uct_iface_addr_t *iface_addr)
{
    *(pid_t*)iface_addr = getpid();
    return UCS_OK;
}

static int
uct_cuda_ipc_iface_mnnvl_supported(uct_cuda_ipc_md_t *md,
                                   const uct_cuda_ipc_device_addr_t *dev_addr,
                                   size_t dev_addr_len)
{
    if (md->enable_mnnvl && (dev_addr_len != sizeof(uint64_t))) {
        ucs_assertv(dev_addr_len >= sizeof(uct_cuda_ipc_device_addr_t),
                    "dev_addr_len=%zu", dev_addr_len);
        return (dev_addr->flags & UCT_CUDA_IPC_DEVICE_ADDR_FLAG_MNNVL);
    }

    return 0;
}

static int
uct_cuda_ipc_iface_is_reachable_v2(const uct_iface_h tl_iface,
                                   const uct_iface_is_reachable_params_t *params)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_ipc_iface_t);
    uct_cuda_ipc_md_t *md       = ucs_derived_of(iface->super.super.md,
                                                 uct_cuda_ipc_md_t);
    const uct_cuda_ipc_device_addr_t *dev_addr;
    size_t dev_addr_len;
    int same_uuid;

    if (!uct_iface_is_reachable_params_addrs_valid(params)) {
        return 0;
    }

    dev_addr_len = UCS_PARAM_VALUE(UCT_IFACE_IS_REACHABLE_FIELD, params,
                                   device_addr_length, DEVICE_ADDR_LENGTH,
                                   sizeof(dev_addr->system_uuid));
    dev_addr     = (const uct_cuda_ipc_device_addr_t *)params->device_addr;
    same_uuid    = (ucs_get_system_id() == dev_addr->system_uuid);

    if ((getpid() == *(pid_t*)params->iface_addr) && same_uuid) {
        uct_iface_fill_info_str_buf(params, "same process");
        return 0;
    }

    if (same_uuid ||
        uct_cuda_ipc_iface_mnnvl_supported(md, dev_addr, dev_addr_len)) {
        return uct_iface_scope_is_reachable(tl_iface, params);
    }

    uct_iface_fill_info_str_buf(params, "MNNVL is not supported");
    return 0;
}

static double uct_cuda_ipc_iface_get_bw()
{
    CUdevice cu_device;
    int major_version;
    ucs_status_t status;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGet(&cu_device, 0));
    if (status != UCS_OK) {
        return 0;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuDeviceGetAttribute(&major_version,
                                 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                 cu_device));
    if (status != UCS_OK) {
        return 0;
    }

    /*
     * TODO: Detect nvswitch
     */
    switch (major_version) {
    case UCT_CUDA_BASE_GEN_P100:
        return 80000.0 * UCS_MBYTE;
    case UCT_CUDA_BASE_GEN_V100:
        return 250000.0 * UCS_MBYTE;
    case UCT_CUDA_BASE_GEN_A100:
        return 300000.0 * UCS_MBYTE;
    case UCT_CUDA_BASE_GEN_H100:
        return 400000.0 * UCS_MBYTE;
    case UCT_CUDA_BASE_GEN_B100:
        return 800000.0 * UCS_MBYTE;
    default:
        return 6911.0  * UCS_MBYTE;
    }
}

static int uct_cuda_ipc_get_device_nvlinks(int ordinal)
{
    static int num_nvlinks = -1;
    unsigned link;
    nvmlDevice_t device;
    nvmlFieldValue_t value;
    nvmlPciInfo_t pci;
    ucs_status_t status;

    if (num_nvlinks != -1) {
        return num_nvlinks;
    }

    status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetHandleByIndex, 0, &device);
    if (status != UCS_OK) {
        goto err;
    }

    value.fieldId = NVML_FI_DEV_NVLINK_LINK_COUNT;

    UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetFieldValues, device, 1, &value);

    num_nvlinks = ((value.nvmlReturn == NVML_SUCCESS) &&
                   (value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)) ?
                  value.value.uiVal : 0;

    /* not enough to check number of nvlinks; need to check if links are active
     * by seeing if remote info can be obtained */
    for (link = 0; link < num_nvlinks; ++link) {
        status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetNvLinkRemotePciInfo,
                                         device, link, &pci);
        if (status != UCS_OK) {
            ucs_debug("could not find remote end info for link %u", link);
            goto err;
        }
    }

    return num_nvlinks;

err:
    return 0;
}

static size_t uct_cuda_ipc_iface_get_max_get_zcopy(uct_cuda_ipc_iface_t *iface)
{
    int num_nvlinks;

    /* assume there is at least >= 1 GPUs on the system; assume uniformity */
    num_nvlinks = uct_cuda_ipc_get_device_nvlinks(0);

    if (!num_nvlinks && (iface->config.enable_get_zcopy != UCS_CONFIG_ON)) {
        ucs_debug("cuda-ipc get zcopy disabled as no nvlinks detected");
        return 0;
    }

    return ULONG_MAX;
}

static ucs_status_t uct_cuda_ipc_iface_query(uct_iface_h tl_iface,
                                             uct_iface_attr_t *iface_attr)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_ipc_iface_t);
    uct_cuda_ipc_md_t *md       = ucs_derived_of(iface->super.super.md,
                                                 uct_cuda_ipc_md_t);

    uct_base_iface_query(&iface->super.super, iface_attr);

    iface_attr->iface_addr_len          = sizeof(pid_t);
    iface_attr->device_addr_len         = md->enable_mnnvl ?
                                          sizeof(uct_cuda_ipc_device_addr_t) :
                                          sizeof(uint64_t);
    iface_attr->ep_addr_len             = 0;
    iface_attr->max_conn_priv           = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE |
                                          UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                          UCT_IFACE_FLAG_PENDING          |
                                          UCT_IFACE_FLAG_GET_ZCOPY        |
                                          UCT_IFACE_FLAG_PUT_ZCOPY;
    if (md->enable_mnnvl) {
        iface_attr->cap.flags |= UCT_IFACE_FLAG_INTER_NODE;
    }

    iface_attr->cap.event_flags         = UCT_IFACE_FLAG_EVENT_SEND_COMP |
                                          UCT_IFACE_FLAG_EVENT_RECV      |
                                          UCT_IFACE_FLAG_EVENT_FD;

    iface_attr->cap.put.max_short       = 0;
    iface_attr->cap.put.max_bcopy       = 0;
    iface_attr->cap.put.min_zcopy       = 0;
    iface_attr->cap.put.max_zcopy       = ULONG_MAX;
    iface_attr->cap.put.opt_zcopy_align = 1;
    iface_attr->cap.put.align_mtu       = iface_attr->cap.put.opt_zcopy_align;
    iface_attr->cap.put.max_iov         = 1;

    iface_attr->cap.get.max_bcopy       = 0;
    iface_attr->cap.get.min_zcopy       = 0;
    iface_attr->cap.get.max_zcopy       = uct_cuda_ipc_iface_get_max_get_zcopy(iface);
    iface_attr->cap.get.opt_zcopy_align = 1;
    iface_attr->cap.get.align_mtu       = iface_attr->cap.get.opt_zcopy_align;
    iface_attr->cap.get.max_iov         = 1;

    /* Latency and overhead don't depend on config values for
     * the sake of wire compatibility */
    iface_attr->latency                 = ucs_linear_func_make(1e-6, 0);
    iface_attr->bandwidth.dedicated     = 0;
    iface_attr->bandwidth.shared        = iface->config.bandwidth;
    iface_attr->overhead                = 7.0e-6;
    iface_attr->priority                = 0;

    return UCS_OK;
}

static void uct_cuda_ipc_complete_event(uct_iface_h tl_iface,
                                        uct_cuda_event_desc_t *cuda_event)
{
    uct_cuda_ipc_iface_t *iface               = ucs_derived_of(tl_iface,
                                                               uct_cuda_ipc_iface_t);
    uct_cuda_ipc_event_desc_t *cuda_ipc_event = ucs_derived_of(cuda_event,
                                                               uct_cuda_ipc_event_desc_t);
    ucs_status_t status;

    status = uct_cuda_ipc_unmap_memhandle(cuda_ipc_event->pid,
                                          cuda_ipc_event->d_bptr,
                                          cuda_ipc_event->mapped_addr,
                                          cuda_ipc_event->cuda_device,
                                          iface->config.enable_cache);
    if (status != UCS_OK) {
        ucs_fatal("failed to unmap addr:%p", cuda_ipc_event->mapped_addr);
    }
}

static uct_iface_ops_t uct_cuda_ipc_iface_ops = {
    .ep_get_zcopy             = uct_cuda_ipc_ep_get_zcopy,
    .ep_put_zcopy             = uct_cuda_ipc_ep_put_zcopy,
    .ep_pending_add           = (uct_ep_pending_add_func_t)ucs_empty_function_return_busy,
    .ep_pending_purge         = (uct_ep_pending_purge_func_t)ucs_empty_function,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_check                 = (uct_ep_check_func_t)ucs_empty_function_return_unsupported,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_cuda_ipc_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_ipc_ep_t),
    .iface_flush              = uct_cuda_base_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_cuda_base_iface_progress,
    .iface_event_fd_get       = uct_cuda_base_iface_event_fd_get,
    .iface_event_arm          = uct_cuda_base_iface_event_fd_arm,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_ipc_iface_t),
    .iface_query              = uct_cuda_ipc_iface_query,
    .iface_get_device_address = uct_cuda_ipc_iface_get_device_address,
    .iface_get_address        = uct_cuda_ipc_iface_get_address,
    .iface_is_reachable       = uct_base_iface_is_reachable,
};

static ucs_status_t
uct_cuda_ipc_estimate_perf(uct_iface_h tl_iface, uct_perf_attr_t *perf_attr)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_ipc_iface_t);

    perf_attr->bandwidth.dedicated = 0;

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_BANDWIDTH) {

        /* TODO:
         *      1. differentiate read vs write perf
         *      2. check if src,dst memory is cuda type;
         *         (currently UCP does not pass mem types so cuda_ipc would
         *          report zero bandwidth)
         *      3. Check nvlinks and report bandwidth;
         *         (for now not checking because we do not want to report zero
         *          bandwidth for PCIe connected IPC-accessible devices)
         */

        perf_attr->bandwidth.shared = uct_cuda_ipc_iface_get_bw();
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_PATH_BANDWIDTH) {
        perf_attr->path_bandwidth.dedicated = 0;
        perf_attr->path_bandwidth.shared    = uct_cuda_ipc_iface_get_bw();
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD) {
        perf_attr->send_pre_overhead = iface->config.overhead;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD) {
        /* In case of sync mem copy, the send operation CPU overhead includes
           the latency of waiting for the copy to complete */
        perf_attr->send_post_overhead = 0;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_RECV_OVERHEAD) {
        perf_attr->recv_overhead = 0;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_LATENCY) {
        /* In case of async mem copy, the latency is not part of the overhead
           and it's a standalone property */
        perf_attr->latency = ucs_linear_func_make(iface->config.latency, 0.0);
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_MAX_INFLIGHT_EPS) {
        perf_attr->max_inflight_eps = SIZE_MAX;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_FLAGS) {
        perf_attr->flags = 0;
    }

    return UCS_OK;
}

static uct_iface_internal_ops_t uct_cuda_ipc_iface_internal_ops = {
    .iface_estimate_perf   = uct_cuda_ipc_estimate_perf,
    .iface_vfs_refresh     = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
    .ep_query              = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
    .ep_invalidate         = (uct_ep_invalidate_func_t)ucs_empty_function_return_unsupported,
    .ep_connect_to_ep_v2   = (uct_ep_connect_to_ep_v2_func_t)ucs_empty_function_return_unsupported,
    .iface_is_reachable_v2 = uct_cuda_ipc_iface_is_reachable_v2,
    .ep_is_connected       = uct_cuda_ipc_ep_is_connected
};

static uct_cuda_ctx_rsc_t * uct_cuda_ipc_ctx_rsc_create(uct_iface_h tl_iface)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_ipc_iface_t);
    uct_cuda_ipc_ctx_rsc_t *ctx_rsc;
    unsigned i;

    ctx_rsc = ucs_malloc(sizeof(*ctx_rsc), "uct_cuda_ipc_ctx_rsc_t");
    if (ctx_rsc == NULL) {
        ucs_error("failed to allocate cuda ipc context resource struct");
        return NULL;
    }

    for (i = 0; i < iface->config.max_streams; i++) {
        uct_cuda_base_queue_desc_init(&ctx_rsc->queue_desc[i]);
    }

    return &ctx_rsc->super;
}

static void uct_cuda_ipc_ctx_rsc_destroy(uct_iface_h tl_iface,
                                         uct_cuda_ctx_rsc_t *cuda_ctx_rsc)
{
    uct_cuda_ipc_iface_t *iface     = ucs_derived_of(tl_iface, uct_cuda_ipc_iface_t);
    uct_cuda_ipc_ctx_rsc_t *ctx_rsc = ucs_derived_of(cuda_ctx_rsc, uct_cuda_ipc_ctx_rsc_t);
    unsigned i;

    for (i = 0; i < iface->config.max_streams; i++) {
        uct_cuda_base_queue_desc_destroy(cuda_ctx_rsc, &ctx_rsc->queue_desc[i]);
    }

    ucs_free(ctx_rsc);
}

static uct_cuda_iface_ops_t uct_cuda_iface_ops = {
    .create_rsc     = uct_cuda_ipc_ctx_rsc_create,
    .destroy_rsc    = uct_cuda_ipc_ctx_rsc_destroy,
    .complete_event = uct_cuda_ipc_complete_event
};

static UCS_CLASS_INIT_FUNC(uct_cuda_ipc_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_cuda_ipc_iface_config_t *config = NULL;
    ucs_status_t status;

    config = ucs_derived_of(tl_config, uct_cuda_ipc_iface_config_t);
    UCS_CLASS_CALL_SUPER_INIT(uct_cuda_iface_t, &uct_cuda_ipc_iface_ops,
                              &uct_cuda_ipc_iface_internal_ops, md, worker, params,
                              tl_config, "cuda_ipc");

    status = uct_cuda_base_check_device_name(params);
    if (status != UCS_OK) {
        return status;
    }

    if (config->params.max_streams > UCT_CUDA_IPC_MAX_PEERS) {
        ucs_error("invalid max streams value (%u > %u)",
                  config->params.max_streams, UCT_CUDA_IPC_MAX_PEERS);
        return UCS_ERR_INVALID_PARAM;
    }

    self->config = config->params;
    if (UCS_CONFIG_DBL_IS_AUTO(config->params.bandwidth)) {
        self->config.bandwidth = uct_cuda_ipc_iface_get_bw() ;
    }

    self->super.ops                    = &uct_cuda_iface_ops;
    self->super.config.max_events      = self->config.max_cuda_ipc_events;
    self->super.config.max_poll        = self->config.max_poll;
    self->super.config.event_desc_size = sizeof(uct_cuda_ipc_event_desc_t);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_ipc_iface_t)
{
}

ucs_status_t
uct_cuda_ipc_query_devices(
        uct_md_h uct_md, uct_tl_device_resource_t **tl_devices_p,
        unsigned *num_tl_devices_p)
{
    /* Declare cuda-ipc as shm device even if MNNVL is supported. In this case
     * UCX_NET_DEVICES doesn't affect cuda-ipc (which is well established
     * interface). When MNNVL is supported cuda-ipc iface sets
     * UCT_IFACE_FLAG_INTER_NODE flag instead. */
    return uct_cuda_base_query_devices_common(uct_md, UCT_DEVICE_TYPE_SHM,
                                              tl_devices_p, num_tl_devices_p);
}

UCS_CLASS_DEFINE(uct_cuda_ipc_iface_t, uct_cuda_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_cuda_ipc_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                          const uct_iface_params_t*, const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_cuda_ipc_iface_t, uct_iface_t);

UCT_TL_DEFINE(&uct_cuda_ipc_component.super, cuda_ipc,
              uct_cuda_ipc_query_devices, uct_cuda_ipc_iface_t, "CUDA_IPC_",
              uct_cuda_ipc_iface_config_table, uct_cuda_ipc_iface_config_t);
