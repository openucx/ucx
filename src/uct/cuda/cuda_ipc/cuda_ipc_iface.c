/**
 * Copyright (C) Mellanox Technologies Ltd. 2018-2019.  ALL RIGHTS RESERVED.
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_ipc_iface.h"
#include "cuda_ipc_md.h"
#include "cuda_ipc_ep.h"

#include <uct/cuda/base/cuda_iface.h>
#include <ucs/type/class.h>
#include <ucs/sys/string.h>
#include <ucs/debug/assert.h>
#include <sys/eventfd.h>
#include <pthread.h>
#include <nvml.h>

static ucs_config_field_t uct_cuda_ipc_iface_config_table[] = {

    {"", "", NULL,
     ucs_offsetof(uct_cuda_ipc_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"MAX_POLL", "16",
     "Max number of event completions to pick during cuda events polling",
      ucs_offsetof(uct_cuda_ipc_iface_config_t, max_poll), UCS_CONFIG_TYPE_UINT},

    {"MAX_STREAMS", "16",
     "Max number of CUDA streams to make concurrent progress on",
      ucs_offsetof(uct_cuda_ipc_iface_config_t, max_streams), UCS_CONFIG_TYPE_UINT},

    {"CACHE", "y",
     "Enable remote endpoint IPC memhandle mapping cache",
     ucs_offsetof(uct_cuda_ipc_iface_config_t, enable_cache),
     UCS_CONFIG_TYPE_BOOL},

    {"ENABLE_GET_ZCOPY", "auto",
     "Enable get operations except for platforms known to have slower performance",
     ucs_offsetof(uct_cuda_ipc_iface_config_t, enable_get_zcopy),
     UCS_CONFIG_TYPE_ON_OFF_AUTO},

    {"MAX_EVENTS", "inf",
     "Max number of cuda events. -1 is infinite",
     ucs_offsetof(uct_cuda_ipc_iface_config_t, max_cuda_ipc_events), UCS_CONFIG_TYPE_UINT},

    {NULL}
};


/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_ipc_iface_t)(uct_iface_t*);


ucs_status_t uct_cuda_ipc_iface_get_device_address(uct_iface_t *tl_iface,
                                                   uct_device_addr_t *addr)
{
    *(uint64_t*)addr = ucs_get_system_id();
    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_iface_get_address(uct_iface_h tl_iface,
                                                   uct_iface_addr_t *iface_addr)
{
    *(pid_t*)iface_addr = getpid();
    return UCS_OK;
}

static int uct_cuda_ipc_iface_is_reachable(const uct_iface_h tl_iface,
                                           const uct_device_addr_t *dev_addr,
                                           const uct_iface_addr_t *iface_addr)
{
    return (ucs_get_system_id() == *((const uint64_t*)dev_addr)) &&
           (getpid() != *(pid_t*)iface_addr);
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
    default:
        return 6911.0  * UCS_MBYTE;
    }
}

/* calls nvmlInit_v2 and nvmlShutdown which are expensive but
 * get_device_nvlinks should be outside critical path */
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

    status = UCT_NVML_FUNC_LOG_ERR(nvmlInit_v2());
    if (status != UCS_OK) {
        goto err;
    }

    status = UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetHandleByIndex(ordinal, &device));
    if (status != UCS_OK) {
        goto err_sd;
    }

    value.fieldId = NVML_FI_DEV_NVLINK_LINK_COUNT;

    UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetFieldValues(device, 1, &value));

    num_nvlinks = ((value.nvmlReturn == NVML_SUCCESS) &&
                   (value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)) ?
                  value.value.uiVal : 0;

    /* not enough to check number of nvlinks; need to check if links are active
     * by seeing if remote info can be obtained */
    for (link = 0; link < num_nvlinks; ++link) {
        status = UCT_NVML_FUNC(nvmlDeviceGetNvLinkRemotePciInfo(device, link,
                                                                &pci),
                               UCS_LOG_LEVEL_DEBUG);
        if (status != UCS_OK) {
            ucs_debug("could not find remote end info for link %u", link);
            goto err_sd;
        }
    }

    UCT_NVML_FUNC_LOG_ERR(nvmlShutdown());
    return num_nvlinks;

err_sd:
    UCT_NVML_FUNC_LOG_ERR(nvmlShutdown());
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

    uct_base_iface_query(&iface->super, iface_attr);

    iface_attr->iface_addr_len          = sizeof(pid_t);
    iface_attr->device_addr_len         = sizeof(uint64_t);
    iface_attr->ep_addr_len             = 0;
    iface_attr->max_conn_priv           = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE |
                                          UCT_IFACE_FLAG_EP_CHECK               |
                                          UCT_IFACE_FLAG_CONNECT_TO_IFACE       |
                                          UCT_IFACE_FLAG_PENDING                |
                                          UCT_IFACE_FLAG_GET_ZCOPY              |
                                          UCT_IFACE_FLAG_PUT_ZCOPY;
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

    iface_attr->latency                 = ucs_linear_func_make(1e-9, 0);
    iface_attr->bandwidth.dedicated     = 0;
    iface_attr->bandwidth.shared        = uct_cuda_ipc_iface_get_bw();
    iface_attr->overhead                = 0;
    iface_attr->priority                = 0;

    return UCS_OK;
}

static ucs_status_t
uct_cuda_ipc_iface_flush(uct_iface_h tl_iface, unsigned flags,
                         uct_completion_t *comp)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_ipc_iface_t);

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (ucs_queue_is_empty(&iface->outstanding_d2d_event_q)) {
        UCT_TL_IFACE_STAT_FLUSH(ucs_derived_of(tl_iface, uct_base_iface_t));
        return UCS_OK;
    }

    UCT_TL_IFACE_STAT_FLUSH_WAIT(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_INPROGRESS;
}

static ucs_status_t uct_cuda_ipc_iface_event_fd_get(uct_iface_h tl_iface, int *fd_p)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_ipc_iface_t);

    if (-1 == iface->eventfd) {
        iface->eventfd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        if (iface->eventfd == -1) {
            ucs_error("Failed to create event fd: %m");
            return UCS_ERR_IO_ERROR;
        }
    }

    *fd_p = iface->eventfd;
    return UCS_OK;
}

static void uct_cuda_ipc_common_cb(void *cuda_ipc_iface)
{
    uct_cuda_ipc_iface_t *iface = cuda_ipc_iface;
    uint64_t dummy = 1;
    int ret;

    /* No error handling yet */
    do {
        ret = write(iface->eventfd, &dummy, sizeof(dummy));
        if (ret == sizeof(dummy)) {
            return;
        } else if (ret == -1) {
            if (errno == EAGAIN) {
                continue;
            } else if (errno != EINTR) {
                ucs_error("Signaling wakeup failed: %m");
                return;
            }
        } else {
            ucs_assert(ret == 0);
        }
    } while (ret == 0);
}

#if (__CUDACC_VER_MAJOR__ >= 100000)
static void CUDA_CB myHostFn(void *iface)
#else
static void CUDA_CB myHostCallback(CUstream hStream,  CUresult status,
                                   void *iface)
#endif
{
    uct_cuda_ipc_common_cb(iface);
}

static UCS_F_ALWAYS_INLINE unsigned
uct_cuda_ipc_progress_event_q(uct_cuda_ipc_iface_t *iface,
                              ucs_queue_head_t *event_q)
{
    unsigned count = 0;
    uct_cuda_ipc_event_desc_t *cuda_ipc_event;
    ucs_queue_iter_t iter;
    ucs_status_t status;
    unsigned max_events = iface->config.max_poll;

    ucs_queue_for_each_safe(cuda_ipc_event, iter, event_q, queue) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(cuEventQuery(cuda_ipc_event->event));
        if (UCS_INPROGRESS == status) {
            continue;
        } else if (UCS_OK != status) {
            return status;
        }

        ucs_queue_del_iter(event_q, iter);
        if (cuda_ipc_event->comp != NULL) {
            uct_invoke_completion(cuda_ipc_event->comp, UCS_OK);
        }

        status = uct_cuda_ipc_unmap_memhandle(cuda_ipc_event->pid,
                                              cuda_ipc_event->d_bptr,
                                              cuda_ipc_event->mapped_addr,
                                              iface->config.enable_cache);
        if (status != UCS_OK) {
            ucs_fatal("failed to unmap addr:%p", cuda_ipc_event->mapped_addr);
        }

        ucs_trace_poll("CUDA_IPC Event Done :%p", cuda_ipc_event);
        iface->stream_refcount[cuda_ipc_event->stream_id]--;
        ucs_mpool_put(cuda_ipc_event);
        count++;

        if (count >= max_events) {
            break;
        }
    }

    return count;
}

static unsigned uct_cuda_ipc_iface_progress(uct_iface_h tl_iface)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_ipc_iface_t);

    return uct_cuda_ipc_progress_event_q(iface, &iface->outstanding_d2d_event_q);
}

static ucs_status_t uct_cuda_ipc_iface_event_fd_arm(uct_iface_h tl_iface,
                                                    unsigned events)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_ipc_iface_t);
    int ret;
    int i;
    uint64_t dummy;
    ucs_status_t status;

    if (uct_cuda_ipc_progress_event_q(iface, &iface->outstanding_d2d_event_q)) {
        return UCS_ERR_BUSY;
    }

    ucs_assert(iface->eventfd != -1);

    do {
        ret = read(iface->eventfd, &dummy, sizeof(dummy));
        if (ret == sizeof(dummy)) {
            status = UCS_ERR_BUSY;
            return status;
        } else if (ret == -1) {
            if (errno == EAGAIN) {
                break;
            } else if (errno != EINTR) {
                ucs_error("read from internal event fd failed: %m");
                status = UCS_ERR_IO_ERROR;
                return status;
            } else {
                return UCS_ERR_BUSY;
            }
        } else {
            ucs_assert(ret == 0);
        }
    } while (ret != 0);

    if (iface->streams_initialized) {
        for (i = 0; i < iface->config.max_streams; i++) {
            if (iface->stream_refcount[i]) {
                status =
#if (__CUDACC_VER_MAJOR__ >= 100000)
                UCT_CUDADRV_FUNC_LOG_ERR(cuLaunchHostFunc(iface->stream_d2d[i],
                                                          myHostFn, iface));
#else
                UCT_CUDADRV_FUNC_LOG_ERR(cuStreamAddCallback(iface->stream_d2d[i],
                                                             myHostCallback,
                                                             iface, 0));
#endif
                if (UCS_OK != status) {
                    return status;
                }
            }
        }
    }
    return UCS_OK;
}

static uct_iface_ops_t uct_cuda_ipc_iface_ops = {
    .ep_get_zcopy             = uct_cuda_ipc_ep_get_zcopy,
    .ep_put_zcopy             = uct_cuda_ipc_ep_put_zcopy,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_check                 = uct_cuda_ipc_ep_check,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_cuda_ipc_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_ipc_ep_t),
    .iface_flush              = uct_cuda_ipc_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_cuda_ipc_iface_progress,
    .iface_event_fd_get       = uct_cuda_ipc_iface_event_fd_get,
    .iface_event_arm          = uct_cuda_ipc_iface_event_fd_arm,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_ipc_iface_t),
    .iface_query              = uct_cuda_ipc_iface_query,
    .iface_get_device_address = uct_cuda_ipc_iface_get_device_address,
    .iface_get_address        = uct_cuda_ipc_iface_get_address,
    .iface_is_reachable       = uct_cuda_ipc_iface_is_reachable,
};

static void uct_cuda_ipc_event_desc_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_cuda_ipc_event_desc_t *base = (uct_cuda_ipc_event_desc_t *) obj;

    memset(base, 0, sizeof(*base));
    UCT_CUDADRV_FUNC_LOG_ERR(cuEventCreate(&base->event, CU_EVENT_DISABLE_TIMING));
}

static void uct_cuda_ipc_event_desc_cleanup(ucs_mpool_t *mp, void *obj)
{
    uct_cuda_ipc_event_desc_t *base = (uct_cuda_ipc_event_desc_t *) obj;
    uct_cuda_ipc_iface_t *iface     = ucs_container_of(mp,
                                                       uct_cuda_ipc_iface_t,
                                                       event_desc);
    CUcontext cuda_context;

    UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetCurrent(&cuda_context));
    if (uct_cuda_base_context_match(cuda_context, iface->cuda_context)) {
        UCT_CUDA_FUNC_LOG_ERR(cudaEventDestroy(base->event));
    }
}

ucs_status_t uct_cuda_ipc_iface_init_streams(uct_cuda_ipc_iface_t *iface)
{
    ucs_status_t status;
    int i;

    for (i = 0; i < iface->config.max_streams; i++) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(cuStreamCreate(&iface->stream_d2d[i],
                                                         CU_STREAM_NON_BLOCKING));
        if (UCS_OK != status) {
            return status;
        }

        iface->stream_refcount[i] = 0;
    }

    iface->streams_initialized = 1;

    return UCS_OK;
}

static ucs_mpool_ops_t uct_cuda_ipc_event_desc_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = uct_cuda_ipc_event_desc_init,
    .obj_cleanup   = uct_cuda_ipc_event_desc_cleanup,
    .obj_str       = NULL
};

static UCS_CLASS_INIT_FUNC(uct_cuda_ipc_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_cuda_ipc_iface_config_t *config = NULL;
    ucs_status_t status;

    config = ucs_derived_of(tl_config, uct_cuda_ipc_iface_config_t);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_cuda_ipc_iface_ops,
                              &uct_base_iface_internal_ops, md, worker, params,
                              tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG("cuda_ipc"));

    if (strncmp(params->mode.device.dev_name,
                UCT_CUDA_DEV_NAME, strlen(UCT_CUDA_DEV_NAME)) != 0) {
        ucs_error("No device was found: %s", params->mode.device.dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    self->config.max_poll            = config->max_poll;
    self->config.max_streams         = config->max_streams;
    self->config.enable_cache        = config->enable_cache;
    self->config.enable_get_zcopy    = config->enable_get_zcopy;
    self->config.max_cuda_ipc_events = config->max_cuda_ipc_events;

    status = ucs_mpool_init(&self->event_desc,
                            0,
                            sizeof(uct_cuda_ipc_event_desc_t),
                            0,
                            UCS_SYS_CACHE_LINE_SIZE,
                            128,
                            self->config.max_cuda_ipc_events,
                            &uct_cuda_ipc_event_desc_mpool_ops,
                            "CUDA_IPC EVENT objects");
    if (UCS_OK != status) {
        ucs_error("mpool creation failed");
        return UCS_ERR_IO_ERROR;
    }

    self->eventfd             = -1;
    self->streams_initialized = 0;
    self->cuda_context        = 0;
    ucs_queue_head_init(&self->outstanding_d2d_event_q);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_ipc_iface_t)
{
    ucs_status_t status;
    int i;
    CUcontext cuda_context;

    UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetCurrent(&cuda_context));

    if (self->streams_initialized &&
        uct_cuda_base_context_match(cuda_context, self->cuda_context)) {
        for (i = 0; i < self->config.max_streams; i++) {
            status = UCT_CUDADRV_FUNC_LOG_ERR(cuStreamDestroy(self->stream_d2d[i]));
            if (UCS_OK != status) {
                continue;
            }

            ucs_assert(self->stream_refcount[i] == 0);
        }
        self->streams_initialized = 0;
    }

    uct_base_iface_progress_disable(&self->super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    ucs_mpool_cleanup(&self->event_desc, 1);
    if (self->eventfd != -1) {
        close(self->eventfd);
    }
}

ucs_status_t
uct_cuda_ipc_query_devices(
        uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
        unsigned *num_tl_devices_p)
{
    return uct_cuda_base_query_devices_common(md, UCT_DEVICE_TYPE_SHM,
                                              tl_devices_p, num_tl_devices_p);
}

UCS_CLASS_DEFINE(uct_cuda_ipc_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_cuda_ipc_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                          const uct_iface_params_t*, const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_cuda_ipc_iface_t, uct_iface_t);

UCT_TL_DEFINE(&uct_cuda_ipc_component.super, cuda_ipc,
              uct_cuda_ipc_query_devices, uct_cuda_ipc_iface_t, "CUDA_IPC_",
              uct_cuda_ipc_iface_config_table, uct_cuda_ipc_iface_config_t);
