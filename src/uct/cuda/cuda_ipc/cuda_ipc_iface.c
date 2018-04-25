/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#include "cuda_ipc_iface.h"
#include "cuda_ipc_md.h"
#include "cuda_ipc_ep.h"
#include <ucs/type/class.h>
#include <ucs/sys/string.h>


static ucs_config_field_t uct_cuda_ipc_iface_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_cuda_ipc_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},
    {"MAX_POLL", "16",
     "Max number of event completions to pick during cuda events polling",
      ucs_offsetof(uct_cuda_ipc_iface_config_t, max_poll), UCS_CONFIG_TYPE_UINT},
    {NULL}
};


/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_ipc_iface_t)(uct_iface_t*);


static uint64_t uct_cuda_ipc_iface_node_guid(uct_base_iface_t *iface)
{
    return ucs_machine_guid() * ucs_string_to_id(iface->md->component->name);
}

ucs_status_t uct_cuda_ipc_iface_get_device_address(uct_iface_t *tl_iface,
                                                   uct_device_addr_t *addr)
{
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);

    *(uint64_t*)addr = uct_cuda_ipc_iface_node_guid(iface);
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
    uct_cuda_ipc_iface_t  *iface = ucs_derived_of(tl_iface, uct_cuda_ipc_iface_t);

    return ((uct_cuda_ipc_iface_node_guid(&iface->super) ==
            *((const uint64_t *)dev_addr)) && ((getpid() != *(pid_t *)iface_addr)));
}

static ucs_status_t uct_cuda_ipc_iface_query(uct_iface_h iface,
                                             uct_iface_attr_t *iface_attr)
{
    memset(iface_attr, 0, sizeof(uct_iface_attr_t));
    iface_attr->iface_addr_len          = sizeof(pid_t);
    iface_attr->device_addr_len         = sizeof(uint64_t);
    iface_attr->ep_addr_len             = 0;
    iface_attr->max_conn_priv           = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                          UCT_IFACE_FLAG_PENDING   |
                                          UCT_IFACE_FLAG_GET_ZCOPY |
                                          UCT_IFACE_FLAG_PUT_ZCOPY;
    iface_attr->cap.put.max_short       = 0;
    iface_attr->cap.put.max_bcopy       = 0;
    iface_attr->cap.put.min_zcopy       = 0;
    iface_attr->cap.put.max_zcopy       = UCT_CUDA_IPC_MAX_ALLOC_SZ;
    iface_attr->cap.put.opt_zcopy_align = 1;
    iface_attr->cap.put.align_mtu       = iface_attr->cap.put.opt_zcopy_align;
    iface_attr->cap.put.max_iov         = 1;
    iface_attr->cap.get.max_bcopy       = 0;
    iface_attr->cap.get.min_zcopy       = 0;
    iface_attr->cap.get.max_zcopy       = UCT_CUDA_IPC_MAX_ALLOC_SZ;
    iface_attr->cap.get.opt_zcopy_align = 1;
    iface_attr->cap.get.align_mtu       = iface_attr->cap.get.opt_zcopy_align;
    iface_attr->cap.get.max_iov         = 1;
    iface_attr->latency.overhead        = 1e-9;
    iface_attr->latency.growth          = 0;
    iface_attr->bandwidth               = 6911 * 1024.0 * 1024.0;
    iface_attr->overhead                = 0;
    iface_attr->priority                = 0;
    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_iface_flush(uct_iface_h tl_iface, unsigned flags,
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

static UCS_F_ALWAYS_INLINE unsigned
uct_cuda_ipc_progress_event_queue(ucs_queue_head_t *event_queue, unsigned max_events)
{
    unsigned count = 0;
    uct_cuda_ipc_event_desc_t *cuda_ipc_event;
    ucs_queue_iter_t iter;
    ucs_status_t status;

    ucs_queue_for_each_safe(cuda_ipc_event, iter, event_queue, queue) {
        status = UCT_CUDADRV_FUNC(cuEventQuery(cuda_ipc_event->event));
        if (UCS_INPROGRESS == status) {
            break;
        }
        else if (UCS_OK != status) {
            return status;
        }
        ucs_queue_del_iter(event_queue, iter);
        if (cuda_ipc_event->comp != NULL) {
            uct_invoke_completion(cuda_ipc_event->comp, UCS_OK);
        }
        ucs_trace_poll("CUDA_IPC Event Done :%p", cuda_ipc_event);
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
    unsigned max_events = iface->config.max_poll;
    unsigned count;

    count =  uct_cuda_ipc_progress_event_queue(&iface->outstanding_d2d_event_q, max_events);
    return count;
}

static uct_iface_ops_t uct_cuda_ipc_iface_ops = {
    .ep_get_zcopy             = uct_cuda_ipc_ep_get_zcopy,
    .ep_put_zcopy             = uct_cuda_ipc_ep_put_zcopy,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create_connected      = UCS_CLASS_NEW_FUNC_NAME(uct_cuda_ipc_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_ipc_ep_t),
    .iface_flush              = uct_cuda_ipc_iface_flush,
    .iface_fence              = uct_base_iface_fence, /*TODO*/
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_cuda_ipc_iface_progress,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_ipc_iface_t),
    .iface_query              = uct_cuda_ipc_iface_query,
    .iface_get_device_address = uct_cuda_ipc_iface_get_device_address,
    .iface_get_address        = uct_cuda_ipc_iface_get_address,
    .iface_is_reachable       = uct_cuda_ipc_iface_is_reachable,
};

static void uct_cuda_ipc_event_desc_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_cuda_ipc_event_desc_t *base = (uct_cuda_ipc_event_desc_t *) obj;
    ucs_status_t status;

    memset(base, 0 , sizeof(*base));
    status = UCT_CUDADRV_FUNC(cuEventCreate(&(base->event), CU_EVENT_DISABLE_TIMING));
    if (UCS_OK != status) {
        return;
    }
}

static void uct_cuda_ipc_event_desc_cleanup(ucs_mpool_t *mp, void *obj)
{
    uct_cuda_ipc_event_desc_t *base = (uct_cuda_ipc_event_desc_t *) obj;
    ucs_status_t status;

    status = UCT_CUDADRV_FUNC(cuEventDestroy(base->event));
    if (UCS_OK != status) {
        return;
    }
}

ucs_status_t uct_cuda_ipc_iface_init_streams(uct_cuda_ipc_iface_t *iface)
{
    ucs_status_t status;
    int i;

    for (i = 0; i < iface->device_count; i++) {
        status = UCT_CUDADRV_FUNC(cuStreamCreate(&iface->stream_d2d[i], CU_STREAM_NON_BLOCKING));
        if (UCS_OK != status) {
            return status;
        }
    }
    iface->streams_initialized = 1;

    return UCS_OK;
}

static ucs_mpool_ops_t uct_cuda_ipc_event_desc_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = uct_cuda_ipc_event_desc_init,
    .obj_cleanup   = uct_cuda_ipc_event_desc_cleanup,
};

static UCS_CLASS_INIT_FUNC(uct_cuda_ipc_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_cuda_ipc_iface_config_t *config = NULL;
    ucs_status_t status;
    int dev_count;
    int i, j;

    config = ucs_derived_of(tl_config, uct_cuda_ipc_iface_config_t);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_cuda_ipc_iface_ops, md, worker,
                              params, tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(UCT_CUDA_IPC_TL_NAME));
    if (strncmp(params->mode.device.dev_name,
                UCT_CUDA_IPC_DEV_NAME, strlen(UCT_CUDA_IPC_DEV_NAME)) != 0) {
        ucs_error("No device was found: %s", params->mode.device.dev_name);
        return UCS_ERR_NO_DEVICE;
    }
    for (i = 0; i < UCT_CUDA_IPC_MAX_PEERS; i++) {
        for (j = 0; j < UCT_CUDA_IPC_MAX_PEERS; j++) {
            self->p2p_map[i][j] = -1;
        }
    }
    status = UCT_CUDADRV_FUNC(cuDeviceGetCount(&dev_count));
    if (UCS_OK != status) {
        return status;
    }
    ucs_assert(dev_count <= UCT_CUDA_IPC_MAX_PEERS);

    self->device_count = dev_count;
    for (i = 0; i < dev_count; i++) {
        for (j = 0; j < dev_count; j++) {
            UCT_CUDADRV_FUNC(cuDeviceCanAccessPeer(&(self->p2p_map[i][j]),
                                                   (CUdevice) i, (CUdevice) j));
            if (UCS_OK != status) {
                return status;
            }
        }
    }
    ucs_trace("cuda_ipc p2p map generated for %d devices", dev_count);
    self->config.max_poll = config->max_poll;
    status = ucs_mpool_init(&self->event_desc,
                            0,
                            sizeof(uct_cuda_ipc_event_desc_t),
                            0,
                            UCS_SYS_CACHE_LINE_SIZE,
                            128,
                            1024,
                            &uct_cuda_ipc_event_desc_mpool_ops,
                            "CUDA_IPC EVENT objects");
    if (UCS_OK != status) {
        ucs_error("mpool creation failed");
        return UCS_ERR_IO_ERROR;
    }
    self->streams_initialized = 0;
    ucs_queue_head_init(&self->outstanding_d2d_event_q);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_ipc_iface_t)
{
    ucs_status_t status;
    int i;

    if (1 == self->streams_initialized) {
        for (i = 0; i < self->device_count; i++) {
            status = UCT_CUDADRV_FUNC(cuStreamDestroy(self->stream_d2d[i]));
            if (UCS_OK != status) {
                return;
            }
        }
        self->streams_initialized = 0;
    }
    uct_base_iface_progress_disable(&self->super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    ucs_mpool_cleanup(&self->event_desc, 1);
}

UCS_CLASS_DEFINE(uct_cuda_ipc_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_cuda_ipc_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                          const uct_iface_params_t*, const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_cuda_ipc_iface_t, uct_iface_t);


static ucs_status_t uct_cuda_ipc_query_tl_resources(uct_md_h md,
                                                    uct_tl_resource_desc_t **resource_p,
                                                    unsigned *num_resources_p)
{
    uct_tl_resource_desc_t *resource;

    resource = ucs_calloc(1, sizeof(uct_tl_resource_desc_t), "resource desc");
    if (NULL == resource) {
        ucs_error("Failed to allocate memory");
        return UCS_ERR_NO_MEMORY;
    }
    ucs_snprintf_zero(resource->tl_name, sizeof(resource->tl_name), "%s",
                      UCT_CUDA_IPC_TL_NAME);
    ucs_snprintf_zero(resource->dev_name, sizeof(resource->dev_name), "%s",
                      UCT_CUDA_IPC_DEV_NAME);
    resource->dev_type = UCT_DEVICE_TYPE_ACC;
    *num_resources_p = 1;
    *resource_p      = resource;
    return UCS_OK;
}

UCT_TL_COMPONENT_DEFINE(uct_cuda_ipc_tl, uct_cuda_ipc_query_tl_resources,
                        uct_cuda_ipc_iface_t, UCT_CUDA_IPC_TL_NAME,
                        "CUDA_IPC_", uct_cuda_ipc_iface_config_table,
                        uct_cuda_ipc_iface_config_t);
UCT_MD_REGISTER_TL(&uct_cuda_ipc_md_component, &uct_cuda_ipc_tl);
