/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gdaki_iface.h"
#include "gdaki_ep.h"

#include <ucs/time/time.h>
#include <ucs/datastruct/string_buffer.h>
#include <uct/ib/mlx5/rc/rc_mlx5.h>

#include <doca_log.h>

typedef struct {
    uct_rc_iface_common_config_t super;
    uct_rc_mlx5_iface_common_config_t mlx5;
} uct_gdaki_iface_config_t;

static int
uct_gdaki_iface_is_reachable_v2(const uct_iface_h iface,
                               const uct_iface_is_reachable_params_t *params)
{
    // TODO implement
    const uct_device_addr_t *dev_addr = params->device_addr;

    if (dev_addr == NULL) {
        return 0;
    }

    return 1;
}

static ucs_status_t uct_gdaki_iface_get_address(uct_iface_h tl_iface,
                                                uct_iface_addr_t *addr)
{
    return UCS_OK;
}

static ucs_status_t uct_gdaki_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_gdaki_iface_t *iface = ucs_derived_of(tl_iface, uct_gdaki_iface_t);
    ucs_status_t status;

    status = uct_ib_iface_query(&iface->super, 0, iface_attr);
    if (status != UCS_OK) {
        return status;
    }

    /* TODO:
     *  - PENDING and PUT_ZCOPY are needed to establish rma_bw lanes
     *  - As this lane does not really support PUT_ZCOPY and PENDING, this could be
     *    causing issue when trying to send standard PUT. Eventually we must proably
     *    introduce another type of lane (rma_batch#x).
     */
    iface_attr->cap.flags = UCT_IFACE_FLAG_PUT_ZCOPY |
                            UCT_IFACE_FLAG_PUT_BATCH |
                            UCT_IFACE_FLAG_PENDING |
                            UCT_IFACE_FLAG_CONNECT_TO_EP;
    iface_attr->ep_addr_len    = sizeof(uct_rc_mlx5_base_ep_address_t);
    iface_attr->iface_addr_len = sizeof(uint8_t);
    iface_attr->overhead = UCT_RC_MLX5_IFACE_OVERHEAD;

    iface_attr->cap.put.min_zcopy = 0;
    iface_attr->cap.put.max_zcopy = uct_ib_iface_port_attr(&iface->super)->max_msg_sz;
    return UCS_OK;
}

static UCS_CLASS_DECLARE_NEW_FUNC(uct_gdaki_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_gdaki_iface_t, uct_iface_t);


static uct_ib_iface_ops_t uct_gdaki_internal_ops = {
    .super = {
        .iface_estimate_perf   = uct_ib_iface_estimate_perf,
        .iface_vfs_refresh     = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
        .ep_query              = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
        .ep_invalidate         = (uct_ep_invalidate_func_t)ucs_empty_function_return_unsupported,
        .ep_connect_to_ep_v2   = uct_gdaki_ep_connect_to_ep_v2,
        .iface_is_reachable_v2 = uct_gdaki_iface_is_reachable_v2,
        .ep_is_connected       = uct_gdaki_base_ep_is_connected,
    },
    .create_cq = (uct_ib_iface_create_cq_func_t)ucs_empty_function_return_success,
    .destroy_cq = (uct_ib_iface_destroy_cq_func_t)ucs_empty_function_return_success,
};

static void uct_gdaki_ep_pending_purge(uct_ep_h ep_h, uct_pending_purge_callback_t cb,
                                       void *arg)
{
}

static uct_iface_ops_t uct_gdaki_iface_tl_ops = {
    .ep_batch_prepare         = uct_gdaki_ep_batch_prepare,
    .ep_batch_release         = uct_gdaki_ep_batch_release,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_gdaki_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_gdaki_ep_t),
    .ep_export_dev            = uct_gdaki_ep_export_dev,
    .ep_get_address           = uct_gdaki_ep_get_address,
    .ep_connect_to_ep         = uct_base_ep_connect_to_ep,
    .ep_pending_purge         = uct_gdaki_ep_pending_purge,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_gdaki_iface_t),
    .iface_query              = uct_gdaki_iface_query,
    .iface_get_address        = uct_gdaki_iface_get_address,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_base_iface_is_reachable,
    .iface_flush              = (uct_iface_flush_func_t)ucs_empty_function_return_success,
    .iface_fence              = (uct_iface_fence_func_t)ucs_empty_function_return_unsupported,
    .iface_progress_enable    = (uct_iface_progress_enable_func_t)ucs_empty_function_return_unsupported,
    .iface_progress_disable   = (uct_iface_progress_disable_func_t)ucs_empty_function_return_unsupported,
    .iface_progress           = (uct_iface_progress_func_t)ucs_empty_function_return_unsupported,
};

static int uct_gdaki_alloc(uct_alloc_t *self, void **ptr, size_t boundary,
                           size_t size, int *dma_fd, const char *name)
{
    uct_gdaki_iface_t *iface = ucs_container_of(self, uct_gdaki_iface_t, alloc);
    doca_error_t derr;

    derr = doca_gpu_mem_alloc(iface->gpu_dev, size, boundary, DOCA_GPU_MEM_TYPE_GPU,
                              ptr, NULL);
    if (derr != DOCA_SUCCESS) {
        ucs_error("doca_gpu_mem_alloc failed: %s", doca_error_get_descr(derr));
        return UCS_ERR_IO_ERROR;
    }

    if (dma_fd) {
#if 0 // TODO
        derr = doca_gpu_dmabuf_fd(iface->gpu_dev, *ptr, size, dma_fd);
        if (derr != DOCA_SUCCESS) {
            ucs_error("doca_gpu_dmabuf_fd failed: %s", doca_error_get_descr(derr));
            return UCS_ERR_IO_ERROR;
        }
#else
        *dma_fd = UCT_DMABUF_FD_INVALID;
#endif
    }

    return UCS_OK;
}

static void uct_gdaki_free(uct_alloc_t *self, void *ptr) {
    uct_gdaki_iface_t *iface = ucs_container_of(self, uct_gdaki_iface_t, alloc);

    doca_gpu_mem_free(iface->gpu_dev, ptr);
}

static UCS_CLASS_INIT_FUNC(uct_gdaki_iface_t, uct_md_h tl_md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_gdaki_iface_config_t *config = ucs_derived_of(tl_config,
                                                      uct_gdaki_iface_config_t);
    uct_ib_mlx5_md_t *md = ucs_derived_of(tl_md, uct_ib_mlx5_md_t);
    uct_ib_iface_init_attr_t init_attr;
    ucs_status_t status;
    doca_error_t derr;
    int cuda_id;
    char gpu_name[UCS_SYS_BDF_NAME_MAX];

    status = uct_rc_mlx5_dp_ordering_ooo_init(md, &self->rc_cfg,
                                              md->dp_ordering_cap.dc,
                                              &config->mlx5, "gdaki");
    if (status != UCS_OK) {
        return status;
    }

    init_attr.seg_size = config->super.super.seg_size;
    init_attr.port_num = 1;
    init_attr.qp_type = IBV_QPT_RC;
    UCS_CLASS_CALL_SUPER_INIT(uct_ib_iface_t, &uct_gdaki_iface_tl_ops,
                              &uct_gdaki_internal_ops,
                              tl_md, worker, params, &config->super.super, &init_attr);

    self->alloc.alloc = uct_gdaki_alloc;
    self->alloc.free = uct_gdaki_free;

    status = uct_set_rc_cfg(&self->rc_cfg, &config->super, &init_attr, &md->super, 64);
    if (status != UCS_OK) {
        goto err;
    }

    self->rc_cfg.max_rd_atomic = 0; // TODO init with uct_rc_iface_init_max_rd_atomic

    status = uct_ib_mlx5_iface_select_sl(&self->super,
                                         &config->mlx5.super,
                                         &config->super.super);
    if (status != UCS_OK) {
        goto err;
    }

    if (memcmp(params->mode.device.dev_name, UCT_DEV_CUDA_NAME,
               UCT_DEV_CUDA_NAME_LEN)) {
        ucs_error("wrong device name: %s\n", params->mode.device.dev_name);
        goto err;
    }

    cuda_id = atoi(params->mode.device.dev_name + UCT_DEV_CUDA_NAME_LEN);
    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetPCIBusId(
                    gpu_name, UCS_SYS_BDF_NAME_MAX, cuda_id));
    if (status != UCS_OK) {
        goto err;
    }

    derr = doca_gpu_create(gpu_name, &self->gpu_dev); // TODO pass global gdrcopy handle
    if (derr != DOCA_SUCCESS) {
        status = UCS_ERR_IO_ERROR;
        ucs_error("doca_gpu_create failed: %s %s", doca_error_get_descr(derr),
                  gpu_name);
        goto err;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGet(&self->cuda_dev, cuda_id));
    if (status != UCS_OK) {
        goto err_doca;
    }

    return UCS_OK;

err_doca:
    doca_gpu_destroy(self->gpu_dev);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_gdaki_iface_t)
{
    doca_gpu_destroy(self->gpu_dev);
}

ucs_config_field_t uct_gdaki_iface_config_table[] = {
  {UCT_IB_CONFIG_PREFIX, "", NULL,
   ucs_offsetof(uct_gdaki_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_rc_iface_common_config_table)},

  {UCT_IB_CONFIG_PREFIX, "", NULL,
   ucs_offsetof(uct_gdaki_iface_config_t, mlx5),
   UCS_CONFIG_TYPE_TABLE(uct_ib_mlx5_iface_config_table)},

  {NULL}
};

UCS_CLASS_DEFINE(uct_gdaki_iface_t, uct_ib_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_gdaki_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_gdaki_iface_t, uct_iface_t);

static ucs_status_t
uct_gdaki_query_tl_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                           unsigned *num_tl_devices_p)

{
    uct_ib_md_t *ib_md = ucs_derived_of(md, uct_ib_md_t);
    uct_tl_device_resource_t *tl_devices;
    unsigned num_tl_devices = 0;
    ucs_status_t status;
    CUdevice device;
    ucs_sys_device_t dev;
    ucs_sys_dev_distance_t dist;
    int num_gpus;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetCount(&num_gpus));
    if (status != UCS_OK) {
        return status;
    }

    tl_devices = ucs_malloc(sizeof(*tl_devices) * num_gpus, "gdaki_tl_devices");
    if (tl_devices == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    for (int i = 0; i < num_gpus; i++) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGet(&device, i));
        if (status != UCS_OK) {
            goto err;
        }

        uct_cuda_base_get_sys_dev(device, &dev);
        status = ucs_topo_get_distance(dev, ib_md->dev.sys_dev, &dist);
        if (status != UCS_OK) {
            goto err;
        }

        if (dist.latency > 300.0/UCS_NSEC_PER_SEC) {
            continue;
        }

        snprintf(tl_devices[num_tl_devices].name,
                 sizeof(tl_devices[num_tl_devices].name),
                 UCT_DEV_CUDA_NAME "%d-%s", device,
                 uct_ib_device_name(&ib_md->dev));
        tl_devices[num_tl_devices].type = UCT_DEVICE_TYPE_NET;
        tl_devices[num_tl_devices].sys_device = dev;
        num_tl_devices++;
    }

    *num_tl_devices_p = num_tl_devices;
    *tl_devices_p     = tl_devices;
    return UCS_OK;

err:
    ucs_free(tl_devices);
    return status;
}

UCT_TL_DEFINE_ENTRY(&uct_ib_component, gdaki, uct_gdaki_query_tl_devices,
                    uct_gdaki_iface_t, "GDAKI_", uct_gdaki_iface_config_table,
                    uct_gdaki_iface_config_t);

void doca_init(void) {
    struct doca_log_backend *sdk_log;
    doca_error_t derr;

    derr = doca_log_backend_create_standard();
    if (derr != DOCA_SUCCESS) {
        printf("%s:%d \n", __func__, __LINE__);
        return;
    }

    derr = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
    if (derr != DOCA_SUCCESS) {
        printf("%s:%d \n", __func__, __LINE__);
        return;
    }

    derr = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_ERROR);
    if (derr != DOCA_SUCCESS) {
        printf("%s:%d \n", __func__, __LINE__);
        return;
    }
}

UCT_TL_INIT(&uct_ib_component, gdaki, ctor, doca_init(),)


