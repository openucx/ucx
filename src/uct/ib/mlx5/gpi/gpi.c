/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gpi.h"

#include <ucs/datastruct/string_buffer.h>
#include <ucs/type/init_once.h>
#include <ucs/type/serialize.h>
#include <uct/ib/base/ib_verbs.h>
#include <uct/ib/mlx5/rc/rc_mlx5.h>
#include <uct/cuda/base/cuda_util.h>
#include <uct/cuda/base/cuda_ctx.h>


typedef struct {
    uct_rc_iface_common_config_t      super;
    uct_rc_mlx5_iface_common_config_t mlx5;
} uct_rc_gpi_iface_config_t;


ucs_config_field_t uct_rc_gpi_iface_config_table[] = {
    {UCT_IB_CONFIG_PREFIX, "", NULL,
     ucs_offsetof(uct_rc_gpi_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_rc_iface_common_config_table)},

    {UCT_IB_CONFIG_PREFIX, "", NULL,
     ucs_offsetof(uct_rc_gpi_iface_config_t, mlx5),
     UCS_CONFIG_TYPE_TABLE(uct_rc_mlx5_common_config_table)},

    {NULL}
};


static UCS_CLASS_INIT_FUNC(uct_rc_gpi_ep_t, const uct_ep_params_t *params)
{
    uct_rc_gpi_iface_t *iface = ucs_derived_of(params->iface,
                                               uct_rc_gpi_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super.super.super);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_gpi_ep_t)
{
}

UCS_CLASS_DEFINE(uct_rc_gpi_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_gpi_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_gpi_ep_t, uct_ep_t);


static ucs_status_t uct_rc_gpi_iface_get_address(uct_iface_h tl_iface,
                                                 uct_iface_addr_t *addr)
{
    *(uint8_t*)addr = UCT_RC_MLX5_IFACE_ADDR_TYPE_BASIC;
    return UCS_OK;
}

static ucs_status_t
uct_rc_gpi_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_gpi_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_gpi_iface_t);
    ucs_status_t status;

    status = uct_ib_iface_query(&iface->super.super.super, iface_attr);
    if (status != UCS_OK) {
        return status;
    }

    iface_attr->cap.flags      = UCT_IFACE_FLAG_CONNECT_TO_EP |
                                 UCT_IFACE_FLAG_INTER_NODE;
    iface_attr->iface_addr_len = sizeof(uint8_t);
    iface_attr->ep_addr_len    = 0;
    iface_attr->overhead       = UCT_RC_MLX5_IFACE_OVERHEAD;
    iface_attr->ctl_device     = uct_cuda_get_sys_dev(iface->cuda_dev);
    iface_attr->dev_num_paths  = 1;

    return UCS_OK;
}

static ucs_status_t
uct_rc_gpi_create_cq(uct_ib_iface_t *ib_iface, uct_ib_dir_t dir,
                     const uct_ib_iface_init_attr_t *init_attr,
                     int preferred_cpu, size_t inl)
{
    uct_rc_gpi_iface_t *iface = ucs_derived_of(ib_iface, uct_rc_gpi_iface_t);

    iface->super.cq[dir].type = UCT_IB_MLX5_OBJ_TYPE_NULL;
    return UCS_OK;
}

static UCS_CLASS_DECLARE_NEW_FUNC(uct_rc_gpi_iface_t, uct_iface_t, uct_md_h,
                                  uct_worker_h, const uct_iface_params_t*,
                                  const uct_iface_config_t*);

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_rc_gpi_iface_t, uct_iface_t);

static uct_rc_iface_ops_t uct_rc_gpi_internal_ops = {
    .super = {
        .super = {
            .iface_query_v2         = uct_iface_base_query_v2,
            .iface_estimate_perf    = uct_ib_iface_estimate_perf,
            .iface_vfs_refresh      = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
            .ep_query               = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
            .ep_invalidate          = (uct_ep_invalidate_func_t)ucs_empty_function_return_unsupported,
            .ep_connect_to_ep_v2    = (uct_ep_connect_to_ep_v2_func_t)ucs_empty_function_return_unsupported,
            .iface_is_reachable_v2  = (uct_iface_is_reachable_v2_func_t)ucs_empty_function_return_one_int,
            .ep_is_connected        = (uct_ep_is_connected_func_t)ucs_empty_function_return_zero_int,
            .ep_get_device_ep       = (uct_ep_get_device_ep_func_t)ucs_empty_function_return_unsupported,
        },
        .create_cq  = uct_rc_gpi_create_cq,
        .destroy_cq = (uct_ib_iface_destroy_cq_func_t)ucs_empty_function_return_success,
    },
    .init_rx    = (uct_rc_iface_init_rx_func_t)ucs_empty_function_return_success,
    .cleanup_rx = (uct_rc_iface_cleanup_rx_func_t)
            ucs_empty_function_return_success,
};

static uct_iface_ops_t uct_rc_gpi_iface_tl_ops = {
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_rc_gpi_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_gpi_ep_t),
    .ep_get_address           = (uct_ep_get_address_func_t)
            ucs_empty_function_return_success,
    .ep_connect_to_ep         = uct_base_ep_connect_to_ep,
    .ep_pending_purge         = (uct_ep_pending_purge_func_t)ucs_empty_function,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_gpi_iface_t),
    .iface_query              = uct_rc_gpi_iface_query,
    .iface_get_address        = uct_rc_gpi_iface_get_address,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_base_iface_is_reachable,
    .iface_flush              = (uct_iface_flush_func_t)
            ucs_empty_function_return_success,
    .iface_fence              = (uct_iface_fence_func_t)
            ucs_empty_function_return_unsupported,
    .iface_progress_enable    = (uct_iface_progress_enable_func_t)
            ucs_empty_function_return_unsupported,
    .iface_progress_disable   = (uct_iface_progress_disable_func_t)
            ucs_empty_function_return_unsupported,
    .iface_progress           = (uct_iface_progress_func_t)
            ucs_empty_function_return_unsupported,
};

static UCS_CLASS_INIT_FUNC(uct_rc_gpi_iface_t, uct_md_h tl_md,
                           uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_rc_gpi_iface_config_t *config =
            ucs_derived_of(tl_config, uct_rc_gpi_iface_config_t);
    uct_ib_mlx5_md_t *md = ucs_derived_of(tl_md, uct_ib_mlx5_md_t);
    uct_ib_iface_init_attr_t init_attr = {};
    UCS_STRING_BUFFER_ONSTACK(strb, 64);
    char *gpu_name, *ib_name;
    ucs_status_t status;
    int cuda_id;

    status = uct_rc_mlx5_dp_ordering_ooo_init(md, &self->super,
                                              md->dp_ordering_cap_devx.rc,
                                              md->ddp_support_dv.rc,
                                              &config->mlx5, "rc_gpi");
    if (status != UCS_OK) {
        return status;
    }

    ucs_string_buffer_appendf(&strb, "%s", params->mode.device.dev_name);
    gpu_name = ucs_string_buffer_next_token(&strb, NULL, "-");
    ib_name  = ucs_string_buffer_next_token(&strb, gpu_name, "-");

    init_attr.seg_size      = config->super.super.seg_size;
    init_attr.qp_type       = IBV_QPT_RC;
    init_attr.dev_name      = ib_name;
    init_attr.max_rd_atomic = IBV_DEV_ATTR(&md->super.dev, max_qp_rd_atom);

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_mlx5_iface_common_t,
                              &uct_rc_gpi_iface_tl_ops,
                              &uct_rc_gpi_internal_ops, tl_md, worker, params,
                              &config->super, &config->mlx5, &init_attr);

    if (memcmp(gpu_name, UCT_GPI_DEVICE_CUDA_NAME,
               UCT_GPI_DEVICE_CUDA_NAME_LEN)) {
        ucs_error("wrong device name: %s", gpu_name);
        return UCS_ERR_INVALID_PARAM;
    }

    cuda_id = atoi(gpu_name + UCT_GPI_DEVICE_CUDA_NAME_LEN);
    status  = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGet(&self->cuda_dev, cuda_id));
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuDevicePrimaryCtxRetain(&self->cuda_ctx, self->cuda_dev));
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;

err_cuda_ctx:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(self->cuda_dev));
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_gpi_iface_t)
{
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(self->cuda_dev));
}

UCS_CLASS_DEFINE(uct_rc_gpi_iface_t, uct_rc_mlx5_iface_common_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_gpi_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_gpi_iface_t, uct_iface_t);

static ucs_status_t
uct_rc_gpi_query_tl_devices(uct_md_h tl_md,
                            uct_tl_device_resource_t **tl_devices_p,
                            unsigned *num_tl_devices_p)
{
    uct_ib_mlx5_md_t *ib_mlx5_md = ucs_derived_of(tl_md, uct_ib_mlx5_md_t);
    uct_ib_md_t *ib_md           = &ib_mlx5_md->super;
    uct_tl_device_resource_t *tl_devices;
    int cudadev_count;
    CUdevice cuda_dev;
    ucs_status_t status;
    int i;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetCount(&cudadev_count));
    if ((status != UCS_OK) || (cudadev_count == 0)) {
        return UCS_ERR_NO_DEVICE;
    }

    tl_devices = ucs_malloc(sizeof(*tl_devices) * cudadev_count,
                            "gpi_tl_devices");
    if (tl_devices == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    for (i = 0; i < cudadev_count; ++i) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGet(&cuda_dev, i));
        if (status != UCS_OK) {
            goto err;
        }

        snprintf(tl_devices[i].name, sizeof(tl_devices[i].name), "%s%d-%s:%d",
                 UCT_GPI_DEVICE_CUDA_NAME, cuda_dev,
                 uct_ib_device_name(&ib_md->dev), ib_md->dev.first_port);
        tl_devices[i].type       = UCT_DEVICE_TYPE_NET;
        tl_devices[i].sys_device = ib_md->dev.sys_dev;
    }

    *num_tl_devices_p = cudadev_count;
    *tl_devices_p     = tl_devices;
    return UCS_OK;

err:
    ucs_free(tl_devices);
    return status;
}

UCT_TL_DEFINE_ENTRY(&uct_ib_component, rc_gpi, uct_rc_gpi_query_tl_devices,
                    uct_rc_gpi_iface_t, "RC_GPI_",
                    uct_rc_gpi_iface_config_table,
                    uct_rc_gpi_iface_config_t);

UCT_TL_INIT(&uct_ib_component, rc_gpi, ctor, , )
