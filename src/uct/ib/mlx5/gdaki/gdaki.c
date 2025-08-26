/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gdaki.h"

#include <ucs/time/time.h>
#include <ucs/datastruct/string_buffer.h>
#include <uct/ib/mlx5/rc/rc_mlx5.h>
#include <uct/cuda/base/cuda_iface.h>

#include <doca_log.h>

typedef struct {
    uct_rc_iface_common_config_t      super;
    uct_rc_mlx5_iface_common_config_t mlx5;
} uct_rc_gdaki_iface_config_t;

ucs_config_field_t uct_rc_gdaki_iface_config_table[] = {
  {UCT_IB_CONFIG_PREFIX, "", NULL,
   ucs_offsetof(uct_rc_gdaki_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_rc_iface_common_config_table)},

  {UCT_IB_CONFIG_PREFIX, "", NULL,
   ucs_offsetof(uct_rc_gdaki_iface_config_t, mlx5),
   UCS_CONFIG_TYPE_TABLE(uct_rc_mlx5_common_config_table)},

  {NULL}
};

static UCS_CLASS_INIT_FUNC(uct_rc_gdaki_iface_t, uct_md_h tl_md,
                           uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_gdaki_iface_t)
{
    doca_gpu_destroy(self->gpu_dev);
}

UCS_CLASS_DEFINE(uct_rc_gdaki_iface_t, uct_rc_mlx5_iface_common_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_gdaki_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static ucs_status_t
uct_gdaki_query_tl_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                           unsigned *num_tl_devices_p)
{
    uct_ib_md_t *ib_md      = ucs_derived_of(md, uct_ib_md_t);
    unsigned num_tl_devices = 0;
    uct_tl_device_resource_t *tl_devices;
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

        /* TODO this logic should be done in UCP */
        if (dist.latency > 300.0 / UCS_NSEC_PER_SEC) {
            continue;
        }

        snprintf(tl_devices[num_tl_devices].name,
                 sizeof(tl_devices[num_tl_devices].name), "cuda%d-%s", device,
                 uct_ib_device_name(&ib_md->dev));
        tl_devices[num_tl_devices].type       = UCT_DEVICE_TYPE_NET;
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
                    uct_rc_gdaki_iface_t, "GDAKI_",
                    uct_rc_gdaki_iface_config_table,
                    uct_rc_gdaki_iface_config_t);

static void uct_ib_doca_init(void)
{
    struct doca_log_backend *sdk_log;
    doca_error_t derr;

    derr = doca_log_backend_create_standard();
    if (derr != DOCA_SUCCESS) {
        ucs_error("doca_log_backend_create_standard failed: %d\n", derr);
        return;
    }

    derr = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
    if (derr != DOCA_SUCCESS) {
        ucs_error("doca_log_backend_create_with_file_sdk failed: %d\n", derr);
        return;
    }

    derr = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_ERROR);
    if (derr != DOCA_SUCCESS) {
        ucs_error("doca_log_backend_set_sdk_level failed: %d\n", derr);
        return;
    }
}

UCT_TL_INIT(&uct_ib_component, gdaki, ctor, uct_ib_doca_init(), )
