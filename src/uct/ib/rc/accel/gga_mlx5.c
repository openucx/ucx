/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/base/uct_iface.h>
#include <uct/ib/base/ib_md.h>
#include <uct/ib/rc/accel/rc_mlx5.h>


/**
 * GGA mlx5 interface configuration
 */
typedef struct uct_gga_mlx5_iface_config {
    uct_rc_iface_config_t             super;
    uct_rc_mlx5_iface_common_config_t rc_mlx5_common;
} uct_gga_mlx5_iface_config_t;


ucs_config_field_t uct_gga_mlx5_iface_config_table[] = {
  {"GGA_", "", NULL,
   ucs_offsetof(uct_gga_mlx5_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_rc_iface_config_table)},

  {"GGA_", "", NULL,
   ucs_offsetof(uct_gga_mlx5_iface_config_t, rc_mlx5_common),
   UCS_CONFIG_TYPE_TABLE(uct_rc_mlx5_common_config_table)},

  {NULL}
};


typedef struct {
    uct_rc_mlx5_iface_common_t  super;
} uct_gga_mlx5_iface_t;


static UCS_CLASS_INIT_FUNC(uct_gga_mlx5_iface_t,
                           uct_md_h tl_md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

static UCS_CLASS_CLEANUP_FUNC(uct_gga_mlx5_iface_t)
{
    ucs_fatal("gga_transport is not implemented yet");
}

UCS_CLASS_DEFINE(uct_gga_mlx5_iface_t, uct_rc_mlx5_iface_common_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_gga_mlx5_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static ucs_status_t
uct_gga_mlx5_query_tl_devices(uct_md_h md,
                              uct_tl_device_resource_t **tl_devices_p,
                              unsigned *num_tl_devices_p)
{
    uct_ib_mlx5_md_t *mlx5_md = ucs_derived_of(md, uct_ib_mlx5_md_t);
    uct_tl_device_resource_t *tl_devices;
    unsigned num_tl_devices;
    ucs_status_t status;

    if (strcmp(mlx5_md->super.name, UCT_IB_MD_NAME(mlx5)) ||
        !ucs_test_all_flags(mlx5_md->flags, UCT_IB_MLX5_MD_FLAG_DEVX           |
                                            UCT_IB_MLX5_MD_FLAG_INDIRECT_XGVMI |
                                            UCT_IB_MLX5_MD_FLAG_MMO_DMA)) {
        return UCS_ERR_NO_DEVICE;
    }

    status = uct_ib_device_query_ports(&mlx5_md->super.dev,
                                       UCT_IB_DEVICE_FLAG_MLX5_PRM, &tl_devices,
                                       &num_tl_devices);
    if (status != UCS_OK) {
        return status;
    }

    /* TODO: del to enable GGA in UCP */
    ucs_free(tl_devices);
    return UCS_ERR_NO_DEVICE;
}

UCT_TL_DEFINE_ENTRY(&uct_ib_component, gga_mlx5, uct_gga_mlx5_query_tl_devices,
                    uct_gga_mlx5_iface_t, "GGA_MLX5_",
                    uct_gga_mlx5_iface_config_table,
                    uct_gga_mlx5_iface_config_t);
