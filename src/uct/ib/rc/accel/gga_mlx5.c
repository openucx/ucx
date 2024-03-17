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


typedef struct {                        
    uct_ib_md_packed_mkey_t packed_mkey;
    uct_ib_mlx5_devx_mem_t  *memh;      
    uct_rkey_bundle_t       rkey_ob;    
} uct_gga_mlx5_rkey_handle_t;           


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

ucs_status_t uct_ib_mlx5_gga_md_open(struct ibv_device *ibv_device,
                                     const uct_ib_md_config_t *md_config,
                                     struct uct_ib_md **md_p)
{
    ucs_status_t status;

    status = uct_ib_mlx5_devx_md_open(ibv_device, md_config, md_p);
    if (status != UCS_OK) {
        return status;
    }

    (*md_p)->name = UCT_IB_MD_NAME(gga);
    return UCS_OK;
}

static ucs_status_t
uct_ib_mlx5_gga_mkey_pack(uct_md_h uct_md, uct_mem_h uct_memh,
                          void *address, size_t length,
                          const uct_md_mkey_pack_params_t *params,
                          void *mkey_buffer)
{
    uct_md_mkey_pack_params_t gga_params = *params;
    ucs_status_t status;
    uct_ib_md_packed_mkey_t *mkey;

    gga_params.field_mask |= UCT_MD_MKEY_PACK_FIELD_FLAGS;
    gga_params.flags      |= UCT_MD_MKEY_PACK_FLAG_EXPORT;

    status = uct_ib_mlx5_devx_mkey_pack(uct_md, uct_memh, address, length,
                                        &gga_params, mkey_buffer);
    if (status != UCS_OK) {
        return status;
    }

    mkey = (uct_ib_md_packed_mkey_t*)mkey_buffer;
    ucs_assert(mkey->flags & UCT_IB_PACKED_MKEY_FLAG_EXPORTED);
    mkey->flags |= UCT_IB_PACKED_MKEY_FLAG_GGA;

    return UCS_OK;
}

ucs_status_t
uct_ib_mlx5_gga_rkey_unpack(const uct_ib_md_packed_mkey_t *mkey,
                            uct_rkey_t *rkey_p, void **handle_p)
{
    uct_rkey_bundle_t *rkey_bundle;
    uct_gga_mlx5_rkey_handle_t *rkey_handle;

    ucs_assert(mkey->flags & UCT_IB_PACKED_MKEY_FLAG_EXPORTED);

    rkey_bundle = ucs_malloc(sizeof(*rkey_bundle), "gga_rkey_bundle");
    if (rkey_bundle == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    rkey_handle = ucs_malloc(sizeof(*rkey_handle), "gga_rkey_handle");
    if (rkey_handle == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    rkey_handle->packed_mkey    = *mkey;
    /* memh and rkey_ob will be initialized on demand */
    rkey_handle->memh           = NULL;
    rkey_handle->rkey_ob.rkey   = UCT_INVALID_RKEY;
    rkey_handle->rkey_ob.handle = NULL;
    rkey_handle->rkey_ob.type   = NULL;

    rkey_bundle->handle      = rkey_handle;
    rkey_bundle->rkey        = (uintptr_t)rkey_bundle;
    rkey_bundle->type        = NULL;

    *rkey_p   = rkey_bundle->rkey;
    *handle_p = rkey_bundle->handle;
    return UCS_OK;
}

static uct_ib_md_ops_t uct_ib_mlx5_gga_md_ops = {
    .super = {
        .close              = uct_ib_mlx5_devx_md_close,
        .query              = uct_ib_mlx5_devx_md_query,
        .mem_alloc          = uct_ib_mlx5_devx_device_mem_alloc,
        .mem_free           = uct_ib_mlx5_devx_device_mem_free,
        .mem_reg            = uct_ib_mlx5_devx_mem_reg,
        .mem_dereg          = uct_ib_mlx5_devx_mem_dereg,
        .mem_attach         = uct_ib_mlx5_devx_mem_attach,
        .mem_advise         = uct_ib_mem_advise,
        .mkey_pack          = uct_ib_mlx5_gga_mkey_pack,
        .detect_memory_type = ucs_empty_function_return_unsupported,
    },
    .open = uct_ib_mlx5_gga_md_open,
};

UCT_IB_MD_DEFINE_ENTRY(gga, uct_ib_mlx5_gga_md_ops);

static ucs_status_t
uct_gga_mlx5_query_tl_devices(uct_md_h md,
                              uct_tl_device_resource_t **tl_devices_p,
                              unsigned *num_tl_devices_p)
{
    uct_ib_mlx5_md_t *mlx5_md = ucs_derived_of(md, uct_ib_mlx5_md_t);
    ucs_status_t status;

    if (strcmp(mlx5_md->super.name, UCT_IB_MD_NAME(mlx5)) ||
        !ucs_test_all_flags(mlx5_md->flags, UCT_IB_MLX5_MD_FLAG_DEVX           |
                                            UCT_IB_MLX5_MD_FLAG_INDIRECT_XGVMI |
                                            UCT_IB_MLX5_MD_FLAG_MMO_DMA)) {
        return UCS_ERR_NO_DEVICE;
    }

    status = uct_ib_device_query_ports(&mlx5_md->super.dev,
                                       UCT_IB_DEVICE_FLAG_MLX5_PRM, tl_devices_p,
                                       num_tl_devices_p);
    if (status != UCS_OK) {
        return status;
    }

    /* TODO: del to enable GGA in UCP */
    return UCS_ERR_NO_DEVICE;
}

UCT_TL_DEFINE_ENTRY(&uct_ib_component, gga_mlx5, uct_gga_mlx5_query_tl_devices,
                    uct_gga_mlx5_iface_t, "GGA_MLX5_",
                    uct_gga_mlx5_iface_config_table,
                    uct_gga_mlx5_iface_config_t);
