/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "vr.h"
#include "vr_inventory.h"

#include <ucs/debug/log.h>


ucs_status_t ucs_topo_vr_init_groups(const ucs_topo_sys_device_info_t *devices,
                                     unsigned num_devices,
                                     ucs_topo_group_t **groups_p,
                                     size_t *num_groups_p)
{
    ucs_topo_vr_inventory_t inventory;
    ucs_status_t status;

    status = ucs_topo_vr_inventory_build(devices, num_devices, &inventory);
    if (status != UCS_OK) {
        return status;
    }

    ucs_debug("built vera-rubin inventory with %zu gpus and %zu cx9 functions",
              (size_t)ucs_array_length(&inventory.gpus),
              (size_t)ucs_array_length(&inventory.cx9_ports));

    /* TODO: Build groups from inventory. */

    ucs_topo_vr_inventory_cleanup(&inventory);

    *groups_p     = NULL;
    *num_groups_p = 0;

    return UCS_OK;
}
