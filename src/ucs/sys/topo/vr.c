/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "vr.h"


ucs_status_t ucs_topo_vr_init_groups(const ucs_topo_sys_device_info_t *devices,
                                     unsigned num_devices,
                                     ucs_topo_group_t **groups_p,
                                     size_t *num_groups_p)
{
    (void)devices;
    (void)num_devices;

    /* TODO: Implement VR groups initialization */

    *groups_p     = NULL;
    *num_groups_p = 0;

    return UCS_OK;
}
