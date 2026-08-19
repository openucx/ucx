/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_TOPO_VR_H
#define UCS_TOPO_VR_H

#include <ucs/datastruct/array.h>
#include <ucs/sys/topo/base/topo_int.h>

BEGIN_C_DECLS


UCS_ARRAY_DECLARE_TYPE(ucs_topo_vr_sys_dev_array_t, size_t, ucs_sys_device_t);


typedef struct {
    ucs_topo_vr_sys_dev_array_t gpus;
    ucs_topo_vr_sys_dev_array_t cx9_ports;
} ucs_topo_vr_inventory_t;


ucs_status_t ucs_topo_vr_init_groups(const ucs_topo_sys_device_info_t *devices,
                                     unsigned num_devices,
                                     ucs_topo_group_t **groups_p,
                                     size_t *num_groups_p);

END_C_DECLS

#endif
