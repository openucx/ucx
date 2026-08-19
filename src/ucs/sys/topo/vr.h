/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_TOPO_VR_H
#define UCS_TOPO_VR_H

#include <ucs/sys/topo/base/topo_int.h>

BEGIN_C_DECLS

ucs_status_t ucs_topo_vr_init_groups(const ucs_topo_sys_device_info_t *devices,
                                     unsigned num_devices,
                                     ucs_topo_group_t **groups_p,
                                     size_t *num_groups_p);

END_C_DECLS

#endif
