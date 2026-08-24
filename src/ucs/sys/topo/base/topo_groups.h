/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_TOPO_GROUPS_H
#define UCS_TOPO_GROUPS_H

#include "topo_int.h"

BEGIN_C_DECLS

ucs_status_t
ucs_topo_init_groups_inner(const ucs_topo_sys_device_info_t *devices,
                           unsigned num_devices, ucs_topo_groups_t *groups_p);

END_C_DECLS

#endif
