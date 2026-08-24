/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "topo_groups.h"

#include <ucs/debug/log.h>


static const char *ucs_topo_groups_type_str(ucs_topo_groups_type_t type)
{
    switch (type) {
    case UCS_TOPO_GROUPS_TYPE_UNKNOWN:
        return "unknown";
    case UCS_TOPO_GROUPS_TYPE_VERA_RUBIN:
        return "vera-rubin";
    default:
        return "<invalid>";
    }
}

ucs_status_t
ucs_topo_init_groups_inner(const ucs_topo_sys_device_info_t *devices,
                           unsigned num_devices, ucs_topo_groups_t *groups_p)
{
    (void)devices;
    (void)num_devices;

    /* TODO: Implement VR groups initialization */

    groups_p->type = UCS_TOPO_GROUPS_TYPE_UNKNOWN;
    ucs_array_init_dynamic(&groups_p->groups);

    ucs_debug("initialized topo groups of type %s with %zu groups",
              ucs_topo_groups_type_str(groups_p->type),
              ucs_array_length(&groups_p->groups));

    return UCS_OK;
}
