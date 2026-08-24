/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "topo_groups.h"

#include <ucs/debug/memtrack_int.h>
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
                           unsigned num_devices,
                           const ucs_topo_groups_t **groups_p)
{
    ucs_topo_groups_t *groups;

    (void)devices;
    (void)num_devices;

    groups = ucs_malloc(sizeof(*groups), "topo_groups");
    if (groups == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    groups->type       = UCS_TOPO_GROUPS_TYPE_UNKNOWN;
    groups->groups     = NULL;
    groups->num_groups = 0;

    /* TODO: Implement VR groups initialization */

    ucs_debug("initialized topo groups of type %s with %zu groups",
              ucs_topo_groups_type_str(groups->type), groups->num_groups);

    *groups_p = groups;

    return UCS_OK;
}
