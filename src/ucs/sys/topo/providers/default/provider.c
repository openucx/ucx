/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucs/sys/topo/base/topo.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>

static ucs_status_t
ucs_topo_get_distance_default(ucs_sys_device_t device1,
                              ucs_sys_device_t device2,
                              ucs_sys_dev_distance_t *distance)
{
    *distance = ucs_topo_default_distance;

    return UCS_OK;
}

static ucs_sys_topo_method_t ucs_sys_topo_default_method = {
    .name         = "default",
    .get_distance = ucs_topo_get_distance_default,
};

void UCS_F_CTOR ucs_topo_default_init()
{
    ucs_topo_register_provider(&ucs_sys_topo_default_method);
}

void UCS_F_DTOR ucs_topo_default_cleanup()
{
    ucs_topo_unregister_provider(&ucs_sys_topo_default_method);
}
