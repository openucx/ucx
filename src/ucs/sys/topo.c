/**
* Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/sys/topo.h>
#include <ucs/type/status.h>
#include <stdio.h>

ucs_status_t ucs_topo_find_device_by_bus_id(const ucs_sys_bus_id_t *bus_id,
                                            const ucs_sys_device_t **sys_dev)
{
    return UCS_OK;
}


ucs_status_t ucs_topo_get_distance(const ucs_sys_device_t *device1,
                                   const ucs_sys_device_t *device2,
                                   ucs_sys_dev_distance_t *distance)
{
    return UCS_OK;
}


void ucs_topo_print_info(FILE *stream)
{
}
