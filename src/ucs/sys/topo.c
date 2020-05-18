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
#include <ucs/debug/log.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>

/* populate with sys fs path and prefix of directory name */
ucs_sys_dev_path_spec_t ucs_sys_dev_specs[] = {
    {"/sys/devices/system/node", "node"},
    {"/sys/bus/pci/drivers/nvidia", "0000"},
    {"/sys/bus/pci/drivers/mlx5_core", "0000"},
    {NULL, NULL},
};

static void ucs_add_bus_id(ucs_sys_device_t *sys_dev, char *name)
{
    char delim[] = {":."};
    char *str1   = name;
    int j        = 1;
    char *token;
    uint16_t uint_vals[4];

    do {
	token = strtok(str1, delim);
	if (token != NULL) {
            uint_vals[j - 1] = (uint16_t) strtoul(token, NULL, 16);
	    j++;
	    str1 = NULL;
	}
    } while (token != NULL);

    ucs_assert(j == 5);

    sys_dev->bus_id.domain   = (uint16_t) uint_vals[0];
    sys_dev->bus_id.bus      = (uint8_t) uint_vals[1];
    sys_dev->bus_id.slot     = (uint8_t) uint_vals[2];
    sys_dev->bus_id.function = (uint8_t) uint_vals[3];
}

static ucs_status_t ucs_add_sys_devices(char *dev_loc, char *match,
                                        ucs_global_sys_dev_array_t *sys_dev_array)
{
    struct dirent **namelist;
    int n, idx;
    unsigned *id_ptr;

    n = scandir(dev_loc, &namelist, NULL, alphasort);
    if (n < 0) {
        perror("scandir");
    }
    else {

        id_ptr = &sys_dev_array->num_entries;
        for (idx = 0; idx < n; idx++) {
            if (!strncmp(namelist[idx]->d_name, match, strlen(match))) {
                ucs_assert(strlen(namelist[idx]->d_name) <= PATH_MAX);
                ucs_trace("sys device full name = %s", namelist[idx]->d_name);
                /* update sys device details */
                sys_dev_array->entry[*id_ptr].id = *id_ptr;
                if (!strncmp(namelist[idx]->d_name, "node", strlen("node"))) {
                    /* bus id invalid for numa node */
                    memset(&(sys_dev_array->entry[*id_ptr].bus_id), 0, sizeof(ucs_sys_bus_id_t));
                } else {
                    ucs_add_bus_id(&(sys_dev_array->entry[*id_ptr]), namelist[idx]->d_name);
                }
                ucs_assert((*id_ptr + 1) < UCS_MAX_SYS_DEV_ENTRIES);
                *id_ptr += 1;
            }
            free(namelist[idx]);
        }
        free(namelist);
    }

    return UCS_OK;
}

ucs_global_sys_dev_array_t ucs_global_sys_devices;

void ucs_sys_topo_init()
{
    ucs_sys_dev_path_spec_t *sys_dev_specs = ucs_sys_dev_specs;

    ucs_global_sys_devices.num_entries = 0;

    while (sys_dev_specs->path != NULL) {
        ucs_trace("device spec: %s %s", sys_dev_specs->path, sys_dev_specs->match);
        ucs_add_sys_devices(sys_dev_specs->path, sys_dev_specs->match,
                            &ucs_global_sys_devices);
        sys_dev_specs++;
    }
    ucs_trace("num sys devices found = %d", ucs_global_sys_devices.num_entries);
}

void ucs_sys_topo_cleanup()
{
}

/* TODO: can this conflict with a valid BDF? */
ucs_sys_bus_id_t ucs_sys_bus_id_unknown = { .domain   = 0xffff,
                                            .bus      = 0xff,
                                            .slot     = 0xff,
                                            .function = 0xff
                                          };

ucs_status_t ucs_topo_find_device_by_bus_id(const ucs_sys_bus_id_t *bus_id,
                                            ucs_sys_device_t *sys_dev)
{
    return UCS_OK;
}


ucs_status_t ucs_topo_get_distance(ucs_sys_device_t device1,
                                   ucs_sys_device_t device2,
                                   ucs_sys_dev_distance_t *distance)
{
    return UCS_OK;
}


void ucs_topo_print_info(FILE *stream)
{
    int i;

    for (i = 0; i < ucs_global_sys_devices.num_entries; i++) {
        fprintf(stream, "found system device with bus id = %u\n",
                ucs_global_sys_devices.entry[i].bus_id.bus);
    }
}
