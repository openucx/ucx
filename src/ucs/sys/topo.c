/**
* Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif


#include <ucs/sys/checker.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/topo.h>
#include <ucs/debug/log.h>
#include <ucs/time/time.h>
#include <ucm/util/sys.h>

#include <sys/ioctl.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <net/if.h>
#include <dirent.h>
#include <sched.h>
#include <ctype.h>
#ifdef HAVE_SYS_THR_H
#include <sys/thr.h>
#endif

#if HAVE_SYS_CAPABILITY_H
#  include <sys/capability.h>
#endif

#include <numaif.h>
#include <math.h>


static char *ucs_mm_unit_paths[] = {
    [UCS_MM_UNIT_CPU]     = "/sys/devices/system/node",
    [UCS_MM_UNIT_CUDA]    = "/sys/bus/pci/drivers/nvidia",
    [UCS_MM_UNIT_LAST] = NULL
};


static char *ucs_mm_unit_match[] = {
    [UCS_MM_UNIT_CPU]     = "node",
    [UCS_MM_UNIT_CUDA]    = "0000",
    [UCS_MM_UNIT_LAST] = NULL
};


static char *ucs_sys_device_paths[] = {
    [UCS_SYS_DEVICE_IB]      = "/sys/bus/pci/drivers/mlx5_core",
    [UCS_SYS_DEVICE_CUDA]    = "/sys/bus/pci/drivers/nvidia",
    [UCS_SYS_DEVICE_LAST] = NULL
};


static char *ucs_sys_device_match[] = {
    [UCS_SYS_DEVICE_IB]      = "0000",
    [UCS_SYS_DEVICE_CUDA]    = "0000",
    [UCS_SYS_DEVICE_LAST] = NULL
};


static ucs_status_t ucs_get_paths(char *dev_loc, char *match, int *num_devices, char **fpaths)
{
    char *dest;
    struct dirent **namelist;
    int n, idx;

    *num_devices = 0;

    n = scandir(dev_loc, &namelist, NULL, alphasort);
    if (n < 0) {
        perror("scandir");
    }
    else {
        *fpaths = ucs_malloc(sizeof(char) * PATH_MAX * n, "mm_fpaths allocation");
        if (NULL == *fpaths) {
            ucs_error("Failed to allocate memory for mm_fpaths");
            return UCS_ERR_NO_MEMORY;
        }

        for (idx = 0; idx < n; idx++) {
            if (!strncmp(namelist[idx]->d_name, match, strlen(match))) {
                ucs_assert(strlen(namelist[idx]->d_name) <= PATH_MAX);
                dest = (char *) *fpaths + (*num_devices * (sizeof(char) * PATH_MAX));
                strcpy(dest, namelist[idx]->d_name);
                *num_devices = *num_devices + 1;
            }
            free(namelist[idx]);
        }
        free(namelist);
    }

    return UCS_OK;
}

static ucs_status_t ucs_release_paths(char *fpaths)
{
    ucs_free(fpaths);
    return UCS_OK;
}

static int ucs_get_bus_id(char *name)
{
    char delim[] = ":";
    char *rval   = NULL;
    char *str    = NULL;
    char *str_p  = NULL;
    int count    = 0;
    int bus_id   = 0;
    size_t idx;
    int value;
    int pow_factor;
    size_t len;

    str = ucs_malloc(sizeof(char) * strlen(name), "ucs_get_bus_id str");
    if (NULL == str) {
        return -1;
    }
    str_p = str;
    strcpy(str, name);

    do {
        rval = strtok(str, delim);
        str = NULL;
        count++;
        if (count == 2) break; /* for 0000:0c:00.0 bus id = 0c */
    } while (rval != NULL);

    len = strlen(rval);
    for (idx = 0; idx < len; idx++) {
        pow_factor = pow(16, len - 1 - idx);
        value = (rval[idx] >= 'a') ? ((rval[idx] - 'a') + 10) : (rval[idx] - '0');
        value *= pow_factor;
        bus_id += value;
    }

    ucs_debug("dev name = %s bus_id = %d", name, bus_id);
    ucs_free(str_p);

    return bus_id;
}

ucs_status_t ucs_sys_get_mm_units(ucs_mm_unit_t **mm_units, int *num_units)
{
    int num_mm_units[UCS_MM_UNIT_LAST];
    char *mm_fpaths[UCS_MM_UNIT_LAST];
    ucs_mm_unit_enum_t mm_idx;
    int i;
    ucs_mm_unit_t *mm_unit_p;
    int mm_unit_idx;
    char *dev_loc;
    char *match;
    char *src;

    *num_units = 0;

    for (mm_idx = UCS_MM_UNIT_CPU; mm_idx < UCS_MM_UNIT_LAST; mm_idx++) {
        dev_loc = ucs_mm_unit_paths[mm_idx];
        match   = ucs_mm_unit_match[mm_idx];
        ucs_get_paths(dev_loc, match, &num_mm_units[mm_idx], &mm_fpaths[mm_idx]);
        *num_units += num_mm_units[mm_idx];
    }

    if (0 == *num_units) {
        goto out;
    }

    *mm_units = ucs_malloc(*num_units * sizeof(ucs_mm_unit_t), "ucs_mm_unit_t array");
    if (*mm_units == NULL) {
        ucs_error("failed to allocate mm_units");
        return UCS_ERR_NO_MEMORY;
    }

    mm_unit_p   = *mm_units;
    mm_unit_idx = 0;

    for (mm_idx = UCS_MM_UNIT_CPU; mm_idx < UCS_MM_UNIT_LAST; mm_idx++) {

        for (i = 0; i < num_mm_units[mm_idx]; i++) {
            strcpy(mm_unit_p->fpath, ucs_mm_unit_paths[mm_idx]);
            src = (char *) mm_fpaths[mm_idx] + (i * PATH_MAX);
            strcat(mm_unit_p->fpath, "/");
            strcat(mm_unit_p->fpath, src);
            mm_unit_p->bus_id       = (mm_idx == UCS_MM_UNIT_CPU) ? -1 : ucs_get_bus_id(src);
            mm_unit_p->id           = mm_unit_idx++;
            mm_unit_p->mm_unit_type = mm_idx;
            mm_unit_p               = mm_unit_p + 1;
        }

    }

out:
    for (mm_idx = UCS_MM_UNIT_CPU; mm_idx < UCS_MM_UNIT_LAST; mm_idx++) {
         ucs_release_paths(mm_fpaths[mm_idx]);
    }

    return UCS_OK;
}

ucs_status_t ucs_sys_free_mm_units(ucs_mm_unit_t *mm_units)
{
    ucs_free(mm_units);
    return UCS_OK;
}

ucs_status_t ucs_sys_get_sys_devices(ucs_sys_device_t **sys_devices, int *num_units)
{
    int num_sys_devices[UCS_SYS_DEVICE_LAST];
    char *sys_fpaths[UCS_SYS_DEVICE_LAST];
    ucs_sys_device_enum_t sys_idx;
    int i;
    ucs_sys_device_t *sys_dev_p;
    int sys_dev_idx;
    char *dev_loc;
    char *match;
    char *src;

    *num_units = 0;

    for (sys_idx = UCS_SYS_DEVICE_IB; sys_idx < UCS_SYS_DEVICE_LAST; sys_idx++) {
        dev_loc = ucs_sys_device_paths[sys_idx];
        match   = ucs_sys_device_match[sys_idx];
        ucs_get_paths(dev_loc, match, &num_sys_devices[sys_idx], &sys_fpaths[sys_idx]);
        *num_units += num_sys_devices[sys_idx];
    }

    if (0 == *num_units) {
        goto out;
    }

    *sys_devices = ucs_malloc(*num_units * sizeof(ucs_sys_device_t), "ucs_sys_device_t array");
    if (*sys_devices == NULL) {
        ucs_error("failed to allocate sys_devices");
        return UCS_ERR_NO_MEMORY;
    }

    sys_dev_p   = *sys_devices;
    sys_dev_idx = 0;

    for (sys_idx = UCS_SYS_DEVICE_IB; sys_idx < UCS_SYS_DEVICE_LAST; sys_idx++) {

        for (i = 0; i < num_sys_devices[sys_idx]; i++) {
            strcpy(sys_dev_p->fpath, ucs_sys_device_paths[sys_idx]);
            src = (char *) sys_fpaths[sys_idx] + (i * PATH_MAX);
            strcat(sys_dev_p->fpath, "/");
            strcat(sys_dev_p->fpath, src);
            sys_dev_p->bus_id       = ucs_get_bus_id(src);
            sys_dev_p->id           = sys_dev_idx++;
            sys_dev_p->sys_dev_type = sys_idx;
            sys_dev_p               = sys_dev_p + 1;
        }

    }

out:
    for (sys_idx = UCS_SYS_DEVICE_IB; sys_idx < UCS_SYS_DEVICE_LAST; sys_idx++) {
         ucs_release_paths(sys_fpaths[sys_idx]);
    }

    return UCS_OK;
}

ucs_status_t ucs_sys_free_sys_devices(ucs_sys_device_t *sys_devices)
{
    ucs_free(sys_devices);
    return UCS_OK;
}
