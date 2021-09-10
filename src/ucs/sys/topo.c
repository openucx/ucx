/**
* Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "topo.h"
#include "string.h"
#include "sys.h"

#include <ucs/datastruct/khash.h>
#include <ucs/type/spinlock.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/time/time.h>
#include <inttypes.h>
#include <float.h>


#define UCS_TOPO_MAX_SYS_DEVICES     256
#define UCS_TOPO_SYSFS_PCI_PREFIX    "/sys/class/pci_bus"
#define UCS_TOPO_DEVICE_NAME_UNKNOWN "<unknown>"
#define UCS_TOPO_DEVICE_NAME_INVALID "<invalid>"

typedef int64_t ucs_bus_id_bit_rep_t;

typedef struct {
    ucs_sys_bus_id_t bus_id;
    char             *name;
} ucs_topo_sys_device_info_t;

KHASH_MAP_INIT_INT64(bus_to_sys_dev, ucs_sys_device_t);

typedef struct ucs_topo_global_ctx {
    ucs_spinlock_t             lock;
    khash_t(bus_to_sys_dev)    bus_to_sys_dev_hash;
    ucs_topo_sys_device_info_t devices[UCS_TOPO_MAX_SYS_DEVICES];
    unsigned                   num_devices;
} ucs_topo_global_ctx_t;


const ucs_sys_dev_distance_t ucs_topo_default_distance = {
    .latency   = 0,
    .bandwidth = DBL_MAX
};
static ucs_topo_global_ctx_t ucs_topo_global_ctx;


static ucs_bus_id_bit_rep_t
ucs_topo_get_bus_id_bit_repr(const ucs_sys_bus_id_t *bus_id)
{
    return (((uint64_t)bus_id->domain << 24) |
            ((uint64_t)bus_id->bus << 16)    |
            ((uint64_t)bus_id->slot << 8)    |
            (bus_id->function));
}

void ucs_topo_init()
{
    ucs_spinlock_init(&ucs_topo_global_ctx.lock, 0);
    kh_init_inplace(bus_to_sys_dev, &ucs_topo_global_ctx.bus_to_sys_dev_hash);
    ucs_topo_global_ctx.num_devices = 0;
}

void ucs_topo_cleanup()
{
    ucs_topo_sys_device_info_t *device;
    while (ucs_topo_global_ctx.num_devices-- > 0) {
        device = &ucs_topo_global_ctx.devices[ucs_topo_global_ctx.num_devices];
        ucs_free(device->name);
    }
    kh_destroy_inplace(bus_to_sys_dev,
                       &ucs_topo_global_ctx.bus_to_sys_dev_hash);
    ucs_spinlock_destroy(&ucs_topo_global_ctx.lock);
}

unsigned ucs_topo_num_devices()
{
    unsigned num_devices;

    ucs_spin_lock(&ucs_topo_global_ctx.lock);
    num_devices = ucs_topo_global_ctx.num_devices;
    ucs_spin_unlock(&ucs_topo_global_ctx.lock);

    return num_devices;
}

static void ucs_topo_bus_id_str(const ucs_sys_bus_id_t *bus_id, int abbreviate,
                                char *str, size_t max)
{
    if (abbreviate && (bus_id->domain == 0)) {
        ucs_snprintf_safe(str, max, "%02x:%02x.%d", bus_id->bus, bus_id->slot,
                          bus_id->function);
    } else {
        ucs_snprintf_safe(str, max, "%04x:%02x:%02x.%d", bus_id->domain,
                          bus_id->bus, bus_id->slot, bus_id->function);
    }
}

ucs_status_t ucs_topo_find_device_by_bus_id(const ucs_sys_bus_id_t *bus_id,
                                            ucs_sys_device_t *sys_dev)
{
    ucs_bus_id_bit_rep_t bus_id_bit_rep;
    ucs_kh_put_t kh_put_status;
    khiter_t hash_it;
    char *name;

    bus_id_bit_rep  = ucs_topo_get_bus_id_bit_repr(bus_id);

    ucs_spin_lock(&ucs_topo_global_ctx.lock);
    hash_it = kh_put(
            bus_to_sys_dev /*name*/,
            &ucs_topo_global_ctx.bus_to_sys_dev_hash /*pointer to hashmap*/,
            bus_id_bit_rep /*key*/, &kh_put_status);

    if (kh_put_status == UCS_KH_PUT_KEY_PRESENT) {
        *sys_dev = kh_value(&ucs_topo_global_ctx.bus_to_sys_dev_hash, hash_it);
        ucs_debug("bus id 0x%"PRIx64" exists. sys_dev = %u", bus_id_bit_rep,
                  *sys_dev);
    } else if ((kh_put_status == UCS_KH_PUT_BUCKET_EMPTY) ||
               (kh_put_status == UCS_KH_PUT_BUCKET_CLEAR)) {
        ucs_assert_always(ucs_topo_global_ctx.num_devices <
                          UCS_TOPO_MAX_SYS_DEVICES);
        *sys_dev = ucs_topo_global_ctx.num_devices;
        ++ucs_topo_global_ctx.num_devices;

        ucs_debug("bus id 0x%"PRIx64" doesn't exist. sys_dev = %u",
                  bus_id_bit_rep, *sys_dev);

        kh_value(&ucs_topo_global_ctx.bus_to_sys_dev_hash, hash_it) = *sys_dev;

        /* Set default name to abbreviated BDF */
        name = ucs_malloc(UCS_SYS_BDF_NAME_MAX, "sys_dev_bdf_name");
        if (name != NULL) {
            ucs_topo_bus_id_str(bus_id, 1, name, UCS_SYS_BDF_NAME_MAX);
        }

        ucs_topo_global_ctx.devices[*sys_dev].bus_id = *bus_id;
        ucs_topo_global_ctx.devices[*sys_dev].name   = name;
    }

    ucs_spin_unlock(&ucs_topo_global_ctx.lock);
    return UCS_OK;
}

ucs_status_t ucs_topo_get_device_bus_id(ucs_sys_device_t sys_dev,
                                        ucs_sys_bus_id_t *bus_id)
{
    if (sys_dev >= ucs_topo_global_ctx.num_devices) {
        return UCS_ERR_NO_ELEM;
    }

    *bus_id = ucs_topo_global_ctx.devices[sys_dev].bus_id;
    return UCS_OK;
}

static void
ucs_topo_get_bus_path(const ucs_sys_bus_id_t *bus_id, char *path, size_t max)
{
    ucs_snprintf_safe(path, max, "%s/%04x:%02x", UCS_TOPO_SYSFS_PCI_PREFIX,
                      bus_id->domain, bus_id->bus);
}

ucs_status_t ucs_topo_get_distance(ucs_sys_device_t device1,
                                   ucs_sys_device_t device2,
                                   ucs_sys_dev_distance_t *distance)
{
    char path1[PATH_MAX], path2[PATH_MAX];
    ssize_t path_distance;

    /* If one of the devices is unknown, we assume near topology */
    if ((device1 == UCS_SYS_DEVICE_ID_UNKNOWN) ||
        (device2 == UCS_SYS_DEVICE_ID_UNKNOWN) || (device1 == device2)) {
        goto default_distance;
    }

    if ((device1 >= ucs_topo_num_devices()) ||
        (device2 >= ucs_topo_num_devices())) {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_topo_get_bus_path(&ucs_topo_global_ctx.devices[device1].bus_id, path1,
                          sizeof(path1));
    ucs_topo_get_bus_path(&ucs_topo_global_ctx.devices[device2].bus_id, path2,
                          sizeof(path2));

    path_distance = ucs_path_calc_distance(path1, path2);
    if (path_distance <= 0) {
        /* Assume default distance for devices that cannot be found in sysfs */
        goto default_distance;
    }

    /* Rough approximation of bandwidth/latency as function of PCI distance in
     * sysfs.
     * TODO implement more accurate estimation, based on system type, PCIe
     * switch, etc.
     */
    distance->latency   = 100e-9 * path_distance;
    distance->bandwidth = (20000 / path_distance) * UCS_MBYTE;
    return UCS_OK;

default_distance:
    *distance = ucs_topo_default_distance;
    return UCS_OK;
}

const char *ucs_topo_distance_str(const ucs_sys_dev_distance_t *distance,
                                  char *buffer, size_t max)
{
    UCS_STRING_BUFFER_FIXED(strb, buffer, max);

    ucs_string_buffer_appendf(&strb, "%.0fns ",
                              distance->latency * UCS_NSEC_PER_SEC);

    if (distance->bandwidth <= UCS_PBYTE) {
        /* Print bandwidth value only if limited */
        ucs_string_buffer_appendf(&strb, "%.2fMB/s",
                                  distance->bandwidth / UCS_MBYTE);
    } else {
        ucs_string_buffer_appendf(&strb, ">1PB/s");
    }

    return ucs_string_buffer_cstr(&strb);
}

const char *
ucs_topo_sys_device_bdf_name(ucs_sys_device_t sys_dev, char *buffer, size_t max)
{
    if (sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
        ucs_strncpy_safe(buffer, UCS_TOPO_DEVICE_NAME_UNKNOWN, max);
    } else {
        ucs_spin_lock(&ucs_topo_global_ctx.lock);
        if (sys_dev < ucs_topo_global_ctx.num_devices) {
            ucs_topo_bus_id_str(&ucs_topo_global_ctx.devices[sys_dev].bus_id, 0,
                                buffer, max);
        } else {
            ucs_strncpy_safe(buffer, UCS_TOPO_DEVICE_NAME_INVALID, max);
        }
        ucs_spin_unlock(&ucs_topo_global_ctx.lock);
    }

    return buffer;
}

ucs_status_t
ucs_topo_find_device_by_bdf_name(const char *name, ucs_sys_device_t *sys_dev)
{
    ucs_sys_bus_id_t bus_id;
    int num_fields;

    /* Try to parse as "<domain>:<bus>:<device>.<function>" */
    num_fields = sscanf(name, "%hx:%hhx:%hhx.%hhx", &bus_id.domain, &bus_id.bus,
                        &bus_id.slot, &bus_id.function);
    if (num_fields == 4) {
        return ucs_topo_find_device_by_bus_id(&bus_id, sys_dev);
    }

    /* Try to parse as "<bus>:<device>.<function>", assume domain is 0 */
    bus_id.domain = 0;
    num_fields    = sscanf(name, "%hhx:%hhx.%hhx", &bus_id.bus, &bus_id.slot,
                           &bus_id.function);
    if (num_fields == 3) {
        return ucs_topo_find_device_by_bus_id(&bus_id, sys_dev);
    }

    return UCS_ERR_INVALID_PARAM;
}

ucs_status_t
ucs_topo_sys_device_set_name(ucs_sys_device_t sys_dev, const char *name)
{
    ucs_spin_lock(&ucs_topo_global_ctx.lock);

    if (sys_dev >= ucs_topo_global_ctx.num_devices) {
        ucs_spin_unlock(&ucs_topo_global_ctx.lock);
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_free(ucs_topo_global_ctx.devices[sys_dev].name);
    ucs_topo_global_ctx.devices[sys_dev].name = ucs_strdup(name,
                                                           "sys_dev_name");
    ucs_spin_unlock(&ucs_topo_global_ctx.lock);

    return UCS_OK;
}

const char *ucs_topo_sys_device_get_name(ucs_sys_device_t sys_dev)
{
    const char *name;

    if (sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
        name = UCS_TOPO_DEVICE_NAME_UNKNOWN;
    } else {
        ucs_spin_lock(&ucs_topo_global_ctx.lock);
        if (sys_dev < ucs_topo_global_ctx.num_devices) {
            name = ucs_topo_global_ctx.devices[sys_dev].name;
        } else {
            name = UCS_TOPO_DEVICE_NAME_INVALID;
        }
        ucs_spin_unlock(&ucs_topo_global_ctx.lock);
    }

    return name;
}

void ucs_topo_print_info(FILE *stream)
{
    char bdf_name[UCS_SYS_BDF_NAME_MAX];
    volatile ucs_sys_device_t sys_dev;

    for (sys_dev = 0; sys_dev < ucs_topo_global_ctx.num_devices; ++sys_dev) {
        fprintf(stream, " %d  %*s  %s\n", sys_dev, UCS_SYS_BDF_NAME_MAX,
                ucs_topo_sys_device_bdf_name(sys_dev, bdf_name,
                                             sizeof(bdf_name)),
                ucs_topo_global_ctx.devices[sys_dev].name);
    }
}
