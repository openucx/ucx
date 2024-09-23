/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019-2022. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/memory/numa.h>
#include <ucs/sys/math.h>
#include <ucs/sys/topo/base/topo.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>

#include <ucs/config/global_opts.h>
#include <ucs/datastruct/khash.h>
#include <ucs/type/spinlock.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/time/time.h>
#include <inttypes.h>


#define UCS_TOPO_MAX_SYS_DEVICES     256
#define UCS_TOPO_SYSFS_PCI_PREFIX    "/sys/bus/pci/devices/"
#define UCS_TOPO_SYSFS_DEVICES_ROOT  "/sys/devices"
#define UCS_TOPO_DEVICE_NAME_UNKNOWN "<unknown>"
#define UCS_TOPO_DEVICE_NAME_INVALID "<invalid>"

/*
 * Function pointer used to refer to specific implementations of
 * ucs_topo_get_memory_distance function by topology modules.
 * This function estimates the distance between the device and the system
 * memory used by the current thread according to its CPU affinity.
 * The function must have a fallback behavior.
 */
typedef void (*ucs_topo_get_memory_distance_func_t)(
        ucs_sys_device_t device, ucs_sys_dev_distance_t *distance);

/*
 * Topology API.
 */
typedef struct {
    /* Provider's ucs_topo_get_distance implementation */
    ucs_topo_get_distance_func_t        get_distance;

    /* Provider's ucs_topo_get_memory_distance implementation */
    ucs_topo_get_memory_distance_func_t get_memory_distance;
} ucs_sys_topo_ops_t;


/*
 * Structure needed to define a topology module implementation
 */
typedef struct {
    /* Name of the topology module */
    const char         *name;

    /* provider's ops */
    ucs_sys_topo_ops_t ops;

    ucs_list_link_t    list;
} ucs_sys_topo_provider_t;

typedef int64_t ucs_bus_id_bit_rep_t;

typedef struct {
    ucs_sys_bus_id_t bus_id;
    char             *name;
    unsigned         name_priority;
    ucs_numa_node_t  numa_node;
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
    .bandwidth = INFINITY
};

static ucs_topo_global_ctx_t ucs_topo_global_ctx;


/* Global list of topology detectors */
UCS_LIST_HEAD(ucs_sys_topo_providers_list);


/* According to NUMA distance definition distances are normalized to 10
 * and the relative distance correlates with the latency.
 * The following translation formula assumes that
 * access to main memory takes 100ns */
static inline double ucs_topo_sysfs_numa_distance_to_latency(double distance)
{
    return distance * 10e-9;
}

static ucs_sys_topo_provider_t *ucs_sys_topo_get_provider()
{
    static ucs_sys_topo_provider_t *provider = NULL;
    ucs_sys_topo_provider_t *list_provider;
    unsigned i;

    if (provider != NULL) {
        return provider;
    }

    for (i = 0; i < ucs_global_opts.topo_prio.count; ++i) {
        ucs_list_for_each(list_provider, &ucs_sys_topo_providers_list, list) {
            if (!strcmp(ucs_global_opts.topo_prio.names[i],
                        list_provider->name)) {
                provider = list_provider;
                return provider;
            }
        }
    }

    return provider;
}

static ucs_status_t
ucs_topo_get_distance_default(ucs_sys_device_t device1,
                              ucs_sys_device_t device2,
                              ucs_sys_dev_distance_t *distance)
{
    *distance = ucs_topo_default_distance;

    return UCS_OK;
}

static void
ucs_topo_get_memory_distance_default(ucs_sys_device_t device,
                                     ucs_sys_dev_distance_t *distance)
{
    *distance = ucs_topo_default_distance;
}

static ucs_sys_topo_provider_t ucs_sys_topo_provider_default = {
    .name = "default",
    .ops = {
        .get_distance        = ucs_topo_get_distance_default,
        .get_memory_distance = ucs_topo_get_memory_distance_default,
    }
};

ucs_status_t ucs_topo_get_distance(ucs_sys_device_t device1,
                                   ucs_sys_device_t device2,
                                   ucs_sys_dev_distance_t *distance)
{
    const ucs_sys_topo_provider_t *provider = ucs_sys_topo_get_provider();

    return provider->ops.get_distance(device1, device2, distance);
}

void ucs_topo_get_memory_distance(ucs_sys_device_t device,
                                  ucs_sys_dev_distance_t *distance)
{
    const ucs_sys_topo_provider_t *provider = ucs_sys_topo_get_provider();

    provider->ops.get_memory_distance(device, distance);
}

static ucs_bus_id_bit_rep_t
ucs_topo_get_bus_id_bit_repr(const ucs_sys_bus_id_t *bus_id)
{
    return (((uint64_t)bus_id->domain << 24) |
            ((uint64_t)bus_id->bus << 16)    |
            ((uint64_t)bus_id->slot << 8)    |
            (bus_id->function));
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

static ucs_status_t
ucs_topo_bus_id_to_sysfs_path(const ucs_sys_bus_id_t *bus_id, char *path,
                              size_t max)
{
    const size_t prefix_length = strlen(UCS_TOPO_SYSFS_PCI_PREFIX);
    ucs_status_t status;
    char *link_path;

    status = ucs_string_alloc_path_buffer(&link_path, "link_path");
    if (status != UCS_OK) {
        goto out;
    }

    if (max < PATH_MAX) {
        status = UCS_ERR_BUFFER_TOO_SMALL;
        goto out_free_link_path;
    }

    ucs_strncpy_safe(link_path, UCS_TOPO_SYSFS_PCI_PREFIX, PATH_MAX);
    ucs_topo_bus_id_str(bus_id, 0, link_path + prefix_length,
                        PATH_MAX - prefix_length);
    if (realpath(link_path, path) == NULL) {
        status = UCS_ERR_IO_ERROR;
    }

out_free_link_path:
    ucs_free(link_path);
out:
    return status;
}

static int
ucs_topo_read_device_numa_node(const ucs_sys_bus_id_t *bus_id)
{
    int numa_node = UCS_NUMA_NODE_UNDEFINED;
    char *path;
    ucs_status_t status;

    status = ucs_string_alloc_path_buffer(&path, "sysfs_path");
    if (status != UCS_OK) {
        goto out;
    }

    status = ucs_topo_bus_id_to_sysfs_path(bus_id, path, PATH_MAX);
    if (status != UCS_OK) {
        goto out_free_path;
    }

    numa_node = ucs_numa_node_of_device(path);

out_free_path:
    ucs_free(path);
out:
    return numa_node;
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
    } else if ((kh_put_status == UCS_KH_PUT_BUCKET_EMPTY) ||
               (kh_put_status == UCS_KH_PUT_BUCKET_CLEAR)) {
        ucs_assert_always(ucs_topo_global_ctx.num_devices <
                          UCS_TOPO_MAX_SYS_DEVICES);
        *sys_dev = ucs_topo_global_ctx.num_devices;
        ++ucs_topo_global_ctx.num_devices;

        kh_value(&ucs_topo_global_ctx.bus_to_sys_dev_hash, hash_it) = *sys_dev;

        /* Set default name to abbreviated BDF */
        name = ucs_malloc(UCS_SYS_BDF_NAME_MAX, "sys_dev_bdf_name");
        if (name != NULL) {
            ucs_topo_bus_id_str(bus_id, 1, name, UCS_SYS_BDF_NAME_MAX);
        }

        ucs_topo_global_ctx.devices[*sys_dev].bus_id        = *bus_id;
        ucs_topo_global_ctx.devices[*sys_dev].name          = name;
        ucs_topo_global_ctx.devices[*sys_dev].name_priority = 0;
        ucs_topo_global_ctx.devices[*sys_dev].numa_node     =
                ucs_topo_read_device_numa_node(bus_id);
        ucs_debug("added sys_dev %d for bus id %s", *sys_dev, name);
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

static ucs_status_t
ucs_topo_sys_dev_to_sysfs_path(ucs_sys_device_t sys_dev, char *path, size_t max)
{
    ucs_status_t status;

    ucs_spin_lock(&ucs_topo_global_ctx.lock);

    if (sys_dev >= ucs_topo_global_ctx.num_devices) {
        ucs_error("system device %d is invalid (max: %d)", sys_dev,
                  ucs_topo_global_ctx.num_devices);
        status = UCS_ERR_INVALID_PARAM;
        goto out_unlock;
    }

    status = ucs_topo_bus_id_to_sysfs_path(
            &ucs_topo_global_ctx.devices[sys_dev].bus_id, path, max);
    if (status != UCS_OK) {
        goto out_unlock;
    }

    status = UCS_OK;

out_unlock:
    ucs_spin_unlock(&ucs_topo_global_ctx.lock);
    return status;
}

static int ucs_topo_is_sys_root(const char *path)
{
    return !strcmp(path, UCS_TOPO_SYSFS_DEVICES_ROOT);
}

static int ucs_topo_is_pci_root(const char *path)
{
    int count = -1;
    sscanf(path, UCS_TOPO_SYSFS_DEVICES_ROOT "/pci%*x:%*x%n", &count);
    return count == strlen(path);
}

static void ucs_topo_set_distance(const ucs_sys_dev_distance_t *distance_cfg,
                                  double bw_auto,
                                  ucs_sys_dev_distance_t *distance)
{
    distance->latency   = distance_cfg->latency;
    distance->bandwidth = UCS_CONFIG_DBL_IS_AUTO(distance_cfg->bandwidth) ?
                                  bw_auto : distance_cfg->bandwidth;
}

static double ucs_topo_pci_root_bw(const char *path1, const char *path2)
{
    size_t path_distance = ucs_path_calc_distance(path1, path2);

    ucs_trace_data("distance between '%s' and '%s' is %zu", path1, path2,
                   path_distance);
    ucs_assertv(path_distance > 0, "path1=%s path2=%s", path1, path2);

    return ucs_min(3500.0 * UCS_MBYTE, (19200.0 * UCS_MBYTE) / path_distance);
}

static int
ucs_topo_is_same_numa_node(ucs_sys_device_t device1,
                             ucs_sys_device_t device2)
{
    ucs_numa_node_t numa1 = ucs_topo_sys_device_get_numa_node(device1);
    ucs_numa_node_t numa2 = ucs_topo_sys_device_get_numa_node(device2);

    return (numa1 == numa2) && (numa1 != UCS_SYS_DEVICE_ID_UNKNOWN);
}

static ucs_status_t
ucs_topo_get_distance_sysfs(ucs_sys_device_t device1,
                            ucs_sys_device_t device2,
                            ucs_sys_dev_distance_t *distance)
{
    ucs_status_t status;
    char *path1, *path2, *common_path;

    status = ucs_string_alloc_path_buffer(&path1, "path1");
    if (status != UCS_OK) {
        goto out;
    }

    status = ucs_string_alloc_path_buffer(&path2, "path2");
    if (status != UCS_OK) {
        goto out_free_path1;
    }

    status = ucs_string_alloc_path_buffer(&common_path, "common_path");
    if (status != UCS_OK) {
        goto out_free_path2;
    }

    /* If one of the devices is unknown, we assume near topology */
    if ((device1 == UCS_SYS_DEVICE_ID_UNKNOWN) ||
        (device2 == UCS_SYS_DEVICE_ID_UNKNOWN) || (device1 == device2)) {
        goto default_distance;
    }

    status = ucs_topo_sys_dev_to_sysfs_path(device1, path1, PATH_MAX);
    if (status != UCS_OK) {
        ucs_debug("failed to get sysfs path for %s",
                  ucs_topo_sys_device_get_name(device1));
        goto default_distance;
    }

    status = ucs_topo_sys_dev_to_sysfs_path(device2, path2, PATH_MAX);
    if (status != UCS_OK) {
        ucs_debug("failed to get sysfs path for %s",
                  ucs_topo_sys_device_get_name(device2));
        goto default_distance;
    }

    ucs_path_get_common_parent(path1, path2, common_path);
    if (ucs_topo_is_pci_root(common_path)) {
        ucs_topo_set_distance(&ucs_global_opts.dist.phb,
                              ucs_topo_pci_root_bw(path1, path2), distance);
        goto out_free_common_path;
    } else if (ucs_topo_is_sys_root(common_path)) {
        if (ucs_topo_is_same_numa_node(device1, device2)) {
            ucs_topo_set_distance(&ucs_global_opts.dist.node, 17000 * UCS_MBYTE,
                                  distance);
            goto out_free_common_path;
        }

        ucs_topo_set_distance(&ucs_global_opts.dist.sys, 220 * UCS_MBYTE,
                              distance);
        goto out_free_common_path;
    }

    /* Report best perf for common PCI bridge or sysfs parsing error */
default_distance:
    status = ucs_topo_get_distance_default(device1, device2, distance);
out_free_common_path:
    ucs_free(common_path);
out_free_path2:
    ucs_free(path2);
out_free_path1:
    ucs_free(path1);
out:
    return status;
}

static void ucs_topo_get_memory_distance_sysfs(ucs_sys_device_t device,
                                               ucs_sys_dev_distance_t *distance)
{
    double total_distance = 0;
    int full_affinity     = 0;
    ucs_sys_cpuset_t thread_cpuset;
    unsigned cpu, num_cpus, cpuset_size;
    ucs_numa_node_t dev_node;
    ucs_status_t status;

    /* If the device is unknown, we assume min distance */
    if (device == UCS_SYS_DEVICE_ID_UNKNOWN) {
        ucs_topo_get_memory_distance_default(device, distance);
        return;
    }

    status = ucs_sys_pthread_getaffinity(&thread_cpuset);
    if (status != UCS_OK) {
        /* If we failed to read thread affinity distance is calculated
         * for a process with full CPU affinity */
        full_affinity = 1;
    }

    dev_node = ucs_topo_sys_device_get_numa_node(device);
    if (dev_node == UCS_NUMA_NODE_UNDEFINED) {
        dev_node = UCS_NUMA_NODE_DEFAULT;
    }

    num_cpus = ucs_numa_num_configured_cpus();
    for (cpu = 0; cpu < num_cpus; ++cpu) {
        if (!full_affinity && !CPU_ISSET(cpu, &thread_cpuset)) {
            continue;
        }

        total_distance += ucs_numa_distance(dev_node,
                                            ucs_numa_node_of_cpu(cpu));
    }

    distance->bandwidth = ucs_topo_default_distance.bandwidth;
    cpuset_size         = full_affinity ? num_cpus : CPU_COUNT(&thread_cpuset);
    distance->latency = ucs_topo_sysfs_numa_distance_to_latency(total_distance /
                                                                cpuset_size);
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

ucs_sys_device_t ucs_topo_get_sysfs_dev(const char *dev_name,
                                        const char *sysfs_path,
                                        unsigned name_priority)
{
    ucs_sys_device_t sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
    const char *bdf_name;
    ucs_status_t status;

    if (sysfs_path == NULL) {
        goto out_unknown;
    }

    bdf_name = strrchr(sysfs_path, '/');
    if (bdf_name == NULL) {
        goto out_unknown;
    }

    ++bdf_name; /* Move past '/' separator */

    status = ucs_topo_find_device_by_bdf_name(bdf_name, &sys_dev);
    if (status != UCS_OK) {
        goto out_unknown;
    }

    status = ucs_topo_sys_device_set_name(sys_dev, dev_name, name_priority);
    if (status != UCS_OK) {
        ucs_debug("%s: ucs_topo_sys_device_set_name failed, using default name "
                  "%s",
                  dev_name, ucs_topo_sys_device_get_name(sys_dev));
    }

    ucs_debug("%s: bdf_name %s sys_dev %d", dev_name, bdf_name, sys_dev);
    return sys_dev;

out_unknown:
    ucs_debug("%s: system device unknown", dev_name);
    return UCS_SYS_DEVICE_ID_UNKNOWN;
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

ucs_status_t ucs_topo_sys_device_set_name(ucs_sys_device_t sys_dev,
                                          const char *name, unsigned priority)
{
    ucs_spin_lock(&ucs_topo_global_ctx.lock);

    if (sys_dev >= ucs_topo_global_ctx.num_devices) {
        ucs_error("system device %d is invalid (max: %d)", sys_dev,
                  ucs_topo_global_ctx.num_devices);
        ucs_spin_unlock(&ucs_topo_global_ctx.lock);
        return UCS_ERR_INVALID_PARAM;
    }

    if (priority > ucs_topo_global_ctx.devices[sys_dev].name_priority) {
        ucs_free(ucs_topo_global_ctx.devices[sys_dev].name);
        ucs_topo_global_ctx.devices[sys_dev].name = ucs_strdup(name,
                                                               "sys_dev_name");
        ucs_topo_global_ctx.devices[sys_dev].name_priority = priority;
    }
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

ucs_numa_node_t ucs_topo_sys_device_get_numa_node(ucs_sys_device_t sys_dev)
{
    int numa_node;

    if (sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
        return UCS_NUMA_NODE_UNDEFINED;
    }

    ucs_spin_lock(&ucs_topo_global_ctx.lock);
    if (sys_dev < ucs_topo_global_ctx.num_devices) {
        numa_node = ucs_topo_global_ctx.devices[sys_dev].numa_node;
    } else {
        numa_node = UCS_NUMA_NODE_UNDEFINED;
    }
    ucs_spin_unlock(&ucs_topo_global_ctx.lock);

    return numa_node;
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

static ucs_sys_topo_provider_t ucs_sys_topo_provider_sysfs = {
    .name = "sysfs",
    .ops = {
        .get_distance        = ucs_topo_get_distance_sysfs,
        .get_memory_distance = ucs_topo_get_memory_distance_sysfs,
    }
};

void ucs_topo_init()
{
    ucs_spinlock_init(&ucs_topo_global_ctx.lock, 0);
    kh_init_inplace(bus_to_sys_dev, &ucs_topo_global_ctx.bus_to_sys_dev_hash);
    ucs_topo_global_ctx.num_devices = 0;
    ucs_list_add_tail(&ucs_sys_topo_providers_list,
                      &ucs_sys_topo_provider_default.list);
    ucs_list_add_tail(&ucs_sys_topo_providers_list,
                      &ucs_sys_topo_provider_sysfs.list);
}

void ucs_topo_cleanup()
{
    ucs_topo_sys_device_info_t *device;

    ucs_list_del(&ucs_sys_topo_provider_sysfs.list);
    ucs_list_del(&ucs_sys_topo_provider_default.list);

    while (ucs_topo_global_ctx.num_devices-- > 0) {
        device = &ucs_topo_global_ctx.devices[ucs_topo_global_ctx.num_devices];
        ucs_free(device->name);
    }

    kh_destroy_inplace(bus_to_sys_dev,
                       &ucs_topo_global_ctx.bus_to_sys_dev_hash);
    ucs_spinlock_destroy(&ucs_topo_global_ctx.lock);
}

typedef struct {
    double     bw_gbps;       /* Link speed */
    uint16_t   payload;       /* Payload used to data transfer */
    uint16_t   tlp_overhead;  /* PHY + data link layer + header + CRC */
    uint16_t   ctrl_ratio;    /* Number of TLC before ACK */
    uint16_t   ctrl_overhead; /* Length of control TLP */
    uint16_t   encoding;      /* Number of encoded symbol bits */
    uint16_t   decoding;      /* Number of decoded symbol bits */
    const char *name;         /* Name of PCI generation */
} ucs_topo_pci_info_t;

/*
 * - TLP (Transaction Layer Packet) overhead calculations (no ECRC):
 *   Gen1/2:
 *     Start   SeqNum   Hdr_64bit   LCRC   End
 *       1   +   2    +   16      +   4  +  1  = 24
 *
 *   Gen3/4:
 *     Start   SeqNum   Hdr_64bit   LCRC
 *       4   +   2    +   16      +   4  = 26
 *
 * - DLLP (Data Link Layer Packet) overhead calculations:
 *    - Control packet 8b ACK + 8b flow control
 *    - ACK/FC ratio: 1 per 4 TLPs
 *
 * References:
 * [1] https://www.xilinx.com/support/documentation/white_papers/wp350.pdf
 * [2] https://xdevs.com/doc/Standards/PCI/PCI_Express_Base_4.0_Rev0.3_February19-2014.pdf
 * [3] https://www.nxp.com/docs/en/application-note/AN3935.pdf
 */
static const ucs_topo_pci_info_t ucs_topo_pci_info[] = {
    {.name          = "gen1",
     .bw_gbps       = 2.5,
     .payload       = 256,
     .tlp_overhead  = 24,
     .ctrl_ratio    = 4,
     .ctrl_overhead = 16,
     .encoding      = 8,
     .decoding      = 10},
    {.name          = "gen2",
     .bw_gbps       = 5,
     .payload       = 256,
     .tlp_overhead  = 24,
     .ctrl_ratio    = 4,
     .ctrl_overhead = 16,
     .encoding      = 8,
     .decoding      = 10},
    {.name          = "gen3",
     .bw_gbps       = 8,
     .payload       = 256,
     .tlp_overhead  = 26,
     .ctrl_ratio    = 4,
     .ctrl_overhead = 16,
     .encoding      = 128,
     .decoding      = 130},
    {.name          = "gen4",
     .bw_gbps       = 16,
     .payload       = 256,
     .tlp_overhead  = 26,
     .ctrl_ratio    = 4,
     .ctrl_overhead = 16,
     .encoding      = 128,
     .decoding      = 130},
    {.name          = "gen5",
     .bw_gbps       = 32,
     .payload       = 256,
     .tlp_overhead  = 26,
     .ctrl_ratio    = 4,
     .ctrl_overhead = 16,
     .encoding      = 128,
     .decoding      = 130},
};

double ucs_topo_get_pci_bw(const char *dev_name, const char *sysfs_path)
{
    const char *pci_width_file_name = "current_link_width";
    const char *pci_speed_file_name = "current_link_speed";

    double bw_gbps, effective_bw, link_utilization;
    const ucs_topo_pci_info_t *p;
    char pci_width_str[16];
    char pci_speed_str[16];
    ucs_status_t status;
    unsigned width;
    char gts[16];
    size_t i;

    status = ucs_sys_read_sysfs_file(dev_name, sysfs_path, pci_width_file_name,
                                     pci_width_str, sizeof(pci_width_str),
                                     UCS_LOG_LEVEL_DEBUG);
    if (status != UCS_OK) {
        goto out_max_bw;
    }

    status = ucs_sys_read_sysfs_file(dev_name, sysfs_path, pci_speed_file_name,
                                     pci_speed_str, sizeof(pci_speed_str),
                                     UCS_LOG_LEVEL_DEBUG);
    if (status != UCS_OK) {
        goto out_max_bw;
    }

    if (sscanf(pci_width_str, "%u", &width) < 1) {
        ucs_debug("%s: incorrect format of %s file: expected: <unsigned "
                  "integer>, actual: %s\n",
                  dev_name, pci_width_file_name, pci_width_str);
        goto out_max_bw;
    }

    if ((sscanf(pci_speed_str, "%lf%s", &bw_gbps, gts) < 2) ||
        strcasecmp("GT/s", ucs_strtrim(gts))) {
        ucs_debug("%s: incorrect format of %s file: expected: <double> GT/s, "
                  "actual: %s\n",
                  dev_name, pci_speed_file_name, pci_speed_str);
        goto out_max_bw;
    }

    for (i = 0; i < ucs_static_array_size(ucs_topo_pci_info); i++) {
        p = &ucs_topo_pci_info[i];
        if ((bw_gbps / p->bw_gbps) > 1.01) { /* floating-point compare */
            continue;
        }

        link_utilization = (double)(p->payload * p->ctrl_ratio) /
                           (((p->payload + p->tlp_overhead) * p->ctrl_ratio) +
                            p->ctrl_overhead);
        /* coverity[overflow] */
        effective_bw     = (p->bw_gbps * 1e9 / 8.0) * width *
                           ((double)p->encoding / p->decoding) * link_utilization;
        ucs_trace("%s: PCIe %s %ux, effective throughput %.3f MB/s %.3f Gb/s",
                  dev_name, p->name, width, effective_bw / UCS_MBYTE,
                  effective_bw * 8e-9);
        return effective_bw;
    }

out_max_bw:
    ucs_debug("%s: pci bandwidth undetected, using maximal value", dev_name);
    return INFINITY;
}

const char *ucs_topo_resolve_sysfs_path(const char *dev_path, char *path_buffer)
{
    const char *detected_type = NULL;
    char *device_file_path, *sysfs_realpath, *sysfs_path;
    struct stat st_buf;
    ucs_status_t status;

    status = ucs_string_alloc_path_buffer(&device_file_path, "device_file_path");
    if (status != UCS_OK) {
        goto out_undetected;
    }

    /* realpath name is expected to be like below:
     * PF: /sys/devices/.../0000:03:00.0/<interface_type>/<dev_name>
     * SF: /sys/devices/.../0000:03:00.0/<UUID>/<interface_type>/<dev_name>
     */

    sysfs_realpath = realpath(dev_path, path_buffer);
    if (sysfs_realpath == NULL) {
        goto out_free_device_file_path;
    }

    /* Try PF: strip 2 components */
    sysfs_path = ucs_dirname(sysfs_realpath, 2);
    ucs_snprintf_safe(device_file_path, PATH_MAX, "%s/device", sysfs_path);

    if (!stat(device_file_path, &st_buf)) {
        detected_type = "PF";
        goto out_detected;
    }

    /* Try SF: strip 3 components (one more) */
    sysfs_path = ucs_dirname(sysfs_path, 1);
    ucs_snprintf_safe(device_file_path, PATH_MAX, "%s/device", sysfs_path);

    if (!stat(device_file_path, &st_buf)) {
        detected_type = "SF";
        goto out_detected;
    }

out_free_device_file_path:
    ucs_free(device_file_path);
out_undetected:
    ucs_debug("%s: sysfs path undetected", dev_path);
    return NULL;
out_detected:
    ucs_debug("%s: %s sysfs path is '%s'\n", dev_path, detected_type,
              sysfs_path);
    ucs_free(device_file_path);
    return sysfs_path;
}
