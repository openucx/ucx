/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "numa.h"

#include <ucs/datastruct/khash.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/type/spinlock.h>
#include <stdint.h>
#include <sched.h>
#include <dirent.h>

#define UCS_NUMA_MIN_DISTANCE       10
#define UCS_NUMA_NODE_MAX           INT16_MAX
#define UCS_NUMA_CORE_DIR_PATH      UCS_SYS_FS_CPUS_PATH "/cpu%d"
#define UCS_NUMA_NODES_DIR_PATH     UCS_SYS_FS_SYSTEM_PATH "/node"
#define UCS_NUMA_NODE_DISTANCE_PATH UCS_NUMA_NODES_DIR_PATH "/node%d/distance"


KHASH_MAP_INIT_INT(numa_distance, ucs_numa_distance_t);

typedef struct {
    unsigned     max_index;
    const char   *prefix;
    const size_t prefix_length;
} ucs_numa_get_max_dirent_ctx_t;

typedef struct {
    ucs_spinlock_t         lock;
    khash_t(numa_distance) numa_distance_hash;
} ucs_numa_global_ctx_t;

static ucs_numa_global_ctx_t ucs_numa_global_ctx;

static inline uint32_t ucs_numa_distance_hash_key(const ucs_numa_node_t node1,
                                                  const ucs_numa_node_t node2)
{
    static const uint8_t shift = sizeof(ucs_numa_node_t) * 8;

    return (ucs_max(node1, node2) << shift) | ucs_min(node1, node2);
}

static ucs_status_t
ucs_numa_get_max_dirent_cb(const struct dirent *entry, void *arg)
{
    ucs_numa_get_max_dirent_ctx_t *ctx = (ucs_numa_get_max_dirent_ctx_t*)arg;
    unsigned entry_index;

    if (!strncmp(entry->d_name, ctx->prefix, ctx->prefix_length)) {
        entry_index    = strtoul(entry->d_name + ctx->prefix_length, 0, 0);
        ctx->max_index = ucs_max(ctx->max_index, entry_index);
    }

    return UCS_OK;
}

/**
 * Iterate the directory entries in @param path of the form
 * @param prefix<index> and return the maximum index (positive number).
 *
 * @param [in]  path          Directory path.
 * @param [in]  prefix        Consider only entries starting with this prefix.
 * @param [in]  limit         Maximum legal value of index.
 * @param [in]  default_value Default value to return in case of an error.
 *
 * @return Maximum index of the entries starting with prefix or -1 on error.
 */
static int ucs_numa_get_max_dirent(const char *path, const char *prefix,
                                   unsigned limit, int default_value)
{
    ucs_numa_get_max_dirent_ctx_t ctx = {0, prefix, strlen(prefix)};

    if (ucs_sys_readdir(path, ucs_numa_get_max_dirent_cb, &ctx) != UCS_OK) {
        ucs_debug("failed to parse sysfs dir %s", path);
        ctx.max_index = default_value;
    } else if (ctx.max_index >= limit) {
        ucs_debug("max index %s/%s%u exceeds limit (%d)", path, prefix,
                  ctx.max_index, limit);
        ctx.max_index = default_value;
    }

    return ctx.max_index;
}

unsigned ucs_numa_num_configured_nodes()
{
    static unsigned num_nodes = 0;
    unsigned max_node;

    if (num_nodes == 0) {
        max_node  = ucs_numa_get_max_dirent(UCS_NUMA_NODES_DIR_PATH, "node",
                                            UCS_NUMA_NODE_MAX,
                                            UCS_NUMA_NODE_DEFAULT);
        num_nodes = max_node + 1;
    }

    return num_nodes;
}

unsigned ucs_numa_num_configured_cpus()
{
    static unsigned num_cpus = 0;
    unsigned max_cpu;

    if (num_cpus == 0) {
        max_cpu  = ucs_numa_get_max_dirent(UCS_SYS_FS_CPUS_PATH, "cpu",
                                           __CPU_SETSIZE, 0);
        num_cpus = max_cpu + 1;
    }

    return num_cpus;
}

ucs_numa_node_t ucs_numa_node_of_cpu(int cpu)
{
    /* Used for caching to improve performance */
    static ucs_numa_node_t cpu_numa_node[__CPU_SETSIZE] = {0};
    ucs_numa_node_t node;
    char core_dir_path[PATH_MAX];

    ucs_assert(cpu < __CPU_SETSIZE);

    if (cpu_numa_node[cpu] == 0) {
        ucs_snprintf_safe(core_dir_path, PATH_MAX, UCS_NUMA_CORE_DIR_PATH, cpu);
        node               = ucs_numa_get_max_dirent(core_dir_path, "node",
                                                     ucs_numa_num_configured_nodes(),
                                                     UCS_NUMA_NODE_DEFAULT);
        cpu_numa_node[cpu] = node + 1;
    }

    return cpu_numa_node[cpu] - 1;
}

ucs_numa_node_t ucs_numa_node_of_device(const char *dev_path)
{
    long parsed_node;
    ucs_status_t status;

    status = ucs_read_file_number(&parsed_node, 1, "%s/numa_node", dev_path);

    if ((status != UCS_OK) || (parsed_node < 0) ||
        (parsed_node >= UCS_NUMA_NODE_MAX)) {
        ucs_debug("failed to discover numa node for device: %s, status %s, \
                  parsed_node %ld", dev_path, ucs_status_string(status),
                  parsed_node);
        return UCS_NUMA_NODE_UNDEFINED;
    }

    return parsed_node;
}

/**
 * Parse and iterate the distance list of @param source.
 * For each neighbor of source put distance(source, neighbor)
 * in @ref numa_distance_hash.
 *
 * @param [in]  source source NUMA node
 * @param [in]  dest   destination NUMA node
 *
 * @return Distance from @param source to @param dest
 */
static ucs_numa_distance_t
ucs_numa_node_parse_distances(ucs_numa_node_t source, ucs_numa_node_t dest)
{
    ucs_numa_distance_t distance_to_dest = UCS_NUMA_MIN_DISTANCE;
    ucs_numa_node_t node                 = 0;
    ucs_numa_distance_t distance;
    ucs_kh_put_t kh_put_status;
    khiter_t hash_it;
    FILE *distance_fp;

    distance_fp = ucs_open_file("r", UCS_LOG_LEVEL_DEBUG,
                                UCS_NUMA_NODE_DISTANCE_PATH, source);
    if (distance_fp == NULL) {
        return distance_to_dest;
    }

    while ((fscanf(distance_fp, "%u", &distance) > 0) &&
           (node < UCS_NUMA_NODE_MAX)) {
        if (distance < UCS_NUMA_MIN_DISTANCE) {
            ucs_debug("node %u parsed NUMA distance %u is "
                      "smaller than the lower bound (%u)",
                      source, distance, UCS_NUMA_MIN_DISTANCE);
            distance = UCS_NUMA_MIN_DISTANCE;
        }

        if (node == dest) {
            distance_to_dest = distance;
        }

        hash_it = kh_put(numa_distance, &ucs_numa_global_ctx.numa_distance_hash,
                         ucs_numa_distance_hash_key(source, node),
                         &kh_put_status);
        if ((kh_put_status != UCS_KH_PUT_FAILED) &&
            (kh_put_status != UCS_KH_PUT_KEY_PRESENT)) {
            kh_value(&ucs_numa_global_ctx.numa_distance_hash,
                     hash_it) = distance;
        }

        node++;
    }

    if (node >= UCS_NUMA_NODE_MAX) {
        ucs_diag("number of nodes in the system is out of range (%u)",
                 UCS_NUMA_NODE_MAX);
    }

    fclose(distance_fp);
    return distance_to_dest;
}

ucs_numa_distance_t
ucs_numa_distance(ucs_numa_node_t node1, ucs_numa_node_t node2)
{
    ucs_numa_distance_t distance = UCS_NUMA_MIN_DISTANCE;
    khiter_t hash_it;

    ucs_assert(node1 < ucs_numa_num_configured_nodes());
    ucs_assert(node2 < ucs_numa_num_configured_nodes());

    ucs_spin_lock(&ucs_numa_global_ctx.lock);
    hash_it = kh_get(numa_distance, &ucs_numa_global_ctx.numa_distance_hash,
                     ucs_numa_distance_hash_key(node1, node2));

    if (ucs_likely(hash_it !=
                   kh_end(&ucs_numa_global_ctx.numa_distance_hash))) {
        distance = kh_value(&ucs_numa_global_ctx.numa_distance_hash, hash_it);
    } else {
        distance = ucs_numa_node_parse_distances(node1, node2);
    }

    ucs_spin_unlock(&ucs_numa_global_ctx.lock);
    return distance;
}

void ucs_numa_init()
{
    ucs_spinlock_init(&ucs_numa_global_ctx.lock, 0);
    kh_init_inplace(numa_distance, &ucs_numa_global_ctx.numa_distance_hash);
}

void ucs_numa_cleanup()
{
    kh_destroy_inplace(numa_distance, &ucs_numa_global_ctx.numa_distance_hash);
    ucs_spinlock_destroy(&ucs_numa_global_ctx.lock);
}
