/**
 * Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucs/config/global_opts.h>
#include <ucs/datastruct/queue.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/math.h>
#include <ucs/sys/string.h>
#include <ucs/vfs/base/vfs_cb.h>
#include <ucs/vfs/base/vfs_obj.h>
#include "rcache_int.h"


#define UCS_RCACHE_VFS_MAX_STR "max"


static void ucs_rcache_vfs_read_inv_q_length(void *obj,
                                             ucs_string_buffer_t *strb,
                                             void *arg_ptr, uint64_t arg_u64)
{
    ucs_rcache_t *rcache = obj;
    size_t rcache_inv_q_length;

    ucs_spin_lock(&rcache->lock);
    rcache_inv_q_length = ucs_queue_length(&rcache->inv_q);
    ucs_spin_unlock(&rcache->lock);

    ucs_string_buffer_appendf(strb, "%zu\n", rcache_inv_q_length);
}

static void ucs_rcache_vfs_read_gc_list_length(void *obj,
                                               ucs_string_buffer_t *strb,
                                               void *arg_ptr, uint64_t arg_u64)
{
    ucs_rcache_t *rcache = obj;
    unsigned long rcache_gc_list_length;

    ucs_spin_lock(&rcache->lock);
    rcache_gc_list_length = ucs_list_length(&rcache->gc_list);
    ucs_spin_unlock(&rcache->lock);

    ucs_string_buffer_appendf(strb, "%lu\n", rcache_gc_list_length);
}

static void ucs_rcache_vfs_show_primitive(void *obj, ucs_string_buffer_t *strb,
                                          void *arg_ptr, uint64_t arg_u64)
{
    ucs_rcache_t *rcache = obj;

    pthread_rwlock_rdlock(&rcache->pgt_lock);
    ucs_vfs_show_primitive(obj, strb, arg_ptr, arg_u64);
    pthread_rwlock_unlock(&rcache->pgt_lock);
}

static void ucs_rcache_vfs_init_regions_distribution(ucs_rcache_t *rcache)
{
    size_t num_bins = ucs_rcache_distribution_get_num_bins();
    char buf[32], *bin_name;
    size_t i;

    for (i = 0; i < num_bins; ++i) {
        if (i != (num_bins - 1)) {
            bin_name = ucs_memunits_to_str(UCS_RCACHE_STAT_MIN_POW2 << i, buf,
                                           sizeof(buf));
        } else {
            bin_name = UCS_RCACHE_VFS_MAX_STR;
        }

        ucs_vfs_obj_add_ro_file(rcache, ucs_rcache_vfs_show_primitive,
                                &rcache->distribution[i].count,
                                UCS_VFS_TYPE_SIZET,
                                "regions_distribution/%s/count", bin_name);
        ucs_vfs_obj_add_ro_file(rcache, ucs_rcache_vfs_show_primitive,
                                &rcache->distribution[i].total_size,
                                UCS_VFS_TYPE_SIZET,
                                "regions_distribution/%s/total_size", bin_name);
    }
}

void ucs_rcache_vfs_init(ucs_rcache_t *rcache)
{
    ucs_vfs_obj_add_dir(NULL, rcache, "ucs/rcache/%s", rcache->name);
    ucs_vfs_obj_add_ro_file(rcache, ucs_vfs_show_primitive,
                            &rcache->num_regions, UCS_VFS_TYPE_ULONG,
                            "num_regions");
    ucs_vfs_obj_add_ro_file(rcache, ucs_vfs_show_primitive, &rcache->total_size,
                            UCS_VFS_TYPE_SIZET, "total_size");
    ucs_vfs_obj_add_ro_file(rcache, ucs_vfs_show_ulunits,
                            &rcache->params.max_regions, 0, "max_regions");
    ucs_vfs_obj_add_ro_file(rcache, ucs_vfs_show_memunits,
                            &rcache->params.max_size, 0, "max_size");
    ucs_vfs_obj_add_ro_file(rcache, ucs_rcache_vfs_read_inv_q_length, NULL, 0,
                            "inv_q/length");
    ucs_vfs_obj_add_ro_file(rcache, ucs_rcache_vfs_read_gc_list_length, NULL, 0,
                            "gc_list/length");

    ucs_rcache_vfs_init_regions_distribution(rcache);
}
