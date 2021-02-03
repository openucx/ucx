/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "memtype_cache.h"

#include <ucs/arch/atomic.h>
#include <ucs/type/class.h>
#include <ucs/datastruct/queue.h>
#include <ucs/debug/log.h>
#include <ucs/profile/profile.h>
#include <ucs/debug/memtrack.h>
#include <ucs/stats/stats.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/math.h>
#include <ucm/api/ucm.h>


typedef enum {
    UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE,
    UCS_MEMTYPE_CACHE_ACTION_REMOVE
} ucs_memtype_cache_action_t;


static UCS_F_ALWAYS_INLINE void
ucs_memory_info_set_unknown(ucs_memory_info_t *mem_info)
{
    mem_info->type    = UCS_MEMORY_TYPE_UNKNOWN;
    mem_info->sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
}

static ucs_pgt_dir_t *ucs_memtype_cache_pgt_dir_alloc(const ucs_pgtable_t *pgtable)
{
    void *ptr;
    int ret;

    ret = ucs_posix_memalign(&ptr,
                             ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                             sizeof(ucs_pgt_dir_t), "memtype_cache_pgdir");
    return (ret == 0) ? ptr : NULL;
}

static void ucs_memtype_cache_pgt_dir_release(const ucs_pgtable_t *pgtable,
                                              ucs_pgt_dir_t *dir)
{
    ucs_free(dir);
}

/*
 * - Lock must be held in write mode
 * - start, end must be aligned to page size
 */
static void ucs_memtype_cache_insert(ucs_memtype_cache_t *memtype_cache,
                                     ucs_pgt_addr_t start, ucs_pgt_addr_t end,
                                     const ucs_memory_info_t *mem_info)
{
    ucs_memtype_cache_region_t *region;
    ucs_status_t status;
    char dev_name[64];
    int ret;

    /* Allocate structure for new region */
    ret = ucs_posix_memalign((void **)&region,
                             ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                             sizeof(ucs_memtype_cache_region_t),
                             "memtype_cache_region");
    if (ret != 0) {
        ucs_warn("failed to allocate memtype_cache region");
        return;
    }

    ucs_assert((start % UCS_PGT_ADDR_ALIGN) == 0);
    ucs_assert((end   % UCS_PGT_ADDR_ALIGN) == 0);

    region->super.start = start;
    region->super.end   = end;
    region->mem_info    = *mem_info;

    status = UCS_PROFILE_CALL(ucs_pgtable_insert, &memtype_cache->pgtable,
                              &region->super);
    if (status != UCS_OK) {
        ucs_error("failed to insert region " UCS_PGT_REGION_FMT ": %s",
                  UCS_PGT_REGION_ARG(&region->super), ucs_status_string(status));
        ucs_free(region);
        return;
    }

    ucs_trace("memtype_cache: insert " UCS_PGT_REGION_FMT " mem_type %s dev %s",
              UCS_PGT_REGION_ARG(&region->super),
              ucs_memory_type_names[mem_info->type],
              ucs_topo_sys_device_bdf_name(mem_info->sys_dev, dev_name,
                                           sizeof(dev_name)));
}

static void ucs_memtype_cache_region_collect_callback(const ucs_pgtable_t *pgtable,
                                                      ucs_pgt_region_t *pgt_region,
                                                      void *arg)
{
    ucs_memtype_cache_region_t *region = ucs_derived_of(pgt_region,
                                                        ucs_memtype_cache_region_t);
    ucs_list_link_t *list = arg;
    ucs_list_add_tail(list, &region->list);
}

UCS_PROFILE_FUNC_VOID(ucs_memtype_cache_update_internal,
                      (memtype_cache, address, size, mem_info, action),
                      ucs_memtype_cache_t *memtype_cache, const void *address,
                      size_t size, const ucs_memory_info_t *mem_info,
                      ucs_memtype_cache_action_t action)
{
    ucs_memtype_cache_region_t *region, *tmp;
    UCS_LIST_HEAD(region_list);
    ucs_pgt_addr_t start, end, search_start, search_end;
    ucs_status_t status;
    char dev_name[64];

    if (!size) {
        return;
    }

    start = ucs_align_down_pow2((uintptr_t)address,        UCS_PGT_ADDR_ALIGN);
    end   = ucs_align_up_pow2  ((uintptr_t)address + size, UCS_PGT_ADDR_ALIGN);

    ucs_trace("%s: [0x%lx..0x%lx] mem_type %s dev %s",
              (action == UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE) ? "update" :
                                                                 "remove",
              start, end, ucs_memory_type_names[mem_info->type],
              ucs_topo_sys_device_bdf_name(mem_info->sys_dev, dev_name,
                                           sizeof(dev_name)));

    if (action == UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE) {
        /* try to find regions that are contiguous and instersected
         * with current one */
        search_start = start - 1;
        search_end   = end;
    } else {
        /* try to find regions that are instersected with current one */
        search_start = start;
        search_end   = end - 1;
    }

    pthread_rwlock_wrlock(&memtype_cache->lock);

    /* find and remove all regions which intersect with new one */
    ucs_pgtable_search_range(&memtype_cache->pgtable, search_start, search_end,
                             ucs_memtype_cache_region_collect_callback,
                             &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        if (action == UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE) {
            if (region->mem_info.type == mem_info->type) {
                /* merge current region with overlapping or adjacent regions
                 * of same memory type */
                start = ucs_min(start, region->super.start);
                end   = ucs_max(end, region->super.end);
            } else if ((region->super.end < start) ||
                       (region->super.start >= end)) {
                /* ignore regions which are not really overlapping and can't
                 * be merged because of different memory types */
                ucs_list_del(&region->list);
                continue;
            }
        }

        status = ucs_pgtable_remove(&memtype_cache->pgtable, &region->super);
        if (status != UCS_OK) {
            ucs_error("failed to remove " UCS_PGT_REGION_FMT
                      " from memtype_cache: %s",
                      UCS_PGT_REGION_ARG(&region->super),
                      ucs_status_string(status));
            goto out_unlock;
        }

        ucs_trace("memtype_cache: removed " UCS_PGT_REGION_FMT " %s dev %s",
                  UCS_PGT_REGION_ARG(&region->super),
                  ucs_memory_type_names[region->mem_info.type],
                  ucs_topo_sys_device_bdf_name(region->mem_info.sys_dev,
                                               dev_name, sizeof(dev_name)));
    }

    if (action == UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE) {
        ucs_memtype_cache_insert(memtype_cache, start, end, mem_info);
    }

    /* slice old regions by the new region, to preserve the previous memory type
     * of the non-overlapping parts
     */
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        if (start > region->super.start) {
            /* create previous region */
            ucs_memtype_cache_insert(memtype_cache, region->super.start, start,
                                     &region->mem_info);
        }
        if (end < region->super.end) {
            /* create next region */
            ucs_memtype_cache_insert(memtype_cache, end, region->super.end,
                                     &region->mem_info);
        }

        ucs_free(region);
    }

out_unlock:
    pthread_rwlock_unlock(&memtype_cache->lock);
}

void ucs_memtype_cache_update(ucs_memtype_cache_t *memtype_cache,
                              const void *address, size_t size,
                              const ucs_memory_info_t *mem_info)
{
    ucs_memtype_cache_update_internal(memtype_cache, address, size, mem_info,
                                      UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE);
}

void ucs_memtype_cache_remove(ucs_memtype_cache_t *memtype_cache,
                              const void *address, size_t size)
{
    ucs_memory_info_t mem_info;

    ucs_memory_info_set_unknown(&mem_info);
    ucs_memtype_cache_update_internal(memtype_cache, address, size, &mem_info,
                                      UCS_MEMTYPE_CACHE_ACTION_REMOVE);
}

static void ucs_memtype_cache_event_callback(ucm_event_type_t event_type,
                                              ucm_event_t *event, void *arg)
{
    ucs_memtype_cache_t *memtype_cache = arg;
    ucs_memory_info_t mem_info         = {
        .type    = event->mem_type.mem_type,
        .sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN
    };
    ucs_memtype_cache_action_t action;

    if (event_type & UCM_EVENT_MEM_TYPE_ALLOC) {
        action = UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE;
    } else if (event_type & UCM_EVENT_MEM_TYPE_FREE) {
        action = UCS_MEMTYPE_CACHE_ACTION_REMOVE;
    } else {
        return;
    }

    ucs_memtype_cache_update_internal(memtype_cache, event->mem_type.address,
                                      event->mem_type.size, &mem_info, action);
}

static void ucs_memtype_cache_purge(ucs_memtype_cache_t *memtype_cache)
{
    ucs_memtype_cache_region_t *region, *tmp;
    UCS_LIST_HEAD(region_list);

    ucs_trace_func("memtype_cache purge");

    ucs_pgtable_purge(&memtype_cache->pgtable,
                      ucs_memtype_cache_region_collect_callback, &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        ucs_free(region);
    }
}

UCS_PROFILE_FUNC(ucs_status_t, ucs_memtype_cache_lookup,
                 (memtype_cache, address, size, mem_info),
                 ucs_memtype_cache_t *memtype_cache, const void *address,
                 size_t size, ucs_memory_info_t *mem_info)
{
    const ucs_pgt_addr_t start = (uintptr_t)address;
    ucs_memtype_cache_region_t *region;
    ucs_pgt_region_t *pgt_region;
    ucs_status_t status;

    pthread_rwlock_rdlock(&memtype_cache->lock);

    pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup, &memtype_cache->pgtable,
                                  start);
    if (pgt_region == NULL) {
        status = UCS_ERR_NO_ELEM;
        goto out_unlock;
    }

    if (ucs_likely((start + size) <= pgt_region->end)) {
        region    = ucs_derived_of(pgt_region, ucs_memtype_cache_region_t);
        *mem_info = region->mem_info;
    } else {
        ucs_memory_info_set_unknown(mem_info);
    }
    status = UCS_OK;

out_unlock:
    pthread_rwlock_unlock(&memtype_cache->lock);
    return status;
}

static UCS_CLASS_INIT_FUNC(ucs_memtype_cache_t)
{
    ucs_status_t status;
    int ret;

    ret = pthread_rwlock_init(&self->lock, NULL);
    if (ret) {
        ucs_error("pthread_rwlock_init() failed: %m");
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    status = ucs_pgtable_init(&self->pgtable, ucs_memtype_cache_pgt_dir_alloc,
                              ucs_memtype_cache_pgt_dir_release);
    if (status != UCS_OK) {
        goto err_destroy_rwlock;
    }

    status = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_ALLOC |
                                   UCM_EVENT_MEM_TYPE_FREE |
                                   UCM_EVENT_FLAG_EXISTING_ALLOC,
                                   1000, ucs_memtype_cache_event_callback,
                                   self);
    if ((status != UCS_OK) && (status != UCS_ERR_UNSUPPORTED)) {
        ucs_error("failed to set UCM memtype event handler: %s",
                  ucs_status_string(status));
        goto err_cleanup_pgtable;
    }

    return UCS_OK;

err_cleanup_pgtable:
    ucs_pgtable_cleanup(&self->pgtable);
err_destroy_rwlock:
    pthread_rwlock_destroy(&self->lock);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(ucs_memtype_cache_t)
{
    ucm_unset_event_handler((UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE),
                            ucs_memtype_cache_event_callback, self);
    ucs_memtype_cache_purge(self);
    ucs_pgtable_cleanup(&self->pgtable);
    pthread_rwlock_destroy(&self->lock);
}

UCS_CLASS_DEFINE(ucs_memtype_cache_t, void);
UCS_CLASS_DEFINE_NAMED_NEW_FUNC(ucs_memtype_cache_create, ucs_memtype_cache_t,
                                ucs_memtype_cache_t)
UCS_CLASS_DEFINE_NAMED_DELETE_FUNC(ucs_memtype_cache_destroy, ucs_memtype_cache_t,
                                   ucs_memtype_cache_t)
