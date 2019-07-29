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
    UCS_MEMTYPE_CACHE_ATION_SET_MEMTYPE,
    UCS_MEMTYPE_CACHE_ATION_REMOVE
} ucs_memtype_cache_action_t;

static ucs_pgt_dir_t *ucs_memtype_cache_pgt_dir_alloc(const ucs_pgtable_t *pgtable)
{
    return ucs_memalign(UCS_PGT_ENTRY_MIN_ALIGN, sizeof(ucs_pgt_dir_t),
                        "memtype_cache_pgdir");
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
                                     ucs_memory_type_t mem_type)
{
    ucs_memtype_cache_region_t *region;
    ucs_status_t status;

    ucs_trace("memtype_cache: insert 0x%lx..0x%lx mem_type %d", start, end,
              mem_type);

    /* Allocate structure for new region */
    region = ucs_memalign(UCS_PGT_ENTRY_MIN_ALIGN,
                          sizeof(ucs_memtype_cache_region_t),
                          "memtype_cache_region");
    if (region == NULL) {
        ucs_warn("failed to allocate memtype_cache region");
        return;
    }

    ucs_assert((start % ucs_get_page_size()) == 0);
    ucs_assert((end   % ucs_get_page_size()) == 0);

    region->super.start = start;
    region->super.end   = end;
    region->mem_type    = mem_type;

    status = UCS_PROFILE_CALL(ucs_pgtable_insert, &memtype_cache->pgtable,
                              &region->super);
    if (status != UCS_OK) {
        ucs_error("failed to insert region " UCS_PGT_REGION_FMT ": %s",
                  UCS_PGT_REGION_ARG(&region->super), ucs_status_string(status));
        ucs_free(region);
    }
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
                      (memtype_cache, address, size, mem_type, action),
                      ucs_memtype_cache_t *memtype_cache, void *address,
                      size_t size, ucs_memory_type_t mem_type,
                      ucs_memtype_cache_action_t action)
{
    const size_t page_size = ucs_get_page_size();
    ucs_memtype_cache_region_t *region, *tmp;
    UCS_LIST_HEAD(region_list);
    ucs_pgt_addr_t start, end;
    ucs_status_t status;

    start = ucs_align_down_pow2((uintptr_t)address,        page_size);
    end   = ucs_align_up_pow2  ((uintptr_t)address + size, page_size);

    pthread_rwlock_wrlock(&memtype_cache->lock);

    /* find and remove all regions which intersect with new one */
    ucs_pgtable_search_range(&memtype_cache->pgtable, start, end - 1,
                             ucs_memtype_cache_region_collect_callback,
                             &region_list);
    ucs_list_for_each(region, &region_list, list) {
        status = ucs_pgtable_remove(&memtype_cache->pgtable, &region->super);
        if (status != UCS_OK) {
            ucs_error("failed to remove address:%p from memtype_cache", address);
            goto out_unlock;
        }
        ucs_trace("memtype_cache: removed 0x%lx..0x%lx mem_type %d",
                  region->super.start, region->super.end, region->mem_type);
    }

    if (action == UCS_MEMTYPE_CACHE_ATION_SET_MEMTYPE) {
        ucs_memtype_cache_insert(memtype_cache, start, end, mem_type);
    }

    /* slice old regions by the new region, to preserve the previous memory type
     * of the non-overlapping parts
     */
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        if (start > region->super.start) {
            /* create previous region */
            ucs_memtype_cache_insert(memtype_cache, region->super.start, start,
                                     region->mem_type);
        }
        if (end < region->super.end) {
            /* create next region */
            ucs_memtype_cache_insert(memtype_cache, end, region->super.end,
                                     region->mem_type);
        }

        ucs_free(region);
    }

out_unlock:
    pthread_rwlock_wrlock(&memtype_cache->lock);
}

void ucs_memtype_cache_update(ucs_memtype_cache_t *memtype_cache, void *address,
                              size_t size, ucs_memory_type_t mem_type)
{
    ucs_memtype_cache_update_internal(memtype_cache, address, size, mem_type,
                                      UCS_MEMTYPE_CACHE_ATION_SET_MEMTYPE);
}

static void ucs_memtype_cache_event_callback(ucm_event_type_t event_type,
                                              ucm_event_t *event, void *arg)
{
    ucs_memtype_cache_t *memtype_cache = arg;
    ucs_memtype_cache_action_t action;

    if (event_type & UCM_EVENT_MEM_TYPE_ALLOC) {
        action = UCS_MEMTYPE_CACHE_ATION_SET_MEMTYPE;
    } else if (event_type & UCM_EVENT_MEM_TYPE_FREE) {
        action = UCS_MEMTYPE_CACHE_ATION_REMOVE;
    } else {
        return;
    }

    ucs_memtype_cache_update_internal(memtype_cache, event->mem_type.address,
                                      event->mem_type.size,
                                      event->mem_type.mem_type, action);
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
                 (memtype_cache, address, size, mem_type_p),
                 ucs_memtype_cache_t *memtype_cache, void *address,
                 size_t size, ucs_memory_type_t *mem_type_p)
{
    const ucs_pgt_addr_t start = (uintptr_t)address;
    ucs_memtype_cache_region_t *region;
    ucs_pgt_region_t *pgt_region;
    ucs_status_t status;

    pthread_rwlock_rdlock(&memtype_cache->lock);

    pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup, &memtype_cache->pgtable,
                                  start);
    if ((pgt_region == NULL) || (pgt_region->end < (start + size))) {
        status = UCS_ERR_NO_ELEM;
        goto out_unlock;
    }

    region      = ucs_derived_of(pgt_region, ucs_memtype_cache_region_t);
    *mem_type_p = region->mem_type;
    status      = UCS_OK;

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

    status = ucm_set_event_handler((UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE),
                                   1000, ucs_memtype_cache_event_callback, self);
    if (status != UCS_OK) {
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
