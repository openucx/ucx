/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

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

static UCS_F_ALWAYS_INLINE void
ucs_memtype_cache_insert(ucs_memtype_cache_t *memtype_cache, void *address,
                         size_t size, ucm_mem_type_t mem_type)
{
    ucs_memtype_cache_region_t *region;
    ucs_pgt_addr_t start, end;
    ucs_status_t status;

    ucs_trace("memtype_cache:insert address:%p length:%zu mem_type:%d",
              address, size, mem_type);

    pthread_rwlock_wrlock(&memtype_cache->lock);

    /* Align to page size */
    start  = ucs_align_down_pow2((uintptr_t)address, UCS_PGT_ADDR_ALIGN);
    end    = ucs_align_up_pow2  ((uintptr_t)address + size, UCS_PGT_ADDR_ALIGN);
    region = NULL;

    /* Allocate structure for new region */
    region = ucs_memalign(UCS_PGT_ENTRY_MIN_ALIGN, sizeof(ucs_memtype_cache_region_t),
                          "memtype_cache_region");
    if (region == NULL) {
        ucs_warn("failed to allocate memtype_cache region");
        goto out_unlock;
    }

    region->super.start = start;
    region->super.end   = end;
    region->mem_type    = mem_type;
    status = UCS_PROFILE_CALL(ucs_pgtable_insert, &memtype_cache->pgtable,
                              &region->super);
    if (status != UCS_OK) {
        ucs_error("failed to insert region " UCS_PGT_REGION_FMT ": %s",
                  UCS_PGT_REGION_ARG(&region->super), ucs_status_string(status));
        ucs_free(region);
        goto out_unlock;
    }

out_unlock:
    pthread_rwlock_unlock(&memtype_cache->lock);
}

static UCS_F_ALWAYS_INLINE void
ucs_memtype_cache_delete(ucs_memtype_cache_t *memtype_cache, void *address,
                         size_t size, ucm_mem_type_t mem_type)
{
    ucs_pgt_addr_t start = (uintptr_t)address;
    ucs_pgt_region_t *pgt_region;
    ucs_memtype_cache_region_t *region;
    ucs_status_t status;

    ucs_trace("memtype_cache:delete address:%p length:%zu mem_type:%d",
              address, size, mem_type);

    pthread_rwlock_rdlock(&memtype_cache->lock);

    pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup, &memtype_cache->pgtable, start);
    assert(pgt_region != NULL);

    region = ucs_derived_of(pgt_region, ucs_memtype_cache_region_t);

    status = ucs_pgtable_remove(&memtype_cache->pgtable, &region->super);
    if (status != UCS_OK) {
        ucs_warn("failed to remove address:%p from memtype_cache", address);
    }
    ucs_free(region);
    pthread_rwlock_unlock(&memtype_cache->lock);
}

static void ucs_memtype_cache_event_callback(ucm_event_type_t event_type,
                                              ucm_event_t *event, void *arg)
{
    ucs_memtype_cache_t *memtype_cache = arg;

    if (event_type & UCM_EVENT_MEM_TYPE_ALLOC) {
        ucs_memtype_cache_insert(memtype_cache, event->mem_type.address,
                                 event->mem_type.size, event->mem_type.mem_type);
    } else if (event_type & UCM_EVENT_MEM_TYPE_FREE) {
        ucs_memtype_cache_delete(memtype_cache, event->mem_type.address,
                                 event->mem_type.size, event->mem_type.mem_type);
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

static void ucs_memtype_cache_purge(ucs_memtype_cache_t *memtype_cache)
{
    ucs_memtype_cache_region_t *region, *tmp;
    ucs_list_link_t region_list;

    ucs_trace_func("memtype_cache purge");

    ucs_list_head_init(&region_list);
    ucs_pgtable_purge(&memtype_cache->pgtable, ucs_memtype_cache_region_collect_callback,
                      &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        ucs_warn("destroying inuse address:%p ", (void *)region->super.start);
        ucs_free(region);
    }
}

UCS_PROFILE_FUNC(ucs_status_t, ucs_memtype_cache_lookup,
                 (memtype_cache, address, length, ucm_mem_type),
                 ucs_memtype_cache_t *memtype_cache, void *address,
                 size_t length, ucm_mem_type_t *ucm_mem_type)
{
    ucs_pgt_addr_t start = (uintptr_t)address;
    ucs_pgt_region_t *pgt_region;
    ucs_memtype_cache_region_t *region;
    ucs_status_t status;

    pthread_rwlock_rdlock(&memtype_cache->lock);

    pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup, &memtype_cache->pgtable, start);
    if (pgt_region && pgt_region->end >= (start + length)) {
        region = ucs_derived_of(pgt_region, ucs_memtype_cache_region_t);
        *ucm_mem_type = region->mem_type;
        status = UCS_OK;
        goto out_unlock;
    }
    status = UCS_ERR_NO_ELEM;
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
