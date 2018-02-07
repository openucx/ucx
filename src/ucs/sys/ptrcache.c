/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ptrcache.h"

#include <ucs/arch/atomic.h>
#include <ucs/type/class.h>
#include <ucs/datastruct/queue.h>
#include <ucs/debug/log.h>
#include <ucs/debug/profile.h>
#include <ucs/debug/memtrack.h>
#include <ucs/stats/stats.h>
#include <ucs/sys/sys.h>
#include <ucm/api/ucm.h>


static ucs_pgt_dir_t *ucs_ptrcache_pgt_dir_alloc(const ucs_pgtable_t *pgtable)
{
    return ucs_memalign(UCS_PGT_ENTRY_MIN_ALIGN, sizeof(ucs_pgt_dir_t),
                        "ptrcache_pgdir");
}

static void ucs_ptrcache_pgt_dir_release(const ucs_pgtable_t *pgtable,
                                       ucs_pgt_dir_t *dir)
{
    ucs_free(dir);
}

static void ucs_ptrcache_insert_callback(ucm_event_type_t event_type, ucm_event_t *event,
                                         void *arg)
{
    ucs_ptrcache_t *ptrcache = arg;
    ucs_ptrcache_region_t *region;
    ucs_pgt_addr_t start, end;
    ucs_status_t status;

    ucs_trace("ptrcache:insert address:%p length:%zu mem_type:%d", event->mem_type.address,
              event->mem_type.size, event->mem_type.mem_type);

    pthread_rwlock_wrlock(&ptrcache->lock);

    /* Align to page size */
    start  = ucs_align_down_pow2((uintptr_t)event->mem_type.address, UCS_PGT_ADDR_ALIGN);
    end    = ucs_align_up_pow2  ((uintptr_t)event->mem_type.address +
                                 event->mem_type.size, UCS_PGT_ADDR_ALIGN);
    region = NULL;

    /* Allocate structure for new region */
    region = ucs_memalign(UCS_PGT_ENTRY_MIN_ALIGN, sizeof(ucs_ptrcache_region_t),
                          "ptrcache_region");
    if (region == NULL) {
        ucs_warn("failed to allocate ptrcache region");
        goto out_unlock;
    }

    region->super.start = start;
    region->super.end   = end;
    region->mem_type    = event->mem_type.mem_type;
    status = UCS_PROFILE_CALL(ucs_pgtable_insert, &ptrcache->pgtable, &region->super);
    if (status != UCS_OK) {
        ucs_error("failed to insert region " UCS_PGT_REGION_FMT ": %s",
                  UCS_PGT_REGION_ARG(&region->super), ucs_status_string(status));
        ucs_free(region);
        goto out_unlock;
    }

out_unlock:
    pthread_rwlock_unlock(&ptrcache->lock);
}

static void ucs_ptrcache_delete_callback(ucm_event_type_t event_type, ucm_event_t *event,
                                         void *arg)
{
    ucs_ptrcache_t *ptrcache = arg;
    ucs_pgt_addr_t start = (uintptr_t)event->mem_type.address;
    ucs_pgt_region_t *pgt_region;
    ucs_ptrcache_region_t *region;
    ucs_status_t status;

    ucs_trace("ptrcache:delete address:%p length:%zu mem_type:%d", event->mem_type.address,
                   event->mem_type.size, event->mem_type.mem_type);

    pthread_rwlock_rdlock(&ptrcache->lock);

    pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup, &ptrcache->pgtable, start);
    assert(pgt_region != NULL);
    region = ucs_derived_of(pgt_region, ucs_ptrcache_region_t);
    assert(region->mem_type == event->mem_type.mem_type);
    status = ucs_pgtable_remove(&ptrcache->pgtable, &region->super);
    if (status != UCS_OK) {
        ucs_warn("failed to remove address:%p from ptrcache", event->mem_type.address);
    }
    pthread_rwlock_unlock(&ptrcache->lock);
}

static void ucs_ptrcache_region_collect_callback(const ucs_pgtable_t *pgtable,
                                                 ucs_pgt_region_t *pgt_region, void *arg)
{
    ucs_ptrcache_region_t *region = ucs_derived_of(pgt_region, ucs_ptrcache_region_t);
    ucs_list_link_t *list = arg;
    ucs_list_add_tail(list, &region->list);
}

static void ucs_ptrcache_purge(ucs_ptrcache_t *ptrcache)
{
    ucs_ptrcache_region_t *region, *tmp;
    ucs_list_link_t region_list;

    ucs_trace_func("rcache=%s", ptrcache->name);

    ucs_list_head_init(&region_list);
    ucs_pgtable_purge(&ptrcache->pgtable, ucs_ptrcache_region_collect_callback,
                      &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        ucs_warn("destroying inuse address:%p ", (void *)region->super.start);
        ucs_free(region);
    }
}

int ucs_ptrcache_lookup(ucs_ptrcache_t *ptrcache, void *address, size_t length,
                        ucm_mem_type_t *ucm_mem_type)
{
    ucs_pgt_addr_t start = (uintptr_t)address;
    ucs_pgt_region_t *pgt_region;
    ucs_ptrcache_region_t *region;
    int ret;

    pthread_rwlock_rdlock(&ptrcache->lock);

    pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup, &ptrcache->pgtable, start);
    if (pgt_region && pgt_region->end >= (start + length)) {
        region = ucs_derived_of(pgt_region, ucs_ptrcache_region_t);
        *ucm_mem_type = region->mem_type;
        ret = 1;
        goto out_unlock;
    }
    ret = 0;
out_unlock:
    pthread_rwlock_unlock(&ptrcache->lock);
    return ret;
}

static UCS_CLASS_INIT_FUNC(ucs_ptrcache_t, const char *name)
{
    ucs_status_t status;
    int ret;

    self->name = strdup(name);
    if (self->name == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    ret = pthread_rwlock_init(&self->lock, NULL);
    if (ret) {
        ucs_error("pthread_rwlock_init() failed: %m");
        status = UCS_ERR_INVALID_PARAM;
        goto err_free_name;
    }

    status = ucs_pgtable_init(&self->pgtable, ucs_ptrcache_pgt_dir_alloc,
                              ucs_ptrcache_pgt_dir_release);
    if (status != UCS_OK) {
        goto err_destroy_rwlock;
    }

    status = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_ALLOC, 1000,
                                   ucs_ptrcache_insert_callback, self);
    if (status != UCS_OK) {
        goto err_cleanup_pgtable;
    }

    status = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_FREE, 1000,
                                   ucs_ptrcache_delete_callback, self);
    if (status != UCS_OK) {
        goto err_cleanup_pgtable;
    }
    return UCS_OK;

err_cleanup_pgtable:
    ucs_pgtable_cleanup(&self->pgtable);
err_destroy_rwlock:
    pthread_rwlock_destroy(&self->lock);
err_free_name:
    free(self->name);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(ucs_ptrcache_t)
{
    ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_FREE, ucs_ptrcache_delete_callback, self);
    ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_ALLOC, ucs_ptrcache_insert_callback, self);
    ucs_ptrcache_purge(self);
    ucs_pgtable_cleanup(&self->pgtable);
    pthread_rwlock_destroy(&self->lock);
    free(self->name);
}

UCS_CLASS_DEFINE(ucs_ptrcache_t, void);
UCS_CLASS_DEFINE_NAMED_NEW_FUNC(ucs_ptrcache_create, ucs_ptrcache_t, ucs_ptrcache_t,
                                const char *)
UCS_CLASS_DEFINE_NAMED_DELETE_FUNC(ucs_ptrcache_destroy, ucs_ptrcache_t, ucs_ptrcache_t)
