/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "rcache.h"

#include <ucs/arch/atomic.h>
#include <ucs/type/class.h>
#include <ucs/datastruct/queue.h>
#include <ucs/debug/log.h>
#include <ucs/debug/profile.h>
#include <ucs/debug/memtrack.h>
#include <ucs/stats/stats.h>
#include <ucs/sys/sys.h>
#include <ucm/api/ucm.h>


#define ucs_rcache_region_log(_level, _message, ...) \
    do { \
        if (ucs_log_is_enabled(_level)) { \
            __ucs_rcache_region_log(__FILE__, __LINE__, __FUNCTION__, (_level), \
                                    _message, ## __VA_ARGS__); \
        } \
    } while (0)

#define ucs_rcache_region_error(_message, ...) \
    ucs_rcache_region_log(UCS_LOG_LEVEL_ERROR, _message, ## __VA_ARGS__)
#define ucs_rcache_region_warn(_message, ...)  \
    ucs_rcache_region_log(UCS_LOG_LEVEL_WARN, _message,  ## __VA_ARGS__)
#define ucs_rcache_region_debug(_message, ...) \
    ucs_rcache_region_log(UCS_LOG_LEVEL_DEBUG, _message, ##  __VA_ARGS__)
#define ucs_rcache_region_trace(_message, ...) \
    ucs_rcache_region_log(UCS_LOG_LEVEL_TRACE, _message, ## __VA_ARGS__)


typedef struct ucs_rcache_inv_entry {
    ucs_queue_elem_t         queue;
    ucs_pgt_addr_t           start;
    ucs_pgt_addr_t           end;
} ucs_rcache_inv_entry_t;


static void __ucs_rcache_region_log(const char *file, int line, const char *function,
                                    ucs_log_level_t level, ucs_rcache_t *rcache,
                                    ucs_rcache_region_t *region, const char *fmt,
                                    ...)
{
    char message[128];
    char region_desc[64];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(message, sizeof(message), fmt, ap);
    va_end(ap);

    if (region->flags & UCS_RCACHE_REGION_FLAG_REGISTERED) {
        rcache->params.ops->dump_region(rcache->params.context, rcache, region,
                                        region_desc, sizeof(region_desc));
    } else {
        strcpy(region_desc, "");
    }

    ucs_log_dispatch(file, line, function, level,
                     "%s: %s region " UCS_PGT_REGION_FMT " %c%c "UCS_RCACHE_PROT_FMT" ref %u %s",
                     rcache->name, message,
                     UCS_PGT_REGION_ARG(&region->super),
                     (region->flags & UCS_RCACHE_REGION_FLAG_REGISTERED) ? 'g' : '-',
                     (region->flags & UCS_RCACHE_REGION_FLAG_PGTABLE)    ? 't' : '-',
                     UCS_RCACHE_PROT_ARG(region->prot),
                     region->refcount,
                     region_desc);
}

static ucs_pgt_dir_t *ucs_rcache_pgt_dir_alloc(const ucs_pgtable_t *pgtable)
{
    return ucs_memalign(UCS_PGT_ENTRY_MIN_ALIGN, sizeof(ucs_pgt_dir_t),
                        "rcache_pgdir");
}

static void ucs_rcache_pgt_dir_release(const ucs_pgtable_t *pgtable,
                                       ucs_pgt_dir_t *dir)
{
    ucs_free(dir);
}

static ucs_status_t ucs_rcache_mp_chunk_alloc(ucs_mpool_t *mp, size_t *size_p,
                                              void **chunk_p)
{
    size_t size;
    void *ptr;

    size = ucs_align_up_pow2(sizeof(size_t) + *size_p, ucs_get_page_size());
    ptr = ucm_orig_mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS,
                        -1, 0);
    if (ptr == MAP_FAILED) {
        ucs_error("mmmap(size=%zu) failed: %m", size);
        return UCS_ERR_NO_MEMORY;
    }

    /* Store the size in the first bytes of the chunk */
    *(size_t*)ptr = size;
    *chunk_p = ptr  + sizeof(size_t);
    *size_p  = size - sizeof(size_t);
    return UCS_OK;
}

static void ucs_rcache_mp_chunk_release(ucs_mpool_t *mp, void *chunk)
{
    size_t size;
    void *ptr;
    int ret;

    ptr = chunk - sizeof(size_t);
    size = *(size_t*)ptr;
    ret = ucm_orig_munmap(ptr, size);
    if (ret) {
        ucs_warn("munmap(%p, %zu) failed: %m", ptr, size);
    }
}

static ucs_mpool_ops_t ucs_rcache_mp_ops = {
    .chunk_alloc   = ucs_rcache_mp_chunk_alloc,
    .chunk_release = ucs_rcache_mp_chunk_release,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};


/* Lock must be held */
static void ucs_rcache_region_collect_callback(const ucs_pgtable_t *pgtable,
                                               ucs_pgt_region_t *pgt_region, void *arg)
{
    ucs_rcache_region_t *region = ucs_derived_of(pgt_region, ucs_rcache_region_t);
    ucs_list_link_t *list = arg;
    ucs_list_add_tail(list, &region->list);
}

/* Lock must be held */
static void ucs_rcache_find_regions(ucs_rcache_t *rcache, ucs_pgt_addr_t from,
                                    ucs_pgt_addr_t to, ucs_list_link_t *list)
{
    ucs_list_head_init(list);
    ucs_pgtable_search_range(&rcache->pgtable, from, to,
                             ucs_rcache_region_collect_callback, list);
}

/* Lock must be held in write mode */
static void ucs_mem_region_destroy_internal(ucs_rcache_t *rcache,
                                            ucs_rcache_region_t *region)
{
    ucs_rcache_region_trace(rcache, region, "destroy");

    ucs_assert(region->refcount == 0);
    ucs_assert(!(region->flags & UCS_RCACHE_REGION_FLAG_PGTABLE));

    if (region->flags & UCS_RCACHE_REGION_FLAG_REGISTERED) {
        UCS_PROFILE_CODE("mem_dereg") {
            rcache->params.ops->mem_dereg(rcache->params.context, rcache, region);
        }
    }

    ucs_free(region);
}

static inline void ucs_rcache_region_put_internal(ucs_rcache_t *rcache,
                                                  ucs_rcache_region_t *region,
                                                  int lock,
                                                  int must_be_destroyed)
{
    ucs_rcache_region_trace(rcache, region, lock ? "put" : "put_nolock");

    ucs_assert(region->refcount > 0);
    if (ucs_unlikely(ucs_atomic_fadd32(&region->refcount, -1) == 1)) {
        if (lock) {
            pthread_rwlock_wrlock(&rcache->lock);
        }
        ucs_mem_region_destroy_internal(rcache, region);
        if (lock) {
            pthread_rwlock_unlock(&rcache->lock);
        }
    } else {
        ucs_assert(!must_be_destroyed);
    }
}

/* Lock must be held in write mode */
static void ucs_rcache_region_invalidate(ucs_rcache_t *rcache,
                                         ucs_rcache_region_t *region,
                                         int must_be_in_pgt,
                                         int must_be_destroyed)
{
    ucs_status_t status;

    ucs_rcache_region_trace(rcache, region, "invalidate");

    /* Remove the memory region from page table, if it's there */
    if (region->flags & UCS_RCACHE_REGION_FLAG_PGTABLE) {
        status = ucs_pgtable_remove(&rcache->pgtable, &region->super);
        if (status != UCS_OK) {
            ucs_rcache_region_warn(rcache, region, "failed to remove (%s)",
                                   ucs_status_string(status));
        }
        region->flags &= ~UCS_RCACHE_REGION_FLAG_PGTABLE;
    } else {
        ucs_assert(!must_be_in_pgt);
    }

     ucs_rcache_region_put_internal(rcache, region, 0, must_be_destroyed);
}

/* Lock must be held in write mode */
static void ucs_rcache_invalidate_range(ucs_rcache_t *rcache, ucs_pgt_addr_t start,
                                        ucs_pgt_addr_t end)
{
    ucs_rcache_region_t *region, *tmp;
    ucs_list_link_t region_list;

    ucs_trace_func("rcache=%s, start=0x%lx, end=0x%lx", rcache->name, start, end);

    ucs_rcache_find_regions(rcache, start, end - 1, &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        /* all regions on the list are in the page table */
        ucs_rcache_region_invalidate(rcache, region, 1, 0);
    }
}

/* Lock must be held in write mode */
static void ucs_rcache_check_inv_queue(ucs_rcache_t *rcache)
{
    ucs_rcache_inv_entry_t *entry;

    ucs_trace_func("rcache=%s", rcache->name);

    pthread_spin_lock(&rcache->inv_lock);
    while (!ucs_queue_is_empty(&rcache->inv_q)) {
        entry = ucs_queue_pull_elem_non_empty(&rcache->inv_q,
                                              ucs_rcache_inv_entry_t, queue);

        /* We need to drop the lock since the following code may trigger memory
         * operations, which could trigger vm_unmapped event which also takes
         * this lock.
         */
        pthread_spin_unlock(&rcache->inv_lock);

        ucs_rcache_invalidate_range(rcache, entry->start, entry->end);

        pthread_spin_lock(&rcache->inv_lock);

        ucs_mpool_put(entry); /* Must be done with the lock held */
    }
    pthread_spin_unlock(&rcache->inv_lock);
}

static void ucs_rcache_unmapped_callback(ucm_event_type_t event_type,
                                         ucm_event_t *event, void *arg)
{
    ucs_rcache_t *rcache = arg;
    ucs_rcache_inv_entry_t *entry;
    ucs_pgt_addr_t start, end;

    ucs_assert(event_type == UCM_EVENT_VM_UNMAPPED);

    start = (uintptr_t)event->vm_unmapped.address;
    end   = (uintptr_t)event->vm_unmapped.address + event->vm_unmapped.size;
    ucs_trace_func("%s: event vm_unmapped 0x%lx..0x%lx", rcache->name, start, end);

    pthread_spin_lock(&rcache->inv_lock);
    entry = ucs_mpool_get(&rcache->inv_mp);
    if (entry != NULL) {
        /* Add region to invalidation list */
        entry->start = start;
        entry->end   = end;
        ucs_queue_push(&rcache->inv_q, &entry->queue);
    } else {
        ucs_error("Failed to allocate invalidation entry for 0x%lx..0x%lx, "
                  "data corruption may occur", start, end);
    }
    pthread_spin_unlock(&rcache->inv_lock);
}

/* Clear all regions
   Lock must be held in write mode (or use it during cleanup)
 */
static void ucs_rcache_purge(ucs_rcache_t *rcache)
{
    ucs_rcache_region_t *region, *tmp;
    ucs_list_link_t region_list;

    ucs_trace_func("rcache=%s", rcache->name);

    ucs_list_head_init(&region_list);
    ucs_pgtable_purge(&rcache->pgtable, ucs_rcache_region_collect_callback,
                      &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        if (region->flags & UCS_RCACHE_REGION_FLAG_PGTABLE) {
            region->flags &= ~UCS_RCACHE_REGION_FLAG_PGTABLE;
            ucs_atomic_add32(&region->refcount, -1);
        }
        if (region->refcount > 0) {
            ucs_rcache_region_warn(rcache, region, "destroying inuse");
        }
        ucs_mem_region_destroy_internal(rcache, region);
    }
}

static inline int ucs_rcache_region_test(ucs_rcache_region_t *region, int prot)
{
    return (region->flags & UCS_RCACHE_REGION_FLAG_REGISTERED) &&
           ucs_test_all_flags(region->prot, prot);
}

/* Lock must be held */
static ucs_status_t
ucs_rcache_check_overlap(ucs_rcache_t *rcache, ucs_pgt_addr_t *start,
                         ucs_pgt_addr_t *end, int *prot, int *merged,
                         ucs_rcache_region_t **region_p)
{
    ucs_rcache_region_t *region, *tmp;
    ucs_list_link_t region_list;
    int mem_prot;

    ucs_trace_func("rcache=%s, *start=0x%lx, *end=0x%lx", rcache->name, *start,
                   *end);

    ucs_rcache_check_inv_queue(rcache);

    ucs_rcache_find_regions(rcache, *start, *end - 1, &region_list);

    /* TODO check if any of the regions is locked */

    ucs_list_for_each_safe(region, tmp, &region_list, list) {

        if ((*start >= region->super.start) && (*end <= region->super.end) &&
            ucs_rcache_region_test(region, *prot))
        {
            /* Found a region which contains the given address range */
            ucs_rcache_region_hold(rcache, region);
            *region_p = region;
            return UCS_ERR_ALREADY_EXISTS;
        }

        /*
         * If we don't provide some of the permissions the other region had,
         * we might want to expand our permissions to support them. We can
         * do that only if the memory range actually has those permissions.
         *  This will prevent the users of the other region to kick us out
         * the next time.
         */
        if (!ucs_test_all_flags(*prot, region->prot)) {
            /* A slow path because searching /proc/maps in order to
             * check memory protection is very expensive.
             *
             * TODO: currently rcache is optimized for the case where most of
             * the regions have same protection.
             */
            mem_prot = UCS_PROFILE_CALL(ucs_get_mem_prot, *start, *end);
            if (!ucs_test_all_flags(mem_prot, *prot)) {
                ucs_rcache_region_trace(rcache, region,
                                        "do not merge "UCS_RCACHE_PROT_FMT
                                        " with mem "UCS_RCACHE_PROT_FMT,
                                        UCS_RCACHE_PROT_ARG(*prot),
                                        UCS_RCACHE_PROT_ARG(mem_prot));
                /* The memory protection can not satisfy that of the
                 * region. However mem_reg still may be able to deal with it.
                 * Do the safest thing: invalidate cached region
                 */
                ucs_rcache_region_invalidate(rcache, region, 1, 0);
                continue;
            } else if (ucs_test_all_flags(mem_prot, region->prot)) {
                *prot |= region->prot;
            } else {
                /* Could not support other region's permissions - so do not merge
                 * with it. If anybody will use the other region, this will kick
                 * out our region, and may potentially lead to ineffective use
                 * of the cache. We can't solve it as long as we have only one
                 * page table, since it does not allow overlap.
                 */
                ucs_rcache_region_trace(rcache, region,
                                        "do not merge mem "UCS_RCACHE_PROT_FMT" with",
                                        UCS_RCACHE_PROT_ARG(mem_prot));
                ucs_rcache_region_invalidate(rcache, region, 1, 0);
                continue;
            }
        }

        ucs_rcache_region_trace(rcache, region,
                                "merge 0x%lx..0x%lx "UCS_RCACHE_PROT_FMT" with",
                                *start, *end, UCS_RCACHE_PROT_ARG(*prot));
        *start  = ucs_min(*start, region->super.start);
        *end    = ucs_max(*end,   region->super.end);
        *merged = 1;
        ucs_rcache_region_invalidate(rcache, region, 1, 0);
    }
    return UCS_OK;
}

static ucs_status_t
ucs_rcache_create_region(ucs_rcache_t *rcache, void *address, size_t length,
                         int prot, void *arg, ucs_rcache_region_t **region_p)
{
    ucs_rcache_region_t *region;
    ucs_pgt_addr_t start, end;
    ucs_status_t status;
    int merged;

    ucs_trace_func("rcache=%s, address=%p, length=%zu", rcache->name, address,
                   length);

    pthread_rwlock_wrlock(&rcache->lock);

retry:
    /* Align to page size */
    start  = ucs_align_down_pow2((uintptr_t)address,
                                 rcache->params.alignment);
    end    = ucs_align_up_pow2  ((uintptr_t)address + length,
                                 rcache->params.alignment);
    region = NULL;
    merged = 0;

    /* Check overlap with existing regions */
    status = UCS_PROFILE_CALL(ucs_rcache_check_overlap, rcache, &start, &end,
                              &prot, &merged, &region);
    if (status == UCS_ERR_ALREADY_EXISTS) {
        /* Found a matching region (it could have been added after we released
         * the lock)
         */
        status = region->status;
        goto out_set_region;
    } else if (status != UCS_OK) {
        /* Could not create a region because there are overlapping regions which
         * cannot be removed.
         */
        goto out_unlock;
    }

    /* Allocate structure for new region */
    region = ucs_memalign(UCS_PGT_ENTRY_MIN_ALIGN, rcache->params.region_struct_size,
                          "rcache_region");
    if (region == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_unlock;
    }

    memset(region, 0, rcache->params.region_struct_size);

    region->super.start = start;
    region->super.end   = end;
    status = UCS_PROFILE_CALL(ucs_pgtable_insert, &rcache->pgtable, &region->super);
    if (status != UCS_OK) {
        ucs_error("failed to insert region " UCS_PGT_REGION_FMT ": %s",
                  UCS_PGT_REGION_ARG(&region->super), ucs_status_string(status));
        ucs_free(region);
        goto out_unlock;
    }

    /* If memory registration failed, keep the region and mark it as invalid,
     * to avoid numerous retries of registering the region.
     */
    region->prot     = prot;
    region->flags    = UCS_RCACHE_REGION_FLAG_PGTABLE;
    region->refcount = 1;
    region->status = status =
        UCS_PROFILE_NAMED_CALL("mem_reg", rcache->params.ops->mem_reg,
                               rcache->params.context, rcache, arg, region);
    if (status != UCS_OK) {
        if (merged) {
            /* failure may be due to merge, because memory of the merged
             * regions has different access permission.
             * Retry with original address: there will be no merge because
             * all merged regions has been invalidated and registration will
             * succeed.
             */
            ucs_debug("failed to register merged region " UCS_PGT_REGION_FMT ": %s, retrying",
                      UCS_PGT_REGION_ARG(&region->super), ucs_status_string(status));
            ucs_rcache_region_invalidate(rcache, region, 1, 1);
            goto retry;
        } else {
            ucs_warn("failed to register region " UCS_PGT_REGION_FMT ": %s",
                     UCS_PGT_REGION_ARG(&region->super), ucs_status_string(status));
            goto out_unlock;
        }
    }

    region->flags   |= UCS_RCACHE_REGION_FLAG_REGISTERED;
    region->refcount = 2; /* Page-table + user */

    ucs_rcache_region_trace(rcache, region, "created");

out_set_region:
    *region_p = region;
out_unlock:
    pthread_rwlock_unlock(&rcache->lock);
    return status;
}

void ucs_rcache_region_hold(ucs_rcache_t *rcache, ucs_rcache_region_t *region)
{
    ucs_atomic_add32(&region->refcount, +1);
    ucs_rcache_region_trace(rcache, region, "hold");
}

ucs_status_t ucs_rcache_get(ucs_rcache_t *rcache, void *address, size_t length,
                            int prot, void *arg, ucs_rcache_region_t **region_p)
{
    ucs_pgt_addr_t start = (uintptr_t)address;
    ucs_pgt_region_t *pgt_region;
    ucs_rcache_region_t *region;

    ucs_trace_func("rcache=%s, address=%p, length=%zu", rcache->name, address,
                   length);

    pthread_rwlock_rdlock(&rcache->lock);
    if (ucs_queue_is_empty(&rcache->inv_q)) {
        pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup, &rcache->pgtable,
                                      start);
        if (ucs_likely(pgt_region != NULL)) {
            region = ucs_derived_of(pgt_region, ucs_rcache_region_t);
            if (((start + length) <= region->super.end) &&
                ucs_rcache_region_test(region, prot))
            {
                ucs_rcache_region_hold(rcache, region);
                *region_p = region;
                pthread_rwlock_unlock(&rcache->lock);
                return UCS_OK;
            }
        }
    }
    pthread_rwlock_unlock(&rcache->lock);

    /* Fall back to slow version (with rw lock) in following cases:
     * - invalidation list not empty
     * - could not find cached region
     * - found unregistered region
     */
    return UCS_PROFILE_CALL(ucs_rcache_create_region, rcache, address, length,
                            prot, arg, region_p);
}

void ucs_rcache_region_put(ucs_rcache_t *rcache, ucs_rcache_region_t *region)
{
    ucs_rcache_region_put_internal(rcache, region, 1, 0);
}

static UCS_CLASS_INIT_FUNC(ucs_rcache_t, const ucs_rcache_params_t *params,
                           const char *name, ucs_stats_node_t *stats_parent)
{
    ucs_status_t status;
    int ret;

    if (params->region_struct_size < sizeof(ucs_rcache_region_t)) {
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    if (!ucs_is_pow2(params->alignment) ||
        (params->alignment < UCS_PGT_ADDR_ALIGN) ||
        (params->alignment > params->max_alignment))
    {
        ucs_error("invalid regcache alignment (%zu): must be a power of 2 "
                  "between %zu and %zu",
                  params->alignment, UCS_PGT_ADDR_ALIGN, params->max_alignment);
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    self->params = *params;

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

    ret = pthread_spin_init(&self->inv_lock, 0);
    if (ret) {
        ucs_error("pthread_spin_init() failed: %m");
        status = UCS_ERR_INVALID_PARAM;
        goto err_destroy_rwlock;
    }

    status = ucs_pgtable_init(&self->pgtable, ucs_rcache_pgt_dir_alloc,
                              ucs_rcache_pgt_dir_release);
    if (status != UCS_OK) {
        goto err_destroy_inv_q_lock;
    }

    status = ucs_mpool_init(&self->inv_mp, 0, sizeof(ucs_rcache_inv_entry_t), 0,
                            1, 1024, -1, &ucs_rcache_mp_ops, "rcache_inv_mp");
    if (status != UCS_OK) {
        goto err_cleanup_pgtable;
    }

    status = ucm_set_event_handler(UCM_EVENT_VM_UNMAPPED, params->ucm_event_priority,
                                   ucs_rcache_unmapped_callback, self);
    if (status != UCS_OK) {
        goto err_destroy_mp;
    }

    ucs_queue_head_init(&self->inv_q);
    return UCS_OK;

err_destroy_mp:
    ucs_mpool_cleanup(&self->inv_mp, 1);
err_cleanup_pgtable:
    ucs_pgtable_cleanup(&self->pgtable);
err_destroy_inv_q_lock:
    pthread_spin_destroy(&self->inv_lock);
err_destroy_rwlock:
    pthread_rwlock_destroy(&self->lock);
err_free_name:
    free(self->name);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(ucs_rcache_t)
{
    ucm_unset_event_handler(UCM_EVENT_VM_UNMAPPED, ucs_rcache_unmapped_callback,
                            self);
    ucs_rcache_check_inv_queue(self);
    ucs_rcache_purge(self);

    ucs_mpool_cleanup(&self->inv_mp, 1);
    ucs_pgtable_cleanup(&self->pgtable);
    pthread_spin_destroy(&self->inv_lock);
    pthread_rwlock_destroy(&self->lock);
    free(self->name);
}

UCS_CLASS_DEFINE(ucs_rcache_t, void);
UCS_CLASS_DEFINE_NAMED_NEW_FUNC(ucs_rcache_create, ucs_rcache_t, ucs_rcache_t,
                                const ucs_rcache_params_t*, const char *,
                                ucs_stats_node_t*)
UCS_CLASS_DEFINE_NAMED_DELETE_FUNC(ucs_rcache_destroy, ucs_rcache_t, ucs_rcache_t)
