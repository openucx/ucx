/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/arch/atomic.h>
#include <ucs/type/class.h>
#include <ucs/datastruct/queue.h>
#include <ucs/debug/log.h>
#include <ucs/profile/profile.h>
#include <ucs/debug/memtrack.h>
#include <ucs/stats/stats.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>
#include <ucs/type/spinlock.h>
#include <ucm/api/ucm.h>

#include "rcache.h"
#include "rcache_int.h"

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

#define ucs_rcache_region_pfn(_region) \
    ((_region)->priv)


enum {
    /* Need to page table lock while destroying */
    UCS_RCACHE_REGION_PUT_FLAG_TAKE_PGLOCK  = UCS_BIT(0),
    /* Instead of actually destroying the region, add it to garbage collection
     * list. This is used when region put is done in the context of memory
     * event callback. */
    UCS_RCACHE_REGION_PUT_FLAG_ADD_TO_GC    = UCS_BIT(1),
#if UCS_ENABLE_ASSERT
    /* Region is expected to reach a reference count of 0 and be destroyed */
    UCS_RCACHE_REGION_PUT_FLAG_MUST_DESTROY = UCS_BIT(2),
    /* Region is expected to be present in the page table  */
    UCS_RCACHE_REGION_PUT_FLAG_IN_PGTABLE   = UCS_BIT(3)
#else
    UCS_RCACHE_REGION_PUT_FLAG_MUST_DESTROY = 0,
    UCS_RCACHE_REGION_PUT_FLAG_IN_PGTABLE   = 0
#endif
};


typedef struct ucs_rcache_inv_entry {
    ucs_queue_elem_t         queue;
    ucs_pgt_addr_t           start;
    ucs_pgt_addr_t           end;
} ucs_rcache_inv_entry_t;


#ifdef ENABLE_STATS
static ucs_stats_class_t ucs_rcache_stats_class = {
    .name = "rcache",
    .num_counters = UCS_RCACHE_STAT_LAST,
    .counter_names = {
        [UCS_RCACHE_GETS]               = "gets",
        [UCS_RCACHE_HITS_FAST]          = "hits_fast",
        [UCS_RCACHE_HITS_SLOW]          = "hits_slow",
        [UCS_RCACHE_MISSES]             = "misses",
        [UCS_RCACHE_MERGES]             = "regions_merged",
        [UCS_RCACHE_UNMAPS]             = "unmap_events",
        [UCS_RCACHE_UNMAP_INVALIDATES]  = "regions_inv_unmap",
        [UCS_RCACHE_PUTS]               = "puts",
        [UCS_RCACHE_REGS]               = "mem_regs",
        [UCS_RCACHE_DEREGS]             = "mem_deregs",
    }
};
#endif


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

    ucs_log_dispatch(file, line, function, level, &ucs_global_opts.log_component,
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
    ucs_rcache_t *rcache = ucs_container_of(pgtable, ucs_rcache_t, pgtable);
    ucs_pgt_dir_t *dir;

    ucs_spin_lock(&rcache->lock);
    dir = ucs_mpool_get(&rcache->mp);
    ucs_spin_unlock(&rcache->lock);

    return dir;
}

static void ucs_rcache_pgt_dir_release(const ucs_pgtable_t *pgtable,
                                       ucs_pgt_dir_t *dir)
{
    ucs_rcache_t *rcache = ucs_container_of(pgtable, ucs_rcache_t, pgtable);

    ucs_spin_lock(&rcache->lock);
    ucs_mpool_put(dir);
    ucs_spin_unlock(&rcache->lock);
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
    *chunk_p      = UCS_PTR_BYTE_OFFSET(ptr, sizeof(size_t));
    *size_p       = size - sizeof(size_t);
    return UCS_OK;
}

static void ucs_rcache_mp_chunk_release(ucs_mpool_t *mp, void *chunk)
{
    size_t size;
    void *ptr;
    int ret;

    ptr  = UCS_PTR_BYTE_OFFSET(chunk, -sizeof(size_t));
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

/* Lock must be held for read */
static void ucs_rcache_region_validate_pfn(ucs_rcache_t *rcache,
                                           ucs_rcache_region_t *region)
{
    unsigned long region_pfn, actual_pfn;

    if (!ucs_unlikely(ucs_global_opts.rcache_check_pfn)) {
        return;
    }

    region_pfn = ucs_rcache_region_pfn(region);
    actual_pfn = ucs_sys_get_pfn(region->super.start);
    if (region_pfn != actual_pfn) {
        ucs_rcache_region_error(rcache, region, "pfn check failed");
        ucs_fatal("%s: page at virtual address 0x%lx moved from pfn 0x%lx to pfn 0x%lx",
                  rcache->name, region->super.start, region_pfn, actual_pfn);
    } else {
        ucs_rcache_region_trace(rcache, region, "pfn ok");
    }
}

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
        UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_DEREGS, 1);
        UCS_PROFILE_CODE("mem_dereg") {
            rcache->params.ops->mem_dereg(rcache->params.context, rcache, region);
        }
    }

    ucs_free(region);
}

static inline void ucs_rcache_region_put_internal(ucs_rcache_t *rcache,
                                                  ucs_rcache_region_t *region,
                                                  unsigned flags)
{
    ucs_rcache_region_trace(rcache, region, "flags 0x%x", flags);

    ucs_assert(region->refcount > 0);
    if (ucs_likely(ucs_atomic_fsub32(&region->refcount, 1) != 1)) {
        ucs_assert(!(flags & UCS_RCACHE_REGION_PUT_FLAG_MUST_DESTROY));
        return;
    }

    if (flags & UCS_RCACHE_REGION_PUT_FLAG_ADD_TO_GC) {
        /* Put the region on garbage collection list */
        ucs_spin_lock(&rcache->lock);
        ucs_rcache_region_trace(rcache, region, "put on GC list", flags);
        ucs_list_add_tail(&rcache->gc_list, &region->list);
        ucs_spin_unlock(&rcache->lock);
        return;
    }

    /* Destroy region and de-register memory */
    if (flags & UCS_RCACHE_REGION_PUT_FLAG_TAKE_PGLOCK) {
        pthread_rwlock_wrlock(&rcache->pgt_lock);
    }

    ucs_mem_region_destroy_internal(rcache, region);

    if (flags & UCS_RCACHE_REGION_PUT_FLAG_TAKE_PGLOCK) {
        pthread_rwlock_unlock(&rcache->pgt_lock);
    }
}

/* Lock must be held in write mode */
static void ucs_rcache_region_invalidate(ucs_rcache_t *rcache,
                                         ucs_rcache_region_t *region,
                                         unsigned flags)
{
    ucs_status_t status;

    ucs_rcache_region_trace(rcache, region, "invalidate");

    ucs_assert(!(flags & UCS_RCACHE_REGION_PUT_FLAG_TAKE_PGLOCK));

    /* Remove the memory region from page table, if it's there */
    if (region->flags & UCS_RCACHE_REGION_FLAG_PGTABLE) {
        status = ucs_pgtable_remove(&rcache->pgtable, &region->super);
        if (status != UCS_OK) {
            ucs_rcache_region_warn(rcache, region, "failed to remove (%s)",
                                   ucs_status_string(status));
        }
        region->flags &= ~UCS_RCACHE_REGION_FLAG_PGTABLE;
    } else {
        ucs_assert(!(flags & UCS_RCACHE_REGION_PUT_FLAG_IN_PGTABLE));
    }

     ucs_rcache_region_put_internal(rcache, region, flags);
}

/* Lock must be held in write mode */
static void ucs_rcache_invalidate_range(ucs_rcache_t *rcache, ucs_pgt_addr_t start,
                                        ucs_pgt_addr_t end, unsigned flags)
{
    ucs_rcache_region_t *region, *tmp;
    ucs_list_link_t region_list;

    ucs_trace_func("rcache=%s, start=0x%lx, end=0x%lx", rcache->name, start, end);

    ucs_rcache_find_regions(rcache, start, end - 1, &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        /* all regions on the list are in the page table */
        ucs_rcache_region_invalidate(rcache, region,
                                     flags | UCS_RCACHE_REGION_PUT_FLAG_IN_PGTABLE);
        UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_UNMAP_INVALIDATES, 1);
    }
}

/* Lock must be held in write mode */
static void ucs_rcache_check_inv_queue(ucs_rcache_t *rcache, unsigned flags)
{
    ucs_rcache_inv_entry_t *entry;

    ucs_trace_func("rcache=%s", rcache->name);

    ucs_spin_lock(&rcache->lock);
    while (!ucs_queue_is_empty(&rcache->inv_q)) {
        entry = ucs_queue_pull_elem_non_empty(&rcache->inv_q,
                                              ucs_rcache_inv_entry_t, queue);

        /* We need to drop the lock since the following code may trigger memory
         * operations, which could trigger vm_unmapped event which also takes
         * this lock.
         */
        ucs_spin_unlock(&rcache->lock);

        ucs_rcache_invalidate_range(rcache, entry->start, entry->end, flags);

        ucs_spin_lock(&rcache->lock);

        ucs_mpool_put(entry); /* Must be done with the lock held */
    }
    ucs_spin_unlock(&rcache->lock);
}

/* Lock must be held in write mode */
static void ucs_rcache_check_gc_list(ucs_rcache_t *rcache)
{
    ucs_rcache_region_t *region;

    ucs_trace_func("rcache=%s", rcache->name);

    ucs_spin_lock(&rcache->lock);
    while (!ucs_list_is_empty(&rcache->gc_list)) {
        region = ucs_list_extract_head(&rcache->gc_list, ucs_rcache_region_t,
                                       list);

        /* We need to drop the lock since the following code may trigger memory
         * operations, which could trigger vm_unmapped event which also takes
         * this lock.
         */
        ucs_spin_unlock(&rcache->lock);

        ucs_mem_region_destroy_internal(rcache, region);

        ucs_spin_lock(&rcache->lock);
    }
    ucs_spin_unlock(&rcache->lock);
}

static void ucs_rcache_unmapped_callback(ucm_event_type_t event_type,
                                         ucm_event_t *event, void *arg)
{
    ucs_rcache_t *rcache = arg;
    ucs_rcache_inv_entry_t *entry;
    ucs_pgt_addr_t start, end;

    ucs_assert(event_type == UCM_EVENT_VM_UNMAPPED ||
               event_type == UCM_EVENT_MEM_TYPE_FREE);

    if (event_type == UCM_EVENT_VM_UNMAPPED) {
        start = (uintptr_t)event->vm_unmapped.address;
        end   = (uintptr_t)event->vm_unmapped.address + event->vm_unmapped.size;
    } else if(event_type == UCM_EVENT_MEM_TYPE_FREE) {
        start = (uintptr_t)event->mem_type.address;
        end   = (uintptr_t)event->mem_type.address + event->mem_type.size;
    } else {
        ucs_warn("%s: unknown event type: %x", rcache->name, event_type);
        return;
    }

    ucs_trace_func("%s: event vm_unmapped 0x%lx..0x%lx", rcache->name, start, end);

    /*
     * Try to lock the page table and invalidate the region immediately.
     * This way we avoid queuing endless events on the invalidation queue when
     * no rcache operations are performed to clean it.
     */
    if (!pthread_rwlock_trywrlock(&rcache->pgt_lock)) {
        ucs_rcache_invalidate_range(rcache, start, end,
                                    UCS_RCACHE_REGION_PUT_FLAG_ADD_TO_GC);
        UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_UNMAPS, 1);
        ucs_rcache_check_inv_queue(rcache, UCS_RCACHE_REGION_PUT_FLAG_ADD_TO_GC);
        pthread_rwlock_unlock(&rcache->pgt_lock);
        return;
    }

    /* Could not lock - add region to invalidation queue */
    ucs_spin_lock(&rcache->lock);
    entry = ucs_mpool_get(&rcache->mp);
    if (entry != NULL) {
        entry->start = start;
        entry->end   = end;
        ucs_queue_push(&rcache->inv_q, &entry->queue);
        UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_UNMAPS, 1);
    } else {
        ucs_error("Failed to allocate invalidation entry for 0x%lx..0x%lx, "
                  "data corruption may occur", start, end);
    }
    ucs_spin_unlock(&rcache->lock);
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
            ucs_atomic_add32(&region->refcount, (uint32_t)-1);
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

    ucs_rcache_check_inv_queue(rcache, 0);
    ucs_rcache_check_gc_list(rcache);

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

        UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_MERGES, 1);
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
                ucs_rcache_region_invalidate(rcache, region,
                                             UCS_RCACHE_REGION_PUT_FLAG_IN_PGTABLE);
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
                ucs_rcache_region_invalidate(rcache, region,
                                             UCS_RCACHE_REGION_PUT_FLAG_IN_PGTABLE);
                continue;
            }
        }

        ucs_rcache_region_trace(rcache, region,
                                "merge 0x%lx..0x%lx "UCS_RCACHE_PROT_FMT" with",
                                *start, *end, UCS_RCACHE_PROT_ARG(*prot));
        *start  = ucs_min(*start, region->super.start);
        *end    = ucs_max(*end,   region->super.end);
        *merged = 1;
        ucs_rcache_region_invalidate(rcache, region,
                                     UCS_RCACHE_REGION_PUT_FLAG_IN_PGTABLE);
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
    int error, merged;

    ucs_trace_func("rcache=%s, address=%p, length=%zu", rcache->name, address,
                   length);

    pthread_rwlock_wrlock(&rcache->pgt_lock);

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
        ucs_rcache_region_validate_pfn(rcache, region);
        status = region->status;
        UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_HITS_SLOW, 1);
        goto out_set_region;
    } else if (status != UCS_OK) {
        /* Could not create a region because there are overlapping regions which
         * cannot be removed.
         */
        goto out_unlock;
    }

    /* Allocate structure for new region */
    error = ucs_posix_memalign((void **)&region,
                               ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                               rcache->params.region_struct_size,
                               "rcache_region");
    if (error != 0) {
        ucs_error("failed to allocate rcache region descriptor: %m");
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
    UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_REGS, 1);

    region->prot     = prot;
    region->flags    = UCS_RCACHE_REGION_FLAG_PGTABLE;
    region->refcount = 1;
    region->status = status =
        UCS_PROFILE_NAMED_CALL("mem_reg", rcache->params.ops->mem_reg,
                               rcache->params.context, rcache, arg, region,
                               merged ? UCS_RCACHE_MEM_REG_HIDE_ERRORS : 0);
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
            ucs_rcache_region_invalidate(rcache, region,
                                         UCS_RCACHE_REGION_PUT_FLAG_IN_PGTABLE |
                                         UCS_RCACHE_REGION_PUT_FLAG_MUST_DESTROY);
            goto retry;
        } else {
            ucs_debug("failed to register region " UCS_PGT_REGION_FMT ": %s",
                      UCS_PGT_REGION_ARG(&region->super), ucs_status_string(status));
            goto out_unlock;
        }
    }

    region->flags   |= UCS_RCACHE_REGION_FLAG_REGISTERED;
    region->refcount = 2; /* Page-table + user */

    if (ucs_global_opts.rcache_check_pfn) {
        ucs_rcache_region_pfn(region) = ucs_sys_get_pfn(region->super.start);
    } else {
        ucs_rcache_region_pfn(region) = 0;
    }

    UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_MISSES, 1);

    ucs_rcache_region_trace(rcache, region, "created");

out_set_region:
    *region_p = region;
out_unlock:
    pthread_rwlock_unlock(&rcache->pgt_lock);
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

    pthread_rwlock_rdlock(&rcache->pgt_lock);
    UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_GETS, 1);
    if (ucs_queue_is_empty(&rcache->inv_q)) {
        pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup, &rcache->pgtable,
                                      start);
        if (ucs_likely(pgt_region != NULL)) {
            region = ucs_derived_of(pgt_region, ucs_rcache_region_t);
            if (((start + length) <= region->super.end) &&
                ucs_rcache_region_test(region, prot))
            {
                ucs_rcache_region_hold(rcache, region);
                ucs_rcache_region_validate_pfn(rcache, region);
                *region_p = region;
                UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_HITS_FAST, 1);
                pthread_rwlock_unlock(&rcache->pgt_lock);
                return UCS_OK;
            }
        }
    }
    pthread_rwlock_unlock(&rcache->pgt_lock);

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
    ucs_rcache_region_put_internal(rcache, region,
                                   UCS_RCACHE_REGION_PUT_FLAG_TAKE_PGLOCK);
    UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_PUTS, 1);
}

static UCS_CLASS_INIT_FUNC(ucs_rcache_t, const ucs_rcache_params_t *params,
                           const char *name, ucs_stats_node_t *stats_parent)
{
    ucs_status_t status, spinlock_status;
    size_t mp_obj_size, mp_align;
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

    status = UCS_STATS_NODE_ALLOC(&self->stats, &ucs_rcache_stats_class,
                                  stats_parent);
    if (status != UCS_OK) {
        goto err;
    }

    self->params = *params;

    self->name = strdup(name);
    if (self->name == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_destroy_stats;
    }

    ret = pthread_rwlock_init(&self->pgt_lock, NULL);
    if (ret) {
        ucs_error("pthread_rwlock_init() failed: %m");
        status = UCS_ERR_INVALID_PARAM;
        goto err_free_name;
    }

    status = ucs_spinlock_init(&self->lock, 0);
    if (status != UCS_OK) {
        goto err_destroy_rwlock;
    }

    status = ucs_pgtable_init(&self->pgtable, ucs_rcache_pgt_dir_alloc,
                              ucs_rcache_pgt_dir_release);
    if (status != UCS_OK) {
        goto err_destroy_inv_q_lock;
    }

    mp_obj_size = ucs_max(sizeof(ucs_pgt_dir_t), sizeof(ucs_rcache_inv_entry_t));
    mp_align    = ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN);
    status      = ucs_mpool_init(&self->mp, 0, mp_obj_size, 0, mp_align, 1024,
                                 UINT_MAX, &ucs_rcache_mp_ops, "rcache_mp");
    if (status != UCS_OK) {
        goto err_cleanup_pgtable;
    }

    ucs_queue_head_init(&self->inv_q);
    ucs_list_head_init(&self->gc_list);

    status = ucm_set_event_handler(params->ucm_events, params->ucm_event_priority,
                                   ucs_rcache_unmapped_callback, self);
    if (status != UCS_OK) {
        goto err_destroy_mp;
    }

    return UCS_OK;

err_destroy_mp:
    ucs_mpool_cleanup(&self->mp, 1);
err_cleanup_pgtable:
    ucs_pgtable_cleanup(&self->pgtable);
err_destroy_inv_q_lock:
    spinlock_status = ucs_spinlock_destroy(&self->lock);
    if (spinlock_status != UCS_OK) {
        ucs_warn("ucs_recursive_spinlock_destroy() failed (%d)", spinlock_status);
    }
err_destroy_rwlock:
    pthread_rwlock_destroy(&self->pgt_lock);
err_free_name:
    free(self->name);
err_destroy_stats:
    UCS_STATS_NODE_FREE(self->stats);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(ucs_rcache_t)
{
    ucs_status_t status;

    ucm_unset_event_handler(self->params.ucm_events, ucs_rcache_unmapped_callback,
                            self);
    ucs_rcache_check_inv_queue(self, 0);
    ucs_rcache_check_gc_list(self);
    ucs_rcache_purge(self);

    ucs_mpool_cleanup(&self->mp, 1);
    ucs_pgtable_cleanup(&self->pgtable);
    status = ucs_spinlock_destroy(&self->lock);
    if (status != UCS_OK) {
        ucs_warn("ucs_recursive_spinlock_destroy() failed (%d)", status);
    }
    pthread_rwlock_destroy(&self->pgt_lock);
    UCS_STATS_NODE_FREE(self->stats);
    free(self->name);
}

UCS_CLASS_DEFINE(ucs_rcache_t, void);
UCS_CLASS_DEFINE_NAMED_NEW_FUNC(ucs_rcache_create, ucs_rcache_t, ucs_rcache_t,
                                const ucs_rcache_params_t*, const char *,
                                ucs_stats_node_t*)
UCS_CLASS_DEFINE_NAMED_DELETE_FUNC(ucs_rcache_destroy, ucs_rcache_t, ucs_rcache_t)
