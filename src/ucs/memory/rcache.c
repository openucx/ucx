/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 * Copyright (C) Huawei Technologies Co., Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/arch/atomic.h>
#include <ucs/async/pipe.h>
#include <ucs/type/class.h>
#include <ucs/datastruct/queue.h>
#include <ucs/debug/log.h>
#include <ucs/profile/profile.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/stats/stats.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>
#include <ucs/type/spinlock.h>
#include <ucs/vfs/base/vfs_obj.h>
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
#define ucs_rcache_region_pfn_ptr(_region) \
    ((_region)->pfn)


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


typedef struct ucs_rcache_comp_entry {
    ucs_list_link_t                   list;
    ucs_rcache_invalidate_comp_func_t func;
    void                              *arg;
} ucs_rcache_comp_entry_t;


typedef struct {
    ucs_rcache_t        *rcache;
    ucs_rcache_region_t *region;
} ucs_rcache_region_validate_pfn_t;


#ifdef ENABLE_STATS
static ucs_stats_class_t ucs_rcache_stats_class = {
    .name          = "rcache",
    .num_counters  = UCS_RCACHE_STAT_LAST,
    .class_id      = UCS_STATS_CLASS_ID_INVALID,
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


/*
 * Global rcache context
 */
typedef struct {
    /* Protects access to context members */
    pthread_mutex_t  lock;

    /* List of all rcaches */
    ucs_list_link_t  list;

    /* Used for triggering an rcache cleanup */
    ucs_async_pipe_t pipe;
} ucs_rcache_global_context_t;

static ucs_rcache_global_context_t ucs_rcache_global_context = {
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .list = UCS_LIST_INITIALIZER(&ucs_rcache_global_context.list,
             &ucs_rcache_global_context.list),
    .pipe = UCS_ASYNC_PIPE_INITIALIZER
};


static void __ucs_rcache_region_log(const char *file, int line, const char *function,
                                    ucs_log_level_t level, ucs_rcache_t *rcache,
                                    ucs_rcache_region_t *region, const char *fmt,
                                    ...) UCS_F_PRINTF(7, 8);

static void __ucs_rcache_region_log(const char *file, int line, const char *function,
                                    ucs_log_level_t level, ucs_rcache_t *rcache,
                                    ucs_rcache_region_t *region, const char *fmt,
                                    ...)
{
    char message[128];
    char region_desc[128];
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

static size_t ucs_rcache_stat_max_pow2()
{
    return ucs_roundup_pow2(ucs_global_opts.rcache_stat_max);
}

static int ucs_rcache_stat_min_lz()
{
    return ucs_count_leading_zero_bits(UCS_RCACHE_STAT_MIN_POW2);
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
        ucs_error("mmap(size=%zu) failed: %m", size);
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
    .obj_cleanup   = NULL,
    .obj_str       = NULL
};

static unsigned ucs_rcache_region_page_count(ucs_rcache_region_t *region)
{
    size_t page_size = ucs_get_page_size();

    return (ucs_align_up(region->super.end, page_size) -
            ucs_align_down(region->super.start, page_size)) /
            ucs_get_page_size();
}

static void ucs_rcache_validate_pfn(ucs_rcache_t *rcache,
                                    ucs_rcache_region_t *region,
                                    unsigned page_num,
                                    unsigned long region_pfn,
                                    unsigned long actual_pfn)
{
    if (region_pfn != actual_pfn) {
        ucs_rcache_region_error(rcache, region, "pfn check failed");
        ucs_fatal("%s: page at virtual address 0x%lx moved from pfn 0x%lx to pfn 0x%lx",
                  rcache->name,
                  region->super.start + (page_num * ucs_get_page_size()),
                  region_pfn, actual_pfn);
    }
}

static void ucs_rcache_region_validate_pfn_cb(unsigned page_num,
                                              unsigned long pfn,
                                              void *ctx)
{
    ucs_rcache_region_validate_pfn_t *data = (ucs_rcache_region_validate_pfn_t*)ctx;

    ucs_rcache_validate_pfn(data->rcache, data->region, page_num,
                            ucs_rcache_region_pfn_ptr(data->region)[page_num],
                            pfn);
}

/* Lock must be held for read */
static void ucs_rcache_region_validate_pfn(ucs_rcache_t *rcache,
                                           ucs_rcache_region_t *region)
{
    unsigned long region_pfn, actual_pfn;
    unsigned page_count;
    ucs_rcache_region_validate_pfn_t ctx;
    ucs_status_t status;

    if ((rcache->params.flags & UCS_RCACHE_FLAG_NO_PFN_CHECK) ||
        (ucs_global_opts.rcache_check_pfn == 0)) {
        return;
    }

    if (ucs_global_opts.rcache_check_pfn == 1) {
        /* in case if only 1 page to check - save PFN value in-place
           in priv section */
        region_pfn = ucs_rcache_region_pfn(region);
        status = ucs_sys_get_pfn(region->super.start, 1, &actual_pfn);
        if (status != UCS_OK) {
            goto out;
        }
        ucs_rcache_validate_pfn(rcache, region, 0, region_pfn, actual_pfn);
        goto out;
    }

    page_count = ucs_min(ucs_global_opts.rcache_check_pfn,
                         ucs_rcache_region_page_count(region));
    ctx.rcache = rcache;
    ctx.region = region;
    status     = ucs_sys_enum_pfn(region->super.start, page_count,
                                  ucs_rcache_region_validate_pfn_cb, &ctx);

out:
    if (status == UCS_OK) {
        ucs_rcache_region_trace(rcache, region, "pfn ok");
    }
}

/* Lock must be held */
static void ucs_rcache_region_collect_callback(const ucs_pgtable_t *pgtable,
                                               ucs_pgt_region_t *pgt_region, void *arg)
{
    ucs_rcache_region_t *region = ucs_derived_of(pgt_region, ucs_rcache_region_t);
    ucs_list_link_t *list = arg;

    ucs_list_add_tail(list, &region->tmp_list);
}

/* Lock must be held */
static void ucs_rcache_find_regions(ucs_rcache_t *rcache, ucs_pgt_addr_t from,
                                    ucs_pgt_addr_t to, ucs_list_link_t *list)
{
    ucs_list_head_init(list);
    ucs_pgtable_search_range(&rcache->pgtable, from, to,
                             ucs_rcache_region_collect_callback, list);
}

/* LRU spinlock must be held */
static inline void
ucs_rcache_region_lru_add(ucs_rcache_t *rcache, ucs_rcache_region_t *region)
{
    if (region->lru_flags & UCS_RCACHE_LRU_FLAG_IN_LRU) {
        return;
    }

    ucs_rcache_region_trace(rcache, region, "lru add");
    ucs_list_add_tail(&rcache->lru.list, &region->lru_list);
    region->lru_flags |= UCS_RCACHE_LRU_FLAG_IN_LRU;
}

/* LRU spinlock must be held */
static inline void
ucs_rcache_region_lru_remove(ucs_rcache_t *rcache, ucs_rcache_region_t *region)
{
    if (!(region->lru_flags & UCS_RCACHE_LRU_FLAG_IN_LRU)) {
        return;
    }

    ucs_rcache_region_trace(rcache, region, "lru remove");
    ucs_list_del(&region->lru_list);
    region->lru_flags &= ~UCS_RCACHE_LRU_FLAG_IN_LRU;
}

static void
ucs_rcache_region_lru_get(ucs_rcache_t *rcache, ucs_rcache_region_t *region)
{
    /* A used region cannot be evicted */
    ucs_spin_lock(&rcache->lru.lock);
    ucs_rcache_region_lru_remove(rcache, region);
    ucs_spin_unlock(&rcache->lru.lock);
}

static void
ucs_rcache_region_lru_put(ucs_rcache_t *rcache, ucs_rcache_region_t *region)
{
    /* When we finish using a region, it's a candidate for LRU eviction */
    ucs_spin_lock(&rcache->lru.lock);
    ucs_rcache_region_lru_add(rcache, region);
    ucs_spin_unlock(&rcache->lru.lock);
}

static ucs_rcache_distribution_t *
ucs_rcache_distribution_get_bin(ucs_rcache_t *rcache, size_t region_size)
{
    size_t bin;

    if (region_size < UCS_RCACHE_STAT_MIN_POW2) {
        bin = 0;
    } else if (region_size >= ucs_rcache_stat_max_pow2()) {
        bin = ucs_rcache_distribution_get_num_bins() - 1;
    } else {
        bin = ucs_rcache_stat_min_lz() + 1 -
              ucs_count_leading_zero_bits(region_size);
    }

    return &rcache->distribution[bin];
}

/* Lock must be held in write mode */
static void ucs_mem_region_destroy_internal(ucs_rcache_t *rcache,
                                            ucs_rcache_region_t *region)
{
    ucs_rcache_comp_entry_t *comp;
    size_t region_size;
    ucs_rcache_distribution_t *distribution_bin;

    ucs_rcache_region_trace(rcache, region, "destroy");

    ucs_assertv(region->refcount == 0, "region %p 0x%lx..0x%lx of %s", region,
                region->super.start, region->super.end, rcache->name);
    ucs_assert(!(region->flags & UCS_RCACHE_REGION_FLAG_PGTABLE));

    if (region->flags & UCS_RCACHE_REGION_FLAG_REGISTERED) {
        UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_DEREGS, 1);
        {
            UCS_PROFILE_CODE("mem_dereg") {
                rcache->params.ops->mem_dereg(rcache->params.context, rcache,
                region);
            }
        }
    }

    if (!(rcache->params.flags & UCS_RCACHE_FLAG_NO_PFN_CHECK) &&
        (ucs_global_opts.rcache_check_pfn > 1)) {
        ucs_free(ucs_rcache_region_pfn_ptr(region));
    }

    ucs_spin_lock(&rcache->lru.lock);
    ucs_rcache_region_lru_remove(rcache, region);
    ucs_spin_unlock(&rcache->lru.lock);

    --rcache->num_regions;
    region_size         = region->super.end - region->super.start;
    rcache->total_size -= region_size;

    distribution_bin = ucs_rcache_distribution_get_bin(rcache, region_size);
    --distribution_bin->count;
    distribution_bin->total_size -= region_size;

    while (!ucs_list_is_empty(&region->comp_list)) {
        comp = ucs_list_extract_head(&region->comp_list,
                                     ucs_rcache_comp_entry_t, list);
        comp->func(comp->arg);
        ucs_spin_lock(&rcache->lock);
        ucs_mpool_put(comp);
        ucs_spin_unlock(&rcache->lock);
    }

    ucs_free(region);
}

static inline void ucs_rcache_region_put_internal(ucs_rcache_t *rcache,
                                                  ucs_rcache_region_t *region,
                                                  unsigned flags)
{
    ucs_rcache_region_trace(rcache, region, "put region, flags 0x%x", flags);

    ucs_assert(region->refcount > 0);
    if (ucs_likely(ucs_atomic_fsub32(&region->refcount, 1) != 1)) {
        ucs_assert(!(flags & UCS_RCACHE_REGION_PUT_FLAG_MUST_DESTROY));
        return;
    }

    if (flags & UCS_RCACHE_REGION_PUT_FLAG_ADD_TO_GC) {
        /* Put the region on garbage collection list */
        ucs_assert(!(flags & UCS_RCACHE_REGION_PUT_FLAG_TAKE_PGLOCK));
        ucs_spin_lock(&rcache->lock);
        ucs_rcache_region_trace(rcache, region, "put on GC list, flags 0x%x",
                                flags);
        rcache->unreleased_size += (region->super.end - region->super.start);
        ucs_list_add_tail(&rcache->gc_list, &region->tmp_list);
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
static void ucs_rcache_region_invalidate_internal(ucs_rcache_t *rcache,
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
        ucs_rcache_region_put_internal(rcache, region, flags);
    } else {
        ucs_assert(!(flags & UCS_RCACHE_REGION_PUT_FLAG_IN_PGTABLE));
    }
}

/* Lock must be held in write mode */
static void ucs_rcache_invalidate_range(ucs_rcache_t *rcache, ucs_pgt_addr_t start,
                                        ucs_pgt_addr_t end, unsigned flags)
{
    ucs_rcache_region_t *region, *tmp;
    ucs_list_link_t region_list;

    ucs_trace_func("rcache=%s, start=0x%lx, end=0x%lx", rcache->name, start, end);

    ucs_rcache_find_regions(rcache, start, end - 1, &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, tmp_list) {
        /* all regions on the list are in the page table */
        ucs_rcache_region_invalidate_internal(
                rcache, region, flags | UCS_RCACHE_REGION_PUT_FLAG_IN_PGTABLE);
        UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_UNMAP_INVALIDATES, 1);
    }
}

static void ucs_rcache_remove_from_unreleased(ucs_rcache_t *rcache,
                                              ucs_pgt_addr_t entry_start,
                                              ucs_pgt_addr_t entry_end)
{
    size_t entry_size = entry_end - entry_start;
    ucs_assert(rcache->unreleased_size >= entry_size);
    rcache->unreleased_size -= entry_size;
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
        ucs_rcache_remove_from_unreleased(rcache, entry->start, entry->end);

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
                                       tmp_list);
        ucs_rcache_remove_from_unreleased(rcache, region->super.start,
                                          region->super.end);

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

    if (rcache->unreleased_size > rcache->params.max_unreleased) {
        /* Trigger a cleanup when the pending size exceeds the threshold */
        ucs_async_pipe_push(&ucs_rcache_global_context.pipe);
    }

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
        /* coverity[double_lock] */
        ucs_rcache_invalidate_range(rcache, start, end,
                                    UCS_RCACHE_REGION_PUT_FLAG_ADD_TO_GC);
        UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_UNMAPS, 1);
        ucs_rcache_check_inv_queue(rcache, UCS_RCACHE_REGION_PUT_FLAG_ADD_TO_GC);
        /* coverity[double_unlock] */
        pthread_rwlock_unlock(&rcache->pgt_lock);
        return;
    }

    /* Could not lock - add region to invalidation queue */
    ucs_spin_lock(&rcache->lock);
    entry = ucs_mpool_get(&rcache->mp);
    if (entry != NULL) {
        entry->start             = start;
        entry->end               = end;
        rcache->unreleased_size += (entry->end - entry->start);
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
    ucs_list_for_each_safe(region, tmp, &region_list, tmp_list) {
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

/* Lock must be held in write mode */
static void ucs_rcache_clean(ucs_rcache_t *rcache)
{
    pthread_rwlock_wrlock(&rcache->pgt_lock);
    /* coverity[double_lock]*/
    ucs_rcache_check_inv_queue(rcache, 0);
    ucs_rcache_check_gc_list(rcache);
    pthread_rwlock_unlock(&rcache->pgt_lock);
}

/* Lock must be held in write mode */
static void ucs_rcache_lru_evict(ucs_rcache_t *rcache)
{
    int num_evicted, num_skipped;
    ucs_rcache_region_t *region;

    num_evicted = 0;
    num_skipped = 0;

    ucs_spin_lock(&rcache->lru.lock);
    while (!ucs_list_is_empty(&rcache->lru.list) &&
           ((rcache->num_regions > rcache->params.max_regions) ||
            (rcache->total_size > rcache->params.max_size))) {
        region = ucs_list_head(&rcache->lru.list, ucs_rcache_region_t,
                               lru_list);
        ucs_assert(region->lru_flags & UCS_RCACHE_LRU_FLAG_IN_LRU);

        if (!(region->flags & UCS_RCACHE_REGION_FLAG_PGTABLE) ||
            (region->refcount > 1)) {
            /* region is in use or not in page table - remove from lru */
            ucs_rcache_region_lru_remove(rcache, region);
            ++num_skipped;
            continue;
        }

        ucs_spin_unlock(&rcache->lru.lock);

        /* The region is expected to have refcount=1 and present in pgt, so it
         * would be destroyed immediately by this function
         */
        ucs_rcache_region_trace(rcache, region, "evict");
        ucs_rcache_region_invalidate_internal(
                rcache, region,
                UCS_RCACHE_REGION_PUT_FLAG_MUST_DESTROY |
                        UCS_RCACHE_REGION_PUT_FLAG_IN_PGTABLE);
        ++num_evicted;

        ucs_spin_lock(&rcache->lru.lock);
    }

    ucs_spin_unlock(&rcache->lru.lock);

    if (num_evicted > 0) {
        ucs_debug("evicted %d regions, skipped %d regions, usage: %lu (%lu)",
                  num_evicted, num_skipped, rcache->num_regions,
                  rcache->params.max_regions);
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

    ucs_list_for_each_safe(region, tmp, &region_list, tmp_list) {
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
                ucs_rcache_region_invalidate_internal(
                        rcache, region, UCS_RCACHE_REGION_PUT_FLAG_IN_PGTABLE);
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
                ucs_rcache_region_invalidate_internal(
                        rcache, region, UCS_RCACHE_REGION_PUT_FLAG_IN_PGTABLE);
                continue;
            }
        }

        ucs_rcache_region_trace(rcache, region,
                                "merge 0x%lx..0x%lx "UCS_RCACHE_PROT_FMT" with",
                                *start, *end, UCS_RCACHE_PROT_ARG(*prot));
        *start  = ucs_min(*start, region->super.start);
        *end    = ucs_max(*end,   region->super.end);
        *merged = 1;
        ucs_rcache_region_invalidate_internal(
                rcache, region, UCS_RCACHE_REGION_PUT_FLAG_IN_PGTABLE);
    }
    return UCS_OK;
}

static ucs_status_t ucs_rcache_fill_pfn(ucs_rcache_region_t *region)
{
    unsigned page_count;
    ucs_status_t status;

    if (ucs_global_opts.rcache_check_pfn == 0) {
        ucs_rcache_region_pfn(region) = 0;
        return UCS_OK;
    }

    if (ucs_global_opts.rcache_check_pfn == 1) {
        return ucs_sys_get_pfn(region->super.start, 1, &ucs_rcache_region_pfn(region));
    }

    page_count = ucs_min(ucs_rcache_region_page_count(region),
                         ucs_global_opts.rcache_check_pfn);
    ucs_rcache_region_pfn_ptr(region) =
        ucs_malloc(sizeof(*ucs_rcache_region_pfn_ptr(region)) * page_count,
                   "pfn list");
    if (ucs_rcache_region_pfn_ptr(region) == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    status = ucs_sys_get_pfn(region->super.start, page_count,
                             ucs_rcache_region_pfn_ptr(region));
    if (status != UCS_OK) {
        ucs_free(ucs_rcache_region_pfn_ptr(region));
    }

    return status;
}

static ucs_status_t
ucs_rcache_create_region(ucs_rcache_t *rcache, void *address, size_t length,
                         int prot, void *arg, ucs_rcache_region_t **region_p)
{
    ucs_rcache_region_t *region;
    ucs_pgt_addr_t start, end;
    ucs_status_t status;
    int error, merged;
    size_t region_size;
    ucs_rcache_distribution_t *distribution_bin;

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
    /* coverity[double_lock] */
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
    ucs_list_head_init(&region->comp_list);

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

    region->prot      = prot;
    region->flags     = UCS_RCACHE_REGION_FLAG_PGTABLE;
    region->lru_flags = 0;
    region->refcount  = 1;
    region->status    = UCS_INPROGRESS;

    ++rcache->num_regions;

    region_size         = region->super.end - region->super.start;
    rcache->total_size += region_size;

    distribution_bin = ucs_rcache_distribution_get_bin(rcache, region_size);
    ++distribution_bin->count;
    distribution_bin->total_size += region_size;

    region->status = status =
        UCS_PROFILE_NAMED_CALL("mem_reg", rcache->params.ops->mem_reg,
                               rcache->params.context, rcache, arg, region,
                               merged ? UCS_RCACHE_MEM_REG_HIDE_ERRORS : 0);
    if (status != UCS_OK) {
        if (merged) {
            /* failure may be due to merge, because memory of the merged
             * regions has different access permission.
             * Retry with original address: there will be no merge because
             * all merged regions have been invalidated and registration will
             * succeed.
             */
            ucs_debug("failed to register merged region " UCS_PGT_REGION_FMT ": %s, retrying",
                      UCS_PGT_REGION_ARG(&region->super), ucs_status_string(status));
            ucs_rcache_region_invalidate_internal(
                    rcache, region,
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

    if (!(rcache->params.flags & UCS_RCACHE_FLAG_NO_PFN_CHECK)) {
        status = ucs_rcache_fill_pfn(region);
        if (status != UCS_OK) {
            ucs_error("failed to allocate pfn list");
            ucs_free(region);
            goto out_unlock;
        }

        ucs_rcache_lru_evict(rcache);
    }

    UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_MISSES, 1);

    ucs_rcache_region_trace(rcache, region, "created");

out_set_region:
    *region_p = region;
out_unlock:
    /* coverity[double_unlock]*/
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
                ucs_rcache_region_lru_get(rcache, region);
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
    ucs_rcache_region_lru_put(rcache, region);
    ucs_rcache_region_put_internal(rcache, region,
                                   UCS_RCACHE_REGION_PUT_FLAG_TAKE_PGLOCK);
    UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_PUTS, 1);
}

void ucs_rcache_region_invalidate(ucs_rcache_t *rcache,
                                  ucs_rcache_region_t *region,
                                  ucs_rcache_invalidate_comp_func_t cb,
                                  void *arg)
{
    ucs_rcache_comp_entry_t *comp;

    /* Completion entry should be added before region is invalidated */
    ucs_spin_lock(&rcache->lock);
    comp = ucs_mpool_get(&rcache->mp);
    ucs_spin_unlock(&rcache->lock);

    pthread_rwlock_wrlock(&rcache->pgt_lock);
    if (comp != NULL) {
        comp->func = cb;
        comp->arg  = arg;
        ucs_list_add_tail(&region->comp_list, &comp->list);
    } else {
        ucs_rcache_region_error(rcache, region,
                                "failed to allocate completion object");
    }

    /* coverity[double_lock] */
    ucs_rcache_region_invalidate_internal(rcache, region, 0);
    /* coverity[double_unlock] */
    pthread_rwlock_unlock(&rcache->pgt_lock);
    UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_PUTS, 1);
}

static void ucs_rcache_before_fork(void)
{
    ucs_rcache_t *rcache;

    pthread_mutex_lock(&ucs_rcache_global_context.lock);
    ucs_list_for_each(rcache, &ucs_rcache_global_context.list, list) {
        if (rcache->params.flags & UCS_RCACHE_FLAG_PURGE_ON_FORK) {
            /* Fork will trigger process memory invalidation. Cache
             * invalidation intended to solve following cases:
             * - Pinned memory with MADV_DONTFORK (e.g IB/MD):
             *   If a registered region shares a page with other allocation in
             *   the process, that allocation won't be available in child
             *   process as expected.
             * - Pinned memory without MADV_DONTFORK (e.g KNEM/MD):
             *   DONTFORK is generally required to avoid registration cache
             *   becoming out of sync with virt-to-phys MMU mapping.
             *   We compensate for the absence of DONTFORK by removing all
             *   registered memory regions, and they could be registered
             *   again on-demand.
             * - Other use cases shouldn't be affected
             */
            pthread_rwlock_wrlock(&rcache->pgt_lock);
            /* coverity[double_lock] */
            ucs_rcache_invalidate_range(rcache, 0, UCS_PGT_ADDR_MAX, 0);
            pthread_rwlock_unlock(&rcache->pgt_lock);
        }
    }
    pthread_mutex_unlock(&ucs_rcache_global_context.lock);
}

static void
ucs_rcache_invalidate_handler(int id, ucs_event_set_types_t events, void *arg)
{
    ucs_rcache_t *rcache;

    ucs_async_pipe_drain(&ucs_rcache_global_context.pipe);

    pthread_mutex_lock(&ucs_rcache_global_context.lock);
    ucs_list_for_each(rcache, &ucs_rcache_global_context.list, list) {
        ucs_rcache_clean(rcache);
    }
    pthread_mutex_unlock(&ucs_rcache_global_context.lock);
}

static ucs_status_t ucs_rcache_global_list_add(ucs_rcache_t *rcache)
{
    ucs_status_t status         = UCS_OK;
    static int atfork_installed = 0;
    int ret;

    pthread_mutex_lock(&ucs_rcache_global_context.lock);
    if (atfork_installed ||
        !(rcache->params.flags & UCS_RCACHE_FLAG_PURGE_ON_FORK)) {
        goto out_list_add;
    }

    ret = pthread_atfork(ucs_rcache_before_fork, NULL, NULL);
    if (ret != 0) {
        ucs_warn("pthread_atfork failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto out_list_add;
    }

    atfork_installed = 1;

out_list_add:
    if (ucs_list_is_empty(&ucs_rcache_global_context.list)) {
        status = ucs_async_pipe_create(&ucs_rcache_global_context.pipe);
        if (status != UCS_OK) {
            goto out;
        }

        status = ucs_async_set_event_handler(
                UCS_ASYNC_MODE_THREAD,
                ucs_async_pipe_rfd(&ucs_rcache_global_context.pipe),
                UCS_EVENT_SET_EVREAD, ucs_rcache_invalidate_handler, NULL,
                NULL);
        if (status != UCS_OK) {
            goto out;
        }
    }

    ucs_list_add_tail(&ucs_rcache_global_context.list, &rcache->list);
    assert(!ucs_list_is_empty(&ucs_rcache_global_context.list));
out:
    pthread_mutex_unlock(&ucs_rcache_global_context.lock);
    return status;
}

static void ucs_rcache_global_list_remove(ucs_rcache_t *rcache)
{
    ucs_async_pipe_t pipe;

    pthread_mutex_lock(&ucs_rcache_global_context.lock);
    pipe = ucs_rcache_global_context.pipe;
    ucs_list_del(&rcache->list);
    if (!ucs_list_is_empty(&ucs_rcache_global_context.list)) {
        pthread_mutex_unlock(&ucs_rcache_global_context.lock);
        return;
    }

    ucs_async_pipe_invalidate(&ucs_rcache_global_context.pipe);
    pthread_mutex_unlock(&ucs_rcache_global_context.lock);
    ucs_async_remove_handler(pipe.read_fd,
                             1);
    ucs_async_pipe_destroy(&pipe);
}

size_t ucs_rcache_distribution_get_num_bins()
{
    return ucs_ilog2(ucs_rcache_stat_max_pow2() / UCS_RCACHE_STAT_MIN_POW2) + 2;
}

static UCS_CLASS_INIT_FUNC(ucs_rcache_t, const ucs_rcache_params_t *params,
                           const char *name, ucs_stats_node_t *stats_parent)
{
    ucs_status_t status;
    size_t mp_obj_size, mp_align;
    int ret;

    if (params->region_struct_size < sizeof(ucs_rcache_region_t)) {
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    if (!ucs_is_pow2(params->alignment) ||
        (params->alignment < UCS_RCACHE_MIN_ALIGNMENT) ||
        (params->alignment > params->max_alignment))
    {
        ucs_error("invalid regcache alignment (%zu): must be a power of 2 "
                  "between %zu and %zu",
                  params->alignment, UCS_RCACHE_MIN_ALIGNMENT,
                  params->max_alignment);
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    self->name = ucs_strdup(name, "ucs rcache name");
    if (self->name == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    status = UCS_STATS_NODE_ALLOC(&self->stats, &ucs_rcache_stats_class,
                                  stats_parent, "-%s", self->name);
    if (status != UCS_OK) {
        goto err_free_name;
    }

    self->params = *params;

    ret = pthread_rwlock_init(&self->pgt_lock, NULL);
    if (ret) {
        ucs_error("pthread_rwlock_init() failed: %m");
        status = UCS_ERR_INVALID_PARAM;
        goto err_destroy_stats;
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
    mp_obj_size = ucs_max(mp_obj_size, sizeof(ucs_rcache_comp_entry_t));

    mp_align    = ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN);
    status      = ucs_mpool_init(&self->mp, 0, mp_obj_size, 0, mp_align, 1024,
                                 UINT_MAX, &ucs_rcache_mp_ops, "rcache_mp");
    if (status != UCS_OK) {
        goto err_cleanup_pgtable;
    }

    ucs_queue_head_init(&self->inv_q);

    /* coverity[missing_lock] */
    self->unreleased_size = 0;
    ucs_list_head_init(&self->gc_list);
    self->num_regions = 0;
    self->total_size  = 0;
    ucs_list_head_init(&self->lru.list);
    ucs_spinlock_init(&self->lru.lock, 0);

    self->distribution = ucs_calloc(ucs_rcache_distribution_get_num_bins(),
                                    sizeof(*self->distribution),
                                    "rcache_distribution");
    if (self->distribution == NULL) {
        ucs_error("failed to allocate rcache regions distribution array");
        status = UCS_ERR_NO_MEMORY;
        goto err_destroy_mp;
    }

    status = ucs_rcache_global_list_add(self);
    if (status != UCS_OK) {
        goto err_destroy_dist;
    }

    ucs_rcache_vfs_init(self);

    status = ucm_set_event_handler(params->ucm_events, params->ucm_event_priority,
                                   ucs_rcache_unmapped_callback, self);
    if (status != UCS_OK) {
        ucs_diag("rcache failed to install UCM event handler: %s",
                 ucs_status_string(status));
        goto err_remove_vfs;
    }

    return UCS_OK;

err_remove_vfs:
    ucs_vfs_obj_remove(self);
    ucs_rcache_global_list_remove(self);
err_destroy_dist:
    ucs_free(self->distribution);
err_destroy_mp:
    ucs_mpool_cleanup(&self->mp, 1);
err_cleanup_pgtable:
    ucs_pgtable_cleanup(&self->pgtable);
err_destroy_inv_q_lock:
    ucs_spinlock_destroy(&self->lock);
err_destroy_rwlock:
    pthread_rwlock_destroy(&self->pgt_lock);
err_destroy_stats:
    UCS_STATS_NODE_FREE(self->stats);
err_free_name:
    ucs_free(self->name);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(ucs_rcache_t)
{
    ucm_unset_event_handler(self->params.ucm_events, ucs_rcache_unmapped_callback,
                            self);
    ucs_vfs_obj_remove(self);
    ucs_rcache_global_list_remove(self);
    ucs_rcache_check_inv_queue(self, 0);
    ucs_rcache_check_gc_list(self);
    ucs_rcache_purge(self);

    if (!ucs_list_is_empty(&self->lru.list)) {
        ucs_warn(
                "rcache %s: %lu regions remained on lru list, first region: %p",
                self->name, ucs_list_length(&self->lru.list),
                ucs_list_head(&self->lru.list, ucs_rcache_region_t, lru_list));
    }

    ucs_spinlock_destroy(&self->lru.lock);

    ucs_mpool_cleanup(&self->mp, 1);
    ucs_pgtable_cleanup(&self->pgtable);
    ucs_spinlock_destroy(&self->lock);
    pthread_rwlock_destroy(&self->pgt_lock);
    UCS_STATS_NODE_FREE(self->stats);
    ucs_free(self->name);
    ucs_free(self->distribution);
}

UCS_CLASS_DEFINE(ucs_rcache_t, void);
UCS_CLASS_DEFINE_NAMED_NEW_FUNC(ucs_rcache_create, ucs_rcache_t, ucs_rcache_t,
                                const ucs_rcache_params_t*, const char *,
                                ucs_stats_node_t*)
UCS_CLASS_DEFINE_NAMED_DELETE_FUNC(ucs_rcache_destroy, ucs_rcache_t, ucs_rcache_t)
