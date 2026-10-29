/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018-2026. ALL RIGHTS RESERVED.
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
#include <ucs/debug/memtrack_int.h>
#include <ucs/stats/stats.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/ptr_arith.h>
#include <ucm/api/ucm.h>


static ucs_spinlock_t ucs_memtype_cache_global_instance_lock;
static int ucs_memtype_cache_failed                    = 0;
ucs_memtype_cache_t *ucs_memtype_cache_global_instance = NULL;


#define UCS_MEMTYPE_CACHE_REGION_FMT UCS_PGT_REGION_FMT " %s dev %s flags 0x%x"
#define UCS_MEMTYPE_CACHE_REGION_ARG(_region) \
            UCS_PGT_REGION_ARG(&(_region)->super), \
            ucs_memory_type_names[(_region)->mem_type], \
            ucs_topo_sys_device_get_name((_region)->sys_dev), \
            (_region)->mem_flags

typedef enum {
    UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE,
    UCS_MEMTYPE_CACHE_ACTION_REMOVE
} ucs_memtype_cache_action_t;

struct ucs_memtype_cache_region {
    ucs_pgt_region_t  super;         /**< Base class - page table region */
    ucs_list_link_t   list;          /**< List element */
    ucs_memory_type_t mem_type;      /**< Memory type (uint8 for compactness) */
    ucs_sys_device_t  sys_dev;       /**< System device index */
    uint8_t           mem_flags;     /**< UCS memory flags */
    size_t            refcount;      /**< Explicit registrations covering
                                          this region, unused with UCM
                                          events */
};


static UCS_CLASS_INIT_FUNC(ucs_memtype_cache_t);
static UCS_CLASS_CLEANUP_FUNC(ucs_memtype_cache_t);

UCS_CLASS_DEFINE(ucs_memtype_cache_t, void);

static UCS_F_ALWAYS_INLINE ucs_memtype_cache_t *ucs_memtype_cache_get_global()
{
    ucs_memtype_cache_t *memtype_cache = NULL;
    ucs_status_t status;

    if (ucs_global_opts.enable_memtype_cache == UCS_NO) {
        return NULL;
    }

    /* Double-check lock scheme */
    if (ucs_unlikely(ucs_memtype_cache_global_instance == NULL) &&
        !ucs_memtype_cache_failed) {
        /* Create the memtype cache outside the lock, to avoid a Coverity error
           of lock inversion with UCS_INIT_ONCE from ucm_set_event_handler() */
        status = UCS_CLASS_NEW(ucs_memtype_cache_t, &memtype_cache);
        if (status != UCS_OK) {
            /* If we failed to create the memtype cache once, do not try again */
            ucs_memtype_cache_failed = 1;
            if (ucs_global_opts.enable_memtype_cache == UCS_YES) {
                ucs_warn("failed to create memtype cache: %s",
                         ucs_status_string(status));
            }
            return NULL;
        }

        ucs_spin_lock(&ucs_memtype_cache_global_instance_lock);
        if (ucs_memtype_cache_global_instance == NULL) {
            ucs_memtype_cache_global_instance = memtype_cache;
        } else {
            /* In case of a race, the memtype cache could already created by
             * another thread, so discard the one created by this thread.
             */
            UCS_CLASS_DELETE(ucs_memtype_cache_t, memtype_cache);
        }
        ucs_spin_unlock(&ucs_memtype_cache_global_instance_lock);
    }

    return ucs_memtype_cache_global_instance;
}

void ucs_memtype_cache_global_create()
{
    ucs_memtype_cache_get_global();
}

static UCS_F_ALWAYS_INLINE void
ucs_memory_info_set_unknown(ucs_memory_info_t *mem_info)
{
    mem_info->type         = UCS_MEMORY_TYPE_UNKNOWN;
    mem_info->sys_dev      = UCS_SYS_DEVICE_ID_UNKNOWN;
    mem_info->base_address = NULL;
    mem_info->alloc_length = -1;
    /* Unknown memory: assume registrable. The type is UNKNOWN so consumers
     * re-detect via the MD anyway, which overwrites this default. */
    mem_info->mem_flags    = UCS_MEM_FLAG_REGISTRABLE;
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
 * - start, end must be aligned to UCS_PGT_ADDR_ALIGN
 */
static ucs_memtype_cache_region_t *
ucs_memtype_cache_insert(ucs_memtype_cache_t *memtype_cache,
                         ucs_pgt_addr_t start, ucs_pgt_addr_t end,
                         ucs_memory_type_t mem_type,
                         ucs_sys_device_t sys_dev, uint8_t mem_flags,
                         size_t refcount)
{
    ucs_memtype_cache_region_t *region;
    ucs_status_t status;
    int ret;

    /* Allocate structure for new region */
    ret = ucs_posix_memalign((void **)&region,
                             ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                             sizeof(ucs_memtype_cache_region_t),
                             "memtype_cache_region");
    if (ret != 0) {
        ucs_warn("failed to allocate memtype_cache region");
        return NULL;
    }

    ucs_assert((start % UCS_PGT_ADDR_ALIGN) == 0);
    ucs_assert((end   % UCS_PGT_ADDR_ALIGN) == 0);

    region->super.start = start;
    region->super.end   = end;
    region->mem_type    = mem_type;
    region->sys_dev     = sys_dev;
    region->mem_flags   = mem_flags;
    region->refcount    = refcount;

    status = UCS_PROFILE_CALL(ucs_pgtable_insert, &memtype_cache->pgtable,
                              &region->super);
    if (status != UCS_OK) {
        ucs_error("failed to insert " UCS_MEMTYPE_CACHE_REGION_FMT ": %s",
                  UCS_MEMTYPE_CACHE_REGION_ARG(region),
                  ucs_status_string(status));
        ucs_free(region);
        return NULL;
    }

    ucs_trace("memtype_cache: insert " UCS_MEMTYPE_CACHE_REGION_FMT,
              UCS_MEMTYPE_CACHE_REGION_ARG(region));
    return region;
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

static int
ucs_memtype_cache_region_same_attributes(const ucs_memtype_cache_region_t *a,
                                         const ucs_memtype_cache_region_t *b)
{
    return (a->mem_type == b->mem_type) && (a->sys_dev == b->sys_dev) &&
           (a->mem_flags == b->mem_flags);
}

/* The cache lock must be held. Change the end of a cached region. */
static void ucs_memtype_cache_region_set_end(ucs_memtype_cache_t *memtype_cache,
                                             ucs_memtype_cache_region_t *region,
                                             ucs_pgt_addr_t end)
{
    ucs_status_t status;

    status = ucs_pgtable_remove(&memtype_cache->pgtable, &region->super);
    ucs_assertv_always(status == UCS_OK,
                       "failed to remove explicit memtype region: %s",
                       ucs_status_string(status));
    region->super.end = end;
    status = ucs_pgtable_insert(&memtype_cache->pgtable, &region->super);
    ucs_assertv_always(status == UCS_OK,
                       "failed to insert explicit memtype region: %s",
                       ucs_status_string(status));
}

/* The cache lock must be held. Make addr a region boundary, keeping the
 * attributes and the reference count on both sides.
 * @return UCS_OK on success, or UCS_ERR_NO_MEMORY if the second half could not
 *         be created. */
static ucs_status_t ucs_memtype_cache_split(ucs_memtype_cache_t *memtype_cache,
                                            ucs_pgt_addr_t addr)
{
    ucs_memtype_cache_region_t *region;
    ucs_pgt_region_t *pgt_region;
    ucs_pgt_addr_t end;

    pgt_region = ucs_pgtable_lookup(&memtype_cache->pgtable, addr);
    if ((pgt_region == NULL) || (pgt_region->start == addr)) {
        return UCS_OK;
    }

    region = ucs_derived_of(pgt_region, ucs_memtype_cache_region_t);
    end    = region->super.end;
    ucs_memtype_cache_region_set_end(memtype_cache, region, addr);
    if (ucs_memtype_cache_insert(memtype_cache, addr, end, region->mem_type,
                                 region->sys_dev, region->mem_flags,
                                 region->refcount) == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

/* The cache lock must be held. Merge neighbors of the range [start, end) which
 * have the same attributes and reference count, to keep the number of regions
 * low. */
static void ucs_memtype_cache_merge(ucs_memtype_cache_t *memtype_cache,
                                    ucs_pgt_addr_t start, ucs_pgt_addr_t end)
{
    ucs_memtype_cache_region_t *prev = NULL;
    ucs_memtype_cache_region_t *region, *tmp;
    UCS_LIST_HEAD(region_list);
    ucs_status_t status;

    ucs_pgtable_search_range(&memtype_cache->pgtable,
                             (start > 0) ? (start - 1) : 0, end,
                             ucs_memtype_cache_region_collect_callback,
                             &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        if ((prev != NULL) && (prev->super.end == region->super.start) &&
            (prev->refcount == region->refcount) &&
            ucs_memtype_cache_region_same_attributes(prev, region)) {
            status = ucs_pgtable_remove(&memtype_cache->pgtable,
                                        &region->super);
            ucs_assertv_always(status == UCS_OK,
                               "failed to remove explicit memtype region: %s",
                               ucs_status_string(status));
            ucs_memtype_cache_region_set_end(memtype_cache, prev,
                                             region->super.end);
            ucs_free(region);
        } else {
            prev = region;
        }
    }
}

void ucs_memtype_cache_register(const void *address, size_t size,
                                ucs_memory_type_t mem_type,
                                ucs_sys_device_t sys_dev, uint8_t mem_flags)
{
    ucs_pgt_addr_t addr = (ucs_pgt_addr_t)address;
    ucs_memtype_cache_region_t *region, *tmp;
    ucs_memtype_cache_t *memtype_cache;
    ucs_pgt_addr_t start, end, pos;
    UCS_LIST_HEAD(region_list);

    if ((ucs_global_opts.enable_memtype_cache != UCS_YES) || (size == 0) ||
        (size > UINTPTR_MAX - addr - (UCS_PGT_ADDR_ALIGN - 1)) ||
        (mem_type == UCS_MEMORY_TYPE_HOST) ||
        (mem_type >= UCS_MEMORY_TYPE_LAST)) {
        return;
    }

    memtype_cache = ucs_memtype_cache_get_global();
    if ((memtype_cache == NULL) || memtype_cache->has_events) {
        return;
    }

    /* The page table requires aligned regions, so cover the whole aligned
     * range */
    start = ucs_align_down_pow2(addr, UCS_PGT_ADDR_ALIGN);
    end   = ucs_align_up_pow2(addr + size, UCS_PGT_ADDR_ALIGN);

    ucs_spin_lock(&memtype_cache->lock);

    if ((ucs_memtype_cache_split(memtype_cache, start) != UCS_OK) ||
        (ucs_memtype_cache_split(memtype_cache, end) != UCS_OK)) {
        goto out_unlock;
    }

    /* Now every existing region intersecting the range lies inside it */
    ucs_pgtable_search_range(&memtype_cache->pgtable, start, end - 1,
                             ucs_memtype_cache_region_collect_callback,
                             &region_list);

    pos = start;
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        if ((pos < region->super.start) &&
            (ucs_memtype_cache_insert(memtype_cache, pos, region->super.start,
                                      mem_type, sys_dev, mem_flags,
                                      1) == NULL)) {
            goto out_unlock;
        }

        /* Same address means the same allocation, so different attributes
         * can only belong to a stale entry. The new registration wins. */
        region->mem_type  = mem_type;
        region->sys_dev   = sys_dev;
        region->mem_flags = mem_flags;
        ++region->refcount;
        pos = region->super.end;
    }

    if ((pos < end) &&
        (ucs_memtype_cache_insert(memtype_cache, pos, end, mem_type, sys_dev,
                                  mem_flags, 1) == NULL)) {
        goto out_unlock;
    }

    ucs_memtype_cache_merge(memtype_cache, start, end);

out_unlock:
    ucs_spin_unlock(&memtype_cache->lock);
}

void ucs_memtype_cache_unregister(const void *address, size_t size)
{
    ucs_memtype_cache_t *memtype_cache = ucs_memtype_cache_global_instance;
    ucs_pgt_addr_t addr                = (ucs_pgt_addr_t)address;
    ucs_memtype_cache_region_t *region, *tmp;
    ucs_pgt_addr_t start, end;
    UCS_LIST_HEAD(region_list);
    ucs_status_t status;

    if ((memtype_cache == NULL) || memtype_cache->has_events || (size == 0) ||
        (size > UINTPTR_MAX - addr - (UCS_PGT_ADDR_ALIGN - 1))) {
        return;
    }

    start = ucs_align_down_pow2(addr, UCS_PGT_ADDR_ALIGN);
    end   = ucs_align_up_pow2(addr + size, UCS_PGT_ADDR_ALIGN);

    ucs_spin_lock(&memtype_cache->lock);

    if ((ucs_memtype_cache_split(memtype_cache, start) != UCS_OK) ||
        (ucs_memtype_cache_split(memtype_cache, end) != UCS_OK)) {
        goto out_unlock;
    }

    ucs_pgtable_search_range(&memtype_cache->pgtable, start, end - 1,
                             ucs_memtype_cache_region_collect_callback,
                             &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        ucs_assert(region->refcount > 0);
        if (--region->refcount > 0) {
            continue;
        }

        status = ucs_pgtable_remove(&memtype_cache->pgtable, &region->super);
        ucs_assertv_always(status == UCS_OK,
                           "failed to remove explicit memtype region: %s",
                           ucs_status_string(status));
        ucs_free(region);
    }

    ucs_memtype_cache_merge(memtype_cache, start, end);

out_unlock:
    ucs_spin_unlock(&memtype_cache->lock);
}

UCS_PROFILE_FUNC_VOID(ucs_memtype_cache_update_internal,
                      (memtype_cache, address, size, mem_type, sys_dev,
                       mem_flags, action),
                      ucs_memtype_cache_t *memtype_cache, const void *address,
                      size_t size, ucs_memory_type_t mem_type,
                      ucs_sys_device_t sys_dev, uint8_t mem_flags,
                      ucs_memtype_cache_action_t action)
{
    ucs_pgt_addr_t start, end, search_start, search_end;
    ucs_memtype_cache_region_t *region, *tmp;
    UCS_LIST_HEAD(region_list);
    ucs_status_t status;

    if (!size) {
        return;
    }

    start = ucs_align_down_pow2((uintptr_t)address,        UCS_PGT_ADDR_ALIGN);
    end   = ucs_align_up_pow2  ((uintptr_t)address + size, UCS_PGT_ADDR_ALIGN);

    ucs_trace("%s: [0x%lx..0x%lx] mem_type %s dev %s flags 0x%x",
              (action == UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE) ? "update" :
                                                                 "remove",
              start, end, ucs_memory_type_names[mem_type],
              ucs_topo_sys_device_get_name(sys_dev), mem_flags);

    search_start = start;
    search_end   = end - 1;

    ucs_spin_lock(&memtype_cache->lock);

    /* find and remove all regions which intersect with new one */
    ucs_pgtable_search_range(&memtype_cache->pgtable, search_start, search_end,
                             ucs_memtype_cache_region_collect_callback,
                             &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        if (action == UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE) {
            if ((region->mem_type == mem_type) &&
                (region->mem_flags == mem_flags)) {
                /* merge current region with overlapping or adjacent regions
                 * of same memory type and flags */
                start = ucs_min(start, region->super.start);
                end   = ucs_max(end, region->super.end);
                ucs_trace("merge with " UCS_MEMTYPE_CACHE_REGION_FMT
                          ": [0x%lx..0x%lx]",
                          UCS_MEMTYPE_CACHE_REGION_ARG(region), start, end);
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
            ucs_error("failed to remove " UCS_MEMTYPE_CACHE_REGION_FMT ": %s",
                      UCS_MEMTYPE_CACHE_REGION_ARG(region),
                      ucs_status_string(status));
            goto out_unlock;
        }

        ucs_trace("memtype_cache: removed " UCS_MEMTYPE_CACHE_REGION_FMT,
                  UCS_MEMTYPE_CACHE_REGION_ARG(region));
    }

    if (action == UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE) {
        ucs_memtype_cache_insert(memtype_cache, start, end, mem_type, sys_dev,
                                 mem_flags, 0);
    }

    /* slice old regions by the new region, to preserve the previous memory type
     * of the non-overlapping parts
     */
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        if (start > region->super.start) {
            /* create previous region */
            ucs_memtype_cache_insert(memtype_cache, region->super.start, start,
                                     region->mem_type, region->sys_dev,
                                     region->mem_flags, 0);
        }
        if (end < region->super.end) {
            /* create next region */
            ucs_memtype_cache_insert(memtype_cache, end, region->super.end,
                                     region->mem_type, region->sys_dev,
                                     region->mem_flags, 0);
        }

        ucs_free(region);
    }

out_unlock:
    ucs_spin_unlock(&memtype_cache->lock);
}

void ucs_memtype_cache_update(const void *address, size_t size,
                              ucs_memory_type_t mem_type,
                              ucs_sys_device_t sys_dev, uint8_t mem_flags)
{
    /* A detected allocation cannot be cached without its free event. */
    if ((ucs_memtype_cache_global_instance == NULL) ||
        !ucs_memtype_cache_global_instance->has_events) {
        return;
    }

    ucs_memtype_cache_update_internal(ucs_memtype_cache_global_instance,
                                      address, size, mem_type, sys_dev,
                                      mem_flags,
                                      UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE);
}

void ucs_memtype_cache_remove(const void *address, size_t size)
{
    if ((ucs_memtype_cache_global_instance == NULL) ||
        !ucs_memtype_cache_global_instance->has_events) {
        return;
    }

    ucs_memtype_cache_update_internal(ucs_memtype_cache_global_instance,
                                      address, size, UCS_MEMORY_TYPE_UNKNOWN,
                                      UCS_SYS_DEVICE_ID_UNKNOWN, 0,
                                      UCS_MEMTYPE_CACHE_ACTION_REMOVE);
}

static void ucs_memtype_cache_event_callback(ucm_event_type_t event_type,
                                             ucm_event_t *event, void *arg)
{
    ucs_memtype_cache_action_t action;

    if (event_type & UCM_EVENT_MEM_TYPE_ALLOC) {
        action = UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE;
    } else if (event_type & UCM_EVENT_MEM_TYPE_FREE) {
        action = UCS_MEMTYPE_CACHE_ACTION_REMOVE;
    } else {
        return;
    }

    ucs_trace("dispatching mem event %d address %p length %zu mem_type %s "
              "sys_dev %u flags 0x%x",
              event_type, event->mem_type.address, event->mem_type.size,
              ucs_memory_type_names[event->mem_type.mem_type],
              event->mem_type.sys_dev, event->mem_type.mem_flags);

    ucs_memtype_cache_update_internal(arg, event->mem_type.address,
                                      event->mem_type.size,
                                      event->mem_type.mem_type,
                                      event->mem_type.sys_dev,
                                      event->mem_type.mem_flags, action);
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

/* The cache lock must be held. Explicit registrations can split an allocation
 * into adjacent regions with different reference counts, so continue into
 * neighbors with the same attributes until the end of the range is covered.
 * @return the end of the contiguous area starting at region. */
static ucs_pgt_addr_t
ucs_memtype_cache_extend_end(ucs_memtype_cache_t *memtype_cache,
                             const ucs_memtype_cache_region_t *region,
                             ucs_pgt_addr_t range_end)
{
    ucs_pgt_addr_t end = region->super.end;
    ucs_pgt_region_t *pgt_region;
    const ucs_memtype_cache_region_t *next;

    while (end < range_end) {
        pgt_region = ucs_pgtable_lookup(&memtype_cache->pgtable, end);
        if (pgt_region == NULL) {
            break;
        }

        next = ucs_derived_of(pgt_region, ucs_memtype_cache_region_t);
        if (!ucs_memtype_cache_region_same_attributes(region, next)) {
            break;
        }

        end = next->super.end;
    }

    return end;
}

UCS_PROFILE_FUNC(ucs_status_t, ucs_memtype_cache_lookup,
                 (address, size, mem_info),
                 const void *address, size_t size, ucs_memory_info_t *mem_info)
{
    ucs_memtype_cache_t *memtype_cache = ucs_memtype_cache_get_global();
    const ucs_pgt_addr_t start         = (uintptr_t)address;
    ucs_memtype_cache_region_t *region;
    ucs_pgt_region_t *pgt_region;
    ucs_pgt_addr_t end;
    ucs_status_t status;

    if (memtype_cache == NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    /* Without UCM events the cache holds only explicit registrations, so skip
     * the lock when there are none. A racing registration is not visible yet,
     * which is the same as a miss. */
    if (!memtype_cache->has_events &&
        (memtype_cache->pgtable.num_regions == 0)) {
        return UCS_ERR_UNSUPPORTED;
    }

    ucs_spin_lock(&memtype_cache->lock);

    pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup, &memtype_cache->pgtable,
                                  start);
    if (pgt_region == NULL) {
        ucs_trace("address 0x%lx not found", start);
        status = memtype_cache->has_events ? UCS_ERR_NO_ELEM :
                                             UCS_ERR_UNSUPPORTED;
        goto out_unlock;
    }

    region = ucs_derived_of(pgt_region, ucs_memtype_cache_region_t);
    end    = pgt_region->end;
    if (ucs_unlikely((start + size) > end) && !memtype_cache->has_events) {
        end = ucs_memtype_cache_extend_end(memtype_cache, region, start + size);
    }

    if (ucs_likely((start + size) <= end)) {
        mem_info->base_address = (void*)region->super.start;
        mem_info->alloc_length = end - region->super.start;
        mem_info->type         = region->mem_type;
        mem_info->sys_dev      = region->sys_dev;
        mem_info->mem_flags    = region->mem_flags;
        ucs_trace_data("0x%lx..0x%lx found in " UCS_MEMTYPE_CACHE_REGION_FMT,
                       start, start + size,
                       UCS_MEMTYPE_CACHE_REGION_ARG(region));
    } else {
        ucs_trace("0x%lx..0x%lx not contained in " UCS_MEMTYPE_CACHE_REGION_FMT,
                  start, start + size, UCS_MEMTYPE_CACHE_REGION_ARG(region));
        ucs_memory_info_set_unknown(mem_info);
    }
    status = UCS_OK;

    /* The memory type cache is not expected to return HOST memory type */
    ucs_assertv(mem_info->type != UCS_MEMORY_TYPE_HOST, "%s (%d)",
                ucs_memory_type_names[mem_info->type], mem_info->type);

out_unlock:
    ucs_spin_unlock(&memtype_cache->lock);
    return status;
}

static UCS_CLASS_INIT_FUNC(ucs_memtype_cache_t)
{
    ucs_status_t status;
    int ret;

    ret = ucs_spinlock_init(&self->lock, 0);
    if (ret) {
        ucs_error("ucs_spinlock_init() failed: %m");
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    status = ucs_pgtable_init(&self->pgtable, ucs_memtype_cache_pgt_dir_alloc,
                              ucs_memtype_cache_pgt_dir_release);
    if (status != UCS_OK) {
        goto err_destroy_rwlock;
    }

    self->has_events = 0;
    status = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_ALLOC |
                                   UCM_EVENT_MEM_TYPE_FREE  |
                                   UCM_EVENT_FLAG_EXISTING_ALLOC,
                                   1000, ucs_memtype_cache_event_callback,
                                   self);
    if (status != UCS_OK) {
        if (ucs_global_opts.enable_memtype_cache != UCS_YES) {
            ucs_diag("failed to set UCM memtype event handler: %s",
                     ucs_status_string(status));
            goto err_cleanup_pgtable;
        }

        if (ucm_global_opts.enable_events) {
            ucs_warn("failed to set UCM memtype event handler: %s; "
                     "using explicit registration", ucs_status_string(status));
        }
    } else {
        self->has_events = 1;
    }

    return UCS_OK;

err_cleanup_pgtable:
    ucs_pgtable_cleanup(&self->pgtable);
err_destroy_rwlock:
    ucs_spinlock_destroy(&self->lock);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(ucs_memtype_cache_t)
{
    if (self->has_events) {
        ucm_unset_event_handler((UCM_EVENT_MEM_TYPE_ALLOC |
                                 UCM_EVENT_MEM_TYPE_FREE),
                                ucs_memtype_cache_event_callback, self);
    }
    ucs_memtype_cache_purge(self);
    ucs_pgtable_cleanup(&self->pgtable);
    ucs_spinlock_destroy(&self->lock);
}

void ucs_memtype_cache_global_init()
{
    ucs_spinlock_init(&ucs_memtype_cache_global_instance_lock, 0);
}

void ucs_memtype_cache_cleanup()
{
    ucs_spinlock_destroy(&ucs_memtype_cache_global_instance_lock);

    if (ucs_memtype_cache_global_instance) {
        UCS_CLASS_DELETE(ucs_memtype_cache_t, ucs_memtype_cache_global_instance);
        ucs_memtype_cache_global_instance = NULL;
    }
}
