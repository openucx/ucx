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

struct ucs_memtype_cache_registration {
    ucs_interval_map_node_t     interval;
    ucs_list_link_t             list;
    ucs_memory_type_t           mem_type;
    ucs_sys_device_t            sys_dev;
    uint8_t                     mem_flags;
    ucs_memtype_cache_region_t *region;
    size_t                      conflicts; /**< Incompatible overlaps */
};

typedef struct {
    ucs_memtype_cache_t              *cache;
    ucs_memtype_cache_registration_t *registration;
    ucs_pgt_addr_t                    restore_start;
    ucs_pgt_addr_t                    restore_end;
} ucs_memtype_cache_registration_ctx_t;

#define UCS_MEMTYPE_CACHE_TOKEN_HASH_KEY(_token) \
    kh_int64_hash_func((uintptr_t)(_token))
KHASH_IMPL(ucs_memtype_cache_tokens, const void*,
           ucs_memtype_cache_registration_t*, 1,
           UCS_MEMTYPE_CACHE_TOKEN_HASH_KEY, kh_int64_hash_equal);

struct ucs_memtype_cache_region {
    ucs_pgt_region_t  super;         /**< Base class - page table region */
    ucs_list_link_t   list;          /**< List element */
    ucs_memory_type_t mem_type;      /**< Memory type (uint8 for compactness) */
    ucs_sys_device_t  sys_dev;       /**< System device index */
    uint8_t           mem_flags;     /**< UCS memory flags */
    ucs_list_link_t   registrations; /**< Registrations sharing this region */
    size_t            full_refs;     /**< Registrations covering this region */
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
                         ucs_sys_device_t sys_dev, uint8_t mem_flags)
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
    ucs_list_head_init(&region->registrations);
    region->full_refs   = 0;

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
ucs_memtype_cache_same_attributes(
        const ucs_memtype_cache_registration_t *registration,
        ucs_memory_type_t mem_type, ucs_sys_device_t sys_dev,
        uint8_t mem_flags)
{
    return (registration->mem_type == mem_type) &&
           (registration->sys_dev == sys_dev) &&
           (registration->mem_flags == mem_flags);
}

static int
ucs_memtype_cache_registrations_compatible(
        const ucs_memtype_cache_registration_t *a,
        const ucs_memtype_cache_registration_t *b)
{
    return ucs_memtype_cache_same_attributes(a, b->mem_type, b->sys_dev,
                                             b->mem_flags) &&
           (((a->interval.start <= b->interval.start) &&
             (a->interval.end >= b->interval.end)) ||
            ((b->interval.start <= a->interval.start) &&
             (b->interval.end >= a->interval.end)));
}

static int
ucs_memtype_cache_region_covers_registration(
        const ucs_memtype_cache_region_t *region,
        const ucs_memtype_cache_registration_t *registration)
{
    return (region->super.start <= registration->interval.start) &&
           (region->super.end >= registration->interval.end) &&
           ucs_memtype_cache_same_attributes(registration, region->mem_type,
                                             region->sys_dev,
                                             region->mem_flags);
}

static void
ucs_memtype_cache_attach_registration(
        ucs_memtype_cache_region_t *region,
        ucs_memtype_cache_registration_t *registration)
{
    ucs_assert(registration->region == NULL);
    ucs_assert(ucs_memtype_cache_region_covers_registration(region,
                                                            registration));

    registration->region = region;
    ucs_list_add_tail(&region->registrations, &registration->list);
    if ((registration->interval.start == region->super.start) &&
        (registration->interval.end == region->super.end)) {
        ++region->full_refs;
    }
}

/* The cache lock must be held. Only registrations sharing this region need
 * their cached entry cleared when a conflicting registration arrives. */
static void
ucs_memtype_cache_remove_explicit(ucs_memtype_cache_t *memtype_cache,
                                  ucs_memtype_cache_region_t *region)
{
    ucs_memtype_cache_registration_t *registration, *tmp;
    ucs_status_t status;

    status = ucs_pgtable_remove(&memtype_cache->pgtable, &region->super);
    ucs_assertv_always(status == UCS_OK,
                       "failed to remove explicit memtype region: %s",
                       ucs_status_string(status));

    ucs_list_for_each_safe(registration, tmp, &region->registrations, list) {
        ucs_list_del(&registration->list);
        registration->region = NULL;
    }
    ucs_free(region);
}

static void
ucs_memtype_cache_attach_covered_callback(ucs_interval_map_node_t *node,
                                          void *arg)
{
    ucs_memtype_cache_region_t *region = arg;
    ucs_memtype_cache_registration_t *registration = ucs_container_of(
            node, ucs_memtype_cache_registration_t, interval);

    if ((registration->region == NULL) && (registration->conflicts == 0) &&
        ucs_memtype_cache_region_covers_registration(region, registration)) {
        ucs_memtype_cache_attach_registration(region, registration);
    }
}

/* The cache lock must be held. Conflict counts include uncached registrations.
 * Each cached region must have an active registration for its full range. */
static void
ucs_memtype_cache_restore_registration(
        ucs_memtype_cache_t *memtype_cache,
        ucs_memtype_cache_registration_t *registration)
{
    ucs_memtype_cache_region_t *region, *tmp;
    ucs_pgt_region_t *pgt_region;
    UCS_LIST_HEAD(region_list);

    if ((registration->region != NULL) || (registration->conflicts != 0)) {
        return;
    }

    pgt_region = ucs_pgtable_lookup(&memtype_cache->pgtable,
                                    registration->interval.start);
    if (pgt_region != NULL) {
        region = ucs_derived_of(pgt_region, ucs_memtype_cache_region_t);
        if (ucs_memtype_cache_region_covers_registration(region,
                                                         registration)) {
            ucs_memtype_cache_attach_registration(region, registration);
            return;
        }
    }

    /* A larger registration can replace cached regions it fully contains. */
    ucs_pgtable_search_range(&memtype_cache->pgtable,
                             registration->interval.start,
                             registration->interval.end - 1,
                             ucs_memtype_cache_region_collect_callback,
                             &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        ucs_assert((registration->interval.start <= region->super.start) &&
                   (registration->interval.end >= region->super.end) &&
                   ucs_memtype_cache_same_attributes(registration,
                                                     region->mem_type,
                                                     region->sys_dev,
                                                     region->mem_flags));
        ucs_list_del(&region->list);
        ucs_memtype_cache_remove_explicit(memtype_cache, region);
    }

    region = ucs_memtype_cache_insert(memtype_cache,
                                      registration->interval.start,
                                      registration->interval.end,
                                      registration->mem_type,
                                      registration->sys_dev,
                                      registration->mem_flags);
    if (region == NULL) {
        return;
    }

    /* Reattach every compatible registration displaced by the new region. */
    ucs_interval_map_foreach_overlapping(
            &memtype_cache->registrations, region->super.start,
            region->super.end, ucs_memtype_cache_attach_covered_callback,
            region);
}

static void
ucs_memtype_cache_restore_overlap_callback(ucs_interval_map_node_t *node,
                                           void *arg)
{
    ucs_memtype_cache_registration_t *registration = ucs_container_of(
            node, ucs_memtype_cache_registration_t, interval);

    ucs_memtype_cache_restore_registration(arg, registration);
}

static void
ucs_memtype_cache_registration_add_overlap(ucs_interval_map_node_t *node,
                                           void *arg)
{
    ucs_memtype_cache_registration_ctx_t *ctx = arg;
    ucs_memtype_cache_registration_t *other = ucs_container_of(
            node, ucs_memtype_cache_registration_t, interval);
    ucs_memtype_cache_registration_t *registration = ctx->registration;

    if (ucs_memtype_cache_registrations_compatible(registration, other)) {
        return;
    }

    ++registration->conflicts;
    ++other->conflicts;
    if (other->region != NULL) {
        ctx->restore_start = ucs_min(ctx->restore_start,
                                     other->region->super.start);
        ctx->restore_end   = ucs_max(ctx->restore_end,
                                     other->region->super.end);
        ucs_trace("memtype_cache: registration [0x%lx..0x%lx] "
                  "conflicts with " UCS_MEMTYPE_CACHE_REGION_FMT,
                  (unsigned long)registration->interval.start,
                  (unsigned long)registration->interval.end,
                  UCS_MEMTYPE_CACHE_REGION_ARG(other->region));
        ucs_memtype_cache_remove_explicit(ctx->cache, other->region);
    }
}

static void
ucs_memtype_cache_registration_remove_overlap(ucs_interval_map_node_t *node,
                                              void *arg)
{
    ucs_memtype_cache_registration_ctx_t *ctx = arg;
    ucs_memtype_cache_registration_t *other = ucs_container_of(
            node, ucs_memtype_cache_registration_t, interval);

    if (ucs_memtype_cache_registrations_compatible(ctx->registration, other)) {
        return;
    }

    ucs_assert(other->conflicts > 0);
    --other->conflicts;
}

void ucs_memtype_cache_register(const void *address, size_t size,
                                ucs_memory_type_t mem_type,
                                ucs_sys_device_t sys_dev, uint8_t mem_flags,
                                const void *token)
{
    ucs_pgt_addr_t start = (ucs_pgt_addr_t)address;
    ucs_memtype_cache_registration_t *registration;
    ucs_memtype_cache_registration_ctx_t ctx;
    ucs_memtype_cache_t *memtype_cache;
    khiter_t hash_it;
    int ret;

    if ((ucs_global_opts.enable_memtype_cache != UCS_YES) ||
        (size == 0) ||
        (size > UINTPTR_MAX - start - (UCS_PGT_ADDR_ALIGN - 1)) ||
        (token == NULL) || (mem_type == UCS_MEMORY_TYPE_HOST) ||
        (mem_type >= UCS_MEMORY_TYPE_LAST)) {
        return;
    }

    memtype_cache = ucs_memtype_cache_get_global();
    if ((memtype_cache == NULL) || memtype_cache->has_events) {
        return;
    }

    registration = ucs_malloc(sizeof(*registration),
                              "memtype_cache_registration");
    if (registration == NULL) {
        return;
    }

    /* The page table requires aligned regions, so cover the whole aligned
     * range. Neighboring allocations cannot share an aligned granule. */
    registration->interval.start = ucs_align_down_pow2(start,
                                                       UCS_PGT_ADDR_ALIGN);
    registration->interval.end   = ucs_align_up_pow2(start + size,
                                                     UCS_PGT_ADDR_ALIGN);
    registration->mem_type       = mem_type;
    registration->sys_dev        = sys_dev;
    registration->mem_flags      = mem_flags;
    registration->region         = NULL;
    registration->conflicts      = 0;

    ucs_spin_lock(&memtype_cache->lock);
    hash_it = kh_put(ucs_memtype_cache_tokens, &memtype_cache->token_hash,
                     token, &ret);
    if ((ret == UCS_KH_PUT_FAILED) ||
        (ret == UCS_KH_PUT_KEY_PRESENT)) {
        goto out_unlock;
    }
    kh_value(&memtype_cache->token_hash, hash_it) = registration;

    ctx.cache         = memtype_cache;
    ctx.registration  = registration;
    ctx.restore_start = registration->interval.start;
    ctx.restore_end   = registration->interval.end;
    ucs_interval_map_foreach_overlapping(
            &memtype_cache->registrations, registration->interval.start,
            registration->interval.end,
            ucs_memtype_cache_registration_add_overlap, &ctx);
    ucs_interval_map_insert(&memtype_cache->registrations,
                            &registration->interval,
                            registration->interval.start,
                            registration->interval.end);
    ucs_interval_map_foreach_overlapping(
            &memtype_cache->registrations, ctx.restore_start, ctx.restore_end,
            ucs_memtype_cache_restore_overlap_callback, memtype_cache);
    ucs_spin_unlock(&memtype_cache->lock);
    return;

out_unlock:
    ucs_spin_unlock(&memtype_cache->lock);
    ucs_free(registration);
}

void ucs_memtype_cache_unregister(const void *token)
{
    ucs_memtype_cache_t *memtype_cache = ucs_memtype_cache_global_instance;
    ucs_memtype_cache_registration_t *found;
    ucs_memtype_cache_region_t *region;
    ucs_memtype_cache_registration_ctx_t ctx;
    khiter_t hash_it;
    int region_was_removed = 0;

    if ((memtype_cache == NULL) || memtype_cache->has_events) {
        return;
    }

    ucs_spin_lock(&memtype_cache->lock);
    hash_it = kh_get(ucs_memtype_cache_tokens, &memtype_cache->token_hash,
                     token);
    if (hash_it == kh_end(&memtype_cache->token_hash)) {
        goto out_unlock;
    }

    found = kh_value(&memtype_cache->token_hash, hash_it);
    kh_del(ucs_memtype_cache_tokens, &memtype_cache->token_hash, hash_it);
    ucs_interval_map_remove(&memtype_cache->registrations, &found->interval);
    if (found->region != NULL) {
        region = found->region;
        ucs_list_del(&found->list);
        found->region = NULL;
        if ((found->interval.start == region->super.start) &&
            (found->interval.end == region->super.end)) {
            ucs_assert(region->full_refs > 0);
            --region->full_refs;
        }
        if (region->full_refs == 0) {
            ucs_memtype_cache_remove_explicit(memtype_cache, region);
            region_was_removed = 1;
        }
    }

    if (found->conflicts != 0) {
        ctx.cache        = memtype_cache;
        ctx.registration = found;
        ucs_interval_map_foreach_overlapping(
                &memtype_cache->registrations, found->interval.start,
                found->interval.end,
                ucs_memtype_cache_registration_remove_overlap, &ctx);
    }
    if ((found->conflicts != 0) || region_was_removed) {
        ucs_interval_map_foreach_overlapping(
                &memtype_cache->registrations, found->interval.start,
                found->interval.end,
                ucs_memtype_cache_restore_overlap_callback, memtype_cache);
    }
    ucs_free(found);

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
                                 mem_flags);
    }

    /* slice old regions by the new region, to preserve the previous memory type
     * of the non-overlapping parts
     */
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        if (start > region->super.start) {
            /* create previous region */
            ucs_memtype_cache_insert(memtype_cache, region->super.start, start,
                                     region->mem_type, region->sys_dev,
                                     region->mem_flags);
        }
        if (end < region->super.end) {
            /* create next region */
            ucs_memtype_cache_insert(memtype_cache, end, region->super.end,
                                     region->mem_type, region->sys_dev,
                                     region->mem_flags);
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
    ucs_memtype_cache_registration_t *registration;
    UCS_LIST_HEAD(region_list);

    ucs_trace_func("memtype_cache purge");

    ucs_pgtable_purge(&memtype_cache->pgtable,
                      ucs_memtype_cache_region_collect_callback, &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        ucs_free(region);
    }

    kh_foreach_value(&memtype_cache->token_hash, registration, {
        ucs_free(registration);
    });
    kh_destroy_inplace(ucs_memtype_cache_tokens, &memtype_cache->token_hash);
}

UCS_PROFILE_FUNC(ucs_status_t, ucs_memtype_cache_lookup,
                 (address, size, mem_info),
                 const void *address, size_t size, ucs_memory_info_t *mem_info)
{
    ucs_memtype_cache_t *memtype_cache = ucs_memtype_cache_get_global();
    const ucs_pgt_addr_t start         = (uintptr_t)address;
    ucs_memtype_cache_region_t *region;
    ucs_pgt_region_t *pgt_region;
    ucs_status_t status;

    if (memtype_cache == NULL) {
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
    if (ucs_likely((start + size) <= pgt_region->end)) {
        mem_info->base_address = (void*)region->super.start;
        mem_info->alloc_length = region->super.end - region->super.start;
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

    ucs_interval_map_init(&self->registrations);
    kh_init_inplace(ucs_memtype_cache_tokens, &self->token_hash);
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
    kh_destroy_inplace(ucs_memtype_cache_tokens, &self->token_hash);
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
