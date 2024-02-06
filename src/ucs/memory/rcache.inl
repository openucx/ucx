/**
 * Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_RCACHE_INL_
#define UCS_RCACHE_INL_

#include "rcache_int.h"

#include <ucs/arch/atomic.h>
#include <ucs/datastruct/queue.h>
#include <ucs/profile/profile.h>


static UCS_F_ALWAYS_INLINE int
ucs_rcache_region_get(ucs_rcache_region_t *region, int prot, size_t alignment)
{
    return ucs_test_all_flags(region->prot, prot) &&
           ((alignment == 1) || (region->alignment >= alignment)) &&
           ucs_lockless_sync_get(&region->lls);
}


/* LRU spinlock must be held */
static UCS_F_ALWAYS_INLINE void
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
static UCS_F_ALWAYS_INLINE void
ucs_rcache_region_lru_remove(ucs_rcache_t *rcache, ucs_rcache_region_t *region)
{
    if (!(region->lru_flags & UCS_RCACHE_LRU_FLAG_IN_LRU)) {
        return;
    }

    ucs_rcache_region_trace(rcache, region, "lru remove");
    ucs_list_del(&region->lru_list);
    region->lru_flags &= ~UCS_RCACHE_LRU_FLAG_IN_LRU;
}


static UCS_F_ALWAYS_INLINE ucs_rcache_region_t *
ucs_rcache_lookup_unsafe(ucs_rcache_t *rcache, void *address, size_t length,
                         size_t alignment, int prot)
{
    ucs_pgt_addr_t start = (uintptr_t)address;
    ucs_pgt_region_t *pgt_region;
    ucs_rcache_region_t *region;

    ucs_trace_func("rcache=%s, address=%p, length=%zu", rcache->name, address,
                   length);

    if (ucs_unlikely(!ucs_queue_is_empty(&rcache->inv_q))) {
        return NULL;
    }

    pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup, &rcache->pgtable, start);
    if (ucs_unlikely(pgt_region == NULL)) {
        return NULL;
    }

    region = ucs_derived_of(pgt_region, ucs_rcache_region_t);
    if (ucs_unlikely(((start + length) > region->super.end) ||
        !ucs_rcache_region_get(region, prot, alignment))) {
        return NULL;
    }

    ucs_rcache_region_lru_remove(rcache, region);
    UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_GETS, 1);
    UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_HITS_FAST, 1);
    return region;
}


static UCS_F_ALWAYS_INLINE void
ucs_rcache_region_put_unsafe(ucs_rcache_t *rcache, ucs_rcache_region_t *region)
{
    ucs_rcache_region_trace(rcache, region, "put unsafe");

    ucs_rcache_region_lru_add(rcache, region);

    if (ucs_unlikely(ucs_lockless_sync_put(&region->lls))) {
        pthread_rwlock_wrlock(&rcache->pgt_lock);
        ucs_mem_region_destroy_internal(rcache, region);
        pthread_rwlock_unlock(&rcache->pgt_lock);
    }

    UCS_STATS_UPDATE_COUNTER(rcache->stats, UCS_RCACHE_PUTS, 1);
}

#endif
