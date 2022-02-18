/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_REG_CACHE_INT_H_
#define UCS_REG_CACHE_INT_H_

#include "rcache.h"

#include <ucs/datastruct/list.h>
#include <ucs/stats/stats.h>
#include <ucs/type/spinlock.h>

/* Names of rcache stats counters */
enum {
    UCS_RCACHE_GETS,                /* number of get operations */
    UCS_RCACHE_HITS_FAST,           /* number of fast path hits */
    UCS_RCACHE_HITS_SLOW,           /* number of slow path hits */
    UCS_RCACHE_MISSES,              /* number of misses */
    UCS_RCACHE_MERGES,              /* number of region merges */
    UCS_RCACHE_UNMAPS,              /* number of memory unmap events */
    UCS_RCACHE_UNMAP_INVALIDATES,   /* number of regions invalidated because
                                       of unmap events */
    UCS_RCACHE_PUTS,                /* number of put operations */
    UCS_RCACHE_REGS,                /* number of memory registrations */
    UCS_RCACHE_DEREGS,              /* number of memory deregistrations */
    UCS_RCACHE_STAT_LAST
};


/* The structure represents a group in regestration cache regions distribution.
   Regions are distributed by thier size.
 */
typedef struct ucs_rcache_distribution {
    size_t count; /**< Number of regions in the group */
    size_t total_size; /**< Total size of regions in the group */
} ucs_rcache_distribution_t;

struct ucs_rcache {
    ucs_rcache_params_t params;          /**< rcache parameters (immutable) */

    pthread_rwlock_t    pgt_lock;        /**< Protects the page table and all
                                              regions whose refcount is 0 */
    ucs_pgtable_t       pgtable;         /**< page table to hold the regions */


    ucs_spinlock_t      lock;            /**< Protects 'mp', 'inv_q' and 'gc_list'.
                                              This is a separate lock because we
                                              may want to invalidate regions
                                              while the page table lock is held by
                                              the calling context.
                                              @note: This lock should always be
                                              taken **after** 'pgt_lock'. */
    ucs_mpool_t         mp;              /**< Memory pool to allocate entries for
                                              inv_q and page table entries, since
                                              we cannot use regular malloc().
                                              The backing storage is original mmap()
                                              which does not generate memory events */
    ucs_queue_head_t    inv_q;           /**< Regions which were invalidated during
                                              memory events */
    ucs_list_link_t     gc_list;         /**< list for regions to destroy, regions
                                              could not be destroyed from memhook */

    unsigned long       num_regions;     /**< Total number of managed regions */
    size_t              total_size;      /**< Total size of registered memory */
    size_t              unreleased_size; /**< Total size of the regions in gc_list and in inv_q */

    struct {
        ucs_spinlock_t  lock;            /**< Lock for this structure */
        ucs_list_link_t list;            /**< List of regions, sorted by usage:
                                              The head of the list is the least
                                              recently used region, and the tail
                                              is the most recently used region. */
    } lru;
    
    char                *name;           /**< Name of the cache, for debug purpose */

    UCS_STATS_NODE_DECLARE(stats)

    ucs_list_link_t           list; /**< List entry in global ucs_rcache list */
    ucs_rcache_distribution_t *distribution; /**< Distribution of registration
                                                  cache regions by size */
};


/**
 * @brief Create objects in VFS to represent registration cache and its
 *        features.
 *
 * @param [in] rcache Registration cache object to be described.
 */
void ucs_rcache_vfs_init(ucs_rcache_t *rcache);


/**
 * @brief Get number of bins in the distribution of registration cache region
 *        sizes.
 *
 * @return Number of bins.
 */
size_t ucs_rcache_distribution_get_num_bins();


#define UCS_RCACHE_STAT_MIN_POW2 \
    ucs_roundup_pow2(ucs_global_opts.rcache_stat_min)

#endif
