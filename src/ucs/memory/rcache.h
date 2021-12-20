/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_REG_CACHE_H_
#define UCS_REG_CACHE_H_

/*
 * Memory registration cache - holds registered memory regions, takes care of
 * memory invalidation (if it's unmapped), merging of regions, protection flags.
 * This data structure is thread safe.
 */
#include <ucs/datastruct/pgtable.h>
#include <ucs/datastruct/list.h>
#include <ucs/datastruct/queue_types.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/stats/stats_fwd.h>
#include <sys/mman.h>


#define UCS_RCACHE_PROT_FMT "%c%c"
#define UCS_RCACHE_PROT_ARG(_prot) \
    ((_prot) & PROT_READ)  ? 'r' : '-', \
    ((_prot) & PROT_WRITE) ? 'w' : '-'

/*
 * Minimal rcache alignment.
 */
#define UCS_RCACHE_MIN_ALIGNMENT UCS_PGT_ADDR_ALIGN


typedef struct ucs_rcache         ucs_rcache_t;
typedef struct ucs_rcache_ops     ucs_rcache_ops_t;
typedef struct ucs_rcache_params  ucs_rcache_params_t;
typedef struct ucs_rcache_region  ucs_rcache_region_t;

/*
 * Memory region flags.
 */
enum {
    UCS_RCACHE_REGION_FLAG_REGISTERED = UCS_BIT(0), /**< Memory registered */
    UCS_RCACHE_REGION_FLAG_PGTABLE    = UCS_BIT(1), /**< In the page table */
};

/*
 * Memory registration flags.
 */
enum {
    UCS_RCACHE_MEM_REG_HIDE_ERRORS = UCS_BIT(0) /**< Hide errors on memory registration */
};

/*
 * Rcache flags.
 */
enum {
    UCS_RCACHE_FLAG_NO_PFN_CHECK  = UCS_BIT(0), /**< PFN check not supported for this rcache */
    UCS_RCACHE_FLAG_PURGE_ON_FORK = UCS_BIT(1), /**< purge rcache on fork */
};

/*
 * Rcache LRU flags.
 */
enum {
    UCS_RCACHE_LRU_FLAG_IN_LRU = UCS_BIT(0) /**< In LRU */
};


typedef void (*ucs_rcache_invalidate_comp_func_t)(void *arg);


/*
 * Registration cache operations.
 */
struct ucs_rcache_ops {
    /**
     * Register a memory region.
     *
     * @param [in]  context    User context, as passed to @ref ucs_rcache_create
     * @param [in]  rcache     Pointer to the registration cache.
     * @param [in]  arg        Custom argument passed to @ref ucs_rcache_get().
     * @param [in]  region     Memory region to register. This may point to a larger
     *                          user-defined structure, as specified by the field
     *                          `region_struct_size' in @ref ucs_rcache_params.
     *                         This function may store relevant information (such
     *                          as memory keys) inside the larger structure.
     * @param [in]  flags      Memory registration flags.
     *
     * @return UCS_OK if registration is successful, error otherwise.
     *
     * @note This function should be able to handle inaccessible memory addresses
     *       and return error status in this case, without any destructive consequences
     *       such as error messages or fatal failure.
     */
    ucs_status_t           (*mem_reg)(void *context, ucs_rcache_t *rcache,
                                      void *arg, ucs_rcache_region_t *region,
                                      uint16_t flags);
   /**
    * Deregister a memory region.
    *
    * @param [in]  context    User context, as passed to @ref ucs_rcache_create
    * @param [in]  rcache     Pointer to the registration cache.
    * @param [in]  region     Memory region to deregister.
    */
    void                   (*mem_dereg)(void *context, ucs_rcache_t *rcache,
                                        ucs_rcache_region_t *region);

    /**
     * Dump memory region information to a string buffer.
     * (Only the user-defined part of the memory region should be dumped)
     *
     * @param [in]  context    User context, as passed to @ref ucs_rcache_create
     * @param [in]  rcache     Pointer to the registration cache.
     * @param [in]  region    Memory region to dump.
     * @param [in]  buf       String buffer to dump to.
     * @param [in]  max       Maximal length of the string buffer.
     */
    void                   (*dump_region)(void *context, ucs_rcache_t *rcache,
                                          ucs_rcache_region_t *region,
                                          char *buf, size_t max);
};


struct ucs_rcache_params {
    size_t                 region_struct_size;  /**< Size of memory region structure,
                                                     must be at least the size
                                                     of @ref ucs_rcache_region_t */
    size_t                 alignment;           /**< Force-align regions to this size.
                                                     Must be smaller or equal to
                                                     system page size. */
    size_t                 max_alignment;       /**< Maximum alignment */
    int                    ucm_events;          /**< UCM events to register. Currently
                                                     UCM_EVENT_VM_UNMAPPED and
                                                     UCM_EVENT_MEM_TYPE_FREE are supported */
    int                    ucm_event_priority;  /**< Priority of memory events */
    const ucs_rcache_ops_t *ops;                /**< Memory operations functions */
    void                   *context;            /**< User-defined context that will
                                                     be passed to mem_reg/mem_dereg */
    int                    flags;               /**< Flags */
    unsigned long          max_regions;         /**< Maximal number of regions */
    size_t                 max_size;            /**< Maximal total size of regions */
    size_t                 max_unreleased;      /**< Threshold for triggering a cleanup */
};


struct ucs_rcache_region {
    ucs_pgt_region_t       super;     /**< Base class - page table region */
    ucs_list_link_t        lru_list;  /**< LRU list element */
    ucs_list_link_t        tmp_list;  /**< Temp list element */
    ucs_list_link_t        comp_list; /**< Completion list element */
    volatile uint32_t      refcount;  /**< Reference count, including +1 if it's
                                           in the page table */
    ucs_status_t           status;    /**< Current status code */
    uint8_t                prot;      /**< Protection bits */
    uint8_t                flags;     /**< Status flags. Protected by page table lock. */
    uint8_t                lru_flags; /**< LRU flags */
    union {
        uint64_t           priv;      /**< Used internally */
        unsigned long     *pfn;       /**< Pointer to PFN array. In case if requested
                                           evaluation more than 1 page - PFN array is
                                           allocated, if 1 page requested - used
                                           in-place priv value. */
    };
};


/**
 * Create a memory registration cache.
 *
 * @param [in]  params        Registration cache parameters.
 * @param [in]  name          Registration cache name, for debugging.
 * @param [in]  stats_parent  Pointer to statistics parent node.
 * @param [out] rcache_p      Filled with a pointer to the registration cache.
 */
ucs_status_t ucs_rcache_create(const ucs_rcache_params_t *params, const char *name,
                               ucs_stats_node_t *stats_parent, ucs_rcache_t **rcache_p);


/**
 * Destroy a memory registration cache.
 *
 * @param [in]  rcache      Registration cache to destroy.
 */
void ucs_rcache_destroy(ucs_rcache_t *rcache);


/**
 * Resolve buffer in the registration cache, or register it if not found.
 * TODO register after N usages.
 *
 * @param [in]  rcache      Memory registration cache.
 * @param [in]  address     Address to register or resolve.
 * @param [in]  length      Length of buffer to register or resolve.
 * @param [in]  prot        Requested access flags, PROT_xx (same as passed to mmap).
 * @param [in]  arg         Custom argument passed down to memory registration
 *                          callback, if a memory registration happens during
 *                          this call.
 * @param [out] region_p    On success, filled with a pointer to the memory
 *                          region. The user could put more data in the region
 *                          structure in mem_reg() function.
 *
 * On success succeeds, the memory region reference count is incremented by 1.
 *
 * @return Error code.
 */
ucs_status_t ucs_rcache_get(ucs_rcache_t *rcache, void *address, size_t length,
                            int prot, void *arg, ucs_rcache_region_t **region_p);


/**
 * Increment memory region reference count.
 *
 * @param [in]  rcache      Memory registration cache.
 * @param [in]  region      Memory region whose reference count to increment.
 */
void ucs_rcache_region_hold(ucs_rcache_t *rcache, ucs_rcache_region_t *region);


/**
 * Decrement memory region reference count and possibly destroy it.
 *
 * @param [in]  rcache      Memory registration cache.
 * @param [in]  region      Memory region to release.
 */
void ucs_rcache_region_put(ucs_rcache_t *rcache, ucs_rcache_region_t *region);


/**
  * Invalidate memory region and possibly destroy it.
  *
  * @param [in] rcache    Memory registration cache.
  * @param [in] region    Memory region to invalidate.
  * @param [in] cb        Completion callback, is called when region is
  *                       released. Callback cannot do any operations which may
  *                       access the rcache.
  * @param [in] arg       Completion argument passed to completion callback.
  */
void ucs_rcache_region_invalidate(ucs_rcache_t *rcache,
                                  ucs_rcache_region_t *region,
                                  ucs_rcache_invalidate_comp_func_t cb,
                                  void *arg);


#endif
