/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_PTR_CACHE_H_
#define UCS_PTR_CACHE_H_

#include <ucs/datastruct/pgtable.h>
#include <ucs/datastruct/list.h>
#include <ucs/stats/stats_fwd.h>
#include <ucm/api/ucm.h>

typedef struct ucs_ptrcache         ucs_ptrcache_t;
typedef struct ucs_ptrcache_region  ucs_ptrcache_region_t;


struct ucs_ptrcache_region {
    ucs_pgt_region_t    super;    /**< Base class - page table region */
    ucs_list_link_t     list;     /**< List element */
    ucm_mem_type_t      mem_type; /**< Memory type the pointer belongs to */
};


struct ucs_ptrcache {
    pthread_rwlock_t      lock;       /**< protests the page table */
    ucs_pgtable_t         pgtable;    /**< Page table to hold the regions */
    char                  *name;
};


/**
 * Create a pointer cache.
 *
 * @param [in]  params        Pointer cache parameters.
 * @param [in]  name          Pointer cache name, for debugging.
 * @param [in]  stats         Pointer to statistics parent node.
 * @param [out] ptrcache_p    Filled with a pointer to the pointer cache.
 */
ucs_status_t ucs_ptrcache_create(const char *name, ucs_ptrcache_t **ptrcache_p);


/**
 * Destroy a pointer cache.
 *
 * @param [in]  ptrcache      Pointer cache to destroy.
 */
void ucs_ptrcache_destroy(ucs_ptrcache_t *ptrcache);


/** Find if address range is in pointer cache.
 *
 * @param [in]  ptrcache      Pointer cache to search
 * @param [in]  address       Address to lookup
 * @param [in]  length        Length of the memory
 * @param [out] ucm_mem_type  memory type of the pointer
 *
 * @return 1 if address cache hit, 0 cache miss.
 */
int ucs_ptrcache_lookup(ucs_ptrcache_t *ptrcache, void *address, size_t length,
                        ucm_mem_type_t *ucm_mem_type);


#endif
