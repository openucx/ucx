/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_MEMTYPE_CACHE_H_
#define UCS_MEMTYPE_CACHE_H_

#include "memory_type.h"

#include <ucs/datastruct/pgtable.h>
#include <ucs/datastruct/list.h>
#include <ucs/stats/stats_fwd.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/sys/topo.h>
#include <pthread.h>


BEGIN_C_DECLS

typedef struct ucs_memtype_cache         ucs_memtype_cache_t;
typedef struct ucs_memtype_cache_region  ucs_memtype_cache_region_t;


/* The single global instance of memory type cache */
extern ucs_memtype_cache_t *ucs_memtype_cache_global_instance;


/* Memory information record */
typedef struct ucs_memory_info {
    ucs_memory_type_t type;          /**< Memory type, use uint8 for compact size */
    ucs_sys_device_t  sys_dev;       /**< System device index */
    void              *base_address; /**< Base address of the underlying allocation */
    size_t            alloc_length;  /**< Whole length of the underlying allocation */
} ucs_memory_info_t;


struct ucs_memtype_cache_region {
    ucs_pgt_region_t    super;    /**< Base class - page table region */
    ucs_list_link_t     list;     /**< List element */
    ucs_memory_info_t   mem_info; /**< Memory type and system device the address
                                       belongs to */
};


struct ucs_memtype_cache {
    pthread_rwlock_t      lock;       /**< protests the page table */
    ucs_pgtable_t         pgtable;    /**< Page table to hold the regions */
};


void ucs_memtype_cache_global_init();
void ucs_memtype_cache_cleanup();


/**
 * Find if address range is in memtype cache.
 *
 * @param [in]  address         Address to lookup.
 * @param [in]  size            Length of the memory.
 * @param [out] mem_info        Set to the memory info of the address range.
 *                              UCS_MEMORY_TYPE_UNKNOWN is a special value which
 *                              means the memory type is an unknown non-host
 *                              memory, and should be detected in another way.
 *
 * @return Error code.
 */
ucs_status_t ucs_memtype_cache_lookup(const void *address, size_t size,
                                      ucs_memory_info_t *mem_info);


/**
 * Update the memory type of an address range.
 * Can be used after @ucs_memtype_cache_lookup returns UCM_MEM_TYPE_LAST, to
 * set the memory type after it was detected.
 *
 * @param [in]  address         Start address to update.
 * @param [in]  size            Size of the memory to update.
 * @param [in]  mem_info        Set the memory info of the address range to this
 *                              value.
 */
void ucs_memtype_cache_update(const void *address, size_t size,
                              const ucs_memory_info_t *mem_info);


/**
 * Remove the address range from a memtype cache.
 *
 * @param [in]  address         Start address to remove.
 * @param [in]  size            Size of the memory to remove.
 */
void ucs_memtype_cache_remove(const void *address, size_t size);


/**
 * Helper function to set memory info structure to host memory type.
 *
 * @param [out] mem_info        Pointer to memory info structure.
 */
void ucs_memory_info_set_host(ucs_memory_info_t *mem_info);


/**
 * Find if global memtype_cache is empty.
 *
 * @return 1 if empty 0 if otherwise.
 */
static UCS_F_ALWAYS_INLINE int ucs_memtype_cache_is_empty()
{
    return (ucs_memtype_cache_global_instance != NULL) &&
           (ucs_memtype_cache_global_instance->pgtable.num_regions == 0);
}

END_C_DECLS

#endif
