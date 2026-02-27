/*
 * Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_ZE_IPC_CACHE_H_
#define UCT_ZE_IPC_CACHE_H_

#include <ucs/datastruct/pgtable.h>
#include <ucs/datastruct/list.h>
#include <ucs/type/spinlock.h>
#include <level_zero/ze_api.h>
#include "ze_ipc_md.h"


typedef struct uct_ze_ipc_cache        uct_ze_ipc_cache_t;
typedef struct uct_ze_ipc_cache_region uct_ze_ipc_cache_region_t;
typedef struct uct_ze_ipc_iface        uct_ze_ipc_iface_t;


/**
 * Cache region structure for storing mapped IPC handles
 */
struct uct_ze_ipc_cache_region {
    ucs_pgt_region_t        super;        /**< Base class - page table region */
    ucs_list_link_t         list;         /**< List element */
    uct_ze_ipc_key_t        key;          /**< Remote memory key */
    void                    *mapped_addr; /**< Local mapped address */
    uint64_t                refcount;     /**< Track in-flight ops before unmapping*/
    ze_context_handle_t     ze_context;   /**< Level Zero context */
    int                     dup_fd;       /**< Duplicated file descriptor */
};


/**
 * Cache structure for managing IPC handle mappings
 */
struct uct_ze_ipc_cache {
    pthread_rwlock_t      lock;       /**< Protects the page table */
    ucs_pgtable_t         pgtable;    /**< Page table to hold the regions */
    char                  *name;      /**< Name for debugging */
};


/**
 * Create a new IPC cache
 *
 * @param cache     Pointer to store the created cache
 * @param name      Name for the cache (for debugging)
 * @return UCS_OK on success, error code otherwise
 */
ucs_status_t uct_ze_ipc_create_cache(uct_ze_ipc_cache_t **cache,
                                     const char *name);


/**
 * Destroy an IPC cache
 *
 * @param cache     Cache to destroy
 */
void uct_ze_ipc_destroy_cache(uct_ze_ipc_cache_t *cache);


/**
 * Map an IPC memory handle to a local address (with caching)
 *
 * @param key          Remote memory key containing IPC handle
 * @param ze_context   Level Zero context
 * @param ze_device    Level Zero device
 * @param mapped_addr  Pointer to store the mapped address
 * @param dup_fd       Pointer to store the duplicated file descriptor
 * @return UCS_OK on success, error code otherwise
 */
ucs_status_t uct_ze_ipc_map_memhandle(uct_ze_ipc_key_t *key,
                                      ze_context_handle_t ze_context,
                                      ze_device_handle_t ze_device,
                                      void **mapped_addr,
                                      int *dup_fd);


/**
 * Unmap an IPC memory handle (with caching)
 *
 * @param pid           Process ID of the remote process
 * @param address       Base address of the remote memory
 * @param mapped_addr   Mapped local address
 * @param ze_context    Level Zero context
 * @param dup_fd        Duplicated file descriptor
 * @param cache_enabled Whether caching is enabled
 * @return UCS_OK on success, error code otherwise
 */
ucs_status_t uct_ze_ipc_unmap_memhandle(pid_t pid, uintptr_t address,
                                        void *mapped_addr,
                                        ze_context_handle_t ze_context,
                                        int dup_fd, int cache_enabled);

#endif /* UCT_ZE_IPC_CACHE_H_ */

