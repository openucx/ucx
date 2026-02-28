/*
 * Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ze_ipc_cache.h"
#include "ze_ipc_iface.h"
#include "ze_ipc_ep.h"
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/profile/profile.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/string.h>
#include <ucs/sys/ptr_arith.h>
#include <ucs/datastruct/khash.h>
#include <unistd.h>


typedef struct uct_ze_ipc_cache_hash_key {
    pid_t               pid;
    uint32_t            _pad;  /* Ensure 8-byte alignment for ze_context */
    ze_context_handle_t ze_context;
} UCS_S_PACKED uct_ze_ipc_cache_hash_key_t;


static UCS_F_ALWAYS_INLINE int
uct_ze_ipc_cache_hash_equal(uct_ze_ipc_cache_hash_key_t key1,
                            uct_ze_ipc_cache_hash_key_t key2)
{
    return (key1.pid == key2.pid) && (key1.ze_context == key2.ze_context);
}


static UCS_F_ALWAYS_INLINE khint32_t
uct_ze_ipc_cache_hash_func(uct_ze_ipc_cache_hash_key_t key)
{
    return kh_int_hash_func((uintptr_t)key.pid ^ (uintptr_t)key.ze_context);
}


KHASH_INIT(ze_ipc_rem_cache, uct_ze_ipc_cache_hash_key_t,
           uct_ze_ipc_cache_t*, 1, uct_ze_ipc_cache_hash_func,
           uct_ze_ipc_cache_hash_equal);


typedef struct uct_ze_ipc_remote_cache {
    khash_t(ze_ipc_rem_cache) hash;
    ucs_recursive_spinlock_t  lock;
} uct_ze_ipc_remote_cache_t;


static uct_ze_ipc_remote_cache_t uct_ze_ipc_remote_cache;


static ucs_pgt_dir_t *uct_ze_ipc_cache_pgt_dir_alloc(const ucs_pgtable_t *pgtable)
{
    void *ptr;
    int ret;

    ret = ucs_posix_memalign(&ptr,
                             ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                             sizeof(ucs_pgt_dir_t), "ze_ipc_cache_pgdir");
    return (ret == 0) ? ptr : NULL;
}


static void uct_ze_ipc_cache_pgt_dir_release(const ucs_pgtable_t *pgtable,
                                             ucs_pgt_dir_t *dir)
{
    ucs_free(dir);
}


static void
uct_ze_ipc_cache_region_collect_callback(const ucs_pgtable_t *pgtable,
                                         ucs_pgt_region_t *pgt_region,
                                         void *arg)
{
    ucs_list_link_t *list = arg;
    uct_ze_ipc_cache_region_t *region;

    region = ucs_derived_of(pgt_region, uct_ze_ipc_cache_region_t);
    ucs_list_add_tail(list, &region->list);
}


static ucs_status_t
uct_ze_ipc_close_memhandle(uct_ze_ipc_cache_region_t *region)
{
    ze_result_t ret;

    ret = zeMemCloseIpcHandle(region->ze_context, region->mapped_addr);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_warn("zeMemCloseIpcHandle failed with error 0x%x", ret);
        return UCS_ERR_IO_ERROR;
    }

    if (region->dup_fd >= 0) {
        close(region->dup_fd);
    }

    return UCS_OK;
}


static void uct_ze_ipc_cache_purge(uct_ze_ipc_cache_t *cache)
{
    uct_ze_ipc_cache_region_t *region, *tmp;
    ucs_list_link_t region_list;

    ucs_list_head_init(&region_list);
    ucs_pgtable_purge(&cache->pgtable, uct_ze_ipc_cache_region_collect_callback,
                      &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        uct_ze_ipc_close_memhandle(region);
        ucs_free(region);
    }
    ucs_trace("%s: ze ipc cache purged", cache->name);
}


static ucs_status_t
uct_ze_ipc_open_memhandle(uct_ze_ipc_key_t *key,
                          ze_context_handle_t ze_context,
                          ze_device_handle_t ze_device,
                          void **mapped_addr, int *dup_fd)
{
    ze_ipc_mem_handle_t local_handle;
    int remote_fd;
    ze_result_t ret;

    /* Extract fd from IPC handle */
    memcpy(&local_handle, &key->ipc_handle, sizeof(local_handle));
    remote_fd = *(int*)local_handle.data;

    /* Duplicate the file descriptor from remote process */
    if (key->pid != getpid() && remote_fd > 0 && remote_fd < 65536) {
        *dup_fd = uct_ze_ipc_dup_fd_from_pid(key->pid, remote_fd);
        if (*dup_fd < 0) {
            ucs_error("failed to duplicate fd %d from pid %d", remote_fd, key->pid);
            return UCS_ERR_IO_ERROR;
        }
        /* Update handle with local fd */
        *(int*)local_handle.data = *dup_fd;
    } else {
        *dup_fd = -1;
    }

    ret = zeMemOpenIpcHandle(ze_context, ze_device, local_handle,
                             0, mapped_addr);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_error("zeMemOpenIpcHandle failed with error 0x%x", ret);
        if (*dup_fd >= 0) {
            close(*dup_fd);
            *dup_fd = -1;
        }
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}


static void uct_ze_ipc_cache_invalidate_regions(uct_ze_ipc_cache_t *cache,
                                                void *from, void *to)
{
    ucs_list_link_t region_list;
    ucs_status_t status;
    uct_ze_ipc_cache_region_t *region, *tmp;

    ucs_list_head_init(&region_list);
    ucs_pgtable_search_range(&cache->pgtable, (ucs_pgt_addr_t)from,
                             (ucs_pgt_addr_t)to - 1,
                             uct_ze_ipc_cache_region_collect_callback,
                             &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        status = ucs_pgtable_remove(&cache->pgtable, &region->super);
        if (status != UCS_OK) {
            ucs_error("failed to remove address:%p from cache (%s)",
                      (void *)region->key.address, ucs_status_string(status));
        }

        status = uct_ze_ipc_close_memhandle(region);
        if (status != UCS_OK) {
            ucs_error("failed to close memhandle for base addr:%p (%s)",
                      (void *)region->key.address, ucs_status_string(status));
        }

        ucs_free(region);
    }
    ucs_trace("%s: closed memhandles in the range [%p..%p]",
              cache->name, from, to);
}


static ucs_status_t
uct_ze_ipc_get_remote_cache(pid_t pid, ze_context_handle_t ze_context,
                            uct_ze_ipc_cache_t **cache)
{
    ucs_status_t status = UCS_OK;
    char target_name[64];
    uct_ze_ipc_cache_hash_key_t key;
    khiter_t khiter;
    int khret;

    ucs_recursive_spin_lock(&uct_ze_ipc_remote_cache.lock);

    key.ze_context = ze_context;
    key.pid        = pid;
    key._pad       = 0;

    khiter = kh_put(ze_ipc_rem_cache, &uct_ze_ipc_remote_cache.hash, key,
                    &khret);
    if ((khret == UCS_KH_PUT_BUCKET_EMPTY) ||
        (khret == UCS_KH_PUT_BUCKET_CLEAR)) {
        ucs_snprintf_safe(target_name, sizeof(target_name), "dest:%d:%p",
                          key.pid, key.ze_context);
        status = uct_ze_ipc_create_cache(cache, target_name);
        if (status != UCS_OK) {
            kh_del(ze_ipc_rem_cache, &uct_ze_ipc_remote_cache.hash, khiter);
            ucs_error("could not create ze ipc cache: %s",
                      ucs_status_string(status));
            goto err_unlock;
        }

        kh_val(&uct_ze_ipc_remote_cache.hash, khiter) = *cache;
    } else if (khret == UCS_KH_PUT_KEY_PRESENT) {
        *cache = kh_val(&uct_ze_ipc_remote_cache.hash, khiter);
    } else {
        ucs_error("unable to use ze_ipc remote_cache hash");
        status = UCS_ERR_NO_RESOURCE;
    }
err_unlock:
    ucs_recursive_spin_unlock(&uct_ze_ipc_remote_cache.lock);
    return status;
}


ucs_status_t uct_ze_ipc_unmap_memhandle(pid_t pid, uintptr_t address,
                                        void *mapped_addr,
                                        ze_context_handle_t ze_context,
                                        int dup_fd, int cache_enabled)
{
    ucs_status_t status = UCS_OK;
    uct_ze_ipc_cache_t *cache;
    ucs_pgt_region_t *pgt_region;
    uct_ze_ipc_cache_region_t *region;

    status = uct_ze_ipc_get_remote_cache(pid, ze_context, &cache);
    if (status != UCS_OK) {
        return status;
    }

    /* use write lock because cache maybe modified */
    pthread_rwlock_wrlock(&cache->lock);
    pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup, &cache->pgtable, address);
    ucs_assert(pgt_region != NULL);
    region = ucs_derived_of(pgt_region, uct_ze_ipc_cache_region_t);

    ucs_assert(region->refcount >= 1);
    region->refcount--;

    /*
     * check refcount to see if an in-flight transfer is using the same mapping
     */
    if (!region->refcount && !cache_enabled) {
        status = ucs_pgtable_remove(&cache->pgtable, &region->super);
        if (status != UCS_OK) {
            ucs_error("failed to remove address:%p from cache (%s)",
                      (void *)region->key.address, ucs_status_string(status));
        }
        ucs_assert(region->mapped_addr == mapped_addr);
        status = uct_ze_ipc_close_memhandle(region);
        ucs_free(region);
    }

    pthread_rwlock_unlock(&cache->lock);
    return status;
}


UCS_PROFILE_FUNC(ucs_status_t, uct_ze_ipc_map_memhandle,
                 (key, ze_context, ze_device, mapped_addr, dup_fd),
                 uct_ze_ipc_key_t *key,
                 ze_context_handle_t ze_context,
                 ze_device_handle_t ze_device,
                 void **mapped_addr, int *dup_fd)
{
    uct_ze_ipc_cache_t *cache;
    ucs_status_t status;
    ucs_pgt_region_t *pgt_region;
    uct_ze_ipc_cache_region_t *region;
    int ret;

    status = uct_ze_ipc_get_remote_cache(key->pid, ze_context, &cache);
    if (status != UCS_OK) {
        return status;
    }

    pthread_rwlock_wrlock(&cache->lock);
    pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup,
                                  &cache->pgtable, key->address);
    if (ucs_likely(pgt_region != NULL)) {
        region = ucs_derived_of(pgt_region, uct_ze_ipc_cache_region_t);

        /* cache hit */
        ucs_trace("%s: ze_ipc cache hit addr:%p size:%lu region:"
                  UCS_PGT_REGION_FMT, cache->name, (void *)key->address,
                  key->length, UCS_PGT_REGION_ARG(&region->super));

        *mapped_addr = region->mapped_addr;
        *dup_fd      = region->dup_fd;
        ucs_assert(region->refcount < UINT64_MAX);
        region->refcount++;
        pthread_rwlock_unlock(&cache->lock);
        return UCS_OK;
    }

    status = uct_ze_ipc_open_memhandle(key, ze_context, ze_device, mapped_addr, dup_fd);
    if (ucs_unlikely(status != UCS_OK)) {
        goto err;
    }

    /* create new cache entry */
    ret = ucs_posix_memalign((void **)&region,
                             ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                             sizeof(uct_ze_ipc_cache_region_t),
                             "uct_ze_ipc_cache_region");
    if (ret != 0) {
        ucs_warn("failed to allocate uct_ze_ipc_cache region");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    region->super.start = ucs_align_down_pow2((uintptr_t)key->address,
                                               UCS_PGT_ADDR_ALIGN);
    region->super.end   = ucs_align_up_pow2((uintptr_t)key->address + key->length,
                                             UCS_PGT_ADDR_ALIGN);
    region->key         = *key;
    region->mapped_addr = *mapped_addr;
    region->refcount    = 1;
    region->ze_context  = ze_context;
    region->dup_fd      = *dup_fd;

    status = UCS_PROFILE_CALL(ucs_pgtable_insert,
                              &cache->pgtable, &region->super);
    if (status == UCS_ERR_ALREADY_EXISTS) {
        /* overlapped region means memory freed at source. remove and try insert */
        uct_ze_ipc_cache_invalidate_regions(cache,
                                            (void *)region->super.start,
                                            (void *)region->super.end);
        status = UCS_PROFILE_CALL(ucs_pgtable_insert,
                                  &cache->pgtable, &region->super);
    }
    if (status != UCS_OK) {
        ucs_error("%s: failed to insert region:"UCS_PGT_REGION_FMT" size:%lu :%s",
                  cache->name, UCS_PGT_REGION_ARG(&region->super), key->length,
                  ucs_status_string(status));
        ucs_free(region);
        goto err;
    }

    ucs_trace("%s: ze_ipc cache new region:"UCS_PGT_REGION_FMT" size:%lu",
              cache->name, UCS_PGT_REGION_ARG(&region->super), key->length);

    status = UCS_OK;

err:
    pthread_rwlock_unlock(&cache->lock);
    return status;
}


ucs_status_t uct_ze_ipc_create_cache(uct_ze_ipc_cache_t **cache,
                                     const char *name)
{
    ucs_status_t status;
    uct_ze_ipc_cache_t *cache_desc;
    int ret;

    cache_desc = ucs_malloc(sizeof(uct_ze_ipc_cache_t), "uct_ze_ipc_cache_t");
    if (cache_desc == NULL) {
        ucs_error("failed to allocate memory for ze_ipc cache");
        return UCS_ERR_NO_MEMORY;
    }

    ret = pthread_rwlock_init(&cache_desc->lock, NULL);
    if (ret) {
        ucs_error("pthread_rwlock_init() failed: %m");
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    status = ucs_pgtable_init(&cache_desc->pgtable,
                              uct_ze_ipc_cache_pgt_dir_alloc,
                              uct_ze_ipc_cache_pgt_dir_release);
    if (status != UCS_OK) {
        goto err_destroy_rwlock;
    }

    cache_desc->name = strdup(name);
    if (cache_desc->name == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_destroy_rwlock;
    }

    *cache = cache_desc;
    return UCS_OK;

err_destroy_rwlock:
    pthread_rwlock_destroy(&cache_desc->lock);
err:
    free(cache_desc);
    return status;
}


void uct_ze_ipc_destroy_cache(uct_ze_ipc_cache_t *cache)
{
    uct_ze_ipc_cache_purge(cache);
    ucs_pgtable_cleanup(&cache->pgtable);
    pthread_rwlock_destroy(&cache->lock);
    free(cache->name);
    ucs_free(cache);
}


UCS_STATIC_INIT {
    ucs_recursive_spinlock_init(&uct_ze_ipc_remote_cache.lock, 0);
    kh_init_inplace(ze_ipc_rem_cache, &uct_ze_ipc_remote_cache.hash);
}


UCS_STATIC_CLEANUP {
    uct_ze_ipc_cache_t *rem_cache;

    kh_foreach_value(&uct_ze_ipc_remote_cache.hash, rem_cache, {
        uct_ze_ipc_destroy_cache(rem_cache);
    })
    kh_destroy_inplace(ze_ipc_rem_cache, &uct_ze_ipc_remote_cache.hash);
    ucs_recursive_spinlock_destroy(&uct_ze_ipc_remote_cache.lock);
}


