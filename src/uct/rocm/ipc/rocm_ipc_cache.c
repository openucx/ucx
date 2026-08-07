/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018. ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rocm_ipc_cache.h"

#include <ucs/datastruct/khash.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/profile/profile.h>
#include <ucs/sys/ptr_arith.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/type/rwlock.h>


typedef struct uct_rocm_ipc_cache_hash_key {
    pid_t        pid;    /* PID of the process that owns the memory */
    ucs_sys_ns_t pid_ns; /* PID namespace of the owner process */
    int          dev_num; /* Device number the memory was allocated on */
} uct_rocm_ipc_cache_hash_key_t;

static UCS_F_ALWAYS_INLINE int
uct_rocm_ipc_cache_hash_equal(uct_rocm_ipc_cache_hash_key_t key1,
                              uct_rocm_ipc_cache_hash_key_t key2)
{
    return (key1.pid == key2.pid) && (key1.pid_ns == key2.pid_ns) &&
           (key1.dev_num == key2.dev_num);
}

static UCS_F_ALWAYS_INLINE khint32_t
uct_rocm_ipc_cache_hash_func(uct_rocm_ipc_cache_hash_key_t key)
{
    uint64_t value = key.pid ^ ((uint64_t)key.pid_ns << 32) ^
                     ((uint32_t)key.dev_num << 24);
    return kh_int64_hash_func(value);
}

KHASH_INIT(rocm_ipc_rem_cache, uct_rocm_ipc_cache_hash_key_t,
           uct_rocm_ipc_cache_t*, 1, uct_rocm_ipc_cache_hash_func,
           uct_rocm_ipc_cache_hash_equal);

/* Per-peer IPC handle caches. Caches are kept per (pid, pid_ns, dev_num). */
typedef struct uct_rocm_ipc_remote_cache {
    khash_t(rocm_ipc_rem_cache) hash;
    ucs_rw_spinlock_t           lock;
} uct_rocm_ipc_remote_cache_t;

static uct_rocm_ipc_remote_cache_t uct_rocm_ipc_remote_cache;

static ucs_pgt_dir_t *uct_rocm_ipc_cache_pgt_dir_alloc(const ucs_pgtable_t *pgtable)
{
    void *ptr;
    int ret;

    ret =  ucs_posix_memalign(&ptr,
                              ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                              sizeof(ucs_pgt_dir_t), "rocm_ipc_cache_pgdir");
    return (ret == 0) ? ptr : NULL;
}

static void uct_rocm_ipc_cache_pgt_dir_release(const ucs_pgtable_t *pgtable,
                                               ucs_pgt_dir_t *dir)
{
    ucs_free(dir);
}

static void
uct_rocm_ipc_cache_region_collect_callback(const ucs_pgtable_t *pgtable,
                                           ucs_pgt_region_t *pgt_region,
                                           void *arg)
{
    ucs_list_link_t *list = arg;
    uct_rocm_ipc_cache_region_t *region;

    region = ucs_derived_of(pgt_region, uct_rocm_ipc_cache_region_t);
    ucs_list_add_tail(list, &region->list);
}

static void uct_rocm_ipc_cache_purge(uct_rocm_ipc_cache_t *cache)
{
    uct_rocm_ipc_cache_region_t *region, *tmp;
    ucs_list_link_t region_list;

    ucs_list_head_init(&region_list);
    ucs_pgtable_purge(&cache->pgtable, uct_rocm_ipc_cache_region_collect_callback,
                      &region_list);

    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        if (hsa_amd_ipc_memory_detach(region->mapped_addr) != HSA_STATUS_SUCCESS) {
            ucs_fatal("failed to unmap addr:%p", region->mapped_addr);
        }

        ucs_free(region);
    }

    ucs_trace("%s: rocm ipc cache purged", cache->name);
}

static void uct_rocm_ipc_cache_invalidate_regions(uct_rocm_ipc_cache_t *cache,
                                                  void *from, void *to)
{
    ucs_list_link_t region_list;
    ucs_status_t status;
    uct_rocm_ipc_cache_region_t *region, *tmp;

    ucs_list_head_init(&region_list);
    ucs_pgtable_search_range(&cache->pgtable, (ucs_pgt_addr_t)from,
                             (ucs_pgt_addr_t)to - 1,
                             uct_rocm_ipc_cache_region_collect_callback,
                             &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        status = ucs_pgtable_remove(&cache->pgtable, &region->super);
        if (status != UCS_OK) {
            ucs_error("failed to remove address:%p from cache (%s)",
                      (void *)region->key.address, ucs_status_string(status));
        }

        if (hsa_amd_ipc_memory_detach(region->mapped_addr) != HSA_STATUS_SUCCESS) {
            ucs_fatal("failed to unmap addr:%p", region->mapped_addr);
        }
        ucs_free(region);
    }
    ucs_trace("%s: closed memhandles in the range [%p..%p]",
              cache->name, from, to);
}

static ucs_status_t
uct_rocm_ipc_cache_map_region(uct_rocm_ipc_cache_t *cache,
                             uct_rocm_ipc_key_t *key, void **mapped_addr)
{
    ucs_status_t status;
    ucs_pgt_region_t *pgt_region;
    uct_rocm_ipc_cache_region_t *region;
    hsa_status_t hsa_status;
    int ret;

    pthread_rwlock_wrlock(&cache->lock);
    pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup,
                                  &cache->pgtable, key->address);
    if (ucs_likely(pgt_region != NULL)) {
        region = ucs_derived_of(pgt_region, uct_rocm_ipc_cache_region_t);
        if (memcmp(&key->ipc, &region->key.ipc, sizeof(key->ipc)) == 0) {
            /*cache hit */
            ucs_trace("%s: rocm_ipc cache hit addr:%p size:%lu region:"
                      UCS_PGT_REGION_FMT, cache->name, (void *)key->address,
                      key->length, UCS_PGT_REGION_ARG(&region->super));

            *mapped_addr = region->mapped_addr;
            pthread_rwlock_unlock(&cache->lock);
            return UCS_OK;
        } else {
            ucs_trace("%s: rocm_ipc cache remove stale region:"
                      UCS_PGT_REGION_FMT " new_addr:%p new_size:%lu",
                      cache->name, UCS_PGT_REGION_ARG(&region->super),
                      (void *)key->address, key->length);

            status = ucs_pgtable_remove(&cache->pgtable, &region->super);
            if (status != UCS_OK) {
                ucs_error("%s: failed to remove address:%p from cache",
                          cache->name, (void *)key->address);
                goto err;
            }

            if (hsa_amd_ipc_memory_detach(region->mapped_addr) != HSA_STATUS_SUCCESS) {
                ucs_fatal("failed to unmap addr:%p", region->mapped_addr);
            }

            ucs_free(region);
        }
    }

    hsa_status = hsa_amd_ipc_memory_attach(&key->ipc, key->length, 0, NULL, mapped_addr);
    if (ucs_unlikely(hsa_status != HSA_STATUS_SUCCESS)) {
        ucs_fatal("%s: failed to open ipc mem handle. addr:%p len:%lu",
                  cache->name, (void *)key->address, key->length);
    }

    /*create new cache entry */
    ret = ucs_posix_memalign((void **)&region,
                             ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                             sizeof(uct_rocm_ipc_cache_region_t),
                             "uct_rocm_ipc_cache_region");
    if (ret != 0) {
        ucs_warn("failed to allocate uct_rocm_ipc_cache region");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    region->super.start = ucs_align_down_pow2(key->address, UCS_PGT_ADDR_ALIGN);
    region->super.end   = ucs_align_up_pow2(key->address + key->length, UCS_PGT_ADDR_ALIGN);
    region->key         = *key;
    region->mapped_addr = *mapped_addr;

    status = UCS_PROFILE_CALL(ucs_pgtable_insert,
                              &cache->pgtable, &region->super);
    if (status == UCS_ERR_ALREADY_EXISTS) {
        /* overlapped region means memory freed at source. remove and try insert */
        uct_rocm_ipc_cache_invalidate_regions(cache,
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

    ucs_trace("%s: rocm_ipc cache new region:"UCS_PGT_REGION_FMT" size:%lu",
              cache->name, UCS_PGT_REGION_ARG(&region->super), key->length);

    pthread_rwlock_unlock(&cache->lock);
    return UCS_OK;
err:
    pthread_rwlock_unlock(&cache->lock);
    return status;
}

ucs_status_t uct_rocm_ipc_create_cache(uct_rocm_ipc_cache_t **cache,
                                       const char *name)
{
    ucs_status_t status;
    uct_rocm_ipc_cache_t *cache_desc;
    int ret;

    cache_desc = ucs_malloc(sizeof(uct_rocm_ipc_cache_t), "uct_rocm_ipc_cache_t");
    if (cache_desc == NULL) {
        ucs_error("failed to allocate memory for rocm_ipc cache");
        return UCS_ERR_NO_MEMORY;
    }

    ret = pthread_rwlock_init(&cache_desc->lock, NULL);
    if (ret) {
        ucs_error("pthread_rwlock_init() failed: %m");
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    status = ucs_pgtable_init(&cache_desc->pgtable,
                              uct_rocm_ipc_cache_pgt_dir_alloc,
                              uct_rocm_ipc_cache_pgt_dir_release);
    if (status != UCS_OK) {
        goto err_destroy_rwlock;
    }

    cache_desc->name = ucs_strdup(name, "rocm_ipc_cache_name");
    if (cache_desc->name == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_destroy_rwlock;
    }

    *cache = cache_desc;
    return UCS_OK;

err_destroy_rwlock:
    pthread_rwlock_destroy(&cache_desc->lock);
err:
    ucs_free(cache_desc);
    return status;
}

/* Look up the per-peer cache for the given hash key, creating it on first use.
 * Caller must hold the remote cache write lock. */
static ucs_status_t
uct_rocm_ipc_remote_cache_get(uct_rocm_ipc_cache_hash_key_t key,
                              uct_rocm_ipc_cache_t **cache_p)
{
    khash_t(rocm_ipc_rem_cache) *hash = &uct_rocm_ipc_remote_cache.hash;
    uct_rocm_ipc_cache_t *cache;
    char target_name[64];
    ucs_status_t status;
    khint_t it;
    int ret;

    it = kh_get(rocm_ipc_rem_cache, hash, key);
    if (ucs_likely(it != kh_end(hash))) {
        *cache_p = kh_val(hash, it);
        return UCS_OK;
    }

    it = kh_put(rocm_ipc_rem_cache, hash, key, &ret);
    if (ret == UCS_KH_PUT_FAILED) {
        ucs_error("failed to allocate rocm_ipc remote_cache hash entry");
        return UCS_ERR_NO_MEMORY;
    }

    ucs_snprintf_safe(target_name, sizeof(target_name), "dest:%d:%u:%d",
                      (int)key.pid, key.pid_ns, key.dev_num);
    status = uct_rocm_ipc_create_cache(&cache, target_name);
    if (status != UCS_OK) {
        kh_del(rocm_ipc_rem_cache, hash, it);
        ucs_error("failed to create rocm ipc cache: %s",
                  ucs_status_string(status));
        return status;
    }

    kh_val(hash, it) = cache;
    *cache_p         = cache;
    return UCS_OK;
}

ucs_status_t uct_rocm_ipc_cache_map_memhandle(uct_rocm_ipc_key_t *key,
                                              void **mapped_addr)
{
    uct_rocm_ipc_cache_hash_key_t hash_key = {key->pid, key->pid_ns,
                                              key->dev_num};
    uct_rocm_ipc_cache_t *cache;
    ucs_status_t status;

    ucs_rw_spinlock_write_lock(&uct_rocm_ipc_remote_cache.lock);
    status = uct_rocm_ipc_remote_cache_get(hash_key, &cache);
    if (status == UCS_OK) {
        status = uct_rocm_ipc_cache_map_region(cache, key, mapped_addr);
    }
    ucs_rw_spinlock_write_unlock(&uct_rocm_ipc_remote_cache.lock);

    return status;
}

void uct_rocm_ipc_destroy_cache(uct_rocm_ipc_cache_t *cache)
{
    uct_rocm_ipc_cache_purge(cache);
    ucs_pgtable_cleanup(&cache->pgtable);
    pthread_rwlock_destroy(&cache->lock);
    ucs_free(cache->name);
    ucs_free(cache);
}

UCS_STATIC_INIT {
    ucs_rw_spinlock_init(&uct_rocm_ipc_remote_cache.lock);
    kh_init_inplace(rocm_ipc_rem_cache, &uct_rocm_ipc_remote_cache.hash);
}

UCS_STATIC_CLEANUP {
    uct_rocm_ipc_cache_t *cache;

    kh_foreach_value(&uct_rocm_ipc_remote_cache.hash, cache, {
        uct_rocm_ipc_destroy_cache(cache);
    })
    kh_destroy_inplace(rocm_ipc_rem_cache, &uct_rocm_ipc_remote_cache.hash);
    ucs_rw_spinlock_cleanup(&uct_rocm_ipc_remote_cache.lock);
}
