/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_ipc_cache.h"
#include "cuda_ipc_iface.h"
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/profile/profile.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/string.h>
#include <ucs/sys/ptr_arith.h>
#include <ucs/datastruct/khash.h>


typedef struct uct_cuda_ipc_cache_hash_key {
    pid_t    pid;
    CUdevice cu_device;
} uct_cuda_ipc_cache_hash_key_t;

static UCS_F_ALWAYS_INLINE int
uct_cuda_ipc_cache_hash_equal(uct_cuda_ipc_cache_hash_key_t key1,
                              uct_cuda_ipc_cache_hash_key_t key2)
{
    return (key1.pid == key2.pid) && (key1.cu_device == key2.cu_device);
}

static UCS_F_ALWAYS_INLINE khint32_t
uct_cuda_ipc_cache_hash_func(uct_cuda_ipc_cache_hash_key_t key)
{
    return kh_int_hash_func((key.pid << 8) | key.cu_device);
}

KHASH_INIT(cuda_ipc_rem_cache, uct_cuda_ipc_cache_hash_key_t,
           uct_cuda_ipc_cache_t*, 1, uct_cuda_ipc_cache_hash_func,
           uct_cuda_ipc_cache_hash_equal);

typedef struct uct_cuda_ipc_remote_cache {
    khash_t(cuda_ipc_rem_cache) hash;
    ucs_recursive_spinlock_t    lock;
} uct_cuda_ipc_remote_cache_t;

uct_cuda_ipc_remote_cache_t uct_cuda_ipc_remote_cache;

#if HAVE_CUDA_FABRIC
static UCS_F_ALWAYS_INLINE int
uct_cuda_ipc_rem_mpool_cache_hash_equal(CUmemFabricHandle key1,
                                        CUmemFabricHandle key2)
{
    return !memcmp(&key1, &key2, sizeof(key1));
}

/* djb2 hash function */
static UCS_F_ALWAYS_INLINE khint_t
uct_cuda_ipc_rem_mpool_cache_hash_func(CUmemFabricHandle key)
{
    khint_t hash = 5381;
    int i;

    for (i = 0; i < sizeof(key.data); i++) {
        hash = ((hash << 5) + hash) + key.data[i];
    }

    return hash;
}

KHASH_INIT(cuda_ipc_rem_mpool_cache, CUmemFabricHandle, CUmemoryPool, 1,
           uct_cuda_ipc_rem_mpool_cache_hash_func,
           uct_cuda_ipc_rem_mpool_cache_hash_equal)

typedef struct {
    khash_t(cuda_ipc_rem_mpool_cache) hash;
    pthread_rwlock_t                  lock;
} uct_cuda_ipc_rem_mpool_cache_t;

static uct_cuda_ipc_rem_mpool_cache_t uct_cuda_ipc_rem_mpool_cache;
#endif

static ucs_pgt_dir_t *uct_cuda_ipc_cache_pgt_dir_alloc(const ucs_pgtable_t *pgtable)
{
    void *ptr;
    int ret;

    ret = ucs_posix_memalign(&ptr,
                             ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                             sizeof(ucs_pgt_dir_t), "cuda_ipc_cache_pgdir");
    return (ret == 0) ? ptr : NULL;
}

static void uct_cuda_ipc_cache_pgt_dir_release(const ucs_pgtable_t *pgtable,
                                               ucs_pgt_dir_t *dir)
{
    ucs_free(dir);
}

static void
uct_cuda_ipc_cache_region_collect_callback(const ucs_pgtable_t *pgtable,
                                           ucs_pgt_region_t *pgt_region,
                                           void *arg)
{
    ucs_list_link_t *list = arg;
    uct_cuda_ipc_cache_region_t *region;

    region = ucs_derived_of(pgt_region, uct_cuda_ipc_cache_region_t);
    ucs_list_add_tail(list, &region->list);
}

static ucs_status_t uct_cuda_ipc_close_memhandle(uct_cuda_ipc_cache_region_t *region)
{
    ucs_status_t status;

#if HAVE_CUDA_FABRIC
    if (region->key.ph.handle_type == UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM) {
        status = UCT_CUDADRV_FUNC_LOG_WARN(cuMemUnmap(
                    (CUdeviceptr)region->mapped_addr, region->key.b_len));
        if (status == UCS_OK) {
            return UCT_CUDADRV_FUNC_LOG_WARN(cuMemAddressFree(
                        (CUdeviceptr)region->mapped_addr, region->key.b_len));
        }
    } else if (region->key.ph.handle_type == UCT_CUDA_IPC_KEY_HANDLE_TYPE_MEMPOOL) {
        return UCT_CUDADRV_FUNC_LOG_WARN(cuMemPoolDestroy(region->key.ph.pool));
    } else
#endif
    {
        return UCT_CUDADRV_FUNC_LOG_WARN(cuIpcCloseMemHandle(
                    (CUdeviceptr)region->mapped_addr));
    }

    return status;
}

static void uct_cuda_ipc_cache_purge(uct_cuda_ipc_cache_t *cache)
{
    int active = uct_cuda_base_is_context_active();
    uct_cuda_ipc_cache_region_t *region, *tmp;
    ucs_list_link_t region_list;

    ucs_list_head_init(&region_list);
    ucs_pgtable_purge(&cache->pgtable, uct_cuda_ipc_cache_region_collect_callback,
                      &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        if (active) {
            uct_cuda_ipc_close_memhandle(region);
        }
        ucs_free(region);
    }
    ucs_trace("%s: cuda ipc cache purged", cache->name);
}

static ucs_status_t
uct_cuda_ipc_open_memhandle_legacy(CUipcMemHandle memh,
                                   CUdeviceptr *mapped_addr)
{
    CUresult cuerr;

    cuerr = cuIpcOpenMemHandle(mapped_addr, memh,
                               CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
    if (cuerr != CUDA_SUCCESS) {
        ucs_debug("cuIpcOpenMemHandle() failed: %s",
                  uct_cuda_base_cu_get_error_string(cuerr));
        return (cuerr == CUDA_ERROR_ALREADY_MAPPED) ?
            UCS_ERR_ALREADY_EXISTS : UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

#if HAVE_CUDA_FABRIC
static ucs_status_t uct_cuda_ipc_init_access_desc(CUmemAccessDesc *access_desc)
{
    access_desc->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc->flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    return UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetDevice(&access_desc->location.id));
}

static ucs_status_t
uct_cuda_ipc_open_memhandle_vmm(uct_cuda_ipc_rkey_t *key,
                                CUdeviceptr *mapped_addr)
{
    CUmemAccessDesc access_desc = {};
    ucs_status_t status;
    CUdeviceptr dptr;
    CUmemGenericAllocationHandle handle;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemImportFromShareableHandle(&handle,
                (void*)&key->ph.handle.fabric_handle,
                CU_MEM_HANDLE_TYPE_FABRIC));
    if (status != UCS_OK) {
        goto out;
    }

    status =
        UCT_CUDADRV_FUNC_LOG_ERR(cuMemAddressReserve(&dptr, key->b_len, 0, 0, 0));
    if (status != UCS_OK) {
        goto release_handle;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemMap(dptr, key->b_len, 0, handle, 0));
    if (status != UCS_OK) {
        goto release_va_range;
    }

    status = uct_cuda_ipc_init_access_desc(&access_desc);
    if (status != UCS_OK) {
        goto unmap_range;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemSetAccess(dptr, key->b_len, &access_desc, 1));
    if (status != UCS_OK) {
        goto unmap_range;
    }

    *mapped_addr = dptr;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemRelease(handle));
    if (status == UCS_OK) {
        goto out;
    }

unmap_range:
    cuMemUnmap(dptr, key->b_len);
release_va_range:
    cuMemAddressFree(dptr, key->b_len);
release_handle:
    cuMemRelease(handle);
out:
    return status;
}

static ucs_status_t cuda_ipc_rem_mpool_cache_create(uct_cuda_ipc_rkey_t *key,
                                                    CUmemoryPool *mpool,
                                                    CUdeviceptr *mapped_addr)
{
    CUmemAccessDesc access_desc = {};
    CUdeviceptr dptr;
    ucs_status_t status;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemPoolImportFromShareableHandle(
                mpool, (void *)&key->ph.handle.fabric_handle,
                CU_MEM_HANDLE_TYPE_FABRIC, 0));
    if (status != UCS_OK) {
        goto err;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemPoolImportPointer(&dptr, *mpool,
                (CUmemPoolPtrExportData*)&key->ph.ptr));
    if (status != UCS_OK) {
        goto err_free_mpool;
    }

    status = uct_cuda_ipc_init_access_desc(&access_desc);
    if (status != UCS_OK) {
        goto err_free_ptr;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuMemPoolSetAccess(*mpool, &access_desc, 1));
    if (status != UCS_OK) {
        goto err_free_ptr;
    }

    *mapped_addr = dptr;
    return UCS_OK;

err_free_ptr:
    cuMemFree(dptr);
err_free_mpool:
    cuMemPoolDestroy(*mpool);
err:
    return status;
}

static ucs_status_t
uct_cuda_ipc_open_memhandle_mempool(uct_cuda_ipc_rkey_t *key,
                                    CUdeviceptr *mapped_addr)
{
    khash_t(cuda_ipc_rem_mpool_cache) *hash = &uct_cuda_ipc_rem_mpool_cache.hash;
    const CUmemFabricHandle *hkey           = &key->ph.handle.fabric_handle;
    ucs_status_t status                     = UCS_OK;
    CUmemoryPool mpool;
    khiter_t khiter;
    int khret;

    pthread_rwlock_rdlock(&uct_cuda_ipc_rem_mpool_cache.lock);
    khiter = kh_get(cuda_ipc_rem_mpool_cache, hash, *hkey);
    if (ucs_likely(khiter != kh_end(hash))) {
        key->ph.pool = kh_val(hash, khiter);
        ucs_assert(key->ph.pool != NULL);
        goto out_import_pointer;
    }

    /* acquire writer lock as we did not find cached mpool in hashmap */
    pthread_rwlock_unlock(&uct_cuda_ipc_rem_mpool_cache.lock);
    pthread_rwlock_wrlock(&uct_cuda_ipc_rem_mpool_cache.lock);

    khiter = kh_put(cuda_ipc_rem_mpool_cache, hash, *hkey, &khret);
    if ((khret == UCS_KH_PUT_BUCKET_EMPTY) ||
        (khret == UCS_KH_PUT_BUCKET_CLEAR)) {
        status = cuda_ipc_rem_mpool_cache_create(key, &mpool, mapped_addr);
        if (status != UCS_OK) {
            /* Remove stale key from the hash if failed to construct new value.
             * Otherwise next lookup will return NULL value. */
            kh_del(cuda_ipc_rem_mpool_cache, hash, khiter);
            goto err;
        }

        kh_val(hash, khiter) = mpool;
        key->ph.pool         = mpool;
    } else if (khret == UCS_KH_PUT_KEY_PRESENT) {
        key->ph.pool = kh_val(hash, khiter);
    } else {
        ucs_error("unable to use cuda_ipc remote_cache hash: %d", khret);
        status = UCS_ERR_NO_RESOURCE;
        goto err;
    }

out_import_pointer:
    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemPoolImportPointer(mapped_addr,
                key->ph.pool, (CUmemPoolPtrExportData*)&key->ph.ptr));

err:
    pthread_rwlock_unlock(&uct_cuda_ipc_rem_mpool_cache.lock);
    return status;
}
#endif

static ucs_status_t uct_cuda_ipc_open_memhandle(uct_cuda_ipc_rkey_t *key,
                                                CUdeviceptr *mapped_addr)
{

#if HAVE_CUDA_FABRIC
    ucs_trace("key handle type %u", key->ph.handle_type);

    if (key->ph.handle_type == UCT_CUDA_IPC_KEY_HANDLE_TYPE_LEGACY) {
        return uct_cuda_ipc_open_memhandle_legacy(key->ph.handle.legacy,
                                                  mapped_addr);
    } else if (key->ph.handle_type == UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM) {
        return uct_cuda_ipc_open_memhandle_vmm(key, mapped_addr);
    } else if (key->ph.handle_type == UCT_CUDA_IPC_KEY_HANDLE_TYPE_MEMPOOL) {
        return uct_cuda_ipc_open_memhandle_mempool(key, mapped_addr);
    } else {
        return UCS_ERR_INVALID_PARAM;
    }
#else
    return uct_cuda_ipc_open_memhandle_legacy(key->ph, mapped_addr);
#endif
}

static void uct_cuda_ipc_cache_invalidate_regions(uct_cuda_ipc_cache_t *cache,
                                                  void *from, void *to)
{
    ucs_list_link_t region_list;
    ucs_status_t status;
    uct_cuda_ipc_cache_region_t *region, *tmp;
    int handle_type;

    ucs_list_head_init(&region_list);
    ucs_pgtable_search_range(&cache->pgtable, (ucs_pgt_addr_t)from,
                             (ucs_pgt_addr_t)to - 1,
                             uct_cuda_ipc_cache_region_collect_callback,
                             &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        status = ucs_pgtable_remove(&cache->pgtable, &region->super);
        if (status != UCS_OK) {
            ucs_error("failed to remove address:%p from cache (%s)",
                      (void *)region->key.d_bptr, ucs_status_string(status));
        }

        status = uct_cuda_ipc_close_memhandle(region);
        if (status != UCS_OK) {
#if HAVE_CUDA_FABRIC
            handle_type = region->key.ph.handle_type;
#else
            handle_type = 1; /* legacy memory type is the only valid type */
#endif
            ucs_error("failed to close memhandle for base addr:%p type:%d (%s)",
                      (void *)region->key.d_bptr, handle_type,
                      ucs_status_string(status));
        }

        ucs_free(region);
    }
    ucs_trace("%s: closed memhandles in the range [%p..%p]",
              cache->name, from, to);
}

static ucs_status_t
uct_cuda_ipc_get_remote_cache(pid_t pid, uct_cuda_ipc_cache_t **cache)
{
    ucs_status_t status = UCS_OK;
    char target_name[64];
    uct_cuda_ipc_cache_hash_key_t key;
    khiter_t khiter;
    int khret;

    UCT_CUDA_IPC_GET_DEVICE(key.cu_device);

    ucs_recursive_spin_lock(&uct_cuda_ipc_remote_cache.lock);

    key.pid = pid;

    khiter = kh_put(cuda_ipc_rem_cache, &uct_cuda_ipc_remote_cache.hash, key,
                    &khret);
    if ((khret == UCS_KH_PUT_BUCKET_EMPTY) ||
        (khret == UCS_KH_PUT_BUCKET_CLEAR)) {
        ucs_snprintf_safe(target_name, sizeof(target_name), "dest:%d:%d",
                          key.pid, key.cu_device);
        status = uct_cuda_ipc_create_cache(cache, target_name);
        if (status != UCS_OK) {
            kh_del(cuda_ipc_rem_cache, &uct_cuda_ipc_remote_cache.hash, khiter);
            ucs_error("could not create create cuda ipc cache: %s",
                      ucs_status_string(status));
            goto err_unlock;
        }

        kh_val(&uct_cuda_ipc_remote_cache.hash, khiter) = *cache;
    } else if (khret == UCS_KH_PUT_KEY_PRESENT) {
        *cache = kh_val(&uct_cuda_ipc_remote_cache.hash, khiter);
    } else {
        ucs_error("unable to use cuda_ipc remote_cache hash");
        status = UCS_ERR_NO_RESOURCE;
    }
err_unlock:
    ucs_recursive_spin_unlock(&uct_cuda_ipc_remote_cache.lock);
    return status;
}

ucs_status_t uct_cuda_ipc_unmap_memhandle(pid_t pid, uintptr_t d_bptr,
                                          void *mapped_addr, int cache_enabled)
{
    ucs_status_t status = UCS_OK;
    uct_cuda_ipc_cache_t *cache;
    ucs_pgt_region_t *pgt_region;
    uct_cuda_ipc_cache_region_t *region;

    status = uct_cuda_ipc_get_remote_cache(pid, &cache);
    if (status != UCS_OK) {
        return status;
    }

    /* use write lock because cache maybe modified */
    pthread_rwlock_wrlock(&cache->lock);
    pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup, &cache->pgtable, d_bptr);
    ucs_assert(pgt_region != NULL);
    region = ucs_derived_of(pgt_region, uct_cuda_ipc_cache_region_t);

    ucs_assert(region->refcount >= 1);
    region->refcount--;

    /*
     * check refcount to see if an in-flight transfer is using the same mapping
     */
    if (!region->refcount && !cache_enabled) {
        status = ucs_pgtable_remove(&cache->pgtable, &region->super);
        if (status != UCS_OK) {
            ucs_error("failed to remove address:%p from cache (%s)",
                      (void *)region->key.d_bptr, ucs_status_string(status));
        }
        ucs_assert(region->mapped_addr == mapped_addr);
        status = uct_cuda_ipc_close_memhandle(region);
        ucs_free(region);
    }

    pthread_rwlock_unlock(&cache->lock);
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_ipc_map_memhandle, (key, mapped_addr),
                 uct_cuda_ipc_rkey_t *key, void **mapped_addr)
{
    uct_cuda_ipc_cache_t *cache;
    ucs_status_t status;
    ucs_pgt_region_t *pgt_region;
    uct_cuda_ipc_cache_region_t *region;
    int ret;
    size_t cmp_size;

#if HAVE_CUDA_FABRIC
    cmp_size = sizeof(key->ph.handle);
#else
    cmp_size = sizeof(key->ph);
#endif

    status = uct_cuda_ipc_get_remote_cache(key->pid, &cache);
    if (status != UCS_OK) {
        return status;
    }

    pthread_rwlock_wrlock(&cache->lock);
    pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup,
                                  &cache->pgtable, key->d_bptr);
    if (ucs_likely(pgt_region != NULL)) {
        region = ucs_derived_of(pgt_region, uct_cuda_ipc_cache_region_t);
        if (memcmp((const void *)&key->ph, (const void *)&region->key.ph,
                   cmp_size) == 0) {
            /*cache hit */
            ucs_trace("%s: cuda_ipc cache hit addr:%p size:%lu region:"
                      UCS_PGT_REGION_FMT, cache->name, (void *)key->d_bptr,
                      key->b_len, UCS_PGT_REGION_ARG(&region->super));

            *mapped_addr = region->mapped_addr;
            ucs_assert(region->refcount < UINT64_MAX);
            region->refcount++;
            pthread_rwlock_unlock(&cache->lock);
            return UCS_OK;
        } else {
            ucs_trace("%s: cuda_ipc cache remove stale region:"
                      UCS_PGT_REGION_FMT " new_addr:%p new_size:%lu",
                      cache->name, UCS_PGT_REGION_ARG(&region->super),
                      (void *)key->d_bptr, key->b_len);

            status = ucs_pgtable_remove(&cache->pgtable, &region->super);
            if (status != UCS_OK) {
                ucs_error("%s: failed to remove address:%p from cache",
                          cache->name, (void *)key->d_bptr);
                goto err;
            }

            /* close memhandle */
            uct_cuda_ipc_close_memhandle(region);
            ucs_free(region);
        }
    }

    status = uct_cuda_ipc_open_memhandle(key, (CUdeviceptr*)mapped_addr);
    if (ucs_unlikely(status != UCS_OK)) {
        if (ucs_likely(status == UCS_ERR_ALREADY_EXISTS)) {
            /* unmap all overlapping regions and retry*/
            uct_cuda_ipc_cache_invalidate_regions(cache, (void *)key->d_bptr,
                                                  UCS_PTR_BYTE_OFFSET(key->d_bptr,
                                                                      key->b_len));
            status = uct_cuda_ipc_open_memhandle(key,
                                                 (CUdeviceptr*)mapped_addr);
            if (ucs_unlikely(status != UCS_OK)) {
                if (ucs_likely(status == UCS_ERR_ALREADY_EXISTS)) {
                    /* unmap all cache entries and retry */
                    uct_cuda_ipc_cache_purge(cache);
                    status =
                        uct_cuda_ipc_open_memhandle(key,
                                                    (CUdeviceptr*)mapped_addr);
                    if (status != UCS_OK) {
                        ucs_fatal("%s: failed to open ipc mem handle. addr:%p "
                                  "len:%lu (%s)", cache->name,
                                  (void *)key->d_bptr, key->b_len,
                                  ucs_status_string(status));
                    }
                } else {
                    ucs_fatal("%s: failed to open ipc mem handle. addr:%p len:%lu",
                              cache->name, (void *)key->d_bptr, key->b_len);
                }
            }
        } else {
            ucs_debug("%s: failed to open ipc mem handle. addr:%p len:%lu",
                      cache->name, (void *)key->d_bptr, key->b_len);
            goto err;
        }
    }

    /*create new cache entry */
    ret = ucs_posix_memalign((void **)&region,
                             ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                             sizeof(uct_cuda_ipc_cache_region_t),
                             "uct_cuda_ipc_cache_region");
    if (ret != 0) {
        ucs_warn("failed to allocate uct_cuda_ipc_cache region");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    region->super.start = ucs_align_down_pow2((uintptr_t)key->d_bptr,
                                               UCS_PGT_ADDR_ALIGN);
    region->super.end   = ucs_align_up_pow2  ((uintptr_t)key->d_bptr + key->b_len,
                                               UCS_PGT_ADDR_ALIGN);
    region->key         = *key;
    region->mapped_addr = *mapped_addr;
    region->refcount    = 1;

    status = UCS_PROFILE_CALL(ucs_pgtable_insert,
                              &cache->pgtable, &region->super);
    if (status == UCS_ERR_ALREADY_EXISTS) {
        /* overlapped region means memory freed at source. remove and try insert */
        uct_cuda_ipc_cache_invalidate_regions(cache,
                                              (void *)region->super.start,
                                              (void *)region->super.end);
        status = UCS_PROFILE_CALL(ucs_pgtable_insert,
                                  &cache->pgtable, &region->super);
    }
    if (status != UCS_OK) {

        ucs_error("%s: failed to insert region:"UCS_PGT_REGION_FMT" size:%lu :%s",
                  cache->name, UCS_PGT_REGION_ARG(&region->super), key->b_len,
                  ucs_status_string(status));
        ucs_free(region);
        goto err;
    }

    ucs_trace("%s: cuda_ipc cache new region:"UCS_PGT_REGION_FMT" size:%lu",
              cache->name, UCS_PGT_REGION_ARG(&region->super), key->b_len);

    status = UCS_OK;

err:
    pthread_rwlock_unlock(&cache->lock);
    return status;
}

ucs_status_t uct_cuda_ipc_create_cache(uct_cuda_ipc_cache_t **cache,
                                       const char *name)
{
    ucs_status_t status;
    uct_cuda_ipc_cache_t *cache_desc;
    int ret;

    cache_desc = ucs_malloc(sizeof(uct_cuda_ipc_cache_t), "uct_cuda_ipc_cache_t");
    if (cache_desc == NULL) {
        ucs_error("failed to allocate memory for cuda_ipc cache");
        return UCS_ERR_NO_MEMORY;
    }

    ret = pthread_rwlock_init(&cache_desc->lock, NULL);
    if (ret) {
        ucs_error("pthread_rwlock_init() failed: %m");
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    status = ucs_pgtable_init(&cache_desc->pgtable,
                              uct_cuda_ipc_cache_pgt_dir_alloc,
                              uct_cuda_ipc_cache_pgt_dir_release);
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

void uct_cuda_ipc_destroy_cache(uct_cuda_ipc_cache_t *cache)
{
    uct_cuda_ipc_cache_purge(cache);
    ucs_pgtable_cleanup(&cache->pgtable);
    pthread_rwlock_destroy(&cache->lock);
    free(cache->name);
    ucs_free(cache);
}

UCS_STATIC_INIT {
    ucs_recursive_spinlock_init(&uct_cuda_ipc_remote_cache.lock, 0);
    kh_init_inplace(cuda_ipc_rem_cache, &uct_cuda_ipc_remote_cache.hash);

#if HAVE_CUDA_FABRIC
    pthread_rwlock_init(&uct_cuda_ipc_rem_mpool_cache.lock, NULL);
    /* Assumption: If import process succeeds, then the two nodes are in the
     * same domain. Within a domain, fabric handles are expected to be unique.
     * For this reason, there is no need to maintain a hashmap per peer OS as
     * key collisions are not expected to occur. */
    kh_init_inplace(cuda_ipc_rem_mpool_cache, &uct_cuda_ipc_rem_mpool_cache.hash);
#endif
}

UCS_STATIC_CLEANUP {
    uct_cuda_ipc_cache_t *rem_cache;

#if HAVE_CUDA_FABRIC
    CUmemoryPool mpool;

    kh_foreach_value(&uct_cuda_ipc_rem_mpool_cache.hash, mpool, {
        cuMemPoolDestroy(mpool);
    });
    kh_destroy_inplace(cuda_ipc_rem_mpool_cache,
                       &uct_cuda_ipc_rem_mpool_cache.hash);
    pthread_rwlock_destroy(&uct_cuda_ipc_rem_mpool_cache.lock);
#endif

    kh_foreach_value(&uct_cuda_ipc_remote_cache.hash, rem_cache, {
        uct_cuda_ipc_destroy_cache(rem_cache);
    })
    kh_destroy_inplace(cuda_ipc_rem_cache, &uct_cuda_ipc_remote_cache.hash);
    ucs_recursive_spinlock_destroy(&uct_cuda_ipc_remote_cache.lock);
}
