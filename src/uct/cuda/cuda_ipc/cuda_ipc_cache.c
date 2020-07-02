/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_ipc_cache.h"
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/profile/profile.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/math.h>

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

static void uct_cuda_ipc_cache_purge(uct_cuda_ipc_cache_t *cache)
{
    uct_cuda_ipc_cache_region_t *region, *tmp;
    ucs_list_link_t region_list;

    ucs_list_head_init(&region_list);
    ucs_pgtable_purge(&cache->pgtable, uct_cuda_ipc_cache_region_collect_callback,
                      &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        UCT_CUDADRV_FUNC_LOG_ERR(
                cuIpcCloseMemHandle((CUdeviceptr)region->mapped_addr));
        ucs_free(region);
    }
    ucs_trace("%s: cuda ipc cache purged", cache->name);
}

static ucs_status_t uct_cuda_ipc_open_memhandle(CUipcMemHandle memh,
                                                CUdeviceptr *mapped_addr)
{
    const char *cu_err_str;
    CUresult cuerr;

    cuerr = cuIpcOpenMemHandle(mapped_addr, memh,
                               CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
    if (cuerr != CUDA_SUCCESS) {
        if (cuerr == CUDA_ERROR_ALREADY_MAPPED) {
            return UCS_ERR_ALREADY_EXISTS;
        }

        cuGetErrorString(cuerr, &cu_err_str);
        ucs_error("cuIpcOpenMemHandle() failed: %s", cu_err_str);

        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static void uct_cuda_ipc_cache_invalidate_regions(uct_cuda_ipc_cache_t *cache,
                                                  void *from, void *to)
{
    ucs_list_link_t region_list;
    ucs_status_t status;
    uct_cuda_ipc_cache_region_t *region, *tmp;

    ucs_list_head_init(&region_list);
    ucs_pgtable_search_range(&cache->pgtable, (ucs_pgt_addr_t)from,
                             (ucs_pgt_addr_t)to,
                             uct_cuda_ipc_cache_region_collect_callback,
                             &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        status = ucs_pgtable_remove(&cache->pgtable, &region->super);
        if (status != UCS_OK) {
            ucs_error("failed to remove address:%p from cache (%s)",
                      (void *)region->key.d_bptr, ucs_status_string(status));
        }
        UCT_CUDADRV_FUNC_LOG_ERR(
                cuIpcCloseMemHandle((CUdeviceptr)region->mapped_addr));
        ucs_free(region);
    }
    ucs_trace("%s: closed memhandles in the range [%p..%p]",
              cache->name, from, to);
}

ucs_status_t uct_cuda_ipc_unmap_memhandle(void *rem_cache, uintptr_t d_bptr,
                                          void *mapped_addr, int cache_enabled)
{
    uct_cuda_ipc_cache_t *cache = (uct_cuda_ipc_cache_t *) rem_cache;
    ucs_status_t status         = UCS_OK;
    ucs_pgt_region_t *pgt_region;
    uct_cuda_ipc_cache_region_t *region;

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
        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuIpcCloseMemHandle((CUdeviceptr)region->mapped_addr));
        ucs_free(region);
    }

    pthread_rwlock_unlock(&cache->lock);
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_ipc_map_memhandle,
                 (arg, key, mapped_addr),
                 void *arg, uct_cuda_ipc_key_t *key, void **mapped_addr)
{
    uct_cuda_ipc_cache_t *cache = (uct_cuda_ipc_cache_t *)arg;
    ucs_status_t status;
    ucs_pgt_region_t *pgt_region;
    uct_cuda_ipc_cache_region_t *region;
    int ret;

    pthread_rwlock_wrlock(&cache->lock);
    pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup,
                                  &cache->pgtable, key->d_bptr);
    if (ucs_likely(pgt_region != NULL)) {
        region = ucs_derived_of(pgt_region, uct_cuda_ipc_cache_region_t);
        if (memcmp((const void *)&key->ph, (const void *)&region->key.ph,
                   sizeof(key->ph)) == 0) {
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
            UCT_CUDADRV_FUNC_LOG_ERR(
                    cuIpcCloseMemHandle((CUdeviceptr)region->mapped_addr));
            ucs_free(region);
        }
    }

    status = uct_cuda_ipc_open_memhandle(key->ph, (CUdeviceptr *)mapped_addr);
    if (ucs_unlikely(status != UCS_OK)) {
        if (ucs_likely(status == UCS_ERR_ALREADY_EXISTS)) {
            /* unmap all overlapping regions and retry*/
            uct_cuda_ipc_cache_invalidate_regions(cache, (void *)key->d_bptr,
                                                  UCS_PTR_BYTE_OFFSET(key->d_bptr,
                                                                      key->b_len));
            status = uct_cuda_ipc_open_memhandle(key->ph, (CUdeviceptr *)mapped_addr);
            if (ucs_unlikely(status != UCS_OK)) {
                if (ucs_likely(status == UCS_ERR_ALREADY_EXISTS)) {
                    /* unmap all cache entries and retry */
                    uct_cuda_ipc_cache_purge(cache);
                    status = uct_cuda_ipc_open_memhandle(key->ph, (CUdeviceptr *)mapped_addr);
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
            ucs_fatal("%s: failed to open ipc mem handle. addr:%p len:%lu",
                      cache->name, (void *)key->d_bptr, key->b_len);
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

    pthread_rwlock_unlock(&cache->lock);
    return UCS_OK;
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
