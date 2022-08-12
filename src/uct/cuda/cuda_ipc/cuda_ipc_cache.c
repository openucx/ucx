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
#include <ucs/sys/math.h>
#include <ucs/memory/rcache.h>
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

static void uct_cuda_ipc_rcache_region_invalidate_cb(void *arg)
{
}

ucs_status_t uct_cuda_ipc_check_rcache(ucs_rcache_t *rcache,
                                       uct_cuda_ipc_key_t *key,
                                       uct_cuda_ipc_rcache_region_t **region_p)
{
    uintptr_t start, end;
    ucs_rcache_region_t *rcache_region;
    ucs_status_t status;

    start = ucs_align_down_pow2((uintptr_t)key->d_bptr, ucs_get_page_size());
    end   = ucs_align_up_pow2  ((uintptr_t)key->d_bptr + key->b_len,
                                ucs_get_page_size());

    status = ucs_rcache_get(rcache, (void*)start, end - start,
                            PROT_READ|PROT_WRITE, (void*)key, &rcache_region);
    if (status != UCS_OK) {
        return UCS_ERR_INVALID_PARAM;
    }

    *region_p = ucs_derived_of(rcache_region, uct_cuda_ipc_rcache_region_t);

    if (memcmp((const void *)&key->ph, (const void *)&((*region_p)->ipc_handle),
                sizeof(key->ph))) {
        /* VA recycling */
        ucs_trace("VA recycling ocurred, removing region %p...%p",
                  (void*)start, (void*)end);

        /* first decrease previous refcount on stale region */
        ucs_rcache_region_put(rcache, rcache_region);

        ucs_rcache_region_invalidate(rcache, rcache_region,
                                     uct_cuda_ipc_rcache_region_invalidate_cb,
                                     NULL);

        status = ucs_rcache_get(rcache, (void*)start, end - start,
                                PROT_READ|PROT_WRITE, (void*)key,
                                &rcache_region);
        if (status != UCS_OK) {
            return UCS_ERR_INVALID_PARAM;
        }

        *region_p = ucs_derived_of(rcache_region, uct_cuda_ipc_rcache_region_t);
    }

    return UCS_OK;
}

static ucs_status_t
uct_cuda_ipc_get_remote_cache(uct_cuda_ipc_md_t *md,
                              pid_t pid, uct_cuda_ipc_cache_t **cache)
{
    ucs_status_t status = UCS_OK;
    char target_name[64];
    uct_cuda_ipc_cache_hash_key_t key;
    khiter_t khiter;
    int khret;

    ucs_recursive_spin_lock(&uct_cuda_ipc_remote_cache.lock);

    key.pid = pid;
    UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetDevice(&key.cu_device));

    khiter = kh_put(cuda_ipc_rem_cache, &uct_cuda_ipc_remote_cache.hash, key,
                    &khret);
    if ((khret == UCS_KH_PUT_BUCKET_EMPTY) ||
        (khret == UCS_KH_PUT_BUCKET_CLEAR)) {
        ucs_snprintf_safe(target_name, sizeof(target_name), "dest:%d:%d",
                          key.pid, key.cu_device);
        status = uct_cuda_ipc_create_cache(md, cache, target_name);
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

void uct_cuda_ipc_close_memhandle(void *mapped_addr,
                                  uct_cuda_ipc_rcache_region_t *cuda_ipc_region)
{
    CUresult cuerr;

    cuerr = cuIpcCloseMemHandle((CUdeviceptr)mapped_addr);
    if (cuerr == CUDA_SUCCESS) {
        if (cuda_ipc_region != NULL) {
            cuda_ipc_region->rcache           = NULL;
            cuda_ipc_region->rem_base_address = NULL;
            cuda_ipc_region->rem_alloc_length = 0;
        }
    }
}

ucs_status_t
uct_cuda_ipc_unmap_memhandle(uct_cuda_ipc_md_t *md,
                             const uct_cuda_ipc_key_t *key,
                             void *mapped_addr,
                             uct_cuda_ipc_rcache_region_t *cuda_ipc_region)
{
    ucs_status_t status;
    uct_cuda_ipc_cache_t *cache;

    if (md->rcache_enable != UCS_NO) {
        status = uct_cuda_ipc_get_remote_cache(md, key->pid, &cache);
        if (status != UCS_OK) {
            return status;
        }

        ucs_rcache_region_put(cache->rcache, &cuda_ipc_region->super);
    } else {
        uct_cuda_ipc_close_memhandle(mapped_addr, NULL);
    }

    return UCS_OK;
}

ucs_status_t uct_cuda_ipc_open_memhandle(void **mapped_addr,
                                         const uct_cuda_ipc_key_t *key,
                                         uct_cuda_ipc_rcache_region_t *cuda_ipc_region,
                                         ucs_rcache_t *rcache)
{
    CUresult cuerr;
    const char *cu_err_str;

    cuerr = cuIpcOpenMemHandle((CUdeviceptr *)mapped_addr,
                               key->ph, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
    if ((cuerr == CUDA_SUCCESS) || (cuerr == CUDA_ERROR_ALREADY_MAPPED)) {
        if (cuda_ipc_region != NULL) {
            cuda_ipc_region->ipc_handle       = key->ph;
            cuda_ipc_region->rcache           = rcache;
            cuda_ipc_region->rem_base_address = (void*)key->d_bptr;
            cuda_ipc_region->rem_alloc_length = key->b_len;
        }
        return UCS_OK;
    } else {
        cuGetErrorString(cuerr, &cu_err_str);
        ucs_debug("cuIpcOpenMemHandle() failed: %s", cu_err_str);
        return UCS_ERR_INVALID_PARAM;
    }
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_ipc_map_memhandle,
                 (md, key, mapped_addr, region_p),
                 uct_cuda_ipc_md_t *md,
                 const uct_cuda_ipc_key_t *key, void **mapped_addr,
                 uct_cuda_ipc_rcache_region_t **region_p)
{
    uct_cuda_ipc_cache_t *cache;
    ucs_status_t status;

    if (md->rcache_enable != UCS_NO) {
        status = uct_cuda_ipc_get_remote_cache(md, key->pid, &cache);
        if (status != UCS_OK) {
            return status;
        }

        status = uct_cuda_ipc_check_rcache(cache->rcache, (uct_cuda_ipc_key_t *)key, region_p);
        if (status != UCS_OK) {
            return status;
        }

        *mapped_addr = (*region_p)->mapping_start;
    } else {
        status = uct_cuda_ipc_open_memhandle(mapped_addr, key, NULL, NULL);
    }

    return status;
}

static ucs_status_t
uct_cuda_ipc_rcache_mem_reg(void *context, ucs_rcache_t *rcache, void *arg,
                            ucs_rcache_region_t *region, uint16_t flags)
{
    uct_cuda_ipc_key_t *key                       = (uct_cuda_ipc_key_t*)arg;
    uct_cuda_ipc_rcache_region_t *cuda_ipc_region =
                    ucs_derived_of(region, uct_cuda_ipc_rcache_region_t);

    return uct_cuda_ipc_open_memhandle(&cuda_ipc_region->mapping_start, key,
                                       cuda_ipc_region, rcache);

}

static void uct_cuda_ipc_rcache_mem_dereg(void *context, ucs_rcache_t *rcache,
                                          ucs_rcache_region_t *region)
{
    uct_cuda_ipc_rcache_region_t *cuda_ipc_region =
                    ucs_derived_of(region, uct_cuda_ipc_rcache_region_t);

    uct_cuda_ipc_close_memhandle(cuda_ipc_region->mapping_start, cuda_ipc_region);

}

static void uct_cuda_ipc_rcache_dump_region(void *context, ucs_rcache_t *rcache,
                                            ucs_rcache_region_t *region, char *buf,
                                            size_t max)
{
    uct_cuda_ipc_rcache_region_t *cuda_ipc_region =
                    ucs_derived_of(region, uct_cuda_ipc_rcache_region_t);

    snprintf(buf, max, "mapping_start %p", cuda_ipc_region->mapping_start);
}

static ucs_rcache_ops_t uct_cuda_ipc_rcache_ops = {
    .mem_reg     = uct_cuda_ipc_rcache_mem_reg,
    .mem_dereg   = uct_cuda_ipc_rcache_mem_dereg,
    .dump_region = uct_cuda_ipc_rcache_dump_region
};


ucs_status_t uct_cuda_ipc_create_cache(uct_cuda_ipc_md_t *md,
                                       uct_cuda_ipc_cache_t **cache,
                                       const char *name)
{
    ucs_status_t status;
    uct_cuda_ipc_cache_t *cache_desc;
    ucs_rcache_params_t rcache_params;

    cache_desc = ucs_malloc(sizeof(uct_cuda_ipc_cache_t), "uct_cuda_ipc_cache_t");
    if (cache_desc == NULL) {
        ucs_error("failed to allocate memory for cuda_ipc cache");
        return UCS_ERR_NO_MEMORY;
    }

    rcache_params.region_struct_size = sizeof(uct_cuda_ipc_rcache_region_t);
    rcache_params.alignment          = ucs_get_page_size();
    rcache_params.max_alignment      = ucs_get_page_size();
    rcache_params.ucm_events         = 0;
    rcache_params.ucm_event_priority = 0;
    rcache_params.ops                = &uct_cuda_ipc_rcache_ops;
    rcache_params.context            = NULL;
    rcache_params.flags              = UCS_RCACHE_FLAG_NO_PFN_CHECK;
    rcache_params.max_regions        = md->rcache_max_regions;
    rcache_params.max_size           = md->rcache_max_size;

    status = ucs_rcache_create(&rcache_params, "cuda_ipc_remote_rcache",
                               ucs_stats_get_root(), &cache_desc->rcache);
    if (status != UCS_OK) {
        ucs_error("failed to create cuda_ipc remote cache: %s",
                  ucs_status_string(status));
        goto err;
    }

    cache_desc->name = strdup(name);
    if (cache_desc->name == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    *cache = cache_desc;
    return UCS_OK;

err:
    free(cache_desc);
    return status;
}

void uct_cuda_ipc_destroy_cache(uct_cuda_ipc_cache_t *cache)
{
    ucs_rcache_destroy(cache->rcache);
    free(cache->name);
    ucs_free(cache);
}

UCS_STATIC_INIT {
    ucs_recursive_spinlock_init(&uct_cuda_ipc_remote_cache.lock, 0);
    kh_init_inplace(cuda_ipc_rem_cache, &uct_cuda_ipc_remote_cache.hash);
}

UCS_STATIC_CLEANUP {
    uct_cuda_ipc_cache_t *rem_cache;

    kh_foreach_value(&uct_cuda_ipc_remote_cache.hash, rem_cache, {
        uct_cuda_ipc_destroy_cache(rem_cache);
    })
    kh_destroy_inplace(cuda_ipc_rem_cache, &uct_cuda_ipc_remote_cache.hash);
    ucs_recursive_spinlock_destroy(&uct_cuda_ipc_remote_cache.lock);
}
