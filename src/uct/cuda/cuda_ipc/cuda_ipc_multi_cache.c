/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_ipc_cache.h"
#include "cuda_ipc_vmm_multi.h"

#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/ptr_arith.h>

#include <string.h>

#if HAVE_CUDA_FABRIC

static void uct_cuda_ipc_multi_cache_free_view(
        uct_cuda_ipc_cache_region_t *region, int close_handle)
{
    ucs_assert(region->refcount == 0);
    ucs_assert(!region->indexed);

    if (close_handle) {
        uct_cuda_ipc_close_memhandle(region);
    }

    ucs_free(region);
}

static void uct_cuda_ipc_multi_cache_retire_internal(
        uct_cuda_ipc_cache_t *cache, uct_cuda_ipc_cache_region_t *region,
        int close_handle)
{
    if (region->indexed) {
        ucs_interval_map_remove(&cache->views, &region->node);
        region->indexed = 0;

        ucs_list_del(&region->lru_list);
        ucs_assert(cache->num_regions > 0);
        cache->num_regions--;
        cache->total_size -= region->key.b_len;
    }

    if (region->refcount == 0) {
        uct_cuda_ipc_multi_cache_free_view(region, close_handle);
    }
}

/*
 * Confirms that the chunks the request names are still the chunks this view
 * mapped.
 */
static int uct_cuda_ipc_multi_cache_verify(
        const uct_cuda_ipc_cache_region_t *region,
        const uct_cuda_ipc_unpacked_rkey_t *unpacked)
{
    const uct_cuda_ipc_rkey_t *key = &unpacked->super.super;
    uint16_t lo = 0, hi = region->num_chunks, mid, i;

    /* Locate the request's first chunk within the view. Chunk sizes may
     * differ, so the index cannot be derived from the byte offset. */
    while (lo < hi) {
        mid = lo + ((hi - lo) / 2);
        if (region->chunks[mid].start < key->d_bptr) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if ((lo + unpacked->num_chunks) > region->num_chunks) {
        return 0;
    }

    for (i = 0; i < unpacked->num_chunks; i++) {
        if ((region->chunks[lo + i].start != unpacked->chunks[i].d_bptr) ||
            (region->chunks[lo + i].buffer_id !=
             unpacked->chunks[i].buffer_id)) {
            return 0;
        }
    }

    return 1;
}

static void uct_cuda_ipc_multi_cache_collect(ucs_interval_map_node_t *node,
                                             void *arg)
{
    uct_cuda_ipc_cache_region_t *region =
            ucs_container_of(node, uct_cuda_ipc_cache_region_t, node);

    ucs_list_add_tail((ucs_list_link_t*)arg, &region->list);
}

static ucs_status_t uct_cuda_ipc_multi_cache_create(
        uct_cuda_ipc_cache_t *cache,
        const uct_cuda_ipc_unpacked_rkey_t *unpacked, CUdevice cu_dev,
        void **mapped_addr_p, uct_cuda_ipc_cache_region_t **region_p,
        ucs_log_level_t log_level)
{
    const uct_cuda_ipc_rkey_t *key = &unpacked->super.super;
    uct_cuda_ipc_cache_region_t *region;
    CUdeviceptr mapped_addr;
    ucs_status_t status;
    uint16_t i;

    region = ucs_calloc(1, sizeof(*region) +
                               (unpacked->num_chunks * sizeof(*region->chunks)),
                        "uct_cuda_ipc_multi_view");
    if (region == NULL) {
        ucs_error("failed to allocate cuda_ipc multi view");
        return UCS_ERR_NO_MEMORY;
    }

    /* The chunk array lives in the same block as the view */
    region->chunks     = (uct_cuda_ipc_chunk_t*)(region + 1);
    region->num_chunks = unpacked->num_chunks;
    for (i = 0; i < unpacked->num_chunks; i++) {
        region->chunks[i].start     = unpacked->chunks[i].d_bptr;
        region->chunks[i].buffer_id = unpacked->chunks[i].buffer_id;
    }

    region->key = *key;
    /* The metadata allocation has rkey lifetime, unlike the cached view */
    memset(&region->key.ph.handle, 0, sizeof(region->key.ph.handle));
    region->cu_dev = cu_dev;

    status = uct_cuda_ipc_open_memhandle_vmm_multi(unpacked, cu_dev,
                                                   &mapped_addr, log_level);
    if (status != UCS_OK) {
        ucs_free(region);
        return status;
    }

    region->mapped_addr = (void*)mapped_addr;
    region->refcount    = 1;

    ucs_interval_map_insert(&cache->views, &region->node, key->d_bptr,
                            key->d_bptr + key->b_len);
    region->indexed = 1;

    ucs_list_add_tail(&cache->lru_list, &region->lru_list);
    cache->num_regions++;
    cache->total_size += key->b_len;
    uct_cuda_ipc_cache_evict_lru(cache);

    ucs_trace("%s: new multi view addr:%p size:%lu chunks:%u", cache->name,
              (void*)key->d_bptr, key->b_len, region->num_chunks);

    *mapped_addr_p = region->mapped_addr;
    *region_p      = region;
    return UCS_OK;
}

ucs_status_t uct_cuda_ipc_multi_cache_get(
        uct_cuda_ipc_cache_t *cache,
        const uct_cuda_ipc_unpacked_rkey_t *unpacked, CUdevice cu_dev,
        void **mapped_addr_p, uct_cuda_ipc_cache_region_t **region_p,
        ucs_log_level_t log_level)
{
    const uct_cuda_ipc_rkey_t *key = &unpacked->super.super;
    uint64_t start                 = key->d_bptr;
    uint64_t end                   = key->d_bptr + key->b_len;
    uct_cuda_ipc_cache_region_t *region;
    ucs_interval_map_node_t *node;

    for (;;) {
        node = ucs_interval_map_find_containing(&cache->views, start, end);
        if (node == NULL) {
            break;
        }

        region = ucs_container_of(node, uct_cuda_ipc_cache_region_t, node);

        if (uct_cuda_ipc_multi_cache_verify(region, unpacked)) {
            ucs_trace("%s: multi cache hit addr:%p size:%lu in view [%p..%p)",
                      cache->name, (void*)start, key->b_len,
                      (void*)region->key.d_bptr,
                      (void*)(region->key.d_bptr + region->key.b_len));

            ucs_list_del(&region->lru_list);
            ucs_list_add_tail(&cache->lru_list, &region->lru_list);

            ucs_assert(region->refcount < UINT64_MAX);
            region->refcount++;

            *mapped_addr_p = UCS_PTR_BYTE_OFFSET(region->mapped_addr,
                                                 start - region->key.d_bptr);
            *region_p      = region;
            return UCS_OK;
        }

        ucs_debug("%s: retiring stale multi view [0x%lx..0x%lx)", cache->name,
                  (unsigned long)region->key.d_bptr,
                  (unsigned long)(region->key.d_bptr + region->key.b_len));
        uct_cuda_ipc_multi_cache_retire(cache, region);
    }

    return uct_cuda_ipc_multi_cache_create(cache, unpacked, cu_dev,
                                           mapped_addr_p, region_p, log_level);
}

void uct_cuda_ipc_multi_cache_retire(uct_cuda_ipc_cache_t *cache,
                                     uct_cuda_ipc_cache_region_t *region)
{
    uct_cuda_ipc_multi_cache_retire_internal(cache, region, 1);
}

void uct_cuda_ipc_multi_cache_put(uct_cuda_ipc_cache_t *cache,
                                  uct_cuda_ipc_cache_region_t *region,
                                  int cache_enabled)
{
    ucs_assert(region->refcount >= 1);
    region->refcount--;

    if (region->refcount > 0) {
        return;
    }

    if (!region->indexed) {
        uct_cuda_ipc_multi_cache_free_view(region, 1);
    } else if (!cache_enabled) {
        uct_cuda_ipc_multi_cache_retire(cache, region);
    }
}

void uct_cuda_ipc_multi_cache_purge(uct_cuda_ipc_cache_t *cache,
                                    int close_handles)
{
    uct_cuda_ipc_cache_region_t *region, *tmp;
    ucs_list_link_t all;

    ucs_list_head_init(&all);
    ucs_interval_map_foreach_overlapping(&cache->views, 0, UINT64_MAX,
                                         uct_cuda_ipc_multi_cache_collect,
                                         &all);

    ucs_list_for_each_safe(region, tmp, &all, list) {
        ucs_list_del(&region->list);
        if (region->refcount > 0) {
            ucs_warn("%s: multi view [0x%lx..0x%lx) still has %lu reference(s)",
                     cache->name, (unsigned long)region->key.d_bptr,
                     (unsigned long)(region->key.d_bptr + region->key.b_len),
                     (unsigned long)region->refcount);
        }
        uct_cuda_ipc_multi_cache_retire_internal(cache, region,
                                                 close_handles);
    }
}

#endif
