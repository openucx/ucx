/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_ipc_vmm_multi.h"
#include "cuda_ipc.inl"

#include <ucs/datastruct/array.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/ptr_arith.h>

#include <string.h>

#if HAVE_CUDA_FABRIC


UCS_ARRAY_DECLARE_TYPE(uct_cuda_ipc_vmm_chunk_array_t, uint32_t,
                       uct_cuda_ipc_vmm_chunk_desc_t);


static ucs_status_t
uct_cuda_ipc_vmm_multi_check_layout(const uct_cuda_ipc_vmm_multi_info_t *info,
                                    ucs_log_level_t log_level)
{
    size_t slots_end;

    /* The inline information and chunk descriptors are part of the protocol */
    UCS_STATIC_ASSERT(sizeof(uct_cuda_ipc_vmm_multi_info_t) == 16);
    UCS_STATIC_ASSERT(sizeof(uct_cuda_ipc_vmm_multi_info_t) <=
                      sizeof(CUmemPoolPtrExportData));
    UCS_STATIC_ASSERT(ucs_offsetof(uct_cuda_ipc_md_handle_t, ptr) ==
                      (ucs_offsetof(uct_cuda_ipc_md_handle_t, handle) +
                       ucs_field_sizeof(uct_cuda_ipc_md_handle_t, handle)));
    UCS_STATIC_ASSERT(ucs_offsetof(uct_cuda_ipc_md_handle_t, pool) ==
                      (ucs_offsetof(uct_cuda_ipc_md_handle_t, ptr) +
                       sizeof(CUmemPoolPtrExportData)));
    UCS_STATIC_ASSERT(sizeof(uct_cuda_ipc_vmm_chunk_desc_t) == 80);

    if (!uct_cuda_ipc_vmm_multi_is_supported(info)) {
        ucs_log(log_level,
                "unsupported multi-chunk VMM metadata version %u handle type "
                "%u",
                info->version, info->handle_type);
        return UCS_ERR_UNSUPPORTED;
    }

    if (info->num_chunks == 0) {
        ucs_log(log_level, "multi-chunk VMM key covers no chunks");
        return UCS_ERR_INVALID_PARAM;
    }

    slots_end = ((size_t)info->first_chunk + info->num_chunks) *
                sizeof(uct_cuda_ipc_vmm_chunk_desc_t);
    if (slots_end > info->meta_size) {
        ucs_log(log_level,
                "multi-chunk VMM key slots %u..%u exceed metadata allocation "
                "size %zu",
                info->first_chunk, info->first_chunk + info->num_chunks,
                info->meta_size);
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

/* The descriptors must tile the key range, in address order and without gaps,
 * since each chunk is mapped at its offset from the key base */
static ucs_status_t
uct_cuda_ipc_vmm_multi_check_chunks(const uct_cuda_ipc_rkey_t *key,
                                    const uct_cuda_ipc_vmm_chunk_desc_t *chunks,
                                    uint16_t num_chunks,
                                    ucs_log_level_t log_level)
{
    CUdeviceptr end = key->d_bptr;
    uint16_t i;

    for (i = 0; i < num_chunks; i++) {
        if ((chunks[i].d_bptr != end) || (chunks[i].b_len == 0)) {
            break;
        }

        end += chunks[i].b_len;
    }

    if ((i < num_chunks) || (end != (key->d_bptr + key->b_len))) {
        ucs_log(log_level,
                "multi-chunk VMM descriptors do not cover key range "
                "0x%llx..0x%llx (chunk %u of %u)",
                key->d_bptr, key->d_bptr + key->b_len, i, num_chunks);
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static void
uct_cuda_ipc_vmm_multi_meta_free_buffer(uct_cuda_ipc_vmm_multi_meta_t *meta)
{
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemUnmap(meta->dev_ptr, meta->alloc_size));
    UCT_CUDADRV_FUNC_LOG_WARN(
            cuMemAddressFree(meta->dev_ptr, meta->alloc_size));
}

void uct_cuda_ipc_vmm_multi_meta_destroy(uct_cuda_ipc_memh_t *memh)
{
    uct_cuda_ipc_vmm_multi_meta_t *meta = memh->vmm_multi;

    if (meta == NULL) {
        return;
    }

    uct_cuda_ipc_vmm_multi_meta_free_buffer(meta);
    ucs_dynamic_bitmap_cleanup(&meta->exported);
    ucs_free(meta->chunks);
    ucs_free(meta);
    memh->vmm_multi = NULL;
}

static ucs_status_t
uct_cuda_ipc_vmm_multi_discover_chunks(CUdeviceptr va_base, size_t va_len,
                                       uct_cuda_ipc_vmm_chunk_desc_t **chunks_p,
                                       uint16_t *num_chunks_p)
{
    uct_cuda_ipc_vmm_chunk_array_t chunks;
    uct_cuda_ipc_vmm_chunk_desc_t *elem;
    uint64_t allowed_handle_types;
    CUdeviceptr pos, chunk_base;
    size_t chunk_size;
    ucs_status_t status;

    ucs_array_init_dynamic(&chunks);

    for (pos = va_base; pos < (va_base + va_len);
         pos = (chunk_base + chunk_size)) {
        if (ucs_array_length(&chunks) == UINT16_MAX) {
            ucs_error("VMM region 0x%llx..0x%llx exceeds %u chunks", va_base,
                      va_base + va_len, UINT16_MAX);
            status = UCS_ERR_EXCEEDS_LIMIT;
            goto err;
        }

        status = UCT_CUDADRV_FUNC_LOG_DEBUG(
                cuMemGetAddressRange(&chunk_base, &chunk_size, pos));
        if (status != UCS_OK) {
            ucs_debug("VMM region 0x%llx..0x%llx is not mapped at 0x%llx",
                      va_base, va_base + va_len, pos);
            status = UCS_ERR_UNSUPPORTED;
            goto err;
        }

        /* Some drivers report a zero base for host-located allocations */
        if ((pos < chunk_base) || (pos >= (chunk_base + chunk_size))) {
            ucs_debug("VMM region 0x%llx..0x%llx: allocation range "
                      "0x%llx..0x%llx does not contain 0x%llx", va_base,
                      va_base + va_len, chunk_base, chunk_base + chunk_size,
                      pos);
            status = UCS_ERR_UNSUPPORTED;
            goto err;
        }

        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuPointerGetAttribute(&allowed_handle_types,
                                      CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
                                      chunk_base));
        if (status != UCS_OK) {
            goto err;
        }

        if (!(allowed_handle_types & CU_MEM_HANDLE_TYPE_FABRIC)) {
            ucs_debug("VMM chunk 0x%llx does not allow fabric handles",
                      chunk_base);
            status = UCS_ERR_UNSUPPORTED;
            goto err;
        }

        elem = ucs_array_append(&chunks, {
            status = UCS_ERR_NO_MEMORY;
            goto err;
        });

        memset(elem, 0, sizeof(*elem));
        elem->d_bptr = chunk_base;
        elem->b_len  = chunk_size;
    }

    *num_chunks_p = (uint16_t)ucs_array_length(&chunks);
    *chunks_p     = ucs_array_extract_buffer(&chunks);
    return UCS_OK;

err:
    ucs_array_cleanup_dynamic(&chunks);
    return status;
}

static ucs_status_t
uct_cuda_ipc_vmm_multi_export_chunk(uct_cuda_ipc_vmm_chunk_desc_t *desc)
{
    CUmemGenericAllocationHandle handle;
    ucs_status_t status;

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemRetainAllocationHandle(&handle, (void*)desc->d_bptr));
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemExportToShareableHandle(
            &desc->fabric_handle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemRelease(handle));
    return status;
}

static ucs_status_t uct_cuda_ipc_vmm_multi_meta_alloc_buffer(
        CUdeviceptr *dev_ptr_p, size_t *alloc_size_p,
        CUmemFabricHandle *fabric_handle_p, size_t data_size,
        const CUmemAllocationProp *prop)
{
    CUmemAccessDesc access = {};
    CUmemGenericAllocationHandle alloc_handle;
    CUdeviceptr dev_ptr;
    ucs_status_t status;
    size_t alloc_size, alloc_granularity;

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemGetAllocationGranularity(&alloc_granularity, prop,
                                          CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    if (status != UCS_OK) {
        return status;
    }

    alloc_size = ucs_align_up(data_size, alloc_granularity);

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemCreate(&alloc_handle, alloc_size, prop, 0));
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemAddressReserve(&dev_ptr, alloc_size, 0, 0, 0));
    if (status != UCS_OK) {
        goto err_release;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemMap(dev_ptr, alloc_size, 0, alloc_handle, 0));
    if (status != UCS_OK) {
        goto err_free_va;
    }

    uct_cuda_ipc_init_access_desc(&access, prop->location.id);
    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemSetAccess(dev_ptr, alloc_size, &access, 1));
    if (status != UCS_OK) {
        goto err_unmap;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemExportToShareableHandle(
            fabric_handle_p, alloc_handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    if (status != UCS_OK) {
        goto err_unmap;
    }

    UCT_CUDADRV_FUNC_LOG_WARN(cuMemRelease(alloc_handle));

    *dev_ptr_p    = dev_ptr;
    *alloc_size_p = alloc_size;
    return UCS_OK;

err_unmap:
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemUnmap(dev_ptr, alloc_size));
err_free_va:
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemAddressFree(dev_ptr, alloc_size));
err_release:
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemRelease(alloc_handle));
    return status;
}

static ucs_status_t
uct_cuda_ipc_vmm_multi_meta_create(uct_cuda_ipc_memh_t *memh)
{
    CUmemAllocationProp prop = {};
    uct_cuda_ipc_vmm_multi_meta_t *meta;
    CUdevice cuda_device;
    int is_ctx_pushed;
    ucs_status_t status;

    status = uct_cuda_ipc_check_and_push_ctx(memh->reg_address, &cuda_device,
                                             &is_ctx_pushed);
    if (status != UCS_OK) {
        return status;
    }

    meta = ucs_calloc(1, sizeof(*meta), "cuda_ipc_vmm_multi_meta");
    if (meta == NULL) {
        ucs_error("failed to allocate VMM metadata record");
        status = UCS_ERR_NO_MEMORY;
        goto out_pop;
    }

    status = uct_cuda_ipc_vmm_multi_discover_chunks(memh->reg_address,
                                                    memh->reg_length,
                                                    &meta->chunks,
                                                    &meta->num_chunks);
    if (status != UCS_OK) {
        goto err_free_meta;
    }

    ucs_dynamic_bitmap_init(&meta->exported);
    ucs_dynamic_bitmap_reserve(&meta->exported, meta->num_chunks);

    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id          = cuda_device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    status = uct_cuda_ipc_vmm_multi_meta_alloc_buffer(
            &meta->dev_ptr, &meta->alloc_size, &meta->fabric_handle,
            meta->num_chunks * sizeof(*meta->chunks), &prop);
    if (status != UCS_OK) {
        goto err_free_chunks;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuPointerGetAttribute(&meta->buffer_id,
                                  CU_POINTER_ATTRIBUTE_BUFFER_ID,
                                  meta->dev_ptr));
    if (status != UCS_OK) {
        goto err_free_buffer;
    }

    ucs_trace("created VMM metadata: %u chunks, allocation size %zu on GPU",
              meta->num_chunks, meta->alloc_size);

    memh->vmm_multi = meta;
    goto out_pop;

err_free_buffer:
    uct_cuda_ipc_vmm_multi_meta_free_buffer(meta);
err_free_chunks:
    ucs_dynamic_bitmap_cleanup(&meta->exported);
    ucs_free(meta->chunks);
err_free_meta:
    ucs_free(meta);
out_pop:
    uct_cuda_ipc_check_and_pop_ctx(is_ctx_pushed);
    return status;
}

/* Index of the chunk containing an address of the registered range */
static unsigned
uct_cuda_ipc_vmm_multi_chunk_index(const uct_cuda_ipc_vmm_multi_meta_t *meta,
                                   CUdeviceptr address)
{
    unsigned low  = 0;
    unsigned high = meta->num_chunks;
    unsigned mid;

    while ((high - low) > 1) {
        mid = low + ((high - low) / 2);
        if (meta->chunks[mid].d_bptr <= address) {
            low = mid;
        } else {
            high = mid;
        }
    }

    return low;
}

/* Whether the slots of the chunks [first, last] are all written */
static int
uct_cuda_ipc_vmm_multi_is_exported(const uct_cuda_ipc_vmm_multi_meta_t *meta,
                                   unsigned first, unsigned last)
{
    return (ucs_dynamic_bitmap_popcount_upto_index(&meta->exported, last + 1) -
            ucs_dynamic_bitmap_popcount_upto_index(&meta->exported, first)) ==
           ((last - first) + 1);
}

/* Export the chunks [first, last] which were not packed before, and write
 * their slots. A slot is never rewritten, since peers may be reading it. */
static ucs_status_t
uct_cuda_ipc_vmm_multi_export_range(uct_cuda_ipc_vmm_multi_meta_t *meta,
                                    unsigned first, unsigned last)
{
    unsigned start, end, i;
    CUdevice cuda_device;
    int is_ctx_pushed;
    ucs_status_t status;

    /* A range packed before needs no driver calls */
    if (uct_cuda_ipc_vmm_multi_is_exported(meta, first, last)) {
        return UCS_OK;
    }

    status = uct_cuda_ipc_check_and_push_ctx(meta->dev_ptr, &cuda_device,
                                             &is_ctx_pushed);
    if (status != UCS_OK) {
        return status;
    }

    for (start = first; start <= last; start = end) {
        if (ucs_dynamic_bitmap_get(&meta->exported, start)) {
            end = start + 1;
            continue;
        }

        /* Find a contiguous range of chunks that were not exported before */
        for (end = start;
             (end <= last) && !ucs_dynamic_bitmap_get(&meta->exported, end);
             ++end) {
            status = uct_cuda_ipc_vmm_multi_export_chunk(&meta->chunks[end]);
            if (status != UCS_OK) {
                goto out_pop;
            }
        }

        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuMemcpyHtoD(meta->dev_ptr + (start * sizeof(*meta->chunks)),
                             &meta->chunks[start],
                             (end - start) * sizeof(*meta->chunks)));
        if (status != UCS_OK) {
            goto out_pop;
        }

        for (i = start; i < end; ++i) {
            ucs_dynamic_bitmap_set(&meta->exported, i);
        }
    }

out_pop:
    uct_cuda_ipc_check_and_pop_ctx(is_ctx_pushed);
    return status;
}

ucs_status_t uct_cuda_ipc_mkey_pack_vmm_multi_chunk(
        uct_cuda_ipc_memh_t *memh, void *address, size_t length,
        const uct_cuda_ipc_vmm_multi_meta_t **meta_p,
        uct_cuda_ipc_vmm_multi_range_t *range_p)
{
    CUdeviceptr start = (CUdeviceptr)address;
    CUdeviceptr end   = start + length;
    const uct_cuda_ipc_vmm_chunk_desc_t *last_chunk;
    uct_cuda_ipc_vmm_multi_info_t *info;
    uct_cuda_ipc_vmm_multi_meta_t *meta;
    unsigned first, last;
    ucs_status_t status;

    ucs_assertv((start >= memh->reg_address) &&
                (end <= (memh->reg_address + memh->reg_length)),
                "buffer 0x%llx..0x%llx registration 0x%llx..0x%llx", start,
                end, memh->reg_address, memh->reg_address + memh->reg_length);

    if (memh->vmm_multi == NULL) {
        status = uct_cuda_ipc_vmm_multi_meta_create(memh);
        if (status != UCS_OK) {
            return status;
        }
    }

    meta  = memh->vmm_multi;
    first = uct_cuda_ipc_vmm_multi_chunk_index(meta, start);
    last  = uct_cuda_ipc_vmm_multi_chunk_index(meta, end - 1);

    status = uct_cuda_ipc_vmm_multi_export_range(meta, first, last);
    if (status != UCS_OK) {
        return status;
    }

    last_chunk = &meta->chunks[last];
    info       = &range_p->info;

    memset(info, 0, sizeof(*info));
    info->version     = UCT_CUDA_IPC_VMM_MULTI_VERSION;
    info->handle_type = UCT_CUDA_IPC_VMM_MULTI_HANDLE_FABRIC;
    info->first_chunk = first;
    info->num_chunks  = (last - first) + 1;
    info->meta_size   = meta->alloc_size;
    range_p->d_bptr   = meta->chunks[first].d_bptr;
    range_p->b_len    = (last_chunk->d_bptr + last_chunk->b_len) -
                        range_p->d_bptr;
    *meta_p           = meta;
    return UCS_OK;
}

ucs_status_t
uct_cuda_ipc_vmm_multi_fetch_chunks(const uct_cuda_ipc_rkey_t *key,
                                    CUdevice cu_dev,
                                    uct_cuda_ipc_vmm_chunk_desc_t **chunks_p,
                                    ucs_log_level_t log_level)
{
    const uct_cuda_ipc_vmm_multi_info_t *info = &key->ph.vmm_multi;
    uct_cuda_ipc_vmm_chunk_desc_t *chunks;
    CUdeviceptr dev_ptr;
    size_t chunks_size;
    ucs_status_t status;
    CUdevice ctx_dev, avail_dev;

    status = uct_cuda_ipc_vmm_multi_check_layout(info, log_level);
    if (status != UCS_OK) {
        return status;
    }

    chunks_size = info->num_chunks * sizeof(*chunks);
    chunks      = ucs_malloc(chunks_size, "vmm_multi_chunks");
    if (chunks == NULL) {
        ucs_error("failed to allocate %zu bytes for VMM chunk descriptors",
                  chunks_size);
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_cuda_ipc_open_memhandle_vmm(
            info->meta_size, cu_dev, &dev_ptr,
            (void*)&key->ph.handle.fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC,
            log_level);
    if (status != UCS_OK) {
        goto err_free;
    }

    status = uct_cuda_ctx_primary_push_avail(0, uct_cuda_get_sys_dev(cu_dev),
                                             &ctx_dev, &avail_dev, log_level);
    if (status == UCS_OK) {
        /* Read only the slots of the key */
        status = UCT_CUDADRV_FUNC(cuMemcpyDtoH(chunks,
                                               dev_ptr + (info->first_chunk *
                                                          sizeof(*chunks)),
                                               chunks_size),
                                  log_level);
        if (ctx_dev != avail_dev) {
            uct_cuda_ctx_primary_pop_and_release(avail_dev);
        }
    }
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemUnmap(dev_ptr, info->meta_size));
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemAddressFree(dev_ptr, info->meta_size));
    if (status != UCS_OK) {
        goto err_free;
    }

    status = uct_cuda_ipc_vmm_multi_check_chunks(key, chunks, info->num_chunks,
                                                 log_level);
    if (status != UCS_OK) {
        goto err_free;
    }

    ucs_trace("fetched %u VMM chunk descriptors from remote", info->num_chunks);
    *chunks_p = chunks;
    return UCS_OK;

err_free:
    ucs_free(chunks);
    return status;
}

#endif
