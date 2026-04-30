/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "cuda_ipc.inl"
#include "cuda_ipc_md.h"
#include "cuda_ipc_vmm_multi.h"

#include <ucs/datastruct/array.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/ptr_arith.h>

#if HAVE_CUDA_FABRIC

void uct_cuda_ipc_vmm_multi_meta_cleanup(uct_cuda_ipc_lkey_t *key)
{
    uct_cuda_ipc_vmm_multi_meta_t *meta = &key->vmm_multi;

    if (meta->chunks_dev_ptr != 0) {
        UCT_CUDADRV_FUNC_LOG_WARN(
                cuMemUnmap(meta->chunks_dev_ptr, meta->chunks_alloc_size));
        UCT_CUDADRV_FUNC_LOG_WARN(
                cuMemAddressFree(meta->chunks_dev_ptr, meta->chunks_alloc_size));
        meta->chunks_dev_ptr = 0;
    }

    if (meta->header_dev_ptr != 0) {
        UCT_CUDADRV_FUNC_LOG_WARN(
                cuMemUnmap(meta->header_dev_ptr, meta->header_alloc_size));
        UCT_CUDADRV_FUNC_LOG_WARN(
                cuMemAddressFree(meta->header_dev_ptr, meta->header_alloc_size));
        meta->header_dev_ptr = 0;
    }
}

UCS_ARRAY_DECLARE_TYPE(uct_cuda_ipc_vmm_chunk_array_t, uint32_t,
                       uct_cuda_ipc_vmm_chunk_desc_t);

static ucs_status_t
uct_cuda_ipc_vmm_multi_discover_chunks(CUdeviceptr va_base, size_t va_len,
                                       uct_cuda_ipc_vmm_chunk_desc_t **chunks_p,
                                       uint16_t *num_chunks_p)
{
    uct_cuda_ipc_vmm_chunk_array_t chunks;
    uct_cuda_ipc_vmm_chunk_desc_t *elem;
    CUmemGenericAllocationHandle handle;
    CUdeviceptr pos, chunk_base;
    unsigned long long buffer_id;
    size_t chunk_size;
    ucs_status_t status;

    ucs_array_init_dynamic(&chunks);

    for (pos = va_base; pos < va_base + va_len; pos = chunk_base + chunk_size) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuMemGetAddressRange(&chunk_base, &chunk_size, pos));
        if (status != UCS_OK) {
            goto err;
        }

        status = UCT_CUDADRV_FUNC(cuMemRetainAllocationHandle(&handle,
                                                              (void*)pos),
                                  UCS_LOG_LEVEL_ERROR);
        if (status != UCS_OK) {
            goto err;
        }

        elem = ucs_array_append(&chunks, {
            UCT_CUDADRV_FUNC_LOG_WARN(cuMemRelease(handle));
            status = UCS_ERR_NO_MEMORY;
            goto err;
        });

        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuMemExportToShareableHandle(&elem->vmm_handle.handle.fabric,
                                             handle, CU_MEM_HANDLE_TYPE_FABRIC,
                                             0));
        UCT_CUDADRV_FUNC_LOG_WARN(cuMemRelease(handle));
        if (status != UCS_OK) {
            goto err;
        }

        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuPointerGetAttribute(&buffer_id,
                                      CU_POINTER_ATTRIBUTE_BUFFER_ID, pos));
        if (status != UCS_OK) {
            goto err;
        }

        elem->vmm_handle.handle_type = UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM;
        elem->d_bptr                 = chunk_base;
        elem->b_len                  = chunk_size;
        elem->buffer_id              = buffer_id;
    }

    if (ucs_array_length(&chunks) > UINT16_MAX) {
        ucs_error("VMM region has %zu chunks, exceeding maximum of %u",
                  (size_t)ucs_array_length(&chunks), UINT16_MAX);
        status = UCS_ERR_EXCEEDS_LIMIT;
        goto err;
    }

    *num_chunks_p = (uint16_t)ucs_array_length(&chunks);
    *chunks_p     = ucs_array_extract_buffer(&chunks);
    return UCS_OK;

err:
    ucs_array_cleanup_dynamic(&chunks);
    return status;
}

static ucs_status_t uct_cuda_ipc_vmm_multi_meta_alloc_buffer(
        CUdeviceptr *dev_ptr_p, size_t *alloc_size_p,
        CUmemFabricHandle *fabric_handle_p, size_t data_size,
        const CUmemAllocationProp *prop, size_t alloc_granularity, int dev_num)
{
    CUmemGenericAllocationHandle alloc_handle;
    CUmemAccessDesc access;
    CUdeviceptr dev_ptr;
    ucs_status_t status;
    size_t alloc_size;

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

    uct_cuda_ipc_init_access_desc(&access, dev_num);
    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemSetAccess(dev_ptr, alloc_size, &access, 1));
    if (status != UCS_OK) {
        goto err_unmap;
    }

    if (fabric_handle_p != NULL) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuMemExportToShareableHandle(fabric_handle_p, alloc_handle,
                                             CU_MEM_HANDLE_TYPE_FABRIC, 0));
        if (status != UCS_OK) {
            goto err_unmap;
        }
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
uct_cuda_ipc_vmm_multi_create_meta_buffer(uct_cuda_ipc_lkey_t *key, int dev_num)
{
    uct_cuda_ipc_vmm_multi_meta_t *meta         = &key->vmm_multi;
    uct_cuda_ipc_vmm_chunk_desc_t *host_chunks  = NULL;
    uint16_t num_chunks                         = 0;
    CUmemAllocationProp prop                    = {};
    uct_cuda_ipc_vmm_meta_header_t header       = {};
    uct_cuda_ipc_vmm_handle_t chunks_vmm_handle;
    CUdeviceptr chunks_dev_ptr, header_dev_ptr;
    size_t chunks_alloc_size, header_alloc_size;
    ucs_status_t status;
    size_t chunks_data_size, alloc_granularity;

    status = uct_cuda_ipc_vmm_multi_discover_chunks(meta->d_bptr, meta->b_len,
                                                    &host_chunks, &num_chunks);
    if (status != UCS_OK) {
        return status;
    }

    chunks_data_size = num_chunks * sizeof(uct_cuda_ipc_vmm_chunk_desc_t);

    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id          = dev_num;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemGetAllocationGranularity(&alloc_granularity, &prop,
                                          CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    if (status != UCS_OK) {
        goto err_free_host;
    }

    status = uct_cuda_ipc_vmm_multi_meta_alloc_buffer(
            &chunks_dev_ptr, &chunks_alloc_size,
            &chunks_vmm_handle.handle.fabric, chunks_data_size, &prop,
            alloc_granularity, dev_num);
    if (status != UCS_OK) {
        goto err_free_host;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemcpyHtoD(chunks_dev_ptr, host_chunks, chunks_data_size));
    if (status != UCS_OK) {
        goto err_cleanup_chunks;
    }

    status = uct_cuda_ipc_vmm_multi_meta_alloc_buffer(
            &header_dev_ptr, &header_alloc_size, &meta->header_fabric_handle,
            sizeof(header), &prop, alloc_granularity, dev_num);
    if (status != UCS_OK) {
        goto err_cleanup_chunks;
    }

    chunks_vmm_handle.handle_type = UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM;
    header.chunks_handle          = chunks_vmm_handle;
    header.num_chunks             = num_chunks;

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemcpyHtoD(header_dev_ptr, &header, sizeof(header)));
    if (status != UCS_OK) {
        goto err_cleanup_header;
    }

    meta->header_dev_ptr    = header_dev_ptr;
    meta->header_alloc_size = header_alloc_size;
    meta->chunks_dev_ptr    = chunks_dev_ptr;
    meta->chunks_alloc_size = chunks_alloc_size;
    meta->num_chunks        = num_chunks;

    ucs_trace("created VMM metadata: %u chunks, chunks_alloc=%zu "
              "header_alloc=%zu on GPU",
              num_chunks, chunks_alloc_size, header_alloc_size);
    ucs_free(host_chunks);
    return UCS_OK;

err_cleanup_header:
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemUnmap(header_dev_ptr, header_alloc_size));
    UCT_CUDADRV_FUNC_LOG_WARN(
            cuMemAddressFree(header_dev_ptr, header_alloc_size));
err_cleanup_chunks:
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemUnmap(chunks_dev_ptr, chunks_alloc_size));
    UCT_CUDADRV_FUNC_LOG_WARN(
            cuMemAddressFree(chunks_dev_ptr, chunks_alloc_size));
err_free_host:
    ucs_free(host_chunks);
    return status;
}

ucs_status_t uct_cuda_ipc_mkey_pack_vmm_multi_chunk(uct_cuda_ipc_memh_t *memh,
                                                    uct_cuda_ipc_lkey_t *key,
                                                    void *address,
                                                    size_t length)
{
    uct_cuda_ipc_vmm_multi_meta_t *meta = &key->vmm_multi;
    unsigned long long buf_id_start, buf_id_end;
    CUdeviceptr first_base, last_base, range_start, range_end;
    size_t first_size, last_size;
    CUdevice cuda_device;
    int is_ctx_pushed;
    ucs_status_t status;

    if (meta->header_dev_ptr != 0) {
        if ((CUdeviceptr)address >= meta->d_bptr &&
            ((CUdeviceptr)address + length) <= (meta->d_bptr + meta->b_len)) {
            return UCS_OK;
        }

        uct_cuda_ipc_vmm_multi_meta_cleanup(key);
    }

    status = uct_cuda_ipc_check_and_push_ctx((CUdeviceptr)address, &cuda_device,
                                             &is_ctx_pushed);
    if (status != UCS_OK) {
        return status;
    }

    range_start = (CUdeviceptr)address;
    range_end   = (CUdeviceptr)address + length;
    if (key->b_len > 0) {
        if (key->d_bptr < range_start) {
            range_start = key->d_bptr;
        }
        if ((key->d_bptr + key->b_len) > range_end) {
            range_end = key->d_bptr + key->b_len;
        }
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuPointerGetAttribute(&buf_id_start, CU_POINTER_ATTRIBUTE_BUFFER_ID,
                                  range_start));
    if (status != UCS_OK) {
        goto out_pop;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuPointerGetAttribute(&buf_id_end, CU_POINTER_ATTRIBUTE_BUFFER_ID,
                                  range_end - 1));
    if (status != UCS_OK) {
        goto out_pop;
    }

    if (buf_id_start == buf_id_end) {
        status = UCS_ERR_UNSUPPORTED;
        goto out_pop;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemGetAddressRange(&first_base, &first_size, range_start));
    if (status != UCS_OK) {
        goto out_pop;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemGetAddressRange(&last_base, &last_size, range_end - 1));
    if (status != UCS_OK) {
        goto out_pop;
    }

    meta->d_bptr = first_base;
    meta->b_len  = (last_base + last_size) - first_base;

    status = uct_cuda_ipc_vmm_multi_create_meta_buffer(key, memh->dev_num);

out_pop:
    uct_cuda_ipc_check_and_pop_ctx(is_ctx_pushed);
    return status;
}

static ucs_status_t
uct_cuda_ipc_vmm_multi_fetch_meta_import(const CUmemFabricHandle *fabric_handle,
                                         CUdevice cu_dev, size_t alloc_size,
                                         CUdeviceptr *dev_ptr_p,
                                         CUmemGenericAllocationHandle *handle_p,
                                         ucs_log_level_t log_level)
{
    CUmemAccessDesc access;
    ucs_status_t status;

    status = UCT_CUDADRV_FUNC(
            cuMemImportFromShareableHandle(handle_p, (void*)fabric_handle,
                                           CU_MEM_HANDLE_TYPE_FABRIC),
            log_level);
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_CUDADRV_FUNC(cuMemAddressReserve(dev_ptr_p, alloc_size, 0, 0,
                                                  0),
                              log_level);
    if (status != UCS_OK) {
        goto err_release;
    }

    status = UCT_CUDADRV_FUNC(cuMemMap(*dev_ptr_p, alloc_size, 0, *handle_p, 0),
                              log_level);
    if (status != UCS_OK) {
        goto err_free_va;
    }

    uct_cuda_ipc_init_access_desc(&access, cu_dev);
    status = UCT_CUDADRV_FUNC(cuMemSetAccess(*dev_ptr_p, alloc_size, &access,
                                             1),
                              log_level);
    if (status != UCS_OK) {
        goto err_unmap;
    }

    return UCS_OK;

err_unmap:
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemUnmap(*dev_ptr_p, alloc_size));
err_free_va:
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemAddressFree(*dev_ptr_p, alloc_size));
err_release:
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemRelease(*handle_p));
    return status;
}

static void
uct_cuda_ipc_vmm_multi_fetch_meta_release(CUdeviceptr dev_ptr,
                                          size_t alloc_size,
                                          CUmemGenericAllocationHandle handle)
{
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemUnmap(dev_ptr, alloc_size));
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemAddressFree(dev_ptr, alloc_size));
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemRelease(handle));
}

ucs_status_t
uct_cuda_ipc_vmm_multi_fetch_chunks(uct_cuda_ipc_unpacked_rkey_t *rkey,
                                    CUdevice cu_dev, ucs_log_level_t log_level)
{
    CUmemAllocationProp prop = {};
    CUmemGenericAllocationHandle header_handle, chunks_handle;
    uct_cuda_ipc_vmm_meta_header_t header;
    CUdeviceptr header_dev_ptr, chunks_dev_ptr;
    size_t alloc_granularity, header_alloc_size, chunks_alloc_size, chunks_size;
    ucs_status_t status;

    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id          = cu_dev;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    status = UCT_CUDADRV_FUNC(
            cuMemGetAllocationGranularity(&alloc_granularity, &prop,
                                          CU_MEM_ALLOC_GRANULARITY_MINIMUM),
            log_level);
    if (status != UCS_OK) {
        return status;
    }

    header_alloc_size = alloc_granularity;
    status            = uct_cuda_ipc_vmm_multi_fetch_meta_import(
            &rkey->super.super.ph.handle.fabric_handle, cu_dev,
            header_alloc_size, &header_dev_ptr, &header_handle, log_level);
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_CUDADRV_FUNC(cuMemcpyDtoH(&header, header_dev_ptr,
                                           sizeof(header)),
                              log_level);
    uct_cuda_ipc_vmm_multi_fetch_meta_release(header_dev_ptr, header_alloc_size,
                                              header_handle);
    if (status != UCS_OK) {
        return status;
    }

    rkey->num_chunks = header.num_chunks;
    chunks_size = header.num_chunks * sizeof(uct_cuda_ipc_vmm_chunk_desc_t);

    rkey->chunks = ucs_malloc(chunks_size, "vmm_multi_chunks");
    if (rkey->chunks == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    chunks_alloc_size = ucs_align_up(chunks_size, alloc_granularity);
    status            = uct_cuda_ipc_vmm_multi_fetch_meta_import(
            &header.chunks_handle.handle.fabric, cu_dev, chunks_alloc_size,
            &chunks_dev_ptr, &chunks_handle, log_level);
    if (status != UCS_OK) {
        goto err_free;
    }

    status = UCT_CUDADRV_FUNC(cuMemcpyDtoH(rkey->chunks, chunks_dev_ptr,
                                           chunks_size),
                              log_level);
    uct_cuda_ipc_vmm_multi_fetch_meta_release(chunks_dev_ptr, chunks_alloc_size,
                                              chunks_handle);
    if (status != UCS_OK) {
        goto err_free;
    }

    ucs_trace("fetched %u VMM chunk descriptors from remote", rkey->num_chunks);
    return UCS_OK;

err_free:
    ucs_free(rkey->chunks);
    rkey->chunks     = NULL;
    rkey->num_chunks = 0;
    return status;
}

#endif
