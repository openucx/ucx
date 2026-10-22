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

#if HAVE_CUDA_FABRIC


void uct_cuda_ipc_vmm_multi_meta_cleanup(uct_cuda_ipc_lkey_t *key)
{
    uct_cuda_ipc_vmm_multi_meta_t *meta, *tmp;

    ucs_list_for_each_safe(meta, tmp, &key->vmm_multi_list, link) {
        UCT_CUDADRV_FUNC_LOG_WARN(
                cuMemUnmap(meta->dev_ptr, meta->info.alloc_size));
        UCT_CUDADRV_FUNC_LOG_WARN(
                cuMemAddressFree(meta->dev_ptr, meta->info.alloc_size));

        ucs_list_del(&meta->link);
        ucs_free(meta);
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
    unsigned long long chunk_buffer_id;
    uint64_t allowed_handle_types;
    CUpointer_attribute attr_type[2];
    void *attr_data[2];
    size_t chunk_size;
    ucs_status_t status;

    attr_type[0] = CU_POINTER_ATTRIBUTE_BUFFER_ID;
    attr_data[0] = &chunk_buffer_id;
    attr_type[1] = CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES;
    attr_data[1] = &allowed_handle_types;

    ucs_array_init_dynamic(&chunks);

    for (pos = va_base; pos < va_base + va_len; pos = chunk_base + chunk_size) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuMemGetAddressRange(&chunk_base, &chunk_size, pos));
        if (status != UCS_OK) {
            goto err;
        }

        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuPointerGetAttributes(ucs_static_array_size(attr_data),
                                       attr_type, attr_data, chunk_base));
        if (status != UCS_OK) {
            goto err;
        }

        if (!(allowed_handle_types & CU_MEM_HANDLE_TYPE_FABRIC)) {
            ucs_debug("VMM chunk 0x%llx does not allow fabric handles",
                      chunk_base);
            status = UCS_ERR_UNSUPPORTED;
            goto err;
        }

        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuMemRetainAllocationHandle(&handle, (void*)pos));
        if (status != UCS_OK) {
            goto err;
        }

        elem = ucs_array_append(&chunks, {
            UCT_CUDADRV_FUNC_LOG_WARN(cuMemRelease(handle));
            status = UCS_ERR_NO_MEMORY;
            goto err;
        });

        status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemExportToShareableHandle(
                &elem->fabric_handle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
        UCT_CUDADRV_FUNC_LOG_WARN(cuMemRelease(handle));
        if (status != UCS_OK) {
            goto err;
        }

        elem->d_bptr    = chunk_base;
        elem->b_len     = chunk_size;
        elem->buffer_id = chunk_buffer_id;
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
uct_cuda_ipc_vmm_multi_create_meta_buffer(uct_cuda_ipc_vmm_multi_meta_t *meta,
                                          int dev_num)
{
    uct_cuda_ipc_vmm_chunk_desc_t *host_chunks = NULL;
    uint16_t num_chunks                        = 0;
    CUmemAllocationProp prop                   = {};
    CUdeviceptr dev_ptr;
    size_t alloc_size;
    ucs_status_t status;
    size_t chunks_data_size;

    status = uct_cuda_ipc_vmm_multi_discover_chunks(meta->d_bptr, meta->b_len,
                                                    &host_chunks, &num_chunks);
    if (status != UCS_OK) {
        return status;
    }

    chunks_data_size = num_chunks * sizeof(*host_chunks);

    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id          = dev_num;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    status = uct_cuda_ipc_vmm_multi_meta_alloc_buffer(
            &dev_ptr, &alloc_size, &meta->fabric_handle, chunks_data_size,
            &prop);
    if (status != UCS_OK) {
        goto err_free_host;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemcpyHtoD(dev_ptr, host_chunks, chunks_data_size));
    if (status != UCS_OK) {
        goto err_cleanup;
    }

    meta->dev_ptr              = dev_ptr;
    meta->info.version         = UCT_CUDA_IPC_VMM_MULTI_VERSION;
    meta->info.num_chunks      = num_chunks;
    meta->info.info_size       = sizeof(meta->info);
    meta->info.chunk_desc_size = sizeof(*host_chunks);
    meta->info.alloc_size      = alloc_size;

    ucs_trace("created VMM metadata: %u chunks, allocation size %zu on GPU",
              num_chunks, alloc_size);
    ucs_free(host_chunks);
    return UCS_OK;

err_cleanup:
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemUnmap(dev_ptr, alloc_size));
    UCT_CUDADRV_FUNC_LOG_WARN(cuMemAddressFree(dev_ptr, alloc_size));
err_free_host:
    ucs_free(host_chunks);
    return status;
}

ucs_status_t uct_cuda_ipc_mkey_pack_vmm_multi_chunk(
        uct_cuda_ipc_lkey_t *key, void *address, size_t length,
        const uct_cuda_ipc_vmm_multi_meta_t **meta_p)
{
    uct_cuda_ipc_vmm_multi_meta_t *meta;
    CUdeviceptr last_base;
    size_t last_size;
    CUdevice cuda_device;
    int is_ctx_pushed;
    ucs_status_t status;

    /* Contained in the lkey's own allocation, so it cannot span chunks */
    if (((CUdeviceptr)address + length) <= (key->d_bptr + key->b_len)) {
        return UCS_ERR_UNSUPPORTED;
    }

    ucs_list_for_each(meta, &key->vmm_multi_list, link) {
        if (((CUdeviceptr)address >= meta->d_bptr) &&
            (((CUdeviceptr)address + length) <=
             (meta->d_bptr + meta->b_len))) {
            *meta_p = meta;
            return UCS_OK;
        }
    }

    status = uct_cuda_ipc_check_and_push_ctx((CUdeviceptr)address, &cuda_device,
                                             &is_ctx_pushed);
    if (status != UCS_OK) {
        return status;
    }

    /* The range starts in the lkey's allocation and ends past it */
    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemGetAddressRange(&last_base, &last_size,
                                 (CUdeviceptr)address + length - 1));
    if (status != UCS_OK) {
        goto out_pop;
    }

    meta = ucs_calloc(1, sizeof(*meta), "cuda_ipc_vmm_multi_meta");
    if (meta == NULL) {
        ucs_error("failed to allocate VMM metadata record");
        status = UCS_ERR_NO_MEMORY;
        goto out_pop;
    }

    meta->d_bptr = key->d_bptr;
    meta->b_len  = (last_base + last_size) - key->d_bptr;

    status = uct_cuda_ipc_vmm_multi_create_meta_buffer(meta, cuda_device);
    if (status != UCS_OK) {
        goto err_free_meta;
    }

    /* Peers may unpack any published metadata until memh deregistration. */
    ucs_list_add_tail(&key->vmm_multi_list, &meta->link);
    *meta_p = meta;
    goto out_pop;

err_free_meta:
    ucs_free(meta);

out_pop:
    uct_cuda_ipc_check_and_pop_ctx(is_ctx_pushed);
    return status;
}

#endif
