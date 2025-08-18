/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "gdaki_mem_handle.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <ucs/sys/ptr_arith.h>
#include <ucs/type/status.h>
#include <ucs/debug/log.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_GDRAPI_H
#include <gdrapi.h>
#endif

struct gdaki_mem_handle {
    void   *gpu_ptr;
    void   *cpu_ptr;
    size_t size;
    bool   owns_gpu_mem;
#ifdef HAVE_GDRAPI_H
    gdr_t gdr;
    gdr_mh_t mh;
    void *aligned_gpu_ptr;
#endif
};

/*
 TODO: Make it cpp class.
       Add sync memory function.
*/

gdaki_mem_handle_t gdaki_mem_create(void *gpu_ptr, size_t size)
{
    gdaki_mem_handle_t handle = (gdaki_mem_handle_t)calloc(1, sizeof(*handle));
    int ret                   = UCS_OK;
    cudaError_t cuda_err;

    if (!handle) {
        ucs_error("Failed to allocate memory\n");
        return NULL;
    }

#ifndef HAVE_GDRAPI_H
    // TODO: Change fallback to cuda pinned memory.
    ucs_diag("GDRcopy is not available, using Managed\n");
    cuda_err = cudaMallocManaged(&handle->gpu_ptr, size);
    if (cuda_err != cudaSuccess) {
        ucs_error("Failed to allocate managed GPU memory: %s\n",
                  cudaGetErrorString(cuda_err));
        return NULL;
    }
    handle->owns_gpu_mem = true;
    handle->cpu_ptr      = handle->gpu_ptr;
    return handle;
#else
    handle->size = ucs_align_up_pow2(size + GPU_PAGE_SIZE, GPU_PAGE_SIZE);
    size_t offset;
    size_t aligned_size;

    // Allocate GPU memory if not provided
    if (gpu_ptr == NULL) {
        cuda_err = cudaMalloc(&handle->gpu_ptr, handle->size);
        if (cuda_err != cudaSuccess) {
            ucs_error("Failed to allocate GPU memory: %s\n",
                      cudaGetErrorString(cuda_err));
            free(handle);
            return NULL;
        }
        handle->owns_gpu_mem = true;
        cuda_err             = cudaMemset(handle->gpu_ptr, 0, handle->size);
        if (cuda_err != cudaSuccess) {
            ucs_error("Failed to initialize GPU memory: %s\n",
                      cudaGetErrorString(cuda_err));
            cudaFree(handle->gpu_ptr);
            free(handle);
            return NULL;
        }
    }
    // TODO: Support receiving gpu_ptr from outside after replacing managed memory with pinned.
    // else {
    //     assert(size % GPU_PAGE_SIZE == 0);
    //     handle->gpu_ptr = gpu_ptr;
    //     handle->owns_gpu_mem = false;
    // }

    // Initialize GDRcopy
    handle->gdr = gdr_open();
    if (!handle->gdr) {
        ucs_error("GDRcopy initialization failed\n");
        goto cleanup;
    }

    // Pin and map the buffer
    handle->aligned_gpu_ptr = (void*)ucs_align_up((uintptr_t)handle->gpu_ptr,
                                                  GPU_PAGE_SIZE);
    offset       = UCS_PTR_BYTE_DIFF(handle->aligned_gpu_ptr, handle->gpu_ptr);
    aligned_size = ucs_align_up(handle->size - offset, GPU_PAGE_SIZE);
    ret = gdr_pin_buffer(handle->gdr, (unsigned long)handle->aligned_gpu_ptr,
                         aligned_size, 0, 0, &handle->mh);
    if (ret) {
        ucs_error("GDRcopy pin buffer failed\n");
        gdr_close(handle->gdr);
        goto cleanup;
    }

    ret = gdr_map(handle->gdr, handle->mh, &handle->cpu_ptr, aligned_size);
    if (ret) {
        ucs_error("GDRcopy map failed\n");
        gdr_unpin_buffer(handle->gdr, handle->mh);
        gdr_close(handle->gdr);
        goto cleanup;
    }

    return handle;

cleanup:
    if (handle->owns_gpu_mem) {
        cudaFree(handle->gpu_ptr);
    }
    free(handle);
    return NULL;
#endif
}

void *gdaki_mem_get_ptr(gdaki_mem_handle_t handle)
{
    return handle ? handle->cpu_ptr : NULL;
}

void *gdaki_mem_get_gpu_ptr(gdaki_mem_handle_t handle)
{
#ifdef HAVE_GDRAPI_H
    return handle ? handle->aligned_gpu_ptr : NULL;
#else
    return handle ? handle->gpu_ptr : NULL;
#endif
}

void gdaki_mem_destroy(gdaki_mem_handle_t handle)
{
#ifdef HAVE_GDRAPI_H
    if (!handle) {
        return;
    }

    gdr_unmap(handle->gdr, handle->mh, handle->cpu_ptr, handle->size);
    gdr_unpin_buffer(handle->gdr, handle->mh);
    gdr_close(handle->gdr);
#endif

    if (handle->owns_gpu_mem) {
        cudaFree(handle->gpu_ptr);
    }

    free(handle);
}
