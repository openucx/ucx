/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <tools/perf/lib/libperf_int.h>

#include <cuda_runtime.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/ptr_arith.h>
#include <uct/api/v2/uct_v2.h>

#define CUDA_CALL(_ret_code, _func, ...) \
    { \
        cudaError_t _cerr = _func(__VA_ARGS__); \
        if (_cerr != cudaSuccess) { \
            ucs_error("%s() failed: %d (%s)", UCS_PP_MAKE_STRING(_func), \
                      _cerr, cudaGetErrorString(_cerr)); \
            return _ret_code; \
        } \
    }

static ucs_status_t ucx_perf_cuda_init(ucx_perf_context_t *perf)
{
    unsigned group_index;
    int num_gpus;
    int gpu_index;

    group_index = rte_call(perf, group_index);

    CUDA_CALL(UCS_ERR_NO_DEVICE, cudaGetDeviceCount, &num_gpus)
    if (num_gpus == 0) {
        ucs_error("no cuda devices available");
        return UCS_ERR_NO_DEVICE;
    }

    gpu_index = (group_index == 0) ? perf->params.recv_device.device_id :
                                     perf->params.send_device.device_id;
    if (gpu_index == UCX_PERF_MEM_DEV_DEFAULT) {
        gpu_index = group_index % num_gpus;
    } else if (gpu_index >= num_gpus) {
        ucs_error("Illegal cuda device %d number of devices %d", gpu_index,
                  num_gpus);
        return UCS_ERR_NO_DEVICE;
    }

    CUDA_CALL(UCS_ERR_NO_DEVICE, cudaSetDevice, gpu_index);

    /* actually set device context as calling cudaSetDevice may result in
     * context being initialized lazily */
    cudaFree(0);

    return UCS_OK;
}

static inline ucs_status_t ucx_perf_cuda_alloc(size_t length,
                                               ucs_memory_type_t mem_type,
                                               void **address_p)
{
    if (mem_type == UCS_MEMORY_TYPE_CUDA) {
        CUDA_CALL(UCS_ERR_NO_MEMORY, cudaMalloc, address_p, length);
    } else if (mem_type == UCS_MEMORY_TYPE_CUDA_MANAGED) {
        CUDA_CALL(UCS_ERR_NO_MEMORY, cudaMallocManaged, address_p, length,
                  cudaMemAttachGlobal);
    } else {
        ucs_error("invalid memory type %s (%d)",
                  ucs_memory_type_names[mem_type], mem_type);
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static inline ucs_status_t
uct_perf_cuda_alloc_reg_mem(const ucx_perf_context_t *perf,
                            size_t length,
                            ucs_memory_type_t mem_type,
                            unsigned flags,
                            uct_allocated_memory_t *alloc_mem)
{
    uct_md_attr_v2_t md_attr = {.field_mask = UCT_MD_ATTR_FIELD_REG_ALIGNMENT};
    void *reg_address;
    ucs_status_t status;

    status = uct_md_query_v2(perf->uct.md, &md_attr);
    if (status != UCS_OK) {
        ucs_error("uct_md_query_v2() returned %d", status);
        return status;
    }

    status = ucx_perf_cuda_alloc(length, mem_type, &alloc_mem->address);
    if (status != UCS_OK) {
        return status;
    }

    /* Register memory respecting MD reg_alignment */
    reg_address = alloc_mem->address;
    ucs_align_ptr_range(&reg_address, &length, md_attr.reg_alignment);

    status = uct_md_mem_reg(perf->uct.md, reg_address, length, flags,
                            &alloc_mem->memh);
    if (status != UCS_OK) {
        cudaFree(alloc_mem->address);
        ucs_error("failed to register memory");
        return status;
    }

    alloc_mem->mem_type = mem_type;
    alloc_mem->md       = perf->uct.md;

    return UCS_OK;
}

static ucs_status_t uct_perf_cuda_alloc(const ucx_perf_context_t *perf,
                                        size_t length, unsigned flags,
                                        uct_allocated_memory_t *alloc_mem)
{
    return uct_perf_cuda_alloc_reg_mem(perf, length, UCS_MEMORY_TYPE_CUDA,
                                       flags, alloc_mem);
}

static ucs_status_t uct_perf_cuda_managed_alloc(const ucx_perf_context_t *perf,
                                                size_t length, unsigned flags,
                                                uct_allocated_memory_t *alloc_mem)
{
    return uct_perf_cuda_alloc_reg_mem(perf, length, UCS_MEMORY_TYPE_CUDA_MANAGED,
                                       flags, alloc_mem);
}

static void uct_perf_cuda_free(const ucx_perf_context_t *perf,
                               uct_allocated_memory_t *alloc_mem)
{
    ucs_status_t status;

    ucs_assert(alloc_mem->md == perf->uct.md);

    status = uct_md_mem_dereg(perf->uct.md, alloc_mem->memh);
    if (status != UCS_OK) {
        ucs_error("failed to deregister memory");
    }

    cudaFree(alloc_mem->address);
}

static void ucx_perf_cuda_memcpy(void *dst, ucs_memory_type_t dst_mem_type,
                                 const void *src, ucs_memory_type_t src_mem_type,
                                 size_t count)
{
    CUDA_CALL(, cudaMemcpy, dst, src, count, cudaMemcpyDefault);
    CUDA_CALL(, cudaDeviceSynchronize);
}

static void* ucx_perf_cuda_memset(void *dst, int value, size_t count)
{
    CUDA_CALL(dst, cudaMemset, dst, value, count);
    CUDA_CALL(dst, cudaDeviceSynchronize);
    return dst;
}

UCS_STATIC_INIT {
    static ucx_perf_allocator_t cuda_allocator = {
        .mem_type  = UCS_MEMORY_TYPE_CUDA,
        .init      = ucx_perf_cuda_init,
        .uct_alloc = uct_perf_cuda_alloc,
        .uct_free  = uct_perf_cuda_free,
        .memcpy    = ucx_perf_cuda_memcpy,
        .memset    = ucx_perf_cuda_memset
    };
    static ucx_perf_allocator_t cuda_managed_allocator = {
        .mem_type  = UCS_MEMORY_TYPE_CUDA_MANAGED,
        .init      = ucx_perf_cuda_init,
        .uct_alloc = uct_perf_cuda_managed_alloc,
        .uct_free  = uct_perf_cuda_free,
        .memcpy    = ucx_perf_cuda_memcpy,
        .memset    = ucx_perf_cuda_memset
    };

    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_CUDA]         = &cuda_allocator;
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_CUDA_MANAGED] = &cuda_managed_allocator;
}

UCS_STATIC_CLEANUP {
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_CUDA]         = NULL;
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_CUDA_MANAGED] = NULL;
}
