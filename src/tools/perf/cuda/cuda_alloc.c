/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_common.h"
#include <tools/perf/lib/libperf_int.h>

#include <cuda_runtime.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/ptr_arith.h>

#include <string.h>


static ucs_status_t ucx_perf_cuda_init(ucx_perf_context_t *perf)
{
    unsigned group_index;
    int num_gpus;
    int gpu_index;

    group_index = rte_call(perf, group_index);

    CUDA_CALL_RET(UCS_ERR_NO_DEVICE, cudaGetDeviceCount, &num_gpus);
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

    CUDA_CALL_RET(UCS_ERR_NO_DEVICE, cudaSetDevice, gpu_index);

    /* actually set device context as calling cudaSetDevice may result in
     * context being initialized lazily */
    cudaFree(0);

    return UCS_OK;
}

static inline ucs_status_t ucx_perf_cuda_mem_alloc(size_t length,
                                                   ucs_memory_type_t mem_type,
                                                   void **address_p)
{
    if (mem_type == UCS_MEMORY_TYPE_CUDA) {
        CUDA_CALL_RET(UCS_ERR_NO_MEMORY, cudaMalloc, address_p, length);
    } else if (mem_type == UCS_MEMORY_TYPE_CUDA_MANAGED) {
        CUDA_CALL_RET(UCS_ERR_NO_MEMORY, cudaMallocManaged, address_p, length,
                      cudaMemAttachGlobal);
    } else {
        ucs_error("invalid memory type %s (%d)",
                  ucs_memory_type_names[mem_type], mem_type);
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static ucs_status_t ucx_perf_cuda_alloc_mem_alloc(
        const ucx_perf_context_t *perf, size_t length, void **address_p)
{
    (void)perf;
    return ucx_perf_cuda_mem_alloc(length, UCS_MEMORY_TYPE_CUDA, address_p);
}

static void ucx_perf_cuda_alloc_mem_free(const ucx_perf_context_t *perf,
                                         void *address)
{
    (void)perf;
    CUDA_CALL_WARN(cudaFree, address);
}

#if CUDART_VERSION >= 11020
static ucs_status_t ucx_perf_cuda_async_mem_alloc(
        const ucx_perf_context_t *perf, size_t length, void **address_p)
{
    cudaError_t cerr;
    void *address;

    (void)perf;

    CUDA_CALL_RET(UCS_ERR_NO_MEMORY, cudaMallocAsync, &address, length, 0);
    cerr = cudaStreamSynchronize(0);
    if (cerr != cudaSuccess) {
        ucs_error("cudaStreamSynchronize() failed: %d (%s)", (int)cerr,
                  cudaGetErrorString(cerr));
        CUDA_CALL_WARN(cudaFreeAsync, address, 0);
        CUDA_CALL_WARN(cudaStreamSynchronize, 0);
        return UCS_ERR_IO_ERROR;
    }

    *address_p = address;
    return UCS_OK;
}

static void ucx_perf_cuda_async_mem_free(
        const ucx_perf_context_t *perf, void *address)
{
    (void)perf;

    CUDA_CALL_WARN(cudaStreamSynchronize, 0);
    CUDA_CALL_WARN(cudaFreeAsync, address, 0);
    CUDA_CALL_WARN(cudaStreamSynchronize, 0);
}
#endif

static ucs_status_t ucx_perf_cuda_uct_reg_mem(
        const ucx_perf_context_t *perf, size_t length,
        ucs_memory_type_t mem_type, unsigned flags,
        uct_allocated_memory_t *alloc_mem)
{
    uct_md_attr_v2_t md_attr = {.field_mask = UCT_MD_ATTR_FIELD_REG_ALIGNMENT};
    size_t reg_length;
    void *reg_address;
    ucs_status_t status;

    status = uct_md_query_v2(perf->uct.md, &md_attr);
    if (status != UCS_OK) {
        ucs_error("uct_md_query_v2() returned %d", status);
        return status;
    }

    /* Register memory respecting MD reg_alignment */
    reg_address = alloc_mem->address;
    reg_length  = length;
    ucs_align_ptr_range(&reg_address, &reg_length, md_attr.reg_alignment);

    status = uct_perf_md_mem_reg(perf->uct.md, reg_address, reg_length, flags,
                                 mem_type, &alloc_mem->memh);
    if (status != UCS_OK) {
        ucs_error("failed to register memory");
        return status;
    }

    alloc_mem->mem_type = mem_type;
    alloc_mem->md       = perf->uct.md;
    alloc_mem->length   = length;
    return UCS_OK;
}

static ucs_status_t ucx_perf_cuda_uct_alloc_reg_mem(
        const ucx_perf_context_t *perf, size_t length,
        ucs_memory_type_t mem_type, unsigned flags,
        uct_allocated_memory_t *alloc_mem)
{
    ucs_status_t status;

    status = ucx_perf_cuda_mem_alloc(length, mem_type, &alloc_mem->address);
    if (status != UCS_OK) {
        return status;
    }

    status = ucx_perf_cuda_uct_reg_mem(perf, length, mem_type, flags,
                                       alloc_mem);
    if (status != UCS_OK) {
        CUDA_CALL_WARN(cudaFree, alloc_mem->address);
        return status;
    }

    return UCS_OK;
}

static ucs_status_t ucx_perf_cuda_uct_alloc(const ucx_perf_context_t *perf,
                                            size_t length, unsigned flags,
                                            uct_allocated_memory_t *alloc_mem)
{
    return ucx_perf_cuda_uct_alloc_reg_mem(perf, length, UCS_MEMORY_TYPE_CUDA,
                                           flags, alloc_mem);
}

static ucs_status_t ucx_perf_cuda_managed_uct_alloc(
        const ucx_perf_context_t *perf, size_t length, unsigned flags,
        uct_allocated_memory_t *alloc_mem)
{
    return ucx_perf_cuda_uct_alloc_reg_mem(perf, length,
                                           UCS_MEMORY_TYPE_CUDA_MANAGED, flags,
                                           alloc_mem);
}

#if CUDART_VERSION >= 11020
/* Resolve async allocation memory type from UCX_CUDA_COPY_ASYNC_MEM_TYPE. */
static ucs_memory_type_t ucx_perf_cuda_async_configured_mem_type(void)
{
    static int initialized                   = 0;
    static ucs_memory_type_t cached_mem_type = UCS_MEMORY_TYPE_CUDA_MANAGED;
    uct_component_h *components              = NULL;
    unsigned num_components                  = 0;
    ucs_memory_type_t result                 = UCS_MEMORY_TYPE_CUDA_MANAGED;
    uct_component_attr_t component_attr;
    uct_md_config_t *md_config;
    char value[64];
    ucs_memory_type_t mem_type;
    ucs_status_t status;
    unsigned i;

    if (initialized) {
        return cached_mem_type;
    }

    status = uct_query_components(&components, &num_components);
    if (status != UCS_OK) {
        ucs_debug("failed to query UCT components: %s",
                  ucs_status_string(status));
        goto out;
    }

    for (i = 0; i < num_components; ++i) {
        component_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_NAME;
        status = uct_component_query(components[i], &component_attr);
        if ((status != UCS_OK) || strcmp(component_attr.name, "cuda_cpy")) {
            continue;
        }

        status = uct_md_config_read(components[i], NULL, NULL, &md_config);
        if (status != UCS_OK) {
            ucs_debug("failed to read cuda_cpy MD config: %s",
                      ucs_status_string(status));
            break;
        }

        status = uct_config_get(md_config, "ASYNC_MEM_TYPE", value,
                                sizeof(value));
        uct_config_release(md_config);
        if (status != UCS_OK) {
            ucs_debug("failed to get ASYNC_MEM_TYPE: %s",
                      ucs_status_string(status));
            break;
        }

        for (mem_type = 0; mem_type < UCS_MEMORY_TYPE_LAST; ++mem_type) {
            if (strcmp(value, ucs_memory_type_names[mem_type])) {
                continue;
            }

            if ((mem_type == UCS_MEMORY_TYPE_CUDA) ||
                (mem_type == UCS_MEMORY_TYPE_CUDA_MANAGED)) {
                result = mem_type;
            } else {
                ucs_warn("wrong memory type for async memory allocations: "
                         "\"%s\"; cuda-managed will be used instead",
                         value);
            }
            break;
        }
        break;
    }

    uct_release_component_list(components);

out:
    cached_mem_type = result;
    initialized     = 1;
    return result;
}

static ucs_memory_type_t
ucx_perf_cuda_async_mem_type(const ucx_perf_context_t *perf,
                             const void *address, size_t length)
{
    ucs_memory_type_t mem_type;
    ucs_status_t status;

    if (perf->params.api != UCX_PERF_API_UCT) {
        return UCS_MEMORY_TYPE_LAST;
    }

    status = uct_md_detect_memory_type(perf->uct.md, address, length,
                                       &mem_type);
    if (status == UCS_OK) {
        return mem_type;
    }

    mem_type = ucx_perf_cuda_async_configured_mem_type();
    ucs_debug("failed to detect cuda async memory type: %s, using %s",
              ucs_status_string(status), ucs_memory_type_names[mem_type]);
    return mem_type;
}

static ucs_memory_type_t
ucx_perf_cuda_async_resolve_mem_type(const ucx_perf_allocator_t *allocator)
{
    (void)allocator;
    return ucx_perf_cuda_async_configured_mem_type();
}

static ucs_status_t ucx_perf_cuda_async_uct_alloc(
        const ucx_perf_context_t *perf, size_t length, unsigned flags,
        uct_allocated_memory_t *alloc_mem)
{
    ucs_memory_type_t mem_type;
    ucs_status_t status;

    status = ucx_perf_cuda_async_mem_alloc(perf, length, &alloc_mem->address);
    if (status != UCS_OK) {
        return status;
    }

    mem_type = ucx_perf_cuda_async_mem_type(perf, alloc_mem->address, length);
    status = ucx_perf_cuda_uct_reg_mem(perf, length, mem_type, flags,
                                       alloc_mem);
    if (status != UCS_OK) {
        ucx_perf_cuda_async_mem_free(perf, alloc_mem->address);
        return status;
    }

    return UCS_OK;
}
#endif

static void ucx_perf_cuda_uct_dereg(const ucx_perf_context_t *perf,
                                    uct_allocated_memory_t *alloc_mem)
{
    ucs_status_t status;

    ucs_assert(alloc_mem->md == perf->uct.md);

    status = uct_md_mem_dereg(perf->uct.md, alloc_mem->memh);
    if (status != UCS_OK) {
        ucs_error("failed to deregister memory");
    }
}

static void ucx_perf_cuda_uct_free(const ucx_perf_context_t *perf,
                                   uct_allocated_memory_t *alloc_mem)
{
    ucx_perf_cuda_uct_dereg(perf, alloc_mem);
    CUDA_CALL_WARN(cudaFree, alloc_mem->address);
}

#if CUDART_VERSION >= 11020
static void ucx_perf_cuda_async_uct_free(const ucx_perf_context_t *perf,
                                         uct_allocated_memory_t *alloc_mem)
{
    ucx_perf_cuda_uct_dereg(perf, alloc_mem);
    ucx_perf_cuda_async_mem_free(perf, alloc_mem->address);
}
#endif

static void ucx_perf_cuda_memcpy(void *dst, ucs_memory_type_t dst_mem_type,
                                 const void *src, ucs_memory_type_t src_mem_type,
                                 size_t count)
{
    CUDA_CALL_ERR(cudaMemcpy, dst, src, count, cudaMemcpyDefault);
    CUDA_CALL_ERR(cudaDeviceSynchronize);
}

static void* ucx_perf_cuda_memset(void *dst, int value, size_t count)
{
    CUDA_CALL_RET(dst, cudaMemset, dst, value, count);
    CUDA_CALL_ERR(cudaDeviceSynchronize);
    return dst;
}

#if CUDART_VERSION >= 11020
static ucx_perf_allocator_t cuda_async_allocator = {
    .name             = "cuda-async",
    .default_mem_type = UCS_MEMORY_TYPE_CUDA_MANAGED,
    .init             = ucx_perf_cuda_init,
    .uct_alloc        = ucx_perf_cuda_async_uct_alloc,
    .uct_free         = ucx_perf_cuda_async_uct_free,
    .mem_alloc        = ucx_perf_cuda_async_mem_alloc,
    .mem_free         = ucx_perf_cuda_async_mem_free,
    .detect_mem_type  = ucx_perf_cuda_async_mem_type,
    .resolve_mem_type = ucx_perf_cuda_async_resolve_mem_type,
    .memcpy           = ucx_perf_cuda_memcpy,
    .memset           = ucx_perf_cuda_memset
};
#endif

static ucx_perf_allocator_t cuda_ucp_allocator = {
    .name             = "cuda",
    .default_mem_type = UCS_MEMORY_TYPE_CUDA,
    .init             = ucx_perf_cuda_init,
    .uct_alloc        = ucx_perf_cuda_uct_alloc,
    .uct_free         = ucx_perf_cuda_uct_free,
    .resolve_mem_type = ucx_perf_allocator_default_resolve_mem_type,
    .memcpy           = ucx_perf_cuda_memcpy,
    .memset           = ucx_perf_cuda_memset
};

static ucx_perf_allocator_t cuda_alloc_allocator = {
    .name             = "cuda-alloc",
    .default_mem_type = UCS_MEMORY_TYPE_CUDA,
    .init             = ucx_perf_cuda_init,
    .uct_alloc        = ucx_perf_cuda_uct_alloc,
    .uct_free         = ucx_perf_cuda_uct_free,
    .mem_alloc        = ucx_perf_cuda_alloc_mem_alloc,
    .mem_free         = ucx_perf_cuda_alloc_mem_free,
    .resolve_mem_type = ucx_perf_allocator_default_resolve_mem_type,
    .memcpy           = ucx_perf_cuda_memcpy,
    .memset           = ucx_perf_cuda_memset
};

static ucx_perf_allocator_t cuda_managed_allocator = {
    .name             = "cuda-managed",
    .default_mem_type = UCS_MEMORY_TYPE_CUDA_MANAGED,
    .init             = ucx_perf_cuda_init,
    .uct_alloc        = ucx_perf_cuda_managed_uct_alloc,
    .uct_free         = ucx_perf_cuda_uct_free,
    .resolve_mem_type = ucx_perf_allocator_default_resolve_mem_type,
    .memcpy           = ucx_perf_cuda_memcpy,
    .memset           = ucx_perf_cuda_memset
};

UCS_STATIC_INIT {
    ucx_perf_allocator_register(&cuda_ucp_allocator);
    ucx_perf_allocator_register(&cuda_alloc_allocator);
    ucx_perf_allocator_register(&cuda_managed_allocator);
#if CUDART_VERSION >= 11020
    ucx_perf_allocator_register(&cuda_async_allocator);
#endif
}

UCS_STATIC_CLEANUP {
#if CUDART_VERSION >= 11020
    ucx_perf_allocator_unregister(&cuda_async_allocator);
#endif
    ucx_perf_allocator_unregister(&cuda_managed_allocator);
    ucx_perf_allocator_unregister(&cuda_alloc_allocator);
    ucx_perf_allocator_unregister(&cuda_ucp_allocator);
}
