/**
 * Copyright (C) Intel Corporation, 2023-2024.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <tools/perf/lib/libperf_int.h>

#include <level_zero/ze_api.h>
#include <ucs/sys/compiler.h>

#define UCX_PERF_ZE_MAX_DEVICES 4

static const size_t gpu_page_size = 65536;
static ze_driver_handle_t gpu_driver;
static ze_context_handle_t gpu_context;
static ze_device_handle_t gpu_devices[UCX_PERF_ZE_MAX_DEVICES];
static ze_command_list_handle_t gpu_cmdlists[UCX_PERF_ZE_MAX_DEVICES];
static unsigned gpu_count;
static unsigned gpu_index;
static int ze_initialized;

static ze_result_t ze_init_devices(void)
{
    ze_context_desc_t ctxt_desc = {};
    ze_result_t ret;
    uint32_t count;

    if (ze_initialized)
        return ZE_RESULT_SUCCESS;

    ret = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (ret != ZE_RESULT_SUCCESS) {
        return ret;
    }

    count = 1;
    ret   = zeDriverGet(&count, &gpu_driver);
    if (ret != ZE_RESULT_SUCCESS) {
        return ret;
    }

    count = UCX_PERF_ZE_MAX_DEVICES;
    ret = zeDeviceGet(gpu_driver, &count, gpu_devices);
    if (ret != ZE_RESULT_SUCCESS) {
        return ret;
    }

    ret = zeContextCreate(gpu_driver, &ctxt_desc, &gpu_context);
    if (ret != ZE_RESULT_SUCCESS) {
        return ret;
    }

    gpu_count = count;
    ze_initialized = 1;
    return ZE_RESULT_SUCCESS;
}

static ucs_status_t ucx_perf_ze_init(ucx_perf_context_t *perf)
{
    ze_command_queue_desc_t cmdq_desc = {
        .stype    = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .ordinal  = 0,
        .index    = 0,
        .flags    = 0,
        .mode     = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
    };
    unsigned group_index, i;
    ze_result_t ret;

    ret = ze_init_devices();
    if (ret != ZE_RESULT_SUCCESS) {
        return UCS_ERR_NO_DEVICE;
    }

    group_index = rte_call(perf, group_index);
    i           = group_index % gpu_count;

    if (!gpu_cmdlists[i]) {
        ret = zeCommandListCreateImmediate(gpu_context, gpu_devices[i],
                                           &cmdq_desc, &gpu_cmdlists[i]);
        if (ret != ZE_RESULT_SUCCESS) {
            return UCS_ERR_NO_DEVICE;
        }
    }

    gpu_index = i;
    return UCS_OK;
}

static ucs_status_t
ucx_perf_ze_alloc(size_t length, ucs_memory_type_t mem_type, void **address_p)
{
    ze_device_mem_alloc_desc_t dev_desc = {};
    ze_host_mem_alloc_desc_t host_desc  = {};
    ze_result_t ret;

    ucs_assert((mem_type == UCS_MEMORY_TYPE_ZE_HOST) ||
               (mem_type == UCS_MEMORY_TYPE_ZE_DEVICE) ||
               (mem_type == UCS_MEMORY_TYPE_ZE_MANAGED));

    if (mem_type == UCS_MEMORY_TYPE_ZE_HOST)
        ret = zeMemAllocHost(gpu_context, &host_desc, length, gpu_page_size,
                             address_p);
    else if (mem_type == UCS_MEMORY_TYPE_ZE_DEVICE)
        ret = zeMemAllocDevice(gpu_context, &dev_desc, length, gpu_page_size,
                               gpu_devices[0], address_p);
    else
        ret = zeMemAllocShared(gpu_context, &dev_desc, &host_desc, length,
                               gpu_page_size, gpu_devices[0], address_p);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_error("failed to allocate memory");
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

static ucs_status_t
uct_perf_ze_alloc_reg_mem(const ucx_perf_context_t *perf, size_t length,
                          ucs_memory_type_t mem_type, unsigned flags,
                          uct_allocated_memory_t *alloc_mem)
{
    ucs_status_t status;

    status = ucx_perf_ze_alloc(length, mem_type, &alloc_mem->address);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_md_mem_reg(perf->uct.md, alloc_mem->address, length, flags,
                            &alloc_mem->memh);
    if (status != UCS_OK) {
        zeMemFree(gpu_context, alloc_mem->address);
        ucs_error("failed to register memory");
        return status;
    }

    alloc_mem->mem_type = mem_type;
    alloc_mem->md       = perf->uct.md;

    return UCS_OK;
}

static ucs_status_t uct_perf_ze_host_alloc(const ucx_perf_context_t *perf,
                                           size_t length, unsigned flags,
                                           uct_allocated_memory_t *alloc_mem)
{
    return uct_perf_ze_alloc_reg_mem(perf, length, UCS_MEMORY_TYPE_ZE_HOST,
                                     flags, alloc_mem);
}

static ucs_status_t uct_perf_ze_device_alloc(const ucx_perf_context_t *perf,
                                             size_t length, unsigned flags,
                                             uct_allocated_memory_t *alloc_mem)
{
    return uct_perf_ze_alloc_reg_mem(perf, length, UCS_MEMORY_TYPE_ZE_DEVICE,
                                     flags, alloc_mem);
}

static ucs_status_t uct_perf_ze_managed_alloc(const ucx_perf_context_t *perf,
                                              size_t length, unsigned flags,
                                              uct_allocated_memory_t *alloc_mem)
{
    return uct_perf_ze_alloc_reg_mem(perf, length, UCS_MEMORY_TYPE_ZE_MANAGED,
                                     flags, alloc_mem);
}

static void uct_perf_ze_free(const ucx_perf_context_t *perf,
                             uct_allocated_memory_t *alloc_mem)
{
    ucs_status_t status;

    ucs_assert(alloc_mem->md == perf->uct.md);

    status = uct_md_mem_dereg(perf->uct.md, alloc_mem->memh);
    if (status != UCS_OK) {
        ucs_error("failed to deregister memory");
    }

    zeMemFree(gpu_context, alloc_mem->address);
}

static void ucx_perf_ze_memcpy(void *dst, ucs_memory_type_t dst_mem_type,
                               const void *src, ucs_memory_type_t src_mem_type,
                               size_t count)
{
    ze_result_t ret;

    ret = zeCommandListAppendMemoryCopy(gpu_cmdlists[gpu_index], dst, src,
                                        count, NULL, 0, NULL);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_error("failed to copy memory: error code %x", ret);
    }

    ret = zeCommandListReset(gpu_cmdlists[gpu_index]);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_error("failed to reset command list: error code %x", ret);
    }
}

static void *ucx_perf_ze_memset(void *dst, int value, size_t count)
{
    ze_result_t ret;

    ret = zeCommandListAppendMemoryFill(gpu_cmdlists[gpu_index], dst, &value, 1,
                                        count, NULL, 0, NULL);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_error("failed to set memory: error code %x", ret);
    }

    ret = zeCommandListReset(gpu_cmdlists[gpu_index]);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_error("failed to reset command list: error code %x", ret);
    }

    return dst;
}

UCS_STATIC_INIT
{
    static ucx_perf_allocator_t ze_host_allocator    = {
        .mem_type  = UCS_MEMORY_TYPE_ZE_HOST,
        .init      = ucx_perf_ze_init,
        .uct_alloc = uct_perf_ze_host_alloc,
        .uct_free  = uct_perf_ze_free,
        .memcpy    = ucx_perf_ze_memcpy,
        .memset    = ucx_perf_ze_memset
    };
    static ucx_perf_allocator_t ze_device_allocator  = {
        .mem_type  = UCS_MEMORY_TYPE_ZE_DEVICE,
        .init      = ucx_perf_ze_init,
        .uct_alloc = uct_perf_ze_device_alloc,
        .uct_free  = uct_perf_ze_free,
        .memcpy    = ucx_perf_ze_memcpy,
        .memset    = ucx_perf_ze_memset
    };
    static ucx_perf_allocator_t ze_managed_allocator = {
        .mem_type  = UCS_MEMORY_TYPE_ZE_MANAGED,
        .init      = ucx_perf_ze_init,
        .uct_alloc = uct_perf_ze_managed_alloc,
        .uct_free  = uct_perf_ze_free,
        .memcpy    = ucx_perf_ze_memcpy,
        .memset    = ucx_perf_ze_memset
    };

    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ZE_HOST] = &ze_host_allocator;
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ZE_DEVICE] =
            &ze_device_allocator;
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ZE_MANAGED] =
            &ze_managed_allocator;
}

UCS_STATIC_CLEANUP
{
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ZE_HOST]    = NULL;
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ZE_DEVICE]  = NULL;
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ZE_MANAGED] = NULL;
}
