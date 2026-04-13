/*
 * Copyright (C) Intel Corporation, 2023-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <tools/perf/lib/libperf_int.h>

#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>
#include <uct/ze/base/ze_base.h>

#include <level_zero/ze_api.h>
#include <pthread.h>


#define ZE_PERF_MAX_DEVICES 32 /* Max root devices (GPUs) */


static ze_result_t ze_init_status  = ZE_RESULT_ERROR_UNINITIALIZED;
static pthread_once_t ze_init_once = PTHREAD_ONCE_INIT;
static ze_context_handle_t gpu_context;
static ze_device_handle_t gpu_devices[ZE_PERF_MAX_DEVICES];
static ze_driver_handle_t gpu_driver;
static unsigned gpu_count;

static __thread unsigned tls_gpu_index;
static __thread ze_command_list_handle_t tls_cmdlist;


static void ucx_perf_ze_destroy_tls_cmdlist(void)
{
    if (tls_cmdlist == NULL) {
        return;
    }

    zeCommandListDestroy(tls_cmdlist);
    tls_cmdlist = NULL;
}

static ze_result_t
ucx_perf_ze_create_tls_cmdlist(unsigned gpu_idx,
                               const ze_command_queue_desc_t *cmdq_desc)
{
    return zeCommandListCreateImmediate(gpu_context, gpu_devices[gpu_idx],
                                        cmdq_desc, &tls_cmdlist);
}

static void ze_do_init(void)
{
    ze_context_desc_t ctxt_desc = {
        .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC
    };
    ze_result_t ret;
    uint32_t count;

    ret = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (ret != ZE_RESULT_SUCCESS) {
        ze_init_status = ret;
        return;
    }

    count = 1;
    ret   = zeDriverGet(&count, &gpu_driver);
    if (ret != ZE_RESULT_SUCCESS) {
        ze_init_status = ret;
        return;
    }

    count = ZE_PERF_MAX_DEVICES;
    ret   = zeDeviceGet(gpu_driver, &count, gpu_devices);
    if (ret != ZE_RESULT_SUCCESS) {
        ze_init_status = ret;
        return;
    }

    if (count == 0) {
        ze_init_status = ZE_RESULT_ERROR_UNINITIALIZED;
        return;
    }

    ret = zeContextCreate(gpu_driver, &ctxt_desc, &gpu_context);
    if (ret != ZE_RESULT_SUCCESS) {
        ze_init_status = ret;
        return;
    }

    gpu_count      = count;
    ze_init_status = ZE_RESULT_SUCCESS;
}

static ze_result_t ze_init_devices(void)
{
    pthread_once(&ze_init_once, ze_do_init);
    return ze_init_status;
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
    ze_result_t ret;
    unsigned group_index, i;

    ret = ze_init_devices();
    if (ret != ZE_RESULT_SUCCESS) {
        return UCS_ERR_NO_DEVICE;
    }

    group_index = rte_call(perf, group_index);
    i           = group_index % gpu_count;

    if ((tls_cmdlist != NULL) && (tls_gpu_index != i)) {
        ucx_perf_ze_destroy_tls_cmdlist();
    }

    tls_gpu_index = i;
    if (tls_cmdlist == NULL) {
        ret = ucx_perf_ze_create_tls_cmdlist(i, &cmdq_desc);
        if (ret != ZE_RESULT_SUCCESS) {
            return UCS_ERR_NO_DEVICE;
        }
    }

    return UCS_OK;
}

static ucs_status_t
ucx_perf_ze_alloc(size_t length, ucs_memory_type_t mem_type, void **address_p)
{
    ze_device_mem_alloc_desc_t dev_desc = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC
    };
    ze_host_mem_alloc_desc_t host_desc  = {
        .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC
    };
    size_t alignment                    = ucs_get_page_size();
    ze_result_t ret;

    if (tls_cmdlist == NULL) {
        return UCS_ERR_NO_DEVICE;
    }

    if (mem_type == UCS_MEMORY_TYPE_ZE_HOST) {
        ret = zeMemAllocHost(gpu_context, &host_desc, length, alignment,
                             address_p);
    } else if (mem_type == UCS_MEMORY_TYPE_ZE_DEVICE) {
        ret = zeMemAllocDevice(gpu_context, &dev_desc, length, alignment,
                               gpu_devices[tls_gpu_index], address_p);
    } else if (mem_type == UCS_MEMORY_TYPE_ZE_MANAGED) {
        ret = zeMemAllocShared(gpu_context, &dev_desc, &host_desc, length,
                               alignment, gpu_devices[tls_gpu_index],
                               address_p);
    } else {
        ucs_error("invalid memory type %s (%d)",
                  ucs_memory_type_names[mem_type], mem_type);
        return UCS_ERR_INVALID_PARAM;
    }

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
        ucs_error("failed to register memory");
        zeMemFree(gpu_context, alloc_mem->address);
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

    if (tls_cmdlist == NULL) {
        ucs_error("ze memcpy called before initialization");
        return;
    }

    ret = zeCommandListAppendMemoryCopy(tls_cmdlist, dst, src, count, NULL, 0,
                                        NULL);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_error("failed to copy memory: error code %x", ret);
    }

    ret = zeCommandListReset(tls_cmdlist);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_error("failed to reset command list: error code %x", ret);
    }
}

static void *ucx_perf_ze_memset(void *dst, int value, size_t count)
{
    ze_result_t ret;

    if (tls_cmdlist == NULL) {
        ucs_error("ze memset called before initialization");
        return dst;
    }

    ret = zeCommandListAppendMemoryFill(tls_cmdlist, dst, &value, 1, count,
                                        NULL, 0, NULL);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_error("failed to set memory: error code %x", ret);
    }

    ret = zeCommandListReset(tls_cmdlist);
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
    ucx_perf_ze_destroy_tls_cmdlist();

    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ZE_HOST]    = NULL;
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ZE_DEVICE]  = NULL;
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_ZE_MANAGED] = NULL;
}
