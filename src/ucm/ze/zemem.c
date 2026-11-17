/**
 * Copyright (C) Intel Corporation, 2023. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "zemem.h"
#include <ucm/event/event.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucm/util/replace.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/preprocessor.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>

UCM_DEFINE_REPLACE_DLSYM_FUNC(zeMemAllocHost, ze_result_t, -1,
                              ze_context_handle_t,
                              const ze_host_mem_alloc_desc_t*, size_t, size_t,
                              void**)
UCM_DEFINE_REPLACE_DLSYM_FUNC(zeMemAllocDevice, ze_result_t, -1,
                              ze_context_handle_t,
                              const ze_device_mem_alloc_desc_t*, size_t, size_t,
                              ze_device_handle_t, void**)
UCM_DEFINE_REPLACE_DLSYM_FUNC(zeMemAllocShared, ze_result_t, -1,
                              ze_context_handle_t,
                              const ze_device_mem_alloc_desc_t*,
                              const ze_host_mem_alloc_desc_t*, size_t, size_t,
                              ze_device_handle_t, void**)
UCM_DEFINE_REPLACE_DLSYM_FUNC(zeMemFree, ze_result_t, -1, ze_context_handle_t,
                              void*)

static void
ucm_zeMemFree_dispatch_events(ze_context_handle_t ze_context, void *ptr)
{
    ze_result_t ret;
    size_t size = 1; /* minimum size by default */
    ucs_memory_type_t mem_type;
    ze_memory_allocation_properties_t props = {};

    if (ptr == NULL) {
        return;
    }

    ret = zeMemGetAllocProperties(ze_context, ptr, &props, NULL);
    if (ret != ZE_RESULT_SUCCESS) {
        ucm_warn("zeMemGetAllocProperties(ptr=%p) failed", ptr);
        mem_type = UCS_MEMORY_TYPE_HOST;
    } else {
        switch (props.type) {
        case ZE_MEMORY_TYPE_HOST:
            mem_type = UCS_MEMORY_TYPE_ZE_HOST;
            break;
        case ZE_MEMORY_TYPE_DEVICE:
            mem_type = UCS_MEMORY_TYPE_ZE_DEVICE;
            break;
        case ZE_MEMORY_TYPE_SHARED:
            mem_type = UCS_MEMORY_TYPE_ZE_MANAGED;
            break;
        default:
            mem_type = UCS_MEMORY_TYPE_HOST;
            break;
        }
        (void)zeMemGetAddressRange(ze_context, ptr, NULL, &size);
    }

    ucm_dispatch_mem_type_free(ptr, size, mem_type);
}

ze_result_t ucm_zeMemAllocHost(ze_context_handle_t context,
                               const ze_host_mem_alloc_desc_t *host_desc,
                               size_t size, size_t alignment, void **ptr)
{
    ze_result_t ret;

    ucm_event_enter();

    ret = ucm_orig_zeMemAllocHost(context, host_desc, size, alignment, ptr);
    if (ret == ZE_RESULT_SUCCESS) {
        ucm_trace("ucm_zeMemAllocHost(ptr=%p size:%lu)", *ptr, size);
        ucm_dispatch_mem_type_alloc(*ptr, size, UCS_MEMORY_TYPE_ZE_HOST,
                                    UCS_MEM_FLAG_REGISTRABLE);
    }

    ucm_event_leave();
    return ret;
}

ze_result_t ucm_zeMemAllocDevice(ze_context_handle_t context,
                                 const ze_device_mem_alloc_desc_t *device_desc,
                                 size_t size, size_t alignment,
                                 ze_device_handle_t device, void **ptr)
{
    ze_result_t ret;

    ucm_event_enter();

    ret = ucm_orig_zeMemAllocDevice(context, device_desc, size, alignment,
                                    device, ptr);
    if (ret == ZE_RESULT_SUCCESS) {
        ucm_trace("ucm_zeMemAllocDevice(ptr=%p size:%lu)", *ptr, size);
        ucm_dispatch_mem_type_alloc(*ptr, size, UCS_MEMORY_TYPE_ZE_DEVICE,
                                    UCS_MEM_FLAG_REGISTRABLE);
    }

    ucm_event_leave();
    return ret;
}

ze_result_t ucm_zeMemAllocShared(ze_context_handle_t context,
                                 const ze_device_mem_alloc_desc_t *device_desc,
                                 const ze_host_mem_alloc_desc_t *host_desc,
                                 size_t size, size_t alignment,
                                 ze_device_handle_t device, void **ptr)
{
    ze_result_t ret;

    ucm_event_enter();

    ret = ucm_orig_zeMemAllocShared(context, device_desc, host_desc, size,
                                    alignment, device, ptr);
    if (ret == ZE_RESULT_SUCCESS) {
        ucm_trace("ucm_zeMemAllocShared(ptr=%p size:%lu)", *ptr, size);
        ucm_dispatch_mem_type_alloc(*ptr, size, UCS_MEMORY_TYPE_ZE_MANAGED,
                                    UCS_MEM_FLAG_REGISTRABLE);
    }

    ucm_event_leave();
    return ret;
}

ze_result_t ucm_zeMemFree(ze_context_handle_t context, void *ptr)
{
    ze_result_t ret;

    ucm_event_enter();

    ucm_trace("ucm_zeMemFree(context=%p, ptr=%p)", context, ptr);

    ucm_zeMemFree_dispatch_events(context, ptr);

    ret = ucm_orig_zeMemFree(context, ptr);

    ucm_event_leave();
    return ret;
}

static ucm_reloc_patch_t patches[] = {{"zeMemAllocHost", ucm_zeMemAllocHost},
                                      {"zeMemAllocDevice",
                                       ucm_zeMemAllocDevice},
                                      {"zeMemAllocShared",
                                       ucm_zeMemAllocShared},
                                      {"zeMemFree", ucm_zeMemFree},
                                      {NULL, NULL}};

static ucs_status_t ucm_zemem_install(int events)
{
    static int ucm_zemem_installed       = 0;
    static pthread_mutex_t install_mutex = PTHREAD_MUTEX_INITIALIZER;
    ucm_reloc_patch_t *patch;
    ucs_status_t status = UCS_OK;

    if (!(events & (UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE))) {
        goto out;
    }

    pthread_mutex_lock(&install_mutex);

    if (ucm_zemem_installed) {
        goto out_unlock;
    }

    for (patch = patches; patch->symbol != NULL; ++patch) {
        status = ucm_reloc_modify(patch);
        if (status != UCS_OK) {
            ucm_warn("failed to install relocation table entry for '%s'",
                     patch->symbol);
            goto out_unlock;
        }
    }

    ucm_info("ze hooks are ready");
    ucm_zemem_installed = 1;

out_unlock:
    pthread_mutex_unlock(&install_mutex);
out:
    return status;
}

static ucm_event_installer_t ucm_ze_initializer = {
    .install      = ucm_zemem_install,
    .path_pattern = "/dev/dri"
};

UCS_STATIC_INIT
{
    ucs_list_add_tail(&ucm_event_installer_list, &ucm_ze_initializer.list);
}

UCS_STATIC_CLEANUP
{
    ucs_list_del(&ucm_ze_initializer.list);
}
