/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucm/rocm/rocmmem.h>

#include <ucm/event/event.h>
#include <ucm/mmap/mmap.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucm/util/replace.h>
#include <ucm/bistro/bistro.h>
#include <ucs/debug/assert.h>

#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>

/* Use the PTR variant so that ucm_orig_<fn> is a function pointer that bistro
 * can redirect to the relocated (trampoline) original, allowing us to intercept
 * callers that resolve the HSA symbol via dlopen/dlsym, which the reloc/GOT
 * hook cannot see. */
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(hsa_amd_memory_pool_allocate, hsa_status_t,
                                  HSA_STATUS_ERROR, hsa_amd_memory_pool_t,
                                  size_t, uint32_t, void**)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(hsa_amd_memory_pool_free, hsa_status_t,
                                  HSA_STATUS_ERROR, void*)

#ifdef HAVE_ROCM_VMM_TYPE
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(hsa_amd_vmem_map, hsa_status_t,
                                  HSA_STATUS_ERROR, void*, size_t, size_t,
                                  hsa_amd_vmem_alloc_handle_t, uint64_t)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(hsa_amd_vmem_unmap, hsa_status_t,
                                  HSA_STATUS_ERROR, void*, size_t)
#endif

static void ucm_hsa_amd_memory_pool_free_dispatch_events(void *ptr)
{
    size_t size;
    hsa_status_t status;
    hsa_device_type_t dev_type;
    ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_ROCM;
    hsa_amd_pointer_info_t info = {
        .size = sizeof(hsa_amd_pointer_info_t),
    };

    if (ptr == NULL) {
        return;
    }

    status = hsa_amd_pointer_info(ptr, &info, NULL, NULL, NULL);
    if (status != HSA_STATUS_SUCCESS) {
        ucm_warn("hsa_amd_pointer_info(dptr=%p) failed", ptr);
        size = 1; /* set minimum length */
    }
    else {
        size = info.sizeInBytes;
    }

    status = hsa_agent_get_info(info.agentOwner, HSA_AGENT_INFO_DEVICE, &dev_type);
    if (status == HSA_STATUS_SUCCESS) {
        if (info.type != HSA_EXT_POINTER_TYPE_HSA) {
            ucm_warn("ucm free non HSA managed memory %p", ptr);
            return;
        }

        if (dev_type != HSA_DEVICE_TYPE_GPU) {
            mem_type = UCS_MEMORY_TYPE_ROCM_MANAGED;
        }
    }

    ucm_dispatch_mem_type_free(ptr, size, mem_type);
}

hsa_status_t ucm_hsa_amd_memory_pool_free(void* ptr)
{
    hsa_status_t status;

    ucm_event_enter();

    ucm_trace("ucm_hsa_amd_memory_pool_free(ptr=%p)", ptr);

    ucm_hsa_amd_memory_pool_free_dispatch_events(ptr);

    status = ucm_orig_hsa_amd_memory_pool_free(ptr);

    ucm_event_leave();
    return status;
}

hsa_status_t ucm_hsa_amd_memory_pool_allocate(
    hsa_amd_memory_pool_t memory_pool, size_t size,
    uint32_t flags, void** ptr)
{
    hsa_status_t status;

    ucm_event_enter();

    status = ucm_orig_hsa_amd_memory_pool_allocate(memory_pool, size, flags, ptr);
    if (status == HSA_STATUS_SUCCESS) {
        ucm_trace("ucm_hsa_amd_memory_pool_allocate(ptr=%p size:%lu)", *ptr, size);
        ucm_dispatch_mem_type_alloc(*ptr, size, UCS_MEMORY_TYPE_UNKNOWN,
                                    UCS_MEM_FLAG_REGISTRABLE);
    }

    ucm_event_leave();
    return status;
}

#ifdef HAVE_ROCM_VMM_TYPE
hsa_status_t ucm_hsa_amd_vmem_map(void* va, size_t size, size_t in_offset,
                                   hsa_amd_vmem_alloc_handle_t handle,
                                   uint64_t flags)
{
    hsa_status_t status;

    ucm_event_enter();

    status = ucm_orig_hsa_amd_vmem_map(va, size, in_offset, handle, flags);
    if (status == HSA_STATUS_SUCCESS) {
        ucm_trace("ucm_hsa_amd_vmem_map(va=%p size:%lu)", va, size);
        /* Keep type unknown for MD detection; the VMM registry hook queries the
         * pointer type to tell a real VMEM mapping from a pool allocation. */
        ucm_dispatch_mem_type_alloc(va, size, UCS_MEMORY_TYPE_UNKNOWN,
                                    UCS_MEM_FLAG_REGISTRABLE);
    }

    ucm_event_leave();
    return status;
}

hsa_status_t ucm_hsa_amd_vmem_unmap(void* va, size_t size)
{
    hsa_status_t status;

    ucm_event_enter();

    ucm_trace("ucm_hsa_amd_vmem_unmap(va=%p size:%lu)", va, size);
    ucm_dispatch_mem_type_free(va, size, UCS_MEMORY_TYPE_ROCM);

    status = ucm_orig_hsa_amd_vmem_unmap(va, size);

    ucm_event_leave();
    return status;
}
#endif

static ucm_hook_func_t ucm_rocm_funcs[] = {
    UCM_HOOK_FUNC_ENTRY(hsa_amd_memory_pool_allocate),
    UCM_HOOK_FUNC_ENTRY(hsa_amd_memory_pool_free),
#ifdef HAVE_ROCM_VMM_TYPE
    UCM_HOOK_FUNC_ENTRY(hsa_amd_vmem_map),
    UCM_HOOK_FUNC_ENTRY(hsa_amd_vmem_unmap),
#endif
    {{NULL, NULL}, NULL}
};

static ucs_status_t ucm_rocmmem_install(int events)
{
    static int installed_hooks           = 0;
    static pthread_mutex_t install_mutex = PTHREAD_MUTEX_INITIALIZER;
    ucs_status_t status                  = UCS_OK;
    ucs_status_t bistro_status, reloc_status;

    if (!(events & (UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE))) {
        goto out;
    }

    if (!(ucm_global_opts.rocm_hook_modes & UCM_GPU_HOOK_MODES_MASK)) {
        ucm_info("rocm memory hooks are disabled by configuration");
        status = UCS_ERR_UNSUPPORTED;
        goto out;
    }

    pthread_mutex_lock(&install_mutex);

    /* Install bistro first: it patches the HSA function body, so it catches
     * callers regardless of how they resolved the symbol (GOT or dlsym). If
     * bistro cannot patch (e.g. an unrelocatable prologue or W^X) */
    bistro_status = ucm_install_hooks(ucm_rocm_funcs, "rocm",
                                      UCM_MMAP_HOOK_BISTRO,
                                      ucm_global_opts.rocm_hook_modes,
                                      &installed_hooks);
    if (bistro_status != UCS_OK) {
        ucm_debug("failed to install rocm bistro hooks");
    }

    /* Install reloc as well: it is harmless (the wrapper calls the bistro
     * trampoline via ucm_orig_*, which bypasses the patch, so no double
     * dispatch) and provides coverage where bistro is not installed. */
    reloc_status = ucm_install_hooks(ucm_rocm_funcs, "rocm",
                                     UCM_MMAP_HOOK_RELOC,
                                     ucm_global_opts.rocm_hook_modes,
                                     &installed_hooks);
    if (reloc_status != UCS_OK) {
        ucm_debug("failed to install rocm reloc hooks");
    }

    /* Success as long as at least one hooking mode was installed. */
    if (installed_hooks & UCM_GPU_HOOK_MODES_MASK) {
        status = UCS_OK;
        if ((bistro_status != UCS_OK) || (reloc_status != UCS_OK)) {
            ucm_warn("rocm hooks installed with one mechanism only (bistro: %d, "
                     "reloc: %d)", bistro_status, reloc_status);
        } else {
            ucm_info("rocm hooks are ready");
        }
    } else {
        status = (bistro_status != UCS_OK) ? bistro_status : reloc_status;
        ucm_diag("failed to install rocm hooks: status %d", status);
    }

    pthread_mutex_unlock(&install_mutex);
out:
    return status;
}

static ucm_event_installer_t ucm_rocm_initializer = {
    .install      = ucm_rocmmem_install,
    .path_pattern = "/dev/dri"
};

UCS_STATIC_INIT {
    ucs_list_add_tail(&ucm_event_installer_list, &ucm_rocm_initializer.list);
}

UCS_STATIC_CLEANUP {
    ucs_list_del(&ucm_rocm_initializer.list);
}
