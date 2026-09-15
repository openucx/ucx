/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
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
#include <ucm/util/sys.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/preprocessor.h>

#include <sys/mman.h>

#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

/* Use the PTR variant so that ucm_orig_<fn> is a function pointer that bistro
 * can redirect to the relocated (trampoline) original, allowing us to intercept
 * callers that resolve the HSA symbol via dlopen/dlsym, which the reloc/GOT 
 * hook cannot see. */
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(hsa_amd_memory_pool_allocate, hsa_status_t,
                                  HSA_STATUS_ERROR, hsa_amd_memory_pool_t,
                                  size_t, uint32_t, void**)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(hsa_amd_memory_pool_free, hsa_status_t,
                                  HSA_STATUS_ERROR, void*)

static UCS_F_ALWAYS_INLINE void
ucm_dispatch_mem_type_alloc(void *addr, size_t length, ucs_memory_type_t mem_type)
{
    ucm_event_t event;

    event.mem_type.address   = addr;
    event.mem_type.size      = length;
    event.mem_type.mem_type  = mem_type;
    event.mem_type.sys_dev   = UCS_SYS_DEVICE_ID_UNKNOWN;
    event.mem_type.mem_flags = UCS_MEM_FLAG_REGISTRABLE;
    ucm_event_dispatch(UCM_EVENT_MEM_TYPE_ALLOC, &event);
}

static UCS_F_ALWAYS_INLINE void
ucm_dispatch_mem_type_free(void *addr, size_t length, ucs_memory_type_t mem_type)
{
    ucm_event_t event;

    event.mem_type.address   = addr;
    event.mem_type.size      = length;
    event.mem_type.mem_type  = mem_type;
    event.mem_type.sys_dev   = UCS_SYS_DEVICE_ID_UNKNOWN;
    event.mem_type.mem_flags = UCS_MEM_FLAG_REGISTRABLE;
    ucm_event_dispatch(UCM_EVENT_MEM_TYPE_FREE, &event);
}

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
        ucm_dispatch_mem_type_alloc(*ptr, size, UCS_MEMORY_TYPE_UNKNOWN);
    }

    ucm_event_leave();
    return status;
}

#define UCM_ROCM_FUNC_ENTRY(_func) \
    { \
        {UCS_PP_MAKE_STRING(_func), ucm_override_##_func}, \
        (void**)&ucm_orig_##_func \
    }

typedef struct {
    ucm_reloc_patch_t patch;
    void              **orig_func_ptr;
} ucm_rocm_func_t;

static ucm_rocm_func_t ucm_rocm_funcs[] = {
    UCM_ROCM_FUNC_ENTRY(hsa_amd_memory_pool_allocate),
    UCM_ROCM_FUNC_ENTRY(hsa_amd_memory_pool_free),
    {{NULL, NULL}, NULL}
};

static ucs_status_t
ucm_rocmmem_install_hooks(ucm_mmap_hook_mode_t mode, int *installed_hooks_p)
{
    ucm_rocm_func_t *func;
    ucs_status_t status;
    void *func_ptr;
    int count;

    if (*installed_hooks_p & UCS_BIT(mode)) {
        return UCS_OK;
    }

    if (!(ucm_global_opts.rocm_hook_modes & UCS_BIT(mode))) {
        /* Disabled by configuration */
        ucm_debug("rocm memory hooks mode %s is disabled",
                  ucm_mmap_hook_modes[mode]);
        return UCS_OK;
    }

    count = 0;
    for (func = ucm_rocm_funcs; func->patch.symbol != NULL; ++func) {
        func_ptr = ucm_reloc_get_orig(func->patch.symbol, func->patch.value);
        if (func_ptr == NULL) {
            /* Symbol not (yet) loaded - e.g. libhsa-runtime64 not mapped */
            continue;
        }

        if (mode == UCM_MMAP_HOOK_BISTRO) {
            status = ucm_bistro_patch(func_ptr, func->patch.value,
                                      func->patch.symbol, func->orig_func_ptr,
                                      NULL);
        } else if (mode == UCM_MMAP_HOOK_RELOC) {
            status = ucm_reloc_modify(&func->patch);
        } else {
            break;
        }

        if (status != UCS_OK) {
            ucm_diag("failed to install %s hook for '%s'",
                     ucm_mmap_hook_modes[mode], func->patch.symbol);
            return status;
        }

        ucm_debug("installed %s hook for '%s'", ucm_mmap_hook_modes[mode],
                  func->patch.symbol);
        ++count;
    }

    *installed_hooks_p |= UCS_BIT(mode);
    ucm_info("rocm memory hooks mode %s: installed %d hooks",
             ucm_mmap_hook_modes[mode], count);
    return UCS_OK;
}

static ucs_status_t ucm_rocmmem_install(int events)
{
    static int installed_hooks           = 0;
    static pthread_mutex_t install_mutex = PTHREAD_MUTEX_INITIALIZER;
    ucs_status_t status                  = UCS_OK;

    if (!(events & (UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE))) {
        goto out;
    }

    pthread_mutex_lock(&install_mutex);

    /* Install bistro first: it patches the HSA function body, so it catches
     * callers regardless of how they resolved the symbol (GOT or dlsym). Then
     * install reloc as well: it is harmless (the wrapper calls the bistro
     * trampoline via ucm_orig_*, which bypasses the patch, so no double
     * dispatch) and provides coverage where bistro cannot patch (e.g. W^X). */
    status = ucm_rocmmem_install_hooks(UCM_MMAP_HOOK_BISTRO, &installed_hooks);
    if (status != UCS_OK) {
        ucm_debug("failed to install rocm bistro hooks");
        goto out_unlock;
    }

    status = ucm_rocmmem_install_hooks(UCM_MMAP_HOOK_RELOC, &installed_hooks);
    if (status != UCS_OK) {
        ucm_debug("failed to install rocm reloc hooks");
        goto out_unlock;
    }

    ucm_info("rocm hooks are ready");

out_unlock:
    pthread_mutex_unlock(&install_mutex);
out:
    return status;
}

static int ucm_rocm_scan_regions_cb(void *arg, void *addr, size_t length,
                                    int prot, const char *path)
{
    static const char rocm_path_pattern[] = "/dev/dri";
    ucm_event_handler_t *handler          = arg;
    ucm_event_t event;

    if ((prot & (PROT_READ | PROT_WRITE | PROT_EXEC)) &&
        strncmp(path, rocm_path_pattern, sizeof(rocm_path_pattern) - 1)) {
        return 0;
    }
    ucm_debug("dispatching initial memtype allocation for %p..%p %s", addr,
              UCS_PTR_BYTE_OFFSET(addr, length), path);

    event.mem_type.address   = addr;
    event.mem_type.size      = length;
    event.mem_type.mem_type  = UCS_MEMORY_TYPE_LAST; /* unknown memory type */
    event.mem_type.sys_dev   = UCS_SYS_DEVICE_ID_UNKNOWN;
    event.mem_type.mem_flags = UCS_MEM_FLAG_REGISTRABLE;

    ucm_event_enter();
    handler->cb(UCM_EVENT_MEM_TYPE_ALLOC, &event, handler->arg);
    ucm_event_leave();

    return 0;
}

static void ucm_rocmmem_get_existing_alloc(ucm_event_handler_t *handler)
{
    if (handler->events & UCM_EVENT_MEM_TYPE_ALLOC) {
        ucm_parse_proc_self_maps(ucm_rocm_scan_regions_cb, handler);
    }
}

static ucm_event_installer_t ucm_rocm_initializer = {
    .install            = ucm_rocmmem_install,
    .get_existing_alloc = ucm_rocmmem_get_existing_alloc
};

UCS_STATIC_INIT {
    ucs_list_add_tail(&ucm_event_installer_list, &ucm_rocm_initializer.list);
}

UCS_STATIC_CLEANUP {
    ucs_list_del(&ucm_rocm_initializer.list);
}
