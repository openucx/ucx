/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucm/rocm/rocmmem.h>

#include <ucm/event/event.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucm/util/replace.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/preprocessor.h>

#include <unistd.h>
#include <pthread.h>

UCM_DEFINE_REPLACE_DLSYM_FUNC(hsa_amd_memory_pool_allocate, hsa_status_t,
                              HSA_STATUS_ERROR, hsa_amd_memory_pool_t,
                              size_t, uint32_t, void**)
UCM_DEFINE_REPLACE_DLSYM_FUNC(hsa_amd_memory_pool_free, hsa_status_t,
                              HSA_STATUS_ERROR, void*)

#if ENABLE_SYMBOL_OVERRIDE
UCM_OVERRIDE_FUNC(hsa_amd_memory_pool_allocate, hsa_status_t)
UCM_OVERRIDE_FUNC(hsa_amd_memory_pool_free, hsa_status_t)
#endif

static UCS_F_ALWAYS_INLINE void
ucm_dispatch_mem_type_alloc(void *addr, size_t length, ucm_mem_type_t mem_type)
{
    ucm_event_t event;

    event.mem_type.address  = addr;
    event.mem_type.size     = length;
    event.mem_type.mem_type = mem_type;
    ucm_event_dispatch(UCM_EVENT_MEM_TYPE_ALLOC, &event);
}

static UCS_F_ALWAYS_INLINE void
ucm_dispatch_mem_type_free(void *addr, size_t length, ucm_mem_type_t mem_type)
{
    ucm_event_t event;

    event.mem_type.address  = addr;
    event.mem_type.size     = length;
    event.mem_type.mem_type = mem_type;
    ucm_event_dispatch(UCM_EVENT_MEM_TYPE_FREE, &event);
}

static void ucm_hsa_amd_memory_pool_free_dispatch_events(void *ptr)
{
    size_t size;
    hsa_status_t status;
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

    ucm_dispatch_mem_type_free(ptr, size, UCM_MEM_TYPE_ROCM);
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
        ucm_dispatch_mem_type_alloc(*ptr, size, UCM_MEM_TYPE_ROCM);
    }

    ucm_event_leave();
    return status;
}

static ucm_reloc_patch_t patches[] = {
    {UCS_PP_MAKE_STRING(hsa_amd_memory_pool_allocate),
     ucm_override_hsa_amd_memory_pool_allocate},
    {UCS_PP_MAKE_STRING(hsa_amd_memory_pool_free),
     ucm_override_hsa_amd_memory_pool_free},
    {NULL, NULL}
};

static ucs_status_t ucm_rocmmem_install(int events)
{
    static int ucm_rocmmem_installed = 0;
    static pthread_mutex_t install_mutex = PTHREAD_MUTEX_INITIALIZER;
    ucm_reloc_patch_t *patch;
    ucs_status_t status = UCS_OK;

    if (!(events & (UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE))) {
        goto out;
    }

    /* TODO: check mem reloc */

    pthread_mutex_lock(&install_mutex);

    if (ucm_rocmmem_installed) {
        goto out_unlock;
    }

    for (patch = patches; patch->symbol != NULL; ++patch) {
        status = ucm_reloc_modify(patch);
        if (status != UCS_OK) {
            ucm_warn("failed to install relocation table entry for '%s'", patch->symbol);
            goto out_unlock;
        }
    }

    ucm_debug("rocm hooks are ready");
    ucm_rocmmem_installed = 1;

out_unlock:
    pthread_mutex_unlock(&install_mutex);
out:
    return status;
}

UCS_STATIC_INIT {
    static ucm_event_installer_t rocm_initializer = {
        .func = ucm_rocmmem_install
    };
    ucs_list_add_tail(&ucm_event_installer_list, &rocm_initializer.list);
}
