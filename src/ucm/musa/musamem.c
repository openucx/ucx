/*
 * Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <musa.h>
#include <musa_runtime_api.h>

#include <ucm/event/event.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucm/util/replace.h>
#include <ucm/util/sys.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/preprocessor.h>

#include <pthread.h>
#include <string.h>
#include <sys/mman.h>

static musaError_t ucm_musaMalloc(void **address, size_t size);
static musaError_t ucm_musaFree(void *address);

UCM_DEFINE_REPLACE_DLSYM_FUNC(musaMalloc, musaError_t, musaErrorUnknown,
                              void **, size_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(musaFree, musaError_t, musaErrorUnknown, void *)

static void
ucm_dispatch_musa_mem_type_event(ucm_event_type_t event_type, void *address,
                                 size_t length)
{
    ucm_event_t event;

    if (address == NULL) {
        return;
    }

    event.mem_type.address   = address;
    event.mem_type.size      = length;
    event.mem_type.mem_type  = UCS_MEMORY_TYPE_MUSA;
    event.mem_type.sys_dev   = UCS_SYS_DEVICE_ID_UNKNOWN;
    event.mem_type.mem_flags = UCS_MEM_FLAG_REGISTRABLE;
    ucm_event_dispatch(event_type, &event);
}

static size_t ucm_musa_mem_size(void *address)
{
    MUdeviceptr base;
    size_t length;

    if ((address == NULL) ||
        (muMemGetAddressRange(&base, &length, (MUdeviceptr)address) !=
         MUSA_SUCCESS)) {
        return 1;
    }

    return length;
}

static musaError_t ucm_musaMalloc(void **address, size_t size)
{
    musaError_t status;

    ucm_event_enter();
    status = ucm_orig_musaMalloc(address, size);
    if ((status == musaSuccess) && (address != NULL) && (*address != NULL)) {
        ucm_dispatch_musa_mem_type_event(UCM_EVENT_MEM_TYPE_ALLOC, *address,
                                         size);
    }
    ucm_event_leave();
    return status;
}

static musaError_t ucm_musaFree(void *address)
{
    musaError_t status;
    size_t length;

    ucm_event_enter();
    length = ucm_musa_mem_size(address);
    ucm_dispatch_musa_mem_type_event(UCM_EVENT_MEM_TYPE_FREE, address, length);
    status = ucm_orig_musaFree(address);
    ucm_event_leave();
    return status;
}

static ucm_reloc_patch_t ucm_musa_patches[] = {
    {UCS_PP_MAKE_STRING(musaMalloc), ucm_override_musaMalloc},
    {UCS_PP_MAKE_STRING(musaFree), ucm_override_musaFree},
    {NULL, NULL}
};

static ucs_status_t ucm_musamem_install(int events)
{
    static int installed;
    static pthread_mutex_t install_mutex = PTHREAD_MUTEX_INITIALIZER;
    ucm_reloc_patch_t *patch;
    ucs_status_t status = UCS_OK;

    if (!(events & (UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE))) {
        return UCS_OK;
    }

    pthread_mutex_lock(&install_mutex);
    if (installed) {
        goto out;
    }

    for (patch = ucm_musa_patches; patch->symbol != NULL; ++patch) {
        status = ucm_reloc_modify(patch);
        if (status != UCS_OK) {
            ucm_warn("failed to install MUSA hook for '%s'", patch->symbol);
            goto out;
        }
    }

    installed = 1;
    ucm_info("MUSA memory hooks are ready");

out:
    pthread_mutex_unlock(&install_mutex);
    return status;
}

static int
ucm_musamem_scan_regions_cb(void *arg, void *addr, size_t length, int prot,
                            const char *path)
{
    static const char musa_path_pattern[] = "/dev/musa";
    ucm_event_handler_t *handler = arg;
    ucm_event_t event;

    if ((prot & (PROT_READ | PROT_WRITE | PROT_EXEC)) &&
        strncmp(path, musa_path_pattern, sizeof(musa_path_pattern) - 1)) {
        return 0;
    }

    event.mem_type.address   = addr;
    event.mem_type.size      = length;
    event.mem_type.mem_type  = UCS_MEMORY_TYPE_LAST;
    event.mem_type.sys_dev   = UCS_SYS_DEVICE_ID_UNKNOWN;
    event.mem_type.mem_flags = UCS_MEM_FLAG_REGISTRABLE;

    ucm_event_enter();
    handler->cb(UCM_EVENT_MEM_TYPE_ALLOC, &event, handler->arg);
    ucm_event_leave();
    return 0;
}

static void ucm_musamem_get_existing_alloc(ucm_event_handler_t *handler)
{
    if (handler->events & UCM_EVENT_MEM_TYPE_ALLOC) {
        ucm_parse_proc_self_maps(ucm_musamem_scan_regions_cb, handler);
    }
}

static ucm_event_installer_t ucm_musa_initializer = {
    .install            = ucm_musamem_install,
    .get_existing_alloc = ucm_musamem_get_existing_alloc
};

UCS_STATIC_INIT
{
    ucs_list_add_tail(&ucm_event_installer_list, &ucm_musa_initializer.list);
}

UCS_STATIC_CLEANUP
{
    ucs_list_del(&ucm_musa_initializer.list);
}
