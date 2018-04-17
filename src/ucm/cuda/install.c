/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cudamem.h"

#include <ucm/api/ucm.h>
#include <ucm/event/event.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucs/sys/math.h>
#include <ucs/sys/preprocessor.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <pthread.h>

static ucm_reloc_patch_t patches[] = {
    {UCS_PP_MAKE_STRING(cuMemFree),                 ucm_override_cuMemFree},
    {UCS_PP_MAKE_STRING(cuMemFreeHost),             ucm_override_cuMemFreeHost},
    {UCS_PP_MAKE_STRING(cuMemAlloc),                ucm_override_cuMemAlloc},
    {UCS_PP_MAKE_STRING(cuMemAllocManaged),         ucm_override_cuMemAllocManaged},
    {UCS_PP_MAKE_STRING(cuMemAllocPitch),           ucm_override_cuMemAllocPitch},
    {UCS_PP_MAKE_STRING(cuMemHostGetDevicePointer), ucm_override_cuMemHostGetDevicePointer},
    {UCS_PP_MAKE_STRING(cuMemHostUnregister),       ucm_override_cuMemHostUnregister},
    {UCS_PP_MAKE_STRING(cudaFree),                  ucm_override_cudaFree},
    {UCS_PP_MAKE_STRING(cudaFreeHost),              ucm_override_cudaFreeHost},
    {UCS_PP_MAKE_STRING(cudaMalloc),                ucm_override_cudaMalloc},
    {UCS_PP_MAKE_STRING(cudaMallocManaged),         ucm_override_cudaMallocManaged},
    {UCS_PP_MAKE_STRING(cudaMallocPitch),           ucm_override_cudaMallocPitch},
    {UCS_PP_MAKE_STRING(cudaHostGetDevicePointer),  ucm_override_cudaHostGetDevicePointer},
    {UCS_PP_MAKE_STRING(cudaHostUnregister),        ucm_override_cudaHostUnregister},
    {NULL,                                          NULL}
};

ucs_status_t ucm_cudamem_install()
{
    static int ucm_cudamem_installed = 0;
    static pthread_mutex_t install_mutex = PTHREAD_MUTEX_INITIALIZER;
    ucm_reloc_patch_t *patch;
    ucs_status_t status;

    if (!ucm_global_opts.enable_cuda_reloc) {
        ucm_debug("installing cudamem relocations is disabled by configuration");
        return UCS_ERR_UNSUPPORTED;
    }
    if (ucm_cudamem_installed) {
        return UCS_OK;
    }

    pthread_mutex_lock(&install_mutex);

    for (patch = patches; patch->symbol != NULL; ++patch) {
        status = ucm_reloc_modify(patch);
        if (status != UCS_OK) {
            ucm_warn("failed to install relocation table entry for '%s'", patch->symbol);
            goto out_unlock;
        }
    }

    ucm_cudamem_installed = 1;

    status = UCS_OK;
out_unlock:
    pthread_mutex_unlock(&install_mutex);
    return status;
}
