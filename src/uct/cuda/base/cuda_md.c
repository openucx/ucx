/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_md.h"
#include "cuda_iface.h"

#include <ucs/sys/module.h>
#include <ucs/profile/profile.h>
#include <ucs/debug/log.h>
#include <cuda_runtime.h>
#include <cuda.h>

static ucs_status_t uct_cuda_base_get_sys_dev(ucs_sys_device_t *sys_dev_p)
{
    CUdevice dev;
    int attrib;

    if (UCS_OK != UCT_CUDADRV_FUNC(cuCtxGetDevice(&dev))) {
        return UCS_ERR_IO_ERROR;
    }

    if (UCS_OK != UCT_CUDADRV_FUNC(cuDeviceGetAttribute(&attrib,
                    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev))) {
        return UCS_ERR_IO_ERROR;
    }
    sys_dev_p->bus_id.domain = (uint16_t) attrib;

    if (UCS_OK != UCT_CUDADRV_FUNC(cuDeviceGetAttribute(&attrib,
                    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev))) {
        return UCS_ERR_IO_ERROR;
    }
    sys_dev_p->bus_id.bus = (uint8_t) attrib;

    return UCS_OK;

}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_base_detect_memory_type,
                 (md, addr, length, mem_type_p, sys_dev_p),
                 uct_md_h md, const void *addr, size_t length,
                 ucs_memory_type_t *mem_type_p,
                 ucs_sys_device_t *sys_dev_p)
{
    CUmemorytype memType = (CUmemorytype)0;
    uint32_t isManaged   = 0;
    unsigned value       = 1;
    void *attrdata[] = {(void *)&memType, (void *)&isManaged};
    CUpointer_attribute attributes[2] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                         CU_POINTER_ATTRIBUTE_IS_MANAGED};
    CUresult cu_err;
    const char *cu_err_str;

    if (addr == NULL) {
        *mem_type_p = UCS_MEMORY_TYPE_HOST;
        return UCS_OK;
    }

    cu_err = cuPointerGetAttributes(2, attributes, attrdata, (CUdeviceptr)addr);
    if ((cu_err == CUDA_SUCCESS) && (memType == CU_MEMORYTYPE_DEVICE)) {
        if (isManaged) {
            *mem_type_p = UCS_MEMORY_TYPE_CUDA_MANAGED;
        } else {
            *mem_type_p = UCS_MEMORY_TYPE_CUDA;
            cu_err = cuPointerSetAttribute(&value, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                           (CUdeviceptr)addr);
            if (cu_err != CUDA_SUCCESS) {
                cuGetErrorString(cu_err, &cu_err_str);
                ucs_warn("cuPointerSetAttribute(%p) error: %s", (void*) addr, cu_err_str);
            }

            return uct_cuda_base_get_sys_dev(sys_dev_p);
        }
        return UCS_OK;
    }

    return UCS_ERR_INVALID_ADDR;
}

ucs_status_t
uct_cuda_base_query_md_resources(uct_component_t *component,
                                 uct_md_resource_desc_t **resources_p,
                                 unsigned *num_resources_p)
{
    cudaError_t cudaErr;
    int num_gpus;

    cudaErr = cudaGetDeviceCount(&num_gpus);
    if ((cudaErr != cudaSuccess) || (num_gpus == 0)) {
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
    }

    return uct_md_query_single_md_resource(component, resources_p,
                                           num_resources_p);
}

UCS_MODULE_INIT() {
    /* TODO make gdrcopy independent of cuda */
    UCS_MODULE_FRAMEWORK_DECLARE(uct_cuda);
    UCS_MODULE_FRAMEWORK_LOAD(uct_cuda, 0);
    return UCS_OK;
}
