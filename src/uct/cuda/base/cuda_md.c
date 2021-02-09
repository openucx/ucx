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


typedef struct uct_cuda_mem_attr {
    uct_mem_attr_t super;
    unsigned long long buf_id;
    void *base_address;
    size_t alloc_length;
} uct_cuda_mem_attr_t;

static int uct_cuda_mem_attr_cmp(const uct_mem_attr_h mem_attr1,
                                 const uct_mem_attr_h mem_attr2)
{
    uct_cuda_mem_attr_t *cuda_mem_attr1, *cuda_mem_attr2;
    cuda_mem_attr1 = ucs_derived_of(mem_attr1, uct_cuda_mem_attr_t);
    cuda_mem_attr2 = ucs_derived_of(mem_attr2, uct_cuda_mem_attr_t);
    return cuda_mem_attr1->buf_id == cuda_mem_attr2->buf_id ? 0 : 1;
}

static void uct_cuda_mem_attr_destroy(uct_mem_attr_h mem_attr)
{
    uct_cuda_mem_attr_t *cuda_mem_attr;
    cuda_mem_attr = ucs_derived_of(mem_attr, uct_cuda_mem_attr_t);
    ucs_free(cuda_mem_attr);
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_mem_attr_query,
                 (address, length, mem_attr_p),
                 const void *address, size_t length,
                 uct_mem_attr_h *mem_attr_p)
{
#define UCT_CUDA_MEM_QUERY_NUM_ATTRS 4
    CUmemorytype cuda_mem_mype = (CUmemorytype)0;
    unsigned long long buf_id  = 0;
    uint32_t is_managed        = 0;
    CUdevice cuda_device       = -1;
    void *base_address         = (void*)address;
    size_t alloc_length        = length;
    ucs_sys_device_t sys_dev   =  UCS_SYS_DEVICE_ID_UNKNOWN;
    CUpointer_attribute attr_type[UCT_CUDA_MEM_QUERY_NUM_ATTRS];
    void *attr_data[UCT_CUDA_MEM_QUERY_NUM_ATTRS];
    ucs_memory_type_t mem_type;
    uct_cuda_mem_attr_t *cuda_mem_attr;
    CUresult cu_err;
    const char *cu_err_str;
    ucs_status_t status;

    if (address == NULL) {
        return UCS_ERR_INVALID_ADDR;
    }

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &cuda_mem_mype;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
    attr_data[2] = &cuda_device;
    attr_type[3] = CU_POINTER_ATTRIBUTE_BUFFER_ID;
    attr_data[3] = &buf_id;

    cu_err = cuPointerGetAttributes(ucs_static_array_size(attr_data),
                                    attr_type, attr_data,
                                    (CUdeviceptr)address);
    if ((cu_err != CUDA_SUCCESS) || (cuda_mem_mype != CU_MEMORYTYPE_DEVICE)) {
        /* pointer not recognized */
        return UCS_ERR_INVALID_ADDR;
    }

    if (is_managed) {
        mem_type = UCS_MEMORY_TYPE_CUDA_MANAGED;
    } else {
        mem_type = UCS_MEMORY_TYPE_CUDA;
    }

    status = uct_cuda_base_get_sys_dev(cuda_device, &sys_dev);
    if (status != UCS_OK) {
        return status;
    }

    cu_err = cuMemGetAddressRange((CUdeviceptr*)&base_address,
                                  &alloc_length, (CUdeviceptr)address);
    if (cu_err != CUDA_SUCCESS) {
        cuGetErrorString(cu_err, &cu_err_str);
        ucs_error("ccuMemGetAddressRange(%p) error: %s", address,
                  cu_err_str);
        return UCS_ERR_INVALID_ADDR;
    }

    cuda_mem_attr = ucs_malloc(sizeof(*cuda_mem_attr), "cuda_mem_attr");
    if (cuda_mem_attr == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    cuda_mem_attr->buf_id             = buf_id;
    cuda_mem_attr->base_address       = base_address;
    cuda_mem_attr->alloc_length       = alloc_length;
    cuda_mem_attr->super.mem_type     = mem_type;
    cuda_mem_attr->super.sys_dev      = sys_dev;
    cuda_mem_attr->super.cmp          = uct_cuda_mem_attr_cmp;
    cuda_mem_attr->super.destroy      = uct_cuda_mem_attr_destroy;

    *mem_attr_p = &cuda_mem_attr->super;
    return UCS_OK;
}

ucs_status_t uct_cuda_base_get_sys_dev(CUdevice cuda_device,
                                       ucs_sys_device_t *sys_dev_p)
{
    ucs_sys_bus_id_t bus_id;
    CUresult cu_err;
    int attrib;

    /* PCI domain id */
    cu_err = cuDeviceGetAttribute(&attrib, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                                  cuda_device);
    if (cu_err != CUDA_SUCCESS) {
         return UCS_ERR_IO_ERROR;
    }
    bus_id.domain = (uint16_t)attrib;

    /* PCI bus id */
    cu_err = cuDeviceGetAttribute(&attrib, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                  cuda_device);
    if (cu_err != CUDA_SUCCESS) {
         return UCS_ERR_IO_ERROR;
    }
    bus_id.bus = (uint8_t)attrib;

    /* PCI slot id */
    cu_err = cuDeviceGetAttribute(&attrib, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                                  cuda_device);
    if (cu_err != CUDA_SUCCESS) {
         return UCS_ERR_IO_ERROR;
    }
    bus_id.slot = (uint8_t)attrib;

    /* Function - always 0 */
    bus_id.function = 0;

    return ucs_topo_find_device_by_bus_id(&bus_id, sys_dev_p);
}


UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_base_detect_memory_type,
                 (md, address, length, mem_type_p),
                 uct_md_h md, const void *address, size_t length,
                 ucs_memory_type_t *mem_type_p)
{
    /* self-initializing to suppress wrong maybe-uninitialized error */
    uct_md_mem_attr_t mem_attr = mem_attr;
    ucs_status_t status;

    mem_attr.field_mask = UCT_MD_MEM_ATTR_FIELD_MEM_TYPE;
    status              = uct_cuda_base_mem_query(md, address, length,
                                                  &mem_attr);
    if (status != UCS_OK) {
        return status;
    }

    *mem_type_p = mem_attr.mem_type;
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_base_mem_query,
                 (md, address, length, mem_attr),
                 uct_md_h md, const void *address, size_t length,
                 uct_md_mem_attr_t *mem_attr)
{
    uct_mem_attr_h mem_attr_h;
    uct_cuda_mem_attr_t *cuda_mem_attr;
    ucs_status_t status;

    if (!(mem_attr->field_mask & (UCT_MD_MEM_ATTR_FIELD_MEM_TYPE     |
                                  UCT_MD_MEM_ATTR_FIELD_SYS_DEV      |
                                  UCT_MD_MEM_ATTR_FIELD_BASE_ADDRESS |
                                  UCT_MD_MEM_ATTR_FIELD_ALLOC_LENGTH))) {
        return UCS_OK;
    }

    status = uct_cuda_mem_attr_query(address, length, &mem_attr_h);
    if (status != UCS_OK) {
        return status;
    }

    if (uct_mem_attr_get_type(mem_attr_h) == UCS_MEMORY_TYPE_CUDA) {
        unsigned value = 1;
        CUresult cu_err;
        const char *cu_err_str;
        /* Synchronize for DMA */
        cu_err = cuPointerSetAttribute(&value,
                                       CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                       (CUdeviceptr)address);
        if (cu_err != CUDA_SUCCESS) {
            cuGetErrorString(cu_err, &cu_err_str);
            ucs_warn("cuPointerSetAttribute(%p) error: %s", address,
                     cu_err_str);
        }
    }

    cuda_mem_attr = ucs_derived_of(mem_attr_h, uct_cuda_mem_attr_t);

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_SYS_DEV) {
        mem_attr->sys_dev = cuda_mem_attr->super.sys_dev;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_MEM_TYPE) {
        mem_attr->mem_type = cuda_mem_attr->super.mem_type;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_BASE_ADDRESS) {
        mem_attr->base_address = cuda_mem_attr->base_address;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_ALLOC_LENGTH) {
        mem_attr->alloc_length = cuda_mem_attr->alloc_length;
    }

    uct_mem_attr_destroy(mem_attr_h);
    return UCS_OK;
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

UCT_MEM_QUERY_REGISTER(uct_cuda_mem_attr_query, UCS_MEMORY_TYPE_CUDA);
