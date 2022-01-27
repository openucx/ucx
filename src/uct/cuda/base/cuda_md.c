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
#include <ucs/sys/string.h>
#include <ucs/memory/memtype_cache.h>
#include <ucs/type/spinlock.h>
#include <ucs/profile/profile.h>
#include <ucs/debug/log.h>
#include <uct/cuda/cuda_copy/cuda_copy_md.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define UCT_CUDA_DEV_NAME_MAX_LEN 64
#define UCT_CUDA_MAX_DEVICES      32

ucs_spinlock_t uct_cuda_base_lock;


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

static size_t
uct_cuda_base_get_total_device_mem(CUdevice cuda_device)
{
    static size_t total_bytes[UCT_CUDA_MAX_DEVICES];
    char dev_name[UCT_CUDA_DEV_NAME_MAX_LEN];
    CUresult cu_err;
    const char *cu_err_str;

    ucs_assert(cuda_device < UCT_CUDA_MAX_DEVICES);

    ucs_spin_lock(&uct_cuda_base_lock);

    if (!total_bytes[cuda_device]) {
        cu_err = cuDeviceTotalMem(&total_bytes[cuda_device], cuda_device);
        if (cu_err != CUDA_SUCCESS) {
            cuGetErrorString(cu_err, &cu_err_str);
            ucs_error("cuDeviceTotalMem error: %s", cu_err_str);
            goto err;
        }

        cu_err = cuDeviceGetName(dev_name, sizeof(dev_name), cuda_device);
        if (cu_err != CUDA_SUCCESS) {
            cuGetErrorString(cu_err, &cu_err_str);
            ucs_error("cuDeviceGetName error: %s", cu_err_str);
            goto err;
        }

        if (!strncmp(dev_name, "T4", 2)) {
            total_bytes[cuda_device] = 1; /* should ensure that whole alloc
                                             registration is not used for t4 */
        }
    }

    ucs_spin_unlock(&uct_cuda_base_lock);
    return total_bytes[cuda_device];

err:
    ucs_spin_unlock(&uct_cuda_base_lock);
    return 1; /* return 1 byte to avoid division by zero */
}

static ucs_status_t
uct_cuda_base_query_attributes(uct_cuda_copy_md_t *md, const void *address,
                               size_t length, ucs_memory_info_t *mem_info)
{
#define UCT_CUDA_MEM_QUERY_NUM_ATTRS 3
    CUmemorytype cuda_mem_mype = (CUmemorytype)0;
    uint32_t is_managed        = 0;
    CUdevice cuda_device       = -1;
    CUpointer_attribute attr_type[UCT_CUDA_MEM_QUERY_NUM_ATTRS];
    void *attr_data[UCT_CUDA_MEM_QUERY_NUM_ATTRS];
    const char *cu_err_str;
    CUdeviceptr base_address;
    size_t alloc_length;
    ucs_status_t status;
    size_t total_bytes;
    CUresult cu_err;

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &cuda_mem_mype;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
    attr_data[2] = &cuda_device;

    cu_err = cuPointerGetAttributes(ucs_static_array_size(attr_data), attr_type,
                                    attr_data, (CUdeviceptr)address);
    if ((cu_err != CUDA_SUCCESS) || (cuda_mem_mype != CU_MEMORYTYPE_DEVICE)) {
        /* pointer not recognized */
        return UCS_ERR_INVALID_ADDR;
    }

    status = uct_cuda_base_get_sys_dev(cuda_device, &mem_info->sys_dev);
    if (status != UCS_OK) {
        return status;
    }

    if (is_managed) {
        /* cuMemGetAddress range does not support managed memory so use provided
         * address and length as base address and alloc length respectively */
        mem_info->type = UCS_MEMORY_TYPE_CUDA_MANAGED;
        goto out_default_range;
    }

    mem_info->type = UCS_MEMORY_TYPE_CUDA;

    /* Extending the registration range is disable by configuration */
    if (md->config.alloc_whole_reg == UCS_CONFIG_OFF) {
        goto out_default_range;
    }

    cu_err = cuMemGetAddressRange(&base_address, &alloc_length,
                                  (CUdeviceptr)address);
    if (cu_err != CUDA_SUCCESS) {
        cuGetErrorString(cu_err, &cu_err_str);
        ucs_error("cuMemGetAddressRange(%p) error: %s", address, cu_err_str);
        return UCS_ERR_INVALID_ADDR;
    }

    ucs_trace("query address %p: 0x%llx..0x%llx length %zu", address,
              base_address, base_address + alloc_length, alloc_length);

    if (md->config.alloc_whole_reg == UCS_CONFIG_AUTO) {
        total_bytes = uct_cuda_base_get_total_device_mem(cuda_device);
        if (alloc_length > (total_bytes * md->config.max_reg_ratio)) {
            goto out_default_range;
        }
    } else {
        ucs_assert(md->config.alloc_whole_reg == UCS_CONFIG_ON);
    }

    mem_info->base_address = (void*)base_address;
    mem_info->alloc_length = alloc_length;
    return UCS_OK;

out_default_range:
    mem_info->base_address = (void*)address;
    mem_info->alloc_length = length;
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_base_detect_memory_type,
                 (md, address, length, mem_type_p),
                 uct_md_h md, const void *address, size_t length,
                 ucs_memory_type_t *mem_type_p)
{
    uct_md_mem_attr_t mem_attr;
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

ucs_status_t uct_cuda_base_mem_query(uct_md_h tl_md, const void *address,
                                     size_t length, uct_md_mem_attr_t *mem_attr)
{
    ucs_memory_info_t default_mem_info = {
        .type              = UCS_MEMORY_TYPE_HOST,
        .sys_dev           = UCS_SYS_DEVICE_ID_UNKNOWN,
        .base_address      = (void*)address,
        .alloc_length      = length
    };
    uct_cuda_copy_md_t *md = ucs_derived_of(tl_md, uct_cuda_copy_md_t);
    unsigned value         = 1;
    ucs_memory_info_t addr_mem_info;
    const char *cu_err_str;
    ucs_status_t status;
    CUresult cu_err;

    if (!(mem_attr->field_mask & (UCT_MD_MEM_ATTR_FIELD_MEM_TYPE     |
                                  UCT_MD_MEM_ATTR_FIELD_SYS_DEV      |
                                  UCT_MD_MEM_ATTR_FIELD_BASE_ADDRESS |
                                  UCT_MD_MEM_ATTR_FIELD_ALLOC_LENGTH))) {
        return UCS_OK;
    }

    if (address != NULL) {
        status = uct_cuda_base_query_attributes(md, address, length,
                                                &addr_mem_info);
        if (status != UCS_OK) {
            return status;
        }

        /* Synchronize for DMA */
        cu_err = cuPointerSetAttribute(&value, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                       (CUdeviceptr)address);
        if (cu_err != CUDA_SUCCESS) {
            cuGetErrorString(cu_err, &cu_err_str);
            ucs_warn("cuPointerSetAttribute(%p, SYNC_MEMOPS) error: %s",
                     address, cu_err_str);
        }

        ucs_memtype_cache_update(addr_mem_info.base_address,
                                 addr_mem_info.alloc_length, addr_mem_info.type,
                                 addr_mem_info.sys_dev);
    } else {
        addr_mem_info = default_mem_info;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_MEM_TYPE) {
        mem_attr->mem_type = addr_mem_info.type;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_SYS_DEV) {
        mem_attr->sys_dev = addr_mem_info.sys_dev;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_BASE_ADDRESS) {
        mem_attr->base_address = addr_mem_info.base_address;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_ALLOC_LENGTH) {
        mem_attr->alloc_length = addr_mem_info.alloc_length;
    }

    return UCS_OK;
}

ucs_status_t
uct_cuda_base_query_md_resources(uct_component_t *component,
                                 uct_md_resource_desc_t **resources_p,
                                 unsigned *num_resources_p)
{
    ucs_sys_device_t sys_dev;
    CUdevice cuda_device;
    cudaError_t cudaErr;
    ucs_status_t status;
    char device_name[10];
    int num_gpus;

    cudaErr = cudaGetDeviceCount(&num_gpus);
    if ((cudaErr != cudaSuccess) || (num_gpus == 0)) {
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
    }

    for (cuda_device = 0; cuda_device < num_gpus; ++cuda_device) {
        status = uct_cuda_base_get_sys_dev(cuda_device, &sys_dev);
        if (status == UCS_OK) {
            ucs_snprintf_safe(device_name, sizeof(device_name), "GPU%d",
                              cuda_device);
            status = ucs_topo_sys_device_set_name(sys_dev, device_name);
            ucs_assert_always(status == UCS_OK);
        }
    }

    return uct_md_query_single_md_resource(component, resources_p,
                                           num_resources_p);
}

UCS_STATIC_INIT {
    ucs_spinlock_init(&uct_cuda_base_lock, 0);
}

UCS_STATIC_CLEANUP {
    ucs_spinlock_destroy(&uct_cuda_base_lock);
}

UCS_MODULE_INIT() {
    /* TODO make gdrcopy independent of cuda */
    UCS_MODULE_FRAMEWORK_DECLARE(uct_cuda);
    UCS_MODULE_FRAMEWORK_LOAD(uct_cuda, 0);
    return UCS_OK;
}
