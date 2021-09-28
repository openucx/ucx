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
#include <ucs/sys/topo.h>
#include <ucs/profile/profile.h>
#include <ucs/debug/log.h>
#include <cuda_runtime.h>
#include <cuda.h>


static ucs_status_t
uct_cuda_base_get_common_ancestor(nvmlDevice_t device1, nvmlDevice_t device2,
                                  ucs_sys_common_ancestor_t *common_ancestor)
{
    nvmlGpuTopologyLevel_t path_info;
    ucs_status_t status;

    status = 
        UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetTopologyCommonAncestor(device1,
                                                                  device2,
                                                                  &path_info));
    if (status == UCS_OK) {
        switch(path_info) {
            case NVML_TOPOLOGY_INTERNAL:
                *common_ancestor = UCS_SYS_TOPO_COMMON_DEVICE;
                break;
            case NVML_TOPOLOGY_SINGLE:
                *common_ancestor = UCS_SYS_TOPO_COMMON_PCIE_SWITCH;
                break;
            case NVML_TOPOLOGY_MULTIPLE:
                *common_ancestor = UCS_SYS_TOPO_COMMON_PCIE_SWITCH_HIERARCHY;
                break;
            case NVML_TOPOLOGY_HOSTBRIDGE:
                *common_ancestor = UCS_SYS_TOPO_COMMON_HOST_BRIDGE;
                break;
            case NVML_TOPOLOGY_NODE:
                *common_ancestor = UCS_SYS_TOPO_COMMON_NUMA_NODE;
                break;
            case NVML_TOPOLOGY_SYSTEM:
                *common_ancestor = UCS_SYS_TOPO_COMMON_SYSTEM;
                break;
            default:
                status = UCS_ERR_UNSUPPORTED;
                break;
        }
    }

    return status;
}

static unsigned uct_cuda_base_p2p_supported(nvmlDevice_t device1,
                                            nvmlDevice_t device2)
{
    nvmlGpuP2PStatus_t p2p_status;

    UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetP2PStatus(device1, device2,
                                                 NVML_P2P_CAPS_INDEX_NVLINK,
                                                 &p2p_status));

    return (p2p_status == NVML_P2P_STATUS_OK) ? 1 : 0;
}

static double uct_cuda_base_get_nvlink_common_bw(nvmlDevice_t device)
{
    double bw;
    nvmlFieldValue_t value;

    value.fieldId = NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON;
    UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetFieldValues(device, 1, &value));

    bw = ((value.nvmlReturn == NVML_SUCCESS) &&
          (value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)) ?
         value.value.uiVal : 0.0;

    return bw;
}

static unsigned uct_cuda_base_get_nvswitch_num_nvlinks(nvmlDevice_t device)
{
    unsigned num_links;
    nvmlFieldValue_t value;

    value.fieldId = NVML_FI_DEV_NVSWITCH_CONNECTED_LINK_COUNT;
    UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetFieldValues(device, 1, &value));

    num_links = ((value.nvmlReturn == NVML_SUCCESS) &&
                 (value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)) ?
                value.value.uiVal : 0;

    return num_links;
}

static int uct_cuda_base_get_num_nvlinks(nvmlDevice_t device1,
                                         nvmlDevice_t device2)
{
    unsigned nvswitch_links = uct_cuda_base_get_nvswitch_num_nvlinks(device1);
    unsigned total_links    = 0;
    unsigned num_links, link;
    nvmlFieldValue_t value;
    nvmlPciInfo_t pci1, pci2;

    if (nvswitch_links) {
        return nvswitch_links;
    }

    value.fieldId = NVML_FI_DEV_NVLINK_LINK_COUNT;

    UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetFieldValues(device1, 1, &value));

    num_links = ((value.nvmlReturn == NVML_SUCCESS) &&
                 (value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)) ?
                value.value.uiVal : 0;

    UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetPciInfo(device2, &pci2));

    for (link = 0; link < num_links; ++link) {
        UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetNvLinkRemotePciInfo(device1, link,
                                                               &pci1));
        if (!strcmp(pci2.busId, pci1.busId)) {
            total_links++;
        }
    }

    return total_links;
}

static double uct_cuda_base_get_nvlink_bw(nvmlDevice_t device1,
                                          nvmlDevice_t device2)
{
    return (uct_cuda_base_p2p_supported(device1, device2) *
            uct_cuda_base_get_num_nvlinks(device1, device2) *
            uct_cuda_base_get_nvlink_common_bw(device1));
}

static void uct_cuda_base_get_nvml_device_bus_id(nvmlDevice_t *device,
                                                 int *bus_id)
{
    nvmlPciInfo_t pci_info;

    UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetPciInfo(*device, &pci_info));
    *bus_id = pci_info.bus;
}

#if 0
static void uct_cuda_base_get_cuda_device_bus_id(CUdevice device,
                                                 int *bus_id)
{
    UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetAttribute(bus_id,
                                                  CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                                  device));
}
#endif

ucs_status_t
uct_cuda_base_get_nvml_device_from_bus_id(int bus_id, nvmlDevice_t *device)
{
    int nvml_bus_id;
    unsigned index, device_count;
    ucs_status_t status;

    UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetCount(&device_count));

    for (index = 0; index < device_count; index++) {
        status = UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetHandleByIndex(index, device));
        if (UCS_OK != status) {
            return status;
        }

        uct_cuda_base_get_nvml_device_bus_id(device, &nvml_bus_id);
        if (nvml_bus_id == bus_id) {
            return UCS_OK;
        }
    }

    return UCS_ERR_NO_ELEM;
}

static void uct_cuda_base_get_nvml_device(ucs_sys_device_t sys_device,
                                          nvmlDevice_t *device)
{
    ucs_sys_bus_id_t bus_id;
    ucs_status_t status;

    if (sys_device != UCS_SYS_DEVICE_ID_UNKNOWN) {
        goto err;
    }

    status = ucs_topo_get_device_bus_id(sys_device, &bus_id);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_cuda_base_get_nvml_device_from_bus_id((int)bus_id.bus, device);
    if (status == UCS_OK) {
        return;
    }

err:
    device = NULL;
    return;
}

static double uct_cuda_base_get_pci_bw(nvmlDevice_t *device)
{
    double bw = UCT_CUDA_BASE_IFACE_DEFAULT_BANDWIDTH;
    unsigned link_gen;
    unsigned link_width;
    nvmlReturn_t nvml_err;


    nvml_err = nvmlDeviceGetCurrPcieLinkGeneration(*device, &link_gen);
    if (nvml_err != NVML_SUCCESS) {
        goto exit;
    }

    nvml_err = nvmlDeviceGetCurrPcieLinkWidth(*device, &link_width);
    if (nvml_err != NVML_SUCCESS) {
        goto exit;
    }

    ucs_trace("nvml device link gen = %d, link width = %d", link_gen, link_width);

    switch(link_gen) {
        case 1:
            bw = link_width * 250 * UCS_MBYTE;
            break;
        case 2:
            bw = link_width * 500 * UCS_MBYTE;
            break;
        case 3:
            bw = link_width * 985 * UCS_MBYTE;
            break;
        case 4:
            bw = link_width * 1970 * UCS_MBYTE;
            break;
        case 5:
            bw = link_width * 3940 * UCS_MBYTE;
            break;
        default:
            bw = UCT_CUDA_BASE_IFACE_DEFAULT_BANDWIDTH;
            break;
    }

exit:
    return bw;
}

double uct_cuda_base_get_bw(ucs_sys_device_t local_sys_device,
                            ucs_sys_device_t remote_sys_device)
{
    double bw                  = 0.0;
    nvmlDevice_t local_device  = 0;
    nvmlDevice_t remote_device = 0;
    ucs_sys_common_ancestor_t common_ancestor;
    ucs_status_t status;

    uct_cuda_base_get_nvml_device(local_sys_device, &local_device);
    uct_cuda_base_get_nvml_device(remote_sys_device, &remote_device);

    if ((local_sys_device != UCS_SYS_DEVICE_ID_UNKNOWN) &&
        (remote_sys_device != UCS_SYS_DEVICE_ID_UNKNOWN) &&
        (local_sys_device != remote_sys_device)) {
        status = uct_cuda_base_get_common_ancestor(local_device, remote_device,
                                                   &common_ancestor);
        if ((status != UCS_OK) ||
            (common_ancestor != UCS_SYS_TOPO_COMMON_SYSTEM)) {
            /* cuda-ipc maybe supported but performance maybe low */
            return 0.0;
        } else {
            bw = uct_cuda_base_get_nvlink_bw(local_device, remote_device);
            if (bw == 0.0) {
                /* if no nvlink/nvswitch, assume peak pcie bw between devices */
                bw = uct_cuda_base_get_pci_bw(&local_device);
                return bw;
            }
        }
    }

    if ((local_sys_device == UCS_SYS_DEVICE_ID_UNKNOWN) &&
        (remote_sys_device == UCS_SYS_DEVICE_ID_UNKNOWN)) {
        /* cuda-ipc not expected to return host bw */
        return 0.0;
    }

    bw = (local_sys_device == UCS_SYS_DEVICE_ID_UNKNOWN) ?
         uct_cuda_base_get_pci_bw(&remote_device) :
         uct_cuda_base_get_pci_bw(&local_device);

    return bw;
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

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_base_mem_query,
                 (md, address, length, mem_attr),
                 uct_md_h md, const void *address, size_t length,
                 uct_md_mem_attr_t *mem_attr)
{
#define UCT_CUDA_MEM_QUERY_NUM_ATTRS 3
    CUmemorytype cuda_mem_mype = (CUmemorytype)0;
    uint32_t is_managed        = 0;
    unsigned value             = 1;
    CUdevice cuda_device       = -1;
    void *base_address         = (void*)address;
    size_t alloc_length        = length;
    ucs_sys_device_t sys_dev   =  UCS_SYS_DEVICE_ID_UNKNOWN;
    CUpointer_attribute attr_type[UCT_CUDA_MEM_QUERY_NUM_ATTRS];
    void *attr_data[UCT_CUDA_MEM_QUERY_NUM_ATTRS];
    ucs_memory_type_t mem_type;
    const char *cu_err_str;
    ucs_status_t status;
    CUresult cu_err;

    if (!(mem_attr->field_mask & (UCT_MD_MEM_ATTR_FIELD_MEM_TYPE     |
                                  UCT_MD_MEM_ATTR_FIELD_SYS_DEV      |
                                  UCT_MD_MEM_ATTR_FIELD_BASE_ADDRESS |
                                  UCT_MD_MEM_ATTR_FIELD_ALLOC_LENGTH))) {
        return UCS_OK;
    }

    if (address == NULL) {
        mem_type              = UCS_MEMORY_TYPE_HOST;
    } else {
        attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
        attr_data[0] = &cuda_mem_mype;
        attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
        attr_data[1] = &is_managed;
        attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
        attr_data[2] = &cuda_device;

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

        if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_SYS_DEV) {
            status = uct_cuda_base_get_sys_dev(cuda_device, &sys_dev);
            if (status != UCS_OK) {
                return status;
            }
        }

        if (mem_attr->field_mask & (UCT_MD_MEM_ATTR_FIELD_ALLOC_LENGTH |
                                    UCT_MD_MEM_ATTR_FIELD_BASE_ADDRESS)) {
            cu_err = cuMemGetAddressRange((CUdeviceptr*)&base_address,
                                          &alloc_length, (CUdeviceptr)address);
            if (cu_err != CUDA_SUCCESS) {
                cuGetErrorString(cu_err, &cu_err_str);
                ucs_error("ccuMemGetAddressRange(%p) error: %s", address,
                          cu_err_str);
                return UCS_ERR_INVALID_ADDR;
            }
        }
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_MEM_TYPE) {
        mem_attr->mem_type = mem_type;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_SYS_DEV) {
        mem_attr->sys_dev = sys_dev;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_BASE_ADDRESS) {
        mem_attr->base_address = base_address;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_ALLOC_LENGTH) {
        mem_attr->alloc_length = alloc_length;
    }

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

UCS_STATIC_INIT {
    UCT_NVML_FUNC_LOG_ERR(nvmlInit_v2());
}

UCS_STATIC_CLEANUP {
    UCT_NVML_FUNC_LOG_ERR(nvmlShutdown());
}
