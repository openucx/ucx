/**
 * Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucs/sys/topo/base/topo.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/time/time.h>
#include <nvml.h>

#define UCT_NVML_FUNC(_func, _log_level)                        \
    ({                                                          \
        ucs_status_t _status = UCS_OK;                          \
        do {                                                    \
            nvmlReturn_t _err = (_func);                        \
            if (NVML_SUCCESS != _err) {                         \
                ucs_log((_log_level), "%s failed: %s",          \
                        UCS_PP_MAKE_STRING(_func),              \
                        nvmlErrorString(_err));                 \
                _status = UCS_ERR_IO_ERROR;                     \
            }                                                   \
        } while (0);                                            \
        _status;                                                \
    })


#define UCT_NVML_FUNC_LOG_ERR(_func) \
    UCT_NVML_FUNC(_func, UCS_LOG_LEVEL_ERROR)

typedef enum {
    UCS_SYS_TOPO_COMMON_DEVICE,
    UCS_SYS_TOPO_COMMON_PCIE_SWITCH,
    UCS_SYS_TOPO_COMMON_PCIE_SWITCH_HIERARCHY,
    UCS_SYS_TOPO_COMMON_HOST_BRIDGE,
    UCS_SYS_TOPO_COMMON_NUMA_NODE,
    UCS_SYS_TOPO_COMMON_SYSTEM,
    UCS_SYS_TOPO_COMMON_UNKNOWN
} ucs_sys_topo_common_ancestor_t;

static ucs_status_t
ucs_sys_nvml_get_common_ancestor(nvmlDevice_t device1, nvmlDevice_t device2,
                                 ucs_sys_topo_common_ancestor_t *common_ancestor)
{
    nvmlGpuTopologyLevel_t path_info;
    ucs_status_t status;

    *common_ancestor = UCS_SYS_TOPO_COMMON_UNKNOWN;

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
            case NVML_TOPOLOGY_SYSTEM: /* nvlink/nvswitch */
                *common_ancestor = UCS_SYS_TOPO_COMMON_SYSTEM;
                break;
            default:
                status = UCS_ERR_UNSUPPORTED;
                break;
        }
    }

    return status;
}

static unsigned ucs_sys_nvml_p2p_supported(nvmlDevice_t device1,
                                           nvmlDevice_t device2)
{
    nvmlGpuP2PStatus_t p2p_status;

    UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetP2PStatus(device1, device2,
                                                 NVML_P2P_CAPS_INDEX_NVLINK,
                                                 &p2p_status));

    return (p2p_status == NVML_P2P_STATUS_OK) ? 1 : 0;
}

static double ucs_sys_nvml_get_nvlink_common_bw(nvmlDevice_t device)
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

static unsigned ucs_sys_nvml_get_nvswitch_num_nvlinks(nvmlDevice_t device)
{
    unsigned num_nvlinks = 0;
    unsigned num_links, link;
    nvmlFieldValue_t value;
    nvmlPciInfo_t pci;
    nvmlDevice_t remote_device;

    value.fieldId = NVML_FI_DEV_NVLINK_LINK_COUNT;

    UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetFieldValues(device, 1, &value));

    num_links = ((value.nvmlReturn == NVML_SUCCESS) &&
                 (value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)) ?
                value.value.uiVal : 0;

    for (link = 0; link < num_links; ++link) {
        UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetNvLinkRemotePciInfo(device, link,
                                                               &pci));
        if (NVML_ERROR_NOT_FOUND ==
                nvmlDeviceGetHandleByPciBusId(pci.busId, &remote_device)) {
            /* nvswitch has bus id but not device */
            num_nvlinks++;
            continue;
        }
    }

    return num_nvlinks;
}

static int ucs_sys_nvml_get_num_nvlinks(nvmlDevice_t device1,
                                        nvmlDevice_t device2)
{
    unsigned nvswitch_links = ucs_sys_nvml_get_nvswitch_num_nvlinks(device1);
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

static double ucs_sys_nvml_get_nvlink_bw(nvmlDevice_t device1,
                                          nvmlDevice_t device2)
{
    return (ucs_sys_nvml_p2p_supported(device1, device2) *
            ucs_sys_nvml_get_num_nvlinks(device1, device2) *
            ucs_sys_nvml_get_nvlink_common_bw(device1));
}

static void ucs_sys_nvml_get_bus_id_from_nvml_device(nvmlDevice_t *device,
                                                     int *bus_id)
{
    nvmlPciInfo_t pci_info;

    UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetPciInfo(*device, &pci_info));
    *bus_id = pci_info.bus;
}

static ucs_status_t
ucs_sys_nvml_get_nvml_device_from_bus_id(int bus_id, nvmlDevice_t *device)
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

        ucs_sys_nvml_get_bus_id_from_nvml_device(device, &nvml_bus_id);
        if (nvml_bus_id == bus_id) {
            return UCS_OK;
        }
    }

    return UCS_ERR_NO_ELEM;
}

static void ucs_sys_nvml_get_nvml_device(ucs_sys_device_t sys_device,
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

    status = ucs_sys_nvml_get_nvml_device_from_bus_id((int)bus_id.bus, device);
    if (status == UCS_OK) {
        return;
    }

err:
    device = NULL;
    return;
}

/* report peak host<->device bandwidth */
static double ucs_sys_nvml_get_pci_bw(nvmlDevice_t *device)
{
    double bw = 0.0; /* set default bw here */
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
            bw = 0.0; /* set default bw here */
            break;
    }

exit:
    return bw;
}

static ucs_status_t 
ucs_topo_get_distance_nvml(ucs_sys_device_t local_sys_device,
                           ucs_sys_device_t remote_sys_device,
                           ucs_sys_dev_distance_t *distance)
{
    double bw                  = 0.0;
    double lat                 = 1e-9;
    nvmlDevice_t local_device  = 0;
    nvmlDevice_t remote_device = 0;
    ucs_sys_topo_common_ancestor_t common_ancestor;
    ucs_status_t status;

    ucs_sys_nvml_get_nvml_device(local_sys_device, &local_device);
    ucs_sys_nvml_get_nvml_device(remote_sys_device, &remote_device);

    /* if neither device is recognizable, return error */
    if ((local_device == 0) && (remote_device == 0)) {
        return UCS_ERR_NO_DEVICE;
    }

    if ((local_device != 0) && (remote_device != 0) &&
        (local_device != remote_device)) {
        /* both are recognized by nvml */
        status = ucs_sys_nvml_get_common_ancestor(local_device, remote_device,
                                                  &common_ancestor);
        if (status == UCS_OK) {
            if (common_ancestor != UCS_SYS_TOPO_COMMON_SYSTEM) {
                /* both devices in the same numa domain */
                bw = ucs_sys_nvml_get_nvlink_bw(local_device, remote_device);
                if (bw == 0.0) {
                    /* if no nvlink/nvswitch, assume peak pcie bw b/w devices */
                    goto pci_bw;
                }
            } else {
                goto out;
            }
        } else {
            goto out;
        }
    }

pci_bw:
    bw = (local_sys_device == UCS_SYS_DEVICE_ID_UNKNOWN) ?
         ucs_sys_nvml_get_pci_bw(&remote_device) :
         ucs_sys_nvml_get_pci_bw(&local_device);

out:
    distance->bandwidth = bw;
    distance->latency   = lat;
    return UCS_OK;
}

static ucs_sys_topo_method_t ucs_sys_topo_nvml_method = {
    .name                = "nvml",
    .get_distance        = ucs_topo_get_distance_nvml
};

UCS_STATIC_INIT
{
    ucs_list_add_tail(&ucs_sys_topo_methods_list,
                      &ucs_sys_topo_nvml_method.list);
    nvmlInit_v2();
}

UCS_STATIC_CLEANUP {
    ucs_list_del(&ucs_sys_topo_nvml_method.list);
    nvmlShutdown();
}
