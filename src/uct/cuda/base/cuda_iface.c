/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_iface.h"

#include <ucs/sys/string.h>


double *uct_cuda_base_nvml_bw;


const char *uct_cuda_base_cu_get_error_string(CUresult result)
{
    static __thread char buf[64];
    const char *error_str;

    if (cuGetErrorString(result, &error_str) != CUDA_SUCCESS) {
        ucs_snprintf_safe(buf, sizeof(buf), "unrecognized error code %d",
                          result);
        error_str = buf;
    }

    return error_str;
}

ucs_status_t
uct_cuda_base_query_devices_common(
        uct_md_h md, uct_device_type_t dev_type,
        uct_tl_device_resource_t **tl_devices_p, unsigned *num_tl_devices_p)
{
    ucs_sys_device_t sys_device = UCS_SYS_DEVICE_ID_UNKNOWN;
    CUdevice cuda_device;

    if (cuCtxGetDevice(&cuda_device) == CUDA_SUCCESS) {
        uct_cuda_base_get_sys_dev(cuda_device, &sys_device);
    }

    return uct_single_device_resource(md, UCT_CUDA_DEV_NAME, dev_type,
                                      sys_device, tl_devices_p,
                                      num_tl_devices_p);
}

ucs_status_t
uct_cuda_base_query_devices(
        uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
        unsigned *num_tl_devices_p)
{
    return uct_cuda_base_query_devices_common(md, UCT_DEVICE_TYPE_ACC,
                                              tl_devices_p, num_tl_devices_p);
}

#if (__CUDACC_VER_MAJOR__ >= 100000)
void CUDA_CB uct_cuda_base_iface_stream_cb_fxn(void *arg)
#else
void CUDA_CB uct_cuda_base_iface_stream_cb_fxn(CUstream hStream, CUresult status,
                                               void *arg)
#endif
{
    uct_cuda_iface_t *cuda_iface = arg;

    ucs_async_eventfd_signal(cuda_iface->eventfd);
}

ucs_status_t uct_cuda_base_iface_event_fd_get(uct_iface_h tl_iface, int *fd_p)
{
    uct_cuda_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_iface_t);
    ucs_status_t status;

    if (iface->eventfd == UCS_ASYNC_EVENTFD_INVALID_FD) {
        status = ucs_async_eventfd_create(&iface->eventfd);
        if (status != UCS_OK) {
            return status;
        }
    }

    *fd_p = iface->eventfd;
    return UCS_OK;
}

int
uct_cuda_base_nvml_nvlink_supported(nvmlDevice_t device1, nvmlDevice_t device2)
{
    nvmlGpuP2PStatus_t p2p_status;
    ucs_status_t status;

    status = UCT_NVML_FUNC_LOG_DEBUG(
            nvmlDeviceGetP2PStatus(device1, device2, NVML_P2P_CAPS_INDEX_NVLINK,
                                   &p2p_status));

    return ((status == UCS_OK) && (p2p_status == NVML_P2P_STATUS_OK));
}

double uct_cuda_base_nvml_get_nvlink_common_bw(nvmlDevice_t device)
{
    unsigned bw;
    nvmlFieldValue_t value;

    value.fieldId = NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON;
    if (UCT_NVML_FUNC_LOG_DEBUG(nvmlDeviceGetFieldValues(device, 1, &value)) !=
        UCS_OK) {
        return 0.;
    }

    bw = ((value.nvmlReturn == NVML_SUCCESS) &&
          (value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)) ?
                 value.value.uiVal :
                 0;

    return (bw * UCS_MBYTE);
}

unsigned uct_cuda_base_nvml_get_nvswitch_num_nvlinks(nvmlDevice_t device)
{
    unsigned num_links = 0;

#if defined(NVML_FI_DEV_NVSWITCH_CONNECTED_LINK_COUNT)
    nvmlFieldValue_t value;

    value.fieldId = NVML_FI_DEV_NVSWITCH_CONNECTED_LINK_COUNT;
    if (UCT_NVML_FUNC_LOG_DEBUG(nvmlDeviceGetFieldValues(device, 1, &value)) !=
        UCS_OK) {
        return 0;
    }

    num_links = ((value.nvmlReturn == NVML_SUCCESS) &&
                 (value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)) ?
                        value.value.uiVal :
                        0;
#endif

    return num_links;
}

unsigned uct_cuda_base_nvml_get_num_nvlinks(nvmlDevice_t device1,
                                            nvmlDevice_t device2)
{
    unsigned nvswitch_links = uct_cuda_base_nvml_get_nvswitch_num_nvlinks(
            device1);
    unsigned total_links    = 0;
    unsigned num_links, link;
    nvmlFieldValue_t value;
    nvmlPciInfo_t pci1, pci2;

    if (nvswitch_links) {
        return nvswitch_links;
    }

    value.fieldId = NVML_FI_DEV_NVLINK_LINK_COUNT;
    if (UCT_NVML_FUNC_LOG_DEBUG(nvmlDeviceGetFieldValues(device1, 1, &value)) !=
        UCS_OK) {
        return 0;
    }

    num_links = ((value.nvmlReturn == NVML_SUCCESS) &&
                 (value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)) ?
                        value.value.uiVal :
                        0;

    if (UCT_NVML_FUNC_LOG_DEBUG(nvmlDeviceGetPciInfo_v3(device2, &pci2)) !=
        UCS_OK) {
        return 0;
    }

    for (link = 0; link < num_links; ++link) {
        if (UCT_NVML_FUNC_LOG_DEBUG(
                nvmlDeviceGetNvLinkRemotePciInfo_v2(device1, link, &pci1)) !=
            UCS_OK) {
            return 0;
        }

        if (!strcmp(pci2.busId, pci1.busId)) {
            total_links++;
        }
    }

    return total_links;
}

double
uct_cuda_base_nvml_get_nvlink_bw(nvmlDevice_t device1, nvmlDevice_t device2)
{
    if (uct_cuda_base_nvml_nvlink_supported(device1, device2)) {
        return uct_cuda_base_nvml_get_num_nvlinks(device1, device2) *
               uct_cuda_base_nvml_get_nvlink_common_bw(device1);
    }

    return 0.0;
}

double
uct_cuda_base_nvml_get_pcie_bw(nvmlDevice_t device1, nvmlDevice_t device2)
{
    unsigned max_link_gen, max_link_width;
    nvmlGpuP2PStatus_t write_status;
    nvmlGpuP2PStatus_t read_status;

    if (UCT_NVML_FUNC_LOG_DEBUG(
            nvmlDeviceGetP2PStatus(device1, device2, NVML_P2P_CAPS_INDEX_WRITE,
                                   &write_status)) != UCS_OK) {
        return 0.;
    }

    if (UCT_NVML_FUNC_LOG_DEBUG(
            nvmlDeviceGetP2PStatus(device1, device2, NVML_P2P_CAPS_INDEX_READ,
                                   &read_status)) != UCS_OK) {
        return 0.;
    }

    if ((write_status != NVML_P2P_STATUS_OK) ||
        (read_status != NVML_P2P_STATUS_OK)) {
        return 0.;
    }

    if (UCT_NVML_FUNC_LOG_DEBUG(
            nvmlDeviceGetMaxPcieLinkGeneration(device1, &max_link_gen)) !=
        UCS_OK) {
        return 0.;
    }

    if (UCT_NVML_FUNC_LOG_DEBUG(
            nvmlDeviceGetMaxPcieLinkWidth(device1, &max_link_width)) !=
        UCS_OK) {
        return 0.;
    }

    switch (max_link_gen) {
    case 4:
        return 1.97 * UCS_GBYTE * max_link_width;
    case 5:
        return 3.94 * UCS_GBYTE * max_link_width;
    case 6:
        return 7.56 * UCS_GBYTE * max_link_width;
    default:
    case 3:
        return 985 * UCS_MBYTE * max_link_width;
    }
}

double uct_cuda_base_nvml_get_p2p_bw(nvmlDevice_t device1, nvmlDevice_t device2)
{
    double bw = uct_cuda_base_nvml_get_nvlink_bw(device1, device2);

    return (bw == 0.0) ? uct_cuda_base_nvml_get_pcie_bw(device1, device2) : bw;
}

double uct_cuda_base_nvml_get_local_bw(nvmlDevice_t device)
{
    nvmlDeviceArchitecture_t arch;
    double bw;

    if (UCT_NVML_FUNC_LOG_DEBUG(nvmlDeviceGetArchitecture(device, &arch)) !=
        UCS_OK) {
        return 720 * UCS_GBYTE;
    }

    switch (arch) {
    case NVML_DEVICE_ARCH_VOLTA:
        bw = 900 * UCS_GBYTE;
        break;
#if defined(NVML_DEVICE_ARCH_AMPERE)
    case NVML_DEVICE_ARCH_AMPERE:
        bw = 1555 * UCS_GBYTE;
        break;
#endif
#if defined(NVML_DEVICE_ARCH_HOPPER)
    case NVML_DEVICE_ARCH_HOPPER:
        bw = 3000 * UCS_GBYTE;
        break;
#endif
    case NVML_DEVICE_ARCH_PASCAL:
    default:
        bw = 720 * UCS_GBYTE;
        break;
    }

    return bw;
}

ucs_status_t
uct_cuda_base_nvml_get_device_index(const char *bus_str, unsigned *index)
{
    nvmlDevice_t device;
    ucs_status_t status;

    status = UCT_NVML_FUNC_LOG_DEBUG(nvmlDeviceGetHandleByPciBusId_v2(bus_str,
                                                                      &device));
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_NVML_FUNC_LOG_DEBUG(nvmlDeviceGetIndex(device, index));
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

ucs_status_t uct_cuda_base_nvml_get_estimate_perf(const char *bus_str1,
                                                  const char *bus_str2,
                                                  double *bw)
{
    ucs_status_t status;
    unsigned index1, index2;
    unsigned device_count;
    double *bw_ptr;
    size_t offset;

    status = uct_cuda_base_nvml_get_device_index(bus_str1, &index1);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_cuda_base_nvml_get_device_index(bus_str2, &index2);
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_NVML_FUNC_LOG_DEBUG(nvmlDeviceGetCount_v2(&device_count));
    if (status != UCS_OK) {
        return status;
    }

    offset = (sizeof(double) * device_count * index1) +
             (sizeof(double) * index2);
    bw_ptr = UCS_PTR_BYTE_OFFSET(uct_cuda_base_nvml_bw, offset);
    *bw    = *bw_ptr;

    return UCS_OK;
}

void uct_cuda_base_nvml_init()
{
    unsigned device_count;
    unsigned i, j;
    size_t offset;
    double *bw_ptr;
    nvmlDevice_t device_i, device_j;

    if (UCT_NVML_FUNC_LOG_DEBUG(nvmlDeviceGetCount_v2(&device_count)) !=
        UCS_OK) {
        return;
    }

    /* Assumes that nvml detects all devices on the system */
    /* TODO: for multi-node nvlink systems nvml may not show reachable devices
     * outside the system */
    uct_cuda_base_nvml_bw = (double*)ucs_malloc(sizeof(double) * device_count *
                                                        device_count,
                                                "nvml_bw");
    if (uct_cuda_base_nvml_bw == NULL) {
        ucs_error("failed to allocate nvml_bw matrix");
        return;
    }

    for (i = 0; i < device_count; ++i) {
        for (j = 0; j < device_count; ++j) {
            offset = (sizeof(double) * device_count * i) + (sizeof(double) * j);
            bw_ptr = UCS_PTR_BYTE_OFFSET(uct_cuda_base_nvml_bw, offset);

            if ((UCT_NVML_FUNC_LOG_DEBUG(
                    nvmlDeviceGetHandleByIndex_v2(i, &device_i)) != UCS_OK) ||
                (UCT_NVML_FUNC_LOG_DEBUG(
                    nvmlDeviceGetHandleByIndex_v2(j, &device_j)) != UCS_OK)) {
                *bw_ptr = 0.;
                continue;
            }

            if (i != j) {
                *bw_ptr = uct_cuda_base_nvml_get_p2p_bw(device_i, device_j);
            } else {
                *bw_ptr = uct_cuda_base_nvml_get_local_bw(device_i);
            }
            ucs_debug("nvml_bw (%u, %u) : %.3lf", i, j, (*bw_ptr / UCS_GBYTE));
        }
    }
}

void uct_cuda_base_nvml_cleanup()
{
    ucs_free(uct_cuda_base_nvml_bw);
}

UCS_CLASS_INIT_FUNC(uct_cuda_iface_t, uct_iface_ops_t *tl_ops,
                    uct_iface_internal_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    const uct_iface_config_t *tl_config,
                    const char *dev_name)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, tl_ops, ops, md, worker, params,
                              tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(dev_name));

    self->eventfd = UCS_ASYNC_EVENTFD_INVALID_FD;

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_iface_t)
{
    ucs_async_eventfd_destroy(self->eventfd);
}

UCS_CLASS_DEFINE(uct_cuda_iface_t, uct_base_iface_t);

UCS_STATIC_INIT
{
    if (UCT_NVML_FUNC_LOG_ERR(nvmlInit_v2()) == UCS_OK) {
        uct_cuda_base_nvml_init();
    }
}

UCS_STATIC_CLEANUP
{
    uct_cuda_base_nvml_cleanup();
    UCT_NVML_FUNC_LOG_ERR(nvmlShutdown());
}
