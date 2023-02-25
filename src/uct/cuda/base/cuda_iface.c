/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_iface.h"

double *uct_cuda_base_nvml_bw;

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

unsigned uct_cuda_base_nvml_nvlink_supported(nvmlDevice_t device1, nvmlDevice_t device2)
{
    nvmlGpuP2PStatus_t p2p_status;
    nvmlReturn_t nvml_err;

    nvml_err = nvmlDeviceGetP2PStatus(device1, device2,
                                      NVML_P2P_CAPS_INDEX_NVLINK, &p2p_status);

    return ((nvml_err == NVML_SUCCESS) && (p2p_status == NVML_P2P_STATUS_OK)) ? 1 : 0;
}

double uct_cuda_base_nvml_get_nvlink_common_bw(nvmlDevice_t device1)
{
    unsigned bw;
    nvmlFieldValue_t value;
    nvmlReturn_t nvml_err;

    value.fieldId = NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON;
    nvml_err = nvmlDeviceGetFieldValues(device1, 1, &value);
    if (nvml_err != NVML_SUCCESS) {
        return 0.0;
    }

    bw = ((value.nvmlReturn == NVML_SUCCESS) &&
          (value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)) ?
         value.value.uiVal : 0;

    return (bw * UCS_MBYTE);
}

unsigned uct_cuda_base_nvml_get_nvswitch_num_nvlinks(nvmlDevice_t device1)
{
    unsigned num_links;
    nvmlFieldValue_t value;
    nvmlReturn_t nvml_err;

    value.fieldId = NVML_FI_DEV_NVSWITCH_CONNECTED_LINK_COUNT;
    nvml_err = nvmlDeviceGetFieldValues(device1, 1, &value);
    if (nvml_err != NVML_SUCCESS) {
        return 0;
    }

    num_links = ((value.nvmlReturn == NVML_SUCCESS) &&
                 (value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)) ?
                value.value.uiVal : 0;

    return num_links;
}

int uct_cuda_base_nvml_get_num_nvlinks(nvmlDevice_t device1, nvmlDevice_t device2)
{
    unsigned nvswitch_links = uct_cuda_base_nvml_get_nvswitch_num_nvlinks(device1);
    unsigned total_links    = 0;
    unsigned num_links, link;
    nvmlFieldValue_t value;
    nvmlPciInfo_t pci1, pci2;
    nvmlReturn_t nvml_err;

    if (nvswitch_links) {
        return nvswitch_links;
    }

    value.fieldId = NVML_FI_DEV_NVLINK_LINK_COUNT;

    nvml_err = nvmlDeviceGetFieldValues(device1, 1, &value);
    if (nvml_err != NVML_SUCCESS) {
        goto err;
    }

    num_links = ((value.nvmlReturn == NVML_SUCCESS) &&
                 (value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)) ?
                value.value.uiVal : 0;

    nvml_err = nvmlDeviceGetPciInfo(device2, &pci2);
    if (nvml_err != NVML_SUCCESS) {
        goto err;
    }

    for (link = 0; link < num_links; ++link) {
        nvml_err = nvmlDeviceGetNvLinkRemotePciInfo(device1, link, &pci1);
        if (nvml_err != NVML_SUCCESS) {
            goto err;
        }

        if (!strcmp(pci2.busId, pci1.busId)) {
            total_links++;
        }
    }

    return total_links;
err:
    return 0;
}

double uct_cuda_base_nvml_get_nvlink_bw(nvmlDevice_t device1, nvmlDevice_t device2)
{
    if (uct_cuda_base_nvml_nvlink_supported(device1, device2)) {
        return uct_cuda_base_nvml_get_num_nvlinks(device1, device2) *
               uct_cuda_base_nvml_get_nvlink_common_bw(device1);
    }

    return 0.0;
}

double uct_cuda_base_nvml_get_pcie_bw(nvmlDevice_t device1, nvmlDevice_t device2)
{
    unsigned max_link_gen, max_link_width;
    double bw;
    nvmlReturn_t nvml_err;
    nvmlGpuP2PStatus_t write_status;
    nvmlGpuP2PStatus_t read_status;

    nvml_err = nvmlDeviceGetP2PStatus(device1, device2,
                                      NVML_P2P_CAPS_INDEX_WRITE, &write_status);
    if (NVML_SUCCESS != nvml_err) {
        goto err;
    }

    nvml_err = nvmlDeviceGetP2PStatus(device1, device2,
                                      NVML_P2P_CAPS_INDEX_READ, &read_status);
    if (NVML_SUCCESS != nvml_err) {
        goto err;
    }

    if ((write_status != NVML_P2P_STATUS_OK) ||
        (read_status != NVML_P2P_STATUS_OK)) {
        goto err;
    }

    nvml_err = nvmlDeviceGetMaxPcieLinkGeneration(device1, &max_link_gen);
    if (NVML_SUCCESS != nvml_err) {
        goto err;
    }

    nvml_err = nvmlDeviceGetMaxPcieLinkWidth(device1, &max_link_width);
    if (NVML_SUCCESS != nvml_err) {
        goto err;
    }

    switch(max_link_gen) {
        case 4:
            bw = 1.97 * UCS_GBYTE * max_link_width;
            break;
        case 5:
            bw = 3.94 * UCS_GBYTE * max_link_width;
            break;
        case 6:
            bw = 7.56 * UCS_GBYTE * max_link_width;
            break;
        default:
        case 3:
            bw = 985 * UCS_MBYTE * max_link_width;
            break;
    }

    return bw;

err:
    return 0.0;
}

double uct_cuda_base_nvml_get_p2p_bw(nvmlDevice_t device1, nvmlDevice_t device2)
{
    double bw = uct_cuda_base_nvml_get_nvlink_bw(device1, device2);

    return (bw == 0.0) ? uct_cuda_base_nvml_get_pcie_bw(device1, device2) : bw;
}

double uct_cuda_base_nvml_get_local_bw(nvmlDevice_t device)
{
    nvmlDeviceArchitecture_t arch;
    nvmlReturn_t nvml_err;
    double bw;

    nvml_err = nvmlDeviceGetArchitecture(device, &arch);
    if (nvml_err != NVML_SUCCESS) {
        return 720 * UCS_GBYTE;
    }

    switch(arch) {
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
    nvmlReturn_t nvml_err;

    nvml_err = nvmlDeviceGetHandleByPciBusId(bus_str, &device);
    if (nvml_err != NVML_SUCCESS) {
        return UCS_ERR_NO_ELEM;
    }

    nvml_err = nvmlDeviceGetIndex(device, index);
    if (nvml_err != NVML_SUCCESS) {
        return UCS_ERR_NO_ELEM;
    }

    return UCS_OK;
}

ucs_status_t
uct_cuda_base_nvml_get_estimate_perf(const char *bus_str1, const char *bus_str2,
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

    UCT_NVML_FUNC(nvmlDeviceGetCount(&device_count), UCS_LOG_LEVEL_DEBUG);

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

    UCT_NVML_FUNC(nvmlDeviceGetCount(&device_count), UCS_LOG_LEVEL_DEBUG);

    /* Assumes that nvml detects all devices on the system */
    /* TODO: for multi-node nvlink systems nvml may not show reachable devices
     * outside the system */
    uct_cuda_base_nvml_bw = NULL;
    uct_cuda_base_nvml_bw = (double*)
        ucs_malloc(sizeof(double) * device_count * device_count, "nvml_bw");

    if (uct_cuda_base_nvml_bw == NULL) {
        ucs_error("failed to allocate nvml_bw matrix");
        goto out;
    }

    for (i = 0; i < device_count; i++) {
        for (j = 0; j < device_count; j++) {
            offset = (sizeof(double) * device_count * i) +
                     (sizeof(double) * j);
            bw_ptr = UCS_PTR_BYTE_OFFSET(uct_cuda_base_nvml_bw, offset);

            UCT_NVML_FUNC(nvmlDeviceGetHandleByIndex(i, &device_i), UCS_LOG_LEVEL_DEBUG);
            UCT_NVML_FUNC(nvmlDeviceGetHandleByIndex(j, &device_j), UCS_LOG_LEVEL_DEBUG);

            if (i != j) {
                *bw_ptr = uct_cuda_base_nvml_get_p2p_bw(device_i, device_j);
            } else {
                *bw_ptr = uct_cuda_base_nvml_get_local_bw(device_i);
            }
            ucs_debug("nvml_bw (%u, %u) : %.3lf", i, j, (*bw_ptr/UCS_GBYTE));
        }
    }

out:
    return;
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

UCS_STATIC_INIT {
    UCT_NVML_FUNC(nvmlInit(), UCS_LOG_LEVEL_DEBUG);
    uct_cuda_base_nvml_init();
}

UCS_STATIC_CLEANUP {
    uct_cuda_base_nvml_cleanup();
    UCT_NVML_FUNC(nvmlShutdown(), UCS_LOG_LEVEL_DEBUG);
}
