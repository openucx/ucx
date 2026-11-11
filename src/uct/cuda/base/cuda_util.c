/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "cuda_util.h"
#include "cuda_nvml.h"
#include <ucs/sys/string.h>
#include <ucs/debug/assert.h>
#include <ucs/type/init_once.h>


#define UCT_CUDA_SYS_DEVICE_NAME_PRIORITY 10


/* GPU visible to the CUDA driver */
typedef struct {
    ucs_bus_id_bit_rep_t bus_id_key;
    ucs_sys_device_t     sys_dev;
} uct_cuda_visible_gpu_t;


const char *uct_cuda_cu_get_error_string(CUresult result)
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

ucs_status_t uct_cuda_driver_get_version(int *version_p)
{
    return UCT_CUDADRV_FUNC_LOG_ERR(cuDriverGetVersion(version_p));
}

ucs_status_t uct_cuda_find_device_by_bus_id(const ucs_sys_bus_id_t *bus_id,
                                            ucs_sys_device_t *sys_dev)
{
    ucs_status_t status;

    status = ucs_topo_find_device_by_bus_id(bus_id, sys_dev);
    if (status != UCS_OK) {
        return status;
    }

    status = ucs_topo_sys_device_set_class(*sys_dev, UCS_TOPO_DEVICE_CLASS_ACC);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

static ucs_status_t
uct_cuda_init_bus_id(CUdevice cuda_device, ucs_sys_bus_id_t *bus_id)
{
    ucs_status_t status;
    int attrib;

    /* PCI domain id */
    status = UCT_CUDADRV_FUNC_LOG_DEBUG(
            cuDeviceGetAttribute(&attrib, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                                 cuda_device));
    if (status != UCS_OK) {
        return status;
    }
    bus_id->domain = (uint16_t)attrib;

    /* PCI bus id */
    status = UCT_CUDADRV_FUNC_LOG_DEBUG(
            cuDeviceGetAttribute(&attrib, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                 cuda_device));
    if (status != UCS_OK) {
        return status;
    }
    bus_id->bus = (uint8_t)attrib;

    /* PCI slot id */
    status = UCT_CUDADRV_FUNC_LOG_DEBUG(
            cuDeviceGetAttribute(&attrib, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                                 cuda_device));
    if (status != UCS_OK) {
        return status;
    }
    bus_id->slot = (uint8_t)attrib;

    /* Function - always 0 */
    bus_id->function = 0;

    return UCS_OK;
}

ucs_status_t uct_cuda_get_sys_dev_and_bus_id(CUdevice cuda_device,
                                             ucs_sys_device_t *sys_dev_p,
                                             ucs_sys_bus_id_t *bus_id_p)
{
    ucs_sys_device_t sys_dev;
    ucs_sys_bus_id_t bus_id;
    ucs_status_t status;

    status = uct_cuda_init_bus_id(cuda_device, &bus_id);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_topo_find_device_by_bus_id_and_user_value(
            &bus_id, (uintptr_t)cuda_device, &sys_dev);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_topo_sys_device_set_class(sys_dev, UCS_TOPO_DEVICE_CLASS_ACC);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_topo_sys_device_enable_aux_path(sys_dev);
    if (status != UCS_OK) {
        goto err;
    }

    if (sys_dev_p != NULL) {
        *sys_dev_p = sys_dev;
    }

    if (bus_id_p != NULL) {
        *bus_id_p = bus_id;
    }

    return UCS_OK;

err:
    return status;
}

ucs_sys_device_t uct_cuda_get_sys_dev(CUdevice cuda_device)
{
    ucs_sys_device_t sys_dev;
    ucs_status_t status;

    status = uct_cuda_get_sys_dev_and_bus_id(cuda_device, &sys_dev, NULL);
    if (status != UCS_OK) {
        return UCS_SYS_DEVICE_ID_UNKNOWN;
    }

    return sys_dev;
}

CUdevice uct_cuda_get_cuda_device(ucs_sys_device_t sys_dev)
{
    uintptr_t user_value;

    user_value = ucs_topo_sys_device_get_user_value(sys_dev);
    if (user_value == UCS_SYS_DEVICE_USER_VALUE_EMPTY) {
        return CU_DEVICE_INVALID;
    }

    return (CUdevice)user_value;
}

static int
uct_cuda_gpu_bus_id_is_visible(const uct_cuda_visible_gpu_t *visible_gpus,
                               unsigned num_visible_gpus,
                               ucs_bus_id_bit_rep_t bus_id_key)
{
    unsigned i;

    for (i = 0; i < num_visible_gpus; ++i) {
        if (visible_gpus[i].bus_id_key == bus_id_key) {
            return 1;
        }
    }

    return 0;
}

static unsigned uct_cuda_nvml_get_device_flags(nvmlDevice_t nvml_dev,
                                               const ucs_sys_bus_id_t *bus_id)
{
    unsigned flags = 0;
    nvmlDeviceArchitecture_t arch;
    ucs_status_t status;

    status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetArchitecture, nvml_dev,
                                     &arch);
    if (status != UCS_OK) {
        ucs_debug("gpu " UCS_SYS_BUS_ID_FMT ": failed to get architecture, "
                  "assuming not rubin",
                  UCS_SYS_BUS_ID_ARG(bus_id));
        return 0;
    }

    if (arch == NVML_DEVICE_ARCH_RUBIN) {
        flags |= UCS_TOPO_DEVICE_FLAG_RUBIN;
    }

    return flags;
}

/* MLOPart partitions of a gpu share its bus id, so all of them get the flags */
static void
uct_cuda_visible_gpus_add_flags(const uct_cuda_visible_gpu_t *visible_gpus,
                                unsigned num_visible_gpus,
                                ucs_bus_id_bit_rep_t bus_id_key, unsigned flags)
{
    ucs_status_t status;
    unsigned i;

    for (i = 0; i < num_visible_gpus; ++i) {
        if (visible_gpus[i].bus_id_key != bus_id_key) {
            continue;
        }

        status = ucs_topo_sys_device_add_flags(visible_gpus[i].sys_dev, flags);
        ucs_assert_always(status == UCS_OK);
    }
}

static unsigned uct_cuda_init_devices_cu(uct_cuda_visible_gpu_t *visible_gpus)
{
    unsigned num_visible_gpus = 0;
    uct_cuda_visible_gpu_t *gpu;
    ucs_sys_bus_id_t bus_id;
    char device_name[10];
    ucs_status_t status;
    CUdevice cuda_dev;
    int i, num_devices;

    status = UCT_CUDADRV_FUNC(cuDeviceGetCount(&num_devices),
                              UCS_LOG_LEVEL_DIAG);
    if (status != UCS_OK) {
        return 0;
    }

    ucs_assert_always(num_devices <= UCT_CUDA_MAX_DEVICES);

    for (i = 0; i < num_devices; ++i) {
        status = UCT_CUDADRV_FUNC(cuDeviceGet(&cuda_dev, i),
                                  UCS_LOG_LEVEL_DIAG);
        if (status != UCS_OK) {
            continue;
        }

        gpu    = &visible_gpus[num_visible_gpus];
        status = uct_cuda_get_sys_dev_and_bus_id(cuda_dev, &gpu->sys_dev,
                                                 &bus_id);
        if (status != UCS_OK) {
            ucs_diag("failed to initialize cuda device %d: %s", cuda_dev,
                     ucs_status_string(status));
            continue;
        }

        gpu->bus_id_key = ucs_topo_get_bus_id_bit_repr(&bus_id);

        ucs_snprintf_safe(device_name, sizeof(device_name), "GPU%d", cuda_dev);
        status = ucs_topo_sys_device_set_name(
                gpu->sys_dev, device_name, UCT_CUDA_SYS_DEVICE_NAME_PRIORITY);
        ucs_assert_always(status == UCS_OK);
        ++num_visible_gpus;
    }

    return num_visible_gpus;
}

static ucs_status_t
uct_cuda_init_devices_nvml(const uct_cuda_visible_gpu_t *visible_gpus,
                           unsigned num_visible_gpus)
{
    unsigned nvml_dev_count, i, device_flags;
    ucs_bus_id_bit_rep_t bus_id_key;
    ucs_sys_device_t sys_dev;
    ucs_sys_bus_id_t bus_id;
    nvmlPciInfo_t nvml_pci;
    nvmlDevice_t nvml_dev;
    char device_name[10];
    ucs_status_t status;

    status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetCount_v2, &nvml_dev_count);
    if (status != UCS_OK) {
        return status;
    }

    for (i = 0; i < nvml_dev_count; ++i) {
        status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetHandleByIndex, i,
                                         &nvml_dev);
        if (status != UCS_OK) {
            return status;
        }

        status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetPciInfo_v3, nvml_dev,
                                         &nvml_pci);
        if (status != UCS_OK) {
            return status;
        }

        bus_id.domain   = nvml_pci.domain;
        bus_id.bus      = nvml_pci.bus;
        bus_id.slot     = nvml_pci.device;
        bus_id.function = 0;

        bus_id_key   = ucs_topo_get_bus_id_bit_repr(&bus_id);
        device_flags = uct_cuda_nvml_get_device_flags(nvml_dev, &bus_id);

        if (uct_cuda_gpu_bus_id_is_visible(visible_gpus, num_visible_gpus,
                                           bus_id_key)) {
            uct_cuda_visible_gpus_add_flags(visible_gpus, num_visible_gpus,
                                            bus_id_key, device_flags);
            continue;
        }

        status = uct_cuda_find_device_by_bus_id(&bus_id, &sys_dev);
        if (status != UCS_OK) {
            return status;
        }

        ucs_snprintf_safe(device_name, sizeof(device_name), "UNKN%u", i);
        status = ucs_topo_sys_device_set_name(
                sys_dev, device_name, UCT_CUDA_SYS_DEVICE_NAME_PRIORITY);
        ucs_assert_always(status == UCS_OK);

        status = ucs_topo_sys_device_add_flags(sys_dev, device_flags);
        ucs_assert_always(status == UCS_OK);
    }

    return UCS_OK;
}

static unsigned uct_cuda_init_devices_internal(void)
{
    uct_cuda_visible_gpu_t visible_gpus[UCT_CUDA_MAX_DEVICES];
    ucs_status_t nvml_status, topo_status;
    unsigned num_visible_gpus;

    /* Init visible devices first using the CUDA driver */
    num_visible_gpus = uct_cuda_init_devices_cu(visible_gpus);

    /* Init the remaining non-visible devices using NVML */
    nvml_status = uct_cuda_init_devices_nvml(visible_gpus, num_visible_gpus);
    if (nvml_status != UCS_OK) {
        topo_status = ucs_topo_device_class_mark_incomplete(
                UCS_TOPO_DEVICE_CLASS_ACC);
        ucs_assert_always(topo_status == UCS_OK);

        ucs_diag("failed to initialize nvml devices: %s; gpus hidden from the "
                 "cuda library are not considered in the topology",
                 ucs_status_string(nvml_status));
    }

    return num_visible_gpus;
}

unsigned uct_cuda_init_devices(void)
{
    static ucs_init_once_t init_once = UCS_INIT_ONCE_INITIALIZER;
    static unsigned num_visible_gpus;

    UCS_INIT_ONCE(&init_once) {
        num_visible_gpus = uct_cuda_init_devices_internal();
    }

    return num_visible_gpus;
}
