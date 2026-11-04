/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "topo.h"

#include <uct/cuda/base/cuda_nvml.h>
#include <uct/cuda/base/cuda_util.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/mlx5/ib_mlx5.h>

#include <ucs/algorithm/qsort_r.h>
#include <ucs/config/parser.h>
#include <ucs/debug/log.h>
#include <ucs/sys/math.h>
#include <ucs/sys/string.h>
#include <ucs/sys/topo/base/topo_int.h>

#include <cuda.h>
#include <infiniband/verbs.h>


#define UCT_GDAKI_TOPO_MAX_DEVICES 64u
#define UCT_GDAKI_TOPO_NAME_PRIO   30

typedef struct {
    unsigned long max_nics_per_gpu;
} uct_gdaki_topo_config_t;

typedef struct {
    ucs_sys_device_t sys_dev;
    ucs_sys_bus_id_t bus_id;
} uct_gdaki_topo_gpu_t;

typedef struct {
    ucs_sys_device_t sys_dev;
    int              is_direct;
} uct_gdaki_topo_nic_t;

typedef struct {
    unsigned               nic_index;
    ucs_sys_dev_distance_t distance;
    unsigned               use_count;
    int                    reachable;
    int                    is_direct;
} uct_gdaki_topo_candidate_t;

static ucs_config_field_t uct_gdaki_topo_config_table[] = {
    {"MAX_NICS_PER_GPU", "auto",
     "Maximum number of NICs assigned to each GPU by the CUDA topology "
     "provider.",
     ucs_offsetof(uct_gdaki_topo_config_t, max_nics_per_gpu),
     UCS_CONFIG_TYPE_ULUNITS},
    {NULL}
};

static ucs_config_global_list_entry_t uct_gdaki_topo_config_entry = {
    .name   = "CUDA topology",
    .prefix = "CUDA_TOPO_",
    .table  = uct_gdaki_topo_config_table,
    .size   = sizeof(uct_gdaki_topo_config_t)
};

UCS_CONFIG_REGISTER_TABLE_ENTRY(&uct_gdaki_topo_config_entry,
                                &ucs_config_global_list);

static struct {
    ucs_sys_topo_provider_t *provider;
    uct_gdaki_topo_gpu_t    gpus[UCT_GDAKI_TOPO_MAX_DEVICES];
    uct_gdaki_topo_nic_t    nics[UCT_GDAKI_TOPO_MAX_DEVICES];
    uint64_t                gpu_nic_map[UCT_GDAKI_TOPO_MAX_DEVICES];
    unsigned                num_gpus;
    unsigned                num_nics;
} uct_gdaki_topo_ctx;

static int uct_gdaki_topo_find_gpu(const ucs_sys_bus_id_t *bus_id);

static int
uct_gdaki_topo_bus_id_equal(const ucs_sys_bus_id_t *bus_id1,
                            const ucs_sys_bus_id_t *bus_id2)
{
    return ucs_topo_get_bus_id_bit_repr(bus_id1) ==
           ucs_topo_get_bus_id_bit_repr(bus_id2);
}

static const uct_gdaki_topo_gpu_t *
uct_gdaki_topo_gpu_lookup(const uct_gdaki_topo_gpu_t *gpus, unsigned count,
                          const ucs_sys_bus_id_t *bus_id)
{
    unsigned gpu_index;

    for (gpu_index = 0; gpu_index < count; ++gpu_index) {
        if (uct_gdaki_topo_bus_id_equal(&gpus[gpu_index].bus_id, bus_id)) {
            return &gpus[gpu_index];
        }
    }

    return NULL;
}

static ucs_status_t
uct_gdaki_topo_get_cuda_gpu(int cuda_index, uct_gdaki_topo_gpu_t *gpu)
{
    CUdevice cuda_device;
    ucs_status_t status;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGet(&cuda_device, cuda_index));
    if (status != UCS_OK) {
        return status;
    }

    return uct_cuda_get_sys_dev_and_bus_id(cuda_device, &gpu->sys_dev,
                                           &gpu->bus_id);
}

static ucs_status_t
uct_gdaki_topo_init_gpu(uct_gdaki_topo_gpu_t *gpu, unsigned index)
{
    ucs_status_t status;
    char name[16];

    status = ucs_topo_sys_device_set_class(gpu->sys_dev,
                                           UCS_TOPO_DEVICE_CLASS_ACC);
    if (status != UCS_OK) {
        return status;
    }

    status = ucs_topo_sys_device_enable_aux_path(gpu->sys_dev);
    if (status != UCS_OK) {
        return status;
    }

    ucs_snprintf_safe(name, sizeof(name), "GPU%u", index);
    return ucs_topo_sys_device_set_name(gpu->sys_dev, name,
                                        UCT_GDAKI_TOPO_NAME_PRIO);
}

static ucs_status_t uct_gdaki_topo_enum_gpus(void)
{
    uct_gdaki_topo_gpu_t cuda_gpus[UCT_GDAKI_TOPO_MAX_DEVICES];
    const uct_gdaki_topo_gpu_t *gpu;
    nvmlPciInfo_t pci_info;
    nvmlDevice_t device;
    unsigned count, index, gpu_index, num_cuda_gpus;
    int cuda_count, cuda_index;
    ucs_status_t status;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetCount(&cuda_count));
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert_always(cuda_count <= UCT_GDAKI_TOPO_MAX_DEVICES);
    num_cuda_gpus = 0;
    for (cuda_index = 0; cuda_index < cuda_count; ++cuda_index) {
        status = uct_gdaki_topo_get_cuda_gpu(
                cuda_index, &cuda_gpus[num_cuda_gpus]);
        if (status != UCS_OK) {
            return status;
        }

        if (uct_gdaki_topo_gpu_lookup(
                    cuda_gpus, num_cuda_gpus,
                    &cuda_gpus[num_cuda_gpus].bus_id) != NULL) {
            ucs_debug("skip cuda device %d with duplicate bdf "
                      UCS_SYS_BUS_ID_FMT,
                      cuda_index,
                      UCS_SYS_BUS_ID_ARG(
                              &cuda_gpus[num_cuda_gpus].bus_id));
            /* Same BDF. TODO: support MLOPart. */
            continue;
        }

        ++num_cuda_gpus;
    }

    status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetCount_v2, &count);
    if (status != UCS_OK) {
        ucs_diag("NVML unavailable: using legacy CUDA-only enumeration");
        memcpy(uct_gdaki_topo_ctx.gpus, cuda_gpus,
               num_cuda_gpus * sizeof(cuda_gpus[0]));
        gpu_index = num_cuda_gpus;
        goto init_gpus;
    }

    count     = ucs_min(count, UCT_GDAKI_TOPO_MAX_DEVICES);
    gpu_index = 0;
    for (index = 0; index < count; ++index) {
        status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetHandleByIndex, index,
                                         &device);
        if (status != UCS_OK) {
            return status;
        }

        status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetPciInfo_v3, device,
                                         &pci_info);
        if (status != UCS_OK) {
            return status;
        }

        uct_gdaki_topo_ctx.gpus[gpu_index].bus_id.domain = pci_info.domain;
        uct_gdaki_topo_ctx.gpus[gpu_index].bus_id.bus    = pci_info.bus;
        uct_gdaki_topo_ctx.gpus[gpu_index].bus_id.slot   = pci_info.device;
        uct_gdaki_topo_ctx.gpus[gpu_index].bus_id.function = 0;

        gpu = uct_gdaki_topo_gpu_lookup(
                cuda_gpus, num_cuda_gpus,
                &uct_gdaki_topo_ctx.gpus[gpu_index].bus_id);
        if (gpu != NULL) {
            uct_gdaki_topo_ctx.gpus[gpu_index++] = *gpu;
            continue;
        }

        status = ucs_topo_find_device_by_bus_id(
                &uct_gdaki_topo_ctx.gpus[gpu_index].bus_id,
                &uct_gdaki_topo_ctx.gpus[gpu_index].sys_dev);
        if (status != UCS_OK) {
            return status;
        }

        ++gpu_index;
    }

init_gpus:
    for (index = 0; index < gpu_index; ++index) {
        status = uct_gdaki_topo_init_gpu(&uct_gdaki_topo_ctx.gpus[index],
                                         index);
        if (status != UCS_OK) {
            return status;
        }
    }

    uct_gdaki_topo_ctx.num_gpus = gpu_index;
    return (gpu_index == 0) ? UCS_ERR_NO_DEVICE : UCS_OK;
}

static ucs_status_t
uct_gdaki_topo_add_nic(struct ibv_device *ibdev, struct ibv_context *context)
{
    uct_gdaki_topo_nic_t *nic;
    uct_ib_device_t dev = {
        .ibv_context = context
    };
    ucs_status_t status;

    status = uct_ib_device_query(&dev, ibdev);
    if (status != UCS_OK) {
        return status;
    }

    if (!uct_ib_device_has_active_port(&dev)) {
        return UCS_ERR_NO_DEVICE;
    }

    nic            = &uct_gdaki_topo_ctx.nics[uct_gdaki_topo_ctx.num_nics];
    nic->sys_dev   = dev.sys_dev;
    nic->is_direct = uct_ib_mlx5dv_check_direct_nic(&dev, 1) !=
                     UCS_SYS_DEVICE_ID_UNKNOWN;
    ++uct_gdaki_topo_ctx.num_nics;
    return UCS_OK;
}

static ucs_status_t uct_gdaki_topo_enum_nics(void)
{
    struct ibv_device **devices;
    struct ibv_context *context;
    ucs_status_t status;
    ucs_status_t fatal_status = UCS_OK;
    int count, index;

    devices = ibv_get_device_list(&count);
    if (devices == NULL) {
        return UCS_ERR_IO_ERROR;
    }

    for (index = 0;
         (index < count) &&
         (uct_gdaki_topo_ctx.num_nics < UCT_GDAKI_TOPO_MAX_DEVICES);
         ++index) {
        context = ibv_open_device(devices[index]);
        if (context == NULL) {
            continue;
        }

        status = uct_gdaki_topo_add_nic(devices[index], context);
        ibv_close_device(context);
        if ((status != UCS_OK) && (status != UCS_ERR_NO_DEVICE)) {
            fatal_status = status;
            break;
        }
    }

    ibv_free_device_list(devices);
    if (fatal_status != UCS_OK) {
        return fatal_status;
    }

    return (uct_gdaki_topo_ctx.num_nics == 0) ? UCS_ERR_NO_DEVICE : UCS_OK;
}

static int
uct_gdaki_topo_candidate_cmp(const void *pa, const void *pb,
                             void *UCS_V_UNUSED arg)
{
    const uct_gdaki_topo_candidate_t *a = pa;
    const uct_gdaki_topo_candidate_t *b = pb;
    int cmp;

    if (a->reachable != b->reachable) {
        return b->reachable - a->reachable;
    }

    if (a->is_direct != b->is_direct) {
        return b->is_direct - a->is_direct;
    }

    cmp = ucs_topo_distance_cmp(&a->distance, &b->distance);
    if (cmp != 0) {
        return cmp;
    }

    if (a->use_count != b->use_count) {
        return (a->use_count < b->use_count) ? -1 : 1;
    }

    return ucs_topo_sys_device_cmp(
            uct_gdaki_topo_ctx.nics[a->nic_index].sys_dev,
            uct_gdaki_topo_ctx.nics[b->nic_index].sys_dev);
}

static ucs_status_t
uct_gdaki_topo_build_assignment(unsigned long max_nics_per_gpu)
{
    uct_gdaki_topo_candidate_t candidates[UCT_GDAKI_TOPO_MAX_DEVICES];
    unsigned use_count[UCT_GDAKI_TOPO_MAX_DEVICES] = {0};
    unsigned gpu_index, nic_index, selected, limit, reachable;
    ucs_status_t status;

    if (max_nics_per_gpu == UCS_ULUNITS_AUTO) {
        limit = ucs_max(1u, uct_gdaki_topo_ctx.num_nics /
                                   uct_gdaki_topo_ctx.num_gpus);
    } else if (max_nics_per_gpu == UCS_ULUNITS_INF) {
        limit = uct_gdaki_topo_ctx.num_nics;
    } else {
        limit = ucs_min(max_nics_per_gpu, uct_gdaki_topo_ctx.num_nics);
    }

    for (gpu_index = 0; gpu_index < uct_gdaki_topo_ctx.num_gpus;
         ++gpu_index) {
        reachable = 0;
        for (nic_index = 0; nic_index < uct_gdaki_topo_ctx.num_nics;
             ++nic_index) {
            candidates[nic_index].nic_index = nic_index;
            candidates[nic_index].use_count = use_count[nic_index];
            candidates[nic_index].is_direct =
                    uct_gdaki_topo_ctx.nics[nic_index].is_direct;
            candidates[nic_index].reachable = ucs_topo_is_reachable(
                    uct_gdaki_topo_ctx.nics[nic_index].sys_dev,
                    uct_gdaki_topo_ctx.gpus[gpu_index].sys_dev);
            if (!candidates[nic_index].reachable) {
                candidates[nic_index].distance = ucs_topo_max_distance;
                continue;
            }

            status = ucs_topo_get_distance(
                    uct_gdaki_topo_ctx.gpus[gpu_index].sys_dev,
                    uct_gdaki_topo_ctx.nics[nic_index].sys_dev,
                    &candidates[nic_index].distance);
            if (status != UCS_OK) {
                return status;
            }
            ++reachable;
        }

        ucs_qsort_r(candidates, uct_gdaki_topo_ctx.num_nics,
                    sizeof(candidates[0]), uct_gdaki_topo_candidate_cmp, NULL);
        selected = ucs_min(limit, reachable);
        for (nic_index = 0; nic_index < selected; ++nic_index) {
            ucs_assert(candidates[nic_index].reachable);
            uct_gdaki_topo_ctx.gpu_nic_map[gpu_index] |=
                    UCS_BIT(candidates[nic_index].nic_index);
            ++use_count[candidates[nic_index].nic_index];
            if (candidates[nic_index].is_direct) {
                break;
            }
        }
    }

    return UCS_OK;
}

static int uct_gdaki_topo_find_gpu(const ucs_sys_bus_id_t *bus_id)
{
    unsigned index;

    for (index = 0; index < uct_gdaki_topo_ctx.num_gpus; ++index) {
        if (uct_gdaki_topo_bus_id_equal(
                    &uct_gdaki_topo_ctx.gpus[index].bus_id, bus_id)) {
            return index;
        }
    }

    return -1;
}

static int uct_gdaki_topo_find_nic(ucs_sys_device_t sys_dev)
{
    unsigned index;

    for (index = 0; index < uct_gdaki_topo_ctx.num_nics; ++index) {
        if (uct_gdaki_topo_ctx.nics[index].sys_dev == sys_dev) {
            return index;
        }
    }

    return -1;
}

static int
uct_gdaki_topo_get_assignment(ucs_sys_device_t gpu_sys_dev,
                              ucs_sys_device_t nic_sys_dev, int *is_assigned_p)
{
    ucs_sys_bus_id_t bus_id;
    int gpu_index, nic_index;

    if (ucs_topo_get_device_bus_id(gpu_sys_dev, &bus_id) != UCS_OK) {
        return 0;
    }

    gpu_index = uct_gdaki_topo_find_gpu(&bus_id);
    nic_index = uct_gdaki_topo_find_nic(nic_sys_dev);
    if ((gpu_index < 0) || (nic_index < 0)) {
        return 0;
    }

    *is_assigned_p = !!(uct_gdaki_topo_ctx.gpu_nic_map[gpu_index] &
                        UCS_BIT(nic_index));
    return 1;
}

static ucs_status_t
uct_gdaki_topo_get_distance(ucs_sys_device_t device1, ucs_sys_device_t device2,
                            ucs_sys_dev_distance_t *distance)
{
    int is_assigned;

    if ((uct_gdaki_topo_get_assignment(device1, device2, &is_assigned) ||
         uct_gdaki_topo_get_assignment(device2, device1, &is_assigned)) &&
        !is_assigned) {
        *distance = ucs_topo_max_distance;
    }

    return UCS_OK;
}

static ucs_status_t uct_gdaki_topo_do_init(void)
{
    uct_gdaki_topo_config_t config;
    const ucs_sys_topo_ops_t ops = {
        .get_distance = uct_gdaki_topo_get_distance
    };
    ucs_status_t status;

    status = ucs_config_parser_fill_opts(&config,
                                         &uct_gdaki_topo_config_entry,
                                         UCS_DEFAULT_ENV_PREFIX, 0);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_gdaki_topo_enum_gpus();
    if (status != UCS_OK) {
        goto out_release_config;
    }

    status = uct_gdaki_topo_enum_nics();
    if (status != UCS_OK) {
        goto out_release_config;
    }

    status = uct_gdaki_topo_build_assignment(config.max_nics_per_gpu);
    if (status != UCS_OK) {
        goto out_release_config;
    }

    status = ucs_sys_topo_provider_register("cuda", &ops,
                                             &uct_gdaki_topo_ctx.provider);
    if (status == UCS_OK) {
        ucs_config_parser_release_opts(&config,
                                       uct_gdaki_topo_config_table);
        return UCS_OK;
    }

out_release_config:
    ucs_config_parser_release_opts(&config, uct_gdaki_topo_config_table);
    return status;
}

void uct_gdaki_topo_init(void)
{
    ucs_status_t status;

    status = uct_gdaki_topo_do_init();
    if (status != UCS_OK) {
        ucs_debug("cuda topology provider is unavailable: %s",
                  ucs_status_string(status));
    }
}

void uct_gdaki_topo_cleanup(void)
{
    if (uct_gdaki_topo_ctx.provider != NULL) {
        ucs_sys_topo_provider_unregister(uct_gdaki_topo_ctx.provider);
        uct_gdaki_topo_ctx.provider = NULL;
    }
}
