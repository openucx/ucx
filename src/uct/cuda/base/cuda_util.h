/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_UTIL_H
#define UCT_CUDA_UTIL_H

#include <ucs/sys/topo/base/topo.h>
#include <ucs/debug/log.h>

#include <cuda.h>


#define UCT_CUDA_MAX_DEVICES 64


const char *uct_cuda_cu_get_error_string(CUresult result);


#define UCT_CUDADRV_LOG(_func, _log_level, _result) \
    ucs_log((_log_level), "%s failed: %s", UCS_PP_MAKE_STRING(_func), \
            uct_cuda_cu_get_error_string(_result))


#define UCT_CUDADRV_FUNC(_func, _log_level) \
    ({ \
        CUresult _result = (_func); \
        ucs_status_t _status; \
        if (ucs_likely(_result == CUDA_SUCCESS)) { \
            _status = UCS_OK; \
        } else { \
            UCT_CUDADRV_LOG(_func, _log_level, _result); \
            _status = UCS_ERR_IO_ERROR; \
        } \
        _status; \
    })


#define UCT_CUDADRV_FUNC_LOG_ERR(_func) \
    UCT_CUDADRV_FUNC(_func, UCS_LOG_LEVEL_ERROR)


#define UCT_CUDADRV_FUNC_LOG_WARN(_func) \
    UCT_CUDADRV_FUNC(_func, UCS_LOG_LEVEL_WARN)


#define UCT_CUDADRV_FUNC_LOG_DEBUG(_func) \
    UCT_CUDADRV_FUNC(_func, UCS_LOG_LEVEL_DEBUG)


/**
 * Get the system device from the CUDA device.
 *
 * @param [in]  cuda_device CUDA device.
 *
 * @return System device corresponding to the CUDA device.
 */
ucs_sys_device_t uct_cuda_get_sys_dev(CUdevice cuda_device);


/**
 * Find CUDA system device by pci bus id and mark it as an accelerator
 * in the topology.
 *
 * @param [in]  bus_id  pointer to bus id of the device of interest.
 * @param [out] sys_dev system device index associated with the bus_id. If the
 *                      device is not found, *sys_dev is set to
 *                      UCS_SYS_DEVICE_ID_UNKNOWN.
 *
 * @return UCS_OK or error in case device cannot be found or marked as an
 *         accelerator.
 */
ucs_status_t uct_cuda_find_device_by_bus_id(const ucs_sys_bus_id_t *bus_id,
                                            ucs_sys_device_t *sys_dev);
/**
 * Get the CUDA device from the system device.
 *
 * @param [in]  sys_dev     System device.
 *
 * @return CUDA device corresponding to the system device.
 */
CUdevice uct_cuda_get_cuda_device(ucs_sys_device_t sys_dev);


/**
 * Enumerate all physical GPUs and register them in the topology, ignoring
 * CUDA_VISIBLE_DEVICES. Uses NVML when available; otherwise falls back to
 * CUDA-visible devices only.
 *
 * The enumeration is performed once and the resulting system device array is
 * cached; subsequent calls return the cached result without re-enumerating.
 *
 * @param [out] sys_devs_p  If non-NULL, set to point to an internally cached,
 *                          read-only array holding the system device of all
 *                          GPUs. The array is owned by this module and remains
 *                          valid for the lifetime of the process.
 * @param [out] count_p     Number of GPUs in the array.
 *
 * @return UCS_OK on success, or an error code otherwise.
 */
ucs_status_t
uct_cuda_enum_gpus(const ucs_sys_device_t **sys_devs_p, unsigned *count_p);

#endif
