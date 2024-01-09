/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_iface.h"
#include "cuda_md.h"

#include <ucs/sys/string.h>


#define UCT_CUDA_DEV_NAME "cuda"


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
    ucs_status_t status;

    if (uct_cuda_base_is_context_active()) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetDevice(&cuda_device));
        if (status != UCS_OK) {
            return status;
        }

        uct_cuda_base_get_sys_dev(cuda_device, &sys_device);
    } else {
        ucs_debug("set cuda sys_device to `unknown` as no context is"
                  " currently active");
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

ucs_status_t uct_cuda_base_check_device_name(const uct_iface_params_t *params)
{
    UCT_CHECK_PARAM(params->field_mask & UCT_IFACE_PARAM_FIELD_DEVICE,
                    "UCT_IFACE_PARAM_FIELD_DEVICE is not defined");

    if (strncmp(params->mode.device.dev_name, UCT_CUDA_DEV_NAME,
                strlen(UCT_CUDA_DEV_NAME)) != 0) {
        ucs_error("no device was found: %s", params->mode.device.dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    return UCS_OK;
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
