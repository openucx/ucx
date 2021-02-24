/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_iface.h"


ucs_status_t
uct_cuda_base_query_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                           unsigned *num_tl_devices_p)
{
    ucs_sys_device_t sys_device = UCS_SYS_DEVICE_ID_UNKNOWN;
    CUdevice cuda_device;

    if (cuCtxGetDevice(&cuda_device) == CUDA_SUCCESS) {
        uct_cuda_base_get_sys_dev(cuda_device, &sys_device);
    }

    return uct_single_device_resource(md, UCT_CUDA_DEV_NAME,
                                      UCT_DEVICE_TYPE_ACC, sys_device,
                                      tl_devices_p, num_tl_devices_p);
}

ucs_status_t uct_cuda_base_get_ctx(CUcontext *ctx)
{
    CUdevice dev;
    int flags;
    int state;

    if (CUDA_SUCCESS == cuCtxGetDevice(&dev)) {
        cuDevicePrimaryCtxGetState(dev, &flags, &state);
        if (state == 0) {
            /* need to retain for malloc purposes */
            if (cuDevicePrimaryCtxRetain(ctx, dev) != CUDA_SUCCESS) {
                ucs_fatal("unable to retain ctx after detecting device");
            }
        }

        if (cuCtxGetCurrent(ctx) != CUDA_SUCCESS) {
            ucs_fatal("unable to get ctx after detecting device");
        }

        return UCS_OK;
    }

    return UCS_ERR_NO_RESOURCE;
}
