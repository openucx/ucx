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
    return uct_single_device_resource(md, UCT_CUDA_DEV_NAME, UCT_DEVICE_TYPE_ACC,
                                      tl_devices_p, num_tl_devices_p);
}

