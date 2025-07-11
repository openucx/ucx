/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_MD_H
#define UCT_CUDA_MD_H

#include <uct/base/uct_iface.h>
#include <uct/base/uct_md.h>


ucs_status_t
uct_cuda_base_query_md_resources(uct_component_t *component,
                                 uct_md_resource_desc_t **resources_p,
                                 unsigned *num_resources_p);


/**
 * Query the list of available cuda devices.
 *
 * @param [in]  md               Memory domain to run the query on.
 * @param [out] tl_devices_p     List of available devices on the md.
 * @param [out] num_tl_devices_p Number of available devices on the md.
 *
 * @return UCS_OK if successful, or UCS_ERR_NO_MEMORY if failed to allocate the
 *         array of device resources.
 */
ucs_status_t
uct_cuda_base_query_devices(uct_md_h md,
                            uct_tl_device_resource_t **tl_devices_p,
                            unsigned *num_tl_devices_p);


/**
 * Check the device name of interface parameter.
 *
 * @param [in]  params           Interface parameters to check.
 *
 * @return UCS_OK if successful, or UCS_ERR_NO_DEVICE if the device name is
 *         invalid.
 */
ucs_status_t uct_cuda_base_check_device_name(const uct_iface_params_t *params);

#endif
