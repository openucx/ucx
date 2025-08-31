/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_DEVICE_IMPL_H
#define UCT_DEVICE_IMPL_H

#include "uct_device_types.h"

#include <uct/api/uct_def.h>
#include <ucs/sys/compiler_def.h>


/**
 * @ingroup UCT_DEVICE
 * @brief Posts one memory put operation.
 *
 * This device routine writes a single memory block from the local address @a address
 * to the remote address @a remote_address using the device endpoint @a device_ep.
 * The memory element @a mem_elem must be valid and contain the local and remote
 * memory regions to be transferred.
 *
 * User can pass @a comp to track execution and completion status.
 * The @a flags parameter can be used to modify the behavior
 * of the routine.
 *
 * @param [in]  device_ep       Device endpoint to be used for the operation.
 * @param [in]  mem_elem        Memory element representing the memory to be transferred.
 * @param [in]  address         Local virtual address to send data from.
 * @param [in]  remote_address  Remote virtual address to write data to.
 * @param [in]  length          Length in bytes of the data to send.
 * @param [in]  flags           Flags to modify the function behavior.
 * @param [in]  comp            Completion object to track the progress of operation.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
template<uct_device_level_t level = UCT_DEVICE_LEVEL_THREAD>
UCS_F_DEVICE ucs_status_t uct_device_ep_put_single(
        uct_device_ep_h device_ep, const uct_device_mem_element_t *mem_elem,
        const void *address, uint64_t remote_address, size_t length,
        uint64_t flags, uct_dev_completion_t *comp)
{
    if (device_ep->uct_tl_id == UCT_DEVICE_TL_RC_MLX5_GDA) {
        // return uct_rc_mlx5_gda_ep_put_single(device_ep, mem_elem, address,
        //                                      remote_address, length, flags,
        //                                      comp);
    } else if (device_ep->uct_tl_id == UCT_DEVICE_TL_CUDA_IPC) {
        // return uct_cuda_ipc_ep_put_single(device_ep, mem_elem, address,
        //                                   remote_address, length, flags, comp);
    }
    return UCS_OK;
}


/**
 * @ingroup UCT_DEVICE
 * @brief Initialize a device completion object.
 *
 * @param [out] comp  Device completion object to initialize.
 */
template<uct_device_level_t level = UCT_DEVICE_LEVEL_THREAD>
UCS_F_DEVICE void uct_device_completion_init(uct_dev_completion_t *comp)
{
    comp->count  = 0;
    comp->status = UCS_OK;
}

#endif
