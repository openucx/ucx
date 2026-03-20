/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025-2026. ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_DEVICE_IMPL_H
#define UCT_DEVICE_IMPL_H

#include "uct_device_types.h"

#include <uct/api/uct_def.h>
<<<<<<< HEAD
#include <ucs/sys/device_code.h>

#if __has_include(<uct/cuda/cuda_ipc/cuda_ipc.cuh>) && \
    __has_include(<cuda/atomic>)
#include <uct/cuda/cuda_ipc/cuda_ipc.cuh>
#define UCT_CUDA_IPC_SUPPORTED 1
#else
#define UCT_CUDA_IPC_SUPPORTED 0
#endif

#if __has_include(<uct/rocm/ipc/rocm_ipc.h>)
#include <uct/rocm/ipc/rocm_ipc.h>
#endif

#if __has_include(<uct/ib/mlx5/gdaki/gdaki.cuh>) && \
    __has_include(<infiniband/mlx5dv.h>)
#include <uct/ib/mlx5/gdaki/gdaki.cuh>
#define UCT_RC_MLX5_GDA_SUPPORTED 1
#else
#define UCT_RC_MLX5_GDA_SUPPORTED 0
#endif

union uct_device_completion {
#if UCT_RC_MLX5_GDA_SUPPORTED
    uct_rc_gda_completion_t   rc_gda;
#endif
#if HAVE_ROCM
    uct_rocm_ipc_completion_t rocm_ipc;
#endif
#if UCT_CUDA_IPC_SUPPORTED
    uct_cuda_ipc_completion_t cuda_ipc;
#endif
};


/**
 * @ingroup UCT_DEVICE
 * @brief Posts one memory put operation.
 *
 * This device routine writes a single memory block from the local address @a address
 * to the remote address @a remote_address using the device endpoint @a device_ep.
 * The local memory element @a src_mem_elem and remote memory element @a mem_elem
 * must be valid and contain the local and remote memory regions to be transferred.
 *
 * User can pass @a comp to track execution and completion status.
 * The @a flags parameter can be used to modify the behavior
 * of the routine.
 *
 * @param [in]  device_ep       Device endpoint to be used for the operation.
 * @param [in]  src_mem_elem    Local memory element representing the memory to be transferred.
 * @param [in]  mem_elem        Remote memory element representing the memory to be transferred.
 * @param [in]  address         Local virtual address to send data from.
 * @param [in]  remote_address  Remote virtual address to write data to.
 * @param [in]  length          Length in bytes of the data to send.
 * @param [in]  channel_id      Channel ID to use for the transfer.
 * @param [in]  flags           Flags to modify the function behavior.
 * @param [in]  comp            Completion object to track the progress of operation.
 *
 * @return UCS_INPROGRESS     - Operation successfully posted, use @ref
 *                              uct_device_ep_progress and @ref
 *                              uct_device_ep_check_completion to
 *                              check for completion.
 * @return UCS_OK             - Operation completed successfully.
 * @return Error code as defined by @ref ucs_status_t
 */
template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t
uct_device_ep_put(uct_device_ep_h device_ep,
                  const uct_device_local_mem_list_elem_t *src_uct_elem,
                  const uct_device_mem_element_t *mem_elem, const void *address,
                  uint64_t remote_address, size_t length, unsigned channel_id,
                  uint64_t flags, uct_device_completion_t *comp)
{
#if UCT_RC_MLX5_GDA_SUPPORTED
    if (device_ep->uct_tl_id == UCT_DEVICE_TL_RC_MLX5_GDA) {
        return uct_rc_mlx5_gda_ep_put<level>(device_ep, src_uct_elem, mem_elem,
                                             address, remote_address, length,
                                             channel_id, flags, comp);
    }
#endif
#if HAVE_ROCM
    if (device_ep->uct_tl_id == UCT_DEVICE_TL_ROCM_IPC) {
        return uct_rocm_ipc_ep_put<level>(device_ep, mem_elem, address,
                                          remote_address, length, flags, comp);
    } else
#elif UCT_CUDA_IPC_SUPPORTED
            if (device_ep->uct_tl_id == UCT_DEVICE_TL_CUDA_IPC) {
        return uct_cuda_ipc_ep_put<level>(device_ep, mem_elem, address,
                                          remote_address, length, flags, comp);
    } else
#else
    {
        return UCS_ERR_UNSUPPORTED;
    }
#endif
}


/**
 * @ingroup UCT_DEVICE
 * @brief Posts one atomic add operation.
 *
 * This device routine increments a single memory value by @a inc_value using the
 * device endpoint @a device_ep. The memory element @a mem_elem must be valid and
 * contain the remote memory region to be modified.
 *
 * User can pass @a comp to track execution and completion status.
 * The @a flags parameter can be used to modify the behavior
 * of the routine.
 *
 * @param [in]  device_ep       Device endpoint to be used for the operation.
 * @param [in]  mem_elem        Memory element representing the memory to be modified.
 * @param [in]  inc_value       Value of the remote increment.
 * @param [in]  remote_address  Remote virtual address to write data to.
 * @param [in]  channel_id      Channel ID to use for the transfer.
 * @param [in]  flags           Flags to modify the function behavior.
 * @param [in]  comp            Completion object to track the progress of operation.
 *
 * @return UCS_INPROGRESS      - Operation successfully posted, use @ref
 *                               uct_device_ep_progress and @ref
 *                               uct_device_ep_check_completion to check
 *                               for completion.
 * @return UCS_OK              - Operation completed successfully.
 * @return Error code as defined by @ref ucs_status_t
 */
template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_device_ep_atomic_add(
        uct_device_ep_h device_ep, const uct_device_mem_element_t *mem_elem,
        uint64_t inc_value, uint64_t remote_address, unsigned channel_id,
        uint64_t flags, uct_device_completion_t *comp)
{
#if UCT_RC_MLX5_GDA_SUPPORTED
    if (device_ep->uct_tl_id == UCT_DEVICE_TL_RC_MLX5_GDA) {
        return uct_rc_mlx5_gda_ep_atomic_add<level>(device_ep, mem_elem,
                                                    inc_value, remote_address,
                                                    channel_id, flags, comp);
    }
#endif
#if HAVE_ROCM
    if (device_ep->uct_tl_id == UCT_DEVICE_TL_ROCM_IPC) {
        return uct_rocm_ipc_ep_atomic_add<level>(device_ep, mem_elem, inc_value,
                                                 remote_address, flags, comp);
    } else
#elif UCT_CUDA_IPC_SUPPORTED
        if (device_ep->uct_tl_id == UCT_DEVICE_TL_CUDA_IPC) {
        return uct_cuda_ipc_ep_atomic_add<level>(device_ep, mem_elem, inc_value,
                                                 remote_address, flags, comp);
    } else
#else
    {
        return UCS_ERR_UNSUPPORTED;
    }
#endif
}


/**
 * @ingroup UCT_DEVICE
 * @brief Gets a local pointer to remote memory.
 *
 * This device routine returns a local pointer to the remote memory if it is available.
 *
 * @param [in]  device_ep  Device endpoint to be used for the operation.
 * @param [in]  mem_elem   Memory element representing the memory to be accessed.
 * @param [in]  address    Local virtual address to get the pointer from.
 * @param [out] addr_p     Local pointer to the remote memory.
 *
 * @return UCS_OK              - Operation completed successfully.
 * @return Error code as defined by @ref ucs_status_t
 */
UCS_F_DEVICE ucs_status_t uct_device_ep_get_ptr(
        uct_device_ep_h device_ep, const uct_device_mem_element_t *mem_elem,
        uint64_t address, void **addr_p)
{
#if HAVE_ROCM
    if (device_ep->uct_tl_id == UCT_DEVICE_TL_ROCM_IPC) {
        return uct_rocm_ipc_ep_get_ptr(device_ep, mem_elem, address, addr_p);
    } else
#elif UCT_CUDA_IPC_SUPPORTED
    if (device_ep->uct_tl_id == UCT_DEVICE_TL_CUDA_IPC) {
        return uct_cuda_ipc_ep_get_ptr(device_ep, mem_elem, address, addr_p);
    } else
#else
    {
        return UCS_ERR_UNSUPPORTED;
    }
#endif
}

/**
 * @ingroup UCT_DEVICE
 * @brief Progress all operations on device endpoint @a device_ep.
 *
 * @param [in]  device_ep       Device endpoint to be used for the operation.
 */
template<ucs_device_level_t level>
UCS_F_DEVICE void uct_device_ep_progress(uct_device_ep_h device_ep)
{
#if UCT_RC_MLX5_GDA_SUPPORTED
    if (device_ep->uct_tl_id == UCT_DEVICE_TL_RC_MLX5_GDA) {
        uct_rc_mlx5_gda_ep_progress<level>(device_ep);
    }
#endif
}


/**
 * @ingroup UCT_DEVICE
 * @brief Check whether opetation executed on device endpoint @a device_ep was
 * completed.
 *
 * @param [in]  device_ep       Device endpoint to be used for the operation.
 * @param [in]  comp            Completion object tracking operation progress.
 *
 * @return UCS_OK           - Some operation was completed.
 * @return UCS_INPROGRESS   - No progress on the endpoint.
 * @return Error code as defined by @ref ucs_status_t
 */
template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_device_ep_check_completion(
        uct_device_ep_h device_ep, uct_device_completion_t *comp)
{
#if UCT_RC_MLX5_GDA_SUPPORTED
    if (device_ep->uct_tl_id == UCT_DEVICE_TL_RC_MLX5_GDA) {
        return uct_rc_mlx5_gda_ep_check_completion<level>(device_ep, comp);
    }
#endif

    return UCS_ERR_UNSUPPORTED;
}

#endif
