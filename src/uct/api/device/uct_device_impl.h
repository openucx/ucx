/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_DEVICE_IMPL_H
#define UCT_DEVICE_IMPL_H

#include "uct_device_types.h"

#include <uct/api/uct_def.h>
#include <uct/cuda/cuda_ipc/cuda_ipc.cuh>
#include <ucs/sys/device_code.h>

#include <uct/ib/mlx5/gdaki/gdaki.cuh>


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
template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_device_ep_put_single(
        uct_device_ep_h device_ep, const uct_device_mem_element_t *mem_elem,
        const void *address, uint64_t remote_address, size_t length,
        uint64_t flags, uct_device_completion_t *comp)
{
    if (device_ep->uct_tl_id == UCT_DEVICE_TL_RC_MLX5_GDA) {
        return uct_rc_mlx5_gda_ep_put_single<level>(device_ep, mem_elem,
                                                    address, remote_address,
                                                    length, flags, comp);
    } else if (device_ep->uct_tl_id == UCT_DEVICE_TL_CUDA_IPC) {
        return uct_cuda_ipc_ep_put_single<level>(device_ep, mem_elem, address,
                                                 remote_address, length, flags,
                                                 comp);
    }

    return UCS_ERR_UNSUPPORTED;
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
 * @param [in]  flags           Flags to modify the function behavior.
 * @param [in]  comp            Completion object to track the progress of operation.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_device_ep_atomic_add(
        uct_device_ep_h device_ep, const uct_device_mem_element_t *mem_elem,
        uint64_t inc_value, uint64_t remote_address, uint64_t flags,
        uct_device_completion_t *comp)
{
    if (device_ep->uct_tl_id == UCT_DEVICE_TL_RC_MLX5_GDA) {
        return uct_rc_mlx5_gda_ep_atomic_add<level>(device_ep, mem_elem,
                                                    inc_value, remote_address,
                                                    flags, comp);
    } else if (device_ep->uct_tl_id == UCT_DEVICE_TL_CUDA_IPC) {
        return uct_cuda_ipc_ep_atomic_add<level>(device_ep, mem_elem,
                                                 inc_value, remote_address,
                                                 flags, comp);
    }

    return UCS_ERR_UNSUPPORTED;
}


/**
 * @ingroup UCP_DEVICE
 * @brief Posts few put operations followed by one atomic increment operation.
 *
 * This device routine posts a batch of put operations, followed by an operation.
 * This increment operation can be polled on the receiver to detect completion
 * of all operations of the batch, started during the same routine call.
 *
 * The content of each entries in the arrays addresses, remote_addresses and
 * lengths must be valid for each corresponding descriptor list entry whose
 * index is referenced in @ref mem_list_indices.
 *
 * The size of the arrays mem_list, addresses, remote_addresses, and lengths
 * are all equal.
 *
 * The routine returns a request that can be progressed and checked for
 * completion with @ref ucp_device_progress_req.
 *
 * This routine can be called repeatedly with the same handle and different
 * mem_list_indices, addresses, lengths and increment related parameters. The
 * flags parameter can be used to modify the behavior of the routine with bit
 * from @ref ucp_device_flags_t.
 *
 * @tparam      level                  Level of cooperation of the transfer.
 * @param [in]  mem_list               Memory descriptor list handle to use.
 * @param [in]  mem_list_count         Number of elements in the array @ref
 *                                     mem_list.
 * @param [in]  addresses              Array of local addresses to send from.
 * @param [in]  remote_addresses       Array of remote addresses to send to.
 * @param [in]  lengths                Array of lengths in bytes for each send.
 * @param [in]  counter_inc_value      Value of the remote increment.
 * @param [in]  counter_remote_address Remote address to increment to.
 * @param [in]  flags                  Flags to modify the function behavior.
 * @param [out] req                    Request populated by the call.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_device_ep_put_multi(
        uct_device_ep_h device_ep, const uct_device_mem_element_t *mem_list,
        unsigned mem_list_count, void *const *addresses,
        const uint64_t *remote_addresses, const size_t *lengths,
        uint64_t counter_inc_value, uint64_t counter_remote_address,
        uint64_t flags, uct_device_completion_t *comp)
{
    if (device_ep->uct_tl_id == UCT_DEVICE_TL_RC_MLX5_GDA) {
        return uct_rc_mlx5_gda_ep_put_multi<level>(device_ep, mem_list,
                                                   mem_list_count, addresses,
                                                   remote_addresses, lengths,
                                                   counter_inc_value,
                                                   counter_remote_address,
                                                   flags, comp);
    } else if (device_ep->uct_tl_id == UCT_DEVICE_TL_CUDA_IPC) {
        return uct_cuda_ipc_ep_put_multi<level>(device_ep, mem_list,
                                                mem_list_count, addresses,
                                                remote_addresses, lengths,
                                                counter_inc_value,
                                                counter_remote_address,
                                                flags, comp);
    }

    return UCS_ERR_UNSUPPORTED;
}


/**
 * @ingroup UCP_DEVICE
 * @brief Posts few put operations followed by one atomic increment operation.
 *
 * This device routine posts a batch of put operations using only some of the
 * descriptor list entries in the input handle, followed by an operation.
 * This increment operation can be polled on the receiver to detect completion
 * of all operations of the batch, started during the same routine call.
 *
 * The set of indices from the descriptor list entries to use are to be passed
 * in the array @ref mem_list_indices. The last entry of the descriptor list is to
 * be used for the final increment operation.
 *
 * The content of each entries in the arrays addresses, remote_addresses and
 * lengths must be valid for each corresponding descriptor list entry whose
 * index is referenced in @ref mem_list_indices.
 *
 * The size of the arrays mem_list_indices, addresses, remote_addresses, and
 * lengths are all equal. They are lower than the size of the descriptor list
 * array from the handle.
 *
 * The caller can then use @ref uct_device_ep_progress and the @a comp object to
 * track completion.
 *
 * This routine can be called repeatedly with the same handle and different
 * mem_list_indices, addresses, lengths and increment related parameters. The
 * flags parameter can be used to modify the behavior of the routine with bit
 * from @ref ucp_device_flags_t.
 *
 * @tparam      level                  Level of cooperation of the transfer.
 * @param [in]  mem_list               Memory descriptor list handle to use.
 * @param [in]  mem_list_indices       Array of indices, to use in descriptor
 *                                     list of entries from handle.
 * @param [in]  mem_list_count         Number of indices in the array @ref
 *                                     mem_list_indices.
 * @param [in]  addresses              Array of local addresses to send from.
 * @param [in]  remote_addresses       Array of remote addresses to send to.
 * @param [in]  lengths                Array of lengths in bytes for each send.
 * @param [in]  counter_index          Index of remote increment descriptor.
 * @param [in]  counter_inc_value      Value of the remote increment.
 * @param [in]  counter_remote_address Remote address to increment to.
 * @param [in]  flags                  Flags to modify the function behavior.
 * @param [in]  comp                   Completion object to track progress.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_device_ep_put_multi_partial(
        uct_device_ep_h device_ep, const uct_device_mem_element_t *mem_list,
        const unsigned *mem_list_indices, unsigned mem_list_count,
        void *const *addresses, const uint64_t *remote_addresses,
        const size_t *lengths, unsigned counter_index,
        uint64_t counter_inc_value, uint64_t counter_remote_address,
        uint64_t flags, uct_device_completion_t *comp)
{
    if (device_ep->uct_tl_id == UCT_DEVICE_TL_RC_MLX5_GDA) {
        return uct_rc_mlx5_gda_ep_put_multi_partial<level>(
                device_ep, mem_list, mem_list_indices, mem_list_count,
                addresses, remote_addresses, lengths, counter_index,
                counter_inc_value, counter_remote_address, flags, comp);
    } else if (device_ep->uct_tl_id == UCT_DEVICE_TL_CUDA_IPC) {
        return uct_cuda_ipc_ep_put_multi_partial<level>(device_ep, mem_list,
                                                        mem_list_indices, mem_list_count,
                                                        addresses, remote_addresses,
                                                        lengths, counter_index,
                                                        counter_inc_value, counter_remote_address,
                                                        flags, comp);
    }
    return UCS_ERR_UNSUPPORTED;
}


/**
 * @ingroup UCT_DEVICE
 * @brief Progress all operations on device endpoint @a device_ep.
 *
 * @param [in]  device_ep       Device endpoint to be used for the operation.
 *
 * @return UCS_OK           - Some operation was completed.
 * @return UCS_INPROGRESS   - No progress on the endpoint.
 * @return Error code as defined by @ref ucs_status_t
 */
template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_device_ep_progress(uct_device_ep_h device_ep)
{
    if (device_ep->uct_tl_id == UCT_DEVICE_TL_RC_MLX5_GDA) {
        return uct_rc_mlx5_gda_ep_progress<level>(device_ep);
    } else if (device_ep->uct_tl_id == UCT_DEVICE_TL_CUDA_IPC) {
        return UCS_OK;
    }

    return UCS_ERR_UNSUPPORTED;
}


/**
 * @ingroup UCT_DEVICE
 * @brief Initialize a device completion object.
 *
 * @param [out] comp  Device completion object to initialize.
 */
UCS_F_DEVICE void uct_device_completion_init(uct_device_completion_t *comp)
{
    comp->count  = 0;
    comp->status = UCS_OK;
}

#endif
