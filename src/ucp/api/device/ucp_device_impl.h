/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_DEVICE_IMPL_H
#define UCP_DEVICE_IMPL_H

#include "ucp_device_types.h"

#include <ucs/sys/compiler_def.h>
#include <ucs/type/status.h>
#include <stdint.h>


/**
 * @ingroup UCP_DEVICE
 * @brief Posts one memory put operation.
 *
 * This device routine posts one put operation using descriptor list handle.
 * The @a mem_list_index is used to point at the @a mem_list entry to be used
 * for the memory transfer. The addresses and length must be valid for the used
 * @a mem_list entry.
 *
 * The routine returns a request that can be progressed and checked for
 * completion with @ref ucp_device_progress_req.
 *
 * This routine can be called repeatedly with the same handle and different
 * addresses and length. The flags parameter can be used to modify the behavior
 * of the routine.
 *
 * @param [in]  mem_list        Memory descriptor list handle to use.
 * @param [in]  mem_list_index  Index in descriptor list pointing to the memory
 * @param [in]  address         Local virtual address to send data from.
 * @param [in]  remote_address  Remote virtual address to send data to.
 * @param [in]  length          Length in bytes of the data to send.
 *                              registration keys to use for the transfer.
 * @param [in]  flags           Flags usable to modify the function behavior.
 * @param [out] req             Request populated by the call.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
UCS_F_DEVICE ucs_status_t
ucp_device_put_single(ucp_device_mem_list_handle_h mem_list,
                      unsigned mem_list_index,
                      const void *address, uint64_t remote_address,
                      size_t length, uint64_t flags, ucp_device_request_t *req)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}


/**
 * @ingroup UCP_DEVICE
 * @brief Posts one memory increment operation.
 *
 * This device routine posts one increment operation using memory descriptor
 * list handle. The @ref mem_list_index is used to point at the @a mem_list
 * entry to be used for the increment operation. The remote address must be
 * valid for the used @a mem_list entry.
 *
 * The routine returns a request that can be progressed and checked for
 * completion with @ref ucp_device_progress_req.
 *
 * This routine can be called repeatedly with the same handle and different
 * address. The flags parameter can be used to modify the behavior of the
 * routine.
 *
 * @param [in]  mem_list        Memory descriptor list handle to use.
 * @param [in]  mem_list_index  Index in descriptor list pointing to the memory
 *                              remote key to use for the increment operation.
 * @param [in]  inc_value       Value used to increment the remote address.
 * @param [in]  remote_address  Remote virtual address to perform the increment
 *                              to.
 * @param [in]  flags           Flags usable to modify the function behavior.
 * @param [out] req             Request populated by the call.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
UCS_F_DEVICE ucs_status_t
ucp_device_counter_inc(ucp_device_mem_list_handle_h mem_list,
                       unsigned mem_list_index, uint64_t inc_value,
                       uint64_t remote_address, uint64_t flags,
                       ucp_device_request_t *req)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}


/**
 * @ingroup UCP_DEVICE
 * @brief Posts multiple put operations followed by one increment operation.
 *
 * This device routine posts a batch of put operations using the descriptor list
 * entries in the input handle, followed by an increment operation. This
 * operation can be polled on the receiver to detect completion of all the
 * operations of the batch, started during the same routine call.
 *
 * The content of each entries in the arrays @a addresses, @a remote_addresses
 * and @a lengths must be valid for each corresponding entry in the descriptor
 * list from the input handle. The last entry in the descriptor list contains
 * the remote memory registration descriptors to be used for the increment
 * operation.
 *
 * The size of the arrays @a addresses, @a remote_addresses, and @a lengths
 * are all equal to the size of the descriptor list array from the handle,
 * minus one.
 *
 * The routine returns a request that can be progressed and checked for
 * completion with @ref ucp_device_progress_req.
 *
 * This routine can be called repeatedly with the same handle and different
 * @a addresses, @a lengths and counter related parameters. The @a flags
 * parameter can be used to modify the behavior of the routine.
 *
 * @param [in]  mem_list               Memory descriptor list handle to use.
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
UCS_F_DEVICE ucs_status_t
ucp_device_put_multi(ucp_device_mem_list_handle_h mem_list,
                     void *const *addresses, const uint64_t *remote_addresses,
                     const size_t *lengths, uint64_t counter_inc_value,
                     uint64_t counter_remote_address, uint64_t flags,
                     ucp_device_request_t *req)
{
    return UCS_ERR_NOT_IMPLEMENTED;
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
 * The routine returns a request that can be progressed and checked for
 * completion with @ref ucp_device_progress_req.
 *
 * This routine can be called repeatedly with the same handle and different
 * mem_list_indices, addresses, lengths and increment related parameters. The
 * flags parameter can be used to modify the behavior of the routine.
 *
 * @param [in]  mem_list               Memory descriptor list handle to use.
 * @param [in]  mem_list_indices       Array of indices, to use in descriptor
 *                                     list of entries from handle.
 * @param [in]  mem_list_count         Number of indices in the array @ref
 *                                     mem_list_indices.
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
UCS_F_DEVICE ucs_status_t
ucp_device_put_multi_partial(ucp_device_mem_list_handle_h mem_list,
                             const unsigned *mem_list_indices,
                             unsigned mem_list_count,
                             void *const *addresses,
                             const uint64_t *remote_addresses,
                             const size_t *lengths,
                             uint64_t counter_inc_value,
                             uint64_t counter_remote_address,
                             uint64_t flags,
                             ucp_device_request_t *req)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}


/**
 * @ingroup UCP_DEVICE
 * @brief Progress a device request containing a batch of operations.
 *
 * This device progress function checks and progresses a request representing a
 * batch of one or many operations in progress.
 *
 * @param [in]  req  Request containing operations in progress.
 *
 * @return UCS_OK           - The request has completed, no more operations are
 *                            in progress.
 * @return UCS_INPROGRESS   - One or more operations in the request batch
 *                            have not completed.
 * @return Error code as defined by @ref ucs_status_t
 */
UCS_F_DEVICE ucs_status_t
ucp_device_progress_req(ucp_device_request_t *req)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

#endif /* UCP_DEVICE_IMPL_H */
