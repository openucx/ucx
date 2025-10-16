/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_DEVICE_IMPL_H
#define UCP_DEVICE_IMPL_H

#include "ucp_device_types.h"

#include <ucp/api/ucp_def.h>
#include <uct/api/device/uct_device_impl.h>
#include <ucs/sys/device_code.h>
#include <ucs/type/status.h>
#include <stdint.h>

/**
 * @ingroup UCP_DEVICE
 * @brief GPU request descriptor of a given batch
 *
 * This request tracks a batch of memory operations in progress. It can be used
 * with @ref ucp_device_progress_req to detect request completion.
 */
typedef struct ucp_device_request {
    uct_device_completion_t comp;
    ucs_status_t            status;
    uct_device_ep_h         device_ep;
    unsigned                channel_id;
} ucp_device_request_t;


/**
 * @ingroup UCP_DEVICE
 * @brief Specify modifier flags for device sending functions.
 */
typedef enum {
    UCP_DEVICE_FLAG_NODELAY =
            UCT_DEVICE_FLAG_NODELAY /**< Complete before return. */
} ucp_device_flags_t;


/**
 * @ingroup UCP_DEVICE
 * @brief Initialize a device request.
 *
 * @param [out] req  Device request to initialize.
 */
UCS_F_DEVICE void ucp_device_request_init(uct_device_ep_t *device_ep,
                                          ucp_device_request_t *req,
                                          uct_device_completion_t *&comp)
{
    if (req != nullptr) {
        comp           = &req->comp;
        req->device_ep = device_ep;
    } else {
        comp = nullptr;
    }
}


/**
 * Macro for device put operations with retry logic
 */
#define UCP_DEVICE_SEND_BLOCKING(_level, _uct_device_ep_send, _device_ep, \
                                 _req, ...) \
    ({ \
        ucs_status_t _status; \
        do { \
            _status = _uct_device_ep_send<_level>(_device_ep, __VA_ARGS__); \
            if (_status != UCS_ERR_NO_RESOURCE) { \
                break; \
            } \
            uct_device_ep_progress<_level>(_device_ep); \
        } while (1); \
        if (_req != nullptr) { \
            _req->status = _status; \
        } \
        _status; \
    })


UCS_F_DEVICE ucs_status_t ucp_device_prepare_send(
        ucp_device_mem_list_handle_h mem_list_h, unsigned first_mem_elem_index,
        ucp_device_request_t *req, uct_device_ep_t *&device_ep,
        const uct_device_mem_element_t *&uct_elem,
        uct_device_completion_t *&comp)
{
    const unsigned lane = 0;
    size_t elem_offset;

    if ((mem_list_h->version != UCP_DEVICE_MEM_LIST_VERSION_V1) ||
        (first_mem_elem_index >= mem_list_h->mem_list_length)) {
        return UCS_ERR_INVALID_PARAM;
    }

    device_ep   = mem_list_h->uct_device_eps[lane];
    elem_offset = first_mem_elem_index * mem_list_h->uct_mem_element_size[lane];
    uct_elem    = (uct_device_mem_element_t*)
            UCS_PTR_BYTE_OFFSET(mem_list_h->uct_mem_elements, elem_offset);
    ucp_device_request_init(device_ep, req, comp);

    return UCS_OK;
}


/**
 * @ingroup UCP_DEVICE
 * @brief Posts one memory put operation.
 *
 * This device routine posts one put operation using descriptor list handle.
 * The @a mem_list_index is used to point at the @a mem_list entry to be used
 * for the memory transfer. The @a local_offset and @a remote_offset parameters
 * specify byte offsets within the selected memory list entry. The @a length,
 * @a local_offset and @a remote_offset parameters must be valid
 * for the used @a mem_list entry.
 *
 * The routine returns a request that can be progressed and checked for
 * completion with @ref ucp_device_progress_req.
 * The routine returns only after the message has been posted or an error has occurred.
 *
 * This routine can be called repeatedly with the same handle and different
 * offsets and length. The flags parameter can be used to modify the behavior
 * of the routine with bit from @ref ucp_device_flags_t.
 *
 * @tparam      level           Level of cooperation of the transfer.
 * @param [in]  mem_list_h      Memory descriptor list handle to use.
 * @param [in]  mem_list_index  Index in descriptor list pointing to the memory
 * @param [in]  local_offset    Local offset to send data from.
 * @param [in]  remote_offset   Remote offset to send data to.
 * @param [in]  length          Length in bytes of the data to send.
 * @param [in]  channel_id      Channel ID to use for the transfer.
 * @param [in]  flags           Flags usable to modify the function behavior.
 * @param [out] req             Request populated by the call.
 *
 * @return UCS_INPROGRESS     - Operation successfully posted. If @a req is not
 *                              NULL, use @ref ucp_device_progress_req to check
 *                              for completion.
 * @return UCS_OK             - Operation completed successfully.
 * @return Error code as defined by @ref ucs_status_t
 */
template<ucs_device_level_t level = UCS_DEVICE_LEVEL_THREAD>
UCS_F_DEVICE ucs_status_t ucp_device_put_single(
        ucp_device_mem_list_handle_h mem_list_h, unsigned mem_list_index,
        size_t local_offset, size_t remote_offset, size_t length,
        unsigned channel_id, uint64_t flags, ucp_device_request_t *req)
{
    const void *address           = UCS_PTR_BYTE_OFFSET(
            mem_list_h->local_addrs[mem_list_index], local_offset);
    const uint64_t remote_address = mem_list_h->remote_addrs[mem_list_index] +
                                    remote_offset;
    const uct_device_mem_element_t *uct_elem;
    uct_device_completion_t *comp;
    uct_device_ep_t *device_ep;
    ucs_status_t status;

    status = ucp_device_prepare_send(mem_list_h, mem_list_index, req, device_ep,
                                     uct_elem, comp);
    if (status != UCS_OK) {
        return status;
    }

    return UCP_DEVICE_SEND_BLOCKING(level, uct_device_ep_put_single, device_ep,
                                    req, uct_elem, address, remote_address,
                                    length, flags, comp);
}


/**
 * @ingroup UCP_DEVICE
 * @brief Posts one memory increment operation.
 *
 * This device routine posts one increment operation using memory descriptor
 * list handle. The @ref mem_list_index is used to point at the @a mem_list
 * entry to be used for the increment operation. The remote offset must be
 * valid for the used @a mem_list entry.
 *
 * The routine returns a request that can be progressed and checked for
 * completion with @ref ucp_device_progress_req.
 *
 * This routine can be called repeatedly with the same handle and different
 * counter offset. The flags parameter can be used to modify the behavior of the
 * routine.
 *
 * @tparam      level           Level of cooperation of the transfer.
 * @param [in]  mem_list_h      Memory descriptor list handle to use.
 * @param [in]  mem_list_index  Index in descriptor list pointing to the memory
 *                              remote key to use for the increment operation.
 * @param [in]  inc_value       Value used to increment the remote address.
 * @param [in]  remote_offset   Remote offset to perform the increment to.
 * @param [in]  channel_id      Channel ID to use for the transfer.
 * @param [in]  flags           Flags usable to modify the function behavior.
 * @param [out] req             Request populated by the call.
 *
 * @return UCS_INPROGRESS     - Operation successfully posted. If @a req is not
 *                              NULL, use @ref ucp_device_progress_req to check
 *                              for completion.
 * @return UCS_OK             - Operation completed successfully.
 * @return Error code as defined by @ref ucs_status_t
 */
template<ucs_device_level_t level = UCS_DEVICE_LEVEL_THREAD>
UCS_F_DEVICE ucs_status_t ucp_device_counter_inc(
        ucp_device_mem_list_handle_h mem_list_h, unsigned mem_list_index,
        uint64_t inc_value, size_t remote_offset, unsigned channel_id,
        uint64_t flags, ucp_device_request_t *req)
{
    uint64_t remote_address = mem_list_h->remote_addrs[mem_list_index] +
                              remote_offset;
    const uct_device_mem_element_t *uct_elem;
    uct_device_completion_t *comp;
    uct_device_ep_t *device_ep;
    ucs_status_t status;

    status = ucp_device_prepare_send(mem_list_h, mem_list_index, req, device_ep,
                                     uct_elem, comp);
    if (status != UCS_OK) {
        return status;
    }

    return UCP_DEVICE_SEND_BLOCKING(level, uct_device_ep_atomic_add, device_ep,
                                    req, uct_elem, inc_value, remote_address,
                                    flags, comp);
}


/**
 * @ingroup UCP_DEVICE
 * @brief Posts multiple put operations followed by one increment operation.
 *
 * This device routine posts a batch of put operations using the descriptor list
 * entries in the input handle, followed by an increment operation if @a counter_inc_value != 0.
 * This operation can be polled on the receiver to detect completion of all the
 * operations of the batch, started during the same routine call.
 *
 * The last entry in the descriptor list contains
 * the remote memory registration descriptors to be used for the increment
 * operation.
 *
 * The routine returns a request that can be progressed and checked for
 * completion with @ref ucp_device_progress_req.
 * The routine returns only after all the messages have been posted or an error has occurred.
 *
 * This routine can be called repeatedly. The @a flags
 * parameter can be used to modify the behavior of the routine with bit from
 * @ref ucp_device_flags_t.
 *
 * @tparam      level                  Level of cooperation of the transfer.
 * @param [in]  mem_list_h             Memory descriptor list handle to use.
 * @param [in]  counter_inc_value      Value of the remote increment.
 * @param [in]  channel_id             Channel ID to use for the transfer.
 * @param [in]  flags                  Flags to modify the function behavior.
 * @param [out] req                    Request populated by the call.
 *
 * @return UCS_INPROGRESS            - Operation successfully posted. If @a req
 *                                     is not NULL, use @ref
 *                                     ucp_device_progress_req to check for
 *                                     completion.
 * @return UCS_OK                    - Operation completed successfully.
 * @return Error code as defined by @ref ucs_status_t
 */
template<ucs_device_level_t level = UCS_DEVICE_LEVEL_THREAD>
UCS_F_DEVICE ucs_status_t ucp_device_put_multi(
        ucp_device_mem_list_handle_h mem_list_h, uint64_t counter_inc_value,
        unsigned channel_id, uint64_t flags, ucp_device_request_t *req)
{
    void *const *addresses           = mem_list_h->local_addrs;
    const uint64_t *remote_addresses = mem_list_h->remote_addrs;
    const size_t *lengths            = mem_list_h->lengths;
    uint64_t counter_remote_address =
            mem_list_h->remote_addrs[mem_list_h->mem_list_length - 1];
    const uct_device_mem_element_t *uct_mem_list;
    uct_device_completion_t *comp;
    uct_device_ep_t *device_ep;
    ucs_status_t status;

    status = ucp_device_prepare_send(mem_list_h, 0, req, device_ep,
                                     uct_mem_list, comp);
    if (status != UCS_OK) {
        return status;
    }

    return UCP_DEVICE_SEND_BLOCKING(level, uct_device_ep_put_multi, device_ep,
                                    req, uct_mem_list,
                                    mem_list_h->mem_list_length, addresses,
                                    remote_addresses, lengths,
                                    counter_inc_value, counter_remote_address,
                                    flags, comp);
}


/**
 * @ingroup UCP_DEVICE
 * @brief Posts few put operations followed by one atomic increment operation.
 *
 * This device routine posts a batch of put operations using only some of the
 * descriptor list entries in the input handle, followed by
 * an increment operation if @a counter_inc_value != 0.
 * This increment operation can be polled on the receiver to detect completion
 * of all operations of the batch, started during the same routine call.
 *
 * The set of indices from the descriptor list entries to use are to be passed
 * in the array @ref mem_list_indices.
 *
 * The content of each entries in the arrays @a local_offsets, @a remote_offsets
 * and @a lengths must be valid for each corresponding descriptor list entry whose
 * index is referenced in @ref mem_list_indices.
 *
 * The size of the arrays @a mem_list_indices, @a local_offsets, @a remote_offsets, and
 * @a lengths are all equal. They are lower than the size of the descriptor list
 * array from the handle.
 *
 * The routine returns a request that can be progressed and checked for
 * completion with @ref ucp_device_progress_req.
 * The routine returns only after all the messages have been posted or an error has occurred.
 *
 * This routine can be called repeatedly with the same handle and different
 * mem_list_indices, local_offsets, remote_offsets, lengths and increment related parameters.
 * The @a flags parameter can be used to modify the behavior of the routine with bit
 * from @ref ucp_device_flags_t.
 *
 * @tparam      level                  Level of cooperation of the transfer.
 * @param [in]  mem_list_h             Memory descriptor list handle to use.
 * @param [in]  mem_list_indices       Array of indices, to use in descriptor
 *                                     list of entries from handle.
 * @param [in]  mem_list_count         Number of indices in the array @ref
 *                                     mem_list_indices.
 * @param [in]  local_offsets          Array of local offsets to send from.
 * @param [in]  remote_offsets         Array of remote offsets to send to.
 * @param [in]  lengths                Array of lengths in bytes for each send.
 * @param [in]  counter_index          Index of remote increment descriptor.
 * @param [in]  counter_inc_value      Value of the remote increment.
 * @param [in]  counter_remote_offset  Remote offset to increment to.
 * @param [in]  channel_id             Channel ID to use for the transfer.
 * @param [in]  flags                  Flags to modify the function behavior.
 * @param [out] req                    Request populated by the call.
 *
 * @return UCS_INPROGRESS            - Operation successfully posted. If @a req
 *                                     is not NULL, use @ref
 *                                     ucp_device_progress_req to check for
 *                                     completion.
 * @return UCS_OK                    - Operation completed successfully.
 * @return Error code as defined by @ref ucs_status_t
 */
template<ucs_device_level_t level = UCS_DEVICE_LEVEL_THREAD>
UCS_F_DEVICE ucs_status_t ucp_device_put_multi_partial(
        ucp_device_mem_list_handle_h mem_list_h,
        const unsigned *mem_list_indices, unsigned mem_list_count,
        const size_t *local_offsets, const size_t *remote_offsets,
        const size_t *lengths, unsigned counter_index,
        uint64_t counter_inc_value, size_t counter_remote_offset,
        unsigned channel_id, uint64_t flags, ucp_device_request_t *req)
{
    void *const *addresses           = mem_list_h->local_addrs;
    const uint64_t *remote_addresses = mem_list_h->remote_addrs;
    uint64_t counter_remote_address = mem_list_h->remote_addrs[counter_index] +
                                      counter_remote_offset;
    const uct_device_mem_element_t *uct_mem_list;
    uct_device_completion_t *comp;
    uct_device_ep_t *device_ep;
    ucs_status_t status;

    status = ucp_device_prepare_send(mem_list_h, 0, req, device_ep,
                                     uct_mem_list, comp);
    if (status != UCS_OK) {
        return status;
    }

    return UCP_DEVICE_SEND_BLOCKING(level, uct_device_ep_put_multi_partial,
                                    device_ep, req, uct_mem_list,
                                    mem_list_indices, mem_list_count, addresses,
                                    remote_addresses, local_offsets,
                                    remote_offsets, lengths, counter_index,
                                    counter_inc_value, counter_remote_address,
                                    flags, comp);
}


/**
 * @ingroup UCP_DEVICE
 * @brief Read a counter value from memory.
 *
 * This function can be used on the receiving side to detect completion of a
 * data transfer.
 *
 * The counter memory area must be initialized with the host function
 * @ref ucp_device_counter_init.
 *
 * @tparam      level       Level of cooperation of the transfer.
 * @param [in]  counter_ptr Counter memory area.
 *
 * @return value of the counter memory area, UINT64_MAX in case of error.
 */
template<ucs_device_level_t level = UCS_DEVICE_LEVEL_THREAD>
UCS_F_DEVICE uint64_t ucp_device_counter_read(const void *counter_ptr)
{
    return ucs_device_atomic64_read(
            reinterpret_cast<const uint64_t*>(counter_ptr));
}


/**
 * @ingroup UCP_DEVICE
 * @brief Write value to the counter memory area.
 *
 * This function can be used to set counter to a specific value.
 *
 * The counter memory area must be initialized with the host function
 * @ref ucp_device_counter_init.
 *
 * @tparam      level       Level of cooperation of the transfer.
 * @param [in]  counter_ptr Counter memory area.
 * @param [in]  value       Value to write.
 *
 */
template<ucs_device_level_t level = UCS_DEVICE_LEVEL_THREAD>
UCS_F_DEVICE void ucp_device_counter_write(void *counter_ptr, uint64_t value)
{
    ucs_device_atomic64_write(reinterpret_cast<uint64_t*>(counter_ptr), value);
}


/**
 * @ingroup UCP_DEVICE
 * @brief Progress a device request containing a batch of operations.
 *
 * This device progress function checks and progresses a request representing a
 * batch of one or many operations in progress.
 *
 * @tparam      level  Level of cooperation of the transfer.
 * @param [in]  req    Request containing operations in progress and channel to progress.
 *
 * @return UCS_OK           - The request has completed, no more operations are
 *                            in progress.
 * @return UCS_INPROGRESS   - One or more operations in the request batch
 *                            have not completed.
 * @return Error code as defined by @ref ucs_status_t
 */
template<ucs_device_level_t level = UCS_DEVICE_LEVEL_THREAD>
UCS_F_DEVICE ucs_status_t ucp_device_progress_req(ucp_device_request_t *req)
{
    if (ucs_likely(req->status != UCS_INPROGRESS)) {
        return req->status;
    }

    uct_device_ep_progress<level>(req->device_ep);
    req->status = uct_device_ep_check_completion<level>(req->device_ep,
                                                        &req->comp);
    return req->status;
}

#endif /* UCP_DEVICE_IMPL_H */
