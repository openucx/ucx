/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025-2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_DEVICE_IMPL_H
#define UCP_DEVICE_IMPL_H

#include "ucp_device_types.h"

#include <ucp/api/ucp_def.h>
#include <uct/api/device/uct_device_impl.h>
#include <ucs/sys/compiler_def.h>
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
 * @brief Check the parameters of the memory list handle.
 *
 * @param [in] mem_list_h  Memory list handle to check.
 * @param [in] index       Index in the memory list to check.
 *
 * @return UCS_OK if the parameters are valid.
 */
template<typename T>
UCS_F_DEVICE ucs_status_t ucp_device_check_params(const T mem_list_h,
                                                  unsigned index)
{
    if (UCP_DEVICE_ENABLE_PARAMS_CHECK) {
        if ((mem_list_h->version != UCP_DEVICE_MEM_LIST_VERSION_V1) ||
            (index >= mem_list_h->length)) {
            ucs_device_error("Invalid parameters for %p\n", mem_list_h);
            return UCS_ERR_INVALID_PARAM;
        }
    }

    return UCS_OK;
}


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


#define UCP_DEVICE_GET_LANE(_handle, _channel_id) \
    ({ \
        unsigned _lane = _channel_id % _handle->num_lanes; \
        _channel_id   /= _handle->num_lanes; \
        _lane; \
    })


#define UCP_DEVICE_GET_ELEM(_handle, _index, _lane) \
    static_cast<ucs_typeof(_handle->mem_elements[0])*>( \
            UCS_PTR_BYTE_OFFSET(_handle->mem_elements, \
                                ((_index * _handle->num_lanes) + _lane) * \
                                        sizeof(_handle->mem_elements[0])))


UCS_F_DEVICE ucs_status_t ucp_device_prepare_send_remote(
        const ucp_device_remote_mem_list_h dst_mem_list_h,
        unsigned dst_mem_list_index, uint64_t &remote_address, unsigned lane,
        ucp_device_request_t *req, uct_device_ep_t *&device_ep,
        const uct_device_mem_element_t *&uct_elem,
        uct_device_completion_t *&comp)
{
    ucs_status_t status;

    status = ucp_device_check_params(dst_mem_list_h, dst_mem_list_index);
    if (status != UCS_OK) {
        return status;
    }

    const auto dst_mem_element = UCP_DEVICE_GET_ELEM(dst_mem_list_h,
                                                     dst_mem_list_index, lane);
    remote_address             = dst_mem_element->addr;
    device_ep                  = dst_mem_element->device_ep;
    uct_elem                   = &dst_mem_element->uct_mem_element;
    ucp_device_request_init(device_ep, req, comp);

    return UCS_OK;
}


UCS_F_DEVICE ucs_status_t
ucp_device_prepare_send(const ucp_device_local_mem_list_h src_mem_list_h,
                        unsigned src_mem_list_index,
                        const ucp_device_remote_mem_list_h dst_mem_list_h,
                        unsigned dst_mem_list_index, const void *&address,
                        uint64_t &remote_address, unsigned lane,
                        ucp_device_request_t *req, uct_device_ep_t *&device_ep,
                        const uct_device_local_mem_list_elem_t *&src_uct_elem,
                        const uct_device_mem_element_t *&uct_elem,
                        uct_device_completion_t *&comp)
{
    ucs_status_t status;

    status = ucp_device_check_params(src_mem_list_h, src_mem_list_index);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_device_prepare_send_remote(dst_mem_list_h, dst_mem_list_index,
                                            remote_address, lane, req,
                                            device_ep, uct_elem, comp);
    if (status != UCS_OK) {
        return status;
    }

    src_uct_elem = UCP_DEVICE_GET_ELEM(src_mem_list_h, src_mem_list_index,
                                       lane);
    address      = src_uct_elem->addr;

    return UCS_OK;
}


/**
 * @ingroup UCP_DEVICE
 * @brief Posts one memory put operation.
 *
 * This device routine posts one put operation using descriptor list handles.
 * The @a src_mem_list_index and @a dst_mem_list_index is used to point at the
 * respected mem_list entry to be used for the memory transfer.
 * The @a local_offset and @a remote_offset parameters specify byte offsets
 * within the selected memory list entry. The @a length, @a local_offset
 * and @a remote_offset parameters must be valid for the used @a mem_list entry.
 *
 * The routine returns a request that can be progressed and checked for
 * completion with @ref ucp_device_progress_req.
 * The routine returns only after the message has been posted or an error has occurred.
 *
 * This routine can be called repeatedly with the same handles and different
 * offsets and length. The flags parameter can be used to modify the behavior
 * of the routine with bit from @ref ucp_device_flags_t.
 *
 * @tparam      level              Level of cooperation of the transfer.
 * @param [in]  src_mem_list_h     Local memory descriptor list handle to use.
 * @param [in]  src_mem_list_index Index in descriptor list pointing to the memory
 * @param [in]  src_offset         Local offset to send data from.
 * @param [in]  dst_mem_list_h     Remote memory descriptor list handle to use.
 * @param [in]  dst_mem_list_index Index in descriptor list pointing to the memory
 * @param [in]  dst_offset         Remote offset to send data to.
 * @param [in]  length             Length in bytes of the data to send.
 * @param [in]  channel_id         Channel ID to use for the transfer.
 * @param [in]  flags              Flags usable to modify the function behavior.
 * @param [out] req                Request populated by the call.
 *
 * @return UCS_INPROGRESS     - Operation successfully posted. If @a req is not
 *                              NULL, use @ref ucp_device_progress_req to check
 *                              for completion.
 * @return UCS_OK             - Operation completed successfully.
 * @return Error code as defined by @ref ucs_status_t
 */
template<ucs_device_level_t level = UCS_DEVICE_LEVEL_THREAD>
UCS_F_DEVICE ucs_status_t
ucp_device_put(const ucp_device_local_mem_list_h src_mem_list_h,
               unsigned src_mem_list_index, const size_t src_offset,
               const ucp_device_remote_mem_list_h dst_mem_list_h,
               unsigned dst_mem_list_index, size_t dst_offset, size_t length,
               unsigned channel_id, uint64_t flags, ucp_device_request_t *req)
{
    const unsigned lane = UCP_DEVICE_GET_LANE(dst_mem_list_h, channel_id);
    const void *address;
    const uct_device_mem_element_t *uct_elem;
    const uct_device_local_mem_list_elem_t *src_uct_elem;
    uint64_t remote_address;
    uct_device_completion_t *comp;
    uct_device_ep_t *device_ep;
    ucs_status_t status;

    status = ucp_device_prepare_send(src_mem_list_h, src_mem_list_index,
                                     dst_mem_list_h, dst_mem_list_index,
                                     address, remote_address, lane, req,
                                     device_ep, src_uct_elem, uct_elem, comp);
    if (status != UCS_OK) {
        return status;
    }

    return UCP_DEVICE_SEND_BLOCKING(level, uct_device_ep_put, device_ep, req,
                                    src_uct_elem, uct_elem,
                                    UCS_PTR_BYTE_OFFSET(address, src_offset),
                                    remote_address + dst_offset, length,
                                    channel_id, flags, comp);
}


/**
 * @ingroup UCP_DEVICE
 * @brief Posts one memory increment operation.
 *
 * This device routine posts one increment operation using memory descriptor
 * list handle. The @a dst_mem_list_index is used to point at the @a dst_mem_list
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
 * @tparam      level          Level of cooperation of the transfer.
 * @param [in]  inc_value      Value used to increment the remote address.
 * @param [in]  mem_list_h     Remote memory descriptor list handle to use.
 * @param [in]  mem_list_index Index in descriptor list pointing to the memory
 *                                 remote key to use for the increment operation.
 * @param [in]  offset         Remote offset to perform the increment to.
 * @param [in]  channel_id     Channel ID to use for the transfer.
 * @param [in]  flags          Flags usable to modify the function behavior.
 * @param [out] req            Request populated by the call.
 *
 * @return UCS_INPROGRESS     - Operation successfully posted. If @a req is not
 *                              NULL, use @ref ucp_device_progress_req to check
 *                              for completion.
 * @return UCS_OK             - Operation completed successfully.
 * @return Error code as defined by @ref ucs_status_t
 */
template<ucs_device_level_t level = UCS_DEVICE_LEVEL_THREAD>
UCS_F_DEVICE ucs_status_t ucp_device_counter_inc(
        const uint64_t inc_value, const ucp_device_remote_mem_list_h mem_list_h,
        unsigned mem_list_index, size_t offset, unsigned channel_id,
        uint64_t flags, ucp_device_request_t *req)
{
    const unsigned lane = UCP_DEVICE_GET_LANE(mem_list_h, channel_id);
    uint64_t remote_address;
    const uct_device_mem_element_t *uct_elem;
    uct_device_completion_t *comp;
    uct_device_ep_t *device_ep;
    ucs_status_t status;

    status = ucp_device_prepare_send_remote(mem_list_h, mem_list_index,
                                            remote_address, lane, req,
                                            device_ep, uct_elem, comp);
    if (status != UCS_OK) {
        return status;
    }

    return UCP_DEVICE_SEND_BLOCKING(level, uct_device_ep_atomic_add, device_ep,
                                    req, uct_elem, inc_value,
                                    remote_address + offset, channel_id, flags,
                                    comp);
}


/**
 * @ingroup UCP_DEVICE
 * @brief Gets a local pointer to remote memory.
 *
 * This device routine returns a local pointer to the remote memory if it is available.
 *
 * @param [in]  mem_list_h     Remote memory descriptor list handle to use.
 * @param [in]  mem_list_index Index in descriptor list pointing to the memory
 * @param [out] addr_p         Local pointer to the remote memory.
 *
 * @return UCS_OK              - Operation completed successfully.
 * @return Error code as defined by @ref ucs_status_t
 */
UCS_F_DEVICE ucs_status_t
ucp_device_get_ptr(const ucp_device_remote_mem_list_h mem_list_h,
                   unsigned mem_list_index, void **addr_p)
{
    const size_t elem_size = sizeof(uct_device_remote_mem_list_elem_t);
    ucs_status_t status;

    status = ucp_device_check_params(mem_list_h, mem_list_index);
    if (status != UCS_OK) {
        return status;
    }

    const auto mem_element = static_cast<uct_device_remote_mem_list_elem_t*>(
            UCS_PTR_BYTE_OFFSET(mem_list_h->mem_elements,
                                mem_list_index * elem_size));

    return uct_device_ep_get_ptr(mem_element->device_ep,
                                 &mem_element->uct_mem_element,
                                 mem_element->addr, addr_p);
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
