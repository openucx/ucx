/*
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_COMPAT_H_
#define UCP_COMPAT_H_


#include <ucp/api/ucp_def.h>
#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

/** @file ucp_compat.h */

/**
 * @ingroup UCP_WORKER
 * @deprecated Replaced by @ref ucp_listener_conn_handler_t.
 */
typedef struct ucp_listener_accept_handler {
   ucp_listener_accept_callback_t  cb;       /**< Endpoint creation callback */
   void                            *arg;     /**< User defined argument for the
                                                  callback */
} ucp_listener_accept_handler_t;


/**
 * @ingroup UCP_COMM
 * @deprecated Replaced by @ref ucp_request_test.
 */
int ucp_request_is_completed(void *request);


/**
 * @ingroup UCP_ENDPOINT
 * @deprecated Replaced by @ref ucp_request_free.
 */
void ucp_request_release(void *request);


/**
 * @ingroup UCP_ENDPOINT
 * @deprecated Replaced by @ref ucp_ep_close_nb.
 */
void ucp_ep_destroy(ucp_ep_h ep);


/**
 * @ingroup UCP_ENDPOINT
 * @deprecated Replaced by @ref ucp_ep_close_nb.
 */
ucs_status_ptr_t ucp_disconnect_nb(ucp_ep_h ep);


/**
 * @ingroup UCP_ENDPOINT
 * @deprecated Replaced by @ref ucp_tag_recv_request_test and
 *             @ref ucp_request_check_status depends on use case.
 *
 * @note Please use @ref ucp_request_check_status for cases that only need to
 *       check the completion status of an outstanding request.
 *       @ref ucp_request_check_status can be used for any type of request.
 *       @ref ucp_tag_recv_request_test should only be used for requests
 *       returned by @ref ucp_tag_recv_nb (or request allocated by user for
 *       @ref ucp_tag_recv_nbr) for which additional information
 *       (returned via the @a info pointer) is needed.
 */
ucs_status_t ucp_request_test(void *request, ucp_tag_recv_info_t *info);


/**
 * @ingroup UCP_MEM
 * @deprecated Replaced by @ref ucp_memh_pack "ucp_memh_pack()".
 * @brief Pack memory region remote access key.
 *
 * This routine allocates a memory buffer and packs a remote access key (RKEY)
 * object into it. RKEY is an opaque object that provides the information that is
 * necessary for remote memory access.
 * This routine packs the RKEY object in a portable format such that the
 * object can be @ref ucp_ep_rkey_unpack "unpacked" on any platform supported by the
 * UCP library. In order to release the memory buffer allocated by this routine,
 * the application is responsible for calling the @ref ucp_rkey_buffer_release
 * "ucp_rkey_buffer_release()" routine.
 *
 *
 * @note
 * @li RKEYs for InfiniBand and Cray Aries networks typically include
 * the InfiniBand and Aries key.
 * @li In order to enable remote direct memory access to the memory associated
 * with the memory handle, the application is responsible for sharing the RKEY with
 * the peers that will initiate the access.
 *
 * @param [in]  context       Application @ref ucp_context_h "context" which was
 *                            used to allocate/map the memory.
 * @param [in]  memh          @ref ucp_mem_h "Handle" to the memory region.
 * @param [out] rkey_buffer_p Memory buffer allocated by the library.
 *                            The buffer contains the packed RKEY.
 * @param [out] size_p        Size (in bytes) of the packed RKEY.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_rkey_pack(ucp_context_h context, ucp_mem_h memh,
                           void **rkey_buffer_p, size_t *size_p);


/**
 * @ingroup UCP_MEM
 * @deprecated Replaced by @ref ucp_memh_buffer_release
 *             "ucp_memh_buffer_release()".
 * @brief Release packed remote key buffer.
 *
 * This routine releases the buffer that was allocated using @ref ucp_rkey_pack
 * "ucp_rkey_pack()".
 *
 * @warning
 * @li Once memory is released, an access to the memory may cause undefined
 * behavior.
 * @li If the input memory address was not allocated using
 * @ref ucp_rkey_pack "ucp_rkey_pack()" routine, the behavior of this routine
 * is undefined.
 *
 * @param [in]  rkey_buffer   Buffer to release.
 */
void ucp_rkey_buffer_release(void *rkey_buffer);


/**
 * @ingroup UCP_ENDPOINT
 * @deprecated Replaced by @ref ucp_ep_flush_nb.
 */
ucs_status_t ucp_ep_flush(ucp_ep_h ep);

/**
 * @ingroup UCP_WORKER
 *
 * @brief Flush outstanding AMO and RMA operations on the @ref ucp_worker_h
 * "worker"
 * @deprecated Replaced by @ref ucp_worker_flush_nb. The following example
 * implements the same functionality using @ref ucp_worker_flush_nb :
 * @code
 * ucs_status_t worker_flush(ucp_worker_h worker)
 * {
 *     void *request = ucp_worker_flush_nb(worker);
 *     if (request == NULL) {
 *         return UCS_OK;
 *     } else if (UCS_PTR_IS_ERR(request)) {
 *         return UCS_PTR_STATUS(request);
 *     } else {
 *         ucs_status_t status;
 *         do {
 *             ucp_worker_progress(worker);
 *             status = ucp_request_check_status(request);
 *         } while (status == UCS_INPROGRESS);
 *         ucp_request_release(request);
 *         return status;
 *     }
 * }
 * @endcode
 *
 *
 * This routine flushes all outstanding AMO and RMA communications on the
 * @ref ucp_worker_h "worker". All the AMO and RMA operations issued on the
 * @a worker prior to this call are completed both at the origin and at the
 * target when this call returns.
 *
 * @note For description of the differences between @ref ucp_worker_flush
 * "flush" and @ref ucp_worker_fence "fence" operations please see
 * @ref ucp_worker_fence "ucp_worker_fence()"
 *
 * @param [in] worker        UCP worker.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_worker_flush(ucp_worker_h worker);


/**
 * @ingroup UCP_COMM
 * @brief Blocking remote memory put operation.
 * @deprecated Replaced by @ref ucp_put_nb. The following example implements
 * the same functionality using @ref ucp_put_nb :
 * @code
 * void empty_callback(void *request, ucs_status_t status)
 * {
 * }
 *
 * ucs_status_t put(ucp_ep_h ep, const void *buffer, size_t length,
 *                   uint64_t remote_addr, ucp_rkey_h rkey)
 * {
 *     void *request = ucp_put_nb(ep, buffer, length, remote_addr, rkey,
 *                                empty_callback),
 *     if (request == NULL) {
 *         return UCS_OK;
 *     } else if (UCS_PTR_IS_ERR(request)) {
 *         return UCS_PTR_STATUS(request);
 *     } else {
 *         ucs_status_t status;
 *         do {
 *             ucp_worker_progress(worker);
 *             status = ucp_request_check_status(request);
 *         } while (status == UCS_INPROGRESS);
 *         ucp_request_release(request);
 *         return status;
 *     }
 * }
 * @endcode
 *
 * This routine stores contiguous block of data that is described by the
 * local address @a buffer in the remote contiguous memory region described by
 * @a remote_addr address and the @ref ucp_rkey_h "memory handle" @a rkey.  The
 * routine returns when it is safe to reuse the source address @e buffer.
 *
 * @param [in]  ep           Remote endpoint handle.
 * @param [in]  buffer       Pointer to the local source address.
 * @param [in]  length       Length of the data (in bytes) stored under the
 *                           source address.
 * @param [in]  remote_addr  Pointer to the destination remote address
 *                           to write to.
 * @param [in]  rkey         Remote memory key associated with the
 *                           remote address.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_put(ucp_ep_h ep, const void *buffer, size_t length,
                     uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @brief Blocking remote memory get operation.
 * @deprecated Replaced by @ref ucp_get_nb. @see ucp_put.
 *
 * This routine loads contiguous block of data that is described by the remote
 * address @a remote_addr and the @ref ucp_rkey_h "memory handle" @a rkey in
 * the local contiguous memory region described by @a buffer address.  The
 * routine returns when remote data is loaded and stored under the local address
 * @e buffer.
 *
 *
 * @param [in]  ep           Remote endpoint handle.
 * @param [in]  buffer       Pointer to the local source address.
 * @param [in]  length       Length of the data (in bytes) stored under the
 *                           source address.
 * @param [in]  remote_addr  Pointer to the destination remote address
 *                           to write to.
 * @param [in]  rkey         Remote memory key associated with the
 *                           remote address.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_get(ucp_ep_h ep, void *buffer, size_t length,
                     uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @brief Blocking atomic add operation for 32 bit integers
 * @deprecated Replaced by @ref ucp_atomic_post with opcode UCP_ATOMIC_POST_OP_ADD.
 * @see ucp_put.
 *
 * This routine performs an add operation on a 32 bit integer value atomically.
 * The remote integer value is described by the combination of the remote
 * memory address @a remote_addr and the @ref ucp_rkey_h "remote memory handle"
 * @a rkey. The @a add value is the value that is used for the add operation.
 * When the operation completes the sum of the original remote value and the
 * operand value (@a add) is stored in remote memory.
 * The call to the routine returns immediately, independent of operation
 * completion.
 *
 * @note The remote address must be aligned to 32 bit.
 *
 * @param [in]  ep           Remote endpoint handle.
 * @param [in]  add          Value to add.
 * @param [in]  remote_addr  Pointer to the destination remote address
 *                           of the atomic variable.
 * @param [in]  rkey         Remote memory key associated with the
 *                           remote address.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_atomic_add32(ucp_ep_h ep, uint32_t add,
                              uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @brief Blocking atomic add operation for 64 bit integers
 * @deprecated Replaced by @ref ucp_atomic_post with opcode UCP_ATOMIC_POST_OP_ADD.
 * @see ucp_put.
 *
 * This routine performs an add operation on a 64 bit integer value atomically.
 * The remote integer value is described by the combination of the remote
 * memory address @a remote_addr and the @ref ucp_rkey_h "remote memory handle"
 * @a rkey. The @a add value is the value that is used for the add operation.
 * When the operation completes the sum of the original remote value and the
 * operand value (@a add) is stored in remote memory.
 * The call to the routine returns immediately, independent of operation
 * completion.
 *
 * @note The remote address must be aligned to 64 bit.
 *
 * @param [in]  ep           Remote endpoint handle.
 * @param [in]  add          Value to add.
 * @param [in]  remote_addr  Pointer to the destination remote address
 *                           of the atomic variable.
 * @param [in]  rkey         Remote memory key associated with the
 *                           remote address.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_atomic_add64(ucp_ep_h ep, uint64_t add,
                              uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @brief Blocking atomic fetch and add operation for 32 bit integers
 * @deprecated Replaced by @ref ucp_atomic_fetch_nb with opcode UCP_ATOMIC_FETCH_OP_FADD.
 * @see ucp_put.
 *
 * This routine performs an add operation on a 32 bit integer value atomically.
 * The remote integer value is described by the combination of the remote
 * memory address @a remote_addr and the @ref ucp_rkey_h "remote memory handle"
 * @a rkey. The @a add value is the value that is used for the add operation.
 * When the operation completes, the original remote value is stored in the
 * local memory @a result, and the sum of the original remote value and the
 * operand value is stored in remote memory.
 * The call to the routine returns when the operation is completed and the
 * @a result value is updated.
 *
 * @note The remote address must be aligned to 32 bit.
 *
 * @param [in]  ep           Remote endpoint handle.
 * @param [in]  add          Value to add.
 * @param [in]  remote_addr  Pointer to the destination remote address
 *                           of the atomic variable.
 * @param [in]  rkey         Remote memory key associated with the
 *                           remote address.
 * @param [out] result       Pointer to the address that is used to store
 *                           the previous value of the atomic variable described
 *                           by the @a remote_addr
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_atomic_fadd32(ucp_ep_h ep, uint32_t add, uint64_t remote_addr,
                               ucp_rkey_h rkey, uint32_t *result);


/**
 * @ingroup UCP_COMM
 * @brief Blocking atomic fetch and add operation for 64 bit integers
 * @deprecated Replaced by @ref ucp_atomic_fetch_nb with opcode UCP_ATOMIC_FETCH_OP_FADD.
 * @see ucp_put.
 *
 * This routine performs an add operation on a 64 bit integer value atomically.
 * The remote integer value is described by the combination of the remote
 * memory address @a remote_addr and the @ref ucp_rkey_h "remote memory handle"
 * @a rkey. The @a add value is the value that is used for the add operation.
 * When the operation completes, the original remote value is stored in the
 * local memory @a result, and the sum of the original remote value and the
 * operand value is stored in remote memory.
 * The call to the routine returns when the operation is completed and the
 * @a result value is updated.
 *
 * @note The remote address must be aligned to 64 bit.
 *
 * @param [in]  ep           Remote endpoint handle.
 * @param [in]  add          Value to add.
 * @param [in]  remote_addr  Pointer to the destination remote address
 *                           of the atomic variable.
 * @param [in]  rkey         Remote memory key associated with the
 *                           remote address.
 * @param [out] result       Pointer to the address that is used to store
 *                           the previous value of the atomic variable described
 *                           by the @a remote_addr
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_atomic_fadd64(ucp_ep_h ep, uint64_t add, uint64_t remote_addr,
                               ucp_rkey_h rkey, uint64_t *result);


/**
 * @ingroup UCP_COMM
 * @brief Blocking atomic swap operation for 32 bit values
 * @deprecated Replaced by @ref ucp_atomic_fetch_nb with opcode UCP_ATOMIC_FETCH_OP_SWAP.
 * @see ucp_put.
 *
 * This routine swaps a 32 bit value between local and remote memory.
 * The remote value is described by the combination of the remote
 * memory address @a remote_addr and the @ref ucp_rkey_h "remote memory handle"
 * @a rkey. The @a swap value is the value that is used for the swap operation.
 * When the operation completes, the remote value is stored in the
 * local memory @a result, and the operand value (@a swap) is stored in remote
 * memory.  The call to the routine returns when the operation is completed and
 * the @a result value is updated.
 *
 * @note The remote address must be aligned to 32 bit.
 *
 * @param [in]  ep           Remote endpoint handle.
 * @param [in]  swap         Value to swap.
 * @param [in]  remote_addr  Pointer to the destination remote address
 *                           of the atomic variable.
 * @param [in]  rkey         Remote memory key associated with the
 *                           remote address.
 * @param [out] result       Pointer to the address that is used to store
 *                           the previous value of the atomic variable described
 *                           by the @a remote_addr
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_atomic_swap32(ucp_ep_h ep, uint32_t swap, uint64_t remote_addr,
                               ucp_rkey_h rkey, uint32_t *result);


/**
 * @ingroup UCP_COMM
 * @brief Blocking atomic swap operation for 64 bit values
 * @deprecated Replaced by @ref ucp_atomic_fetch_nb with opcode UCP_ATOMIC_FETCH_OP_SWAP.
 * @see ucp_put.
 *
 * This routine swaps a 64 bit value between local and remote memory.
 * The remote value is described by the combination of the remote
 * memory address @a remote_addr and the @ref ucp_rkey_h "remote memory handle"
 * @a rkey. The @a swap value is the value that is used for the swap operation.
 * When the operation completes, the remote value is stored in the
 * local memory @a result, and the operand value (@a swap) is stored in remote
 * memory.  The call to the routine returns when the operation is completed and
 * the @a result value is updated.
 *
 * @note The remote address must be aligned to 64 bit.
 *
 * @param [in]  ep           Remote endpoint handle.
 * @param [in]  swap         Value to swap.
 * @param [in]  remote_addr  Pointer to the destination remote address
 *                           of the atomic variable.
 * @param [in]  rkey         Remote memory key associated with the
 *                           remote address.
 * @param [out] result       Pointer to the address that is used to store
 *                           the previous value of the atomic variable described
 *                           by the @a remote_addr
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_atomic_swap64(ucp_ep_h ep, uint64_t swap, uint64_t remote_addr,
                               ucp_rkey_h rkey, uint64_t *result);


/**
 * @ingroup UCP_COMM
 * @brief Blocking atomic conditional swap (cswap) operation for 32 bit values.
 * @deprecated Replaced by @ref ucp_atomic_fetch_nb with opcode UCP_ATOMIC_FETCH_OP_CSWAP.
 * @see ucp_put.
 *
 * This routine conditionally swaps a 32 bit value between local and remote
 * memory. The swap occurs only if the condition value (@a continue) is equal
 * to the remote value, otherwise the remote memory is not modified.  The
 * remote value is described by the combination of the remote memory address @p
 * remote_addr and the @ref ucp_rkey_h "remote memory handle" @a rkey. The @p
 * swap value is the value that is used to update the remote memory if the
 * condition is true.  The call to the routine returns when the operation is
 * completed and the @a result value is updated.
 *
 * @note The remote address must be aligned to 32 bit.
 *
 * @param [in]  ep           Remote endpoint handle.
 * @param [in]  compare      Value to compare to.
 * @param [in]  swap         Value to swap.
 * @param [in]  remote_addr  Pointer to the destination remote address
 *                           of the atomic variable.
 * @param [in]  rkey         Remote memory key associated with the
 *                           remote address.
 * @param [out] result       Pointer to the address that is used to store
 *                           the previous value of the atomic variable described
 *                           by the @a remote_addr
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_atomic_cswap32(ucp_ep_h ep, uint32_t compare, uint32_t swap,
                                uint64_t remote_addr, ucp_rkey_h rkey,
                                uint32_t *result);


/**
 * @ingroup UCP_COMM
 * @brief Blocking atomic conditional swap (cswap) operation for 64 bit values.
 * @deprecated Replaced by @ref ucp_atomic_fetch_nb with opcode UCP_ATOMIC_FETCH_OP_CSWAP.
 * @see ucp_put.
 *
 * This routine conditionally swaps a 64 bit value between local and remote
 * memory. The swap occurs only if the condition value (@a continue) is equal
 * to the remote value, otherwise the remote memory is not modified.  The
 * remote value is described by the combination of the remote memory address @p
 * remote_addr and the @ref ucp_rkey_h "remote memory handle" @a rkey. The @p
 * swap value is the value that is used to update the remote memory if the
 * condition is true.  The call to the routine returns when the operation is
 * completed and the @a result value is updated.
 *
 * @note The remote address must be aligned to 64 bit.
 *
 * @param [in]  ep           Remote endpoint handle.
 * @param [in]  compare      Value to compare to.
 * @param [in]  swap         Value to swap.
 * @param [in]  remote_addr  Pointer to the destination remote address
 *                           of the atomic variable.
 * @param [in]  rkey         Remote memory key associated with the
 *                           remote address.
 * @param [out] result       Pointer to the address that is used to store
 *                           the previous value of the atomic variable described
 *                           by the @a remote_addr
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_atomic_cswap64(ucp_ep_h ep, uint64_t compare, uint64_t swap,
                                uint64_t remote_addr, ucp_rkey_h rkey,
                                uint64_t *result);


/**
 * @ingroup UCP_ENDPOINT
 * @brief Modify endpoint parameters.
 *
 * @deprecated Use @ref ucp_listener_conn_handler_t instead of @ref
 *             ucp_listener_accept_handler_t, if you have other use case please
 *             submit an issue on https://github.com/openucx/ucx or report to
 *             ucx-group@elist.ornl.gov
 *
 * This routine modifies @ref ucp_ep_h "endpoint" created by @ref ucp_ep_create
 * or @ref ucp_listener_accept_callback_t. For example, this API can be used
 * to setup custom parameters like @ref ucp_ep_params_t::user_data or
 * @ref ucp_ep_params_t::err_handler to endpoint created by
 * @ref ucp_listener_accept_callback_t.
 *
 * @param [in]  ep          A handle to the endpoint.
 * @param [in]  params      User defined @ref ucp_ep_params_t configurations
 *                          for the @ref ucp_ep_h "UCP endpoint".
 *
 * @return NULL             - The endpoint is modified successfully.
 * @return UCS_PTR_IS_ERR(_ptr) - The reconfiguration failed and an error code
 *                                indicates the status. However, the @a endpoint
 *                                is not modified and can be used further.
 * @return otherwise        - The reconfiguration process is started, and can be
 *                            completed at any point in time. A request handle
 *                            is returned to the application in order to track
 *                            progress of the endpoint modification.
 *                            The application is responsible for releasing the
 *                            handle using the @ref ucp_request_free routine.
 *
 * @note See the documentation of @ref ucp_ep_params_t for details, only some of
 *       the parameters can be modified.
 */
ucs_status_ptr_t ucp_ep_modify_nb(ucp_ep_h ep, const ucp_ep_params_t *params);


/**
 * @ingroup UCP_WORKER
 * @brief Get the address of the worker object.
 *
 * @deprecated Use @ref ucp_worker_query with the flag
 *             @ref UCP_WORKER_ATTR_FIELD_ADDRESS in order to obtain the worker
 *             address.
 *
 * This routine returns the address of the worker object.  This address can be
 * passed to remote instances of the UCP library in order to connect to this
 * worker. The memory for the address handle is allocated by this function, and
 * must be released by using @ref ucp_worker_release_address
 * "ucp_worker_release_address()" routine.
 *
 * @param [in]  worker            Worker object whose address to return.
 * @param [out] address_p         A pointer to the worker address.
 * @param [out] address_length_p  The size in bytes of the address.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_worker_get_address(ucp_worker_h worker,
                                    ucp_address_t **address_p,
                                    size_t *address_length_p);


/**
 * @ingroup UCP_ENDPOINT
 *
 * @brief Non-blocking @ref ucp_ep_h "endpoint" closure.
 *
 * @deprecated Use @ref ucp_ep_close_nbx instead.
 *
 * This routine releases the @ref ucp_ep_h "endpoint". The endpoint closure
 * process depends on the selected @a mode.
 *
 * @param [in]  ep      Handle to the endpoint to close.
 * @param [in]  mode    One from @ref ucp_ep_close_mode value.
 *
 * @return UCS_OK           - The endpoint is closed successfully.
 * @return UCS_PTR_IS_ERR(_ptr) - The closure failed and an error code indicates
 *                                the transport level status. However, resources
 *                                are released and the @a endpoint can no longer
 *                                be used.
 * @return otherwise        - The closure process is started, and can be
 *                            completed at any point in time. A request handle
 *                            is returned to the application in order to track
 *                            progress of the endpoint closure. The application
 *                            is responsible for releasing the handle using the
 *                            @ref ucp_request_free routine.
 *
 * @note @ref ucp_ep_close_nb replaces deprecated @ref ucp_disconnect_nb and
 *       @ref ucp_ep_destroy
 */
ucs_status_ptr_t ucp_ep_close_nb(ucp_ep_h ep, unsigned mode);


/**
 * @ingroup UCP_ENDPOINT
 *
 * @brief Non-blocking flush of outstanding AMO and RMA operations on the
 * @ref ucp_ep_h "endpoint".
 *
 * @deprecated Use @ref ucp_ep_flush_nbx instead.
 *
 * This routine flushes all outstanding AMO and RMA communications on the
 * @ref ucp_ep_h "endpoint". All the AMO and RMA operations issued on the
 * @a ep prior to this call are completed both at the origin and at the target
 * @ref ucp_ep_h "endpoint" when this call returns.
 *
 * @param [in] ep        UCP endpoint.
 * @param [in] flags     Flags for flush operation. Reserved for future use.
 * @param [in] cb        Callback which will be called when the flush operation
 *                       completes.
 *
 * @return NULL             - The flush operation was completed immediately.
 * @return UCS_PTR_IS_ERR(_ptr) - The flush operation failed.
 * @return otherwise        - Flush operation was scheduled and can be completed
 *                          in any point in time. The request handle is returned
 *                          to the application in order to track progress. The
 *                          application is responsible for releasing the handle
 *                          using @ref ucp_request_free "ucp_request_free()"
 *                          routine.
 */
ucs_status_ptr_t ucp_ep_flush_nb(ucp_ep_h ep, unsigned flags,
                                 ucp_send_callback_t cb);


/**
 * @ingroup UCP_WORKER
 * @brief Add user defined callback for Active Message.
 *
 * @deprecated Use @ref ucp_worker_set_am_recv_handler instead.
 *
 * This routine installs a user defined callback to handle incoming Active
 * Messages with a specific id. This callback is called whenever an Active
 * Message that was sent from the remote peer by @ref ucp_am_send_nb is
 * received on this worker.
 *
 * @param [in]  worker      UCP worker on which to set the Active Message
 *                          handler.
 * @param [in]  id          Active Message id.
 * @param [in]  cb          Active Message callback. NULL to clear.
 * @param [in]  arg         Active Message argument, which will be passed
 *                          in to every invocation of the callback as the
 *                          arg argument.
 * @param [in]  flags       Dictates how an Active Message is handled on the
 *                          remote endpoint. Currently only
 *                          UCP_AM_FLAG_WHOLE_MSG is supported, which
 *                          indicates the callback will not be invoked
 *                          until all data has arrived.
 *
 * @return error code if the worker does not support Active Messages or
 *         requested callback flags.
 */
ucs_status_t ucp_worker_set_am_handler(ucp_worker_h worker, uint16_t id,
                                       ucp_am_callback_t cb, void *arg,
                                       uint32_t flags);


/**
 * @ingroup UCP_COMM
 * @brief Send Active Message.
 *
 * @deprecated Use @ref ucp_am_send_nbx instead.
 *
 * This routine sends an Active Message to an ep. It does not support
 * CUDA memory.
 *
 * @param [in]  ep          UCP endpoint where the Active Message will be run.
 * @param [in]  id          Active Message id. Specifies which registered
 *                          callback to run.
 * @param [in]  buffer      Pointer to the data to be sent to the target node
 *                          of the Active Message.
 * @param [in]  count       Number of elements to send.
 * @param [in]  datatype    Datatype descriptor for the elements in the buffer.
 * @param [in]  cb          Callback that is invoked upon completion of the
 *                          data transfer if it is not completed immediately.
 * @param [in]  flags       Operation flags as defined by @ref ucp_send_am_flags.
 *
 * @return NULL             Active Message was sent immediately.
 * @return UCS_PTR_IS_ERR(_ptr) Error sending Active Message.
 * @return otherwise        Pointer to request, and Active Message is known
 *                          to be completed after cb is run.
 */
ucs_status_ptr_t ucp_am_send_nb(ucp_ep_h ep, uint16_t id,
                                const void *buffer, size_t count,
                                ucp_datatype_t datatype,
                                ucp_send_callback_t cb, unsigned flags);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking stream send operation.
 *
 * @deprecated Use @ref ucp_stream_send_nbx instead.
 *
 * This routine sends data that is described by the local address @a buffer,
 * size @a count, and @a datatype object to the destination endpoint @a ep.
 * The routine is non-blocking and therefore returns immediately, however
 * the actual send operation may be delayed. The send operation is considered
 * completed when it is safe to reuse the source @e buffer. If the send
 * operation is completed immediately the routine returns UCS_OK and the
 * callback function @a cb is @b not invoked. If the operation is
 * @b not completed immediately and no error reported, then the UCP library will
 * schedule invocation of the callback @a cb upon completion of the send
 * operation. In other words, the completion of the operation will be signaled
 * either by the return code or by the callback.
 *
 * @note The user should not modify any part of the @a buffer after this
 *       operation is called, until the operation completes.
 *
 * @param [in]  ep          Destination endpoint handle.
 * @param [in]  buffer      Pointer to the message buffer (payload).
 * @param [in]  count       Number of elements to send.
 * @param [in]  datatype    Datatype descriptor for the elements in the buffer.
 * @param [in]  cb          Callback function that is invoked whenever the
 *                          send operation is completed. It is important to note
 *                          that the callback is only invoked in the event that
 *                          the operation cannot be completed in place.
 * @param [in]  flags       Reserved for future use.
 *
 * @return NULL             - The send operation was completed immediately.
 * @return UCS_PTR_IS_ERR(_ptr) - The send operation failed.
 * @return otherwise        - Operation was scheduled for send and can be
 *                          completed in any point in time. The request handle
 *                          is returned to the application in order to track
 *                          progress of the message. The application is
 *                          responsible for releasing the handle using
 *                          @ref ucp_request_free routine.
 */
ucs_status_ptr_t ucp_stream_send_nb(ucp_ep_h ep, const void *buffer, size_t count,
                                    ucp_datatype_t datatype, ucp_send_callback_t cb,
                                    unsigned flags);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking stream receive operation of structured data into a
 *        user-supplied buffer.
 *
 * @deprecated Use @ref ucp_stream_recv_nbx instead.
 *
 * This routine receives data that is described by the local address @a buffer,
 * size @a count, and @a datatype object on the endpoint @a ep. The routine is
 * non-blocking and therefore returns immediately. The receive operation is
 * considered complete when the message is delivered to the buffer. If data is
 * not immediately available, the operation will be scheduled for receive and
 * a request handle will be returned. In order to notify the application about
 * completion of a scheduled receive operation, the UCP library will invoke
 * the call-back @a cb when data is in the receive buffer and ready for
 * application access. If the receive operation cannot be started, the routine
 * returns an error.
 *
 * @param [in]     ep       UCP endpoint that is used for the receive operation.
 * @param [in]     buffer   Pointer to the buffer to receive the data.
 * @param [in]     count    Number of elements to receive into @a buffer.
 * @param [in]     datatype Datatype descriptor for the elements in the buffer.
 * @param [in]     cb       Callback function that is invoked whenever the
 *                          receive operation is completed and the data is ready
 *                          in the receive @a buffer. It is important to note
 *                          that the call-back is only invoked in a case when
 *                          the operation cannot be completed immediately.
 * @param [out]    length   Size of the received data in bytes. The value is
 *                          valid only if return code is UCS_OK.
 * @note                    The amount of data received, in bytes, is always an
 *                          integral multiple of the @a datatype size.
 * @param [in]     flags    Flags defined in @ref ucp_stream_recv_flags_t.
 *
 * @return NULL                 - The receive operation was completed
 *                                immediately.
 * @return UCS_PTR_IS_ERR(_ptr) - The receive operation failed.
 * @return otherwise            - Operation was scheduled for receive. A request
 *                                handle is returned to the application in order
 *                                to track progress of the operation.
 *                                The application is responsible for releasing
 *                                the handle by calling the
 *                                @ref ucp_request_free routine.
 */
ucs_status_ptr_t ucp_stream_recv_nb(ucp_ep_h ep, void *buffer, size_t count,
                                    ucp_datatype_t datatype,
                                    ucp_stream_recv_callback_t cb,
                                    size_t *length, unsigned flags);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking tagged-send operations
 *
 * @deprecated Use @ref ucp_tag_send_nbx instead.
 *
 * This routine sends a messages that is described by the local address @a
 * buffer, size @a count, and @a datatype object to the destination endpoint
 * @a ep. Each message is associated with a @a tag value that is used for
 * message matching on the @ref ucp_tag_recv_nb "receiver". The routine is
 * non-blocking and therefore returns immediately, however the actual send
 * operation may be delayed. The send operation is considered completed when
 * it is safe to reuse the source @e buffer. If the send operation is
 * completed immediately the routine return UCS_OK and the call-back function
 * @a cb is @b not invoked. If the operation is @b not completed immediately
 * and no error reported then the UCP library will schedule to invoke the
 * call-back @a cb whenever the send operation will be completed. In other
 * words, the completion of a message can be signaled by the return code or
 * the call-back.
 *
 * @note The user should not modify any part of the @a buffer after this
 *       operation is called, until the operation completes.
 *
 * @param [in]  ep          Destination endpoint handle.
 * @param [in]  buffer      Pointer to the message buffer (payload).
 * @param [in]  count       Number of elements to send
 * @param [in]  datatype    Datatype descriptor for the elements in the buffer.
 * @param [in]  tag         Message tag.
 * @param [in]  cb          Callback function that is invoked whenever the
 *                          send operation is completed. It is important to note
 *                          that the call-back is only invoked in a case when
 *                          the operation cannot be completed in place.
 *
 * @return NULL            - The send operation was completed immediately.
 * @return UCS_PTR_IS_ERR(_ptr) - The send operation failed.
 * @return otherwise        - Operation was scheduled for send and can be
 *                          completed in any point in time. The request handle
 *                          is returned to the application in order to track
 *                          progress of the message. The application is
 *                          responsible for releasing the handle using
 *                          @ref ucp_request_free "ucp_request_free()" routine.
 */
ucs_status_ptr_t ucp_tag_send_nb(ucp_ep_h ep, const void *buffer, size_t count,
                                 ucp_datatype_t datatype, ucp_tag_t tag,
                                 ucp_send_callback_t cb);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking tagged-send operations with user provided request
 *
 * @deprecated Use @ref ucp_tag_send_nbx with the flag
 *             @ref UCP_OP_ATTR_FIELD_REQUEST instead.
 *
 * This routine provides a convenient and efficient way to implement a
 * blocking send pattern. It also completes requests faster than
 * @ref ucp_tag_send_nb() because:
 * @li it always uses eager protocol to send data up to the
 *     rendezvous threshold.
 * @li its rendezvous threshold is higher than the one used by
 *     the @ref ucp_tag_send_nb(). The threshold is controlled by
 *     the @b UCX_SEND_NBR_RNDV_THRESH environment variable.
 * @li its request handling is simpler. There is no callback and no need
 *     to allocate and free requests. In fact request can be allocated by
 *     caller on the stack.
 *
 * This routine sends a messages that is described by the local address @a
 * buffer, size @a count, and @a datatype object to the destination endpoint
 * @a ep. Each message is associated with a @a tag value that is used for
 * message matching on the @ref ucp_tag_recv_nbr "receiver".
 *
 * The routine is non-blocking and therefore returns immediately, however
 * the actual send operation may be delayed. The send operation is considered
 * completed when it is safe to reuse the source @e buffer. If the send
 * operation is completed immediately the routine returns UCS_OK.
 *
 * If the operation is @b not completed immediately and no error reported
 * then the UCP library will fill a user provided @a req and
 * return UCS_INPROGRESS status. In order to monitor completion of the
 * operation @ref ucp_request_check_status() should be used.
 *
 * Following pseudo code implements a blocking send function:
 * @code
 * MPI_send(...)
 * {
 *     char *request;
 *     ucs_status_t status;
 *
 *     // allocate request on the stack
 *     // ucp_context_query() was used to get ucp_request_size
 *     request = alloca(ucp_request_size);
 *
 *     // note: make sure that there is enough memory before the
 *     // request handle
 *     status = ucp_tag_send_nbr(ep, ..., request + ucp_request_size);
 *     if (status != UCS_INPROGRESS) {
 *         return status;
 *     }
 *
 *     do {
 *         ucp_worker_progress(worker);
 *         status = ucp_request_check_status(request + ucp_request_size);
 *     } while (status == UCS_INPROGRESS);
 *
 *     return status;
 * }
 * @endcode
 *
 * @note The user should not modify any part of the @a buffer after this
 *       operation is called, until the operation completes.
 *
 *
 * @param [in]  ep          Destination endpoint handle.
 * @param [in]  buffer      Pointer to the message buffer (payload).
 * @param [in]  count       Number of elements to send
 * @param [in]  datatype    Datatype descriptor for the elements in the buffer.
 * @param [in]  tag         Message tag.
 * @param [in]  req         Request handle allocated by the user. There should
 *                          be at least UCP request size bytes of available
 *                          space before the @a req. The size of UCP request
 *                          can be obtained by @ref ucp_context_query function.
 *
 * @return UCS_OK           - The send operation was completed immediately.
 * @return UCS_INPROGRESS   - The send was not completed and is in progress.
 *                            @ref ucp_request_check_status() should be used to
 *                            monitor @a req status.
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_tag_send_nbr(ucp_ep_h ep, const void *buffer, size_t count,
                              ucp_datatype_t datatype, ucp_tag_t tag, void *req);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking synchronous tagged-send operation.
 *
 * @deprecated Use @ref ucp_tag_send_sync_nbx instead.
 *
 * Same as @ref ucp_tag_send_nb, except the request completes only after there
 * is a remote tag match on the message (which does not always mean the remote
 * receive has been completed). This function never completes "in-place", and
 * always returns a request handle.
 *
 * @note The user should not modify any part of the @a buffer after this
 *       operation is called, until the operation completes.
 * @note Returns @ref UCS_ERR_UNSUPPORTED if @ref UCP_ERR_HANDLING_MODE_PEER is
 *       enabled. This is a temporary implementation-related constraint that
 *       will be addressed in future releases.
 *
 * @param [in]  ep          Destination endpoint handle.
 * @param [in]  buffer      Pointer to the message buffer (payload).
 * @param [in]  count       Number of elements to send
 * @param [in]  datatype    Datatype descriptor for the elements in the buffer.
 * @param [in]  tag         Message tag.
 * @param [in]  cb          Callback function that is invoked whenever the
 *                          send operation is completed.
 *
 * @return UCS_PTR_IS_ERR(_ptr) - The send operation failed.
 * @return otherwise        - Operation was scheduled for send and can be
 *                          completed in any point in time. The request handle
 *                          is returned to the application in order to track
 *                          progress of the message. The application is
 *                          responsible for releasing the handle using
 *                          @ref ucp_request_free "ucp_request_free()" routine.
 */
ucs_status_ptr_t ucp_tag_send_sync_nb(ucp_ep_h ep, const void *buffer, size_t count,
                                      ucp_datatype_t datatype, ucp_tag_t tag,
                                      ucp_send_callback_t cb);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking tagged-receive operation.
 *
 * @deprecated Use @ref ucp_tag_recv_nbx instead.
 *
 * This routine receives a message that is described by the local address @a
 * buffer, size @a count, and @a datatype object on the @a worker. The tag
 * value of the receive message has to match the @a tag and @a tag_mask values,
 * where the @a tag_mask indicates which bits of the tag have to be matched. The
 * routine is non-blocking and therefore returns immediately. The receive
 * operation is considered completed when the message is delivered to the @a
 * buffer.  In order to notify the application about completion of the receive
 * operation the UCP library will invoke the call-back @a cb when the received
 * message is in the receive buffer and ready for application access.  If the
 * receive operation cannot be stated the routine returns an error.
 *
 * @note This routine cannot return UCS_OK. It always returns a request
 *       handle or an error.
 *
 * @param [in]  worker      UCP worker that is used for the receive operation.
 * @param [in]  buffer      Pointer to the buffer to receive the data.
 * @param [in]  count       Number of elements to receive
 * @param [in]  datatype    Datatype descriptor for the elements in the buffer.
 * @param [in]  tag         Message tag to expect.
 * @param [in]  tag_mask    Bit mask that indicates the bits that are used for
 *                          the matching of the incoming tag
 *                          against the expected tag.
 * @param [in]  cb          Callback function that is invoked whenever the
 *                          receive operation is completed and the data is ready
 *                          in the receive @a buffer.
 *
 * @return UCS_PTR_IS_ERR(_ptr) - The receive operation failed.
 * @return otherwise          - Operation was scheduled for receive. The request
 *                              handle is returned to the application in order
 *                              to track progress of the operation. The
 *                              application is responsible for releasing the
 *                              handle using @ref ucp_request_free
 *                              "ucp_request_free()" routine.
 */
ucs_status_ptr_t ucp_tag_recv_nb(ucp_worker_h worker, void *buffer, size_t count,
                                 ucp_datatype_t datatype, ucp_tag_t tag,
                                 ucp_tag_t tag_mask, ucp_tag_recv_callback_t cb);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking tagged-receive operation.
 *
 * @deprecated Use @ref ucp_tag_recv_nbx with the flag
 *             @ref UCP_OP_ATTR_FIELD_REQUEST instead.
 *
 * This routine receives a message that is described by the local address @a
 * buffer, size @a count, and @a datatype object on the @a worker. The tag
 * value of the receive message has to match the @a tag and @a tag_mask values,
 * where the @a tag_mask indicates which bits of the tag have to be matched. The
 * routine is non-blocking and therefore returns immediately. The receive
 * operation is considered completed when the message is delivered to the @a
 * buffer. In order to monitor completion of the operation
 * @ref ucp_request_check_status or @ref ucp_tag_recv_request_test should be
 * used.
 *
 * @param [in]  worker      UCP worker that is used for the receive operation.
 * @param [in]  buffer      Pointer to the buffer to receive the data.
 * @param [in]  count       Number of elements to receive
 * @param [in]  datatype    Datatype descriptor for the elements in the buffer.
 * @param [in]  tag         Message tag to expect.
 * @param [in]  tag_mask    Bit mask that indicates the bits that are used for
 *                          the matching of the incoming tag
 *                          against the expected tag.
 * @param [in]  req         Request handle allocated by the user. There should
 *                          be at least UCP request size bytes of available
 *                          space before the @a req. The size of UCP request
 *                          can be obtained by @ref ucp_context_query function.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_tag_recv_nbr(ucp_worker_h worker, void *buffer, size_t count,
                              ucp_datatype_t datatype, ucp_tag_t tag,
                              ucp_tag_t tag_mask, void *req);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking receive operation for a probed message.
 *
 * @deprecated Use @ref ucp_tag_recv_nbx instead.
 *
 * This routine receives a message that is described by the local address @a
 * buffer, size @a count, @a message handle, and @a datatype object on the @a
 * worker. The @a message handle can be obtained by calling the @ref
 * ucp_tag_probe_nb "ucp_tag_probe_nb()" routine. The @ref ucp_tag_msg_recv_nb
 * "ucp_tag_msg_recv_nb()" routine is non-blocking and therefore returns
 * immediately. The receive operation is considered completed when the message
 * is delivered to the @a buffer. In order to notify the application about
 * completion of the receive operation the UCP library will invoke the
 * call-back @a cb when the received message is in the receive buffer and ready
 * for application access. If the receive operation cannot be started the
 * routine returns an error.
 *
 * @param [in]  worker      UCP worker that is used for the receive operation.
 * @param [in]  buffer      Pointer to the buffer that will receive the data.
 * @param [in]  count       Number of elements to receive
 * @param [in]  datatype    Datatype descriptor for the elements in the buffer.
 * @param [in]  message     Message handle.
 * @param [in]  cb          Callback function that is invoked whenever the
 *                          receive operation is completed and the data is ready
 *                          in the receive @a buffer.
 *
 * @return UCS_PTR_IS_ERR(_ptr) - The receive operation failed.
 * @return otherwise          - Operation was scheduled for receive. The request
 *                              handle is returned to the application in order
 *                              to track progress of the operation. The
 *                              application is responsible for releasing the
 *                              handle using @ref ucp_request_free
 *                              "ucp_request_free()" routine.
 */
ucs_status_ptr_t ucp_tag_msg_recv_nb(ucp_worker_h worker, void *buffer,
                                     size_t count, ucp_datatype_t datatype,
                                     ucp_tag_message_h message,
                                     ucp_tag_recv_callback_t cb);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking implicit remote memory put operation.
 *
 * @deprecated Use @ref ucp_put_nbx without passing the flag
 *             @ref UCP_OP_ATTR_FIELD_CALLBACK instead. If a request pointer
 *             is returned, release it immediately by @ref ucp_request_free.
 *
 * This routine initiates a storage of contiguous block of data that is
 * described by the local address @a buffer in the remote contiguous memory
 * region described by @a remote_addr address and the @ref ucp_rkey_h "memory
 * handle" @a rkey. The routine returns immediately and @b does @b not
 * guarantee re-usability of the source address @e buffer. If the operation is
 * completed immediately the routine return UCS_OK, otherwise UCS_INPROGRESS
 * or an error is returned to user.
 *
 * @note A user can use @ref ucp_worker_flush_nb "ucp_worker_flush_nb()"
 * in order to guarantee re-usability of the source address @e buffer.
 *
 * @param [in]  ep           Remote endpoint handle.
 * @param [in]  buffer       Pointer to the local source address.
 * @param [in]  length       Length of the data (in bytes) stored under the
 *                           source address.
 * @param [in]  remote_addr  Pointer to the destination remote memory address
 *                           to write to.
 * @param [in]  rkey         Remote memory key associated with the
 *                           remote memory address.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_put_nbi(ucp_ep_h ep, const void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking remote memory put operation.
 *
 * @deprecated Use @ref ucp_put_nbx instead.
 *
 * This routine initiates a storage of contiguous block of data that is
 * described by the local address @a buffer in the remote contiguous memory
 * region described by @a remote_addr address and the @ref ucp_rkey_h "memory
 * handle" @a rkey.  The routine returns immediately and @b does @b not
 * guarantee re-usability of the source address @e buffer. If the operation is
 * completed immediately the routine return UCS_OK, otherwise UCS_INPROGRESS
 * or an error is returned to user. If the put operation completes immediately,
 * the routine returns UCS_OK and the call-back routine @a cb is @b not
 * invoked. If the operation is @b not completed immediately and no error is
 * reported, then the UCP library will schedule invocation of the call-back
 * routine @a cb upon completion of the put operation. In other words, the
 * completion of a put operation can be signaled by the return code or
 * execution of the call-back.
 *
 * @note A user can use @ref ucp_worker_flush_nb "ucp_worker_flush_nb()"
 * in order to guarantee re-usability of the source address @e buffer.
 *
 * @param [in]  ep           Remote endpoint handle.
 * @param [in]  buffer       Pointer to the local source address.
 * @param [in]  length       Length of the data (in bytes) stored under the
 *                           source address.
 * @param [in]  remote_addr  Pointer to the destination remote memory address
 *                           to write to.
 * @param [in]  rkey         Remote memory key associated with the
 *                           remote memory address.
 * @param [in]  cb           Call-back function that is invoked whenever the
 *                           put operation is completed and the local buffer
 *                           can be modified. Does not guarantee remote
 *                           completion.
 *
 * @return NULL                 - The operation was completed immediately.
 * @return UCS_PTR_IS_ERR(_ptr) - The operation failed.
 * @return otherwise            - Operation was scheduled and can be
 *                              completed at any point in time. The request handle
 *                              is returned to the application in order to track
 *                              progress of the operation. The application is
 *                              responsible for releasing the handle using
 *                              @ref ucp_request_free "ucp_request_free()" routine.
 */
ucs_status_ptr_t ucp_put_nb(ucp_ep_h ep, const void *buffer, size_t length,
                            uint64_t remote_addr, ucp_rkey_h rkey,
                            ucp_send_callback_t cb);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking implicit remote memory get operation.
 *
 * @deprecated Use @ref ucp_get_nbx without passing the flag
 *             @ref UCP_OP_ATTR_FIELD_CALLBACK instead. If a request pointer
 *             is returned, release it immediately by @ref ucp_request_free.
 *
 * This routine initiate a load of contiguous block of data that is described
 * by the remote memory address @a remote_addr and the @ref ucp_rkey_h "memory handle"
 * @a rkey in the local contiguous memory region described by @a buffer
 * address. The routine returns immediately and @b does @b not guarantee that
 * remote data is loaded and stored under the local address @e buffer.
 *
 * @note A user can use @ref ucp_worker_flush_nb "ucp_worker_flush_nb()" in order
 * guarantee that remote data is loaded and stored under the local address
 * @e buffer.
 *
 * @param [in]  ep           Remote endpoint handle.
 * @param [in]  buffer       Pointer to the local destination address.
 * @param [in]  length       Length of the data (in bytes) stored under the
 *                           destination address.
 * @param [in]  remote_addr  Pointer to the source remote memory address
 *                           to read from.
 * @param [in]  rkey         Remote memory key associated with the
 *                           remote memory address.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_get_nbi(ucp_ep_h ep, void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking remote memory get operation.
 *
 * @deprecated Use @ref ucp_get_nbx instead.
 *
 * This routine initiates a load of a contiguous block of data that is
 * described by the remote memory address @a remote_addr and the @ref ucp_rkey_h
 * "memory handle" @a rkey in the local contiguous memory region described
 * by @a buffer address. The routine returns immediately and @b does @b not
 * guarantee that remote data is loaded and stored under the local address @e
 * buffer. If the operation is completed immediately the routine return UCS_OK,
 * otherwise UCS_INPROGRESS or an error is returned to user. If the get
 * operation completes immediately, the routine returns UCS_OK and the
 * call-back routine @a cb is @b not invoked. If the operation is @b not
 * completed immediately and no error is reported, then the UCP library will
 * schedule invocation of the call-back routine @a cb upon completion of the
 * get operation. In other words, the completion of a get operation can be
 * signaled by the return code or execution of the call-back.
 *
 * @note A user can use @ref ucp_worker_flush_nb "ucp_worker_flush_nb()"
 * in order to guarantee re-usability of the source address @e buffer.
 *
 * @param [in]  ep           Remote endpoint handle.
 * @param [in]  buffer       Pointer to the local destination address.
 * @param [in]  length       Length of the data (in bytes) stored under the
 *                           destination address.
 * @param [in]  remote_addr  Pointer to the source remote memory address
 *                           to read from.
 * @param [in]  rkey         Remote memory key associated with the
 *                           remote memory address.
 * @param [in]  cb           Call-back function that is invoked whenever the
 *                           get operation is completed and the data is
 *                           visible to the local process.
 *
 * @return NULL                 - The operation was completed immediately.
 * @return UCS_PTR_IS_ERR(_ptr) - The operation failed.
 * @return otherwise            - Operation was scheduled and can be
 *                              completed at any point in time. The request handle
 *                              is returned to the application in order to track
 *                              progress of the operation. The application is
 *                              responsible for releasing the handle using
 *                              @ref ucp_request_free "ucp_request_free()" routine.
 */
ucs_status_ptr_t ucp_get_nb(ucp_ep_h ep, void *buffer, size_t length,
                            uint64_t remote_addr, ucp_rkey_h rkey,
                            ucp_send_callback_t cb);


/**
 * @ingroup UCP_COMM
 * @brief Atomic operation requested for ucp_atomic_post
 *
 * @deprecated Use @ref ucp_atomic_op_nbx and @ref ucp_atomic_op_t instead.
 *
 * This enumeration defines which atomic memory operation should be
 * performed by the ucp_atomic_post family of functions. All of these are
 * non-fetching atomics and will not result in a request handle.
 */
typedef enum {
    UCP_ATOMIC_POST_OP_ADD, /**< Atomic add */
    UCP_ATOMIC_POST_OP_AND, /**< Atomic and */
    UCP_ATOMIC_POST_OP_OR,  /**< Atomic or  */
    UCP_ATOMIC_POST_OP_XOR, /**< Atomic xor */
    UCP_ATOMIC_POST_OP_LAST
} ucp_atomic_post_op_t;


/**
 * @ingroup UCP_COMM
 * @brief Post an atomic memory operation.
 *
 * @deprecated Use @ref ucp_atomic_op_nbx without the flag
 *             @ref UCP_OP_ATTR_FIELD_REPLY_BUFFER instead.
 *
 * This routine posts an atomic memory operation to a remote value.
 * The remote value is described by the combination of the remote
 * memory address @a remote_addr and the @ref ucp_rkey_h "remote memory handle"
 * @a rkey.
 * Return from the function does not guarantee completion. A user must
 * call @ref ucp_ep_flush_nb or @ref ucp_worker_flush_nb to guarantee that the
 * remote value has been updated.
 *
 * @param [in] ep          UCP endpoint.
 * @param [in] opcode      One of @ref ucp_atomic_post_op_t.
 * @param [in] value       Source operand for the atomic operation.
 * @param [in] op_size     Size of value in bytes
 * @param [in] remote_addr Remote address to operate on.
 * @param [in] rkey        Remote key handle for the remote memory address.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_atomic_post(ucp_ep_h ep, ucp_atomic_post_op_t opcode, uint64_t value,
                             size_t op_size, uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @brief Atomic operation requested for ucp_atomic_fetch
 *
 * @deprecated Use @ref ucp_atomic_op_nbx and @ref ucp_atomic_op_t instead.
 *
 * This enumeration defines which atomic memory operation should be performed
 * by the ucp_atomic_fetch family of functions. All of these functions
 * will fetch data from the remote node.
 */
typedef enum {
    UCP_ATOMIC_FETCH_OP_FADD,  /**< Atomic Fetch and add    */
    UCP_ATOMIC_FETCH_OP_SWAP,  /**< Atomic swap             */
    UCP_ATOMIC_FETCH_OP_CSWAP, /**< Atomic conditional swap */
    UCP_ATOMIC_FETCH_OP_FAND,  /**< Atomic Fetch and and    */
    UCP_ATOMIC_FETCH_OP_FOR,   /**< Atomic Fetch and or     */
    UCP_ATOMIC_FETCH_OP_FXOR,  /**< Atomic Fetch and xor    */
    UCP_ATOMIC_FETCH_OP_LAST
} ucp_atomic_fetch_op_t;


/**
 * @ingroup UCP_COMM
 * @brief Post an atomic fetch operation.
 *
 * @deprecated Use @ref ucp_atomic_op_nbx with the flag
 *             @ref UCP_OP_ATTR_FIELD_REPLY_BUFFER instead.
 *
 * This routine will post an atomic fetch operation to remote memory.
 * The remote value is described by the combination of the remote
 * memory address @a remote_addr and the @ref ucp_rkey_h "remote memory handle"
 * @a rkey.
 * The routine is non-blocking and therefore returns immediately. However the
 * actual atomic operation may be delayed. The atomic operation is not considered complete
 * until the values in remote and local memory are completed. If the atomic operation
 * completes immediately, the routine returns UCS_OK and the call-back routine
 * @a cb is @b not invoked. If the operation is @b not completed immediately and no
 * error is reported, then the UCP library will schedule invocation of the call-back
 * routine @a cb upon completion of the atomic operation. In other words, the completion
 * of an atomic operation can be signaled by the return code or execution of the call-back.
 *
 * @note The user should not modify any part of the @a result after this
 *       operation is called, until the operation completes.
 *
 * @param [in] ep          UCP endpoint.
 * @param [in] opcode      One of @ref ucp_atomic_fetch_op_t.
 * @param [in] value       Source operand for atomic operation. In the case of CSWAP
 *                         this is the conditional for the swap. For SWAP this is
 *                         the value to be placed in remote memory.
 * @param [inout] result   Local memory address to store resulting fetch to.
 *                         In the case of CSWAP the value in result will be
 *                         swapped into the @a remote_addr if the condition
 *                         is true.
 * @param [in] op_size     Size of value in bytes and pointer type for result
 * @param [in] remote_addr Remote address to operate on.
 * @param [in] rkey        Remote key handle for the remote memory address.
 * @param [in] cb          Call-back function that is invoked whenever the
 *                         send operation is completed. It is important to note
 *                         that the call-back function is only invoked in a case when
 *                         the operation cannot be completed in place.
 *
 * @return NULL                 - The operation was completed immediately.
 * @return UCS_PTR_IS_ERR(_ptr) - The operation failed.
 * @return otherwise            - Operation was scheduled and can be
 *                              completed at any point in time. The request handle
 *                              is returned to the application in order to track
 *                              progress of the operation. The application is
 *                              responsible for releasing the handle using
 *                              @ref ucp_request_free "ucp_request_free()" routine.
 */
ucs_status_ptr_t
ucp_atomic_fetch_nb(ucp_ep_h ep, ucp_atomic_fetch_op_t opcode,
                    uint64_t value, void *result, size_t op_size,
                    uint64_t remote_addr, ucp_rkey_h rkey,
                    ucp_send_callback_t cb);


/**
 * @ingroup UCP_WORKER
 *
 * @brief Flush outstanding AMO and RMA operations on the @ref ucp_worker_h
 * "worker"
 *
 * @deprecated Use @ref ucp_worker_flush_nbx instead.
 *
 * This routine flushes all outstanding AMO and RMA communications on the
 * @ref ucp_worker_h "worker". All the AMO and RMA operations issued on the
 * @a worker prior to this call are completed both at the origin and at the
 * target when this call returns.
 *
 * @note For description of the differences between @ref ucp_worker_flush_nb
 * "flush" and @ref ucp_worker_fence "fence" operations please see
 * @ref ucp_worker_fence "ucp_worker_fence()"
 *
 * @param [in] worker    UCP worker.
 * @param [in] flags     Flags for flush operation. Reserved for future use.
 * @param [in] cb        Callback which will be called when the flush operation
 *                       completes.
 *
 * @return NULL             - The flush operation was completed immediately.
 * @return UCS_PTR_IS_ERR(_ptr) - The flush operation failed.
 * @return otherwise        - Flush operation was scheduled and can be completed
 *                          in any point in time. The request handle is returned
 *                          to the application in order to track progress. The
 *                          application is responsible for releasing the handle
 *                          using @ref ucp_request_free "ucp_request_free()"
 *                          routine.
 */
ucs_status_ptr_t ucp_worker_flush_nb(ucp_worker_h worker, unsigned flags,
                                     ucp_send_callback_t cb);


/**
 * @ingroup UCP_ENDPOINT
 * @brief Close UCP endpoint modes.
 * 
 * @deprecated Use @ref ucp_ep_close_nbx and @ref ucp_ep_close_flags_t instead.
 *
 * The enumeration is used to specify the behavior of @ref ucp_ep_close_nb.
 */
enum ucp_ep_close_mode {
    UCP_EP_CLOSE_MODE_FORCE         = 0, /**< @ref ucp_ep_close_nb releases
                                              the endpoint without any
                                              confirmation from the peer. All
                                              outstanding requests will be
                                              completed with
                                              @ref UCS_ERR_CANCELED error.
                                              @note This mode may cause
                                              transport level errors on remote
                                              side, so it requires set
                                              @ref UCP_ERR_HANDLING_MODE_PEER
                                              for all endpoints created on
                                              both (local and remote) sides to
                                              avoid undefined behavior. */
    UCP_EP_CLOSE_MODE_FLUSH         = 1  /**< @ref ucp_ep_close_nb schedules
                                              flushes on all outstanding
                                              operations. */
};


END_C_DECLS

#endif
