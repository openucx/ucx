/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_DEVICE_IMPL_H
#define UCP_DEVICE_IMPL_H

#include <ucs/type/status.h>
#include <uct/api/uct.h>
#include <ucp/api/ucp.h>


/**
 * @ingroup UCP_COMM
 * @brief GPU request descriptor of a given batch
 *
 * This request tracks a batch of memory operations in progress. It can be used
 * with @ref ucp_gpu_request_progress to detect request completion.
 */
typedef struct ucp_gpu_request {
} ucp_gpu_request_t;


/**
 * @ingroup UCP_COMM
 * @brief Opaque descriptor list stored on GPU.
 */
typedef struct ucp_dlist_handle {
} ucp_dlist_handle_t;


typedef ucp_dlist_handle_t *ucp_dlist_handle_h;


/**
 * @ingroup UCP_COMM
 * @brief Posts one memory put operation.
 *
 * This GPU routine posts one put operation using descriptor list handle.
 * The @ref dlist_index is used to point at the dlist entry to be used for the
 * memory transfer. The addresses and length must be valid for the used dlist
 * entry.
 *
 * The routine returns a request that can be progressed and checked for
 * completion with @ref ucp_gpu_request_progress.
 *
 * This routine can be called repeatedly with the same handle and different
 * addresses and length. The flags parameter can be used to modify the behavior
 * of the routine.
 *
 * @param [in]  handle      Exported descriptor list handle to use for transfer.
 * @param [in]  addr        Local virtual address to send data from.
 * @param [in]  remote_addr Remote virtual address to send data to.
 * @param [in]  length      Length in bytes of the data to send.
 * @param [in]  dlist_index Index in descriptor list pointing to the memory
 *                          registration keys to use for the transfer.
 * @param [in]  flags       Bitfield usable to modify the function behavior.
 * @param [out] req         Request populated by the call.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
UCS_DEVICE_FUNC ucs_status_t
ucp_gpu_put_single(ucp_dlist_handle_h handle,
                   const void *addr, uint64_t remote_addr,
                   size_t length, int dlist_index, uint64_t flags,
                   ucp_gpu_request_t *req);


/**
 * @ingroup UCP_COMM
 * @brief Posts one memory atomic increment operation.
 *
 * This GPU routine posts one atomic increment operation using descriptor list
 * handle. The @ref dlist_index is used to point at the dlist entry to be used
 * for the atomic operation. The remote address must be valid for the used dlist
 * entry.
 *
 * The routine returns a request that can be progressed and checked for
 * completion with @ref ucp_gpu_request_progress.
 *
 * This routine can be called repeatedly with the same handle and different
 * address. The flags parameter can be used to modify the behavior of the
 * routine.
 *
 * @param [in]  handle      Exported descriptor list handle to use for transfer.
 * @param [in]  value       Value used to increment the remote address.
 * @param [in]  remote_addr Remote virtual address to perform the increment to.
 * @param [in]  dlist_index Index in descriptor list pointing to the memory
 *                          remote key to use for the atomic operation.
 * @param [in]  flags       Bitfield usable to modify the function behavior.
 * @param [out] req         Request populated by the call.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
UCS_DEVICE_FUNC ucs_status_t
ucp_gpu_atomic_inc(ucp_dlist_handle_h handle,
                   uint64_t value, uint64_t remote_addr,
                   int dlist_index, uint64_t flags,
                   ucp_gpu_request_t *req);


/**
 * @ingroup UCP_COMM
 * @brief Posts multiple put operations followed by one atomic operation.
 *
 * This GPU routine posts a batch of put operations using the descriptor list
 * entries in the input handle, followed by an atomic operation. This atomic
 * operation can be polled on the receiver to detect completion of all the
 * operations of the batch, started during the same routine call.
 *
 * The content of each entries in the arrays addrs, remote_addrs and lengths
 * must be valid for each corresponding entry in the descriptor list from the
 * input handle. The last entry in the descriptor list contains the remote
 * memory registration descriptors to be used for the atomic operation.
 *
 * The size of the arrays addrs, remote_addrs, and lengths are all equal to
 * the size of the descriptor list array from the handle, minus one.
 *
 * The routine returns a request that can be progressed and checked for
 * completion with @ref ucp_gpu_request_progress.
 *
 * This routine can be called repeatedly with the same handle and different
 * addresses, lengths and atomic related parameters. The flags parameter can be
 * used to modify the behavior of the routine.
 *
 * @param [in]  handle              Exported descriptor list handle to use.
 * @param [in]  addrs               Array of local addresses to send from.
 * @param [in]  remote_addrs        Array of remote addresses to send to.
 * @param [in]  lengths             Array of lengths in bytes for each send.
 * @param [in]  atomic_value        Value of the remote increment.
 * @param [in]  atomic_remote_addr  Remote address to increment to.
 * @param [in]  flags               Bitfield to modify the function behavior.
 * @param [out] req                 Request populated by the call.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
UCS_DEVICE_FUNC ucs_status_t
ucp_gpu_put_multi(ucp_dlist_handle_h handle,
                  void *const *addrs, const uint64_t *remote_addrs,
                  const size_t *lengths,
                  uint64_t atomic_value, uint64_t atomic_remote_addr,
                  uint64_t flags,
                  ucp_gpu_request_t *req);


/**
 * @ingroup UCP_COMM
 * @brief Posts few put operations followed by one atomic operation.
 *
 * This GPU routine posts a batch of put operations using only some of the
 * descriptor list entries in the input handle, followed by an atomic operation.
 * This atomic operation can be polled on the receiver to detect completion of
 * all operations of the batch, started during the same routine call.
 *
 * The set of indexes from the descriptor list entries to use are to be passed
 * in the array @ref dlist_indexes. The last entry of the descriptor list is to
 * be used for the final atomic increment operation.
 *
 * The content of each entries in the arrays addrs, remote_addrs and lengths
 * must be valid for each corresponding descriptor list entry whose index is
 * referenced in @ref dlist_indexes.
 *
 * The size of the arrays dlist_indexes, addrs, remote_addrs, and lengths are
 * all equal. They are lower than the size of the descriptor list array from
 * the handle.
 *
 * The routine returns a request that can be progressed and checked for
 * completion with @ref ucp_gpu_request_progress.
 *
 * This routine can be called repeatedly with the same handle and different
 * dlist_indexes, addresses, lengths and atomic related parameters. The flags
 * parameter can be used to modify the behavior of the routine.
 *
 * @param [in]  handle              Exported descriptor list handle to use.
 * @param [in]  dlist_indexes       Array of indexes, to use in descriptor list
 *                                  of entries from handle.
 * @param [in]  dlist_count         Number of indexes in the array @ref
 *                                  dlist_indexes.
 * @param [in]  addrs               Array of local addresses to send from.
 * @param [in]  remote_addrs        Array of remote addresses to send to.
 * @param [in]  lengths             Array of lengths in bytes for each send.
 * @param [in]  atomic_value        Value of the remote increment.
 * @param [in]  atomic_remote_addr  Remote address to increment to.
 * @param [in]  flags               Bitfield to modify the function behavior.
 * @param [out] req                 Request populated by the call.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
UCS_DEVICE_FUNC ucs_status_t
ucp_gpu_put_multi_partial(ucp_dlist_handle_h handle,
                          const int *dlist_indexes,
                          size_t dlist_count,
                          void *const *addrs, const uint64_t *remote_addrs,
                          const size_t *lengths,
                          uint64_t atomic_value,
                          uint64_t atomic_remote_addr,
                          uint64_t flags,
                          ucp_gpu_request_t *req);


/**
 * @ingroup UCP_COMM
 * @brief Progress a GPU request containing a batch of operations.
 *
 * This GPU progress function checks and progresses a request representing a
 * batch of one or many operations in progress.
 *
 * @param [in]  req  Request containing operations in progress.
 *
 * @return UCS_OK           - The request has completed, no more operations are
 *                            in progress.
 * @return UCS_INPROGRESS   - One or many operations from the batch of the
 *                            request have not completed.
 * @return Error code as defined by @ref ucs_status_t
 */
UCS_DEVICE_FUNC ucs_status_t
ucp_gpu_progress_req(ucp_gpu_request_t *req);

#endif /* UCP_DEVICE_IMPL_H */
