/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCP_H_
#define UCP_H_

#include <ucp/api/ucp_def.h>
#include <ucs/type/status.h>
#include <ucs/type/thread_mode.h>
#include <ucs/datastruct/queue.h> /* TODO remove, needed for req priv */


/**
 * @ingroup CONTEXT
 * @brief UCP configuration
 *
 *  This structure defines the configuration for UCP context.
 */
struct ucp_config {
    struct {
        char             **names;     /**< Array of device names to use */
        unsigned         count;       /**< Number of devices in the array */
    } devices;

    int                  device_policy_force; /**< Whether to force using all devices */

    struct {
       char              **names;     /**< Array of transport names to use */
       unsigned          count;       /**< Number of transports in the array */
   } tls;
};


/**
 * @ingroup CONTEXT
 * @brief Completion status of a tag-matched receive.
 */
typedef struct ucp_tag_recv_completion {
    ucp_tag_t             sender_tag;  /**< Full sender tag */
    size_t                rcvd_len;    /**< How much data was received */
} ucp_tag_recv_completion_t;


/**
 * @ingroup CONTEXT
 * @brief Non-blocking tag-match receive request.
 */
typedef struct ucp_recv_request {
    ucs_status_t               status; /**< Current request status */
    ucp_tag_recv_completion_t  comp;   /**< Completion information. Filled if
                                            status is != INPROGRESS.*/
    ucs_queue_elem_t           queue;
    void                       *buffer;
    size_t                     length;
    uint64_t                   tag;
    uint64_t                   tag_mask;
} ucp_recv_request_t;


/**
 * @ingroup CONTEXT
 * @brief Read UCP configuration.
 *
 * @param [in]  env_prefix    If non-NULL, search for environment variables
 *                            starting with this UCX_<prefix>_. Otherwise, search
 *                            for environment variables starting with just UCX_.
 * @param [in]  filename      If non-NULL, read configuration from this file. If
 *                            the file does not exist, it will be ignored.
 * @param [out] config_p      Filled with a pointer to configuration.
 *
 * @return Error code.
 */
ucs_status_t ucp_config_read(const char *env_prefix, const char *filename,
                             ucp_config_t **config_p);


/**
 * @ingroup CONTEXT
 * @brief Release configuration memory returned from @ref ucp_config_read().
 *
 * @param [in]  config        Configuration to release.
 */
void ucp_config_release(ucp_config_t *config);


/**
 * @ingroup CONTEXT
 * @brief Initialize global ucp context.
 *
 * @param [in]  config            UCP configuration returned from @ref ucp_config_read().
 * @param [in]  request_headroom  How many bytes to reserve before every request
 *                                 returned by non-blocking operations.
 * @param [out] context_p         Filled with a ucp context handle.
 *
 * @return Error code.
 */
ucs_status_t ucp_init(const ucp_config_t *config, size_t request_headroom,
                      ucp_context_h *context_p);


/**
 * @ingroup CONTEXT
 * @brief Destroy global ucp context.
 *
 * @param [in] context_p   Handle to the ucp context.
 */
void ucp_cleanup(ucp_context_h context_p);


/**
 * @ingroup CONTEXT
 * @brief Create a worker object.
 *
 *  The worker represents a progress engine. Multiple progress engines can be
 * created in an application, for example to be used by multiple threads.
 * Every worker can be progressed independently of others.
 *
 * @param [in]  context       Handle to context.
 * @param [in]  thread_mode   Thread access mode to the worker and resources
 *                             created on it.
 * @param [out] worker_p      Filled with a pointer to the worker object.
 */
ucs_status_t ucp_worker_create(ucp_context_h context, ucs_thread_mode_t thread_mode,
                               ucp_worker_h *worker_p);


/**
 * @ingroup CONTEXT
 * @brief Destroy a worker object.
 *
 * @param [in]  worker        Worker object to destroy.
 */
void ucp_worker_destroy(ucp_worker_h worker);


/**
 * @ingroup CONTEXT
 * @brief Get the address of worker object.
 *
 *  Returns the address of a worker object. This address should be passed to any
 * remote entity wishing to connect to this worker.
 *  The address buffer is allocated by this function, and should be released by
 * calling @ref ucp_worker_release_address().
 *
 * @param [in]  worker            Worker object whose address to return.
 * @param [out] address_p         Filled with a pointer to worker address.
 * @param [out] address_length_p  Filled with the size of the address.
 */
ucs_status_t ucp_worker_get_address(ucp_worker_h worker, ucp_address_t **address_p,
                                    size_t *address_length_p);


/**
 * @ingroup CONTEXT
 * @brief Release an address of worker object.
 *
 * @param [in] address            Address to release, returned from
 *                                @ref ucp_worker_get_address()
 */
void ucp_worker_release_address(ucp_worker_h worker, ucp_address_t *address);


/**
 * @ingroup CONTEXT
 * @brief Progress all communications on a specific worker.
 *
 *  This function progresses all communications and returns handles to completed
 * requests.
 *
 * @param [in]  worker    Worker to progress.
 * @param [out] reqs      Pointer to an array of request handlers to be filled
 *                         with completed requests.
 * @param [in]  max       Size of request array.
 *
 * @return How many request handless were written to `reqs'.
 */
void ucp_worker_progress(ucp_worker_h worker);


/**
 * @ingroup CONTEXT
 * @brief Create and open an endpoint.
 *
 *  Create an endpoint to remote peer. This function is non-blocking, and
 * communications may begin immediately after it returns. If the connection
 * process is not completed, communications will be dealyed.
 *
 * @param [in]  worker      Handle to the worker on which the endpoint is created.
 * @param [in]  address     Destination address, originally obtained from @ref
 *                            ucp_worker_get_address().
 * @param [out] ep_p        Filled with a handle to the opened endpoint.
 *
 * @return Error code.
 */
ucs_status_t ucp_ep_create(ucp_worker_h worker, ucp_address_t *address,
                           ucp_ep_h *ep_p);


/**
 * @ingroup CONTEXT
 * @brief Close the remote endpoint.
 *
 * @param [in]  ep   Handle to the remote endpoint.
 */
void ucp_ep_destroy(ucp_ep_h ep);


/**
 * @ingroup CONTEXT
 * @brief Map or allocate memory for zero-copy sends and remote access.
 * 
 * @param [in]     context    UCP context to map memory on.
 * @param [out]    address_p  If != NULL, memory region to map.
 *                            If == NULL, filled with a pointer to allocated region.
 * @param [in]     length     How many bytes to allocate. 
 * @param [in]     flags      Allocation flags (currently reserved - set to 0).
 * @param [out]    memh_p     Filled with handle for allocated region.
 */
ucs_status_t ucp_mem_map(ucp_context_h context, void **address_p, size_t length,
                         unsigned flags, ucp_mem_h *memh_p);


/**
 * @ingroup CONTEXT
 * @brief Undo the operation of uct_mem_map().
 *
 * @param [in]  context     UCP context which was used to allocate/map the memory.
 * @paran [in]  memh        Handle to memory region.
 */
ucs_status_t ucp_mem_unmap(ucp_context_h context, ucp_mem_h memh);


/**
 * @ingroup CONTEXT
 * @brief Serialize memory region remote access key
 *
 * @param [in]  memh          memory region handle.
 * @param [out] rkey_buffer   contains serialized rkey. Caller is responsible to free() it.
 * @param [out] size          length of serialized rkey. 
 */
ucs_status_t ucp_rkey_pack(ucp_mem_h memh, void **rkey_buffer_p, size_t *size_p);


/**
 * @ingroup CONTEXT
 * @brief Create rkey from serialized data
 *
 * @param [in]  context       UCP context
 * @param [in]  rkey_buffer   serialized rkey
 * @param [out] rkey          filled with rkey
 */
ucs_status_t ucp_rkey_unpack(ucp_context_h context, void *rkey_buffer,
                             ucp_rkey_h *rkey_p);


/**
 * @ingroup CONTEXT
 * @brief Destroy remote key.
 *
 * param [in] rkey
 */
ucs_status_t ucp_rkey_destroy(ucp_rkey_h rk_rkey_destrey);


/**
 * @ingroup CONTEXT
 * @brief If possible translate remote address into local address which can be used for direct memory access
 * 
 * @param [in]  ep              endpoint address
 * @param [in]  remote_addr     address to translate
 * @param [in]  rkey            remote memory key
 * @param [out] local_addr      filled by local memory key
 */
ucs_status_t ucp_rmem_ptr(ucp_ep_h ep, void *remote_addr, ucp_rkey_h rkey,
                          void **local_addr_p);


/**
 * @ingroup CONTEXT
 * @brief Send tagged message.
 *
 * This function is blocking - it returns only after the buffer can be reused.
 *
 * @param [in]  ep          Destination to send to.
 * @param [in]  buffer      Message payload to send.
 * @param [in]  length      Message length to send.
 * @param [in]  tag         Message tag to send.
 */
ucs_status_t ucp_tag_send(ucp_ep_h ep, const void *buffer, size_t length, ucp_tag_t tag);


/**
 * @ingroup CONTEXT
 * @brief Receive-match a tagged message.
 *
 * @param [in]  worker      UCP worker.
 * @param [in]  buffer      Buffer to receive the data to.
 * @param [in]  length      Size of the buffer.
 * @param [in]  tag         Message tag to expect.
 * @param [in]  tag_mask    Mask of which bits to match from the incoming tag
 *                           against the expected tag.
 */
ucs_status_t ucp_tag_recv(ucp_worker_h worker, void *buffer, size_t length,
                          ucp_tag_t tag, ucp_tag_t tag_mask,
                          ucp_tag_recv_completion_t *comp);


/**
 * @ingroup CONTEXT
 * @brief Remote put. Returns when local buffer is safe for reuse.
 *
 *  Write a buffer to remote memory.
 *
 * @param [in]  ep           Remote endpoint.
 * @param [in]  buffer       Buffer to write.
 * @param [in]  length       Buffer size.
 * @param [in]  remote_addr  Remote address to write to.
 * @param [in]  rkey         Remote memory key.
 */
ucs_status_t ucp_rma_put(ucp_ep_h ep, const void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup CONTEXT
 * @brief Remote get. Returns when data are in local buffer
 *
 *  Read a buffer from remote memory.
 *
 * @param [in]  ep           Remote endpoint.
 * @param [out] buffer       Buffer to read into.
 * @param [in]  length       Buffer size.
 * @param [in]  remote_addr  Remote address to read from.
 * @param [in]  rkey         Remote memory key.
 */
ucs_status_t ucp_rma_get(ucp_ep_h ep, void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup CONTEXT
 *
 * @brief Force ordering between operations
 *
 * All operations started before fence will be completed before those
 * issued after.
 *
 * @param [in] context  UCP context
 */
ucs_status_t ucp_rma_fence(ucp_context_h context);


/**
 * @ingroup CONTEXT
 *
 * @brief Force remote completion.
 *
 * All operations that were started before ucp_quiet will be completed on
 * remote when ucp_quiet returns
 *
 * @param [in] context  UCP context
 */
ucs_status_t ucp_rma_flush(ucp_context_h context);


#endif
