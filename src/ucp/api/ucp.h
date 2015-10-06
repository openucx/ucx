/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCP_H_
#define UCP_H_

#include <ucp/api/ucp_def.h>
#include <ucs/type/thread_mode.h>
#include <ucs/datastruct/queue_types.h>
#include <ucs/config/types.h>
#include <ucs/sys/math.h>
#include <stdio.h>


/**
 * @defgroup UCP_CONTEXT UCP Application Context
 * @{
 * Application  context is a primary concept of UCP design which
 * provides an isolation mechanism, allowing resources associated
 * with the context to separate or share network communication context
 * across multiple instances of applications.
 *
 * This section provides a detailed description of this concept and
 * routines associated with it.
 *
 * @}
 */


 /**
 * @defgroup UCP_WORKER UCP Worker
 * @{
 * UCP Worker routines
 * @}
 */


 /**
 * @defgroup UCP_MEM UCP Memory routines
 * @{
 * UCP Memory routines
 * @}
 */


 /**
 * @defgroup UCP_ENDPOINT UCP Endpoint
 * @{
 * UCP Endpoint routines
 * @}
 */


 /**
 * @defgroup UCP_COMM UCP Communication routines
 * @{
 * UCP Communication routines
 * @}
 */


 /**
 * @defgroup UCP_CONFIG UCP Configuration
 * @{
 * This section describes routines for configuration
 * of the UCP network layer
 * @}
 */


 /**
 * @defgroup UCP_DATATYPE UCP Data type routines
 * @{
 * UCP Data type routines
 * @}
 */


/**
 * @ingroup UCP_CONTEXT
 * @brief UCP features
 */
enum {
    UCP_FEATURE_TAG   = UCS_BIT(0),  /**< Request tag matching support */
    UCP_FEATURE_RMA   = UCS_BIT(1),  /**< Request remote memory access support */
    UCP_FEATURE_AMO32 = UCS_BIT(2),  /**< Request 32-bit atomic operations support */
    UCP_FEATURE_AMO64 = UCS_BIT(3)   /**< Request 64-bit atomic operations support */
};


/**
 * @ingroup UCP_DATATYPE
 * @brief Data type classification - used internally
 */
enum {
    UCP_DATATYPE_CONTIG  = 0,      /**< Contiguous type */
    UCP_DATATYPE_STRIDED = 1,      /**< Strided type */
    UCP_DATATYPE_GENERIC = 7,      /**< Generic type with user-defined pack/unpack routines */

    UCP_DATATYPE_SHIFT   = 3,      /**< How many bits define the data-type classification */
    UCP_DATATYPE_CLASS_MASK = UCS_MASK(UCP_DATATYPE_SHIFT)
};


/**
 * @ingroup UCP_DATATYPE
 * @brief Generate an identifier for contiguous data type.
 *
 * Create an identifier for contiguous data-type, which is defined by the size
 * of the basic element.
 *
 * @param [in]  _elem_size    Size of the basic element of the type.
 *
 * @return Data-type identifier.
 */
#define ucp_dt_make_contig(_elem_size) \
    (((ucp_datatype_t)(_elem_size) << UCP_DATATYPE_SHIFT) | UCP_DATATYPE_CONTIG)


/**
 * @ingroup UCP_DATATYPE
 * @brief Represents a generic data type.
 */
struct ucp_generic_dt_ops {

    /**
     * @brief Start a packing request.
     *
     * @param [in]  context        User-defined context.
     * @param [in]  buffer         Buffer to pack.
     * @param [in]  count          Number of elements to pack in the buffer.
     *
     * @return  A custom "state" which would be passed to the pack function later.
     */
    void* (*start_pack)(void *context, const void *buffer, size_t count);

    /**
     * @ingroup UCP_DATATYPE
     * @brief Start an unpacking request.
     *
     * @param [in]  context        User-defined context.
     * @param [in]  buffer         Buffer to unpack to.
     * @param [in]  count          Number of elements to unpack in the buffer.
     *
     * @return  A custom "state" which would be passed to the unpack function later.
     */
    void* (*start_unpack)(void *context, void *buffer, size_t count);

    /**
     * @ingroup UCP_DATATYPE
     * @brief Get the total size of packed data.
     *
     * For packing return is the output size, for unpacking - the maximal input size.
     *
     * @param [in]  state          State as returned from start_pack().
     *
     * @return Size of data in packed form.
     */
    size_t (*packed_size)(void *state);

    /**
     * @ingroup UCP_DATATYPE
     * @brief Pack some data.
     *
     * @param [in]  state          State as returned from start_pack().
     * @param [in]  offset         Virtual offset in the output stream.
     * @param [in]  dest           Destination to pack data to.
     * @param [in]  max_length     Maximal length to pack.
     *
     * @return How much was actually written to the destination buffer. Must be
     *         less than or equal to "max_length".
     */
    size_t (*pack) (void *state, size_t offset, void *dest, size_t max_length);

    /**
     * @ingroup UCP_DATATYPE
     * @brief Unpack some data.
     *
     * @param [in]  state          State as returned from start_unpack().
     * @param [in]  offset         Virtual offset in the input stream.
     * @param [in]  src            Source to unpack data from.
     * @param [in]  length         How much to unpack.
     *
     * @return UCS_OK or an error if unpacking failed.
     */
    ucs_status_t (*unpack)(void *state, size_t offset, const void *src, size_t count);

    /**
     * @ingroup UCP_DATATYPE
     * @brief Finish packing/unpacking.
     *
     * @param [in]  state          State as returned from start_pack()/start_unpack().
     */
    void (*finish)(void *state);
};


/**
 * @ingroup UCP_CONFIG
 * @brief Parameters for UCP configuration.
 */
struct ucp_params {
    uint64_t                    features;        /**< Which UCP features to activate. Using other
                                                      features would result in undefined behavior. */
    size_t                      request_size;    /**< How much space to reserve in non-blocking requests. */
    ucp_request_init_callback_t request_init;    /**< Callback for initializing a request May be NULL. */
    ucp_request_cleanup_callback_t request_cleanup; /**< Callback for cleaning-up a request. May be NULL. */
};


/**
 * @ingroup UCP_CONFIG
 * @brief UCP configuration
 *
 * This structure defines the configuration for UCP context.
 */
typedef struct ucp_config {
    UCS_CONFIG_STRING_ARRAY_FIELD(names)   devices; /**< Array of device names to use */
    UCS_CONFIG_STRING_ARRAY_FIELD(names)   tls;     /**< Array of device names to use */
    int                                    force_all_devices; /**< Whether to force using all devices */
    UCS_CONFIG_STRING_ARRAY_FIELD(methods) alloc_prio;   /**< Array of allocation methods */
    size_t                                 bcopy_thresh;  /**< Threshold for switching to bcopy protocol */
    size_t                                 rndv_thresh;  /** Threshold for using rendezvous protocol */
} ucp_config_t;


/**
 * @ingroup UCP_CONTEXT
 * @brief Completion status of a tag-matched receive.
 */
typedef struct ucp_tag_recv_completion {
    ucp_tag_t             sender_tag;  /**< Full sender tag */
    size_t                rcvd_len;    /**< How much data was received */
} ucp_tag_recv_completion_t;


/**
 * @ingroup UCP_CONTEXT
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
 * @ingroup UCP_CONTEXT
 * @brief Progress callback. Used to progress user context during blocking operations.
 */
struct ucp_tag_recv_info {
    ucp_tag_t                              sender_tag;  /**< Full sender tag */
    size_t                                 length;      /**< How much data was received */
};


/**
 * @ingroup UCP_CONFIG
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
 * @ingroup UCP_CONFIG
 * @brief Release configuration memory returned from @ref ucp_config_read().
 *
 * @param [in]  config        Configuration to release.
 */
void ucp_config_release(ucp_config_t *config);


/**
 * @ingroup UCP_CONFIG
 * @brief Print UCP configuration.
 *
 * @param [in]  config        Configuration to print.
 * @param [in]  stream        Output stream to print the configuration to.
 * @param [in]  title         Configuration title to print.
 * @param [in]  print_flags   Control printing options.
 */
void ucp_config_print(const ucp_config_t *config, FILE *stream,
                      const char *title, ucs_config_print_flags_t print_flags);


/**
 * @ingroup UCP_CONTEXT
 * @brief Initialize global ucp context.
 *
 * @param [in]  config        UCP configuration returned from @ref ucp_config_read().
 * @param [out] context_p     Filled with a ucp context handle.
 *
 * @return Error code.
 */
ucs_status_t ucp_init(const ucp_params_t *params, const ucp_config_t *config,
                      ucp_context_h *context_p);


/**
 * @ingroup UCP_CONTEXT
 * @brief Destroy global ucp context.
 *
 * @param [in] context_p   Handle to the ucp context.
 */
void ucp_cleanup(ucp_context_h context_p);


/**
 * @ingroup UCP_WORKER
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
 * @ingroup UCP_WORKER
 * @brief Destroy a worker object.
 *
 * @param [in]  worker        Worker object to destroy.
 */
void ucp_worker_destroy(ucp_worker_h worker);


/**
 * @ingroup UCP_WORKER
 * @brief Register user worker progress callback. The callback is called
 * from @ref ucp_worker_progress().
 *
 * @param [in]  worker     Worker object.
 * @param [in]  func       Callback function to add.
 * @param [in]  arg        Custom argument that is passed to callback function.
 */
void ucp_worker_progress_register(ucp_worker_h worker,
                                  ucp_user_progress_func_t func, void *arg);


/**
 * @ingroup UCP_WORKER
 * @brief Remove a previously registered user worker progress callback.
 *
 * @param [in]  worker     Worker object.
 * @param [in]  func       Callback function to remove.
 * @param [in]  arg        Custom argument that is passed to callback function.
 */
void ucp_worker_progress_unregister(ucp_worker_h worker,
                                    ucp_user_progress_func_t func, void *arg);


/**
 * @ingroup UCP_WORKER
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
 * @ingroup UCP_WORKER
 * @brief Release an address of worker object.
 *
 * @param [in] address            Address to release, returned from
 *                                @ref ucp_worker_get_address()
 */
void ucp_worker_release_address(ucp_worker_h worker, ucp_address_t *address);


/**
 * @ingroup UCP_WORKER
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
 * @ingroup UCP_ENDPOINT
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
 * @ingroup UCP_ENDPOINT
 * @brief Close the remote endpoint.
 *
 * @param [in]  ep   Handle to the remote endpoint.
 */
void ucp_ep_destroy(ucp_ep_h ep);


/**
 * @ingroup UCP_MEM
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
 * @ingroup UCP_MEM
 * @brief Undo the operation of uct_mem_map().
 *
 * @param [in]  context     UCP context which was used to allocate/map the memory.
 * @paran [in]  memh        Handle to memory region.
 */
ucs_status_t ucp_mem_unmap(ucp_context_h context, ucp_mem_h memh);


/**
 * @ingroup UCP_MEM
 * @brief Serialize memory region remote access key.
 *
 * @param [in]  context       UCP context.
 * @param [in]  memh          memory region handle.
 * @param [out] rkey_buffer   contains serialized rkey. Caller is responsible to
 *                            release it using ucp_rkey_buffer_release().
 * @param [out] size          length of serialized rkey. 
 */
ucs_status_t ucp_rkey_pack(ucp_context_h context, ucp_mem_h memh,
                           void **rkey_buffer_p, size_t *size_p);


/**
 * @ingroup UCP_MEM
 * @brief Release packed remote key buffer, returned from @ref ucp_rkey_pack().
 *
 * @param [in]  rkey_buffer   Buffer to release.
 */
void ucp_rkey_buffer_release(void *rkey_buffer);


/**
 * @ingroup UCP_MEM
 * @brief Create remote access key from serialized data.
 *
 * @param [in]  ep            Endpoint to access using the remote key.
 * @param [in]  rkey_buffer   Serialized rkey.
 * @param [out] rkey          Filled with rkey handle.
 */
ucs_status_t ucp_ep_rkey_unpack(ucp_ep_h ep, void *rkey_buffer, ucp_rkey_h *rkey_p);


/**
 * @ingroup UCP_MEM
 * @brief Destroy remote key returned from @ref ucp_rkey_unpack().
 *
 * @param [in]  ep           Endpoint which has created the remote key.
 * @param [in]  rkey         Romote key to destroy.
 */
void ucp_rkey_destroy(ucp_rkey_h rkey);


/**
 * @ingroup UCP_MEM
 * @brief If possible translate remote address into local address which can be
 *        used for direct memory access
 * 
 * @param [in]  ep              endpoint address
 * @param [in]  remote_addr     address to translate
 * @param [in]  rkey            remote memory key
 * @param [out] local_addr      filled by local memory key
 */
ucs_status_t ucp_rmem_ptr(ucp_ep_h ep, void *remote_addr, ucp_rkey_h rkey,
                          void **local_addr_p);


/**
 * @ingroup UCP_COMM
 * @brief Send tagged message in a non-blocking fashion.
 *
 *  Non-blocking tag send. The function returns immediately, however the actual
 * send may be delayed.
 *
 * @param [in]  ep          Destination to send to.
 * @param [in]  buffer      Message payload to send.
 * @param [in]  count       Number of elements in the buffer.
 * @param [in]  datatype    Type of elements in the buffer.
 * @param [in]  tag         Message tag to send.
 * @param [in]  cb          Callback function which would be called when the
 *                          send is completed (buffer can be reused), in case
 *                          the return value is a request handle.
 *
 * @return NULL/UCS_OK            - The send is completed.
 *         UCS_PTR_IS_ERR(_ptr)   - Error during send.
 *         otherwise              - A request handle. the handle should be released
 *                                  by calling ucp_request_release().
 */
ucs_status_ptr_t ucp_tag_send_nb(ucp_ep_h ep, const void *buffer, size_t count,
                                 ucp_datatype_t datatype, ucp_tag_t tag,
                                 ucp_send_callback_t cb);


/**
 * @ingroup UCP_COMM
 * @brief Receive tagged message in a non-blocking fashion.
 *
 *  Non-blocking tag receive. The function returns immediately, however the actual
 * receive may occur later.
 *
 * @param [in]  worker      UCP worker.
 * @param [in]  buffer      Buffer to receive the data to.
 * @param [in]  count       Number of elements in the buffer.
 * @param [in]  datatype    Type of elements in the buffer.
 * @param [in]  tag         Message tag to expect.
 * @param [in]  tag_mask    Mask of which bits to match from the incoming tag
 *                           against the expected tag.
 * @param [in]  cb          Callback function which would be called when the
 *                           data is ready in the receive buffer.
 *
 * @return UCS_PTR_IS_ERR(_ptr) - Error during receive.
 *         otherwise            - A request handle. the handle should be released
 *                                 by calling ucp_request_release().
 *
 * @note This function cannot return UCS_OK/NULL. It always returns a request
 *       handle or an error.
 */
ucs_status_ptr_t ucp_tag_recv_nb(ucp_worker_h worker, void *buffer, size_t count,
                                 ucp_datatype_t datatype, ucp_tag_t tag,
                                 ucp_tag_t tag_mask, ucp_tag_recv_callback_t cb);


/**
 * @ingroup UCP_COMM
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
ucs_status_t ucp_put(ucp_ep_h ep, const void *buffer, size_t length,
                     uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
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
ucs_status_t ucp_get(ucp_ep_h ep, void *buffer, size_t length,
                     uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @brief Atomic 32-bit add.
 *
 * Atomically add a value to a remote 32-bit integer variable.
 * Remote address must be aligned to 32 bit.
 *
 * @param [in]  ep           Remote endpoint.
 * @param [in]  add          Value to add.
 * @param [in]  remote_addr  Remote address of the atomic variable.
 * @param [in]  rkey         Remote memory key.
 */
ucs_status_t ucp_atomic_add32(ucp_ep_h ep, uint32_t add,
                              uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @brief Atomic 64-bit add.
 *
 * Atomically add a value to a remote 64-bit integer variable.
 * Remote address must be aligned to 64 bit.
 *
 * @param [in]  ep           Remote endpoint.
 * @param [in]  add          Value to add.
 * @param [in]  remote_addr  Remote address of the atomic variable.
 * @param [in]  rkey         Remote memory key.
 */
ucs_status_t ucp_atomic_add64(ucp_ep_h ep, uint64_t add,
                              uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @brief Atomic 32-bit fetch-and-add.
 *
 * Atomically add a value to a remote 32-bit integer variable, and put the
 * previous variable value in "result". This function is blocking.
 * Remote address must be aligned to 32 bit.
 *
 * @param [in]  ep           Remote endpoint.
 * @param [in]  add          Value to add.
 * @param [in]  remote_addr  Remote address of the atomic variable.
 * @param [in]  rkey         Remote memory key.
 * @param [out] result       Filled with the previous value of the variable.
 */
ucs_status_t ucp_atomic_fadd32(ucp_ep_h ep, uint32_t add, uint64_t remote_addr,
                               ucp_rkey_h rkey, uint32_t *result);


/**
 * @ingroup UCP_COMM
 * @brief Atomic 64-bit fetch-and-add.
 *
 * Atomically add a value to a remote 64-bit integer variable, and put the
 * previous variable value in "result". This function is blocking.
 * Remote address must be aligned to 64 bit.
 *
 * @param [in]  ep           Remote endpoint.
 * @param [in]  add          Value to add.
 * @param [in]  remote_addr  Remote address of the atomic variable.
 * @param [in]  rkey         Remote memory key.
 * @param [out] result       Filled with the previous value of the variable.
 */
ucs_status_t ucp_atomic_fadd64(ucp_ep_h ep, uint64_t add, uint64_t remote_addr,
                               ucp_rkey_h rkey, uint64_t *result);

/**
 * @ingroup UCP_COMM
 * @brief Atomic 32-bit swap.
 *
 * Atomically assign a new value to a remote 32-bit variable, and put the
 * previous variable value in "result". This function is blocking.
 * Remote address must be aligned to 32 bit.
 *
 * @param [in]  ep           Remote endpoint.
 * @param [in]  swap         Value to swap the remote variable to.
 * @param [in]  remote_addr  Remote address of the atomic variable.
 * @param [in]  rkey         Remote memory key.
 * @param [out] result       Filled with the previous value of the variable.
 */
ucs_status_t ucp_atomic_swap32(ucp_ep_h ep, uint32_t swap, uint64_t remote_addr,
                               ucp_rkey_h rkey, uint32_t *result);


/**
 * @ingroup UCP_COMM
 * @brief Atomic 64-bit swap.
 *
 * Atomically assign a new value to a remote 64-bit variable, and put the
 * previous variable value in "result". This function is blocking.
 * Remote address must be aligned to 64 bit.
 *
 * @param [in]  ep           Remote endpoint.
 * @param [in]  swap         Value to swap the remote variable to.
 * @param [in]  remote_addr  Remote address of the atomic variable.
 * @param [in]  rkey         Remote memory key.
 * @param [out] result       Filled with the previous value of the variable.
 */
ucs_status_t ucp_atomic_swap64(ucp_ep_h ep, uint64_t swap, uint64_t remote_addr,
                               ucp_rkey_h rkey, uint64_t *result);


/**
 * @ingroup UCP_COMM
 * @brief Atomic 32-bit compare-and-swap.
 *
 * Atomically compare a remote 32-bit variable to "compare", if it equals - assign
 * a new value to it, and in any case return previous variable value in "result".
 * This function is blocking.
 * Remote address must be aligned to 32 bit.
 *
 * @param [in]  ep           Remote endpoint.
 * @param [in]  compare      Value to compare to.
 * @param [in]  swap         Value to swap the remote variable to.
 * @param [in]  remote_addr  Remote address of the atomic variable.
 * @param [in]  rkey         Remote memory key.
 * @param [out] result       Filled with the previous value of the variable.
 */
ucs_status_t ucp_atomic_cswap32(ucp_ep_h ep, uint32_t compare, uint32_t swap,
                                uint64_t remote_addr, ucp_rkey_h rkey,
                                uint32_t *result);


/**
 * @ingroup UCP_COMM
 * @brief Atomic 64-bit compare-and-swap.
 *
 * Atomically compare a remote 64-bit variable to "compare", if it equals - assign
 * a new value to it, and in any case return previous variable value in "result".
 * This function is blocking.
 * Remote address must be aligned to 64 bit.
 *
 * @param [in]  ep           Remote endpoint.
 * @param [in]  compare      Value to compare to.
 * @param [in]  swap         Value to swap the remote variable to.
 * @param [in]  remote_addr  Remote address of the atomic variable.
 * @param [in]  rkey         Remote memory key.
 * @param [out] result       Filled with the previous value of the variable.
 */
ucs_status_t ucp_atomic_cswap64(ucp_ep_h ep, uint64_t compare, uint64_t swap,
                                uint64_t remote_addr, ucp_rkey_h rkey,
                                uint64_t *result);


/**
 * @ingroup UCP_COMM
 * @brief Check if a non-blocking request is completed.
 *
 * @param [in]  request      Non-blocking request to check.
 *
 * @return Whether the request is completed.
 */
int ucp_request_is_completed(void *request);


/**
 * @ingroup UCP_COMM
 * @brief Release a communications request.
 *
 * @param [in]  request      Non-blocking request to release.
 *
 * @note If the request is not completed yet, it will actually be released when
 *       completed.
 */
void ucp_request_release(void *request);


/**
 * @ingroup UCP_DATATYPE
 * @brief Create a generic data type.
 *
 * @param [in]  ops          Generic datatype function table.
 * @param [in]  context      User-defined context passed to the functions.
 * @param [out] datatype_p   Filled with a pointer to datatype.
 */
ucs_status_t ucp_dt_create_generic(ucp_generic_dt_ops_t *ops, void *context,
                                   ucp_datatype_t *datatype_p);


/**
 * @ingroup UCP_DATATYPE
 * @brief Destroy a datatype and release its resources.
 *
 * @param [in]  datatype     Data-type to destroy.
 */
void ucp_dt_destroy(ucp_datatype_t datatype);


/**
 * @ingroup UCP_COMM
 *
 * @brief Force ordering between operations
 *
 * All operations started before fence will be completed before those
 * issued after.
 *
 * @param [in] worker        UCP worker.
 */
ucs_status_t ucp_worker_fence(ucp_worker_h worker);


/**
 * @ingroup UCP_COMM
 *
 * @brief Force remote completion.
 *
 * All operations that were started before ucp_quiet will be completed on
 * remote when ucp_quiet returns
 *
 * @param [in] worker        UCP worker.
 */
ucs_status_t ucp_worker_flush(ucp_worker_h worker);


#endif
