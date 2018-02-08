/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCP_H_
#define UCP_H_

#include <ucp/api/ucp_def.h>
#include <ucp/api/ucp_compat.h>
#include <ucp/api/ucp_version.h>
#include <ucs/type/thread_mode.h>
#include <ucs/type/cpu_set.h>
#include <ucs/config/types.h>
#include <ucs/sys/compiler_def.h>
#include <stdio.h>
#include <sys/types.h>

BEGIN_C_DECLS

/**
 * @defgroup UCP_API Unified Communication Protocol (UCP) API
 * @{
 * This section describes UCP API.
 * @}
 */

/**
 * @defgroup UCP_CONTEXT UCP Application Context
 * @ingroup UCP_API
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
 * @ingroup UCP_API
 * @{
 * UCP Worker routines
 * @}
 */


 /**
 * @defgroup UCP_MEM UCP Memory routines
 * @ingroup UCP_API
 * @{
 * UCP Memory routines
 * @}
 */


 /**
 * @defgroup UCP_WAKEUP UCP Wake-up routines
 * @ingroup UCP_API
 * @{
 * UCP Wake-up routines
 * @}
 */


 /**
 * @defgroup UCP_ENDPOINT UCP Endpoint
 * @ingroup UCP_API
 * @{
 * UCP Endpoint routines
 * @}
 */


 /**
 * @defgroup UCP_COMM UCP Communication routines
 * @ingroup UCP_API
 * @{
 * UCP Communication routines
 * @}
 */


 /**
 * @defgroup UCP_CONFIG UCP Configuration
 * @ingroup UCP_API
 * @{
 * This section describes routines for configuration
 * of the UCP network layer
 * @}
 */


 /**
 * @defgroup UCP_DATATYPE UCP Data type routines
 * @ingroup UCP_API
 * @{
 * UCP Data type routines
 * @}
 */


/**
 * @ingroup UCP_CONTEXT
 * @brief UCP context parameters field mask.
 *
 * The enumeration allows specifying which fields in @ref ucp_params_t are
 * present. It is used for the enablement of backward compatibility support.
 */
enum ucp_params_field {
    UCP_PARAM_FIELD_FEATURES          = UCS_BIT(0), /**< features */
    UCP_PARAM_FIELD_REQUEST_SIZE      = UCS_BIT(1), /**< request_size */
    UCP_PARAM_FIELD_REQUEST_INIT      = UCS_BIT(2), /**< request_init */
    UCP_PARAM_FIELD_REQUEST_CLEANUP   = UCS_BIT(3), /**< request_cleanup */
    UCP_PARAM_FIELD_TAG_SENDER_MASK   = UCS_BIT(4), /**< tag_sender_mask */
    UCP_PARAM_FIELD_MT_WORKERS_SHARED = UCS_BIT(5), /**< mt_workers_shared */
    UCP_PARAM_FIELD_ESTIMATED_NUM_EPS = UCS_BIT(6)  /**< estimated_num_eps */
};


/**
 * @ingroup UCP_CONTEXT
 * @brief UCP configuration features
 *
 * The enumeration list describes the features supported by UCP.  An
 * application can request the features using @ref ucp_params_t "UCP parameters"
 * during @ref ucp_init "UCP initialization" process.
 */
enum ucp_feature {
    UCP_FEATURE_TAG    = UCS_BIT(0),  /**< Request tag matching support */
    UCP_FEATURE_RMA    = UCS_BIT(1),  /**< Request remote memory
                                           access support */
    UCP_FEATURE_AMO32  = UCS_BIT(2),  /**< Request 32-bit atomic
                                           operations support */
    UCP_FEATURE_AMO64  = UCS_BIT(3),  /**< Request 64-bit atomic
                                           operations support */
    UCP_FEATURE_WAKEUP = UCS_BIT(4),  /**< Request interrupt notification
                                           support */
    UCP_FEATURE_STREAM = UCS_BIT(5)   /**< Request stream support */
};


/**
 * @ingroup UCP_WORKER
 * @brief UCP worker parameters field mask.
 *
 * The enumeration allows specifying which fields in @ref ucp_worker_params_t are
 * present. It is used for the enablement of backward compatibility support.
 */
enum ucp_worker_params_field {
    UCP_WORKER_PARAM_FIELD_THREAD_MODE  = UCS_BIT(0), /**< UCP thread mode */
    UCP_WORKER_PARAM_FIELD_CPU_MASK     = UCS_BIT(1), /**< Worker's CPU bitmap */
    UCP_WORKER_PARAM_FIELD_EVENTS       = UCS_BIT(2), /**< Worker's events bitmap */
    UCP_WORKER_PARAM_FIELD_USER_DATA    = UCS_BIT(3), /**< User data */
    UCP_WORKER_PARAM_FIELD_EVENT_FD     = UCS_BIT(4)  /**< External event file
                                                           descriptor */
};


/**
 * @ingroup UCP_WORKER
 * @brief UCP listener parameters field mask.
 *
 * The enumeration allows specifying which fields in @ref ucp_listener_params_t
 * are present. It is used for the enablement of backward compatibility support.
 */
enum ucp_listener_params_field {
    UCP_LISTENER_PARAM_FIELD_SOCK_ADDR       = UCS_BIT(0), /**< Sock address and
                                                                length */
    UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER  = UCS_BIT(1)  /**< User's callback
                                                                and argument
                                                                for handling the
                                                                creation of an
                                                                endpoint */
};


/**
 * @ingroup UCP_ENDPOINT
 * @brief UCP endpoint parameters field mask.
 *
 * The enumeration allows specifying which fields in @ref ucp_ep_params_t are
 * present. It is used for the enablement of backward compatibility support.
 */
enum ucp_ep_params_field {
    UCP_EP_PARAM_FIELD_REMOTE_ADDRESS    = UCS_BIT(0), /**< Address of remote
                                                            peer */
    UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE = UCS_BIT(1), /**< Error handling mode.
                                                            @ref ucp_err_handling_mode_t */
    UCP_EP_PARAM_FIELD_ERR_HANDLER       = UCS_BIT(2), /**< Handler to process
                                                            transport level errors */
    UCP_EP_PARAM_FIELD_USER_DATA         = UCS_BIT(3), /**< User data pointer */
    UCP_EP_PARAM_FIELD_SOCK_ADDR         = UCS_BIT(4), /**< Socket address field */
    UCP_EP_PARAM_FIELD_FLAGS             = UCS_BIT(5)  /**< Endpoint flags */
};


/**
 * @ingroup UCP_ENDPOINT
 * @brief UCP endpoint parameters flags.
 *
 * The enumeration list describes the endpoint's parameters flags supported by
 * @ref ucp_ep_create() function.
 */
enum ucp_ep_params_flags_field {
    UCP_EP_PARAMS_FLAGS_CLIENT_SERVER  = UCS_BIT(0)   /**< Using a client-server
                                                           connection establishment
                                                           mechanism.
                                                           @ref ucs_sock_addr_t
                                                           sockaddr field
                                                           must be provided and
                                                           contain the address
                                                           of the remote peer */
};


/**
 * @ingroup UCP_ENDPOINT
 * @brief Close UCP endpoint modes.
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


/**
 * @ingroup UCP_MEM
 * @brief UCP memory mapping parameters field mask.
 *
 * The enumeration allows specifying which fields in @ref ucp_mem_map_params_t are
 * present. It is used for the enablement of backward compatibility support.
 */
enum ucp_mem_map_params_field {
    UCP_MEM_MAP_PARAM_FIELD_ADDRESS = UCS_BIT(0), /**< Address of the memory that
                                                       would be used in the
                                                       @ref ucp_mem_map routine,
                                                       see @ref ucp_mem_map_matrix
                                                       for details */
    UCP_MEM_MAP_PARAM_FIELD_LENGTH  = UCS_BIT(1), /**< The size of memory that
                                                       would be allocated or
                                                       registered in the
                                                       @ref ucp_mem_map routine.*/
    UCP_MEM_MAP_PARAM_FIELD_FLAGS   = UCS_BIT(2)  /**< Allocation flags */
};

/**
 * @ingroup UCP_MEM
 * @brief UCP memory advice parameters field mask.
 *
 * The enumeration allows specifying which fields in @ref ucp_mem_advise_params_t are
 * present. It is used for the enablement of backward compatibility support.
 */
enum ucp_mem_advise_params_field {
    UCP_MEM_ADVISE_PARAM_FIELD_ADDRESS = UCS_BIT(0), /**< Address of the memory */
    UCP_MEM_ADVISE_PARAM_FIELD_LENGTH  = UCS_BIT(1), /**< The size of memory */ 
    UCP_MEM_ADVISE_PARAM_FIELD_ADVICE  = UCS_BIT(2)  /**< Advice on memory usage */
};


/**
 * @ingroup UCP_CONTEXT
 * @brief UCP context attributes field mask.
 *
 * The enumeration allows specifying which fields in @ref ucp_context_attr_t are
 * present. It is used for the enablement of backward compatibility support.
 */
enum ucp_context_attr_field {
    UCP_ATTR_FIELD_REQUEST_SIZE = UCS_BIT(0), /**< UCP request size */
    UCP_ATTR_FIELD_THREAD_MODE  = UCS_BIT(1)  /**< UCP context thread flag */
};

/**
 * @ingroup UCP_WORKER
 * @brief UCP worker attributes field mask.
 *
 * The enumeration allows specifying which fields in @ref ucp_worker_attr_t are
 * present. It is used for the enablement of backward compatibility support.
 */
enum ucp_worker_attr_field {
    UCP_WORKER_ATTR_FIELD_THREAD_MODE = UCS_BIT(0)  /**< UCP thread mode */
};

/**
 * @ingroup UCP_DATATYPE
 * @brief UCP data type classification
 *
 * The enumeration list describes the datatypes supported by UCP.
 */
enum ucp_dt_type {
    UCP_DATATYPE_CONTIG  = 0,      /**< Contiguous datatype */
    UCP_DATATYPE_STRIDED = 1,      /**< Strided datatype */
    UCP_DATATYPE_IOV     = 2,      /**< Scatter-gather list with multiple pointers */
    UCP_DATATYPE_GENERIC = 7,      /**< Generic datatype with
                                        user-defined pack/unpack routines */
    UCP_DATATYPE_SHIFT   = 3,      /**< Number of bits defining
                                        the datatype classification */
    UCP_DATATYPE_CLASS_MASK = UCS_MASK(UCP_DATATYPE_SHIFT) /**< Data-type class
                                                                mask */
};


/**
 * @ingroup UCP_MEM
 * @brief UCP memory mapping flags.
 *
 * The enumeration list describes the memory mapping flags supported by @ref
 * ucp_mem_map() function.
 */
enum {
    UCP_MEM_MAP_NONBLOCK = UCS_BIT(0), /**< Complete the mapping faster, possibly by
                                            not populating the pages in the mapping
                                            up-front, and mapping them later when
                                            they are accessed by communication
                                            routines. */
    UCP_MEM_MAP_ALLOCATE = UCS_BIT(1), /**< Identify requirement for allocation,
                                            if passed address is not a null-pointer
                                            then it will be used as a hint or direct
                                            address for allocation. */
    UCP_MEM_MAP_FIXED    = UCS_BIT(2)  /**< Don't interpret address as a hint:
                                            place the mapping at exactly that
                                            address. The address must be a multiple
                                            of the page size. */
};


/**
 * @ingroup UCP_COMM
 * @brief Atomic operation requested for ucp_atomic_post
 *
 * This enumeration defines which atomic memory operation should be
 * performed by the ucp_atomic_post family of fuctions. All of these are
 * non-fetching atomics and will not result in a request handle.
 */
typedef enum {
    UCP_ATOMIC_POST_OP_ADD, /**< Atomic add */
    UCP_ATOMIC_POST_OP_LAST
} ucp_atomic_post_op_t;


/**
 * @ingroup UCP_COMM
 * @brief Atomic operation requested for ucp_atomic_fetch
 *
 * This enumeration defines which atomic memory operation should be performed
 * by the ucp_atomic_fetch family of functions. All of these functions
 * will fetch data from the remote node.
 */
typedef enum {
    UCP_ATOMIC_FETCH_OP_FADD, /**< Atomic Fetch and add */
    UCP_ATOMIC_FETCH_OP_SWAP, /**< Atomic swap */
    UCP_ATOMIC_FETCH_OP_CSWAP, /**< Atomic conditional swap */
    UCP_ATOMIC_FETCH_OP_LAST
} ucp_atomic_fetch_op_t;


/**
 * @ingroup UCP_DATATYPE
 * @brief Generate an identifier for contiguous data type.
 *
 * This macro creates an identifier for contiguous datatype that is defined by
 * the size of the basic element.
 *
 * @param [in]  _elem_size    Size of the basic element of the type.
 *
 * @return Data-type identifier.
 *
 * @note In case of partial receive, the buffer will be filled with integral
 *       count of elements.
 */
#define ucp_dt_make_contig(_elem_size) \
    (((ucp_datatype_t)(_elem_size) << UCP_DATATYPE_SHIFT) | UCP_DATATYPE_CONTIG)


/**
 * @ingroup UCP_DATATYPE
 * @brief Generate an identifier for Scatter-gather IOV data type.
 *
 * This macro creates an identifier for datatype of scatter-gather list
 * with multiple pointers
 *
 * @return Data-type identifier.
 *
 * @note In case of partial receive, @ref ucp_dt_iov_t::buffer can be filled
 *       with any number of bytes according to its @ref ucp_dt_iov_t::length.
 */
#define ucp_dt_make_iov() (UCP_DATATYPE_IOV)


/**
 * @ingroup UCP_DATATYPE
 * @brief Structure for scatter-gather I/O.
 *
 * This structure is used to specify a list of buffers which can be used
 * within a single data transfer function call.
 *
 * @note If @a length is zero, the memory pointed to by @a buffer
 *       will not be accessed. Otherwise, @a buffer must point to valid memory.
 */
typedef struct ucp_dt_iov {
    void   *buffer;   /**< Pointer to a data buffer */
    size_t  length;   /**< Length of the @a buffer in bytes */
} ucp_dt_iov_t;


/**
 * @ingroup UCP_DATATYPE
 * @brief UCP generic data type descriptor
 *
 * This structure provides a generic datatype descriptor that
 * is used for definition of application defined datatypes.

 * Typically, the descriptor is used for an integration with datatype
 * engines implemented within MPI and SHMEM implementations.
 *
 * @note In case of partial receive, any amount of received data is acceptable
 *       which matches buffer size.
 */
typedef struct ucp_generic_dt_ops {

    /**
     * @ingroup UCP_DATATYPE
     * @brief Start a packing request.
     *
     * The pointer refers to application defined start-to-pack routine. It will
     * be called from the @ref ucp_tag_send_nb routine.
     *
     * @param [in]  context        User-defined context.
     * @param [in]  buffer         Buffer to pack.
     * @param [in]  count          Number of elements to pack into the buffer.
     *
     * @return  A custom state that is passed to the following
     *          @ref ucp_generic_dt_ops::unpack "pack()" routine.
     */
    void* (*start_pack)(void *context, const void *buffer, size_t count);

    /**
     * @ingroup UCP_DATATYPE
     * @brief Start an unpacking request.
     *
     * The pointer refers to application defined start-to-unpack routine. It will
     * be called from the @ref ucp_tag_recv_nb routine.
     *
     * @param [in]  context        User-defined context.
     * @param [in]  buffer         Buffer to unpack to.
     * @param [in]  count          Number of elements to unpack in the buffer.
     *
     * @return  A custom state that is passed later to the following
     *          @ref ucp_generic_dt_ops::unpack "unpack()" routine.
     */
    void* (*start_unpack)(void *context, void *buffer, size_t count);

    /**
     * @ingroup UCP_DATATYPE
     * @brief Get the total size of packed data.
     *
     * The pointer refers to user defined routine that returns the size of data
     * in a packed format.
     *
     * @param [in]  state          State as returned by
     *                             @ref ucp_generic_dt_ops::start_pack
     *                             "start_pack()" routine.
     *
     * @return  The size of the data in a packed form.
     */
    size_t (*packed_size)(void *state);

    /**
     * @ingroup UCP_DATATYPE
     * @brief Pack data.
     *
     * The pointer refers to application defined pack routine.
     *
     * @param [in]  state          State as returned by
     *                             @ref ucp_generic_dt_ops::start_pack
     *                             "start_pack()" routine.
     * @param [in]  offset         Virtual offset in the output stream.
     * @param [in]  dest           Destination to pack the data to.
     * @param [in]  max_length     Maximal length to pack.
     *
     * @return The size of the data that was written to the destination buffer.
     *         Must be less than or equal to @e max_length.
     */
    size_t (*pack) (void *state, size_t offset, void *dest, size_t max_length);

    /**
     * @ingroup UCP_DATATYPE
     * @brief Unpack data.
     *
     * The pointer refers to application defined unpack routine.
     *
     * @param [in]  state          State as returned by
     *                             @ref ucp_generic_dt_ops::start_pack
     *                             "start_pack()" routine.
     * @param [in]  offset         Virtual offset in the input stream.
     * @param [in]  src            Source to unpack the data from.
     * @param [in]  length         Length to unpack.
     *
     * @return UCS_OK or an error if unpacking failed.
     */
    ucs_status_t (*unpack)(void *state, size_t offset, const void *src, size_t count);

    /**
     * @ingroup UCP_DATATYPE
     * @brief Finish packing/unpacking.
     *
     * The pointer refers to application defined finish routine.
     *
     * @param [in]  state          State as returned by
     *                             @ref ucp_generic_dt_ops::start_pack
     *                             "start_pack()"
     *                             and
     *                             @ref ucp_generic_dt_ops::start_unpack
     *                             "start_unpack()"
     *                             routines.
     */
    void (*finish)(void *state);
} ucp_generic_dt_ops_t;


/**
 * @ingroup UCP_CONFIG
 * @brief Tuning parameters for UCP library.
 *
 * The structure defines the parameters that are used for
 * UCP library tuning during UCP library @ref ucp_init "initialization".
 *
 * @note UCP library implementation uses the @ref ucp_feature "features"
 * parameter to optimize the library functionality that minimize memory
 * footprint. For example, if the application does not require send/receive
 * semantics UCP library may avoid allocation of expensive resources associated with
 * send/receive queues.
 */
typedef struct ucp_params {
    /**
     * Mask of valid fields in this structure, using bits from @ref ucp_params_field.
     * Fields not specified in this mask would be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t                           field_mask;

    /**
     * UCP @ref ucp_feature "features" that are used for library
     * initialization. It is recommended for applications only to request
     * the features that are required for an optimal functionality
     * This field must be specified.
     */
    uint64_t                           features;

    /**
     * The size of a reserved space in a non-blocking requests. Typically
     * applications use this space for caching own structures in order to avoid
     * costly memory allocations, pointer dereferences, and cache misses.
     * For example, MPI implementation can use this memory for caching MPI
     * descriptors
     * This field defaults to 0 if not specified.
     */
    size_t                             request_size;

    /**
     * Pointer to a routine that is used for the request initialization.
     * This function will be called only on the very first time a request memory
     * is initialized, and may not be called again if a request is reused.
     * If a request should be reset before the next reuse, it can be done before
     * calling @ref ucp_request_free.
     *
     * @e NULL can be used if no such is function required, which is also the
     * default if this field is not specified by @ref field_mask.
     */
    ucp_request_init_callback_t        request_init;

    /**
     * Pointer to a routine that is responsible for final cleanup of the memory
     * associated with the request. This routine may not be called every time a
     * request is released. For some implementations, the cleanup call may be
     * delayed and only invoked at @ref ucp_worker_cleanup.
     *
     * @e NULL can be used if no such function is required, which is also the
     * default if this field is not specified by @ref field_mask.
     */
    ucp_request_cleanup_callback_t     request_cleanup;

    /**
     * Mask which specifies particular bits of the tag which can uniquely
     * identify the sender (UCP endpoint) in tagged operations.
     * This field defaults to 0 if not specified.
     */
    uint64_t                           tag_sender_mask;

    /**
     * This flag indicates if this context is shared by multiple workers
     * from different threads. If so, this context needs thread safety
     * support; otherwise, the context does not need to provide thread
     * safety.
     * For example, if the context is used by single worker, and that
     * worker is shared by multiple threads, this context does not need
     * thread safety; if the context is used by worker 1 and worker 2,
     * and worker 1 is used by thread 1 and worker 2 is used by thread 2,
     * then this context needs thread safety.
     */
    int                                mt_workers_shared;

    /**
     * An optimization hint of how many endpoints would be created on this context.
     * For example, when used from MPI or SHMEM libraries, this number would specify
     * the number of ranks (or processing elements) in the job.
     * Does not affect semantics, but only transport selection criteria and the
     * resulting performance.
     * The value can be also set by UCX_NUM_EPS environment variable. In such case
     * it will override the number of endpoints set by @e estimated_num_eps
     */
    size_t                             estimated_num_eps;

} ucp_params_t;


/**
 * @ingroup UCP_CONTEXT
 * @brief Context attributes.
 *
 * The structure defines the attributes which characterize
 * the particular context.
 */
typedef struct ucp_context_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucp_context_attr_field.
     * Fields not specified in this mask will be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t              field_mask;

    /**
     * Size of UCP non-blocking request. When pre-allocated request is used
     * (e.g. in @ref ucp_tag_recv_nbr) it should have enough space to fit
     * UCP request data, which is defined by this value.
     */
    size_t                request_size;

    /**
     * Thread safe level of the context. For supported thread levels please
     * see @ref ucs_thread_mode_t.
     */
    ucs_thread_mode_t     thread_mode;
} ucp_context_attr_t;

/**
 * @ingroup UCP_WORKER
 * @brief UCP worker attributes.
 *
 * The structure defines the attributes which characterize
 * the particular worker.
 */
typedef struct ucp_worker_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucp_worker_attr_field.
     * Fields not specified in this mask will be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t              field_mask;

    /**
     * Thread safe level of the worker.
     */
    ucs_thread_mode_t     thread_mode;
} ucp_worker_attr_t;


/**
 * @ingroup UCP_WORKER
 * @brief Tuning parameters for the UCP worker.
 *
 * The structure defines the parameters that are used for the
 * UCP worker tuning during the UCP worker @ref ucp_worker_create "creation".
 */
typedef struct ucp_worker_params {
    /**
     * Mask of valid fields in this structure, using bits from @ref ucp_worker_params_field.
     * Fields not specified in this mask would be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t                field_mask;

    /**
     * Thread safety "mode" for the worker object and resources associated with it.
     * This value is optional.
     * If it's not set (along with its corresponding bit in the field_mask -
     * UCP_WORKER_PARAM_FIELD_THREAD_MODE), the UCS_THREAD_MODE_SINGLE mode
     * will be used.
     */
    ucs_thread_mode_t       thread_mode;

    /**
     * Mask of which CPUs worker resources should preferably be allocated on.
     * This value is optional.
     * If it's not set (along with its corresponding bit in the field_mask -
     * UCP_WORKER_PARAM_FIELD_CPU_MASK), resources are allocated according to
     * system's default policy.
     */
    ucs_cpu_set_t           cpu_mask;

    /**
     * Mask of events (@ref ucp_wakeup_event_t) which are expected on wakeup.
     * This value is optional.
     * If it's not set (along with its corresponding bit in the field_mask -
     * UCP_WORKER_PARAM_FIELD_EVENTS), all types of events will trigger on
     * wakeup.
     */
    unsigned                events;

    /**
     * User data associated with the current worker.
     * This value is optional.
     * If it's not set (along with its corresponding bit in the field_mask -
     * UCP_WORKER_PARAM_FIELD_USER_DATA), it will default to NULL.
     */
    void                    *user_data;

    /**
     * External event file descriptor.
     * This value is optional.
     * If @ref UCP_WORKER_PARAM_FIELD_EVENT_FD is set in the field_mask, events
     * on the worker will be reported on the provided event file descriptor. In
     * this case, calling @ref ucp_worker_get_efd() will result in an error.
     * The provided file descriptor must be capable of aggregating notifications
     * for arbitrary events, for example @c epoll(7) on Linux systems.
     * @ref user_data will be used as the event user-data on systems which
     * support it. For example, on Linux, it will be placed in
     * @ref epoll_data_t::ptr, when returned from @ref epoll_wait().
     *
     * Otherwise, events would be reported to the event file descriptor returned
     * from @ref ucp_worker_get_efd().
     */
    int                     event_fd;

} ucp_worker_params_t;


/**
 * @ingroup UCP_WORKER
 * @brief Parameters for a UCP listener object.
 *
 * This structure defines parameters for @ref ucp_listener_create, which is used to
 * listen for incoming client/server connections.
 */
typedef struct ucp_listener_params {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucp_listener_params_field.
     * Fields not specified in this mask would be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t                       field_mask;

    /**
     * An address in the form of a sockaddr.
     * This field is mandatory for filling (along with its corresponding bit
     * in the field_mask - @ref UCP_LISTENER_PARAM_FIELD_SOCK_ADDR).
     * The @ref ucp_listener_create routine will return with an error if sockaddr
     * is not specified.
     */
    ucs_sock_addr_t                sockaddr;

    /**
     * Handler to endpoint creation in a client-server connection flow.
     * In order for the callback inside this handler to be invoked, the
     * UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER needs to be set in the
     * field_mask.
     */
    ucp_listener_accept_handler_t  accept_handler;
} ucp_listener_params_t;


/**
 * @ingroup UCP_ENDPOINT
 * @brief Tuning parameters for the UCP endpoint.
 *
 * The structure defines the parameters that are used for the
 * UCP endpoint tuning during the UCP ep @ref ucp_ep_create "creation".
 */
typedef struct ucp_ep_params {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucp_ep_params_field.
     * Fields not specified in this mask would be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t                field_mask;

    /**
     * Destination address; if the @ref UCP_EP_PARAMS_FLAGS_CLIENT_SERVER flag
     * is not set, this address is mandatory for filling
     * (along with its corresponding bit in the field_mask - @ref
     * UCP_EP_PARAM_FIELD_REMOTE_ADDRESS) and must be obtained using
     * @ref ucp_worker_get_address. This field cannot be changed by
     * @ref ucp_ep_modify_nb.
     */
    const ucp_address_t     *address;

    /**
     * Desired error handling mode, optional parameter. Default value is
     * @ref UCP_ERR_HANDLING_MODE_NONE. This field cannot be changed by
     * @ref ucp_ep_modify_nb.
     */
    ucp_err_handling_mode_t err_mode;

    /**
     * Handler to process transport level failure.
     */
    ucp_err_handler_t       err_handler;

    /**
     * User data associated with an endpoint. See @ref ucp_stream_poll_ep_t and
     * @ref ucp_err_handler_t
     */
    void                    *user_data;

    /**
     * Endpoint flags from @ref ucp_ep_params_flags_field.
     * This value is optional.
     * If it's not set (along with its corresponding bit in the field_mask -
     * @ref UCP_EP_PARAM_FIELD_FLAGS), the @ref ucp_ep_create() routine will
     * consider the flags as set to zero.
     */
     unsigned               flags;

    /**
     * Destination address in the form of a sockaddr;
     * if the @ref UCP_EP_PARAMS_FLAGS_CLIENT_SERVER flag is set, this address
     * is mandatory for filling (along with its corresponding bit in the
     * field_mask - @ref UCP_EP_PARAM_FIELD_SOCK_ADDR) and should be obtained
     * from the user. This field cannot be changed by @ref ucp_ep_modify_nb.
     */
    ucs_sock_addr_t         sockaddr;

} ucp_ep_params_t;


/**
 * @ingroup UCP_ENDPOINT
 * @brief Output parameter of @ref ucp_stream_worker_poll function.
 *
 * The structure defines the endpoint and its user data.
 */
typedef struct {
    /**
     * Endpoint handle.
     */
    ucp_ep_h    ep;

    /**
     * User data associated with an endpoint passed in
     * @ref ucp_ep_params_t::user_data.
     */
    void        *user_data;

    /**
     * Reserved for future use.
     */
    unsigned    flags;

    /**
     * Reserved for future use.
     */
    uint8_t     reserved[16];
} ucp_stream_poll_ep_t;

/**
 * @ingroup UCP_MEM
 * @brief Tuning parameters for the UCP memory mapping.
 *
 * The structure defines the parameters that are used for the
 * UCP memory mapping tuning during the @ref ucp_mem_map "ucp_mem_map" routine.
 */
typedef struct ucp_mem_map_params {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucp_mem_map_params_field.
     * Fields not specified in this mask would be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t                field_mask;

    /**
     * If the address is not NULL, the routine maps (registers) the memory segment
     * pointed to by this address.
     * If the pointer is NULL, the library allocates mapped (registered) memory
     * segment and returns its address in this argument.
     * Therefore, this value is optional.
     * If it's not set (along with its corresponding bit in the field_mask -
     * @ref UCP_MEM_MAP_PARAM_FIELD_ADDRESS), the ucp_mem_map routine will consider
     * address as set to NULL and will allocate memory.
     */
     void                   *address;

     /**
      * Length (in bytes) to allocate or map (register).
      * This field is mandatory for filling (along with its corresponding bit
      * in the field_mask - @ref UCP_MEM_MAP_PARAM_FIELD_LENGTH).
      * The @ref ucp_mem_map routine will return with an error if the length isn't
      * specified.
      */
     size_t                 length;

     /**
      * Allocation flags, e.g. @ref UCP_MEM_MAP_NONBLOCK.
      * This value is optional.
      * If it's not set (along with its corresponding bit in the field_mask -
      * @ref UCP_MEM_MAP_PARAM_FIELD_FLAGS), the @ref ucp_mem_map routine will
      * consider the flags as set to zero.
      */
     unsigned               flags;
} ucp_mem_map_params_t;


/**
 * @ingroup UCP_CONTEXT
 * @brief UCP receive information descriptor
 *
 * The UCP receive information descriptor is allocated by application and filled
 * in with the information about the received message by @ref ucp_tag_probe_nb
 * or @ref ucp_tag_recv_request_test routines or
 * @ref ucp_tag_recv_callback_t callback argument.
 */
struct ucp_tag_recv_info {
    /** Sender tag */
    ucp_tag_t                              sender_tag;
    /** The size of the received data */
    size_t                                 length;
};


/**
 * @ingroup UCP_CONFIG
 * @brief Read UCP configuration descriptor
 *
 * The routine fetches the information about UCP library configuration from
 * the run-time environment. Then, the fetched descriptor is used for
 * UCP library @ref ucp_init "initialization". The Application can print out the
 * descriptor using @ref ucp_config_print "print" routine. In addition
 * the application is responsible to @ref ucp_config_free "free" the
 * descriptor back to UCP library.
 *
 * @param [in]  env_prefix    If non-NULL, the routine searches for the
 *                            environment variables that start with
 *                            @e UCX_<env_prefix>_ prefix.
 *                            Otherwise, the routine searches for the
 *                            environment variables that start with
 *                            @e UCX_ prefix.
 * @param [in]  filename      If non-NULL, read configuration from the file
 *                            defined by @e filename. If the file does not
 *                            exist, it will be ignored and no error reported
 *                            to the application.
 * @param [out] config_p      Pointer to configuration descriptor as defined by
 *                            @ref ucp_config_t "ucp_config_t".
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_config_read(const char *env_prefix, const char *filename,
                             ucp_config_t **config_p);


/**
 * @ingroup UCP_CONFIG
 * @brief Release configuration descriptor
 *
 * The routine releases the configuration descriptor that was allocated through
 * @ref ucp_config_read "ucp_config_read()" routine.
 *
 * @param [out] config        Configuration descriptor as defined by
 *                            @ref ucp_config_t "ucp_config_t".
 */
void ucp_config_release(ucp_config_t *config);


/**
 * @ingroup UCP_CONFIG
 * @brief Modify context configuration.
 *
 * The routine changes one configuration setting stored in @ref ucp_config_t
 * "configuration" descriptor.
 *
 * @param [in]  config        Configuration to modify.
 * @param [in]  name          Configuration variable name.
 * @param [in]  value         Value to set.
 *
 * @return Error code.
 */
ucs_status_t ucp_config_modify(ucp_config_t *config, const char *name,
                               const char *value);


/**
 * @ingroup UCP_CONFIG
 * @brief Print configuration information
 *
 * The routine prints the configuration information that is stored in
 * @ref ucp_config_t "configuration" descriptor.
 *
 * @todo Expose ucs_config_print_flags_t
 *
 * @param [in]  config        @ref ucp_config_t "Configuration descriptor"
 *                            to print.
 * @param [in]  stream        Output stream to print the configuration to.
 * @param [in]  title         Configuration title to print.
 * @param [in]  print_flags   Flags that control various printing options.
 */
void ucp_config_print(const ucp_config_t *config, FILE *stream,
                      const char *title, ucs_config_print_flags_t print_flags);


/**
 * @ingroup UCP_CONTEXT
 * @brief Get UCP library version.
 *
 * This routine returns the UCP library version.
 *
 * @param [out] major_version       Filled with library major version.
 * @param [out] minor_version       Filled with library minor version.
 * @param [out] release_number      Filled with library release number.
 */
void ucp_get_version(unsigned *major_version, unsigned *minor_version,
                     unsigned *release_number);


/**
 * @ingroup UCP_CONTEXT
 * @brief Get UCP library version as a string.
 *
 * This routine returns the UCP library version as a string which consists of:
 * "major.minor.release".
 */
const char *ucp_get_version_string(void);


/** @cond PRIVATE_INTERFACE */
/**
 * @ingroup UCP_CONTEXT
 * @brief UCP context initialization with particular API version.
 *
 *  This is an internal routine used to check compatibility with a particular
 * API version. @ref ucp_init should be used to create UCP context.
 */
ucs_status_t ucp_init_version(unsigned api_major_version, unsigned api_minor_version,
                              const ucp_params_t *params, const ucp_config_t *config,
                              ucp_context_h *context_p);
/** @endcond */


/**
 * @ingroup UCP_CONTEXT
 * @brief UCP context initialization.
 *
 * This routine creates and initializes a @ref ucp_context_h
 * "UCP application context".
 *
 * @warning This routine must be called before any other UCP function
 * call in the application.
 *
 * This routine checks API version compatibility, then discovers the available
 * network interfaces, and initializes the network resources required for
 * discovering of the network and memory related devices.
 *  This routine is responsible for initialization all information required for
 * a particular application scope, for example, MPI application, OpenSHMEM
 * application, etc.
 *
 * @note
 * @li Higher level protocols can add additional communication isolation, as
 * MPI does with it's communicator object. A single communication context may
 * be used to support multiple MPI communicators.
 * @li The context can be used to isolate the communication that corresponds to
 * different protocols. For example, if MPI and OpenSHMEM are using UCP to
 * isolate the MPI communication from the OpenSHMEM communication, users should
 * use different application context for each of the communication libraries.
 *
 * @param [in]  config        UCP configuration descriptor allocated through
 *                            @ref ucp_config_read "ucp_config_read()" routine.
 * @param [in]  params        User defined @ref ucp_params_t configurations for the
 *                            @ref ucp_context_h "UCP application context".
 * @param [out] context_p     Initialized @ref ucp_context_h
 *                            "UCP application context".
 *
 * @return Error code as defined by @ref ucs_status_t
 */
static inline ucs_status_t ucp_init(const ucp_params_t *params,
                                    const ucp_config_t *config,
                                    ucp_context_h *context_p)
{
    return ucp_init_version(UCP_API_MAJOR, UCP_API_MINOR, params, config,
                            context_p);
}


/**
 * @ingroup UCP_CONTEXT
 * @brief Release UCP application context.
 *
 * This routine finalizes and releases the resources associated with a
 * @ref ucp_context_h "UCP application context".
 *
 * @warning An application cannot call any UCP routine
 * once the UCP application context released.
 *
 * The cleanup process releases and shuts down all resources associated    with
 * the application context. After calling this routine, calling any UCP
 * routine without calling @ref ucp_init "UCP initialization routine" is invalid.
 *
 * @param [in] context_p   Handle to @ref ucp_context_h
 *                         "UCP application context".
 */
void ucp_cleanup(ucp_context_h context_p);


/**
 * @ingroup UCP_CONTEXT
 * @brief Get attributes specific to a particular context.
 *
 * This routine fetches an information about the context.
 *
 * @param [in]  context_p  Handle to @ref ucp_context_h
 *                         "UCP application context".
 *
 * @param [out] attr       Filled with attributes of @p context_p context.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_context_query(ucp_context_h context_p,
                               ucp_context_attr_t *attr);


/**
 * @ingroup UCP_CONTEXT
 * @brief Print context information.
 *
 * This routine prints information about the context configuration, including
 * memory domains, transport resources, and other useful information associated
 * with the context.
 *
 * @param [in] context      Context object whose configuration to print.
 * @param [in] stream       Output stream to print the information to.
 */
void ucp_context_print_info(ucp_context_h context, FILE *stream);


/**
 * @ingroup UCP_WORKER
 * @brief Create a worker object.
 *
 * This routine allocates and initializes a @ref ucp_worker_h "worker" object.
 * Each worker is associated with one and only one @ref ucp_context_h
 * "application" context.  In the same time, an application context can create
 * multiple @ref ucp_worker_h "workers" in order to enable concurrent access to
 * communication resources. For example, application can allocate a dedicated
 * worker for each application thread, where every worker can be progressed
 * independently of others.
 *
 * @note The worker object is allocated within context of the calling thread
 *
 * @param [in] context     Handle to @ref ucp_context_h
 *                         "UCP application context".
 * @param [in] params      User defined @ref ucp_worker_params_t configurations for the
 *                         @ref ucp_worker_h "UCP worker".
 * @param [out] worker_p   A pointer to the worker object allocated by the
 *                         UCP library
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_worker_create(ucp_context_h context,
                               const ucp_worker_params_t *params,
                               ucp_worker_h *worker_p);


/**
 * @ingroup UCP_WORKER
 * @brief Destroy a worker object.
 *
 * This routine releases the resources associated with a
 * @ref ucp_worker_h "UCP worker".
 *
 * @warning Once the UCP worker destroy the worker handle cannot be used with any
 * UCP routine.
 *
 * The destroy process releases and shuts down all resources associated    with
 * the @ref ucp_worker_h "worker".
 *
 * @param [in]  worker        Worker object to destroy.
 */
void ucp_worker_destroy(ucp_worker_h worker);

/**
 * @ingroup UCP_WORKER
 * @brief Get attributes specific to a particular worker.
 *
 * This routine fetches information about the worker.
 *
 * @param [in]  worker     Worker object to query.
 * @param [out] attr       Filled with attributes of worker.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_worker_query(ucp_worker_h worker,
                              ucp_worker_attr_t *attr);

/**
 * @ingroup UCP_WORKER
 * @brief Print information about the worker.
 *
 * This routine prints information about the protocols being used, thresholds,
 * UCT transport methods, and other useful information associated with the worker.
 *
 * @param [in] worker       Worker object to print information for.
 * @param [in] stream       Output stream to print the information to.
 */
void ucp_worker_print_info(ucp_worker_h worker, FILE *stream);


/**
 * @ingroup UCP_WORKER
 * @brief Get the address of the worker object.
 *
 * This routine returns the address of the worker object.  This address can be
 * passed to remote instances of the UCP library in order to to connect to this
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
 * @ingroup UCP_WORKER
 * @brief Release an address of the worker object.
 *
 * This routine release an @ref ucp_address_t "address handle" associated within
 * the @ref ucp_worker_h "worker" object.
 *
 * @warning Once the address released the address handle cannot be used with any
 * UCP routine.
 *
 * @param [in]  worker            Worker object that is associated with the
 *                                address object.
 * @param [in] address            Address to release; the address object has to
 *                                be allocated using @ref ucp_worker_get_address
 *                                "ucp_worker_get_address()" routine.
 *
 * @todo We should consider to change it to return int so we can catch the
 * errors when worker != address
 */
void ucp_worker_release_address(ucp_worker_h worker, ucp_address_t *address);


/**
 * @ingroup UCP_WORKER
 * @brief Progress all communications on a specific worker.
 *
 * This routine explicitly progresses all communication operations on a worker.
 *
 * @note
 * @li Typically, request wait and test routines call @ref
 * ucp_worker_progress "this routine" to progress any outstanding operations.
 * @li Transport layers, implementing asynchronous progress using threads,
 * require callbacks and other user code to be thread safe.
 * @li The state of communication can be advanced (progressed) by blocking
 * routines. Nevertheless, the non-blocking routines can not be used for
 * communication progress.
 *
 * @param [in]  worker    Worker to progress.
 *
 * @return Non-zero if any communication was progressed, zero otherwise.
 */
unsigned ucp_worker_progress(ucp_worker_h worker);


/**
 * @ingroup UCP_WORKER
 * @brief Poll for endpoints that are ready to consume streaming data.
 *
 * This non-blocking routine returns endpoints on a worker which are ready
 * to consume streaming data. The ready endpoints are placed in @poll_eps array,
 * and the function return value indicates how many are there.
 *
 * @param [in]   worker    Worker to poll.
 * @param [out]  poll_eps  Pointer to array of endpoints, should be
 *                         allocated by user.
 * @param [in]   max_eps   Maximal number of endpoints which should be filled
 *                         in @a poll_eps.
 * @param [in]   flags     Reserved for future use.
 *
 * @return Negative value indicates an error according to @ref ucp_status_t.
 *         On success, non-negative value (less or equal @a max_eps) indicates
 *         actual number of endpoints filled in @a poll_eps array.
 *
 */
ssize_t ucp_stream_worker_poll(ucp_worker_h worker,
                               ucp_stream_poll_ep_t *poll_eps, size_t max_eps,
                               unsigned flags);


/**
 * @ingroup UCP_WAKEUP
 * @brief Obtain an event file descriptor for event notification.
 *
 * This routine returns a valid file descriptor for polling functions.
 * The file descriptor will get signaled when an event occurs, as part of the
 * wake-up mechanism. Signaling means a call to poll() or select() with this
 * file descriptor will return at this point, with this descriptor marked as the
 * reason (or one of the reasons) the function has returned. The user does not
 * need to release the obtained file descriptor.
 *
 * The wake-up mechanism exists to allow for the user process to register for
 * notifications on events of the underlying interfaces, and wait until such
 * occur. This is an alternative to repeated polling for request completion.
 * The goal is to allow for waiting while consuming minimal resources from the
 * system. This is recommended for cases where traffic is infrequent, and
 * latency can be traded for lower resource consumption while waiting for it.
 *
 * There are two alternative ways to use the wakeup mechanism: the first is the
 * file descriptor obtained per worker (this function) and the second is the
 * @ref ucp_worker_wait function for waiting on the next event internally.
 *
 * @note UCP @ref ucp_feature "features" have to be triggered
 *   with @ref UCP_FEATURE_WAKEUP to select proper transport
 *
 * @param [in]  worker    Worker of notified events.
 * @param [out] fd        File descriptor.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_worker_get_efd(ucp_worker_h worker, int *fd);


/**
 * @ingroup UCP_WAKEUP
 * @brief Wait for an event of the worker.
 *
 * This routine waits (blocking) until an event has happened, as part of the
 * wake-up mechanism.
 *
 * This function is guaranteed to return only if new communication events occur
 * on the @a worker. Therefore one must drain all existing events before waiting
 * on the file descriptor. This can be achieved by calling
 * @ref ucp_worker_progress repeatedly until it returns 0.
 *
 * There are two alternative ways to use the wakeup mechanism. The first is by
 * polling on a per-worker file descriptor obtained from @ref ucp_worker_get_efd.
 * The second is by using this function to perform an internal wait for the next
 * event associated with the specified worker.
 *
 * @note During the blocking call the wake-up mechanism relies on other means of
 * notification and may not progress some of the requests as it would when
 * calling @ref ucp_worker_progress (which is not invoked in that duration).
 *
 * @note UCP @ref ucp_feature "features" have to be triggered
 *   with @ref UCP_FEATURE_WAKEUP to select proper transport
 *
 * @param [in]  worker    Worker to wait for events on.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_worker_wait(ucp_worker_h worker);


/**
 * @ingroup UCP_WAKEUP
 * @brief Wait for memory update on the address
 *
 * This routine waits for a memory update at the local memory @a address.  This
 * is a blocking routine. The routine returns when the memory address is
 * updated ("write") or an event occurs in the system.
 *
 * This function is guaranteed to return only if new communication events occur
 * on the worker or @a address is modified. Therefore one must drain all existing
 * events before waiting on the file descriptor. This can be achieved by calling
 * @ref ucp_worker_progress repeatedly until it returns 0.
 *
 * @note This routine can be used by an application that executes busy-waiting
 * loop checking for a memory update. Instead of continuous busy-waiting on an
 * address the application can use @a ucp_worker_wait_mem, which may suspend
 * execution until the memory is updated. The goal of the routine is to provide
 * an opportunity for energy savings for architectures that support this
 * functionality.
 *
 * @param [in] address          Local memory address
 */
void ucp_worker_wait_mem(ucp_worker_h worker, void *address);


/**
 * @ingroup UCP_WAKEUP
 * @brief Turn on event notification for the next event.
 *
 * This routine needs to be called before waiting on each notification on this
 * worker, so will typically be called once the processing of the previous event
 * is over, as part of the wake-up mechanism.
 *
 * The worker must be armed before waiting on an event (must be re-armed after
 * it has been signaled for re-use) with @ref ucp_worker_arm.
 * The events triggering a signal of the file descriptor from
 * @ref ucp_worker_get_efd depend on the interfaces used by the worker and
 * defined in the transport layer, and typically represent a request completion
 * or newly available resources. It can also be triggered by calling
 * @ref ucp_worker_signal .
 *
 * The file descriptor is guaranteed to become signaled only if new communication
 * events occur on the @a worker. Therefore one must drain all existing events
 * before waiting on the file descriptor. This can be achieved by calling
 * @ref ucp_worker_progress repeatedly until it returns 0.
 *
 * @code {.c}
 * void application_initialization() {
 *     ...
 *     status = ucp_worker_get_efd(worker, &fd);
 *     ...
 * }
 *
 * void process_comminucation() {
 *     for (;;) {
 *         // check for events as long as progress is made
 *         if (check_for_events()) {
 *              break;                    // got something interesting, exit
 *         } else if (ucp_worker_progress(worker)) {
 *              continue;                 // got something uninteresting, retry
 *         }
 *
 *         // arm the worker and clean-up fd
 *         status = ucp_worker_arm(worker);
 *         if (UCS_OK == status) {
 *             poll(&fds, nfds, timeout);  // wait for events
 *         } else if (UCS_ERR_BUSY == status) {
 *             continue;                   // poll for more events
 *         } else {
 *             abort();
 *         }
 *     }
 * }
 * @endcode
 *
 * @note UCP @ref ucp_feature "features" have to be triggered
 *   with @ref UCP_FEATURE_WAKEUP to select proper transport
 *
 * @param [in]  worker    Worker of notified events.
 *
 * @return ::UCS_OK        The operation completed successfully. File descriptor
 *                         will be signaled by new events.
 * @return ::UCS_ERR_BUSY  There are unprocessed events which prevent the
 *                         file descriptor from being armed. These events should
 *                         be removed by calling @ref ucp_worker_progress().
 *                         The operation is not completed. File descriptor
 *                         will not be signaled by new events.
 * @return @ref ucs_status_t "Other" different error codes in case of issues.
 */
ucs_status_t ucp_worker_arm(ucp_worker_h worker);


/**
 * @ingroup UCP_WAKEUP
 * @brief Cause an event of the worker.
 *
 * This routine signals that the event has happened, as part of the wake-up
 * mechanism. This function causes a blocking call to @ref ucp_worker_wait or
 * waiting on a file descriptor from @ref ucp_worker_get_efd to return, even
 * if no event from the underlying interfaces has taken place.
 *
 * @param [in]  worker    Worker to wait for events on.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_worker_signal(ucp_worker_h worker);


/**
 * @ingroup UCP_WORKER
 * @brief Accept connections on a local address of the worker object.
 *
 * This routine binds the worker object to a @ref ucs_sock_addr_t sockaddr
 * which is set by the user.
 * The worker will listen to incoming connection requests and upon receiving such
 * a request from the remote peer, an endpoint to it will be created.
 * The user's call-back will be invoked once the endpoint is created.
 *
 * @param [in]  worker           Worker object that is associated with the
 *                               params object.
 * @param [in]  params           User defined @ref ucp_listener_params_t
 *                               configurations for the @ref ucp_listener_h.
 * @param [out] listener_p       A handle to the created listener, can be released
 *                               by calling @ref ucp_listener_destroy
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_listener_create(ucp_worker_h worker,
                                 const ucp_listener_params_t *params,
                                 ucp_listener_h *listener_p);


/**
 * @ingroup UCP_WORKER
 * @brief Stop accepting connections on a local address of the worker object.
 *
 * This routine unbinds the worker from the given handle and stops
 * listening for incoming connection requests on it.
 *
 * @param [in] listener        A handle to the listener to stop listening on.
 */
void ucp_listener_destroy(ucp_listener_h listener);


/**
 * @ingroup UCP_ENDPOINT
 * @brief Create and connect an endpoint.
 *
 * This routine creates and connects an @ref ucp_ep_h "endpoint" on a @ref
 * ucp_worker_h "local worker" for a destination @ref ucp_address_t "address"
 * that identifies the remote @ref ucp_worker_h "worker". This function is
 * non-blocking, and communications may begin immediately after it returns. If
 * the connection process is not completed, communications may be delayed.
 * The created @ref ucp_ep_h "endpoint" is associated with one and only one
 * @ref ucp_worker_h "worker".
 *
 * @param [in]  worker      Handle to the worker; the endpoint
 *                          is associated with the worker.
 * @param [in]  params      User defined @ref ucp_ep_params_t configurations
 *                          for the @ref ucp_ep_h "UCP endpoint".
 * @param [out] ep_p        A handle to the created endpoint.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_ep_create(ucp_worker_h worker, const ucp_ep_params_t *params,
                           ucp_ep_h *ep_p);


/**
 * @ingroup UCP_ENDPOINT
 * @brief Modify endpoint parameters.
 *
 * This routine modifies @ref ucp_ep_h "endpoint" created by @ref ucp_ep_create
 * or @ref ucp_listener_accept_callback_t. For example, this API can be used
 * to setup custom parameters like @ref ucp_ep_params_t::user_data or
 * @ref ucp_ep_params_t::err_handler_cb to endpoint created by 
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
 * @ingroup UCP_ENDPOINT
 *
 * @brief Non-blocking @ref ucp_ep_h "endpoint" closure.
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
 * @brief Print endpoint information.
 *
 * This routine prints information about the endpoint transport methods, their
 * thresholds, and other useful information associated with the endpoint.
 *
 * @param [in] ep           Endpoint object whose configuration to print.
 * @param [in] stream       Output stream to print the information to.
 */
void ucp_ep_print_info(ucp_ep_h ep, FILE *stream);


/**
 * @ingroup UCP_ENDPOINT
 *
 * @brief Non-blocking flush of outstanding AMO and RMA operations on the
 * @ref ucp_ep_h "endpoint".
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
 * @return UCS_OK           - The flush operation was completed immediately.
 * @return UCS_PTR_IS_ERR(_ptr) - The flush operation failed.
 * @return otherwise        - Flush operation was scheduled and can be completed
 *                          in any point in time. The request handle is returned
 *                          to the application in order to track progress. The
 *                          application is responsible to release the handle
 *                          using @ref ucp_request_free "ucp_request_free()"
 *                          routine.
 *
 *
 * The following example demonstrates how blocking flush can be implemented
 * using non-blocking flush:
 * @code {.c}
 * void empty_function(void *request, ucs_status_t status)
 * {
 * }
 *
 * ucs_status_t blocking_ep_flush(ucp_ep_h ep, ucp_worker_h worker)
 * {
 *     void *request;
 *
 *     request = ucp_ep_flush_nb(ep, 0, empty_function);
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
 * @endcode */
ucs_status_ptr_t ucp_ep_flush_nb(ucp_ep_h ep, unsigned flags,
                                 ucp_send_callback_t cb);


/**
 * @ingroup UCP_MEM
 * @brief Map or allocate memory for zero-copy operations.
 *
 * This routine maps or/and allocates a user-specified memory segment with @ref
 * ucp_context_h "UCP application context" and the network resources associated
 * with it. If the application specifies NULL as an address for the memory
 * segment, the routine allocates a mapped memory segment and returns its
 * address in the @a address_p argument.  The network stack associated with an
 * application context can typically send and receive data from the mapped
 * memory without CPU intervention; some devices and associated network stacks
 * require the memory to be mapped to send and receive data. The @ref ucp_mem_h
 * "memory handle" includes all information required to access the memory
 * locally using UCP routines, while @ref ucp_rkey_h
 * "remote registration handle" provides an information that is necessary for
 * remote memory access.
 *
 * @note
 * Another well know terminology for the "map" operation that is typically
 * used in the context of networking is memory "registration" or "pinning". The
 * UCP library registers the memory the available hardware so it can be
 * assessed directly by the hardware.
 *
 * Memory mapping assumptions:
 * @li A given memory segment can be mapped by several different communication
 * stacks, if these are compatible.
 * @li The @a memh_p handle returned may be used with any sub-region of the
 * mapped memory.
 * @li If a large segment is registered, and then segmented for subsequent use
 * by a user, then the user is responsible for segmentation and subsequent
 * management.
 *
 * <table>
 * <caption id="ucp_mem_map_matrix">Matrix of behavior</caption>
 * <tr><th>parameter/flag <td align="center">@ref UCP_MEM_MAP_NONBLOCK "NONBLOCK"</td>
 *                        <td align="center">@ref UCP_MEM_MAP_ALLOCATE "ALLOCATE"</td>
 *                        <td align="center">@ref UCP_MEM_MAP_FIXED "FIXED"</td>
 *                        <td align="center">@ref ucp_mem_map_params.address "address"</td>
 *                        <td align="center">@b result
 * <tr><td rowspan="8" align="center">@b value <td rowspan="8" align="center">0/1 - the value\n only affects the\n register/map\n phase</td>
 *                                               <td align="center">0 <td align="center">0 <td align="center">0 <td align="center">@ref anch_err "error"
 * <tr>                                          <td align="center">1 <td align="center">0 <td align="center">0 <td align="center">@ref anch_alloc_reg "alloc+register"
 * <tr>                                          <td align="center">0 <td align="center">1 <td align="center">0 <td align="center">@ref anch_err "error"</td>
 * <tr>                                          <td align="center">0 <td align="center">0 <td align="center">defined <td align="center">@ref anch_reg "register"
 * <tr>                                          <td align="center">1 <td align="center">1 <td align="center">0 <td align="center">@ref anch_err "error"</td>
 * <tr>                                          <td align="center">1 <td align="center">0 <td align="center">defined <td align="center">@ref anch_alloc_hint_reg "alloc+register,hint"
 * <tr>                                          <td align="center">0 <td align="center">1 <td align="center">defined <td align="center">@ref anch_err "error"</td>
 * <tr>                                          <td align="center">1 <td align="center">1 <td align="center">defined <td align="center">@ref anch_alloc_fixed_reg "alloc+register,fixed"
 * </table>
 *
 * @note
 * @li \anchor anch_reg @b register means that the memory will be registered in
 *     corresponding transports for RMA/AMO operations. This case intends that
 *     the memory was allocated by user before.
 * @li \anchor anch_alloc_reg @b alloc+register means that the memory will be allocated
 *     in the memory provided by the system and registered in corresponding
 *     transports for RMA/AMO operations.
 * @li \anchor anch_alloc_hint_reg <b>alloc+register,hint</b> means that
 *     the memory will be allocated with using @ref ucp_mem_map_params.address
 *     as a hint and registered in corresponding transports for RMA/AMO operations.
 * @li \anchor anch_alloc_fixed_reg <b>alloc+register,fixed</b> means that the memory
 *     will be allocated and registered in corresponding transports for RMA/AMO
 *     operations.
 * @li \anchor anch_err @b error is an erroneous combination of the parameters.
 *
 * @param [in]     context    Application @ref ucp_context_h "context" to map
 *                            (register) and allocate the memory on.
 * @param [in]     params     User defined @ref ucp_mem_map_params_t configurations
 *                            for the @ref ucp_mem_h "UCP memory handle".
 * @param [out]    memh_p     UCP @ref ucp_mem_h "handle" for the allocated
 *                            segment.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_mem_map(ucp_context_h context, const ucp_mem_map_params_t *params,
                         ucp_mem_h *memh_p);


/**
 * @ingroup UCP_MEM
 * @brief Unmap memory segment
 *
 * This routine unmaps a user specified memory segment, that was previously
 * mapped using the @ref ucp_mem_map "ucp_mem_map()" routine.  The unmap
 * routine will also release the resources associated with the memory
 * @ref ucp_mem_h "handle".  When the function returns, the @ref ucp_mem_h
 * and associated @ref ucp_rkey_h "remote key" will be invalid and cannot be
 * used with any UCP routine.
 *
 * @note
 * Another well know terminology for the "unmap" operation that is typically
 * used in the context of networking is memory "de-registration". The UCP
 * library de-registers the memory the available hardware so it can be returned
 * back to the operation system.
 *
 * Error cases:
 * @li Once memory is unmapped a network access to the region may cause a
 * failure.
 *
 * @param [in]  context     Application @ref ucp_context_h "context" which was
 *                          used to allocate/map the memory.
 * @param [in]  memh        @ref ucp_mem_h "Handle" to memory region.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_mem_unmap(ucp_context_h context, ucp_mem_h memh);


/**
 * @ingroup UCP_MEM
 * @brief query mapped memory segment
 *
 * This routine returns address and length of memory segment mapped with
 * @ref ucp_mem_map "ucp_mem_map()" routine.
 *
 * @param [in]  memh    @ref ucp_mem_h "Handle" to memory region.
 * @param [out] attr    Filled with attributes of the @ref ucp_mem_h
 *                      "UCP memory handle".
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_mem_query(const ucp_mem_h memh, ucp_mem_attr_t *attr);


/**
 * @ingroup UCP_MEM
 * @brief list of UCP memory use advice.
 *
 * The enumeration list describes memory advice supported by @ref
 * ucp_mem_advise() function.
 */
typedef enum ucp_mem_advice {
    UCP_MADV_NORMAL   = 0,  /**< No special treatment */
    UCP_MADV_WILLNEED       /**< can be used on the memory mapped with
                                 @ref UCP_MEM_MAP_NONBLOCK to speed up memory
                                 mapping and to avoid page faults when 
                                 the memory is accessed for the first time. */
} ucp_mem_advice_t;


/**
 * @ingroup UCP_MEM
 * @brief Tuning parameters for the UCP memory advice.
 *
 * This structure defines the parameters that are used for the
 * UCP memory advice tuning during the @ref ucp_mem_advise "ucp_mem_advise" 
 * routine.
 */
typedef struct ucp_mem_advise_params {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucp_mem_advise_params_field. All fields are mandatory.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t                field_mask;

    /**
     * Memory base address. 
     */
     void                   *address;

     /**
      * Length (in bytes) to allocate or map (register).
      */
     size_t                 length;

     /**
      * Memory use advice @ref ucp_mem_advice
      */
     ucp_mem_advice_t       advice;
} ucp_mem_advise_params_t;


/**
 * @ingroup UCP_MEM
 * @brief give advice about the use of memory
 *
 * This routine advises the UCP about how to handle memory range beginning at
 * address and size of length bytes. This call does not influence the semantics
 * of the application, but may influence its performance. The UCP may ignore 
 * the advice.
 *
 * @param [in]  context     Application @ref ucp_context_h "context" which was
 *                          used to allocate/map the memory.
 * @param [in]  memh        @ref ucp_mem_h "Handle" to memory region.
 * @param [in]  params      Memory base address and length. The advice field 
 *                          is used to pass memory use advice as defined in 
 *                          the @ref ucp_mem_advice list
 *                          The memory range must belong to the @a memh
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_mem_advise(ucp_context_h context, ucp_mem_h memh,  
                            ucp_mem_advise_params_t *params);


/**
 * @ingroup UCP_MEM
 * @brief Pack memory region remote access key.
 *
 * This routine allocates memory buffer and packs into the buffer
 * a remote access key (RKEY) object. RKEY is an opaque object that provides
 * the information that is necessary for remote memory access.
 * This routine packs the RKEY object in a portable format such that the
 * object can be @ref ucp_ep_rkey_unpack "unpacked" on any platform supported by the
 * UCP library. In order to release the memory buffer allocated by this routine
 * the application is responsible to call the @ref ucp_rkey_buffer_release
 * "ucp_rkey_buffer_release()" routine.
 *
 *
 * @note
 * @li RKEYs for InfiniBand and Cray Aries networks typically includes
 * InifiniBand and Aries key.
 * @li In order to enable remote direct memory access to the memory associated
 * with the memory handle the application is responsible to share the RKEY with
 * the peers that will initiate the access.
 *
 * @param [in]  context       Application @ref ucp_context_h "context" which was
 *                            used to allocate/map the memory.
 * @param [in]  memh          @ref ucp_mem_h "Handle" to memory region.
 * @param [out] rkey_buffer_p Memory buffer allocated by the library.
 *                            The buffer contains packed RKEY.
 * @param [out] size_p        Size (in bytes) of the packed RKEY.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_rkey_pack(ucp_context_h context, ucp_mem_h memh,
                           void **rkey_buffer_p, size_t *size_p);


/**
 * @ingroup UCP_MEM
 * @brief Release packed remote key buffer.
 *
 * This routine releases the buffer that was allocated using @ref ucp_rkey_pack
 * "ucp_rkey_pack()".
 *
 * @warning
 * @li Once memory is released an access to the memory may cause a
 * failure.
 * @li If the input memory address was not allocated using
 * @ref ucp_rkey_pack "ucp_rkey_pack()" routine the behaviour of this routine
 * is undefined.
 *
 * @param [in]  rkey_buffer   Buffer to release.
 */
void ucp_rkey_buffer_release(void *rkey_buffer);


/**
 * @ingroup UCP_MEM
 * @brief Create remote access key from packed buffer.
 *
 * This routine unpacks the remote key (RKEY) object into the local memory
 * such that it can be accesses and used  by UCP routines. The RKEY object has
 * to be packed using the @ref ucp_rkey_pack "ucp_rkey_pack()" routine.
 * Application code should not make any alternations to the content of the RKEY
 * buffer.
 *
 * @param [in]  ep            Endpoint to access using the remote key.
 * @param [in]  rkey_buffer   Packed rkey.
 * @param [out] rkey_p        Remote key handle.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_ep_rkey_unpack(ucp_ep_h ep, const void *rkey_buffer,
                                ucp_rkey_h *rkey_p);


/**
 * @ingroup UCP_MEM
 * @brief Get a local pointer to remote memory.
 *
 * This routine returns a local pointer to the remote memory described
 * by the rkey.
 *
 * @note This routine can return a valid pointer only for the endpoints
 * that are reacheble via shared memory.
 *
 * @param [in]  rkey          A remote key handle.
 * @param [in]  raddr         A remote address within the memory area
 *                            described by the rkey.
 * @param [out] addr_p        A pointer that can be used for direct
 *                            access to the remote memory.
 *
 * @return Error code as defined by @ref ucs_status_t if the remote memory
 *         cannot be accessed directly or the remote address is not valid.
 */
ucs_status_t ucp_rkey_ptr(ucp_rkey_h rkey, uint64_t raddr, void **addr_p);


/**
 * @ingroup UCP_MEM
 * @brief Destroy the remote key
 *
 * This routine destroys the RKEY object and the memory that was allocated
 * using the @ref ucp_ep_rkey_unpack "ucp_ep_rkey_unpack()" routine. This
 * routine also releases any resources that are associated with the RKEY
 * object.
 *
 * @warning
 * @li Once the RKEY object is released an access to the memory will cause an
 * undefined failure.
 * @li If the RKEY object was not created using
 * @ref ucp_ep_rkey_unpack "ucp_ep_rkey_unpack()" routine the behaviour of this
 * routine is undefined.
 *
 * @param [in]  rkey         Remote key to destroy.
 */
void ucp_rkey_destroy(ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking stream send operation.
 *
 * This routine sends data that is described by the local address @a buffer,
 * size @a count, and @a datatype object to the destination endpoint @a ep.
 * The routine is non-blocking and therefore returns immediately, however
 * the actual send operation may be delayed. The send operation is considered
 * completed when it is safe to reuse the source @e buffer. If the send
 * operation is completed immediately the routine returns UCS_OK and the
 * call-back function @a cb is @b not invoked. If the operation is
 * @b not completed immediately and no error reported, then the UCP library will
 * schedule invocation of the call-back @a cb upon completion of the send
 * operation. In other words, the completion of the operation will be signaled
 * either by the return code or by the call-back.
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
 *                          that the call-back is only invoked in a case when
 *                          the operation cannot be completed in place.
 * @param [in]  flags       Reserved for future use.
 *
 * @return UCS_OK           - The send operation was completed immediately.
 * @return UCS_PTR_IS_ERR(_ptr) - The send operation failed.
 * @return otherwise        - Operation was scheduled for send and can be
 *                          completed in any point in time. The request handle
 *                          is returned to the application in order to track
 *                          progress of the message. The application is
 *                          responsible to release the handle using
 *                          @ref ucp_request_free routine.
 */
ucs_status_ptr_t ucp_stream_send_nb(ucp_ep_h ep, const void *buffer, size_t count,
                                    ucp_datatype_t datatype, ucp_send_callback_t cb,
                                    unsigned flags);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking tagged-send operations
 *
 * This routine sends a messages that is described by the local address @a
 * buffer, size @a count, and @a datatype object to the destination endpoint
 * @a ep. Each message is associated with a @a tag value that is used for
 * message matching on the @ref ucp_tag_recv_nb "receiver".  The routine is
 * non-blocking and therefore returns immediately, however the actual send
 * operation may be delayed.  The send operation is considered completed when
 * it is safe to reuse the source @e buffer.  If the send operation is
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
 * @return UCS_OK           - The send operation was completed immediately.
 * @return UCS_PTR_IS_ERR(_ptr) - The send operation failed.
 * @return otherwise        - Operation was scheduled for send and can be
 *                          completed in any point in time. The request handle
 *                          is returned to the application in order to track
 *                          progress of the message. The application is
 *                          responsible to released the handle using
 *                          @ref ucp_request_free "ucp_request_free()" routine.
 */
ucs_status_ptr_t ucp_tag_send_nb(ucp_ep_h ep, const void *buffer, size_t count,
                                 ucp_datatype_t datatype, ucp_tag_t tag,
                                 ucp_send_callback_t cb);

/**
 * @ingroup UCP_COMM
 * @brief Non-blocking tagged-send operations with user provided request
 *
 * This routine provides a convenient and efficient way to implement a
 * blocking send pattern. It also completes requests faster than
 * @ref ucp_tag_send_nbr() because:
 * @li it always uses @ref uct_ep_am_bcopy() to send data up to the
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
 *                          responsible to release the handle using
 *                          @ref ucp_request_free "ucp_request_free()" routine.
 */
ucs_status_ptr_t ucp_tag_send_sync_nb(ucp_ep_h ep, const void *buffer, size_t count,
                                      ucp_datatype_t datatype, ucp_tag_t tag,
                                      ucp_send_callback_t cb);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking stream receive operation of structured data into a 
 *        user-supplied buffer.
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
 * @param [in]     buffer   Pointer to the buffer to receive the data to.
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
 * @param [in]     flags    Reserved for future use.
 *
 * @return UCS_OK               - The receive operation was completed
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
 * @brief Non-blocking stream receive operation of unstructured data into
 *        a UCP-supplied buffer.
 *
 * This routine receives any available data from endpoint @a ep.
 * Unlike @ref ucp_stream_recv_nb, the returned data is unstructured and is
 * treated as an array of bytes. If data is immediately available,
 * UCS_STATUS_PTR(_ptr) is returned as a pointer to the data, and @a length
 * is set to the size of the returned data buffer. The routine is non-blocking
 * and therefore returns immediately.
 *
 * @param [in]   ep               UCP endpoint that is used for the receive
 *                                operation.
 * @param [out]  length           Length of received data.
 *
 * @return UCS_OK               - No received data available on the @a ep.
 * @return UCS_PTR_IS_ERR(_ptr) - the receive operation failed and
 *                                UCS_PTR_STATUS(_ptr) indicates an error.
 * @return otherwise            - The pointer to the data UCS_STATUS_PTR(_ptr)
 *                                is returned to the application. After the data
 *                                is processed, the application is responsible
 *                                for releasing the data buffer by calling the
 *                                @ref ucp_stream_data_release routine.
 *
 * @note This function returns packed data (equivalent to ucp_dt_make_contig(1)).
 * @note This function returns a pointer to a UCP-supplied buffer, whereas
 *       @ref ucp_stream_recv_nb places the data into a user-provided buffer.
 *       In some cases, receiving data directly into a UCP-supplied buffer can
 *       be more optimal, for example by processing the incoming data in-place
 *       and thus avoiding extra memory copy operations.
 */
ucs_status_ptr_t ucp_stream_recv_data_nb(ucp_ep_h ep, size_t *length);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking tagged-receive operation.
 *
 * This routine receives a messages that is described by the local address @a
 * buffer, size @a count, and @a datatype object on the @a worker.  The tag
 * value of the receive message has to match the @a tag and @a tag_mask values,
 * where the @a tag_mask indicates what bits of the tag have to be matched. The
 * routine is a non-blocking and therefore returns immediately. The receive
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
 * @param [in]  buffer      Pointer to the buffer to receive the data to.
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
 *                              application is responsible to released the
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
 * This routine receives a message that is described by the local address @a
 * buffer, size @a count, and @a datatype object on the @a worker.  The tag
 * value of the receive message has to match the @a tag and @a tag_mask values,
 * where the @a tag_mask indicates what bits of the tag have to be matched. The
 * routine is a non-blocking and therefore returns immediately. The receive
 * operation is considered completed when the message is delivered to the @a
 * buffer. In order to monitor completion of the operation
 * @ref ucp_request_check_status or @ref ucp_tag_recv_request_test should be
 * used.
 *
 * @param [in]  worker      UCP worker that is used for the receive operation.
 * @param [in]  buffer      Pointer to the buffer to receive the data to.
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
 * @brief Non-blocking probe and return a message.
 *
 * This routine probes (checks) if a messages described by the @a tag and
 * @a tag_mask was received (fully or partially) on the @a worker. The tag
 * value of the received message has to match the @a tag and @a tag_mask
 * values, where the @a tag_mask indicates what bits of the tag have to be
 * matched. The function returns immediately and if the message is matched it
 * returns a handle for the message.
 *
 * @param [in]  worker      UCP worker that is used for the probe operation.
 * @param [in]  tag         Message tag to probe for.
 * @param [in]  tag_mask    Bit mask that indicates the bits that are used for
 *                          the matching of the incoming tag
 *                          against the expected tag.
 * @param [in]  remove      The flag indicates if the matched message has to
 *                          be removed from UCP library.
 *                          If true (1), the message handle is removed from
 *                          the UCP library and the application is responsible
 *                          to call @ref ucp_tag_msg_recv_nb
 *                          "ucp_tag_msg_recv_nb()" in order to receive the data
 *                          and release the resources associated with the
 *                          message handle.
 *                          If false (0), the return value is merely an indication
 *                          to whether a matching message is present, and it cannot
 *                          be used in any other way, and in particular it cannot
 *                          be passed to @ref ucp_tag_msg_recv_nb().
 * @param [out] info        If the matching message is found the descriptor is
 *                          filled with the details about the message.
 *
 * @return NULL                      - No match found.
 * @return Message handle (not NULL) - If message is matched the message handle
 *                                     is returned.
 *
 * @note This function does not advance the communication state of the network.
 *       If this routine is used in busy-poll mode, need to make sure
 *       @ref ucp_worker_progress() is called periodically to extract messages
 *       from the transport.
 */
ucp_tag_message_h ucp_tag_probe_nb(ucp_worker_h worker, ucp_tag_t tag,
                                   ucp_tag_t tag_mask, int remove,
                                   ucp_tag_recv_info_t *info);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking receive operation for a probed message.
 *
 * This routine receives a messages that is described by the local address @a
 * buffer, size @a count, @a message handle, and @a datatype object on the @a
 * worker.  The @a message handle can be obtain by calling the @ref
 * ucp_tag_probe_nb "ucp_tag_probe_nb()" routine.  @ref ucp_tag_msg_recv_nb
 * "ucp_tag_msg_recv_nb()" routine is a non-blocking and therefore returns
 * immediately. The receive operation is considered completed when the message
 * is delivered to the @a buffer.  In order to notify the application about
 * completion of the receive operation the UCP library will invoke the
 * call-back @a cb when the received message is in the receive buffer and ready
 * for application access.  If the receive operation cannot be stated the
 * routine returns an error.
 *
 * @param [in]  worker      UCP worker that is used for the receive operation.
 * @param [in]  buffer      Pointer to the buffer to receive the data to.
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
 *                              application is responsible to released the
 *                              handle using @ref ucp_request_free
 *                              "ucp_request_free()" routine.
 */
ucs_status_ptr_t ucp_tag_msg_recv_nb(ucp_worker_h worker, void *buffer,
                                     size_t count, ucp_datatype_t datatype,
                                     ucp_tag_message_h message,
                                     ucp_tag_recv_callback_t cb);


/**
 * @ingroup UCP_COMM
 * @brief Blocking remote memory put operation.
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
 * @brief Non-blocking implicit remote memory put operation.
 *
 * This routine initiates a storage of contiguous block of data that is
 * described by the local address @a buffer in the remote contiguous memory
 * region described by @a remote_addr address and the @ref ucp_rkey_h "memory
 * handle" @a rkey.  The routine returns immediately and @b does @b not
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
 * @param [in]  remote_addr  Pointer to the destination remote address
 *                           to write to.
 * @param [in]  rkey         Remote memory key associated with the
 *                           remote address.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_put_nbi(ucp_ep_h ep, const void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @brief Blocking remote memory get operation.
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
 * @brief Non-blocking implicit remote memory get operation.
 *
 * This routine initiate a load of contiguous block of data that is described
 * by the remote address @a remote_addr and the @ref ucp_rkey_h "memory handle"
 * @a rkey in the local contiguous memory region described by @a buffer
 * address.  The routine returns immediately and @b does @b not guarantee that
 * remote data is loaded and stored under the local address @e buffer.
 *
 * @note A user can use @ref ucp_worker_flush_nb "ucp_worker_flush_nb()" in order
 * guarantee that remote data is loaded and stored under the local address
 * @e buffer.
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
ucs_status_t ucp_get_nbi(ucp_ep_h ep, void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @brief Blocking atomic add operation for 32 bit integers
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
 * @ingroup UCP_COMM
 * @brief Post an atomic memory operation.
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
 * @param [in] rkey        Remote key handle for the remote address.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_atomic_post(ucp_ep_h ep, ucp_atomic_post_op_t opcode, uint64_t value,
                             size_t op_size, uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @brief Post an atomic fetch operation.
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
 * @param [in] rkey        Remote key handle for the remote address.
 * @param [in] cb          Call-back function that is invoked whenever the
 *                         send operation is completed. It is important to note
 *                         that the call-back function is only invoked in a case when
 *                         the operation cannot be completed in place.
 *
 * @return UCS_OK               - The operation was completed immediately.
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
 * @ingroup UCP_COMM
 * @brief Check the status of non-blocking request.
 *
 * This routine checks the state of the request and returns its current status.
 * Any value different from UCS_INPROGRESS means that request is in a completed
 * state.
 *
 * @param [in]  request     Non-blocking request to check.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_request_check_status(void *request);


/**
 * @ingroup UCP_COMM
 * @brief Check the status and currently available state of non-blocking request
 *        returned from @ref ucp_tag_recv_nb routine.
 *
 * This routine checks the state and returns current status of the request
 * returned from @ref ucp_tag_recv_nb routine or the user allocated request
 * for @ref ucp_tag_recv_nbr. Any value different from UCS_INPROGRESS means
 * that the request is in a completed state.
 *
 * @param [in]  request     Non-blocking request to check.
 * @param [out] info        It is filled with the details about the message
 *                          available at the moment of calling.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_tag_recv_request_test(void *request, ucp_tag_recv_info_t *info);


/**
 * @ingroup UCP_COMM
 * @brief Check the status and currently available state of non-blocking request
 *        returned from @ref ucp_stream_recv_nb routine.
 *
 * This routine checks the state and returns current status of the request
 * returned from @ref ucp_stream_recv_nb routine. Any value different from
 * UCS_INPROGRESS means that the request is in a completed state.
 *
 * @param [in]  request     Non-blocking request to check.
 * @param [out] length_p    The size of the received data in bytes. This value
 *                          is only valid if the status is UCS_OK. If valid, it
 *                          is always an integral multiple of the datatype size
 *                          associated with the request.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_stream_recv_request_test(void *request, size_t *length_p);

/**
 * @ingroup UCP_COMM
 * @brief Cancel an outstanding communications request.
 *
 * @param [in]  worker       UCP worker.
 * @param [in]  request      Non-blocking request to cancel.
 *
 * This routine tries to cancels an outstanding communication request.  After
 * calling this routine, the @a request will be in completed or canceled (but
 * not both) state regardless of the status of the target endpoint associated
 * with the communication request.  If the request is completed successfully,
 * the @ref ucp_send_callback_t "send" or @ref ucp_tag_recv_callback_t
 * "receive" completion callbacks (based on the type of the request) will be
 * called with the @a status argument of the callback set to UCS_OK, and in a
 * case it is canceled the @a status argument is set to UCS_ERR_CANCELED.  It is
 * important to note that in order to release the request back to the library
 * the application is responsible to call @ref ucp_request_free
 * "ucp_request_free()".
 */
void ucp_request_cancel(ucp_worker_h worker, void *request);


/**
 * @ingroup UCP_COMM
 * @brief Release UCP data buffer returned by @ref ucp_stream_recv_data_nb.
 *
 * @param [in]  ep        Endpoint @a data received from.
 * @param [in]  data      Data pointer to release, which was returned from
 *                        @ref ucp_stream_recv_data_nb.
 *
 * This routine releases internal UCP data buffer returned by
 * @ref ucp_stream_recv_data_nb when @a data is processed, the application can't
 * use this buffer after calling this function.
 */
void ucp_stream_data_release(ucp_ep_h ep, void *data);


/**
 * @ingroup UCP_COMM
 * @brief Release a communications request.
 *
 * @param [in]  request      Non-blocking request to release.
 *
 * This routine releases the non-blocking request back to the library, regardless
 * of its current state. Communications operations associated with this request
 * will make progress internally, however no further notifications or callbacks
 * would be invoked for this request.
 */
void ucp_request_free(void *request);


/**
 * @ingroup UCP_DATATYPE
 * @brief Create a generic datatype.
 *
 * This routine create a generic datatype object.
 * The generic datatype is described by the @a ops @ref ucp_generic_dt_ops_t
 * "object" which provides a table of routines defining the operations for
 * generic datatype manipulation. Typically, generic datatypes are used for
 * integration with datatype engines provided with MPI implementations (MPICH,
 * Open MPI, etc).
 * The application is responsible to release the @a datatype_p  object using
 * @ref ucp_dt_destroy "ucp_dt_destroy()" routine.
 *
 * @param [in]  ops          Generic datatype function table as defined by
 *                           @ref ucp_generic_dt_ops_t .
 * @param [in]  context      Application defined context passed to this
 *                           routine.  The context is passed as a parameter
 *                           to the routines in the @a ops table.
 * @param [out] datatype_p   A pointer to datatype object.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_dt_create_generic(const ucp_generic_dt_ops_t *ops, void *context,
                                   ucp_datatype_t *datatype_p);


/**
 * @ingroup UCP_DATATYPE
 * @brief Destroy a datatype and release its resources.
 *
 * This routine destroys the @a datatype object and
 * releases any resources that are associated with the object.
 * The @a datatype object must be allocated using @ref ucp_dt_create_generic
 * "ucp_dt_create_generic()" routine.
 *
 * @warning
 * @li Once the @a datatype object is released an access to this object may
 * cause an undefined failure.
 *
 * @param [in]  datatype     Datatype object to destroy.
 */
void ucp_dt_destroy(ucp_datatype_t datatype);


/**
 * @ingroup UCP_WORKER
 *
 * @brief Assures ordering between non-blocking operations
 *
 * This routine ensures ordering of non-blocking communication operations on
 * the @ref ucp_worker_h "UCP worker".  Communication operations issued on the
 * @a worker prior to this call are guaranteed to be completed before any
 * subsequent communication operations to the same @ref ucp_worker_h "worker"
 * which follow the call to @ref ucp_worker_fence "fence".
 *
 * @note The primary difference between @ref ucp_worker_fence "ucp_worker_fence()"
 * and the @ref ucp_worker_flush_nb "ucp_worker_flush_nb()" is the fact the fence
 * routine does not guarantee completion of the operations on the call return but
 * only ensures the order between communication operations. The
 * @ref ucp_worker_flush_nb "flush" operation on return guarantees that all
 * operations are completed and corresponding memory regions were updated.
 *
 * @param [in] worker        UCP worker.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_worker_fence(ucp_worker_h worker);


/**
 * @ingroup UCP_WORKER
 *
 * @brief Flush outstanding AMO and RMA operations on the @ref ucp_worker_h
 * "worker"
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
 * @return UCS_OK           - The flush operation was completed immediately.
 * @return UCS_PTR_IS_ERR(_ptr) - The flush operation failed.
 * @return otherwise        - Flush operation was scheduled and can be completed
 *                          in any point in time. The request handle is returned
 *                          to the application in order to track progress. The
 *                          application is responsible to release the handle
 *                          using @ref ucp_request_free "ucp_request_free()"
 *                          routine.
 */
ucs_status_ptr_t ucp_worker_flush_nb(ucp_worker_h worker, unsigned flags,
                                     ucp_send_callback_t cb);


/**
 * @example ucp_hello_world.c
 * UCP hello world client / server example utility.
 */

END_C_DECLS

#endif
