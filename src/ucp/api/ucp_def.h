/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
* Copyright (C) IBM 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCP_DEF_H_
#define UCP_DEF_H_

#include <ucs/type/status.h>
#include <ucs/config/types.h>
#include <stddef.h>
#include <stdint.h>


/**
 * @ingroup UCP_CONTEXT
 * @brief UCP receive information descriptor
 *
 * The UCP receive information descriptor is allocated by application and filled
 * in with the information about the received message by @ref ucp_tag_probe_nb
 * or @ref ucp_tag_recv_request_test routines or
 * @ref ucp_tag_recv_callback_t callback argument.
 */
typedef struct ucp_tag_recv_info             ucp_tag_recv_info_t;


/**
 * @ingroup UCP_CONTEXT
 * @brief UCP Application Context
 *
 * UCP application context (or just a context) is an opaque handle that holds a
 * UCP communication instance's global information.  It represents a single UCP
 * communication instance.  The communication instance could be an OS process
 * (an application) that uses UCP library.  This global information includes
 * communication resources, endpoints, memory, temporary file storage, and
 * other communication information directly associated with a specific UCP
 * instance.  The context also acts as an isolation mechanism, allowing
 * resources associated with the context to manage multiple concurrent
 * communication instances. For example, users using both MPI and OpenSHMEM
 * sessions simultaneously can isolate their communication by allocating and
 * using separate contexts for each of them. Alternatively, users can share the
 * communication resources (memory, network resource context, etc.) between
 * them by using the same application context. A message sent or a RMA
 * operation performed in one application context cannot be received in any
 * other application context.
 */
typedef struct ucp_context               *ucp_context_h;


/**
 * @ingroup UCP_CONFIG
 * @brief UCP configuration descriptor
 *
 * This descriptor defines the configuration for @ref ucp_context_h
 * "UCP application context". The configuration is loaded from the run-time
 * environment (using configuration files of environment variables)
 * using @ref ucp_config_read "ucp_config_read" routine and can be printed
 * using @ref ucp_config_print "ucp_config_print" routine. In addition,
 * application is responsible to release the descriptor using
 * @ref ucp_config_release "ucp_config_release" routine.
 *
 * @todo This structure will be modified through a dedicated function.
 */
typedef struct ucp_config                ucp_config_t;


/**
 * @ingroup UCP_ENDPOINT
 * @brief UCP Endpoint
 *
 * The endpoint handle is an opaque object that is used to address a remote
 * @ref ucp_worker_h "worker". It typically provides a description of source,
 * destination, or both. All UCP communication routines address a destination
 * with the endpoint handle. The endpoint handle is associated with only one
 * @ref ucp_context_h "UCP context". UCP provides the @ref ucp_ep_create
 * "endpoint create" routine to create the endpoint handle and the @ref
 * ucp_ep_destroy "destroy" routine to destroy the endpoint handle.
 */
typedef struct ucp_ep                    *ucp_ep_h;


/**
 * @ingroup UCP_ENDPOINT
 * @brief UCP connection request
 *
 * A server-side handle to incoming connection request. Can be used to create an
 * endpoint which connects back to the client.
 */
typedef struct ucp_conn_request          *ucp_conn_request_h;


/**
 * @ingroup UCP_WORKER
 * @brief UCP worker address
 *
 * The address handle is an opaque object that is used as an identifier for a
 * @ref ucp_worker_h "worker" instance.
 */
typedef struct ucp_address               ucp_address_t;


/**
 * @ingroup UCP_ENDPOINT
 * @brief Error handling mode for the UCP endpoint.
 *
 * Specifies error handling mode for the UCP endpoint.
 */
typedef enum {
    UCP_ERR_HANDLING_MODE_NONE,             /**< No guarantees about error
                                             *   reporting, imposes minimal
                                             *   overhead from a performance
                                             *   perspective. @note In this
                                             *   mode, any error reporting will
                                             *   not generate calls to @ref
                                             *   ucp_ep_params_t::err_handler.
                                             */
    UCP_ERR_HANDLING_MODE_PEER              /**< Guarantees that send requests
                                             *   are always completed
                                             *   (successfully or error) even in
                                             *   case of remote failure, disables
                                             *   protocols and APIs which may
                                             *   cause a hang or undefined
                                             *   behavior in case of peer failure,
                                             *   may affect performance and
                                             *   memory footprint */
} ucp_err_handling_mode_t;


/**
 * @ingroup UCP_MEM
 * @brief UCP Remote memory handle
 *
 * Remote memory handle is an opaque object representing remote memory access
 * information. Typically, the handle includes a memory access key and other
 * network hardware specific information, which are input to remote memory
 * access operations, such as PUT, GET, and ATOMIC. The object is
 * communicated to remote peers to enable an access to the memory region.
 */
typedef struct ucp_rkey                  *ucp_rkey_h;


/**
 * @ingroup UCP_MEM
 * @brief UCP Memory handle
 *
 * Memory handle is an opaque object representing a memory region allocated
 * through UCP library, which is optimized for remote memory access
 * operations (zero-copy operations).  The memory handle is a self-contained
 * object, which includes the information required to access the memory region
 * locally, while @ref ucp_rkey_h "remote key" is used to access it
 * remotely. The memory could be registered to one or multiple network resources
 * that are supported by UCP, such as InfiniBand, Gemini, and others.
 */
typedef struct ucp_mem                   *ucp_mem_h;


/**
 * @ingroup UCP_WORKER
 * @brief UCP listen handle.
 *
 * The listener handle is an opaque object that is used for listening on a
 * specific address and accepting connections from clients.
 */
typedef struct ucp_listener              *ucp_listener_h;


/**
 * @ingroup UCP_MEM
 * @brief Attributes of the @ref ucp_mem_h "UCP Memory handle", filled by
 *        @ref ucp_mem_query function.
 */
typedef struct ucp_mem_attr {
   /**
     * Mask of valid fields in this structure, using bits from @ref ucp_mem_attr_field.
     * Fields not specified in this mask would be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t                field_mask;

    /**
     * Address of the memory segment.
     */
     void                   *address;

    /**
     * Size of the memory segment.
     */
     size_t                 length;
} ucp_mem_attr_t;


/**
 * @ingroup UCP_MEM
 * @brief UCP Memory handle attributes field mask.
 *
 * The enumeration allows specifying which fields in @ref ucp_mem_attr_t are
 * present. It is used for the enablement of backward compatibility support.
 */
enum ucp_mem_attr_field {
    UCP_MEM_ATTR_FIELD_ADDRESS = UCS_BIT(0), /**< Virtual address */
    UCP_MEM_ATTR_FIELD_LENGTH  = UCS_BIT(1)  /**< The size of memory region */
};


/**
 * @ingroup UCP_WORKER
 * @brief UCP Worker
 *
 * UCP worker is an opaque object representing the communication context.  The
 * worker represents an instance of a local communication resource and progress
 * engine associated with it. Progress engine is a construct that is
 * responsible for asynchronous and independent progress of communication
 * directives. The progress engine could be implement in hardware or software.
 * The worker object abstract an instance of network resources such as a host
 * channel adapter port, network interface, or multiple resources such as
 * multiple network interfaces or communication ports. It could also represent
 * virtual communication resources that are defined across multiple devices.
 * Although the worker can represent multiple network resources, it is
 * associated with a single @ref ucp_context_h "UCX application context".
 * All communication functions require a context to perform the operation on
 * the dedicated hardware resource(s) and an @ref ucp_ep_h "endpoint" to address the
 * destination.
 *
 * @note Worker are parallel "threading points" that an upper layer may use to
 * optimize concurrent communications.
 */
 typedef struct ucp_worker                *ucp_worker_h;


/**
 * @ingroup UCP_COMM
 * @brief UCP Tag Identifier
 *
 * UCP tag identifier is a 64bit object used for message identification.
 * UCP tag send and receive operations use the object for an implementation
 * tag matching semantics (derivative of MPI tag matching semantics).
 */
typedef uint64_t                         ucp_tag_t;


/**
 * @ingroup UCP_COMM
 * @brief UCP Message descriptor.
 *
 * UCP Message descriptor is an opaque handle for a message returned by
 * @ref ucp_tag_probe_nb. This handle can be passed to @ref ucp_tag_msg_recv_nb
 * in order to receive the message data to a specific buffer.
 */
typedef struct ucp_recv_desc             *ucp_tag_message_h;


/**
 * @ingroup UCP_COMM
 * @brief UCP Datatype Identifier
 *
 * UCP datatype identifier is a 64bit object used for datatype identification.
 * Predefined UCP identifiers are defined by @ref ucp_dt_type.
 */
typedef uint64_t                         ucp_datatype_t;


/**
 * @ingroup UCP_CONTEXT
 * @brief Request initialization callback.
 *
 * This callback routine is responsible for the request initialization.
 *
 * @param [in]  request   Request handle to initialize.
 */
typedef void (*ucp_request_init_callback_t)(void *request);


/**
 * @ingroup UCP_CONTEXT
 * @brief Request cleanup callback.
 *
 * This callback routine is responsible for cleanup of the memory
 * associated with the request.
 *
 * @param [in]  request   Request handle to cleanup.
 */
typedef void (*ucp_request_cleanup_callback_t)(void *request);


/**
 * @ingroup UCP_COMM
 * @brief Completion callback for non-blocking sends.
 *
 * This callback routine is invoked whenever the @ref ucp_tag_send_nb
 * "send operation" is completed. It is important to note that the call-back is
 * only invoked in a case when the operation cannot be completed in place.
 *
 * @param [in]  request   The completed send request.
 * @param [in]  status    Completion status. If the send operation was completed
 *                        successfully UCX_OK is returned. If send operation was
 *                        canceled UCS_ERR_CANCELED is returned.
 *                        Otherwise, an @ref ucs_status_t "error status" is
 *                        returned.
 */
typedef void (*ucp_send_callback_t)(void *request, ucs_status_t status);


 /**
 * @ingroup UCP_COMM
 * @brief Callback to process peer failure.
 *
 * This callback routine is invoked when transport level error detected.
 *
 * @param [in]  arg      User argument to be passed to the callback.
 * @param [in]  ep       Endpoint to handle transport level error. Upon return
 *                       from the callback, this @a ep is no longer usable and
 *                       all subsequent operations on this @a ep will fail with
 *                       the error code passed in @a status.
 * @param [in]  status   @ref ucs_status_t "error status".
 */
typedef void (*ucp_err_handler_cb_t)(void *arg, ucp_ep_h ep, ucs_status_t status);


 /**
 * @ingroup UCP_COMM
 * @brief UCP endpoint error handling context.
 *
 * This structure should be initialized in @ref ucp_ep_params_t to handle peer failure
 */
typedef struct ucp_err_handler {
    ucp_err_handler_cb_t cb;       /**< Error handler callback, if NULL, will
                                        not be called. */
    void                 *arg;     /**< User defined argument associated with
                                        an endpoint, it will be overridden by
                                        @ref ucp_ep_params_t::user_data if both
                                        are set. */
} ucp_err_handler_t;


/**
 * @ingroup UCP_WORKER
 * @brief A callback for accepting client/server connections on a listener
 *        @ref ucp_listener_h.
 *
 *  This callback routine is invoked on the server side upon creating a connection
 *  to a remote client. The user can pass an argument to this callback.
 *  The user is responsible for releasing the @a ep handle using the
 *  @ref ucp_ep_destroy "ucp_ep_destroy()" routine.
 *
 *  @param [in]  ep      Handle to a newly created endpoint which is connected
 *                       to the remote peer which has initiated the connection.
 *  @param [in]  arg     User's argument for the callback.
 */
typedef void (*ucp_listener_accept_callback_t)(ucp_ep_h ep, void *arg);


/**
 * @ingroup UCP_WORKER
 * @brief A callback for handling of incoming connection request @a conn_request
 * from a client.
 *
 * This callback routine is invoked on the server side to handle incoming
 * connections from remote clients. The user can pass an argument to this
 * callback. The @a conn_request handle has to be released, either by @ref
 * ucp_ep_create or @ref ucp_ep_reject routine.
 *
 *  @param [in]  conn_request   Connection request handle.
 *  @param [in]  arg            User's argument for the callback.
 */
typedef void
(*ucp_listener_conn_callback_t)(ucp_conn_request_h conn_request, void *arg);


/**
 * @ingroup UCP_WORKER
 * @brief UCP callback to handle the connection request in a client-server
 * connection establishment flow.
 *
 * This structure is used for handling an incoming connection request on
 * the listener. Setting this type of handler allows creating an endpoint on
 * any other worker and not limited to the worker on which the listener was
 * created.
 * @note
 * - Other than communication progress routines, it is allowed to call all
 *   other communication routines from the callback in the struct.
 * - The callback is thread safe with respect to the worker it is invoked on.
 * - It is the user's responsibility to avoid potential dead lock accessing
 *   different worker.
 */
typedef struct ucp_listener_conn_handler {
   ucp_listener_conn_callback_t cb;      /**< Connection request callback */
   void                         *arg;    /**< User defined argument for the
                                              callback */
} ucp_listener_conn_handler_t;


/**
 * @ingroup UCP_COMM
 * @brief Completion callback for non-blocking stream oriented receives.
 *
 * This callback routine is invoked whenever the @ref ucp_stream_recv_nb
 * "receive operation" is completed and the data is ready in the receive buffer.
 *
 * @param [in]  request   The completed receive request.
 * @param [in]  status    Completion status. If the send operation was completed
 *                        successfully UCX_OK is returned. Otherwise,
 *                        an @ref ucs_status_t "error status" is returned.
 * @param [in]  length    The size of the received data in bytes, always
 *                        boundary of base datatype size. The value is valid
 *                        only if the status is UCS_OK.
 */
typedef void (*ucp_stream_recv_callback_t)(void *request, ucs_status_t status,
                                           size_t length);


/**
 * @ingroup UCP_COMM
 * @brief Completion callback for non-blocking tag receives.
 *
 * This callback routine is invoked whenever the @ref ucp_tag_recv_nb
 * "receive operation" is completed and the data is ready in the receive buffer.
 *
 * @param [in]  request   The completed receive request.
 * @param [in]  status    Completion status. If the send operation was completed
 *                        successfully UCX_OK is returned. If send operation was
 *                        canceled UCS_ERR_CANCELED is returned. If the data can
 *                        not fit into the receive buffer the
 *                        @ref UCS_ERR_MESSAGE_TRUNCATED error code is returned.
 *                        Otherwise, an @ref ucs_status_t "error status" is
 *                        returned.
 * @param [in]  info      @ref ucp_tag_recv_info_t "Completion information"
 *                        The @a info descriptor is Valid only if the status is
 *                        UCS_OK.
 */
typedef void (*ucp_tag_recv_callback_t)(void *request, ucs_status_t status,
                                        ucp_tag_recv_info_t *info);

/**
 * @ingroup UCP_WORKER
 * @brief UCP worker wakeup events mask.
 *
 * The enumeration allows specifying which events are expected on wakeup. Empty
 * events are possible for any type of event except for @ref UCP_WAKEUP_TX and
 * @ref UCP_WAKEUP_RX.
 *
 * @note Send completions are reported by POLLIN-like events (see poll man
 * page). Since outgoing operations can be initiated at any time, UCP does not
 * generate POLLOUT-like events, although it must be noted that outgoing
 * operations may be queued depending upon resource availability.
 */
typedef enum ucp_wakeup_event_types {
    UCP_WAKEUP_RMA         = UCS_BIT(0), /**< Remote memory access send completion */
    UCP_WAKEUP_AMO         = UCS_BIT(1), /**< Atomic operation send completion */
    UCP_WAKEUP_TAG_SEND    = UCS_BIT(2), /**< Tag send completion  */
    UCP_WAKEUP_TAG_RECV    = UCS_BIT(3), /**< Tag receive completion */
    UCP_WAKEUP_TX          = UCS_BIT(10),/**< This event type will generate an
                                              event on completion of any
                                              outgoing operation (complete or
                                              partial, according to the
                                              underlying protocol) for any type
                                              of transfer (send, atomic, or
                                              RMA). */
    UCP_WAKEUP_RX          = UCS_BIT(11),/**< This event type will generate an
                                              event on completion of any receive
                                              operation (complete or partial,
                                              according to the underlying
                                              protocol). */
    UCP_WAKEUP_EDGE        = UCS_BIT(16) /**< Use edge-triggered wakeup. The event
                                              file descriptor will be signaled only
                                              for new events, rather than existing
                                              ones. */
} ucp_wakeup_event_t;


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
     * Destination address; this field should be set along with its
     * corresponding bit in the field_mask - @ref
     * UCP_EP_PARAM_FIELD_REMOTE_ADDRESS and must be obtained using @ref
     * ucp_worker_get_address.
     */
    const ucp_address_t     *address;

    /**
     * Desired error handling mode, optional parameter. Default value is
     * @ref UCP_ERR_HANDLING_MODE_NONE.
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
     * Destination address in the form of a sockaddr; this field should be set
     * along with its corresponding bit in the field_mask - @ref
     * UCP_EP_PARAM_FIELD_SOCK_ADDR and must be obtained from the user, it means
     * that this type of the endpoint creation is possible only on client side
     * in client-server connection establishment flow.
     */
    ucs_sock_addr_t         sockaddr;

    /**
     * Connection request from client; this field should be set along with its
     * corresponding bit in the field_mask - @ref
     * UCP_EP_PARAM_FIELD_CONN_REQUEST and must be obtained from @ref
     * ucp_listener_accept_addr_callback_t, it means that this type of the
     * endpoint creation is possible only on server side in client-server
     * connection establishment flow.
     */
    ucp_conn_request_h      conn_request;

} ucp_ep_params_t;


#endif
