/**
 * @file        uct.h
 * @date        2014-2020
 * @copyright   NVIDIA Corporation. All rights reserved.
 * @copyright   Mellanox Technologies Ltd. All rights reserved.
 * @copyright   Oak Ridge National Laboratory. All rights received.
 * @copyright   Advanced Micro Devices, Inc. All rights received.
 * @brief       Unified Communication Transport
 */

#ifndef UCT_H_
#define UCT_H_

#include <uct/api/uct_def.h>
#include <uct/api/tl.h>
#include <uct/api/version.h>
#include <ucs/async/async_fwd.h>
#include <ucs/datastruct/callbackq.h>
#include <ucs/datastruct/linear_func.h>
#include <ucs/memory/memory_type.h>
#include <ucs/type/status.h>
#include <ucs/type/thread_mode.h>
#include <ucs/type/cpu_set.h>
#include <ucs/stats/stats_fwd.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/sys/topo.h>

#include <sys/socket.h>
#include <stdio.h>
#include <sched.h>

BEGIN_C_DECLS

/** @file uct.h */

/**
 * @defgroup UCT_API Unified Communication Transport (UCT) API
 * @{
 * This section describes UCT API.
 * @}
 */

/**
* @defgroup UCT_RESOURCE   UCT Communication Resource
* @ingroup UCT_API
* @{
* This section describes a concept of the Communication Resource and routines
* associated with the concept.
* @}
*/

/**
 * @defgroup UCT_CONTEXT    UCT Communication Context
 * @ingroup UCT_API
 * @{
 *
 * UCT context abstracts all the resources required for network communication.
 * It is designed to enable either share or isolate resources for multiple
 * programming models used by an application.
 *
 * This section provides a detailed description of this concept and
 * routines associated with it.
 *
 * @}
 */

/**
 * @defgroup UCT_MD    UCT Memory Domain
 * @ingroup UCT_API
 * @{
 * The Memory Domain abstracts resources required for network communication,
 * which typically includes memory, transport mechanisms, compute and
 * network resources. It is an isolation  mechanism that can be employed
 * by the applications for isolating resources between multiple programming models.
 * The attributes of the Memory Domain are defined by the structure @ref uct_md_attr().
 * The communication and memory operations are defined in the context of Memory Domain.
 *
 * @}
 */

/**
 * @defgroup UCT_AM   UCT Active messages
 * @ingroup UCT_API
 * @{
 * Defines active message functions.
 * @}
 */

/**
 * @defgroup UCT_RMA  UCT Remote memory access operations
 * @ingroup UCT_API
 * @{
 * Defines remote memory access operations.
 * @}
 */

/**
 * @defgroup UCT_AMO   UCT Atomic operations
 * @ingroup UCT_API
 * @{
 * Defines atomic operations.
 * @}
 */

/**
 * @defgroup UCT_TAG   UCT Tag matching operations
 * @ingroup UCT_API
 * @{
 * Defines tag matching operations.
 * @}
 */

/**
 * @defgroup UCT_CLIENT_SERVER   UCT client-server operations
 * @ingroup UCT_API
 * @{
 * Defines client-server operations.
 * The client-server API allows the connection establishment between an active
 * side - a client, and its peer - the passive side - a server.
 * The connection can be established through a UCT transport that supports
 * listening and connecting via IP address and port (listening can also be on INADDR_ANY).
 *
 * The following is a general overview of the operations on the server side:
 *
 * Connecting:
 * @ref uct_cm_open
 *      Open a connection manager.
 * @ref uct_listener_create
 *      Create a listener on the CM and start listening on a given IP,port / INADDR_ANY.
 * @ref uct_cm_listener_conn_request_callback_t
 *      This callback is invoked by the UCT transport to handle an incoming connection
 *      request from a client.
 *      Accept or reject the client's connection request.
 * @ref uct_ep_create
 *      Connect to the client by creating an endpoint if the request is accepted.
 *      The server creates a new endpoint for every connection request that it accepts.
 * @ref uct_cm_ep_priv_data_pack_callback_t
 *      This callback is invoked by the UCT transport to fill auxiliary data in
 *      the connection acknowledgement or reject notification back to the client.
 *      Send the client a connection acknowledgement or reject notification.
 *      Wait for an acknowledgment from the client, indicating that it is connected.
 * @ref uct_cm_ep_server_conn_notify_callback_t
 *      This callback is invoked by the UCT transport to handle the connection
 *      notification from the client.
 *
 * Disconnecting:
 * @ref uct_ep_disconnect
 *      Disconnect the server's endpoint from the client.
 *      Can be called when initiating a disconnect or when receiving a disconnect
 *      notification from the remote side.
 * @ref uct_ep_disconnect_cb_t
 *      This callback is invoked by the UCT transport when the client side calls
 *      uct_ep_disconnect as well.
 * @ref uct_ep_destroy
 *      Destroy the endpoint connected to the remote peer.
 *      If this function is called before the endpoint was disconnected, the
 *      @ref uct_ep_disconnect_cb_t will not be invoked.
 *
 * Destroying the server's resources:
 * @ref uct_listener_destroy
 *      Destroy the listener object.
 * @ref uct_cm_close
 *      Close the connection manager.
 *
 * The following is a general overview of the operations on the client side:
 *
 * Connecting:
 * @ref uct_cm_open
 *      Open a connection manager.
 * @ref uct_ep_create
 *      Create an endpoint for establishing a connection to the server.
 * @ref uct_cm_ep_priv_data_pack_callback_t
 *      This callback is invoked by the UCT transport to fill the user's private data
 *      in the connection request to be sent to the server. This connection request
 *      should be created by the transport.
 *      Send the connection request to the server.
 *      Wait for an acknowledgment from the server, indicating that it is connected.
 * @ref uct_cm_ep_client_connect_callback_t
 *      This callback is invoked by the UCT transport to handle a connection response
 *      from the server.
 *      After invoking this callback, the UCT transport will finalize the client's
 *      connection to the server.
 * @ref uct_cm_client_ep_conn_notify
 *      After the client's connection establishment is completed, the client
 *      should call this function in which it sends a notification message to
 *      the server stating that it (the client) is connected.
 *      The notification message that is sent depends on the transport's
 *      implementation.
 *
 * Disconnecting:
 * @ref uct_ep_disconnect
 *      Disconnect the client's endpoint from the server.
 *      Can be called when initiating a disconnect or when receiving a disconnect
 *      notification from the remote side.
 * @ref uct_ep_disconnect_cb_t
 *      This callback is invoked by the UCT transport when the server side calls
 *      uct_ep_disconnect as well.
 * @ref uct_ep_destroy
 *      Destroy the endpoint connected to the remote peer.
 *
 * Destroying the client's resources:
 * @ref uct_cm_close
 *      Close the connection manager.
 *
 * @}
 */

/**
 * @ingroup UCT_RESOURCE
 * @brief Memory domain resource descriptor.
 *
 * This structure describes a memory domain resource.
 */
typedef struct uct_md_resource_desc {
    char                     md_name[UCT_MD_NAME_MAX]; /**< Memory domain name */
} uct_md_resource_desc_t;


/**
 * @ingroup UCT_RESOURCE
 * @brief UCT component attributes field mask
 *
 * The enumeration allows specifying which fields in @ref uct_component_attr_t
 * are present. It is used for backward compatibility support.
 */
enum uct_component_attr_field {
    UCT_COMPONENT_ATTR_FIELD_NAME              = UCS_BIT(0), /**< Component name */
    UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT = UCS_BIT(1), /**< MD resource count */
    UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES      = UCS_BIT(2), /**< MD resources array */
    UCT_COMPONENT_ATTR_FIELD_FLAGS             = UCS_BIT(3)  /**< Capability flags */
};


/**
 * @ingroup UCT_RESOURCE
 * @brief UCT component attributes
 *
 * This structure defines the attributes for UCT component. It is used for
 * @ref uct_component_query
 */
typedef struct uct_component_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_component_attr_field.
     * Fields not specified in this mask will be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t               field_mask;

    /** Component name */
    char                   name[UCT_COMPONENT_NAME_MAX];

    /** Number of memory-domain resources */
    unsigned               md_resource_count;

    /**
     * Array of memory domain resources. When used, it should be initialized
     * prior to calling @ref uct_component_query with a pointer to an array,
     * which is large enough to hold all memory domain resource entries. After
     * the call, this array will be filled with information about existing
     * memory domain resources.
     * In order to allocate this array, you can call @ref uct_component_query
     * twice: The first time would only obtain the amount of entries required,
     * by specifying @ref UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT in
     * field_mask. Then the array could be allocated with the returned number of
     * entries, and passed to a second call to @ref uct_component_query, this
     * time setting field_mask to @ref UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES.
     */
    uct_md_resource_desc_t *md_resources;

    /**
     * Flags as defined by UCT_COMPONENT_FLAG_xx.
     */
    uint64_t               flags;
} uct_component_attr_t;


/**
 * @ingroup UCT_RESOURCE
 * @brief Capability flags of @ref uct_component_h.
 *
 * The enumeration defines bit mask of @ref uct_component_h capabilities in
 * @ref uct_component_attr_t::flags which is set by @ref uct_component_query.
 */
enum {
    /**
     * If set, the component supports @ref uct_cm_h functionality.
     * See @ref uct_cm_open for details.
     */
    UCT_COMPONENT_FLAG_CM = UCS_BIT(0)
};


/**
 * @ingroup UCT_RESOURCE
 * @brief  List of UCX device types.
 */
typedef enum {
    UCT_DEVICE_TYPE_NET,     /**< Network devices */
    UCT_DEVICE_TYPE_SHM,     /**< Shared memory devices */
    UCT_DEVICE_TYPE_ACC,     /**< Acceleration devices */
    UCT_DEVICE_TYPE_SELF,    /**< Loop-back device */
    UCT_DEVICE_TYPE_LAST
} uct_device_type_t;


/**
 * @ingroup UCT_RESOURCE
 * @brief Communication resource descriptor.
 *
 * Resource descriptor is an object representing the network resource.
 * Resource descriptor could represent a stand-alone communication resource
 * such as an HCA port, network interface, or multiple resources such as
 * multiple network interfaces or communication ports. It could also represent
 * virtual communication resources that are defined over a single physical
 * network interface.
 */
typedef struct uct_tl_resource_desc {
    char                     tl_name[UCT_TL_NAME_MAX];   /**< Transport name */
    char                     dev_name[UCT_DEVICE_NAME_MAX]; /**< Hardware device name */
    uct_device_type_t        dev_type;     /**< The device represented by this resource
                                                (e.g. UCT_DEVICE_TYPE_NET for a network interface) */
    ucs_sys_device_t         sys_device;   /**< The identifier associated with the device
                                                bus_id as captured in ucs_sys_bus_id_t struct */
} uct_tl_resource_desc_t;

#define UCT_TL_RESOURCE_DESC_FMT              "%s/%s"
#define UCT_TL_RESOURCE_DESC_ARG(_resource)   (_resource)->tl_name, (_resource)->dev_name


/**
 * @brief Atomic operation requested for uct_ep_atomic32_post, uct_ep_atomic64_post,
 * uct_ep_atomic32_fetch and uct_ep_atomic64_fetch.
 *
 * This enumeration defines which atomic memory operation should be
 * performed by the uct_ep_atomic family of fuctions.
 */
typedef enum uct_atomic_op {
    UCT_ATOMIC_OP_ADD,   /**< Atomic add  */
    UCT_ATOMIC_OP_AND,   /**< Atomic and  */
    UCT_ATOMIC_OP_OR,    /**< Atomic or   */
    UCT_ATOMIC_OP_XOR,   /**< Atomic xor  */
    UCT_ATOMIC_OP_SWAP,  /**< Atomic swap */
    UCT_ATOMIC_OP_CSWAP, /**< Atomic compare-and-swap */
    UCT_ATOMIC_OP_LAST
} uct_atomic_op_t;


/**
 * @defgroup UCT_RESOURCE_IFACE_CAP   UCT interface operations and capabilities
 * @ingroup UCT_RESOURCE
 *
 * @brief  List of capabilities supported by UCX API
 *
 * The definition list presents a full list of operations and capabilities
 * exposed by UCX API.
 * @{
 */
        /* Active message capabilities */
#define UCT_IFACE_FLAG_AM_SHORT       UCS_BIT(0)  /**< Short active message */
#define UCT_IFACE_FLAG_AM_BCOPY       UCS_BIT(1)  /**< Buffered active message */
#define UCT_IFACE_FLAG_AM_ZCOPY       UCS_BIT(2)  /**< Zero-copy active message */

#define UCT_IFACE_FLAG_PENDING        UCS_BIT(3)  /**< Pending operations */

        /* PUT capabilities */
#define UCT_IFACE_FLAG_PUT_SHORT      UCS_BIT(4)  /**< Short put */
#define UCT_IFACE_FLAG_PUT_BCOPY      UCS_BIT(5)  /**< Buffered put */
#define UCT_IFACE_FLAG_PUT_ZCOPY      UCS_BIT(6)  /**< Zero-copy put */

        /* GET capabilities */
#define UCT_IFACE_FLAG_GET_SHORT      UCS_BIT(8)  /**< Short get */
#define UCT_IFACE_FLAG_GET_BCOPY      UCS_BIT(9)  /**< Buffered get */
#define UCT_IFACE_FLAG_GET_ZCOPY      UCS_BIT(10) /**< Zero-copy get */

        /* Atomic operations domain */
#define UCT_IFACE_FLAG_ATOMIC_CPU     UCS_BIT(30) /**< Atomic communications are consistent
                                                       with respect to CPU operations. */
#define UCT_IFACE_FLAG_ATOMIC_DEVICE  UCS_BIT(31) /**< Atomic communications are consistent
                                                       only with respect to other atomics
                                                       on the same device. */

        /* Error handling capabilities */
#define UCT_IFACE_FLAG_ERRHANDLE_SHORT_BUF    UCS_BIT(32) /**< Invalid buffer for short operation */
#define UCT_IFACE_FLAG_ERRHANDLE_BCOPY_BUF    UCS_BIT(33) /**< Invalid buffer for buffered operation */
#define UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF    UCS_BIT(34) /**< Invalid buffer for zero copy operation */
#define UCT_IFACE_FLAG_ERRHANDLE_AM_ID        UCS_BIT(35) /**< Invalid AM id on remote */
#define UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM   UCS_BIT(36) /**< Remote memory access */
#define UCT_IFACE_FLAG_ERRHANDLE_BCOPY_LEN    UCS_BIT(37) /**< Invalid length for buffered operation */
#define UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE UCS_BIT(38) /**< Remote peer failures/outage */

#define UCT_IFACE_FLAG_EP_CHECK               UCS_BIT(39) /**< Endpoint check */

        /* Connection establishment */
#define UCT_IFACE_FLAG_CONNECT_TO_IFACE       UCS_BIT(40) /**< Supports connecting to interface */
#define UCT_IFACE_FLAG_CONNECT_TO_EP          UCS_BIT(41) /**< Supports connecting to specific endpoint */
#define UCT_IFACE_FLAG_CONNECT_TO_SOCKADDR    UCS_BIT(42) /**< Supports connecting to sockaddr */

        /* Special transport flags */
#define UCT_IFACE_FLAG_AM_DUP         UCS_BIT(43) /**< Active messages may be received with duplicates
                                                       This happens if the transport does not keep enough
                                                       information to detect retransmissions */

        /* Callback invocation */
#define UCT_IFACE_FLAG_CB_SYNC        UCS_BIT(44) /**< Interface supports setting a callback
                                                       which is invoked only from the calling context of
                                                       uct_worker_progress() */
#define UCT_IFACE_FLAG_CB_ASYNC       UCS_BIT(45) /**< Interface supports setting a callback
                                                       which will be invoked within a reasonable amount of
                                                       time if uct_worker_progress() is not being called.
                                                       The callback can be invoked from any progress context
                                                       and it may also be invoked when uct_worker_progress()
                                                       is called. */

        /* Keepalive */
#define UCT_IFACE_FLAG_EP_KEEPALIVE   UCS_BIT(46) /**< Transport endpoint has built-in keepalive feature,
                                                       which guarantees the error callback on the transport
                                                       interface will be called if the communication
                                                       channel with remote peer is broken, even if there
                                                       are no outstanding send operations */

        /* Tag matching operations */
#define UCT_IFACE_FLAG_TAG_EAGER_SHORT UCS_BIT(50) /**< Hardware tag matching short eager support */
#define UCT_IFACE_FLAG_TAG_EAGER_BCOPY UCS_BIT(51) /**< Hardware tag matching bcopy eager support */
#define UCT_IFACE_FLAG_TAG_EAGER_ZCOPY UCS_BIT(52) /**< Hardware tag matching zcopy eager support */
#define UCT_IFACE_FLAG_TAG_RNDV_ZCOPY  UCS_BIT(53) /**< Hardware tag matching rendezvous zcopy support */
/**
 * @}
 */


/**
 * @defgroup UCT_RESOURCE_IFACE_EVENT_CAP   UCT interface for asynchronous event capabilities
 * @ingroup UCT_RESOURCE
 *
 * @brief List of capabilities supported by UCT iface event API
 *
 * The definition list presents a full list of operations and capabilities
 * supported by UCT iface event.
 * @{
   */
        /* Event types */
#define UCT_IFACE_FLAG_EVENT_SEND_COMP UCS_BIT(0) /**< Event notification of send completion is
                                                       supported */
#define UCT_IFACE_FLAG_EVENT_RECV      UCS_BIT(1) /**< Event notification of tag and active message
                                                       receive is supported */
#define UCT_IFACE_FLAG_EVENT_RECV_SIG  UCS_BIT(2) /**< Event notification of signaled tag and active
                                                       message is supported */
        /* Event notification mechanisms */
#define UCT_IFACE_FLAG_EVENT_FD        UCS_BIT(3) /**< Event notification through File Descriptor
                                                       is supported */
#define UCT_IFACE_FLAG_EVENT_ASYNC_CB  UCS_BIT(4) /**< Event notification through asynchronous
                                                       callback invocation is supported */
/**
 * @}
 */


/**
 * @ingroup UCT_CONTEXT
 * @brief  Memory allocation methods.
 */
typedef enum {
    UCT_ALLOC_METHOD_THP,  /**< Allocate from OS using libc allocator with
                                Transparent Huge Pages enabled*/
    UCT_ALLOC_METHOD_MD,   /**< Allocate using memory domain */
    UCT_ALLOC_METHOD_HEAP, /**< Allocate from heap using libc allocator */
    UCT_ALLOC_METHOD_MMAP, /**< Allocate from OS using mmap() syscall */
    UCT_ALLOC_METHOD_HUGE, /**< Allocate huge pages */
    UCT_ALLOC_METHOD_LAST,
    UCT_ALLOC_METHOD_DEFAULT = UCT_ALLOC_METHOD_LAST /**< Use default method */
} uct_alloc_method_t;


/**
 * @ingroup UCT_RESOURCE
 * @brief  Asynchronous event types.
 *
 * @note The UCT_EVENT_RECV and UCT_EVENT_RECV_SIG event types are used to
 *       indicate receive-side completions for both tag matching and active
 *       messages. If the interface supports signaled receives
 *       (@ref UCT_IFACE_FLAG_EVENT_RECV_SIG), then for the messages sent with
 *       UCT_SEND_FLAG_SIGNALED flag, UCT_EVENT_RECV_SIG should be triggered
 *       on the receiver. Otherwise, UCT_EVENT_RECV should be triggered.
 */
enum uct_iface_event_types {
    UCT_EVENT_SEND_COMP     = UCS_BIT(0), /**< Send completion event */
    UCT_EVENT_RECV          = UCS_BIT(1), /**< Tag or active message received */
    UCT_EVENT_RECV_SIG      = UCS_BIT(2)  /**< Signaled tag or active message
                                               received */
};


/**
 * @ingroup UCT_RESOURCE
 * @brief  Flush modifiers.
 */
enum uct_flush_flags {
    UCT_FLUSH_FLAG_LOCAL    = 0,            /**< Guarantees that the data
                                                 transfer is completed but the
                                                 target buffer may not be
                                                 updated yet.*/
    UCT_FLUSH_FLAG_CANCEL   = UCS_BIT(0)    /**< The library will make a best
                                                 effort attempt to cancel all
                                                 uncompleted operations.
                                                 However, there is a chance that
                                                 some operations will not be
                                                 canceled in which case the user
                                                 will need to handle their
                                                 completions through
                                                 the relevant callbacks.
                                                 After @ref uct_ep_flush
                                                 with this flag is completed,
                                                 the endpoint will be set to
                                                 error state, and it becomes
                                                 unusable for send operations
                                                 and should be destroyed. */
};


/**
 * @ingroup UCT_RESOURCE
 * @brief UCT progress types
 */
enum uct_progress_types {
    UCT_PROGRESS_SEND        = UCS_BIT(0),  /**< Progress send operations */
    UCT_PROGRESS_RECV        = UCS_BIT(1),  /**< Progress receive operations */
    UCT_PROGRESS_THREAD_SAFE = UCS_BIT(7)   /**< Enable/disable progress while
                                                 another thread may be calling
                                                 @ref ucp_worker_progress(). */
};


/**
 * @ingroup UCT_AM
 * @brief Flags for active message send operation.
 */
enum uct_msg_flags {
    UCT_SEND_FLAG_SIGNALED = UCS_BIT(0) /**< Trigger @ref UCT_EVENT_RECV_SIG
                                             event on remote side. Make best
                                             effort attempt to avoid triggering
                                             @ref UCT_EVENT_RECV event.
                                             Ignored if not supported by interface. */
};


/**
 * @ingroup UCT_RESOURCE
 * @brief Callback flags.
 *
 * List of flags for a callback.
 */
enum uct_cb_flags {
    UCT_CB_FLAG_RESERVED = UCS_BIT(1), /**< Reserved for future use. */
    UCT_CB_FLAG_ASYNC    = UCS_BIT(2)  /**< Callback is allowed to be called
                                            from any thread in the process, and
                                            therefore should be thread-safe. For
                                            example, it may be called from a
                                            transport async progress thread. To
                                            guarantee async invocation, the
                                            interface must have the @ref
                                            UCT_IFACE_FLAG_CB_ASYNC flag set. If
                                            async callback is requested on an
                                            interface which only supports sync
                                            callback (i.e., only the @ref
                                            UCT_IFACE_FLAG_CB_SYNC flag is set),
                                            the callback will be invoked only
                                            from the context that called @ref
                                            uct_iface_progress). */
};


/**
 * @ingroup UCT_RESOURCE
 * @brief Mode in which to open the interface.
 */
enum uct_iface_open_mode {
   /** Interface is opened on a specific device */
   UCT_IFACE_OPEN_MODE_DEVICE          = UCS_BIT(0),

   /** Interface is opened on a specific address on the server side. This mode
       will be deprecated in the near future for a better API. */
   UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER = UCS_BIT(1),

   /** Interface is opened on a specific address on the client side This mode
       will be deprecated in the near future for a better API. */
   UCT_IFACE_OPEN_MODE_SOCKADDR_CLIENT = UCS_BIT(2)
};


/**
 * @ingroup UCT_RESOURCE
 * @brief UCT interface created by @ref uct_iface_open parameters field mask.
 *
 * The enumeration allows specifying which fields in @ref uct_iface_params_t are
 * present, for backward compatibility support.
 */
enum uct_iface_params_field {
    /** Enables @ref uct_iface_params_t::cpu_mask */
    UCT_IFACE_PARAM_FIELD_CPU_MASK           = UCS_BIT(0),

    /** Enables @ref uct_iface_params_t::open_mode */
    UCT_IFACE_PARAM_FIELD_OPEN_MODE          = UCS_BIT(1),

    /** Enables @ref uct_iface_params_t_mode_device
     *  "uct_iface_params_t::mode::device" */
    UCT_IFACE_PARAM_FIELD_DEVICE             = UCS_BIT(2),

    /** Enables @ref uct_iface_params_t_mode_sockaddr
     *  "uct_iface_params_t::mode::sockaddr" */
    UCT_IFACE_PARAM_FIELD_SOCKADDR           = UCS_BIT(3),

    /** Enables @ref uct_iface_params_t::stats_root */
    UCT_IFACE_PARAM_FIELD_STATS_ROOT         = UCS_BIT(4),

    /** Enables @ref uct_iface_params_t::rx_headroom */
    UCT_IFACE_PARAM_FIELD_RX_HEADROOM        = UCS_BIT(5),

    /** Enables @ref uct_iface_params_t::err_handler_arg */
    UCT_IFACE_PARAM_FIELD_ERR_HANDLER_ARG    = UCS_BIT(6),

    /** Enables @ref uct_iface_params_t::err_handler */
    UCT_IFACE_PARAM_FIELD_ERR_HANDLER        = UCS_BIT(7),

    /** Enables @ref uct_iface_params_t::err_handler_flags */
    UCT_IFACE_PARAM_FIELD_ERR_HANDLER_FLAGS  = UCS_BIT(8),

    /** Enables @ref uct_iface_params_t::eager_arg */
    UCT_IFACE_PARAM_FIELD_HW_TM_EAGER_ARG    = UCS_BIT(9),

    /** Enables @ref uct_iface_params_t::eager_cb */
    UCT_IFACE_PARAM_FIELD_HW_TM_EAGER_CB     = UCS_BIT(10),

    /** Enables @ref uct_iface_params_t::rndv_arg */
    UCT_IFACE_PARAM_FIELD_HW_TM_RNDV_ARG     = UCS_BIT(11),

    /** Enables @ref uct_iface_params_t::rndv_cb */
    UCT_IFACE_PARAM_FIELD_HW_TM_RNDV_CB      = UCS_BIT(12),

    /** Enables @ref uct_iface_params_t::async_event_arg */
    UCT_IFACE_PARAM_FIELD_ASYNC_EVENT_ARG    = UCS_BIT(13),

    /** Enables @ref uct_iface_params_t::async_event_cb */
    UCT_IFACE_PARAM_FIELD_ASYNC_EVENT_CB     = UCS_BIT(14),

    /** Enables @ref uct_iface_params_t::keepalive_interval */
    UCT_IFACE_PARAM_FIELD_KEEPALIVE_INTERVAL = UCS_BIT(15)
};

/**
 * @ingroup UCT_MD
 * @brief Socket address accessibility type.
 */
typedef enum {
   UCT_SOCKADDR_ACC_LOCAL,  /**< Check if local address exists.
                                 Address should belong to a local
                                 network interface */
   UCT_SOCKADDR_ACC_REMOTE  /**< Check if remote address can be reached.
                                 Address is routable from one of the
                                 local network interfaces */
} uct_sockaddr_accessibility_t;


/**
 * @ingroup UCT_MD
 * @brief  Memory domain capability flags.
 */
enum {
    UCT_MD_FLAG_ALLOC      = UCS_BIT(0),  /**< MD supports memory allocation */
    UCT_MD_FLAG_REG        = UCS_BIT(1),  /**< MD supports memory registration */
    UCT_MD_FLAG_NEED_MEMH  = UCS_BIT(2),  /**< The transport needs a valid local
                                               memory handle for zero-copy operations */
    UCT_MD_FLAG_NEED_RKEY  = UCS_BIT(3),  /**< The transport needs a valid
                                               remote memory key for remote memory
                                               operations */
    UCT_MD_FLAG_ADVISE     = UCS_BIT(4),  /**< MD supports memory advice */
    UCT_MD_FLAG_FIXED      = UCS_BIT(5),  /**< MD supports memory allocation with
                                               fixed address */
    UCT_MD_FLAG_RKEY_PTR   = UCS_BIT(6),  /**< MD supports direct access to
                                               remote memory via a pointer that
                                               is returned by @ref uct_rkey_ptr */
    UCT_MD_FLAG_SOCKADDR   = UCS_BIT(7)   /**< MD support for client-server
                                               connection establishment via
                                               sockaddr */
};

/**
 * @ingroup UCT_MD
 * @brief  Memory allocation/registration flags.
 */
enum uct_md_mem_flags {
    UCT_MD_MEM_FLAG_NONBLOCK    = UCS_BIT(0), /**< Hint to perform non-blocking
                                                   allocation/registration: page
                                                   mapping may be deferred until
                                                   it is accessed by the CPU or a
                                                   transport. */
    UCT_MD_MEM_FLAG_FIXED       = UCS_BIT(1), /**< Place the mapping at exactly
                                                   defined address */
    UCT_MD_MEM_FLAG_LOCK        = UCS_BIT(2), /**< Registered memory should be
                                                   locked. May incur extra cost for
                                                   registration, but memory access
                                                   is usually faster. */
    UCT_MD_MEM_FLAG_HIDE_ERRORS = UCS_BIT(3), /**< Hide errors on memory registration.
                                                   In some cases registration failure
                                                   is not an error (e. g. for merged
                                                   memory regions). */

    /* memory access flags */
    UCT_MD_MEM_ACCESS_REMOTE_PUT    = UCS_BIT(5), /**< enable remote put access */
    UCT_MD_MEM_ACCESS_REMOTE_GET    = UCS_BIT(6), /**< enable remote get access */
    UCT_MD_MEM_ACCESS_REMOTE_ATOMIC = UCS_BIT(7), /**< enable remote atomic access */
    UCT_MD_MEM_ACCESS_LOCAL_READ    = UCS_BIT(8), /**< enable local read access */
    UCT_MD_MEM_ACCESS_LOCAL_WRITE   = UCS_BIT(9), /**< enable local write access */

    /** enable local and remote access for all operations */
    UCT_MD_MEM_ACCESS_ALL =  (UCT_MD_MEM_ACCESS_REMOTE_PUT|
                              UCT_MD_MEM_ACCESS_REMOTE_GET|
                              UCT_MD_MEM_ACCESS_REMOTE_ATOMIC|
                              UCT_MD_MEM_ACCESS_LOCAL_READ|
                              UCT_MD_MEM_ACCESS_LOCAL_WRITE),

    /** enable local and remote access for put and get operations */
    UCT_MD_MEM_ACCESS_RMA = (UCT_MD_MEM_ACCESS_REMOTE_PUT|
                             UCT_MD_MEM_ACCESS_REMOTE_GET|
                             UCT_MD_MEM_ACCESS_LOCAL_READ|
                             UCT_MD_MEM_ACCESS_LOCAL_WRITE)
};


/**
 * @ingroup UCT_MD
 * @brief list of UCT memory use advice
 */
typedef enum {
    UCT_MADV_NORMAL  = 0,  /**< No special treatment */
    UCT_MADV_WILLNEED      /**< can be used on the memory mapped with
                                @ref UCT_MD_MEM_FLAG_NONBLOCK to speed up
                                memory mapping and to avoid page faults when
                                the memory is accessed for the first time. */
} uct_mem_advice_t;


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief UCT connection manager attributes field mask.
 *
 * The enumeration allows specifying which fields in @ref uct_cm_attr_t are
 * present, for backward compatibility support.
 */
enum uct_cm_attr_field {
    /** Enables @ref uct_cm_attr::max_conn_priv */
    UCT_CM_ATTR_FIELD_MAX_CONN_PRIV = UCS_BIT(0)
};


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief UCT listener attributes field mask.
 *
 * The enumeration allows specifying which fields in @ref uct_listener_attr_t are
 * present, for backward compatibility support.
 */
enum uct_listener_attr_field {
    /** Enables @ref uct_listener_attr::sockaddr */
    UCT_LISTENER_ATTR_FIELD_SOCKADDR = UCS_BIT(0)
};


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief UCT listener created by @ref uct_listener_create parameters field mask.
 *
 * The enumeration allows specifying which fields in @ref uct_listener_params_t
 * are present, for backward compatibility support.
 */
enum uct_listener_params_field {
    /** Enables @ref uct_listener_params::backlog */
    UCT_LISTENER_PARAM_FIELD_BACKLOG         = UCS_BIT(0),

    /** Enables @ref uct_listener_params::conn_request_cb */
    UCT_LISTENER_PARAM_FIELD_CONN_REQUEST_CB = UCS_BIT(1),

    /** Enables @ref uct_listener_params::user_data */
    UCT_LISTENER_PARAM_FIELD_USER_DATA       = UCS_BIT(2)
};


/**
 * @ingroup UCT_RESOURCE
 * @brief UCT endpoint created by @ref uct_ep_create parameters field mask.
 *
 * The enumeration allows specifying which fields in @ref uct_ep_params_t are
 * present, for backward compatibility support.
 */
enum uct_ep_params_field {
    /** Enables @ref uct_ep_params::iface */
    UCT_EP_PARAM_FIELD_IFACE                      = UCS_BIT(0),

    /** Enables @ref uct_ep_params::user_data */
    UCT_EP_PARAM_FIELD_USER_DATA                  = UCS_BIT(1),

    /** Enables @ref uct_ep_params::dev_addr */
    UCT_EP_PARAM_FIELD_DEV_ADDR                   = UCS_BIT(2),

    /** Enables @ref uct_ep_params::iface_addr */
    UCT_EP_PARAM_FIELD_IFACE_ADDR                 = UCS_BIT(3),

    /** Enables @ref uct_ep_params::sockaddr */
    UCT_EP_PARAM_FIELD_SOCKADDR                   = UCS_BIT(4),

    /** Enables @ref uct_ep_params::sockaddr_cb_flags */
    UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS          = UCS_BIT(5),

    /** Enables @ref uct_ep_params::sockaddr_pack_cb */
    UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB           = UCS_BIT(6),

    /** Enables @ref uct_ep_params::cm */
    UCT_EP_PARAM_FIELD_CM                         = UCS_BIT(7),

    /** Enables @ref uct_ep_params::conn_request */
    UCT_EP_PARAM_FIELD_CONN_REQUEST               = UCS_BIT(8),

    /** Enables @ref uct_ep_params::sockaddr_cb_client */
    UCT_EP_PARAM_FIELD_SOCKADDR_CONNECT_CB_CLIENT = UCS_BIT(9),

    /** Enables @ref uct_ep_params::sockaddr_cb_server */
    UCT_EP_PARAM_FIELD_SOCKADDR_NOTIFY_CB_SERVER  = UCS_BIT(10),

    /** Enables @ref uct_ep_params::disconnect_cb */
    UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB     = UCS_BIT(11),

    /** Enables @ref uct_ep_params::path_index */
    UCT_EP_PARAM_FIELD_PATH_INDEX                 = UCS_BIT(12)
};


/*
 * @ingroup UCT_RESOURCE
 * @brief Process Per Node (PPN) bandwidth specification: f(ppn) = dedicated + shared / ppn
 *
 *  This structure specifies a function which is used as basis for bandwidth
 * estimation of various UCT operations. This information can be used to select
 * the best performing combination of UCT operations.
 */
typedef struct uct_ppn_bandwidth {
    double                   dedicated; /**< Dedicated bandwidth, bytes/second */
    double                   shared;    /**< Shared bandwidth, bytes/second */
} uct_ppn_bandwidth_t;


/**
 * @ingroup UCT_RESOURCE
 * @brief Interface attributes: capabilities and limitations.
 */
struct uct_iface_attr {
    struct {
        struct {
            size_t           max_short;  /**< Maximal size for put_short */
            size_t           max_bcopy;  /**< Maximal size for put_bcopy */
            size_t           min_zcopy;  /**< Minimal size for put_zcopy (total
                                              of @ref uct_iov_t::length of the
                                              @a iov parameter) */
            size_t           max_zcopy;  /**< Maximal size for put_zcopy (total
                                              of @ref uct_iov_t::length of the
                                              @a iov parameter) */
            size_t           opt_zcopy_align; /**< Optimal alignment for zero-copy
                                              buffer address */
            size_t           align_mtu;       /**< MTU used for alignment */
            size_t           max_iov;    /**< Maximal @a iovcnt parameter in
                                              @ref ::uct_ep_put_zcopy
                                              @anchor uct_iface_attr_cap_put_max_iov */
        } put;                           /**< Attributes for PUT operations */

        struct {
            size_t           max_short;  /**< Maximal size for get_short */
            size_t           max_bcopy;  /**< Maximal size for get_bcopy */
            size_t           min_zcopy;  /**< Minimal size for get_zcopy (total
                                              of @ref uct_iov_t::length of the
                                              @a iov parameter) */
            size_t           max_zcopy;  /**< Maximal size for get_zcopy (total
                                              of @ref uct_iov_t::length of the
                                              @a iov parameter) */
            size_t           opt_zcopy_align; /**< Optimal alignment for zero-copy
                                              buffer address */
            size_t           align_mtu;       /**< MTU used for alignment */
            size_t           max_iov;    /**< Maximal @a iovcnt parameter in
                                              @ref uct_ep_get_zcopy
                                              @anchor uct_iface_attr_cap_get_max_iov */
        } get;                           /**< Attributes for GET operations */

        struct {
            size_t           max_short;  /**< Total maximum size (incl. the header)
                                              @anchor uct_iface_attr_cap_am_max_short */
            size_t           max_bcopy;  /**< Total maximum size (incl. the header) */
            size_t           min_zcopy;  /**< Minimal size for am_zcopy (incl. the
                                              header and total of @ref uct_iov_t::length
                                              of the @a iov parameter) */
            size_t           max_zcopy;  /**< Total max. size (incl. the header
                                              and total of @ref uct_iov_t::length
                                              of the @a iov parameter) */
            size_t           opt_zcopy_align; /**< Optimal alignment for zero-copy
                                              buffer address */
            size_t           align_mtu;       /**< MTU used for alignment */
            size_t           max_hdr;    /**< Max. header size for zcopy */
            size_t           max_iov;    /**< Maximal @a iovcnt parameter in
                                              @ref ::uct_ep_am_zcopy
                                              @anchor uct_iface_attr_cap_am_max_iov */
        } am;                            /**< Attributes for AM operations */

        struct {
            struct {
                size_t       min_recv;   /**< Minimal allowed length of posted receive buffer */
                size_t       max_zcopy;  /**< Maximal allowed data length in
                                              @ref uct_iface_tag_recv_zcopy */
                size_t       max_iov;    /**< Maximal @a iovcnt parameter in
                                              @ref uct_iface_tag_recv_zcopy
                                              @anchor uct_iface_attr_cap_tag_recv_iov */
                size_t       max_outstanding; /**< Maximal number of simultaneous
                                                   receive operations */
            } recv;

            struct {
                  size_t     max_short;  /**< Maximal allowed data length in
                                              @ref uct_ep_tag_eager_short */
                  size_t     max_bcopy;  /**< Maximal allowed data length in
                                              @ref uct_ep_tag_eager_bcopy */
                  size_t     max_zcopy;  /**< Maximal allowed data length in
                                              @ref uct_ep_tag_eager_zcopy */
                  size_t     max_iov;    /**< Maximal @a iovcnt parameter in
                                              @ref uct_ep_tag_eager_zcopy */
            } eager;                     /**< Attributes related to eager protocol */

            struct {
                  size_t     max_zcopy;  /**< Maximal allowed data length in
                                              @ref uct_ep_tag_rndv_zcopy */
                  size_t     max_hdr;    /**< Maximal allowed header length in
                                              @ref uct_ep_tag_rndv_zcopy and
                                              @ref uct_ep_tag_rndv_request */
                  size_t     max_iov;    /**< Maximal @a iovcnt parameter in
                                              @ref uct_ep_tag_rndv_zcopy */
            } rndv;                      /**< Attributes related to rendezvous protocol */
        } tag;                           /**< Attributes for TAG operations */

        struct {
            uint64_t         op_flags;   /**< Attributes for atomic-post operations */
            uint64_t         fop_flags;  /**< Attributes for atomic-fetch operations */
        } atomic32, atomic64;            /**< Attributes for atomic operations */

        uint64_t             flags;      /**< Flags from @ref UCT_RESOURCE_IFACE_CAP */
        uint64_t             event_flags;/**< Flags from @ref UCT_RESOURCE_IFACE_EVENT_CAP */
    } cap;                               /**< Interface capabilities */

    size_t                   device_addr_len;/**< Size of device address */
    size_t                   iface_addr_len; /**< Size of interface address */
    size_t                   ep_addr_len;    /**< Size of endpoint address */
    size_t                   max_conn_priv;  /**< Max size of the iface's private data.
                                                  used for connection
                                                  establishment with sockaddr */
    struct sockaddr_storage  listen_sockaddr; /**< Sockaddr on which this iface
                                                   is listening. */
    /*
     * The following fields define expected performance of the communication
     * interface, this would usually be a combination of device and system
     * characteristics and determined at run time.
     */
    double                   overhead;     /**< Message overhead, seconds */
    uct_ppn_bandwidth_t      bandwidth;    /**< Bandwidth model */
    ucs_linear_func_t        latency;      /**< Latency as function of number of
                                                active endpoints */
    uint8_t                  priority;     /**< Priority of device */
    size_t                   max_num_eps;  /**< Maximum number of endpoints */
    unsigned                 dev_num_paths;/**< How many network paths can be
                                                utilized on the device used by
                                                this interface for optimal
                                                performance. Endpoints that connect
                                                to the same remote address but use
                                                different paths can potentially
                                                achieve higher total bandwidth
                                                compared to using only a single
                                                endpoint. */
};


/**
 * @ingroup UCT_RESOURCE
 * @brief Parameters used for interface creation.
 *
 * This structure should be allocated by the user and should be passed to
 * @ref uct_iface_open. User has to initialize all fields of this structure.
 */
struct uct_iface_params {
    /** Mask of valid fields in this structure, using bits from
     *  @ref uct_iface_params_field. Fields not specified in this mask will be
     *  ignored. */
    uint64_t                                     field_mask;
    /** Mask of CPUs to use for resources */
    ucs_cpu_set_t                                cpu_mask;
    /** Interface open mode bitmap. @ref uct_iface_open_mode */
    uint64_t                                     open_mode;
    /** Mode-specific parameters */
    union {
        /** @anchor uct_iface_params_t_mode_device
         *  The fields in this structure (tl_name and dev_name) need to be set only when
         *  the @ref UCT_IFACE_OPEN_MODE_DEVICE bit is set in @ref
         *  uct_iface_params_t.open_mode This will make @ref uct_iface_open
         *  open the interface on the specified device.
         */
        struct {
            const char                           *tl_name;  /**< Transport name */
            const char                           *dev_name; /**< Device Name */
        } device;
        /** @anchor uct_iface_params_t_mode_sockaddr
         *  These callbacks and address are only relevant for client-server
         *  connection establishment with sockaddr and are needed on the server side.
         *  The callbacks and address need to be set when the @ref
         *  UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER bit is set in @ref
         *  uct_iface_params_t.open_mode. This will make @ref uct_iface_open
         *  open the interface on the specified address as a server. */
        struct {
            ucs_sock_addr_t                      listen_sockaddr;
            /** Argument for connection request callback */
            void                                 *conn_request_arg;
            /** Callback for an incoming connection request on the server */
            uct_sockaddr_conn_request_callback_t conn_request_cb;
            /** Callback flags to indicate where the callback can be invoked from.
             * @ref uct_cb_flags */
            uint32_t                             cb_flags;
        } sockaddr;
    } mode;

    /** Root in the statistics tree. Can be NULL. If non NULL, it will be
        a root of @a uct_iface object in the statistics tree. */
    ucs_stats_node_t                             *stats_root;
    /** How much bytes to reserve before the receive segment.*/
    size_t                                       rx_headroom;

    /** Custom argument of @a err_handler. */
    void                                         *err_handler_arg;
    /** The callback to handle transport level error.*/
    uct_error_handler_t                          err_handler;
    /** Callback flags to indicate where the @a err_handler callback can be
     * invoked from. @ref uct_cb_flags */
    uint32_t                                     err_handler_flags;

    /** These callbacks are only relevant for HW Tag Matching */
    void                                         *eager_arg;
    /** Callback for tag matching unexpected eager messages */
    uct_tag_unexp_eager_cb_t                     eager_cb;
    void                                         *rndv_arg;
    /** Callback for tag matching unexpected rndv messages */
    uct_tag_unexp_rndv_cb_t                      rndv_cb;

    void                                         *async_event_arg;
    /** Callback for asynchronous event handling. The callback will be
     * invoked from UCT transport when there are new events to be
     * read by user if the iface has @ref UCT_IFACE_FLAG_EVENT_ASYNC_CB
     * capability */
    uct_async_event_cb_t                         async_event_cb;

    /* Time period between keepalive rounds */
    ucs_time_t                                   keepalive_interval;
};


/**
 * @ingroup UCT_RESOURCE
 * @brief Parameters for creating a UCT endpoint by @ref uct_ep_create
 */
struct uct_ep_params {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_ep_params_field. Fields not specified by this mask will be
     * ignored.
     */
    uint64_t                          field_mask;

    /**
     * Interface to create the endpoint on.
     * Either @a iface or @a cm field must be initialized but not both.
     */
    uct_iface_h                       iface;

    /**
     * User data associated with the endpoint.
     */
    void                              *user_data;

    /**
     * The device address to connect to on the remote peer. This must be defined
     * together with @ref uct_ep_params_t::iface_addr to create an endpoint
     * connected to a remote interface.
     */
    const uct_device_addr_t           *dev_addr;

    /**
     * This specifies the remote address to use when creating an endpoint that
     * is connected to a remote interface.
     * @note This requires @ref UCT_IFACE_FLAG_CONNECT_TO_IFACE capability.
     */
    const uct_iface_addr_t            *iface_addr;

    /**
     * The sockaddr to connect to on the remote peer. If set, @ref uct_ep_create
     * will create an endpoint for a connection to the remote peer, specified by
     * its socket address.
     * @note The interface in this routine requires the
     * @ref UCT_IFACE_FLAG_CONNECT_TO_SOCKADDR capability.
     */
    const ucs_sock_addr_t             *sockaddr;

    /**
     * @ref uct_cb_flags to indicate @ref uct_ep_params_t::sockaddr_pack_cb
     * behavior. If @ref uct_ep_params_t::sockaddr_pack_cb is not set, this
     * field will be ignored.
     */
    uint32_t                          sockaddr_cb_flags;

    /**
     * Callback that will be used for filling the user's private data to be
     * delivered to the remote peer by the callback on the server or client side.
     * This field is only valid if @ref uct_ep_params_t::sockaddr is set.
     * @note It is never guaranteed that the callaback will be called. If, for
     * example, the endpoint goes into error state before issuing the connection
     * request, the callback will not be invoked.
     */
    uct_cm_ep_priv_data_pack_callback_t sockaddr_pack_cb;

    /**
     * The connection manager object as created by @ref uct_cm_open.
     * Either @a cm or @a iface field must be initialized but not both.
     */
    uct_cm_h                          cm;

    /**
     * Connection request that was passed to
     * @ref uct_cm_listener_conn_request_args_t::conn_request.
     * @note After a call to @ref uct_ep_create, @a params.conn_request is
     *       consumed and should not be used anymore, even if the call returns
     *       with an error.
     */
    uct_conn_request_h                conn_request;

    /**
     * Callback that will be invoked when the endpoint on the client side
     * is being connected to the server by a connection manager @ref uct_cm_h .
     */
    uct_cm_ep_client_connect_callback_t      sockaddr_cb_client;

    /**
     * Callback that will be invoked when the endpoint on the server side
     * is being connected to a client by a connection manager @ref uct_cm_h .
     */
    uct_cm_ep_server_conn_notify_callback_t  sockaddr_cb_server;

    /**
     * Callback that will be invoked when the endpoint is disconnected.
     */
    uct_ep_disconnect_cb_t              disconnect_cb;

    /**
     * Index of the path which the endpoint should use, must be in the range
     * 0..(@ref uct_iface_attr_t.dev_num_paths - 1).
     */
    unsigned                            path_index;
};


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Connection manager attributes, capabilities and limitations.
 */
struct uct_cm_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_cm_attr_field. Fields not specified by this mask
     * will be ignored.
     */
    uint64_t    field_mask;

    /**
     * Max size of the connection manager's private data used for connection
     * establishment with sockaddr.
     */
    size_t      max_conn_priv;
};


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief UCT listener attributes, capabilities and limitations.
 */
struct uct_listener_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_listener_attr_field. Fields not specified by this mask
     * will be ignored.
     */
    uint64_t                field_mask;

    /**
     * Sockaddr on which this listener is listening.
     */
    struct sockaddr_storage sockaddr;
};


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Parameters for creating a listener object @ref uct_listener_h by
 * @ref uct_listener_create
 */
struct uct_listener_params {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_listener_params_field. Fields not specified by this mask
     * will be ignored.
     */
    uint64_t                                field_mask;

    /**
     * Backlog of incoming connection requests. If specified, must be a positive value.
     * If not specified, each CM component will use its maximal allowed value,
     * based on the system's setting.
     */
    int                                     backlog;

    /**
     * Callback function for handling incoming connection requests.
     */
    uct_cm_listener_conn_request_callback_t conn_request_cb;

    /**
     * User data associated with the listener.
     */
    void                                    *user_data;
};


/**
 * @ingroup UCT_MD
 * @brief  Memory domain attributes.
 *
 * This structure defines the attributes of a Memory Domain which includes
 * maximum memory that can be allocated, credentials required for accessing the memory,
 * CPU mask indicating the proximity of CPUs, and bitmaps indicating the types
 * of memory (CPU/CUDA/ROCM) that can be detected, allocated and accessed.
 */
struct uct_md_attr {
    struct {
        size_t               max_alloc; /**< Maximal allocation size */
        size_t               max_reg;   /**< Maximal registration size */
        uint64_t             flags;     /**< UCT_MD_FLAG_xx */
        uint64_t             reg_mem_types; /**< Bitmap of memory types that Memory Domain can be registered with */
        uint64_t             detect_mem_types; /**< Bitmap of memory types that Memory Domain can detect if address belongs to it */
        uint64_t             alloc_mem_types;  /**< Bitmap of memory types that Memory Domain can allocate memory on */
        uint64_t             access_mem_types; /**< Memory types that Memory Domain can access */
    } cap;

    ucs_linear_func_t        reg_cost;  /**< Memory registration cost estimation
                                             (time,seconds) as a linear function
                                             of the buffer size. */

    char                     component_name[UCT_COMPONENT_NAME_MAX]; /**< Component name */
    size_t                   rkey_packed_size; /**< Size of buffer needed for packed rkey */
    ucs_cpu_set_t            local_cpus;    /**< Mask of CPUs near the resource */
};


/**
 * @ingroup UCT_MD
 * @brief UCT MD memory attributes field mask
 *
 * The enumeration allows specifying which fields in @ref uct_md_mem_attr_t
 * are present.
 */
enum uct_md_mem_attr_field {
    UCT_MD_MEM_ATTR_FIELD_MEM_TYPE = UCS_BIT(0), /**< Indicate if memory type
                                                      is populated. E.g. CPU/GPU */
    UCT_MD_MEM_ATTR_FIELD_SYS_DEV  = UCS_BIT(1)  /**< Indicate if details of
                                                      system device backing
                                                      the pointer are populated.
                                                      E.g. NUMA/GPU */
};


/**
 * @ingroup UCT_MD
 * @brief  Memory domain attributes.
 *
 * This structure defines the attributes of a memory pointer which may
 * include the memory type of the pointer, and the system device that backs
 * the pointer depending on the bit fields populated in field_mask.
 */
typedef struct uct_md_mem_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_md_mem_attr_t. Note that the field mask is
     * populated upon return from uct_md_mem_query and not set by user.
     * Subsequent use of members of the structure are valid after ensuring that
     * relevant bits in the field_mask are set.
     */
    uint64_t          field_mask;

    /**
     * The type of memory. E.g. CPU/GPU memory or some other valid type
     */
    ucs_memory_type_t mem_type;

    /**
     * Index of the system device on which the buffer resides. eg: NUMA/GPU
     */
    ucs_sys_device_t  sys_dev;
} uct_md_mem_attr_t;


/**
 * @ingroup UCT_MD
 * @brief Query attributes of a given pointer
 *
 * Return attributes such as memory type, and system device for the
 * given pointer of specific length.
 *
 * @param [in]     md          Memory domain to run the query on. This function
 *                             returns an error if the md does not recognize the
 *                             pointer.
 * @param [in]     address     The address of the pointer. Must be non-NULL
 *                             else UCS_ERR_INVALID_PARAM error is returned.
 * @param [in]     length      Length of the memory region to examine.
 *                             Must be nonzero else UCS_ERR_INVALID_PARAM error
 *                             is returned.
 * @param [out]    mem_attr    If successful, filled with ptr attributes.
 *
 * @return Error code.
 */
ucs_status_t uct_md_mem_query(uct_md_h md, const void *address, const size_t length,
                              uct_md_mem_attr_t *mem_attr);


/**
 * @ingroup UCT_MD
 * @brief Describes a memory allocated by UCT.
 *
 * This structure describes the memory block which includes the address, size, and
 * Memory Domain used for allocation. This structure is passed to interface
 * and the memory is allocated by memory allocation functions @ref uct_mem_alloc.
 */
typedef struct uct_allocated_memory {
    void                     *address; /**< Address of allocated memory */
    size_t                   length;   /**< Real size of allocated memory */
    uct_alloc_method_t       method;   /**< Method used to allocate the memory */
    ucs_memory_type_t        mem_type; /**< type of allocated memory */
    uct_md_h                 md;       /**< if method==MD: MD used to allocate the memory */
    uct_mem_h                memh;     /**< if method==MD: MD memory handle */
} uct_allocated_memory_t;


/**
 * @ingroup UCT_MD
 * @brief Remote key with its type
 *
 * This structure describes the credentials (typically key) and information
 * required to access the remote memory by the communication interfaces.
 */
typedef struct uct_rkey_bundle {
    uct_rkey_t               rkey;    /**< Remote key descriptor, passed to RMA functions */
    void                     *handle; /**< Handle, used internally for releasing the key */
    void                     *type;   /**< Remote key type */
} uct_rkey_bundle_t;


/**
 * @ingroup UCT_RESOURCE
 * @brief Completion handle.
 *
 * This structure should be allocated by the user and can be passed to communication
 * primitives. The user must initialize all fields of the structure.
 *  If the operation returns UCS_INPROGRESS, this structure will be in use by the
 * transport until the operation completes. When the operation completes, "count"
 * field is decremented by 1, and whenever it reaches 0 - the callback is called.
 *
 * Notes:
 *  - The same structure can be passed multiple times to communication functions
 *    without the need to wait for completion.
 *  - If the number of operations is smaller than the initial value of the counter,
 *    the callback will not be called at all, so it may be left undefined.
 *  - status field is required to track the first time the error occurred, and
 *    report it via a callback when count reaches 0.
 */
struct uct_completion {
    uct_completion_callback_t func;    /**< User callback function */
    int                       count;   /**< Completion counter */
    ucs_status_t              status;  /**< Completion status, this field must
                                            be initialized with UCS_OK before
                                            first operation is started. */
};


/**
 * @ingroup UCT_RESOURCE
 * @brief Pending request.
 *
 * This structure should be passed to @ref uct_ep_pending_add() and is used to signal
 * new available resources back to user.
 */
struct uct_pending_req {
    uct_pending_callback_t    func;   /**< User callback function */
    char                      priv[UCT_PENDING_REQ_PRIV_LEN]; /**< Used internally by UCT */
};


/**
 * @ingroup UCT_TAG
 * @brief Posted tag context.
 *
 * Tag context is an object which tracks a tag posted to the transport. It
 * contains callbacks for matching events on this tag.
 */
struct uct_tag_context {
    /**
     * Tag is consumed by the transport and should not be matched in software.
     *
     * @param [in]  self    Pointer to relevant context structure, which was
     *                      initially passed to @ref uct_iface_tag_recv_zcopy.
     */
    void (*tag_consumed_cb)(uct_tag_context_t *self);

    /**
     * Tag processing is completed by the transport.
     *
     * @param [in]  self        Pointer to relevant context structure, which was
     *                          initially passed to @ref uct_iface_tag_recv_zcopy.
     * @param [in]  stag        Tag from sender.
     * @param [in]  imm         Immediate data from sender. For rendezvous, it's always 0.
     * @param [in]  length      Completed length.
     * @param [in]  inline_data If non-null, points to a temporary buffer which contains
                                the received data. In this case the received data was not
                                placed directly in the receive buffer. This callback routine
                                is responsible for copy-out the inline data, otherwise it is
                                released.
     * @param [in]  status  Completion status:
     * (a)   UCS_OK - Success, data placed in provided buffer.
     * (b)   UCS_ERR_TRUNCATED - Sender's length exceed posted
                                 buffer, no data is copied.
     * (c)   UCS_ERR_CANCELED - Canceled by user.
     */
     void (*completed_cb)(uct_tag_context_t *self, uct_tag_t stag, uint64_t imm,
                          size_t length, void *inline_data, ucs_status_t status);

    /**
     * Tag was matched by a rendezvous request, which should be completed by
     * the protocol layer.
     *
     * @param [in]  self          Pointer to relevant context structure, which was
     *                            initially passed to @ref uct_iface_tag_recv_zcopy.
     * @param [in]  stag          Tag from sender.
     * @param [in]  header        User defined header.
     * @param [in]  header_length User defined header length in bytes.
     * @param [in]  status        Completion status.
     * @param [in]  flags         Flags defined by UCT_TAG_RECV_CB_xx.
     */
     void (*rndv_cb)(uct_tag_context_t *self, uct_tag_t stag, const void *header,
                     unsigned header_length, ucs_status_t status, unsigned flags);

     /** A placeholder for the private data used by the transport */
     char priv[UCT_TAG_PRIV_LEN];
};


/**
 * @ingroup UCT_RESOURCE
 * @brief flags of @ref uct_tag_context.
 */
enum {
    /* If set, header points to inline data, otherwise it is user buffer. */
    UCT_TAG_RECV_CB_INLINE_DATA = UCS_BIT(0)
};


extern const char *uct_alloc_method_names[];


/**
 * @ingroup UCT_RESOURCE
 * @brief Query for list of components.
 *
 * Obtain the list of transport components available on the current system.
 *
 * @param [out] components_p      Filled with a pointer to an array of component
 *                                handles.
 * @param [out] num_components_p  Filled with the number of elements in the array.
 *
 * @return UCS_OK if successful, or UCS_ERR_NO_MEMORY if failed to allocate the
 *         array of component handles.
 */
ucs_status_t uct_query_components(uct_component_h **components_p,
                                  unsigned *num_components_p);

/**
 * @ingroup UCT_RESOURCE
 * @brief Release the list of components returned from @ref uct_query_components.
 *
 * This routine releases the memory associated with the list of components
 * allocated by @ref uct_query_components.
 *
 * @param [in] components  Array of component handles to release.
 */
void uct_release_component_list(uct_component_h *components);


/**
 * @ingroup UCT_RESOURCE
 * @brief Get component attributes
 *
 * Query various attributes of a component.
 *
 * @param [in] component          Component handle to query attributes for. The
 *                                handle can be obtained from @ref uct_query_components.
 * @param [inout] component_attr  Filled with component attributes.
 *
 * @return UCS_OK if successful, or nonzero error code in case of failure.
 */
ucs_status_t uct_component_query(uct_component_h component,
                                 uct_component_attr_t *component_attr);


/**
 * @ingroup UCT_RESOURCE
 * @brief Open a memory domain.
 *
 * Open a specific memory domain. All communications and memory operations
 * are performed in the context of a specific memory domain. Therefore it
 * must be created before communication resources.
 *
 * @param [in]  component       Component on which to open the memory domain,
 *                              as returned from @ref uct_query_components.
 * @param [in]  md_name         Memory domain name, as returned from @ref
 *                              uct_component_query.
 * @param [in]  config          MD configuration options. Should be obtained
 *                              from uct_md_config_read() function, or point to
 *                              MD-specific structure which extends uct_md_config_t.
 * @param [out] md_p            Filled with a handle to the memory domain.
 *
 * @return Error code.
 */
ucs_status_t uct_md_open(uct_component_h component, const char *md_name,
                         const uct_md_config_t *config, uct_md_h *md_p);

/**
 * @ingroup UCT_RESOURCE
 * @brief Close a memory domain.
 *
 * @param [in]  md               Memory domain to close.
 */
void uct_md_close(uct_md_h md);


/**
 * @ingroup UCT_RESOURCE
 * @brief Query for transport resources.
 *
 * This routine queries the @ref uct_md_h "memory domain" for communication
 * resources that are available for it.
 *
 * @param [in]  md              Handle to memory domain.
 * @param [out] resources_p     Filled with a pointer to an array of resource
 *                              descriptors.
 * @param [out] num_resources_p Filled with the number of resources in the array.
 *
 * @return Error code.
 */
ucs_status_t uct_md_query_tl_resources(uct_md_h md,
                                       uct_tl_resource_desc_t **resources_p,
                                       unsigned *num_resources_p);


/**
 * @ingroup UCT_RESOURCE
 * @brief Release the list of resources returned from @ref uct_md_query_tl_resources.
 *
 * This routine releases the memory associated with the list of resources
 * allocated by @ref uct_md_query_tl_resources.
 *
 * @param [in] resources  Array of resource descriptors to release.
 */
void uct_release_tl_resource_list(uct_tl_resource_desc_t *resources);


/**
 * @ingroup UCT_CONTEXT
 * @brief Create a worker object.
 *
 *  The worker represents a progress engine. Multiple progress engines can be
 * created in an application, for example to be used by multiple threads.
 *  Transports can allocate separate communication resources for every worker,
 * so that every worker can be progressed independently of others.
 *
 * @param [in]  async         Context for async event handlers. Must not be NULL.
 * @param [in]  thread_mode   Thread access mode to the worker and all interfaces
 *                             and endpoints associated with it.
 * @param [out] worker_p      Filled with a pointer to the worker object.
 */
ucs_status_t uct_worker_create(ucs_async_context_t *async,
                               ucs_thread_mode_t thread_mode,
                               uct_worker_h *worker_p);


/**
 * @ingroup UCT_CONTEXT
 * @brief Destroy a worker object.
 *
 * @param [in]  worker        Worker object to destroy.
 */
void uct_worker_destroy(uct_worker_h worker);


/**
 * @ingroup UCT_CONTEXT
 * @brief Add a slow path callback function to a worker progress.
 *
 * If *id_p is equal to UCS_CALLBACKQ_ID_NULL, this function will add a callback
 * which will be invoked every time progress is made on the worker. *id_p will
 * be updated with an id which refers to this callback and can be used in
 * @ref uct_worker_progress_unregister_safe to remove it from the progress path.
 *
 * @param [in]    worker        Handle to the worker whose progress should invoke
 *                              the callback.
 * @param [in]    func          Pointer to the callback function.
 * @param [in]    arg           Argument for the callback function.
 * @param [in]    flags         Callback flags, see @ref ucs_callbackq_flags.
 * @param [inout] id_p          Points to a location to store a callback identifier.
 *                              If *id_p is equal to UCS_CALLBACKQ_ID_NULL, a
 *                              callback will be added and *id_p will be replaced
 *                              with a callback identifier which can be subsequently
 *                              used to remove the callback. Otherwise, no callback
 *                              will be added and *id_p will be left unchanged.
 *
 * @note This function is thread safe.
 */
void uct_worker_progress_register_safe(uct_worker_h worker, ucs_callback_t func,
                                       void *arg, unsigned flags,
                                       uct_worker_cb_id_t *id_p);


/**
 * @ingroup UCT_CONTEXT
 * @brief Remove a slow path callback function from worker's progress.
 *
 * If *id_p is not equal to UCS_CALLBACKQ_ID_NULL, remove a callback which was
 * previously added by @ref uct_worker_progress_register_safe. *id_p will be reset
 * to UCS_CALLBACKQ_ID_NULL.
 *
 * @param [in]    worker        Handle to the worker whose progress should invoke
 *                              the callback.
 * @param [inout] id_p          Points to a callback identifier which indicates
 *                              the callback to remove. If *id_p is not equal to
 *                              UCS_CALLBACKQ_ID_NULL, the callback will be removed
 *                              and *id_p will be reset to UCS_CALLBACKQ_ID_NULL.
 *                              If *id_p is equal to UCS_CALLBACKQ_ID_NULL, no
 *                              operation will be performed and *id_p will be
 *                              left unchanged.
 *
 * @note This function is thread safe.
 */
void uct_worker_progress_unregister_safe(uct_worker_h worker,
                                         uct_worker_cb_id_t *id_p);


/**
 * @ingroup UCT_RESOURCE
 * @brief Read transport-specific interface configuration.
 *
 * @param [in]  md            Memory domain on which the transport's interface
 *                            was registered.
 * @param [in]  tl_name       Transport name. If @e md supports
 *                            @ref UCT_MD_FLAG_SOCKADDR, the transport name
 *                            is allowed to be NULL. In this case, the configuration
 *                            returned from this routine should be passed to
 *                            @ref uct_iface_open with
 *                            @ref UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER or
 *                            @ref UCT_IFACE_OPEN_MODE_SOCKADDR_CLIENT set in
 *                            @ref uct_iface_params_t.open_mode.
 *                            In addition, if tl_name is not NULL, the configuration
 *                            returned from this routine should be passed to
 *                            @ref uct_iface_open with @ref UCT_IFACE_OPEN_MODE_DEVICE
 *                            set in @ref uct_iface_params_t.open_mode.
 * @param [in]  env_prefix    If non-NULL, search for environment variables
 *                            starting with this UCT_<prefix>_. Otherwise, search
 *                            for environment variables starting with just UCT_.
 * @param [in]  filename      If non-NULL, read configuration from this file. If
 *                            the file does not exist, it will be ignored.
 * @param [out] config_p      Filled with a pointer to configuration.
 *
 * @return Error code.
 */
ucs_status_t uct_md_iface_config_read(uct_md_h md, const char *tl_name,
                                      const char *env_prefix, const char *filename,
                                      uct_iface_config_t **config_p);


/**
 * @ingroup UCT_RESOURCE
 * @brief Release configuration memory returned from uct_md_iface_config_read(),
 * uct_md_config_read(), or from uct_cm_config_read().
 *
 * @param [in]  config        Configuration to release.
 */
void uct_config_release(void *config);


/**
 * @ingroup UCT_CONTEXT
 * @brief Get value by name from interface configuration (@ref uct_iface_config_t),
 *        memory domain configuration (@ref uct_md_config_t)
 *        or connection manager configuration (@ref uct_cm_config_t).
 *
 * @param [in]  config        Configuration to get from.
 * @param [in]  name          Configuration variable name.
 * @param [out] value         Pointer to get value. Should be allocated/freed by
 *                            caller.
 * @param [in]  max           Available memory space at @a value pointer.
 *
 * @return UCS_OK if found, otherwise UCS_ERR_INVALID_PARAM or UCS_ERR_NO_ELEM
 *         if error.
 */
ucs_status_t uct_config_get(void *config, const char *name, char *value,
                            size_t max);


/**
 * @ingroup UCT_CONTEXT
 * @brief Modify interface configuration (@ref uct_iface_config_t),
 *        memory domain configuration (@ref uct_md_config_t)
 *        or connection manager configuration (@ref uct_cm_config_t).
 *
 * @param [in]  config        Configuration to modify.
 * @param [in]  name          Configuration variable name.
 * @param [in]  value         Value to set.
 *
 * @return Error code.
 */
ucs_status_t uct_config_modify(void *config, const char *name, const char *value);


/**
 * @ingroup UCT_RESOURCE
 * @brief Open a communication interface.
 *
 * @param [in]  md            Memory domain to create the interface on.
 * @param [in]  worker        Handle to worker which will be used to progress
 *                            communications on this interface.
 * @param [in]  params        User defined @ref uct_iface_params_t parameters.
 * @param [in]  config        Interface configuration options. Should be obtained
 *                            from uct_md_iface_config_read() function, or point to
 *                            transport-specific structure which extends uct_iface_config_t.
 * @param [out] iface_p       Filled with a handle to opened communication interface.
 *
 * @return Error code.
 */
ucs_status_t uct_iface_open(uct_md_h md, uct_worker_h worker,
                            const uct_iface_params_t *params,
                            const uct_iface_config_t *config,
                            uct_iface_h *iface_p);


/**
 * @ingroup UCT_RESOURCE
 * @brief Close and destroy an interface.
 *
 * @param [in]  iface  Interface to close.
 */
void uct_iface_close(uct_iface_h iface);


/**
 * @ingroup UCT_RESOURCE
 * @brief Get interface attributes.
 *
 * @param [in]  iface      Interface to query.
 * @param [out] iface_attr Filled with interface attributes.
 */
ucs_status_t uct_iface_query(uct_iface_h iface, uct_iface_attr_t *iface_attr);


/**
 * @ingroup UCT_RESOURCE
 * @brief Get address of the device the interface is using.
 *
 *  Get underlying device address of the interface. All interfaces using the same
 * device would return the same address.
 *
 * @param [in]  iface       Interface to query.
 * @param [out] addr        Filled with device address. The size of the buffer
 *                          provided must be at least @ref uct_iface_attr_t::device_addr_len.
 */
ucs_status_t uct_iface_get_device_address(uct_iface_h iface, uct_device_addr_t *addr);


/**
 * @ingroup UCT_RESOURCE
 * @brief Get interface address.
 *
 * requires @ref UCT_IFACE_FLAG_CONNECT_TO_IFACE.
 *
 * @param [in]  iface       Interface to query.
 * @param [out] addr        Filled with interface address. The size of the buffer
 *                          provided must be at least @ref uct_iface_attr_t::iface_addr_len.
 */
ucs_status_t uct_iface_get_address(uct_iface_h iface, uct_iface_addr_t *addr);


/**
 * @ingroup UCT_RESOURCE
 * @brief Check if remote iface address is reachable.
 *
 * This function checks if a remote address can be reached from a local interface.
 * If the function returns true, it does not necessarily mean a connection and/or
 * data transfer would succeed, since the reachability check is a local operation
 * it does not detect issues such as network mis-configuration or lack of connectivity.
 *
 * @param [in]  iface      Interface to check reachability from.
 * @param [in]  dev_addr   Device address to check reachability to. It is NULL
 *                         if iface_attr.dev_addr_len == 0, and must be non-NULL otherwise.
 * @param [in]  iface_addr Interface address to check reachability to. It is
 *                         NULL if iface_attr.iface_addr_len == 0, and must
 *                         be non-NULL otherwise.
 *
 * @return Nonzero if reachable, 0 if not.
 */
int uct_iface_is_reachable(const uct_iface_h iface, const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr);


/**
 * @ingroup UCT_RESOURCE
 * @brief check if the destination endpoint is alive in respect to UCT library
 *
 * This function checks if the destination endpoint is alive with respect to the
 * UCT library. If the status of @a ep is known, either @ref UCS_OK or an error
 * is returned immediately. Otherwise, @ref UCS_INPROGRESS is returned,
 * indicating that synchronization on the status is needed. In this case, the
 * status will be be propagated by @a comp callback.
 *
 * @param [in]  ep      Endpoint to check
 * @param [in]  flags   Flags that define level of check
 *                      (currently unsupported - set to 0).
 * @param [in]  comp    Handler to process status of @a ep
 *
 * @return              Error code.
 */
ucs_status_t uct_ep_check(const uct_ep_h ep, unsigned flags,
                          uct_completion_t *comp);


/**
 * @ingroup UCT_RESOURCE
 * @brief Obtain a notification file descriptor for polling.
 *
 * Only interfaces that support at least one of the UCT_IFACE_FLAG_EVENT* flags
 * will implement this function.
 *
 * @param [in]  iface      Interface to get the notification descriptor.
 * @param [out] fd_p       Location to write the notification file descriptor.
 *
 * @return Error code.
 */
ucs_status_t uct_iface_event_fd_get(uct_iface_h iface, int *fd_p);


/**
 * @ingroup UCT_RESOURCE
 * @brief Turn on event notification for the next event.
 *
 * This routine needs to be called before waiting on each notification on this
 * interface, so will typically be called once the processing of the previous
 * event is over.
 *
 * @param [in] iface       Interface to arm.
 * @param [in] events      Events to wakeup on. See @ref uct_iface_event_types
 *
 * @return ::UCS_OK        The operation completed successfully. File descriptor
 *                         will be signaled by new events.
 * @return ::UCS_ERR_BUSY  There are unprocessed events which prevent the
 *                         file descriptor from being armed.
 *                         The operation is not completed. File descriptor
 *                         will not be signaled by new events.
 * @return @ref ucs_status_t "Other" different error codes in case of issues.
 */
ucs_status_t uct_iface_event_arm(uct_iface_h iface, unsigned events);


/**
 * @ingroup UCT_RESOURCE
 * @brief Allocate memory which can be used for zero-copy communications.
 *
 * Allocate a region of memory which can be used for zero-copy data transfer or
 * remote access on a particular transport interface.
 *
 * @param [in]  iface    Interface to allocate memory on.
 * @param [in]  length   Size of memory region to allocate.
 * @param [in]  flags    Memory allocation flags, see @ref uct_md_mem_flags.
 * @param [in]  name     Allocation name, for debug purposes.
 * @param [out] mem      Descriptor of allocated memory.
 *
 * @return UCS_OK if allocation was successful, error code otherwise.
 */
ucs_status_t uct_iface_mem_alloc(uct_iface_h iface, size_t length, unsigned flags,
                                 const char *name, uct_allocated_memory_t *mem);


/**
 * @ingroup UCT_RESOURCE
 * @brief Release memory allocated with @ref uct_iface_mem_alloc().
 *
 * @param [in]  mem      Descriptor of memory to release.
 */
void uct_iface_mem_free(const uct_allocated_memory_t *mem);


/**
 * @ingroup UCT_AM
 * @brief Set active message handler for the interface.
 *
 * Only one handler can be set of each active message ID, and setting a handler
 * replaces the previous value. If cb == NULL, the current handler is removed.
 *
 *
 * @param [in]  iface    Interface to set the active message handler for.
 * @param [in]  id       Active message id. Must be 0..UCT_AM_ID_MAX-1.
 * @param [in]  cb       Active message callback. NULL to clear.
 * @param [in]  arg      Active message argument.
 * @param [in]  flags    Required @ref uct_cb_flags "callback flags"
 *
 * @return error code if the interface does not support active messages or
 *         requested callback flags
 */
ucs_status_t uct_iface_set_am_handler(uct_iface_h iface, uint8_t id,
                                      uct_am_callback_t cb, void *arg, uint32_t flags);


/**
 * @ingroup UCT_AM
 * @brief Set active message tracer for the interface.
 *
 * Sets a function which dumps active message debug information to a buffer,
 * which is printed every time an active message is sent or received, when
 * data tracing is on. Without the tracer, only transport-level information is
 * printed.
 *
 * @param [in]  iface    Interface to set the active message tracer for.
 * @param [in]  tracer   Active message tracer. NULL to clear.
 * @param [in]  arg      Tracer custom argument.
 */
ucs_status_t uct_iface_set_am_tracer(uct_iface_h iface, uct_am_tracer_t tracer,
                                     void *arg);


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Accept connection request.
 *
 * @param [in] iface        Transport interface which generated connection
 *                          request @a conn_request.
 * @param [in] conn_request Connection establishment request passed as parameter
 *                          of @ref uct_sockaddr_conn_request_callback_t.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t uct_iface_accept(uct_iface_h iface,
                              uct_conn_request_h conn_request);


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Reject connection request. Will invoke an error handler @ref
 *        uct_error_handler_t on the remote transport interface, if set.
 *
 * @param [in] iface        Interface which generated connection establishment
 *                          request @a conn_request.
 * @param [in] conn_request Connection establishment request passed as parameter
 *                          of @ref uct_sockaddr_conn_request_callback_t.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t uct_iface_reject(uct_iface_h iface,
                              uct_conn_request_h conn_request);


/**
 * @ingroup UCT_RESOURCE
 * @brief Create new endpoint.
 *
 * Create a UCT endpoint in one of the available modes:
 * -# Unconnected endpoint: If no any address is present in @ref uct_ep_params,
 *    this creates an unconnected endpoint. To establish a connection to a
 *    remote endpoint, @ref uct_ep_connect_to_ep will need to be called. Use of
 *    this mode requires @ref uct_ep_params_t::iface has the
 *    @ref UCT_IFACE_FLAG_CONNECT_TO_EP capability flag. It may be obtained by
 *    @ref uct_iface_query .
 * -# Connect to a remote interface: If @ref uct_ep_params_t::dev_addr and
 *    @ref uct_ep_params_t::iface_addr are set, this will establish an endpoint
 *    that is connected to a remote interface. This requires that
 *    @ref uct_ep_params_t::iface has the @ref UCT_IFACE_FLAG_CONNECT_TO_IFACE
 *    capability flag. It may be obtained by @ref uct_iface_query.
 * -# Connect to a remote socket address: If @ref uct_ep_params_t::sockaddr is
 *    set, this will create an endpoint that is connected to a remote socket.
 *    This requires that either @ref uct_ep_params::cm, or
 *    @ref uct_ep_params::iface will be set. In the latter case, the interface
 *    has to support @ref UCT_IFACE_FLAG_CONNECT_TO_SOCKADDR flag, which can be
 *    checked by calling @ref uct_iface_query.
 * @param [in]  params  User defined @ref uct_ep_params_t configuration for the
 *                      @a ep_p.
 * @param [out] ep_p    Filled with handle to the new endpoint.
 *
 * @return UCS_OK       The endpoint is created successfully. This does not
 *                      guarantee that the endpoint has been connected to
 *                      the destination defined in @a params; in case of failure,
 *                      the error will be reported to the interface error
 *                      handler callback provided to @ref uct_iface_open
 *                      via @ref uct_iface_params_t.err_handler.
 * @return              Error code as defined by @ref ucs_status_t
 */
ucs_status_t uct_ep_create(const uct_ep_params_t *params, uct_ep_h *ep_p);


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Initiate a disconnection of an endpoint connected to a
 *        sockaddr by a connection manager @ref uct_cm_h.
 *
 * This non-blocking routine will send a disconnect notification on the endpoint,
 * so that @ref uct_ep_disconnect_cb_t will be called on the remote peer.
 * The remote side should also call this routine when handling the initiator's
 * disconnect.
 * After a call to this function, the given endpoint may not be used for
 * communications anymore.
 * The @ref uct_ep_flush / @ref uct_iface_flush routines will guarantee that the
 * disconnect notification is delivered to the remote peer.
 * @ref uct_ep_destroy should be called on this endpoint after invoking this
 * routine and @ref uct_ep_params::disconnect_cb was called.
 *
 * @param [in] ep       Endpoint to disconnect.
 * @param [in] flags    Reserved for future use.
 *
 * @return UCS_OK                Operation has completed successfully.
 *         UCS_ERR_BUSY          The @a ep is not connected yet (either
 *                               @ref uct_cm_ep_client_connect_callback_t or
 *                               @ref uct_cm_ep_server_conn_notify_callback_t
 *                               was not invoked).
 *         UCS_INPROGRESS        The disconnect request has been initiated, but
 *                               the remote peer has not yet responded to this
 *                               request, and consequently the registered
 *                               callback @ref uct_ep_disconnect_cb_t has not
 *                               been invoked to handle the request.
 *         UCS_ERR_NOT_CONNECTED The @a ep is disconnected locally and remotely.
 *         Other error codes as defined by @ref ucs_status_t .
 */
ucs_status_t uct_ep_disconnect(uct_ep_h ep, unsigned flags);


/**
 * @ingroup UCT_RESOURCE
 * @brief Destroy an endpoint.
 *
 * @param [in] ep       Endpoint to destroy.
 */
void uct_ep_destroy(uct_ep_h ep);


/**
 * @ingroup UCT_RESOURCE
 * @brief Get endpoint address.
 *
 * @param [in]  ep       Endpoint to query.
 * @param [out] addr     Filled with endpoint address. The size of the buffer
 *                       provided must be at least @ref uct_iface_attr_t::ep_addr_len.
 */
ucs_status_t uct_ep_get_address(uct_ep_h ep, uct_ep_addr_t *addr);


/**
 * @ingroup UCT_RESOURCE
 * @brief Connect endpoint to a remote endpoint.
 *
 * requires @ref UCT_IFACE_FLAG_CONNECT_TO_EP capability.
 *
 * @param [in] ep           Endpoint to connect.
 * @param [in] dev_addr     Remote device address.
 * @param [in] ep_addr      Remote endpoint address.
 */
ucs_status_t uct_ep_connect_to_ep(uct_ep_h ep, const uct_device_addr_t *dev_addr,
                                  const uct_ep_addr_t *ep_addr);


/**
 * @ingroup UCT_MD
 * @brief Query for memory domain attributes.
 *
 * @param [in]  md       Memory domain to query.
 * @param [out] md_attr  Filled with memory domain attributes.
 */
ucs_status_t uct_md_query(uct_md_h md, uct_md_attr_t *md_attr);


/**
 * @ingroup UCT_MD
 * @brief UCT allocation parameters specification field mask
 *
 * The enumeration allows specifying which fields in @ref uct_mem_alloc_params_t
 * are present.
 */
typedef enum {
    /** Enables @ref uct_mem_alloc_params_t::flags */
    UCT_MEM_ALLOC_PARAM_FIELD_FLAGS          = UCS_BIT(0),

    /** Enables @ref uct_mem_alloc_params_t::address */
    UCT_MEM_ALLOC_PARAM_FIELD_ADDRESS        = UCS_BIT(1),

    /** Enables @ref uct_mem_alloc_params_t::mem_type */
    UCT_MEM_ALLOC_PARAM_FIELD_MEM_TYPE       = UCS_BIT(2),

    /** Enables @ref uct_mem_alloc_params_t::mds */
    UCT_MEM_ALLOC_PARAM_FIELD_MDS            = UCS_BIT(3),

    /** Enables @ref uct_mem_alloc_params_t::name */
    UCT_MEM_ALLOC_PARAM_FIELD_NAME           = UCS_BIT(4)
} uct_mem_alloc_params_field_t;


/**
 * @ingroup UCT_MD
 * @brief Parameters for allocating memory using @ref uct_mem_alloc
 */
typedef struct {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_mem_alloc_params_field_t. Fields not specified in this mask will
     * be ignored.
     */
    uint64_t                     field_mask;

    /**
     * Memory allocation flags, see @ref uct_md_mem_flags
     * If UCT_MEM_ALLOC_PARAM_FIELD_FLAGS is not specified in field_mask, then
     * (UCT_MD_MEM_ACCESS_LOCAL_READ | UCT_MD_MEM_ACCESS_LOCAL_WRITE) is used by
     * default.
     */
    unsigned                     flags;

    /**
     * If @a address is NULL, the underlying allocation routine will
     * choose the address at which to create the mapping. If @a address
     * is non-NULL and UCT_MD_MEM_FLAG_FIXED is not set, the address
     * will be interpreted as a hint as to where to establish the mapping. If
     * @a address is non-NULL and UCT_MD_MEM_FLAG_FIXED is set, then the
     * specified address is interpreted as a requirement. In this case, if the
     * mapping to the exact address cannot be made, the allocation request
     * fails.
     */
    void                         *address;

    /**
     * Type of memory to be allocated.
     */
    ucs_memory_type_t            mem_type;

    struct {
        /**
         * Array of memory domains to attempt to allocate
         * the memory with, for MD allocation method.
         */
        const uct_md_h           *mds;

        /**
         *  Length of 'mds' array. May be empty, in such case
         *  'mds' may be NULL, and MD allocation method will
         *  be skipped.
         */
        unsigned                 count;
    } mds;

    /**
     * Name of the allocated region, used to track memory
     * usage for debugging and profiling.
     * If UCT_MEM_ALLOC_PARAM_FIELD_NAME is not specified in field_mask, then
     * "anonymous-uct_mem_alloc" is used by default.
     */
    const char                   *name;
} uct_mem_alloc_params_t;


/**
 * @ingroup UCT_MD
 * @brief Give advice about the use of memory
 *
 * This routine advises the UCT about how to handle memory range beginning at
 * address and size of length bytes. This call does not influence the semantics
 * of the application, but may influence its performance. The advice may be
 * ignored.
 *
 * @param [in]     md          Memory domain memory was allocated or registered on.
 * @param [in]     memh        Memory handle, as returned from @ref uct_mem_alloc
 * @param [in]     addr        Memory base address. Memory range must belong to the
 *                             @a memh
 * @param [in]     length      Length of memory to advise. Must be >0.
 * @param [in]     advice      Memory use advice as defined in the
 *                             @ref uct_mem_advice_t list
 */
ucs_status_t uct_md_mem_advise(uct_md_h md, uct_mem_h memh, void *addr,
                               size_t length, uct_mem_advice_t advice);


/**
 * @ingroup UCT_MD
 * @brief Register memory for zero-copy sends and remote access.
 *
 *  Register memory on the memory domain. In order to use this function, MD
 * must support @ref UCT_MD_FLAG_REG flag.
 *
 * @param [in]     md        Memory domain to register memory on.
 * @param [out]    address   Memory to register.
 * @param [in]     length    Size of memory to register. Must be >0.
 * @param [in]     flags     Memory allocation flags, see @ref uct_md_mem_flags.
 * @param [out]    memh_p    Filled with handle for allocated region.
 */
ucs_status_t uct_md_mem_reg(uct_md_h md, void *address, size_t length,
                            unsigned flags, uct_mem_h *memh_p);


/**
 * @ingroup UCT_MD
 * @brief Undo the operation of @ref uct_md_mem_reg().
 *
 * @param [in]  md          Memory domain which was used to register the memory.
 * @param [in]  memh        Local access key to memory region.
 */
ucs_status_t uct_md_mem_dereg(uct_md_h md, uct_mem_h memh);


/**
 * @ingroup UCT_MD
 * @brief Detect memory type
 *
 *
 * @param [in]     md           Memory domain to detect memory type
 * @param [in]     addr         Memory address to detect.
 * @param [in]     length       Size of memory
 * @param [out]    mem_type_p   Filled with memory type of the address range if
                                function succeeds
 * @return UCS_OK               If memory type is successfully detected
 *         UCS_ERR_INVALID_ADDR If failed to detect memory type
 */
ucs_status_t uct_md_detect_memory_type(uct_md_h md, const void *addr,
                                       size_t length,
                                       ucs_memory_type_t *mem_type_p);


/**
 * @ingroup UCT_MD
 * @brief Allocate memory for zero-copy communications and remote access.
 *
 * Allocate potentially registered memory.
 *
 * @param [in]     length      The minimal size to allocate. The actual size may
 *                             be larger, for example because of alignment
 *                             restrictions. Must be >0.
 * @param [in]     methods     Array of memory allocation methods to attempt.
 *                             Each of the provided allocation methods will be
 *                             tried in array order, to perform the allocation,
 *                             until one succeeds. Whenever the MD method is
 *                             encountered, each of the provided MDs will be
 *                             tried in array order, to allocate the memory,
 *                             until one succeeds, or they are exhausted. In
 *                             this case the next allocation method from the
 *                             initial list will be attempted.
 * @param [in]     num_methods Length of 'methods' array.
 * @param [in]     params      Memory allocation characteristics, see
 *                             @ref uct_mem_alloc_params_t.
 * @param [out]    mem         In case of success, filled with information about
 *                             the allocated memory. @ref uct_allocated_memory_t
 */
ucs_status_t uct_mem_alloc(size_t length, const uct_alloc_method_t *methods,
                           unsigned num_methods,
                           const uct_mem_alloc_params_t *params,
                           uct_allocated_memory_t *mem);


/**
 * @ingroup UCT_MD
 * @brief Release allocated memory.
 *
 * Release the memory allocated by @ref uct_mem_alloc.
 *
 * @param [in]  mem         Description of allocated memory, as returned from
 *                          @ref uct_mem_alloc.
 */
ucs_status_t uct_mem_free(const uct_allocated_memory_t *mem);

/**
 * @ingroup UCT_MD
 * @brief Read the configuration for a memory domain.
 *
 * @param [in]  component     Read the configuration of this component.
 * @param [in]  env_prefix    If non-NULL, search for environment variables
 *                            starting with this UCT_<prefix>_. Otherwise, search
 *                            for environment variables starting with just UCT_.
 * @param [in]  filename      If non-NULL, read configuration from this file. If
 *                            the file does not exist, it will be ignored.
 * @param [out] config_p      Filled with a pointer to the configuration.
 *
 * @return Error code.
 */
ucs_status_t uct_md_config_read(uct_component_h component,
                                const char *env_prefix, const char *filename,
                                uct_md_config_t **config_p);



/**
 * @ingroup UCT_MD
 * @brief Check if remote sock address is accessible from the memory domain.
 *
 * This function checks if a remote sock address can be accessed from a local
 * memory domain. Accessibility can be checked in local or remote mode.
 *
 * @param [in]  md         Memory domain to check accessibility from.
 *                         This memory domain must support the @ref
 *                         UCT_MD_FLAG_SOCKADDR flag.
 * @param [in]  sockaddr   Socket address to check accessibility to.
 * @param [in]  mode       Mode for checking accessibility, as defined in @ref
 *                         uct_sockaddr_accessibility_t.
 *                         Indicates if accessibility is tested on the server side -
 *                         for binding to the given sockaddr, or on the
 *                         client side - for connecting to the given remote
 *                         peer's sockaddr.
 *
 * @return Nonzero if accessible, 0 if inaccessible.
 */
int uct_md_is_sockaddr_accessible(uct_md_h md, const ucs_sock_addr_t *sockaddr,
                                  uct_sockaddr_accessibility_t mode);


/**
 * @ingroup UCT_MD
 *
 * @brief Pack a remote key.
 *
 * @param [in]  md           Handle to memory domain.
 * @param [in]  memh         Local key, whose remote key should be packed.
 * @param [out] rkey_buffer  Filled with packed remote key.
 *
 * @return Error code.
 */
ucs_status_t uct_md_mkey_pack(uct_md_h md, uct_mem_h memh, void *rkey_buffer);


/**
 * @ingroup UCT_MD
 *
 * @brief Unpack a remote key.
 *
 * @param [in]  component    Component on which to unpack the remote key.
 * @param [in]  rkey_buffer  Packed remote key buffer.
 * @param [out] rkey_ob      Filled with the unpacked remote key and its type.
 *
 * @note The remote key must be unpacked with the same component that was used
 *       to pack it. For example, if a remote device address on the remote
 *       memory domain which was used to pack the key is reachable by a
 *       transport on a local component, then that component is eligible to
 *       unpack the key.
 *       If the remote key buffer cannot be unpacked with the given component,
 *       UCS_ERR_INVALID_PARAM will be returned.
 *
 * @return Error code.
 */
ucs_status_t uct_rkey_unpack(uct_component_h component, const void *rkey_buffer,
                             uct_rkey_bundle_t *rkey_ob);


/**
 * @ingroup UCT_MD
 *
 * @brief Get a local pointer to remote memory.
 *
 * This routine returns a local pointer to the remote memory
 * described by the rkey bundle. The MD must support
 * @ref UCT_MD_FLAG_RKEY_PTR flag.
 *
 * @param [in]  component    Component on which to obtain the pointer to the
 *                           remote key.
 * @param [in]  rkey_ob      A remote key bundle as returned by
 *                           the @ref uct_rkey_unpack function.
 * @param [in]  remote_addr  A remote address within the memory area described
 *                           by the rkey_ob.
 * @param [out] addr_p       A pointer that can be used for direct access to
 *                           the remote memory.
 *
 * @note The component used to obtain a local pointer to the remote memory must
 *       be the same component that was used to pack the remote key. See notes
 *       section for @ref uct_rkey_unpack.
 *
 * @return Error code if the remote memory cannot be accessed directly or
 *         the remote address is not valid.
 */
ucs_status_t uct_rkey_ptr(uct_component_h component, uct_rkey_bundle_t *rkey_ob,
                          uint64_t remote_addr, void **addr_p);


/**
 * @ingroup UCT_MD
 *
 * @brief Release a remote key.
 *
 * @param [in]  component    Component which was used to unpack the remote key.
 * @param [in]  rkey_ob      Remote key to release.
 */
ucs_status_t uct_rkey_release(uct_component_h component,
                              const uct_rkey_bundle_t *rkey_ob);


/**
 * @ingroup UCT_CONTEXT
 * @brief Explicit progress for UCT worker.
 *
 * This routine explicitly progresses any outstanding communication operations
 * and active message requests.
 *
 * @note @li In the current implementation, users @b MUST call this routine
 * to receive the active message requests.
 *
 * @param [in]  worker        Handle to worker.
 *
 * @return Nonzero if any communication was progressed, zero otherwise.
 */
UCT_INLINE_API unsigned uct_worker_progress(uct_worker_h worker)
{
    return ucs_callbackq_dispatch(&worker->progress_q);
}


/**
 * @ingroup UCT_RESOURCE
 * @brief Flush outstanding communication operations on an interface.
 *
 * Flushes all outstanding communications issued on the interface prior to
 * this call. The operations are completed at the origin or at the target
 * as well. The exact completion semantic depends on @a flags parameter.
 *
 * @note Currently only one completion type is supported. It guarantees that
 * the data transfer is completed but the target buffer may not be updated yet.
 *
 * @param [in]    iface  Interface to flush communications from.
 * @param [in]    flags  Flags that control completion semantic (currently only
 *                       @ref UCT_FLUSH_FLAG_LOCAL is supported).
 * @param [inout] comp   Completion handle as defined by @ref uct_completion_t.
 *                       Can be NULL, which means that the call will return the
 *                       current state of the interface and no completion will
 *                       be generated in case of outstanding communications.
 *                       If it is not NULL completion counter is decremented
 *                       by 1 when the call completes. Completion callback is
 *                       called when the counter reaches 0.
 *
 *
 * @return UCS_OK         - No outstanding communications left.
 *         UCS_INPROGRESS - Some communication operations are still in progress.
 *                          If non-NULL 'comp' is provided, it will be updated
 *                          upon completion of these operations.
 */
UCT_INLINE_API ucs_status_t uct_iface_flush(uct_iface_h iface, unsigned flags,
                                            uct_completion_t *comp)
{
    return iface->ops.iface_flush(iface, flags, comp);
}

/**
 * @ingroup UCT_RESOURCE
 * @brief Ensures ordering of outstanding communications on the interface.
 * Operations issued on the interface prior to this call are guaranteed to
 * be completed before any subsequent communication operations to the same
 * interface which follow the call to fence.
 *
 * @param [in]    iface  Interface to issue communications from.
 * @param [in]    flags  Flags that control ordering semantic (currently
 *                       unsupported - set to 0).
 * @return UCS_OK         - Ordering is inserted.
 */

UCT_INLINE_API ucs_status_t uct_iface_fence(uct_iface_h iface, unsigned flags)
{
    return iface->ops.iface_fence(iface, flags);
}

/**
 * @ingroup UCT_AM
 * @brief Release AM descriptor
 *
 * Release active message descriptor @a desc, which was passed to
 * @ref uct_am_callback_t "the active message callback", and owned by the callee.
 *
 * @param [in]  desc  Descriptor to release.
 */
UCT_INLINE_API void uct_iface_release_desc(void *desc)
{
    uct_recv_desc_t *release_desc = uct_recv_desc(desc);
    release_desc->cb(release_desc, desc);
}


/**
 * @ingroup UCT_RMA
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_put_short(uct_ep_h ep, const void *buffer, unsigned length,
                                             uint64_t remote_addr, uct_rkey_t rkey)
{
    return ep->iface->ops.ep_put_short(ep, buffer, length, remote_addr, rkey);
}


/**
 * @ingroup UCT_RMA
 * @brief
 */
UCT_INLINE_API ssize_t uct_ep_put_bcopy(uct_ep_h ep, uct_pack_callback_t pack_cb,
                                        void *arg, uint64_t remote_addr,
                                        uct_rkey_t rkey)
{
    return ep->iface->ops.ep_put_bcopy(ep, pack_cb, arg, remote_addr, rkey);
}


/**
 * @ingroup UCT_RMA
 * @brief Write data to remote memory while avoiding local memory copy
 *
 * The input data in @a iov array of @ref ::uct_iov_t structures sent to remote
 * address ("gather output"). Buffers in @a iov are processed in array order.
 * This means that the function complete iov[0] before proceeding to
 * iov[1], and so on.
 *
 *
 * @param [in] ep          Destination endpoint handle.
 * @param [in] iov         Points to an array of @ref ::uct_iov_t structures.
 *                         The @a iov pointer must be a valid address of an array
 *                         of @ref ::uct_iov_t structures. A particular structure
 *                         pointer must be a valid address. A NULL terminated
 *                         array is not required.
 * @param [in] iovcnt      Size of the @a iov data @ref ::uct_iov_t structures
 *                         array. If @a iovcnt is zero, the data is considered empty.
 *                         @a iovcnt is limited by @ref uct_iface_attr_cap_put_max_iov
 *                         "uct_iface_attr::cap::put::max_iov".
 * @param [in] remote_addr Remote address to place the @a iov data.
 * @param [in] rkey        Remote key descriptor provided by @ref ::uct_rkey_unpack
 * @param [in] comp        Completion handle as defined by @ref ::uct_completion_t.
 *
 * @return UCS_INPROGRESS  Some communication operations are still in progress.
 *                         If non-NULL @a comp is provided, it will be updated
 *                         upon completion of these operations.
 *
 */
UCT_INLINE_API ucs_status_t uct_ep_put_zcopy(uct_ep_h ep,
                                             const uct_iov_t *iov, size_t iovcnt,
                                             uint64_t remote_addr, uct_rkey_t rkey,
                                             uct_completion_t *comp)
{
    return ep->iface->ops.ep_put_zcopy(ep, iov, iovcnt, remote_addr, rkey, comp);
}


/**
 * @ingroup UCT_RMA
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_get_short(uct_ep_h ep, void *buffer, unsigned length,
                                             uint64_t remote_addr, uct_rkey_t rkey)
{
    return ep->iface->ops.ep_get_short(ep, buffer, length, remote_addr, rkey);
}


/**
 * @ingroup UCT_RMA
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_get_bcopy(uct_ep_h ep, uct_unpack_callback_t unpack_cb,
                                             void *arg, size_t length,
                                             uint64_t remote_addr, uct_rkey_t rkey,
                                             uct_completion_t *comp)
{
    return ep->iface->ops.ep_get_bcopy(ep, unpack_cb, arg, length, remote_addr,
                                       rkey, comp);
}


/**
 * @ingroup UCT_RMA
 * @brief Read data from remote memory while avoiding local memory copy
 *
 * The output data in @a iov array of @ref ::uct_iov_t structures received from
 * remote address ("scatter input"). Buffers in @a iov are processed in array order.
 * This means that the function complete iov[0] before proceeding to
 * iov[1], and so on.
 *
 *
 * @param [in] ep          Destination endpoint handle.
 * @param [in] iov         Points to an array of @ref ::uct_iov_t structures.
 *                         The @a iov pointer must be a valid address of an array
 *                         of @ref ::uct_iov_t structures. A particular structure
 *                         pointer must be a valid address. A NULL terminated
 *                         array is not required.
 * @param [in] iovcnt      Size of the @a iov data @ref ::uct_iov_t structures
 *                         array. If @a iovcnt is zero, the data is considered empty.
 *                         @a iovcnt is limited by @ref uct_iface_attr_cap_get_max_iov
 *                         "uct_iface_attr::cap::get::max_iov".
 * @param [in] remote_addr Remote address of the data placed to the @a iov.
 * @param [in] rkey        Remote key descriptor provided by @ref ::uct_rkey_unpack
 * @param [in] comp        Completion handle as defined by @ref ::uct_completion_t.
 *
 * @return UCS_INPROGRESS  Some communication operations are still in progress.
 *                         If non-NULL @a comp is provided, it will be updated
 *                         upon completion of these operations.
 *
 */
UCT_INLINE_API ucs_status_t uct_ep_get_zcopy(uct_ep_h ep,
                                             const uct_iov_t *iov, size_t iovcnt,
                                             uint64_t remote_addr, uct_rkey_t rkey,
                                             uct_completion_t *comp)
{
    return ep->iface->ops.ep_get_zcopy(ep, iov, iovcnt, remote_addr, rkey, comp);
}


/**
 * @ingroup UCT_AM
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                                            const void *payload, unsigned length)
{
    return ep->iface->ops.ep_am_short(ep, id, header, payload, length);
}


/**
 * @ingroup UCT_AM
 * @brief Short io-vector send operation.
 *
 * This routine sends a message using @ref uct_short_protocol_desc "short" protocol.
 * The input data in @a iov array of @ref ::uct_iov_t structures is sent to remote
 * side to contiguous buffer keeping the order of the data in the array.
 *
 * @param [in] ep              Destination endpoint handle.
 * @param [in] id              Active message id. Must be in range 0..UCT_AM_ID_MAX-1.
 * @param [in] iov             Points to an array of @ref ::uct_iov_t structures.
 *                             The @a iov pointer must be a valid address of an array
 *                             of @ref ::uct_iov_t structures. A particular structure
 *                             pointer must be a valid address. A NULL terminated
 *                             array is not required. @a stride and @a count fields in
 *                             @ref ::uct_iov_t structure are ignored in current
 *                             implementation. The total size of the data buffers in
 *                             the array is limited by
 *                             @ref uct_iface_attr_cap_am_max_short
 *                             "uct_iface_attr::cap::am::max_short".
 * @param [in] iovcnt          Size of the @a iov data @ref ::uct_iov_t structures
 *                             array. If @a iovcnt is zero, the data is considered empty.
 *                             @a iovcnt is limited by @ref uct_iface_attr_cap_am_max_iov
 *                             "uct_iface_attr::cap::am::max_iov".
 *
 * @return UCS_OK              Operation completed successfully.
 * @return UCS_ERR_NO_RESOURCE Could not start the operation due to lack of
 *                             send resources.
 * @return otherwise           Error code.
 */
UCT_INLINE_API ucs_status_t uct_ep_am_short_iov(uct_ep_h ep, uint8_t id,
                                                const uct_iov_t *iov, size_t iovcnt)
{
    return ep->iface->ops.ep_am_short_iov(ep, id, iov, iovcnt);
}


/**
 * @ingroup UCT_AM
 * @brief
 */
UCT_INLINE_API ssize_t uct_ep_am_bcopy(uct_ep_h ep, uint8_t id,
                                       uct_pack_callback_t pack_cb, void *arg,
                                       unsigned flags)
{
    return ep->iface->ops.ep_am_bcopy(ep, id, pack_cb, arg, flags);
}


/**
 * @ingroup UCT_AM
 * @brief Send active message while avoiding local memory copy
 *
 * The input data in @a iov array of @ref ::uct_iov_t structures sent to remote
 * side ("gather output"). Buffers in @a iov are processed in array order.
 * This means that the function complete iov[0] before proceeding to
 * iov[1], and so on.
 *
 *
 * @param [in] ep              Destination endpoint handle.
 * @param [in] id              Active message id. Must be in range 0..UCT_AM_ID_MAX-1.
 * @param [in] header          Active message header.
 * @param [in] header_length   Active message header length in bytes.
 * @param [in] iov             Points to an array of @ref ::uct_iov_t structures.
 *                             The @a iov pointer must be a valid address of an array
 *                             of @ref ::uct_iov_t structures. A particular structure
 *                             pointer must be a valid address. A NULL terminated
 *                             array is not required.
 * @param [in] iovcnt          Size of the @a iov data @ref ::uct_iov_t structures
 *                             array. If @a iovcnt is zero, the data is considered empty.
 *                             @a iovcnt is limited by @ref uct_iface_attr_cap_am_max_iov
 *                             "uct_iface_attr::cap::am::max_iov".
 * @param [in] flags           Active message flags, see @ref uct_msg_flags.
 * @param [in] comp            Completion handle as defined by @ref ::uct_completion_t.
 *
 * @return UCS_OK              Operation completed successfully.
 * @return UCS_INPROGRESS      Some communication operations are still in progress.
 *                             If non-NULL @a comp is provided, it will be updated
 *                             upon completion of these operations.
 * @return UCS_ERR_NO_RESOURCE Could not start the operation due to lack of send
 *                             resources.
 *
 * @note If the operation returns @a UCS_INPROGRESS, the memory buffers
 *       pointed to by @a iov array must not be modified until the operation
 *       is completed by @a comp. @a header can be released or changed.
 */
UCT_INLINE_API ucs_status_t uct_ep_am_zcopy(uct_ep_h ep, uint8_t id,
                                            const void *header,
                                            unsigned header_length,
                                            const uct_iov_t *iov, size_t iovcnt,
                                            unsigned flags,
                                            uct_completion_t *comp)
{
    return ep->iface->ops.ep_am_zcopy(ep, id, header, header_length, iov, iovcnt,
                                      flags, comp);
}

/**
 * @ingroup UCT_AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_cswap64(uct_ep_h ep, uint64_t compare, uint64_t swap,
                                                  uint64_t remote_addr, uct_rkey_t rkey,
                                                  uint64_t *result, uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_cswap64(ep, compare, swap, remote_addr, rkey, result, comp);
}


/**
 * @ingroup UCT_AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_cswap32(uct_ep_h ep, uint32_t compare, uint32_t swap,
                                                  uint64_t remote_addr, uct_rkey_t rkey,
                                                  uint32_t *result, uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_cswap32(ep, compare, swap, remote_addr, rkey, result, comp);
}


/**
 * @ingroup UCT_AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic32_post(uct_ep_h ep, uct_atomic_op_t opcode,
                                                 uint32_t value, uint64_t remote_addr,
                                                 uct_rkey_t rkey)
{
    return ep->iface->ops.ep_atomic32_post(ep, opcode, value, remote_addr, rkey);
}


/**
 * @ingroup UCT_AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic64_post(uct_ep_h ep, uct_atomic_op_t opcode,
                                                 uint64_t value, uint64_t remote_addr,
                                                 uct_rkey_t rkey)
{
    return ep->iface->ops.ep_atomic64_post(ep, opcode, value, remote_addr, rkey);
}


/**
 * @ingroup UCT_AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic32_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                                  uint32_t value, uint32_t *result,
                                                  uint64_t remote_addr, uct_rkey_t rkey,
                                                  uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic32_fetch(ep, opcode, value, result,
                                            remote_addr, rkey, comp);
}


/**
 * @ingroup UCT_AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic64_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                                  uint64_t value, uint64_t *result,
                                                  uint64_t remote_addr, uct_rkey_t rkey,
                                                  uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic64_fetch(ep, opcode, value, result,
                                            remote_addr, rkey, comp);
}


/**
 * @ingroup UCT_RESOURCE
 * @brief Add a pending request to an endpoint.
 *
 * Add a pending request to the endpoint pending queue. The request will be
 * dispatched when the endpoint could potentially have additional send resources.
 *
 * @param [in]  ep    Endpoint to add the pending request to.
 * @param [in]  req   Pending request, which would be dispatched when more
 *                    resources become available. The user is expected to initialize
 *                    the "func" field.
 *                    After being passed to the function, the request is owned by UCT,
 *                    until the callback is called and returns UCS_OK.
 * @param [in]  flags Flags that control pending request processing (see @ref uct_cb_flags)
 *
 * @return UCS_OK       - request added to pending queue
 *         UCS_ERR_BUSY - request was not added to pending queue, because send
 *                        resources are available now. The user is advised to
 *                        retry.
 */
UCT_INLINE_API ucs_status_t uct_ep_pending_add(uct_ep_h ep,
                                               uct_pending_req_t *req,
                                               unsigned flags)
{
    return ep->iface->ops.ep_pending_add(ep, req, flags);
}


/**
 * @ingroup UCT_RESOURCE
 * @brief Remove all pending requests from an endpoint.
 *
 *  Remove pending requests from the given endpoint and pass them to the provided
 * callback function. The callback return value is ignored.
 *
 * @param [in]  ep  Endpoint to remove pending requests from.
 * @param [in]  cb  Callback to pass the removed requests to.
 * @param [in]  arg Argument to pass to the @a cb callback.
 */
UCT_INLINE_API void uct_ep_pending_purge(uct_ep_h ep,
                                         uct_pending_purge_callback_t cb,
                                         void *arg)
{
    ep->iface->ops.ep_pending_purge(ep, cb, arg);
}


/**
 * @ingroup UCT_RESOURCE
 * @brief Flush outstanding communication operations on an endpoint.
 *
 * Flushes all outstanding communications issued on the endpoint prior to
 * this call. The operations are completed at the origin or at the target
 * as well. The exact completion semantic depends on @a flags parameter.
 *
 * @param [in]    ep     Endpoint to flush communications from.
 * @param [in]    flags  Flags @ref uct_flush_flags that control completion
 *                       semantic.
 * @param [inout] comp   Completion handle as defined by @ref uct_completion_t.
 *                       Can be NULL, which means that the call will return the
 *                       current state of the endpoint and no completion will
 *                       be generated in case of outstanding communications.
 *                       If it is not NULL completion counter is decremented
 *                       by 1 when the call completes. Completion callback is
 *                       called when the counter reaches 0.
 *
 * @return UCS_OK              - No outstanding communications left.
 *         UCS_ERR_NO_RESOURCE - Flush operation could not be initiated. A subsequent
 *                               call to @ref uct_ep_pending_add would add a pending
 *                               operation, which provides an opportunity to retry
 *                               the flush.
 *         UCS_INPROGRESS      - Some communication operations are still in progress.
 *                               If non-NULL 'comp' is provided, it will be updated
 *                               upon completion of these operations.
 */
UCT_INLINE_API ucs_status_t uct_ep_flush(uct_ep_h ep, unsigned flags,
                                         uct_completion_t *comp)
{
    return ep->iface->ops.ep_flush(ep, flags, comp);
}


/**
 * @ingroup UCT_RESOURCE
 * @brief Ensures ordering of outstanding communications on the endpoint.
 * Operations issued on the endpoint prior to this call are guaranteed to
 * be completed before any subsequent communication operations to the same
 * endpoint which follow the call to fence.
 *
 * @param [in]    ep     Endpoint to issue communications from.
 * @param [in]    flags  Flags that control ordering semantic (currently
 *                       unsupported - set to 0).
 * @return UCS_OK         - Ordering is inserted.
 */
UCT_INLINE_API ucs_status_t uct_ep_fence(uct_ep_h ep, unsigned flags)
{
    return ep->iface->ops.ep_fence(ep, flags);
}


/**
 * @ingroup UCT_TAG
 * @brief Short eager tagged-send operation.
 *
 * This routine sends a message using @ref uct_short_protocol_desc "short"
 * eager protocol. Eager protocol means that the whole data is sent to the peer
 * immediately without any preceding notification.
 * The data is provided as buffer and its length,and must not be larger than the
 * corresponding @a max_short value in @ref uct_iface_attr.
 * The immediate value delivered to the receiver is implicitly equal to 0.
 * If it's required to pass nonzero imm value, @ref uct_ep_tag_eager_bcopy
 * should be used.
 *
 * @param [in]  ep        Destination endpoint handle.
 * @param [in]  tag       Tag to use for the eager message.
 * @param [in]  data      Data to send.
 * @param [in]  length    Data length.
 *
 * @return UCS_OK              - operation completed successfully.
 * @return UCS_ERR_NO_RESOURCE - could not start the operation due to lack of
 *                               send resources.
 */
UCT_INLINE_API ucs_status_t uct_ep_tag_eager_short(uct_ep_h ep, uct_tag_t tag,
                                                   const void *data, size_t length)
{
    return ep->iface->ops.ep_tag_eager_short(ep, tag, data, length);
}


/**
 * @ingroup UCT_TAG
 * @brief Bcopy eager tagged-send operation.
 *
 * This routine sends a message using @ref uct_bcopy_protocol_desc "bcopy"
 * eager protocol. Eager protocol means that the whole data is sent to the peer
 * immediately without any preceding notification.
 * Custom data callback is used to copy the data to the network buffers.
 *
 * @note The resulted data length must not be larger than the corresponding
 *       @a max_bcopy value in @ref uct_iface_attr.
 *
 * @param [in]  ep        Destination endpoint handle.
 * @param [in]  tag       Tag to use for the eager message.
 * @param [in]  imm       Immediate value which will be available to the
 *                        receiver.
 * @param [in]  pack_cb   User callback to pack the data.
 * @param [in]  arg       Custom argument to @a pack_cb.
 * @param [in]  flags     Tag message flags, see @ref uct_msg_flags.
 *
 * @return >=0       - The size of the data packed by @a pack_cb.
 * @return otherwise - Error code.
 */
UCT_INLINE_API ssize_t uct_ep_tag_eager_bcopy(uct_ep_h ep, uct_tag_t tag,
                                              uint64_t imm,
                                              uct_pack_callback_t pack_cb,
                                              void *arg, unsigned flags)
{
    return ep->iface->ops.ep_tag_eager_bcopy(ep, tag, imm, pack_cb, arg, flags);
}


/**
 * @ingroup UCT_TAG
 * @brief Zcopy eager tagged-send operation.
 *
 * This routine sends a message using @ref uct_zcopy_protocol_desc "zcopy"
 * eager protocol. Eager protocol means that the whole data is sent to the peer
 * immediately without any preceding notification.
 * The input data (which has to be previously registered) in @a iov array of
 * @ref uct_iov_t structures sent to remote side ("gather output"). Buffers in
 * @a iov are processed in array order, so the function complete @a iov[0]
 * before proceeding to @a iov[1], and so on.
 *
 * @note The resulted data length must not be larger than the corresponding
 *       @a max_zcopy value in @ref uct_iface_attr.
 *
 * @param [in]  ep        Destination endpoint handle.
 * @param [in]  tag       Tag to use for the eager message.
 * @param [in]  imm       Immediate value which will be available to the
 *                        receiver.
 * @param [in]  iov       Points to an array of @ref uct_iov_t structures.
 *                        A particular structure pointer must be a valid address.
 *                        A NULL terminated array is not required.
 * @param [in]  iovcnt    Size of the @a iov array. If @a iovcnt is zero, the
 *                        data is considered empty. Note that @a iovcnt is
 *                        limited by the corresponding @a max_iov value in
 *                        @ref uct_iface_attr.
 * @param [in]  flags     Tag message flags, see @ref uct_msg_flags.
 * @param [in]  comp      Completion callback which will be called when the data
 *                        is reliably received by the peer, and the buffer
 *                        can be reused or invalidated.
 *
 * @return UCS_OK              - operation completed successfully.
 * @return UCS_ERR_NO_RESOURCE - could not start the operation due to lack of
 *                               send resources.
 * @return UCS_INPROGRESS      - operation started, and @a comp will be used to
 *                               notify when it's completed.
 */
UCT_INLINE_API ucs_status_t uct_ep_tag_eager_zcopy(uct_ep_h ep, uct_tag_t tag,
                                                   uint64_t imm,
                                                   const uct_iov_t *iov,
                                                   size_t iovcnt,
                                                   unsigned flags,
                                                   uct_completion_t *comp)
{
    return ep->iface->ops.ep_tag_eager_zcopy(ep, tag, imm, iov, iovcnt, flags,
                                             comp);
}


/**
 * @ingroup UCT_TAG
 * @brief Rendezvous tagged-send operation.
 *
 * This routine sends a message using rendezvous protocol. Rendezvous protocol
 * means that only a small notification is sent at first, and the data itself
 * is transferred later (when there is a match) to avoid extra memory copy.
 *
 * @note The header will be available to the receiver in case of unexpected
 *       rendezvous operation only, i.e. the peer has not posted tag for this
 *       message yet (by means of @ref uct_iface_tag_recv_zcopy), when it is
 *       arrived.
 *
 * @param [in]  ep            Destination endpoint handle.
 * @param [in]  tag           Tag to use for the eager message.
 * @param [in]  header        User defined header.
 * @param [in]  header_length User defined header length in bytes. Note that
 *                            it is limited by the corresponding @a max_hdr
 *                            value in @ref uct_iface_attr.
 * @param [in]  iov           Points to an array of @ref uct_iov_t structures.
 *                            A particular structure pointer must be valid
 *                            address. A NULL terminated array is not required.
 * @param [in]  iovcnt        Size of the @a iov array. If @a iovcnt is zero,
 *                            the data is considered empty. Note that @a iovcnt
 *                            is limited by the corresponding @a max_iov value
 *                            in @ref uct_iface_attr.
 * @param [in]  flags         Tag message flags, see @ref uct_msg_flags.
 * @param [in]  comp          Completion callback which will be called when the
 *                            data is reliably received by the peer, and the
 *                            buffer can be reused or invalidated.
 *
 * @return >=0       - The operation is in progress and the return value is a
 *                     handle which can be used to cancel the outstanding
 *                     rendezvous operation.
 * @return otherwise - Error code.
 */
UCT_INLINE_API ucs_status_ptr_t uct_ep_tag_rndv_zcopy(uct_ep_h ep, uct_tag_t tag,
                                                      const void *header,
                                                      unsigned header_length,
                                                      const uct_iov_t *iov,
                                                      size_t iovcnt,
                                                      unsigned flags,
                                                      uct_completion_t *comp)
{
    return ep->iface->ops.ep_tag_rndv_zcopy(ep, tag, header, header_length,
                                            iov, iovcnt, flags, comp);
}


/**
 * @ingroup UCT_TAG
 * @brief Cancel outstanding rendezvous operation.
 *
 * This routine signals the underlying transport disregard the outstanding
 * operation without calling completion callback provided in
 * @ref uct_ep_tag_rndv_zcopy.
 *
 * @note The operation handle should be valid at the time the routine is
 * invoked. I.e. it should be a handle of the real operation which is not
 * completed yet.
 *
 * @param [in] ep Destination endpoint handle.
 * @param [in] op Rendezvous operation handle, as returned from
 *                @ref uct_ep_tag_rndv_zcopy.
 *
 * @return UCS_OK - The operation has been canceled.
 */
UCT_INLINE_API ucs_status_t uct_ep_tag_rndv_cancel(uct_ep_h ep, void *op)
{
    return ep->iface->ops.ep_tag_rndv_cancel(ep, op);
}


/**
 * @ingroup UCT_TAG
 * @brief Send software rendezvous request.
 *
 * This routine sends a rendezvous request only, which indicates that the data
 * transfer should be completed in software.
 *
 * @param [in]  ep            Destination endpoint handle.
 * @param [in]  tag           Tag to use for matching.
 * @param [in]  header        User defined header
 * @param [in]  header_length User defined header length in bytes. Note that it
 *                            is limited by the corresponding @a max_hdr value
 *                            in @ref uct_iface_attr.
 * @param [in]  flags         Tag message flags, see @ref uct_msg_flags.
 *
 * @return UCS_OK              - operation completed successfully.
 * @return UCS_ERR_NO_RESOURCE - could not start the operation due to lack of
 *                               send resources.
 */
UCT_INLINE_API ucs_status_t uct_ep_tag_rndv_request(uct_ep_h ep, uct_tag_t tag,
                                                    const void* header,
                                                    unsigned header_length,
                                                    unsigned flags)
{
    return ep->iface->ops.ep_tag_rndv_request(ep, tag, header, header_length,
                                              flags);
}


/**
 * @ingroup UCT_TAG
 * @brief Post a tag to a transport interface.
 *
 * This routine posts a tag to be matched on a transport interface. When a
 * message with the corresponding tag arrives it is stored in the user buffer
 * (described by @a iov and @a iovcnt) directly. The operation completion is
 * reported using callbacks on the @a ctx structure.
 *
 * @param [in]    iface     Interface to post the tag on.
 * @param [in]    tag       Tag to expect.
 * @param [in]    tag_mask  Mask which specifies what bits of the tag to
 *                          compare.
 * @param [in]    iov       Points to an array of @ref ::uct_iov_t structures.
 *                          The @a iov pointer must be a valid address of an array
 *                          of @ref ::uct_iov_t structures. A particular structure
 *                          pointer must be a valid address. A NULL terminated
 *                          array is not required.
 * @param [in]    iovcnt    Size of the @a iov data @ref ::uct_iov_t structures
 *                          array. If @a iovcnt is zero, the data is considered empty.
 *                          @a iovcnt is limited by @ref uct_iface_attr_cap_tag_recv_iov
 *                          "uct_iface_attr::cap::tag::max_iov".
 * @param [inout] ctx       Context associated with this particular tag, "priv" field
 *                          in this structure is used to track the state internally.
 *
 * @return UCS_OK                - The tag is posted to the transport.
 * @return UCS_ERR_NO_RESOURCE   - Could not start the operation due to lack of
 *                                 resources.
 * @return UCS_ERR_EXCEEDS_LIMIT - No more room for tags in the transport.
 */
UCT_INLINE_API ucs_status_t uct_iface_tag_recv_zcopy(uct_iface_h iface,
                                                     uct_tag_t tag,
                                                     uct_tag_t tag_mask,
                                                     const uct_iov_t *iov,
                                                     size_t iovcnt,
                                                     uct_tag_context_t *ctx)
{
    return iface->ops.iface_tag_recv_zcopy(iface, tag, tag_mask, iov, iovcnt, ctx);
}


/**
 * @ingroup UCT_TAG
 * @brief Cancel a posted tag.
 *
 * This routine cancels a tag, which was previously posted by
 * @ref uct_iface_tag_recv_zcopy. The tag would be either matched or canceled,
 * in a bounded time, regardless of the peer actions. The original completion
 * callback of the tag would be called with the status if @a force is not set.
 *
 * @param [in]  iface     Interface to cancel the tag on.
 * @param [in]  ctx       Tag context which was used for posting the tag. If
 *                        force is 0, @a ctx->completed_cb will be called with
 *                        either UCS_OK which means the tag was matched and data
 *                        received despite the cancel request, or
 *                        UCS_ERR_CANCELED which means the tag was successfully
 *                        canceled before it was matched.
 * @param [in]  force     Whether to report completions to @a ctx->completed_cb.
 *                        If nonzero, the cancel is assumed to be successful,
 *                        and the callback is not called.
 *
 * @return UCS_OK  -      The tag is canceled in the transport.
 */
UCT_INLINE_API ucs_status_t uct_iface_tag_recv_cancel(uct_iface_h iface,
                                                      uct_tag_context_t *ctx,
                                                      int force)
{
    return iface->ops.iface_tag_recv_cancel(iface, ctx, force);
}


/**
 * @ingroup UCT_RESOURCE
 * @brief Enable synchronous progress for the interface
 *
 * Notify the transport that it should actively progress communications during
 * @ref uct_worker_progress().
 *
 * When the interface is created, its progress is initially disabled.
 *
 * @param [in]  iface    The interface to enable progress.
 * @param [in]  flags    The type of progress to enable as defined by
 *                       @ref uct_progress_types
 *
 * @note This function is not thread safe with respect to
 *       @ref ucp_worker_progress(), unless the flag
 *       @ref UCT_PROGRESS_THREAD_SAFE is specified.
 *
 */
UCT_INLINE_API void uct_iface_progress_enable(uct_iface_h iface, unsigned flags)
{
    iface->ops.iface_progress_enable(iface, flags);
}


/**
 * @ingroup UCT_RESOURCE
 * @brief Disable synchronous progress for the interface
 *
 * Notify the transport that it should not progress its communications during
 * @ref uct_worker_progress(). Thus the latency of other transports may be
 * improved.
 *
 * By default, progress is disabled when the interface is created.
 *
 * @param [in]  iface    The interface to disable progress.
 * @param [in]  flags    The type of progress to disable as defined by
 *                       @ref uct_progress_types.
 *
 * @note This function is not thread safe with respect to
 *       @ref ucp_worker_progress(), unless the flag
 *       @ref UCT_PROGRESS_THREAD_SAFE is specified.
 *
 */
UCT_INLINE_API void uct_iface_progress_disable(uct_iface_h iface, unsigned flags)
{
    iface->ops.iface_progress_disable(iface, flags);
}


/**
 * @ingroup UCT_RESOURCE
 * @brief Perform a progress on an interface.
 */
UCT_INLINE_API unsigned uct_iface_progress(uct_iface_h iface)
{
    return iface->ops.iface_progress(iface);
}


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Open a connection manager.
 *
 * Open a connection manager. All client server connection
 * establishment operations are performed in the context of a specific
 * connection manager.
 * @note This is an alternative API for
 *       @ref uct_iface_open_mode::UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER and
 *       @ref uct_iface_open_mode::UCT_IFACE_OPEN_MODE_SOCKADDR_CLIENT .
 *
 * @param [in]  component   Component on which to open the connection manager,
 *                          as returned from @ref uct_query_components.
 * @param [in]  worker      Worker on which to open the connection manager.
 * @param [in]  config      CM configuration options. Either obtained
 *                          from @ref uct_cm_config_read() function, or pointer
 *                          to CM-specific structure that extends
 *                          @ref uct_cm_config_t.
 * @param [out] cm_p        Filled with a handle to the connection manager.
 *
 * @return Error code.
 */
ucs_status_t uct_cm_open(uct_component_h component, uct_worker_h worker,
                         const uct_cm_config_t *config, uct_cm_h *cm_p);


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Close a connection manager.
 *
 * @param [in]  cm    Connection manager to close.
 */
void uct_cm_close(uct_cm_h cm);


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Get connection manager attributes.
 *
 * This routine queries the @ref uct_cm_h "cm" for its attributes
 * @ref uct_cm_attr_t.
 *
 * @param [in]  cm      Connection manager to query.
 * @param [out] cm_attr Filled with connection manager attributes.
 */
ucs_status_t uct_cm_query(uct_cm_h cm, uct_cm_attr_t *cm_attr);


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Read the configuration for a connection manager.
 *
 * @param [in]  component     Read the configuration of the connection manager
 *                            on this component.
 * @param [in]  env_prefix    If non-NULL, search for environment variables
 *                            starting with this UCT_<prefix>_. Otherwise, search
 *                            for environment variables starting with just UCT_.
 * @param [in]  filename      If non-NULL, read configuration from this file. If
 *                            the file does not exist, or exists but cannot be
 *                            opened or read, it will be ignored.
 * @param [out] config_p      Filled with a pointer to the configuration.
 *
 * @return Error code.
 */
ucs_status_t uct_cm_config_read(uct_component_h component,
                                const char *env_prefix, const char *filename,
                                uct_cm_config_t **config_p);


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Notify the server about client-side connection establishment.
 *
 * This routine should be called on the client side after the client completed
 * establishing its connection to the server. The routine will send a
 * notification message to the server indicating that the client is connected.
 *
 * @param [in]  ep      The connected endpoint on the client side.
 *
 * @return Error code.
 */
ucs_status_t uct_cm_client_ep_conn_notify(uct_ep_h ep);


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Create a new transport listener object.
 *
 * This routine creates a new listener on the given CM which will start
 * listening on a given sockaddr.
 *
 * @param [in]  cm          Connection manager on which to open the listener.
 *                          This cm should not be closed as long as there are
 *                          open listeners on it.
 * @param [in]  saddr       The socket address to listen on.
 * @param [in]  socklen     The saddr length.
 * @param [in]  params      User defined @ref uct_listener_params_t
 *                          configurations for the @a listener_p.
 * @param [out] listener_p  Filled with handle to the new listener.
 *
 * @return Error code.
 */
ucs_status_t uct_listener_create(uct_cm_h cm, const struct sockaddr *saddr,
                                 socklen_t socklen,
                                 const uct_listener_params_t *params,
                                 uct_listener_h *listener_p);


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Destroy a transport listener.
 *
 * @param [in]  listener    Listener to destroy.
 */
void uct_listener_destroy(uct_listener_h listener);


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Reject a connection request.
 *
 * This routine can be invoked on the server side. It rejects a connection request
 * from the client.
 *
 * @param [in] listener     Listener which will reject the connection request.
 * @param [in] conn_request Connection establishment request passed as parameter
 *                          of @ref uct_cm_listener_conn_request_callback_t in
 *                          @ref uct_cm_listener_conn_request_args_t::conn_request.
 *
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t uct_listener_reject(uct_listener_h listener,
                                 uct_conn_request_h conn_request);


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Get attributes specific to a particular listener.
 *
 * This routine queries the @ref uct_listener_h "listener" for its attributes
 * @ref uct_listener_attr_t.
 *
 * @param [in]  listener      Listener object to query.
 * @param [out] listener_attr Filled with attributes of the listener.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t uct_listener_query(uct_listener_h listener,
                                uct_listener_attr_t *listener_attr);


/**
 * @ingroup UCT_RESOURCE
 * @brief Update status of UCT completion handle.
 *
 * @param comp   [in]         Completion handle to update.
 * @param status [in]         Status to update @a comp handle.
 */
static UCS_F_ALWAYS_INLINE
void uct_completion_update_status(uct_completion_t *comp, ucs_status_t status)
{
    if (ucs_unlikely(status != UCS_OK) && (comp->status == UCS_OK)) {
        /* store first failure status */
        comp->status = status;
    }
}


/**
 * @example uct_hello_world.c
 * UCT hello world client / server example utility.
 */

END_C_DECLS

#endif
