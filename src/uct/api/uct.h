/**
 * @file        uct.h
 * @date        2014-2015
 * @copyright   Mellanox Technologies Ltd. All rights reserved.
 * @copyright   Oak Ridge National Laboratory. All rights received.
 * @brief       Unified Communication Transport
 */

#ifndef UCT_H_
#define UCT_H_

#include <uct/api/uct_def.h>
#include <uct/api/tl.h>
#include <uct/api/version.h>
#include <ucs/async/async_fwd.h>
#include <ucs/config/types.h>
#include <ucs/datastruct/callbackq.h>
#include <ucs/type/status.h>
#include <ucs/type/thread_mode.h>
#include <ucs/type/cpu_set.h>
#include <ucs/stats/stats_fwd.h>
#include <ucs/sys/compiler_def.h>

#include <sys/socket.h>
#include <stdio.h>
#include <sched.h>

BEGIN_C_DECLS

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
    uct_device_type_t        dev_type;     /**< Device type. To which UCT group it belongs to */
} uct_tl_resource_desc_t;

#define UCT_TL_RESOURCE_DESC_FMT              "%s/%s"
#define UCT_TL_RESOURCE_DESC_ARG(_resource)   (_resource)->tl_name, (_resource)->dev_name


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

        /* Atomic operations capabilities */
#define UCT_IFACE_FLAG_ATOMIC_ADD32   UCS_BIT(16) /**< 32bit atomic add */
#define UCT_IFACE_FLAG_ATOMIC_ADD64   UCS_BIT(17) /**< 64bit atomic add */
#define UCT_IFACE_FLAG_ATOMIC_FADD32  UCS_BIT(18) /**< 32bit atomic fetch-and-add */
#define UCT_IFACE_FLAG_ATOMIC_FADD64  UCS_BIT(19) /**< 64bit atomic fetch-and-add */
#define UCT_IFACE_FLAG_ATOMIC_SWAP32  UCS_BIT(20) /**< 32bit atomic swap */
#define UCT_IFACE_FLAG_ATOMIC_SWAP64  UCS_BIT(21) /**< 64bit atomic swap */
#define UCT_IFACE_FLAG_ATOMIC_CSWAP32 UCS_BIT(22) /**< 32bit atomic compare-and-swap */
#define UCT_IFACE_FLAG_ATOMIC_CSWAP64 UCS_BIT(23) /**< 64bit atomic compare-and-swap */

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

        /* Event notification */
#define UCT_IFACE_FLAG_EVENT_SEND_COMP UCS_BIT(46) /**< Event notification of send completion is
                                                        supported */
#define UCT_IFACE_FLAG_EVENT_RECV      UCS_BIT(47) /**< Event notification of tag and active message
                                                        receive is supported */
#define UCT_IFACE_FLAG_EVENT_RECV_SIG  UCS_BIT(48) /**< Event notification of signaled tag and active
                                                        message is supported */

        /* Tag matching operations */
#define UCT_IFACE_FLAG_TAG_EAGER_SHORT UCS_BIT(50) /**< Hardware tag matching short eager support */
#define UCT_IFACE_FLAG_TAG_EAGER_BCOPY UCS_BIT(51) /**< Hardware tag matching bcopy eager support */
#define UCT_IFACE_FLAG_TAG_EAGER_ZCOPY UCS_BIT(52) /**< Hardware tag matching zcopy eager support */
#define UCT_IFACE_FLAG_TAG_RNDV_ZCOPY  UCS_BIT(53) /**< Hardware tag matching rendezvous zcopy support */
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
                                                 the relevant callbacks. */
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
 * A callback must have either the SYNC or ASYNC flag set.
 */
enum uct_cb_flags {
    UCT_CB_FLAG_SYNC  = UCS_BIT(1), /**< Callback is always invoked from the context (thread, process)
                                         that called uct_iface_progress(). An interface must
                                         have the @ref UCT_IFACE_FLAG_CB_SYNC flag set to support sync
                                         callback invocation. */

    UCT_CB_FLAG_ASYNC = UCS_BIT(2)  /**< Callback may be invoked from any context. For example,
                                         it may be called from a transport async progress thread. To guarantee
                                         async invocation, the interface must have the @ref UCT_IFACE_FLAG_CB_ASYNC
                                         flag set.
                                         If async callback is requested on an interface
                                         which only supports sync callback
                                         (i.e., only the @ref UCT_IFACE_FLAG_CB_SYNC flag is set),
                                         it will behave exactly like a sync callback.  */
};


/**
 * @ingroup UCT_RESOURCE
 * @brief Mode in which to open the interface.
 */
enum uct_iface_open_mode {
   UCT_IFACE_OPEN_MODE_DEVICE          = UCS_BIT(0),  /**< Interface is opened on a specific device */
   UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER = UCS_BIT(1),  /**< Interface is opened on a specific address
                                                           on the server side */
   UCT_IFACE_OPEN_MODE_SOCKADDR_CLIENT = UCS_BIT(2)   /**< Interface is opened on a specific address
                                                           on the client side */
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

/*
 * @ingroup UCT_MD
 * @brief  Memory types
 */
typedef enum {
    UCT_MD_MEM_TYPE_HOST = 0,      /**< Default system memory */
    UCT_MD_MEM_TYPE_CUDA,          /**< NVIDIA CUDA memory */
    UCT_MD_MEM_TYPE_LAST
} uct_memory_type_t;


/**
 * @ingroup UCT_MD
 * @brief  Memory allocation/registration flags.
 */
enum uct_md_mem_flags {
    UCT_MD_MEM_FLAG_NONBLOCK = UCS_BIT(0), /**< Hint to perform non-blocking
                                                allocation/registration: page
                                                mapping may be deferred until
                                                it is accessed by the CPU or a
                                                transport. */
    UCT_MD_MEM_FLAG_FIXED    = UCS_BIT(1), /**< Place the mapping at exactly
                                                defined address */
    UCT_MD_MEM_FLAG_LOCK     = UCS_BIT(2), /**< Registered memory should be
                                                locked. May incur extra cost for
                                                registration, but memory access
                                                is usually faster. */

    /* memory access flags */
    UCT_MD_MEM_ACCESS_REMOTE_PUT    = UCS_BIT(5), /**< enable remote put access */
    UCT_MD_MEM_ACCESS_REMOTE_GET    = UCS_BIT(6), /**< enable remote get access */
    UCT_MD_MEM_ACCESS_REMOTE_ATOMIC = UCS_BIT(7), /**< enable remote atomic access */

    /** enable local and remote access for all operations */
    UCT_MD_MEM_ACCESS_ALL =  (UCT_MD_MEM_ACCESS_REMOTE_PUT|
                              UCT_MD_MEM_ACCESS_REMOTE_GET|
                              UCT_MD_MEM_ACCESS_REMOTE_ATOMIC),

    /** enable local and remote access for put and get operations */
    UCT_MD_MEM_ACCESS_RMA = (UCT_MD_MEM_ACCESS_REMOTE_PUT|
                             UCT_MD_MEM_ACCESS_REMOTE_GET)
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


/*
 * @ingroup UCT_RESOURCE
 * @brief Linear growth specification: f(x) = overhead + growth * x
 *
 *  This structure specifies a linear function which is used as basis for time
 * estimation of various UCT operations. This information can be used to select
 * the best performing combination of UCT operations.
 */
typedef struct uct_linear_growth {
    double                   overhead;  /**< Constant overhead factor */
    double                   growth;    /**< Growth rate factor */
} uct_linear_growth_t;


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
            size_t           max_short;  /**< Total max. size (incl. the header) */
            size_t           max_bcopy;  /**< Total max. size (incl. the header) */
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

        uint64_t             flags;      /**< Flags from @ref UCT_RESOURCE_IFACE_CAP */
    } cap;                               /**< Interface capabilities */

    size_t                   device_addr_len;/**< Size of device address */
    size_t                   iface_addr_len; /**< Size of interface address */
    size_t                   ep_addr_len;    /**< Size of endpoint address */
    size_t                   max_conn_priv;  /**< Max size of the iface's private data.
                                                  used for connection
                                                  establishment with sockaddr */
    /*
     * The following fields define expected performance of the communication
     * interface, this would usually be a combination of device and system
     * characteristics and determined at run time.
     */
    double                   overhead;     /**< Message overhead, seconds */
    double                   bandwidth;    /**< Maximal bandwidth, bytes/second */
    uct_linear_growth_t      latency;      /**< Latency model */
    uint8_t                  priority;     /**< Priority of device */
};


/**
 * @ingroup UCT_RESOURCE
 * @brief Parameters used for interface creation.
 *
 * This structure should be allocated by the user and should be passed to
 * @ref uct_iface_open. User has to initialize all fields of this structure.
 */
struct uct_iface_params {
    /** Mask of CPUs to use for resources */
    ucs_cpu_set_t                                cpu_mask;
    /** Interface open mode bitmap. @ref uct_iface_open_mode */
    uint64_t                                     open_mode;
    /** Mode-specific parameters */
    union {
        /** The fields in this structure (tl_name and dev_name) need to be set only when
         *  the @ref UCT_IFACE_OPEN_MODE_DEVICE bit is set in @ref
         *  uct_iface_params_t.open_mode This will make @ref uct_iface_open
         *  open the interface on the specified device.
         */
        struct {
            const char                           *tl_name;  /**< Transport name */
            const char                           *dev_name; /**< Device Name */
        } device;
        /** These callbacks and address are only relevant for client-server
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

    /** These callbacks are only relevant for HW Tag Matching */
    void                                         *eager_arg;
    /** Callback for tag matching unexpected eager messages */
    uct_tag_unexp_eager_cb_t                     eager_cb;
    void                                         *rndv_arg;
    /** Callback for tag matching unexpected rndv messages */
    uct_tag_unexp_rndv_cb_t                      rndv_cb;
};


/**
 * @ingroup UCT_MD
 * @brief  Memory domain attributes.
 *
 * This structure defines the attributes of a Memory Domain which includes
 * maximum memory that can be allocated, credentials required for accessing the memory,
 * and CPU mask indicating the proximity of CPUs.
 */
struct uct_md_attr {
    struct {
        size_t               max_alloc; /**< Maximal allocation size */
        size_t               max_reg;   /**< Maximal registration size */
        uint64_t             flags;     /**< UCT_MD_FLAG_xx */
        uint64_t             reg_mem_types; /** UCS_BIT(uct_memory_type_t) */
        uct_memory_type_t    mem_type;  /**< Supported(owned) memory type */
    } cap;

    uct_linear_growth_t      reg_cost;  /**< Memory registration cost estimation
                                             (time,seconds) as a linear function
                                             of the buffer size. */

    char                     component_name[UCT_MD_COMPONENT_NAME_MAX]; /**< MD component name */
    size_t                   rkey_packed_size; /**< Size of buffer needed for packed rkey */
    cpu_set_t                local_cpus;    /**< Mask of CPUs near the resource */
};


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
    uct_memory_type_t        mem_type; /**< type of allocated memory */
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
 * primitives. User has to initializes both fields of the structure.
 *  If the operation returns UCS_INPROGRESS, this structure will be in use by the
 * transport until the operation completes. When the operation completes, "count"
 * field is decremented by 1, and whenever it reaches 0 - the callback is called.
 *
 * Notes:
 *  - The same structure can be passed multiple times to communication functions
 *    without the need to wait for completion.
 *  - If the number of operations is smaller than the initial value of the counter,
 *    the callback will not be called at all, so it may be left undefined.
 */
struct uct_completion {
    uct_completion_callback_t func;    /**< User callback function */
    int                       count;   /**< Completion counter */
};


/**
 * @ingroup UCT_RESOURCE
 * @brief Pending request.
 *
 * This structure should be passed to uct_pending_add() and is used to signal
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
     * @param [in]  self    Pointer to relevant context structure, which was
     *                      initially passed to @ref uct_iface_tag_recv_zcopy.
     * @param [in]  stag    Tag from sender.
     * @param [in]  imm     Immediate data from sender. For rendezvous, it's always 0.
     * @param [in]  length  Completed length.
     * @param [in]  status  Completion status:
     * (a)   UCS_OK - Success, data placed in provided buffer.
     * (b)   UCS_ERR_TRUNCATED - Sender's length exceed posted
                                 buffer, no data is copied.
     * (c)   UCS_ERR_CANCELED - Canceled by user.
     */
     void (*completed_cb)(uct_tag_context_t *self, uct_tag_t stag, uint64_t imm,
                          size_t length, ucs_status_t status);

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
     */
     void (*rndv_cb)(uct_tag_context_t *self, uct_tag_t stag, const void *header,
                     unsigned header_length, ucs_status_t status);

     /** A placeholder for the private data used by the transport */
     char priv[UCT_TAG_PRIV_LEN];
};


extern const char *uct_alloc_method_names[];


/**
 * @ingroup UCT_RESOURCE
 * @brief Query for memory resources.
 *
 * Obtain the list of memory domain resources available on the current system.
 *
 * @param [out] resources_p     Filled with a pointer to an array of resource
 *                              descriptors.
 * @param [out] num_resources_p Filled with the number of resources in the array.
 *
 * @return Error code.
 */
ucs_status_t uct_query_md_resources(uct_md_resource_desc_t **resources_p,
                                    unsigned *num_resources_p);

/**
 * @ingroup UCT_RESOURCE
 * @brief Release the list of resources returned from @ref uct_query_md_resources.
 *
 * This routine releases the memory associated with the list of resources
 * allocated by @ref uct_query_md_resources.
 *
 * @param [in] resources  Array of resource descriptors to release.
 */
void uct_release_md_resource_list(uct_md_resource_desc_t *resources);


/**
 * @ingroup UCT_RESOURCE
 * @brief Open a memory domain.
 *
 * Open a specific memory domain. All communications and memory operations
 * are performed in the context of a specific memory domain. Therefore it
 * must be created before communication resources.
 *
 * @param [in]  md_name         Memory domain name, as returned from @ref
 *                              uct_query_md_resources.
 * @param [in]  config          MD configuration options. Should be obtained
 *                              from uct_md_config_read() function, or point to
 *                              MD-specific structure which extends uct_md_config_t.
 * @param [out] md_p            Filled with a handle to the memory domain.
 *
 * @return Error code.
 */
ucs_status_t uct_md_open(const char *md_name, const uct_md_config_t *config,
                         uct_md_h *md_p);

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
 * @param [in]  async         Context for async event handlers.
  *                            Can be NULL, which means that event handlers will
 *                             not have particular context.
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
 * @brief Release configuration memory returned from uct_md_iface_config_read() or
 * from uct_md_config_read().
 *
 * @param [in]  config        Configuration to release.
 */
void uct_config_release(void *config);


/**
 * @ingroup UCT_RESOURCE
 * @brief Print interface/MD configuration to a stream.
 *
 * @param [in]  config        Configuration to print.
 * @param [in]  stream        Output stream to print to.
 * @param [in]  title         Title to the output.
 * @param [in]  print_flags   Controls how the configuration is printed.
 */
void uct_config_print(const void *config, FILE *stream, const char *title,
                      ucs_config_print_flags_t print_flags);


/**
 * @ingroup UCT_CONTEXT
 * @brief Get value by name from interface/MD configuration.
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
 * @brief Modify interface/MD configuration.
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
 * Only interfaces supporting the @ref UCT_IFACE_FLAG_EVENT_FD implement this
 * function.
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
 * @ingroup UCT_RESOURCE
 * @brief Create new endpoint.
 *
 * @param [in]  iface   Interface to create the endpoint on.
 * @param [out] ep_p    Filled with handle to the new endpoint.
 */
ucs_status_t uct_ep_create(uct_iface_h iface, uct_ep_h *ep_p);


/**
 * @ingroup UCT_RESOURCE
 * @brief Create an endpoint which is connected to remote interface.
 *
 * requires @ref UCT_IFACE_FLAG_CONNECT_TO_IFACE capability.
 *
 * @param [in]  iface       Interface to create the endpoint on.
 * @param [in]  dev_addr    Remote device address to connect to.
 * @param [in]  iface_addr  Remote interface address to connect to.
 * @param [out] ep_p        Filled with handle to the new endpoint.
 */
ucs_status_t uct_ep_create_connected(uct_iface_h iface,
                                     const uct_device_addr_t *dev_addr,
                                     const uct_iface_addr_t *iface_addr,
                                     uct_ep_h *ep_p);


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
 * @ingroup UCT_RESOURCE
 * @brief Initiate a client-server connection to a remote peer.
 *
 * This routine will create an endpoint for a connection to the remote peer,
 * specified by its socket address.
 * The user may provide private data to be sent on a connection request to the
 * remote peer.
 *
 * @note The interface in this routine requires the
 * @ref UCT_IFACE_FLAG_CONNECT_TO_SOCKADDR capability.
 *
 * @param [in]  iface            Interface to create the endpoint on.
 * @param [in]  sockaddr         The sockaddr to connect to on the remote peer.
 * @param [in]  priv_data        User's private data for connecting to the
 *                               remote peer.
 * @param [in]  length           Length of the private data.
 * @param [out] ep_p             Handle to the created endpoint.
 *
 * @return UCS_OK              - Connection request was sent to the server.
 *                               This does not guarantee that the server has
 *                               received the message; in case of failure, the
 *                               error will be reported to the interface error
 *                               handler callback provided to @ref uct_iface_open
 *                               via @ref uct_iface_params_t.err_handler.
 *
 * @return error code          - In case of an error. (@ref ucs_status_t)
 */
ucs_status_t uct_ep_create_sockaddr(uct_iface_h iface,
                                    const ucs_sock_addr_t *sockaddr,
                                    const void *priv_data, size_t length,
                                    uct_ep_h *ep_p);


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
 * @brief Allocate memory for zero-copy sends and remote access.
 *
 * Allocate memory on the memory domain. In order to use this function, MD
 * must support @ref UCT_MD_FLAG_ALLOC flag.
 *
 * @param [in]     md          Memory domain to allocate memory on.
 * @param [in,out] length_p    Points to the size of memory to allocate. Upon successful
 *                             return, filled with the actual size that was allocated,
 *                             which may be larger than the one requested. Must be >0.
 * @param [in,out] address_p   The address
 * @param [in]     flags       Memory allocation flags, see @ref uct_md_mem_flags.
 * @param [in]     name        Name of the allocated region, used to track memory
 *                             usage for debugging and profiling.
 * @param [out]    memh_p      Filled with handle for allocated region.
 */
ucs_status_t uct_md_mem_alloc(uct_md_h md, size_t *length_p, void **address_p,
                              unsigned flags, const char *name, uct_mem_h *memh_p);


/**
 * @ingroup UCT_MD
 * @brief Release memory allocated by @ref uct_md_mem_alloc.
 *
 * @param [in]     md          Memory domain memory was allocated on.
 * @param [in]     memh        Memory handle, as returned from @ref uct_md_mem_alloc.
 */
ucs_status_t uct_md_mem_free(uct_md_h md, uct_mem_h memh);


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
 * @param [in]     memh        Memory handle, as returned from @ref uct_md_mem_alloc
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
 * @brief Check if memory type is owned by MD
 *
 *  Check memory type.
 *  @return Nonzero if memory is owned, 0 if not owned
 *
 * @param [in]     md        Memory domain to detect if memory belongs to.
 * @param [in]     address   Memory address to detect.
 * @param [in]     length    Size of memory
 */
int uct_md_is_mem_type_owned(uct_md_h md, void *addr, size_t length);

/**
 * @ingroup UCT_MD
 * @brief Allocate memory for zero-copy communications and remote access.
 *
 * Allocate potentially registered memory. Every one of the provided allocation
 * methods will be used, in turn, to perform the allocation, until one succeeds.
 *  Whenever the MD method is encountered, every one of the provided MDs will be
 * used, in turn, to allocate the memory, until one succeeds, or they are
 * exhausted. In this case the next allocation method from the initial list will
 * be attempted.
 *
 * @param [in]     addr        If @a addr is NULL, the underlying allocation routine
 *                             will choose the address at which to create the mapping.
 *                             If @a addr is non-NULL but UCT_MD_MEM_FLAG_FIXED is
 *                             not set, the address will be interpreted as a hint
 *                             as to where to establish the mapping. If @a addr is
 *                             non-NULL and UCT_MD_MEM_FLAG_FIXED is set, then
 *                             the specified address is interpreted as a requirement.
 *                             In this case, if the mapping to the exact address
 *                             cannot be made, the allocation request fails.
 * @param [in]     min_length  Minimal size to allocate. The actual size may be
 *                             larger, for example because of alignment restrictions.
 * @param [in]     flags       Memory allocation flags, see @ref uct_md_mem_flags.
 * @param [in]     methods     Array of memory allocation methods to attempt.
 * @param [in]     num_methods Length of 'methods' array.
 * @param [in]     mds         Array of memory domains to attempt to allocate
 *                             the memory with, for MD allocation method.
 * @param [in]     num_mds     Length of 'mds' array. May be empty, in such case
 *                             'mds' may be NULL, and MD allocation method will
 *                             be skipped.
 * @param [in]     name        Name of the allocation. Used for memory statistics.
 * @param [out]    mem         In case of success, filled with information about
 *                              the allocated memory. @ref uct_allocated_memory_t.
 */
ucs_status_t uct_mem_alloc(void *addr, size_t min_length, unsigned flags,
                           uct_alloc_method_t *methods, unsigned num_methods,
                           uct_md_h *mds, unsigned num_mds, const char *name,
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
 * @brief Read the configuration of the MD component.
 *
 * @param [in]  name          Name of the MD or the MD component.
 * @param [in]  env_prefix    If non-NULL, search for environment variables
 *                            starting with this UCT_<prefix>_. Otherwise, search
 *                            for environment variables starting with just UCT_.
 * @param [in]  filename      If non-NULL, read configuration from this file. If
 *                            the file does not exist, it will be ignored.
 * @param [out] config_p      Filled with a pointer to the configuration.
 *
 * @return Error code.
 */
ucs_status_t uct_md_config_read(const char *name, const char *env_prefix,
                                const char *filename,
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
 * @param [in]  rkey_buffer  Packed remote key buffer.
 * @param [out] rkey_ob      Filled with the unpacked remote key and its type.
 *
 * @return Error code.
 */
ucs_status_t uct_rkey_unpack(const void *rkey_buffer, uct_rkey_bundle_t *rkey_ob);


/**
 * @ingroup UCT_MD
 *
 * @brief Get a local pointer to remote memory.
 *
 * This routine returns a local pointer to the remote memory
 * described by the rkey bundle. The MD must support
 * @ref UCT_MD_FLAG_RKEY_PTR flag.
 *
 * @param [in]  rkey_ob      A remote key bundle as returned by
 *                           the @ref uct_rkey_unpack function.
 * @param [in]  remote_addr  A remote address within the memory area described
 *                           by the rkey_ob.
 * @param [out] addr_p       A pointer that can be used for direct access to
 *                           the remote memory.
 *
 * @return Error code if the remote memory cannot be accessed directly or
 *         the remote address is not valid.
 */
ucs_status_t uct_rkey_ptr(uct_rkey_bundle_t *rkey_ob, uint64_t remote_addr,
                          void **addr_p);


/**
 * @ingroup UCT_MD
 *
 * @brief Release a remote key.
 *
 * @param [in]  rkey_ob      Remote key to release.
 */
ucs_status_t uct_rkey_release(const uct_rkey_bundle_t *rkey_ob);


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
 * @return Non-zero if any communication was progressed, zero otherwise.
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
 *                         The @a iov pointer must be valid address of an array
 *                         of @ref ::uct_iov_t structures. A particular structure
 *                         pointer must be valid address. NULL terminated pointer
 *                         is not required.
 * @param [in] iovcnt      Size of the @a iov data @ref ::uct_iov_t structures
 *                         array. If @a iovcnt is zero, the data is considered empty.
 *                         @a iovcnt is limited by @ref uct_iface_attr_cap_put_max_iov
 *                         "uct_iface_attr::cap::put::max_iov"
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
 *                         The @a iov pointer must be valid address of an array
 *                         of @ref ::uct_iov_t structures. A particular structure
 *                         pointer must be valid address. NULL terminated pointer
 *                         is not required.
 * @param [in] iovcnt      Size of the @a iov data @ref ::uct_iov_t structures
 *                         array. If @a iovcnt is zero, the data is considered empty.
 *                         @a iovcnt is limited by @ref uct_iface_attr_cap_get_max_iov
 *                         "uct_iface_attr::cap::get::max_iov"
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
 * @param [in] ep            Destination endpoint handle.
 * @param [in] id            Active message id. Must be in range 0..UCT_AM_ID_MAX-1.
 * @param [in] header        Active message header.
 * @param [in] header_length Active message header length in bytes.
 * @param [in] iov           Points to an array of @ref ::uct_iov_t structures.
 *                           The @a iov pointer must be valid address of an array
 *                           of @ref ::uct_iov_t structures. A particular structure
 *                           pointer must be valid address. NULL terminated pointer
 *                           is not required.
 * @param [in] iovcnt        Size of the @a iov data @ref ::uct_iov_t structures
 *                           array. If @a iovcnt is zero, the data is considered empty.
 *                           @a iovcnt is limited by @ref uct_iface_attr_cap_am_max_iov
 *                           "uct_iface_attr::cap::am::max_iov"
 * @param [in] flags         Active message flags, see @ref uct_msg_flags.
 * @param [in] comp          Completion handle as defined by @ref ::uct_completion_t.
 *
 * @return UCS_INPROGRESS    Some communication operations are still in progress.
 *                           If non-NULL @a comp is provided, it will be updated
 *                           upon completion of these operations.
 *
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
UCT_INLINE_API ucs_status_t uct_ep_atomic_add64(uct_ep_h ep, uint64_t add,
                                                uint64_t remote_addr, uct_rkey_t rkey)
{
    return ep->iface->ops.ep_atomic_add64(ep, add, remote_addr, rkey);
}


/**
 * @ingroup UCT_AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_fadd64(uct_ep_h ep, uint64_t add,
                                                 uint64_t remote_addr, uct_rkey_t rkey,
                                                 uint64_t *result, uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_fadd64(ep, add, remote_addr, rkey, result, comp);
}


/**
 * @ingroup UCT_AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_swap64(uct_ep_h ep, uint64_t swap,
                                                 uint64_t remote_addr, uct_rkey_t rkey,
                                                 uint64_t *result, uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_swap64(ep, swap, remote_addr, rkey, result, comp);
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
UCT_INLINE_API ucs_status_t uct_ep_atomic_add32(uct_ep_h ep, uint32_t add,
                                                uint64_t remote_addr, uct_rkey_t rkey)
{
    return ep->iface->ops.ep_atomic_add32(ep, add, remote_addr, rkey);
}


/**
 * @ingroup UCT_AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_fadd32(uct_ep_h ep, uint32_t add,
                                                 uint64_t remote_addr, uct_rkey_t rkey,
                                                 uint32_t *result, uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_fadd32(ep, add, remote_addr, rkey, result, comp);
}


/**
 * @ingroup UCT_AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_swap32(uct_ep_h ep, uint32_t swap,
                                                 uint64_t remote_addr, uct_rkey_t rkey,
                                                 uint32_t *result, uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_swap32(ep, swap, remote_addr, rkey, result, comp);
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
 * @ingroup UCT_RESOURCE
 * @brief Add a pending request to an endpoint.
 *
 *  Add a pending request to the endpoint pending queue. The request will be
 * dispatched when the endpoint could potentially have additional send resources.
 *
 * @param [in]  ep    Endpoint to add the pending request to.
 * @param [in]  req   Pending request, which would be dispatched when more
 *                    resources become available. The user is expected to initialize
 *                    the "func" field.
 *                    After passed to the function, the request is owned by UCT,
 *                    until the callback is called and returns UCS_OK.
 *
 * @return UCS_OK       - request added to pending queue
 *         UCS_ERR_BUSY - request was not added to pending queue, because send
 *                        resources are available now. The user is advised to
 *                        retry.
 */
UCT_INLINE_API ucs_status_t uct_ep_pending_add(uct_ep_h ep, uct_pending_req_t *req)
{
    return ep->iface->ops.ep_pending_add(ep, req);
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
 * If it's required to pass non-zero imm value, @ref uct_ep_tag_eager_bcopy
 * should be used.
 *
 * @param [in]  ep        Destination endpoint handle.
 * @param [in]  tag       Tag to use for the eager message.
 * @param [in]  data      Data to send.
 * @param [in]  length    Data length.
 *
 * @return UCS_OK              - operation completed successfully.
 * @return UCS_ERR_NO_RESOURCE - could not start the operation now due to lack
 *                               of send resources.
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
 *                        A particular structure pointer must be valid address.
 *                        NULL terminated pointer is not required.
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
 * @return UCS_ERR_NO_RESOURCE - could not start the operation now due to lack
 *                               of send resources.
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
 *                            address. NULL terminated pointer is not required.
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
 * @return UCS_ERR_NO_RESOURCE - could not start the operation now due to lack of
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
 *                          The @a iov pointer must be valid address of an array
 *                          of @ref ::uct_iov_t structures. A particular structure
 *                          pointer must be valid address. NULL terminated pointer
 *                          is not required.
 * @param [in]    iovcnt    Size of the @a iov data @ref ::uct_iov_t structures
 *                          array. If @a iovcnt is zero, the data is considered empty.
 *                          @a iovcnt is limited by @ref uct_iface_attr_cap_tag_recv_iov
 *                          "uct_iface_attr::cap::tag::max_iov"
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
 * Perform a progress on an interface.
 */
UCT_INLINE_API unsigned uct_iface_progress(uct_iface_h iface)
{
    return iface->ops.iface_progress(iface);
}


/**
 * @example uct_hello_world.c
 * UCT hello world client / server example utility.
 */

END_C_DECLS

#endif
