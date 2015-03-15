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
#include <ucs/config/types.h>

#include <sys/socket.h>
#include <stdio.h>
#include <sched.h>

/**
* @defgroup RESOURCE   Communication Resource
* @{
* This section describes a concept of the Communication Resource and routines
* associated with the concept.
* @}
*/

/**
 * @defgroup CONTEXT    UCT Communication Context
 * @{
 * UCT context is a primary concept of UCX design which provides an isolation
 * mechanism, allowing resources associated with the context to separate or
 * share network communication context across multiple instances of parallel
 * programming models.
 *
 * This section provides a detailed description of this concept and
 * routines associated with it.
 *
 * @}
 */

/**
 * @defgroup PD    UCT Protection Domain
 * @{
 * The protection domain defines memory allocation, registration, key exchange
 * operations.
 * @}
 */

/**
 * @defgroup AM   Active messages
 * @{
 * Defines active message functions.
 * @}
 */

/**
 * @defgroup RMA  Remote memeory access operations.
 * @{
 * Defines remote memory access operairons.
 * @}
 */

/**
 * @defgroup AMO   Atomic operations.
 * @{
 * Defines atomic operations..
 * @}
 */

/**
 * @ingroup RESOURCE
 * @brief Communication resource descriptor
 *
 * Resource descriptor is an object representing the network resource.
 * Resource descriptor could represent a stand-alone communication resource
 * such as a HCA port, network interface, or multiple resources such as
 * multiple network interfaces or communication ports. It could also represent
 * virtual communication resources that are defined over a single physical
 * network interface.
 */
typedef struct uct_resource_desc {
    char                     tl_name[UCT_MAX_NAME_LEN];   /**< Transport name */
    char                     dev_name[UCT_MAX_NAME_LEN];  /**< Hardware device name */
    uint64_t                 latency;      /**< Latency, nanoseconds */
    size_t                   bandwidth;    /**< Bandwidth, bytes/second */
    cpu_set_t                local_cpus;   /**< Mask of CPUs near the resource */
    struct sockaddr_storage  subnet_addr;  /**< Subnet address. Devices which can
                                                reach each other have same address */
} uct_resource_desc_t;


/**
 * @ingroup RESOURCE
 * @brief Thread mode.
 *
 * Specifies thread sharing mode of the object.
 */
typedef enum {
    UCT_THREAD_MODE_SINGLE,   /**< Only one thread can access */
    UCT_THREAD_MODE_FUNNELED, /**< Multiple threads can access, but only one at a time */
    UCT_THREAD_MODE_MULTI,    /**< Multiple threads can access concurrently */
    UCT_THREAD_MODE_LAST
} uct_thread_mode_t;


/**
 * Opaque type for interface address.
 */
struct uct_iface_addr {
};


/**
 * Opaque type for endpoint address.
 */
struct uct_ep_addr {
};

/**
 * @ingroup RESOURCE
 * @brief  List of capabilities supported by UCX API
 *
 * The enumeration list presents a full list of operations and capabilities
 * exposed by UCX API.
 */
enum {
    /* Active message capabilities */
    UCT_IFACE_FLAG_AM_SHORT       = UCS_BIT(0), /**< Short active message */
    UCT_IFACE_FLAG_AM_BCOPY       = UCS_BIT(1), /**< Buffered active message */
    UCT_IFACE_FLAG_AM_ZCOPY       = UCS_BIT(2), /**< Zero-copy active message */

    /* PUT capabilities */
    UCT_IFACE_FLAG_PUT_SHORT      = UCS_BIT(4), /**< Short put */
    UCT_IFACE_FLAG_PUT_BCOPY      = UCS_BIT(5), /**< Buffered put */
    UCT_IFACE_FLAG_PUT_ZCOPY      = UCS_BIT(6), /**< Zero-copy put */

    /* GET capabilities */
    UCT_IFACE_FLAG_GET_SHORT      = UCS_BIT(8), /**< Short get */
    UCT_IFACE_FLAG_GET_BCOPY      = UCS_BIT(9), /**< Buffered get */
    UCT_IFACE_FLAG_GET_ZCOPY      = UCS_BIT(10) /**< Zero-copy get */,

    /* Atomic operations capabilities */
    UCT_IFACE_FLAG_ATOMIC_ADD32   = UCS_BIT(16), /**< 32bit atomic add */
    UCT_IFACE_FLAG_ATOMIC_ADD64   = UCS_BIT(17), /**< 64bit atomic add */
    UCT_IFACE_FLAG_ATOMIC_FADD32  = UCS_BIT(18), /**< 32bit atomic fetch-and-add */
    UCT_IFACE_FLAG_ATOMIC_FADD64  = UCS_BIT(19), /**< 64bit atomic fetch-and-add */
    UCT_IFACE_FLAG_ATOMIC_SWAP32  = UCS_BIT(20), /**< 32bit atomic swap */
    UCT_IFACE_FLAG_ATOMIC_SWAP64  = UCS_BIT(21), /**< 64bit atomic swap */
    UCT_IFACE_FLAG_ATOMIC_CSWAP32 = UCS_BIT(22), /**< 32bit atomic compare-and-swap */
    UCT_IFACE_FLAG_ATOMIC_CSWAP64 = UCS_BIT(23), /**< 64bit atomic compare-and-swap */

    /* Error handling capabilities */
    UCT_IFACE_FLAG_ERRHANDLE_SHORT_BUF  = UCS_BIT(32), /**< Invalid buffer for short operation */
    UCT_IFACE_FLAG_ERRHANDLE_BCOPY_BUF  = UCS_BIT(33), /**< Invalid buffer for buffered operation */
    UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF  = UCS_BIT(34), /**< Invalid buffer for zero copy operation */
    UCT_IFACE_FLAG_ERRHANDLE_AM_ID      = UCS_BIT(35), /**< Invalid AM id on remote */
    UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM = UCS_BIT(35), /**<  Remote memory access */
};


/**
 * @ingroup CONTEXT
 * @brief  Memory allocation methods.
 */
typedef enum {
    UCT_ALLOC_METHOD_PD,   /**< Allocate using protection domain */
    UCT_ALLOC_METHOD_HEAP, /**< Allocate from heap usign libc allocator */
    UCT_ALLOC_METHOD_MMAP, /**< Allocate from OS using mmap() syscall */
    UCT_ALLOC_METHOD_HUGE, /**< Allocate huge pages */
    UCT_ALLOC_METHOD_LAST,
    UCT_ALLOC_METHOD_DEFAULT = UCT_ALLOC_METHOD_LAST /**< Use default method */
} uct_alloc_method_t;


/**
 * @ingroup RESOURCE
 * @brief Interface attributes: capabilities and limitations.
 */
struct uct_iface_attr {
    struct {
        struct {
            size_t           max_short;  /**< Maximal size for put_short */
            size_t           max_bcopy;  /**< Maximal size for put_bcopy */
            size_t           max_zcopy;  /**< Maximal size for put_zcopy */
        } put;

        struct {
            size_t           max_bcopy;  /**< Maximal size for get_bcopy */
            size_t           max_zcopy;  /**< Maximal size for get_zcopy */
        } get;

        struct {
            size_t           max_short;  /**< Total max. size (incl. the header) */
            size_t           max_bcopy;  /**< Total max. size (incl. the header) */
            size_t           max_zcopy;  /**< Total max. size (incl. the header) */
            size_t           max_hdr;    /**< Max. header size for bcopy/zcopy */
        } am;

        uint64_t             flags;      /**< Flags from UCT_IFACE_FLAG_xx */
    } cap;

    size_t                   iface_addr_len; /**< Size of interface address */
    size_t                   ep_addr_len;    /**< Size of endpoint address */
    size_t                   completion_priv_len;  /**< Size of private data in @ref uct_completion_t */
};


/**
 * @ingroup PD
 * @brief  Protection domain capability flags.
 */
enum {
    UCT_PD_FLAG_ALLOC     = UCS_BIT(0),  /**< PD support memory allocation */
    UCT_PD_FLAG_REG       = UCS_BIT(1),  /**< PD support memory registration */
};


/**
 * @ingroup PD
 * @brief  List of allocation methods, in order of priority (high to low).
 */
typedef struct uct_alloc_methods {
    uint8_t                  count;  /**< Number of allocation methods in the array */
    uint8_t                  methods[UCT_ALLOC_METHOD_LAST];
                                     /**< Array of allocation methods */
} uct_alloc_methods_t;


/**
 * @ingroup PD
 * @brief  Protection domain attributes.
 */
struct uct_pd_attr {
    char                     name[UCT_MAX_NAME_LEN]; /**< Protection domain name */

    struct {
        size_t               max_alloc;     /**< Maximal allocation size */
        size_t               max_reg;       /**< Maximal registration size */
        uint64_t             flags;         /**< UCT_PD_FLAG_xx */
    } cap;

    uct_alloc_methods_t      alloc_methods; /**< Allocation methods priority */
    size_t                   rkey_packed_size; /**< Size of buffer needed for packed rkey */
};


/**
 * @ingroup PD
 * @brief Remote key with its type
 */
typedef struct uct_rkey_bundle {
    uct_rkey_t               rkey;   /**< Remote key descriptor, passed to RMA functions */
    void                     *type;  /**< Remote key type */
} uct_rkey_bundle_t;


/**
 * @ingroup RESOURCE
 * @brief Completion handle.
 *
 * This structure should be allocated by the user, while reserving at least @ref
 * uct_iface_attr_t::completion_priv_len bytes for the 'priv' field. If the send
 * operation returns UCT_INPROGRESS, this structure will be owned by the transport
 * until the send completes. This completion is signaled by calling the callback
 * function specified in the 'func' field of this structure.
 */
struct uct_completion {
    uct_completion_callback_t func;    /**< User callback function */
    char                      priv[0]; /**< Actual size of this field is
                                            returned in completion_priv_len
                                            by @ref uct_iface_query() */
};


/**
 * @ingroup CONTEXT
 * @brief   UCT global context initialization
 *
 * This routine creates and initializes a UCT @ref uct_context "global context".
 *
 * @warning The function must be called before any other UCT function call in
 * the application.
 *
 * This routine discovers the available network interfaces, and initializes the
 * network resources required for discovering the device.  This routine is
 * responsible for inializing all information required for a particular
 * communication scope, for example, MPI instance, OpenSHMEM instance.
 *
 * @note @li Higher level protocols can add additional communication isolation,
 * as MPI does with it's communicator object. A single communication context
 * may be used to support multiple MPI communicators.  @li The context can be
 * used to isolate the communication that corresponds to different protocols.
 * For example, if MPI and OpenSHMEM are using UCCS to isolate the MPI
 * communication from the OpenSHMEM communication, users should use different
 * communication context for each of the protocol.
 *
 * @param [out] context_p   Filled with context handle.
 *
 * @return Error code.
 */
ucs_status_t uct_init(uct_context_h *context_p);


/**
 * @ingroup CONTEXT
 * @brief   UCT global context finalization
 *
 * This routine finalizes and releases the resources associated with a UCT
 * global context.
 *
 * @warning Users cannot call any communication routines using the finalized
 * UCT context.
 *
 * The finalization process releases and shuts down all resources associated
 * with the @ref uct_context "context".  After calling this routine, calling
 * any UCT routine without calling initialization routine is invalid.
 *
 * @param [in] context   Handle to context.
 *
 * @return void.
 */
void uct_cleanup(uct_context_h context);


/**
 * @ingroup RESOURCE
 * @brief Query for transport resources.
 *
 * This routine queries the @ref uct_context "global context" for communication
 * that are available for the context.
 * As an input, users provide the @ref uct_context "global context" ,
 * and as an output the routine returns an array of the resource @ref
 * uct_resource_desc_t "descriptors".
 *
 * @param [in]  context         Handle to context.
 * @param [out] resources_p     Filled with a pointer to an array of resource
 *                              descriptors.
 * @param [out] num_resources_p Filled with the number of resources in the array.
 *
 * @return Error code.
 */
ucs_status_t uct_query_resources(uct_context_h context,
                                 uct_resource_desc_t **resources_p,
                                 unsigned *num_resources_p);


/**
 * @ingroup RESOURCE
 * @brief Release the list of resources returned from @ref uct_query_resources.
 *
 * This routine releases the memory associated with the list of resources
 * allocated by @ref uct_query_resources.
 *
 * @param [in] resources  Array of resource descriptors to release.
 *
 * @return void.
 */
void uct_release_resource_list(uct_resource_desc_t *resources);


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
ucs_status_t uct_worker_create(uct_context_h context, uct_thread_mode_t thread_mode,
                               uct_worker_h *worker_p);


/**
 * @ingroup CONTEXT
 * @brief Destroy a worker object.
 *
 * @param [in]  worker        Worker object to destroy.
 */
void uct_worker_destroy(uct_worker_h worker);


/**
 * @ingroup CONTEXT
 * @brief Explicit progress for UCT worker.
 *
 * This routine explicitly progresses any outstanding communication operations
 * and active message requests.
 *
 * @note @li In the current implementation, users @b MUST call this routine
 * to receive the active message requests.
 *
 * @param [in] worker   Handle to worker.
 */
void uct_worker_progress(uct_worker_h worker);


/**
 * @ingroup RESOURCE
 * @brief Read transport-specific interface configuration.
 *
 * @param [in]  context       Handle to context.
 * @param [in]  tl_name       Transport name.
 * @param [in]  env_prefix    If non-NULL, search for environment variables
 *                            starting with this UCT_<prefix>_. Otherwise, search
 *                            for environment variables starting with just UCT_.
 * @param [in]  filename      If non-NULL, read configuration from this file. If
 *                            the file does not exist, it will be ignored.
 * @param [out] config_p      Filled with a pointer to configuration.
 *
 * @return Error code.
 */
ucs_status_t uct_iface_config_read(uct_context_h context, const char *tl_name,
                                   const char *env_prefix, const char *filename,
                                   uct_iface_config_t **config_p);


/**
 * @ingroup RESOURCE
 * @brief Release configuration memory returned from uct_iface_read_config().
 *
 * @param [in]  config        Configuration to release.
 */
void uct_iface_config_release(uct_iface_config_t *config);


/**
 * @ingroup RESOURCE
 * @brief Print interface configuration to a stream.
 *
 * @param [in]  config        Configuration to print.
 * @param [in]  stream        Output stream to print to.
 * @param [in]  title         Title to the output.
 * @param [in]  print_flags   Controls how the configuration is printed.
 */
void uct_iface_config_print(uct_iface_config_t *config, FILE *stream,
                            const char *title, ucs_config_print_flags_t print_flags);


/**
 * @ingroup CONTEXT
 * @brief Print interface configuration to a stream.
 *
 * @param [in]  config        Configuration to release.
 * @param [in]  name          Configuration variable name.
 * @param [in]  value         Value to set.
 *
 * @return Error code.
 */
ucs_status_t uct_iface_config_modify(uct_iface_config_t *config,
                                     const char *name, const char *value);


/**
 * @ingroup RESOURCE
 * @brief Open a communication interface.
 *
 * @param [in]  worker        Handle to worker which will be used to progress
 *                             communications on this interface.
 * @param [in]  tl_name       Transport name.
 * @param [in]  dev_name      Hardware device name,
 * @param [in]  rx_headroom   How much bytes to reserve before the receive segment.
 * @param [in]  config        Interface configuration options. Should be obtained
 *                            from uct_iface_read_config() function, or point to
 *                            transport-specific structure which extends uct_iface_config_t.
 * @param [out] iface_p       Filled with a handle to opened communication interface.
 *
 * @return Error code.
 */
ucs_status_t uct_iface_open(uct_worker_h worker, const char *tl_name,
                            const char *dev_name, size_t rx_headroom,
                            uct_iface_config_t *config, uct_iface_h *iface_p);


/**
 * @ingroup RESOURCE
 * @brief Close and destroy an interface.
 *
 * @param [in]  iface  Interface to close.
 */
void uct_iface_close(uct_iface_h iface);


/**
 * @ingroup RESOURCE
 * @brief Get interface attributes.
 *
 * @param [in]  iface   Interface to query.
 */
ucs_status_t uct_iface_query(uct_iface_h iface, uct_iface_attr_t *iface_attr);


/**
 * @ingroup RESOURCE
 * @brief Get interface address.
 *
 * @param [in]  iface       Interface to query.
 * @param [out] iface_addr  Filled with interface address. The size of the buffer
 *                           provided must be at least @ref uct_iface_attr_t::iface_addr_len.
 */
ucs_status_t uct_iface_get_address(uct_iface_h iface, uct_iface_addr_t *iface_addr);


/**
 * @ingroup AM
 * @brief Set active message handler for the interface.
 *
 * Only one handler can be set of each active message ID, and setting a handler
 * replaces the previous value. If cb == NULL, the current handler is removed.
 *
 * @param [in]  iface    Interface to set the active message handler for.
 * @param [in]  id       Active message id. Must be 0..UCT_AM_ID_MAX-1.
 * @param [in]  cb       Active message callback. NULL to clear.
 * @param [in]  arg      Active message argument.
 */
ucs_status_t uct_iface_set_am_handler(uct_iface_h iface, uint8_t id,
                                      uct_am_callback_t cb, void *arg);


/**
 * @ingroup RESOURCE
 * @brief Create new endpoint.
 *
 * @param [in]  iface   Interface to create the endpoint on.
 * @param [out] ep_p    Filled with handle to the new endpoint.
 */
ucs_status_t uct_ep_create(uct_iface_h iface, uct_ep_h *ep_p);


/**
 * @ingroup RESOURCE
 * @brief Destroy an endpoint.
 *
 * @param [in] ep       Endpoint to destroy.
 */
void uct_ep_destroy(uct_ep_h ep);


/**
 * @ingroup RESOURCE
 * @brief Get endpoint address.
 *
 * @param [in]  ep       Endpoint to query.
 * @param [out] ep_addr  Filled with endpoint address. The size of the buffer
 *                        provided must be at least @ref uct_iface_attr_t::ep_addr_len.
 */
ucs_status_t uct_ep_get_address(uct_ep_h ep, uct_ep_addr_t *ep_addr);


/**
 * @ingroup RESOURCE
 * @brief Connect endpoint to a remote interface.
 *
 * TODO
 */
ucs_status_t uct_ep_connect_to_iface(uct_ep_h ep, uct_iface_addr_t *iface_addr);


/**
 * @ingroup RESOURCE
 * @brief Connect endpoint to a remote endpoint.
 *
 * @param [in] ep           Endpoint to connect.
 * @param [in] iface_addr   Remote interface address.
 * @param [in] ep_addr      Remote endpoint address.
 */
ucs_status_t uct_ep_connect_to_ep(uct_ep_h ep, uct_iface_addr_t *iface_addr,
                                  uct_ep_addr_t *ep_addr);


/**
 * @ingroup PD
 * @brief Query for protection domain attributes..
 *
 * @param [in]  pd       Protection domain to query.
 * @param [out] pd_attr  Filled with protection domain attributes.
 */
ucs_status_t uct_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr);


/**
 * @ingroup PD
 * @brief Register memory for zero-copy sends and remote access.
 *
 * @param [in]     pd        Protection domain to register memory on.
 * @param [out]    address   Memory to register.
 * @param [in,out] length    Size of memory to register. Must be >0.
 * @param [out]    memh_p    Filled with handle for allocated region.
 */
ucs_status_t uct_pd_mem_reg(uct_pd_h pd, void *address, size_t length,
                            uct_mem_h *memh_p);


/**
 * @ingroup PD
 * @brief Undo the operation of uct_pd_mem_reg().
 *
 * @param [in]  pd          Protection domain which was used to register the memory.
 * @paran [in]  memh        Local access key to memory region.
 */
ucs_status_t uct_pd_mem_dereg(uct_pd_h pd, uct_mem_h memh);


/**
 * @ingroup PD
 * @brief Allocate memory for zero-copy sends and remote access.
 *
 * Allocate registered memory. The memory would either be allocated with the
 * protection domain, or allocated using other method and then registered with
 * the protection domain.
 *
 * TODO allow passing multiple protection domains.
 *
 * @param [in]     pd          Protection domain to allocate memory on.
 * @param [in]     method      Memory allocation method. Can be UCT_ALLOC_METHOD_DEFAULT.
 * @param [in,out] length_p    Points to a value which specifies how many bytes to
 *                              allocate. Filled with the actual allocated size,
 *                              which is larger than or equal to the requested size.
                                Must be >0.
 * @param [in]     alignment   Allocation alignment, must be power-of-2.
 * @param [out]    address_p   Filled with a pointer to allocated memory.
 * @param [out]    memh_p      Filled with a handle for allocated memory, which can
 *                              be used for zero-copy communications.
 * @param [in]     alloc_name  Name of the allocation. Used for memory statistics.
 */
ucs_status_t uct_pd_mem_alloc(uct_pd_h pd, uct_alloc_method_t method,
                              size_t *length_p, size_t alignment, void **address_p,
                              uct_mem_h *memh_p, const char *alloc_name);


/**
 * @ingroup PD
 * @brief Release allocated memory.
 *
 * Release the memory allocated by @ref uct_pd_mem_alloc. pd should be the same
 * as passed to @ref uct_pd_mem_alloc, and address should be one returned from it.
 *
 * @param [in]  pd          Protection domain which was used to allocate the memory.
 * @param [in]  address     Address of allocated memory.
 * @paran [in]  memh        Local access key to memory region.
 */
ucs_status_t uct_pd_mem_free(uct_pd_h pd, void *address, uct_mem_h memh);


/**
 * @ingroup PD
 *
 * @brief Pack a remote key.
 *
 * @param [in]  pd           Handle to protection domain.
 * @param [in]  memh         Local key, whose remote key should be packed.
 * @param [out] rkey_buffer  Filled with packed remote key.
 *
 * @return Error code.
 */
ucs_status_t uct_pd_rkey_pack(uct_pd_h pd, uct_mem_h memh, void *rkey_buffer);


/**
 * @ingroup PD
 *
 * @brief Unpack a remote key.
 *
 * @param [in]  pd           Handle to protection domain.
 * @param [in]  rkey_buffer  Packed remote key buffer.
 * @param [out] rkey_ob      Filled with the unpacked remote key and its type.
 *
 * @return Error code.
 */
ucs_status_t uct_pd_rkey_unpack(uct_pd_h pd, void *rkey_buffer,
                                uct_rkey_bundle_t *rkey_ob);


/**
 * @ingroup PD
 *
 * @brief Release a remote key.
 *
 * @param [in]  pd           Handle to protection domain.
 * @param [in]  rkey_ob      Remote key to release.
 */
void uct_pd_rkey_release(uct_pd_h pd, uct_rkey_bundle_t *rkey_ob);


/**
 * @ingroup RESOURCE
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_iface_flush(uct_iface_h iface)
{
    return iface->ops.iface_flush(iface);
}


/**
 * @ingroup AM
 * @brief Release active message descriptor, which was passed to the active
 * message callback, and owned by the callee.
 */
UCT_INLINE_API void uct_iface_release_am_desc(uct_iface_h iface, void *desc)
{
    iface->ops.iface_release_am_desc(iface, desc);
}


/**
 * @ingroup RMA
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_put_short(uct_ep_h ep, void *buffer, unsigned length,
                                             uint64_t remote_addr, uct_rkey_t rkey)
{
    return ep->iface->ops.ep_put_short(ep, buffer, length, remote_addr, rkey);
}


/**
 * @ingroup RMA
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_put_bcopy(uct_ep_h ep, uct_pack_callback_t pack_cb,
                                             void *arg, size_t length, uint64_t remote_addr,
                                             uct_rkey_t rkey)
{
    return ep->iface->ops.ep_put_bcopy(ep, pack_cb, arg, length, remote_addr, rkey);
}


/**
 * @ingroup RMA
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_put_zcopy(uct_ep_h ep, void *buffer, size_t length,
                                             uct_mem_h memh, uint64_t remote_addr,
                                             uct_rkey_t rkey, uct_completion_t *comp)
{
    return ep->iface->ops.ep_put_zcopy(ep, buffer, length, memh, remote_addr,
                                       rkey, comp);
}


/**
 * @ingroup RMA
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_get_bcopy(uct_ep_h ep, size_t length,
                                             uint64_t remote_addr, uct_rkey_t rkey,
                                             uct_completion_t *comp)
{
    return ep->iface->ops.ep_get_bcopy(ep, length, remote_addr, rkey, comp);
}


/**
 * @ingroup RMA
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_get_zcopy(uct_ep_h ep, void *buffer, size_t length,
                                             uct_mem_h memh, uint64_t remote_addr,
                                             uct_rkey_t rkey, uct_completion_t *comp)
{
    return ep->iface->ops.ep_get_zcopy(ep, buffer, length, memh, remote_addr,
                                       rkey, comp);
}


/**
 * @ingroup AM
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                                            void *payload, unsigned length)
{
    return ep->iface->ops.ep_am_short(ep, id, header, payload, length);
}


/**
 * @ingroup AM
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_am_bcopy(uct_ep_h ep, uint8_t id,
                                            uct_pack_callback_t pack_cb,
                                            void *arg, size_t length)
{
    return ep->iface->ops.ep_am_bcopy(ep, id, pack_cb, arg, length);
}


/**
 * @ingroup AM
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_am_zcopy(uct_ep_h ep, uint8_t id, void *header,
                                            unsigned header_length, void *payload,
                                            size_t length, uct_mem_h memh,
                                            uct_completion_t *comp)
{
    return ep->iface->ops.ep_am_zcopy(ep, id, header, header_length, payload,
                                      length, memh, comp);
}

/**
 * @ingroup AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_add64(uct_ep_h ep, uint64_t add,
                                                uint64_t remote_addr, uct_rkey_t rkey)
{
    return ep->iface->ops.ep_atomic_add64(ep, add, remote_addr, rkey);
}


/**
 * @ingroup AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_fadd64(uct_ep_h ep, uint64_t add,
                                                 uint64_t remote_addr, uct_rkey_t rkey,
                                                 uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_fadd64(ep, add, remote_addr, rkey, comp);
}


/**
 * @ingroup AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_swap64(uct_ep_h ep, uint64_t swap,
                                                 uint64_t remote_addr, uct_rkey_t rkey,
                                                 uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_swap64(ep, swap, remote_addr, rkey, comp);
}


/**
 * @ingroup AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_cswap64(uct_ep_h ep, uint64_t compare, uint64_t swap,
                                                  uint64_t remote_addr, uct_rkey_t rkey,
                                                  uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_cswap64(ep, compare, swap, remote_addr, rkey, comp);
}


/**
 * @ingroup AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_add32(uct_ep_h ep, uint32_t add,
                                                uint64_t remote_addr, uct_rkey_t rkey)
{
    return ep->iface->ops.ep_atomic_add32(ep, add, remote_addr, rkey);
}


/**
 * @ingroup AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_fadd32(uct_ep_h ep, uint32_t add,
                                                 uint64_t remote_addr, uct_rkey_t rkey,
                                                 uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_fadd32(ep, add, remote_addr, rkey, comp);
}


/**
 * @ingroup AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_swap32(uct_ep_h ep, uint32_t swap,
                                                 uint64_t remote_addr, uct_rkey_t rkey,
                                                 uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_swap32(ep, swap, remote_addr, rkey, comp);
}


/**
 * @ingroup AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_cswap32(uct_ep_h ep, uint32_t compare, uint32_t swap,
                                                  uint64_t remote_addr, uct_rkey_t rkey,
                                                  uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_cswap32(ep, compare, swap, remote_addr, rkey, comp);
}


/**
 * @ingroup RESOURCE
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_flush(uct_ep_h ep)
{
    return ep->iface->ops.ep_flush(ep);
}

#endif
