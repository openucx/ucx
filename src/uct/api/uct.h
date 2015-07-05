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
#include <uct/api/addr.h>
#include <uct/api/tl.h>
#include <uct/api/version.h>
#include <ucs/async/async_fwd.h>
#include <ucs/config/types.h>
#include <ucs/type/status.h>
#include <ucs/type/thread_mode.h>

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
 * @brief Protection domain resource descriptor.
 *
 * This structure describes a protection domain resource.
 */
typedef struct uct_pd_resource_desc {
    char                     pd_name[UCT_PD_NAME_MAX]; /**< Protection domain name */
} uct_pd_resource_desc_t;


/**
 * @ingroup RESOURCE
 * @brief Communication resource descriptor.
 *
 * Resource descriptor is an object representing the network resource.
 * Resource descriptor could represent a stand-alone communication resource
 * such as a HCA port, network interface, or multiple resources such as
 * multiple network interfaces or communication ports. It could also represent
 * virtual communication resources that are defined over a single physical
 * network interface.
 */
typedef struct uct_tl_resource_desc {
    char                     tl_name[UCT_TL_NAME_MAX];   /**< Transport name */
    char                     dev_name[UCT_DEVICE_NAME_MAX]; /**< Hardware device name */
    uint64_t                 latency;      /**< Latency, nanoseconds */
    size_t                   bandwidth;    /**< Bandwidth, bytes/second */
} uct_tl_resource_desc_t;

#define UCT_TL_RESOURCE_DESC_FMT              "%s/%s"
#define UCT_TL_RESOURCE_DESC_ARG(_resource)   (_resource)->tl_name, (_resource)->dev_name

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
    UCT_IFACE_FLAG_GET_ZCOPY      = UCS_BIT(10), /**< Zero-copy get */

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

    /* Connection establishment */
    UCT_IFACE_FLAG_CONNECT_TO_IFACE = UCS_BIT(40), /**< Supports connecting to interface */
    UCT_IFACE_FLAG_CONNECT_TO_EP    = UCS_BIT(41), /**< Supports connecting to specific endpoint */

    /* Thread safety */
    UCT_IFACE_FLAG_AM_THREAD_SINGLE = UCS_BIT(44), /**< Active messages callback is invoked only from main thread */

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
 * @brief  Protection domain attributes.
 */
struct uct_pd_attr {
    struct {
        size_t               max_alloc;     /**< Maximal allocation size */
        size_t               max_reg;       /**< Maximal registration size */
        uint64_t             flags;         /**< UCT_PD_FLAG_xx */
    } cap;

    char                     component_name[UCT_PD_COMPONENT_NAME_MAX]; /**< PD component name */
    size_t                   rkey_packed_size; /**< Size of buffer needed for packed rkey */
    cpu_set_t                local_cpus;    /**< Mask of CPUs near the resource */
};


/**
 * @ingroup PD
 * @brief Describes a memory allocated by UCT.
 *
 * This structure describes a memory block allocated by UCT layer. The block
 * could be allocated with one of several methods by @ref uct_mem_alloc().
 */
typedef struct uct_allocated_memory {
    void                     *address; /**< Address of allocated memory */
    size_t                   length;   /**< Real size of allocated memory */
    uct_alloc_method_t       method;   /**< Method used to allocate the memory */
    uct_pd_h                 pd;       /**< if method==PD: PD used to allocate the memory */
    uct_mem_h                memh;     /**< if method==PD: PD memory handle */
} uct_allocated_memory_t;


/**
 * @ingroup PD
 * @brief Remote key with its type
 */
typedef struct uct_rkey_bundle {
    uct_rkey_t               rkey;    /**< Remote key descriptor, passed to RMA functions */
    void                     *handle; /**< Handle, used internally for releasing the key */
    void                     *type;   /**< Remote key type */
} uct_rkey_bundle_t;


/**
 * @ingroup RESOURCE
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


extern const char *uct_alloc_method_names[];


/**
 * @ingroup RESOURCE
 * @brief Query for memory resources.
 *
 * Obtain the list of protection domain resources available on the current system.
 *
 * @param [out] resources_p     Filled with a pointer to an array of resource
 *                              descriptors.
 * @param [out] num_resources_p Filled with the number of resources in the array.
 *
 * @return Error code.
 */
ucs_status_t uct_query_pd_resources(uct_pd_resource_desc_t **resources_p,
                                    unsigned *num_resources_p);

/**
 * @ingroup RESOURCE
 * @brief Release the list of resources returned from @ref uct_query_pd_resources.
 *
 * This routine releases the memory associated with the list of resources
 * allocated by @ref uct_query_pd_resources.
 *
 * @param [in] resources  Array of resource descriptors to release.
 */
void uct_release_pd_resource_list(uct_pd_resource_desc_t *resources);


/**
 * @ingroup RESOURCE
 * @brief Open a protection domain.
 *
 * Open a specific protection domain. All communications and memory operations
 * are performed in the context of a specific protection domain. Therefore it
 * must be created before communication resources.
 *
 * @param [in]  pd_name         Protection domain name, as returned from @ref
 *                              uct_query_pd_resources.
 * @param [out] pd_p            Filled with a handle to the protection domain.
 *
 * @return Error code.
 */
ucs_status_t uct_pd_open(const char *pd_name, uct_pd_h *pd_p);

/**
 * @ingroup RESOURCE
 * @brief Close a protection domain.
 *
 * @param [in]  pd               Protection domain to close.
 */
void uct_pd_close(uct_pd_h pd);


/**
 * @ingroup RESOURCE
 * @brief Query for transport resources.
 *
 * This routine queries the @ref uct_pd_t "protection domain" for communication
 * resources that are available for it.
 *
 * @param [in]  pd              Handle to protection domain.
 * @param [out] resources_p     Filled with a pointer to an array of resource
 *                              descriptors.
 * @param [out] num_resources_p Filled with the number of resources in the array.
 *
 * @return Error code.
 */
ucs_status_t uct_pd_query_tl_resources(uct_pd_h pd,
                                       uct_tl_resource_desc_t **resources_p,
                                       unsigned *num_resources_p);


/**
 * @ingroup RESOURCE
 * @brief Release the list of resources returned from @ref uct_pd_query_tl_resources.
 *
 * This routine releases the memory associated with the list of resources
 * allocated by @ref uct_query_tl_resources.
 *
 * @param [in] resources  Array of resource descriptors to release.
 */
void uct_release_tl_resource_list(uct_tl_resource_desc_t *resources);


/**
 * @ingroup CONTEXT
 * @brief Create a worker object.
 *
 *  The worker represents a progress engine. Multiple progress engines can be
 * created in an application, for example to be used by multiple threads.
 * Every worker can be progressed independently of others.
 *
 * @param [in]  async         Context for async event handlers.
  *                            Can be NULL, which means that event handlers will
 *                             not have particular context.
 * @param [in]  thread_mode   Thread access mode to the worker and resources
 *                             created on it.
 * @param [out] worker_p      Filled with a pointer to the worker object.
 */
ucs_status_t uct_worker_create(ucs_async_context_t *async,
                               ucs_thread_mode_t thread_mode,
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
ucs_status_t uct_iface_config_read(const char *tl_name, const char *env_prefix,
                                   const char *filename,
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
void uct_iface_config_print(const uct_iface_config_t *config, FILE *stream,
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
 * @param [in]  pd            Protection domain to create the interface on.
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
ucs_status_t uct_iface_open(uct_pd_h pd, uct_worker_h worker,
                            const char *tl_name, const char *dev_name,
                            size_t rx_headroom, const uct_iface_config_t *config,
                            uct_iface_h *iface_p);


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
ucs_status_t uct_iface_get_address(uct_iface_h iface, struct sockaddr *addr);


/**
 * @ingroup RESOURCE
 * @brief Check if remote iface address is reachable.
 *
 * This function checks if a remote address can be reached from a local interface.
 * If the function returns true, it does not necessarily mean a connection and/or
 * data transfer would succeed, since the reachability check is a local operation
 * it does not detect issues such as network mis-configuration or lack of connectivity.
 *
 * @param [in]  iface      Interface to check reachability from.
 * @param [in]  addr       Address to check reachability to.
 *
 * @return Nonzero if reachable, 0 if not.
 */
int uct_iface_is_reachable(uct_iface_h iface, const struct sockaddr *addr);


/**
 * @ingroup RESOURCE
 */
ucs_status_t uct_iface_mem_alloc(uct_iface_h iface, size_t length,
                                 const char *name, uct_allocated_memory_t *mem);

void uct_iface_mem_free(const uct_allocated_memory_t *mem);


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
 * @brief Create an endpoint which is connected to remote interface.
 *
 * @param [in]  iface   Interface to create the endpoint on.
 * @param [in]  addr    Remote interface address to connect to.
 * @param [out] ep_p    Filled with handle to the new endpoint.
 *
 *
 */
static inline ucs_status_t uct_ep_create_connected(uct_iface_h iface,
                                                   const struct sockaddr *addr,
                                                   uct_ep_h* ep_p)
{
    return iface->ops.ep_create_connected(iface, addr, ep_p);
}


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
ucs_status_t uct_ep_get_address(uct_ep_h ep, struct sockaddr *addr);


/**
 * @ingroup RESOURCE
 * @brief Connect endpoint to a remote endpoint.
 *
 * @param [in] ep           Endpoint to connect.
 * @param [in] iface_addr   Remote interface address.
 * @param [in] ep_addr      Remote endpoint address.
 */
ucs_status_t uct_ep_connect_to_ep(uct_ep_h ep, const struct sockaddr *addr);


/**
 * @ingroup PD
 * @brief Query for protection domain attributes. *
 *
 * @param [in]  pd       Protection domain to query.
 * @param [out] pd_attr  Filled with protection domain attributes.
 */
ucs_status_t uct_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr);


/**
 * @ingroup PD
 * @brief Allocate memory for zero-copy sends and remote access.
 *
 *  Allocate memory on the protection domain. In order to use this function, PD
 * must support @ref UCT_PD_FLAG_ALLOC flag.
 *
 * @param [in]     pd          Protection domain to allocate memory on.
 * @param [in,out] length_p    Points to the size of memory to allocate. Upon successful
 *                              return, filled with the actual size that was allocated,
 *                              which may be larger than the one requested. Must be >0.
 * @param [in]     name        Name of the allocated region, used to track memory
 *                              usage for debugging and profiling.
 * @param [out]    memh_p      Filled with handle for allocated region.
 */
ucs_status_t uct_pd_mem_alloc(uct_pd_h pd, size_t *length_p, void **address_p,
                              const char *name, uct_mem_h *memh_p);

/**
 * @ingroup PD
 * @brief Release memory allocated by @ref uct_pd_mem_alloc.
 *
 * @param [in]     pd          Protection domain memory was allocateed on.
 * @param [in]     memh        Memory handle, as returned from @ref uct_pd_mem_alloc.
 */
ucs_status_t uct_pd_mem_free(uct_pd_h pd, uct_mem_h memh);


/**
 * @ingroup PD
 * @brief Register memory for zero-copy sends and remote access.
 *
 *  Register memory on the protection domain. In order to use this function, PD
 * must support @ref UCT_PD_FLAG_REG flag.
 *
 * @param [in]     pd        Protection domain to register memory on.
 * @param [out]    address   Memory to register.
 * @param [in]     length    Size of memory to register. Must be >0.
 * @param [out]    memh_p    Filled with handle for allocated region.
 */
ucs_status_t uct_pd_mem_reg(uct_pd_h pd, void *address, size_t length,
                            uct_mem_h *memh_p);


/**
 * @ingroup PD
 * @brief Undo the operation of @ref uct_pd_mem_reg().
 *
 * @param [in]  pd          Protection domain which was used to register the memory.
 * @paran [in]  memh        Local access key to memory region.
 */
ucs_status_t uct_pd_mem_dereg(uct_pd_h pd, uct_mem_h memh);


/**
 * @ingroup PD
 * @brief Allocate memory for zero-copy communications and remote access.
 *
 * Allocate potentially registered memory. Every one of the provided allocation
 * methods will be used, in turn, to perform the allocation, until one succeeds.
 *  Whenever the PD method is encountered, every one of the provided PDs will be
 * used, in turn, to allocate the memory, until one succeeds, or they are
 * exhausted. In this case the next allocation method from the initial list will
 * be attempted.
 *
 * @param [in]     min_length  Minimal size to allocate. The actual size may be
 *                             larger, for example because of alignment restrictions.
 * @param [in]     methods     Array of memory allocation methods to attempt.
 * @param [in]     num_method  Length of 'methods' array.
 * @param [in]     pds         Array of protection domains to attempt to allocate
 *                             the memory with, for PD allocation method.
 * @param [in]     num_pds     Length of 'pds' array. May be empty, in such case
 *                             'pds' may be NULL, and PD allocation method will
 *                             be skipped.
 * @param [in]     name        Name of the allocation. Used for memory statistics.
 * @param [out]    mem         In case of success, filled with information about
 *                              the allocated memory. @ref uct_allocated_memory_t.
 */
ucs_status_t uct_mem_alloc(size_t min_length, uct_alloc_method_t *methods,
                           unsigned num_methods, uct_pd_h *pds, unsigned num_pds,
                           const char *name, uct_allocated_memory_t *mem);


/**
 * @ingroup PD
 * @brief Release allocated memory.
 *
 * Release the memory allocated by @ref uct_mem_alloc.
 *
 * @param [in]  mem         Description of allocated memory, as returned from
 *                          @ref uct_mem_alloc.
 */
ucs_status_t uct_mem_free(const uct_allocated_memory_t *mem);


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
ucs_status_t uct_pd_mkey_pack(uct_pd_h pd, uct_mem_h memh, void *rkey_buffer);


/**
 * @ingroup PD
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
 * @ingroup PD
 *
 * @brief Release a remote key.
 *
 * @param [in]  rkey_ob      Remote key to release.
 */
void uct_rkey_release(const uct_rkey_bundle_t *rkey_ob);


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
 *
 * @param [in]  desc         Descriptor to release.
 */
UCT_INLINE_API void uct_iface_release_am_desc(void *desc)
{
    uct_iface_h iface = uct_recv_desc_iface(desc);
    iface->ops.iface_release_am_desc(iface, desc);
}


/**
 * @ingroup RMA
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_put_short(uct_ep_h ep, const void *buffer, unsigned length,
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
UCT_INLINE_API ucs_status_t uct_ep_put_zcopy(uct_ep_h ep, const void *buffer, size_t length,
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
UCT_INLINE_API ucs_status_t uct_ep_get_bcopy(uct_ep_h ep, uct_unpack_callback_t unpack_cb,
                                             void *arg, size_t length,
                                             uint64_t remote_addr, uct_rkey_t rkey,
                                             uct_completion_t *comp)
{
    return ep->iface->ops.ep_get_bcopy(ep, unpack_cb, arg, length, remote_addr,
                                       rkey, comp);
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
                                            const void *payload, unsigned length)
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
                                            unsigned header_length, const void *payload,
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
                                                 uint64_t *result, uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_fadd64(ep, add, remote_addr, rkey, result, comp);
}


/**
 * @ingroup AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_swap64(uct_ep_h ep, uint64_t swap,
                                                 uint64_t remote_addr, uct_rkey_t rkey,
                                                 uint64_t *result, uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_swap64(ep, swap, remote_addr, rkey, result, comp);
}


/**
 * @ingroup AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_cswap64(uct_ep_h ep, uint64_t compare, uint64_t swap,
                                                  uint64_t remote_addr, uct_rkey_t rkey,
                                                  uint64_t *result, uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_cswap64(ep, compare, swap, remote_addr, rkey, result, comp);
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
                                                 uint32_t *result, uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_fadd32(ep, add, remote_addr, rkey, result, comp);
}


/**
 * @ingroup AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_swap32(uct_ep_h ep, uint32_t swap,
                                                 uint64_t remote_addr, uct_rkey_t rkey,
                                                 uint32_t *result, uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_swap32(ep, swap, remote_addr, rkey, result, comp);
}


/**
 * @ingroup AMO
 * @brief
 */
UCT_INLINE_API ucs_status_t uct_ep_atomic_cswap32(uct_ep_h ep, uint32_t compare, uint32_t swap,
                                                  uint64_t remote_addr, uct_rkey_t rkey,
                                                  uint32_t *result, uct_completion_t *comp)
{
    return ep->iface->ops.ep_atomic_cswap32(ep, compare, swap, remote_addr, rkey, result, comp);
}


/**
 * @ingroup RESOURCE
 * @brief Request a notification when more send resources are available on this
 *        endpoint.
 *
 *  This function can be used when send operations fail with ERR_NO_RESOURCE, and
 * the user would like to know when send resources become available (opposed to
 * repeatedly calling progress/send until it returns OK).
 *
 * @param ep    Endpoint to request notification on.
 * @param cb    Completion callback, which would be invoked (once) when more
 *              resources become available.
 */
UCT_INLINE_API ucs_status_t uct_ep_req_notify(uct_ep_h ep, ucs_callback_t *cb)
{
    return ep->iface->ops.ep_req_notify(ep, cb);
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
