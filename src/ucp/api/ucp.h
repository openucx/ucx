/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCP_H_
#define UCP_H_

#include <ucp/api/ucp_def.h>
#include <ucs/type/thread_mode.h>
#include <ucs/config/types.h>
#include <ucs/sys/math.h>
#include <stdio.h>


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
 * @brief UCP configuration features
 *
 * The enumeration list describes the features supported by UCP.  An
 * application can request the features using @ref ucp_params_t "UCP parameters"
 * during @ref ucp_init "UCP initialization" process.
 */
enum ucp_feature {
    UCP_FEATURE_TAG   = UCS_BIT(0),  /**< Request tag matching support */
    UCP_FEATURE_RMA   = UCS_BIT(1),  /**< Request remote memory
                                          access support */
    UCP_FEATURE_AMO32 = UCS_BIT(2),  /**< Request 32-bit atomic
                                          operations support */
    UCP_FEATURE_AMO64 = UCS_BIT(3)   /**< Request 64-bit atomic
                                          operations support */
};


/**
 * @ingroup UCP_DATATYPE
 * @brief UCP data type classification
 *
 * The enumeration list describes the data-types supported by UCP.
 */
enum ucp_dt_type {
    UCP_DATATYPE_CONTIG  = 0,      /**< Contiguous datatype */
    UCP_DATATYPE_STRIDED = 1,      /**< Strided data-type */
    UCP_DATATYPE_GENERIC = 7,      /**< Generic data-type with
                                        user-defined pack/unpack routines */
    UCP_DATATYPE_SHIFT   = 3,      /**< Number of bits defining
                                        the data-type classification */
    UCP_DATATYPE_CLASS_MASK = UCS_MASK(UCP_DATATYPE_SHIFT) /**< Data-type class
                                                                mask */
};


/**
 * @ingroup UCP_DATATYPE
 * @brief Generate an identifier for contiguous data type.
 *
 * This macro creates an identifier for contiguous data-type that is defined by
 * the size of the basic element.
 *
 * @param [in]  _elem_size    Size of the basic element of the type.
 *
 * @return Data-type identifier.
 */
#define ucp_dt_make_contig(_elem_size) \
    (((ucp_datatype_t)(_elem_size) << UCP_DATATYPE_SHIFT) | UCP_DATATYPE_CONTIG)


/**
 * @ingroup UCP_DATATYPE
 * @brief UCP generic data type descriptor
 *
 * This structure provides a generic datatype descriptor that
 * is used for definition of application defined datatypes.

 * Typically, the descriptor is used for an integratoion with data-type
 * engines implemented within MPI and SHMEM implementations.
 *
 */
typedef struct ucp_generic_dt_ops {

    /**
     * @ingroup UCP_DATATYPE
     * @brief Start a packing request.
     *
     * The pointer refers to application defined start-to-pack routine.
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
     * The pointer refers to application defined start-to-unpack routine.
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
     * UCP @ref ucp_feature "features" that are used for library
     * initialization.  It is recommend for applications only request
     * the features that are required for an optimal functionality
     */
    uint64_t                    features;
    /**
     * The size of a reserved space in a non-blocking requests. Typically
     * applications use the this space for caching own structures in order
     * avoid costly memory allocations, pointer dereferences, and cache misses.
     * For example, MPI implementation can use this memory for caching MPI
     * descriptors
     */
    size_t                      request_size;
    /**
     * Pointer to a routine that is used for the request initialization.
     * @e NULL can be used if no such function required.
     */
    ucp_request_init_callback_t request_init;
    /**
     * Pointer to a routine that is responsible for cleanup the memory
     * associated with the request.  @e NULL can be used if no such function
     * required.
     */
    ucp_request_cleanup_callback_t request_cleanup;
} ucp_params_t;


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
 * @todo This structure will be converted to an opaque object that will
 * be modified through a dedicated function.
 */
typedef struct ucp_config {
    /** Array of device names to use */
    UCS_CONFIG_STRING_ARRAY_FIELD(names)   devices;
    /** Array of transport names to use */
    UCS_CONFIG_STRING_ARRAY_FIELD(names)   tls;
    /** Whether to force using of all available devices */
    int                                    force_all_devices;
    /** Array of memory allocation methods */
    UCS_CONFIG_STRING_ARRAY_FIELD(methods) alloc_prio;
    /** Threshold for switching UCP to buffered copy(bcopy) protocol */
    size_t                                 bcopy_thresh;
    /** Threshold for switching UCP to rendezvous protocol */
    size_t                                 rndv_thresh;
} ucp_config_t;


/**
 * @ingroup UCP_CONTEXT
 * @brief Completion status of a tag-matched receive.
 *
 * @todo This declaration should be removed from public API
 */
typedef struct ucp_tag_recv_completion {
    ucp_tag_t             sender_tag;  /**< Full sender tag */
    size_t                rcvd_len;    /**< How much data was received */
} ucp_tag_recv_completion_t;


/**
 * @ingroup UCP_CONTEXT
 * @brief UCP receive information descriptor
 *
 * The UCP receive information descriptor is allocated by application and filled
 * in with the information about the received message by @ref ucp_tag_probe_nb
 * "ucp_tag_probe_nb" routine.
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
 * the application is responsible to @ref ucp_config_release "release" the
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
 * @brief UCP context initialization.
 *
 * This routine creates and initializes a @ref ucp_context_t
 * "UCP application context".
 *
 * @warning This routine must be called before any other UCP function
 * call in the application.
 *
 * This routine discovers the available network interfaces, and initializes the
 * network resources required for discovering of the network and memory
 * related devices.  This routine is responsible for initialization all
 * information required for a particular application scope, for example, MPI
 * application, OpenSHMEM application, etc.
 *
 * @note
 * @li Higher level protocols can add additional communication isolation, as
 * MPI does with it’s communicator object. A single communication context may
 * be used to support multiple MPI communicators.
 * @li The context can be used to isolate the communication that corresponds to
 * different protocols. For example, if MPI and OpenSHMEM are using UCP to
 * isolate the MPI communication from the OpenSHMEM communication, users should
 * use different application context for each of the communication libraries.
 *
 * @param [in]  config        UCP configuration descriptor allocated through
 *                            @ref ucp_config_read "ucp_config_read()" routine.
 * @param [in]  params        User defined @ref ucp_params_t "tunings" for the
 *                            @ref ucp_context_h "UCP application context".
 * @param [out] context_p     Initialized @ref ucp_context_h
 *                            "UCP application context".
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_init(const ucp_params_t *params, const ucp_config_t *config,
                      ucp_context_h *context_p);


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
 * @param [in] context_p   Handle to @ref ucp_context_h
 *                         "UCP application context".
 * @param [in] thread_mode Thread safety @ref ucs_thread_mode_t "mode" for
 *                         the worker object and resources associated with it.
 * @param [out] worker_p   A pointer to the worker object allocated by the
 *                         UCP library
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_worker_create(ucp_context_h context, ucs_thread_mode_t thread_mode,
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

/** @cond PRIVATE_INTERFACE */
/* This set of functions is expected to be removed */
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
/** @endcond */

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
 *
 * @param [in]  worker    Worker to progress.
 */
void ucp_worker_progress(ucp_worker_h worker);


/**
 * @ingroup UCP_ENDPOINT
 * @brief Create and connect an endpoint.
 *
 * This routine creates and connects an @ref ucp_ep_h "endpoint" on a @ref
 * ucp_worker_h "local worker" for a destination @ref ucp_address_t "address"
 * that identifies the remote @ref ucp_worker_h "worker".  This function is
 * non-blocking, and communications may begin immediately after it returns. If
 * the connection process is not completed, communications may be delayed.
 * The created @ref ucp_ep_h "endpoint" is associated with one and only one
 * @ref ucp_worker_t "worker".
 *
 * @param [in]  worker      Handle to the worker; the endpoint
 *                          is associated with the worker.
 * @param [in]  address     Destination address; the address must be obtained
 *                          using @ref ucp_worker_get_address
 *                          "ucp_worker_get_address()" routine.
 * @param [out] ep_p        A handle to the created endpoint.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_ep_create(ucp_worker_h worker, ucp_address_t *address,
                           ucp_ep_h *ep_p);


/**
 * @ingroup UCP_ENDPOINT
 * @brief Destroy and release the endpoint.
 *
 * This routine releases an @ref ucp_ep_h "endpoint". The release process
 * flushes, locally, all outstanding communication operations and releases all
 * memory context associated with the endpoints.  Once the endpoint is
 * destroyed, application cannot access it anymore.  Nevertheless, if the
 * application is interested, to re-initiate communication with a particular
 * endpoint it can use @ref ucp_ep_create "endpoints create routine" to
 * re-open the endpoint.
 *
 * @param [in]  ep   Handle to the remote endpoint.
 */
void ucp_ep_destroy(ucp_ep_h ep);


/**
 * @ingroup UCP_MEM
 * @brief Map or allocate memory for zero-copy operations.
 *
 * This routine maps or/and allocates a user-specified memory segment with @ref
 * ucp_context_h "UCP application context" and the network resources associated
 * with it. If the application specifies NULL as an address for the memory
 * segment, the routine allocates and maps the memory segment.  The network
 * stack associated with an application context can typically send and receive
 * data from the mapped memory without CPU intervention; some devices and
 * associated network stacks require the memory to be mapped to send and
 * receive data. The @ref ucp_mem_h "memory handle" includes all information
 * required to access the memory locally using UCP routines, , while @ref
 * ucp_rkey_h "remote registration handle" provides an information that is
 * necessary for remote memory access.
 *
 * @note
 * Another well know terminology for the "map" operation that is typically
 * used in the context of networking is memory "registration". The UCP library
 * registers the memory the available hardware so it can be assessed directly
 * by the hardware.
 *
 * Memory mapping assumptions:
 * @li A given memory segment can be mapped by several different communication
 * stacks, if these are compatible.
 * @li The pointer returned may be used with any sub-region of the mapped
 * memory.
 * @li If a large segment is registered, and then segmented for subsequent use
 * by a user, then the user is responsible for segmentation and subsequent
 * management.

 * @param [in]     context    Application @ref ucp_context_h "context" to map
 *                            (register) and allocate the memory on.
 * @param [out]    address_p  If the pointer to the address is not NULL,
 *                            the routine maps (registers) the memory segment.
 *                            if the pointer is NULL, the library allocates
 *                            mapped (registered) memory segment.
 * @param [in]     length     Length (in bytes) to allocate.
 * @param [in]     flags      Allocation flags (currently reserved - set to 0).
 * @param [out]    memh_p     UCP @ref ucp_mem_h "handle" for the allocated
 *                            segment.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_mem_map(ucp_context_h context, void **address_p, size_t length,
                         unsigned flags, ucp_mem_h *memh_p);


/**
 * @ingroup UCP_MEM
 * @brief Unmap memory segment
 *
 * This routine unmaps a user specified memory segment, that was previously
 * mapped using the @ref ucp_mem_map "ucp_mem_map()" routine.  The unmap
 * routine will also release the resources associated with the memory
 * @ucp_mem_h "handle".  When the function returns, the @ref ucp_mem_h will be
 * invalid and cannot be used with any UCP routine.
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
 * @paran [in]  memh        @ref ucp_mem_h "Handle" to memory region.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_mem_unmap(ucp_context_h context, ucp_mem_h memh);


/**
 * @ingroup UCP_MEM
 * @brief Pack memory region remote access key.
 *
 * This routine allocates memory buffer and packs into the buffer
 * a remote access key (RKEY) object. RKEY is an opaque object that provides
 * the information that is necessary for remote memory access.
 * This routine packs the RKEY object in a portable format such that the
 * object can be @ucp_rkey_unpack "unpacked" on any platform supported by the
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
 * @paran [in]  memh          @ref ucp_mem_h "Handle" to memory region.
 * @param [out] rkey_buffer   Memory buffer allocated by the library.
 *                            The buffer contains packed RKEY.
 * @param [out] size          Size (in bytes) of the packed RKEY.
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
 * @param [out] rkey          Remote key handle.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_ep_rkey_unpack(ucp_ep_h ep, void *rkey_buffer, ucp_rkey_h *rkey_p);


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
 * @ref ucp_rkey_unpack "ucp_rkey_unpack()" routine the behaviour of this
 * routine is undefined.
 *
 * @param [in]  rkey         Romote key to destroy.
 */
void ucp_rkey_destroy(ucp_rkey_h rkey);


/**
 * @ingroup UCP_MEM
 * @brief If possible translate remote address into local address which can be
 *        accessed using direct load and store operations.
 *
 * This routine returns a local memory address for the remote address such that
 * application can use the local address for direct memory load and store
 * operations. If the underlaying hardware does not support this capability
 * this routine will return a corresponding error.
 *
 * @param [in]  ep              Endpoint handle that was used for rkey object
 *                              creation and is used for the remote memory address.
 * @param [in]  remote_addr     Remote address to translate.
 * @param [out] rkey            Remote key handle for the remote address.
 * @param [out] local_addr      Local memory address that can by accessed
 *                              directly using memory load and store operations.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_rmem_ptr(ucp_ep_h ep, void *remote_addr, ucp_rkey_h rkey,
                          void **local_addr_p);


/**
 * @ingroup UCP_COMM
 * @brief Non-blocking tagged-send operations
 *
 * Non-blocking tag send. The function returns immediately, however the actual
 * send may be delayed.
 *
 * @param [in]  ep          Destination endpoint to send to.
 * @param [in]  buffer      Message payload to send.
 * @param [in]  count       Number of elements in the buffer.
 * @param [in]  datatype    Type of elements in the buffer.
 * @param [in]  tag         Message tag to send.
 * @param [in]  cb          Callback function which is called when the
 *                          send is completed (buffer can be reused), in case
 *                          the return value is a request handle.
 *
 * @return NULL/UCS_OK          The send is completed completed in place.
 * @return UCS_PTR_IS_ERR(_ptr) Error during send.
 * @return otherwise            A request handle. the handle should be released
 *                              by calling ucp_request_release().
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
 * @brief Non-blocking probe and return a message.
 *
 *  Probe for incoming unexpected message. The function returns immediately and
 * possibly returns a handle to the message.
 *
 * @param [in]  worker      UCP worker.
 * @param [in]  tag         Message tag to expect.
 * @param [in]  tag_mask    Mask of which bits to match from the incoming tag
 *                           against the expected tag.
 * @param [in]  remove      Whether to remove the probed message from unexpected
 *                           queue (true), or keep it there (false).
 * @param [out] info        Filled with details about the received message.
 *
 * @return NULL       - No match found.
 *         otherwise  - A message handle. If remove==1, the handle should be
 *                      passed to ucp_tag_msg_recv_nb().
 */
ucp_tag_message_h ucp_tag_probe_nb(ucp_worker_h worker, ucp_tag_t tag,
                                   ucp_tag_t tag_mask, int remove,
                                   ucp_tag_recv_info_t *info);


/**
 * @ingroup UCP_COMM
 * @brief Receive probed message in a non-blocking fashion.
 *
 *  Non-blocking probed message receive. The function returns immediately,
 * however the actual receive may occur later.
 *
 * @param [in]  worker      UCP worker.
 * @param [in]  buffer      Buffer to receive the data to.
 * @param [in]  count       Number of elements in the buffer.
 * @param [in]  datatype    Type of elements in the buffer.
 * @param [in]  message     Handle to probed message.
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
 * @brief Blocking remote memory get operation.
 *
 * This routine loads contiguous block of data that is described by the remote
 * address @a remote_addr and the @ref ucp_rkey_h "memory handle" @a rkey in
 * the local contiguous memory region described by @a buffer address.  The
 * routine returns when remote data is loaded and stored under the local address
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
ucs_status_t ucp_get(ucp_ep_h ep, void *buffer, size_t length,
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
 * local memory @a result, and the operand value (@a swap) is stored in remote memory.
 * The call to the routine returns when the operation is completed and the
 * @a result value is updated.
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
 * local memory @a result, and the operand value (@a swap) is stored in remote memory.
 * The call to the routine returns when the operation is completed and the
 * @a result value is updated.
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
 * @brief Create a generic datatype.
 *
 * This routine create a generic datatype object.
 *
 * @param [in]  ops          Generic datatype function table as defined by
 *                           @ref ucp_generic_dt_ops_t .
 * @param [in]  context      Application defined context passed to this routine.
 * @param [out] datatype_p   A pointer to datatype object.
 */
ucs_status_t ucp_dt_create_generic(ucp_generic_dt_ops_t *ops, void *context,
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
 * subsequent communication operations to the same @ucp_worker_h "worker" which
 * follow the call to @ref ucp_worker_fence "fence".
 *
 * @note The primary diference between @ref ucp_worker_fence "ucp_worker_fence()"
 * and the @ref ucp_worker_flush "ucp_worker_flush()" is the fact the fence
 * routine does not gurantee completion of the operations on the call return but
 * only ensures the order between communication operations. The
 * @ref ucp_worker_flush "flush" operation on return grantees that all
 * operation are completed and corresponding memory regions were updated.
 *
 * @param [in] worker        UCP worker.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_worker_fence(ucp_worker_h worker);


/**
 * @ingroup UCP_WORKER
 *
 * @brief Flush all outstanding communication on the @ref ucp_worker_h "worker"
 *
 * This routine flushes all outstanding communication on the @ref ucp_worker_h
 * "worker".  Communication operations issued by on the @a worker prior to this
 * call will have completed both at the origin and at the target @ref
 * ucp_worker_h "worker" when this call returns.
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


#endif
