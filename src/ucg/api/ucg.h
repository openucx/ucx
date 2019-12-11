/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCG_H_
#define UCG_H_

#include <ucg/api/ucg_def.h>
#include <ucg/api/ucg_version.h>

#include <ucp/api/ucp.h>

BEGIN_C_DECLS

/**
 * @defgroup UCG_API Unified Communication Protocol (UCG) API
 * @{
 * This section describes UCG API.
 * @}
 */


/**
 * @defgroup UCG_CONTEXT UCG Application Context
 * @ingroup UCG_API
 * @{
 * Application context is a primary concept of UCG design which
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
 * @defgroup UCG_GROUP UCG Group
 * @ingroup UCG_API
 * @{
 * UCG Group routines
 * @}
 */


/*
 * Since UCG works on top of UCP, most of the functionality overlaps. For API
 * completeness, UCG presents a full-featured API with the "ucg_" prefix.
 */
#define ucg_context_h               ucp_context_h
#define ucg_config_t                ucp_config_t
#define ucg_address_t               ucp_address_t
#define ucg_worker_h                ucp_worker_h
#define ucg_params_t                ucp_params_t
#define ucg_context_attr_t          ucp_context_attr_t
#define ucg_worker_attr_t           ucp_worker_attr_t
#define ucg_worker_params_t         ucp_worker_params_t
#define ucg_config_read             ucp_config_read
#define ucg_config_release          ucp_config_release
#define ucg_config_modify           ucp_config_modify
#define ucg_config_print            ucp_config_print
#define ucg_get_version             ucp_get_version
#define ucg_get_version_string      ucp_get_version_string
#define ucg_cleanup                 ucp_cleanup
#define ucg_context_query           ucp_context_query
#define ucg_context_print_info      ucp_context_print_info
#define ucg_worker_create           ucp_worker_create
#define ucg_worker_destroy          ucp_worker_destroy
#define ucg_worker_query            ucp_worker_query
#define ucg_worker_print_info       ucp_worker_print_info
#define ucg_worker_get_address      ucp_worker_get_address
#define ucg_worker_release_address  ucp_worker_release_address


/**
 * @ingroup UCG_GROUP
 * @brief UCG group collective operation characteristics.
 *
 * The enumeration allows specifying modifiers to describe the requested
 * collective operation, as part of @ref ucg_collective_params_t
 * passed to @ref ucg_collective_start . For example, for MPI_Reduce:
 *
 * modifiers = UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE |
 *             UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_DESTINATION;
 *
 * The premise is that (a) any collective type can be described as a combination
 * of the flags below, and (b) the implementation can benefit from applying
 * logic based on these flags. For example, we can check if a collective has
 * a single rank as the source, which will be true for both MPI_Bcast and
 * MPI_Scatterv today, and potentially other types in the future.
 *
 * @note
 * For simplicity, some rarely used collectives were intentionally omitted. For
 * instance, MPI_Scan and MPI_Exscan could be supported using additional flags,
 * which are not part of the API at this time.
 */
enum ucg_collective_modifiers {
    /* Network Pattern Considerations */
    UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_SOURCE      = UCS_BIT( 0), /* otherwise from all */
    UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_DESTINATION = UCS_BIT( 1), /* otherwise to all */
    UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE          = UCS_BIT( 2), /* otherwise gather */
    UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST          = UCS_BIT( 3), /* otherwise scatter */
    UCG_GROUP_COLLECTIVE_MODIFIER_VARIABLE_LENGTH    = UCS_BIT( 4), /* otherwise fixed length */
    UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE_PARTIAL  = UCS_BIT( 5), /* MPI_Scan */
    UCG_GROUP_COLLECTIVE_MODIFIER_NEIGHBOR           = UCS_BIT( 6), /* Neighbor collectives */

    /* Buffer/Data Management Considerations */
    UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE_STABLE   = UCS_BIT( 7), /* stable reduction */
    UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE_EXCLUDE  = UCS_BIT( 8), /* MPI_Exscan */
    UCG_GROUP_COLLECTIVE_MODIFIER_IN_PLACE           = UCS_BIT( 9), /* otherwise two buffers */
    UCG_GROUP_COLLECTIVE_MODIFIER_VARIABLE_DATATYPE  = UCS_BIT(10), /* otherwise fixed data-type */
    UCG_GROUP_COLLECTIVE_MODIFIER_PERSISTENT         = UCS_BIT(11), /* otherwise destroy coll_h */
    UCG_GROUP_COLLECTIVE_MODIFIER_BARRIER            = UCS_BIT(12), /* prevent others from starting */

    /* This mask indicates the extent required to store the modifiers,
     * which must also match the */
    UCG_GROUP_COLLECTIVE_MODIFIER_MASK               = UCS_MASK(16)
};

/**
 * @ingroup UCG_GROUP
 * @brief UCG group collective operation description.
 *
 * Some collective opertions have one special rank. For example MPI_Bcast has
 * the root of the broadcast, and MPI_Reduce has the root where the final result
 * must be written. The "root" field is used in cases where "modifiers" includes:
 *   (a) UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_SOURCE
 *   (b) UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_DESTINATION
 * In other cases, the "root" field is ignored.
 */
typedef struct ucg_collective_type {
    enum ucg_collective_modifiers modifiers :16; /* Collective description, using
                                                    @ref ucg_collective_modifiers */
    ucg_group_member_index_t      root :48;      /* Root rank, if applicable */
} ucg_collective_type_t;

/**
 * @ingroup UCG_GROUP
 * @brief UCG group member distance.
 *
 * During group creation, the caller can pass information about the distance of
 * each other member of the group. This information may be used to select the
 * best logical topology for collective operations inside UCG.
 */
enum ucg_group_member_distance {
    UCG_GROUP_MEMBER_DISTANCE_SELF   = 0, /* This is the calling member */
    /* Reserved for in-socket proximity values */
    UCG_GROUP_MEMBER_DISTANCE_SOCKET = UCS_MASK(3), /* member is on the same socket */
    /* Reserved for in-host proximity values */
    UCG_GROUP_MEMBER_DISTANCE_HOST   = UCS_MASK(4), /* member is on the same host */
    /* Reserved for network proximity values */
    UCG_GROUP_MEMBER_DISTANCE_NET    = UCS_MASK(8) - 1, /* member is on the network */

    UCG_GROUP_MEMBER_DISTANCE_LAST   = UCS_MASK(8)
} UCS_S_PACKED;

/**
 * @ingroup UCG_GROUP
 * @brief UCG group parameters field mask.
 *
 * The enumeration allows specifying which fields in @ref ucg_group_params_t are
 * present. It is used to enable backward compatibility support.
 */
enum ucg_group_params_field {
    UCG_GROUP_PARAM_FIELD_MEMBER_COUNT = UCS_BIT(0), /**< Number of members */
    UCG_GROUP_PARAM_FIELD_MEMBER_INDEX = UCS_BIT(1), /**< My member index */
    UCG_GROUP_PARAM_FIELD_DISTANCES    = UCS_BIT(2), /**< Member distance array */
    UCG_GROUP_PARAM_FIELD_REDUCE_CB    = UCS_BIT(3), /**< Callback for reduce ops */
    UCG_GROUP_PARAM_FIELD_RESOLVER_CB  = UCS_BIT(4)  /**< Callback for address
                                                          resolution/destruction */
};

/**
 * @ingroup UCG_GROUP
 * @brief Creation parameters for the UCG group.
 *
 * The structure defines the parameters that are used during the UCG group
 * @ref ucg_group_create "creation".
 */
typedef struct ucg_group_params {
    /**
     * Mask of valid fields in this structure, using bits from @ref ucg_group_params_field.
     * Fields not specified in this mask will be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t field_mask;

    ucg_group_member_index_t member_count; /* Number of group members */
    ucg_group_member_index_t member_index; /* My member index within the group */

    /*
     * This array contains information about the process placement of different
     * group members, which is used to select the best topology for collectives.
     *
     *
     * For example, for 2 nodes, 3 sockets each, 4 cores per socket, each member
     * should be passed the distance array contents as follows:
     *   1st group member distance array:  0111222222223333333333333333
     *   2nd group member distance array:  1011222222223333333333333333
     *   3rd group member distance array:  1101222222223333333333333333
     *   4th group member distance array:  1110222222223333333333333333
     *   5th group member distance array:  2222011122223333333333333333
     *   6th group member distance array:  2222101122223333333333333333
     *   7th group member distance array:  2222110122223333333333333333
     *   8th group member distance array:  2222111022223333333333333333
     *    ...
     *   12th group member distance array: 3333333333333333011122222222
     *   13th group member distance array: 3333333333333333101122222222
     *    ...
     */
    enum ucg_group_member_distance *distance;

    /* MPI passes its own reduction function, used for complex data-types */
    void (*mpi_reduce_f)(void *mpi_op, char *src, char *dst, unsigned count, void *mpi_dtype);

    /* Callback function for connection establishment */
    ucs_status_t (*resolve_address_f)(void *cb_group_obj, ucg_group_member_index_t index,
                                      ucg_address_t **addr, size_t *addr_len);
    void         (*release_address_f)(ucg_address_t *addr);

    void *cb_group_obj;  /* external group object for call-backs (MPI_Comm) */
} ucg_group_params_t;


/**
 * @ingroup UCG_GROUP
 * @brief Creation parameters for the UCG collective operation.
 *
 * The structure defines the parameters that are used during the UCG collective
 * @ref ucg_collective_create "creation". The size of this structure is critical
 * to performance, as well as it being contiguous, because its entire contents
 * are accessed during run-time.
 */
typedef struct ucg_collective {
    ucg_collective_type_t     type;    /* the type (and root) of the collective */

    struct {
        void                 *buf;     /* buffer location to use */
        union {
            int               count;   /* item count */
            int              *counts;  /* item count array */
        };
        union {
            size_t            dt_len;  /* external datatype length */
            size_t           *dts_len; /* external datatype length array */
        };
        union {
            void             *dt_ext;  /* external datatype context */
            void             *dts_ext; /* external datatype context array */
        };
        union {
            size_t            stride;  /* item stride */
            int              *displs;  /* item displacement array */
            void             *op_ext;  /* external reduce operation handle */
        };
    } send, recv;

    ucg_collective_callback_t comp_cb; /* completion callback */
} ucg_collective_params_t;


/**
 * @ingroup UCG_GROUP
 * @brief Create a group object.
 *
 * This routine allocates and initializes a @ref ucg_group_h "group" object.
 * This routine is a "collective operation", meaning it has to be called for
 * each worker participating in the group - before the first call on the group
 * is invoked on any of those workers. The call does not contain a barrier,
 * meaning a call on one worker can complete regardless of call on others.
 *
 * @note The group object is allocated within context of the calling thread
 *
 * @param [in] worker      Worker to create a group on top of.
 * @param [in] params      User defined @ref ucg_group_params_t configurations for the
 *                         @ref ucg_group_h "UCG group".
 * @param [out] group_p    A pointer to the group object allocated by the
 *                         UCG library
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucg_group_create(ucg_worker_h worker,
                              const ucg_group_params_t *params,
                              ucg_group_h *group_p);


/**
 * @ingroup UCG_GROUP
 * @brief Destroy a group object.
 *
 * This routine releases the resources associated with a @ref ucg_group_h
 * "UCG group". This routine is also a "collective operation", similarly to
 * @ref ucg_group_create, meaning it must be called on each worker participating
 * in the group.
 *
 * @warning Once the UCG group handle is destroyed, it cannot be used with any
 * UCG routine.
 *
 * The destroy process releases and shuts down all resources associated with
 * the @ref ucg_group_h "group".
 *
 * @param [in]  group       Group object to destroy.
 */
void ucg_group_destroy(ucg_group_h group);


/**
 * @ingroup UCG_GROUP
 * @brief Progresses a Group object.
 *
 * @param [in]  group       Group object to progress.
 */
unsigned ucg_group_progress(ucg_group_h group);


/**
 * @ingroup UCG_GROUP
 * @brief Progresses a Worker object with the groups (UCG) extension.
 *
 * @param [in]  group       Group object to progress.
 */
unsigned ucg_worker_progress(ucg_worker_h worker);


/**
 * @ingroup UCG_GROUP
 * @brief Creates a collective operation on a group object.
 * The parameters are intentionally non-constant, to allow UCG to write-back some
 * information and avoid redundant actions on the next call. For example, memory
 * registration handles are written back to the parameters pointer passed to the
 * function, and are re-used in subsequent calls.
 *
 * @param [in]  group       Group object to use.
 * @param [in]  params      Collective operation parameters.
 * @param [out] coll        Collective operation handle.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucg_collective_create(ucg_group_h group,
                                   const ucg_collective_params_t *params,
                                   ucg_coll_h *coll);


/**
 * @ingroup UCG_GROUP
 * @brief Starts a collective operation.
 *
 * @param [in]  coll        Collective operation handle.
 *
 * @return UCS_OK           - The collective operation was completed immediately.
 * @return UCS_PTR_IS_ERR(_ptr) - The collective operation failed.
 * @return otherwise        - Operation was scheduled for send and can be
 *                          completed in any point in time. The request handle
 *                          is returned to the application in order to track
 *                          progress of the message. The application is
 *                          responsible to release the handle using
 *                          @ref ucg_request_free routine.
 */
ucs_status_ptr_t ucg_collective_start_nb(ucg_coll_h coll);


/**
 * @ingroup UCG_GROUP
 * @brief Starts a collective operation.
 *
 * @param [in]  coll        Collective operation handle.
 * @param [in]  req         Request handle allocated by the user. There should
 *                          be at least UCG request size bytes of available
 *                          space before the @a req. The size of UCG request
 *                          can be obtained by @ref ucg_context_query function.
 *
 * @return UCS_OK           - The collective operation was completed immediately.
 * @return UCS_INPROGRESS   - The collective was not completed and is in progress.
 *                            @ref ucg_request_check_status() should be used to
 *                            monitor @a req status.
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucg_collective_start_nbr(ucg_coll_h coll, void *req);


/**
 * @ingroup UCG_GROUP
 * @brief Destroys a collective operation handle.
 *
 * This is only required for persistent collectives, where the flag
 * UCG_GROUP_COLLECTIVE_MODIFIER_PERSISTENT is passed when calling
 * @ref ucg_collective_create. Otherwise, the handle is
 * destroyed when the collective operation is completed.
 *
 * @param [in]  coll         Collective operation handle.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
void ucg_collective_destroy(ucg_coll_h coll);


/**
 * @ingroup UCG_GROUP
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
ucs_status_t ucg_request_check_status(void *request);


/**
 * @ingroup UCG_GROUP
 * @brief Cancel an outstanding communications request.
 *
 * @param [in]  worker       UCG worker.
 * @param [in]  request      Non-blocking request to cancel.
 *
 */
void ucg_request_cancel(ucg_worker_h worker, void *request);


/**
 * @ingroup UCG_GROUP
 * @brief Release a communications request.
 *
 * @param [in]  request      Non-blocking request to release.
 *
 * This routine releases the non-blocking request back to the library, regardless
 * of its current state. Communications operations associated with this request
 * will make progress internally, however no further notifications or callbacks
 * will be invoked for this request.
 */
void ucg_request_free(void *request);


/** @cond PRIVATE_INTERFACE */
/**
 * @ingroup UCG_CONTEXT
 * @brief UCG context initialization with particular API version.
 *
 * This is an internal routine used to check compatibility with a particular
 * API version. @ref ucg_init should be used to create UCG context.
 */
ucs_status_t ucg_init_version(unsigned api_major_version,
                              unsigned api_minor_version,
                              const ucg_params_t *params,
                              const ucg_config_t *config,
                              ucg_context_h *context_p);
/** @endcond */


/**
 * @ingroup UCG_CONTEXT
 * @brief UCG context initialization.
 *
 * This routine creates and initializes a @ref ucg_context_h
 * "UCG application context".
 *
 * @warning This routine must be called before any other UCG function
 * call in the application.
 *
 * This routine checks API version compatibility, then discovers the available
 * network interfaces, and initializes the network resources required for
 * discovering of the network and memory related devices.
 *  This routine is responsible for initialization all information required for
 * a particular application scope, for example, MPI application, OpenSHMEM
 * application, etc.
 *
 * @param [in]  config        UCG configuration descriptor allocated through
 *                            @ref ucg_config_read "ucg_config_read()" routine.
 * @param [in]  params        User defined @ref ucg_params_t configurations for the
 *                            @ref ucg_context_h "UCG application context".
 * @param [out] context_p     Initialized @ref ucg_context_h
 *                            "UCG application context".
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucg_init(const ucg_params_t *params,
                      const ucg_config_t *config,
                      ucg_context_h *context_p);

END_C_DECLS

#endif
