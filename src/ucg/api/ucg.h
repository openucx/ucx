/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
* Copyright (C) Los Alamos National Security, LLC. 2018 ALL RIGHTS RESERVED.
* Copyright (C) Huawei Technologies Co., Ltd. 2018.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCG_H_
#define UCG_H_

#include <ucp/api/ucp.h>
#include <ucc/api/ucg_version.h>
#include <ucs/type/thread_mode.h>
#include <ucs/type/cpu_set.h>
#include <ucs/config/types.h>
#include <ucs/sys/compiler_def.h>
#include <stdio.h>
#include <sys/types.h>
#include "../../ucg/api/ucg_def.h"

BEGIN_C_DECLS

#define ucg_context_h               ucp_context_h
#define ucg_config_t                ucp_config_t
#define ucg_address_t               ucp_address_t
#define ucg_worker_h                ucp_worker_h
#define ucg_request_t               ucp_request_t
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
#define ucg_init_version            ucp_init_version
#define ucg_init                    ucp_init
#define ucg_cleanup                 ucp_cleanup
#define ucg_context_query           ucp_context_query
#define ucg_context_print_info      ucp_context_print_info
#define ucg_worker_create           ucp_worker_create
#define ucg_worker_destroy          ucp_worker_destroy
#define ucg_worker_query            ucp_worker_query
#define ucg_worker_print_info       ucp_worker_print_info
#define ucg_worker_get_address      ucp_worker_get_address
#define ucg_worker_release_address  ucp_worker_release_address
#define ucg_worker_progress         ucp_worker_progress
#define ucg_request_check_status    ucp_request_check_status
#define ucg_request_cancel          ucp_request_cancel
#define ucg_request_free            ucp_request_free

/**
 * @defgroup UCG_API Unified Communication Protocol (UCG) API
 * @{
 * This section describes UCG API.
 * @}
 */

 /**
 * @defgroup UCG_GROUP UCG Group
 * @ingroup UCG_API
 * @{
 * UCG Group routines
 * @}
 */

/**
 * @ingroup UCG_GROUP
 * @brief UCG group collective operation description.
 *
 * The enumeration allows specifying modifiers to describe the requested
 * collective operation, as part of @ref ucg_group_collective_params_t
 * passed to @ref ucg_group_collective_start .
 */
enum ucg_group_collective_modifiers {
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
    UCG_GROUP_COLLECTIVE_MODIFIER_PERSISTENT         = UCS_BIT(11)  /* otherwise destroy coll_h */
};

enum ucg_group_member_distance {
    UCG_GROUP_MEMBER_DISTANCE_SELF = 0,
    UCG_GROUP_MEMBER_DISTANCE_SOCKET,
    UCG_GROUP_MEMBER_DISTANCE_HOST,
    UCG_GROUP_MEMBER_DISTANCE_NET,
    UCG_GROUP_MEMBER_DISTANCE_LAST
};

typedef struct ucg_group_params {
    ucg_group_member_index_t member_count; /* number of group members */

    /* For each member - its distance is used to determine the topology */
    enum ucg_group_member_distance *distance;

    /* MPI passes its own reduction function, used for complex data-types */
    void   (*mpi_reduce_f)(void *mpi_op, void *src, void *dst, unsigned count, void *mpi_dtype);

    /* Callback function for connection establishment */
    ucs_status_t (*mpi_get_ep_f)(void *cb_group_obj, ucg_group_member_index_t index, ucg_address_h **addr);
                 //TODO: (*mpi_release_address);

    void *cb_group_obj;  /* external group object for call-backs (MPI_Comm) */
} ucg_group_params_t;

typedef struct ucg_group_collective {
    enum ucg_group_collective_modifiers flags;
    ucg_group_member_index_t            root;       /* root member index */
    const void                         *sbuf;       /* data to submit */
    void                               *rbuf;       /* buffer to receive the result */
    size_t                              count;      /* item count */
    ucp_datatype_t                      datatype;   /* item type */
    void                               *cb_r_op;    /* external reduce op, for (MPI) callbacks */
    void                               *cb_r_dtype; /* external reduce dtype, for (MPI) callbacks */
    ucg_group_collective_callback_t     comp_cb;    /* completion callback */
} ucg_group_collective_params_t;



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
 * @brief Creates a collective operation on a group object.
 *
 * @param [in]  group       Group object to use.
 * @param [in]  params      Collective operation parameters.
 * @param [out] coll        Collective operation handle.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucg_group_collective_create(ucg_group_h group,
                                         ucg_group_collective_params_t *params,
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
ucs_status_ptr_t ucg_group_collective_start_nb(ucg_coll_h coll);


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
ucs_status_t ucg_group_collective_start_nbr(ucg_coll_h coll, void *req);


/**
 * @ingroup UCG_GROUP
 * @brief Destroys a collective operation handle.
 *
 * This is only required for persistent collectives, where the flag
 * UCG_GROUP_COLLECTIVE_MODIFIER_PERSISTENT is passed when calling
 * @ref ucg_group_collective_create. Otherwise, the handle is
 * destroyed when the collective operation is completed.
 *
 * @param [in]  coll         Collective operation handle.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucg_group_collective_destroy(ucg_coll_h coll);







#define UCG_COLL_PARAMS_DTYPE(_flags, _sbuf, _rbuf, _count, _dtype,        \
                              _mpi_op, _mpi_dtype, _group, _root, _cb) {   \
            .flags = _flags,                                               \
            .sbuf = _sbuf,                                                 \
            .rbuf = _rbuf,                                                 \
            .count = _count,                                               \
            .datatype = _dtype,                                            \
            .cb_r_op = _mpi_op,                                            \
            .cb_r_dtype = _mpi_dtype,                                      \
            .root = _root,                                                 \
            .comp_cb = cb                                                  \
}

static inline ucs_status_t ucg_coll_allreduce_init(const void *sbuf,
        void *rbuf, int count, ucp_datatype_t dtype, void *mpi_op,
        void *mpi_dtype, ucg_group_h group,
        enum ucg_group_collective_modifiers extra_flags,
        ucg_group_collective_callback_t cb,
        ucg_coll_h *coll_p)
{
    ucg_group_collective_params_t params = UCG_COLL_PARAMS_DTYPE(
            UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE | UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST,
            sbuf, rbuf, count, dtype, mpi_op, mpi_dtype, group, 0, cb);

    return ucg_group_collective_create(group, &params, coll_p);
}

static inline ucs_status_t ucg_coll_reduce_init(const void *sbuf,
        void *rbuf, int count, ucp_datatype_t dtype, void *mpi_op,
        void *mpi_dtype, ucg_group_h group, int root,
        enum ucg_group_collective_modifiers extra_flags,
        ucg_group_collective_callback_t cb,
        ucg_coll_h *coll_p)
{
    ucg_group_collective_params_t params = UCG_COLL_PARAMS_DTYPE(
            UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE,
            sbuf, rbuf, count, dtype, mpi_op, mpi_dtype, group, root, cb);

    return ucg_group_collective_create(group, &params, coll_p);
}

static inline ucs_status_t ucg_coll_bcast_init(const void *sbuf,
        void *rbuf, int count, ucp_datatype_t dtype,
        void *mpi_dtype, ucg_group_h group, int root,
        enum ucg_group_collective_modifiers extra_flags,
        ucg_group_collective_callback_t cb,
        ucg_coll_h *coll_p)
{
    ucg_group_collective_params_t params = UCG_COLL_PARAMS_DTYPE(
            UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST,
            sbuf, rbuf, count, dtype, 0, mpi_dtype, group, root, cb);

    return ucg_group_collective_create(group, &params, coll_p);
}

static inline ucs_status_t ucg_coll_barrier_init(ucg_group_h group,
        enum ucg_group_collective_modifiers extra_flags,
        ucg_group_collective_callback_t cb,
        ucg_coll_h *coll_p)
{
    ucg_group_collective_params_t params = {
            .flags = UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE | UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST,
            .count = 0,
            .comp_cb = cb
    };

    return ucg_group_collective_create(group, &params, coll_p);
}

END_C_DECLS

#endif
