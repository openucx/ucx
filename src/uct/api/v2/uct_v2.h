/**
 * @file        uct_v2.h
 * @date        2021
 * @copyright   Mellanox Technologies Ltd. All rights reserved.
 * @brief       Unified Communication Transport
 */

#ifndef UCT_V2_H_
#define UCT_V2_H_

#include <ucs/sys/compiler_def.h>
#include <ucs/memory/memory_type.h>
#include <uct/api/uct.h>

#include <stdint.h>

BEGIN_C_DECLS

/** @file uct_v2.h */

/**
* @defgroup UCT_RESOURCE   UCT Communication Resource
* @ingroup UCT_API
* @{
* This section describes a concept of the Communication Resource and routines
* associated with the concept.
* @}
*/

/**
 * @brief All existing UCT operations
 *
 * This enumeration defines all available UCT operations.
 */
typedef enum uct_ep_operation {
    UCT_OP_AM_SHORT,     /**< Short active message */
    UCT_OP_AM_BCOPY,     /**< Buffered active message */
    UCT_OP_AM_ZCOPY,     /**< Zero-copy active message */
    UCT_OP_PUT_SHORT,    /**< Short put */
    UCT_OP_PUT_BCOPY,    /**< Buffered put */
    UCT_OP_PUT_ZCOPY,    /**< Zero-copy put */
    UCT_OP_GET_SHORT,    /**< Short get */
    UCT_OP_GET_BCOPY,    /**< Buffered get */
    UCT_OP_GET_ZCOPY,    /**< Zero-copy get */
    UCT_OP_EAGER_SHORT,  /**< Tag matching short eager */
    UCT_OP_EAGER_BCOPY,  /**< Tag matching bcopy eager */
    UCT_OP_EAGER_ZCOPY,  /**< Tag matching zcopy eager */
    UCT_OP_RNDV_ZCOPY,   /**< Tag matching rendezvous eager */
    UCT_OP_ATOMIC_POST,  /**< Atomic post */
    UCT_OP_ATOMIC_FETCH  /**< Atomic fetch */
} uct_ep_operation_t;


/**
 * @ingroup UCT_RESOURCE
 * @brief UCT interface query by @ref uct_iface_estimate_perf parameters field mask.
 *
 * The enumeration allows specifying which fields in @ref uct_perf_attr_t are
 * present, for backward compatibility support.
 */
enum uct_perf_attr_field {
    /** Enables @ref uct_perf_attr_t::operation */
    UCT_PERF_ATTR_FIELD_OPERATION          = UCS_BIT(0),

    /** Enables @ref uct_perf_attr_t::local_memory_type */
    UCT_PERF_ATTR_FIELD_LOCAL_MEMORY_TYPE  = UCS_BIT(1),

    /** Enables @ref uct_perf_attr_t::remote_memory_type */
    UCT_PERF_ATTR_FIELD_REMOTE_MEMORY_TYPE = UCS_BIT(2),

    /** Enables @ref uct_perf_attr_t::overhead */
    UCT_PERF_ATTR_FIELD_OVERHEAD           = UCS_BIT(3),

    /** Enables @ref uct_perf_attr_t::bandwidth */
    UCT_PERF_ATTR_FIELD_BANDWIDTH          = UCS_BIT(4)
};


/**
 * @ingroup UCT_RESOURCE
 * @brief Parameters for querying a UCT interface by @ref uct_iface_estimate_perf
 *
 * This structure must be allocated and initialized by the user
 */
typedef struct {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_perf_attr_field. Fields not specified by this mask will be
     * ignored. This field must be initialized by the caller.
     */
    uint64_t            field_mask;

    /**
     * Operation to report performance for.
     * This field must be initialized by the caller.
     */
    uct_ep_operation_t  operation;

    /**
     * Local memory type to use for determining performance.
     * This field must be initialized by the caller.
     */
    ucs_memory_type_t   local_memory_type;

    /**
     * Remote memory type to use for determining performance.
     * Relevant only for operations that have remote memory access.
     * This field must be initialized by the caller.
     */
    ucs_memory_type_t   remote_memory_type;

    /**
     * Message overhead time, in seconds. This field is set by the UCT layer.
     */
    double              overhead;

    /**
     * Bandwidth model. This field is set by the UCT layer.
     */
    uct_ppn_bandwidth_t bandwidth;
} uct_perf_attr_t;


/**
 * @ingroup UCT_MD
 * @brief MD memory de-registration operation flags.
 */
typedef enum {
    UCT_MD_MEM_DEREG_FIELD_MEMH       = UCS_BIT(0), /**< memh field */
    UCT_MD_MEM_DEREG_FIELD_FLAGS      = UCS_BIT(1), /**< flags field */
    UCT_MD_MEM_DEREG_FIELD_COMPLETION = UCS_BIT(2)  /**< comp field */
} uct_md_mem_dereg_field_mask_t;


typedef enum {
    /**
     * Invalidate the memory region. If this flag is set then memory region is
     * invalidated after de-registration and the callback (see @ref
     * uct_md_mem_dereg_params_t) is called when the memory is fully
     * invalidated and will not be accessed anymore by zero-copy or remote
     * memory access operations.
     */
    UCT_MD_MEM_DEREG_FLAG_INVALIDATE = UCS_BIT(0) 
} uct_md_mem_dereg_flags_t;


/**
 * @ingroup UCT_MD
 * @brief Completion callback for memory region invalidation.
 *
 * This callback routine is invoked when is no longer accessible by remote peer.
 * 
 * $note: in some implementations this callback may be called immediately after
 *        @ref uct_md_mem_dereg_v2 is called, but it is possible that the
 *        callback call will be delayed until all references to the memory
 *        region, for example, when rcache is used in the implementation of the
 *        memory domains.
 *
 * @param [in]  arg User data passed to "arg" value, see
 *                  @ref uct_md_mem_dereg_params_t
 */
typedef void (*uct_md_mem_invalidate_cb_t)(void *arg);


/**
 * @ingroup UCT_MD
 * @brief Operation parameters passed to @ref uct_md_mem_dereg_v2.
 */
typedef struct uct_md_mem_dereg_params {
    /**
     * Mask of valid fields in this structure and operation flags, using
     * bits from @ref uct_md_mem_dereg_field_mask_t. Fields not specified
     * in this mask will be ignored. Provides ABI compatibility with respect
     * to adding new fields.
     */
    uint64_t                     field_mask;

    /**
     * Operation specific flags, using bits from @ref uct_md_mem_dereg_flags_t.
     */
    unsigned                     flags;

    /**
     * Memory handle to deregister.
     */
    uct_mem_h                    memh;

    /**
     * Pointer to UCT completion object that is invoked when region is
     * invalidated.
     */
    uct_completion_t             *comp;
} uct_md_mem_dereg_params_t;


/**
 * @ingroup UCT_RESOURCE
 * @brief Get interface performance attributes, by memory types and operation.
 *        A pointer to uct_perf_attr_t struct must be passed, with the memory
 *        types and operation members initialized. Overhead and bandwidth
 *        for the operation on the given memory types will be reported.
 *
 * @param [in]    tl_iface  Interface to query.
 * @param [inout] perf_attr Filled with performance attributes.
 */
ucs_status_t
uct_iface_estimate_perf(uct_iface_h tl_iface, uct_perf_attr_t *perf_attr);


/**
 * @ingroup UCT_MD
 * @brief Undo the operation of @ref uct_md_mem_reg() and invalidate memory
 *        region.
 *
 * This routine deregisters the memory region registered by @ref uct_md_mem_reg
 * and allow the memory region to be invalidated with callback called when the
 * memory region is unregistered.
 * 
 * @param [in]  md          Memory domain that was used to register the memory.
 * @param [in]  params      Operation parameters, see @ref 
 *                          uct_md_mem_dereg_params_t.
 */
ucs_status_t uct_md_mem_dereg_v2(uct_md_h md,
                                 const uct_md_mem_dereg_params_t *params);

END_C_DECLS

#endif
