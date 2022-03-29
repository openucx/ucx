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
#include <ucs/sys/topo/base/topo.h>

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
 * @ingroup UCT_RESOURCE
 * @brief All existing UCT operations
 *
 * This enumeration defines all available UCT operations.
 */
typedef enum uct_ep_operation {
    UCT_EP_OP_AM_SHORT,     /**< Short active message */
    UCT_EP_OP_AM_BCOPY,     /**< Buffered active message */
    UCT_EP_OP_AM_ZCOPY,     /**< Zero-copy active message */
    UCT_EP_OP_PUT_SHORT,    /**< Short put */
    UCT_EP_OP_PUT_BCOPY,    /**< Buffered put */
    UCT_EP_OP_PUT_ZCOPY,    /**< Zero-copy put */
    UCT_EP_OP_GET_SHORT,    /**< Short get */
    UCT_EP_OP_GET_BCOPY,    /**< Buffered get */
    UCT_EP_OP_GET_ZCOPY,    /**< Zero-copy get */
    UCT_EP_OP_EAGER_SHORT,  /**< Tag matching short eager */
    UCT_EP_OP_EAGER_BCOPY,  /**< Tag matching bcopy eager */
    UCT_EP_OP_EAGER_ZCOPY,  /**< Tag matching zcopy eager */
    UCT_EP_OP_RNDV_ZCOPY,   /**< Tag matching rendezvous eager */
    UCT_EP_OP_ATOMIC_POST,  /**< Atomic post */
    UCT_EP_OP_ATOMIC_FETCH, /**< Atomic fetch */
    UCT_EP_OP_LAST
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

    /** Enables @ref uct_perf_attr_t::local_sys_device */
    UCT_PERF_ATTR_FIELD_LOCAL_SYS_DEVICE   = UCS_BIT(3),

    /** Enables @ref uct_perf_attr_t::remote_sys_device */
    UCT_PERF_ATTR_FIELD_REMOTE_SYS_DEVICE  = UCS_BIT(4),

    /** Enables @ref uct_perf_attr_t::send_pre_overhead */
    UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD  = UCS_BIT(5),

    /** Enables @ref uct_perf_attr_t::send_post_overhead */
    UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD = UCS_BIT(6),

    /** Enables @ref uct_perf_attr_t::recv_overhead */
    UCT_PERF_ATTR_FIELD_RECV_OVERHEAD      = UCS_BIT(7),

    /** Enables @ref uct_perf_attr_t::bandwidth */
    UCT_PERF_ATTR_FIELD_BANDWIDTH          = UCS_BIT(8),

    /** Enables @ref uct_perf_attr_t::latency */
    UCT_PERF_ATTR_FIELD_LATENCY            = UCS_BIT(9),

    /** Enable @ref uct_perf_attr_t::max_inflight_eps */
    UCT_PERF_ATTR_FIELD_MAX_INFLIGHT_EPS   = UCS_BIT(10)
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
     * System device where the local memory type resides.
     * Can be UCS_SYS_DEVICE_ID_UNKNOWN.
     * This field must be initialized by the caller.
     */
    ucs_sys_device_t    local_sys_device;

    /**
     * System device where the remote memory type resides.
     * Can be UCS_SYS_DEVICE_ID_UNKNOWN and be the same as local system device.
     * This field must be initialized by the caller.
     */
    ucs_sys_device_t    remote_sys_device;

    /**
     * This is the time spent in the UCT layer to prepare message request and
     * pass it to the hardware or system software layers, in seconds.
     * This field is set by the UCT layer.
     */
    double              send_pre_overhead;

    /**
     * This is the time spent in the UCT layer after the message request has
     * been passed to the hardware or system software layers and before
     * operation has been finalized, in seconds.
     * This value has no effect on how long it takes to deliver the message to
     * remote side.
     * This field is set by the UCT layer.
     */
    double              send_post_overhead;

    /**
     * Message receive overhead time, in seconds.
     * This field is set by the UCT layer.
     */
    double              recv_overhead;

    /**
     * Bandwidth model. This field is set by the UCT layer.
     */
    uct_ppn_bandwidth_t bandwidth;

    /**
     * Latency as a function of number of endpoints.
     * This field is set by the UCT layer.
     */
    ucs_linear_func_t   latency;

    /**
     * Approximate maximum number of endpoints that could have outstanding
     * operations simultaneously.
     * Protocols that require sending to multiple destinations at the same time
     * (such as keepalive) could benefit from using a transport that has a
     * large number of maximum inflight endpoints.
     * This field is set by the UCT layer.
     */
    size_t              max_inflight_eps;
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


/**
 * @ingroup UCT_MD
 * @brief MD memory key pack parameters field mask.
 */
typedef enum {
    UCT_MD_MKEY_PACK_FIELD_FLAGS = UCS_BIT(0)  /**< flags field */
} uct_md_mkey_pack_field_mask_t;


/**
 * @ingroup UCT_RESOURCE
 * @brief UCT endpoint attributes field mask.
 *
 * The enumeration allows specifying which fields in @ref uct_ep_attr_t are
 * present, for backward compatibility support.
 */
enum uct_ep_attr_field {
    /** Enables @ref uct_ep_attr::local_address */
    UCT_EP_ATTR_FIELD_LOCAL_SOCKADDR  = UCS_BIT(0),
    /** Enables @ref uct_ep_attr::remote_address */
    UCT_EP_ATTR_FIELD_REMOTE_SOCKADDR = UCS_BIT(1)
};


/**
 * @ingroup UCT_RESOURCE
 * @brief field mask of @ref uct_iface_is_reachable_v2
 */
typedef enum {
    UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR        = UCS_BIT(0), /**< device_addr field */
    UCT_IFACE_IS_REACHABLE_FIELD_IFACE_ADDR         = UCS_BIT(1), /**< iface_addr field */
    UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING        = UCS_BIT(2), /**< info_string field */
    UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING_LENGTH = UCS_BIT(3)  /**< info_string_length field */
} uct_iface_is_reachable_field_mask_t;


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


typedef enum {
    /**
     * The flag is used indicate that remote access to a memory region
     * associated with the remote key must fail once the memory region is
     * deregister using @ref uct_md_mem_dereg_v2 with
     * @ref UCT_MD_MEM_DEREG_FLAG_INVALIDATE flag set. Using
     * @ref uct_md_mem_dereg_v2 deregistration routine with
     * @ref UCT_MD_MEM_DEREG_FLAG_INVALIDATE flag set on an rkey that was
     * generated without @ref UCT_MD_MKEY_PACK_FLAG_INVALIDATE flag will
     * not function correctly, and may result in data corruption. In other words
     * in order for @ref UCT_MD_MEM_DEREG_FLAG_INVALIDATE flag to function
     * the @ref UCT_MD_MKEY_PACK_FLAG_INVALIDATE flag must be set.
     */
    UCT_MD_MKEY_PACK_FLAG_INVALIDATE = UCS_BIT(0)
} uct_md_mkey_pack_flags_t;


/**
 * @ingroup UCT_RESOURCE
 * @brief Endpoint attributes, capabilities and limitations.
 */
struct uct_ep_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_ep_attr_field. Fields not specified by this mask
     * will be ignored.
     */
    uint64_t                field_mask;

    /**
     * Local sockaddr used by the endpoint.
     */
    struct sockaddr_storage local_address;

    /**
     * Remote sockaddr the endpoint is connected to.
     */
    struct sockaddr_storage remote_address;
};


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


typedef struct uct_md_mkey_pack_params {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_md_mkey_pack_field_mask_t. Fields not specified in this mask
     * will be ignored. Provides ABI compatibility with respect to adding new
     * fields.
     */
    uint64_t field_mask;

    /**
     * Remote key packing flags, using bits from @ref uct_md_mkey_pack_flags_t.
     */
    unsigned flags;
} uct_md_mkey_pack_params_t;


/**
 * @ingroup UCT_RESOURCE
 * @brief Names for UCT endpoint operations.
 */
extern const char *uct_ep_operation_names[];


/**
 * @ingroup UCT_RESOURCE
 * @brief Operation parameters passed to @ref uct_iface_is_reachable_v2.
 */
typedef struct uct_iface_is_reachable_params {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_iface_is_reachable_field_mask_t. Fields not specified in this
     * mask will be ignored. Provides ABI compatibility with respect to adding
     * new fields.
     */
    uint64_t                     field_mask;

    /**
     * Device address to check for reachability.
     * This field must not be passed if iface_attr.dev_addr_len == 0.
     */
    const uct_device_addr_t      *device_addr;

    /**
     * Interface address to check for reachability.
     * This field must not be passed if iface_attr.iface_addr_len == 0.
     */
    const uct_iface_addr_t       *iface_addr;

    /**
     * User-provided pointer to a string buffer.
     * The function @ref uct_iface_is_reachable_v2 fills this buffer with a
     * null-terminated information string explaining why the remote address is
     * not reachable if the return value is 0.
     */
    char                         *info_string;

    /**
     * The length of the @a info_string is provided in bytes.
     * This value must be specified in conjunction with @a info_string.
     */
    size_t                        info_string_length;
} uct_iface_is_reachable_params_t;


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


/**
 * @ingroup UCT_MD
 *
 * @brief Pack a remote key.
 *
 * @param [in]  md          Handle to memory domain.
 * @param [in]  memh        Pack a remote key for this memory handle.
 * @param [in]  params      Operation parameters, see @ref
 *                          uct_md_mkey_pack_params_t.
 * @param [out] rkey_buffer Pointer to a buffer to hold the packed remote key.
 *                          The size of this buffer has should be at least
 *                          @ref uct_md_attr_t::rkey_packed_size, as returned by
 *                          @ref uct_md_query.
 * @return                  Error code.
 */
ucs_status_t uct_md_mkey_pack_v2(uct_md_h md, uct_mem_h memh,
                                 const uct_md_mkey_pack_params_t *params,
                                 void *rkey_buffer);


/**
 * @ingroup UCT_RESOURCE
 * @brief Get ep's attributes.
 *
 * This routine fetches information about the endpoint.
 *
 * @param [in]  ep         Endpoint to query.
 * @param [out] ep_attr    Filled with endpoint attributes.
 *
 * @return Error code.
 */
ucs_status_t uct_ep_query(uct_ep_h ep, uct_ep_attr_t *ep_attr);


/**
 * @ingroup UCT_RESOURCE
 * @brief Check if remote iface address is reachable.
 *
 * This function checks if a remote address can be reached from a local
 * interface. If the function returns a non-zero value, it does not necessarily
 * mean a connection and/or data transfer would succeed; as the reachability
 * check is a local operation it does not detect issues such as network
 * mis-configuration or lack of connectivity.
 *
 * @param [in]  iface       Local interface to check reachability from.
 * @param [in]  params      Operation parameters, see @ref
 *                          uct_iface_is_reachable_params_t.
 *
 * @return Nonzero if reachable, 0 if not.
 */
int uct_iface_is_reachable_v2(uct_iface_h iface,
                              const uct_iface_is_reachable_params_t *params);

END_C_DECLS

#endif
