/**
 * @file        uct_v2.h
 * @date        2021
 * @copyright   NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
    UCT_EP_OP_RNDV_ZCOPY,   /**< Tag matching rendezvous */
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
    UCT_PERF_ATTR_FIELD_MAX_INFLIGHT_EPS   = UCS_BIT(10),

    /** Enable @ref uct_perf_attr_t::flags */
    UCT_PERF_ATTR_FIELD_FLAGS              = UCS_BIT(11)
};

/**
 * @ingroup UCT_RESOURCE
 * @brief Flags of supported performance attributes functionalities
 *
 * This is used in @ref uct_perf_attr_t::flags.
 */
typedef enum {
    /** TX operations can depend on unrelated RX operation completion */
    UCT_PERF_ATTR_FLAGS_TX_RX_SHARED = UCS_BIT(0)
} uct_perf_attr_flags_t;

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
     * large number of maximum in-flight endpoints.
     * This field is set by the UCT layer.
     */
    size_t              max_inflight_eps;

    /**
     * Performance characteristics of the network interface.
     */
    uint64_t            flags;
} uct_perf_attr_t;


/**
 * @ingroup UCT_MD
 * @brief MD memory registration operation flags.
 */
typedef enum {
    UCT_MD_MEM_REG_FIELD_FLAGS         = UCS_BIT(0),
    UCT_MD_MEM_REG_FIELD_DMABUF_FD     = UCS_BIT(1),
    UCT_MD_MEM_REG_FIELD_DMABUF_OFFSET = UCS_BIT(2)
} uct_md_mem_reg_field_mask_t;


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
 * @ingroup UCT_MD
 * @brief MD memory attach operation parameters field mask.
 */
typedef enum {
    /** Enables @ref uct_md_mem_attach_params_t.flags field */
    UCT_MD_MEM_ATTACH_FIELD_FLAGS = UCS_BIT(0)
} uct_md_mem_attach_field_mask_t;


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
    UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING_LENGTH = UCS_BIT(3), /**< info_string_length field */
    UCT_IFACE_IS_REACHABLE_FIELD_SCOPE              = UCS_BIT(4) /**<  scope field */
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
     * deregistered using @ref uct_md_mem_dereg_v2 with
     * @ref UCT_MD_MEM_DEREG_FLAG_INVALIDATE flag set. Using
     * @ref uct_md_mem_dereg_v2 deregistration routine with
     * @ref UCT_MD_MEM_DEREG_FLAG_INVALIDATE flag set on an rkey that was
     * generated without
     * @ref UCT_MD_MKEY_PACK_FLAG_INVALIDATE_RMA and/or
     * @ref UCT_MD_MKEY_PACK_FLAG_INVALIDATE_AMO flags will not function
     * correctly, and may result in data corruption. In other words, in order
     * for @ref UCT_MD_MEM_DEREG_FLAG_INVALIDATE flag to function
     * the @ref UCT_MD_MKEY_PACK_FLAG_INVALIDATE_RMA and/or
     * the @ref UCT_MD_MKEY_PACK_FLAG_INVALIDATE_AMO flag must be set.
     */
    UCT_MD_MKEY_PACK_FLAG_INVALIDATE_RMA = UCS_BIT(0),

    /**
     * The flag is used to indicate that the atomic operations must fail once
     * the memory region is deregistered using @ref uct_md_mem_dereg_v2 with
     * @ref UCT_MD_MEM_DEREG_FLAG_INVALIDATE flag set.
     * Using @ref uct_md_mem_dereg_v2 deregistration routine with
     * @ref UCT_MD_MEM_DEREG_FLAG_INVALIDATE flag set on an rkey that was
     * generated without
     * @ref UCT_MD_MKEY_PACK_FLAG_INVALIDATE_RMA and/or
     * @ref UCT_MD_MKEY_PACK_FLAG_INVALIDATE_AMO flags will not function
     * correctly, and may result in data corruption. In other words, in order
     * for @ref UCT_MD_MEM_DEREG_FLAG_INVALIDATE flag to function
     * the @ref UCT_MD_MKEY_PACK_FLAG_INVALIDATE_RMA and/or
     * the @ref UCT_MD_MKEY_PACK_FLAG_INVALIDATE_AMO flag must be set.
     */
    UCT_MD_MKEY_PACK_FLAG_INVALIDATE_AMO = UCS_BIT(1),

    /**
     * The flag is used to indicate that the memory region should be packed in
     * order to be accessed by another process using the same device to perform
     * UCT operations.
     */
    UCT_MD_MKEY_PACK_FLAG_EXPORT         = UCS_BIT(2)
} uct_md_mkey_pack_flags_t;


/**
 * @ingroup UCT_MD
 * @brief Flags used in @ref uct_md_mem_attach
 */
typedef enum {
    /** Hide errors on memory attach. */
    UCT_MD_MEM_ATTACH_FLAG_HIDE_ERRORS = UCS_BIT(0)
} uct_md_mem_attach_flags_t;


/**
 * @ingroup UCT_RESOURCE
 * @brief uct_ep_connect_to_ep_v2 operation fields and flags
 * 
 * The enumeration allows specifying which fields in @ref
 * uct_ep_connect_to_ep_params_t are present and operation flags are used. It is
 * used to enable backward compatibility support.
 */
typedef enum {
    /** Device address length */
    UCT_EP_CONNECT_TO_EP_PARAM_FIELD_DEVICE_ADDR_LENGTH = UCS_BIT(0),

    /** Endpoint address length */
    UCT_EP_CONNECT_TO_EP_PARAM_FIELD_EP_ADDR_LENGTH     = UCS_BIT(1)
} uct_ep_connect_to_ep_param_field_t;


/**
 * @ingroup UCT_RESOURCE
 * @brief field mask of @ref uct_ep_is_connected_params_t
 *
 * The enumeration allows specifying which fields in @ref
 * uct_ep_is_connected_params_t are present.
 */
typedef enum {
    /** Device address */
    UCT_EP_IS_CONNECTED_FIELD_DEVICE_ADDR = UCS_BIT(0),

    /** Interface address */
    UCT_EP_IS_CONNECTED_FIELD_IFACE_ADDR  = UCS_BIT(1),

    /** Endpoint address */
    UCT_EP_IS_CONNECTED_FIELD_EP_ADDR     = UCS_BIT(2)
} uct_ep_is_connected_field_mask_t;


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
 * @brief Operation parameters passed to @ref uct_md_mem_reg_v2.
 */
typedef struct uct_md_mem_reg_params {
    /**
     * Mask of valid fields in this structure and operation flags, using
     * bits from @ref uct_md_mem_reg_field_mask_t. Fields not specified
     * in this mask will be ignored. Provides ABI compatibility with respect
     * to adding new fields.
     */
    uint64_t                     field_mask;

    /**
     * Operation specific flags, using bits from @ref uct_md_mem_flags.
     */
    uint64_t                     flags;

    /**
     * dmabuf file descriptor of the memory region to register.
     *
     * If is set (along with its corresponding bit in the field_mask -
     * @ref UCT_MD_MEM_REG_FIELD_DMABUF_FD), the memory region will be
     * registered using dmabuf mechanism.
     * Can be used only if the memory domain supports dmabuf registration by
     * returning @ref UCT_MD_FLAG_REG_DMABUF flag from @ref uct_md_query_v2.
     *
     * When registering memory using dmabuf mechanism, any memory type is supported
     * (assuming a valid dmabuf file descriptor could be created).
     * Therefore, @ref uct_md_attr_v2_t.reg_mem_types field is not relevant for
     * such registrations.
     *
     * If not set, it's assumed to be @ref UCT_DMABUF_FD_INVALID, and the memory
     * region will be registered without using dmabuf mechanism.
     *
     * More information about dmabuf registration can be found in
     * https://01.org/linuxgraphics/gfx-docs/drm/driver-api/dma-buf.html
     */
    int                          dmabuf_fd;

    /**
     * When @ref uct_md_mem_reg_params_t.dmabuf_fd is provided, this field
     * specifies the offset of the region to register relative to the start of
     * the underlying dmabuf region.
     *
     * If not set (along with its corresponding bit in the field_mask -
     * @ref UCT_MD_MEM_REG_FIELD_DMABUF_OFFSET) it's assumed to be 0.
     *
     * @note The value of this field must be equal to the offset of the virtual
     * address provided by the address parameter to @ref uct_md_mem_reg_v2 from
     * the beginning of the memory region associated with
     * @ref uct_md_mem_reg_params_t.dmabuf_fd.
     * For example, if the virtual address is equal to the beginning of the
     * dmabuf region, then this field must be omitted or set to 0.
     */
    size_t                       dmabuf_offset;
} uct_md_mem_reg_params_t;


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
 * @ingroup UCT_MD
 * @brief Operation parameters passed to @ref uct_md_mem_attach.
 */
typedef struct uct_md_mem_attach_params {
    /**
     * Mask of valid fields in this structure and operation flags, using
     * bits from @ref uct_md_mem_attach_field_mask_t. Fields not specified in
     * this mask will be ignored. Provides ABI compatibility with respect to
     * adding new fields.
     */
    uint64_t                     field_mask;

    /**
     * Operation specific flags, using bits from
     * @ref uct_md_mem_attach_flags_t.
     */
    uint64_t                     flags;
} uct_md_mem_attach_params_t;


/**
 * @ingroup UCT_RESOURCE
 * @brief Names for UCT endpoint operations.
 */
extern const char *uct_ep_operation_names[];


/**
 * @ingroup UCS_RESOURCE
 * @brief Reachability scope
 *
 * Reachability scope. Can be one of the following values:
 * Device scope: Checks if addresses describe the same device.
 * Network scope: Checks reachability between different devices.
 */
typedef enum {
    UCT_IFACE_REACHABILITY_SCOPE_DEVICE, /**< Local device scope */
    UCT_IFACE_REACHABILITY_SCOPE_NETWORK /**< Network scope */
} uct_iface_reachability_scope_t;


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
    uint64_t                       field_mask;

    /**
     * Device address to check for reachability.
     * This field must not be passed if iface_attr.dev_addr_len == 0.
     */
    const uct_device_addr_t       *device_addr;

    /**
     * Interface address to check for reachability.
     * This field must not be passed if iface_attr.iface_addr_len == 0.
     */
    const uct_iface_addr_t        *iface_addr;

    /**
     * User-provided pointer to a string buffer.
     * The function @ref uct_iface_is_reachable_v2 fills this buffer with a
     * null-terminated information string explaining why the remote address is
     * not reachable if the return value is 0.
     */
    char                          *info_string;

    /**
     * The length of the @a info_string is provided in bytes.
     * This value must be specified in conjunction with @a info_string.
     */
    size_t                         info_string_length;

    /**
     * Reachability scope.
     */
    uct_iface_reachability_scope_t scope;
} uct_iface_is_reachable_params_t;


/**
 * @ingroup UCT_RESOURCE
 * @brief Operation parameters passed to @ref uct_ep_is_connected.
 *
 * This struct is used to pass the required arguments to
 * @ref uct_ep_is_connected.
 */
typedef struct uct_ep_is_connected_params {
    /**
     * Mask of valid fields in this structure, using
     * bits from @ref uct_ep_is_connected_field_mask_t. Fields not specified
     * in this mask will be ignored. Provides ABI compatibility with respect to
     * adding new fields.
     */
    uint64_t                 field_mask;

    /**
     * Device address to check for connectivity.
     * This field must be passed if @ref uct_iface_query returned
     * @ref uct_iface_attr_t::dev_addr_len > 0 on the remote side.
     */
    const uct_device_addr_t *device_addr;

    /**
     * Interface address to check for connectivity.
     * This field must be passed if this endpoint was created by calling
     * @ref uct_ep_create with @ref uct_ep_params_t::iface_addr.
     */
    const uct_iface_addr_t  *iface_addr;

    /**
     * Endpoint address to check for connectivity.
     * This field must be passed if @ref uct_ep_connect_to_ep_v2 was
     * called on this endpoint.
     */
    const uct_ep_addr_t     *ep_addr;
} uct_ep_is_connected_params_t;


/**
 * @ingroup UCT_RESOURCE
 * @brief Parameters for connecting a UCT endpoint by @ref
 * uct_ep_connect_to_ep_v2.
 */
typedef struct uct_ep_connect_to_ep_params {
    /**
     * Mask of valid fields in this structure and operation flags, using
     * bits from @ref uct_ep_connect_to_ep_param_field_t. Fields not specified
     * in this mask will be ignored. Provides ABI compatibility with respect to
     * adding new fields.
     */
    uint64_t                      field_mask;

    /**
     * Device address length. If not provided, the transport will assume a
     * default minimal length according to the address buffer contents.
     */
    size_t                        device_addr_length;

    /**
     * Endpoint address length. If not provided, the transport will assume a
     * default minimal length according to the address buffer contents.
     */
    size_t                        ep_addr_length;
} uct_ep_connect_to_ep_params_t;


/**
 * @ingroup UCT_MD
 * @brief Parameters for comparing remote keys using @ref uct_rkey_compare.
 */
typedef struct uct_rkey_compare_params {
    /**
     * Mask of valid fields in this structure. Must currently be equal to zero.
     * Fields not specified in this mask will be ignored. Provides ABI
     * compatibility with respect to adding new fields.
     */
    uint64_t                      field_mask;
} uct_rkey_compare_params_t;

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
 * @brief Register memory for zero-copy sends and remote access.
 *
 * Register memory on the memory domain. In order to use this function, @a md
 * must support the @ref UCT_MD_FLAG_REG flag.
 *
 * @param [in]  md          Memory domain that was used to register the memory.
 * @param [in]  address     Memory to register.
 * @param [in]  length      Size of memory to register. Must be > 0.
 * @param [in]  params      Operation parameters, see @ref
 *                          uct_md_mem_reg_params_t.
 * @param [out] memh_p      Filled with handle for allocated region.
 *
 * @return Error code.
 */
ucs_status_t uct_md_mem_reg_v2(uct_md_h md, void *address, size_t length,
                               const uct_md_mem_reg_params_t *params,
                               uct_mem_h *memh_p);


/**
 * @ingroup UCT_MD
 * @brief Undo the operation of @ref uct_md_mem_reg() or
 *        @ref uct_md_mem_reg_v2() and invalidate memory region.
 *
 * This routine deregisters the memory region registered by @ref uct_md_mem_reg
 * and allows the memory region to be invalidated with callback called when the
 * memory region is unregistered.
 *
 * @param [in]  md          Memory domain that was used to register the memory.
 * @param [in]  params      Operation parameters, see @ref
 *                          uct_md_mem_dereg_params_t.
 *
 * @return Error code.
 */
ucs_status_t uct_md_mem_dereg_v2(uct_md_h md,
                                 const uct_md_mem_dereg_params_t *params);


/**
 * @ingroup UCT_MD
 * @brief UCT MD attributes field mask.
 *
 * The enumeration allows specifying which fields in @ref uct_md_attr_v2_t
 * are present.
 */
typedef enum uct_md_attr_field {
    /** Indicate max allocation size. */
    UCT_MD_ATTR_FIELD_MAX_ALLOC                 = UCS_BIT(0),

    /** Indicate max registration size. */
    UCT_MD_ATTR_FIELD_MAX_REG                   = UCS_BIT(1),

    /** Indicate capability flags. */
    UCT_MD_ATTR_FIELD_FLAGS                     = UCS_BIT(2),

    /** Indicate memory types that the MD can register. */
    UCT_MD_ATTR_FIELD_REG_MEM_TYPES             = UCS_BIT(3),

    /** Indicate memory types that are suitable for non-blocking registration. */
    UCT_MD_ATTR_FIELD_REG_NONBLOCK_MEM_TYPES    = UCS_BIT(4),

    /** Indicate memory types that the MD can cache. */
    UCT_MD_ATTR_FIELD_CACHE_MEM_TYPES           = UCS_BIT(5),

    /** Indicate memory types that the MD can detect. */
    UCT_MD_ATTR_FIELD_DETECT_MEM_TYPES          = UCS_BIT(6),

    /** Indicate memory types that the MD can allocate. */
    UCT_MD_ATTR_FIELD_ALLOC_MEM_TYPES           = UCS_BIT(7),

    /** Indicate memory types that the MD can access. */
    UCT_MD_ATTR_FIELD_ACCESS_MEM_TYPES          = UCS_BIT(8),

    /** Indicate memory types for which the MD can return a dmabuf_fd. */
    UCT_MD_ATTR_FIELD_DMABUF_MEM_TYPES          = UCS_BIT(9),

    /** Indicate registration cost. */
    UCT_MD_ATTR_FIELD_REG_COST                  = UCS_BIT(10),

    /** Indicate component name. */
    UCT_MD_ATTR_FIELD_COMPONENT_NAME            = UCS_BIT(11),

    /** Indicate size of buffer needed for packed rkey. */
    UCT_MD_ATTR_FIELD_RKEY_PACKED_SIZE          = UCS_BIT(12),

    /** Indicate CPUs closest to the resource. */
    UCT_MD_ATTR_FIELD_LOCAL_CPUS                = UCS_BIT(13),

    /** Indicate size of buffer needed for packed exported memory key. */
    UCT_MD_ATTR_FIELD_EXPORTED_MKEY_PACKED_SIZE = UCS_BIT(14),

    /** Unique global identifier of the memory domain. */
    UCT_MD_ATTR_FIELD_GLOBAL_ID                 = UCS_BIT(15),

    /** Indicate registration alignment. */
    UCT_MD_ATTR_FIELD_REG_ALIGNMENT             = UCS_BIT(16),

    /** Indicate memory types that the MD can register using global VA MR. */
    UCT_MD_ATTR_FIELD_GVA_MEM_TYPES             = UCS_BIT(17)
} uct_md_attr_field_t;


/**
 * @ingroup UCT_MD
 * @brief  Memory domain attributes.
 *
 * This structure defines the attributes of a Memory Domain which include
 * maximum memory that can be allocated, credentials required for accessing the memory,
 * CPU mask indicating the proximity of CPUs, and bitmaps indicating the types
 * of memory (CPU/CUDA/ROCM) that can be detected, allocated, accessed, and
 * memory types for which dmabuf attributes can be returned.
 */
typedef struct {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_md_attr_field_t.
     */
    uint64_t          field_mask;

    /**
     * Maximal allocation size.
     */
    uint64_t          max_alloc;

    /**
     * Maximal registration size.
     */
    size_t            max_reg;

    /**
     * Memory domain capability flags such as UCT_MD_FLAG_REG. Refer
     * @ref uct_md_attr_t.
     */
    uint64_t          flags;

    /**
     * Bitmap of memory types which the Memory Domain can be register.
     */
    uint64_t          reg_mem_types;

    /**
     * Bitmap of memory types that are suitable for non-blocking registration
     */
    uint64_t          reg_nonblock_mem_types;

    /**
     * Bitmap of memory types that can be cached for this memory domain.
     */
    uint64_t          cache_mem_types;

    /**
     * Bitmap of memory types that can create global memory handle.
     */
    uint64_t          gva_mem_types;

    /**
     * Bitmap of memory types that Memory Domain can detect if address belongs
     * to it.
     */
    uint64_t          detect_mem_types;

    /**
     * Bitmap of memory types that Memory Domain can allocate memory on.
     */
    uint64_t          alloc_mem_types;

    /**
     * Memory types that Memory Domain can access.
     */
    uint64_t          access_mem_types;

    /**
     * Memory types for which MD can provide DMABUF fd.
     */
    uint64_t          dmabuf_mem_types;

    /**
     * Memory registration cost estimation (time,seconds) as a linear function
     * of the buffer size.
     */
    ucs_linear_func_t reg_cost;

    /**
     * Component name.
     */
    char              component_name[UCT_COMPONENT_NAME_MAX];

    /**
     * Size of buffer needed for packed rkey.
     */
    size_t            rkey_packed_size;

    /**
     * Mask of CPUs closest to the resource.
     */
    ucs_cpu_set_t     local_cpus;

    /**
     * Size of buffer needed for packing an exported mkey. Valid only if
     * @ref UCT_MD_FLAG_EXPORTED_MKEY is supported by Memory Domain.
     */
    size_t            exported_mkey_packed_size;

    /**
     * Byte array that holds a globally unique device identifier (for example,
     * a MAC address or a GUID). If global identifiers are equal, it means that
     * Memory Domains belong to the same device.
     */
    char              global_id[UCT_MD_GLOBAL_ID_MAX];

    /**
     * Registration alignment.
     */
    size_t            reg_alignment;
} uct_md_attr_v2_t;


/**
 * @ingroup UCT_MD
 * @brief  Memory domain capability flags.
 */
typedef enum {
    UCT_MD_FLAG_V2_FIRST       = UCT_MD_FLAG_LAST,

    /**
     * Memory domain supports invalidation of memory handle registered by
     * @ref uct_md_mem_reg_v2 with @ref UCT_MD_MEM_ACCESS_RMA flag and packed
     * key by @ref uct_md_mkey_pack_v2 with
     * @ref UCT_MD_MKEY_PACK_FLAG_INVALIDATE_RMA flag.
     */
    UCT_MD_FLAG_INVALIDATE_RMA = UCT_MD_FLAG_V2_FIRST,

    /**
     * Memory domain supports invalidation of memory handle registered by
     * @ref uct_md_mem_reg_v2 with @ref UCT_MD_MEM_ACCESS_REMOTE_ATOMIC flag and
     * packed key by @ref uct_md_mkey_pack_v2 with
     * @ref UCT_MD_MKEY_PACK_FLAG_INVALIDATE_AMO flag.
     */
    UCT_MD_FLAG_INVALIDATE_AMO = UCS_BIT(12)
} uct_md_flags_v2_t;


/**
 * @ingroup UCT_MD
 * @brief Query for memory domain attributes.
 *
 * @param [in]  md       Memory domain to query.
 * @param [out] md_attr  Filled with memory domain attributes.
 * @return               Error code.
 */
ucs_status_t uct_md_query_v2(uct_md_h md, uct_md_attr_v2_t *md_attr);


/**
 * @ingroup UCT_MD
 *
 * @brief Pack a memory key as a remote or shared one.
 *
 * This routine packs a local memory handle registered by @ref uct_md_mem_reg
 * into a memory buffer, which then could be deserialized by a peer and used in
 * UCT operations.
 *
 * @param [in]  md          Handle to memory domain.
 * @param [in]  memh        Pack a remote key for this memory handle.
 * @param [in]  address     Memory address to expose for remote access.
 * @param [in]  length      The size (in bytes) of memory that will be exposed
 *                          for remote access.
 * @param [in]  params      Operation parameters, see @ref
 *                          uct_md_mkey_pack_params_t.
 * @param [out] mkey_buffer Pointer to a buffer to hold the packed memory key.
 *                          The size of this buffer should be at least
 *                          @ref uct_md_attr_t::rkey_packed_size or
 *                          @ref uct_md_attr_t::exported_mkey_packed_size, as
 *                          returned by @ref uct_md_query.
 * @return                  Error code.
 */
ucs_status_t uct_md_mkey_pack_v2(uct_md_h md, uct_mem_h memh,
                                 void *address, size_t length,
                                 const uct_md_mkey_pack_params_t *params,
                                 void *mkey_buffer);


/**
 * @ingroup UCT_MD
 *
 * @brief Locally attach to a remote memory.
 *
 * This routine attaches a local memory handle to a memory region
 * registered by @ref uct_md_mem_reg and packed by
 * @ref uct_md_mem_pack_v2 by a peer to allow performing local operations
 * on a remote memory.
 *
 * @param [in]  md            Handle to memory domain.
 * @param [in]  mkey_buffer   Buffer with a packed remote memory handle as
 *                            returned from @ref uct_md_mkey_pack_v2.
 * @param [in]  params        Attach parameters, see @ref
 *                            uct_md_mem_attach_params_t.
 * @param [out] memh_p        Memory handle attached to a remote memory.
 *
 * @return                    Error code.
 */
ucs_status_t uct_md_mem_attach(uct_md_h md, const void *mkey_buffer,
                               uct_md_mem_attach_params_t *params,
                               uct_mem_h *memh_p);


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


/**
 * @ingroup UCT_RESOURCE
 * @brief Connect endpoint to a remote endpoint.
 *
 * requires @ref UCT_IFACE_FLAG_CONNECT_TO_EP capability.
 *
 * @param [in] ep           Endpoint to connect.
 * @param [in] device_addr  Remote device address.
 * @param [in] iface_addr   Remote interface address or NULL if such address is
 *                          not available.
 * @param [in] ep_addr      Remote endpoint address.
 * @param [in] params       Parameters as defined in @ref
 *                          uct_ep_connect_to_ep_params_t.
 *
 * @return UCS_OK           Operation has been initiated successfully.
 *         Other            Error codes as defined by @ref ucs_status_t.
 */
ucs_status_t uct_ep_connect_to_ep_v2(uct_ep_h ep,
                                     const uct_device_addr_t *device_addr,
                                     const uct_ep_addr_t *ep_addr,
                                     const uct_ep_connect_to_ep_params_t *params);

/**
 * @ingroup UCT_RESOURCE
 * @brief Checks if an endpoint is connected to a remote address.
 *
 * This function checks if a local endpoint is connected to a remote address.
 *
 * @param [in] ep      Endpoint to check.
 * @param [in] params  Parameters as defined in @ref
 *                     uct_ep_is_connected_params_t.
 *
 * @return Nonzero if connected, 0 otherwise.
 */
int uct_ep_is_connected(uct_ep_h ep,
                        const uct_ep_is_connected_params_t *params);

/**
 * @ingroup UCT_MD
 *
 * @brief This routine compares two remote keys.
 *
 * It sets the @a result argument to < 0 if rkey1 is lower than rkey2, 0 if they
 * are equal or > 0 if rkey1 is greater than rkey2. The result value can be used
 * for sorting remote keys.
 *
 * @param[in]  component  Component to use for the comparison
 * @param[in]  rkey1      First rkey to compare
 * @param[in]  rkey2      Second rkey to compare
 * @param[in]  params     Additional parameters for comparison
 * @param[out] result     Result of the comparison
 *
 * @return UCS_OK         @a result contains the comparison result
 *         Other          Error codes as defined by @ref ucs_status_t.
 */
ucs_status_t
uct_rkey_compare(uct_component_h component, uct_rkey_t rkey1, uct_rkey_t rkey2,
                 const uct_rkey_compare_params_t *params, int *result);

END_C_DECLS

#endif
