/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IFACE_H_
#define UCT_IFACE_H_

#include "uct_worker.h"

#include <uct/api/uct.h>
#include <uct/base/uct_component.h>
#include <ucs/config/parser.h>
#include <ucs/datastruct/arbiter.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/queue.h>
#include <ucs/debug/log.h>
#include <ucs/debug/debug_int.h>
#include <ucs/stats/stats.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/uid.h>
#include <ucs/type/class.h>
#include <uct/api/v2/uct_v2.h>
#include <ucs/type/param.h>

#include <ucs/datastruct/mpool.inl>


/* UCT IFACE local address flag which packed to ID and indicates if an address
 * is extended by a system namespace information */
#define UCT_IFACE_LOCAL_ADDR_FLAG_NS UCS_BIT(63)


enum {
    UCT_EP_STAT_AM,
    UCT_EP_STAT_PUT,
    UCT_EP_STAT_GET,
    UCT_EP_STAT_ATOMIC,
#if IBV_HW_TM
    UCT_EP_STAT_TAG,
#endif
    UCT_EP_STAT_BYTES_SHORT,
    UCT_EP_STAT_BYTES_BCOPY,
    UCT_EP_STAT_BYTES_ZCOPY,
    UCT_EP_STAT_NO_RES,
    UCT_EP_STAT_FLUSH,
    UCT_EP_STAT_FLUSH_WAIT,  /* number of times flush called while in progress */
    UCT_EP_STAT_PENDING,
    UCT_EP_STAT_FENCE,
    UCT_EP_STAT_LAST
};

enum {
    UCT_IFACE_STAT_RX_AM,
    UCT_IFACE_STAT_RX_AM_BYTES,
    UCT_IFACE_STAT_TX_NO_DESC,
    UCT_IFACE_STAT_FLUSH,
    UCT_IFACE_STAT_FLUSH_WAIT,  /* number of times flush called while in progress */
    UCT_IFACE_STAT_FENCE,
    UCT_IFACE_STAT_LAST
};


/*
 * Statistics macros
 */
#define UCT_TL_EP_STAT_OP(_ep, _op, _method, _size) \
    UCS_STATS_UPDATE_COUNTER((_ep)->stats, UCT_EP_STAT_##_op, 1); \
    UCS_STATS_UPDATE_COUNTER((_ep)->stats, UCT_EP_STAT_BYTES_##_method, _size);
#define UCT_TL_EP_STAT_OP_IF_SUCCESS(_status, _ep, _op, _method, _size) \
    if (_status >= 0) { \
        UCT_TL_EP_STAT_OP(_ep, _op, _method, _size) \
    }
#define UCT_TL_EP_STAT_ATOMIC(_ep) \
    UCS_STATS_UPDATE_COUNTER((_ep)->stats, UCT_EP_STAT_ATOMIC, 1);
#define UCT_TL_EP_STAT_FLUSH(_ep) \
    UCS_STATS_UPDATE_COUNTER((_ep)->stats, UCT_EP_STAT_FLUSH, 1);
#define UCT_TL_EP_STAT_FLUSH_WAIT(_ep) \
    UCS_STATS_UPDATE_COUNTER((_ep)->stats, UCT_EP_STAT_FLUSH_WAIT, 1);
#define UCT_TL_EP_STAT_FENCE(_ep) \
    UCS_STATS_UPDATE_COUNTER((_ep)->stats, UCT_EP_STAT_FENCE, 1);
#define UCT_TL_EP_STAT_PEND(_ep) \
    UCS_STATS_UPDATE_COUNTER((_ep)->stats, UCT_EP_STAT_PENDING, 1);

#define UCT_TL_IFACE_STAT_FLUSH(_iface) \
    UCS_STATS_UPDATE_COUNTER((_iface)->stats, UCT_IFACE_STAT_FLUSH, 1);
#define UCT_TL_IFACE_STAT_FLUSH_WAIT(_iface) \
    UCS_STATS_UPDATE_COUNTER((_iface)->stats, UCT_IFACE_STAT_FLUSH_WAIT, 1);
#define UCT_TL_IFACE_STAT_FENCE(_iface) \
    UCS_STATS_UPDATE_COUNTER((_iface)->stats, UCT_IFACE_STAT_FENCE, 1);
#define UCT_TL_IFACE_STAT_TX_NO_DESC(_iface) \
    UCS_STATS_UPDATE_COUNTER((_iface)->stats, UCT_IFACE_STAT_TX_NO_DESC, 1);


#define UCT_CB_FLAGS_CHECK(_flags) \
    do { \
        if ((_flags) & UCT_CB_FLAG_RESERVED) { \
            ucs_error("Unsupported callback flag 0x%x", UCT_CB_FLAG_RESERVED); \
            return UCS_ERR_INVALID_PARAM; \
        } \
    } while (0)


/**
 * In release mode - do nothing.
 *
 * In debug mode, if _condition is not true, return an error. This could be less
 * optimal because of additional checks, and that compiler needs to generate code
 * for error flow as well.
 */
#define UCT_CHECK_PARAM(_condition, _err_message, ...) \
    if (ENABLE_PARAMS_CHECK && !(_condition)) { \
        ucs_error(_err_message, ## __VA_ARGS__); \
        return UCS_ERR_INVALID_PARAM; \
    }


/**
 * In release mode - do nothing.
 *
 * In debug mode, if @a _params field mask does not have set
 * @ref UCT_EP_PARAM_FIELD_DEV_ADDR and @ref UCT_EP_PARAM_FIELD_IFACE_ADDR
 * flags, return an error.
 */
#define UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(_params) \
    UCT_CHECK_PARAM(ucs_test_all_flags((_params)->field_mask, \
                                       UCT_EP_PARAM_FIELD_DEV_ADDR | \
                                       UCT_EP_PARAM_FIELD_IFACE_ADDR), \
                    "UCT_EP_PARAM_FIELD_DEV_ADDR and UCT_EP_PARAM_FIELD_IFACE_ADDR are not defined")


#define UCT_ATTR_VALUE(_obj, _attrs, _name, _flag, _default) \
    UCS_PARAM_VALUE(UCS_PP_TOKENPASTE3(UCT_, _obj, _ATTR_FIELD), _attrs, \
                    _name, _flag, _default)

#define UCT_EP_PARAM_VALUE(_params, _name, _flag, _default) \
    UCS_PARAM_VALUE(UCT_EP_PARAM_FIELD, _params, _name, _flag, _default)


#define UCT_IFACE_PARAM_VALUE(_params, _name, _flag, _default) \
    UCS_PARAM_VALUE(UCT_IFACE_PARAM_FIELD, _params, _name, _flag, _default)


#define UCT_EP_PARAMS_GET_PATH_INDEX(_params) \
    UCT_EP_PARAM_VALUE(_params, path_index, PATH_INDEX, 0)


/**
 * Check the condition and return status as a pointer if not true.
 */
#define UCT_CHECK_PARAM_PTR(_condition, _err_message, ...) \
    if (ENABLE_PARAMS_CHECK && !(_condition)) { \
        ucs_error(_err_message, ## __VA_ARGS__); \
        return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM); \
    }


/**
 * Check the size of the IOV array
 */
#define UCT_CHECK_IOV_SIZE(_iovcnt, _max_iov, _name) \
    UCT_CHECK_PARAM((_iovcnt) <= (_max_iov), \
                    "iovcnt(%lu) should be limited by %lu in %s", \
                    _iovcnt, _max_iov, _name)


/**
 * In debug mode, if _condition is not true, generate 'Invalid length' error.
 */
#define UCT_CHECK_LENGTH(_length, _min_length, _max_length, _name) \
    { \
        typeof(_length) __length = _length; \
        UCT_CHECK_PARAM((_length) <= (_max_length), \
                        "Invalid %s length: %zu (expected: <= %zu)", \
                        _name, (size_t)(__length), (size_t)(_max_length)); \
        UCT_CHECK_PARAM((ssize_t)(_length) >= (_min_length), \
                        "Invalid %s length: %zu (expected: >= %zu)", \
                        _name, (size_t)(__length), (size_t)(_min_length)); \
    }

/**
 * Skip if this is a zero-length operation.
 */
#define UCT_SKIP_ZERO_LENGTH(_length, ...) \
    if (0 == (_length)) { \
        ucs_trace_data("Zero length request: skip it"); \
        UCS_PP_FOREACH(_UCT_RELEASE_DESC, _, __VA_ARGS__)  \
        return UCS_OK; \
    }
#define _UCT_RELEASE_DESC(_, _desc) \
    ucs_mpool_put(_desc);


/**
 * In debug mode, check that active message ID is valid.
 */
#define UCT_CHECK_AM_ID(_am_id) \
    UCT_CHECK_PARAM((_am_id) < UCT_AM_ID_MAX, \
                    "Invalid active message id (valid range: 0..%d)", \
                    (int)UCT_AM_ID_MAX - 1)


/**
 * In debug mode, check that keepalive params are valid
 */
#define UCT_EP_KEEPALIVE_CHECK_PARAM(_flags, _comp) \
    UCT_CHECK_PARAM((_comp) == NULL, "Unsupported completion on ep_check"); \
    UCT_CHECK_PARAM((_flags) == 0, "Unsupported flags: %x", (_flags));


/**
 * Declare classes for structures defined in api/tl.h
 */
UCS_CLASS_DECLARE(uct_iface_h, uct_iface_ops_t, uct_md_h);
UCS_CLASS_DECLARE(uct_ep_t, uct_iface_h);


/**
 * Active message handle table entry
 */
typedef struct uct_am_handler {
    uct_am_callback_t cb;
    void              *arg;
    uint32_t          flags;
} uct_am_handler_t;


/* Performance estimation operation */
typedef ucs_status_t (*uct_iface_estimate_perf_func_t)(
        uct_iface_h iface, uct_perf_attr_t *perf_attr);


/* Refresh the VFS representation of the interface */
typedef void (*uct_iface_vfs_refresh_func_t)(uct_iface_h iface);


/* Query the attributes of the ep */
typedef ucs_status_t (*uct_ep_query_func_t)(uct_ep_h ep, uct_ep_attr_t *ep_attr);


/* Invalidate the ep to emulate transport level error */
typedef ucs_status_t (*uct_ep_invalidate_func_t)(uct_ep_h ep, unsigned flags);


/* Internal operations, not exposed by the external API */
typedef struct uct_iface_internal_ops {
    uct_iface_estimate_perf_func_t iface_estimate_perf;
    uct_iface_vfs_refresh_func_t   iface_vfs_refresh;
    uct_ep_query_func_t            ep_query;
    uct_ep_invalidate_func_t       ep_invalidate;
} uct_iface_internal_ops_t;


/**
 * Base structure of all interfaces.
 * Includes the AM table which we don't want to expose.
 */
typedef struct uct_base_iface {
    uct_iface_t              super;
    uct_iface_internal_ops_t *internal_ops;    /* Internal operations */
    uct_md_h                 md;               /* MD this interface is using */
    uct_priv_worker_t        *worker;          /* Worker this interface is on */
    uct_am_handler_t         am[UCT_AM_ID_MAX];/* Active message table */
    uct_am_tracer_t          am_tracer;        /* Active message tracer */
    void                     *am_tracer_arg;   /* Tracer argument */
    uct_error_handler_t      err_handler;      /* Error handler */
    void                     *err_handler_arg; /* Error handler argument */
    uint32_t                 err_handler_flags; /* Error handler callback flags */
    uct_worker_progress_t    prog;             /* Will be removed once all transports
                                                  support progress control */
    unsigned                 progress_flags;   /* Which progress is currently enabled */

    struct {
        unsigned             num_alloc_methods;
        uct_alloc_method_t   alloc_methods[UCT_ALLOC_METHOD_LAST];
        ucs_log_level_t      failure_level;
        size_t               max_num_eps;
    } config;

    UCS_STATS_NODE_DECLARE(stats)            /* Statistics */
} uct_base_iface_t;

UCS_CLASS_DECLARE(uct_base_iface_t, uct_iface_ops_t*, uct_iface_internal_ops_t*,
                  uct_md_h, uct_worker_h, const uct_iface_params_t*,
                  const uct_iface_config_t *UCS_STATS_ARG(ucs_stats_node_t*)
                  UCS_STATS_ARG(const char*));


/**
 * Stub interface used for failed endpoints
 */
typedef struct uct_failed_iface {
    uct_iface_t       super;
    ucs_queue_head_t  pend_q;
} uct_failed_iface_t;


/**
 * Keepalive info used by EP
 */
typedef struct uct_keepalive_info {
    unsigned long start_time; /* Process start time */
} uct_keepalive_info_t;


/**
 * Base structure of all endpoints.
 */
typedef struct uct_base_ep {
    uct_ep_t          super;
    UCS_STATS_NODE_DECLARE(stats)
} uct_base_ep_t;
UCS_CLASS_DECLARE(uct_base_ep_t, uct_base_iface_t*);


/**
 * Internal resource descriptor of a transport device
 */
typedef struct uct_tl_device_resource {
    char                     name[UCT_DEVICE_NAME_MAX]; /**< Hardware device name */
    uct_device_type_t        type;       /**< The device represented by this resource
                                              (e.g. UCT_DEVICE_TYPE_NET for a network interface) */
    ucs_sys_device_t         sys_device; /**< The identifier associated with the device
                                              bus_id as captured in ucs_sys_bus_id_t struct */
} uct_tl_device_resource_t;


/**
 * UCT transport definition. This structure should not be used directly; use
 * @ref UCT_TL_DEFINE macro to define a transport.
 */
typedef struct uct_tl {
    char                   name[UCT_TL_NAME_MAX]; /**< Transport name */

    ucs_status_t           (*query_devices)(uct_md_h md,
                                            uct_tl_device_resource_t **tl_devices_p,
                                            unsigned *num_tl_devices_p);

    ucs_status_t           (*iface_open)(uct_md_h md, uct_worker_h worker,
                                         const uct_iface_params_t *params,
                                         const uct_iface_config_t *config,
                                         uct_iface_h *iface_p);

    ucs_config_global_list_entry_t config; /**< Transport configuration entry */
    ucs_list_link_t                list;   /**< Entry in component's transports list */
} uct_tl_t;


/**
 * Base UCT IFACE local address
 */
typedef struct uct_iface_local_addr_base {
    uint64_t id; /* System ID + @ref UCT_IFACE_LOCAL_ADDR_FLAG_NS if a local
                    address is extended by a system namespace information */
} UCS_S_PACKED uct_iface_local_addr_base_t;


/**
 * Extended UCT IFACE local address
 */
typedef struct uct_iface_local_addr_ns {
    uct_iface_local_addr_base_t super; /* Base UCT IFACE local address */
    ucs_sys_ns_t                sys_ns; /* System namespace (IPC or network) */
} UCS_S_PACKED uct_iface_local_addr_ns_t;


/**
 * Define a transport
 *
 * @param _component      Component to add the transport to
 * @param _name           Name of the transport (should be a token, not a string)
 * @param _query_devices  Function to query the list of available devices
 * @param _iface_class    Struct type defining the uct_iface class
 * @param _cfg_prefix     Prefix for configuration variables
 * @param _cfg_table      Transport configuration table
 * @param _cfg_struct     Struct type defining transport configuration
 */
#define UCT_TL_DEFINE(_component, _name, _query_devices, _iface_class, \
                      _cfg_prefix, _cfg_table, _cfg_struct) \
    \
    uct_tl_t uct_##_name##_tl = { \
        .name               = #_name, \
        .query_devices      = _query_devices, \
        .iface_open         = UCS_CLASS_NEW_FUNC_NAME(_iface_class), \
        .config = { \
            .name           = #_name" transport", \
            .prefix         = _cfg_prefix, \
            .table          = _cfg_table, \
            .size           = sizeof(_cfg_struct), \
         } \
    }; \
    UCS_CONFIG_REGISTER_TABLE_ENTRY(&(uct_##_name##_tl).config, &ucs_config_global_list); \
    UCS_STATIC_INIT { \
        ucs_list_add_tail(&(_component)->tl_list, &(uct_##_name##_tl).list); \
    }


/**
 * Declare TL constructor and destructor
 *
 * @param [in] _name   TL name
 */
#define UCT_TL_DECL(_name) \
    void uct_##_name##_init(void); \
    void uct_##_name##_cleanup(void);


/* Helper macro to provide ctor/dtor scope */
#define _UCT_IFACE_CTOR_
#define _UCT_IFACE_DTOR_
#define _UCT_IFACE_CTOR_ctor UCS_F_CTOR
#define _UCT_IFACE_DTOR_ctor UCS_F_DTOR


/**
 * Register/unregister TL
 *
 * @param [in] _name          Component and TL name
 * @param [in] _scope         Scope for functions, must be ctor or empty
 * @param [in] _init_code     Initialization code
 * @param [in] _cleanup_code  Cleanup code
 */
#define UCT_TL_INIT(_component, _name, _scope, _init_code, _cleanup_code) \
    UCS_PP_EXPAND(_UCT_IFACE_CTOR_##_scope) void uct_##_name##_init(void) \
    { \
        _init_code; \
        uct_tl_register(_component, &UCT_TL_NAME(_name)); \
    } \
    UCS_PP_EXPAND(_UCT_IFACE_DTOR_##_scope) void uct_##_name##_cleanup(void) \
    { \
        uct_tl_unregister(&UCT_TL_NAME(_name)); \
        _cleanup_code; \
    }


/**
 * Register/unregister component and TL
 *
 * @param [in] _name          Component and TL name
 * @param [in] _scope         Scope for functions, must be ctor or empty
 * @param [in] _init_code     Initialization code
 * @param [in] _cleanup_code  Cleanup code
 */
#define UCT_SINGLE_TL_INIT(_component, _name, _scope, _init_code, \
                           _cleanup_code) \
    UCT_TL_INIT(_component, _name, _scope, \
                {_init_code; uct_component_register(_component);}, \
                {uct_component_unregister(_component); _cleanup_code;})


#define UCT_TL_NAME(_name) uct_##_name##_tl


/**
 * Transport registration routines
 *
 * @param _component      Component to add the transport to
 * @param _name           Name of the transport (should be a token, not a string)
 * @param _query_devices  Function to query the list of available devices
 * @param _iface_class    Struct type defining the uct_iface class
 * @param _cfg_prefix     Prefix for configuration variables
 * @param _cfg_table      Transport configuration table
 * @param _cfg_struct     Struct type defining transport configuration
 */
#define UCT_TL_DEFINE_ENTRY(_component, _name, _query_devices, _iface_class, \
                            _cfg_prefix, _cfg_table, _cfg_struct) \
    \
    uct_tl_t UCT_TL_NAME(_name) = { \
        .name               = #_name, \
        .query_devices      = _query_devices, \
        .iface_open         = UCS_CLASS_NEW_FUNC_NAME(_iface_class), \
        .config = { \
            .name           = #_name" transport", \
            .prefix         = _cfg_prefix, \
            .table          = _cfg_table, \
            .size           = sizeof(_cfg_struct), \
         } \
    };


/**
 * "Base" structure which defines interface configuration options.
 * Specific transport extend this structure.
 */
struct uct_iface_config {
    struct {
        uct_alloc_method_t  *methods;
        unsigned            count;
    } alloc_methods;

    int               failure;   /* Level of failure reports */
    size_t            max_num_eps;
};


/**
 * Memory pool configuration.
 */
typedef struct uct_iface_mpool_config {
    unsigned          max_bufs;  /* Upper limit to number of buffers */
    unsigned          bufs_grow; /* How many buffers (approx.) are allocated every time */
} uct_iface_mpool_config_t;


/**
 * Define configuration fields for memory pool parameters.
 */
#define UCT_IFACE_MPOOL_CONFIG_FIELDS(_prefix, _dfl_max, _dfl_grow, _mp_name, _offset, _desc) \
    {_prefix "MAX_BUFS", UCS_PP_QUOTE(_dfl_max), \
     "Maximal number of " _mp_name " buffers for the interface. -1 is infinite." \
     _desc, \
     (_offset) + ucs_offsetof(uct_iface_mpool_config_t, max_bufs), UCS_CONFIG_TYPE_INT}, \
    \
    {_prefix "BUFS_GROW", UCS_PP_QUOTE(_dfl_grow), \
     "How much buffers are added every time the " _mp_name " memory pool grows.\n" \
     "0 means the value is chosen by the transport.", \
     (_offset) + ucs_offsetof(uct_iface_mpool_config_t, bufs_grow), UCS_CONFIG_TYPE_UINT}


/**
 * Get a descriptor from memory pool, tell valgrind it's already defined, return
 * error if the memory pool is empty.
 *
 * @param _mp       Memory pool to get descriptor from.
 * @param _desc     Variable to assign descriptor to.
 * @param _failure  What do to if memory poll is empty.
 *
 * @return TX descriptor fetched from memory pool.
 */
#define UCT_TL_IFACE_GET_TX_DESC(_iface, _mp, _desc, _failure) \
    { \
        _desc = ucs_mpool_get_inline(_mp); \
        if (ucs_unlikely((_desc) == NULL)) { \
            UCT_TL_IFACE_STAT_TX_NO_DESC(_iface); \
            _failure; \
        } \
        \
        VALGRIND_MAKE_MEM_DEFINED(_desc, sizeof(*(_desc))); \
    }


#define UCT_TL_IFACE_GET_RX_DESC(_iface, _mp, _desc, _failure) \
    { \
        _desc = ucs_mpool_get_inline(_mp); \
        if (ucs_unlikely((_desc) == NULL)) { \
            uct_iface_mpool_empty_warn(_iface, _mp); \
            _failure; \
        } \
        \
        VALGRIND_MAKE_MEM_DEFINED(_desc, sizeof(*(_desc))); \
    }


#define UCT_TL_IFACE_PUT_DESC(_desc) \
    { \
        ucs_mpool_put_inline(_desc); \
        VALGRIND_MAKE_MEM_UNDEFINED(_desc, sizeof(*(_desc))); \
    }


/**
 * TL Memory pool object initialization callback.
 */
typedef void (*uct_iface_mpool_init_obj_cb_t)(uct_iface_h iface, void *obj,
                                              uct_mem_h memh);


/**
 * Base structure for private data held inside a pending request for TLs
 * which use ucs_arbiter_t to progress pending requests.
 */
typedef struct {
    ucs_arbiter_elem_t  arb_elem;
} uct_pending_req_priv_arb_t;


static UCS_F_ALWAYS_INLINE ucs_arbiter_elem_t *
uct_pending_req_priv_arb_elem(uct_pending_req_t *req)
{
    uct_pending_req_priv_arb_t *priv_arb_p =
        (uct_pending_req_priv_arb_t *)&req->priv;

    return &priv_arb_p->arb_elem;
}


/**
 * Add a pending request to the arbiter.
 */
#define uct_pending_req_arb_group_push(_arbiter_group, _req) \
    do { \
        ucs_arbiter_elem_init(uct_pending_req_priv_arb_elem(_req)); \
        ucs_arbiter_group_push_elem_always(_arbiter_group, \
                                           uct_pending_req_priv_arb_elem(_req)); \
    } while (0)


/**
 * Add a pending request to the head of group in arbiter.
 */
#define uct_pending_req_arb_group_push_head(_arbiter_group, _req) \
    do { \
        ucs_arbiter_elem_init(uct_pending_req_priv_arb_elem(_req)); \
        ucs_arbiter_group_push_head_elem_always(_arbiter_group, \
                                                uct_pending_req_priv_arb_elem(_req)); \
    } while (0)


/**
 * Base structure for private data held inside a pending request for TLs
 * which use ucs_queue_t to progress pending requests.
 */
typedef struct {
    ucs_queue_elem_t    queue_elem;
} uct_pending_req_priv_queue_t;


static UCS_F_ALWAYS_INLINE ucs_queue_elem_t *
uct_pending_req_priv_queue_elem(uct_pending_req_t* req)
{
    uct_pending_req_priv_queue_t *priv_queue_p =
        (uct_pending_req_priv_queue_t *)&req->priv;

    return &priv_queue_p->queue_elem;
}


/**
 * Add a pending request to the queue.
 */
#define uct_pending_req_queue_push(_queue, _req) \
    ucs_queue_push((_queue), uct_pending_req_priv_queue_elem(_req))


typedef struct {
    uct_pending_purge_callback_t cb;
    void                         *arg;
} uct_purge_cb_args_t;


/**
 * Dispatch all requests in the pending queue, as long as _cond holds true.
 * _cond is an expression which can use "_priv" variable.
 *
 * @param _priv   Variable which will hold a pointer to request private data.
 * @param _queue  The pending queue.
 * @param _cond   Condition which should be true in order to keep dispatching.
 */
#define uct_pending_queue_dispatch(_priv, _queue, _cond) \
    while (!ucs_queue_is_empty(_queue)) { \
        uct_pending_req_priv_queue_t *_base_priv; \
        uct_pending_req_t *_req; \
        ucs_status_t _status; \
        \
        _base_priv = ucs_queue_head_elem_non_empty( \
                (_queue), uct_pending_req_priv_queue_t, queue_elem); \
        \
        UCS_STATIC_ASSERT(sizeof(*(_priv)) <= UCT_PENDING_REQ_PRIV_LEN); \
        _priv = (typeof(_priv))(_base_priv); \
        \
        if (!(_cond)) { \
            break; \
        } \
        \
        _req = ucs_container_of(_priv, uct_pending_req_t, priv); \
        ucs_queue_pull_non_empty(_queue); \
        _status = _req->func(_req); \
        if ((_status) == UCS_OK) { \
            /* pending element should be removed from queue */ \
            continue; \
        } \
        \
        /* pending element did not complete; return it to the queue */ \
        ucs_queue_push_head(_queue, &_base_priv->queue_elem); \
        if (UCS_STATUS_IS_ERR(_status)) { \
            break; \
        } \
    }


/**
 * Purge messages from the pending queue.
 *
 * @param _priv   Variable which will hold a pointer to request private data.
 * @param _queue  The pending queue.
 * @param _cond   Condition which should be true in order to remove a request.
 * @param _cb     Callback for purging the request.
 * @return Callback return value.
 */
#define uct_pending_queue_purge(_priv, _queue, _cond, _cb, _arg) \
    { \
        uct_pending_req_priv_queue_t *_base_priv; \
        ucs_queue_iter_t             _iter; \
        \
        ucs_queue_for_each_safe(_base_priv, _iter, _queue, queue_elem) { \
            _priv = (typeof(_priv))(_base_priv); \
            if (_cond) { \
                ucs_queue_del_iter(_queue, _iter); \
                (void)_cb(ucs_container_of(_base_priv, uct_pending_req_t, priv), _arg); \
            } \
        } \
    }


/**
 * Helper macro to trace active message send/receive.
 *
 * @param _iface    Interface.
 * @param _type     Message type (send/receive)
 * @param _am_id    Active message ID.
 * @param _payload  Active message payload.
 * @param _length   Active message length
 */
#define uct_iface_trace_am(_iface, _type, _am_id, _payload, _length, _fmt, ...) \
    if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        char buf[256] = {0}; \
        uct_iface_dump_am(_iface, _type, _am_id, _payload, _length, \
                          buf, sizeof(buf) - 1); \
        ucs_trace_data(_fmt " am_id %d len %zu %s", ## __VA_ARGS__, \
                       _am_id, (size_t)(_length), buf); \
    }


/**
 * Helper macro to invoke the function from iface operations.
 *
 * @param _iface    UCT interface.
 * @param _ops_type Type of iface operations.
 * @param _func     Function to call.
 * @param ...       Parameters that is passed to the function.
 */
#define uct_iface_invoke_ops_func(_iface, _ops_type, _func, ...) \
    ({ \
        _ops_type *__ops = ucs_derived_of((_iface)->ops, _ops_type); \
        __ops->_func(__VA_ARGS__); \
    })


extern ucs_config_field_t uct_iface_config_table[];


extern uct_iface_internal_ops_t uct_base_iface_internal_ops;


/**
 * Initialize a memory pool for buffers used by TL interface.
 *
 * @param mp
 * @param elem_size
 * @param align_offset
 * @param alignment    Data will be aligned to these units.
 * @param config       Memory pool configuration.
 * @param grow         Default number of buffers added for every chunk.
 * @param init_obj_cb  Object constructor.
 * @param name         Memory pool name.
 */
ucs_status_t uct_iface_mpool_init(uct_base_iface_t *iface, ucs_mpool_t *mp,
                                  size_t elem_size, size_t align_offset, size_t alignment,
                                  const uct_iface_mpool_config_t *config, unsigned grow,
                                  uct_iface_mpool_init_obj_cb_t init_obj_cb,
                                  const char *name);


/**
 * Dump active message contents using the user-defined tracer callback.
 */
void uct_iface_dump_am(uct_base_iface_t *iface, uct_am_trace_type_t type,
                       uint8_t id, const void *data, size_t length,
                       char *buffer, size_t max);

void uct_iface_mpool_empty_warn(uct_base_iface_t *iface, ucs_mpool_t *mp);

void uct_iface_set_async_event_params(const uct_iface_params_t *params,
                                      uct_async_event_cb_t *event_cb,
                                      void **event_arg);

ucs_status_t uct_iface_handle_ep_err(uct_iface_h iface, uct_ep_h ep,
                                      ucs_status_t status);

/**
 * Initialize AM data alignment and its offset based on the user configuration
 * provided in interface parameters.
 *
 * @param [in]  params         User defined interface parameters.
 * @param [in]  elem_size      Transport receive buffer size.
 * @param [in]  base_offset    Default offset in the transport receive buffer,
 *                             which should be aligned to the certain boundary.
 * @param [in]  payload_offset Offset to the payload in the transport receive
 *                             buffer.
 * @param [out] align          Alignment of the Active Message data on the
 *                             receiver.
 * @param [out] align_offset   Offset in the incoming Active Message which
 *                             should be aligned to the @a align boundary.
 *
 * @return UCS_OK on success or UCS_ERR_INVALID_PARAM if user specified invalid
 *         combination of @a am_alignment and @a am_align_offset in @a params.
 */
ucs_status_t
uct_iface_param_am_alignment(const uct_iface_params_t *params, size_t elem_size,
                             size_t base_offset, size_t payload_offset,
                             size_t *align, size_t *align_offset);

void uct_base_iface_query(uct_base_iface_t *iface, uct_iface_attr_t *iface_attr);

ucs_status_t uct_single_device_resource(uct_md_h md, const char *dev_name,
                                        uct_device_type_t dev_type,
                                        ucs_sys_device_t sys_device,
                                        uct_tl_device_resource_t **tl_devices_p,
                                        unsigned *num_tl_devices_p);

ucs_status_t uct_base_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                  uct_completion_t *comp);

ucs_status_t uct_base_iface_fence(uct_iface_h tl_iface, unsigned flags);

void uct_base_iface_progress_enable(uct_iface_h tl_iface, unsigned flags);

void uct_base_iface_progress_enable_cb(uct_base_iface_t *iface,
                                       ucs_callback_t cb, unsigned flags);

void uct_base_iface_progress_disable(uct_iface_h tl_iface, unsigned flags);

ucs_status_t
uct_base_iface_estimate_perf(uct_iface_h iface, uct_perf_attr_t *perf_attr);

ucs_status_t uct_base_ep_flush(uct_ep_h tl_ep, unsigned flags,
                               uct_completion_t *comp);

ucs_status_t uct_base_ep_fence(uct_ep_h tl_ep, unsigned flags);

void uct_iface_get_local_address(uct_iface_local_addr_ns_t *addr_ns,
                                 ucs_sys_namespace_type_t sys_ns_type);

int uct_iface_local_is_reachable(uct_iface_local_addr_ns_t *addr_ns,
                                 ucs_sys_namespace_type_t sys_ns_type);

/*
 * Invoke active message handler.
 *
 * @param iface    Interface to invoke the handler for.
 * @param id       Active message ID.
 * @param data     Received data.
 * @param length   Length of received data.
 * @param flags    Mask with @ref uct_cb_param_flags
 */
static inline ucs_status_t
uct_iface_invoke_am(uct_base_iface_t *iface, uint8_t id, void *data,
                    unsigned length, unsigned flags)
{
    ucs_status_t     status;
    uct_am_handler_t *handler;

    ucs_assertv(id < UCT_AM_ID_MAX, "invalid am id: %d (max: %lu)",
                id, UCT_AM_ID_MAX - 1);

    UCS_STATS_UPDATE_COUNTER(iface->stats, UCT_IFACE_STAT_RX_AM, 1);
    UCS_STATS_UPDATE_COUNTER(iface->stats, UCT_IFACE_STAT_RX_AM_BYTES, length);

    handler = &iface->am[id];
    status = handler->cb(handler->arg, data, length, flags);
    ucs_assertv((status == UCS_OK) ||
                ((status == UCS_INPROGRESS) && (flags &
                                                UCT_CB_PARAM_FLAG_DESC)),
                "%s(arg=%p data=%p length=%u flags=0x%x) returned %s",
                ucs_debug_get_symbol_name((void*)handler->cb), handler->arg,
                data, length, flags, ucs_status_string(status));
    return status;
}


/**
 * Invoke send completion.
 *
 * @param comp   Completion to invoke.
 * @param status Status of completed operation.
 */
static UCS_F_ALWAYS_INLINE
void uct_invoke_completion(uct_completion_t *comp, ucs_status_t status)
{
    ucs_trace_func("comp=%p (%s) count=%d status=%d", comp,
                   ucs_debug_get_symbol_name((void*)comp->func), comp->count,
                   status);
    ucs_assertv(comp->count > 0, "comp=%p (%s) count=%d status=%d", comp,
                ucs_debug_get_symbol_name((void*)comp->func), comp->count,
                status);

    uct_completion_update_status(comp, status);
    if (--comp->count == 0) {
        comp->func(comp);
    }
}


/**
 * Copy data to target am_short buffer
 */
static UCS_F_ALWAYS_INLINE
void uct_am_short_fill_data(void *buffer, uint64_t header, const void *payload,
                            size_t length)
{
    /**
     * Helper structure to fill send buffer of short messages for
     * non-accelerated transports
     */
    struct uct_am_short_packet {
        uint64_t header;
        char     payload[];
    } UCS_S_PACKED *packet = (struct uct_am_short_packet*)buffer;

    packet->header = header;
    /* suppress false positive diagnostic from uct_mm_ep_am_common_send call */
    /* cppcheck-suppress ctunullpointer */
    memcpy(packet->payload, payload, length);
}


static UCS_F_ALWAYS_INLINE
ucs_log_level_t uct_base_iface_failure_log_level(uct_base_iface_t *iface,
                                                 ucs_status_t err_handler_status,
                                                 ucs_status_t status)
{
    if (err_handler_status != UCS_OK) {
        return UCS_LOG_LEVEL_FATAL;
    } else if ((status == UCS_ERR_ENDPOINT_TIMEOUT) ||
               (status == UCS_ERR_CONNECTION_RESET)) {
        return iface->config.failure_level;
    } else {
        return UCS_LOG_LEVEL_ERROR;
    }
}


ucs_status_t uct_base_ep_am_short_iov(uct_ep_h ep, uint8_t id, const uct_iov_t *iov,
                                      size_t iovcnt);

int uct_ep_get_process_proc_dir(char *buffer, size_t max_len, pid_t pid);

ucs_status_t uct_ep_keepalive_init(uct_keepalive_info_t *ka, pid_t pid);

void uct_ep_keepalive_check(uct_ep_h ep, uct_keepalive_info_t *ka, pid_t pid,
                            unsigned flags, uct_completion_t *comp);

void uct_ep_set_iface(uct_ep_h ep, uct_iface_t *iface);

ucs_status_t uct_base_ep_stats_reset(uct_base_ep_t *ep, uct_base_iface_t *iface);

void uct_iface_vfs_refresh(void *obj);

ucs_status_t uct_ep_invalidate(uct_ep_h ep, unsigned flags);

void uct_tl_register(uct_component_t *component, uct_tl_t *tl);

void uct_tl_unregister(uct_tl_t *tl);

static UCS_F_ALWAYS_INLINE int uct_ep_op_is_zcopy(uct_ep_operation_t op)
{
    return UCS_BIT(op) & (UCS_BIT(UCT_EP_OP_AM_ZCOPY) |
                          UCS_BIT(UCT_EP_OP_PUT_ZCOPY) |
                          UCS_BIT(UCT_EP_OP_GET_ZCOPY) |
                          UCS_BIT(UCT_EP_OP_EAGER_ZCOPY));
}

#endif
