/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IFACE_H_
#define UCT_IFACE_H_

#include "uct_worker.h"

#include <uct/api/uct.h>
#include <ucs/config/parser.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/queue.h>
#include <ucs/debug/log.h>
#include <ucs/stats/stats.h>
#include <ucs/sys/sys.h>
#include <ucs/type/class.h>

#include <ucs/datastruct/mpool.inl>


enum {
    UCT_EP_STAT_AM,
    UCT_EP_STAT_PUT,
    UCT_EP_STAT_GET,
    UCT_EP_STAT_ATOMIC,
    UCT_EP_STAT_BYTES_SHORT,
    UCT_EP_STAT_BYTES_BCOPY,
    UCT_EP_STAT_BYTES_ZCOPY,
    UCT_EP_STAT_NO_RES,
    UCT_EP_STAT_FLUSH,
    UCT_EP_STAT_FLUSH_WAIT,  /* number of times flush called while in progress */
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

#define UCT_TL_IFACE_STAT_FLUSH(_iface) \
    UCS_STATS_UPDATE_COUNTER((_iface)->stats, UCT_IFACE_STAT_FLUSH, 1);
#define UCT_TL_IFACE_STAT_FLUSH_WAIT(_iface) \
    UCS_STATS_UPDATE_COUNTER((_iface)->stats, UCT_IFACE_STAT_FLUSH_WAIT, 1);
#define UCT_TL_IFACE_STAT_FENCE(_iface) \
    UCS_STATS_UPDATE_COUNTER((_iface)->stats, UCT_IFACE_STAT_FENCE, 1);
#define UCT_TL_IFACE_STAT_TX_NO_DESC(_iface) \
    UCS_STATS_UPDATE_COUNTER((_iface)->stats, UCT_IFACE_STAT_TX_NO_DESC, 1);


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


/**
 * Base structure of all interfaces.
 * Includes the AM table which we don't want to expose.
 */
typedef struct uct_base_iface {
    uct_iface_t             super;
    uct_md_h                md;               /* MD this interface is using */
    uct_priv_worker_t       *worker;          /* Worker this interface is on */
    uct_am_handler_t        am[UCT_AM_ID_MAX];/* Active message table */
    uct_am_tracer_t         am_tracer;        /* Active message tracer */
    void                    *am_tracer_arg;   /* Tracer argument */
    uct_error_handler_t     err_handler;      /* Error handler */
    void                    *err_handler_arg; /* Error handler argument */
    uct_worker_progress_t   prog;             /* Will be removed once all transports
                                                 support progress control */
    unsigned                progress_flags;   /* Which progress is currently enabled */

    struct {
        unsigned            num_alloc_methods;
        uct_alloc_method_t  alloc_methods[UCT_ALLOC_METHOD_LAST];
        ucs_log_level_t     failure_level;
    } config;

    UCS_STATS_NODE_DECLARE(stats);           /* Statistics */
} uct_base_iface_t;

UCS_CLASS_DECLARE(uct_base_iface_t, uct_iface_ops_t*,  uct_md_h, uct_worker_h,
                  const uct_iface_params_t*, const uct_iface_config_t*
                  UCS_STATS_ARG(ucs_stats_node_t*) UCS_STATS_ARG(const char*));


/**
 * Stub interface used for failed endpoints
 */
typedef struct uct_failed_iface {
    uct_iface_t       super;
    ucs_queue_head_t  pend_q;
} uct_failed_iface_t;


/**
 * Base structure of all endpoints.
 */
typedef struct uct_base_ep {
    uct_ep_t          super;
    UCS_STATS_NODE_DECLARE(stats);
} uct_base_ep_t;
UCS_CLASS_DECLARE(uct_base_ep_t, uct_base_iface_t*);


/**
 * Transport component.
 */
typedef struct uct_tl_component {
    ucs_status_t           (*query_resources)(uct_md_h md,
                                              uct_tl_resource_desc_t **resources_p,
                                              unsigned *num_resources_p);

    ucs_status_t           (*iface_open)(uct_md_h md, uct_worker_h worker,
                                         const uct_iface_params_t *params,
                                         const uct_iface_config_t *config,
                                         uct_iface_h *iface_p);

    char                   name[UCT_TL_NAME_MAX];/**< Transport name */
    const char             *cfg_prefix;         /**< Prefix for configuration environment vars */
    ucs_config_field_t     *iface_config_table; /**< Defines transport configuration options */
    size_t                 iface_config_size;   /**< Transport configuration structure size */
} uct_tl_component_t;


/**
 * Define a transport component.
 */
#define UCT_TL_COMPONENT_DEFINE(_tlc, _query, _iface_struct, _name, \
                                _cfg_prefix, _cfg_table, _cfg_struct) \
    \
    uct_tl_component_t _tlc = { \
        .query_resources     = _query, \
        .iface_open          = UCS_CLASS_NEW_FUNC_NAME(_iface_struct), \
        .name                = _name, \
        .cfg_prefix          = _cfg_prefix, \
        .iface_config_table  = _cfg_table, \
        .iface_config_size   = sizeof(_cfg_struct) \
    };


/**
 * "Base" structure which defines interface configuration options.
 * Specific transport extend this structure.
 */
struct uct_iface_config {
    size_t            max_short;
    size_t            max_bcopy;

    struct {
        uct_alloc_method_t  *methods;
        unsigned            count;
    } alloc_methods;

    int               failure;   /* Level of failure reports */
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
 * Base structure for private data held inside a pending request. Contains
 * a queue element so we can put this on a queue.
 */
typedef struct {
    ucs_queue_elem_t  queue;
} uct_pending_req_priv_t;


typedef struct {
    uct_pending_purge_callback_t cb;
    void *arg;
} uct_purge_cb_args_t;


/**
 * Add a pending request to the queue.
 */
#define uct_pending_req_push(_queue, _req) \
    ucs_queue_push((_queue), &uct_pending_req_priv(_req)->queue);


/**
 * Dispatch all requests in the pending queue, as long as _cond holds true.
 * _cond is an expression which can use "_priv" variable.
 *
 * @param _priv   Variable which will hold a pointer to request private data.
 * @param _queue  The pending queue.
 * @param _cond   Condition which should be true in order to keep dispatching.
 *
 * TODO support a callback returning UCS_INPROGRESS.
 */
#define uct_pending_queue_dispatch(_priv, _queue, _cond) \
    while (!ucs_queue_is_empty(_queue)) { \
        uct_pending_req_priv_t *_base_priv; \
        uct_pending_req_t *_req; \
        ucs_status_t _status; \
        \
        _base_priv = ucs_queue_head_elem_non_empty((_queue), uct_pending_req_priv_t, \
                                                   queue); \
        \
        UCS_STATIC_ASSERT(sizeof(*(_priv)) <= UCT_PENDING_REQ_PRIV_LEN); \
        _priv = (typeof(_priv))(_base_priv); \
        \
        if (!(_cond)) { \
            break; \
        } \
        \
        _req = ucs_container_of(priv, uct_pending_req_t, priv); \
        ucs_queue_pull_non_empty(_queue); \
        _status = _req->func(_req); \
        if (_status != UCS_OK) { \
            ucs_queue_push_head(_queue, &_base_priv->queue); \
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
        uct_pending_req_priv_t *_base_priv; \
        ucs_queue_iter_t _iter; \
        \
        ucs_queue_for_each_safe(_base_priv, _iter, _queue, queue) { \
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
 * @paral _length   Active message length
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
 * @return Private data field of a pending request.
 */
static inline uct_pending_req_priv_t* uct_pending_req_priv(uct_pending_req_t *req)
{
    UCS_STATIC_ASSERT(sizeof(uct_pending_req_priv_t) <= UCT_PENDING_REQ_PRIV_LEN);
    return (uct_pending_req_priv_t*)&req->priv;
}


extern ucs_config_field_t uct_iface_config_table[];


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

ucs_status_t uct_set_ep_failed(ucs_class_t* cls, uct_ep_h tl_ep, uct_iface_h
                               tl_iface, ucs_status_t status);

ucs_status_t uct_base_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                  uct_completion_t *comp);

ucs_status_t uct_base_iface_fence(uct_iface_h tl_iface, unsigned flags);

void uct_base_iface_progress_enable(uct_iface_h tl_iface, unsigned flags);

void uct_base_iface_progress_enable_cb(uct_base_iface_t *iface,
                                       ucs_callback_t cb, unsigned flags);

void uct_base_iface_progress_disable(uct_iface_h tl_iface, unsigned flags);

ucs_status_t uct_base_ep_flush(uct_ep_h tl_ep, unsigned flags,
                               uct_completion_t *comp);

ucs_status_t uct_base_ep_fence(uct_ep_h tl_ep, unsigned flags);

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
    uct_am_handler_t *handler = &iface->am[id];

    ucs_assert(id < UCT_AM_ID_MAX);
    UCS_STATS_UPDATE_COUNTER(iface->stats, UCT_IFACE_STAT_RX_AM, 1);
    UCS_STATS_UPDATE_COUNTER(iface->stats, UCT_IFACE_STAT_RX_AM_BYTES, length);
    status = handler->cb(handler->arg, data, length, flags);
    ucs_assert((status == UCS_OK) ||
               ((status == UCS_INPROGRESS) && (flags & UCT_CB_PARAM_FLAG_DESC)));
    return status;
}


/**
 * Invoke send completion.
 *
 * @param comp   Completion to invoke.
 * @param data   Optional completion data (operation reply).
 */
static UCS_F_ALWAYS_INLINE
void uct_invoke_completion(uct_completion_t *comp, ucs_status_t status)
{
    ucs_trace_func("comp=%p, count=%d, status=%d", comp, comp->count, status);
    if (--comp->count == 0) {
        comp->func(comp, status);
    }
}

/**
 * Calculates total length of particular iov data buffer.
 * Currently has no support for stride.
 * If stride supported it should be like: length + ((count - 1) * stride)
 */
static UCS_F_ALWAYS_INLINE
size_t uct_iov_get_length(const uct_iov_t *iov)
{
    return iov->count * iov->length;
}

/**
 * Calculates total length of the iov array buffers.
 */
static UCS_F_ALWAYS_INLINE
size_t uct_iov_total_length(const uct_iov_t *iov, size_t iovcnt)
{
    size_t iov_it, total_length = 0;

    for (iov_it = 0; iov_it < iovcnt; ++iov_it) {
        total_length += uct_iov_get_length(&iov[iov_it]);
    }

    return total_length;
}

#endif
