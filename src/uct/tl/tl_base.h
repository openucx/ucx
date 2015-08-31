/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef TL_BASE_H_
#define TL_BASE_H_

#include <uct/api/uct.h>
#include <ucs/config/parser.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/queue.h>
#include <ucs/debug/log.h>
#include <ucs/stats/stats.h>
#include <ucs/sys/sys.h>
#include <ucs/type/class.h>


enum {
    UCT_EP_STAT_AM,
    UCT_EP_STAT_PUT,
    UCT_EP_STAT_GET,
    UCT_EP_STAT_ATOMIC,
    UCT_EP_STAT_BYTES_SHORT,
    UCT_EP_STAT_BYTES_BCOPY,
    UCT_EP_STAT_BYTES_ZCOPY,
    UCT_EP_STAT_FLUSH,
    UCT_EP_STAT_LAST
};

enum {
    UCT_IFACE_STAT_RX_AM,
    UCT_IFACE_STAT_RX_AM_BYTES,
    UCT_IFACE_STAT_TX_NO_DESC,
    UCT_IFACE_STAT_RX_NO_DESC,
    UCT_IFACE_STAT_FLUSH,
    UCT_IFACE_STAT_LAST
};


/*
 * Statistics macors
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
#define UCT_TL_IFACE_STAT_FLUSH(_iface) \
    UCS_STATS_UPDATE_COUNTER((_iface)->stats, UCT_IFACE_STAT_FLUSH, 1);


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
 * In debug mode, if _condition is not true, generate 'Invalid length' error.
 */
#define UCT_CHECK_LENGTH(_length, _max_length, _name) \
    UCT_CHECK_PARAM((_length) <= (_max_length), \
                    "Invalid %s length: %zu (expected: <= %zu)", \
                    _name, (size_t)(_length), (size_t)(_max_length))


/**
 * In debug mode, check that active message ID is valid.
 */
#define UCT_CHECK_AM_ID(_am_id) \
    UCT_CHECK_PARAM((_am_id) < UCT_AM_ID_MAX, \
                    "Invalid active message id (valid range: 0..%d)", (int)UCT_AM_ID_MAX - 1)


/**
 * Declare classes for structs defined in api/tl.h
 */
UCS_CLASS_DECLARE(uct_iface_h, uct_iface_ops_t, uct_pd_h);
UCS_CLASS_DECLARE(uct_ep_t, uct_iface_h);


/**
 * Active message handle table entry
 */
typedef struct uct_am_handler {
    uct_am_callback_t cb;
    void              *arg;
} uct_am_handler_t;


/**
 * Base structure of all interfaces.
 * Includes the AM table which we don't want to expose.
 */
typedef struct uct_base_iface {
    uct_iface_t       super;
    uct_pd_h          pd;                    /* PD this interface is using */
    uct_worker_h      worker;                /* Worker this interface is on */
    UCS_STATS_NODE_DECLARE(stats);           /* Statistics */
    uct_am_handler_t  am[UCT_AM_ID_MAX];     /* Active message table */

    struct {
        unsigned            num_alloc_methods;
        uct_alloc_method_t  alloc_methods[UCT_ALLOC_METHOD_LAST];
    } config;

} uct_base_iface_t;
UCS_CLASS_DECLARE(uct_base_iface_t, uct_iface_ops_t*,  uct_pd_h, uct_worker_h,
                  const uct_iface_config_t* UCS_STATS_ARG(ucs_stats_node_t*));


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
    ucs_status_t           (*query_resources)(uct_pd_h pd,
                                              uct_tl_resource_desc_t **resources_p,
                                              unsigned *num_resources_p);

    ucs_status_t           (*iface_open)(uct_pd_h pd, uct_worker_h worker,
                                         const char *dev_name, size_t rx_headroom,
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
    UCS_CONFIG_ARRAY_FIELD(ucs_range_spec_t, signals) lid_path_bits;
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
        _desc = ucs_mpool_get(_mp); \
        if (ucs_unlikely((_desc) == NULL)) { \
            UCS_STATS_UPDATE_COUNTER((_iface)->stats, UCT_IFACE_STAT_TX_NO_DESC, 1); \
            _failure; \
        } \
        \
        VALGRIND_MAKE_MEM_DEFINED(_desc, sizeof(*(_desc))); \
    }


#define UCT_TL_IFACE_GET_RX_DESC(_iface, _mp, _desc, _failure) \
    { \
        _desc = ucs_mpool_get(_mp); \
        if (ucs_unlikely((_desc) == NULL)) { \
            UCS_STATS_UPDATE_COUNTER((_iface)->stats, UCT_IFACE_STAT_RX_NO_DESC, 1); \
            _failure; \
        } \
        \
        VALGRIND_MAKE_MEM_DEFINED(_desc, sizeof(*(_desc))); \
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
        _priv = ucs_derived_of(_base_priv, typeof(*(_priv))); \
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
#define uct_pending_queue_purge(_priv, _queue, _cond, _cb) \
    ({ \
        uct_pending_req_priv_t *_base_priv; \
        ucs_queue_iter_t _iter; \
        ucs_status_t _status; \
        \
        ucs_queue_for_each_safe(_base_priv, _iter, _queue, queue) { \
            _priv = ucs_derived_of(_base_priv, typeof(*_priv)); \
            if (_cond) { \
                ucs_queue_del_iter(_queue, _iter); \
                _status = _cb(ucs_container_of(_base_priv, uct_pending_req_t, priv)); \
                if (_status != UCS_OK) { \
                    ucs_queue_push_head(_queue, &_base_priv->queue); \
                    goto out; \
                } \
            } \
        } \
        _status = UCS_OK; \
    out: \
        _status; \
    })


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
 * Create a memory pool for buffers used by TL interface.
 *
 * @param elem_size
 * @param align_offset
 * @param alignment    Data will be aligned to these units.
 * @param config       Memory pool configuration.
 * @param grow         Default number of buffers added for every chunk.
 * @param init_obj_cb  Object constructor.
 * @param name         Memory pool name.
 * @param mp_p         Filled with memory pool handle.
 */
ucs_status_t uct_iface_mpool_create(uct_iface_h iface, size_t elem_size,
                                    size_t align_offset, size_t alignment,
                                    uct_iface_mpool_config_t *config, unsigned grow,
                                    uct_iface_mpool_init_obj_cb_t init_obj_cb,
                                    const char *name, ucs_mpool_h *mp_p);


/**
 * Invoke active message handler.
 *
 * @param iface    Interface to invoke the handler for.
 * @param id       Active message ID.
 * @param data     Received data.
 * @param length   Length of received data.
 * @param desc     Receive descriptor, as passed to user callback.
 */
static inline ucs_status_t
uct_iface_invoke_am(uct_base_iface_t *iface, uint8_t id, void *data,
                    unsigned length, void *desc)
{
    uct_am_handler_t *handler = &iface->am[id];
    UCS_STATS_UPDATE_COUNTER(iface->stats, UCT_IFACE_STAT_RX_AM, 1);
    UCS_STATS_UPDATE_COUNTER(iface->stats, UCT_IFACE_STAT_RX_AM_BYTES, length);
    return handler->cb(handler->arg, data, length, desc);
}


/**
 * Invoke send completion.
 *
 * @param comp   Completion to invoke.
 * @param data   Optional completion data (operation reply).
 */
static UCS_F_ALWAYS_INLINE
void uct_invoke_completion(uct_completion_t *comp)
{
    ucs_trace_func("comp=%p, count=%d", comp, comp->count);
    if (--comp->count == 0) {
        comp->func(comp);
    }
}

#endif
