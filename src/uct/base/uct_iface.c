/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "uct_iface.h"
#include "uct_md.h"

#include <uct/api/uct.h>
#include <ucs/time/time.h>


#if ENABLE_STATS
static ucs_stats_class_t uct_ep_stats_class = {
    .name = "uct_ep",
    .num_counters = UCT_EP_STAT_LAST,
    .counter_names = {
        [UCT_EP_STAT_AM]          = "am",
        [UCT_EP_STAT_PUT]         = "put",
        [UCT_EP_STAT_GET]         = "get",
        [UCT_EP_STAT_ATOMIC]      = "atomic",
        [UCT_EP_STAT_BYTES_SHORT] = "bytes_short",
        [UCT_EP_STAT_BYTES_BCOPY] = "bytes_bcopy",
        [UCT_EP_STAT_BYTES_ZCOPY] = "bytes_zcopy",
        [UCT_EP_STAT_NO_RES]      = "no_res",
        [UCT_EP_STAT_FLUSH]       = "flush",
        [UCT_EP_STAT_FLUSH_WAIT]  = "flush_wait",
        [UCT_EP_STAT_FENCE]       = "fence"
    }
};

static ucs_stats_class_t uct_iface_stats_class = {
    .name = "uct_iface",
    .num_counters = UCT_IFACE_STAT_LAST,
    .counter_names = {
        [UCT_IFACE_STAT_RX_AM]       = "rx_am",
        [UCT_IFACE_STAT_RX_AM_BYTES] = "rx_am_bytes",
        [UCT_IFACE_STAT_TX_NO_DESC]  = "tx_no_desc",
        [UCT_IFACE_STAT_FLUSH]       = "flush",
        [UCT_IFACE_STAT_FLUSH_WAIT]  = "flush_wait",
        [UCT_IFACE_STAT_FENCE]       = "fence"
    }
};
#endif


static ucs_status_t uct_iface_stub_am_handler(void *arg, void *data,
                                              size_t length, void *desc)
{
    uint8_t id = (uintptr_t)arg;
    ucs_warn("got active message id %d, but no handler installed", id);
    return UCS_OK;
}

static void uct_iface_set_stub_am_handler(uct_base_iface_t *iface, uint8_t id)
{
    iface->am[id].cb    = uct_iface_stub_am_handler;
    iface->am[id].arg   = (void*)(uintptr_t)id;
    iface->am[id].flags = UCT_AM_CB_FLAG_ASYNC;
}

ucs_status_t uct_iface_set_am_handler(uct_iface_h tl_iface, uint8_t id,
                                      uct_am_callback_t cb, void *arg,
                                      uint32_t flags)
{
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);
    ucs_status_t status;
    uct_iface_attr_t attr;

    if (id >= UCT_AM_ID_MAX) {
        ucs_error("active message id out-of-range (got: %d max: %d)", id,
                  (int)UCT_AM_ID_MAX);
        return UCS_ERR_INVALID_PARAM;
    }

    if (cb == NULL) {
        uct_iface_set_stub_am_handler(iface, id);
        return UCS_OK;
    }

    if (!(flags & (UCT_AM_CB_FLAG_SYNC|UCT_AM_CB_FLAG_ASYNC))) {
        ucs_error("invalid active message flags 0x%x", flags);
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_iface_query(tl_iface, &attr);
    if (status != UCS_OK) {
        return status;
    }

    /* If user wants a synchronous callback, it must be supported, or the
     * callback could be called from another thread.
     */
    if ((flags & UCT_AM_CB_FLAG_SYNC) && !(attr.cap.flags & UCT_IFACE_FLAG_AM_CB_SYNC)) {
        ucs_error("Synchronous active message callback requested, but not supported");
        return UCS_ERR_INVALID_PARAM;
    }

    iface->am[id].cb    = cb;
    iface->am[id].arg   = arg;
    iface->am[id].flags = flags;
    return UCS_OK;
}

ucs_status_t uct_iface_set_am_tracer(uct_iface_h tl_iface, uct_am_tracer_t tracer,
                                     void *arg)
{
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);

    iface->am_tracer     = tracer;
    iface->am_tracer_arg = arg;
    return UCS_OK;
}

void uct_iface_dump_am(uct_base_iface_t *iface, uct_am_trace_type_t type,
                       uint8_t id, const void *data, size_t length,
                       char *buffer, size_t max)
{
    if (iface->am_tracer != NULL) {
        iface->am_tracer(iface->am_tracer_arg, type, id, data, length, buffer, max);
    }
}

void uct_iface_mpool_empty_warn(uct_base_iface_t *iface, ucs_mpool_t *mp)
{
    static ucs_time_t warn_time = 0;
    ucs_time_t now = ucs_get_time();

    /* Limit the rate of warning to once in 30 seconds. This gives reasonable
     * indication about a deadlock without flooding with warnings messages. */
    if (warn_time == 0) {
        warn_time = now;
    }
    if (now - warn_time > ucs_time_from_sec(30)) {
        ucs_warn("Memory pool %s is empty", ucs_mpool_name(mp));
        warn_time = now;
    }
}

ucs_status_t uct_iface_query(uct_iface_h iface, uct_iface_attr_t *iface_attr)
{
    return iface->ops.iface_query(iface, iface_attr);
}

ucs_status_t uct_iface_get_device_address(uct_iface_h iface, uct_device_addr_t *addr)
{
    return iface->ops.iface_get_device_address(iface, addr);
}

ucs_status_t uct_iface_get_address(uct_iface_h iface, uct_iface_addr_t *addr)
{
    return iface->ops.iface_get_address(iface, addr);
}

int uct_iface_is_reachable(const uct_iface_h iface, const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    return iface->ops.iface_is_reachable(iface, dev_addr, iface_addr);
}

ucs_status_t uct_wakeup_open(uct_iface_h iface, unsigned events,
                             uct_wakeup_h *wakeup_p)
{
    if ((events == 0) || (wakeup_p == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    *wakeup_p = ucs_malloc(sizeof(**wakeup_p), "iface_wakeup_context");
    if (*wakeup_p == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    (*wakeup_p)->fd = -1;
    (*wakeup_p)->iface = iface;
    (*wakeup_p)->events = events;

    return iface->ops.iface_wakeup_open(iface, events, *wakeup_p);
}

ucs_status_t uct_wakeup_efd_get(uct_wakeup_h wakeup, int *fd_p)
{
    return wakeup->iface->ops.iface_wakeup_get_fd(wakeup, fd_p);
}

ucs_status_t uct_wakeup_efd_arm(uct_wakeup_h wakeup)
{
    return wakeup->iface->ops.iface_wakeup_arm(wakeup);
}

ucs_status_t uct_wakeup_wait(uct_wakeup_h wakeup)
{
    return wakeup->iface->ops.iface_wakeup_wait(wakeup);
}

ucs_status_t uct_wakeup_signal(uct_wakeup_h wakeup)
{
    return wakeup->iface->ops.iface_wakeup_signal(wakeup);
}

void uct_wakeup_close(uct_wakeup_h wakeup)
{
    wakeup->iface->ops.iface_wakeup_close(wakeup);
    ucs_free(wakeup);
}

void uct_iface_close(uct_iface_h iface)
{
    iface->ops.iface_close(iface);
}

static ucs_status_t uct_base_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                         uct_completion_t *comp)
{
    UCT_TL_IFACE_STAT_FLUSH(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_OK;
}

static ucs_status_t uct_base_iface_fence(uct_iface_h tl_iface, unsigned flags)
{
    UCT_TL_IFACE_STAT_FENCE(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_OK;
}

static ucs_status_t uct_base_ep_flush(uct_ep_h tl_ep, unsigned flags,
                                      uct_completion_t *comp)
{
    UCT_TL_EP_STAT_FLUSH(ucs_derived_of(tl_ep, uct_base_ep_t));
    return UCS_OK;
}

static ucs_status_t uct_base_ep_fence(uct_ep_h tl_ep, unsigned flags)
{
    UCT_TL_EP_STAT_FENCE(ucs_derived_of(tl_ep, uct_base_ep_t));
    return UCS_OK;
}

static void uct_ep_failed_purge_cb(uct_pending_req_t *self, void *arg)
{
    uct_pending_req_push((ucs_queue_head_t*)arg, self);
}

static void uct_ep_failed_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb,
                                void *arg)
{
    uct_failed_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                               uct_failed_iface_t);
    uct_pending_req_t *req;

    ucs_queue_for_each_extract(req, &iface->pend_q, priv, 1) {
        if (cb != NULL) {
            cb(req, arg);
        } else {
            ucs_warn("ep=%p cancelling user pending request %p", tl_ep, req);
        }
    }
}

static void uct_ep_failed_destroy(uct_ep_h tl_ep)
{
    /* Warn user if some pending reqs left*/
    uct_ep_failed_purge (tl_ep, NULL, NULL);

    ucs_free(tl_ep->iface);
    ucs_free(tl_ep);
}

void uct_set_ep_failed(ucs_class_t *cls, uct_ep_h tl_ep, uct_iface_h tl_iface)
{
    uct_failed_iface_t *f_iface;
    uct_iface_ops_t    *ops;

    /* TBD: consider allocating one instance per interface
     * rather than for each endpoint */
    f_iface = ucs_malloc(sizeof(*f_iface), "failed iface");
    if (f_iface == NULL) {
        ucs_error("Could not create failed iface (nomem)");
        return;
    }

    ucs_queue_head_init(&f_iface->pend_q);
    ops = &f_iface->super.ops;

    /* Move all pending requests to the queue.
     * Failed ep will use that queue for purge. */
    uct_ep_pending_purge(tl_ep, uct_ep_failed_purge_cb, &f_iface->pend_q);

    ops->ep_get_address     = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_connect_to_ep   = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_flush           = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_destroy         = uct_ep_failed_destroy;
    ops->ep_pending_add     = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_pending_purge   = uct_ep_failed_purge;
    ops->ep_put_short       = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_put_bcopy       = (void*)ucs_empty_function_return_bc_ep_timeout;
    ops->ep_put_zcopy       = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_get_bcopy       = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_get_zcopy       = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_am_short        = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_am_bcopy        = (void*)ucs_empty_function_return_bc_ep_timeout;
    ops->ep_am_zcopy        = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_atomic_add64    = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_atomic_fadd64   = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_atomic_swap64   = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_atomic_cswap64  = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_atomic_add32    = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_atomic_fadd32   = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_atomic_swap32   = (void*)ucs_empty_function_return_ep_timeout;
    ops->ep_atomic_cswap32  = (void*)ucs_empty_function_return_ep_timeout;

    ucs_class_call_cleanup_chain(cls, tl_ep, -1);

    tl_ep->iface = &f_iface->super;
}

UCS_CLASS_INIT_FUNC(uct_iface_t, uct_iface_ops_t *ops)
{

    self->ops = *ops;
    if (ops->ep_flush == NULL) {
        self->ops.ep_flush = uct_base_ep_flush;
    }

    if (ops->ep_fence == NULL) {
        self->ops.ep_fence = uct_base_ep_fence;
    }

    if (ops->iface_flush == NULL) {
        self->ops.iface_flush = uct_base_iface_flush;
    }

    if (ops->iface_fence == NULL) {
        self->ops.iface_fence = uct_base_iface_fence;
    }

    if (ops->iface_progress_disable == NULL) {
        self->ops.iface_progress_disable = (void *)ucs_empty_function_return_unsupported;
    }

    if (ops->iface_progress_enable == NULL) {
        self->ops.iface_progress_disable = (void *)ucs_empty_function_return_unsupported;
    }
    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(uct_iface_t)
{
}

UCS_CLASS_DEFINE(uct_iface_t, void);


UCS_CLASS_INIT_FUNC(uct_base_iface_t, uct_iface_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_config_t *config
                    UCS_STATS_ARG(ucs_stats_node_t *stats_parent)
                    UCS_STATS_ARG(const char *iface_name))
{
    uint64_t alloc_methods_bitmap;
    uct_alloc_method_t method;
    unsigned i;
    uint8_t id;

    UCS_CLASS_CALL_SUPER_INIT(uct_iface_t, ops);

    self->md            = md;
    self->worker        = worker;
    self->am_tracer     = NULL;
    self->am_tracer_arg = NULL;

    for (id = 0; id < UCT_AM_ID_MAX; ++id) {
        uct_iface_set_stub_am_handler(self, id);
    }

    /* Copy allocation methods configuration. In the process, remove duplicates. */
    UCS_STATIC_ASSERT(sizeof(alloc_methods_bitmap) * 8 >= UCT_ALLOC_METHOD_LAST);
    self->config.num_alloc_methods = 0;
    alloc_methods_bitmap = 0;
    for (i = 0; i < config->alloc_methods.count; ++i) {
        method = config->alloc_methods.methods[i];
        if (alloc_methods_bitmap & UCS_BIT(method)) {
            continue;
        }

        ucs_assert(self->config.num_alloc_methods < UCT_ALLOC_METHOD_LAST);
        self->config.alloc_methods[self->config.num_alloc_methods++] = method;
        alloc_methods_bitmap |= UCS_BIT(method);
    }

    self->config.failure_level = config->failure;

    return UCS_STATS_NODE_ALLOC(&self->stats, &uct_iface_stats_class,
                                stats_parent, "-%s-%p", iface_name, self);
}

static UCS_CLASS_CLEANUP_FUNC(uct_base_iface_t)
{
    UCS_STATS_NODE_FREE(self->stats);
}

UCS_CLASS_DEFINE(uct_base_iface_t, uct_iface_t);


ucs_status_t uct_ep_create(uct_iface_h iface, uct_ep_h *ep_p)
{
    return iface->ops.ep_create(iface, ep_p);
}

ucs_status_t
uct_ep_create_connected(uct_iface_h iface, const uct_device_addr_t *dev_addr,
                        const uct_iface_addr_t *iface_addr, uct_ep_h *ep_p)
{
    return iface->ops.ep_create_connected(iface, dev_addr, iface_addr, ep_p);
}

void uct_ep_destroy(uct_ep_h ep)
{
    ep->iface->ops.ep_destroy(ep);
}

ucs_status_t uct_ep_get_address(uct_ep_h ep, uct_ep_addr_t *addr)
{
    return ep->iface->ops.ep_get_address(ep, addr);
}

ucs_status_t uct_ep_connect_to_ep(uct_ep_h ep, const uct_device_addr_t *dev_addr,
                                  const uct_ep_addr_t *ep_addr)
{
    return ep->iface->ops.ep_connect_to_ep(ep, dev_addr, ep_addr);
}

ucs_status_t uct_iface_progress_disable(uct_iface_h iface, uint32_t flags)
{
    return iface->ops.iface_progress_disable(iface, flags);
}

ucs_status_t uct_iface_progress_enable(uct_iface_h iface, uint32_t flags)
{
    return iface->ops.iface_progress_enable(iface, flags);
}

UCS_CLASS_INIT_FUNC(uct_ep_t, uct_iface_t *iface)
{
    self->iface = iface;
    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(uct_ep_t)
{
}

UCS_CLASS_DEFINE(uct_ep_t, void);


UCS_CLASS_INIT_FUNC(uct_base_ep_t, uct_base_iface_t *iface)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_ep_t, &iface->super);

    return UCS_STATS_NODE_ALLOC(&self->stats, &uct_ep_stats_class, iface->stats,
                                "-%p", self);
}

static UCS_CLASS_CLEANUP_FUNC(uct_base_ep_t)
{
    UCS_STATS_NODE_FREE(self->stats);
}

UCS_CLASS_DEFINE(uct_base_ep_t, uct_ep_t);


UCS_CONFIG_DEFINE_ARRAY(alloc_methods, sizeof(uct_alloc_method_t),
                        UCS_CONFIG_TYPE_ENUM(uct_alloc_method_names));

ucs_config_field_t uct_iface_config_table[] = {
  {"MAX_SHORT", "128",
   "Maximal size of short sends. The transport is allowed to support any size up\n"
   "to this limit, the actual size can be lower due to transport constraints.",
   ucs_offsetof(uct_iface_config_t, max_short), UCS_CONFIG_TYPE_MEMUNITS},

  {"MAX_BCOPY", "8192",
   "Maximal size of copy-out sends. The transport is allowed to support any size\n"
   "up to this limit, the actual size can be lower due to transport constraints.",
   ucs_offsetof(uct_iface_config_t, max_bcopy), UCS_CONFIG_TYPE_MEMUNITS},

  {"ALLOC", "huge,md,mmap,heap",
   "Priority of methods to allocate intermediate buffers for communication",
   ucs_offsetof(uct_iface_config_t, alloc_methods), UCS_CONFIG_TYPE_ARRAY(alloc_methods)},

  {"FAILURE", "error",
   "Level of network failure reporting",
   ucs_offsetof(uct_iface_config_t, failure), UCS_CONFIG_TYPE_ENUM(ucs_log_level_names)},

  {NULL}
};
