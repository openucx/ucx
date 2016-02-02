/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <uct/api/uct.h>
#include "uct_iface.h"
#include "uct_pd.h"


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
        [UCT_EP_STAT_FLUSH]       = "flush",
        [UCT_EP_STAT_FLUSH_WAIT]  = "flush_wait"
    }
};

static ucs_stats_class_t uct_iface_stats_class = {
    .name = "uct_iface",
    .num_counters = UCT_IFACE_STAT_LAST,
    .counter_names = {
        [UCT_IFACE_STAT_RX_AM]       = "rx_am",
        [UCT_IFACE_STAT_RX_AM_BYTES] = "rx_am_bytes",
        [UCT_IFACE_STAT_TX_NO_RES]   = "tx_no_res",
        [UCT_IFACE_STAT_FLUSH]       = "flush",
        [UCT_IFACE_STAT_FLUSH_WAIT]  = "flush_wait",
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
                                      uct_am_callback_t cb, void *arg, uint32_t flags)
{
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);
    ucs_status_t status;
    uct_iface_attr_t attr;

    if (id >= UCT_AM_ID_MAX) {
        return UCS_ERR_INVALID_PARAM;
    }

    if (cb == NULL) {
        uct_iface_set_stub_am_handler(iface, id);
        return UCS_OK;
    }

    if (flags == 0) {
        ucs_debug("am cb flags must not be empty");
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_iface_query(tl_iface, &attr);
    if (status != UCS_OK) {
        return status;
    }

    /* interface must have am capabilities */
    if (!(attr.cap.flags & 
        (UCT_IFACE_FLAG_AM_SHORT|
         UCT_IFACE_FLAG_AM_BCOPY|
         UCT_IFACE_FLAG_AM_ZCOPY))) {
        ucs_debug("attempt to set am cb on iface without active message capabilites");
        return UCS_ERR_INVALID_PARAM;
    }

    if (flags & UCT_AM_CB_FLAG_SYNC) {
        /* to use sync cb iface must have sync flag */
        if (!(attr.cap.flags & UCT_IFACE_FLAG_AM_CB_SYNC)) {
            ucs_debug("am cb sync requested on the interface that does not support it");
            return UCS_ERR_INVALID_PARAM;
        }
    } else if (flags & UCT_AM_CB_FLAG_ASYNC) {
        /* async cb can be set for any iface. In that case it will act as a sync one */ 
        if (!(attr.cap.flags & UCT_IFACE_FLAG_AM_CB_ASYNC)) {
            ucs_debug("am cb async requested on the interface that does not support it. It may be an error");
        }
    } else {
        ucs_debug("am cb invalid capabilites requested 0x%x", flags);
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

ucs_status_t uct_iface_query(uct_iface_h iface, uct_iface_attr_t *iface_attr)
{
    return iface->ops.iface_query(iface, iface_attr);
}

ucs_status_t uct_iface_get_address(uct_iface_h iface, struct sockaddr *addr)
{
    return iface->ops.iface_get_address(iface, addr);
}

int uct_iface_is_reachable(uct_iface_h iface, const struct sockaddr *addr)
{
    return iface->ops.iface_is_reachable(iface, addr);

}

void uct_iface_close(uct_iface_h iface)
{
    iface->ops.iface_close(iface);
}

static ucs_status_t uct_base_iface_flush(uct_iface_h tl_iface)
{
    UCT_TL_IFACE_STAT_FLUSH(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_OK;
}

static ucs_status_t uct_base_ep_flush(uct_ep_h tl_ep)
{
    UCT_TL_EP_STAT_FLUSH(ucs_derived_of(tl_ep, uct_base_ep_t));
    return UCS_OK;
}

UCS_CLASS_INIT_FUNC(uct_iface_t, uct_iface_ops_t *ops)
{

    self->ops = *ops;
    if (ops->ep_flush == NULL) {
        self->ops.ep_flush = uct_base_ep_flush;
    }

    if (ops->iface_flush == NULL) {
        self->ops.iface_flush = uct_base_iface_flush;
    }
    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(uct_iface_t)
{
}

UCS_CLASS_DEFINE(uct_iface_t, void);


UCS_CLASS_INIT_FUNC(uct_base_iface_t, uct_iface_ops_t *ops, uct_pd_h pd,
                    uct_worker_h worker, const uct_iface_config_t *config
                    UCS_STATS_ARG(ucs_stats_node_t *stats_parent))
{
    uint64_t alloc_methods_bitmap;
    uct_alloc_method_t method;
    ucs_status_t status;
    unsigned i;
    uint8_t id;

    UCS_CLASS_CALL_SUPER_INIT(uct_iface_t, ops);

    self->pd            = pd;
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

    status = UCS_STATS_NODE_ALLOC(&self->stats, &uct_iface_stats_class,
                                  stats_parent);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
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

void uct_ep_destroy(uct_ep_h ep)
{
    ep->iface->ops.ep_destroy(ep);
}

ucs_status_t uct_ep_get_address(uct_ep_h ep, struct sockaddr *addr)
{
    return ep->iface->ops.ep_get_address(ep, addr);
}

ucs_status_t uct_ep_connect_to_ep(uct_ep_h ep, const struct sockaddr *addr)
{
    return ep->iface->ops.ep_connect_to_ep(ep, addr);
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
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_ep_t, &iface->super);

    status = UCS_STATS_NODE_ALLOC(&self->stats, &uct_ep_stats_class, iface->stats);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
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

  {"ALLOC", "huge,pd,mmap,heap",
   "Priority of methods to allocate intermediate buffers for communication",
   ucs_offsetof(uct_iface_config_t, alloc_methods), UCS_CONFIG_TYPE_ARRAY(alloc_methods)},

  {NULL}
};
