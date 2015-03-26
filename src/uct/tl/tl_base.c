/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include "tl_base.h"
#include "context.h"

#include <uct/api/uct.h>


typedef struct {
    uct_mem_h memh;
} uct_iface_mp_chunk_hdr_t;


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
    }
};

static ucs_stats_class_t uct_iface_stats_class = {
    .name = "uct_iface",
    .num_counters = UCT_IFACE_STAT_LAST,
    .counter_names = {
        [UCT_IFACE_STAT_RX_AM]       = "rx_am",
        [UCT_IFACE_STAT_RX_AM_BYTES] = "rx_am_bytes",
        [UCT_IFACE_STAT_TX_NO_DESC]  = "tx_no_desc",
        [UCT_IFACE_STAT_RX_NO_DESC]  = "rx_no_desc",
        [UCT_IFACE_STAT_FLUSH]       = "flush",
    }
};
#endif


ucs_status_t uct_iface_mp_chunk_alloc(void *mp_context, size_t *size, void **chunk_p
                                      UCS_MEMTRACK_ARG)
{
    uct_base_iface_t *iface = mp_context;
    uct_iface_mp_chunk_hdr_t *hdr;
    ucs_status_t status;
    uct_mem_h memh;
    size_t length;
    void *ptr;

    length = sizeof(*hdr) + *size;
    status = uct_pd_mem_alloc(iface->super.pd, UCT_ALLOC_METHOD_DEFAULT,
                              &length, 1, &ptr, &memh, UCS_MEMTRACK_VAL_ALWAYS);
    if (status != UCS_OK) {
        return status;
    }

     hdr         = ptr;
    hdr->memh   = memh;
    *size       = length - sizeof(*hdr);
    *chunk_p    = hdr + 1;
    return UCS_OK;
}

void uct_iface_mp_chunk_free(void *mp_context, void *chunk)
{
    uct_base_iface_t *iface = mp_context;
    uct_iface_mp_chunk_hdr_t *hdr;

    hdr = chunk - sizeof(*hdr);
    uct_pd_mem_free(iface->super.pd, hdr, hdr->memh);
}

void uct_iface_mp_init_obj(void *mp_context, void *obj, void *chunk, void *arg)
{
    uct_iface_t *iface = mp_context;
    uct_iface_mpool_init_obj_cb_t cb = arg;
    uct_iface_mp_chunk_hdr_t *hdr;

    hdr = chunk - sizeof(*hdr);
    if (cb) {
        cb(iface, obj, hdr->memh);
    }
}

ucs_status_t uct_iface_mpool_create(uct_iface_h iface, size_t elem_size,
                                    size_t align_offset, size_t alignment,
                                    uct_iface_mpool_config_t *config, unsigned grow,
                                    uct_iface_mpool_init_obj_cb_t init_obj_cb,
                                    const char *name, ucs_mpool_h *mp_p)
{
    unsigned elems_per_chunk;

    elems_per_chunk = (config->bufs_grow != 0) ? config->bufs_grow : grow;
    return ucs_mpool_create(name, elem_size, align_offset, alignment,
                            elems_per_chunk, config->max_bufs, iface,
                            uct_iface_mp_chunk_alloc, uct_iface_mp_chunk_free,
                            uct_iface_mp_init_obj, init_obj_cb, mp_p);
}

static ucs_status_t uct_iface_stub_am_handler(void *arg, void *data,
                                              size_t length, void *desc)
{
    uint8_t id = (uintptr_t)arg;
    ucs_warn("got active message id %d, but no handler installed", id);
    return UCS_OK;
}

static void uct_iface_set_stub_am_handler(uct_base_iface_t *iface, uint8_t id)
{
    iface->am[id].cb  = uct_iface_stub_am_handler;
    iface->am[id].arg = (void*)(uintptr_t)id;
}

ucs_status_t uct_iface_set_am_handler(uct_iface_h tl_iface, uint8_t id,
                                      uct_am_callback_t cb, void *arg)
{
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);

    if (id >= UCT_AM_ID_MAX) {
        return UCS_ERR_INVALID_PARAM;
    }

    if (cb == NULL) {
        uct_iface_set_stub_am_handler(iface, id);
    } else {
        iface->am[id].cb  = cb;
        iface->am[id].arg = arg;
    }
    return UCS_OK;
}

ucs_status_t uct_iface_query(uct_iface_h iface, uct_iface_attr_t *iface_attr)
{
    return iface->ops.iface_query(iface, iface_attr);
}

void uct_iface_close(uct_iface_h iface)
{
    iface->ops.iface_close(iface);
}

ucs_status_t uct_iface_get_address(uct_iface_h iface, uct_iface_addr_t *iface_addr)
{
    return iface->ops.iface_get_address(iface, iface_addr);
}

UCS_CLASS_INIT_FUNC(uct_iface_t, uct_iface_ops_t *ops, uct_pd_h pd)
{
    self->ops = *ops;
    self->pd  = pd;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_iface_t)
{
}

UCS_CLASS_DEFINE(uct_iface_t, void);


UCS_CLASS_INIT_FUNC(uct_base_iface_t, uct_iface_ops_t *ops, uct_worker_h worker,
                    uct_pd_h pd, uct_iface_config_t *config
                    UCS_STATS_ARG(ucs_stats_node_t *stats_parent))
{
    ucs_status_t status;
    uint8_t id;

    UCS_CLASS_CALL_SUPER_INIT(uct_iface_t, ops, pd);

    self->worker = worker;

    for (id = 0; id < UCT_AM_ID_MAX; ++id) {
        uct_iface_set_stub_am_handler(self, id);
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

ucs_status_t uct_ep_get_address(uct_ep_h ep, uct_ep_addr_t *ep_addr)
{
    return ep->iface->ops.ep_get_address(ep, ep_addr);
}

ucs_status_t uct_ep_connect_to_iface(uct_ep_h ep, uct_iface_addr_t *iface_addr)
{
    return ep->iface->ops.ep_connect_to_iface(ep, iface_addr);
}

ucs_status_t uct_ep_connect_to_ep(uct_ep_h ep, uct_iface_addr_t *iface_addr,
                                  uct_ep_addr_t *ep_addr)
{
    return ep->iface->ops.ep_connect_to_ep(ep, iface_addr, ep_addr);
}

static UCS_CLASS_INIT_FUNC(uct_ep_t, uct_iface_t *iface)
{
    self->iface = iface;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ep_t)
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


ucs_config_field_t uct_iface_config_table[] = {
  {"MAX_SHORT", "128",
   "Maximal size of short sends. The transport is allowed to support any size up\n"
   "to this limit, the actual size can be lower due to transport constraints.",
   ucs_offsetof(uct_iface_config_t, max_short), UCS_CONFIG_TYPE_MEMUNITS},

  {"MAX_BCOPY", "8192",
   "Maximal size of copy-out sends. The transport is allowed to support any size\n"
   "up to this limit, the actual size can be lower due to transport constraints.",
   ucs_offsetof(uct_iface_config_t, max_bcopy), UCS_CONFIG_TYPE_MEMUNITS},

  {NULL}
};
