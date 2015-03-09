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
#include <ucs/type/class.h>


typedef struct {
    uct_alloc_method_t method;
    uct_mem_h                memh;
    size_t                   length;
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
    uct_alloc_method_t method;
    uct_iface_mp_chunk_hdr_t *hdr;
    ucs_status_t status;
    uct_mem_h memh;
    size_t length;
    uint8_t i;
    void *ptr;

    if (iface->config.alloc_methods_count == 0) {
        ucs_warn("no allocation methods selected");
        return UCS_ERR_INVALID_PARAM;
    }

    for (i = 0; i < iface->config.alloc_methods_count; ++i) {
        ptr    = NULL;
        length = sizeof(*hdr) + *size;
        method = iface->config.alloc_methods[i];

        status = uct_pd_mem_alloc(iface->super.pd, method, 1,
                                  UCS_MEMTRACK_VAL_ALWAYS, &length, &ptr, &memh);
        if (status == UCS_OK) {
            ucs_debug("allocated %zu->%zu bytes using %s: %p",
                      sizeof(*hdr) + *size, length, uct_alloc_method_names[method],
                      ptr);
            hdr         = ptr;
            hdr->method = method;
            hdr->length = length;
            hdr->memh   = memh;
            *size       = length - sizeof(*hdr);
            *chunk_p    = hdr + 1;
            return UCS_OK;
        }

        /* Fallback to next method in the list */
        ucs_debug("failed to allocate %zu bytes using %s, %s",
                  length, uct_alloc_method_names[method],
                  ((i + 1) < iface->config.alloc_methods_count) ?
                                  "trying next method" :
                                  "no more methods to try");
    }

    return UCS_ERR_NO_MEMORY;
}

void uct_iface_mp_chunk_free(void *mp_context, void *chunk)
{
    uct_base_iface_t *iface = mp_context;
    uct_iface_mp_chunk_hdr_t *hdr;

    hdr = chunk - sizeof(*hdr);
    uct_pd_mem_free(iface->super.pd, hdr->method, hdr, hdr->length, hdr->memh);
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

static ucs_status_t uct_iface_stub_am_handler(void *desc, void *data,
                                              size_t length, void *arg)
{
    uint8_t id = (uintptr_t)arg;
    ucs_warn("got active message id %d, but no handler installed", id);
    return UCS_OK;
}

ucs_status_t uct_set_am_handler(uct_iface_h tl_iface, uint8_t id,
                                uct_am_callback_t cb, void *arg)
{
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);

    if (id >= UCT_AM_ID_MAX) {
        return UCS_ERR_INVALID_PARAM;
    }

    if (cb == NULL) {
        cb = uct_iface_stub_am_handler;
    }

    iface->am[id].cb  = cb;
    iface->am[id].arg = arg;
    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_iface_t, uct_iface_ops_t *ops, uct_pd_h pd)
{
    self->ops = *ops;
    self->pd  = pd;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_iface_t)
{
}

UCS_CLASS_DEFINE(uct_iface_t, void);


static UCS_CLASS_INIT_FUNC(uct_base_iface_t, uct_iface_ops_t *ops, uct_pd_h pd,
                           uct_iface_config_t *config
                           UCS_STATS_ARG(ucs_stats_node_t *stats_parent))
{
    ucs_status_t status;
    unsigned i;
    uint8_t id;

    UCS_CLASS_CALL_SUPER_INIT(ops, pd);

    for (id = 0; id < UCT_AM_ID_MAX; ++id) {
        self->am[id].cb  = uct_iface_stub_am_handler;
        self->am[id].arg = (void*)(uintptr_t)id;
    }

    ucs_assert(config->alloc.count <= UINT8_MAX);
    self->config.alloc_methods_count = config->alloc.count;
    for (i = 0; i < config->alloc.count; ++i) {
        self->config.alloc_methods[i] = config->alloc.prio[i];
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


static UCS_CLASS_INIT_FUNC(uct_ep_t, uct_iface_t *iface)
{
    self->iface = iface;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ep_t)
{
}

UCS_CLASS_DEFINE(uct_ep_t, void);


static UCS_CLASS_INIT_FUNC(uct_base_ep_t, uct_base_iface_t *iface)
{
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(&iface->super);

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


UCS_CONFIG_DEFINE_ARRAY(alloc_method, sizeof(uct_alloc_method_t),
                        UCS_CONFIG_TYPE_ENUM(uct_alloc_method_names));

ucs_config_field_t uct_iface_config_table[] = {
   {"ALLOC", "huge,pd,mmap,heap",
    "How to allocate bounce buffers for the interface. Several allocation\n"
    "methods can be specified, ordered by priority. The allocation methods are\n"
    "attempted one-by-one until one is successful. Allocation methods are:\n"
    " - pd    : Use the protection domain memory allocator.\n"
    " - heap  : Allocate memory from the heap.\n"
    " - mmap  : Request memory from the OS using mmap() call.\n"
    " - huge  : Allocate huge pages.\n",
    ucs_offsetof(uct_iface_config_t, alloc), UCS_CONFIG_TYPE_ARRAY(alloc_method)},

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
