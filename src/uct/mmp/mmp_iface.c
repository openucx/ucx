/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "ucs/type/class.h"
#include "uct/tl/context.h"

#include "mmp_iface.h"
#include "mmp_ep.h"

unsigned mmp_domain_global_counter = 0;

static void uct_mmp_progress(void *arg)
{
}

static ucs_status_t uct_mmp_iface_flush(uct_iface_h tl_iface)
{
}

/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_mmp_iface_t)(uct_iface_t*);

ucs_status_t uct_mmp_iface_get_address(uct_iface_h tl_iface, 
                                       uct_iface_addr_t *iface_addr)
{
}

ucs_status_t uct_mmp_iface_query(uct_iface_h iface, uct_iface_attr_t *iface_attr)
{
}

#define UCT_mmp_RKEY_MAGIC  0xdeadbeefLL

static ucs_status_t uct_mmp_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
}

static ucs_status_t uct_mmp_mem_map(uct_pd_h pd, void **address_p, 
                                    size_t *length_p, unsigned flags, 
                                    uct_lkey_t *lkey_p UCS_MEMTRACK_ARG)
{
}

static ucs_status_t uct_mmp_mem_unmap(uct_pd_h pd, uct_lkey_t lkey)
{
}

static ucs_status_t uct_mmp_rkey_pack(uct_pd_h pd, uct_lkey_t lkey,
                                      void *rkey_buffer)
{
}

static void uct_mmp_rkey_release(uct_context_h context, uct_rkey_t key)
{
    ucs_free((void *)key);
}

ucs_status_t uct_mmp_rkey_unpack(uct_context_h context, void *rkey_buffer,
                                 uct_rkey_bundle_t *rkey_ob)
{
}

uct_iface_ops_t uct_mmp_iface_ops = {
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_mmp_iface_t),
    .iface_get_address   = uct_mmp_iface_get_address,
    .iface_flush         = uct_mmp_iface_flush,
    .ep_get_address      = uct_mmp_ep_get_address,
    .ep_connect_to_iface = NULL,
    .ep_connect_to_ep    = uct_mmp_ep_connect_to_ep,
    .iface_query         = uct_mmp_iface_query,
    .ep_put_short        = uct_mmp_ep_put_short,
    .ep_am_short         = uct_mmp_ep_am_short,
    .ep_create           = UCS_CLASS_NEW_FUNC_NAME(uct_mmp_ep_t),
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_mmp_ep_t),
};

uct_pd_ops_t uct_mmp_pd_ops = {
    .query        = uct_mmp_pd_query,
    .mem_map      = uct_mmp_mem_map,
    .mem_unmap    = uct_mmp_mem_unmap,
    .rkey_pack    = uct_mmp_rkey_pack,
};

static void uct_mmp_free_fma_out_init(void *mp_context, void *obj, 
                                      void *chunk, void *arg)
{
}

static UCS_CLASS_INIT_FUNC(uct_mmp_iface_t, uct_context_h context,
                           const char *dev_name, size_t rx_headroom,
                           uct_iface_config_t *tl_config)
{
}

static UCS_CLASS_CLEANUP_FUNC(uct_mmp_iface_t)
{
}

UCS_CLASS_DEFINE(uct_mmp_iface_t, uct_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_mmp_iface_t, uct_iface_t, uct_context_h,
                                 const char*, size_t, uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_mmp_iface_t, uct_iface_t);

uct_tl_ops_t uct_mmp_tl_ops = {
    .query_resources     = uct_mmp_query_resources,
    .iface_open          = UCS_CLASS_NEW_FUNC_NAME(uct_mmp_iface_t),
    .rkey_unpack         = uct_mmp_rkey_unpack,
};

#define UCT_mmp_LOCAL_CQ (8192)
ucs_status_t mmp_activate_iface(uct_mmp_iface_t *iface, 
                                uct_mmp_context_t *mmp_ctx)
{
}
