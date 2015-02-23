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

/* called in the notifier chain and iface_flush */
static void uct_mmp_progress(void *arg)
{
    /* FIXME not sure what this will do yet */
}

static ucs_status_t uct_mmp_iface_flush(uct_iface_h tl_iface)
{
    uct_mmp_iface_t *iface = ucs_derived_of(tl_iface, uct_mmp_iface_t);
    if (0 == iface->outstanding) {
        return UCS_OK;
    }
    uct_mmp_progress(iface);
    return UCS_ERR_WOULD_BLOCK;
}

/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_mmp_iface_t)(uct_iface_t*);

ucs_status_t uct_mmp_iface_get_address(uct_iface_h tl_iface, 
                                       uct_iface_addr_t *iface_addr)
{
    uct_mmp_iface_t *iface = ucs_derived_of(tl_iface, uct_mmp_iface_t);

    *(uct_mmp_iface_addr_t*)iface_addr = iface->address;
    return UCS_OK;
}

ucs_status_t uct_mmp_iface_query(uct_iface_h iface, uct_iface_attr_t *iface_attr)
{
    /* FIXME all of these values */
    iface_attr->cap.put.max_short      = 2048;
    iface_attr->cap.put.max_bcopy      = 2048;
    iface_attr->cap.put.max_zcopy      = 0;
    iface_attr->iface_addr_len         = sizeof(uct_mmp_iface_addr_t);
    iface_attr->ep_addr_len            = sizeof(uct_mmp_ep_addr_t);
    iface_attr->cap.flags              = UCT_IFACE_FLAG_PUT_SHORT;
    return UCS_OK;
}

static ucs_status_t uct_mmp_mem_map(uct_pd_h pd, void **address_p, 
                                    size_t *length_p, unsigned flags, 
                                    uct_lkey_t *lkey_p UCS_MEMTRACK_ARG)
{
    ucs_status_t rc;
    uct_mmp_pd_t *mmp_pd = ucs_derived_of(pd, uct_mmp_pd_t);
    bool inter_allocation = false;

    if (0 == *length_p) {
        return UCS_ERR_INVALID_PARAM;
    }

    if (NULL == *address_p) {
        *address_p = ucs_malloc(*length_p, "uct_mmp_mem_map");
        if (NULL == *address_p) {
            ucs_error("Failed to allocate %zu bytes", *length_p);
            rc = UCS_ERR_NO_MEMORY;
            goto mem_err;
        }
        ucs_memtrack_allocated(address_p, length_p UCS_MEMTRACK_VAL);
        inter_allocation = true;
    }

    /* FIXME register mapped memory */

mem_err:
    if (inter_allocation) {
        free(*address_p);
    }
    free(mem_hndl);
    return rc;
}

static ucs_status_t uct_mmp_mem_unmap(uct_pd_h pd, uct_lkey_t lkey)
{
    uct_mmp_pd_t *mmp_pd = ucs_derived_of(pd, uct_mmp_pd_t);
    ucs_status_t rc = UCS_OK;

    /* FIXME derive 'mem_hndl' from 'lkey' */

    /* FIXME unregister the memory */

    ucs_free(mem_hndl);
    return rc;
}

#define UCT_MMP_RKEY_MAGIC  0xdeadbeefLL /* FIXME change this for mmp */

static ucs_status_t uct_mmp_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
    /* FIXME what are we going to use for keys here? */
    return UCS_OK;

}

static ucs_status_t uct_mmp_rkey_pack(uct_pd_h pd, uct_lkey_t lkey,
                                      void *rkey_buffer)
{
    /* FIXME what are we going to use for keys here? */
}

static void uct_mmp_rkey_release(uct_context_h context, uct_rkey_t key)
{
    ucs_free((void *)key);
}

ucs_status_t uct_mmp_rkey_unpack(uct_context_h context, void *rkey_buffer,
                                 uct_rkey_bundle_t *rkey_ob)
{
    /* FIXME what are we going to use for keys here? */
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

static UCS_CLASS_INIT_FUNC(uct_mmp_iface_t, uct_context_h context,
                           const char *dev_name, size_t rx_headroom,
                           uct_iface_config_t *tl_config)
{
    uct_mmp_iface_config_t *config = 
        ucs_derived_of(tl_config, uct_mmp_iface_config_t);
    uct_mmp_context_t *mmp_ctx = 
        ucs_component_get(context, mmp, uct_mmp_context_t);
    uct_mmp_device_t *dev;
    ucs_status_t rc;

    UCS_CLASS_CALL_SUPER_INIT(&uct_mmp_iface_ops);

    dev = uct_mmp_device_by_name(mmp_ctx, dev_name);
    if (NULL == dev) {
        ucs_error("No device was found: %s", dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    self->pd.super.ops = &uct_mmp_pd_ops;
    self->pd.super.context = context;
    self->pd.iface = self;

    self->super.super.pd   = &self->pd.super;
    self->dev              = dev;
    self->address.nic_addr = dev->address;

    /* FIXME what else needs to happen for iface init? */

    ucs_notifier_chain_add(&context->progress_chain, uct_mmp_progress, self);

    self->activated = false;
    self->outstanding = 0;
    /* TBD: atomic increment */
    ++mmp_ctx->num_ifaces;
    return mmp_activate_iface(self, mmp_ctx);
}

static UCS_CLASS_CLEANUP_FUNC(uct_mmp_iface_t)
{
    uct_context_h context = self->super.super.pd->context;
    ucs_notifier_chain_remove(&context->progress_chain, uct_mmp_progress, self);

    if (!self->activated) {
        /* We done with release */
        return;
    }

    /* TBD: Clean endpoints first (unbind and destroy) ?*/
    ucs_atomic_add32(&mmp_domain_global_counter, -1);

    /* FIXME tasks to tear down the domain */

    self->activated = false;
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

#define UCT_mmp_LOCAL_CQ (8192) /* FIXME does mmp have a local cq? */
ucs_status_t mmp_activate_iface(uct_mmp_iface_t *iface, 
                                uct_mmp_context_t *mmp_ctx)
{
    int rc, d_id;

    if(iface->activated) {
        return UCS_OK;
    }

    /* Make sure that context is activated */
    rc = mmp_activate_domain(mmp_ctx);
    if (UCS_OK != rc) {
        ucs_error("Failed to activate context, Error status: %d", rc);
        return rc;
    }

    d_id = ucs_atomic_fadd32(&mmp_domain_global_counter, 1);

    /* FIXME with rank info as decided in mmp context */
    iface->domain_id = mmp_ctx->pmi_rank_id + mmp_ctx->pmi_num_of_ranks * d_id;

    /* FIXME tasks to activate the domain */

    iface->activated = true;

    /* iface is activated */
    return UCS_OK;
}
