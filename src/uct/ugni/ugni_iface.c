/**
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include <uct/tl/context.h>
#include <ucs/type/class.h>

#include "ugni_iface.h"
#include "ugni_context.h"

uct_iface_ops_t uct_ugni_iface_ops;

static void uct_ugni_progress(void *arg)
{
    /* TBD */
}

static UCS_CLASS_INIT_FUNC(uct_ugni_iface_t, uct_context_h context,
        const char *dev_name, uct_iface_config_t *config)
{
    uct_ugni_context_t *ugni_ctx = ucs_component_get(context, ugni, uct_ugni_context_t);
    uct_ugni_device_t *dev;

    UCS_CLASS_CALL_SUPER_INIT(uct_ugni_iface_ops);

    dev = uct_ugni_device_by_name(ugni_ctx, dev_name);
    if (NULL == dev) {
        ucs_warn("No device was found: %s", dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    self->super.pd = &(dev->super);
    self->dev = dev;

    ucs_notifier_chain_add(&context->progress_chain, uct_ugni_progress,
            self);
    /* TBD: Resources are allocted later on */
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ugni_iface_t)
{
    uct_context_h context = self->super.pd->context;
    ucs_notifier_chain_remove(&context->progress_chain, uct_ugni_progress, self);
}

static UCS_CLASS_DEFINE_NEW_FUNC(uct_ugni_iface_t, uct_iface_t, uct_context_h, const char*, uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ugni_iface_t, uct_iface_t);
UCS_CLASS_DEFINE(uct_ugni_iface_t, uct_iface_t);

uct_tl_ops_t uct_ugni_tl_ops = {
    .query_resources     = uct_ugni_query_resources,
    .iface_open          = UCS_CLASS_NEW_FUNC_NAME(uct_ugni_iface_t),
    .rkey_unpack         = uct_ugni_rkey_unpack,
};

uct_iface_ops_t uct_ugni_iface_ops = {
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_iface_t),
    .iface_get_address   = NULL,//uct_ugni_iface_get_address,
    .iface_flush         = NULL,//uct_ugni_iface_flush,
    .ep_get_address      = NULL,//uct_ugni_ep_get_address,
    .ep_connect_to_iface = NULL,//NULL,
    .ep_connect_to_ep    = NULL,//uct_ugni_ep_connect_to_ep,
    .iface_query         = NULL,//uct_ugni_iface_query,
    .ep_put_short        = NULL,//uct_ugni_ep_put_short,
    .ep_create           = NULL,//UCS_CLASS_NEW_FUNC_NAME(uct_ugni_ep_t),
    .ep_destroy          = NULL,//UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_ep_t),
};
