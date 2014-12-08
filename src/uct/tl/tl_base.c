/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "tl_base.h"
#include "context.h"

#include <uct/api/uct.h>
#include <ucs/type/class.h>


static ucs_status_t uct_iface_stub_am_handler(void *desc, unsigned length, void *arg)
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

static UCS_CLASS_INIT_FUNC(uct_iface_t, uct_iface_ops_t *ops)
{

    self->ops = *ops;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_iface_t)
{
}

UCS_CLASS_DEFINE(uct_iface_t, void);


static UCS_CLASS_INIT_FUNC(uct_base_iface_t, uct_iface_ops_t *ops)
{
    uint8_t id;

    UCS_CLASS_CALL_SUPER_INIT(ops);

    for (id = 0; id < UCT_AM_ID_MAX; ++id) {
        self->am[id].cb  = uct_iface_stub_am_handler;
        self->am[id].arg = (void*)(uintptr_t)id;
    }
    return UCS_OK;
}


static UCS_CLASS_CLEANUP_FUNC(uct_base_iface_t)
{
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
