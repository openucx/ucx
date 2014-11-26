/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "context.h"

#include <uct/api/uct.h>
#include <ucs/type/class.h>


static UCS_CLASS_INIT_FUNC(uct_iface_t, uct_iface_ops_t *ops)
{
    self->ops = *ops;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_iface_t)
{
}

UCS_CLASS_DEFINE(uct_iface_t, void);


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
